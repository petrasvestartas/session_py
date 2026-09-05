from __future__ import annotations
from typing import List
from typing import Tuple
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from typing import overload
import numpy as np
import math
import uuid

if TYPE_CHECKING:
    from .proto import nurbscurve_pb2
    from pathlib import Path

from .point import Point
from .vector import Vector
from .plane import Plane
from .tolerance import Tolerance
from .xform import Xform
from .color import Color
from . import nurbsknot
from .nurbsknot import CurveNurbsKnotStyle
from .nurbsknot import CurveInterpStyle


def _evaluate_nurbs_blossom(cvdim, order, cv_stride, CV, nurbsknot, t):
    if CV is None or t is None or nurbsknot is None:
        return None
    if cv_stride < cvdim:
        return None
    degree = order - 1
    for i in range(1, 2 * degree):
        if nurbsknot[i] - nurbsknot[i - 1] < 0.0:
            return None
    if nurbsknot[degree] - nurbsknot[degree - 1] < Tolerance.ZERO_TOLERANCE:
        return None
    P = np.zeros(cvdim)
    space = np.zeros(order)
    for coord in range(cvdim):
        for j in range(order):
            space[j] = CV[j * cv_stride + coord]
        for j in range(1, order):
            for k in range(j, order):
                denom = nurbsknot[degree + k - j] - nurbsknot[k - 1]
                space[k - j] = ((nurbsknot[degree + k - j] - t[j - 1]) / denom * space[k - j] +
                    (t[j - 1] - nurbsknot[k - 1]) / denom * space[k - j + 1])
        P[coord] = space[0]
    return P


def _get_raised_degree_cv(old_order, cvdim, old_cv_stride, oldCV, oldkn, newkn, cv_id):
    if oldCV is None or oldkn is None or newkn is None or cv_id < 0 or cv_id > old_order:
        return None
    old_degree = old_order - 1
    new_degree = old_degree + 1
    newCV = np.zeros(cvdim)
    kn = newkn[cv_id:]
    t = np.zeros(old_degree)
    for i in range(new_degree):
        k = 0
        for j in range(new_degree):
            if j != i:
                t[k] = kn[j]
                k += 1
        P = _evaluate_nurbs_blossom(cvdim, old_order, old_cv_stride, oldCV, oldkn, t)
        if P is None:
            return None
        newCV += P
    newCV /= new_degree
    return newCV


def _next_span_index(order, cv_count, nurbsknot, span_index):
    if span_index < 0 or span_index > cv_count - order or nurbsknot is None:
        return -1
    if span_index < cv_count - order:
        span_index += 1
        while (span_index < cv_count - order and
               nurbsknot[span_index + order - 2] == nurbsknot[span_index + order - 1]):
            span_index += 1
    return span_index


class NurbsCurve:
    """A Non-Uniform Rational B-Spline (NURBS) curve.

    Parameters
    ----------
    dimension : int, optional
        The dimension of the curve (typically 3 for 3D). Defaults to 3.
    is_rational : bool, optional
        Whether the curve is rational (has weights). Defaults to False.
    order : int, optional
        The order of the curve (degree + 1). Defaults to 4 (cubic).
    cv_count : int, optional
        Number of control vertices. Defaults to 0.

    Attributes
    ----------
    guid : str
        Unique identifier of the curve.
    name : str
        Name of the curve.
    m_dim : int
        Dimension of the curve.
    m_is_rat : int
        1 if rational, 0 if non-rational.
    m_order : int
        Order of the curve (degree + 1).
    m_cv_count : int
        Number of control vertices.
    m_cv_stride : int
        Stride between control vertices in array.
    m_nurbsknot : np.ndarray
        NurbsKnot vector array.
    m_cv : np.ndarray
        Control vertex data array (homogeneous if rational).
    """
    

    # ═══════════════════════════════════════════════════════════════════════════
    # Static Factory Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def create(periodic: bool, degree: int, points: list[Point], 
               dimension: int = 3, nurbsknot_delta: float = 1.0) -> 'NurbsCurve':
        """Create NURBS curve from points.

        Parameters
        ----------
        periodic : bool
            If True, creates a periodic curve; otherwise clamped.
        degree : int
            The degree of the curve.
        points : list[Point]
            Control points for the curve.
        dimension : int, optional
            Dimension of the curve. Defaults to 3.
        nurbsknot_delta : float, optional
            Spacing between nurbsknots. Defaults to 1.0.

        Returns
        -------
        NurbsCurve
            The created NURBS curve.
        """
        curve = NurbsCurve()
        if periodic:
            curve.create_periodic_uniform(dimension, degree + 1, points, nurbsknot_delta)
        else:
            curve.create_clamped_uniform(dimension, degree + 1, points, nurbsknot_delta)
        if curve.is_valid():
            # A degree-1 curve is a polyline: its arc length is the exact sum of segment lengths
            # (plus the closing segment when periodic). Computing it directly avoids the general
            # quadrature length(), which dominated every polyline build (lift / mesh / split).
            if degree == 1:
                np_ = len(points)
                def _seg(a, b):
                    dx = points[b][0]-points[a][0]; dy = points[b][1]-points[a][1]; dz = points[b][2]-points[a][2]
                    return (dx*dx + dy*dy + dz*dz) ** 0.5
                L = sum(_seg(i - 1, i) for i in range(1, np_))
                if periodic and np_ > 1:
                    L += _seg(np_ - 1, 0)
            else:
                L = curve.length()
            if L > 0.0:
                curve.set_domain(0.0, L)
        return curve

    @staticmethod
    def create_interpolated(points: list["Point"], parameterization: CurveNurbsKnotStyle = CurveNurbsKnotStyle.Chord,
                            end_condition: CurveInterpStyle = CurveInterpStyle.Rhino) -> "NurbsCurve":
        # parameterization maps to Rhino's CurveKnotStyle: Uniform/Chord/ChordSquareRoot
        # (centripetal). Rhino's CreateInterpolatedCurve(points, degree) API defaults to Uniform;
        # the InterpCrv command commonly uses Chord. Pass the style explicitly to match Rhino.
        # end_condition selects the boundary tangent rule: Rhino (Bessel, default) or
        # Occt (cubic Lagrange derivative, reproduces OCCT GeomAPI_Interpolate exactly).
        n = len(points)
        if n < 2:
            return NurbsCurve()
        dim = 3
        degree = 3
        order = degree + 1

        periodic = parameterization in (CurveNurbsKnotStyle.UniformPeriodic,
                                        CurveNurbsKnotStyle.ChordPeriodic,
                                        CurveNurbsKnotStyle.ChordSquareRootPeriodic)

        if periodic and n < 3:
            return NurbsCurve()

        # Two points: Rhino emits a degree-1 line (2 CVs), not a cubic.
        if n == 2 and not periodic:
            return NurbsCurve.create(False, 1, list(points))

        def pdist(a, b):
            dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
            return math.sqrt(dx*dx + dy*dy + dz*dz)

        if periodic:
            cv_count = n + 3
            kc = cv_count + order - 2

            base_map = {CurveNurbsKnotStyle.UniformPeriodic: CurveNurbsKnotStyle.Uniform,
                        CurveNurbsKnotStyle.ChordSquareRootPeriodic: CurveNurbsKnotStyle.ChordSquareRoot}
            base_style = base_map.get(parameterization, CurveNurbsKnotStyle.Chord)

            params = [0.0] * (n + 1)
            if base_style == CurveNurbsKnotStyle.Uniform:
                for i in range(1, n + 1):
                    params[i] = float(i)
            else:
                for i in range(1, n):
                    d = pdist(points[i-1], points[i])
                    if base_style == CurveNurbsKnotStyle.ChordSquareRoot:
                        d = math.sqrt(d)
                    params[i] = params[i-1] + d
                d_close = pdist(points[n-1], points[0])
                if base_style == CurveNurbsKnotStyle.ChordSquareRoot:
                    d_close = math.sqrt(d_close)
                params[n] = params[n-1] + d_close

            dmin, dmax = 1e300, 0.0
            for i in range(n):
                d = params[i+1] - params[i]
                if d < dmin: dmin = d
                if d > dmax: dmax = d
            if dmax <= 0.0 or dmax * 1.490116119385e-8 >= dmin:
                return NurbsCurve()

            nurbsknots_vec = [0.0] * kc
            for i in range(n + 1):
                nurbsknots_vec[i + 2] = params[i]
            nurbsknots_vec[cv_count]     = nurbsknots_vec[3] - nurbsknots_vec[2] + nurbsknots_vec[cv_count - 1]
            nurbsknots_vec[1]            = nurbsknots_vec[cv_count - 2] - nurbsknots_vec[cv_count - 1] + nurbsknots_vec[2]
            nurbsknots_vec[cv_count + 1] = nurbsknots_vec[4] - nurbsknots_vec[3] + nurbsknots_vec[cv_count]
            nurbsknots_vec[0]            = nurbsknots_vec[cv_count - 3] - nurbsknots_vec[cv_count - 2] + nurbsknots_vec[1]

            A = [[0.0] * n for _ in range(n)]
            rhs = [0.0] * (n * dim)

            for i in range(n):
                basis = nurbsknot.eval_basis(order, nurbsknots_vec, i, params[i])
                c0 = i % n
                c1 = (i + 1) % n
                c2 = (i + 2) % n
                A[i][c0] += basis[0]
                A[i][c1] += basis[1]
                A[i][c2] += basis[2]
                for d in range(dim):
                    rhs[i * dim + d] = points[i][d]

            cv = [0.0] * (n * dim)
            for i in range(n):
                for d in range(dim):
                    cv[i * dim + d] = rhs[i * dim + d]

            for col in range(n):
                pivot = col
                for row in range(col + 1, n):
                    if abs(A[row][col]) > abs(A[pivot][col]):
                        pivot = row
                if pivot != col:
                    A[col], A[pivot] = A[pivot], A[col]
                    for d in range(dim):
                        cv[col*dim+d], cv[pivot*dim+d] = cv[pivot*dim+d], cv[col*dim+d]
                if abs(A[col][col]) < 1e-300:
                    return NurbsCurve()
                for row in range(col + 1, n):
                    factor = A[row][col] / A[col][col]
                    for j in range(col, n):
                        A[row][j] -= factor * A[col][j]
                    for d in range(dim):
                        cv[row*dim+d] -= factor * cv[col*dim+d]

            for i in range(n - 1, -1, -1):
                for d in range(dim):
                    s = cv[i*dim+d]
                    for j in range(i + 1, n):
                        s -= A[i][j] * cv[j*dim+d]
                    cv[i*dim+d] = s / A[i][i]

            curve = NurbsCurve(dimension=dim, is_rational=False, order=order, cv_count=cv_count)
            curve.m_nurbsknot = np.array(nurbsknots_vec, dtype=np.float64)
            for i in range(n):
                curve.set_cv(i, Point(cv[i*3], cv[i*3+1], cv[i*3+2]))
            curve.set_cv(n, curve.get_cv(0))
            curve.set_cv(n + 1, curve.get_cv(1))
            curve.set_cv(n + 2, curve.get_cv(2))
            return curve

        # Open interpolation
        cv_count = n + 2

        pts = np.zeros((n, dim))
        for i in range(n):
            pts[i, 0] = points[i][0]
            pts[i, 1] = points[i][1]
            pts[i, 2] = points[i][2]

        params = nurbsknot.compute_parameters(pts, parameterization)
        nurbsknots_vec = nurbsknot.build_interp_nurbsknots(params, degree)
        kc = len(nurbsknots_vec)

        def estimate_tangent(i0, i1, i2):
            d01 = pdist(points[i0], points[i1])
            d21 = pdist(points[i2], points[i1])
            if d01 + d21 < 1e-300:
                return Vector(0, 0, 0)
            s = d01 / (d01 + d21)
            t = 1.0 - s
            denom = 2.0 * s * t
            if denom < 1e-16:
                dx = points[i1][0] - points[i0][0]
                dy = points[i1][1] - points[i0][1]
                dz = points[i1][2] - points[i0][2]
                l = math.sqrt(dx*dx + dy*dy + dz*dz)
                return Vector(dx/l, dy/l, dz/l) if l > 0 else Vector(0, 0, 0)
            cvx = (-t*t*points[i0][0] + points[i1][0] - s*s*points[i2][0]) / denom
            cvy = (-t*t*points[i0][1] + points[i1][1] - s*s*points[i2][1]) / denom
            cvz = (-t*t*points[i0][2] + points[i1][2] - s*s*points[i2][2]) / denom
            dx = cvx - points[i0][0]
            dy = cvy - points[i0][1]
            dz = cvz - points[i0][2]
            l = math.sqrt(dx*dx + dy*dy + dz*dz)
            return Vector(dx/l, dy/l, dz/l) if l > 0 else Vector(0, 0, 0)

        # Un-normalized derivative of the cubic (or quadratic, when n==3) Lagrange
        # polynomial through `m` consecutive points, evaluated at parameter t.
        # Reproduces OCCT GeomAPI_Interpolate::BuildTangents (PLib::EvalLagrange).
        def lagrange_tangent(i0, m, t):
            res = [0.0, 0.0, 0.0]
            for j in range(m):
                uj = params[i0 + j]
                dsum = 0.0
                for i in range(m):
                    if i == j:
                        continue
                    term = 1.0 / (uj - params[i0 + i])
                    for k in range(m):
                        if k == j or k == i:
                            continue
                        term *= (t - params[i0 + k]) / (uj - params[i0 + k])
                    dsum += term
                Pj = points[i0 + j]
                for d in range(3):
                    res[d] += Pj[d] * dsum
            return Vector(res[0], res[1], res[2])

        if end_condition == CurveInterpStyle.Occt and n >= 3:
            # OCCT mode: un-normalized Lagrange derivative at the endpoints. The
            # derivative-constraint poles satisfy C'(u0) = 3/(params[1]-params[0])*(P1-P0),
            # so P1 = P0 + (params[1]-params[0])/3 * tan_start (symmetric at the end).
            deg_t = 2 if n == 3 else 3
            tan_start = lagrange_tangent(0, deg_t + 1, params[0])
            tan_end = lagrange_tangent(n - 1 - deg_t, deg_t + 1, params[n - 1])
            s0 = (params[1] - params[0]) / 3.0
            s1 = -(params[n-1] - params[n-2]) / 3.0
        elif n >= 3:
            tan_start = estimate_tangent(0, 1, 2)
            end_raw = estimate_tangent(n-1, n-2, n-3)
            tan_end = Vector(-end_raw[0], -end_raw[1], -end_raw[2])
            s0 = pdist(points[0], points[1]) / 3.0
            s1 = -pdist(points[n-1], points[n-2]) / 3.0
        else:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            dz = points[1][2] - points[0][2]
            l = math.sqrt(dx*dx + dy*dy + dz*dz)
            if l > 0:
                tan_start = Vector(dx/l, dy/l, dz/l)
                tan_end = tan_start
            else:
                tan_start = Vector(0, 0, 0)
                tan_end = Vector(0, 0, 0)
            s0 = pdist(points[0], points[1]) / 3.0
            s1 = -pdist(points[n-1], points[n-2]) / 3.0

        cv = [0.0] * (cv_count * dim)
        for d in range(dim):
            cv[d] = points[0][d]
        for d in range(dim):
            cv[dim + d] = points[0][d] + s0 * tan_start[d]
        for i in range(1, n-1):
            for d in range(dim):
                cv[(i+1) * dim + d] = points[i][d]
        for d in range(dim):
            cv[n * dim + d] = points[n-1][d] + s1 * tan_end[d]
        for d in range(dim):
            cv[(n+1) * dim + d] = points[n-1][d]

        sys_n = n
        lower = [0.0] * sys_n
        diag_arr = [0.0] * sys_n
        upper = [0.0] * sys_n
        rhs = [0.0] * (sys_n * dim)

        diag_arr[0] = 1.0
        for d in range(dim):
            rhs[d] = cv[dim + d]

        for i in range(1, n-1):
            basis = nurbsknot.eval_basis(order, nurbsknots_vec, i, params[i])
            lower[i] = basis[0]
            diag_arr[i] = basis[1]
            upper[i] = basis[2]
            for d in range(dim):
                rhs[i * dim + d] = points[i][d]

        diag_arr[n-1] = 1.0
        for d in range(dim):
            rhs[(n-1) * dim + d] = cv[n * dim + d]

        solution = nurbsknot.solve_tridiagonal(dim, sys_n, lower, diag_arr, upper, rhs)
        if solution is None:
            return NurbsCurve()

        for i in range(sys_n):
            for d in range(dim):
                cv[(i+1) * dim + d] = solution[i * dim + d]

        curve = NurbsCurve(dimension=dim, is_rational=False, order=order, cv_count=cv_count)
        curve.m_nurbsknot = np.array(nurbsknots_vec, dtype=np.float64)
        curve.m_cv = np.zeros(cv_count * dim, dtype=np.float64)
        for i in range(cv_count):
            curve.set_cv(i, Point(cv[i*3], cv[i*3+1], cv[i*3+2]))

        return curve

    @staticmethod
    def create_from_parameters(points: list["Point"], weights: list[float], knots: list[float], mults: list[int], degree: int, periodic: bool = False) -> "NurbsCurve":
        """Create a NURBS curve from explicit parameters (OCCT / compas_occt convention:
        distinct knots + per-knot multiplicities). Mirrors OCCNurbsCurve.from_parameters and
        underlies from_points / from_line / from_circle / from_ellipse. The internal (OpenNURBS)
        knot vector is the expanded full knot vector with first and last entries dropped; the
        domain becomes [knots[0], knots[-1]]."""
        n = len(points)
        order = degree + 1
        if n < order:
            return NurbsCurve()
        if len(weights) != n or len(knots) != len(mults) or len(knots) == 0:
            return NurbsCurve()
        if periodic:
            return NurbsCurve()  # periodic from_parameters not yet supported

        rational = any(abs(w - 1.0) > Tolerance.ZERO_TOLERANCE for w in weights)

        # Expand distinct knots by multiplicity into the full (OCCT-style) knot vector.
        full = []
        for v, m in zip(knots, mults):
            full.extend([float(v)] * int(m))

        kc = order + n - 2
        if len(full) != kc + 2:  # must equal n + order
            return NurbsCurve()

        dim = 3
        curve = NurbsCurve(dimension=dim, is_rational=rational, order=order, cv_count=n)
        curve.m_nurbsknot = np.array(full[1:kc + 1], dtype=np.float64)
        curve.m_cv = np.zeros(n * curve.m_cv_stride, dtype=np.float64)
        for i in range(n):
            if rational:
                w = weights[i]
                curve.set_cv_4d(i, points[i][0] * w, points[i][1] * w, points[i][2] * w, w)
            else:
                curve.set_cv(i, Point(points[i][0], points[i][1], points[i][2]))
        return curve

    @staticmethod
    def create_fitted(points: list["Point"], num_cvs: int, degree: int = 3, is_periodic: bool = False) -> "NurbsCurve":
        m = len(points)
        dim = 3
        order = degree + 1

        def pdist(a, b):
            dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
            return math.sqrt(dx*dx + dy*dy + dz*dz)

        if is_periodic:
            n = m
            if n >= 2 and pdist(points[0], points[n-1]) < 1e-10:
                n -= 1
            if n <= num_cvs or num_cvs < order:
                if n < 3:
                    return NurbsCurve()
                return NurbsCurve.create_interpolated(points[:n], nurbsknot.CurveNurbsKnotStyle.ChordPeriodic)

            cv_count = num_cvs + degree
            kc = cv_count + order - 2

            params = [0.0] * (n + 1)
            for i in range(1, n):
                params[i] = params[i-1] + pdist(points[i-1], points[i])
            params[n] = params[n-1] + pdist(points[n-1], points[0])
            T = params[n]
            if T < 1e-14:
                return NurbsCurve()

            ppts = []
            for i in range(n):
                ppts.extend([points[i][0], points[i][1], points[i][2]])
            nurbsknots_vec = nurbsknot.build_fitted_nurbsknots_periodic_adaptive(params, ppts, n, dim, num_cvs, degree)

            NtN = [[0.0] * num_cvs for _ in range(num_cvs)]
            NtQ = [0.0] * (num_cvs * dim)

            for k in range(n):
                span = nurbsknot.find_span(order, cv_count, nurbsknots_vec, params[k])
                basis = nurbsknot.eval_basis(order, nurbsknots_vec, span, params[k])
                for a in range(order):
                    ci = (span + a) % num_cvs
                    for d in range(dim):
                        NtQ[ci * dim + d] += basis[a] * points[k][d]
                    for b in range(order):
                        cj = (span + b) % num_cvs
                        NtN[ci][cj] += basis[a] * basis[b]

            cv = list(NtQ)

            for col in range(num_cvs):
                pivot = col
                for row in range(col + 1, num_cvs):
                    if abs(NtN[row][col]) > abs(NtN[pivot][col]):
                        pivot = row
                if pivot != col:
                    NtN[col], NtN[pivot] = NtN[pivot], NtN[col]
                    for d in range(dim):
                        cv[col*dim+d], cv[pivot*dim+d] = cv[pivot*dim+d], cv[col*dim+d]
                if abs(NtN[col][col]) < 1e-300:
                    return NurbsCurve()
                for row in range(col + 1, num_cvs):
                    factor = NtN[row][col] / NtN[col][col]
                    for j in range(col, num_cvs):
                        NtN[row][j] -= factor * NtN[col][j]
                    for d in range(dim):
                        cv[row*dim+d] -= factor * cv[col*dim+d]
            for i in range(num_cvs - 1, -1, -1):
                for d in range(dim):
                    s = cv[i*dim+d]
                    for j in range(i + 1, num_cvs):
                        s -= NtN[i][j] * cv[j*dim+d]
                    cv[i*dim+d] = s / NtN[i][i]

            curve = NurbsCurve(dimension=dim, is_rational=False, order=order, cv_count=cv_count)
            curve.m_nurbsknot = np.array(nurbsknots_vec, dtype=np.float64)
            curve.m_cv = np.zeros(cv_count * dim, dtype=np.float64)
            for i in range(num_cvs):
                curve.set_cv(i, Point(cv[i*3], cv[i*3+1], cv[i*3+2]))
            for i in range(degree):
                curve.set_cv(num_cvs + i, curve.get_cv(i))
            return curve

        # Open fitting
        if m <= num_cvs or num_cvs < order:
            return NurbsCurve.create_interpolated(points)

        pts = np.zeros((m, dim))
        for i in range(m):
            pts[i, 0] = points[i][0]
            pts[i, 1] = points[i][1]
            pts[i, 2] = points[i][2]

        params = nurbsknot.compute_parameters(pts, nurbsknot.CurveNurbsKnotStyle.Chord)
        flat_pts = []
        for i in range(m):
            flat_pts.extend([points[i][0], points[i][1], points[i][2]])
        nurbsknots_vec = nurbsknot.build_fitted_nurbsknots_adaptive(params, flat_pts, m, dim, num_cvs, degree)
        n = num_cvs - 1
        sys_n = num_cvs - 2
        bw = degree
        bw1 = bw + 1

        band = [0.0] * (sys_n * bw1)
        rhs = [0.0] * (sys_n * dim)

        for k in range(1, m - 1):
            span = nurbsknot.find_span(order, num_cvs, nurbsknots_vec, params[k])
            basis = nurbsknot.eval_basis(order, nurbsknots_vec, span, params[k])

            rk = [points[k][d] for d in range(dim)]
            for a in range(order):
                ci = span + a
                if ci == 0:
                    for d in range(dim):
                        rk[d] -= basis[a] * points[0][d]
                if ci == n:
                    for d in range(dim):
                        rk[d] -= basis[a] * points[m-1][d]

            for a in range(order):
                ci = span + a
                if ci < 1 or ci > n - 1:
                    continue
                ri = ci - 1
                for d in range(dim):
                    rhs[ri * dim + d] += basis[a] * rk[d]
                for b in range(a, order):
                    cj = span + b
                    if cj < 1 or cj > n - 1:
                        continue
                    rj = cj - 1
                    band[rj * bw1 + (rj - ri)] += basis[a] * basis[b]

        if not nurbsknot.solve_banded_spd(dim, sys_n, bw, band, rhs):
            return NurbsCurve.create_interpolated(points)

        curve = NurbsCurve(dimension=dim, is_rational=False, order=order, cv_count=num_cvs)
        curve.m_nurbsknot = np.array(nurbsknots_vec, dtype=np.float64)
        curve.m_cv = np.zeros(num_cvs * dim, dtype=np.float64)
        curve.set_cv(0, points[0])
        for i in range(sys_n):
            curve.set_cv(i + 1, Point(rhs[i*3], rhs[i*3+1], rhs[i*3+2]))
        curve.set_cv(n, points[m-1])
        return curve

    @staticmethod
    def join(curves: list["NurbsCurve"], tolerance: float | None = None) -> list["NurbsCurve"]:
        """Join curve segments into chains by endpoint matching.

        Segments are greedily chained (reversed as needed), made compatible
        (common degree, common rationality), and concatenated with C0
        continuity (junction nurbsknot at multiplicity = degree).

        Parameters
        ----------
        curves : list[NurbsCurve]
            Segments to join. Inputs are not modified.
        tolerance : float, optional
            Endpoint matching distance. Defaults to Tolerance.ZERO_TOLERANCE.

        Returns
        -------
        list[NurbsCurve]
            One curve per chain (singletons returned as duplicates).
        """
        tol = tolerance if tolerance is not None else Tolerance.ZERO_TOLERANCE
        segs = []
        for c in curves:
            if c is not None and c.is_valid():
                segs.append(c.duplicate())
        chains = []
        used = [False] * len(segs)
        for i in range(len(segs)):
            if used[i]:
                continue
            used[i] = True
            chain = [segs[i]]
            if not segs[i].is_closed():
                grown = True
                while grown:
                    grown = False
                    start = chain[0].point_at_start()
                    end = chain[-1].point_at_end()
                    for j in range(len(segs)):
                        if used[j] or segs[j].is_closed():
                            continue
                        s = segs[j].point_at_start()
                        e = segs[j].point_at_end()
                        if s.distance(end) <= tol:
                            chain.append(segs[j])
                        elif e.distance(end) <= tol:
                            segs[j].reverse()
                            chain.append(segs[j])
                        elif e.distance(start) <= tol:
                            chain.insert(0, segs[j])
                        elif s.distance(start) <= tol:
                            segs[j].reverse()
                            chain.insert(0, segs[j])
                        else:
                            continue
                        used[j] = True
                        grown = True
                        break
            chains.append(chain)
        result = []
        for chain in chains:
            if len(chain) == 1:
                result.append(chain[0])
                continue
            rational = False
            max_degree = 1
            for c in chain:
                if c.is_rational():
                    rational = True
                if c.degree() > max_degree:
                    max_degree = c.degree()
            for c in chain:
                if rational:
                    c.make_rational()
                c.clamp_end(2)
                c.increase_degree(max_degree)
            joined = chain[0]
            for c in chain[1:]:
                stride = joined.m_cv_stride
                cvdim = joined.cv_size()
                _, a1 = joined.domain()
                s0, s1 = c.domain()
                c.set_domain(a1, a1 + (s1 - s0))
                if rational:
                    w_end = joined.weight(joined.m_cv_count - 1)
                    w_start = c.weight(0)
                    if abs(w_start) > Tolerance.ZERO_TOLERANCE:
                        scale = w_end / w_start
                        for k in range(len(c.m_cv)):
                            c.m_cv[k] = c.m_cv[k] * scale
                joined_cv = np.asarray(joined.m_cv, dtype=np.float64)
                c_cv = np.asarray(c.m_cv, dtype=np.float64)
                last = (joined.m_cv_count - 1) * stride
                for k in range(cvdim):
                    joined_cv[last + k] = 0.5 * (joined_cv[last + k] + c_cv[k])
                joined.m_nurbsknot = np.concatenate([np.asarray(joined.m_nurbsknot, dtype=np.float64), np.asarray(c.m_nurbsknot, dtype=np.float64)[joined.m_order - 1:]])
                joined.m_cv = np.concatenate([joined_cv, c_cv[stride:]])
                joined.m_cv_count = joined.m_cv_count + c.m_cv_count - 1
            joined._invalidate_rmf_cache()
            result.append(joined)
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors & Destructor
    # ═══════════════════════════════════════════════════════════════════════════

    def __init__(self, dimension: int = 3, is_rational: bool = False,
                 order: int = 4, cv_count: int = 0):
        self._guid = None
        self.name = "my_nurbscurve"
        self.width = 1.0
        self.pointcolors = []
        self.linecolors = []

        self.m_dim = dimension
        self.m_is_rat = 1 if is_rational else 0
        self.m_order = order
        self.m_cv_count = cv_count
        self.m_cv_stride = (dimension + 1) if is_rational else dimension

        if cv_count > 0 and order > 0 and cv_count >= order:
            nurbsknot_count = order + cv_count - 2
            self.m_nurbsknot = np.zeros(nurbsknot_count, dtype=np.float64)
            self.m_cv = np.zeros(cv_count * self.m_cv_stride, dtype=np.float64)
        else:
            self.m_nurbsknot = np.array([], dtype=np.float64)
            self.m_cv = np.array([], dtype=np.float64)

        self._rmf_cache = None

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def refresh_guid(self) -> None:
        """Clear the guid so a FRESH one mints lazily on next read — the duplicate/copy enabler."""
        self._guid = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, NurbsCurve):
            return False
        if self.m_dim != other.m_dim or self.m_is_rat != other.m_is_rat:
            return False
        if self.m_order != other.m_order or self.m_cv_count != other.m_cv_count:
            return False
        if self.m_cv_stride != other.m_cv_stride:
            return False
        if self.name != other.name:
            return False
        if abs(self.width - other.width) > Tolerance.ZERO_TOLERANCE:
            return False
        if self.pointcolors != other.pointcolors:
            return False
        if self.linecolors != other.linecolors:
            return False
        if len(self.m_nurbsknot) != len(other.m_nurbsknot):
            return False
        for i in range(len(self.m_nurbsknot)):
            if abs(float(self.m_nurbsknot[i]) - float(other.m_nurbsknot[i])) > Tolerance.ZERO_TOLERANCE:
                return False
        if len(self.m_cv) != len(other.m_cv):
            return False
        for i in range(len(self.m_cv)):
            if abs(float(self.m_cv[i]) - float(other.m_cv[i])) > Tolerance.ZERO_TOLERANCE:
                return False
        return True

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def duplicate(self) -> "NurbsCurve":
        """Create a duplicate with a new GUID.

        Returns
        -------
        NurbsCurve
            A copy of the curve with a new GUID.
        """
        import copy
        import uuid
        new_curve = copy.deepcopy(self)
        new_curve.guid = str(uuid.uuid4())
        return new_curve


    # ═══════════════════════════════════════════════════════════════════════════
    # Initialization & Creation
    # ═══════════════════════════════════════════════════════════════════════════

    def initialize(self) -> None:
        """Initialize all fields to zero/empty.
        
        Returns
        -------
        None
        """
        self.m_dim = 0
        self.m_is_rat = 0
        self.m_order = 0
        self.m_cv_count = 0
        self.m_cv_stride = 0
        self.m_nurbsknot = np.array([], dtype=np.float64)
        self.m_cv = np.array([], dtype=np.float64)

    def create_curve(self, dimension: int, is_rational: bool, 
                    order: int, cv_count: int) -> bool:
        """Create NURBS curve with specified parameters"""
        if dimension < 1 or order < 2 or cv_count < order:
            return False
        
        self.m_dim = dimension
        self.m_is_rat = 1 if is_rational else 0
        self.m_order = order
        self.m_cv_count = cv_count
        self.m_cv_stride = (dimension + 1) if is_rational else dimension
        
        # Allocate arrays
        nurbsknot_count = order + cv_count - 2
        self.m_nurbsknot = np.zeros(nurbsknot_count, dtype=np.float64)
        self.m_cv = np.zeros(cv_count * self.m_cv_stride, dtype=np.float64)
        
        # Set weights to 1.0 if rational
        if is_rational:
            for i in range(cv_count):
                self.m_cv[i * self.m_cv_stride + dimension] = 1.0
        
        return True

    def create_clamped_uniform(self, dimension: int, order: int, 
                              points: list[Point], nurbsknot_delta: float = 1.0) -> bool:
        """Create clamped uniform NURBS curve from control points"""
        if not points or len(points) < order:
            return False
        
        if not self.create_curve(dimension, False, order, len(points)):
            return False
        
        # Set control points
        for i, pt in enumerate(points):
            self.set_cv(i, pt)
        
        self.m_nurbsknot = nurbsknot.make_clamped_uniform(self.m_order, self.m_cv_count, nurbsknot_delta)
        
        return True

    def create_periodic_uniform(self, dimension: int, order: int,
                               points: list[Point], nurbsknot_delta: float = 1.0) -> bool:
        """Create periodic uniform NURBS curve from control points"""
        point_count = len(points) if points else 0
        if point_count < order:
            return False

        cv_count = point_count + order - 1
        if not self.create_curve(dimension, False, order, cv_count):
            return False

        for i in range(point_count):
            self.set_cv(i, points[i])
        for i in range(order - 1):
            self.set_cv(point_count + i, points[i % point_count])

        self.m_nurbsknot = nurbsknot.make_periodic_uniform(self.m_order, self.m_cv_count, nurbsknot_delta)

        return True

    def destroy(self) -> None:
        """Deallocate all memory and reset to empty state"""
        self.initialize()


    # ═══════════════════════════════════════════════════════════════════════════
    # Boolean Queries
    # ═══════════════════════════════════════════════════════════════════════════

    def is_valid(self) -> bool:
        """Check if NURBS curve is valid"""
        if self.m_dim < 1:
            return False
        if self.m_order < 2:
            return False
        if self.m_cv_count < self.m_order:
            return False
        if len(self.m_nurbsknot) != self.m_order + self.m_cv_count - 2:
            return False
        if len(self.m_cv) < self.m_cv_count * self.m_cv_stride:
            return False
        
        # Check nurbsknot vector is non-decreasing
        for i in range(len(self.m_nurbsknot) - 1):
            if self.m_nurbsknot[i] > self.m_nurbsknot[i + 1] + Tolerance.ZERO_TOLERANCE:
                return False
        
        return True

    def is_rational(self) -> bool:
        return self.m_is_rat != 0

    def is_closed(self) -> bool:
        """Check if curve is closed"""
        if not self.is_valid():
            return False

        p_start = self.point_at_start()
        p_end = self.point_at_end()
        return p_start.distance(p_end) < Tolerance.ZERO_TOLERANCE

    def is_periodic(self) -> bool:
        """Check if curve is periodic"""
        if not self.is_valid():
            return False

        # Check if nurbsknots and CVs wrap around
        if not self.is_closed():
            return False

        # Check if first order-1 CVs match last order-1 CVs
        for i in range(self.m_order - 1):
            p1 = self.get_cv(i)
            p2 = self.get_cv(self.m_cv_count - self.m_order + 1 + i)
            if p1 and p2 and p1.distance(p2) > Tolerance.ZERO_TOLERANCE:
                return False

        return True

    def is_linear(self, tolerance: float | None = None) -> bool:
        """Check if curve is a straight line.

        Parameters
        ----------
        tolerance : float, optional
            Maximum deviation from line. Defaults to Tolerance.ZERO_TOLERANCE.

        Returns
        -------
        bool
            True if curve is linear within tolerance.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE

        if not self.is_valid() or self.m_cv_count < 2:
            return False

        p_start = self.point_at_start()
        p_end = self.point_at_end()
        line_length = p_start.distance(p_end)

        if line_length < tolerance:
            return True

        num_samples = max(20, self.m_cv_count * 2)
        t0, t1 = self.domain()

        sample_params = np.linspace(t0, t1, num_samples + 1)[1:-1]
        pts = self._batch_point_at(sample_params)

        v = np.array([p_end.x - p_start.x, p_end.y - p_start.y, p_end.z - p_start.z])
        c2 = np.dot(v, v)

        if c2 <= Tolerance.ZERO_TOLERANCE:
            return True

        origin = np.array([p_start.x, p_start.y, p_start.z])
        w = pts - origin[None, :]
        c1 = w @ v
        b = c1 / c2
        projected = origin[None, :] + b[:, None] * v[None, :]
        diffs = pts - projected
        dists_sq = np.sum(diffs * diffs, axis=1)

        return bool(np.all(dists_sq <= tolerance * tolerance))

    def is_planar(self, tolerance: float | None = None) -> bool:
        """Check if curve lies in a plane.

        Parameters
        ----------
        tolerance : float, optional
            Maximum deviation from plane. Defaults to Tolerance.ZERO_TOLERANCE.

        Returns
        -------
        bool
            True if curve is planar within tolerance.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE

        if not self.is_valid() or self.m_cv_count < 3:
            return True

        p0 = self.get_cv(0)
        p1 = self.get_cv(self.m_cv_count // 2)
        p2 = self.get_cv(self.m_cv_count - 1)

        if not (p0 and p1 and p2):
            return False

        v1 = Vector(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
        v2 = Vector(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z)
        normal = v1.cross(v2)

        if normal.magnitude() < Tolerance.ZERO_TOLERANCE:
            return True

        normal = normal.normalized()
        n = np.array([normal[0], normal[1], normal[2]])
        origin = np.array([p0.x, p0.y, p0.z])

        cv = self.m_cv
        stride = self.m_cv_stride
        dim = self.m_dim
        pts = np.empty((self.m_cv_count, 3))
        for i in range(self.m_cv_count):
            base = i * stride
            pts[i, 0] = cv[base]
            pts[i, 1] = cv[base + 1] if dim > 1 else 0.0
            pts[i, 2] = cv[base + 2] if dim > 2 else 0.0
            if self.m_is_rat:
                wi = cv[base + dim]
                if abs(wi) > 1e-10:
                    pts[i] /= wi

        diffs = pts - origin[None, :]
        dists = np.abs(diffs @ n)
        return bool(np.all(dists <= tolerance))

    def is_arc(self, tolerance: float | None = None) -> bool:
        """Check if curve is an arc.

        Parameters
        ----------
        tolerance : float, optional
            Tolerance for arc test. Defaults to Tolerance.ZERO_TOLERANCE.

        Returns
        -------
        bool
            True if curve is an arc.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE
        if not self.is_valid():
            return False
        if self.m_dim != 2 and self.m_dim != 3:
            return False
        if self.m_order < 3:
            return False
        if self.is_linear(tolerance):
            return False
        if not self.is_planar(tolerance):
            return False

        t0, t1 = self.domain()
        tmid = (t0 + t1) * 0.5
        p0 = self.point_at(t0)
        p1 = self.point_at(tmid)
        p2 = self.point_at(t1)

        from session_py.vector import Vector as Vec
        d1 = Vec(p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
        d2 = Vec(p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
        normal = d1.cross(d2)
        if normal.magnitude() < Tolerance.ZERO_TOLERANCE:
            return False
        normal = normal.normalized()

        from session_py.point import Point as Pt
        m1 = Pt((p0[0]+p1[0])*0.5, (p0[1]+p1[1])*0.5, (p0[2]+p1[2])*0.5)
        m2 = Pt((p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5, (p1[2]+p2[2])*0.5)
        perp1 = d1.cross(normal).normalized()
        perp2 = d2.cross(normal).normalized()

        denom = perp1[0] * perp2[1] - perp1[1] * perp2[0]
        if abs(denom) < Tolerance.ZERO_TOLERANCE:
            denom = perp1[0] * perp2[2] - perp1[2] * perp2[0]
        if abs(denom) < Tolerance.ZERO_TOLERANCE:
            return False

        dx = m2[0] - m1[0]
        dy = m2[1] - m1[1]
        s = (dx * perp2[1] - dy * perp2[0]) / denom
        center = Pt(m1[0]+s*perp1[0], m1[1]+s*perp1[1], m1[2]+s*perp1[2])
        radius = center.distance(p0)
        if radius < Tolerance.ZERO_TOLERANCE:
            return False

        samples_per_span = max(2 * self.degree() + 1, 4)
        num_samples = self.span_count() * samples_per_span
        for i in range(num_samples + 1):
            t = t0 + (t1 - t0) * i / num_samples
            pt = self.point_at(t)
            if abs(pt.distance(center) - radius) > tolerance:
                return False
        return True

    def is_in_plane(self, test_plane: Plane, tolerance: float | None = None) -> bool:
        """Check if curve lies in a specific plane.

        Parameters
        ----------
        test_plane : Plane
            The plane to test against.
        tolerance : float, optional
            Maximum deviation. Defaults to Tolerance.ZERO_TOLERANCE.

        Returns
        -------
        bool
            True if curve lies in the plane.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE

        if not self.is_valid():
            return False

        n = np.array([test_plane.z_axis[0], test_plane.z_axis[1], test_plane.z_axis[2]])
        origin = np.array([test_plane.origin.x, test_plane.origin.y, test_plane.origin.z])

        cv = self.m_cv
        stride = self.m_cv_stride
        dim = self.m_dim
        pts = np.empty((self.m_cv_count, 3))
        for i in range(self.m_cv_count):
            base = i * stride
            pts[i, 0] = cv[base]
            pts[i, 1] = cv[base + 1] if dim > 1 else 0.0
            pts[i, 2] = cv[base + 2] if dim > 2 else 0.0
            if self.m_is_rat:
                wi = cv[base + dim]
                if abs(wi) > 1e-10:
                    pts[i] /= wi

        diffs = pts - origin[None, :]
        dists = np.abs(diffs @ n)
        return bool(np.all(dists <= tolerance))

    def is_natural(self, end: int = 2) -> bool:
        """Test if curve has natural end (zero 2nd derivative).

        Parameters
        ----------
        end : int, optional
            0 for start, 1 for end, 2 for both. Defaults to 2.

        Returns
        -------
        bool
            True if has natural end.
        """
        if not self.is_valid():
            return False

        tol_factor = 1e-8
        t0, t1 = self.domain()

        # Check start (pass=0) and/or end (pass=1)
        start_pass = 0 if (end == 0 or end == 2) else 1
        end_pass = 2 if (end == 1 or end == 2) else 1

        for pass_idx in range(start_pass, end_pass):
            t = t0 if pass_idx == 0 else t1

            # Evaluate 2nd derivative
            derivs = self.evaluate(t, 2)
            if len(derivs) < 3:
                return False

            d2 = derivs[2]
            d2_len = d2.magnitude()

            # Get control polygon length for tolerance
            if pass_idx == 0:
                cv0 = self.get_cv(0)
                cv2 = self.get_cv(min(2, self.m_cv_count - 1))
            else:
                cv0 = self.get_cv(self.m_cv_count - 1)
                cv2 = self.get_cv(max(0, self.m_cv_count - 3))

            tol = cv0.distance(cv2) * tol_factor

            if d2_len > tol:
                return False

        return True

    def is_polyline(self) -> tuple[bool, list[Point], list[float]]:
        """Check if curve can be represented as a polyline.

        Returns
        -------
        tuple[bool, list[Point], list[float]]
            (is_polyline, points, parameters) or (False, [], []).
        """
        if not self.is_valid():
            return False, [], []

        # Check if curve is linear
        if self.is_linear():
            points = [self.point_at_start(), self.point_at_end()]
            t0, t1 = self.domain()
            params = [t0, t1]
            return True, points, params

        return False, [], []

    def is_singular(self) -> bool:
        """Check if entire curve is singular (collapsed to a point).

        Returns
        -------
        bool
            True if curve is singular.
        """
        if not self.is_valid():
            return False

        p_first = self.point_at_start()

        # Check if all sample points are at same location
        t0, t1 = self.domain()
        num_samples = max(10, self.m_cv_count)
        dt = (t1 - t0) / num_samples

        for i in range(1, num_samples + 1):
            t = t0 + i * dt
            p = self.point_at(t)
            if p_first.distance(p) > Tolerance.ZERO_TOLERANCE:
                return False

        return True

    def is_duplicate(self, other: "NurbsCurve", ignore_parameterization: bool = False, tolerance: float | None = None) -> bool:
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE
        if not self.is_valid() or not other.is_valid():
            return False
        if self.m_dim != other.m_dim:
            return False
        if self.m_is_rat != other.m_is_rat:
            return False
        if self.m_order != other.m_order:
            return False
        if self.m_cv_count != other.m_cv_count:
            return False
        for i in range(self.m_cv_count):
            p1 = self.get_cv(i)
            p2 = other.get_cv(i)
            if p1.distance(p2) > tolerance:
                return False
            if self.m_is_rat:
                if abs(self.weight(i) - other.weight(i)) > tolerance:
                    return False
        if not ignore_parameterization:
            for i in range(self.nurbsknot_count()):
                if abs(self.m_nurbsknot[i] - other.m_nurbsknot[i]) > tolerance:
                    return False
        return True

    def is_continuous(self, continuity_type: int, t: float) -> bool:
        if not self.is_valid():
            return False
        d0, d1 = self.domain()
        if t < d0 or t > d1:
            return False
        at_nurbsknot = False
        nurbsknot_idx = 0
        for i in range(self.nurbsknot_count()):
            if abs(self.m_nurbsknot[i] - t) < Tolerance.ZERO_TOLERANCE:
                at_nurbsknot = True
                nurbsknot_idx = i
                break
        if not at_nurbsknot:
            return True
        mult = self.nurbsknot_multiplicity(nurbsknot_idx)
        if continuity_type == 0:
            return mult < self.m_order
        elif continuity_type == 1:
            return mult < self.m_order - 1
        elif continuity_type == 2:
            return mult < self.m_order - 2
        else:
            return mult < self.m_order - 1

    def is_valid_nurbsknot_vector(self) -> bool:
        """Check if nurbsknot vector is valid"""
        if len(self.m_nurbsknot) != self.nurbsknot_count():
            return False

        for i in range(len(self.m_nurbsknot) - 1):
            if self.m_nurbsknot[i] > self.m_nurbsknot[i + 1] + Tolerance.ZERO_TOLERANCE:
                return False

        return True

    def is_clamped(self, end: int = 2) -> bool:
        """Check if nurbsknot vector is clamped at ends.

        Parameters
        ----------
        end : int, optional
            0 for start, 1 for end, 2 for both. Defaults to 2.

        Returns
        -------
        bool
            True if clamped at specified end(s).
        """
        if not self.is_valid():
            return False

        # Use nurbsknot module function
        return nurbsknot.is_clamped(self.m_order, self.m_cv_count, self.m_nurbsknot, end)


    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════

    def dimension(self) -> int:
        return self.m_dim

    def order(self) -> int:
        return self.m_order

    def degree(self) -> int:
        return self.m_order - 1

    def cv_count(self) -> int:
        return self.m_cv_count

    def cv_size(self) -> int:
        """Size of each control vertex"""
        return (self.m_dim + 1) if self.m_is_rat else self.m_dim

    def nurbsknot_count(self) -> int:
        return self.m_order + self.m_cv_count - 2

    def span_count(self) -> int:
        return self.m_cv_count - self.m_order + 1

    def get_nurbsknots(self) -> np.ndarray:
        """Get all nurbsknot values"""
        return self.m_nurbsknot.copy()

    def nurbsknot_array(self) -> np.ndarray:
        """Get pointer to nurbsknot array"""
        return self.m_nurbsknot

    def cv_array(self) -> np.ndarray:
        """Get pointer to CV array"""
        return self.m_cv


    # ═══════════════════════════════════════════════════════════════════════════
    # Control Vertex Access
    # ═══════════════════════════════════════════════════════════════════════════

    def get_cv(self, cv_index: int) -> Point | None:
        """Get control point at index as Point (Euclidean coordinates)"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return None

        idx = cv_index * self.m_cv_stride
        x = self.m_cv[idx] if self.m_dim > 0 else 0.0
        y = self.m_cv[idx + 1] if self.m_dim > 1 else 0.0
        z = self.m_cv[idx + 2] if self.m_dim > 2 else 0.0
        if self.m_is_rat:
            w = self.m_cv[idx + self.m_dim]
            if abs(w) < 1e-14:
                return Point(0.0, 0.0, 0.0)
            return Point(x / w, y / w, z / w)
        return Point(x, y, z)

    def cv(self, cv_index: int) -> list[float] | None:
        """Get raw CV data at index (like C++ double* cv(int))"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return None
        idx = cv_index * self.m_cv_stride
        return list(self.m_cv[idx:idx + self.m_cv_stride])

    def set_cv(self, cv_index: int, point: Point) -> bool:
        """Set control point at index from Point"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return False

        idx = cv_index * self.m_cv_stride
        if self.m_dim > 0:
            self.m_cv[idx] = point.x
        if self.m_dim > 1:
            self.m_cv[idx + 1] = point.y
        if self.m_dim > 2:
            self.m_cv[idx + 2] = point.z

        if self.m_is_rat:
            w = self.m_cv[idx + self.m_dim]
            if self.m_dim > 0:
                self.m_cv[idx] *= w
            if self.m_dim > 1:
                self.m_cv[idx + 1] *= w
            if self.m_dim > 2:
                self.m_cv[idx + 2] *= w

        self._invalidate_rmf_cache()
        return True

    def get_cv_4d(self, cv_index: int) -> tuple[float, float, float, float] | None:
        """Get control point as homogeneous coordinates (x, y, z, w)"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return None
        
        idx = cv_index * self.m_cv_stride
        x = self.m_cv[idx] if self.m_dim > 0 else 0.0
        y = self.m_cv[idx + 1] if self.m_dim > 1 else 0.0
        z = self.m_cv[idx + 2] if self.m_dim > 2 else 0.0
        w = self.m_cv[idx + self.m_dim] if self.m_is_rat else 1.0
        
        return (x, y, z, w)

    def set_cv_4d(self, cv_index: int, x: float, y: float, z: float, w: float) -> bool:
        """Set control point from homogeneous coordinates"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return False

        # Make rational if w != 1.0 (matches C++ implementation)
        if not self.m_is_rat and w != 1.0:
            self.make_rational()

        idx = cv_index * self.m_cv_stride
        if self.m_dim > 0:
            self.m_cv[idx] = x
        if self.m_dim > 1:
            self.m_cv[idx + 1] = y
        if self.m_dim > 2:
            self.m_cv[idx + 2] = z
        if self.m_is_rat:
            self.m_cv[idx + self.m_dim] = w

        self._invalidate_rmf_cache()
        return True

    def weight(self, cv_index: int) -> float:
        """Get weight at control vertex index"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return 1.0
        
        if not self.m_is_rat:
            return 1.0
        
        idx = cv_index * self.m_cv_stride
        return self.m_cv[idx + self.m_dim]

    def set_weight(self, cv_index: int, weight: float) -> bool:
        """Set weight at control vertex index"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return False

        if not self.m_is_rat:
            if abs(weight - 1.0) > Tolerance.ZERO_TOLERANCE:
                self.make_rational()

        if self.m_is_rat:
            idx = cv_index * self.m_cv_stride
            self.m_cv[idx + self.m_dim] = weight

        self._invalidate_rmf_cache()
        return True


    # ═══════════════════════════════════════════════════════════════════════════
    # NurbsKnot Access
    # ═══════════════════════════════════════════════════════════════════════════

    def nurbsknot(self, nurbsknot_index: int) -> float:
        """Get nurbsknot value at index"""
        if nurbsknot_index < 0 or nurbsknot_index >= len(self.m_nurbsknot):
            return 0.0
        return self.m_nurbsknot[nurbsknot_index]

    def set_nurbsknot(self, nurbsknot_index: int, nurbsknot_value: float) -> bool:
        """Set nurbsknot value at index"""
        if nurbsknot_index < 0 or nurbsknot_index >= len(self.m_nurbsknot):
            return False
        self.m_nurbsknot[nurbsknot_index] = nurbsknot_value
        self._invalidate_rmf_cache()
        return True

    def nurbsknot_multiplicity(self, nurbsknot_index: int) -> int:
        """Get nurbsknot multiplicity at index"""
        if nurbsknot_index < 0 or nurbsknot_index >= len(self.m_nurbsknot):
            return 0
        
        nurbsknot_value = self.m_nurbsknot[nurbsknot_index]
        mult = 1
        
        # Count after
        for i in range(nurbsknot_index + 1, len(self.m_nurbsknot)):
            if abs(self.m_nurbsknot[i] - nurbsknot_value) < Tolerance.ZERO_TOLERANCE:
                mult += 1
            else:
                break
        
        # Count before
        for i in range(nurbsknot_index - 1, -1, -1):
            if abs(self.m_nurbsknot[i] - nurbsknot_value) < Tolerance.ZERO_TOLERANCE:
                mult += 1
            else:
                break
        
        return mult

    def superfluous_nurbsknot(self, end: int) -> float:
        """Get superfluous nurbsknot value at end.

        Parameters
        ----------
        end : int
            0 for start, 1 for end.

        Returns
        -------
        float
            The superfluous nurbsknot value.
        """
        if not self.is_valid():
            return 0.0

        kc = self.nurbsknot_count()
        if end == 0:
            # First superfluous nurbsknot: reflect first nurbsknot across nurbsknot[order-2]
            return 2.0 * self.m_nurbsknot[0] - self.m_nurbsknot[self.m_order - 2]
        else:
            # Last superfluous nurbsknot: reflect last nurbsknot across nurbsknot[cv_count-order]
            return 2.0 * self.m_nurbsknot[kc - 1] - self.m_nurbsknot[self.m_cv_count - self.m_order]

    def insert_nurbsknot(self, nurbsknot_value: float, nurbsknot_multiplicity: int = 1) -> bool:
        if not self.is_valid():
            return False

        p = self.degree()
        if nurbsknot_multiplicity < 1 or nurbsknot_multiplicity > p:
            return False

        d0, d1 = self.domain()
        if nurbsknot_value < d0 or nurbsknot_value > d1:
            return False

        # Handle end nurbsknots
        if nurbsknot_value == d0:
            if nurbsknot_multiplicity == p:
                return self.clamp_end(0)
            if nurbsknot_multiplicity == 1:
                return True
            return False
        if nurbsknot_value == d1:
            if nurbsknot_multiplicity == p:
                return self.clamp_end(1)
            if nurbsknot_multiplicity == 1:
                return True
            return False

        import numpy as np
        import math

        n = self.m_cv_count - 1
        full_nurbsknot_count = self.m_cv_count + self.m_order

        for insert_iter in range(nurbsknot_multiplicity):
            # Build full nurbsknot vector
            U = np.zeros(full_nurbsknot_count)
            U[0] = self.m_nurbsknot[0]
            for i in range(len(self.m_nurbsknot)):
                U[i + 1] = self.m_nurbsknot[i]
            U[full_nurbsknot_count - 1] = self.m_nurbsknot[-1]

            # Count current multiplicity
            tol = (abs(d0) + abs(d1) + abs(d1 - d0)) * math.sqrt(np.finfo(float).eps)
            mult = sum(1 for i in range(full_nurbsknot_count) if abs(U[i] - nurbsknot_value) <= tol)
            if mult >= nurbsknot_multiplicity:
                # Already at the requested multiplicity (e.g. splitting a degree-1 polyline
                # exactly at a vertex nurbsknot) -- nothing to insert, and that is success.
                return True
            if mult >= p:
                # Cannot increase multiplicity beyond degree for interior nurbsknots
                return False

            # Find span
            span = self._find_span(nurbsknot_value)
            k = span + self.m_order - 1

            # Single-nurbsknot insertion
            m_full = full_nurbsknot_count - 1
            new_full_nurbsknot_count = full_nurbsknot_count + 1
            new_cv_count = self.m_cv_count + 1

            U_new = np.zeros(new_full_nurbsknot_count)
            cv_new = np.zeros(new_cv_count * self.m_cv_stride)

            # Copy unaffected nurbsknots
            for i in range(k + 1):
                U_new[i] = U[i]
            U_new[k + 1] = nurbsknot_value
            for i in range(k + 1, m_full + 1):
                U_new[i + 1] = U[i]

            # Copy unaffected CVs before
            for i in range(k - p + 1):
                src = i * self.m_cv_stride
                dst = i * self.m_cv_stride
                cv_new[dst:dst + self.m_cv_stride] = self.m_cv[src:src + self.m_cv_stride]

            # Copy unaffected CVs after
            for i in range(k + 1, n + 2):
                src = (i - 1) * self.m_cv_stride
                dst = i * self.m_cv_stride
                cv_new[dst:dst + self.m_cv_stride] = self.m_cv[src:src + self.m_cv_stride]

            # Compute new CVs in affected region
            for i in range(k - p + 1, k + 1):
                denom = U[i + p] - U[i]
                alpha = (nurbsknot_value - U[i]) / denom if denom != 0.0 else 0.0

                src_prev = (i - 1) * self.m_cv_stride
                src_curr = i * self.m_cv_stride
                dst = i * self.m_cv_stride

                for d in range(self.m_cv_stride):
                    cv_new[dst + d] = (1.0 - alpha) * self.m_cv[src_prev + d] + alpha * self.m_cv[src_curr + d]

            # Update internal state
            self.m_cv_count = new_cv_count
            self.m_cv = cv_new

            new_compressed_nurbsknot_count = self.m_order + self.m_cv_count - 2
            self.m_nurbsknot = np.array([U_new[i + 1] for i in range(new_compressed_nurbsknot_count)])

            full_nurbsknot_count = new_full_nurbsknot_count
            n = self.m_cv_count - 1

        return True

    def greville_abcissa(self, cv_index: int) -> float:
        """Get Greville abcissa for a control point.

        Parameters
        ----------
        cv_index : int
            Index of the control vertex.

        Returns
        -------
        float
            The Greville abcissa parameter value.
        """
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return 0.0

        nurbsknot = self.m_nurbsknot[cv_index:]
        order = self.m_order

        if order <= 2 or nurbsknot[0] == nurbsknot[order - 2]:
            return float(nurbsknot[0])

        p = order - 1
        k0 = nurbsknot[0]
        k = nurbsknot[p // 2]
        k1 = nurbsknot[p - 1]
        tol = (k1 - k0) * 1.490116119385e-8

        g = sum(nurbsknot[i] for i in range(p)) / p

        if abs(2.0 * k - (k0 + k1)) <= tol and abs(g - k) <= (abs(g) * 1.490116119385e-8 + tol):
            g = k

        return float(g)

    def get_greville_abcissae(self) -> list[float]:
        """Get all Greville abcissae.
        
        Returns
        -------
        list[float]
            Greville parameters for all control vertices.
        """
        return [self.greville_abcissa(i) for i in range(self.m_cv_count)]


    # ═══════════════════════════════════════════════════════════════════════════
    # Domain & Parameterization
    # ═══════════════════════════════════════════════════════════════════════════

    def domain(self) -> tuple[float, float]:
        """Get curve domain [start_param, end_param]"""
        if not self.is_valid():
            return (0.0, 0.0)
        return (self.m_nurbsknot[self.m_order - 2], self.m_nurbsknot[self.m_cv_count - 1])

    def domain_start(self) -> float:
        """Get start of domain"""
        t0, _ = self.domain()
        return t0

    def domain_end(self) -> float:
        """Get end of domain"""
        _, t1 = self.domain()
        return t1

    def domain_middle(self) -> float:
        """Get middle of domain"""
        t0, t1 = self.domain()
        return (t0 + t1) * 0.5

    def set_domain(self, t0: float, t1: float) -> bool:
        """Set curve domain"""
        if not self.is_valid():
            return False
        if t0 >= t1:
            return False

        old_t0, old_t1 = self.domain()
        if abs(old_t1 - old_t0) < Tolerance.ZERO_TOLERANCE:
            return False

        clamped_start = (self.m_order >= 2 and
            abs(self.m_nurbsknot[0] - self.m_nurbsknot[self.m_order - 2]) < Tolerance.ZERO_TOLERANCE)
        clamped_end = (self.m_cv_count < len(self.m_nurbsknot) and
            abs(self.m_nurbsknot[-1] - self.m_nurbsknot[self.m_cv_count - 1]) < Tolerance.ZERO_TOLERANCE)

        scale = (t1 - t0) / (old_t1 - old_t0)
        for i in range(len(self.m_nurbsknot)):
            self.m_nurbsknot[i] = t0 + (self.m_nurbsknot[i] - old_t0) * scale

        if clamped_start:
            for i in range(self.m_order - 1):
                self.m_nurbsknot[i] = t0
        if clamped_end:
            for i in range(self.m_cv_count - 1, len(self.m_nurbsknot)):
                self.m_nurbsknot[i] = t1

        self._invalidate_rmf_cache()
        return True

    def get_span_vector(self) -> list[float]:
        """Get span (distinct nurbsknot intervals) values"""
        if not self.is_valid():
            return []
        
        spans = []
        for i in range(self.m_order - 2, self.m_cv_count):
            if i == self.m_order - 2 or abs(self.m_nurbsknot[i] - self.m_nurbsknot[i-1]) > Tolerance.ZERO_TOLERANCE:
                spans.append(self.m_nurbsknot[i])
        
        return spans

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometric Queries
    # ═══════════════════════════════════════════════════════════════════════════

    def get_next_discontinuity(self, continuity_type: int, t0: float, t1: float) -> tuple[bool, float]:
        if not self.is_valid():
            return False, 0.0
        d0, d1 = self.domain()
        t0 = max(t0, d0)
        t1 = min(t1, d1)
        if t0 >= t1:
            return False, 0.0
        for i in range(self.m_order - 1, self.m_cv_count - 1):
            t = float(self.m_nurbsknot[i])
            if t <= t0 or t >= t1:
                continue
            mult = self.nurbsknot_multiplicity(i)
            found = False
            if continuity_type == 0 and mult >= self.m_order:
                found = True
            elif continuity_type == 1 and mult >= self.m_order - 1:
                found = True
            elif continuity_type == 2 and mult >= self.m_order - 2:
                found = True
            elif continuity_type in (3, 4) and mult >= self.m_order - 1:
                found = True
            if found:
                return True, t
        return False, 0.0

    def _span_is_singular(self, span_index: int) -> bool:
        """Check if span is singular (collapsed to a point).

        Parameters
        ----------
        span_index : int
            Index of the span.

        Returns
        -------
        bool
            True if span is singular.
        """
        if not self.is_valid():
            return False

        spans = self.get_span_vector()
        if span_index < 0 or span_index >= len(spans) - 1:
            return False

        t0 = spans[span_index]
        t1 = spans[span_index + 1]

        p0 = self.point_at(t0)
        p1 = self.point_at(t1)

        return p0.distance(p1) < Tolerance.ZERO_TOLERANCE


    # ═══════════════════════════════════════════════════════════════════════════
    # Conversion Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def length(self) -> float:
        """Compute curve length using Gauss-Legendre quadrature"""
        if not self.is_valid():
            return 0.0

        GL_X = np.array([
            -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
            -0.4333953941292472, -0.1488743389816312,
             0.1488743389816312,  0.4333953941292472,  0.6794095682990244,
             0.8650633666889845,  0.9739065285171717
        ])
        GL_W = np.array([
            0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
            0.2692667193099963, 0.2955242247147529,
            0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
            0.1494513491505806, 0.0666713443086881
        ])

        # Count nurbsknot INTERVALS, not span_count(): a repeated interior nurbsknot makes
        # span_count() smaller than the interval count, and the trailing spans go unintegrated.
        n_spans = self.m_cv_count - self.m_order + 1
        SUBDIVISIONS = 4

        # Collect all GL sample params
        mids_list = []
        halfs_list = []
        for span in range(n_spans):
            span_a = self.m_nurbsknot[self.m_order - 2 + span]
            span_b = self.m_nurbsknot[self.m_order - 1 + span]
            if span_b <= span_a:
                continue
            span_width = (span_b - span_a) / SUBDIVISIONS
            for sub in range(SUBDIVISIONS):
                a = span_a + sub * span_width
                b = a + span_width
                mids_list.append((a + b) * 0.5)
                halfs_list.append((b - a) * 0.5)

        if not mids_list:
            return 0.0

        mids = np.array(mids_list)
        halfs = np.array(halfs_list)
        n_intervals = len(mids)

        # Shape (n_intervals, 10) -> flat
        all_params = (mids[:, None] + halfs[:, None] * GL_X[None, :]).ravel()

        # Batch evaluate first derivatives
        derivs = self._batch_evaluate_deriv1(all_params)
        speeds = np.sqrt(np.sum(derivs * derivs, axis=1))

        # Reshape, apply GL weights, sum
        speeds = speeds.reshape(n_intervals, 10)
        integrals = halfs * np.sum(GL_W[None, :] * speeds, axis=1)

        return float(np.sum(integrals))

    def to_polyline_adaptive(self, angle_tolerance: float = 0.1, 
                            min_edge_length: float = 0.0,
                            max_edge_length: float = 0.0) -> tuple[list[Point], list[float]]:
        """Convert curve to polyline with adaptive sampling (curvature-based).

        Parameters
        ----------
        angle_tolerance : float, optional
            Maximum angle between segments in radians. Defaults to 0.1.
        min_edge_length : float, optional
            Minimum distance between points. Defaults to 0.0 (auto).
        max_edge_length : float, optional
            Maximum distance between points. Defaults to 0.0 (auto).

        Returns
        -------
        tuple[list[Point], list[float]]
            Points and parameters.
        """
        if not self.is_valid():
            return [], []

        if angle_tolerance <= 0.0:
            angle_tolerance = 0.1

        t0, t1 = self.domain()
        curve_len = self.length()

        # Set default edge lengths if not specified (matches C++ implementation)
        if max_edge_length <= 0.0:
            max_edge_length = curve_len / 10.0
        if min_edge_length <= 0.0:
            min_edge_length = curve_len / 1000.0
        if min_edge_length > max_edge_length:
            min_edge_length = max_edge_length * 0.1

        # Collect (param, point) pairs using binary subdivision
        samples = [(t0, self.point_at(t0)), (t1, self.point_at(t1))]

        # Work queue: segments to potentially subdivide (ta, tb)
        work_queue = [(t0, t1)]

        max_iterations = 10000
        iterations = 0

        while work_queue and iterations < max_iterations:
            iterations += 1
            ta, tb = work_queue.pop()

            pa = self.point_at(ta)
            pb = self.point_at(tb)
            chord_length = pa.distance(pb)

            if chord_length < min_edge_length:
                continue

            tm = (ta + tb) * 0.5
            pm = self.point_at(tm)

            # Check deviation: distance from midpoint to chord
            chord = Vector(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z)
            to_mid = Vector(pm.x - pa.x, pm.y - pa.y, pm.z - pa.z)
            chord_len_sq = chord.dot(chord)
            deviation = 0.0

            if chord_len_sq > 1e-20:
                proj = to_mid.dot(chord) / chord_len_sq
                projected = Point(pa.x + proj * chord.x, pa.y + proj * chord.y, pa.z + proj * chord.z)
                deviation = pm.distance(projected)

            # Convert angle tolerance to approximate deviation tolerance
            # For small angles: deviation ≈ chord_length * sin(angle/2) ≈ chord_length * angle/2
            deviation_tolerance = chord_length * angle_tolerance * 0.5

            need_subdivide = (deviation > deviation_tolerance) or (chord_length > max_edge_length)

            if need_subdivide:
                samples.append((tm, pm))
                work_queue.append((ta, tm))
                work_queue.append((tm, tb))

        # Sort by parameter
        samples.sort(key=lambda x: x[0])

        # Extract results
        points = [p for _, p in samples]
        params = [t for t, _ in samples]

        return points, params

    def divide_by_count(self, count: int, include_endpoints: bool = True) -> tuple[list[Point], list[float]]:
        """Divide curve into uniform arc-length segments."""
        points = []
        params = []

        if not self.is_valid() or count < 2:
            return points, params

        t0, t1 = self.domain()
        dom_len = t1 - t0
        h = dom_len * 1e-8

        if self.m_order == 2 and not self.m_is_rat and self.m_cv_count >= 2:
            if include_endpoints:
                ts = np.linspace(t0, t1, count)
            else:
                step = dom_len / (count + 1)
                ts = t0 + step * np.arange(1, count + 1)
            pts_arr = self._batch_point_at(ts)
            points = [Point(pts_arr[i, 0], pts_arr[i, 1], pts_arr[i, 2]) for i in range(len(ts))]
            params = ts.tolist()
            return points, params

        GL_NODES = np.array([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640])
        GL_WEIGHTS = np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891])

        # Build arc-length table using batch evaluation
        n_samples = max(1000, count * 100)
        t_vals = np.linspace(t0, t1, n_samples + 1)
        mids = (t_vals[:-1] + t_vals[1:]) * 0.5
        halfs = (t_vals[1:] - t_vals[:-1]) * 0.5

        # GL sample parameters: shape (n_samples, 5) -> flat
        gl_params = mids[:, None] + halfs[:, None] * GL_NODES[None, :]
        all_params = gl_params.ravel()

        # Replicate original boundary logic for finite-difference derivatives
        n_all = len(all_params)
        tp = np.empty(n_all)
        tm = np.empty(n_all)
        dt = np.empty(n_all)
        near_start = all_params <= t0 + h
        near_end = all_params >= t1 - h
        mid_mask = ~near_start & ~near_end
        # Near start: forward difference
        tp[near_start] = t0 + h
        tm[near_start] = t0
        dt[near_start] = h
        # Near end: backward difference
        tp[near_end] = t1
        tm[near_end] = t1 - h
        dt[near_end] = h
        # Middle: central difference
        tp[mid_mask] = all_params[mid_mask] + h
        tm[mid_mask] = all_params[mid_mask] - h
        dt[mid_mask] = 2.0 * h

        pp = self._batch_point_at(tp)
        pm = self._batch_point_at(tm)
        deriv = (pp - pm) / dt[:, None]
        speeds = np.sqrt(np.sum(deriv * deriv, axis=1))

        # Apply GL weights and cumsum
        speeds = speeds.reshape(n_samples, 5)
        arc_lens = halfs * np.sum(GL_WEIGHTS[None, :] * speeds, axis=1)
        s_arr = np.empty(n_samples + 1)
        s_arr[0] = 0.0
        np.cumsum(arc_lens, out=s_arr[1:])

        total_len = s_arr[-1]
        n_segs = (count - 1) if include_endpoints else (count + 1)
        seg_len = total_len / n_segs

        def _deriv_params(params_arr):
            n_ = len(params_arr)
            tp_ = np.empty(n_)
            tm_ = np.empty(n_)
            dt_ = np.empty(n_)
            for i_ in range(n_):
                ti = params_arr[i_]
                if ti <= t0 + h:
                    tp_[i_] = t0 + h
                    tm_[i_] = t0
                    dt_[i_] = h
                elif ti >= t1 - h:
                    tp_[i_] = t1
                    tm_[i_] = t1 - h
                    dt_[i_] = h
                else:
                    tp_[i_] = ti + h
                    tm_[i_] = ti - h
                    dt_[i_] = 2.0 * h
            return tp_, tm_, dt_

        def speed_at(t):
            arr = np.array([t])
            tp_, tm_, dt_ = _deriv_params(arr)
            both = self._batch_point_at(np.concatenate([tp_, tm_]))
            d = (both[0] - both[1]) / dt_[0]
            return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])

        def arc_length_gauss(ta, tb):
            mid = (ta + tb) * 0.5
            half = (tb - ta) * 0.5
            gl_t = mid + half * GL_NODES
            tp_, tm_, dt_ = _deriv_params(gl_t)
            k = len(tp_)
            both = self._batch_point_at(np.concatenate([tp_, tm_]))
            d = (both[:k] - both[k:]) / dt_[:, None]
            spd = np.sqrt(np.sum(d * d, axis=1))
            return half * float(np.dot(GL_WEIGHTS, spd))

        def find_t_at_s(s_target):
            if s_target <= 0.0:
                return t0
            if s_target >= total_len:
                return t1

            idx = np.searchsorted(s_arr, s_target, side='right') - 1
            lo = max(0, min(idx, n_samples - 1))
            hi = lo + 1

            frac = (s_target - s_arr[lo]) / (s_arr[hi] - s_arr[lo])
            t = t_vals[lo] + frac * (t_vals[hi] - t_vals[lo])

            t_lo, t_hi = t_vals[lo], t_vals[hi]
            for _ in range(20):
                s_cur = s_arr[lo] + arc_length_gauss(t_vals[lo], t)
                error = s_cur - s_target

                if abs(error) < 1e-12:
                    break

                spd = speed_at(t)
                if spd < 1e-14:
                    if error > 0:
                        t_hi = t
                        t = (t_lo + t_hi) * 0.5
                    else:
                        t_lo = t
                        t = (t_lo + t_hi) * 0.5
                    continue

                t_new = t - error / spd
                if t_new <= t_lo or t_new >= t_hi:
                    if error > 0:
                        t_hi = t
                        t = (t_lo + t_hi) * 0.5
                    else:
                        t_lo = t
                        t = (t_lo + t_hi) * 0.5
                else:
                    t = t_new

            return t

        if include_endpoints:
            s_targets = seg_len * np.arange(count, dtype=np.float64)
        else:
            s_targets = seg_len * np.arange(1, count + 1, dtype=np.float64)

        clamped = np.clip(s_targets, 0.0, total_len)
        idxs = np.clip(np.searchsorted(s_arr, clamped, side='right') - 1, 0, n_samples - 1)
        s_lo_arr = s_arr[idxs]
        s_hi_arr = s_arr[idxs + 1]
        denom = np.where(s_hi_arr > s_lo_arr, s_hi_arr - s_lo_arr, 1.0)
        frac = np.where(s_hi_arr > s_lo_arr, (clamped - s_lo_arr) / denom, 0.0)
        t_cur = t_vals[idxs] + frac * (t_vals[idxs + 1] - t_vals[idxs])
        t_lo_b = t_vals[idxs].copy()
        t_hi_b = t_vals[idxs + 1].copy()

        cn = len(t_cur)
        for _ in range(20):
            ta = t_vals[idxs]
            tb = t_cur
            mid_b = (ta + tb) * 0.5
            half_b = (tb - ta) * 0.5
            gl_t = mid_b[:, None] + half_b[:, None] * GL_NODES[None, :]
            gl_flat = gl_t.ravel()
            nc = len(gl_flat)
            all_t = np.concatenate([gl_flat, t_cur])
            near_s_all = all_t <= t0 + h
            near_e_all = all_t >= t1 - h
            tp_all = np.where(near_s_all, t0 + h, np.where(near_e_all, t1, all_t + h))
            tm_all = np.where(near_s_all, t0, np.where(near_e_all, t1 - h, all_t - h))
            dt_all = np.where(near_s_all | near_e_all, h, 2.0 * h)
            k_all = len(all_t)
            both = self._batch_point_at(np.concatenate([tp_all, tm_all]))
            d_all = (both[:k_all] - both[k_all:]) / dt_all[:, None]
            spd_all = np.sqrt(np.sum(d_all * d_all, axis=1))
            spd_gl = spd_all[:nc].reshape(-1, 5)
            arc_vals = half_b * np.sum(GL_WEIGHTS[None, :] * spd_gl, axis=1)
            spd_cur = spd_all[nc:]

            s_cur = s_lo_arr + arc_vals
            error = s_cur - clamped
            if np.max(np.abs(error)) < 1e-12:
                break

            safe_spd = np.where(spd_cur > 1e-14, spd_cur, 1.0)
            t_new = t_cur - error / safe_spd
            bad = (t_new <= t_lo_b) | (t_new >= t_hi_b) | (spd_cur < 1e-14)
            bisect = np.where(error > 0, (t_lo_b + t_cur) * 0.5, (t_cur + t_hi_b) * 0.5)
            t_next = np.where(bad, bisect, t_new)
            pos_err = error > 0
            t_hi_b = np.where(pos_err & bad, t_cur, t_hi_b)
            t_lo_b = np.where((~pos_err) & bad, t_cur, t_lo_b)
            t_cur = t_next

        t_cur = np.where(s_targets <= 0.0, t0, np.where(s_targets >= total_len, t1, t_cur))

        pts = self._batch_point_at(t_cur)
        points = [Point(pts[i, 0], pts[i, 1], pts[i, 2]) for i in range(len(t_cur))]
        params = t_cur.tolist()

        return points, params

    def divide_by_length(self, segment_length: float) -> tuple[list[Point], list[float]]:
        """Divide curve by arc length using Gauss-Legendre quadrature.

        Parameters
        ----------
        segment_length : float
            Target length between points.

        Returns
        -------
        tuple[list[Point], list[float]]
            Points and parameters spaced by segment_length.
        """
        points = []
        params = []

        if not self.is_valid() or segment_length <= 0.0:
            return points, params

        t0, t1 = self.domain()
        dom_len = t1 - t0
        h = dom_len * 1e-8

        GL_NODES = np.array([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640])
        GL_WEIGHTS = np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891])

        # Build arc-length table using batch evaluation (same pattern as divide_by_count)
        curve_len = self.length()
        n_samples = max(1000, int(curve_len / segment_length) * 100)

        t_vals = np.linspace(t0, t1, n_samples + 1)
        mids = (t_vals[:-1] + t_vals[1:]) * 0.5
        halfs = (t_vals[1:] - t_vals[:-1]) * 0.5

        gl_params = mids[:, None] + halfs[:, None] * GL_NODES[None, :]
        all_params = gl_params.ravel()

        n_all = len(all_params)
        tp = np.empty(n_all)
        tm = np.empty(n_all)
        dt_arr = np.empty(n_all)
        near_start = all_params <= t0 + h
        near_end = all_params >= t1 - h
        mid_mask = ~near_start & ~near_end
        tp[near_start] = t0 + h; tm[near_start] = t0; dt_arr[near_start] = h
        tp[near_end] = t1; tm[near_end] = t1 - h; dt_arr[near_end] = h
        tp[mid_mask] = all_params[mid_mask] + h
        tm[mid_mask] = all_params[mid_mask] - h
        dt_arr[mid_mask] = 2.0 * h

        pp = self._batch_point_at(tp)
        pm = self._batch_point_at(tm)
        deriv = (pp - pm) / dt_arr[:, None]
        speeds = np.sqrt(np.sum(deriv * deriv, axis=1))

        speeds = speeds.reshape(n_samples, 5)
        arc_lens = halfs * np.sum(GL_WEIGHTS[None, :] * speeds, axis=1)
        s_arr = np.empty(n_samples + 1)
        s_arr[0] = 0.0
        np.cumsum(arc_lens, out=s_arr[1:])

        total_len = s_arr[-1]

        def _deriv_params(params_arr):
            n_ = len(params_arr)
            tp_ = np.empty(n_); tm_ = np.empty(n_); dt_ = np.empty(n_)
            for i_ in range(n_):
                ti = params_arr[i_]
                if ti <= t0 + h:
                    tp_[i_] = t0 + h; tm_[i_] = t0; dt_[i_] = h
                elif ti >= t1 - h:
                    tp_[i_] = t1; tm_[i_] = t1 - h; dt_[i_] = h
                else:
                    tp_[i_] = ti + h; tm_[i_] = ti - h; dt_[i_] = 2.0 * h
            return tp_, tm_, dt_

        def speed_at(t):
            arr = np.array([t])
            tp_, tm_, dt_ = _deriv_params(arr)
            pp_ = self._batch_point_at(tp_)
            pm_ = self._batch_point_at(tm_)
            d = (pp_[0] - pm_[0]) / dt_[0]
            return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])

        def arc_length_gauss(ta, tb):
            mid = (ta + tb) * 0.5
            half_w = (tb - ta) * 0.5
            gl_t = mid + half_w * GL_NODES
            tp_, tm_, dt_ = _deriv_params(gl_t)
            pp_ = self._batch_point_at(tp_)
            pm_ = self._batch_point_at(tm_)
            d = (pp_ - pm_) / dt_[:, None]
            spd = np.sqrt(np.sum(d * d, axis=1))
            return half_w * float(np.dot(GL_WEIGHTS, spd))

        def find_t_at_s(s_target):
            if s_target <= 0.0:
                return t0
            if s_target >= total_len:
                return t1

            idx = np.searchsorted(s_arr, s_target, side='right') - 1
            lo = max(0, min(idx, n_samples - 1))
            hi_idx = lo + 1

            frac = (s_target - s_arr[lo]) / (s_arr[hi_idx] - s_arr[lo])
            t = t_vals[lo] + frac * (t_vals[hi_idx] - t_vals[lo])

            t_lo, t_hi = t_vals[lo], t_vals[hi_idx]
            for _ in range(20):
                s_cur = s_arr[lo] + arc_length_gauss(t_vals[lo], t)
                error = s_cur - s_target

                if abs(error) < 1e-12:
                    break

                spd = speed_at(t)
                if spd < 1e-14:
                    if error > 0:
                        t_hi = t; t = (t_lo + t_hi) * 0.5
                    else:
                        t_lo = t; t = (t_lo + t_hi) * 0.5
                    continue

                t_new = t - error / spd
                if t_new <= t_lo or t_new >= t_hi:
                    if error > 0:
                        t_hi = t; t = (t_lo + t_hi) * 0.5
                    else:
                        t_lo = t; t = (t_lo + t_hi) * 0.5
                else:
                    t = t_new

            return t

        # Collect all t-values first, then batch evaluate points
        t_results = []
        s = 0.0
        while s <= total_len + 1e-10:
            t_results.append(find_t_at_s(s))
            s += segment_length

        if t_results:
            t_arr = np.array(t_results)
            pts = self._batch_point_at(t_arr)
            for i in range(len(t_results)):
                points.append(Point(pts[i, 0], pts[i, 1], pts[i, 2]))
                params.append(t_results[i])

        return points, params


    # ═══════════════════════════════════════════════════════════════════════════
    # Evaluation
    # ═══════════════════════════════════════════════════════════════════════════

    def point_at(self, t: float) -> Point:
        """Evaluate point at parameter t.
        
        Implementation matches OpenNURBS evaluation approach.
        """
        if not self.is_valid():
            return Point(0, 0, 0)
        
        # Find span (returns index relative to shifted nurbsknot array)
        span = self._find_span(t)
        if span < 0:
            return Point(0, 0, 0)
        
        # Evaluate using Cox-de Boor algorithm
        N = self._basis_functions(span, t)
        
        # Compute point
        pt = np.zeros(self.m_dim)
        
        if self.m_is_rat:
            # Rational: CVs stored as (x*w, y*w, z*w, w) - homogeneous form
            w = 0.0
            for i in range(self.m_order):
                cv_idx = span + i
                if cv_idx < 0 or cv_idx >= self.m_cv_count:
                    continue
                idx = cv_idx * self.m_cv_stride
                weight = self.m_cv[idx + self.m_dim]
                w += N[i] * weight
                for j in range(self.m_dim):
                    pt[j] += N[i] * self.m_cv[idx + j]

            if abs(w) > 1e-10:
                pt /= w
        else:
            # Non-rational curve
            # In OpenNURBS, span index directly corresponds to CV starting index
            for i in range(self.m_order):
                cv_idx = span + i
                if cv_idx < 0 or cv_idx >= self.m_cv_count:
                    continue
                idx = cv_idx * self.m_cv_stride
                for j in range(self.m_dim):
                    pt[j] += N[i] * self.m_cv[idx + j]
        
        return Point(pt[0], pt[1] if self.m_dim > 1 else 0, pt[2] if self.m_dim > 2 else 0)

    def evaluate(self, t: float, derivative_count: int = 0) -> list[Vector]:
        """Evaluate point and derivatives on curve at parameter t.

        Parameters
        ----------
        t : float
            Parameter value.
        derivative_count : int, optional
            Number of derivatives to compute. Defaults to 0 (point only).

        Returns
        -------
        list[Vector]
            [point, 1st_derivative, 2nd_derivative, ...].
        """
        result = []

        if not self.is_valid():
            result.append(Vector(0, 0, 0))
            return result

        # Clamp derivative order to degree
        max_derivs = min(derivative_count, self.degree())

        span = self._find_span(t)
        ders = self._basis_functions_derivatives(span, t, max_derivs)

        # Evaluate non-rational or homogeneous coordinates and derivatives
        p = self.degree()
        Aders = [[0.0, 0.0, 0.0, 0.0] for _ in range(max_derivs + 1)]

        for k in range(max_derivs + 1):
            for j in range(p + 1):
                cv_idx = span + j
                if cv_idx < 0 or cv_idx >= self.m_cv_count:
                    continue
                idx = cv_idx * self.m_cv_stride

                Nx = ders[k, j]
                cx = self.m_cv[idx]
                cy = self.m_cv[idx + 1] if self.m_dim > 1 else 0.0
                cz = self.m_cv[idx + 2] if self.m_dim > 2 else 0.0
                wv = self.m_cv[idx + self.m_dim] if self.m_is_rat else 1.0

                # CVs stored in homogeneous form: cx=x*w, cy=y*w, cz=z*w
                Aders[k][0] += Nx * cx
                Aders[k][1] += Nx * cy
                Aders[k][2] += Nx * cz
                Aders[k][3] += Nx * wv

        # Convert from homogeneous derivatives (Aders) to Cartesian derivatives
        Cders = [[0.0, 0.0, 0.0] for _ in range(max_derivs + 1)]

        if not self.m_is_rat:
            # Non-rational: derivatives are directly Aders (w == 1)
            for k in range(max_derivs + 1):
                Cders[k] = [Aders[k][0], Aders[k][1], Aders[k][2]]
        else:
            # Rational: use standard formula (Piegl & Tiller, Eq. 2.28)
            for k in range(max_derivs + 1):
                w = Aders[0][3]
                inv_w = 1.0 / w if w != 0.0 else 0.0

                # Initialize derivative to homogeneous derivative
                Ck_x = Aders[k][0]
                Ck_y = Aders[k][1]
                Ck_z = Aders[k][2]

                # Subtract contributions of weight derivatives
                for j_idx in range(1, k + 1):
                    # Binomial coefficient: k! / (j! * (k-j)!)
                    coeff = 1.0
                    for ii in range(1, j_idx + 1):
                        coeff = coeff * (k - ii + 1) / ii
                    wj = Aders[j_idx][3]
                    Ck_x -= coeff * wj * Cders[k - j_idx][0]
                    Ck_y -= coeff * wj * Cders[k - j_idx][1]
                    Ck_z -= coeff * wj * Cders[k - j_idx][2]

                Ck_x *= inv_w
                Ck_y *= inv_w
                Ck_z *= inv_w
                Cders[k] = [Ck_x, Ck_y, Ck_z]

        # Fill result vectors (0th derivative = point)
        for k in range(max_derivs + 1):
            result.append(Vector(Cders[k][0], Cders[k][1], Cders[k][2]))

        # If caller requested more derivatives than degree, pad with zeros
        for k in range(max_derivs + 1, derivative_count + 1):
            result.append(Vector(0.0, 0.0, 0.0))

        return result

    def curvature_at(self, t: float) -> float:
        """Curvature magnitude (1/radius) at parameter t.

        Uses analytic 1st/2nd derivatives: kappa = |C' x C''| / |C'|^3.
        Matches OCCT GeomLProp_CLProps.Curvature.
        """
        d = self.evaluate(t, 2)
        if len(d) < 3:
            return 0.0
        s = d[1].magnitude()
        if s < 1e-12:
            return 0.0
        return d[1].cross(d[2]).magnitude() / (s * s * s)

    def closest_parameter(self, test_point: 'Point') -> float:
        """Parameter of the closest point on the curve to test_point (grid seed + Newton).

        Matches OCCT GeomAPI_ProjectPointOnCurve.
        """
        from session_py.closest import Closest
        return Closest.curve_point(self, test_point)[0]

    def closest_point(self, test_point: 'Point') -> 'Point':
        """Closest point on the curve to test_point."""
        return self.point_at(self.closest_parameter(test_point))

    @overload
    def closest_parameters_curve(self, other: 'NurbsCurve', return_distance: bool = False) -> tuple[float, float]: ...

    @overload
    def closest_parameters_curve(self, other: 'NurbsCurve', return_distance: bool = True) -> tuple[tuple[float, float], float]: ...

    def closest_parameters_curve(self, other: 'NurbsCurve', return_distance: bool = False) -> tuple[float, float] | tuple[tuple[float, float], float]:
        """Parameters (u, v) where this curve is closest to another curve.

        Matches OCCT GeomAPI_ExtremaCurveCurve. If return_distance, returns ((u, v), dist).
        """
        from session_py.closest import Closest
        u, v, dist = Closest.curve_curve(self, other)
        if return_distance:
            return (u, v), dist
        return (u, v)

    @overload
    def closest_points_curve(self, other: 'NurbsCurve', return_distance: bool = False) -> tuple['Point', 'Point']: ...

    @overload
    def closest_points_curve(self, other: 'NurbsCurve', return_distance: bool = True) -> tuple[tuple['Point', 'Point'], float]: ...

    def closest_points_curve(self, other: 'NurbsCurve', return_distance: bool = False) -> tuple['Point', 'Point'] | tuple[tuple['Point', 'Point'], float]:
        """Points (pa, pb) where this curve is closest to another curve."""
        from session_py.closest import Closest
        u, v, dist = Closest.curve_curve(self, other)
        pa = self.point_at(u)
        pb = other.point_at(v)
        if return_distance:
            return (pa, pb), dist
        return (pa, pb)

    def tangent_at(self, t: float) -> Vector:
        """Evaluate tangent vector at parameter t (normalized)"""
        if not self.is_valid():
            return Vector(0, 0, 0)

        t0, t1 = self.domain()
        h = (t1 - t0) * 1e-7

        if t <= t0 + h:
            p1 = self.point_at(t0)
            p2 = self.point_at(t0 + h)
        elif t >= t1 - h:
            p1 = self.point_at(t1 - h)
            p2 = self.point_at(t1)
        else:
            p1 = self.point_at(t - h)
            p2 = self.point_at(t + h)

        tan = Vector(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
        mag = tan.magnitude()
        if mag > 1e-14:
            tan.normalize_self()
        return tan

    def plane_at(self, t: float, normalized: bool = True) -> 'Plane':
        """Get Frenet frame at parameter t (tangent, normal, binormal)"""
        if not self.is_valid():
            return Plane.invalid()

        t0, t1 = self.domain()
        if normalized:
            if t < 0.0 or t > 1.0:
                return Plane.invalid()
            param = t0 + t * (t1 - t0)
        else:
            if t < t0 or t > t1:
                return Plane.invalid()
            param = t

        h = (t1 - t0) * 1e-5
        origin = self.point_at(param)

        # Handle endpoints with one-sided differences
        if param <= t0 + h:
            p0 = self.point_at(t0)
            pp = self.point_at(t0 + h)
            pp2 = self.point_at(t0 + 2 * h)
            d1 = Vector(pp[0] - p0[0], pp[1] - p0[1], pp[2] - p0[2])
            d2 = Vector(
                (pp2[0] - 2 * pp[0] + p0[0]) / (h * h),
                (pp2[1] - 2 * pp[1] + p0[1]) / (h * h),
                (pp2[2] - 2 * pp[2] + p0[2]) / (h * h)
            )
        elif param >= t1 - h:
            pm = self.point_at(t1 - h)
            p0 = self.point_at(t1)
            pm2 = self.point_at(t1 - 2 * h)
            d1 = Vector(p0[0] - pm[0], p0[1] - pm[1], p0[2] - pm[2])
            d2 = Vector(
                (p0[0] - 2 * pm[0] + pm2[0]) / (h * h),
                (p0[1] - 2 * pm[1] + pm2[1]) / (h * h),
                (p0[2] - 2 * pm[2] + pm2[2]) / (h * h)
            )
        else:
            # Central difference for interior points
            pm = self.point_at(param - h)
            p0 = self.point_at(param)
            pp = self.point_at(param + h)
            d1 = Vector(
                (pp[0] - pm[0]) / (2 * h),
                (pp[1] - pm[1]) / (2 * h),
                (pp[2] - pm[2]) / (2 * h)
            )
            d2 = Vector(
                (pp[0] - 2 * p0[0] + pm[0]) / (h * h),
                (pp[1] - 2 * p0[1] + pm[1]) / (h * h),
                (pp[2] - 2 * p0[2] + pm[2]) / (h * h)
            )

        d1_mag = d1.magnitude()
        if d1_mag < 1e-14:
            return Plane.invalid()

        T = d1.normalized()

        d2_dot_T = d2.dot(T)
        N = Vector(d2[0] - d2_dot_T * T[0], d2[1] - d2_dot_T * T[1], d2[2] - d2_dot_T * T[2])
        n_mag = N.magnitude()

        if n_mag < 1e-14:
            world_z = Vector(0, 0, 1)
            N = T.cross(world_z)
            n_mag = N.magnitude()
            if n_mag < 1e-14:
                world_y = Vector(0, 1, 0)
                N = T.cross(world_y)
                n_mag = N.magnitude()

        if n_mag > 1e-14:
            N = N.normalized()

        B = T.cross(N).normalized()

        return Plane.from_frame(origin, T, N, B)

    def _invalidate_rmf_cache(self):
        self._rmf_cache = None

    def _ensure_rmf_cache(self):
        if self._rmf_cache is not None:
            return

        num_samples = max(20, self.span_count() * 4)
        t0, t1 = self.domain()
        dt = (t1 - t0) / (num_samples - 1)

        params = []
        quaternions = []
        origins = []

        for i in range(num_samples):
            t = t0 + i * dt
            params.append(t)

            pl = self.perpendicular_plane_at(t, False)
            if pl.is_valid():
                origins.append(pl.origin)
                quaternions.append(self._frame_to_quaternion(pl.x_axis, pl.y_axis, pl.z_axis))
            else:
                origins.append(Point(0, 0, 0))
                quaternions.append([1.0, 0.0, 0.0, 0.0])

        self._rmf_cache = {'params': params, 'quaternions': quaternions, 'origins': origins}

    def _frame_to_quaternion(r: Vector, s: Vector, t: Vector) -> list[float]:
        trace = r.x + s.y + t.z

        if trace > 0:
            big_s = math.sqrt(trace + 1.0) * 2
            return [0.25 * big_s, (s.z - t.y) / big_s, (t.x - r.z) / big_s, (r.y - s.x) / big_s]
        elif r.x > s.y and r.x > t.z:
            big_s = math.sqrt(1.0 + r.x - s.y - t.z) * 2
            return [(s.z - t.y) / big_s, 0.25 * big_s, (s.x + r.y) / big_s, (t.x + r.z) / big_s]
        elif s.y > t.z:
            big_s = math.sqrt(1.0 + s.y - r.x - t.z) * 2
            return [(t.x - r.z) / big_s, (s.x + r.y) / big_s, 0.25 * big_s, (t.y + s.z) / big_s]
        else:
            big_s = math.sqrt(1.0 + t.z - r.x - s.y) * 2
            return [(r.y - s.x) / big_s, (t.x + r.z) / big_s, (t.y + s.z) / big_s, 0.25 * big_s]

    def _quaternion_to_frame(q: list[float]) -> tuple[Vector, Vector, Vector]:
        w, x, y, z = q
        r = Vector(1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y))
        s = Vector(2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x))
        t = Vector(2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y))
        return (r, s, t)

    def _slerp(q0: list[float], q1: list[float], u: float) -> list[float]:
        dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3]

        q1_adj = q1
        if dot < 0:
            dot = -dot
            q1_adj = [-q1[0], -q1[1], -q1[2], -q1[3]]

        if dot > 0.9995:
            result = [
                q0[0] + u * (q1_adj[0] - q0[0]),
                q0[1] + u * (q1_adj[1] - q0[1]),
                q0[2] + u * (q1_adj[2] - q0[2]),
                q0[3] + u * (q1_adj[3] - q0[3])
            ]
            norm = math.sqrt(sum(r*r for r in result))
            return [r / norm for r in result]

        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        w0 = math.sin((1 - u) * theta) / sin_theta
        w1 = math.sin(u * theta) / sin_theta

        return [
            w0*q0[0] + w1*q1_adj[0],
            w0*q0[1] + w1*q1_adj[1],
            w0*q0[2] + w1*q1_adj[2],
            w0*q0[3] + w1*q1_adj[3]
        ]

    def perpendicular_plane_at(self, t: float, normalized: bool = True) -> 'Plane':
        """Get rotation minimizing perpendicular frame at parameter t"""
        if not self.is_valid():
            return Plane.invalid()

        t0, t1 = self.domain()
        param = t0 + t * (t1 - t0) if normalized else t
        if normalized and (t < 0.0 or t > 1.0):
            return Plane.invalid()
        if not normalized and (t < t0 or t > t1):
            return Plane.invalid()

        # Get initial frame at t0 using Frenet (curvature-based)
        derivs0 = self.evaluate(t0, 2)
        D1_0 = Vector(derivs0[1].x, derivs0[1].y, derivs0[1].z)
        D2_0 = Vector(derivs0[2].x, derivs0[2].y, derivs0[2].z)

        D1_0_mag = D1_0.magnitude()
        if D1_0_mag < 1e-14:
            return Plane.invalid()

        tangent0 = D1_0 / D1_0_mag

        # Initial normal from curvature (Frenet)
        D2_dot_D1 = D2_0.dot(D1_0)
        D1_0_mag_sq = D1_0_mag * D1_0_mag
        N0_unnorm = Vector(
            D2_0.x - (D2_dot_D1 / D1_0_mag_sq) * D1_0.x,
            D2_0.y - (D2_dot_D1 / D1_0_mag_sq) * D1_0.y,
            D2_0.z - (D2_dot_D1 / D1_0_mag_sq) * D1_0.z
        )

        N0_mag = N0_unnorm.magnitude()
        if N0_mag < 1e-14:
            world_z = Vector(0, 0, 1)
            N0_unnorm = world_z.cross(tangent0)
            N0_mag = N0_unnorm.magnitude()
            if N0_mag < 1e-14:
                world_y = Vector(0, 1, 0)
                N0_unnorm = world_y.cross(tangent0)
                N0_mag = N0_unnorm.magnitude()
        r0 = N0_unnorm / N0_mag

        origin = self.point_at(param)

        # If at start, return Frenet frame directly
        if abs(param - t0) < 1e-14:
            s0 = tangent0.cross(r0).normalized()
            return Plane.from_frame(origin, r0, s0, tangent0)

        # Propagate frame using Double Reflection (RMF) algorithm
        num_steps = max(10, int((param - t0) / (t1 - t0) * 100))
        dt = (param - t0) / num_steps

        ri = r0
        ti = t0
        xi = self.point_at(ti)
        tangent_i = tangent0

        for _ in range(num_steps):
            if ti >= param - 1e-14:
                break
            ti_next = min(ti + dt, param)
            xi_next = self.point_at(ti_next)
            tangent_next = self.tangent_at(ti_next).normalized()

            v1 = Vector(xi_next.x - xi.x, xi_next.y - xi.y, xi_next.z - xi.z)
            c1 = v1.dot(v1)
            if c1 < 1e-28:
                ti, xi, tangent_i = ti_next, xi_next, tangent_next
                continue

            ri_dot_v1 = ri.dot(v1)
            r_l = Vector(
                ri.x - 2.0 * ri_dot_v1 / c1 * v1.x,
                ri.y - 2.0 * ri_dot_v1 / c1 * v1.y,
                ri.z - 2.0 * ri_dot_v1 / c1 * v1.z
            )

            ti_dot_v1 = tangent_i.dot(v1)
            t_l = Vector(
                tangent_i.x - 2.0 * ti_dot_v1 / c1 * v1.x,
                tangent_i.y - 2.0 * ti_dot_v1 / c1 * v1.y,
                tangent_i.z - 2.0 * ti_dot_v1 / c1 * v1.z
            )

            v2 = Vector(tangent_next.x - t_l.x, tangent_next.y - t_l.y, tangent_next.z - t_l.z)
            c2 = v2.dot(v2)
            if c2 < 1e-28:
                ri = r_l
            else:
                rl_dot_v2 = r_l.dot(v2)
                ri = Vector(
                    r_l.x - 2.0 * rl_dot_v2 / c2 * v2.x,
                    r_l.y - 2.0 * rl_dot_v2 / c2 * v2.y,
                    r_l.z - 2.0 * rl_dot_v2 / c2 * v2.z
                )

            ri = ri.normalized()
            ti, xi, tangent_i = ti_next, xi_next, tangent_next

        tangent = self.tangent_at(param).normalized()
        ri_dot_t = ri.dot(tangent)
        ri = Vector(ri.x - ri_dot_t * tangent.x, ri.y - ri_dot_t * tangent.y, ri.z - ri_dot_t * tangent.z).normalized()
        s = tangent.cross(ri).normalized()

        return Plane.from_frame(origin, ri, s, tangent)

    def get_perpendicular_planes(self, count: int) -> list['Plane']:
        """Get multiple rotation minimizing frames along the curve.
        count = number of subdivisions (returns count+1 frames at arc-length equidistant points)"""
        pts, params = self.divide_by_count(count + 1, True)
        return [self.perpendicular_plane_at(t, False) for t in params]

    def _batch_evaluate_deriv1(self, params: np.ndarray) -> np.ndarray:
        kn = self.m_nurbsknot
        cv = self.m_cv
        order = self.m_order
        dim = self.m_dim
        stride = self.m_cv_stride
        cv_count = self.m_cv_count
        is_rat = self.m_is_rat
        p = order - 1
        pp1 = p + 1

        params = np.asarray(params, dtype=np.float64)
        n = len(params)
        if n == 0:
            return np.empty((0, 3))
        kn_arr = np.asarray(kn, dtype=np.float64)
        cv_arr = np.asarray(cv, dtype=np.float64)

        if cv_count >= order and len(kn_arr) >= cv_count:
            interior = kn_arr[order - 2:cv_count]
            spans = np.clip(np.searchsorted(interior, params, side='right') - 1, 0, cv_count - order)
        else:
            spans = np.zeros(n, dtype=np.int64)
        offset = order - 2 + spans

        ndu = np.zeros((n, pp1, pp1))
        left = np.zeros((n, pp1))
        right = np.zeros((n, pp1))
        ndu[:, 0, 0] = 1.0
        for j in range(1, pp1):
            left[:, j] = params - kn_arr[offset + 1 - j]
            right[:, j] = kn_arr[offset + j] - params
            saved = np.zeros(n)
            for r in range(j):
                denom = right[:, r + 1] + left[:, j - r]
                nz = np.abs(denom) > 1e-14
                safe = np.where(nz, denom, 1.0)
                ndu[:, j, r] = denom
                temp = np.where(nz, ndu[:, r, j - 1] / safe, 0.0)
                ndu[:, r, j] = saved + right[:, r + 1] * temp
                saved = left[:, j - r] * temp
            ndu[:, j, j] = saved

        pk = p - 1
        N1 = np.zeros((n, pp1))
        for r in range(pp1):
            d = np.zeros(n)
            rk = r - 1
            if r >= 1:
                denom_a0 = ndu[:, pk + 1, rk]
                nz0 = np.abs(denom_a0) > 1e-14
                a0 = np.where(nz0, 1.0 / np.where(nz0, denom_a0, 1.0), 0.0)
                d = a0 * ndu[:, rk, pk]
            if r <= pk:
                denom_ak = ndu[:, pk + 1, r]
                nzk = np.abs(denom_ak) > 1e-14
                ak = np.where(nzk, -1.0 / np.where(nzk, denom_ak, 1.0), 0.0)
                d = d + ak * ndu[:, r, pk]
            N1[:, r] = d * p

        ci = np.clip(spans[:, None] + np.arange(order)[None, :], 0, cv_count - 1)
        base = ci * stride
        result = np.zeros((n, 3))
        if not is_rat:
            result[:, 0] = np.sum(N1 * cv_arr[base], axis=1)
            if dim > 1:
                result[:, 1] = np.sum(N1 * cv_arr[base + 1], axis=1)
            if dim > 2:
                result[:, 2] = np.sum(N1 * cv_arr[base + 2], axis=1)
        else:
            N0 = ndu[:, :, p]
            w_cv = cv_arr[base + dim]
            # Control points are stored in homogeneous form (x*w, y*w, z*w, w), so the
            # numerator sums N * Pw as stored; weighting it again squares the weight and
            # shortens the derivative (a radius-2 circle then integrates to 11.55, not 4*pi).
            Aw0x = np.sum(N0 * cv_arr[base], axis=1)
            Aw0y = np.sum(N0 * cv_arr[base + 1], axis=1) if dim > 1 else np.zeros(n)
            Aw0z = np.sum(N0 * cv_arr[base + 2], axis=1) if dim > 2 else np.zeros(n)
            Aw0w = np.sum(N0 * w_cv, axis=1)
            Aw1x = np.sum(N1 * cv_arr[base], axis=1)
            Aw1y = np.sum(N1 * cv_arr[base + 1], axis=1) if dim > 1 else np.zeros(n)
            Aw1z = np.sum(N1 * cv_arr[base + 2], axis=1) if dim > 2 else np.zeros(n)
            Aw1w = np.sum(N1 * w_cv, axis=1)
            mask = np.abs(Aw0w) > 1e-10
            inv_w = np.where(mask, 1.0 / np.where(mask, Aw0w, 1.0), 0.0)
            C0x = Aw0x * inv_w
            C0y = Aw0y * inv_w
            C0z = Aw0z * inv_w
            result[:, 0] = (Aw1x - Aw1w * C0x) * inv_w
            if dim > 1:
                result[:, 1] = (Aw1y - Aw1w * C0y) * inv_w
            if dim > 2:
                result[:, 2] = (Aw1z - Aw1w * C0z) * inv_w
        return result

    def _batch_evaluate_deriv1_scalar(self, params: np.ndarray) -> np.ndarray:
        n = len(params)
        result = np.empty((n, 3))
        kn = self.m_nurbsknot
        cv = self.m_cv
        order = self.m_order
        dim = self.m_dim
        stride = self.m_cv_stride
        cv_count = self.m_cv_count
        is_rat = self.m_is_rat
        p = order - 1
        find_span_fn = nurbsknot.find_span

        for idx in range(n):
            t = params[idx]
            span = find_span_fn(order, cv_count, kn, t)
            offset = order - 2 + span

            # Build ndu table (Algorithm A2.3 from The NURBS Book)
            ndu_flat = [0.0] * ((p + 1) * (p + 1))
            left = [0.0] * (p + 1)
            right = [0.0] * (p + 1)
            pp1 = p + 1
            ndu_flat[0] = 1.0
            for j in range(1, pp1):
                left[j] = t - kn[offset + 1 - j]
                right[j] = kn[offset + j] - t
                saved = 0.0
                for r in range(j):
                    ndu_flat[j * pp1 + r] = right[r + 1] + left[j - r]
                    denom = ndu_flat[j * pp1 + r]
                    temp = ndu_flat[r * pp1 + j - 1] / denom if abs(denom) > 1e-14 else 0.0
                    ndu_flat[r * pp1 + j] = saved + right[r + 1] * temp
                    saved = left[j - r] * temp
                ndu_flat[j * pp1 + j] = saved

            # Extract basis functions (k=0) and compute first derivatives (k=1)
            N0 = [ndu_flat[j * pp1 + p] for j in range(pp1)]
            N1 = [0.0] * pp1
            pk = p - 1
            for r in range(pp1):
                d = 0.0
                rk = r - 1
                if r >= 1:
                    a0 = 1.0 / ndu_flat[(pk + 1) * pp1 + rk]
                    d = a0 * ndu_flat[rk * pp1 + pk]
                if r <= pk:
                    ak = -1.0 / ndu_flat[(pk + 1) * pp1 + r]
                    d += ak * ndu_flat[r * pp1 + pk]
                N1[r] = d * p

            if not is_rat:
                dx = dy = dz = 0.0
                for i in range(order):
                    ci = span + i
                    if ci >= cv_count:
                        break
                    base = ci * stride
                    ni = N1[i]
                    dx += ni * cv[base]
                    dy += ni * cv[base + 1] if dim > 1 else 0
                    dz += ni * cv[base + 2] if dim > 2 else 0
                result[idx, 0] = dx
                result[idx, 1] = dy
                result[idx, 2] = dz
            else:
                Aw0x = Aw0y = Aw0z = Aw0w = 0.0
                Aw1x = Aw1y = Aw1z = Aw1w = 0.0
                for i in range(order):
                    ci = span + i
                    if ci >= cv_count:
                        break
                    base = ci * stride
                    cx = cv[base]
                    cy = cv[base + 1] if dim > 1 else 0.0
                    cz = cv[base + 2] if dim > 2 else 0.0
                    wi = cv[base + dim]
                    n0 = N0[i]
                    n1 = N1[i]
                    Aw0x += n0 * cx * wi; Aw0y += n0 * cy * wi
                    Aw0z += n0 * cz * wi; Aw0w += n0 * wi
                    Aw1x += n1 * cx * wi; Aw1y += n1 * cy * wi
                    Aw1z += n1 * cz * wi; Aw1w += n1 * wi
                inv_w = 1.0 / Aw0w if abs(Aw0w) > 1e-10 else 0.0
                C0x = Aw0x * inv_w; C0y = Aw0y * inv_w; C0z = Aw0z * inv_w
                result[idx, 0] = (Aw1x - Aw1w * C0x) * inv_w
                result[idx, 1] = (Aw1y - Aw1w * C0y) * inv_w
                result[idx, 2] = (Aw1z - Aw1w * C0z) * inv_w

        return result

    def _batch_point_at(self, params: np.ndarray) -> np.ndarray:
        kn = self.m_nurbsknot
        cv = self.m_cv
        order = self.m_order
        dim = self.m_dim
        stride = self.m_cv_stride
        cv_count = self.m_cv_count
        is_rat = self.m_is_rat

        if order == 2 and not is_rat and cv_count >= 2 and len(kn) >= cv_count:
            if not isinstance(params, np.ndarray) or params.dtype != np.float64:
                params = np.asarray(params, dtype=np.float64)
            xp = kn[:cv_count]
            if xp[-1] >= xp[0] and np.all(np.diff(xp) >= 0):
                n = len(params)
                result = np.zeros((n, 3))
                for d in range(min(dim, 3)):
                    fp = cv[d:cv_count * stride:stride]
                    result[:, d] = np.interp(params, xp, fp)
                return result

        if not isinstance(params, np.ndarray) or params.dtype != np.float64:
            params = np.asarray(params, dtype=np.float64)
        n = len(params)
        if n == 0:
            return np.empty((0, 3))
        # The general (degree>1) path below indexes kn/cv with an integer array, which only
        # works on an ndarray; pcurves rebuilt during split/boolean may store m_nurbsknot /
        # m_cv as plain Python lists, so coerce defensively.
        kn_arr = kn if isinstance(kn, np.ndarray) else np.asarray(kn, dtype=np.float64)
        cv_arr = cv if isinstance(cv, np.ndarray) else np.asarray(cv, dtype=np.float64)

        if cv_count >= order and len(kn_arr) >= order - 2 + (cv_count - order + 2):
            interior = kn_arr[order - 2: cv_count]
            spans = np.clip(np.searchsorted(interior, params, side='right') - 1, 0, cv_count - order)
        else:
            spans = np.zeros(n, dtype=np.int64)

        offset = order - 2 + spans
        N = np.zeros((n, order))
        N[:, 0] = 1.0
        left = np.empty((n, order))
        right = np.empty((n, order))
        for j in range(1, order):
            left[:, j] = params - kn_arr[offset + 1 - j]
            right[:, j] = kn_arr[offset + j] - params
            saved = np.zeros(n)
            for r in range(j):
                denom = right[:, r + 1] + left[:, j - r]
                temp = np.zeros(n)
                np.divide(N[:, r], denom, out=temp, where=denom != 0.0)
                N[:, r] = saved + right[:, r + 1] * temp
                saved = left[:, j - r] * temp
            N[:, j] = saved

        ci = np.clip(spans[:, None] + np.arange(order)[None, :], 0, cv_count - 1)
        base = ci * stride
        result = np.zeros((n, 3))
        if is_rat:
            # Control points are stored in homogeneous form (x*w, y*w, z*w, w), so the
            # numerator is sum(N * Pw) and the denominator sum(N * w) — multiplying the
            # numerator by w again double-counts the weight and collapses non-unit-weight
            # spans toward the control polygon.
            w_cv = cv_arr[base + dim]
            x = np.einsum('nj,nj->n', N, cv_arr[base])
            y = np.einsum('nj,nj->n', N, cv_arr[base + 1]) if dim > 1 else np.zeros(n)
            z = np.einsum('nj,nj->n', N, cv_arr[base + 2]) if dim > 2 else np.zeros(n)
            w = np.einsum('nj,nj->n', N, w_cv)
            mask = w != 0.0
            safe_w = np.where(mask, w, 1.0)
            result[:, 0] = np.where(mask, x / safe_w, x)
            if dim > 1:
                result[:, 1] = np.where(mask, y / safe_w, y)
            if dim > 2:
                result[:, 2] = np.where(mask, z / safe_w, z)
        else:
            result[:, 0] = np.einsum('nj,nj->n', N, cv_arr[base])
            if dim > 1:
                result[:, 1] = np.einsum('nj,nj->n', N, cv_arr[base + 1])
            if dim > 2:
                result[:, 2] = np.einsum('nj,nj->n', N, cv_arr[base + 2])
        return result

    def point_at_start(self) -> Point:
        """Evaluate point at curve start"""
        t0, _ = self.domain()
        return self.point_at(t0)

    def point_at_end(self) -> Point:
        """Evaluate point at curve end"""
        _, t1 = self.domain()
        return self.point_at(t1)

    def point_at_middle(self) -> Point:
        """Evaluate point at curve middle"""
        return self.point_at(self.domain_middle())

    def set_start_point(self, start_point: Point) -> bool:
        """Force curve to start at specified point.
        
        Parameters
        ----------
        start_point : Point
            New start point.
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        
        return self.set_cv(0, start_point)

    def set_end_point(self, end_point: Point) -> bool:
        """Force curve to end at specified point.
        
        Parameters
        ----------
        end_point : Point
            New end point.
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        
        return self.set_cv(self.m_cv_count - 1, end_point)


    # ═══════════════════════════════════════════════════════════════════════════
    # Modification Operations
    # ═══════════════════════════════════════════════════════════════════════════

    def reverse(self) -> bool:
        """Reverse curve direction"""
        if not self.is_valid():
            return False

        t0, t1 = self.domain()
        for i in range(len(self.m_nurbsknot)):
            self.m_nurbsknot[i] = t0 + t1 - self.m_nurbsknot[i]
        self.m_nurbsknot = np.flip(self.m_nurbsknot).copy()

        cvs = self.cv_size()
        for i in range(self.m_cv_count // 2):
            j = self.m_cv_count - 1 - i
            for k in range(cvs):
                temp = self.m_cv[i * cvs + k]
                self.m_cv[i * cvs + k] = self.m_cv[j * cvs + k]
                self.m_cv[j * cvs + k] = temp

        self._invalidate_rmf_cache()
        return True

    def swap_coordinates(self, axis_i: int, axis_j: int) -> bool:
        """Swap two coordinate axes.
        
        Parameters
        ----------
        axis_i : int
            First axis index (0=x, 1=y, 2=z).
        axis_j : int
            Second axis index (0=x, 1=y, 2=z).
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        if axis_i < 0 or axis_i >= self.m_dim or axis_j < 0 or axis_j >= self.m_dim:
            return False
        if axis_i == axis_j:
            return True
        
        for i in range(self.m_cv_count):
            idx = i * self.m_cv_stride
            temp = self.m_cv[idx + axis_i]
            self.m_cv[idx + axis_i] = self.m_cv[idx + axis_j]
            self.m_cv[idx + axis_j] = temp
        
        return True

    def trim(self, t0: float, t1: float) -> bool:
        if not self.is_valid() or t0 >= t1:
            return False

        d0, d1 = self.domain()
        if t0 < d0 - Tolerance.ZERO_TOLERANCE or t1 > d1 + Tolerance.ZERO_TOLERANCE:
            return False
        t0 = max(t0, d0)
        t1 = min(t1, d1)
        if abs(t0 - d0) < Tolerance.ZERO_TOLERANCE and abs(t1 - d1) < Tolerance.ZERO_TOLERANCE:
            return True

        p = self.degree()
        trim_start = t0 > d0 + Tolerance.ZERO_TOLERANCE
        trim_end = t1 < d1 - Tolerance.ZERO_TOLERANCE

        # Snap the trim parameters to an existing nurbsknot within the insertion tolerance:
        # inserting at an already-present value is a no-op, and the strict span search below
        # would miss a near-hit (a boundary landing on a polyline vertex nurbsknot).
        stol = (abs(d0) + abs(d1) + abs(d1 - d0)) * math.sqrt(np.finfo(float).eps)
        for k in self.m_nurbsknot:
            if trim_start and 0.0 < abs(k - t0) <= stol:
                t0 = float(k)
            if trim_end and 0.0 < abs(k - t1) <= stol:
                t1 = float(k)
        if t0 >= t1:
            return False

        # Insert nurbsknots at trim boundaries to multiplicity = degree
        if trim_start:
            if not self.insert_nurbsknot(t0, p):
                return False
        if trim_end:
            if not self.insert_nurbsknot(t1, p):
                return False

        full_nurbsknot_count = self.m_cv_count + self.m_order
        U = [0.0] * full_nurbsknot_count
        U[0] = self.m_nurbsknot[0]
        for i in range(len(self.m_nurbsknot)):
            U[i + 1] = self.m_nurbsknot[i]
        U[full_nurbsknot_count - 1] = self.m_nurbsknot[-1]

        tol = Tolerance.ZERO_TOLERANCE

        start_span = -1
        for i in range(full_nurbsknot_count - 1, -1, -1):
            if abs(U[i] - t0) < tol:
                start_span = i
                break

        end_span = -1
        for i in range(full_nurbsknot_count):
            if abs(U[i] - t1) < tol:
                end_span = i
                break

        if start_span < 0 or end_span < 0 or start_span >= end_span:
            return False

        first_cv = start_span - p if start_span >= p else 0
        last_cv = end_span - 1
        if last_cv >= self.m_cv_count:
            last_cv = self.m_cv_count - 1

        new_cv_count = last_cv - first_cv + 1
        if new_cv_count < self.m_order:
            new_cv_count = self.m_order
            if first_cv + new_cv_count - 1 < self.m_cv_count:
                last_cv = first_cv + new_cv_count - 1
            else:
                return False

        new_nurbsknot_count = new_cv_count + self.m_order - 2
        new_nurbsknot = [0.0] * new_nurbsknot_count

        for i in range(max(p, 1) - 1):
            new_nurbsknot[i] = t0

        mid_count = new_nurbsknot_count - 2 * (p - 1)
        if mid_count > 0:
            for i in range(mid_count):
                src_idx = start_span + i
                if src_idx < full_nurbsknot_count:
                    new_nurbsknot[p - 1 + i] = U[src_idx]
                else:
                    new_nurbsknot[p - 1 + i] = t1

        for i in range(max(p, 1) - 1):
            new_nurbsknot[new_nurbsknot_count - p + 1 + i] = t1

        cvs = self.m_cv_stride
        new_cv = [0.0] * (new_cv_count * cvs)
        for i in range(new_cv_count):
            src = (first_cv + i) * cvs
            dst = i * cvs
            new_cv[dst:dst + cvs] = self.m_cv[src:src + cvs]

        self.m_cv_count = new_cv_count
        self.m_cv = new_cv
        self.m_nurbsknot = new_nurbsknot

        self._invalidate_rmf_cache()
        return True

    def split(self, t: float) -> tuple[Optional['NurbsCurve'], Optional['NurbsCurve']]:
        if not self.is_valid():
            return None, None

        t0, t1 = self.domain()
        if t <= t0 or t >= t1:
            return None, None

        left = self.duplicate()
        right = self.duplicate()
        left.trim(t0, t)
        right.trim(t, t1)

        return left, right

    def extend(self, t0: float, t1: float) -> bool:
        """Extend curve to include domain [t0, t1].

        Uses de Boor extrapolation matching C++ implementation.

        Parameters
        ----------
        t0 : float
            New start parameter (can be before current start).
        t1 : float
            New end parameter (can be after current end).

        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid() or self.is_closed():
            return False

        domain_t0, domain_t1 = self.domain()
        cv_dim = self.cv_size()
        changed = False

        # Extend start (t0 < current domain start)
        if t0 < domain_t0:
            self.clamp_end(0)
            # Extrapolate using de Boor algorithm
            self._evaluate_nurbs_de_boor_inplace(cv_dim, self.m_order, 0, 1, t0)
            for i in range(self.m_order - 1):
                self.m_nurbsknot[i] = t0
            changed = True

        # Extend end (t1 > current domain end)
        if t1 > domain_t1:
            self.clamp_end(1)
            # Extrapolate using de Boor algorithm
            i0 = self.m_cv_count - self.m_order
            self._evaluate_nurbs_de_boor_inplace(cv_dim, self.m_order, i0, -1, t1)
            kc = self.nurbsknot_count()
            for i in range(self.m_cv_count - 1, kc):
                self.m_nurbsknot[i] = t1
            changed = True

        return changed

    def make_rational(self) -> bool:
        """Convert to rational curve"""
        if self.m_is_rat:
            return True
        
        new_stride = self.m_dim + 1
        new_cv = np.zeros(self.m_cv_count * new_stride)
        
        for i in range(self.m_cv_count):
            old_idx = i * self.m_cv_stride
            new_idx = i * new_stride
            
            for j in range(self.m_dim):
                new_cv[new_idx + j] = self.m_cv[old_idx + j]
            new_cv[new_idx + self.m_dim] = 1.0  # Weight
        
        self.m_is_rat = 1
        self.m_cv_stride = new_stride
        self.m_cv = new_cv
        
        return True

    def make_non_rational(self, force: bool = False) -> bool:
        """Convert to non-rational curve.

        If force=False (default), fails when weights differ.
        If force=True, sets all weights to 1.0 (changes geometry!).
        """
        if not self.m_is_rat:
            return True

        if force:
            for i in range(self.m_cv_count):
                idx = i * self.m_cv_stride
                self.m_cv[idx + self.m_dim] = 1.0
        else:
            w0 = self.weight(0)
            for i in range(1, self.m_cv_count):
                if abs(self.weight(i) - w0) > Tolerance.ZERO_TOLERANCE:
                    return False

        new_stride = self.m_dim
        new_cv = np.zeros(self.m_cv_count * new_stride)

        for i in range(self.m_cv_count):
            p = self.get_cv(i)
            new_idx = i * new_stride
            new_cv[new_idx] = p.x
            if self.m_dim > 1:
                new_cv[new_idx + 1] = p.y
            if self.m_dim > 2:
                new_cv[new_idx + 2] = p.z

        self.m_is_rat = 0
        self.m_cv_stride = new_stride
        self.m_cv = new_cv

        return True

    def clamp_end(self, end: int) -> bool:
        """Clamp ends (add multiplicity to end nurbsknots).

        Parameters
        ----------
        end : int
            0 for start, 1 for end, 2 for both.

        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        if end < 0 or end > 2:
            return False

        # Clamp start
        if end == 0 or end == 2:
            t = self.m_nurbsknot[self.m_order - 2]
            for i in range(self.m_order - 2):
                self.m_nurbsknot[i] = t

        # Clamp end
        if end == 1 or end == 2:
            t = self.m_nurbsknot[self.m_cv_count - 1]
            kc = self.nurbsknot_count()
            for i in range(self.m_cv_count, kc):
                self.m_nurbsknot[i] = t

        return True

    def increase_degree(self, desired_degree: int) -> bool:
        if not self.is_valid():
            return False
        if desired_degree < 1 or desired_degree < self.degree():
            return False
        if desired_degree == self.degree():
            return True
        if not self.clamp_end(2):
            return False

        degree_delta = desired_degree - self.degree()
        for _ in range(degree_delta):
            if not self._increment_degree():
                return False
        return True

    def _increment_degree(self) -> bool:
        import copy
        M_order = self.m_order
        M_cv_count = self.m_cv_count
        M_nurbsknot = self.m_nurbsknot.copy()
        M_cv = self.m_cv.copy()
        M_cv_stride = self.m_cv_stride
        cvdim = self.cv_size()

        # Count non-degenerate spans
        sc = 0
        deg = M_order - 1
        for i in range(deg, M_cv_count - 1):
            if M_nurbsknot[i] < M_nurbsknot[i + 1] - Tolerance.ZERO_TOLERANCE:
                sc += 1
        if sc == 0:
            sc = 1

        new_order = M_order + 1
        mkc = len(M_nurbsknot)

        # Build new nurbsknot vector: each distinct nurbsknot gets mult+1 copies
        new_nurbsknots = []
        ki = 0
        while ki < mkc:
            kn = M_nurbsknot[ki]
            mult = 1
            while ki + mult < mkc and abs(M_nurbsknot[ki + mult] - kn) < Tolerance.ZERO_TOLERANCE:
                mult += 1
            for _ in range(mult + 1):
                new_nurbsknots.append(kn)
            ki += mult
        new_nurbsknots = np.array(new_nurbsknots, dtype=float)
        new_cv_count = len(new_nurbsknots) - new_order + 2

        self.m_order = new_order
        self.m_cv_count = new_cv_count
        self.m_nurbsknot = new_nurbsknots
        self.m_cv = np.zeros(new_cv_count * self.m_cv_stride)

        # Compute new CVs per span using blossom
        siN = 0
        siM = 0
        for _ in range(sc):
            nurbsknotN = self.m_nurbsknot[siN:]
            nurbsknotM = M_nurbsknot[siM:]
            cvM = M_cv[siM * M_cv_stride:]
            # Get span multiplicity at the span boundary in new nurbsknot vector
            span_mult = self._forward_nurbsknot_mult(siN + self.degree() - 1)
            skip = self.m_order - span_mult
            for j in range(skip, self.m_order):
                cv_idx = siN + j
                P = _get_raised_degree_cv(M_order, cvdim, M_cv_stride, cvM, nurbsknotM, nurbsknotN, j)
                if P is None:
                    return False
                idx = cv_idx * self.m_cv_stride
                self.m_cv[idx:idx + cvdim] = P
            siN = _next_span_index(self.m_order, self.m_cv_count, self.m_nurbsknot, siN)
            siM = _next_span_index(M_order, M_cv_count, M_nurbsknot, siM)

        # Copy first and last CVs from original
        self.m_cv[0:cvdim] = M_cv[0:cvdim]
        last_new = (self.m_cv_count - 1) * self.m_cv_stride
        last_old = (M_cv_count - 1) * M_cv_stride
        self.m_cv[last_new:last_new + cvdim] = M_cv[last_old:last_old + cvdim]
        return True

    def _forward_nurbsknot_mult(self, nurbsknot_index: int) -> int:
        if nurbsknot_index < 0 or nurbsknot_index >= len(self.m_nurbsknot):
            return 0
        val = self.m_nurbsknot[nurbsknot_index]
        mult = 1
        i = nurbsknot_index + 1
        while i < len(self.m_nurbsknot) and abs(self.m_nurbsknot[i] - val) < Tolerance.ZERO_TOLERANCE:
            mult += 1
            i += 1
        i = nurbsknot_index - 1
        while i >= 0 and abs(self.m_nurbsknot[i] - val) < Tolerance.ZERO_TOLERANCE:
            mult += 1
            i -= 1
        return mult

    def change_closed_curve_seam(self, t: float) -> bool:
        if not self.is_valid() or not self.is_closed():
            return False

        t0, t1 = self.domain()
        dom_len = t1 - t0

        s = (t - t0) / dom_len
        if s < 0.0 or s > 1.0:
            s = s % 1.0
            if s < 0.0:
                s += 1.0
            t = t0 + s * dom_len

        if abs(t - t0) < Tolerance.ZERO_TOLERANCE or abs(t - t1) < Tolerance.ZERO_TOLERANCE:
            return True
        if t <= t0 or t >= t1:
            return True

        p = self.degree()
        order = self.m_order

        if self.is_periodic():
            sc = self.span_count()
            kc = self.nurbsknot_count()
            if sc >= kc - 2 * p + 1:
                nurbsknot_index = -1
                for i in range(kc):
                    if self.m_nurbsknot[i] > t:
                        nurbsknot_index = i
                        break
                if nurbsknot_index >= p and nurbsknot_index <= kc - p:
                    k0 = self.m_nurbsknot[nurbsknot_index - 1]
                    k1 = self.m_nurbsknot[nurbsknot_index]
                    d0 = t - k0
                    d1_val = k1 - t
                    need_insert = True
                    if d0 <= d1_val:
                        if d0 < Tolerance.ZERO_TOLERANCE:
                            nurbsknot_index -= 1
                            need_insert = False
                    else:
                        if d1_val < Tolerance.ZERO_TOLERANCE:
                            need_insert = False
                    if need_insert:
                        if not self.insert_nurbsknot(t, 1):
                            return False
                        kc = self.nurbsknot_count()
                        sc = self.span_count()
                        nurbsknot_index = -1
                        for i in range(kc):
                            if self.m_nurbsknot[i] > t + Tolerance.ZERO_TOLERANCE:
                                nurbsknot_index = i
                                break
                        if nurbsknot_index < 0:
                            return False
                    if nurbsknot_index >= p and nurbsknot_index < kc - p:
                        cvc = self.m_cv_count
                        distinct_cvc = cvc - p
                        cvdim = self.cv_size()
                        old_nurbsknots = list(self.m_nurbsknot)
                        old_cv = list(self.m_cv)

                        curr = p - 1
                        for i in range(nurbsknot_index, sc + p - 1):
                            self.m_nurbsknot[curr] = old_nurbsknots[i]
                            curr += 1
                        for i in range(nurbsknot_index - p + 2):
                            self.m_nurbsknot[curr] = old_nurbsknots[p - 1 + i] + dom_len
                            curr += 1
                        for i in range(p - 1):
                            self.m_nurbsknot[curr + i] = self.m_nurbsknot[curr + i - 1] + self.m_nurbsknot[p + i] - self.m_nurbsknot[p + i - 1]
                            self.m_nurbsknot[p - 2 - i] = self.m_nurbsknot[p - i - 1] - self.m_nurbsknot[curr - 1 - i] + self.m_nurbsknot[curr - 2 - i]

                        cv_id = nurbsknot_index - p + 1
                        for i in range(cvc):
                            src = cv_id % distinct_cvc
                            if src < 0:
                                src += distinct_cvc
                            for j in range(cvdim):
                                self.m_cv[i * self.m_cv_stride + j] = old_cv[src * self.m_cv_stride + j]
                            cv_id += 1

                        self.set_domain(t, t + dom_len)
                        self._invalidate_rmf_cache()
                        return True

        left, right = self.split(t)
        if left is None or right is None:
            return False

        shift = t1 - t0
        cvdim = self.cv_size()
        new_cv_count = right.m_cv_count + left.m_cv_count - 1
        new_kc = order + new_cv_count - 2

        new_cv = [0.0] * (new_cv_count * self.m_cv_stride)
        new_nurbsknots = [0.0] * new_kc

        for i in range(right.m_cv_count):
            for j in range(cvdim):
                new_cv[i * self.m_cv_stride + j] = right.m_cv[i * right.m_cv_stride + j]

        for i in range(1, left.m_cv_count):
            dst = right.m_cv_count + i - 1
            for j in range(cvdim):
                new_cv[dst * self.m_cv_stride + j] = left.m_cv[i * left.m_cv_stride + j]

        rkc = right.nurbsknot_count()
        for i in range(rkc):
            new_nurbsknots[i] = right.m_nurbsknot[i]

        lkc = left.nurbsknot_count()
        for i in range(order - 1, lkc):
            new_nurbsknots[rkc + i - (order - 1)] = left.m_nurbsknot[i] + shift

        self.m_cv_count = new_cv_count
        self.m_cv = new_cv
        self.m_nurbsknot = new_nurbsknots

        self.set_domain(t, t + dom_len)
        self._invalidate_rmf_cache()
        return True


    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════

    def transform(self, xform: Xform) -> bool:
        """Apply transformation to the curve.

        Parameters
        ----------
        xform : Xform
            Transformation to apply.

        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False

        m = xform.m
        cv = self.m_cv
        stride = self.m_cv_stride
        dim = self.m_dim

        for i in range(self.m_cv_count):
            base = i * stride
            if self.m_is_rat:
                wi = cv[base + dim]
                x = cv[base] / wi if abs(wi) > 1e-10 else cv[base]
                y = cv[base + 1] / wi if dim > 1 and abs(wi) > 1e-10 else (cv[base + 1] if dim > 1 else 0.0)
                z = cv[base + 2] / wi if dim > 2 and abs(wi) > 1e-10 else (cv[base + 2] if dim > 2 else 0.0)
            else:
                x = cv[base]
                y = cv[base + 1] if dim > 1 else 0.0
                z = cv[base + 2] if dim > 2 else 0.0

            w = m[3] * x + m[7] * y + m[11] * z + m[15]
            w_inv = 1.0 / w if abs(w) > 1e-10 else 1.0
            nx = (m[0] * x + m[4] * y + m[8] * z + m[12]) * w_inv
            ny = (m[1] * x + m[5] * y + m[9] * z + m[13]) * w_inv
            nz = (m[2] * x + m[6] * y + m[10] * z + m[14]) * w_inv

            if self.m_is_rat:
                cv[base] = nx * wi
                if dim > 1: cv[base + 1] = ny * wi
                if dim > 2: cv[base + 2] = nz * wi
            else:
                cv[base] = nx
                if dim > 1: cv[base + 1] = ny
                if dim > 2: cv[base + 2] = nz

        return True

    def transformed(self, xform: Xform) -> 'NurbsCurve':
        """Get transformed copy of the curve.

        Parameters
        ----------
        xform : Xform
            Transformation to apply.

        Returns
        -------
        NurbsCurve
            Transformed copy of the curve.
        """
        result = self.duplicate()
        result.transform(xform)

        return result


    # ═══════════════════════════════════════════════════════════════════════════
    # JSON Serialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self):
        """Return a JSON-serializable dictionary representation (matches C++ format)."""
        control_points = []
        for i in range(self.m_cv_count):
            if self.m_is_rat:
                # 4D for rational curves: dropping w loses the weights and the reloaded
                # curve is invalid (cv array too short for its stride).
                x, y, z, w = self.get_cv_4d(i)
                control_points.append([float(x), float(y), float(z), float(w)])
            else:
                p = self.get_cv(i)
                control_points.append([p[0], p[1], p[2]] if p else [0.0, 0.0, 0.0])
        return {
            "control_points": control_points,
            "cv_count": int(self.m_cv_count),
            "cv_stride": int(self.m_cv_stride),
            "dimension": int(self.m_dim),
            "guid": self.guid,
            "is_rational": self.m_is_rat != 0,
            "nurbsknots": self.m_nurbsknot.tolist() if hasattr(self.m_nurbsknot, 'tolist') else list(self.m_nurbsknot),
            "linecolors": [v for c in self.linecolors for v in (c.r, c.g, c.b, c.a)],
            "name": self.name,
            "order": int(self.m_order),
            "pointcolors": [v for c in self.pointcolors for v in (c.r, c.g, c.b, c.a)],
            "type": "NurbsCurve",
            "width": float(self.width),
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Create NurbsCurve from JSON dictionary (accepts C++ format)."""
        curve = cls()
        curve.guid = guid if guid is not None else data.get("guid", curve.guid)
        curve.name = name if name is not None else data.get("name", curve.name)
        curve.width = data.get("width", 1.0)
        if "pointcolors" in data:
            arr = data["pointcolors"]
            curve.pointcolors = [Color(arr[i], arr[i+1], arr[i+2], arr[i+3]) for i in range(0, len(arr) - 3, 4)]
        if "linecolors" in data:
            arr = data["linecolors"]
            curve.linecolors = [Color(arr[i], arr[i+1], arr[i+2], arr[i+3]) for i in range(0, len(arr) - 3, 4)]
        curve.m_dim = data.get("dimension", 0)
        curve.m_is_rat = 1 if data.get("is_rational", False) else 0
        curve.m_order = data.get("order", 0)
        curve.m_cv_count = data.get("cv_count", 0)
        curve.m_cv_stride = data.get("cv_stride", curve.m_dim + (1 if curve.m_is_rat else 0))
        curve.m_nurbsknot = np.array(data.get("nurbsknots", []), dtype=np.float64)
        control_points = data.get("control_points", [])
        flat_cv = []
        for cp in control_points:
            flat_cv.extend(cp[:curve.m_cv_stride])
        curve.m_cv = np.array(flat_cv, dtype=np.float64)
        return curve

    def file_json_dumps(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "NurbsCurve":
        """Load from JSON string."""
        import json
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write JSON to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "NurbsCurve":
        """Read JSON from file."""
        import json
        with open(filepath) as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def pb_dumps(self) -> bytes:
        """Convert to protobuf binary bytes."""
        from .proto import nurbscurve_pb2
        proto = nurbscurve_pb2.NurbsCurve()
        proto.guid = self.guid
        proto.name = self.name
        proto.dimension = int(self.m_dim)
        proto.is_rational = bool(self.m_is_rat)
        proto.order = int(self.m_order)
        proto.cv_count = int(self.m_cv_count)
        proto.cv_stride = int(self.m_cv_stride)
        proto.nurbsknots.extend(self.m_nurbsknot.tolist() if hasattr(self.m_nurbsknot, 'tolist') else list(self.m_nurbsknot))
        proto.cvs.extend(self.m_cv.tolist() if hasattr(self.m_cv, 'tolist') else list(self.m_cv))
        proto.width = float(self.width)
        from .proto import color_pb2
        for c in self.pointcolors:
            cp = proto.pointcolors.add()
            cp.r = int(c.r); cp.g = int(c.g); cp.b = int(c.b); cp.a = int(c.a)
        for c in self.linecolors:
            cp = proto.linecolors.add()
            cp.r = int(c.r); cp.g = int(c.g); cp.b = int(c.b); cp.a = int(c.a)
        return proto.SerializeToString()

    def pb_fill(self, proto: "nurbscurve_pb2.NurbsCurve") -> None:
        """Fill an existing NurbsCurve proto message directly (avoids serialize/deserialize cycle)."""
        proto.guid = self.guid
        proto.name = self.name
        proto.dimension = int(self.m_dim)
        proto.is_rational = bool(self.m_is_rat)
        proto.order = int(self.m_order)
        proto.cv_count = int(self.m_cv_count)
        proto.cv_stride = int(self.m_cv_stride)
        proto.nurbsknots.extend(self.m_nurbsknot.tolist() if hasattr(self.m_nurbsknot, 'tolist') else list(self.m_nurbsknot))
        proto.cvs.extend(self.m_cv.tolist() if hasattr(self.m_cv, 'tolist') else list(self.m_cv))
        proto.width = float(self.width)
        for c in self.pointcolors:
            cp = proto.pointcolors.add()
            cp.r = int(c.r); cp.g = int(c.g); cp.b = int(c.b); cp.a = int(c.a)
        for c in self.linecolors:
            cp = proto.linecolors.add()
            cp.r = int(c.r); cp.g = int(c.g); cp.b = int(c.b); cp.a = int(c.a)

    @classmethod
    def pb_loads(cls, data: bytes) -> "NurbsCurve":
        """Load from protobuf binary bytes."""
        from .proto import nurbscurve_pb2
        proto = nurbscurve_pb2.NurbsCurve()
        proto.ParseFromString(data)
        curve = cls()
        curve.guid = proto.guid
        curve.name = proto.name
        curve.m_dim = proto.dimension
        curve.m_is_rat = 1 if proto.is_rational else 0
        curve.m_order = proto.order
        curve.m_cv_count = proto.cv_count
        curve.m_cv_stride = proto.cv_stride
        curve.m_nurbsknot = np.array(list(proto.nurbsknots), dtype=np.float64)
        curve.m_cv = np.array(list(proto.cvs), dtype=np.float64)
        curve.width = proto.width if proto.width != 0.0 else 1.0
        curve.pointcolors = [Color(c.r, c.g, c.b, c.a) for c in proto.pointcolors]
        curve.linecolors = [Color(c.r, c.g, c.b, c.a) for c in proto.linecolors]
        return curve

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write protobuf to file."""
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "NurbsCurve":
        """Read protobuf from file."""
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())


    # ═══════════════════════════════════════════════════════════════════════════
    # String Representation
    # ═══════════════════════════════════════════════════════════════════════════

    def __str__(self) -> str:
        """String representation."""
        return f"NurbsCurve(name={self.name}, degree={self.degree()}, cvs={self.m_cv_count})"

    def __repr__(self) -> str:
        """Representation string."""
        rational_str = "true" if self.m_is_rat else "false"
        lines = [
            "NurbsCurve(",
            f"  name={self.name},",
            f"  degree={self.degree()},",
            f"  cvs={self.m_cv_count},",
            f"  rational={rational_str},",
            "  control_points=["
        ]
        for i in range(self.m_cv_count):
            p = self.get_cv(i)
            lines.append(f"    {p[0]}, {p[1]}, {p[2]}")
        lines.append("  ]")
        lines.append(")")
        return "\n".join(lines)


    # ═══════════════════════════════════════════════════════════════════════════
    # Internal Helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _find_span(self, t: float) -> int:
        """Find nurbsknot span index for parameter t using binary search.

        Implementation matches OpenNURBS ON_NurbsSpanIndex.

        Returns
        -------
        int
            Span index relative to shifted nurbsknot array (0-based from domain start)
        """
        if not self.is_valid():
            return -1

        # Use nurbsknot module function
        return nurbsknot.find_span(self.m_order, self.m_cv_count, self.m_nurbsknot, t)

    def _basis_functions(self, span: int, t: float) -> np.ndarray:
        """Compute non-zero basis functions at parameter t.
        
        Implementation matches OpenNURBS Cox-de Boor algorithm.
        
        Parameters
        ----------
        span : int
            NurbsKnot span index from _find_span() (relative to shifted array).
        t : float
            Parameter value.
            
        Returns
        -------
        np.ndarray
            Array of m_order non-zero basis function values.
        """
        N = np.zeros(self.m_order)
        left = np.zeros(self.m_order)
        right = np.zeros(self.m_order)
        
        # Offset nurbsknot pointer like OpenNURBS does
        offset = self.m_order - 2 + span
        
        N[0] = 1.0
        
        for j in range(1, self.m_order):
            left[j] = t - self.m_nurbsknot[offset + 1 - j]
            right[j] = self.m_nurbsknot[offset + j] - t
            saved = 0.0
            
            for r in range(j):
                denom = right[r + 1] + left[j - r]
                temp = N[r] / denom if denom != 0.0 else 0.0
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            
            N[j] = saved
        
        return N

    def _basis_functions_derivatives(self, span: int, t: float, deriv_order: int) -> np.ndarray:
        """Compute basis function derivatives at parameter t.

        Algorithm A2.3 from "The NURBS Book" (Piegl & Tiller).
        Matches OpenNURBS/Rhino implementation.

        Parameters
        ----------
        span : int
            NurbsKnot span index from _find_span().
        t : float
            Parameter value.
        deriv_order : int
            Maximum derivative order.

        Returns
        -------
        np.ndarray
            2D array [deriv_order+1, m_order] of basis function derivatives.
        """
        p = self.degree()
        n_der = min(deriv_order, p)

        ders = np.zeros((n_der + 1, p + 1))
        left = np.zeros(p + 1)
        right = np.zeros(p + 1)
        ndu = np.zeros((p + 1, p + 1))

        # Offset nurbsknot pointer like OpenNURBS
        offset = self.m_order - 2 + span

        ndu[0, 0] = 1.0
        for j in range(1, p + 1):
            left[j] = t - self.m_nurbsknot[offset + 1 - j]
            right[j] = self.m_nurbsknot[offset + j] - t
            saved = 0.0
            for r in range(j):
                # Store nurbsknot differences in ndu[j, r] for derivative computation
                ndu[j, r] = right[r + 1] + left[j - r]
                temp = ndu[r, j - 1] / ndu[j, r] if abs(ndu[j, r]) > 1e-14 else 0.0
                ndu[r, j] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            ndu[j, j] = saved

        # Load basis functions
        for j in range(p + 1):
            ders[0, j] = ndu[j, p]

        # Compute derivatives using Eq. 2.10 from The NURBS Book
        a = np.zeros((2, p + 1))
        for r in range(p + 1):
            s1 = 0
            s2 = 1
            a[0, 0] = 1.0

            for k in range(1, n_der + 1):
                d = 0.0
                rk = r - k
                pk = p - k

                if r >= k:
                    a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                    d = a[s2, 0] * ndu[rk, pk]

                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else p - r

                for j in range(j1, j2 + 1):
                    a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                    d += a[s2, j] * ndu[rk + j, pk]

                if r <= pk:
                    a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                    d += a[s2, k] * ndu[r, pk]

                ders[k, r] = d
                s1, s2 = s2, s1

        # Apply factorial scaling: p!/(p-k)! (falling factorial)
        factor = float(p)
        for k in range(1, n_der + 1):
            for j in range(p + 1):
                ders[k, j] *= factor
            factor *= (p - k)

        return ders

    def _evaluate_nurbs_de_boor_inplace(self, cvdim: int, order: int, cv_start: int, direction: int, t: float):
        """Internal de Boor evaluation for curve extension (modifies CVs in place)."""
        if order < 2:
            return

        stride = self.m_cv_stride
        for i in range(1, order):
            k0 = cv_start + i - 1 if direction > 0 else cv_start + order - i
            k1 = k0 + direction

            a = self.m_nurbsknot[cv_start + (order - 1 if direction > 0 else 0)]
            b = self.m_nurbsknot[cv_start + (i if direction > 0 else order - 1 - i)]

            if abs(b - a) < 1e-14:
                continue

            s = (t - a) / (b - a)

            for j in range(cvdim):
                cv0_val = self.m_cv[k0 * stride + j]
                cv1_val = self.m_cv[k1 * stride + j]
                self.m_cv[k0 * stride + j] = cv0_val + s * (cv0_val - cv1_val)

    def _zero_cvs(self) -> bool:
        """Zero all control vertices and set weights to 1 if rational.
        
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        
        self.m_cv.fill(0.0)
        
        if self.m_is_rat:
            for i in range(self.m_cv_count):
                self.m_cv[i * self.m_cv_stride + self.m_dim] = 1.0
        
        return True

    def _clean_nurbsknots(self, tolerance: float = 0.0) -> bool:
        """Clean up invalid nurbsknots (remove duplicates within tolerance).
        
        Parameters
        ----------
        tolerance : float, optional
            NurbsKnot comparison tolerance. Defaults to 0.0.
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        
        if tolerance <= 0.0:
            tolerance = Tolerance.ZERO_TOLERANCE
        
        # Remove nurbsknots that are too close together
        cleaned_nurbsknots = [self.m_nurbsknot[0]]
        for i in range(1, len(self.m_nurbsknot)):
            if abs(self.m_nurbsknot[i] - cleaned_nurbsknots[-1]) > tolerance:
                cleaned_nurbsknots.append(self.m_nurbsknot[i])
        
        if len(cleaned_nurbsknots) != len(self.m_nurbsknot):
            self.m_nurbsknot = np.array(cleaned_nurbsknots)
        
        return True

    def _span_is_linear(self, span_index: int, min_length: float = 0.0,
                       tolerance: float | None = None) -> bool:
        """Check if span is linear within tolerance.
        
        Parameters
        ----------
        span_index : int
            Index of the span.
        min_length : float, optional
            Minimum length to consider. Defaults to 0.0.
        tolerance : float, optional
            Tolerance for linearity. Defaults to Tolerance.ZERO_TOLERANCE.
            
        Returns
        -------
        bool
            True if span is linear.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE
        
        if not self.is_valid():
            return False
        
        spans = self.get_span_vector()
        if span_index < 0 or span_index >= len(spans) - 1:
            return False
        
        t0 = spans[span_index]
        t1 = spans[span_index + 1]
        
        p0 = self.point_at(t0)
        p1 = self.point_at(t1)
        
        length = p0.distance(p1)
        if length < min_length:
            return False
        
        # Check deviation from line
        num_samples = 5
        dt = (t1 - t0) / (num_samples - 1)
        
        for i in range(1, num_samples - 1):
            t = t0 + i * dt
            p = self.point_at(t)
            
            # Distance from point to line
            v = Vector(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
            w = Vector(p.x - p0.x, p.y - p0.y, p.z - p0.z)
            
            c1 = w.dot(v)
            c2 = v.dot(v)
            
            if c2 > Tolerance.ZERO_TOLERANCE:
                b = c1 / c2
                pb = Point(p0.x + b * v.x, p0.y + b * v.y, p0.z + b * v.z)
                dist = p.distance(pb)
                if dist > tolerance:
                    return False
        
        return True

    def _repair_bad_nurbsknots(self, tolerance: float = 0.0, repair: bool = True) -> bool:
        """Repair bad nurbsknots (too close, high multiplicity).
        
        Parameters
        ----------
        tolerance : float, optional
            NurbsKnot tolerance. Defaults to 0.0.
        repair : bool, optional
            If True, repairs nurbsknots; if False, only checks. Defaults to True.
            
        Returns
        -------
        bool
            True if nurbsknots are valid or repaired.
        """
        if not self.is_valid():
            return False
        
        if repair:
            return self._clean_nurbsknots(tolerance)
        
        # Just check
        for i in range(len(self.m_nurbsknot) - 1):
            if self.m_nurbsknot[i] > self.m_nurbsknot[i + 1] + Tolerance.ZERO_TOLERANCE:
                return False
        
        return True

    def _get_parameter_tolerance(self, t: float) -> tuple[float, float]:
        """Get parameter tolerance at point.
        
        Parameters
        ----------
        t : float
            Parameter value.
            
        Returns
        -------
        tuple[float, float]
            (t_minus, t_plus) tolerance bounds.
        """
        if not self.is_valid():
            return (0.0, 0.0)
        
        # Simple implementation: use small epsilon
        eps = Tolerance.ZERO_TOLERANCE * 10.0
        return (t - eps, t + eps)

