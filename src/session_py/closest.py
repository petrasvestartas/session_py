"""Closest point operations for geometry classes."""

from typing import Tuple
import math
import numpy as np

from session_py.point import Point
from session_py.vector import Vector
from session_py.line import Line
from session_py.polyline import Polyline


class Closest:
    """Static methods for finding closest points between geometry objects."""

    @staticmethod
    def curve_point(curve, test_point: Point, t0: float = 0.0, t1: float = 0.0) -> Tuple[float, float]:
        """Find closest point on NURBS curve to test point.

        Parameters
        ----------
        curve : NurbsCurve
            The NURBS curve.
        test_point : Point
            Point to find closest curve point to.
        t0 : float
            Start of search interval. 0 means curve start.
        t1 : float
            End of search interval. 0 means curve end.

        Returns
        -------
        tuple of (float, float)
            (parameter, distance) of closest point.
        """
        if not curve.is_valid():
            return (0.0, float('inf'))

        domain_start, domain_end = curve.domain()
        if t0 <= 0.0:
            t0 = domain_start
        if t1 <= 0.0:
            t1 = domain_end

        t0 = max(t0, domain_start)
        t1 = min(t1, domain_end)

        # Dense seed grid: sample every knot span several times so the global
        # minimum's basin is captured before Newton refines (matches OCCT's robust
        # initial sampling in GeomAPI_ProjectPointOnCurve).
        num_samples = max(50, curve.cv_count() * 10)
        dt = (t1 - t0) / num_samples

        best_t = t0
        best_dist = curve.point_at(t0).distance(test_point)

        for i in range(num_samples + 1):
            t = t0 + i * dt
            dist = curve.point_at(t).distance(test_point)
            if dist < best_dist:
                best_dist = dist
                best_t = t

        max_iterations = 32
        step_tolerance = (t1 - t0) * 1e-12

        t = best_t

        # Newton on h(t) = (C(t) - P) . C'(t)  (= 0 at a foot of perpendicular).
        # h'(t) = |C'(t)|^2 + (C(t) - P) . C''(t).  Use the RAW derivatives C', C''.
        for _ in range(max_iterations):
            derivs = curve.evaluate(t, 2)
            if len(derivs) < 3:
                break
            pt = derivs[0]
            d1 = derivs[1]
            d2 = derivs[2]

            rx = pt[0] - test_point[0]
            ry = pt[1] - test_point[1]
            rz = pt[2] - test_point[2]

            f = rx * d1[0] + ry * d1[1] + rz * d1[2]

            if abs(f) < step_tolerance:
                break

            df = (d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
                  + rx * d2[0] + ry * d2[1] + rz * d2[2])

            if abs(df) < 1e-14:
                break

            dt_step = -f / df

            if abs(dt_step) > (t1 - t0) * 0.5:
                dt_step = math.copysign((t1 - t0) * 0.5, dt_step)

            t += dt_step

            if t < t0:
                t = t0
            if t > t1:
                t = t1

            if abs(dt_step) < step_tolerance:
                break

        final_dist = curve.point_at(t).distance(test_point)

        dist_start = curve.point_at(t0).distance(test_point)
        dist_end = curve.point_at(t1).distance(test_point)

        if dist_start < final_dist:
            t = t0
            final_dist = dist_start
        if dist_end < final_dist:
            t = t1
            final_dist = dist_end

        return (t, final_dist)

    @staticmethod
    def curve_curve(curve0, curve1) -> Tuple[float, float, float]:
        """Closest approach between two NURBS curves (dense grid seed + 2D Newton).

        Minimizes f(u,v) = |C0(u) - C1(v)|^2. Returns (u, v, distance).
        Matches OCCT GeomAPI_ExtremaCurveCurve.
        """
        if not curve0.is_valid() or not curve1.is_valid():
            return (0.0, 0.0, float('inf'))

        u0, u1 = curve0.domain()
        v0, v1 = curve1.domain()
        n0 = max(40, curve0.cv_count() * 8)
        n1 = max(40, curve1.cv_count() * 8)

        us = np.linspace(u0, u1, n0 + 1)
        vs = np.linspace(v0, v1, n1 + 1)
        p0 = curve0._batch_point_at(us)   # (n0+1, 3)
        p1 = curve1._batch_point_at(vs)   # (n1+1, 3)
        # Pairwise squared distances: (n0+1, n1+1)
        diff = p0[:, None, :] - p1[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        u, v = float(us[i]), float(vs[j])

        for _ in range(64):
            e0 = curve0.evaluate(u, 2)
            e1 = curve1.evaluate(v, 2)
            c0, c0p, c0pp = e0[0], e0[1], e0[2]
            c1, c1p, c1pp = e1[0], e1[1], e1[2]
            rx = c0[0] - c1[0]; ry = c0[1] - c1[1]; rz = c0[2] - c1[2]

            gu = rx * c0p[0] + ry * c0p[1] + rz * c0p[2]          # 0.5 df/du
            gv = -(rx * c1p[0] + ry * c1p[1] + rz * c1p[2])        # 0.5 df/dv

            huu = (c0p[0] ** 2 + c0p[1] ** 2 + c0p[2] ** 2
                   + rx * c0pp[0] + ry * c0pp[1] + rz * c0pp[2])
            huv = -(c0p[0] * c1p[0] + c0p[1] * c1p[1] + c0p[2] * c1p[2])
            hvv = (c1p[0] ** 2 + c1p[1] ** 2 + c1p[2] ** 2
                   - (rx * c1pp[0] + ry * c1pp[1] + rz * c1pp[2]))

            det = huu * hvv - huv * huv
            if abs(det) < 1e-14:
                break
            du = -(hvv * gu - huv * gv) / det
            dv = -(-huv * gu + huu * gv) / det

            # Limit step to half the domain to stay in the basin.
            du = math.copysign(min(abs(du), (u1 - u0) * 0.5), du) if du else 0.0
            dv = math.copysign(min(abs(dv), (v1 - v0) * 0.5), dv) if dv else 0.0

            u = min(max(u + du, u0), u1)
            v = min(max(v + dv, v0), v1)
            if max(abs(du), abs(dv)) < 1e-13:
                break

        dist = curve0.point_at(u).distance(curve1.point_at(v))
        return (u, v, dist)

    @staticmethod
    def line_point(line: Line, test_point: Point) -> Tuple[Point, float, float]:
        """Find closest point on line to test point.

        Parameters
        ----------
        line : Line
            The line segment.
        test_point : Point
            Point to find closest line point to.

        Returns
        -------
        tuple of (Point, float, float)
            (closest_point, parameter, distance).
        """
        start = line.start()
        end = line.end()

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]

        len_sq = dx * dx + dy * dy + dz * dz

        if len_sq < 1e-20:
            dist = start.distance(test_point)
            return (start, 0.0, dist)

        t = ((test_point[0] - start[0]) * dx +
             (test_point[1] - start[1]) * dy +
             (test_point[2] - start[2]) * dz) / len_sq

        t = max(0.0, min(1.0, t))

        closest = Point(
            start[0] + t * dx,
            start[1] + t * dy,
            start[2] + t * dz
        )

        dist = closest.distance(test_point)

        return (closest, t, dist)

    @staticmethod
    def polyline_point(polyline: Polyline, test_point: Point) -> Tuple[Point, float, float]:
        """Find closest point on polyline to test point.

        Parameters
        ----------
        polyline : Polyline
            The polyline.
        test_point : Point
            Point to find closest polyline point to.

        Returns
        -------
        tuple of (Point, float, float)
            (closest_point, parameter, distance).
        """
        points = polyline.get_points()

        if not points:
            return (Point(0, 0, 0), 0.0, float('inf'))

        if len(points) == 1:
            dist = points[0].distance(test_point)
            return (points[0], 0.0, dist)

        best_point = points[0]
        best_param = 0.0
        best_dist = float('inf')

        cumulative_length = 0.0
        total_length = polyline.length()

        for i in range(len(points) - 1):
            segment = Line.from_points(points[i], points[i + 1])
            closest, t, dist = Closest.line_point(segment, test_point)

            if dist < best_dist:
                best_dist = dist
                best_point = closest
                segment_length = segment.length()
                if total_length > 1e-20:
                    best_param = (cumulative_length + t * segment_length) / total_length
                else:
                    best_param = float(i) / (len(points) - 1)

            cumulative_length += segment.length()

        return (best_point, best_param, best_dist)

    @staticmethod
    def surface_point(surface, test_point: Point, u0: float = 0.0, u1: float = 0.0,
                      v0: float = 0.0, v1: float = 0.0) -> Tuple[float, float, float]:
        """Find closest point on NURBS surface to test point.

        Parameters
        ----------
        surface : NurbsSurface
            The NURBS surface.
        test_point : Point
            Point to find closest surface point to.
        u0, u1 : float
            U parameter search interval. 0 means use surface domain.
        v0, v1 : float
            V parameter search interval. 0 means use surface domain.

        Returns
        -------
        tuple of (float, float, float)
            (u_param, v_param, distance).
        """
        if not surface.is_valid():
            return (0.0, 0.0, float('inf'))

        domain_u0, domain_u1 = surface.domain(0)
        domain_v0, domain_v1 = surface.domain(1)

        if u0 <= 0.0:
            u0 = domain_u0
        if u1 <= 0.0:
            u1 = domain_u1
        if v0 <= 0.0:
            v0 = domain_v0
        if v1 <= 0.0:
            v1 = domain_v1

        u0 = max(u0, domain_u0)
        u1 = min(u1, domain_u1)
        v0 = max(v0, domain_v0)
        v1 = min(v1, domain_v1)

        u_samples = max(10, surface.order(0))
        v_samples = max(10, surface.order(1))

        du_param = (u1 - u0) / u_samples
        dv_param = (v1 - v0) / v_samples

        best_u = u0
        best_v = v0
        best_dist = float('inf')

        for i in range(u_samples + 1):
            for j in range(v_samples + 1):
                uu = u0 + i * du_param
                vv = v0 + j * dv_param
                pt = surface.point_at(uu, vv)
                dist = pt.distance(test_point)
                if dist < best_dist:
                    best_dist = dist
                    best_u = uu
                    best_v = vv

        max_iterations = 20
        step_tolerance = min(u1 - u0, v1 - v0) * 1e-10

        u = best_u
        v = best_v

        for _ in range(max_iterations):
            derivs = surface.evaluate(u, v, 1)
            if len(derivs) < 3:
                break

            pt = surface.point_at(u, v)
            du_vec = derivs[2]  # evaluate returns [S, Sv, Su, ...]
            dv_vec = derivs[1]

            delta = Vector(
                test_point[0] - pt[0],
                test_point[1] - pt[1],
                test_point[2] - pt[2]
            )

            fu = -delta.dot(du_vec)
            fv = -delta.dot(dv_vec)

            if abs(fu) < step_tolerance and abs(fv) < step_tolerance:
                break

            duu = du_vec.dot(du_vec)
            dvv = dv_vec.dot(dv_vec)
            duv = du_vec.dot(dv_vec)

            det = duu * dvv - duv * duv
            if abs(det) < 1e-12:
                break

            du_step = (dvv * fu - duv * fv) / det
            dv_step = (duu * fv - duv * fu) / det

            max_step = min(u1 - u0, v1 - v0) * 0.5
            if abs(du_step) > max_step:
                du_step = math.copysign(max_step, du_step)
            if abs(dv_step) > max_step:
                dv_step = math.copysign(max_step, dv_step)

            u -= du_step
            v -= dv_step

            u = max(u0, min(u1, u))
            v = max(v0, min(v1, v))

            if abs(du_step) < step_tolerance and abs(dv_step) < step_tolerance:
                break

        final_dist = surface.point_at(u, v).distance(test_point)

        return (u, v, final_dist)

    @staticmethod
    def surface_curve(surface, curve, t0: float = 0.0, t1: float = 0.0, tolerance=None):
        """Project a 3D curve onto a surface (curve pullback).

        Samples the curve, inverts each sample with warm-started windowed
        point inversion, unwraps across seams of closed surfaces, refines
        adaptively, and refits seam-split UV pcurves (x=u, y=v, z=0) with
        domain [0, 1]. Returns an empty list if the curve does not lie on the
        surface within the rejection tolerance.

        Parameters
        ----------
        surface : NurbsSurface
            The surface to project onto.
        curve : NurbsCurve
            The 3D curve to pull back.
        t0, t1 : float
            Curve sub-domain. 0 means use the curve domain end.
        tolerance : float, optional
            Fit deviation budget. Defaults to a trace-step heuristic.

        Returns
        -------
        list of NurbsCurve
            Seam-split UV pcurves.
        """
        from session_py.nurbscurve import NurbsCurve
        from session_py.nurbsknot import CurveNurbsKnotStyle

        if not surface.is_valid() or not curve.is_valid():
            return []

        u0, u1 = surface.domain(0)
        v0, v1 = surface.domain(1)
        range_u = u1 - u0
        range_v = v1 - v0
        closed_u = surface.is_closed(0)
        closed_v = surface.is_closed(1)

        ct0, ct1 = curve.domain()
        if t0 <= 0.0:
            t0 = ct0
        if t1 <= 0.0:
            t1 = ct1
        t0 = max(t0, ct0)
        t1 = min(t1, ct1)
        if t1 - t0 < 1e-14:
            return []

        spans_u = surface.get_span_vector(0)
        spans_v = surface.get_span_vector(1)
        nu = max(len(spans_u) - 1, 1) * 4
        nv = max(len(spans_v) - 1, 1) * 4
        du = range_u / nu
        dv = range_v / nv

        mu = (u0 + u1) * 0.5
        mv = (v0 + v1) * 0.5
        pmid = surface.point_at(mu, mv)
        wu_probe = min(mu + du, u1)
        wv_probe = min(mv + dv, v1)
        uv_to_3d_u = pmid.distance(surface.point_at(wu_probe, mv)) / du
        uv_to_3d_v = pmid.distance(surface.point_at(mu, wv_probe)) / dv
        uv_to_3d = max(uv_to_3d_u, uv_to_3d_v)
        uv_to_3d_min = min(uv_to_3d_u, uv_to_3d_v)
        if uv_to_3d < 1e-10:
            uv_to_3d = 1.0
        if uv_to_3d_min < 1e-10:
            uv_to_3d_min = 1.0

        step = min(du, dv) * 0.25
        if tolerance is not None and tolerance > 0.0:
            fit_tol = tolerance
        else:
            fit_tol = step * (uv_to_3d + uv_to_3d_min) * 0.5
        reject_tol = fit_tol * 100.0
        # Absolute "lies on the surface" gate (fraction of the surface size).
        # Used to (a) reject a curve that nowhere touches the surface and
        # (b) stop bisecting stick-out portions of a curve that extends past
        # the face, both of which otherwise burn a full 4096-sample bisection.
        corner_diag = surface.point_at(u0, v0).distance(surface.point_at(u1, v1))
        if corner_diag < 1e-12:
            corner_diag = max(range_u, range_v)
        on_surf_tol = corner_diag * 0.05

        def wrap_u(u):
            if closed_u:
                t = math.fmod(u - u0, range_u)
                if t < 0:
                    t += range_u
                return u0 + t
            return max(u0, min(u, u1))

        def wrap_v(v):
            if closed_v:
                t = math.fmod(v - v0, range_v)
                if t < 0:
                    t += range_v
                return v0 + t
            return max(v0, min(v, v1))

        def invert_near(pt, up, vp, wu, wv):
            # Windowed inversion with seam-aware candidate windows
            u_centers = [up]
            if closed_u:
                if up - wu < u0:
                    u_centers.append(up + range_u)
                if up + wu > u1:
                    u_centers.append(up - range_u)
            v_centers = [vp]
            if closed_v:
                if vp - wv < v0:
                    v_centers.append(vp + range_v)
                if vp + wv > v1:
                    v_centers.append(vp - range_v)
            best = (up, vp, float('inf'))
            for uc in u_centers:
                for vc in v_centers:
                    wu0 = max(uc - wu, u0)
                    wu1 = min(uc + wu, u1)
                    wv0 = max(vc - wv, v0)
                    wv1 = min(vc + wv, v1)
                    if wu1 - wu0 < 1e-14 or wv1 - wv0 < 1e-14:
                        continue
                    res = Closest.surface_point(surface, pt, wu0, wu1, wv0, wv1)
                    if res[2] < best[2]:
                        best = res
                    if best[2] < fit_tol * 0.01:
                        break
            return best

        def unwrap_to(prev_u, prev_v, u, v):
            if closed_u:
                while u - prev_u > range_u * 0.5:
                    u -= range_u
                while u - prev_u < -range_u * 0.5:
                    u += range_u
            if closed_v:
                while v - prev_v > range_v * 0.5:
                    v -= range_v
                while v - prev_v < -range_v * 0.5:
                    v += range_v
            return u, v

        # 1. Initial samples with warm-started inversion
        n0 = max(16, 4 * curve.span_count())
        samples = []  # list of [t, u_unwrapped, v_unwrapped, residual]
        max_residual = 0.0
        min_residual = float('inf')
        for i in range(n0 + 1):
            t = t0 + (t1 - t0) * i / n0
            pt = curve.point_at(t)
            if i == 0:
                ru, rv, rd = Closest.surface_point(surface, pt, 0.0, 0.0, 0.0, 0.0)
                uu, vv = ru, rv
            else:
                prev = samples[-1]
                wu = max(du, dv) * 2.0 + abs(prev[1] - samples[max(0, len(samples)-2)][1])
                wv = max(du, dv) * 2.0 + abs(prev[2] - samples[max(0, len(samples)-2)][2])
                ru, rv, rd = invert_near(pt, wrap_u(prev[1]), wrap_v(prev[2]), wu, wv)
                if rd > reject_tol:
                    ru, rv, rd = Closest.surface_point(surface, pt, 0.0, 0.0, 0.0, 0.0)
                uu, vv = unwrap_to(prev[1], prev[2], ru, rv)
            samples.append([t, uu, vv, rd])
            max_residual = max(max_residual, rd)
            min_residual = min(min_residual, rd)

        # Reject a curve that nowhere lies on the surface (no sample touches it).
        if max_residual > reject_tol or min_residual > on_surf_tol:
            return []

        # 2. Adaptive bisection where the lifted UV midpoint strays from the curve
        depth = 0
        while depth < 8:
            inserted = 0
            i = 0
            while i < len(samples) - 1:
                a = samples[i]
                b = samples[i + 1]
                tm = (a[0] + b[0]) * 0.5
                um = (a[1] + b[1]) * 0.5
                vm = (a[2] + b[2]) * 0.5
                pm = curve.point_at(tm)
                lift = surface.point_at(wrap_u(um), wrap_v(vm))
                if lift.distance(pm) > fit_tol and len(samples) < 4096:
                    wu = max(abs(b[1] - a[1]), du) * 1.0
                    wv = max(abs(b[2] - a[2]), dv) * 1.0
                    ru, rv, rd = invert_near(pm, wrap_u(um), wrap_v(vm), wu, wv)
                    if rd > on_surf_tol:
                        # Midpoint is off the surface: this is a stick-out
                        # portion of a curve that extends past the face, not a
                        # curvature stray. Do not refine it (avoids an
                        # unbounded bisection of an off-surface segment).
                        i += 1
                        continue
                    uu, vv = unwrap_to(a[1], a[2], ru, rv)
                    samples.insert(i + 1, [tm, uu, vv, rd])
                    inserted += 1
                    i += 2
                else:
                    i += 1
            if inserted == 0:
                break
            depth += 1

        pts = [[s[1], s[2]] for s in samples]

        # 3. Closed-loop closure and seam-crossing split (same scheme as surface_plane_uv)
        p_first = curve.point_at(samples[0][0])
        p_last = curve.point_at(samples[-1][0])
        is_loop = p_first.distance(p_last) < fit_tol * 4.0 and len(pts) >= 6

        closure_du = 0.0
        closure_dv = 0.0
        if is_loop:
            pts = pts[:-1]
            du_j = pts[0][0] - pts[-1][0]
            dv_j = pts[0][1] - pts[-1][1]
            if closed_u:
                while du_j > range_u * 0.5:
                    du_j -= range_u
                while du_j < -range_u * 0.5:
                    du_j += range_u
            if closed_v:
                while dv_j > range_v * 0.5:
                    dv_j -= range_v
                while dv_j < -range_v * 0.5:
                    dv_j += range_v
            closure_du = (pts[-1][0] + du_j) - pts[0][0]
            closure_dv = (pts[-1][1] + dv_j) - pts[0][1]
            pts.append([pts[0][0] + closure_du, pts[0][1] + closure_dv])

        out_pts = [pts[0]]
        cross_idx = []
        for i in range(1, len(pts)):
            pa = pts[i - 1]
            pb = pts[i]
            crossings = []
            if closed_u and abs(pb[0] - pa[0]) > 1e-15:
                k0 = math.floor((pa[0] - u0) / range_u)
                k1 = math.floor((pb[0] - u0) / range_u)
                for k in range(min(k0, k1) + 1, max(k0, k1) + 1):
                    L = u0 + k * range_u
                    t = (L - pa[0]) / (pb[0] - pa[0])
                    if 0.0 < t < 1.0:
                        crossings.append((t, 0, L))
            if closed_v and abs(pb[1] - pa[1]) > 1e-15:
                k0 = math.floor((pa[1] - v0) / range_v)
                k1 = math.floor((pb[1] - v0) / range_v)
                for k in range(min(k0, k1) + 1, max(k0, k1) + 1):
                    L = v0 + k * range_v
                    t = (L - pa[1]) / (pb[1] - pa[1])
                    if 0.0 < t < 1.0:
                        crossings.append((t, 1, L))
            crossings.sort()
            for t, axis, L in crossings:
                cu = pa[0] + (pb[0] - pa[0]) * t
                cv_ = pa[1] + (pb[1] - pa[1]) * t
                if axis == 0:
                    cu = L
                else:
                    cv_ = L
                out_pts.append([cu, cv_])
                cross_idx.append(len(out_pts) - 1)
            out_pts.append([pb[0], pb[1]])
            # An interior sample sitting exactly on a seam level is a crossing
            if i < len(pts) - 1:
                on_seam = False
                if closed_u:
                    k = round((pb[0] - u0) / range_u)
                    L = u0 + k * range_u
                    if abs(pb[0] - L) < range_u * 1e-9 and abs(pb[0] - pa[0]) > range_u * 1e-9:
                        out_pts[-1][0] = L
                        on_seam = True
                if closed_v:
                    k = round((pb[1] - v0) / range_v)
                    L = v0 + k * range_v
                    if abs(pb[1] - L) < range_v * 1e-9 and abs(pb[1] - pa[1]) > range_v * 1e-9:
                        out_pts[-1][1] = L
                        on_seam = True
                if on_seam:
                    cross_idx.append(len(out_pts) - 1)

        wrap_drift = abs(closure_du) > range_u * 0.5 or abs(closure_dv) > range_v * 0.5
        if len(cross_idx) == 0:
            pieces = [(out_pts, is_loop and not wrap_drift)]
        else:
            pieces = []
            if is_loop:
                for a, b in zip(cross_idx, cross_idx[1:]):
                    pieces.append((out_pts[a:b + 1], False))
                wrap_piece = [list(p) for p in out_pts[cross_idx[-1]:]]
                for p in out_pts[1:cross_idx[0] + 1]:
                    wrap_piece.append([p[0] + closure_du, p[1] + closure_dv])
                pieces.append((wrap_piece, False))
            else:
                bounds = [0] + cross_idx + [len(out_pts) - 1]
                for a, b in zip(bounds, bounds[1:]):
                    if b > a:
                        pieces.append((out_pts[a:b + 1], False))

        # 4. Refit each piece as a UV pcurve
        result = []
        for piece_pts, piece_loop in pieces:
            if len(piece_pts) < 2:
                continue
            mid = piece_pts[len(piece_pts) // 2]
            if closed_u:
                k_u = math.floor((mid[0] - u0) / range_u)
                if k_u != 0:
                    for p in piece_pts:
                        p[0] -= k_u * range_u
            if closed_v:
                k_v = math.floor((mid[1] - v0) / range_v)
                if k_v != 0:
                    for p in piece_pts:
                        p[1] -= k_v * range_v

            pts_uv = [Point(p[0], p[1], 0.0) for p in piece_pts]
            mp = len(pts_uv)
            fit_tol_uv = step
            total_turning = 0.0
            for i in range(1, mp - 1):
                dx1 = pts_uv[i][0] - pts_uv[i-1][0]
                dy1 = pts_uv[i][1] - pts_uv[i-1][1]
                dx2 = pts_uv[i+1][0] - pts_uv[i][0]
                dy2 = pts_uv[i+1][1] - pts_uv[i][1]
                l1 = math.hypot(dx1, dy1)
                l2 = math.hypot(dx2, dy2)
                if l1 > 1e-14 and l2 > 1e-14:
                    c = (dx1*dx2 + dy1*dy2) / (l1*l2)
                    c = max(-1.0, min(1.0, c))
                    total_turning += math.acos(c)

            chords = [0.0] * mp
            total_len = 0.0
            for i in range(1, mp):
                total_len += pts_uv[i].distance(pts_uv[i-1])
                chords[i] = total_len
            if piece_loop and mp > 1:
                total_len += pts_uv[0].distance(pts_uv[mp-1])
            if total_len > 1e-14:
                for i in range(1, mp):
                    chords[i] /= total_len

            target_cvs = max(8, int(total_turning / 0.5) + 6)
            max_cvs = mp - 1
            pcurve = NurbsCurve()
            for attempt in range(5):
                if target_cvs > max_cvs:
                    break
                pcurve = NurbsCurve.create_fitted(pts_uv, target_cvs, 3, piece_loop)
                if not pcurve.is_valid():
                    break
                ft0, ft1 = pcurve.domain()
                max_dev = 0.0
                for i in range(mp):
                    t = ft0 + (ft1 - ft0) * chords[i]
                    max_dev = max(max_dev, pcurve.point_at(t).distance(pts_uv[i]))
                if max_dev < fit_tol_uv:
                    break
                target_cvs = min(target_cvs * 2, max_cvs)

            if not pcurve.is_valid():
                if piece_loop:
                    pcurve = NurbsCurve.create_interpolated(pts_uv, CurveNurbsKnotStyle.ChordPeriodic)
                else:
                    pcurve = NurbsCurve.create_interpolated(pts_uv)
            if not pcurve.is_valid() and len(pts_uv) >= 2:
                # Last resort: a degree-1 polyline through the inverted UV samples
                # (always valid; lies on the surface piecewise-linearly in UV).
                pcurve = NurbsCurve.create(False, 1, pts_uv)
            if not pcurve.is_valid():
                continue

            pcurve.set_domain(0.0, 1.0)
            result.append(pcurve)

        return result

    @staticmethod
    def _closest_point_on_triangle(p, a, b, c):
        abx, aby, abz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
        acx, acy, acz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
        apx, apy, apz = p[0]-a[0], p[1]-a[1], p[2]-a[2]

        d1 = abx*apx + aby*apy + abz*apz
        d2 = acx*apx + acy*apy + acz*apz
        if d1 <= 0.0 and d2 <= 0.0:
            return a

        bpx, bpy, bpz = p[0]-b[0], p[1]-b[1], p[2]-b[2]
        d3 = abx*bpx + aby*bpy + abz*bpz
        d4 = acx*bpx + acy*bpy + acz*bpz
        if d3 >= 0.0 and d4 <= d3:
            return b

        vc = d1*d4 - d3*d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return Point(a[0] + v*abx, a[1] + v*aby, a[2] + v*abz)

        cpx, cpy, cpz = p[0]-c[0], p[1]-c[1], p[2]-c[2]
        d5 = abx*cpx + aby*cpy + abz*cpz
        d6 = acx*cpx + acy*cpy + acz*cpz
        if d6 >= 0.0 and d5 <= d6:
            return c

        vb = d5*d2 - d1*d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return Point(a[0] + w*acx, a[1] + w*acy, a[2] + w*acz)

        va = d3*d6 - d5*d4
        if va <= 0.0 and (d4-d3) >= 0.0 and (d5-d6) >= 0.0:
            w = (d4-d3) / ((d4-d3) + (d5-d6))
            return Point(b[0] + w*(c[0]-b[0]), b[1] + w*(c[1]-b[1]), b[2] + w*(c[2]-b[2]))

        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return Point(a[0] + abx*v + acx*w, a[1] + aby*v + acy*w, a[2] + abz*v + acz*w)

    @staticmethod
    def _aabb_min_distance(aabb, p):
        dx = max(0.0, max(aabb.cx - aabb.hx - p[0], p[0] - aabb.cx - aabb.hx))
        dy = max(0.0, max(aabb.cy - aabb.hy - p[1], p[1] - aabb.cy - aabb.hy))
        dz = max(0.0, max(aabb.cz - aabb.hz - p[2], p[2] - aabb.cz - aabb.hz))
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    @staticmethod
    def mesh_point(mesh, test_point):
        import heapq

        if mesh.number_of_faces() == 0:
            return (Point(0, 0, 0), 0, float('inf'))

        mesh.build_triangle_bvh()
        bvh = mesh.get_cached_bvh()

        face_keys = sorted(mesh.face.keys())

        best_point = Point(0, 0, 0)
        best_face_key = 0
        best_dist = float('inf')

        if bvh is None or bvh.root is None:
            return (best_point, best_face_key, best_dist)

        counter = 0
        pq = [(Closest._aabb_min_distance(bvh.root.aabb, test_point), counter, bvh.root)]

        while pq:
            d, _, node = heapq.heappop(pq)
            if d >= best_dist:
                break

            if node.is_leaf():
                found, face_idx, sub_idx, v0, v1, v2 = mesh.get_triangle_by_id(node.object_id)
                if found:
                    cp = Closest._closest_point_on_triangle(test_point, v0, v1, v2)
                    dist = cp.distance(test_point)
                    if dist < best_dist:
                        best_dist = dist
                        best_point = cp
                        best_face_key = face_keys[face_idx]
            else:
                if node.left is not None:
                    ld = Closest._aabb_min_distance(node.left.aabb, test_point)
                    if ld < best_dist:
                        counter += 1
                        heapq.heappush(pq, (ld, counter, node.left))
                if node.right is not None:
                    rd = Closest._aabb_min_distance(node.right.aabb, test_point)
                    if rd < best_dist:
                        counter += 1
                        heapq.heappush(pq, (rd, counter, node.right))

        return (best_point, best_face_key, best_dist)

    @staticmethod
    def mesh_point_aabb(mesh, test_point):
        if mesh.number_of_faces() == 0:
            return (Point(0, 0, 0), 0, float('inf'))

        vertices, faces = mesh.to_vertices_and_faces()
        sorted_face_keys = sorted(mesh.face.keys())

        tris = []
        tri_face_idx = []
        for fi, fv in enumerate(faces):
            if len(fv) < 3:
                continue
            v0 = vertices[fv[0]]
            for j in range(1, len(fv) - 1):
                tris.append((v0, vertices[fv[j]], vertices[fv[j + 1]]))
                tri_face_idx.append(fi)

        if not tris:
            return (Point(0, 0, 0), 0, float('inf'))

        boxes = []
        for v0, v1, v2 in tris:
            lx = min(v0[0], v1[0], v2[0])
            ly = min(v0[1], v1[1], v2[1])
            lz = min(v0[2], v1[2], v2[2])
            hx = max(v0[0], v1[0], v2[0])
            hy = max(v0[1], v1[1], v2[1])
            hz = max(v0[2], v1[2], v2[2])
            boxes.append(((lx+hx)*0.5, (ly+hy)*0.5, (lz+hz)*0.5,
                          (hx-lx)*0.5, (hy-ly)*0.5, (hz-lz)*0.5))

        nodes = []

        def build_node(ids):
            ni = len(nodes)
            nodes.append([None, -1, -1])
            lx = ly = lz = 1e308
            hx = hy = hz = -1e308
            for i in ids:
                b = boxes[i]
                lx = min(lx, b[0]-b[3]); hx = max(hx, b[0]+b[3])
                ly = min(ly, b[1]-b[4]); hy = max(hy, b[1]+b[4])
                lz = min(lz, b[2]-b[5]); hz = max(hz, b[2]+b[5])
            nodes[ni][0] = ((lx+hx)*0.5, (ly+hy)*0.5, (lz+hz)*0.5,
                            (hx-lx)*0.5, (hy-ly)*0.5, (hz-lz)*0.5)
            if len(ids) == 1:
                nodes[ni][2] = ids[0]
                return
            dx, dy, dz = hx-lx, hy-ly, hz-lz
            axis = 0 if dx >= dy and dx >= dz else (1 if dy >= dz else 2)
            ids.sort(key=lambda i: boxes[i][axis])
            mid = len(ids) // 2
            build_node(ids[:mid])
            nodes[ni][1] = len(nodes)
            build_node(ids[mid:])

        build_node(list(range(len(tris))))

        def aabb_min_dist(aabb, pt):
            dx = max(0.0, abs(pt[0] - aabb[0]) - aabb[3])
            dy = max(0.0, abs(pt[1] - aabb[1]) - aabb[4])
            dz = max(0.0, abs(pt[2] - aabb[2]) - aabb[5])
            return (dx*dx + dy*dy + dz*dz) ** 0.5

        best = [Point(0, 0, 0), 0, float('inf')]

        def dfs(ni):
            aabb, right, obj = nodes[ni]
            if aabb_min_dist(aabb, test_point) >= best[2]:
                return
            if obj >= 0:
                v0, v1, v2 = tris[obj]
                cp = Closest._closest_point_on_triangle(test_point, v0, v1, v2)
                d = cp.distance(test_point)
                if d < best[2]:
                    best[0] = cp
                    best[1] = sorted_face_keys[tri_face_idx[obj]]
                    best[2] = d
                return
            left = ni + 1
            ld = aabb_min_dist(nodes[left][0], test_point)
            rd = aabb_min_dist(nodes[right][0], test_point)
            if ld <= rd:
                if ld < best[2]: dfs(left)
                if rd < best[2]: dfs(right)
            else:
                if rd < best[2]: dfs(right)
                if ld < best[2]: dfs(left)

        dfs(0)
        return tuple(best)

    @staticmethod
    def pointcloud_point(cloud, test_point):
        if cloud.point_count() == 0:
            return (Point(0, 0, 0), 0, float('inf'))

        best_point = cloud.get_point(0)
        best_index = 0
        best_dist = best_point.distance(test_point)

        for i in range(1, cloud.point_count()):
            p = cloud.get_point(i)
            dist = p.distance(test_point)
            if dist < best_dist:
                best_dist = dist
                best_point = p
                best_index = i

        return (best_point, best_index, best_dist)

    @staticmethod
    def pointcloud_point_kdtree(cloud, test_point):
        if cloud.point_count() == 0:
            return (Point(0, 0, 0), 0, float('inf'))
        from session_py import SpatialKDTree
        pts = [cloud.get_point(i) for i in range(cloud.point_count())]
        tree = SpatialKDTree(pts)
        idx, dist = tree.nearest(test_point)
        return (cloud.get_point(idx), idx, dist)

    @staticmethod
    def _build_raw_boxes(aabbs):
        return [(b.cx, b.cy, b.cz, b.hx, b.hy, b.hz) for b in aabbs]

    @staticmethod
    def _build_aabb_nodes(raw_boxes):
        nodes = []

        def build(ids):
            ni = len(nodes)
            nodes.append([None, -1, -1])
            lx = ly = lz = 1e308
            hx = hy = hz = -1e308
            for i in ids:
                b = raw_boxes[i]
                lx = min(lx, b[0]-b[3]); hx = max(hx, b[0]+b[3])
                ly = min(ly, b[1]-b[4]); hy = max(hy, b[1]+b[4])
                lz = min(lz, b[2]-b[5]); hz = max(hz, b[2]+b[5])
            nodes[ni][0] = ((lx+hx)*0.5, (ly+hy)*0.5, (lz+hz)*0.5,
                            (hx-lx)*0.5, (hy-ly)*0.5, (hz-lz)*0.5)
            if len(ids) == 1:
                nodes[ni][2] = ids[0]
                return
            dx = hx-lx; dy = hy-ly; dz = hz-lz
            axis = 0 if dx >= dy and dx >= dz else (1 if dy >= dz else 2)
            ids.sort(key=lambda i: raw_boxes[i][axis])
            mid = len(ids) // 2
            build(ids[:mid])
            nodes[ni][1] = len(nodes)
            build(ids[mid:])

        build(list(range(len(raw_boxes))))
        return nodes

    @staticmethod
    def _query_aabb_nodes(nodes, query, result):
        def overlaps(a, b):
            return (abs(a[0]-b[0]) <= a[3]+b[3] and
                    abs(a[1]-b[1]) <= a[4]+b[4] and
                    abs(a[2]-b[2]) <= a[5]+b[5])

        def dfs(ni):
            aabb, right, obj = nodes[ni]
            if not overlaps(aabb, query):
                return
            if obj >= 0:
                result.append(obj)
                return
            dfs(ni + 1)
            dfs(right)

        dfs(0)

    @staticmethod
    def _aabb_to_aabb_min_dist(a, b):
        dx = max(0.0, abs(a[0]-b[0]) - a[3] - b[3])
        dy = max(0.0, abs(a[1]-b[1]) - a[4] - b[4])
        dz = max(0.0, abs(a[2]-b[2]) - a[5] - b[5])
        return (dx*dx + dy*dy + dz*dz) ** 0.5

    @staticmethod
    def lines_closest(lines, threshold=0.0):
        if len(lines) < 2:
            return []
        from session_py import AABB
        raw = Closest._build_raw_boxes([AABB.from_line(ln, threshold) for ln in lines])
        nodes = Closest._build_aabb_nodes(raw)
        pairs = []
        for i in range(len(lines)):
            candidates = []
            Closest._query_aabb_nodes(nodes, raw[i], candidates)
            for j in candidates:
                if j <= i:
                    continue
                _, _, d_a = Closest.line_point(lines[j], lines[i].start())
                _, _, d_b = Closest.line_point(lines[j], lines[i].end())
                _, _, d_c = Closest.line_point(lines[i], lines[j].start())
                _, _, d_d = Closest.line_point(lines[i], lines[j].end())
                if min(d_a, d_b, d_c, d_d) <= threshold:
                    pairs.append((i, j))
        return pairs

    @staticmethod
    def polylines_closest(polylines, threshold=0.0):
        if len(polylines) < 2:
            return []
        from session_py import AABB
        raw = Closest._build_raw_boxes([AABB.from_polyline(pl, threshold) for pl in polylines])
        nodes = Closest._build_aabb_nodes(raw)
        pairs = []
        for i in range(len(polylines)):
            candidates = []
            Closest._query_aabb_nodes(nodes, raw[i], candidates)
            for j in candidates:
                if j <= i:
                    continue
                pts_a = polylines[i].get_points()
                dist = min(Closest.polyline_point(polylines[j], pt)[2] for pt in pts_a)
                if dist <= threshold:
                    pairs.append((i, j))
        return pairs

    @staticmethod
    def nurbscurves_closest(curves, threshold=0.0):
        if len(curves) < 2:
            return []
        from session_py import AABB
        raw = Closest._build_raw_boxes([AABB.from_nurbscurve(crv, threshold, False) for crv in curves])
        nodes = Closest._build_aabb_nodes(raw)
        pairs = []
        for i in range(len(curves)):
            candidates = []
            Closest._query_aabb_nodes(nodes, raw[i], candidates)
            for j in candidates:
                if j <= i:
                    continue
                t0, t1 = curves[i].domain()
                p_start = curves[i].point_at(t0)
                p_end = curves[i].point_at(t1)
                _, d_a = Closest.curve_point(curves[j], p_start)
                _, d_b = Closest.curve_point(curves[j], p_end)
                if min(d_a, d_b) <= threshold:
                    pairs.append((i, j))
        return pairs

    @staticmethod
    def boxes_closest(boxes, threshold=0.0):
        if len(boxes) < 2:
            return []
        inflated = []
        for b in boxes:
            from session_py import AABB
            inf = AABB(b.cx, b.cy, b.cz, b.hx + threshold, b.hy + threshold, b.hz + threshold)
            inflated.append(inf)
        raw = Closest._build_raw_boxes(inflated)
        nodes = Closest._build_aabb_nodes(raw)
        raw_orig = Closest._build_raw_boxes(boxes)
        pairs = []
        for i in range(len(boxes)):
            candidates = []
            Closest._query_aabb_nodes(nodes, raw[i], candidates)
            for j in candidates:
                if j <= i:
                    continue
                dist = Closest._aabb_to_aabb_min_dist(raw_orig[i], raw_orig[j])
                if dist <= threshold:
                    pairs.append((i, j))
        return pairs
