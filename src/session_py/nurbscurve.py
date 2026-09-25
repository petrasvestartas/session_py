from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import functools
import json
import math
import uuid
import numpy as np
from .color import Color
from .plane import Plane
from .point import Point
from .tolerance import Tolerance
from .tolerance import TOLERANCE
from .vector import Vector
from .xform import Xform
from . import nurbsknot
from .nurbsknot import CurveInterpStyle
from .nurbsknot import CurveNurbsKnotStyle

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import nurbscurve_pb2

SQRT_EPSILON = 1.490116119385e-8
GL_X = [
    -0.9739065285171717,
    -0.8650633666889845,
    -0.6794095682990244,
    -0.4333953941292472,
    -0.1488743389816312,
    0.1488743389816312,
    0.4333953941292472,
    0.6794095682990244,
    0.8650633666889845,
    0.9739065285171717,
]
GL_W = [
    0.0666713443086881,
    0.1494513491505806,
    0.2190863625159820,
    0.2692667193099963,
    0.2955242247147529,
    0.2955242247147529,
    0.2692667193099963,
    0.2190863625159820,
    0.1494513491505806,
    0.0666713443086881,
]
GL_NODES = [
    -0.9061798459386640,
    -0.5384693101056831,
    0.0,
    0.5384693101056831,
    0.9061798459386640,
]
GL_WEIGHTS = [
    0.2369268850561891,
    0.4786286704993665,
    0.5688888888888889,
    0.4786286704993665,
    0.2369268850561891,
]


class NurbsCurve:
    """A NURBS curve: OpenNURBS layout, nurbsknot count = order + cv_count - 2, homogeneous CVs when rational."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, dimension: int = 0, is_rational: bool = False, order: int = 0, cv_count: int = 0):
        """Construct an unset curve with the given layout."""

        self._guid = None  # Lazily minted GUID.
        self.name = "my_nurbscurve"  # Curve name.
        self.width = 1.0  # Display width.
        self.pointcolors: list[Color] = []  # Display color per control point.
        self.linecolors: list[Color] = []  # Display color per control polygon segment.
        self.m_dim = 0  # Coordinate dimension.
        self.m_is_rat = 0  # 1 when rational, 0 otherwise.
        self.m_order = 0  # Degree + 1.
        self.m_cv_count = 0  # Number of control vertices.
        self.m_cv_stride = 0  # Doubles between consecutive CVs.
        self.m_nurbsknot = np.zeros(0, dtype=np.float64)  # NurbsKnot vector, order + cv_count - 2 values.
        self.m_cv = np.zeros(0, dtype=np.float64)  # Flat CV array, homogeneous when rational.
        self.create_curve(dimension, is_rational, order, cv_count)

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = NurbsCurve()
        result._deep_copy_from(self)
        memo[id(self)] = result

        return result

    def duplicate(self) -> NurbsCurve:
        """Copy with a new guid and the same data."""
        return copy.deepcopy(self)

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def create(
        periodic: bool,
        degree: int,
        points: list[Point],
        dimension: int = 3,
        nurbsknot_delta: float = 1.0,
    ) -> NurbsCurve:
        """Construct a clamped or periodic uniform curve through control points, domain rescaled to [0, arc length]."""

        curve = NurbsCurve()
        order = degree + 1

        if periodic:
            curve.create_periodic_uniform(dimension, order, points, nurbsknot_delta)
        else:
            curve.create_clamped_uniform(dimension, order, points, nurbsknot_delta)

        if not curve.is_valid():
            return curve

        length = 0.0

        if degree == 1:
            np_ = len(points)

            for i in range(1, np_):
                length += points[i - 1].distance(points[i])

            if periodic and np_ > 1:
                length += points[np_ - 1].distance(points[0])
        else:
            length = curve.length()

        if length > 0.0:
            curve.set_domain(0.0, length)

        return curve

    @staticmethod
    def create_interpolated(
        points: list[Point],
        parameterization: CurveNurbsKnotStyle = CurveNurbsKnotStyle.Chord,
        end_condition: CurveInterpStyle = CurveInterpStyle.Rhino,
    ) -> NurbsCurve:
        """Construct an interpolated cubic through points; Rhino (Bessel) or Occt (Lagrange) end tangents."""

        n = len(points)

        if n < 2:
            return NurbsCurve()

        periodic = parameterization in (
            CurveNurbsKnotStyle.UniformPeriodic,
            CurveNurbsKnotStyle.ChordPeriodic,
            CurveNurbsKnotStyle.ChordSquareRootPeriodic,
        )

        if periodic and n < 3:
            return NurbsCurve()

        if n == 2 and not periodic:
            return NurbsCurve.create(False, 1, points)

        if periodic:
            return NurbsCurve._create_interpolated_periodic(points, parameterization)

        return NurbsCurve._create_interpolated_clamped(points, parameterization, end_condition)

    @staticmethod
    def create_from_parameters(
        points: list[Point],
        weights: list[float],
        knots: list[float],
        mults: list[int],
        degree: int,
        periodic: bool = False,
    ) -> NurbsCurve:
        """Construct from poles, weights, distinct knots and multiplicities (OCCT convention)."""

        n = len(points)
        order = degree + 1

        if n < order:
            return NurbsCurve()

        if len(weights) != n:
            return NurbsCurve()

        if len(knots) != len(mults) or len(knots) == 0:
            return NurbsCurve()

        if periodic:
            return NurbsCurve()

        rational = False

        for w in weights:
            if abs(w - 1.0) > Tolerance.ZERO_TOLERANCE:
                rational = True

        full = []

        for i in range(len(knots)):
            for m in range(mults[i]):
                full.append(knots[i])

        kc = order + n - 2

        if len(full) != kc + 2:
            return NurbsCurve()

        curve = NurbsCurve()

        if not curve.create_curve(3, rational, order, n):
            return NurbsCurve()

        for i in range(kc):
            curve.set_nurbsknot(i, full[i + 1])

        for i in range(n):
            if rational:
                w = weights[i]
                curve.set_cv_4d(
                    i, points[i][0] * w, points[i][1] * w, points[i][2] * w, w
                )
            else:
                curve.set_cv(i, points[i])

        return curve

    @staticmethod
    def create_fitted(
        points: list[Point], num_cvs: int, degree: int = 3, is_periodic: bool = False
    ) -> NurbsCurve:
        """Construct a least-squares fit with num_cvs control points (Piegl & Tiller 9.4)."""

        if is_periodic:
            return NurbsCurve._create_fitted_periodic(points, num_cvs, degree)

        return NurbsCurve._create_fitted_clamped(points, num_cvs, degree)

    @staticmethod
    def join(
        curves: list[NurbsCurve], tolerance: float = Tolerance.ZERO_TOLERANCE
    ) -> list[NurbsCurve]:
        """Chain segments by endpoint matching, raise to a common degree and merge with C0 junctions."""

        segs = []

        for c in curves:
            if c.is_valid():
                segs.append(c.duplicate())

        NurbsCurve._promote_to_3d(segs)

        chains = NurbsCurve._chain_segments(segs, tolerance)
        result = []

        for chain in chains:
            NurbsCurve._join_chain(chain, result)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __eq__(self, other) -> bool:
        """Compare name, width, colors, layout, nurbsknots and CVs to 1e-12; guid ignored."""

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
            if (
                abs(float(self.m_nurbsknot[i]) - float(other.m_nurbsknot[i]))
                > Tolerance.ZERO_TOLERANCE
            ):
                return False

        if len(self.m_cv) != len(other.m_cv):
            return False

        for i in range(len(self.m_cv)):
            if (
                abs(float(self.m_cv[i]) - float(other.m_cv[i]))
                > Tolerance.ZERO_TOLERANCE
            ):
                return False

        return True

    def __ne__(self, other) -> bool:
        """Compare name, width, colors, layout, nurbsknots and CVs to 1e-12; guid ignored."""
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════
    def transform(self, xform: Xform) -> bool:
        """Transform in place."""

        for i in range(self.m_cv_count):
            p = self.get_cv(i)
            x = xform.m[0] * p[0] + xform.m[4] * p[1] + xform.m[8] * p[2] + xform.m[12]
            y = xform.m[1] * p[0] + xform.m[5] * p[1] + xform.m[9] * p[2] + xform.m[13]
            z = xform.m[2] * p[0] + xform.m[6] * p[1] + xform.m[10] * p[2] + xform.m[14]

            if self.m_is_rat:
                w = self.weight(i)
                self.set_cv_4d(i, x * w, y * w, z * w, w)
            else:
                self.set_cv(i, Point(x, y, z))

        return True

    def transformed(self, xform: Xform) -> NurbsCurve:
        """Return a transformed copy."""

        result = self.duplicate()
        result.transform(xform)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Initialization
    # ═══════════════════════════════════════════════════════════════════════════
    def initialize(self) -> None:
        """Zero every field."""

        self.m_dim = 0
        self.m_is_rat = 0
        self.m_order = 0
        self.m_cv_count = 0
        self.m_cv_stride = 0
        self.m_nurbsknot = np.zeros(0, dtype=np.float64)
        self.m_cv = np.zeros(0, dtype=np.float64)

    def create_curve(self, dimension: int, is_rational: bool, order: int, cv_count: int) -> bool:
        """Allocate layout for dimension, rationality, order and cv_count."""

        if dimension < 1 or order < 2 or cv_count < order:
            return False

        self.destroy()
        self.m_dim = dimension
        self.m_is_rat = 1 if is_rational else 0
        self.m_order = order
        self.m_cv_count = cv_count
        self.m_cv_stride = (dimension + 1) if is_rational else dimension
        self.m_nurbsknot = np.zeros(
            self.m_order + self.m_cv_count - 2, dtype=np.float64
        )
        self.m_cv = np.zeros(self.m_cv_count * self.m_cv_stride, dtype=np.float64)

        return True

    def create_clamped_uniform(
        self,
        dimension: int,
        order: int,
        points: list[Point],
        nurbsknot_delta: float = 1.0,
    ) -> bool:
        """Set clamped uniform nurbsknots over control points."""

        point_count = len(points)

        if not self.create_curve(dimension, False, order, point_count):
            return False

        for i in range(point_count):
            self.set_cv(i, points[i])

        kc = self.m_order + self.m_cv_count - 2
        k = 0.0

        for i in range(self.m_order - 2, self.m_cv_count):
            self.m_nurbsknot[i] = k
            k += nurbsknot_delta

        i0 = self.m_order - 2

        for i in range(i0):
            self.m_nurbsknot[i] = self.m_nurbsknot[i0]

        i0 = self.m_cv_count - 1

        for i in range(i0 + 1, kc):
            self.m_nurbsknot[i] = self.m_nurbsknot[i0]

        return True

    def create_periodic_uniform(
        self,
        dimension: int,
        order: int,
        points: list[Point],
        nurbsknot_delta: float = 1.0,
    ) -> bool:
        """Set periodic uniform nurbsknots over control points wrapped by order - 1."""

        point_count = len(points)

        if point_count < order:
            return False

        if not self.create_curve(dimension, False, order, point_count + order - 1):
            return False

        for i in range(point_count):
            self.set_cv(i, points[i])

        for i in range(order - 1):
            self.set_cv(point_count + i, points[i])

        kc = self.m_order + self.m_cv_count - 2

        for i in range(kc):
            self.m_nurbsknot[i] = (i - self.m_order + 1) * nurbsknot_delta

        return True

    def destroy(self) -> None:
        """Reset to the empty state."""
        self.initialize()

    # ═══════════════════════════════════════════════════════════════════════════
    # Boolean queries
    # ═══════════════════════════════════════════════════════════════════════════
    def is_valid(self) -> bool:
        """Return whether the layout, nurbsknots and CVs are consistent."""

        if self.m_dim <= 0:
            return False

        if self.m_order < 2:
            return False

        if self.m_cv_count < self.m_order:
            return False

        if self.m_cv_stride < self.cv_size():
            return False

        if len(self.m_cv) == 0 or len(self.m_nurbsknot) == 0:
            return False

        if len(self.m_cv) < (self.m_cv_count - 1) * self.m_cv_stride + self.cv_size():
            return False

        if not self.is_valid_nurbsknot_vector():
            return False

        for i in range(len(self.m_cv)):
            if not math.isfinite(self.m_cv[i]):
                return False

        return True

    def is_rational(self) -> bool:
        """Return whether the CVs carry weights."""
        return self.m_is_rat != 0

    def is_closed(self) -> bool:
        """Return whether the start point equals the end point."""

        if not self.is_valid():
            return False

        return (
            self.point_at_start().distance(self.point_at_end())
            < Tolerance.ZERO_TOLERANCE
        )

    def is_periodic(self) -> bool:
        """Return whether the last degree CVs repeat the first and nurbsknots are uniform."""

        if self.m_order < 2:
            return False

        deg = self.degree()

        for i in range(deg):
            if (
                self.get_cv(i).distance(self.get_cv(self.m_cv_count - deg + i))
                > Tolerance.ZERO_TOLERANCE
            ):
                return False

        kc = self.nurbsknot_count()

        if kc < 2:
            return False

        delta = self.m_nurbsknot[self.m_order - 1] - self.m_nurbsknot[self.m_order - 2]

        if delta < Tolerance.ZERO_TOLERANCE:
            return False

        for i in range(1, kc):
            if (
                abs((self.m_nurbsknot[i] - self.m_nurbsknot[i - 1]) - delta)
                > Tolerance.ZERO_TOLERANCE
            ):
                return False

        return True

    def is_linear(self, tolerance: float = Tolerance.ZERO_TOLERANCE) -> bool:
        """Return whether every CV is within tolerance of the chord."""

        if not self.is_valid() or self.m_cv_count < 2:
            return False

        p0 = self.get_cv(0)
        p1 = self.get_cv(self.m_cv_count - 1)
        line_vec = p1 - p0
        line_length = line_vec.magnitude()

        if line_length < tolerance:
            return True

        for i in range(1, self.m_cv_count - 1):
            p = self.get_cv(i)
            v = p - p0

            if line_vec.cross(v).magnitude() / line_length > tolerance:
                return False

        return True

    def is_planar(
        self, plane: Plane | None = None, tolerance: float = Tolerance.ZERO_TOLERANCE
    ) -> bool:
        """Return whether every CV is within tolerance of one plane, written to plane when given."""

        if not self.is_valid() or self.m_cv_count < 3:
            return True

        p0 = self.get_cv(0)
        p1 = self.get_cv(self.m_cv_count // 2)
        p2 = self.get_cv(self.m_cv_count - 1)
        v1 = p1 - p0
        v2 = p2 - p0
        normal = v1.cross(v2)

        if normal.magnitude() < tolerance:
            return True

        for i in range(self.m_cv_count):
            p = self.get_cv(i)
            v = p - p0

            if abs(v.dot(normal)) / normal.magnitude() > tolerance:
                return False

        if plane is not None:
            normal.normalize_self()

            x_axis = Vector(v1[0], v1[1], v1[2])
            x_axis.normalize_self()
            NurbsCurve._assign_plane(plane, Plane(p0, x_axis, normal.cross(x_axis)))

        return True

    def is_arc(
        self, plane: Plane | None = None, tolerance: float = Tolerance.ZERO_TOLERANCE
    ) -> bool:
        """Return whether the curve is planar and equidistant from one center, plane written when given."""

        if not self.is_valid():
            return False

        if self.m_dim != 2 and self.m_dim != 3:
            return False

        if self.m_order < 3:
            return False

        if self.is_linear(tolerance):
            return False

        test_plane = Plane()

        if not self.is_planar(test_plane, tolerance):
            return False

        t0, t1 = self.domain()
        p0 = self.point_at(t0)
        center = NurbsCurve._circle_center(p0, self.point_at((t0 + t1) * 0.5), self.point_at(t1))

        if center is None:
            return False

        radius = center.distance(p0)

        if radius < Tolerance.ZERO_TOLERANCE:
            return False

        samples_per_span = max(4, 2 * self.degree() + 1)
        num_samples = self.span_count() * samples_per_span

        for i in range(num_samples + 1):
            t = t0 + (t1 - t0) * i / num_samples

            if abs(self.point_at(t).distance(center) - radius) > tolerance:
                return False

        if plane is not None:
            NurbsCurve._assign_plane(plane, test_plane)

        return True

    def is_in_plane(
        self, test_plane: Plane, tolerance: float = Tolerance.ZERO_TOLERANCE
    ) -> bool:
        """Return whether every CV is within tolerance of test_plane."""

        if not self.is_valid():
            return False

        for i in range(self.m_cv_count):
            pt = self.get_cv(i)
            v = pt - test_plane.origin

            if abs(v.dot(test_plane.z_axis)) > tolerance:
                return False

        return True

    def is_natural(self, end: int = 2) -> bool:
        """Return whether the second derivative is zero at end (0 = start, 1 = end, 2 = both)."""

        if not self.is_valid():
            return False

        tol_factor = 1e-8
        t0, t1 = self.domain()
        first = 0 if (end == 0 or end == 2) else 1
        stop = 2 if (end == 1 or end == 2) else 1

        for pass_ in range(first, stop):
            t = t0 if pass_ == 0 else t1
            derivs = self.evaluate(t, 2)

            if len(derivs) < 3:
                return False

            d2_len = derivs[2].magnitude()
            cv0 = self.get_cv(0 if pass_ == 0 else self.m_cv_count - 1)
            cv2 = self.get_cv(
                min(2, self.m_cv_count - 1)
                if pass_ == 0
                else max(0, self.m_cv_count - 3)
            )

            if d2_len > cv0.distance(cv2) * tol_factor:
                return False

        return True

    def is_polyline(self) -> tuple[int, list[Point], list[float]]:
        """Return the vertex count when every span is a line, else 0; vertices and params written when given."""

        points: list[Point] = []
        params: list[float] = []

        if not self.is_valid():
            return 0, points, params

        if self.m_order == 2:
            for i in range(self.m_cv_count):
                points.append(self.get_cv(i))

            for i in range(self.m_cv_count):
                params.append(float(self.m_nurbsknot[i]))

            return self.m_cv_count, points, params

        if self.m_order > 2 and 2 <= self.m_dim <= 3:
            span_cnt = self.span_count()
            all_linear = True

            for i in range(span_cnt):
                if not self._span_is_linear(
                    i, Tolerance.ZERO_TOLERANCE, Tolerance.ZERO_TOLERANCE
                ):
                    all_linear = False
                    break

            if all_linear and span_cnt > 0:
                points.append(self.get_cv(0))

                for i in range(span_cnt):
                    points.append(
                        self.get_cv(i * (self.m_order - 1) + (self.m_order - 1))
                    )

                params = self.get_span_vector()

                return span_cnt + 1, points, params

        return 0, points, params

    def is_singular(self) -> bool:
        """Return whether every span is collapsed to a point."""

        if not self.is_valid():
            return False

        span_cnt = self.span_count()

        for i in range(span_cnt):
            if not self._span_is_singular(i):
                return False

        return True

    def is_duplicate(
        self,
        other: NurbsCurve,
        ignore_parameterization: bool,
        tolerance: float = Tolerance.ZERO_TOLERANCE,
    ) -> bool:
        """Return whether layout, CVs and weights match to tolerance, and nurbsknots unless ignore_parameterization."""

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
            if self.get_cv(i).distance(other.get_cv(i)) > tolerance:
                return False

            if self.m_is_rat and abs(self.weight(i) - other.weight(i)) > tolerance:
                return False

        if not ignore_parameterization:
            for i in range(self.nurbsknot_count()):
                if abs(self.m_nurbsknot[i] - other.m_nurbsknot[i]) > tolerance:
                    return False

        return True

    def is_continuous(
        self,
        continuity_type: int,
        t: float,
        point_tolerance: float = Tolerance.ZERO_TOLERANCE,
        d1_tolerance: float = Tolerance.ZERO_TOLERANCE,
        d2_tolerance: float = Tolerance.ZERO_TOLERANCE,
        cos_angle_tolerance: float = 0.99984769515639123,
        curvature_tolerance: float = 1e-8,
    ) -> bool:
        """Return the continuity at t from nurbsknot multiplicity (0 = C0, 1 = C1, 2 = C2, 3 = G1, 4 = G2)."""

        if not self.is_valid():
            return False

        d0, d1 = self.domain()

        if t < d0 or t > d1:
            return False

        nurbsknot_idx = -1

        for i in range(self.nurbsknot_count()):
            if abs(self.m_nurbsknot[i] - t) < Tolerance.ZERO_TOLERANCE:
                nurbsknot_idx = i
                break

        if nurbsknot_idx < 0:
            return True

        mult = self.nurbsknot_multiplicity(nurbsknot_idx)

        if continuity_type == 0:
            return mult < self.m_order

        if continuity_type == 1:
            return mult < self.m_order - 1

        if continuity_type == 2:
            return mult < self.m_order - 2

        return mult < self.m_order - 1

    def is_valid_nurbsknot_vector(self) -> bool:
        """Return whether the nurbsknots have the right count, are non-decreasing and span a non-empty domain."""

        kc = self.nurbsknot_count()

        if len(self.m_nurbsknot) != kc:
            return False

        for i in range(1, kc):
            if self.m_nurbsknot[i] < self.m_nurbsknot[i - 1]:
                return False

        if self.m_nurbsknot[self.m_order - 2] >= self.m_nurbsknot[self.m_cv_count - 1]:
            return False

        return True

    def is_clamped(self, end: int = 2) -> bool:
        """Return whether end has full multiplicity (0 = start, 1 = end, 2 = both)."""

        if not self.is_valid():
            return False

        return nurbsknot.is_clamped(
            self.m_order, self.m_cv_count, self.m_nurbsknot, end
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def has_guid(self) -> bool:
        """Return whether the lazy guid has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the guid, creating it on first access."""

        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        """Set the guid."""
        self._guid = value

    def refresh_guid(self) -> None:
        """Clear the guid so a fresh one mints lazily on the next read."""
        self._guid = None

    def dimension(self) -> int:
        """Return the coordinate dimension."""
        return self.m_dim

    def order(self) -> int:
        """Return the order (degree + 1)."""
        return self.m_order

    def degree(self) -> int:
        """Return the degree (order - 1)."""
        return self.m_order - 1

    def cv_count(self) -> int:
        """Return the number of control vertices."""
        return self.m_cv_count

    def cv_size(self) -> int:
        """Return the doubles per CV: dimension + 1 when rational."""

        if self.m_dim <= 0:
            return 0

        return self.m_dim + 1 if self.m_is_rat else self.m_dim

    def nurbsknot_count(self) -> int:
        """Return order + cv_count - 2."""
        return self.m_order + self.m_cv_count - 2

    def span_count(self) -> int:
        """Return the number of distinct nurbsknot intervals inside the domain."""

        count = 0
        kc = self.nurbsknot_count()

        for i in range(self.m_order - 2, self.m_cv_count - 1):
            if i >= 0 and i + 1 < kc and self.m_nurbsknot[i] < self.m_nurbsknot[i + 1]:
                count += 1

        return count

    # ═══════════════════════════════════════════════════════════════════════════
    # Control vertex access
    # ═══════════════════════════════════════════════════════════════════════════
    def cv(self, cv_index: int) -> np.ndarray | None:
        """Return the pointer to the CV doubles, nullptr when out of range."""

        if cv_index < 0 or cv_index >= self.m_cv_count:
            return None

        idx = cv_index * self.m_cv_stride

        return self.m_cv[idx : idx + self.m_cv_stride]

    def get_cv(self, cv_index: int) -> Point:
        """Return the Euclidean CV (divided by weight when rational)."""

        cv_ptr = self.cv(cv_index)

        if cv_ptr is None:
            return Point(0.0, 0.0, 0.0)

        if self.m_is_rat:
            w = cv_ptr[self.m_dim]

            if abs(w) < 1e-14:
                return Point(0.0, 0.0, 0.0)

            return Point(
                cv_ptr[0] / w, cv_ptr[1] / w, cv_ptr[2] / w if self.m_dim > 2 else 0.0
            )

        return Point(cv_ptr[0], cv_ptr[1], cv_ptr[2] if self.m_dim > 2 else 0.0)

    def get_cv_4d(self, cv_index: int) -> tuple[float, float, float, float]:
        """Return the homogeneous CV (x, y, z, w)."""

        cv_ptr = self.cv(cv_index)

        if cv_ptr is None:
            return 0.0, 0.0, 0.0, 1.0

        x = float(cv_ptr[0])
        y = float(cv_ptr[1]) if self.m_dim > 1 else 0.0
        z = float(cv_ptr[2]) if self.m_dim > 2 else 0.0
        w = float(cv_ptr[self.m_dim]) if self.m_is_rat else 1.0

        return x, y, z, w

    def set_cv(self, cv_index: int, point: Point) -> bool:
        """Set the CV from a point, weight reset to 1."""

        cv_ptr = self.cv(cv_index)

        if cv_ptr is None:
            return False

        cv_ptr[0] = point[0]

        if self.m_dim > 1:
            cv_ptr[1] = point[1]

        if self.m_dim > 2:
            cv_ptr[2] = point[2]

        if self.m_is_rat:
            cv_ptr[self.m_dim] = 1.0

        return True

    def set_cv_4d(self, cv_index: int, x: float, y: float, z: float, w: float) -> bool:
        """Set the homogeneous CV, making the curve rational when w != 1."""

        if cv_index < 0 or cv_index >= self.m_cv_count:
            return False

        if not self.m_is_rat and w != 1.0 and not self.to_rational():
            return False

        cv_ptr = self.cv(cv_index)

        if cv_ptr is None:
            return False

        cv_ptr[0] = x

        if self.m_dim > 1:
            cv_ptr[1] = y

        if self.m_dim > 2:
            cv_ptr[2] = z

        if self.m_is_rat:
            cv_ptr[self.m_dim] = w

        return True

    def weight(self, cv_index: int) -> float:
        """Return the weight of a CV, 1 when non-rational."""

        if not self.m_is_rat:
            return 1.0

        cv_ptr = self.cv(cv_index)

        return float(cv_ptr[self.m_dim]) if cv_ptr is not None else 1.0

    def set_weight(self, cv_index: int, weight: float) -> bool:
        """Set the weight, making the curve rational first."""

        if not self.m_is_rat and not self.to_rational():
            return False

        cv_ptr = self.cv(cv_index)

        if cv_ptr is None:
            return False

        cv_ptr[self.m_dim] = weight

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # NurbsKnot access
    # ═══════════════════════════════════════════════════════════════════════════
    def nurbsknot(self, nurbsknot_index: int) -> float:
        """Return the nurbsknot at nurbsknot_index."""

        if nurbsknot_index < 0 or nurbsknot_index >= len(self.m_nurbsknot):
            return 0.0

        return float(self.m_nurbsknot[nurbsknot_index])

    def set_nurbsknot(self, nurbsknot_index: int, nurbsknot_value: float) -> bool:
        """Set the nurbsknot at nurbsknot_index."""

        if nurbsknot_index < 0 or nurbsknot_index >= len(self.m_nurbsknot):
            return False

        self.m_nurbsknot[nurbsknot_index] = nurbsknot_value

        return True

    def nurbsknot_multiplicity(self, nurbsknot_index: int) -> int:
        """Return the count of nurbsknots equal to the one at nurbsknot_index."""

        if nurbsknot_index < 0 or nurbsknot_index >= self.nurbsknot_count():
            return 0

        nurbsknot_value = self.m_nurbsknot[nurbsknot_index]
        mult = 1

        for i in range(nurbsknot_index + 1, self.nurbsknot_count()):
            if abs(self.m_nurbsknot[i] - nurbsknot_value) >= Tolerance.ZERO_TOLERANCE:
                break

            mult += 1

        for i in range(nurbsknot_index - 1, -1, -1):
            if abs(self.m_nurbsknot[i] - nurbsknot_value) >= Tolerance.ZERO_TOLERANCE:
                break

            mult += 1

        return mult

    def superfluous_nurbsknot(self, end: int) -> float:
        """Return the reflected end nurbsknot (0 = start, 1 = end)."""

        if not self.is_valid():
            return 0.0

        if end == 0:
            return float(2.0 * self.m_nurbsknot[0] - self.m_nurbsknot[self.m_order - 2])

        return float(
            2.0 * self.m_nurbsknot[self.nurbsknot_count() - 1]
            - self.m_nurbsknot[self.m_cv_count - self.m_order]
        )

    def nurbsknot_array(self) -> np.ndarray:
        """Return the nurbsknot array pointer."""
        return self.m_nurbsknot

    def cv_array(self) -> np.ndarray:
        """Return the CV array pointer."""
        return self.m_cv

    def get_nurbsknots(self) -> list[float]:
        """Return a copy of the nurbsknot vector."""
        return self.m_nurbsknot.tolist()

    def insert_nurbsknot(
        self, nurbsknot_value: float, nurbsknot_multiplicity: int = 1
    ) -> bool:
        """Insert a nurbsknot by Boehm to the given multiplicity."""

        if not self.is_valid():
            return False

        p = self.degree()

        if nurbsknot_multiplicity < 1 or nurbsknot_multiplicity > p:
            return False

        d0, d1 = self.domain()

        if nurbsknot_value < d0 or nurbsknot_value > d1:
            return False

        if nurbsknot_value == d0:
            if nurbsknot_multiplicity == p:
                return self.clamp_end(0)

            return nurbsknot_multiplicity == 1

        if nurbsknot_value == d1:
            if nurbsknot_multiplicity == p:
                return self.clamp_end(1)

            return nurbsknot_multiplicity == 1

        tol = (abs(d0) + abs(d1) + abs(d1 - d0)) * SQRT_EPSILON

        for insert_iter in range(nurbsknot_multiplicity):
            U = self._full_nurbsknots()
            mult = 0

            for knot in U:
                if abs(knot - nurbsknot_value) <= tol:
                    mult += 1

            if mult >= nurbsknot_multiplicity:
                return True

            if mult >= p:
                return False

            self._insert_nurbsknot_once(nurbsknot_value, U)

        return True

    def greville_abcissa(self, cv_index: int) -> float:
        """Return the Greville abcissa of a CV."""

        if cv_index < 0 or cv_index >= self.m_cv_count:
            return 0.0

        nurbsknot_ = self.m_nurbsknot[cv_index:]
        order = self.m_order

        if order <= 2 or nurbsknot_[0] == nurbsknot_[order - 2]:
            return float(nurbsknot_[0])

        p = order - 1
        k0 = nurbsknot_[0]
        k = nurbsknot_[p // 2]
        k1 = nurbsknot_[p - 1]
        tol = (k1 - k0) * SQRT_EPSILON
        g = 0.0

        for i in range(p):
            g += nurbsknot_[i]

        g /= float(p)

        if abs(2.0 * k - (k0 + k1)) <= tol and abs(g - k) <= (
            abs(g) * SQRT_EPSILON + tol
        ):
            g = k

        return float(g)

    def get_greville_abcissae(self) -> list[float]:
        """Return the Greville abcissae of every CV."""

        abcissae: list[float] = []

        if not self.is_valid():
            return abcissae

        for i in range(self.m_cv_count):
            abcissae.append(self.greville_abcissa(i))

        return abcissae

    # ═══════════════════════════════════════════════════════════════════════════
    # Domain
    # ═══════════════════════════════════════════════════════════════════════════
    def domain(self) -> tuple[float, float]:
        """Return the domain (t0, t1)."""

        if len(self.m_nurbsknot) == 0:
            return 0.0, 0.0

        return float(self.m_nurbsknot[self.m_order - 2]), float(
            self.m_nurbsknot[self.m_cv_count - 1]
        )

    def domain_start(self) -> float:
        """Return the domain start."""

        if len(self.m_nurbsknot) == 0:
            return 0.0

        return float(self.m_nurbsknot[self.m_order - 2])

    def domain_end(self) -> float:
        """Return the domain end."""

        if len(self.m_nurbsknot) == 0:
            return 0.0

        return float(self.m_nurbsknot[self.m_cv_count - 1])

    def domain_middle(self) -> float:
        """Return the domain midpoint."""

        if len(self.m_nurbsknot) == 0:
            return 0.0

        return (
            float(
                self.m_nurbsknot[self.m_order - 2]
                + self.m_nurbsknot[self.m_cv_count - 1]
            )
            * 0.5
        )

    def set_domain(self, t0: float, t1: float) -> bool:
        """Rescale the nurbsknots to [t0, t1]."""

        if t0 >= t1 or not self.is_valid():
            return False

        d0, d1 = self.domain()

        if d0 >= d1:
            return False

        clamped_start = (
            self.m_order >= 2
            and abs(self.m_nurbsknot[0] - self.m_nurbsknot[self.m_order - 2])
            < Tolerance.ZERO_TOLERANCE
        )
        clamped_end = (
            self.m_cv_count < len(self.m_nurbsknot)
            and abs(self.m_nurbsknot[-1] - self.m_nurbsknot[self.m_cv_count - 1])
            < Tolerance.ZERO_TOLERANCE
        )
        scale = (t1 - t0) / (d1 - d0)

        for i in range(len(self.m_nurbsknot)):
            self.m_nurbsknot[i] = t0 + (self.m_nurbsknot[i] - d0) * scale

        if clamped_start:
            for i in range(self.m_order - 1):
                self.m_nurbsknot[i] = t0

        if clamped_end:
            for i in range(self.m_cv_count - 1, len(self.m_nurbsknot)):
                self.m_nurbsknot[i] = t1

        return True

    def get_span_vector(self) -> list[float]:
        """Return the distinct nurbsknot values inside the domain."""

        if not self.is_valid():
            return []

        spans = [float(self.m_nurbsknot[self.m_order - 2])]

        for i in range(self.m_order - 1, self.m_cv_count):
            if self.m_nurbsknot[i] > spans[-1]:
                spans.append(float(self.m_nurbsknot[i]))

        return spans

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometry
    # ═══════════════════════════════════════════════════════════════════════════
    def get_next_discontinuity(self, continuity_type: int, t0: float, t1: float) -> tuple[bool, float]:
        """Return (found, t) for the first interior nurbsknot in (t0, t1) whose multiplicity breaks continuity_type."""

        if not self.is_valid():
            return False, 0.0

        if t0 >= t1:
            return False, 0.0

        d0, d1 = self.domain()

        if t0 < d0:
            t0 = d0

        if t1 > d1:
            t1 = d1

        if t0 >= t1:
            return False, 0.0

        for i in range(self.m_order - 1, self.m_cv_count - 1):
            t = float(self.m_nurbsknot[i])

            if t <= t0 or t >= t1:
                continue

            mult = self.nurbsknot_multiplicity(i)
            found = False

            if continuity_type == 0:
                found = mult >= self.m_order
            elif continuity_type == 1 or continuity_type == 3 or continuity_type == 4:
                found = mult >= self.m_order - 1
            elif continuity_type == 2:
                found = mult >= self.m_order - 2

            if not found:
                continue

            return True, t

        return False, 0.0

    def length(self, tolerance: float = 1e-6) -> float:
        """Return the arc length by 10-point Gauss-Legendre over 4 subdivisions per span."""

        if not self.is_valid():
            return 0.0

        SUBDIVISIONS = 4
        total = 0.0
        n_spans = self.m_cv_count - self.m_order + 1

        for span in range(n_spans):
            span_a = self.m_nurbsknot[self.m_order - 2 + span]
            span_b = self.m_nurbsknot[self.m_order - 1 + span]

            if span_b <= span_a:
                continue

            span_width = (span_b - span_a) / SUBDIVISIONS

            for sub in range(SUBDIVISIONS):
                a = span_a + sub * span_width
                b = a + span_width
                mid = (a + b) * 0.5
                half = (b - a) * 0.5
                s = 0.0

                for i in range(10):
                    s += GL_W[i] * self.evaluate(mid + half * GL_X[i], 1)[1].magnitude()

                total += half * s

        return total

    def to_polyline_adaptive(
        self,
        angle_tolerance: float = 0.1,
        min_edge_length: float = 0.0,
        max_edge_length: float = 0.0,
    ) -> tuple[list[Point], list[float]]:
        """Return the chord-deviation subdivision points and parameters."""

        points: list[Point] = []
        params: list[float] = []

        if not self.is_valid():
            return points, params

        if angle_tolerance <= 0.0:
            angle_tolerance = 0.1

        curve_len = self.length()

        if max_edge_length <= 0.0:
            max_edge_length = curve_len / 10.0

        if min_edge_length <= 0.0:
            min_edge_length = curve_len / 1000.0

        if min_edge_length > max_edge_length:
            min_edge_length = max_edge_length * 0.1

        samples = self._adaptive_samples(angle_tolerance, min_edge_length, max_edge_length)

        for t, p in samples:
            points.append(p)
            params.append(t)

        return points, params

    def divide_by_count(
        self, count: int, include_endpoints: bool = True
    ) -> tuple[list[Point], list[float]]:
        """Return count points at equal arc length and their parameters."""

        points: list[Point] = []
        params: list[float] = []

        if not self.is_valid():
            return points, params

        if count < 2:
            return points, params

        t0, t1 = self.domain()
        h = (t1 - t0) * 1e-8
        n_samples = max(1000, count * 100)
        dt = (t1 - t0) / n_samples
        t_vals = [0.0] * (n_samples + 1)
        s_vals = [0.0] * (n_samples + 1)
        t_vals[0] = t0
        s_vals[0] = 0.0

        for i in range(1, n_samples + 1):
            t_vals[i] = t0 + i * dt
            s_vals[i] = s_vals[i - 1] + self._arc_length_gauss(
                t_vals[i - 1], t_vals[i], h
            )

        n_segs = (count - 1) if include_endpoints else (count + 1)
        seg_len = s_vals[n_samples] / n_segs

        for i in range(count):
            s_target = seg_len * i if include_endpoints else seg_len * (i + 1)
            t = self._find_t_at_s(s_target, t_vals, s_vals, h)
            points.append(self.point_at(t))
            params.append(t)

        return points, params

    def divide_by_length(
        self, segment_length: float
    ) -> tuple[list[Point], list[float]]:
        """Return points every segment_length of arc length and their parameters."""

        points: list[Point] = []
        params: list[float] = []

        if not self.is_valid():
            return points, params

        if segment_length <= 0.0:
            return points, params

        t0, t1 = self.domain()
        h = (t1 - t0) * 1e-8
        n_samples = max(1000, int(self.length() / segment_length) * 100)
        dt = (t1 - t0) / n_samples
        t_vals = [0.0] * (n_samples + 1)
        s_vals = [0.0] * (n_samples + 1)
        t_vals[0] = t0
        s_vals[0] = 0.0

        for i in range(1, n_samples + 1):
            t_vals[i] = t0 + i * dt
            s_vals[i] = s_vals[i - 1] + self._arc_length_gauss(
                t_vals[i - 1], t_vals[i], h
            )

        total_len = s_vals[n_samples]
        s = 0.0

        while s <= total_len + 1e-10:
            t = self._find_t_at_s(s, t_vals, s_vals, h)
            points.append(self.point_at(t))
            params.append(t)
            s += segment_length

        return points, params

    # ═══════════════════════════════════════════════════════════════════════════
    # Evaluation
    # ═══════════════════════════════════════════════════════════════════════════
    def point_at(self, t: float) -> Point:
        """Return the point at parameter t."""

        if not self.is_valid():
            return Point(0.0, 0.0, 0.0)

        span = self._find_span(t)
        basis = self._basis_functions(span, t)
        x = 0.0
        y = 0.0
        z = 0.0
        w = 0.0

        for i in range(self.m_order):
            cv_ptr = self.cv(span + i)

            if cv_ptr is None:
                continue

            N = basis[i]
            x += N * cv_ptr[0]
            y += N * (cv_ptr[1] if self.m_dim > 1 else 0.0)
            z += N * (cv_ptr[2] if self.m_dim > 2 else 0.0)

            if self.m_is_rat:
                w += N * cv_ptr[self.m_dim]
            else:
                w = 1.0

        if self.m_is_rat and w != 0.0:
            return Point(x / w, y / w, z / w)

        return Point(x, y, z)

    def evaluate(self, t: float, derivative_count: int = 0) -> list[Vector]:
        """Return [point, first derivative, ..., derivative_count] with zeros past the degree."""

        result: list[Vector] = []

        if not self.is_valid():
            result.append(Vector(0.0, 0.0, 0.0))

            return result

        max_derivs = min(derivative_count, self.degree())
        span = self._find_span(t)
        ders = self._basis_functions_derivatives(span, t, max_derivs)
        Aders = self._homogeneous_derivatives(span, ders)
        Cders = [[0.0, 0.0, 0.0] for _ in range(max_derivs + 1)]

        if not self.m_is_rat:
            for k in range(max_derivs + 1):
                Cders[k] = [Aders[k][0], Aders[k][1], Aders[k][2]]
        else:
            for k in range(max_derivs + 1):
                w = Aders[0][3]
                inv_w = 1.0 / w if w != 0.0 else 0.0
                Ck_x = Aders[k][0]
                Ck_y = Aders[k][1]
                Ck_z = Aders[k][2]

                for j in range(1, k + 1):
                    coeff = math.gamma(k + 1) / (
                        math.gamma(j + 1) * math.gamma(k - j + 1)
                    )
                    wj = Aders[j][3]
                    Ck_x -= coeff * wj * Cders[k - j][0]
                    Ck_y -= coeff * wj * Cders[k - j][1]
                    Ck_z -= coeff * wj * Cders[k - j][2]

                Cders[k] = [Ck_x * inv_w, Ck_y * inv_w, Ck_z * inv_w]

        for k in range(max_derivs + 1):
            result.append(Vector(Cders[k][0], Cders[k][1], Cders[k][2]))

        for k in range(max_derivs + 1, derivative_count + 1):
            result.append(Vector(0.0, 0.0, 0.0))

        return result

    def tangent_at(self, t: float) -> Vector:
        """Return the unit tangent by central difference."""

        if not self.is_valid():
            return Vector(0.0, 0.0, 0.0)

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

        tan = p2 - p1

        if tan.magnitude() > 1e-14:
            tan.normalize_self()

        return tan

    def curvature_at(self, t: float) -> float:
        """Return |C' x C''| / |C'|^3."""

        d = self.evaluate(t, 2)

        if len(d) < 3:
            return 0.0

        s = d[1].magnitude()

        if s < Tolerance.ZERO_TOLERANCE:
            return 0.0

        return d[1].cross(d[2]).magnitude() / (s * s * s)

    def closest_parameter(self, test_point: Point) -> float:
        """Return the parameter of the closest point to test_point."""

        from .closest import Closest

        return Closest.curve_point(self, test_point)[0]

    def closest_point(self, test_point: Point) -> Point:
        """Return the closest point to test_point."""
        return self.point_at(self.closest_parameter(test_point))

    def closest_parameters_curve(self, other: NurbsCurve) -> tuple[float, float]:
        """Return the parameters (u, v) where this curve and other are closest."""

        from .closest import Closest

        u, v, dist = Closest.curve_curve(self, other)

        return u, v

    def closest_points_curve(self, other: NurbsCurve) -> tuple[Point, Point]:
        """Return the points where this curve and other are closest."""

        u, v = self.closest_parameters_curve(other)

        return self.point_at(u), other.point_at(v)

    def plane_at(self, t: float, normalized: bool) -> Plane:
        """Return the Frenet frame (tangent, normal, binormal); normalized maps t from [0, 1]."""

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

        if param <= t0 + h:
            p0 = self.point_at(t0)
            pp = self.point_at(t0 + h)
            pp2 = self.point_at(t0 + 2 * h)
            d1 = pp - p0
            d2 = Vector(
                (pp2[0] - 2 * pp[0] + p0[0]) / (h * h),
                (pp2[1] - 2 * pp[1] + p0[1]) / (h * h),
                (pp2[2] - 2 * pp[2] + p0[2]) / (h * h),
            )

            return NurbsCurve._frenet_frame(origin, d1, d2)

        if param >= t1 - h:
            pm = self.point_at(t1 - h)
            p0 = self.point_at(t1)
            pm2 = self.point_at(t1 - 2 * h)
            d1 = p0 - pm
            d2 = Vector(
                (p0[0] - 2 * pm[0] + pm2[0]) / (h * h),
                (p0[1] - 2 * pm[1] + pm2[1]) / (h * h),
                (p0[2] - 2 * pm[2] + pm2[2]) / (h * h),
            )

            return NurbsCurve._frenet_frame(origin, d1, d2)

        pm = self.point_at(param - h)
        p0 = self.point_at(param)
        pp = self.point_at(param + h)
        d1 = (pp - pm) / (2 * h)
        d2 = Vector(
            (pp[0] - 2 * p0[0] + pm[0]) / (h * h),
            (pp[1] - 2 * p0[1] + pm[1]) / (h * h),
            (pp[2] - 2 * p0[2] + pm[2]) / (h * h),
        )

        return NurbsCurve._frenet_frame(origin, d1, d2)

    def perpendicular_plane_at(self, t: float, normalized: bool) -> Plane:
        """Return the rotation minimizing frame by double reflection (Wang et al. 2008)."""

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

        frame = self._start_frame()

        if frame is None:
            return Plane.invalid()

        T0, r0 = frame
        origin = self.point_at(param)

        if abs(param - t0) < 1e-14:
            s0 = T0.cross(r0)
            s0.normalize_self()

            return Plane.from_frame(origin, r0, s0, T0)

        ri = self._double_reflection(param, r0, T0)
        T = self.tangent_at(param)
        T.normalize_self()

        ri_dot_T = ri.dot(T)
        ri -= T * ri_dot_T

        if ri.magnitude() > 1e-14:
            ri.normalize_self()

        s = T.cross(ri)
        s.normalize_self()

        return Plane.from_frame(origin, ri, s, T)

    def get_perpendicular_planes(self, count: int) -> list[Plane]:
        """Return count + 1 rotation minimizing frames at equal arc length."""

        frames: list[Plane] = []
        pts, params = self.divide_by_count(count + 1, True)

        for t in params:
            frames.append(self.perpendicular_plane_at(t, False))

        return frames

    def point_at_start(self) -> Point:
        """Return the point at the domain start."""
        return self.point_at(self.domain_start())

    def point_at_middle(self) -> Point:
        """Return the point at the domain midpoint."""
        return self.point_at(self.domain_middle())

    def point_at_end(self) -> Point:
        """Return the point at the domain end."""
        return self.point_at(self.domain_end())

    def set_start_point(self, start_point: Point) -> bool:
        """Clamp and move the first CV."""

        if not self.is_valid() or not self.clamp_end(2):
            return False

        w = self.weight(0) if self.m_is_rat else 1.0

        if self.m_is_rat and w != 1.0:
            self.set_cv_4d(
                0, start_point[0] * w, start_point[1] * w, start_point[2] * w, w
            )
        else:
            self.set_cv(0, start_point)

            if self.m_is_rat:
                self.set_weight(0, w)

        return True

    def set_end_point(self, end_point: Point) -> bool:
        """Clamp and move the last CV."""

        if not self.is_valid() or not self.clamp_end(2):
            return False

        last = self.m_cv_count - 1
        w = self.weight(last) if self.m_is_rat else 1.0

        if self.m_is_rat and w != 1.0:
            self.set_cv_4d(
                last, end_point[0] * w, end_point[1] * w, end_point[2] * w, w
            )
        else:
            self.set_cv(last, end_point)

            if self.m_is_rat:
                self.set_weight(last, w)

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # Modifications
    # ═══════════════════════════════════════════════════════════════════════════
    def reverse(self) -> bool:
        """Reverse the direction keeping the domain."""

        if not self.is_valid():
            return False

        d0, d1 = self.domain()

        for i in range(len(self.m_nurbsknot)):
            self.m_nurbsknot[i] = d0 + d1 - self.m_nurbsknot[i]

        self.m_nurbsknot = self.m_nurbsknot[::-1].copy()

        for i in range(self.m_cv_count // 2):
            j = self.m_cv_count - 1 - i
            xi, yi, zi, wi = self.get_cv_4d(i)
            xj, yj, zj, wj = self.get_cv_4d(j)
            self.set_cv_4d(i, xj, yj, zj, wj)
            self.set_cv_4d(j, xi, yi, zi, wi)

        return True

    def swap_coordinates(self, axis_i: int, axis_j: int) -> bool:
        """Swap two coordinate axes of every CV."""

        if not self.is_valid():
            return False

        if axis_i < 0 or axis_i >= self.m_dim:
            return False

        if axis_j < 0 or axis_j >= self.m_dim:
            return False

        if axis_i == axis_j:
            return True

        for cv_idx in range(self.m_cv_count):
            cv_ptr = self.cv(cv_idx)
            cv_ptr[axis_i], cv_ptr[axis_j] = cv_ptr[axis_j], cv_ptr[axis_i]

        return True

    def trim(self, t0: float, t1: float) -> bool:
        """Keep [t0, t1] by nurbsknot insertion."""

        if not self.is_valid() or t0 >= t1:
            return False

        d0, d1 = self.domain()

        if t0 < d0 - Tolerance.ZERO_TOLERANCE or t1 > d1 + Tolerance.ZERO_TOLERANCE:
            return False

        t0 = max(t0, d0)
        t1 = min(t1, d1)

        if (
            abs(t0 - d0) < Tolerance.ZERO_TOLERANCE
            and abs(t1 - d1) < Tolerance.ZERO_TOLERANCE
        ):
            return True

        p = self.degree()
        trim_start = t0 > d0 + Tolerance.ZERO_TOLERANCE
        trim_end = t1 < d1 - Tolerance.ZERO_TOLERANCE

        stol = (abs(d0) + abs(d1) + abs(d1 - d0)) * SQRT_EPSILON

        for k in self.m_nurbsknot:
            if trim_start and abs(k - t0) <= stol and abs(k - t0) > 0.0:
                t0 = float(k)

            if trim_end and abs(k - t1) <= stol and abs(k - t1) > 0.0:
                t1 = float(k)

        if t0 >= t1:
            return False

        if trim_start and not self.insert_nurbsknot(t0, p):
            return False

        if trim_end and not self.insert_nurbsknot(t1, p):
            return False

        return self._keep_span_range(t0, t1)

    def split(self, t: float) -> tuple[NurbsCurve, NurbsCurve]:
        """Return trimmed copies on both sides of t."""

        left_curve = NurbsCurve()
        right_curve = NurbsCurve()

        if not self.is_valid():
            return left_curve, right_curve

        t0, t1 = self.domain()

        if t <= t0 or t >= t1:
            return left_curve, right_curve

        left_curve = self.duplicate()
        right_curve = self.duplicate()

        if not left_curve.trim(t0, t):
            return left_curve, right_curve

        right_curve.trim(t, t1)

        return left_curve, right_curve

    def extend(self, t0: float, t1: float) -> bool:
        """Extrapolate the domain to cover [t0, t1] by de Boor."""

        if not self.is_valid() or self.is_closed():
            return False

        d0, d1 = self.domain()
        cvdim = self.cv_size()
        changed = False

        if t0 < d0:
            if not self.clamp_end(0) or not NurbsCurve._evaluate_nurbs_de_boor(
                cvdim,
                self.m_order,
                self.m_cv_stride,
                self.m_cv,
                0,
                self.m_nurbsknot,
                0,
                1,
                t0,
            ):
                return False

            for i in range(self.m_order - 1):
                self.m_nurbsknot[i] = t0

            changed = True

        if t1 > d1:
            if not self.clamp_end(1):
                return False

            i0 = self.m_cv_count - self.m_order

            if not NurbsCurve._evaluate_nurbs_de_boor(
                cvdim,
                self.m_order,
                self.m_cv_stride,
                self.m_cv,
                i0 * self.m_cv_stride,
                self.m_nurbsknot,
                i0,
                -1,
                t1,
            ):
                return False

            kc = self.nurbsknot_count()

            for i in range(self.m_cv_count - 1, kc):
                self.m_nurbsknot[i] = t1

            changed = True

        return changed

    def to_rational(self) -> bool:
        """Add unit weights."""

        if self.m_is_rat:
            return True

        new_stride = self.m_dim + 1
        new_cv = np.zeros(self.m_cv_count * new_stride, dtype=np.float64)

        for i in range(self.m_cv_count):
            old_cv = self.cv(i)

            for j in range(self.m_dim):
                new_cv[i * new_stride + j] = old_cv[j]

            new_cv[i * new_stride + self.m_dim] = 1.0

        self.m_cv = new_cv
        self.m_is_rat = 1
        self.m_cv_stride = new_stride

        return True

    def to_non_rational(self, force: bool = False) -> bool:
        """Drop the weights; fails when they differ unless force."""

        if not self.m_is_rat:
            return True

        if force:
            for i in range(self.m_cv_count):
                cv_ptr = self.cv(i)

                if cv_ptr is not None:
                    cv_ptr[self.m_dim] = 1.0
        else:
            w0 = self.weight(0)

            for i in range(1, self.m_cv_count):
                if abs(self.weight(i) - w0) > Tolerance.ZERO_TOLERANCE:
                    return False

        new_stride = self.m_dim
        new_cv = np.zeros(self.m_cv_count * new_stride, dtype=np.float64)

        for i in range(self.m_cv_count):
            p = self.get_cv(i)
            new_cv[i * new_stride] = p[0]

            if self.m_dim > 1:
                new_cv[i * new_stride + 1] = p[1]

            if self.m_dim > 2:
                new_cv[i * new_stride + 2] = p[2]

        self.m_cv = new_cv
        self.m_is_rat = 0
        self.m_cv_stride = new_stride

        return True

    def clamp_end(self, end: int) -> bool:
        """Set full multiplicity at end (0 = start, 1 = end, 2 = both) with CVs adjusted."""

        if not self.is_valid():
            return False

        if end < 0 or end > 2:
            return False

        cvdim = self.cv_size()
        rc = True

        if end == 0 or end == 2:
            t = float(self.m_nurbsknot[self.m_order - 2])

            if NurbsCurve._evaluate_nurbs_de_boor(
                cvdim,
                self.m_order,
                self.m_cv_stride,
                self.m_cv,
                0,
                self.m_nurbsknot,
                0,
                1,
                t,
            ):
                for i in range(self.m_order - 2):
                    self.m_nurbsknot[i] = t
            else:
                rc = False

        if end == 1 or end == 2:
            i0 = self.m_cv_count - self.m_order
            t = float(self.m_nurbsknot[self.m_cv_count - 1])

            if NurbsCurve._evaluate_nurbs_de_boor(
                cvdim,
                self.m_order,
                self.m_cv_stride,
                self.m_cv,
                i0 * self.m_cv_stride,
                self.m_nurbsknot,
                i0,
                -1,
                t,
            ):
                kc = self.nurbsknot_count()

                for i in range(self.m_cv_count, kc):
                    self.m_nurbsknot[i] = t
            else:
                rc = False

        return rc

    def increase_degree(self, desired_degree: int) -> bool:
        """Raise the degree by blossoming without changing the shape."""

        if not self.is_valid():
            return False

        if desired_degree < 1 or desired_degree < self.degree():
            return False

        if desired_degree == self.degree():
            return True

        if not self.clamp_end(2):
            return False

        del_ = desired_degree - self.degree()

        for i in range(del_):
            if not _increment_nurbs_degree(self):
                return False

        return True

    def change_closed_curve_seam(self, t: float) -> bool:
        """Move the seam of a closed curve to t."""

        if not self.is_valid():
            return False

        if not self.is_closed():
            return False

        t0, t1 = self.domain()
        dom_len = t1 - t0
        s = (t - t0) / dom_len

        if s < 0.0 or s > 1.0:
            s = math.fmod(s, 1.0)

            if s < 0.0:
                s += 1.0

            t = t0 + s * dom_len

        if (
            abs(t - t0) < Tolerance.ZERO_TOLERANCE
            or abs(t - t1) < Tolerance.ZERO_TOLERANCE
        ):
            return True

        if t <= t0 or t >= t1:
            return True

        p = self.degree()

        if self.is_periodic():
            kc = self.nurbsknot_count()

            if self.span_count() + 2 * p > kc:
                nurbsknot_index = self._first_nurbsknot_above(t)

                if p <= nurbsknot_index <= kc - p:
                    nurbsknot_index = self._seam_nurbsknot_index(t, nurbsknot_index)

                    if nurbsknot_index < 0:
                        return False

                    if p <= nurbsknot_index < self.nurbsknot_count() - p:
                        return self._rotate_periodic_seam(nurbsknot_index, t, dom_len)

        return self._split_seam(t, dom_len)

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        control_points = []

        for i in range(self.m_cv_count):
            if self.m_is_rat:
                x, y, z, w = self.get_cv_4d(i)
                control_points.append([x, y, z, w])
            else:
                p = self.get_cv(i)
                control_points.append([p[0], p[1], p[2]])

        linecolors_arr = []

        for c in self.linecolors:
            linecolors_arr.extend([c.r, c.g, c.b, c.a])

        pointcolors_arr = []

        for c in self.pointcolors:
            pointcolors_arr.extend([c.r, c.g, c.b, c.a])

        return {
            "control_points": control_points,
            "cv_count": int(self.m_cv_count),
            "cv_stride": int(self.m_cv_stride),
            "dimension": int(self.m_dim),
            "guid": self.guid,
            "is_rational": self.m_is_rat != 0,
            "linecolors": linecolors_arr,
            "name": self.name,
            "nurbsknots": self.m_nurbsknot.tolist(),
            "order": int(self.m_order),
            "pointcolors": pointcolors_arr,
            "type": "NurbsCurve",
            "width": float(self.width),
        }

    @classmethod
    def __jsonload__(cls, data: dict, guid: str | None = None, name: str | None = None) -> NurbsCurve:
        """Deserialize from a JSON object."""

        curve = cls()

        if "dimension" not in data or "order" not in data or "cv_count" not in data:
            return curve

        dim = int(data["dimension"])
        is_rat = bool(data.get("is_rational", False))
        order = int(data["order"])
        cv_count = int(data["cv_count"])
        curve.create_curve(dim, is_rat, order, cv_count)

        if "nurbsknots" in data:
            curve.m_nurbsknot = np.array(data["nurbsknots"], dtype=np.float64)

        if "control_points" in data:
            cps = data["control_points"]
            n = min(cv_count, len(cps))

            for i in range(n):
                x = cps[i][0]
                y = cps[i][1]
                z = cps[i][2] if len(cps[i]) > 2 else 0.0

                if is_rat and len(cps[i]) > 3:
                    curve.set_cv_4d(i, x, y, z, cps[i][3])
                else:
                    curve.set_cv(i, Point(x, y, z))

        curve.guid = guid if guid is not None else data.get("guid", str(uuid.uuid4()))
        curve.name = name if name is not None else data.get("name", "my_nurbscurve")
        curve.width = data.get("width", 1.0)

        arr = data.get("pointcolors", [])

        for i in range(0, len(arr) - 3, 4):
            curve.pointcolors.append(Color(arr[i], arr[i + 1], arr[i + 2], arr[i + 3]))

        arr = data.get("linecolors", [])

        for i in range(0, len(arr) - 3, 4):
            curve.linecolors.append(Color(arr[i], arr[i + 1], arr[i + 2], arr[i + 3]))

        return curve

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> NurbsCurve:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> NurbsCurve:
        """Read from a JSON file."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> nurbscurve_pb2.NurbsCurve:
        """Convert to the protobuf message."""

        from .proto import nurbscurve_pb2

        proto = nurbscurve_pb2.NurbsCurve()

        if self.has_guid():
            proto.guid = self.guid

        proto.name = self.name
        proto.dimension = int(self.m_dim)
        proto.is_rational = self.m_is_rat != 0
        proto.order = int(self.m_order)
        proto.cv_count = int(self.m_cv_count)
        proto.cv_stride = int(self.m_cv_stride)
        proto.nurbsknots.extend(self.m_nurbsknot.tolist())
        proto.cvs.extend(self.m_cv.tolist())
        proto.width = float(self.width)

        for c in self.pointcolors:
            cp = proto.pointcolors.add()
            cp.r = c.r
            cp.g = c.g
            cp.b = c.b
            cp.a = c.a

        for c in self.linecolors:
            cp = proto.linecolors.add()
            cp.r = c.r
            cp.g = c.g
            cp.b = c.b
            cp.a = c.a

        return proto

    @classmethod
    def from_proto(cls, proto: nurbscurve_pb2.NurbsCurve) -> NurbsCurve:
        """Construct from the protobuf message."""

        curve = cls(proto.dimension, proto.is_rational, proto.order, proto.cv_count)

        if proto.guid:
            curve.guid = proto.guid

        curve.name = proto.name
        curve.width = proto.width if proto.width != 0.0 else 1.0
        curve.m_nurbsknot = np.array(list(proto.nurbsknots), dtype=np.float64)
        curve.m_cv = np.array(list(proto.cvs), dtype=np.float64)

        for c in proto.pointcolors:
            curve.pointcolors.append(Color(c.r, c.g, c.b, c.a))

        for c in proto.linecolors:
            curve.linecolors.append(Color(c.r, c.g, c.b, c.a))

        return curve

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> NurbsCurve:
        """Deserialize from protobuf bytes."""

        from .proto import nurbscurve_pb2

        proto = nurbscurve_pb2.NurbsCurve()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write to a protobuf file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> NurbsCurve:
        """Read from a protobuf file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "NurbsCurve(name=..., degree=..., cvs=...)"."""
        return f"NurbsCurve(name={self.name}, degree={self.degree()}, cvs={self.cv_count()})"

    def __repr__(self) -> str:
        """Return the multi-line form with every control point."""

        prec = Tolerance.ROUNDING
        rational = "true" if self.m_is_rat else "false"
        result = f"NurbsCurve(\n  name={self.name},\n  degree={self.degree()},\n  cvs={self.m_cv_count},\n  rational={rational},\n  control_points=[\n"

        for i in range(self.m_cv_count):
            p = self.get_cv(i)
            result += f"    {TOLERANCE.format_number(p[0], prec)}, {TOLERANCE.format_number(p[1], prec)}, {TOLERANCE.format_number(p[2], prec)}\n"

        result += "  ]\n)"

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════════════════════════
    def _span_is_linear(
        self, span_index: int, min_length: float, tolerance: float
    ) -> bool:
        """Return whether the span has full end multiplicity and its CVs lie on its chord."""

        if not self.is_valid():
            return False

        if span_index < 0 or span_index >= self.m_cv_count - self.m_order:
            return False

        if self.m_dim < 2 or self.m_dim > 3:
            return False

        ki = span_index + self.m_order - 2

        if self.m_nurbsknot[ki] >= self.m_nurbsknot[ki + 1]:
            return False

        mult_start = 1
        i = ki - 1

        while i >= 0 and self.m_nurbsknot[i] == self.m_nurbsknot[ki]:
            mult_start += 1
            i -= 1

        mult_end = 1
        kc = self.nurbsknot_count()
        i = ki + 2

        while i < kc and self.m_nurbsknot[i] == self.m_nurbsknot[ki + 1]:
            mult_end += 1
            i += 1

        if mult_start < self.m_order - 1 or mult_end < self.m_order - 1:
            return False

        p0 = self.get_cv(span_index)
        p1 = self.get_cv(span_index + self.m_order - 1)
        line_vec = p1 - p0
        line_length = line_vec.magnitude()

        if line_length < min_length:
            return False

        for i in range(1, self.m_order - 1):
            p = self.get_cv(span_index + i)
            v = p - p0

            if line_vec.cross(v).magnitude() / line_length > tolerance:
                return False

            t = v.dot(line_vec) / (line_length * line_length)

            if t < -0.01 or t > 1.01:
                return False

        return True

    def _span_is_singular(self, span_index: int) -> bool:
        """Return whether the span is collapsed to a point."""

        if not self.is_valid():
            return False

        if span_index < 0 or span_index >= self.m_cv_count - self.m_order:
            return False

        ki = span_index + self.m_order - 2

        if self.m_nurbsknot[ki] >= self.m_nurbsknot[ki + 1]:
            return True

        p0 = self.get_cv(span_index)

        for i in range(1, self.m_order):
            if p0.distance(self.get_cv(span_index + i)) > Tolerance.ZERO_TOLERANCE:
                return False

        return True

    def _find_span(self, t: float) -> int:
        """Return the span index of t relative to nurbsknot[order - 2] by binary search."""

        offset = self.m_order - 2
        len_ = self.m_cv_count - self.m_order + 2

        if t <= self.m_nurbsknot[offset]:
            return 0

        if t >= self.m_nurbsknot[offset + len_ - 1]:
            return len_ - 2

        low = 0
        high = len_ - 1

        while high > low + 1:
            mid = (low + high) // 2

            if t < self.m_nurbsknot[offset + mid]:
                high = mid
            else:
                low = mid

        return low

    def _basis_functions(self, span: int, t: float) -> list[float]:
        """Compute the Cox-de Boor basis at t."""

        basis = [0.0] * self.m_order
        left = [0.0] * self.m_order
        right = [0.0] * self.m_order
        offset = self.m_order - 2 + span
        basis[0] = 1.0

        for j in range(1, self.m_order):
            left[j] = t - self.m_nurbsknot[offset + 1 - j]
            right[j] = self.m_nurbsknot[offset + j] - t

            saved = 0.0

            for r in range(j):
                denom = right[r + 1] + left[j - r]
                temp = basis[r] / denom if denom != 0.0 else 0.0
                basis[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp

            basis[j] = saved

        return basis

    def _basis_functions_derivatives(
        self, span: int, t: float, deriv_order: int
    ) -> list[list[float]]:
        """Compute the basis derivatives (Piegl & Tiller A2.3)."""

        p = self.degree()
        n_der = min(deriv_order, p)
        ders = [[0.0] * (p + 1) for _ in range(n_der + 1)]
        ndu = self._basis_functions_ndu(span, t)

        for j in range(p + 1):
            ders[0][j] = ndu[j][p]

        a = [[0.0] * (p + 1) for _ in range(2)]

        for r in range(p + 1):
            s1 = 0
            s2 = 1
            a[0][0] = 1.0

            for k in range(1, n_der + 1):
                d = 0.0
                rk = r - k
                pk = p - k

                if r >= k:
                    a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                    d = a[s2][0] * ndu[rk][pk]

                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else p - r

                for j in range(j1, j2 + 1):
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
                    d += a[s2][j] * ndu[rk + j][pk]

                if r <= pk:
                    a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
                    d += a[s2][k] * ndu[r][pk]

                ders[k][r] = d
                s1, s2 = s2, s1

        scale = float(p)

        for k in range(1, n_der + 1):
            for j in range(p + 1):
                ders[k][j] *= scale

            scale *= float(p - k)

        return ders

    def _basis_functions_ndu(self, span: int, t: float) -> list[list[float]]:
        """Compute the triangular table of basis functions and nurbsknot differences (Piegl & Tiller A2.3)."""

        p = self.degree()
        left = [0.0] * (p + 1)
        right = [0.0] * (p + 1)
        offset = self.m_order - 2 + span
        ndu = [[0.0] * (p + 1) for _ in range(p + 1)]
        ndu[0][0] = 1.0

        for j in range(1, p + 1):
            left[j] = t - self.m_nurbsknot[offset + 1 - j]
            right[j] = self.m_nurbsknot[offset + j] - t

            saved = 0.0

            for r in range(j):
                ndu[j][r] = right[r + 1] + left[j - r]
                temp = ndu[r][j - 1] / ndu[j][r]
                ndu[r][j] = saved + right[r + 1] * temp
                saved = left[j - r] * temp

            ndu[j][j] = saved

        return ndu

    def _deep_copy_from(self, src: NurbsCurve) -> None:
        """Copy every field but the guid."""

        self.m_dim = src.m_dim
        self.m_is_rat = src.m_is_rat
        self.m_order = src.m_order
        self.m_cv_count = src.m_cv_count
        self.m_cv_stride = src.m_cv_stride
        self.m_nurbsknot = np.array(src.m_nurbsknot, dtype=np.float64)
        self.m_cv = np.array(src.m_cv, dtype=np.float64)
        self._guid = None
        self.name = src.name
        self.width = src.width
        self.pointcolors = list(src.pointcolors)
        self.linecolors = list(src.linecolors)

    @staticmethod
    def _evaluate_nurbs_de_boor(
        cv_dim: int,
        order: int,
        cv_stride: int,
        cv: np.ndarray,
        cv0: int,
        nurbsknots: np.ndarray,
        kn0: int,
        side: int,
        t: float,
    ) -> bool:
        """Reshape one span's CVs so it starts (side > 0) or ends (side < 0) at t (OpenNURBS ON_EvaluateNurbsDeBoor)."""

        degree = order - 1

        if nurbsknots[kn0 + degree - 1] == nurbsknots[kn0 + degree]:
            return False

        if side < 0:
            return NurbsCurve._de_boor_end(cv_dim, order, cv_stride, cv, cv0, nurbsknots, kn0, t)

        return NurbsCurve._de_boor_start(cv_dim, order, cv_stride, cv, cv0, nurbsknots, kn0, t)

    @staticmethod
    def _de_boor_end(
        cv_dim: int,
        order: int,
        cv_stride: int,
        cv: np.ndarray,
        cv0: int,
        nurbsknots: np.ndarray,
        kn0: int,
        t: float,
    ) -> bool:
        """Reshape one span's CVs so it ends at t."""

        degree = order - 1
        t0 = float(nurbsknots[kn0 + degree - 1])
        t1 = float(nurbsknots[kn0 + degree])

        if t == t1 and t1 == nurbsknots[kn0 + 2 * degree - 1]:
            return True

        fully_multiple = t0 == nurbsknots[kn0]
        kn = kn0 + degree - 1
        delta_t = [0.0] * degree

        if not fully_multiple:
            for idx in range(degree):
                delta_t[idx] = t - float(nurbsknots[kn - idx])

        for k in range(order - 1, 0, -1):
            for i in range(k - 1, -1, -1):
                di = k - 1 - i

                if fully_multiple:
                    alpha1 = (t - t0) / (float(nurbsknots[kn + k - di]) - t0)
                else:
                    alpha1 = delta_t[di] / (
                        float(nurbsknots[kn + k - di]) - float(nurbsknots[kn - di])
                    )

                alpha0 = 1.0 - alpha1
                row1 = cv0 + (order - k + i) * cv_stride
                row0 = row1 - cv_stride

                for j in range(cv_dim):
                    cv[row1 + j] = cv[row0 + j] * alpha0 + cv[row1 + j] * alpha1

        return True

    @staticmethod
    def _de_boor_start(
        cv_dim: int,
        order: int,
        cv_stride: int,
        cv: np.ndarray,
        cv0: int,
        nurbsknots: np.ndarray,
        kn0: int,
        t: float,
    ) -> bool:
        """Reshape one span's CVs so it starts at t."""

        degree = order - 1
        t0 = float(nurbsknots[kn0 + degree - 1])
        t1 = float(nurbsknots[kn0 + degree])

        if t == t0 and t0 == nurbsknots[kn0]:
            return True

        fully_multiple = t1 == nurbsknots[kn0 + 2 * degree - 1]
        kn = kn0 + degree
        delta_t = [0.0] * degree

        if not fully_multiple:
            for idx in range(degree):
                delta_t[idx] = float(nurbsknots[kn + idx]) - t

        for k in range(order - 1, 0, -1):
            for i in range(k):
                if fully_multiple:
                    alpha0 = (t1 - t) / (t1 - float(nurbsknots[kn - k + i]))
                else:
                    alpha0 = delta_t[i] / (
                        float(nurbsknots[kn + i]) - float(nurbsknots[kn - k + i])
                    )

                alpha1 = 1.0 - alpha0
                row0 = cv0 + i * cv_stride
                row1 = row0 + cv_stride

                for j in range(cv_dim):
                    cv[row0 + j] = cv[row0 + j] * alpha0 + cv[row1 + j] * alpha1

        return True

    @staticmethod
    def _solve_dense(matrix: list[list[float]], rhs: list[float], n: int, dim: int) -> bool:
        """Solve matrix * x = rhs in place by Gaussian elimination with partial pivoting, dim values per row."""

        for col in range(n):
            pivot = col

            for row in range(col + 1, n):
                if abs(matrix[row][col]) > abs(matrix[pivot][col]):
                    pivot = row

            if pivot != col:
                matrix[col], matrix[pivot] = matrix[pivot], matrix[col]

                for d in range(dim):
                    rhs[col * dim + d], rhs[pivot * dim + d] = rhs[pivot * dim + d], rhs[col * dim + d]

            if abs(matrix[col][col]) < 1e-300:
                return False

            for row in range(col + 1, n):
                factor = matrix[row][col] / matrix[col][col]

                for j in range(col, n):
                    matrix[row][j] -= factor * matrix[col][j]

                for d in range(dim):
                    rhs[row * dim + d] -= factor * rhs[col * dim + d]

        for i in range(n - 1, -1, -1):
            for d in range(dim):
                sum_ = rhs[i * dim + d]

                for j in range(i + 1, n):
                    sum_ -= matrix[i][j] * rhs[j * dim + d]

                rhs[i * dim + d] = sum_ / matrix[i][i]

        return True

    def _derivative_at(self, t: float, h: float) -> Vector:
        """Return the un-normalized derivative by finite difference with step h."""

        t0, t1 = self.domain()

        if t <= t0 + h:
            p1 = self.point_at(t0)
            p2 = self.point_at(t0 + h)
            dt = h
        elif t >= t1 - h:
            p1 = self.point_at(t1 - h)
            p2 = self.point_at(t1)
            dt = h
        else:
            p1 = self.point_at(t - h)
            p2 = self.point_at(t + h)
            dt = 2.0 * h

        return (p2 - p1) / dt

    def _arc_length_gauss(self, ta: float, tb: float, h: float) -> float:
        """Return the arc length of [ta, tb] by 5-point Gauss-Legendre."""

        mid = (ta + tb) * 0.5
        half = (tb - ta) * 0.5
        sum_ = 0.0

        for i in range(5):
            sum_ += (
                GL_WEIGHTS[i]
                * self._derivative_at(mid + half * GL_NODES[i], h).magnitude()
            )

        return half * sum_

    def _find_t_at_s(
        self, s_target: float, t_vals: list[float], s_vals: list[float], h: float
    ) -> float:
        """Return the parameter at arc length s_target from the (t, s) table by bracketed Newton."""

        n_samples = len(t_vals) - 1

        if s_target <= 0.0:
            return t_vals[0]

        if s_target >= s_vals[n_samples]:
            return t_vals[n_samples]

        lo = 0
        hi = n_samples

        while hi - lo > 1:
            mid = (lo + hi) // 2

            if s_vals[mid] < s_target:
                lo = mid
            else:
                hi = mid

        frac = (s_target - s_vals[lo]) / (s_vals[hi] - s_vals[lo])
        t = t_vals[lo] + frac * (t_vals[hi] - t_vals[lo])
        t_lo = t_vals[lo]
        t_hi = t_vals[hi]

        for iter_ in range(20):
            error = s_vals[lo] + self._arc_length_gauss(t_vals[lo], t, h) - s_target

            if abs(error) < 1e-12:
                break

            speed = self._derivative_at(t, h).magnitude()
            t_new = t - error / speed if speed >= 1e-14 else t

            if speed < 1e-14 or t_new <= t_lo or t_new >= t_hi:
                if error > 0:
                    t_hi = t
                else:
                    t_lo = t

                t = (t_lo + t_hi) * 0.5
            else:
                t = t_new

        return t

    @staticmethod
    def _frenet_frame(origin: Point, d1: Vector, d2: Vector) -> Plane:
        """Return the Frenet frame from first and second derivatives, world Z then Y as normal fallback."""

        if d1.magnitude() < 1e-14:
            return Plane.invalid()

        T = Vector(d1[0], d1[1], d1[2])
        T.normalize_self()

        d2_dot_T = d2.dot(T)
        N = d2 - T * d2_dot_T
        n_mag = N.magnitude()

        if n_mag < 1e-14:
            N = T.cross(Vector(0.0, 0.0, 1.0))
            n_mag = N.magnitude()

            if n_mag < 1e-14:
                N = T.cross(Vector(0.0, 1.0, 0.0))
                n_mag = N.magnitude()

        if n_mag > 1e-14:
            N.normalize_self()

        B = T.cross(N)
        B.normalize_self()

        return Plane.from_frame(origin, T, N, B)

    @staticmethod
    def _bessel_tangent(points: list[Point], i0: int, i1: int, i2: int) -> Vector:
        """Return the unit Bessel tangent at points[i0] from the parabola through i0, i1, i2."""

        d01 = points[i0].distance(points[i1])
        d21 = points[i2].distance(points[i1])

        if d01 + d21 < 1e-300:
            return Vector(0.0, 0.0, 0.0)

        s = d01 / (d01 + d21)
        t = 1.0 - s
        denom = 2.0 * s * t

        if denom < 1e-16:
            chord = points[i1] - points[i0]

            return chord if chord.normalize_self() else Vector(0.0, 0.0, 0.0)

        cvx = (-t * t * points[i0][0] + points[i1][0] - s * s * points[i2][0]) / denom
        cvy = (-t * t * points[i0][1] + points[i1][1] - s * s * points[i2][1]) / denom
        cvz = (-t * t * points[i0][2] + points[i1][2] - s * s * points[i2][2]) / denom
        tangent = Point(cvx, cvy, cvz) - points[i0]

        return tangent if tangent.normalize_self() else Vector(0.0, 0.0, 0.0)

    @staticmethod
    def _lagrange_tangent(
        points: list[Point], params: list[float], i0: int, m: int, t: float
    ) -> Vector:
        """Return the derivative at t of the Lagrange polynomial through m points from i0 (OCCT BuildTangents)."""

        result = Vector(0.0, 0.0, 0.0)

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
            result += Vector(Pj[0], Pj[1], Pj[2]) * dsum

        return result

    @staticmethod
    def _create_interpolated_periodic(
        points: list[Point], parameterization: CurveNurbsKnotStyle
    ) -> NurbsCurve:
        """Construct the closed interpolated cubic through points, wrapped by three CVs."""

        n = len(points)
        dim = 3
        order = 4
        cv_count = n + 3
        kc = cv_count + order - 2
        params = NurbsCurve._periodic_interpolation_parameters(points, parameterization)
        dmin = 1e300
        dmax = 0.0

        for i in range(n):
            d = params[i + 1] - params[i]

            if d < dmin:
                dmin = d

            if d > dmax:
                dmax = d

        if dmax <= 0.0 or dmax * SQRT_EPSILON >= dmin:
            return NurbsCurve()

        nurbsknots = NurbsCurve._periodic_interpolation_nurbsknots(params, cv_count)
        A = [[0.0] * n for _ in range(n)]
        cv = [0.0] * (n * dim)

        for i in range(n):
            basis = nurbsknot.eval_basis(order, nurbsknots, i, params[i])
            A[i][i % n] += basis[0]
            A[i][(i + 1) % n] += basis[1]
            A[i][(i + 2) % n] += basis[2]

            for d in range(dim):
                cv[i * dim + d] = points[i][d]

        if not NurbsCurve._solve_dense(A, cv, n, dim):
            return NurbsCurve()

        curve = NurbsCurve(dim, False, order, cv_count)

        for i in range(kc):
            curve.set_nurbsknot(i, nurbsknots[i])

        for i in range(n):
            curve.set_cv(i, Point(cv[i * 3], cv[i * 3 + 1], cv[i * 3 + 2]))

        curve.set_cv(n, curve.get_cv(0))
        curve.set_cv(n + 1, curve.get_cv(1))
        curve.set_cv(n + 2, curve.get_cv(2))

        return curve

    @staticmethod
    def _periodic_interpolation_parameters(
        points: list[Point], parameterization: CurveNurbsKnotStyle
    ) -> list[float]:
        """Return the n + 1 parameters of the closed point loop, uniform or (square root) chord spaced."""

        n = len(points)
        base_style = CurveNurbsKnotStyle.Chord

        if parameterization == CurveNurbsKnotStyle.UniformPeriodic:
            base_style = CurveNurbsKnotStyle.Uniform

        if parameterization == CurveNurbsKnotStyle.ChordSquareRootPeriodic:
            base_style = CurveNurbsKnotStyle.ChordSquareRoot

        params = [0.0] * (n + 1)

        if base_style == CurveNurbsKnotStyle.Uniform:
            for i in range(1, n + 1):
                params[i] = float(i)

            return params

        for i in range(1, n):
            d = points[i - 1].distance(points[i])

            if base_style == CurveNurbsKnotStyle.ChordSquareRoot:
                d = math.sqrt(d)

            params[i] = params[i - 1] + d

        d_close = points[n - 1].distance(points[0])

        if base_style == CurveNurbsKnotStyle.ChordSquareRoot:
            d_close = math.sqrt(d_close)

        params[n] = params[n - 1] + d_close

        return params

    @staticmethod
    def _periodic_interpolation_nurbsknots(params: list[float], cv_count: int) -> list[float]:
        """Return the periodic nurbsknots over params, extended by the wrapped spans at both ends."""

        n = len(params) - 1
        nurbsknots = [0.0] * (cv_count + 2)

        for i in range(n + 1):
            nurbsknots[i + 2] = params[i]

        nurbsknots[cv_count] = nurbsknots[3] - nurbsknots[2] + nurbsknots[cv_count - 1]
        nurbsknots[1] = nurbsknots[cv_count - 2] - nurbsknots[cv_count - 1] + nurbsknots[2]
        nurbsknots[cv_count + 1] = nurbsknots[4] - nurbsknots[3] + nurbsknots[cv_count]
        nurbsknots[0] = nurbsknots[cv_count - 3] - nurbsknots[cv_count - 2] + nurbsknots[1]

        return nurbsknots

    @staticmethod
    def _create_interpolated_clamped(
        points: list[Point],
        parameterization: CurveNurbsKnotStyle,
        end_condition: CurveInterpStyle,
    ) -> NurbsCurve:
        """Construct the open interpolated cubic through points with end tangents from end_condition."""

        n = len(points)
        dim = 3
        degree = 3
        cv_count = n + 2
        pts = NurbsCurve._flatten_points(points, n)
        params = nurbsknot.compute_parameters(pts, n, dim, parameterization)
        nurbsknots = nurbsknot.build_interp_nurbsknots(params, degree)
        kc = len(nurbsknots)
        cv = NurbsCurve._interpolation_end_cvs(points, params, end_condition)

        if not NurbsCurve._solve_interpolation_cvs(points, params, nurbsknots, cv):
            return NurbsCurve()

        curve = NurbsCurve(dim, False, degree + 1, cv_count)

        for i in range(kc):
            curve.set_nurbsknot(i, nurbsknots[i])

        for i in range(cv_count):
            curve.set_cv(i, Point(cv[i * 3], cv[i * 3 + 1], cv[i * 3 + 2]))

        return curve

    @staticmethod
    def _interpolation_end_cvs(
        points: list[Point], params: list[float], end_condition: CurveInterpStyle
    ) -> list[float]:
        """Return the n + 2 CVs: the points with an end tangent CV after the first and before the last."""

        n = len(points)
        dim = 3

        if end_condition == CurveInterpStyle.Occt:
            deg_t = 2 if n == 3 else 3
            tan_start = NurbsCurve._lagrange_tangent(
                points, params, 0, deg_t + 1, params[0]
            )
            tan_end = NurbsCurve._lagrange_tangent(
                points, params, n - 1 - deg_t, deg_t + 1, params[n - 1]
            )
            s0 = (params[1] - params[0]) / 3.0
            s1 = -(params[n - 1] - params[n - 2]) / 3.0
        else:
            tan_start = NurbsCurve._bessel_tangent(points, 0, 1, 2)
            end_raw = NurbsCurve._bessel_tangent(points, n - 1, n - 2, n - 3)
            tan_end = -end_raw
            s0 = points[0].distance(points[1]) / 3.0
            s1 = -points[n - 1].distance(points[n - 2]) / 3.0

        cv = [0.0] * ((n + 2) * dim)

        for d in range(dim):
            cv[d] = points[0][d]

        for d in range(dim):
            cv[dim + d] = points[0][d] + s0 * tan_start[d]

        for i in range(1, n - 1):
            for d in range(dim):
                cv[(i + 1) * dim + d] = points[i][d]

        for d in range(dim):
            cv[n * dim + d] = points[n - 1][d] + s1 * tan_end[d]

        for d in range(dim):
            cv[(n + 1) * dim + d] = points[n - 1][d]

        return cv

    @staticmethod
    def _solve_interpolation_cvs(
        points: list[Point], params: list[float], nurbsknots: list[float], cv: list[float]
    ) -> bool:
        """Solve the tridiagonal interpolation system and write the interior CVs into cv."""

        n = len(points)
        dim = 3
        order = 4
        sys_n = n
        lower = [0.0] * sys_n
        diag = [0.0] * sys_n
        upper = [0.0] * sys_n
        rhs = [0.0] * (sys_n * dim)
        diag[0] = 1.0

        for d in range(dim):
            rhs[d] = cv[dim + d]

        for i in range(1, n - 1):
            basis = nurbsknot.eval_basis(order, nurbsknots, i, params[i])
            lower[i] = basis[0]
            diag[i] = basis[1]
            upper[i] = basis[2]

            for d in range(dim):
                rhs[i * dim + d] = points[i][d]

        diag[n - 1] = 1.0

        for d in range(dim):
            rhs[(n - 1) * dim + d] = cv[n * dim + d]

        solution = nurbsknot.solve_tridiagonal(dim, sys_n, lower, diag, upper, rhs)

        if solution is None:
            return False

        for i in range(sys_n):
            for d in range(dim):
                cv[(i + 1) * dim + d] = solution[i * dim + d]

        return True

    @staticmethod
    def _flatten_points(points: list[Point], count: int) -> list[float]:
        """Return the x, y, z of the first count points as one flat array."""

        flat = [0.0] * (count * 3)

        for i in range(count):
            flat[i * 3] = points[i][0]
            flat[i * 3 + 1] = points[i][1]
            flat[i * 3 + 2] = points[i][2]

        return flat

    @staticmethod
    def _create_fitted_periodic(points: list[Point], num_cvs: int, degree: int) -> NurbsCurve:
        """Construct the closed least-squares fit with num_cvs distinct CVs."""

        dim = 3
        order = degree + 1
        n = len(points)

        if n >= 2 and points[0].distance(points[n - 1]) < 1e-10:
            n -= 1

        if n <= num_cvs or num_cvs < order:
            return (
                NurbsCurve()
                if n < 3
                else NurbsCurve.create_interpolated(
                    points[:n], CurveNurbsKnotStyle.ChordPeriodic
                )
            )

        cv_count = num_cvs + degree
        kc = cv_count + order - 2
        params = [0.0] * (n + 1)

        for i in range(1, n):
            params[i] = params[i - 1] + points[i - 1].distance(points[i])

        params[n] = params[n - 1] + points[n - 1].distance(points[0])

        if params[n] < 1e-14:
            return NurbsCurve()

        ppts = NurbsCurve._flatten_points(points, n)
        nurbsknots = nurbsknot.build_fitted_nurbsknots_periodic_adaptive(
            params, ppts, n, dim, num_cvs, degree
        )
        NtN = [[0.0] * num_cvs for _ in range(num_cvs)]
        cv = [0.0] * (num_cvs * dim)

        for k in range(n):
            span = nurbsknot.find_span(order, cv_count, nurbsknots, params[k])
            basis = nurbsknot.eval_basis(order, nurbsknots, span, params[k])

            for a in range(order):
                ci = (span + a) % num_cvs

                for d in range(dim):
                    cv[ci * dim + d] += basis[a] * points[k][d]

                for b in range(order):
                    NtN[ci][(span + b) % num_cvs] += basis[a] * basis[b]

        if not NurbsCurve._solve_dense(NtN, cv, num_cvs, dim):
            return NurbsCurve()

        curve = NurbsCurve(dim, False, order, cv_count)

        for i in range(kc):
            curve.set_nurbsknot(i, nurbsknots[i])

        for i in range(num_cvs):
            curve.set_cv(i, Point(cv[i * 3], cv[i * 3 + 1], cv[i * 3 + 2]))

        for i in range(degree):
            curve.set_cv(num_cvs + i, curve.get_cv(i))

        return curve

    @staticmethod
    def _create_fitted_clamped(points: list[Point], num_cvs: int, degree: int) -> NurbsCurve:
        """Construct the open least-squares fit through the first and last point."""

        m = len(points)
        dim = 3

        if m <= num_cvs or num_cvs < degree + 1:
            return NurbsCurve.create_interpolated(points)

        pts = NurbsCurve._flatten_points(points, m)
        params = nurbsknot.compute_parameters(pts, m, dim, CurveNurbsKnotStyle.Chord)
        nurbsknots = nurbsknot.build_fitted_nurbsknots_adaptive(
            params, pts, m, dim, num_cvs, degree
        )
        sys_n = num_cvs - 2
        band = [0.0] * (sys_n * (degree + 1))
        rhs = [0.0] * (sys_n * dim)
        NurbsCurve._fitted_band_system(points, params, nurbsknots, num_cvs, degree, band, rhs)

        if not nurbsknot.solve_banded_spd(dim, sys_n, degree, band, rhs):
            return NurbsCurve.create_interpolated(points)

        kc = len(nurbsknots)
        curve = NurbsCurve(dim, False, degree + 1, num_cvs)

        for i in range(kc):
            curve.set_nurbsknot(i, nurbsknots[i])

        curve.set_cv(0, points[0])

        for i in range(sys_n):
            curve.set_cv(i + 1, Point(rhs[i * 3], rhs[i * 3 + 1], rhs[i * 3 + 2]))

        curve.set_cv(num_cvs - 1, points[m - 1])

        return curve

    @staticmethod
    def _fitted_band_system(
        points: list[Point],
        params: list[float],
        nurbsknots: list[float],
        num_cvs: int,
        degree: int,
        band: list[float],
        rhs: list[float],
    ) -> None:
        """Accumulate the banded normal equations of the open fit, end CVs fixed."""

        m = len(points)
        dim = 3
        order = degree + 1
        n = num_cvs - 1
        bw1 = degree + 1

        for k in range(1, m - 1):
            span = nurbsknot.find_span(order, num_cvs, nurbsknots, params[k])
            basis = nurbsknot.eval_basis(order, nurbsknots, span, params[k])
            rk = [points[k][0], points[k][1], points[k][2]]

            for a in range(order):
                ci = span + a

                if ci == 0:
                    for d in range(dim):
                        rk[d] -= basis[a] * points[0][d]

                if ci == n:
                    for d in range(dim):
                        rk[d] -= basis[a] * points[m - 1][d]

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

    @staticmethod
    def _promote_to_3d(segs: list[NurbsCurve]) -> None:
        """Lift 2D segments to 3D when 2D and 3D segments are mixed."""

        any2 = False
        any3 = False

        for c in segs:
            if c.m_dim == 2:
                any2 = True
            elif c.m_dim == 3:
                any3 = True

        if not any2 or not any3:
            return

        for c in segs:
            if c.m_dim != 2:
                continue

            os_ = c.m_cv_stride
            ns = os_ + 1
            cv = np.zeros(c.m_cv_count * ns, dtype=np.float64)

            for i in range(c.m_cv_count):
                cv[i * ns] = c.m_cv[i * os_]
                cv[i * ns + 1] = c.m_cv[i * os_ + 1]

                if c.m_is_rat:
                    cv[i * ns + 3] = c.m_cv[i * os_ + 2]

            c.m_cv = cv
            c.m_cv_stride = ns
            c.m_dim = 3

    @staticmethod
    def _chain_segments(segs: list[NurbsCurve], tolerance: float) -> list[list[NurbsCurve]]:
        """Group segments into chains by endpoint matching, reversing where needed."""

        chains = []
        used = [False] * len(segs)

        for i in range(len(segs)):
            if used[i]:
                continue

            used[i] = True

            chain = [segs[i]]
            grown = not segs[i].is_closed()

            while grown:
                grown = False
                start = chain[0].point_at_start()
                end = chain[-1].point_at_end()

                for j in range(len(segs)):
                    if used[j] or segs[j].is_closed():
                        continue

                    s = segs[j].point_at_start()
                    e = segs[j].point_at_end()

                    if s.distance(end) <= tolerance:
                        chain.append(segs[j])
                    elif e.distance(end) <= tolerance:
                        r = segs[j].duplicate()
                        r.reverse()
                        chain.append(r)
                    elif e.distance(start) <= tolerance:
                        chain.insert(0, segs[j])
                    elif s.distance(start) <= tolerance:
                        r = segs[j].duplicate()
                        r.reverse()
                        chain.insert(0, r)
                    else:
                        continue

                    used[j] = True
                    grown = True
                    break

            chains.append(chain)

        return chains

    @staticmethod
    def _join_chain(chain: list[NurbsCurve], result: list[NurbsCurve]) -> None:
        """Append the chain merged into one curve to result, or its segments when they cannot be merged."""

        if len(chain) == 1:
            result.append(chain[0])

            return

        rational = False
        max_degree = 1

        for c in chain:
            if c.is_rational():
                rational = True

            if c.degree() > max_degree:
                max_degree = c.degree()

        aligned = True

        for c in chain:
            if rational:
                c.to_rational()

            if not c.clamp_end(2) or not c.increase_degree(max_degree):
                aligned = False

        joined = chain[0].duplicate()

        if aligned:
            for ci in range(1, len(chain)):
                NurbsCurve._append_segment(joined, chain[ci], rational)

        if (
            not aligned
            or len(joined.m_cv)
            < (joined.m_cv_count - 1) * joined.m_cv_stride + joined.cv_size()
            or len(joined.m_nurbsknot) != joined.m_cv_count + joined.m_order - 2
        ):
            for c in chain:
                result.append(c)

            return

        result.append(joined)

    @staticmethod
    def _append_segment(joined: NurbsCurve, segment: NurbsCurve, rational: bool) -> None:
        """Append segment to joined with a C0 junction at the averaged shared CV."""

        stride = joined.m_cv_stride
        cvdim = joined.cv_size()
        a1 = joined.domain_end()
        s0, s1 = segment.domain()
        segment.set_domain(a1, a1 + (s1 - s0))

        if rational:
            w_end = joined.weight(joined.m_cv_count - 1)
            w_start = segment.weight(0)

            if abs(w_start) > Tolerance.ZERO_TOLERANCE:
                scale = w_end / w_start

                for k in range(len(segment.m_cv)):
                    segment.m_cv[k] = segment.m_cv[k] * scale

        last = (joined.m_cv_count - 1) * stride

        if (
            stride <= 0
            or cvdim <= 0
            or segment.m_order != joined.m_order
            or segment.m_cv_stride != stride
            or segment.cv_size() != cvdim
            or len(joined.m_cv) < last + cvdim
            or len(segment.m_cv) < segment.m_cv_count * stride
            or len(segment.m_cv) <= stride
            or len(segment.m_nurbsknot) != segment.m_cv_count + segment.m_order - 2
        ):
            return

        for k in range(cvdim):
            joined.m_cv[last + k] = 0.5 * (joined.m_cv[last + k] + segment.m_cv[k])

        joined.m_nurbsknot = np.concatenate(
            [joined.m_nurbsknot, segment.m_nurbsknot[joined.m_order - 1 :]]
        )
        joined.m_cv = np.concatenate([joined.m_cv, segment.m_cv[stride:]])
        joined.m_cv_count = joined.m_cv_count + segment.m_cv_count - 1

    @staticmethod
    def _circle_center(p0: Point, p1: Point, p2: Point) -> Point | None:
        """Return the center of the circle through three points, None when they are collinear."""

        d1 = p1 - p0
        d2 = p2 - p1
        normal = d1.cross(d2)

        if normal.magnitude() < Tolerance.ZERO_TOLERANCE:
            return None

        normal = normal.normalized()

        m1 = Point.sum(p0, p1) * 0.5
        m2 = Point.sum(p1, p2) * 0.5
        perp1 = d1.cross(normal).normalized()
        perp2 = d2.cross(normal).normalized()
        denom = perp1[0] * perp2[1] - perp1[1] * perp2[0]

        if abs(denom) < Tolerance.ZERO_TOLERANCE:
            denom = perp1[0] * perp2[2] - perp1[2] * perp2[0]

        if abs(denom) < Tolerance.ZERO_TOLERANCE:
            return None

        dx = m2[0] - m1[0]
        dy = m2[1] - m1[1]
        s = (dx * perp2[1] - dy * perp2[0]) / denom

        return m1 + perp1 * s

    def _full_nurbsknots(self) -> list[float]:
        """Return the nurbsknots padded with one superfluous value at each end."""

        full_nurbsknot_count = self.m_cv_count + self.m_order
        U = [0.0] * full_nurbsknot_count
        U[0] = float(self.m_nurbsknot[0])

        for i in range(len(self.m_nurbsknot)):
            U[i + 1] = float(self.m_nurbsknot[i])

        U[full_nurbsknot_count - 1] = float(self.m_nurbsknot[-1])

        return U

    def _insert_nurbsknot_once(self, nurbsknot_value: float, U: list[float]) -> None:
        """Insert one nurbsknot by Boehm, U the padded nurbsknots."""

        p = self.degree()
        n = self.m_cv_count - 1
        full_nurbsknot_count = self.m_cv_count + self.m_order
        k = self._find_span(nurbsknot_value) + self.m_order - 1
        new_cv_count = self.m_cv_count + 1
        U_new = [0.0] * (full_nurbsknot_count + 1)
        cv_new = np.zeros(new_cv_count * self.m_cv_stride, dtype=np.float64)

        for i in range(k + 1):
            U_new[i] = U[i]

        U_new[k + 1] = nurbsknot_value

        for i in range(k + 1, full_nurbsknot_count):
            U_new[i + 1] = U[i]

        for i in range(k - p + 1):
            cv_new[i * self.m_cv_stride : (i + 1) * self.m_cv_stride] = self.m_cv[
                i * self.m_cv_stride : (i + 1) * self.m_cv_stride
            ]

        for i in range(k + 1, n + 2):
            cv_new[i * self.m_cv_stride : (i + 1) * self.m_cv_stride] = self.m_cv[
                (i - 1) * self.m_cv_stride : i * self.m_cv_stride
            ]

        for i in range(k - p + 1, k + 1):
            alpha = 0.0
            denom = U[i + p] - U[i]

            if denom != 0.0:
                alpha = (nurbsknot_value - U[i]) / denom

            for d in range(self.m_cv_stride):
                cv_new[i * self.m_cv_stride + d] = (1.0 - alpha) * self.m_cv[
                    (i - 1) * self.m_cv_stride + d
                ] + alpha * self.m_cv[i * self.m_cv_stride + d]

        self.m_cv_count = new_cv_count
        self.m_cv = cv_new

        kc = self.m_order + self.m_cv_count - 2
        nurbsknot_new = np.zeros(kc, dtype=np.float64)

        for i in range(kc):
            nurbsknot_new[i] = U_new[i + 1]

        self.m_nurbsknot = nurbsknot_new

    def _adaptive_samples(
        self, angle_tolerance: float, min_edge_length: float, max_edge_length: float
    ) -> list[tuple[float, Point]]:
        """Return the (t, point) samples of the chord-deviation bisection, sorted by t."""

        t0, t1 = self.domain()
        samples = [(t0, self.point_at(t0)), (t1, self.point_at(t1))]
        work_queue = [(t0, t1)]
        closed = samples[0][1].distance(samples[1][1]) < 1e-6

        if closed and self.length() > max_edge_length:
            t_third = t0 + (t1 - t0) / 3.0
            t_two_thirds = t0 + 2.0 * (t1 - t0) / 3.0
            samples.append((t_third, self.point_at(t_third)))
            samples.append((t_two_thirds, self.point_at(t_two_thirds)))
            work_queue = [(t0, t_third), (t_third, t_two_thirds), (t_two_thirds, t1)]

        max_iterations = 10000
        iterations = 0

        while len(work_queue) > 0 and iterations < max_iterations:
            iterations += 1

            ta, tb = work_queue.pop()
            pa = self.point_at(ta)
            pb = self.point_at(tb)
            chord_length = pa.distance(pb)

            if chord_length < min_edge_length:
                continue

            tm = (ta + tb) * 0.5
            pm = self.point_at(tm)
            chord = pb - pa
            to_mid = pm - pa
            chord_len_sq = chord.dot(chord)
            deviation = 0.0

            if chord_len_sq > 1e-20:
                proj = to_mid.dot(chord) / chord_len_sq
                deviation = pm.distance(pa + chord * proj)

            deviation_tolerance = chord_length * angle_tolerance * 0.5

            if deviation > deviation_tolerance or chord_length > max_edge_length:
                samples.append((tm, pm))
                work_queue.append((ta, tm))
                work_queue.append((tm, tb))

        samples.sort(key=functools.cmp_to_key(_sample_before))

        return samples

    def _homogeneous_derivatives(self, span: int, ders: list[list[float]]) -> list[list[float]]:
        """Return the homogeneous derivatives (x, y, z, w) at span from the basis derivatives."""

        p = self.degree()
        count = len(ders)
        Aders = [[0.0, 0.0, 0.0, 0.0] for _ in range(count)]

        for k in range(count):
            for j in range(p + 1):
                cv_ptr = self.cv(span + j)

                if cv_ptr is None:
                    continue

                Nx = ders[k][j]
                Aders[k][0] += Nx * cv_ptr[0]
                Aders[k][1] += Nx * (cv_ptr[1] if self.m_dim > 1 else 0.0)
                Aders[k][2] += Nx * (cv_ptr[2] if self.m_dim > 2 else 0.0)
                Aders[k][3] += Nx * (cv_ptr[self.m_dim] if self.m_is_rat else 1.0)

        return Aders

    def _start_frame(self) -> tuple[Vector, Vector] | None:
        """Return the unit tangent and normal at the domain start, None when the derivative vanishes."""

        derivs0 = self.evaluate(self.domain_start(), 2)
        D1_0 = derivs0[1]
        D2_0 = derivs0[2]
        D1_0_mag = D1_0.magnitude()

        if D1_0_mag < 1e-14:
            return None

        T0 = D1_0 / D1_0_mag
        D2_dot_D1 = D2_0.dot(D1_0)
        D1_0_mag_sq = D1_0_mag * D1_0_mag
        N0_unnorm = D2_0 - D1_0 * (D2_dot_D1 / D1_0_mag_sq)
        N0_mag = N0_unnorm.magnitude()

        if N0_mag < 1e-14:
            N0_unnorm = Vector(0.0, 0.0, 1.0).cross(T0)
            N0_mag = N0_unnorm.magnitude()

            if N0_mag < 1e-14:
                N0_unnorm = Vector(0.0, 1.0, 0.0).cross(T0)
                N0_mag = N0_unnorm.magnitude()

        return T0, N0_unnorm / N0_mag

    def _double_reflection(self, param: float, r0: Vector, T0: Vector) -> Vector:
        """Return r0 carried from the domain start to param by double reflection."""

        t0, t1 = self.domain()
        num_steps = max(10, int((param - t0) / (t1 - t0) * 100))
        dt = (param - t0) / num_steps
        ri = r0
        ti = t0
        xi = self.point_at(ti)
        Ti = T0

        for i in range(num_steps):
            if ti >= param - 1e-14:
                break

            ti_next = min(ti + dt, param)
            xi_next = self.point_at(ti_next)
            Ti_next = self.tangent_at(ti_next)
            Ti_next.normalize_self()

            v1 = xi_next - xi
            c1 = v1.dot(v1)

            if c1 < 1e-28:
                ti = ti_next
                xi = xi_next
                Ti = Ti_next
                continue

            ri_dot_v1 = ri.dot(v1)
            rL = ri - v1 * (2.0 * ri_dot_v1 / c1)
            Ti_dot_v1 = Ti.dot(v1)
            TL = Ti - v1 * (2.0 * Ti_dot_v1 / c1)
            v2 = Ti_next - TL
            c2 = v2.dot(v2)

            if c2 < 1e-28:
                ri = rL
            else:
                rL_dot_v2 = rL.dot(v2)
                ri = rL - v2 * (2.0 * rL_dot_v2 / c2)

            if ri.magnitude() > 1e-14:
                ri.normalize_self()

            ti = ti_next
            xi = xi_next
            Ti = Ti_next

        return ri

    def _keep_span_range(self, t0: float, t1: float) -> bool:
        """Keep the CVs and nurbsknots between the full-multiplicity nurbsknots t0 and t1."""

        p = self.degree()
        U = self._full_nurbsknots()
        full_nurbsknot_count = len(U)
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

        first_cv = max(0, start_span - p)
        last_cv = min(end_span - 1, self.m_cv_count - 1)
        new_cv_count = last_cv - first_cv + 1

        if new_cv_count < self.m_order:
            new_cv_count = self.m_order

            if first_cv + new_cv_count > self.m_cv_count:
                return False

        new_nurbsknot = self._trimmed_nurbsknots(U, start_span, new_cv_count, t0, t1)
        new_cv = np.zeros(new_cv_count * self.m_cv_stride, dtype=np.float64)

        for i in range(new_cv_count):
            new_cv[i * self.m_cv_stride : (i + 1) * self.m_cv_stride] = self.m_cv[
                (first_cv + i) * self.m_cv_stride : (first_cv + i + 1)
                * self.m_cv_stride
            ]

        self.m_cv_count = new_cv_count
        self.m_cv = new_cv
        self.m_nurbsknot = new_nurbsknot

        return True

    def _trimmed_nurbsknots(
        self, U: list[float], start_span: int, new_cv_count: int, t0: float, t1: float
    ) -> np.ndarray:
        """Return the nurbsknots of the kept range, clamped at t0 and t1."""

        p = self.degree()
        full_nurbsknot_count = len(U)
        new_nurbsknot_count = new_cv_count + self.m_order - 2
        new_nurbsknot = np.zeros(new_nurbsknot_count, dtype=np.float64)

        for i in range(p - 1):
            new_nurbsknot[i] = t0

        mid_count = new_nurbsknot_count - 2 * (p - 1)

        for i in range(mid_count):
            src_idx = start_span + i
            new_nurbsknot[p - 1 + i] = (
                U[src_idx] if src_idx < full_nurbsknot_count else t1
            )

        for i in range(p - 1):
            new_nurbsknot[new_nurbsknot_count - p + 1 + i] = t1

        return new_nurbsknot

    def _first_nurbsknot_above(self, value: float) -> int:
        """Return the index of the first nurbsknot greater than value, -1 when none."""

        kc = self.nurbsknot_count()

        for i in range(kc):
            if self.m_nurbsknot[i] > value:
                return i

        return -1

    def _seam_nurbsknot_index(self, t: float, nurbsknot_index: int) -> int:
        """Return the seam nurbsknot index near t, snapping to an existing nurbsknot or inserting one; -1 on failure."""

        d0 = t - self.m_nurbsknot[nurbsknot_index - 1]
        d1 = self.m_nurbsknot[nurbsknot_index] - t

        if d0 <= d1 and d0 < Tolerance.ZERO_TOLERANCE:
            return nurbsknot_index - 1

        if d0 > d1 and d1 < Tolerance.ZERO_TOLERANCE:
            return nurbsknot_index

        if not self.insert_nurbsknot(t, 1):
            return -1

        return self._first_nurbsknot_above(t + Tolerance.ZERO_TOLERANCE)

    def _rotate_periodic_seam(self, nurbsknot_index: int, t: float, dom_len: float) -> bool:
        """Rotate the nurbsknots and CVs of a periodic curve so the domain starts at t."""

        p = self.degree()
        sc = self.span_count()
        cvc = self.m_cv_count
        distinct_cvc = cvc - p
        cvdim = self.cv_size()
        old_nurbsknots = self.m_nurbsknot.copy()
        old_cv = self.m_cv.copy()
        curr = p - 1

        for i in range(nurbsknot_index, sc + p - 1):
            self.m_nurbsknot[curr] = old_nurbsknots[i]
            curr += 1

        for i in range(nurbsknot_index - p + 2):
            self.m_nurbsknot[curr] = old_nurbsknots[p - 1 + i] + dom_len
            curr += 1

        for i in range(p - 1):
            self.m_nurbsknot[curr + i] = (
                self.m_nurbsknot[curr + i - 1]
                + self.m_nurbsknot[p + i]
                - self.m_nurbsknot[p + i - 1]
            )
            self.m_nurbsknot[p - 2 - i] = (
                self.m_nurbsknot[p - i - 1]
                - self.m_nurbsknot[curr - 1 - i]
                + self.m_nurbsknot[curr - 2 - i]
            )

        cv_id = nurbsknot_index - p + 1

        for i in range(cvc):
            src = cv_id % distinct_cvc

            if src < 0:
                src += distinct_cvc

            for j in range(cvdim):
                self.m_cv[i * self.m_cv_stride + j] = old_cv[src * self.m_cv_stride + j]

            cv_id += 1

        return self.set_domain(t, t + dom_len)

    def _split_seam(self, t: float, dom_len: float) -> bool:
        """Split at t and join the right part before the left so the domain starts at t."""

        left_crv, right_crv = self.split(t)

        if not left_crv.is_valid() or not right_crv.is_valid():
            return False

        order = self.m_order
        cvdim = self.cv_size()
        new_cv_count = right_crv.m_cv_count + left_crv.m_cv_count - 1
        new_kc = order + new_cv_count - 2
        new_cv = np.zeros(new_cv_count * self.m_cv_stride, dtype=np.float64)
        new_nurbsknots = np.zeros(new_kc, dtype=np.float64)

        for i in range(right_crv.m_cv_count):
            for j in range(cvdim):
                new_cv[i * self.m_cv_stride + j] = right_crv.m_cv[
                    i * right_crv.m_cv_stride + j
                ]

        for i in range(1, left_crv.m_cv_count):
            dst = right_crv.m_cv_count + i - 1

            for j in range(cvdim):
                new_cv[dst * self.m_cv_stride + j] = left_crv.m_cv[
                    i * left_crv.m_cv_stride + j
                ]

        rkc = right_crv.nurbsknot_count()

        for i in range(rkc):
            new_nurbsknots[i] = right_crv.m_nurbsknot[i]

        lkc = left_crv.nurbsknot_count()

        for i in range(order - 1, lkc):
            new_nurbsknots[rkc + i - (order - 1)] = left_crv.m_nurbsknot[i] + dom_len

        self.m_cv_count = new_cv_count
        self.m_cv = new_cv
        self.m_nurbsknot = new_nurbsknots

        return self.set_domain(t, t + dom_len)

    @staticmethod
    def _assign_plane(dst: Plane, src: Plane) -> None:
        """Copy the frame of src into dst (the C++ Plane* out-parameter)."""

        dst._origin = src.origin
        dst._x_axis = src.x_axis
        dst._y_axis = src.y_axis
        dst._z_axis = src.z_axis
        dst._update_equation()


# ═══════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════
def _sample_before(a: tuple[float, Point], b: tuple[float, Point]) -> int:
    """Order two (t, point) samples by parameter."""
    return (a[0] > b[0]) - (a[0] < b[0])


# ═══════════════════════════════════════════════════════════════════════════
# Degree elevation
# ═══════════════════════════════════════════════════════════════════════════
def _evaluate_nurbs_blossom(
    cvdim: int,
    order: int,
    cv_stride: int,
    CV: np.ndarray,
    cv0: int,
    nurbsknot_: np.ndarray,
    kn0: int,
    t: list[float],
    P: list[float],
) -> bool:
    """Compute the blossom of one span at order - 1 parameters by the de Boor recurrence."""

    if cv_stride < cvdim:
        return False

    degree = order - 1

    for i in range(1, 2 * degree):
        if nurbsknot_[kn0 + i] - nurbsknot_[kn0 + i - 1] < 0.0:
            return False

    if (
        nurbsknot_[kn0 + degree] - nurbsknot_[kn0 + degree - 1]
        < Tolerance.ZERO_TOLERANCE
    ):
        return False

    space = [0.0] * order

    for i in range(cvdim):
        for j in range(order):
            space[j] = CV[cv0 + j * cv_stride + i]

        for j in range(1, order):
            for k in range(j, order):
                denom = nurbsknot_[kn0 + degree + k - j] - nurbsknot_[kn0 + k - 1]
                space[k - j] = (
                    nurbsknot_[kn0 + degree + k - j] - t[j - 1]
                ) / denom * space[k - j] + (
                    t[j - 1] - nurbsknot_[kn0 + k - 1]
                ) / denom * space[k - j + 1]

        P[i] = space[0]

    return True


def _get_raised_degree_cv(
    old_order: int,
    cvdim: int,
    old_cv_stride: int,
    oldCV: np.ndarray,
    cv0: int,
    oldkn: np.ndarray,
    okn0: int,
    newkn: np.ndarray,
    nkn0: int,
    cv_id: int,
    newCV: np.ndarray,
    ncv0: int,
) -> bool:
    """Compute one CV of the degree-raised span as the average of blossoms."""

    if cv_id < 0 or cv_id > old_order:
        return False

    old_degree = old_order - 1
    new_degree = old_degree + 1
    t = [0.0] * old_degree
    P = [0.0] * cvdim

    for i in range(cvdim):
        newCV[ncv0 + i] = 0.0

    for i in range(new_degree):
        k = 0

        for j in range(new_degree):
            if j != i:
                t[k] = newkn[nkn0 + cv_id + j]
                k += 1

        if not _evaluate_nurbs_blossom(
            cvdim, old_order, old_cv_stride, oldCV, cv0, oldkn, okn0, t, P
        ):
            return False

        for k in range(cvdim):
            newCV[ncv0 + k] += P[k]

    for i in range(cvdim):
        newCV[ncv0 + i] /= float(new_degree)

    return True


def _next_span_index(
    order: int, cv_count: int, nurbsknot_: np.ndarray, span_index: int
) -> int:
    """Return the next span index past degenerate spans."""

    if span_index < 0 or span_index > cv_count - order:
        return -1

    if span_index < cv_count - order:
        span_index += 1

        while (
            span_index < cv_count - order
            and nurbsknot_[span_index + order - 2] == nurbsknot_[span_index + order - 1]
        ):
            span_index += 1

    return span_index


def _increment_nurbs_degree(N: NurbsCurve) -> bool:
    """Raise the degree of N by one."""

    M = N.duplicate()
    sc = M.span_count()
    new_kcount = M.nurbsknot_count() + sc + 1
    new_order = M.order() + 1
    new_cv_count = new_kcount - new_order + 2
    cvdim = M.cv_size()
    N.m_order = new_order
    N.m_cv_count = new_cv_count
    N.m_nurbsknot = np.zeros(new_order + new_cv_count - 2, dtype=np.float64)
    N.m_cv = np.zeros(new_cv_count * N.m_cv_stride, dtype=np.float64)

    ki = 0
    ko = 0
    mkc = M.nurbsknot_count()

    while ki < mkc:
        kn = M.m_nurbsknot[ki]
        mult = 1

        while (
            ki + mult < mkc
            and abs(M.m_nurbsknot[ki + mult] - kn) < Tolerance.ZERO_TOLERANCE
        ):
            mult += 1

        for j in range(mult + 1):
            N.m_nurbsknot[ko] = kn
            ko += 1

        ki += mult

    siN = 0
    siM = 0

    for i in range(sc):
        span_mult = N.nurbsknot_multiplicity(siN + N.degree() - 1)
        skip = N.order() - span_mult

        for j in range(skip, N.order()):
            _get_raised_degree_cv(
                M.order(),
                cvdim,
                M.m_cv_stride,
                M.m_cv,
                siM * M.m_cv_stride,
                M.m_nurbsknot,
                siM,
                N.m_nurbsknot,
                siN,
                j,
                N.m_cv,
                (siN + j) * N.m_cv_stride,
            )

        siN = _next_span_index(N.order(), N.cv_count(), N.m_nurbsknot, siN)
        siM = _next_span_index(M.order(), M.cv_count(), M.m_nurbsknot, siM)

    for i in range(cvdim):
        N.m_cv[i] = M.m_cv[i]
        N.m_cv[(N.cv_count() - 1) * N.m_cv_stride + i] = M.m_cv[
            (M.cv_count() - 1) * M.m_cv_stride + i
        ]

    return True
