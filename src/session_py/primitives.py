import numpy as np
import math

from .nurbscurve import NurbsCurve
from .point import Point
from .tolerance import Tolerance
from . import knot


class Primitives:
    """Static factory methods for creating NURBS curve primitives."""

    @staticmethod
    def circle(cx: float, cy: float, cz: float, radius: float) -> NurbsCurve:
        """Create a circle as a rational NURBS curve (9 control points)."""
        w = math.sqrt(2.0) / 2.0

        cx_pat = [1, 1, 0, -1, -1, -1, 0, 1, 1]
        cy_pat = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        weights = [1, w, 1, w, 1, w, 1, w, 1]

        curve = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=9)
        curve.m_knot = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float64)
        curve.m_cv = np.zeros(9 * 4, dtype=np.float64)

        for i in range(9):
            px = cx + radius * cx_pat[i]
            py = cy + radius * cy_pat[i]
            curve.set_cv_4d(i, px * weights[i], py * weights[i], cz * weights[i], weights[i])

        return curve

    @staticmethod
    def ellipse(cx: float, cy: float, cz: float, major_radius: float, minor_radius: float) -> NurbsCurve:
        """Create an ellipse as a rational NURBS curve."""
        w = math.sqrt(2.0) / 2.0
        ex = [1, 1, 0, -1, -1, -1, 0, 1, 1]
        ey = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        weights = [1, w, 1, w, 1, w, 1, w, 1]

        curve = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=9)
        curve.m_knot = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float64)
        curve.m_cv = np.zeros(9 * 4, dtype=np.float64)

        for i in range(9):
            px = cx + major_radius * ex[i]
            py = cy + minor_radius * ey[i]
            curve.set_cv_4d(i, px * weights[i], py * weights[i], cz * weights[i], weights[i])

        return curve

    @staticmethod
    def arc(start: Point, mid: Point, end: Point) -> NurbsCurve:
        """Create an arc through three points as a rational NURBS curve."""
        d1 = [mid[0] - start[0], mid[1] - start[1], mid[2] - start[2]]
        d2 = [end[0] - mid[0], end[1] - mid[1], end[2] - mid[2]]

        m1 = [(start[0] + mid[0]) / 2, (start[1] + mid[1]) / 2, (start[2] + mid[2]) / 2]
        m2 = [(mid[0] + end[0]) / 2, (mid[1] + end[1]) / 2, (mid[2] + end[2]) / 2]

        normal = [d1[1]*d2[2] - d1[2]*d2[1],
                  d1[2]*d2[0] - d1[0]*d2[2],
                  d1[0]*d2[1] - d1[1]*d2[0]]
        normal_len = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

        if normal_len < Tolerance.ZERO_TOLERANCE:
            return NurbsCurve.create(periodic=False, degree=1, points=[start, end])

        # Calculate weight from arc geometry
        chord_mid = Point((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2)
        sagitta = chord_mid.distance(mid)
        chord_len = start.distance(end)

        if sagitta < Tolerance.ZERO_TOLERANCE:
            return NurbsCurve.create(periodic=False, degree=1, points=[start, end])

        # w = cos(theta/2) where theta is the arc angle
        # For a circular arc: sagitta = r(1 - cos(theta/2))
        # Using the relation: w relates to how much the shoulder point is pushed out
        half_chord = chord_len / 2
        r_approx = (half_chord**2 + sagitta**2) / (2 * sagitta) if sagitta > 0 else float('inf')

        if r_approx > 0:
            cos_half = (r_approx - sagitta) / r_approx
            cos_half = max(-1.0, min(1.0, cos_half))
            w = abs(cos_half) if cos_half > 0 else 0.5
        else:
            w = 0.5

        w = max(0.1, min(1.0, w))

        curve = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=3)
        curve.m_knot = np.array([0, 0, 1, 1], dtype=np.float64)
        curve.m_cv = np.zeros(3 * 4, dtype=np.float64)

        shoulder = Point(
            (start[0] + end[0]) / 2 + (mid[0] - (start[0] + end[0]) / 2) / w,
            (start[1] + end[1]) / 2 + (mid[1] - (start[1] + end[1]) / 2) / w,
            (start[2] + end[2]) / 2 + (mid[2] - (start[2] + end[2]) / 2) / w
        )

        curve.set_cv_4d(0, start[0], start[1], start[2], 1.0)
        curve.set_cv_4d(1, shoulder[0] * w, shoulder[1] * w, shoulder[2] * w, w)
        curve.set_cv_4d(2, end[0], end[1], end[2], 1.0)

        return curve

    @staticmethod
    def parabola(p0: Point, p1: Point, p2: Point) -> NurbsCurve:
        """Create a parabola through 3 points as a non-rational quadratic NURBS."""
        curve = NurbsCurve(dimension=3, is_rational=False, order=3, cv_count=3)
        curve.m_knot = np.array([0, 0, 1, 1], dtype=np.float64)
        curve.m_cv = np.zeros(3 * 3, dtype=np.float64)

        cv1 = Point(
            2 * p1[0] - (p0[0] + p2[0]) / 2,
            2 * p1[1] - (p0[1] + p2[1]) / 2,
            2 * p1[2] - (p0[2] + p2[2]) / 2
        )

        curve.set_cv(0, p0)
        curve.set_cv(1, cv1)
        curve.set_cv(2, p2)

        return curve

    @staticmethod
    def hyperbola(center: Point, a: float, b: float, extent: float) -> NurbsCurve:
        """Create a hyperbola segment as a NURBS curve."""
        num_segments = 8
        cv_count = num_segments + 1

        curve = NurbsCurve(dimension=3, is_rational=False, order=4, cv_count=cv_count)
        curve.m_cv = np.zeros(cv_count * 3, dtype=np.float64)

        for i in range(cv_count):
            t = -extent + 2 * extent * i / num_segments
            x = center[0] + a * math.cosh(t)
            y = center[1] + b * math.sinh(t)
            z = center[2]
            curve.set_cv(i, Point(x, y, z))

        curve.m_knot = knot.make_clamped_uniform(curve.m_order, curve.m_cv_count, 1.0)
        return curve

    @staticmethod
    def spiral(start_radius: float, end_radius: float, pitch: float, turns: float) -> NurbsCurve:
        """Create a spiral (helix with varying radius)."""
        segments_per_turn = 8
        total_segments = max(4, int(turns * segments_per_turn))
        cv_count = total_segments + 1

        curve = NurbsCurve(dimension=3, is_rational=False, order=4, cv_count=cv_count)
        curve.m_cv = np.zeros(cv_count * 3, dtype=np.float64)

        total_angle = turns * 2 * math.pi

        for i in range(cv_count):
            t = i / total_segments
            angle = t * total_angle
            r = start_radius + t * (end_radius - start_radius)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = t * turns * pitch
            curve.set_cv(i, Point(x, y, z))

        curve.m_knot = knot.make_clamped_uniform(curve.m_order, curve.m_cv_count, 1.0)
        return curve
