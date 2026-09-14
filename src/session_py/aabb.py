from __future__ import annotations
from typing import TYPE_CHECKING
import math
from .line import Line
from .point import Point
from .tolerance import Tolerance
from .tolerance import TOLERANCE

if TYPE_CHECKING:
    from .mesh import Mesh
    from .nurbscurve import NurbsCurve
    from .nurbssurface import NurbsSurface
    from .pointcloud import PointCloud
    from .polyline import Polyline

NUM_SAMPLES = 20
MAX_ITER = 20


class AABB:
    """Axis-aligned bounding box as center and half-size"""

    __slots__ = ("cx", "cy", "cz", "hx", "hy", "hz")

    def __init__(
        self,
        cx: float = 0.0,
        cy: float = 0.0,
        cz: float = 0.0,
        hx: float = 0.0,
        hy: float = 0.0,
        hz: float = 0.0,
    ):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.hx = hx
        self.hy = hy
        self.hz = hz

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def from_point(point: Point, inflate: float = 0.0) -> "AABB":
        """Box of half-size inflate around point"""
        return AABB(point[0], point[1], point[2], inflate, inflate, inflate)

    @staticmethod
    def from_points(points: list[Point], inflate: float = 0.0) -> "AABB":
        """Tight box of points grown by inflate"""
        if not points:
            return AABB()
        min_x = math.inf
        min_y = math.inf
        min_z = math.inf
        max_x = -math.inf
        max_y = -math.inf
        max_z = -math.inf
        for pt in points:
            min_x = min(min_x, pt[0])
            min_y = min(min_y, pt[1])
            min_z = min(min_z, pt[2])
            max_x = max(max_x, pt[0])
            max_y = max(max_y, pt[1])
            max_z = max(max_z, pt[2])
        return AABB(
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
            (max_x - min_x) * 0.5 + inflate,
            (max_y - min_y) * 0.5 + inflate,
            (max_z - min_z) * 0.5 + inflate,
        )

    @staticmethod
    def from_line(line: Line, inflate: float = 0.0) -> "AABB":
        """Tight box of the two ends grown by inflate"""
        return AABB.from_points([line.start(), line.end()], inflate)

    @staticmethod
    def from_polyline(polyline: "Polyline", inflate: float = 0.0) -> "AABB":
        """Tight box of the vertices grown by inflate"""
        return AABB.from_points(polyline.get_points(), inflate)

    @staticmethod
    def from_mesh(mesh: "Mesh", inflate: float = 0.0) -> "AABB":
        """Tight box of the vertices grown by inflate"""
        vertices, faces = mesh.to_vertices_and_faces()
        return AABB.from_points(vertices, inflate)

    @staticmethod
    def from_pointcloud(pointcloud: "PointCloud", inflate: float = 0.0) -> "AABB":
        """Tight box of the points grown by inflate"""
        return AABB.from_points(pointcloud.get_points(), inflate)

    @staticmethod
    def from_nurbscurve(
        curve: "NurbsCurve", inflate: float = 0.0, tight: bool = False
    ) -> "AABB":
        """Box of the control points, or of the curve extrema when tight"""
        if not curve.is_valid() or curve.cv_count() == 0:
            return AABB()
        points = []
        if not tight:
            for i in range(curve.cv_count()):
                points.append(curve.get_cv(i))
            return AABB.from_points(points, inflate)
        t0, t1 = curve.domain()
        points.append(curve.point_at(t0))
        points.append(curve.point_at(t1))
        for t in curve.get_span_vector():
            if t > t0 and t < t1:
                points.append(curve.point_at(t))
        dt = (t1 - t0) / NUM_SAMPLES
        for axis in range(3):
            for i in range(NUM_SAMPLES):
                t_start = t0 + i * dt
                t_end = t_start + dt
                deriv_start = curve.evaluate(t_start, 1)
                deriv_end = curve.evaluate(t_end, 1)
                if len(deriv_start) < 2 or len(deriv_end) < 2:
                    continue
                d_start = deriv_start[1][axis]
                d_end = deriv_end[1][axis]
                if d_start * d_end < 0:
                    t_root = AABB._compute_extremum(curve, axis, t_start, t_end, d_start)
                    points.append(curve.point_at(t_root))
        return AABB.from_points(points, inflate)

    @staticmethod
    def from_nurbssurface(surface: "NurbsSurface", inflate: float = 0.0) -> "AABB":
        """Box of the control points grown by inflate"""
        if (
            not surface.is_valid()
            or surface.cv_count(0) == 0
            or surface.cv_count(1) == 0
        ):
            return AABB()
        points = []
        for i in range(surface.cv_count(0)):
            for j in range(surface.cv_count(1)):
                points.append(surface.get_cv(i, j))
        return AABB.from_points(points, inflate)

    @staticmethod
    def merge(a: "AABB", b: "AABB") -> "AABB":
        """Box enclosing both a and b"""
        min_x = min(a.cx - a.hx, b.cx - b.hx)
        min_y = min(a.cy - a.hy, b.cy - b.hy)
        min_z = min(a.cz - a.hz, b.cz - b.hz)
        max_x = max(a.cx + a.hx, b.cx + b.hx)
        max_y = max(a.cy + a.hy, b.cy + b.hy)
        max_z = max(a.cz + a.hz, b.cz + b.hz)
        return AABB(
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
            (max_x - min_x) * 0.5,
            (max_y - min_y) * 0.5,
            (max_z - min_z) * 0.5,
        )

    @staticmethod
    def _compute_extremum(
        curve: "NurbsCurve", axis: int, t_lo: float, t_hi: float, d_start: float
    ) -> float:
        """Parameter in [t_lo, t_hi] where the axis derivative crosses zero, by Newton steps bracketed by bisection"""
        t_root = (t_lo + t_hi) * 0.5
        for it in range(MAX_ITER):
            deriv = curve.evaluate(t_root, 2)
            if len(deriv) < 3:
                break
            f = deriv[1][axis]
            fp = deriv[2][axis]
            if abs(f) < 1e-12:
                break
            if abs(fp) > 1e-14:
                t_new = t_root - f / fp
                if t_new >= t_lo and t_new <= t_hi:
                    t_root = t_new
                else:
                    if f * d_start < 0:
                        t_hi = t_root
                    else:
                        t_lo = t_root
                    t_root = (t_lo + t_hi) * 0.5
            else:
                t_root = (t_lo + t_hi) * 0.5
            deriv_check = curve.evaluate(t_root, 1)
            if len(deriv_check) < 2:
                continue
            f_check = deriv_check[1][axis]
            if f_check * d_start < 0:
                t_hi = t_root
            else:
                t_lo = t_root
                d_start = f_check
        return t_root

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════

    def __eq__(self, other) -> bool:
        """Center and half-size to 1e-6"""
        if not isinstance(other, AABB):
            return False
        return (
            round(self.cx * 1000000.0) == round(other.cx * 1000000.0)
            and round(self.cy * 1000000.0) == round(other.cy * 1000000.0)
            and round(self.cz * 1000000.0) == round(other.cz * 1000000.0)
            and round(self.hx * 1000000.0) == round(other.hx * 1000000.0)
            and round(self.hy * 1000000.0) == round(other.hy * 1000000.0)
            and round(self.hz * 1000000.0) == round(other.hz * 1000000.0)
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometry
    # ═══════════════════════════════════════════════════════════════════════════

    def min_point(self) -> Point:
        return Point(self.cx - self.hx, self.cy - self.hy, self.cz - self.hz)

    def max_point(self) -> Point:
        return Point(self.cx + self.hx, self.cy + self.hy, self.cz + self.hz)

    def center(self) -> Point:
        return Point(self.cx, self.cy, self.cz)

    def area(self) -> float:
        """Surface area"""
        return 8.0 * (self.hx * self.hy + self.hy * self.hz + self.hz * self.hx)

    def diagonal(self) -> float:
        """Length of the space diagonal"""
        return 2.0 * math.sqrt(
            self.hx * self.hx + self.hy * self.hy + self.hz * self.hz
        )

    def volume(self) -> float:
        return 8.0 * self.hx * self.hy * self.hz

    def is_valid(self) -> bool:
        """No negative half-size"""
        return self.hx >= 0.0 and self.hy >= 0.0 and self.hz >= 0.0

    def closest_point(self, pt: Point) -> Point:
        """pt clamped to the box"""
        x = max(self.cx - self.hx, min(self.cx + self.hx, pt[0]))
        y = max(self.cy - self.hy, min(self.cy + self.hy, pt[1]))
        z = max(self.cz - self.hz, min(self.cz + self.hz, pt[2]))
        return Point(x, y, z)

    def contains(self, pt: Point) -> bool:
        return (
            pt[0] >= self.cx - self.hx
            and pt[0] <= self.cx + self.hx
            and pt[1] >= self.cy - self.hy
            and pt[1] <= self.cy + self.hy
            and pt[2] >= self.cz - self.hz
            and pt[2] <= self.cz + self.hz
        )

    def intersects(self, other: "AABB") -> bool:
        return (
            self.cx - self.hx <= other.cx + other.hx
            and self.cx + self.hx >= other.cx - other.hx
            and self.cy - self.hy <= other.cy + other.hy
            and self.cy + self.hy >= other.cy - other.hy
            and self.cz - self.hz <= other.cz + other.hz
            and self.cz + self.hz >= other.cz - other.hz
        )

    def corner(self, x_max: bool, y_max: bool, z_max: bool) -> Point:
        """Corner picked by the sign of each half-size"""
        return Point(
            self.cx + (self.hx if x_max else -self.hx),
            self.cy + (self.hy if y_max else -self.hy),
            self.cz + (self.hz if z_max else -self.hz),
        )

    def corners(self) -> list[Point]:
        """Bottom loop then top loop, counter-clockwise from +x+y"""
        return [
            Point(self.cx + self.hx, self.cy + self.hy, self.cz - self.hz),
            Point(self.cx - self.hx, self.cy + self.hy, self.cz - self.hz),
            Point(self.cx - self.hx, self.cy - self.hy, self.cz - self.hz),
            Point(self.cx + self.hx, self.cy - self.hy, self.cz - self.hz),
            Point(self.cx + self.hx, self.cy + self.hy, self.cz + self.hz),
            Point(self.cx - self.hx, self.cy + self.hy, self.cz + self.hz),
            Point(self.cx - self.hx, self.cy - self.hy, self.cz + self.hz),
            Point(self.cx + self.hx, self.cy - self.hy, self.cz + self.hz),
        ]

    def get_corners(self) -> list[Point]:
        return self.corners()

    def get_edges(self) -> list[Line]:
        """Bottom loop, top loop, then the four verticals"""
        c = self.corners()
        return [
            Line.from_points(c[0], c[1]),
            Line.from_points(c[1], c[2]),
            Line.from_points(c[2], c[3]),
            Line.from_points(c[3], c[0]),
            Line.from_points(c[4], c[5]),
            Line.from_points(c[5], c[6]),
            Line.from_points(c[6], c[7]),
            Line.from_points(c[7], c[4]),
            Line.from_points(c[0], c[4]),
            Line.from_points(c[1], c[5]),
            Line.from_points(c[2], c[6]),
            Line.from_points(c[3], c[7]),
        ]

    def point_at(self, x: float, y: float, z: float) -> Point:
        """Center offset by x, y, z"""
        return Point(self.cx + x, self.cy + y, self.cz + z)

    def inflate(self, amount: float) -> None:
        """Grow every half-size by amount"""
        self.hx += amount
        self.hy += amount
        self.hz += amount

    def union_with(self, other: "AABB") -> None:
        """Grow to enclose other"""
        merged = AABB.merge(self, other)
        self.cx = merged.cx
        self.cy = merged.cy
        self.cz = merged.cz
        self.hx = merged.hx
        self.hy = merged.hy
        self.hz = merged.hz

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════

    def __str__(self) -> str:
        """cx, cy, cz, hx, hy, hz"""
        prec = Tolerance.ROUNDING
        return f"{TOLERANCE.format_number(self.cx, prec)}, {TOLERANCE.format_number(self.cy, prec)}, {TOLERANCE.format_number(self.cz, prec)}, {TOLERANCE.format_number(self.hx, prec)}, {TOLERANCE.format_number(self.hy, prec)}, {TOLERANCE.format_number(self.hz, prec)}"

    def __repr__(self) -> str:
        """AABB(cx, cy, cz, hx, hy, hz)"""
        return f"AABB({self})"
