from __future__ import annotations
from typing import List
from typing import NamedTuple
from typing import TYPE_CHECKING
# AABB — axis-aligned bounding box primitive (center + half-size).
# Use for: containment tests, intersection tests, tight bounds of geometry.

if TYPE_CHECKING:
    from .mesh import Mesh
    from .nurbscurve import NurbsCurve
    from .nurbssurface import NurbsSurface
    from .pointcloud import PointCloud
    from .polyline import Polyline
    from .line import Line
    from .point import Point


class AABB(NamedTuple):
    """Axis-aligned bounding box (center + half-size)."""

    cx: float
    cy: float
    cz: float
    hx: float
    hy: float
    hz: float

    @classmethod
    def from_point(cls, point: "Point", inflate: float = 0.0) -> "AABB":
        return cls(point[0], point[1], point[2], inflate, inflate, inflate)

    @classmethod
    def from_points(cls, points: list, inflate: float = 0.0) -> "AABB":
        if not points:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        min_z = min(p[2] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        max_z = max(p[2] for p in points)
        return cls(
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
            (max_x - min_x) * 0.5 + inflate,
            (max_y - min_y) * 0.5 + inflate,
            (max_z - min_z) * 0.5 + inflate,
        )

    @classmethod
    def from_coords_stride3(cls, coords: list[float], inflate: float = 0.0) -> "AABB":
        # Build an AABB directly from a stride-3 coord buffer (e.g. Polyline.coords)
        # without constructing an intermediate list of Points. Used on hot paths
        # like Session.add_polyline where the caller already has raw coords.
        if len(coords) < 3:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        n = len(coords) // 3
        min_x = max_x = coords[0]
        min_y = max_y = coords[1]
        min_z = max_z = coords[2]
        for i in range(1, n):
            x = coords[i * 3]
            y = coords[i * 3 + 1]
            z = coords[i * 3 + 2]
            if x < min_x: min_x = x
            elif x > max_x: max_x = x
            if y < min_y: min_y = y
            elif y > max_y: max_y = y
            if z < min_z: min_z = z
            elif z > max_z: max_z = z
        return cls(
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
            (max_x - min_x) * 0.5 + inflate,
            (max_y - min_y) * 0.5 + inflate,
            (max_z - min_z) * 0.5 + inflate,
        )

    @classmethod
    def from_line(cls, line: "Line", inflate: float = 0.0) -> "AABB":
        return cls.from_points([line.start(), line.end()], inflate)

    @classmethod
    def from_polyline(cls, polyline: "Polyline", inflate: float = 0.0) -> "AABB":
        return cls.from_points(polyline.points, inflate)

    @classmethod
    def from_mesh(cls, mesh: "Mesh", inflate: float = 0.0) -> "AABB":
        vertices, _ = mesh.to_vertices_and_faces()
        return cls.from_points(vertices, inflate)

    @classmethod
    def from_pointcloud(cls, pointcloud: "PointCloud", inflate: float = 0.0) -> "AABB":
        return cls.from_points(pointcloud.get_points(), inflate)

    @classmethod
    def from_nurbscurve(cls, curve: "NurbsCurve", inflate: float = 0.0, tight: bool = False) -> "AABB":
        from .vector import Vector
        if not curve.is_valid() or curve.cv_count() == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if not tight:
            points = [curve.get_cv(i) for i in range(curve.cv_count())]
            return cls.from_points(points, inflate)
        t0, t1 = curve.domain()
        extrema_points = [curve.point_at(t0), curve.point_at(t1)]
        for t in curve.get_span_vector():
            if t0 < t < t1:
                extrema_points.append(curve.point_at(t))
        NUM_SAMPLES = 20
        dt = (t1 - t0) / NUM_SAMPLES
        for i in range(NUM_SAMPLES):
            t_start = t0 + i * dt
            t_end = t_start + dt
            deriv_start = curve.evaluate(t_start, 1)
            deriv_end = curve.evaluate(t_end, 1)
            if len(deriv_start) < 2 or len(deriv_end) < 2:
                continue
            for axis_idx in range(3):
                d_start = deriv_start[1][axis_idx]
                d_end = deriv_end[1][axis_idx]
                if d_start * d_end < 0:
                    t_lo, t_hi = t_start, t_end
                    t_root = (t_lo + t_hi) * 0.5
                    for _ in range(20):
                        deriv = curve.evaluate(t_root, 2)
                        if len(deriv) < 3:
                            break
                        f = deriv[1][axis_idx]
                        fp = deriv[2][axis_idx]
                        if abs(f) < 1e-12:
                            break
                        if abs(fp) > 1e-14:
                            t_new = t_root - f / fp
                            if t_lo <= t_new <= t_hi:
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
                        if len(deriv_check) >= 2:
                            f_check = deriv_check[1][axis_idx]
                            if f_check * d_start < 0:
                                t_hi = t_root
                                d_end = f_check
                            else:
                                t_lo = t_root
                                d_start = f_check
                    extrema_points.append(curve.point_at(t_root))
        return cls.from_points(extrema_points, inflate)

    @classmethod
    def from_nurbssurface(cls, surface: "NurbsSurface", inflate: float = 0.0) -> "AABB":
        if not surface.is_valid() or surface.cv_count(0) == 0 or surface.cv_count(1) == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        points = []
        for i in range(surface.cv_count(0)):
            for j in range(surface.cv_count(1)):
                points.append(surface.get_cv(i, j))
        return cls.from_points(points, inflate)

    def min_point(self) -> "Point":
        from .point import Point
        return Point(self.cx - self.hx, self.cy - self.hy, self.cz - self.hz)

    def max_point(self) -> "Point":
        from .point import Point
        return Point(self.cx + self.hx, self.cy + self.hy, self.cz + self.hz)

    def corners(self) -> list["Point"]:
        from .point import Point
        cx, cy, cz, hx, hy, hz = self
        return [
            Point(cx + hx, cy + hy, cz - hz),
            Point(cx - hx, cy + hy, cz - hz),
            Point(cx - hx, cy - hy, cz - hz),
            Point(cx + hx, cy - hy, cz - hz),
            Point(cx + hx, cy + hy, cz + hz),
            Point(cx - hx, cy + hy, cz + hz),
            Point(cx - hx, cy - hy, cz + hz),
            Point(cx + hx, cy - hy, cz + hz),
        ]

    def inflate(self, amount: float) -> "AABB":
        return self._replace(hx=self.hx + amount, hy=self.hy + amount, hz=self.hz + amount)

    def intersects(self, other: "AABB") -> bool:
        return (
            self.cx - self.hx <= other.cx + other.hx
            and self.cx + self.hx >= other.cx - other.hx
            and self.cy - self.hy <= other.cy + other.hy
            and self.cy + self.hy >= other.cy - other.hy
            and self.cz - self.hz <= other.cz + other.hz
            and self.cz + self.hz >= other.cz - other.hz
        )

    @classmethod
    def merge(cls, a: "AABB", b: "AABB") -> "AABB":
        min_x = min(a.cx - a.hx, b.cx - b.hx)
        min_y = min(a.cy - a.hy, b.cy - b.hy)
        min_z = min(a.cz - a.hz, b.cz - b.hz)
        max_x = max(a.cx + a.hx, b.cx + b.hx)
        max_y = max(a.cy + a.hy, b.cy + b.hy)
        max_z = max(a.cz + a.hz, b.cz + b.hz)
        return cls(
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
            (max_x - min_x) * 0.5,
            (max_y - min_y) * 0.5,
            (max_z - min_z) * 0.5,
        )

    def center(self) -> "Point":
        from .point import Point
        return Point(self.cx, self.cy, self.cz)

    def area(self) -> float:
        return 8.0 * (self.hx * self.hy + self.hy * self.hz + self.hz * self.hx)

    def diagonal(self) -> float:
        import math
        return 2.0 * math.sqrt(self.hx * self.hx + self.hy * self.hy + self.hz * self.hz)

    def is_valid(self) -> bool:
        return self.hx >= 0.0 and self.hy >= 0.0 and self.hz >= 0.0

    def volume(self) -> float:
        return 8.0 * self.hx * self.hy * self.hz

    def closest_point(self, pt: "Point") -> "Point":
        from .point import Point
        x = max(self.cx - self.hx, min(self.cx + self.hx, pt[0]))
        y = max(self.cy - self.hy, min(self.cy + self.hy, pt[1]))
        z = max(self.cz - self.hz, min(self.cz + self.hz, pt[2]))
        return Point(x, y, z)

    def contains(self, pt: "Point") -> bool:
        return (self.cx - self.hx <= pt[0] <= self.cx + self.hx and
                self.cy - self.hy <= pt[1] <= self.cy + self.hy and
                self.cz - self.hz <= pt[2] <= self.cz + self.hz)

    def corner(self, x_max: bool, y_max: bool, z_max: bool) -> "Point":
        from .point import Point
        return Point(
            self.cx + (self.hx if x_max else -self.hx),
            self.cy + (self.hy if y_max else -self.hy),
            self.cz + (self.hz if z_max else -self.hz),
        )

    def get_corners(self) -> list["Point"]:
        return self.corners()

    def get_edges(self) -> list["Line"]:
        from .line import Line
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

    def point_at(self, x: float, y: float, z: float) -> "Point":
        from .point import Point
        return Point(self.cx + x, self.cy + y, self.cz + z)

    def union_with(self, other: "AABB") -> "AABB":
        min_x = min(self.cx - self.hx, other.cx - other.hx)
        min_y = min(self.cy - self.hy, other.cy - other.hy)
        min_z = min(self.cz - self.hz, other.cz - other.hz)
        max_x = max(self.cx + self.hx, other.cx + other.hx)
        max_y = max(self.cy + self.hy, other.cy + other.hy)
        max_z = max(self.cz + self.hz, other.cz + other.hz)
        return self._replace(
            cx=(min_x + max_x) * 0.5,
            cy=(min_y + max_y) * 0.5,
            cz=(min_z + max_z) * 0.5,
            hx=(max_x - min_x) * 0.5,
            hy=(max_y - min_y) * 0.5,
            hz=(max_z - min_z) * 0.5,
        )
