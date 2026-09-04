from __future__ import annotations
from typing import TYPE_CHECKING
import uuid
from .point import Point
from .vector import Vector
from .plane import Plane
from .aabb import AABB

if TYPE_CHECKING:
    from .mesh import Mesh
    from .nurbscurve import NurbsCurve
    from .nurbssurface import NurbsSurface
    from pathlib import Path
    from .pointcloud import PointCloud
    from .polyline import Polyline
    from .line import Line
    from .xform import Xform


class OBB:
    def __init__(
        self,
        center: Point | None = None,
        x_axis: Vector | None = None,
        y_axis: Vector | None = None,
        z_axis: Vector | None = None,
        half_size: Vector | None = None,
    ):
        self.center = center if center is not None else Point(0.0, 0.0, 0.0)
        self.x_axis = x_axis if x_axis is not None else Vector(1.0, 0.0, 0.0)
        self.y_axis = y_axis if y_axis is not None else Vector(0.0, 1.0, 0.0)
        self.z_axis = z_axis if z_axis is not None else Vector(0.0, 0.0, 1.0)
        self.half_size = half_size if half_size is not None else Vector(0.5, 0.5, 0.5)
        self._guid = None
        self.name = "my_obb"

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    @classmethod
    def from_plane(cls, plane: Plane, dx: float, dy: float, dz: float) -> "OBB":
        return cls(
            center=plane.origin,
            x_axis=plane.x_axis,
            y_axis=plane.y_axis,
            z_axis=plane.z_axis,
            half_size=Vector(dx * 0.5, dy * 0.5, dz * 0.5),
        )

    @classmethod
    def from_aabb(cls, a: AABB) -> "OBB":
        return cls(
            center=Point(a.cx, a.cy, a.cz),
            x_axis=Vector(1.0, 0.0, 0.0),
            y_axis=Vector(0.0, 1.0, 0.0),
            z_axis=Vector(0.0, 0.0, 1.0),
            half_size=Vector(a.hx, a.hy, a.hz),
        )

    @classmethod
    def from_point(cls, point: Point, inflate: float = 0.0) -> "OBB":
        return cls.from_aabb(AABB.from_point(point, inflate))

    @classmethod
    def from_points(cls, points: list[Point], inflate: float = 0.0) -> "OBB":
        return cls.from_aabb(AABB.from_points(points, inflate))

    @classmethod
    def from_line(cls, line: "Line", inflate: float = 0.0) -> "OBB":
        return cls.from_aabb(AABB.from_line(line, inflate))

    @classmethod
    def from_polyline(cls, polyline: "Polyline", inflate: float = 0.0, plane: Plane | None = None) -> "OBB":
        """Create bounding box from polyline.

        Parameters
        ----------
        polyline : Polyline
            The polyline to bound.
        inflate : float, optional
            Amount to inflate the bounding box (default 0.0).
        plane : Plane, optional
            If provided, creates an OOBB aligned to the plane.

        Returns
        -------
        OBB
            Axis-aligned or oriented bounding box containing the polyline.
        """
        if plane is not None:
            return cls.from_points_with_plane(polyline.points, plane, inflate)
        return cls.from_aabb(AABB.from_polyline(polyline, inflate))

    @classmethod
    def from_mesh(cls, mesh: "Mesh", inflate: float = 0.0, plane: Plane | None = None) -> "OBB":
        """Create bounding box from mesh.

        Parameters
        ----------
        mesh : Mesh
            The mesh to bound.
        inflate : float, optional
            Amount to inflate the bounding box (default 0.0).
        plane : Plane, optional
            If provided, creates an OOBB aligned to the plane.

        Returns
        -------
        OBB
            Axis-aligned or oriented bounding box containing the mesh.
        """
        if plane is not None:
            vertices, faces = mesh.to_vertices_and_faces()
            return cls.from_points_with_plane(vertices, plane, inflate)
        return cls.from_aabb(AABB.from_mesh(mesh, inflate))

    @classmethod
    def from_pointcloud(cls, pointcloud: "PointCloud", inflate: float = 0.0, plane: Plane | None = None) -> "OBB":
        if plane is not None:
            return cls.from_points_with_plane(pointcloud.get_points(), plane, inflate)
        return cls.from_aabb(AABB.from_pointcloud(pointcloud, inflate))

    @classmethod
    def from_nurbssurface(cls, surface: "NurbsSurface", inflate: float = 0.0, plane: Plane | None = None) -> "OBB":
        if plane is None:
            return cls.from_aabb(AABB.from_nurbssurface(surface, inflate))
        if not surface.is_valid() or surface.cv_count(0) == 0 or surface.cv_count(1) == 0:
            return cls()
        points = []
        for i in range(surface.cv_count(0)):
            for j in range(surface.cv_count(1)):
                points.append(surface.get_cv(i, j))
        return cls.from_points_with_plane(points, plane, inflate)

    @classmethod
    def from_nurbscurve(cls, curve: "NurbsCurve", inflate: float = 0.0, tight: bool = False, plane: Plane | None = None) -> "OBB":
        if plane is None:
            return cls.from_aabb(AABB.from_nurbscurve(curve, inflate, tight))

        if not curve.is_valid() or curve.cv_count() == 0:
            return cls()

        if not tight:
            points = [curve.get_cv(i) for i in range(curve.cv_count())]
            return cls.from_points_with_plane(points, plane, inflate)

        t0, t1 = curve.domain()
        extrema_points = [curve.point_at(t0), curve.point_at(t1)]

        spans = curve.get_span_vector()
        for t in spans:
            if t > t0 and t < t1:
                extrema_points.append(curve.point_at(t))

        axes = [plane.x_axis, plane.y_axis, plane.z_axis]
        NUM_SAMPLES = 20
        dt = (t1 - t0) / NUM_SAMPLES

        for axis in axes:
            for i in range(NUM_SAMPLES):
                t_start = t0 + i * dt
                t_end = t_start + dt

                deriv_start = curve.evaluate(t_start, 1)
                deriv_end = curve.evaluate(t_end, 1)
                if len(deriv_start) < 2 or len(deriv_end) < 2:
                    continue

                d_start = deriv_start[1].dot(axis)
                d_end = deriv_end[1].dot(axis)

                if d_start * d_end < 0:
                    t_lo, t_hi = t_start, t_end
                    t_root = (t_lo + t_hi) * 0.5

                    for _ in range(20):
                        deriv = curve.evaluate(t_root, 2)
                        if len(deriv) < 3:
                            break

                        f = deriv[1].dot(axis)
                        fp = deriv[2].dot(axis)

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
                            f_check = deriv_check[1].dot(axis)
                            if f_check * d_start < 0:
                                t_hi = t_root
                                d_end = f_check
                            else:
                                t_lo = t_root
                                d_start = f_check

                    extrema_points.append(curve.point_at(t_root))

        return cls.from_points_with_plane(extrema_points, plane, inflate)

    @classmethod
    def from_points_with_plane(cls, points: list[Point], plane: Plane, inflate: float = 0.0) -> "OBB":
        if not points:
            return cls()

        from .xform import Xform
        origin = plane.origin
        x_axis = plane.x_axis
        y_axis = plane.y_axis
        z_axis = plane.z_axis
        # plane_to_xy stores the basis as matrix COLUMNS, so it rotates local->world in
        # both directions and its forward call yields the inverse frame for every plane
        # that is not axis-aligned. world_to_frame / frame_to_world are the real pair.
        world_to_local = Xform.world_to_frame(origin, x_axis, y_axis, z_axis)
        local_to_world = Xform.frame_to_world(origin, x_axis, y_axis, z_axis)

        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for pt in points:
            local_pt = pt.transformed(world_to_local)
            min_x = min(min_x, local_pt[0])
            min_y = min(min_y, local_pt[1])
            min_z = min(min_z, local_pt[2])
            max_x = max(max_x, local_pt[0])
            max_y = max(max_y, local_pt[1])
            max_z = max(max_z, local_pt[2])

        local_center = Point((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5)
        half_size = Vector(
            (max_x - min_x) * 0.5 + inflate,
            (max_y - min_y) * 0.5 + inflate,
            (max_z - min_z) * 0.5 + inflate
        )

        world_center = local_center.transformed(local_to_world)

        return cls(world_center, x_axis, y_axis, z_axis, half_size)

    def aabb(self) -> "AABB":
        ex, ey, ez = self.half_size[0], self.half_size[1], self.half_size[2]
        hx = abs(self.x_axis[0]) * ex + abs(self.y_axis[0]) * ey + abs(self.z_axis[0]) * ez
        hy = abs(self.x_axis[1]) * ex + abs(self.y_axis[1]) * ey + abs(self.z_axis[1]) * ez
        hz = abs(self.x_axis[2]) * ex + abs(self.y_axis[2]) * ey + abs(self.z_axis[2]) * ez
        return AABB(self.center[0], self.center[1], self.center[2], hx, hy, hz)

    def point_at(self, x: float, y: float, z: float) -> Point:
        return Point(
            self.center[0] + x * self.x_axis[0] + y * self.y_axis[0] + z * self.z_axis[0],
            self.center[1] + x * self.x_axis[1] + y * self.y_axis[1] + z * self.z_axis[1],
            self.center[2] + x * self.x_axis[2] + y * self.y_axis[2] + z * self.z_axis[2],
        )

    def min_point(self) -> Point:
        return self.aabb().min_point()

    def max_point(self) -> Point:
        return self.aabb().max_point()

    def corners(self) -> list[Point]:
        """Get all 8 corner points of the bounding box.

        Returns
        -------
        List[Point]
            List of 8 corner points in a specific order.
        """
        return [
            self.point_at(self.half_size[0], self.half_size[1], -self.half_size[2]),
            self.point_at(-self.half_size[0], self.half_size[1], -self.half_size[2]),
            self.point_at(-self.half_size[0], -self.half_size[1], -self.half_size[2]),
            self.point_at(self.half_size[0], -self.half_size[1], -self.half_size[2]),
            self.point_at(self.half_size[0], self.half_size[1], self.half_size[2]),
            self.point_at(-self.half_size[0], self.half_size[1], self.half_size[2]),
            self.point_at(-self.half_size[0], -self.half_size[1], self.half_size[2]),
            self.point_at(self.half_size[0], -self.half_size[1], self.half_size[2]),
        ]

    def two_rectangles(self) -> list[Point]:
        return [
            self.point_at(self.half_size[0], self.half_size[1], -self.half_size[2]),
            self.point_at(-self.half_size[0], self.half_size[1], -self.half_size[2]),
            self.point_at(-self.half_size[0], -self.half_size[1], -self.half_size[2]),
            self.point_at(self.half_size[0], -self.half_size[1], -self.half_size[2]),
            self.point_at(self.half_size[0], self.half_size[1], -self.half_size[2]),
            self.point_at(self.half_size[0], self.half_size[1], self.half_size[2]),
            self.point_at(-self.half_size[0], self.half_size[1], self.half_size[2]),
            self.point_at(-self.half_size[0], -self.half_size[1], self.half_size[2]),
            self.point_at(self.half_size[0], -self.half_size[1], self.half_size[2]),
            self.point_at(self.half_size[0], self.half_size[1], self.half_size[2]),
        ]

    def inflate(self, amount: float) -> None:
        self.half_size = Vector(
            self.half_size[0] + amount,
            self.half_size[1] + amount,
            self.half_size[2] + amount,
        )

    def area(self) -> float:
        hx, hy, hz = self.half_size[0], self.half_size[1], self.half_size[2]
        return 8.0 * (hx * hy + hy * hz + hz * hx)

    def diagonal(self) -> float:
        import math
        hx, hy, hz = self.half_size[0], self.half_size[1], self.half_size[2]
        return 2.0 * math.sqrt(hx * hx + hy * hy + hz * hz)

    def is_valid(self) -> bool:
        return self.half_size[0] >= 0.0 and self.half_size[1] >= 0.0 and self.half_size[2] >= 0.0

    def volume(self) -> float:
        return 8.0 * self.half_size[0] * self.half_size[1] * self.half_size[2]

    def closest_point(self, pt: Point) -> Point:
        dx = pt[0] - self.center[0]
        dy = pt[1] - self.center[1]
        dz = pt[2] - self.center[2]
        lx = dx * self.x_axis[0] + dy * self.x_axis[1] + dz * self.x_axis[2]
        ly = dx * self.y_axis[0] + dy * self.y_axis[1] + dz * self.y_axis[2]
        lz = dx * self.z_axis[0] + dy * self.z_axis[1] + dz * self.z_axis[2]
        hx, hy, hz = self.half_size[0], self.half_size[1], self.half_size[2]
        lx = max(-hx, min(hx, lx))
        ly = max(-hy, min(hy, ly))
        lz = max(-hz, min(hz, lz))
        return Point(
            self.center[0] + lx * self.x_axis[0] + ly * self.y_axis[0] + lz * self.z_axis[0],
            self.center[1] + lx * self.x_axis[1] + ly * self.y_axis[1] + lz * self.z_axis[1],
            self.center[2] + lx * self.x_axis[2] + ly * self.y_axis[2] + lz * self.z_axis[2],
        )

    def contains(self, pt: Point) -> bool:
        dx = pt[0] - self.center[0]
        dy = pt[1] - self.center[1]
        dz = pt[2] - self.center[2]
        lx = abs(dx * self.x_axis[0] + dy * self.x_axis[1] + dz * self.x_axis[2])
        ly = abs(dx * self.y_axis[0] + dy * self.y_axis[1] + dz * self.y_axis[2])
        lz = abs(dx * self.z_axis[0] + dy * self.z_axis[1] + dz * self.z_axis[2])
        return lx <= self.half_size[0] and ly <= self.half_size[1] and lz <= self.half_size[2]

    def corner(self, x_max: bool, y_max: bool, z_max: bool) -> Point:
        ox = self.half_size[0] if x_max else -self.half_size[0]
        oy = self.half_size[1] if y_max else -self.half_size[1]
        oz = self.half_size[2] if z_max else -self.half_size[2]
        return self.point_at(ox, oy, oz)

    def get_corners(self) -> list[Point]:
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

    def union_with(self, other: "OBB") -> None:
        min_x, max_x = -self.half_size[0], self.half_size[0]
        min_y, max_y = -self.half_size[1], self.half_size[1]
        min_z, max_z = -self.half_size[2], self.half_size[2]
        for c in other.corners():
            dx = c[0] - self.center[0]
            dy = c[1] - self.center[1]
            dz = c[2] - self.center[2]
            lx = dx * self.x_axis[0] + dy * self.x_axis[1] + dz * self.x_axis[2]
            ly = dx * self.y_axis[0] + dy * self.y_axis[1] + dz * self.y_axis[2]
            lz = dx * self.z_axis[0] + dy * self.z_axis[1] + dz * self.z_axis[2]
            min_x = min(min_x, lx)
            max_x = max(max_x, lx)
            min_y = min(min_y, ly)
            max_y = max(max_y, ly)
            min_z = min(min_z, lz)
            max_z = max(max_z, lz)
        ox = (min_x + max_x) * 0.5
        oy = (min_y + max_y) * 0.5
        oz = (min_z + max_z) * 0.5
        self.center = Point(
            self.center[0] + ox * self.x_axis[0] + oy * self.y_axis[0] + oz * self.z_axis[0],
            self.center[1] + ox * self.x_axis[1] + oy * self.y_axis[1] + oz * self.z_axis[1],
            self.center[2] + ox * self.x_axis[2] + oy * self.y_axis[2] + oz * self.z_axis[2],
        )
        self.half_size = Vector((max_x - min_x) * 0.5, (max_y - min_y) * 0.5, (max_z - min_z) * 0.5)

    @staticmethod
    def _separating_plane_exists(
        relative_position: Vector,
        axis: Vector,
        box1: "OBB",
        box2: "OBB",
    ) -> bool:
        dot_rp = abs(relative_position.dot(axis))

        v1 = box1.x_axis * box1.half_size[0]
        v2 = box1.y_axis * box1.half_size[1]
        v3 = box1.z_axis * box1.half_size[2]
        proj1 = abs(v1.dot(axis)) + abs(v2.dot(axis)) + abs(v3.dot(axis))

        v4 = box2.x_axis * box2.half_size[0]
        v5 = box2.y_axis * box2.half_size[1]
        v6 = box2.z_axis * box2.half_size[2]
        proj2 = abs(v4.dot(axis)) + abs(v5.dot(axis)) + abs(v6.dot(axis))

        return dot_rp > (proj1 + proj2)

    def collides_with_broad(self, other: "OBB") -> bool:
        if not self.aabb().intersects(other.aabb()):
            return False
        return self.collides_with(other)

    def collides_with(self, other: "OBB") -> bool:
        return self.collides_with_rtcd(other)

    def collides_with_rtcd(self, other: "OBB") -> bool:
        EPS = 1e-9
        A0, A1, A2 = self.x_axis, self.y_axis, self.z_axis
        B0, B1, B2 = other.x_axis, other.y_axis, other.z_axis
        a0, a1, a2 = self.half_size[0], self.half_size[1], self.half_size[2]
        b0, b1, b2 = other.half_size[0], other.half_size[1], other.half_size[2]
        R00, R01, R02 = A0.dot(B0), A0.dot(B1), A0.dot(B2)
        R10, R11, R12 = A1.dot(B0), A1.dot(B1), A1.dot(B2)
        R20, R21, R22 = A2.dot(B0), A2.dot(B1), A2.dot(B2)
        d = Vector(other.center[0] - self.center[0], other.center[1] - self.center[1], other.center[2] - self.center[2])
        t0, t1, t2 = d.dot(A0), d.dot(A1), d.dot(A2)
        AbsR00, AbsR01, AbsR02 = abs(R00) + EPS, abs(R01) + EPS, abs(R02) + EPS
        AbsR10, AbsR11, AbsR12 = abs(R10) + EPS, abs(R11) + EPS, abs(R12) + EPS
        AbsR20, AbsR21, AbsR22 = abs(R20) + EPS, abs(R21) + EPS, abs(R22) + EPS
        if abs(t0) > a0 + b0 * AbsR00 + b1 * AbsR01 + b2 * AbsR02: return False
        if abs(t1) > a1 + b0 * AbsR10 + b1 * AbsR11 + b2 * AbsR12: return False
        if abs(t2) > a2 + b0 * AbsR20 + b1 * AbsR21 + b2 * AbsR22: return False
        if abs(t0 * R00 + t1 * R10 + t2 * R20) > a0 * AbsR00 + a1 * AbsR10 + a2 * AbsR20 + b0: return False
        if abs(t0 * R01 + t1 * R11 + t2 * R21) > a0 * AbsR01 + a1 * AbsR11 + a2 * AbsR21 + b1: return False
        if abs(t0 * R02 + t1 * R12 + t2 * R22) > a0 * AbsR02 + a1 * AbsR12 + a2 * AbsR22 + b2: return False
        if abs(t2 * R10 - t1 * R20) > a1 * AbsR20 + a2 * AbsR10 + b1 * AbsR02 + b2 * AbsR01: return False
        if abs(t2 * R11 - t1 * R21) > a1 * AbsR21 + a2 * AbsR11 + b0 * AbsR02 + b2 * AbsR00: return False
        if abs(t2 * R12 - t1 * R22) > a1 * AbsR22 + a2 * AbsR12 + b0 * AbsR01 + b1 * AbsR00: return False
        if abs(t0 * R20 - t2 * R00) > a0 * AbsR20 + a2 * AbsR00 + b1 * AbsR12 + b2 * AbsR11: return False
        if abs(t0 * R21 - t2 * R01) > a0 * AbsR21 + a2 * AbsR01 + b0 * AbsR12 + b2 * AbsR10: return False
        if abs(t0 * R22 - t2 * R02) > a0 * AbsR22 + a2 * AbsR02 + b0 * AbsR11 + b1 * AbsR10: return False
        if abs(t1 * R00 - t0 * R10) > a0 * AbsR10 + a1 * AbsR00 + b1 * AbsR22 + b2 * AbsR21: return False
        if abs(t1 * R01 - t0 * R11) > a0 * AbsR11 + a1 * AbsR01 + b0 * AbsR22 + b2 * AbsR20: return False
        if abs(t1 * R02 - t0 * R12) > a0 * AbsR12 + a1 * AbsR02 + b0 * AbsR21 + b1 * AbsR20: return False
        return True

    def collides_with_naive(self, other: "OBB") -> bool:
        relative_position = Vector(
            other.center[0] - self.center[0],
            other.center[1] - self.center[1],
            other.center[2] - self.center[2],
        )
        x1, y1, z1 = self.x_axis, self.y_axis, self.z_axis
        x2, y2, z2 = other.x_axis, other.y_axis, other.z_axis
        if self._separating_plane_exists(relative_position, x1, self, other): return False
        if self._separating_plane_exists(relative_position, y1, self, other): return False
        if self._separating_plane_exists(relative_position, z1, self, other): return False
        if self._separating_plane_exists(relative_position, x2, self, other): return False
        if self._separating_plane_exists(relative_position, y2, self, other): return False
        if self._separating_plane_exists(relative_position, z2, self, other): return False
        if self._separating_plane_exists(relative_position, x1.cross(x2), self, other): return False
        if self._separating_plane_exists(relative_position, x1.cross(y2), self, other): return False
        if self._separating_plane_exists(relative_position, x1.cross(z2), self, other): return False
        if self._separating_plane_exists(relative_position, y1.cross(x2), self, other): return False
        if self._separating_plane_exists(relative_position, y1.cross(y2), self, other): return False
        if self._separating_plane_exists(relative_position, y1.cross(z2), self, other): return False
        if self._separating_plane_exists(relative_position, z1.cross(x2), self, other): return False
        if self._separating_plane_exists(relative_position, z1.cross(y2), self, other): return False
        if self._separating_plane_exists(relative_position, z1.cross(z2), self, other): return False
        return True

    ###########################################################################################
    # Transformation
    ###########################################################################################

    def transform(self, xform: "Xform") -> None:
        """Apply a transformation to the bounding box, in place."""
        self.center.transform(xform)
        self.x_axis.transform(xform)
        self.y_axis.transform(xform)
        self.z_axis.transform(xform)

    def transformed(self, xform: "Xform") -> "OBB":
        """Return a transformed copy of the bounding box."""
        import copy

        result = copy.deepcopy(self)
        result.transform(xform)
        return result

    ###########################################################################################
    # Polymorphic JSON Serialization (COMPAS-style)
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field."""
        # Alphabetical order to match Rust's serde_json
        return {
            "center": self.center.__jsondump__(),
            "guid": self.guid,
            "half_size": self.half_size.__jsondump__(),
            "name": self.name,
            "type": f"{self.__class__.__name__}",
            "x_axis": self.x_axis.__jsondump__(),
            "y_axis": self.y_axis.__jsondump__(),
            "z_axis": self.z_axis.__jsondump__(),
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format."""
        from .file_encoders import file_decode_node

        center = file_decode_node(data["center"])
        x_axis = file_decode_node(data["x_axis"])
        y_axis = file_decode_node(data["y_axis"])
        z_axis = file_decode_node(data["z_axis"])
        half_size = file_decode_node(data["half_size"])

        bbox = cls(center, x_axis, y_axis, z_axis, half_size)
        bbox.guid = guid if guid is not None else data.get("guid", bbox.guid)
        bbox.name = name if name is not None else data.get("name", bbox.name)

        return bbox

    def file_json_dumps(self) -> str:
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, s: str) -> "OBB":
        import json
        return cls.__jsonload__(json.loads(s))

    def file_json_dump(self, filepath: str | Path) -> None:
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: str | Path) -> "OBB":
        import json
        with open(filepath) as f:
            return cls.__jsonload__(json.load(f))

    def pb_dumps(self) -> bytes:
        from .proto import boundingbox_pb2
        proto = boundingbox_pb2.BoundingBox()
        proto.center.ParseFromString(self.center.pb_dumps())
        proto.x_axis.ParseFromString(self.x_axis.pb_dumps())
        proto.y_axis.ParseFromString(self.y_axis.pb_dumps())
        proto.z_axis.ParseFromString(self.z_axis.pb_dumps())
        proto.half_size.ParseFromString(self.half_size.pb_dumps())
        proto.guid = self.guid
        proto.name = self.name
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "OBB":
        from .proto import boundingbox_pb2
        proto = boundingbox_pb2.BoundingBox()
        proto.ParseFromString(data)
        center = Point.pb_loads(proto.center.SerializeToString())
        x_axis = Vector.pb_loads(proto.x_axis.SerializeToString())
        y_axis = Vector.pb_loads(proto.y_axis.SerializeToString())
        z_axis = Vector.pb_loads(proto.z_axis.SerializeToString())
        half_size = Vector.pb_loads(proto.half_size.SerializeToString())
        bbox = cls(center, x_axis, y_axis, z_axis, half_size)
        bbox.guid = proto.guid
        bbox.name = proto.name
        return bbox

    def pb_dump(self, filepath: str | Path) -> None:
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: str | Path) -> "OBB":
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())
