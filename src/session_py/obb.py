from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union
import copy
import json
import math
import uuid
from .aabb import AABB
from .line import Line
from .plane import Plane
from .point import Point
from .vector import Vector
from .xform import Xform

if TYPE_CHECKING:
    from pathlib import Path
    from .mesh import Mesh
    from .nurbscurve import NurbsCurve
    from .nurbssurface import NurbsSurface
    from .pointcloud import PointCloud
    from .polyline import Polyline

NUM_SAMPLES = 20
MAX_ITER = 20


class OBB:
    """Oriented bounding box as center, three axes and half-size."""

    __slots__ = ("_guid", "center", "x_axis", "y_axis", "z_axis", "half_size", "name")

    def __init__(
        self,
        center: Point | None = None,
        x_axis: Vector | None = None,
        y_axis: Vector | None = None,
        z_axis: Vector | None = None,
        half_size: Vector | None = None,
    ):
        """Construct from center, axes and half-size; defaults to the unit box at the origin."""

        self._guid = None
        self.center = center if center is not None else Point(0.0, 0.0, 0.0)
        self.x_axis = x_axis if x_axis is not None else Vector(1.0, 0.0, 0.0)
        self.y_axis = y_axis if y_axis is not None else Vector(0.0, 1.0, 0.0)
        self.z_axis = z_axis if z_axis is not None else Vector(0.0, 0.0, 1.0)
        self.half_size = half_size if half_size is not None else Vector(0.5, 0.5, 0.5)
        self.name = "my_obb"

    @staticmethod
    def from_plane(plane: Plane, dx: float, dy: float, dz: float) -> "OBB":
        """Construct on the plane frame with full sizes dx, dy, dz."""

        return OBB(
            plane.origin,
            plane.x_axis,
            plane.y_axis,
            plane.z_axis,
            Vector(dx * 0.5, dy * 0.5, dz * 0.5),
        )

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = OBB(
            copy.deepcopy(self.center, memo),
            copy.deepcopy(self.x_axis, memo),
            copy.deepcopy(self.y_axis, memo),
            copy.deepcopy(self.z_axis, memo),
            copy.deepcopy(self.half_size, memo),
        )
        result.name = self.name
        memo[id(self)] = result

        return result

    def duplicate(self) -> "OBB":
        """Copy with a new guid and the same data."""
        return copy.deepcopy(self)

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

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _from_aabb(aabb: AABB) -> "OBB":
        """Construct the world-aligned box with the center and half-size of aabb."""

        return OBB(
            Point(aabb.cx, aabb.cy, aabb.cz),
            Vector(1.0, 0.0, 0.0),
            Vector(0.0, 1.0, 0.0),
            Vector(0.0, 0.0, 1.0),
            Vector(aabb.hx, aabb.hy, aabb.hz),
        )

    @staticmethod
    def from_point(point: Point, inflate: float = 0.0) -> "OBB":
        """Construct the world-aligned box of half-size inflate around point."""
        return OBB._from_aabb(AABB.from_point(point, inflate))

    @staticmethod
    def from_points(
        points: list[Point], inflate: float = 0.0, plane: Plane | None = None
    ) -> "OBB":
        """Construct the world-aligned tight box of points, or tight in the plane frame, grown by inflate."""

        if plane is None:
            return OBB._from_aabb(AABB.from_points(points, inflate))

        if not points:
            return OBB()

        origin = plane.origin
        x_axis = plane.x_axis
        y_axis = plane.y_axis
        z_axis = plane.z_axis
        world_to_local = Xform.world_to_frame(origin, x_axis, y_axis, z_axis)
        local_to_world = Xform.frame_to_world(origin, x_axis, y_axis, z_axis)
        min_x = math.inf
        min_y = math.inf
        min_z = math.inf
        max_x = -math.inf
        max_y = -math.inf
        max_z = -math.inf

        for pt in points:
            local = pt.transformed(world_to_local)
            min_x = min(min_x, local[0])
            min_y = min(min_y, local[1])
            min_z = min(min_z, local[2])
            max_x = max(max_x, local[0])
            max_y = max(max_y, local[1])
            max_z = max(max_z, local[2])

        local_center = Point(
            (min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5
        )
        half_size = Vector(
            (max_x - min_x) * 0.5 + inflate,
            (max_y - min_y) * 0.5 + inflate,
            (max_z - min_z) * 0.5 + inflate,
        )

        return OBB(
            local_center.transformed(local_to_world), x_axis, y_axis, z_axis, half_size
        )

    @staticmethod
    def from_line(
        line: Line, inflate: float = 0.0, plane: Plane | None = None
    ) -> "OBB":
        """Construct the tight box of the two ends, world-aligned or in the plane frame, grown by inflate."""
        if plane is None:
            return OBB._from_aabb(AABB.from_line(line, inflate))

        return OBB.from_points([line.start(), line.end()], inflate, plane)

    @staticmethod
    def from_polyline(
        polyline: "Polyline", inflate: float = 0.0, plane: Plane | None = None
    ) -> "OBB":
        """Construct the tight box of the vertices, world-aligned or in the plane frame, grown by inflate."""
        if plane is None:
            return OBB._from_aabb(AABB.from_polyline(polyline, inflate))

        return OBB.from_points(polyline.get_points(), inflate, plane)

    @staticmethod
    def from_mesh(
        mesh: "Mesh", inflate: float = 0.0, plane: Plane | None = None
    ) -> "OBB":
        """Construct the tight box of the vertices, world-aligned or in the plane frame, grown by inflate."""

        if plane is None:
            return OBB._from_aabb(AABB.from_mesh(mesh, inflate))

        vertices, faces = mesh.to_vertices_and_faces()

        return OBB.from_points(vertices, inflate, plane)

    @staticmethod
    def from_pointcloud(
        pointcloud: "PointCloud", inflate: float = 0.0, plane: Plane | None = None
    ) -> "OBB":
        """Construct the tight box of the points, world-aligned or in the plane frame, grown by inflate."""
        if plane is None:
            return OBB._from_aabb(AABB.from_pointcloud(pointcloud, inflate))

        return OBB.from_points(pointcloud.get_points(), inflate, plane)

    @staticmethod
    def from_nurbscurve(
        curve: "NurbsCurve",
        inflate: float = 0.0,
        tight: bool = False,
        plane: Plane | None = None,
    ) -> "OBB":
        """Construct the box of the control points, or of the curve extrema when tight, world-aligned or in the plane frame."""

        if plane is None:
            return OBB._from_aabb(AABB.from_nurbscurve(curve, inflate, tight))

        if not curve.is_valid() or curve.cv_count() == 0:
            return OBB()

        points = []

        if not tight:
            for i in range(curve.cv_count()):
                points.append(curve.get_cv(i))

            return OBB.from_points(points, inflate, plane)

        t0, t1 = curve.domain()
        points.append(curve.point_at(t0))
        points.append(curve.point_at(t1))

        for t in curve.get_span_vector():
            if t > t0 and t < t1:
                points.append(curve.point_at(t))

        axes = [plane.x_axis, plane.y_axis, plane.z_axis]
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
                    t_root = OBB._compute_extremum(curve, axis, t_start, t_end, d_start)
                    points.append(curve.point_at(t_root))

        return OBB.from_points(points, inflate, plane)

    @staticmethod
    def from_nurbssurface(
        surface: "NurbsSurface", inflate: float = 0.0, plane: Plane | None = None
    ) -> "OBB":
        """Construct the box of the control points, world-aligned or in the plane frame, grown by inflate."""

        if plane is None:
            return OBB._from_aabb(AABB.from_nurbssurface(surface, inflate))

        if (
            not surface.is_valid()
            or surface.cv_count(0) == 0
            or surface.cv_count(1) == 0
        ):
            return OBB()

        points = []

        for i in range(surface.cv_count(0)):
            for j in range(surface.cv_count(1)):
                points.append(surface.get_cv(i, j))

        return OBB.from_points(points, inflate, plane)

    @staticmethod
    def _compute_extremum(
        curve: "NurbsCurve", axis: Vector, t_lo: float, t_hi: float, d_start: float
    ) -> float:
        """Compute the parameter in [t_lo, t_hi] where the derivative along axis crosses zero, by Newton steps bracketed by bisection."""

        t_root = (t_lo + t_hi) * 0.5

        for it in range(MAX_ITER):
            deriv = curve.evaluate(t_root, 2)

            if len(deriv) < 3:
                break

            f = deriv[1].dot(axis)
            fp = deriv[2].dot(axis)

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

            f_check = deriv_check[1].dot(axis)

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
        """Compare name, center, axes and half-size to 1e-6; guid ignored."""

        if not isinstance(other, OBB):
            return NotImplemented

        if self.name != other.name:
            return False

        for i in range(3):
            if round(self.center[i] * 1000000.0) != round(other.center[i] * 1000000.0):
                return False

            if round(self.x_axis[i] * 1000000.0) != round(other.x_axis[i] * 1000000.0):
                return False

            if round(self.y_axis[i] * 1000000.0) != round(other.y_axis[i] * 1000000.0):
                return False

            if round(self.z_axis[i] * 1000000.0) != round(other.z_axis[i] * 1000000.0):
                return False

            if round(self.half_size[i] * 1000000.0) != round(
                other.half_size[i] * 1000000.0
            ):
                return False

        return True

    def __ne__(self, other) -> bool:
        """Compare name, center, axes and half-size to 1e-6; guid ignored."""
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════

    def transform(self, xform: Xform) -> None:
        """Transform center and axes in place."""

        self.center.transform(xform)
        self.x_axis.transform(xform)
        self.y_axis.transform(xform)
        self.z_axis.transform(xform)

    def transformed(self, xform: Xform) -> "OBB":
        """Return a transformed copy."""
        result = copy.deepcopy(self)
        result.transform(xform)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometry
    # ═══════════════════════════════════════════════════════════════════════════

    def aabb(self) -> AABB:
        """Return the world-aligned box enclosing the corners."""

        ex = self.half_size[0]
        ey = self.half_size[1]
        ez = self.half_size[2]
        hx = (
            abs(self.x_axis[0]) * ex
            + abs(self.y_axis[0]) * ey
            + abs(self.z_axis[0]) * ez
        )
        hy = (
            abs(self.x_axis[1]) * ex
            + abs(self.y_axis[1]) * ey
            + abs(self.z_axis[1]) * ez
        )
        hz = (
            abs(self.x_axis[2]) * ex
            + abs(self.y_axis[2]) * ey
            + abs(self.z_axis[2]) * ez
        )

        return AABB(self.center[0], self.center[1], self.center[2], hx, hy, hz)

    def min_point(self) -> Point:
        """Return the min corner of the world-aligned box."""
        return self.aabb().min_point()

    def max_point(self) -> Point:
        """Return the max corner of the world-aligned box."""
        return self.aabb().max_point()

    def area(self) -> float:
        """Return the surface area."""

        hx = self.half_size[0]
        hy = self.half_size[1]
        hz = self.half_size[2]

        return 8.0 * (hx * hy + hy * hz + hz * hx)

    def diagonal(self) -> float:
        """Return the length of the space diagonal."""

        hx = self.half_size[0]
        hy = self.half_size[1]
        hz = self.half_size[2]

        return 2.0 * math.sqrt(hx * hx + hy * hy + hz * hz)

    def volume(self) -> float:
        """Return the volume."""
        return 8.0 * self.half_size[0] * self.half_size[1] * self.half_size[2]

    def is_valid(self) -> bool:
        """Return whether no half-size is negative."""

        return (
            self.half_size[0] >= 0.0
            and self.half_size[1] >= 0.0
            and self.half_size[2] >= 0.0
        )

    def closest_point(self, pt: Point) -> Point:
        """Return pt clamped to the box in its own frame."""

        d = pt - self.center
        lx = max(-self.half_size[0], min(self.half_size[0], d.dot(self.x_axis)))
        ly = max(-self.half_size[1], min(self.half_size[1], d.dot(self.y_axis)))
        lz = max(-self.half_size[2], min(self.half_size[2], d.dot(self.z_axis)))

        return self.point_at(lx, ly, lz)

    def contains(self, pt: Point) -> bool:
        """Return whether pt lies inside or on the box."""

        d = pt - self.center
        lx = abs(d.dot(self.x_axis))
        ly = abs(d.dot(self.y_axis))
        lz = abs(d.dot(self.z_axis))

        return (
            lx <= self.half_size[0]
            and ly <= self.half_size[1]
            and lz <= self.half_size[2]
        )

    def corner(self, x_max: bool, y_max: bool, z_max: bool) -> Point:
        """Return the corner picked by the sign of each half-size."""

        ox = self.half_size[0] if x_max else -self.half_size[0]
        oy = self.half_size[1] if y_max else -self.half_size[1]
        oz = self.half_size[2] if z_max else -self.half_size[2]

        return self.point_at(ox, oy, oz)

    def corners(self) -> list[Point]:
        """Return the bottom loop then the top loop, counter-clockwise from +x+y."""

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

    def get_corners(self) -> list[Point]:
        """Return the bottom loop then the top loop, counter-clockwise from +x+y."""
        return self.corners()

    def get_edges(self) -> list[Line]:
        """Return the bottom loop, the top loop, then the four verticals."""

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

    def two_rectangles(self) -> list[Point]:
        """Return the bottom loop and the top loop, each closed by repeating its first corner."""

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

    def point_at(self, x: float, y: float, z: float) -> Point:
        """Return the center offset by x, y, z along the axes."""

        return self.center + self.x_axis * x + self.y_axis * y + self.z_axis * z

    def inflate(self, amount: float) -> None:
        """Grow every half-size by amount."""

        self.half_size = self.half_size + Vector(amount, amount, amount)

    def union_with(self, other: "OBB") -> None:
        """Grow in place to enclose the corners of other."""

        min_x = -self.half_size[0]
        min_y = -self.half_size[1]
        min_z = -self.half_size[2]
        max_x = self.half_size[0]
        max_y = self.half_size[1]
        max_z = self.half_size[2]

        for c in other.corners():
            d = c - self.center
            lx = d.dot(self.x_axis)
            ly = d.dot(self.y_axis)
            lz = d.dot(self.z_axis)
            min_x = min(min_x, lx)
            min_y = min(min_y, ly)
            min_z = min(min_z, lz)
            max_x = max(max_x, lx)
            max_y = max(max_y, ly)
            max_z = max(max_z, lz)

        self.center = self.point_at(
            (min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5
        )
        self.half_size = Vector(
            (max_x - min_x) * 0.5, (max_y - min_y) * 0.5, (max_z - min_z) * 0.5
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Collision
    # ═══════════════════════════════════════════════════════════════════════════

    def collides_with(self, other: "OBB") -> bool:
        """Return whether the boxes overlap by the separating axis test (collides_with_rtcd)."""
        return self.collides_with_rtcd(other)

    def collides_with_broad(self, other: "OBB") -> bool:
        """Return whether the boxes overlap, rejecting by AABB before collides_with."""
        if not self.aabb().intersects(other.aabb()):
            return False

        return self.collides_with(other)

    def collides_with_rtcd(self, other: "OBB") -> bool:
        """Return whether the boxes overlap by the fifteen-axis test in the frame of this box (Real-Time Collision Detection)."""

        eps = 1e-9
        a0 = self.half_size[0]
        a1 = self.half_size[1]
        a2 = self.half_size[2]
        b0 = other.half_size[0]
        b1 = other.half_size[1]
        b2 = other.half_size[2]
        r00 = self.x_axis.dot(other.x_axis)
        r01 = self.x_axis.dot(other.y_axis)
        r02 = self.x_axis.dot(other.z_axis)
        r10 = self.y_axis.dot(other.x_axis)
        r11 = self.y_axis.dot(other.y_axis)
        r12 = self.y_axis.dot(other.z_axis)
        r20 = self.z_axis.dot(other.x_axis)
        r21 = self.z_axis.dot(other.y_axis)
        r22 = self.z_axis.dot(other.z_axis)
        d = other.center - self.center
        t0 = d.dot(self.x_axis)
        t1 = d.dot(self.y_axis)
        t2 = d.dot(self.z_axis)
        ar00 = abs(r00) + eps
        ar01 = abs(r01) + eps
        ar02 = abs(r02) + eps
        ar10 = abs(r10) + eps
        ar11 = abs(r11) + eps
        ar12 = abs(r12) + eps
        ar20 = abs(r20) + eps
        ar21 = abs(r21) + eps
        ar22 = abs(r22) + eps

        if abs(t0) > a0 + b0 * ar00 + b1 * ar01 + b2 * ar02:
            return False

        if abs(t1) > a1 + b0 * ar10 + b1 * ar11 + b2 * ar12:
            return False

        if abs(t2) > a2 + b0 * ar20 + b1 * ar21 + b2 * ar22:
            return False

        if abs(t0 * r00 + t1 * r10 + t2 * r20) > a0 * ar00 + a1 * ar10 + a2 * ar20 + b0:
            return False

        if abs(t0 * r01 + t1 * r11 + t2 * r21) > a0 * ar01 + a1 * ar11 + a2 * ar21 + b1:
            return False

        if abs(t0 * r02 + t1 * r12 + t2 * r22) > a0 * ar02 + a1 * ar12 + a2 * ar22 + b2:
            return False

        if abs(t2 * r10 - t1 * r20) > a1 * ar20 + a2 * ar10 + b1 * ar02 + b2 * ar01:
            return False

        if abs(t2 * r11 - t1 * r21) > a1 * ar21 + a2 * ar11 + b0 * ar02 + b2 * ar00:
            return False

        if abs(t2 * r12 - t1 * r22) > a1 * ar22 + a2 * ar12 + b0 * ar01 + b1 * ar00:
            return False

        if abs(t0 * r20 - t2 * r00) > a0 * ar20 + a2 * ar00 + b1 * ar12 + b2 * ar11:
            return False

        if abs(t0 * r21 - t2 * r01) > a0 * ar21 + a2 * ar01 + b0 * ar12 + b2 * ar10:
            return False

        if abs(t0 * r22 - t2 * r02) > a0 * ar22 + a2 * ar02 + b0 * ar11 + b1 * ar10:
            return False

        if abs(t1 * r00 - t0 * r10) > a0 * ar10 + a1 * ar00 + b1 * ar22 + b2 * ar21:
            return False

        if abs(t1 * r01 - t0 * r11) > a0 * ar11 + a1 * ar01 + b0 * ar22 + b2 * ar20:
            return False

        if abs(t1 * r02 - t0 * r12) > a0 * ar12 + a1 * ar02 + b0 * ar21 + b1 * ar20:
            return False

        return True

    def collides_with_naive(self, other: "OBB") -> bool:
        """Return whether the boxes overlap by the fifteen-axis test on projected extents."""

        rp = other.center - self.center
        axes = [
            self.x_axis,
            self.y_axis,
            self.z_axis,
            other.x_axis,
            other.y_axis,
            other.z_axis,
            self.x_axis.cross(other.x_axis),
            self.x_axis.cross(other.y_axis),
            self.x_axis.cross(other.z_axis),
            self.y_axis.cross(other.x_axis),
            self.y_axis.cross(other.y_axis),
            self.y_axis.cross(other.z_axis),
            self.z_axis.cross(other.x_axis),
            self.z_axis.cross(other.y_axis),
            self.z_axis.cross(other.z_axis),
        ]

        for axis in axes:
            if OBB._separating_plane_exists(rp, axis, self, other):
                return False

        return True

    @staticmethod
    def _separating_plane_exists(
        relative_position: Vector, axis: Vector, box1: "OBB", box2: "OBB"
    ) -> bool:
        """Return whether the extents of both boxes projected on axis do not reach their center distance."""

        proj1 = (
            abs((box1.x_axis * box1.half_size[0]).dot(axis))
            + abs((box1.y_axis * box1.half_size[1]).dot(axis))
            + abs((box1.z_axis * box1.half_size[2]).dot(axis))
        )
        proj2 = (
            abs((box2.x_axis * box2.half_size[0]).dot(axis))
            + abs((box2.y_axis * box2.half_size[1]).dot(axis))
            + abs((box2.z_axis * box2.half_size[2]).dot(axis))
        )

        return abs(relative_position.dot(axis)) > proj1 + proj2

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        return {
            "center": self.center.__jsondump__(),
            "guid": self.guid,
            "half_size": self.half_size.__jsondump__(),
            "name": self.name,
            "type": "OBB",
            "x_axis": self.x_axis.__jsondump__(),
            "y_axis": self.y_axis.__jsondump__(),
            "z_axis": self.z_axis.__jsondump__(),
        }

    @classmethod
    def __jsonload__(cls, data: dict, guid: str = None, name: str = None) -> "OBB":
        """Deserialize from a JSON object."""

        from .file_encoders import file_decode_node

        obb = cls(
            file_decode_node(data["center"]),
            file_decode_node(data["x_axis"]),
            file_decode_node(data["y_axis"]),
            file_decode_node(data["z_axis"]),
            file_decode_node(data["half_size"]),
        )
        obb.guid = guid if guid is not None else data["guid"]
        obb.name = name if name is not None else data["name"]

        return obb

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "OBB":
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write to a JSON file."""
        with open(filepath, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "OBB":
        """Read from a JSON file."""
        with open(filepath) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""

        from .proto import boundingbox_pb2

        proto = boundingbox_pb2.BoundingBox()
        proto.center.ParseFromString(self.center.pb_dumps())
        proto.x_axis.ParseFromString(self.x_axis.pb_dumps())
        proto.y_axis.ParseFromString(self.y_axis.pb_dumps())
        proto.z_axis.ParseFromString(self.z_axis.pb_dumps())
        proto.half_size.ParseFromString(self.half_size.pb_dumps())

        if self.has_guid():
            proto.guid = self._guid

        proto.name = self.name

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "OBB":
        """Deserialize from protobuf bytes."""

        from .proto import boundingbox_pb2

        proto = boundingbox_pb2.BoundingBox()
        proto.ParseFromString(data)
        obb = cls(
            Point.pb_loads(proto.center.SerializeToString()),
            Vector.pb_loads(proto.x_axis.SerializeToString()),
            Vector.pb_loads(proto.y_axis.SerializeToString()),
            Vector.pb_loads(proto.z_axis.SerializeToString()),
            Vector.pb_loads(proto.half_size.SerializeToString()),
        )

        if proto.guid:
            obb.guid = proto.guid

        obb.name = proto.name

        return obb

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write to a protobuf file."""
        with open(filepath, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "OBB":
        """Read from a protobuf file."""
        with open(filepath, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════

    def __str__(self) -> str:
        """Return "center."""
        return f"{self.center}\n{self.x_axis}\n{self.y_axis}\n{self.z_axis}\n{self.half_size}"

    def __repr__(self) -> str:
        """Return "OBB(name, center, x_axis, y_axis, z_axis, half_size)"."""
        return f"OBB({self.name}, {self.center}, {self.x_axis}, {self.y_axis}, {self.z_axis}, {self.half_size})"
