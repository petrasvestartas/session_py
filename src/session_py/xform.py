from __future__ import annotations
from typing import TYPE_CHECKING
import json
import math
import uuid

if TYPE_CHECKING:
    from pathlib import Path
    from .line import Line
    from .plane import Plane
    from .point import Point
    from .polyline import Polyline
    from .proto import xform_pb2

from .tolerance import TO_RADIANS
from .vector import Vector


class Xform:
    """A 4x4 column-major transformation matrix."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, matrix: list[float] | None = None):
        """Construct the identity, or from column-major values."""

        self._guid = None
        self.name = "my_xform"

        if matrix is not None:
            self.m = list(matrix)
            return

        self.m = [0.0] * 16
        self.m[0] = 1.0
        self.m[5] = 1.0
        self.m[10] = 1.0
        self.m[15] = 1.0

    def duplicate(self) -> Xform:
        """Copy with a new guid and the same data."""

        copy = Xform(self.m)
        copy.name = self.name

        return copy

    @staticmethod
    def identity() -> Xform:
        """Construct the identity."""
        return Xform()

    @staticmethod
    def from_matrix(matrix: list[float]) -> Xform:
        """Construct from column-major values."""
        return Xform(matrix)

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

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __mul__(self, other: Xform) -> Xform:
        """Multiply two transforms."""

        result = Xform()

        for i in range(4):
            for j in range(4):
                total = 0.0

                for k in range(4):
                    total += self.m[k * 4 + i] * other.m[j * 4 + k]

                result.m[j * 4 + i] = total

        return result

    def __imul__(self, other: Xform) -> Xform:
        """Multiply in place."""

        self.m = (self * other).m

        return self

    def __getitem__(self, index: tuple[int, int]) -> float:
        """Return the element at (row, col)."""

        row, col = index

        if not (0 <= row < 4 and 0 <= col < 4):
            raise IndexError(f"Index out of bounds: ({row}, {col})")

        return self.m[col * 4 + row]

    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        """Set the element at (row, col)."""

        row, col = index

        if not (0 <= row < 4 and 0 <= col < 4):
            raise IndexError(f"Index out of bounds: ({row}, {col})")

        self.m[col * 4 + row] = value

    def __eq__(self, other: object) -> bool:
        """Compare all elements within tolerance."""

        if not isinstance(other, Xform):
            return False

        for i in range(16):
            if abs(self.m[i] - other.m[i]) > 1e-10:
                return False

        return True

    def __ne__(self, other: object) -> bool:
        """Compare all elements within tolerance."""
        return not self == other

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformations
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def from_axes(col_x: Vector, col_y: Vector, col_z: Vector) -> Xform:
        """Construct a pure rotation from three column axis vectors."""

        xform = Xform()
        xform.m[0] = col_x[0]
        xform.m[1] = col_x[1]
        xform.m[2] = col_x[2]
        xform.m[4] = col_y[0]
        xform.m[5] = col_y[1]
        xform.m[6] = col_y[2]
        xform.m[8] = col_z[0]
        xform.m[9] = col_z[1]
        xform.m[10] = col_z[2]

        return xform

    @staticmethod
    def translation(x: float, y: float, z: float) -> Xform:
        """Construct a translation."""

        xform = Xform()
        xform.m[12] = x
        xform.m[13] = y
        xform.m[14] = z

        return xform

    @staticmethod
    def rotation_x(angle: float, degrees: bool = False) -> Xform:
        """Construct a rotation about the x axis."""

        if degrees:
            angle *= TO_RADIANS

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        xform = Xform()
        xform.m[5] = cos_angle
        xform.m[6] = sin_angle
        xform.m[9] = -sin_angle
        xform.m[10] = cos_angle

        return xform

    @staticmethod
    def rotation_y(angle: float, degrees: bool = False) -> Xform:
        """Construct a rotation about the y axis."""

        if degrees:
            angle *= TO_RADIANS

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        xform = Xform()
        xform.m[0] = cos_angle
        xform.m[2] = -sin_angle
        xform.m[8] = sin_angle
        xform.m[10] = cos_angle

        return xform

    @staticmethod
    def rotation_z(angle: float, degrees: bool = False) -> Xform:
        """Construct a rotation about the z axis."""

        if degrees:
            angle *= TO_RADIANS

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        xform = Xform()
        xform.m[0] = cos_angle
        xform.m[1] = sin_angle
        xform.m[4] = -sin_angle
        xform.m[5] = cos_angle

        return xform

    @staticmethod
    def rotation(axis: Vector, angle: float, degrees: bool = False) -> Xform:
        """Construct a rotation about an arbitrary axis through the origin."""

        if degrees:
            angle *= TO_RADIANS

        if axis.is_zero():
            return Xform.identity()

        unit = axis.normalized()
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        one_minus_cos = 1.0 - cos_angle
        xx = unit[0] * unit[0]
        xy = unit[0] * unit[1]
        xz = unit[0] * unit[2]
        yy = unit[1] * unit[1]
        yz = unit[1] * unit[2]
        zz = unit[2] * unit[2]

        xform = Xform()
        xform.m[0] = cos_angle + xx * one_minus_cos
        xform.m[1] = xy * one_minus_cos + unit[2] * sin_angle
        xform.m[2] = xz * one_minus_cos - unit[1] * sin_angle
        xform.m[4] = xy * one_minus_cos - unit[2] * sin_angle
        xform.m[5] = cos_angle + yy * one_minus_cos
        xform.m[6] = yz * one_minus_cos + unit[0] * sin_angle
        xform.m[8] = xz * one_minus_cos + unit[1] * sin_angle
        xform.m[9] = yz * one_minus_cos - unit[0] * sin_angle
        xform.m[10] = cos_angle + zz * one_minus_cos

        return xform

    @staticmethod
    def rotation_around_line(line: Line, angle: float, degrees: bool = False) -> Xform:
        """Construct a rotation about a line."""

        p = line.start()
        d = line.to_direction()
        t0 = Xform.translation(-p[0], -p[1], -p[2])
        r = Xform.rotation(d, angle, degrees)
        t1 = Xform.translation(p[0], p[1], p[2])

        return t1 * (r * t0)

    @staticmethod
    def _change_basis_pivot(r: list[list[float]], p: int, a: int, b: int) -> bool:
        """Scale row p to a unit pivot, then clear column p in rows a and b; false on a zero pivot."""

        if r[p][p] == 0.0:
            return False

        d = 1.0 / r[p][p]

        for j in range(6):
            r[p][j] *= d

        r[p][p] = 1.0

        if r[a][p] != 0.0:
            d = -r[a][p]

            for j in range(6):
                r[a][j] += d * r[p][j]

            r[a][p] = 0.0

        if r[b][p] != 0.0:
            d = -r[b][p]

            for j in range(6):
                r[b][j] += d * r[p][j]

            r[b][p] = 0.0

        return True

    @staticmethod
    def change_basis(
        origin_1: Point,
        x_axis_1: Vector,
        y_axis_1: Vector,
        z_axis_1: Vector,
        origin_0: Point,
        x_axis_0: Vector,
        y_axis_0: Vector,
        z_axis_0: Vector,
    ) -> Xform:
        """Construct a change of basis from frame 1 to frame 0."""

        a = x_axis_1.dot(y_axis_1)
        b = x_axis_1.dot(z_axis_1)
        c = y_axis_1.dot(z_axis_1)
        r = [
            [
                x_axis_1.dot(x_axis_1),
                a,
                b,
                x_axis_1.dot(x_axis_0),
                x_axis_1.dot(y_axis_0),
                x_axis_1.dot(z_axis_0),
            ],
            [
                a,
                y_axis_1.dot(y_axis_1),
                c,
                y_axis_1.dot(x_axis_0),
                y_axis_1.dot(y_axis_0),
                y_axis_1.dot(z_axis_0),
            ],
            [
                b,
                c,
                z_axis_1.dot(z_axis_1),
                z_axis_1.dot(x_axis_0),
                z_axis_1.dot(y_axis_0),
                z_axis_1.dot(z_axis_0),
            ],
        ]

        i0 = 0 if r[0][0] >= r[1][1] else 1

        if r[2][2] > r[i0][i0]:
            i0 = 2

        i1 = (i0 + 1) % 3
        i2 = (i1 + 1) % 3

        if not Xform._change_basis_pivot(r, i0, i1, i2):
            return Xform.identity()

        if abs(r[i1][i1]) < abs(r[i2][i2]):
            i1, i2 = i2, i1

        if not Xform._change_basis_pivot(r, i1, i0, i2):
            return Xform.identity()

        if not Xform._change_basis_pivot(r, i2, i0, i1):
            return Xform.identity()

        m_xform = Xform()
        m_xform.m[0] = r[0][3]
        m_xform.m[4] = r[0][4]
        m_xform.m[8] = r[0][5]
        m_xform.m[1] = r[1][3]
        m_xform.m[5] = r[1][4]
        m_xform.m[9] = r[1][5]
        m_xform.m[2] = r[2][3]
        m_xform.m[6] = r[2][4]
        m_xform.m[10] = r[2][5]

        t0 = Xform.translation(-origin_1[0], -origin_1[1], -origin_1[2])
        t2 = Xform.translation(origin_0[0], origin_0[1], origin_0[2])

        return t2 * (m_xform * t0)

    @staticmethod
    def from_change_of_basis(rect0: Polyline, rect1: Polyline) -> Xform:
        """Map the unit cube [-0.5, 0.5]^3 to the joint volume frame spanned by rect0 (x, y) and rect1[0] (z)."""

        from .point import Point

        if rect0.point_count() < 4 or rect1.point_count() < 1:
            return Xform.identity()

        origin_1 = Point(-0.5, -0.5, -0.5)
        x_axis_1 = Vector(1, 0, 0)
        y_axis_1 = Vector(0, 1, 0)
        z_axis_1 = Vector(0, 0, 1)
        origin_0 = rect0.get_point(0)
        x_axis_0 = rect0.get_point(1) - origin_0
        y_axis_0 = rect0.get_point(3) - origin_0
        z_axis_0 = rect1.get_point(0) - origin_0

        return Xform.change_basis(
            origin_1,
            x_axis_1,
            y_axis_1,
            z_axis_1,
            origin_0,
            x_axis_0,
            y_axis_0,
            z_axis_0,
        )

    @staticmethod
    def plane_to_plane(plane_from: Plane, plane_to: Plane) -> Xform:
        """Construct the transform taking one plane to another."""

        x0 = plane_from.x_axis.normalized()
        y0 = plane_from.y_axis.normalized()
        z0 = plane_from.z_axis.normalized()
        x1 = plane_to.x_axis.normalized()
        y1 = plane_to.y_axis.normalized()
        z1 = plane_to.z_axis.normalized()
        origin_0 = plane_from.origin
        origin_1 = plane_to.origin

        f0 = Xform()
        f0.m[0] = x0[0]
        f0.m[1] = x0[1]
        f0.m[2] = x0[2]
        f0.m[4] = y0[0]
        f0.m[5] = y0[1]
        f0.m[6] = y0[2]
        f0.m[8] = z0[0]
        f0.m[9] = z0[1]
        f0.m[10] = z0[2]

        f1 = Xform()
        f1.m[0] = x1[0]
        f1.m[4] = x1[1]
        f1.m[8] = x1[2]
        f1.m[1] = y1[0]
        f1.m[5] = y1[1]
        f1.m[9] = y1[2]
        f1.m[2] = z1[0]
        f1.m[6] = z1[1]
        f1.m[10] = z1[2]

        t0 = Xform.translation(-origin_0[0], -origin_0[1], -origin_0[2])
        r = f1 * f0
        t1 = Xform.translation(origin_1[0], origin_1[1], origin_1[2])

        return t1 * (r * t0)

    @staticmethod
    def world_to_frame(
        origin: Point, x_axis: Vector, y_axis: Vector, z_axis: Vector
    ) -> Xform:
        """Construct the world point to frame coordinates transform (axes as rows)."""

        x = x_axis.normalized()
        y = y_axis.normalized()
        z = z_axis.normalized()

        f = Xform()
        f.m[0] = x[0]
        f.m[4] = x[1]
        f.m[8] = x[2]
        f.m[1] = y[0]
        f.m[5] = y[1]
        f.m[9] = y[2]
        f.m[2] = z[0]
        f.m[6] = z[1]
        f.m[10] = z[2]

        t = Xform.translation(-origin[0], -origin[1], -origin[2])

        return f * t

    @staticmethod
    def frame_to_world(
        origin: Point, x_axis: Vector, y_axis: Vector, z_axis: Vector
    ) -> Xform:
        """Construct the frame coordinates to world point transform (axes as columns)."""

        x = x_axis.normalized()
        y = y_axis.normalized()
        z = z_axis.normalized()

        f = Xform()
        f.m[0] = x[0]
        f.m[1] = x[1]
        f.m[2] = x[2]
        f.m[4] = y[0]
        f.m[5] = y[1]
        f.m[6] = y[2]
        f.m[8] = z[0]
        f.m[9] = z[1]
        f.m[10] = z[2]

        t = Xform.translation(origin[0], origin[1], origin[2])

        return t * f

    @staticmethod
    def to_frame(frame: Plane) -> Xform:
        """Construct the world XY to frame plane transform (COMPAS from_frame)."""

        x = frame.x_axis.normalized()
        y = frame.y_axis.normalized()
        z = frame.z_axis.normalized()
        o = frame.origin

        xform = Xform()
        xform.m[0] = x[0]
        xform.m[4] = y[0]
        xform.m[8] = z[0]
        xform.m[12] = o[0]
        xform.m[1] = x[1]
        xform.m[5] = y[1]
        xform.m[9] = z[1]
        xform.m[13] = o[1]
        xform.m[2] = x[2]
        xform.m[6] = y[2]
        xform.m[10] = z[2]
        xform.m[14] = o[2]

        return xform

    @staticmethod
    def scale_xyz(scale_x: float, scale_y: float, scale_z: float) -> Xform:
        """Construct a scale about the origin."""

        xform = Xform()
        xform.m[0] = scale_x
        xform.m[5] = scale_y
        xform.m[10] = scale_z

        return xform

    @staticmethod
    def scale_uniform(origin: Point, scale_value: float) -> Xform:
        """Construct a uniform scale about a point."""

        t0 = Xform.translation(-origin[0], -origin[1], -origin[2])
        t1 = Xform.scale_xyz(scale_value, scale_value, scale_value)
        t2 = Xform.translation(origin[0], origin[1], origin[2])

        return t2 * (t1 * t0)

    @staticmethod
    def scale_non_uniform(
        origin: Point, scale_x: float, scale_y: float, scale_z: float
    ) -> Xform:
        """Construct a non-uniform scale about a point."""

        t0 = Xform.translation(-origin[0], -origin[1], -origin[2])
        t1 = Xform.scale_xyz(scale_x, scale_y, scale_z)
        t2 = Xform.translation(origin[0], origin[1], origin[2])

        return t2 * (t1 * t0)

    @staticmethod
    def axis_rotation(angle: float, axis: Vector, degrees: bool = False) -> Xform:
        """Construct a Rodrigues rotation about a unit axis."""

        if degrees:
            angle *= TO_RADIANS

        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        ux = axis[0]
        uy = axis[1]
        uz = axis[2]

        xform = Xform()
        xform.m[0] = t * ux * ux + c
        xform.m[4] = t * ux * uy - uz * s
        xform.m[8] = t * ux * uz + uy * s
        xform.m[1] = t * ux * uy + uz * s
        xform.m[5] = t * uy * uy + c
        xform.m[9] = t * uy * uz - ux * s
        xform.m[2] = t * ux * uz - uy * s
        xform.m[6] = t * uy * uz + ux * s
        xform.m[10] = t * uz * uz + c

        return xform

    @staticmethod
    def look_at_right_handed(eye: Point, target: Point, up: Vector) -> Xform:
        """Construct a right-handed view matrix looking at a target (camera looks down -Z, up must not be parallel to the view)."""
        return Xform.look_to_right_handed(eye, target - eye, up)

    @staticmethod
    def look_to_right_handed(eye: Point, direction: Vector, up: Vector) -> Xform:
        """Construct a right-handed view matrix looking along a direction."""

        f = direction.normalized()
        s = f.cross(up.normalized()).normalized()
        u = s.cross(f)
        eye_vector = Vector(eye[0], eye[1], eye[2])

        xform = Xform()
        xform.m[0] = s[0]
        xform.m[4] = s[1]
        xform.m[8] = s[2]
        xform.m[1] = u[0]
        xform.m[5] = u[1]
        xform.m[9] = u[2]
        xform.m[2] = -f[0]
        xform.m[6] = -f[1]
        xform.m[10] = -f[2]
        xform.m[12] = -s.dot(eye_vector)
        xform.m[13] = -u.dot(eye_vector)
        xform.m[14] = f.dot(eye_vector)

        return xform

    @staticmethod
    def perspective(fov_y: float, aspect: float, near: float, far: float) -> Xform:
        """Construct a right-handed perspective projection with depth [0, 1]."""

        f = 1.0 / math.tan(fov_y / 2.0)
        nf = near - far

        xform = Xform([0.0] * 16)
        xform.m[0] = f / aspect
        xform.m[5] = f
        xform.m[10] = far / nf
        xform.m[11] = -1.0
        xform.m[14] = (near * far) / nf

        return xform

    @staticmethod
    def orthographic(
        left: float, right: float, bottom: float, top: float, near: float, far: float
    ) -> Xform:
        """Construct a right-handed orthographic projection with depth [0, 1]."""

        rl = right - left
        tb = top - bottom
        nf = near - far

        xform = Xform([0.0] * 16)
        xform.m[0] = 2.0 / rl
        xform.m[5] = 2.0 / tb
        xform.m[10] = 1.0 / nf
        xform.m[12] = (left + right) / (left - right)
        xform.m[13] = (bottom + top) / (bottom - top)
        xform.m[14] = near / nf
        xform.m[15] = 1.0

        return xform

    @staticmethod
    def project_to_plane(plane: Plane) -> Xform:
        """Construct an orthogonal projection onto a plane."""

        n = plane.z_axis
        o = plane.origin
        nx = n[0]
        ny = n[1]
        nz = n[2]
        d = o[0] * nx + o[1] * ny + o[2] * nz

        xform = Xform()
        xform.m[0] = 1.0 - nx * nx
        xform.m[4] = -nx * ny
        xform.m[8] = -nx * nz
        xform.m[12] = nx * d
        xform.m[1] = -ny * nx
        xform.m[5] = 1.0 - ny * ny
        xform.m[9] = -ny * nz
        xform.m[13] = ny * d
        xform.m[2] = -nz * nx
        xform.m[6] = -nz * ny
        xform.m[10] = 1.0 - nz * nz
        xform.m[14] = nz * d

        return xform

    @staticmethod
    def project_to_plane_by_axis(plane: Plane, direction: Vector) -> Xform:
        """Construct a projection onto a plane along a direction."""

        n = plane.z_axis
        o = plane.origin
        nx = n[0]
        ny = n[1]
        nz = n[2]
        dx = direction[0]
        dy = direction[1]
        dz = direction[2]
        s = 1.0 / (nx * dx + ny * dy + nz * dz)
        d = o[0] * nx + o[1] * ny + o[2] * nz

        xform = Xform()
        xform.m[0] = 1.0 - dx * s * nx
        xform.m[4] = -dx * s * ny
        xform.m[8] = -dx * s * nz
        xform.m[12] = dx * s * d
        xform.m[1] = -dy * s * nx
        xform.m[5] = 1.0 - dy * s * ny
        xform.m[9] = -dy * s * nz
        xform.m[13] = dy * s * d
        xform.m[2] = -dz * s * nx
        xform.m[6] = -dz * s * ny
        xform.m[10] = 1.0 - dz * s * nz
        xform.m[14] = dz * s * d

        return xform

    # ═══════════════════════════════════════════════════════════════════════════
    # Apply transformations
    # ═══════════════════════════════════════════════════════════════════════════
    def transform_point(self, p: Point) -> Point:
        """Transform a point with a homogeneous multiply, dividing by w when projective."""

        from .point import Point

        x = self.m[0] * p[0] + self.m[4] * p[1] + self.m[8] * p[2] + self.m[12]
        y = self.m[1] * p[0] + self.m[5] * p[1] + self.m[9] * p[2] + self.m[13]
        z = self.m[2] * p[0] + self.m[6] * p[1] + self.m[10] * p[2] + self.m[14]
        w = self.m[3] * p[0] + self.m[7] * p[1] + self.m[11] * p[2] + self.m[15]

        if abs(w) < 1e-12:
            return Point(x, y, z)

        return Point(x / w, y / w, z / w)

    def transform_vector(self, v: Vector) -> Vector:
        """Transform a vector with rotation and scale only."""

        x = self.m[0] * v[0] + self.m[4] * v[1] + self.m[8] * v[2]
        y = self.m[1] * v[0] + self.m[5] * v[1] + self.m[9] * v[2]
        z = self.m[2] * v[0] + self.m[6] * v[1] + self.m[10] * v[2]

        return Vector(x, y, z)

    # ═══════════════════════════════════════════════════════════════════════════
    # Details
    # ═══════════════════════════════════════════════════════════════════════════
    def inverse(self) -> Xform | None:
        """Return the inverse, or None when singular."""

        s0 = self.m[0] * self.m[5] - self.m[1] * self.m[4]
        s1 = self.m[0] * self.m[9] - self.m[1] * self.m[8]
        s2 = self.m[0] * self.m[13] - self.m[1] * self.m[12]
        s3 = self.m[4] * self.m[9] - self.m[5] * self.m[8]
        s4 = self.m[4] * self.m[13] - self.m[5] * self.m[12]
        s5 = self.m[8] * self.m[13] - self.m[9] * self.m[12]
        c5 = self.m[10] * self.m[15] - self.m[11] * self.m[14]
        c4 = self.m[6] * self.m[15] - self.m[7] * self.m[14]
        c3 = self.m[6] * self.m[11] - self.m[7] * self.m[10]
        c2 = self.m[2] * self.m[15] - self.m[3] * self.m[14]
        c1 = self.m[2] * self.m[11] - self.m[3] * self.m[10]
        c0 = self.m[2] * self.m[7] - self.m[3] * self.m[6]
        det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0

        if abs(det) < 1e-12:
            return None

        inv_det = 1.0 / det

        result = Xform()
        result.m[0] = (self.m[5] * c5 - self.m[9] * c4 + self.m[13] * c3) * inv_det
        result.m[4] = (-self.m[4] * c5 + self.m[8] * c4 - self.m[12] * c3) * inv_det
        result.m[8] = (self.m[7] * s5 - self.m[11] * s4 + self.m[15] * s3) * inv_det
        result.m[12] = (-self.m[6] * s5 + self.m[10] * s4 - self.m[14] * s3) * inv_det
        result.m[1] = (-self.m[1] * c5 + self.m[9] * c2 - self.m[13] * c1) * inv_det
        result.m[5] = (self.m[0] * c5 - self.m[8] * c2 + self.m[12] * c1) * inv_det
        result.m[9] = (-self.m[3] * s5 + self.m[11] * s2 - self.m[15] * s1) * inv_det
        result.m[13] = (self.m[2] * s5 - self.m[10] * s2 + self.m[14] * s1) * inv_det
        result.m[2] = (self.m[1] * c4 - self.m[5] * c2 + self.m[13] * c0) * inv_det
        result.m[6] = (-self.m[0] * c4 + self.m[4] * c2 - self.m[12] * c0) * inv_det
        result.m[10] = (self.m[3] * s4 - self.m[7] * s2 + self.m[15] * s0) * inv_det
        result.m[14] = (-self.m[2] * s4 + self.m[6] * s2 - self.m[14] * s0) * inv_det
        result.m[3] = (-self.m[1] * c3 + self.m[5] * c1 - self.m[9] * c0) * inv_det
        result.m[7] = (self.m[0] * c3 - self.m[4] * c1 + self.m[8] * c0) * inv_det
        result.m[11] = (-self.m[3] * s3 + self.m[7] * s1 - self.m[11] * s0) * inv_det
        result.m[15] = (self.m[2] * s3 - self.m[6] * s1 + self.m[10] * s0) * inv_det

        return result

    def is_identity(self) -> bool:
        """Return whether the matrix is the identity."""
        return self == Xform()

    def to_cols(self) -> list[list[float]]:
        """Return four columns of four rows."""

        return [
            [self.m[0], self.m[1], self.m[2], self.m[3]],
            [self.m[4], self.m[5], self.m[6], self.m[7]],
            [self.m[8], self.m[9], self.m[10], self.m[11]],
            [self.m[12], self.m[13], self.m[14], self.m[15]],
        ]

    def uniform_scale(self) -> float:
        """Return the length of the first column: the uniform scale the matrix applies."""
        return math.sqrt(
            self.m[0] * self.m[0] + self.m[1] * self.m[1] + self.m[2] * self.m[2]
        )

    def eye(self) -> Point:
        """Return the eye of a view-projection: where clip x, y and w vanish at once; orthographic has none, so the view direction pushed far back."""

        from .point import Point

        rows = [
            [self.m[0], self.m[4], self.m[8]],
            [self.m[1], self.m[5], self.m[9]],
            [self.m[3], self.m[7], self.m[11]],
        ]
        rhs = [-self.m[12], -self.m[13], -self.m[15]]
        d = Xform._det3(rows)
        norm = 1.0

        for row in rows:
            norm *= math.sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2])

        if abs(d) <= 1e-9 * max(norm, 1e-30):
            length = max(math.sqrt(self.m[2] * self.m[2] + self.m[6] * self.m[6] + self.m[10] * self.m[10]), 1e-30)

            return Point(
                self.m[2] / length * 1.0e9, self.m[6] / length * 1.0e9, self.m[10] / length * 1.0e9
            )

        eye = [0.0, 0.0, 0.0]

        for k in range(3):
            replaced = [rows[0][:], rows[1][:], rows[2][:]]

            for row in range(3):
                replaced[row][k] = rhs[row]

            eye[k] = Xform._det3(replaced) / d

        return Point(eye[0], eye[1], eye[2])

    def ortho_half_height(self) -> float:
        """Return the half-height of an orthographic view-projection in world units, 0 in perspective."""

        w2 = self.m[3] * self.m[3] + self.m[7] * self.m[7] + self.m[11] * self.m[11]

        if w2 > 1e-12:
            return 0.0

        r1 = self.m[1] * self.m[1] + self.m[5] * self.m[5] + self.m[9] * self.m[9]

        if r1 <= 1e-30:
            return 0.0

        return 1.0 / math.sqrt(r1)

    @staticmethod
    def _det3(rows: list[list[float]]) -> float:
        """Return the determinant of a 3x3 given by rows."""

        return (
            rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to an ordered JSON object."""

        return {
            "guid": self.guid,
            "m": self.m,
            "name": self.name,
            "type": "Xform",
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Xform:
        """Deserialize from a JSON object."""

        xform = cls(data["m"])
        xform.guid = guid or data["guid"]
        xform.name = name or data["name"]

        return xform

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> Xform:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write JSON to a file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Xform:
        """Read JSON from a file."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> xform_pb2.Xform:
        """Convert to the protobuf message."""

        from .proto import xform_pb2

        proto = xform_pb2.Xform()

        if self.has_guid():
            proto.guid = self.guid

        proto.name = self.name
        proto.matrix.extend(self.m)

        return proto

    @classmethod
    def from_proto(cls, proto: xform_pb2.Xform) -> Xform:
        """Construct from the protobuf message."""

        xform = cls()

        if proto.guid:
            xform.guid = proto.guid

        xform.name = proto.name

        for i in range(min(16, len(proto.matrix))):
            xform.m[i] = proto.matrix[i]

        return xform

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Xform:
        """Deserialize from protobuf bytes."""

        from .proto import xform_pb2

        proto = xform_pb2.Xform()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write protobuf bytes to a file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Xform:
        """Read protobuf bytes from a file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return the four matrix rows."""

        rows = []

        for i in range(4):
            rows.append(
                f"[{self.m[i]:.6f}, {self.m[4 + i]:.6f}, {self.m[8 + i]:.6f}, {self.m[12 + i]:.6f}]"
            )

        return "\n".join(rows)

    def __repr__(self) -> str:
        """Return the name and guid prefix."""
        return f"Xform({self.name}, {self.guid[:8]})"
