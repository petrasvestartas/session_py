from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union
import copy
import json
import math
import uuid
from .color import Color
from .point import Point
from .tolerance import Tolerance
from .tolerance import TOLERANCE
from .vector import Vector

if TYPE_CHECKING:
    from pathlib import Path
    from .polyline import Polyline
    from .xform import Xform


class Plane:
    """A plane defined by an origin and an orthonormal x, y, z frame."""

    __slots__ = (
        "_guid",
        "name",
        "width",
        "linecolor",
        "_origin",
        "_x_axis",
        "_y_axis",
        "_z_axis",
        "_a",
        "_b",
        "_c",
        "_d",
    )

    def __init__(
        self,
        point: Point | None = None,
        x_axis: Vector | None = None,
        y_axis: Vector | None = None,
        name: str = "my_plane",
    ):
        """Construct from an origin and two axes; x is normalized, y is made orthogonal to x, z = x × y."""

        self._guid = None
        self.name = name
        self.width = 1.0
        self.linecolor = Color.blue()
        self._origin = Point() if point is None else Point(point[0], point[1], point[2])
        self._x_axis = (
            Vector.x_axis()
            if x_axis is None
            else Vector(x_axis[0], x_axis[1], x_axis[2])
        )
        self._x_axis.normalize_self()
        self._y_axis = (
            Vector.y_axis()
            if y_axis is None
            else y_axis - self._x_axis * y_axis.dot(self._x_axis)
        )
        self._y_axis.normalize_self()
        self._z_axis = self._x_axis.cross(self._y_axis)
        self._z_axis.normalize_self()
        self._update_equation()

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = Plane.from_frame(
            self._origin, self._x_axis, self._y_axis, self._z_axis
        )
        result.name = self.name
        result.width = self.width
        result.linecolor = copy.deepcopy(self.linecolor, memo)
        memo[id(self)] = result

        return result

    def duplicate(self) -> "Plane":
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

    @property
    def origin(self) -> Point:
        """Return the origin."""
        return self._origin

    @property
    def x_axis(self) -> Vector:
        """Return the unit x axis."""
        return self._x_axis

    @property
    def y_axis(self) -> Vector:
        """Return the unit y axis."""
        return self._y_axis

    @property
    def z_axis(self) -> Vector:
        """Return the unit z axis."""
        return self._z_axis

    @property
    def a(self) -> float:
        """Return plane equation coefficient a."""
        return self._a

    @property
    def b(self) -> float:
        """Return plane equation coefficient b."""
        return self._b

    @property
    def c(self) -> float:
        """Return plane equation coefficient c."""
        return self._c

    @property
    def d(self) -> float:
        """Return plane equation coefficient d."""
        return self._d

    def _update_equation(self) -> None:
        """Recompute a, b, c, d from z_axis and origin."""

        self._a = self._z_axis[0]
        self._b = self._z_axis[1]
        self._c = self._z_axis[2]
        self._d = -(
            self._a * self._origin[0]
            + self._b * self._origin[1]
            + self._c * self._origin[2]
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def from_frame(
        origin: Point, x_axis: Vector, y_axis: Vector, z_axis: Vector
    ) -> "Plane":
        """Construct from a frame taken as given, no normalization."""

        plane = Plane()
        plane._origin = Point(origin[0], origin[1], origin[2])
        plane._x_axis = Vector(x_axis[0], x_axis[1], x_axis[2])
        plane._y_axis = Vector(y_axis[0], y_axis[1], y_axis[2])
        plane._z_axis = Vector(z_axis[0], z_axis[1], z_axis[2])
        plane._update_equation()

        return plane

    @staticmethod
    def from_point_normal(
        point: Point, normal: Vector, normalize: bool = True
    ) -> "Plane":
        """Construct the plane through point with normal as z axis."""

        z_axis = Vector(normal[0], normal[1], normal[2])

        if normalize:
            z_axis.normalize_self()

        x_axis = Vector()
        x_axis.perpendicular_to(z_axis)

        if normalize:
            x_axis.normalize_self()

        y_axis = z_axis.cross(x_axis)

        if normalize:
            y_axis.normalize_self()

        return Plane.from_frame(point, x_axis, y_axis, z_axis)

    @staticmethod
    def from_points(points: list[Point]) -> "Plane":
        """Construct the plane through the first three points, x axis along the first edge."""

        if len(points) < 3:
            return Plane()

        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        z_axis = v1.cross(v2)
        z_axis.normalize_self()
        x_axis = Vector(v1[0], v1[1], v1[2])
        x_axis.normalize_self()
        y_axis = z_axis.cross(x_axis)
        y_axis.normalize_self()

        return Plane.from_frame(points[0], x_axis, y_axis, z_axis)

    @staticmethod
    def from_points_pca(points: list[Point]) -> "Plane":
        """Construct the least-squares plane through points by power-iteration PCA."""

        if len(points) < 3:
            return Plane()

        n = float(len(points))
        cx = 0.0
        cy = 0.0
        cz = 0.0

        for p in points:
            cx += p[0]
            cy += p[1]
            cz += p[2]

        cx /= n
        cy /= n
        cz /= n
        cxx = 0.0
        cyy = 0.0
        czz = 0.0
        cxy = 0.0
        cxz = 0.0
        cyz = 0.0

        for p in points:
            dx = p[0] - cx
            dy = p[1] - cy
            dz = p[2] - cz
            cxx += dx * dx
            cyy += dy * dy
            czz += dz * dz
            cxy += dx * dy
            cxz += dx * dz
            cyz += dy * dz

        eigvec = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        eigval = [0.0, 0.0, 0.0]
        cov = [[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]]

        for e in range(3):
            vx = 1.0 if e == 0 else 0.0
            vy = 1.0 if e == 1 else 0.0
            vz = 1.0 if e == 2 else 0.0

            for iter in range(100):
                nx = cov[0][0] * vx + cov[0][1] * vy + cov[0][2] * vz
                ny = cov[1][0] * vx + cov[1][1] * vy + cov[1][2] * vz
                nz = cov[2][0] * vx + cov[2][1] * vy + cov[2][2] * vz
                mag = math.sqrt(nx * nx + ny * ny + nz * nz)

                if mag < 1e-15:
                    break

                vx = nx / mag
                vy = ny / mag
                vz = nz / mag

            eigvec[e][0] = vx
            eigvec[e][1] = vy
            eigvec[e][2] = vz
            eigval[e] = (
                cov[0][0] * vx * vx
                + cov[1][1] * vy * vy
                + cov[2][2] * vz * vz
                + 2.0 * cov[0][1] * vx * vy
                + 2.0 * cov[0][2] * vx * vz
                + 2.0 * cov[1][2] * vy * vz
            )

            for i in range(3):
                for j in range(3):
                    cov[i][j] -= eigval[e] * eigvec[e][i] * eigvec[e][j]

        x_axis = Vector(eigvec[0][0], eigvec[0][1], eigvec[0][2])
        y_axis = Vector(eigvec[1][0], eigvec[1][1], eigvec[1][2])
        z_axis = x_axis.cross(y_axis)
        z_axis.normalize_self()
        y_axis = z_axis.cross(x_axis)
        y_axis.normalize_self()
        x_axis.normalize_self()

        return Plane.from_frame(Point(cx, cy, cz), x_axis, y_axis, z_axis)

    @staticmethod
    def from_two_points(point1: Point, point2: Point) -> "Plane":
        """Construct the plane with x axis from point1 to point2."""

        x_axis = point2 - point1
        x_axis.normalize_self()
        z_axis = Vector()
        z_axis.perpendicular_to(x_axis)
        z_axis.normalize_self()
        y_axis = z_axis.cross(x_axis)
        y_axis.normalize_self()

        return Plane.from_frame(point1, x_axis, y_axis, z_axis)

    @staticmethod
    def invalid() -> "Plane":
        """Construct an all-zero frame that fails is_valid()."""

        return Plane.from_frame(
            Point(0.0, 0.0, 0.0),
            Vector(0.0, 0.0, 0.0),
            Vector(0.0, 0.0, 0.0),
            Vector(0.0, 0.0, 0.0),
        )

    @staticmethod
    def xy_plane() -> "Plane":
        """Construct the world XY plane."""

        plane = Plane.from_frame(
            Point(0.0, 0.0, 0.0), Vector.x_axis(), Vector.y_axis(), Vector.z_axis()
        )
        plane.name = "xy_plane"

        return plane

    @staticmethod
    def yz_plane() -> "Plane":
        """Construct the world YZ plane."""

        plane = Plane.from_frame(
            Point(0.0, 0.0, 0.0), Vector.y_axis(), Vector.z_axis(), Vector.x_axis()
        )
        plane.name = "yz_plane"

        return plane

    @staticmethod
    def xz_plane() -> "Plane":
        """Construct the world XZ plane."""

        plane = Plane.from_frame(
            Point(0.0, 0.0, 0.0),
            Vector.x_axis(),
            Vector(0.0, 0.0, -1.0),
            Vector(0.0, 1.0, 0.0),
        )
        plane.name = "xz_plane"

        return plane

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════

    def __getitem__(self, index: int) -> Vector:
        """Return the axis by index (0=x, 1=y, 2=z)."""

        if index == 0:
            return self._x_axis

        if index == 1:
            return self._y_axis

        if index == 2:
            return self._z_axis

        raise IndexError("Plane index out of range")

    def __eq__(self, other) -> bool:
        """Compare name, frame and linecolor; guid ignored."""

        if not isinstance(other, Plane):
            return False

        return (
            self.name == other.name
            and self._origin == other._origin
            and self._x_axis == other._x_axis
            and self._y_axis == other._y_axis
            and self._z_axis == other._z_axis
            and self.linecolor == other.linecolor
        )

    def __ne__(self, other) -> bool:
        """Compare name, frame and linecolor; guid ignored."""
        return not self.__eq__(other)

    def __iadd__(self, other: Vector) -> "Plane":
        """Translate in place."""
        self._origin += other
        self._update_equation()

        return self

    def __isub__(self, other: Vector) -> "Plane":
        """Translate back in place."""
        self._origin -= other
        self._update_equation()

        return self

    def __add__(self, other: Vector) -> "Plane":
        """Return a translated copy."""
        result = copy.deepcopy(self)
        result += other

        return result

    def __sub__(self, other: Vector) -> "Plane":
        """Return a copy translated back."""
        result = copy.deepcopy(self)
        result -= other

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════

    def transform(self, xform: "Xform") -> None:
        """Transform in place."""

        self._origin.transform(xform)
        self._x_axis.transform(xform)
        self._y_axis.transform(xform)
        self._z_axis.transform(xform)
        self._update_equation()

    def transformed(self, xform: "Xform") -> "Plane":
        """Return a transformed copy."""
        result = copy.deepcopy(self)
        result.transform(xform)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometry
    # ═══════════════════════════════════════════════════════════════════════════

    def is_valid(self) -> bool:
        """Return whether the frame is non-zero."""

        return (
            self._x_axis.magnitude() > 1e-14
            and self._y_axis.magnitude() > 1e-14
            and self._z_axis.magnitude() > 1e-14
        )

    def reverse(self) -> None:
        """Swap x and y and flip z in place."""

        temp = self._x_axis
        self._x_axis = self._y_axis
        self._y_axis = temp
        self._z_axis.reverse()
        self._update_equation()

    def rotate(self, angles_in_radians: float) -> None:
        """Rotate x and y around z in place."""

        cos_angle = math.cos(angles_in_radians)
        sin_angle = math.sin(angles_in_radians)
        new_x = self._x_axis * cos_angle + self._y_axis * sin_angle
        new_y = self._y_axis * cos_angle - self._x_axis * sin_angle
        self._x_axis = new_x
        self._y_axis = new_y

    def is_right_hand(self) -> bool:
        """Return whether x × y points along z."""
        return self._x_axis.cross(self._y_axis).dot(self._z_axis) > 0.999

    @staticmethod
    def is_same_direction(
        plane0: "Plane", plane1: "Plane", can_be_flipped: bool = True
    ) -> bool:
        """Return whether the normals are parallel (can_be_flipped) or exactly opposite (!can_be_flipped)."""

        parallel = plane0._z_axis.is_parallel_to(plane1._z_axis)

        if can_be_flipped:
            return parallel != 0

        return parallel == -1

    @staticmethod
    def is_same_position(plane0: "Plane", plane1: "Plane") -> bool:
        """Return whether each origin lies on the other plane."""

        dist0 = abs(
            plane0._a * plane1._origin[0]
            + plane0._b * plane1._origin[1]
            + plane0._c * plane1._origin[2]
            + plane0._d
        )
        dist1 = abs(
            plane1._a * plane0._origin[0]
            + plane1._b * plane0._origin[1]
            + plane1._c * plane0._origin[2]
            + plane1._d
        )
        tolerance = Tolerance.APPROXIMATION

        return dist0 < tolerance and dist1 < tolerance

    @staticmethod
    def is_coplanar(
        plane0: "Plane", plane1: "Plane", can_be_flipped: bool = True
    ) -> bool:
        """Return whether the planes share direction and position."""
        return Plane.is_same_direction(
            plane0, plane1, can_be_flipped
        ) and Plane.is_same_position(plane0, plane1)

    @staticmethod
    def is_coplanar_from_normals(
        origin0: Point,
        normal0: Vector,
        origin1: Point,
        normal1: Vector,
        can_be_flipped: bool = True,
        tolerance: float = -1.0,
    ) -> bool:
        """Return is_coplanar from origin and normal pairs without building planes; tolerance < 0 uses APPROXIMATION."""

        parallel = normal0.is_parallel_to(normal1)

        if parallel == 0 if can_be_flipped else parallel != -1:
            return False

        d0 = -(
            normal0[0] * origin0[0] + normal0[1] * origin0[1] + normal0[2] * origin0[2]
        )
        d1 = -(
            normal1[0] * origin1[0] + normal1[1] * origin1[1] + normal1[2] * origin1[2]
        )
        dist0 = abs(
            normal0[0] * origin1[0]
            + normal0[1] * origin1[1]
            + normal0[2] * origin1[2]
            + d0
        )
        dist1 = abs(
            normal1[0] * origin0[0]
            + normal1[1] * origin0[1]
            + normal1[2] * origin0[2]
            + d1
        )
        tol = Tolerance.APPROXIMATION if tolerance < 0.0 else tolerance

        return dist0 < tol and dist1 < tol

    def translate_by_normal(self, distance: float) -> "Plane":
        """Return a copy moved along z by distance."""

        normal = Vector(self._z_axis[0], self._z_axis[1], self._z_axis[2])
        normal.normalize_self()

        return Plane(
            self._origin + normal * distance, self._x_axis, self._y_axis, self.name
        )

    def project(self, p: Point) -> Point:
        """Return the orthogonal projection of p onto the plane."""

        dist = self._a * p[0] + self._b * p[1] + self._c * p[2] + self._d

        return Point(
            p[0] - dist * self._a, p[1] - dist * self._b, p[2] - dist * self._c
        )

    def axis_point(self) -> Point:
        """Return the plane point on the axis of the largest normal component, the other two coordinates zero."""

        n = self._z_axis
        d = -n.dot(Vector(self._origin[0], self._origin[1], self._origin[2]))
        fa = abs(n[0])
        fb = abs(n[1])
        fc = abs(n[2])

        if fa > fb and fa > fc:
            return Point(-d / n[0], 0.0, 0.0)
        if fb > fc:
            return Point(0.0, -d / n[1], 0.0)

        return Point(0.0, 0.0, -d / n[2])

    def has_on_negative_side(self, p: Point) -> bool:
        """Return whether a*p[0] + b*p[1] + c*p[2] + d < 0."""
        return self._a * p[0] + self._b * p[1] + self._c * p[2] + self._d < 0.0

    def squared_distance(self, p: Point) -> float:
        """Return the squared distance from p to the plane."""
        value = self._a * p[0] + self._b * p[1] + self._c * p[2] + self._d
        normal_sq = self._a * self._a + self._b * self._b + self._c * self._c

        return value * value / normal_sq if normal_sq > 1e-20 else value * value

    def base1(self) -> Vector:
        """Return the canonical in-plane axis from the normal alone: zero the smallest normal coordinate, negate-swap the other two."""

        nx = self._z_axis[0]
        ny = self._z_axis[1]
        nz = self._z_axis[2]
        ax = abs(nx)
        ay = abs(ny)
        az = abs(nz)

        if ax <= ay and ax <= az:
            b = Vector(0.0, -nz, ny)
        elif ay <= ax and ay <= az:
            b = Vector(-nz, 0.0, nx)
        else:
            b = Vector(-ny, nx, 0.0)

        b.normalize_self()

        return b

    def base2(self) -> Vector:
        """Return z × base1 at unit length."""
        b2 = self._z_axis.cross(self.base1())
        b2.normalize_self()

        return b2

    def to_polylines(self, scale: float = 1.0) -> list["Polyline"]:
        """Return the square outline of side scale plus the three axes as polylines."""

        from .polyline import Polyline

        s = scale * 0.5
        o = self._origin
        x = self._x_axis
        y = self._y_axis
        z = self._z_axis
        c0 = o - x * s - y * s
        c1 = o + x * s - y * s
        c2 = o + x * s + y * s
        c3 = o - x * s + y * s
        rect = Polyline([c0, c1, c2, c3, c0])
        rect.linecolor = copy.deepcopy(self.linecolor)
        origin_pt = Point(o[0], o[1], o[2])
        x_line = Polyline([origin_pt, o + x * s])
        x_line.linecolor = Color.red()
        y_line = Polyline([origin_pt, o + y * s])
        y_line.linecolor = Color.green()
        z_line = Polyline([origin_pt, o + z * s])
        z_line.linecolor = Color.blue()

        return [rect, x_line, y_line, z_line]

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        return {
            "frame": [
                self._origin[0],
                self._origin[1],
                self._origin[2],
                self._x_axis[0],
                self._x_axis[1],
                self._x_axis[2],
                self._y_axis[0],
                self._y_axis[1],
                self._y_axis[2],
                self._z_axis[0],
                self._z_axis[1],
                self._z_axis[2],
            ],
            "guid": self.guid,
            "linecolor": self.linecolor.__jsondump__(),
            "name": self.name,
            "type": "Plane",
            "width": self.width,
        }

    @classmethod
    def __jsonload__(cls, data: dict, guid: str = None, name: str = None) -> "Plane":
        """Deserialize from a JSON object."""

        from .file_encoders import file_decode_node

        frame = data["frame"]
        plane = cls.from_frame(
            Point(frame[0], frame[1], frame[2]),
            Vector(frame[3], frame[4], frame[5]),
            Vector(frame[6], frame[7], frame[8]),
            Vector(frame[9], frame[10], frame[11]),
        )
        plane.guid = guid if guid is not None else data["guid"]
        plane.name = name if name is not None else data["name"]

        if "linecolor" in data:
            plane.linecolor = file_decode_node(data["linecolor"])

        if "width" in data:
            plane.width = data["width"]

        return plane

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Plane":
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write to a JSON file."""
        with open(filepath, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "Plane":
        """Read from a JSON file."""
        with open(filepath) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""

        from .proto import plane_pb2

        proto = plane_pb2.Plane()

        if self.has_guid():
            proto.guid = self._guid

        proto.name = self.name
        proto.width = self.width

        for i in range(3):
            proto.frame.append(self._origin[i])

        for i in range(3):
            proto.frame.append(self._x_axis[i])

        for i in range(3):
            proto.frame.append(self._y_axis[i])

        for i in range(3):
            proto.frame.append(self._z_axis[i])

        proto.linecolor.name = self.linecolor.name
        proto.linecolor.r = self.linecolor.r
        proto.linecolor.g = self.linecolor.g
        proto.linecolor.b = self.linecolor.b
        proto.linecolor.a = self.linecolor.a

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Plane":
        """Deserialize from protobuf bytes."""

        from .proto import plane_pb2

        proto = plane_pb2.Plane()
        proto.ParseFromString(data)
        plane = cls()

        if len(proto.frame) >= 12:
            plane = cls.from_frame(
                Point(proto.frame[0], proto.frame[1], proto.frame[2]),
                Vector(proto.frame[3], proto.frame[4], proto.frame[5]),
                Vector(proto.frame[6], proto.frame[7], proto.frame[8]),
                Vector(proto.frame[9], proto.frame[10], proto.frame[11]),
            )
        if proto.guid:
            plane.guid = proto.guid

        plane.name = proto.name

        if proto.width > 0.0:
            plane.width = proto.width

        plane.linecolor.name = proto.linecolor.name
        plane.linecolor.r = proto.linecolor.r
        plane.linecolor.g = proto.linecolor.g
        plane.linecolor.b = proto.linecolor.b
        plane.linecolor.a = proto.linecolor.a

        return plane

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write to a protobuf file."""
        with open(filepath, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "Plane":
        """Read from a protobuf file."""
        with open(filepath, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════

    def __str__(self) -> str:
        """Return "origin\nx_axis\ny_axis\nz_axis"."""
        return f"{self._origin}\n{self._x_axis}\n{self._y_axis}\n{self._z_axis}"

    def __repr__(self) -> str:
        """Return "Plane(name, ox, oy, oz, zx, zy, zz, Color(...))"."""
        prec = Tolerance.ROUNDING

        return f"Plane({self.name}, {TOLERANCE.format_number(self._origin[0], prec)}, {TOLERANCE.format_number(self._origin[1], prec)}, {TOLERANCE.format_number(self._origin[2], prec)}, {TOLERANCE.format_number(self._z_axis[0], prec)}, {TOLERANCE.format_number(self._z_axis[1], prec)}, {TOLERANCE.format_number(self._z_axis[2], prec)}, {repr(self.linecolor)})"
