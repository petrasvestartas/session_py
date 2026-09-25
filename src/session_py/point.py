from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import json
import math
import uuid
from .color import Color
from .tolerance import Tolerance
from .tolerance import TOLERANCE
from .vector import Vector

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import point_pb2
    from .xform import Xform


class Point:
    """A 3D point with display width and color."""

    __slots__ = ("_guid", "_x", "_y", "_z", "name", "width", "pointcolor")

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, name: str = "my_point"):
        """Construct from coordinates and a name."""

        self._guid = None  # Lazily minted GUID.
        self._x = x  # X coordinate.
        self._y = y  # Y coordinate.
        self._z = z  # Z coordinate.
        self.name = name  # Point name.
        self.width = 1.0  # Display width.
        self.pointcolor = Color.black()  # Display color.

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = Point(self._x, self._y, self._z, self.name)
        result.width = self.width
        result.pointcolor = copy.deepcopy(self.pointcolor, memo)
        memo[id(self)] = result

        return result

    def duplicate(self) -> Point:
        """Copy with a new guid and the same data."""
        return copy.deepcopy(self)

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

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __getitem__(self, index: int) -> float:
        """Return the coordinate by index (0=x, 1=y, 2=z)."""

        if index == 0:
            return self._x

        if index == 1:
            return self._y

        if index == 2:
            return self._z

        raise IndexError("Index out of range")

    def __setitem__(self, index: int, value: float) -> None:
        """Set the coordinate by index (0=x, 1=y, 2=z)."""

        if index == 0:
            self._x = value
        elif index == 1:
            self._y = value
        elif index == 2:
            self._z = value
        else:
            raise IndexError("Index out of range")

    def __eq__(self, other) -> bool:
        """Compare name, coordinates, width and color within rounding."""

        if not isinstance(other, Point):
            return False

        return (
            self.name == other.name
            and round(self._x * 1000000.0) == round(other._x * 1000000.0)
            and round(self._y * 1000000.0) == round(other._y * 1000000.0)
            and round(self._z * 1000000.0) == round(other._z * 1000000.0)
            and round(self.width * 1000000.0) == round(other.width * 1000000.0)
            and self.pointcolor == other.pointcolor
        )

    def __ne__(self, other) -> bool:
        """Compare name, coordinates, width and color within rounding."""
        return not self == other

    def __imul__(self, factor: float) -> Point:
        """Scale in place."""

        self._x *= factor
        self._y *= factor
        self._z *= factor

        return self

    def __itruediv__(self, factor: float) -> Point:
        """Divide in place."""

        self._x /= factor
        self._y /= factor
        self._z /= factor

        return self

    def __iadd__(self, other: Vector) -> Point:
        """Translate in place."""

        self._x += other[0]
        self._y += other[1]
        self._z += other[2]

        return self

    def __isub__(self, other: Vector) -> Point:
        """Translate back in place."""

        self._x -= other[0]
        self._y -= other[1]
        self._z -= other[2]

        return self

    def __mul__(self, factor: float) -> Point:
        """Return a scaled copy."""
        return Point(self._x * factor, self._y * factor, self._z * factor)

    def __truediv__(self, factor: float) -> Point:
        """Return a divided copy."""
        return Point(self._x / factor, self._y / factor, self._z / factor)

    def __add__(self, other: Vector) -> Point:
        """Return a translated copy."""
        return Point(self._x + other[0], self._y + other[1], self._z + other[2])

    def __sub__(self, other: Point | Vector) -> Point | Vector:
        """Return a copy translated back by a Vector, or the vector from another Point."""

        if isinstance(other, Point):
            return Vector(self._x - other._x, self._y - other._y, self._z - other._z)

        return Point(self._x - other[0], self._y - other[1], self._z - other[2])

    @staticmethod
    def sum(p0: Point, p1: Point) -> Point:
        """Return the coordinate-wise sum of two points."""
        return Point(p0[0] + p1[0], p0[1] + p1[1], p0[2] + p1[2])

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════
    def transform(self, xform: Xform) -> None:
        """Transform in place."""

        x = self._x
        y = self._y
        z = self._z
        m = xform.m
        w = m[3] * x + m[7] * y + m[11] * z + m[15]
        w_inv = 1.0 / w if abs(w) > 1e-10 else 1.0

        self._x = (m[0] * x + m[4] * y + m[8] * z + m[12]) * w_inv
        self._y = (m[1] * x + m[5] * y + m[9] * z + m[13]) * w_inv
        self._z = (m[2] * x + m[6] * y + m[10] * z + m[14]) * w_inv

    def transformed(self, xform: Xform) -> Point:
        """Return a transformed copy."""

        result = self.duplicate()
        result.transform(xform)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometry
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def is_ccw(a: Point, b: Point, c: Point) -> bool:
        """Return whether a, b, c turn counter-clockwise in the xy plane."""
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def mid_point(self, p: Point) -> Point:
        """Return the mid point between this point and p."""
        return Point((self._x + p[0]) / 2.0, (self._y + p[1]) / 2.0, (self._z + p[2]) / 2.0)

    def distance(self, p: Point, double_min: float = 1e-12) -> float:
        """Return the distance to p, scaled to stay finite for large coordinates."""

        dx = abs(self._x - p[0])
        dy = abs(self._y - p[1])
        dz = abs(self._z - p[2])

        if dy >= dx and dy >= dz:
            dx, dy = dy, dx
        elif dz >= dx and dz >= dy:
            dx, dz = dz, dx

        if dx > double_min:
            dy /= dx
            dz /= dx

            return dx * math.sqrt(1.0 + dy * dy + dz * dz)

        if dx > 0.0 and math.isfinite(dx):
            return dx

        return 0.0

    def squared_distance(self, p: Point, double_min: float = 1e-12) -> float:
        """Return the squared distance to p, scaled to stay finite for large coordinates."""

        dx = abs(self._x - p[0])
        dy = abs(self._y - p[1])
        dz = abs(self._z - p[2])

        if dy >= dx and dy >= dz:
            dx, dy = dy, dx
        elif dz >= dx and dz >= dy:
            dx, dz = dz, dx

        if dx > double_min:
            dy /= dx
            dz /= dx

            return dx * dx * (1.0 + dy * dy + dz * dz)

        if dx > 0.0 and math.isfinite(dx):
            return dx * dx

        return 0.0

    @staticmethod
    def lerp(a: Point, b: Point, t: float) -> Point:
        """Return the point at parameter t in [0, 1] between a and b."""
        return a + (b - a) * t

    @staticmethod
    def interpolate(from_pt: Point, to_pt: Point, steps: int, kind: int = 0) -> list[Point]:
        """Return evenly spaced points between from and to (kind: 0=no endpoints, 1=both, 2=start only)."""

        points = []

        if kind == 1 or kind == 2:
            points.append(from_pt.duplicate())

        for i in range(1, steps + 1):
            points.append(Point.lerp(from_pt, to_pt, float(i) / float(steps + 1)))

        if kind == 1:
            points.append(to_pt.duplicate())

        return points

    @staticmethod
    def area(points: list[Point]) -> float:
        """Return the shoelace area of a polygon in the xy plane."""

        n = len(points)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        return abs(area) / 2.0

    @staticmethod
    def centroid_quad(vertices: list[Point]) -> Point:
        """Return the area-weighted centroid of a quadrilateral."""

        if len(vertices) != 4:
            raise ValueError("Polygon must have exactly 4 vertices.")

        total_area = 0.0
        centroid_sum = Vector(0.0, 0.0, 0.0)

        for i in range(4):
            p0 = vertices[i]
            p1 = vertices[(i + 1) % 4]
            p2 = vertices[(i + 2) % 4]
            tri_area = abs(p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1])) / 2.0
            tri_centroid = Vector((p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0, (p0[2] + p1[2] + p2[2]) / 3.0)

            total_area += tri_area
            centroid_sum += tri_centroid * tri_area

        result = centroid_sum / total_area

        return Point(result[0], result[1], result[2])

    @staticmethod
    def centroid(points: list[Point]) -> Point:
        """Return the arithmetic mean of points; empty input returns the origin."""

        if not points:
            return Point(0.0, 0.0, 0.0)

        cx = 0.0
        cy = 0.0
        cz = 0.0

        for p in points:
            cx += p[0]
            cy += p[1]
            cz += p[2]

        n = float(len(points))

        return Point(cx / n, cy / n, cz / n)

    @staticmethod
    def dihedral_angle_deg(p: Point, q: Point, r: Point, s: Point) -> float:
        """Return the unsigned dihedral angle in degrees of edge pq between half-planes pqr and pqs."""

        pq = q - p
        pr = r - p
        ps = s - p
        n1 = pq.cross(pr)
        n2 = pq.cross(ps)
        m1 = n1.magnitude()
        m2 = n2.magnitude()

        if m1 < Tolerance.ZERO_TOLERANCE or m2 < Tolerance.ZERO_TOLERANCE:
            return 0.0

        cos_t = max(-1.0, min(1.0, n1.dot(n2) / (m1 * m2)))

        return math.acos(cos_t) * (180.0 / 3.141592653589793)

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to an ordered JSON object."""

        return {
            "guid": self.guid,
            "name": self.name,
            "pointcolor": self.pointcolor.__jsondump__(),
            "type": "Point",
            "width": self.width,
            "x": self._x,
            "y": self._y,
            "z": self._z,
        }

    @classmethod
    def __jsonload__(cls, data: dict, guid: str | None = None, name: str | None = None) -> Point:
        """Deserialize from a JSON object."""

        from .file_encoders import file_decode_node

        point = cls(data["x"], data["y"], data["z"])
        point.guid = guid or data["guid"]
        point.name = name or data["name"]
        point.pointcolor = file_decode_node(data["pointcolor"])
        point.width = data["width"]

        return point

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> Point:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write JSON to a file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Point:
        """Read JSON from a file."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> point_pb2.Point:
        """Convert to the protobuf message."""

        from .proto import point_pb2

        proto = point_pb2.Point()

        if self.has_guid():
            proto.guid = self.guid

        proto.name = self.name
        proto.x = self._x
        proto.y = self._y
        proto.z = self._z
        proto.width = self.width
        proto.pointcolor.CopyFrom(self.pointcolor.to_proto())

        return proto

    @classmethod
    def from_proto(cls, proto: point_pb2.Point) -> Point:
        """Construct from the protobuf message."""

        point = cls(proto.x, proto.y, proto.z)

        if proto.guid:
            point.guid = proto.guid

        point.name = proto.name
        point.width = proto.width
        point.pointcolor = Color.from_proto(proto.pointcolor)

        return point

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Point:
        """Deserialize from protobuf bytes."""

        from .proto import point_pb2

        proto = point_pb2.Point()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write protobuf bytes to a file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Point:
        """Read protobuf bytes from a file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "x, y, z"."""

        prec = Tolerance.ROUNDING

        return f"{TOLERANCE.format_number(self._x, prec)}, {TOLERANCE.format_number(self._y, prec)}, {TOLERANCE.format_number(self._z, prec)}"

    def __repr__(self) -> str:
        """Return "Point(name, x, y, z, Color(...), width)"."""

        prec = Tolerance.ROUNDING

        return f"Point({self.name}, {TOLERANCE.format_number(self._x, prec)}, {TOLERANCE.format_number(self._y, prec)}, {TOLERANCE.format_number(self._z, prec)}, {repr(self.pointcolor)}, {TOLERANCE.format_number(self.width, prec)})"
