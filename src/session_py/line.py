from __future__ import annotations
from typing import TYPE_CHECKING
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
    from .proto import line_pb2
    from .xform import Xform


class Line:
    """A 3D line segment with display width, dash pattern and color."""

    __slots__ = (
        "_guid",
        "_x0",
        "_y0",
        "_z0",
        "_x1",
        "_y1",
        "_z1",
        "name",
        "width",
        "dash",
        "linecolor",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        z0: float = 0.0,
        x1: float = 0.0,
        y1: float = 0.0,
        z1: float = 1.0,
    ):
        """Construct from start and end coordinates."""

        self._guid = None  # Lazily minted GUID.
        self._x0 = x0  # Start x.
        self._y0 = y0  # Start y.
        self._z0 = z0  # Start z.
        self._x1 = x1  # End x.
        self._y1 = y1  # End y.
        self._z1 = z1  # End z.
        self.name = "my_line"  # Line name.
        self.width = 1.0  # Display width.
        self.dash = []  # Dash pattern lengths.
        self.linecolor = Color.black()  # Display color.

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = Line(self._x0, self._y0, self._z0, self._x1, self._y1, self._z1)
        result.name = self.name
        result.width = self.width
        result.dash = list(self.dash)
        result.linecolor = copy.deepcopy(self.linecolor, memo)
        memo[id(self)] = result

        return result

    def duplicate(self) -> Line:
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
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def from_points(p1: Point, p2: Point) -> Line:
        """Construct from two points."""
        return Line(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])

    @staticmethod
    def from_point_and_vector(point: Point, vector: Vector) -> Line:
        """Construct from point to point + vector."""
        return Line.from_points(point, point + vector)

    @staticmethod
    def from_point_direction_length(
        point: Point, direction: Vector, length: float
    ) -> Line:
        """Construct from point along the normalized direction."""
        return Line.from_points(point, point + direction.normalized() * length)

    @staticmethod
    def _fit_points_axis(points: list[Point], center: Point) -> Vector:
        """Principal direction of the points about center by power iteration on their covariance."""

        cxx = 0.0
        cyy = 0.0
        czz = 0.0
        cxy = 0.0
        cxz = 0.0
        cyz = 0.0

        for p in points:
            d = p - center
            cxx += d[0] * d[0]
            cyy += d[1] * d[1]
            czz += d[2] * d[2]
            cxy += d[0] * d[1]
            cxz += d[0] * d[2]
            cyz += d[1] * d[2]

        axis = Vector(1.0, 0.0, 0.0)

        if cyy > cxx and cyy >= czz:
            axis = Vector(0.0, 1.0, 0.0)
        elif czz > cxx and czz > cyy:
            axis = Vector(0.0, 0.0, 1.0)

        for _ in range(100):
            next = Vector(
                cxx * axis[0] + cxy * axis[1] + cxz * axis[2],
                cxy * axis[0] + cyy * axis[1] + cyz * axis[2],
                cxz * axis[0] + cyz * axis[1] + czz * axis[2],
            )
            mag = math.sqrt(next.magnitude_squared())

            if mag < 1e-15:
                break

            axis = next / mag

        return axis

    @staticmethod
    def _fit_points_half(
        points: list[Point], center: Point, axis: Vector, length: float
    ) -> float:
        """Half length of the fitted line: length / 2, or the projected extent when length <= 0."""

        half = length / 2.0

        if length <= 0.0:
            t_min = 0.0
            t_max = 0.0

            for p in points:
                t = (p - center).dot(axis)
                t_min = min(t_min, t)
                t_max = max(t_max, t)

            half = max(abs(t_min), abs(t_max))

            if half < 1e-10:
                half = 0.5

        return half

    @staticmethod
    def fit_points(points: list[Point], length: float = 0.0) -> Line:
        """Construct the least-squares line through points by power-iteration PCA; length <= 0 spans the projected extent."""

        if len(points) < 2:
            raise ValueError("At least 2 points are required for line fitting")

        center = Point.centroid(points)
        axis = Line._fit_points_axis(points, center)
        half = Line._fit_points_half(points, center, axis, length)

        return Line.from_points(center - axis * half, center + axis * half)

    @staticmethod
    def with_name(
        name: str, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float
    ) -> Line:
        """Construct a named line from coordinates."""

        line = Line(x0, y0, z0, x1, y1, z1)
        line.name = name

        return line

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __getitem__(self, index: int) -> float:
        """Return the coordinate by index (0=x0, 1=y0, 2=z0, 3=x1, 4=y1, 5=z1)."""

        if index == 0:
            return self._x0

        if index == 1:
            return self._y0

        if index == 2:
            return self._z0

        if index == 3:
            return self._x1

        if index == 4:
            return self._y1

        if index == 5:
            return self._z1

        raise IndexError("Index out of bounds")

    def __setitem__(self, index: int, value: float) -> None:
        """Set the coordinate by index (0=x0, 1=y0, 2=z0, 3=x1, 4=y1, 5=z1)."""

        if index == 0:
            self._x0 = value
        elif index == 1:
            self._y0 = value
        elif index == 2:
            self._z0 = value
        elif index == 3:
            self._x1 = value
        elif index == 4:
            self._y1 = value
        elif index == 5:
            self._z1 = value
        else:
            raise IndexError("Index out of bounds")

    def __eq__(self, other) -> bool:
        """Compare name, coordinates to 1e-6, width and linecolor; guid ignored."""

        if not isinstance(other, Line):
            return False

        return (
            self.name == other.name
            and round(self._x0 * 1000000.0) == round(other._x0 * 1000000.0)
            and round(self._y0 * 1000000.0) == round(other._y0 * 1000000.0)
            and round(self._z0 * 1000000.0) == round(other._z0 * 1000000.0)
            and round(self._x1 * 1000000.0) == round(other._x1 * 1000000.0)
            and round(self._y1 * 1000000.0) == round(other._y1 * 1000000.0)
            and round(self._z1 * 1000000.0) == round(other._z1 * 1000000.0)
            and round(self.width * 1000000.0) == round(other.width * 1000000.0)
            and self.linecolor == other.linecolor
        )

    def __ne__(self, other) -> bool:
        """Compare name, coordinates to 1e-6, width and linecolor; guid ignored."""
        return not self == other

    def __iadd__(self, other: Vector) -> Line:
        """Translate in place."""

        self._x0 += other[0]
        self._y0 += other[1]
        self._z0 += other[2]
        self._x1 += other[0]
        self._y1 += other[1]
        self._z1 += other[2]

        return self

    def __isub__(self, other: Vector) -> Line:
        """Translate back in place."""

        self._x0 -= other[0]
        self._y0 -= other[1]
        self._z0 -= other[2]
        self._x1 -= other[0]
        self._y1 -= other[1]
        self._z1 -= other[2]

        return self

    def __imul__(self, factor: float) -> Line:
        """Scale both ends in place."""

        self._x0 *= factor
        self._y0 *= factor
        self._z0 *= factor
        self._x1 *= factor
        self._y1 *= factor
        self._z1 *= factor

        return self

    def __itruediv__(self, factor: float) -> Line:
        """Divide both ends in place."""

        self._x0 /= factor
        self._y0 /= factor
        self._z0 /= factor
        self._x1 /= factor
        self._y1 /= factor
        self._z1 /= factor

        return self

    def __add__(self, other: Vector) -> Line:
        """Return a translated copy."""

        result = copy.deepcopy(self)
        result += other

        return result

    def __sub__(self, other: Vector) -> Line:
        """Return a copy translated back."""

        result = copy.deepcopy(self)
        result -= other

        return result

    def __mul__(self, factor: float) -> Line:
        """Return a copy with both ends scaled."""

        result = copy.deepcopy(self)
        result *= factor

        return result

    def __truediv__(self, factor: float) -> Line:
        """Return a copy with both ends divided."""

        result = copy.deepcopy(self)
        result /= factor

        return result

    def __neg__(self) -> Line:
        """Return a flipped copy (end to start)."""
        return Line(self._x1, self._y1, self._z1, self._x0, self._y0, self._z0)

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════
    def transform(self, xform: Xform) -> None:
        """Transform in place."""

        s = self.start().transformed(xform)
        e = self.end().transformed(xform)

        self._x0 = s[0]
        self._y0 = s[1]
        self._z0 = s[2]
        self._x1 = e[0]
        self._y1 = e[1]
        self._z1 = e[2]

    def transformed(self, xform: Xform) -> Line:
        """Return a transformed copy."""

        result = copy.deepcopy(self)
        result.transform(xform)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometry
    # ═══════════════════════════════════════════════════════════════════════════
    def length(self) -> float:
        """Return the length."""
        return math.sqrt(self.squared_length())

    def squared_length(self) -> float:
        """Return the squared length."""
        return self.to_vector().magnitude_squared()

    def to_vector(self) -> Vector:
        """Return the vector from start to end."""
        return Vector(self._x1 - self._x0, self._y1 - self._y0, self._z1 - self._z0)

    def to_direction(self) -> Vector:
        """Return the unit vector from start to end."""
        return self.to_vector().normalized()

    def start(self) -> Point:
        """Return the start point."""
        return Point(self._x0, self._y0, self._z0)

    def end(self) -> Point:
        """Return the end point."""
        return Point(self._x1, self._y1, self._z1)

    def center(self) -> Point:
        """Return the midpoint."""
        return Point(
            (self._x0 + self._x1) * 0.5,
            (self._y0 + self._y1) * 0.5,
            (self._z0 + self._z1) * 0.5,
        )

    def point_at(self, t: float) -> Point:
        """Return the point at parameter t (0 = start, 1 = end)."""

        s = 1.0 - t

        return Point(
            s * self._x0 + t * self._x1,
            s * self._y0 + t * self._y1,
            s * self._z0 + t * self._z1,
        )

    def subdivide(self, n: int) -> list[Point]:
        """Return n evenly spaced points including both ends."""

        if n < 2:
            raise ValueError("n must be at least 2")

        points = []

        for i in range(n):
            points.append(self.point_at(i / (n - 1)))

        return points

    def subdivide_by_distance(self, distance: float) -> list[Point]:
        """Return points spaced approximately distance apart including both ends."""

        if distance <= 0.0:
            raise ValueError("distance must be positive")

        total = self.length()

        if total < 1e-10:
            return [self.start(), self.end()]

        n = max(2, int(total / distance + 0.5) + 1)

        return self.subdivide(n)

    def closest_point(self, point: Point, limited: bool = True) -> tuple[float, Point]:
        """Return the parameter and closest point; limited clamps t to [0, 1]."""

        d = self.to_vector()
        len_sq = d.magnitude_squared()

        if len_sq < 1e-20:
            return (0.0, self.start())

        t = (point - self.start()).dot(d) / len_sq

        if limited:
            t = max(0.0, min(1.0, t))

        return (t, self.point_at(t))

    @staticmethod
    def get_middle_line(
        line0_start: Point, line0_end: Point, line1_start: Point, line1_end: Point
    ) -> tuple[Point, Point]:
        """Compute the midpoints of the paired starts and ends."""

        output_start = Point(
            (line0_start[0] + line1_start[0]) * 0.5,
            (line0_start[1] + line1_start[1]) * 0.5,
            (line0_start[2] + line1_start[2]) * 0.5,
        )

        output_end = Point(
            (line0_end[0] + line1_end[0]) * 0.5,
            (line0_end[1] + line1_end[1]) * 0.5,
            (line0_end[2] + line1_end[2]) * 0.5,
        )

        return (output_start, output_end)

    @staticmethod
    def from_projected_points(line: Line, points: list[Point]) -> Line | None:
        """Compute the extreme sub-segment of line spanned by the projected points; None when empty."""

        from .polyline import Polyline

        result = Polyline.line_from_projected_points(line.start(), line.end(), points)

        if result is None:
            return None

        return Line.from_points(result[0], result[1])

    def overlap(self, other: Line) -> Line | None:
        """Compute the collinear overlap with other; None when none or a single point."""

        from .polyline import Polyline

        result = Polyline.line_line_overlap(
            self.start(), self.end(), other.start(), other.end()
        )

        if result is None:
            return None

        return Line.from_points(result[0], result[1])

    def overlap_average(self, other: Line) -> Line | None:
        """Compute the longer of the two midpoint pairings of overlap(other) and other.overlap(self); None when empty."""

        from .polyline import Polyline

        output_start, output_end = Polyline.line_line_overlap_average(
            self.start(), self.end(), other.start(), other.end()
        )

        out = Line.from_points(output_start, output_end)

        if out.squared_length() <= 0.0:
            return None

        return out

    def extend(self, ext_start: float, ext_end: float) -> None:
        """Grow start by ext_start and end by ext_end."""

        from .polyline import Polyline

        s = self.start()
        e = self.end()
        Polyline.extend_line_segment(s, e, ext_start, ext_end)

        self._x0 = s[0]
        self._y0 = s[1]
        self._z0 = s[2]
        self._x1 = e[0]
        self._y1 = e[1]
        self._z1 = e[2]

    def extend_equally(self, dist: float = 0.0, proportion: float = 0.0) -> None:
        """Grow both ends by dist, or by proportion of the length when non-zero."""

        from .polyline import Polyline

        if dist == 0.0 and proportion == 0.0:
            return

        s = self.start()
        e = self.end()
        Polyline.extend_segment_equally_static(s, e, dist, proportion)

        self._x0 = s[0]
        self._y0 = s[1]
        self._z0 = s[2]
        self._x1 = e[0]
        self._y1 = e[1]
        self._z1 = e[2]

    def scale(self, dist: float) -> None:
        """Shrink both ends by dist as a fraction of the length."""

        from .polyline import Polyline

        s = self.start()
        e = self.end()
        Polyline.shrink_line_segment(s, e, dist)

        self._x0 = s[0]
        self._y0 = s[1]
        self._z0 = s[2]
        self._x1 = e[0]
        self._y1 = e[1]
        self._z1 = e[2]

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        return {
            "dash": list(self.dash),
            "guid": self.guid,
            "linecolor": self.linecolor.__jsondump__(),
            "name": self.name,
            "type": "Line",
            "width": self.width,
            "x0": self._x0,
            "x1": self._x1,
            "y0": self._y0,
            "y1": self._y1,
            "z0": self._z0,
            "z1": self._z1,
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Line:
        """Deserialize from a JSON object."""

        from .file_encoders import file_decode_node

        line = cls(
            data["x0"], data["y0"], data["z0"], data["x1"], data["y1"], data["z1"]
        )
        line.guid = guid if guid is not None else data["guid"]
        line.name = name if name is not None else data["name"]

        if "dash" in data:
            line.dash = list(data["dash"])

        if "linecolor" in data:
            line.linecolor = file_decode_node(data["linecolor"])

        if "width" in data:
            line.width = data["width"]

        return line

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> Line:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Line:
        """Read from a JSON file."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> line_pb2.Line:
        """Convert to the protobuf message."""

        from .proto import line_pb2

        proto = line_pb2.Line()

        if self.has_guid():
            proto.guid = self.guid

        proto.name = self.name
        proto.width = self.width

        for i in range(6):
            proto.coords.append(self[i])

        for d in self.dash:
            proto.dash.append(d)

        proto.linecolor_rgba.append(self.linecolor.r)
        proto.linecolor_rgba.append(self.linecolor.g)
        proto.linecolor_rgba.append(self.linecolor.b)
        proto.linecolor_rgba.append(self.linecolor.a)
        proto.linecolor_name = self.linecolor.name

        return proto

    @classmethod
    def from_proto(cls, proto: line_pb2.Line) -> Line:
        """Construct from the protobuf message."""

        line = cls()

        if len(proto.coords) == 6:
            line = cls(
                proto.coords[0],
                proto.coords[1],
                proto.coords[2],
                proto.coords[3],
                proto.coords[4],
                proto.coords[5],
            )

        if proto.guid:
            line.guid = proto.guid

        line.name = proto.name

        if proto.width > 0.0:
            line.width = proto.width

        line.dash = list(proto.dash)

        if len(proto.linecolor_rgba) == 4:
            line.linecolor.r = proto.linecolor_rgba[0]
            line.linecolor.g = proto.linecolor_rgba[1]
            line.linecolor.b = proto.linecolor_rgba[2]
            line.linecolor.a = proto.linecolor_rgba[3]

            if proto.linecolor_name:
                line.linecolor.name = proto.linecolor_name

        return line

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Line:
        """Deserialize from protobuf bytes."""

        from .proto import line_pb2

        proto = line_pb2.Line()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write to a protobuf file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Line:
        """Read from a protobuf file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "x0, y0, z0, x1, y1, z1"."""

        prec = Tolerance.ROUNDING

        return f"{TOLERANCE.format_number(self._x0, prec)}, {TOLERANCE.format_number(self._y0, prec)}, {TOLERANCE.format_number(self._z0, prec)}, {TOLERANCE.format_number(self._x1, prec)}, {TOLERANCE.format_number(self._y1, prec)}, {TOLERANCE.format_number(self._z1, prec)}"

    def __repr__(self) -> str:
        """Return "Line(name, x0, y0, z0, x1, y1, z1, Color(...), width)"."""

        prec = Tolerance.ROUNDING

        return f"Line({self.name}, {TOLERANCE.format_number(self._x0, prec)}, {TOLERANCE.format_number(self._y0, prec)}, {TOLERANCE.format_number(self._z0, prec)}, {TOLERANCE.format_number(self._x1, prec)}, {TOLERANCE.format_number(self._y1, prec)}, {TOLERANCE.format_number(self._z1, prec)}, {repr(self.linecolor)}, {TOLERANCE.format_number(self.width, prec)})"
