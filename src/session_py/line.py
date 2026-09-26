from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import json
import math
import uuid
from enum import Enum
from .color import Color
from .point import Point
from .tolerance import Tolerance
from .tolerance import TOLERANCE
from .vector import Vector

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import line_pb2
    from .xform import Xform


class Arrowhead(Enum):
    """Which ends of a curve carry an arrowhead."""

    NONE = "none"
    START = "start"
    END = "end"
    BOTH = "both"

    def flipped(self) -> Arrowhead:
        """Return the arrowhead with start and end swapped."""

        if self == Arrowhead.START:
            return Arrowhead.END

        if self == Arrowhead.END:
            return Arrowhead.START

        return self

    def joined(self, last: Arrowhead) -> Arrowhead:
        """Return this start head combined with the end head of last."""

        start = self in (Arrowhead.START, Arrowhead.BOTH)
        end = last in (Arrowhead.END, Arrowhead.BOTH)

        if start and end:
            return Arrowhead.BOTH

        if start:
            return Arrowhead.START

        return Arrowhead.END if end else Arrowhead.NONE

    def piece(self, first: bool, last: bool) -> Arrowhead:
        """Return the heads a split piece keeps: the start head when first, the end head when last."""

        none = Arrowhead.NONE

        return (self if first else none).joined(self if last else none)

    @classmethod
    def _missing_(cls, value) -> Arrowhead:
        """Return none for an unknown name."""
        return cls.NONE


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
        "arrowhead",
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
        self.arrowhead = Arrowhead.NONE  # Arrowhead ends.

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = Line(self._x0, self._y0, self._z0, self._x1, self._y1, self._z1)
        result.name = self.name
        result.width = self.width
        result.dash = list(self.dash)
        result.linecolor = copy.deepcopy(self.linecolor, memo)
        result.arrowhead = self.arrowhead
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
    def _fit_points_power(
        row0: Vector, row1: Vector, row2: Vector, seed: Vector
    ) -> tuple[Vector, float]:
        """Power iteration on the covariance rows from seed: the unit axis and its eigenvalue estimate."""

        axis = seed
        eigen = 0.0

        for _ in range(100):
            next = Vector(row0.dot(axis), row1.dot(axis), row2.dot(axis))
            eigen = math.sqrt(next.magnitude_squared())

            if eigen < 1e-15:
                break

            axis = next / eigen

        return (axis, eigen)

    @staticmethod
    def _fit_points_axis(points: list[Point], center: Point) -> Vector:
        """Principal direction of the points about center: power iteration from each of X, Y and Z, largest eigenvalue kept."""

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

        row0 = Vector(cxx, cxy, cxz)
        row1 = Vector(cxy, cyy, cyz)
        row2 = Vector(cxz, cyz, czz)
        first = 0

        if cyy > cxx and cyy >= czz:
            first = 1
        elif czz > cxx and czz > cyy:
            first = 2

        axis = Vector(1.0, 0.0, 0.0)
        best = -1.0

        for k in range(3):
            seed = Vector(0.0, 0.0, 0.0)
            seed[(first + k) % 3] = 1.0
            power = Line._fit_points_power(row0, row1, row2, seed)

            if power[1] > best * (1.0 + Tolerance.RELATIVE):
                axis = power[0]
                best = power[1]

        return axis

    @staticmethod
    def _fit_points_extent(
        points: list[Point], center: Point, axis: Vector, length: float
    ) -> tuple[float, float]:
        """Parameter range of the fitted line along axis: +-length / 2, or the projected extent when length <= 0."""

        if length > 0.0:
            return (-length / 2.0, length / 2.0)

        t_min = 0.0
        t_max = 0.0

        for p in points:
            t = (p - center).dot(axis)
            t_min = min(t_min, t)
            t_max = max(t_max, t)

        if t_max - t_min < 1e-10:
            return (-0.5, 0.5)

        return (t_min, t_max)

    @staticmethod
    def fit_points(points: list[Point], length: float = 0.0) -> Line:
        """Construct the least-squares line through points by power-iteration PCA; length <= 0 spans the projected extent."""

        if len(points) < 2:
            raise ValueError("At least 2 points are required for line fitting")

        center = Point.centroid(points)
        axis = Line._fit_points_axis(points, center)
        extent = Line._fit_points_extent(points, center, axis, length)

        return Line.from_points(center + axis * extent[0], center + axis * extent[1])

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
        """Compare name, coordinates to 1e-6, width, linecolor and arrowhead; guid ignored."""

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
            and self.arrowhead == other.arrowhead
        )

    def __ne__(self, other) -> bool:
        """Compare name, coordinates to 1e-6, width, linecolor and arrowhead; guid ignored."""
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
        """Return a flipped copy (end to start, arrowhead ends swapped)."""

        result = self.duplicate()
        result._x0, result._x1 = self._x1, self._x0
        result._y0, result._y1 = self._y1, self._y0
        result._z0, result._z1 = self._z1, self._z0
        result.arrowhead = self.arrowhead.flipped()

        return result

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

    @staticmethod
    def split_at_crossings(
        lines: list[Line], boundary: list[Line], tolerance: float, merge: float
    ) -> tuple[list[Line], list[int]]:
        """Split lines and boundary lines in xy at every crossing within tolerance, lines shorter than tolerance dropped, a collinear overlap kept by the boundary, else by the earlier line; split points within merge welded onto boundary ends, then boundary crossings, then the rest; dangling pieces dropped. Returns the pieces, a piece two lines share kept once, and the index of the line of each, boundary lines numbered after lines."""

        segments = []

        for i in range(len(lines) + len(boundary)):
            line = lines[i] if i < len(lines) else boundary[i - len(lines)]
            start = Point(line.start()[0], line.start()[1], 0.0)
            end = Point(line.end()[0], line.end()[1], 0.0)

            if _split_distance(start, end) >= tolerance:
                segments.append(_SplitSegment(start, end, i, i >= len(lines)))

        _split_overlaps(segments, tolerance)
        points, kept = _split_welds(_split_stops(segments, tolerance), merge)
        pairs, sources = _split_pieces(segments, kept)
        pairs, sources = _split_pruned(pairs, sources, len(points))

        pieces = []

        for k in range(len(pairs)):
            pieces.append(Line.from_points(points[pairs[k][0]], points[pairs[k][1]]))

        return pieces, sources

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

        data = {}

        if self.arrowhead != Arrowhead.NONE:
            data["arrowhead"] = self.arrowhead.value

        return data | {
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

        if "arrowhead" in data:
            line.arrowhead = Arrowhead(data["arrowhead"])

        return line

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__(), separators=(",", ":"))

    @classmethod
    def file_json_loads(cls, json_string: str) -> Line:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=4)

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
        proto.arrowhead = list(Arrowhead).index(self.arrowhead)

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

        line.arrowhead = (
            list(Arrowhead)[proto.arrowhead]
            if 0 <= proto.arrowhead < 4
            else Arrowhead.NONE
        )

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


# ═══════════════════════════════════════════════════════════════════════════
# Crossings
# ═══════════════════════════════════════════════════════════════════════════
class _SplitSegment:
    """A piece of an input line while the crossings are computed."""

    def __init__(self, start: Point, end: Point, source: int, boundary: bool):
        self.start = start  # Start at z 0.
        self.end = end  # End at z 0.
        self.source = source  # Index of the input line, boundary lines after lines.
        self.boundary = boundary  # True for a boundary line.
        self.alive = True  # False once a stronger collinear segment took it.
        self.stops = []  # Distance along it and index of each of its stops.


class _SplitStop:
    """A point every piece ends on: a segment end or a crossing."""

    def __init__(self, point: Point, order: int):
        self.point = point  # At z 0.
        self.order = order  # Weld order: 0 boundary end, 1 boundary crossing, 2 rest.


def _split_direction(start: Point, end: Point) -> Vector:
    """Unit xy direction from start to end."""
    return Vector(end[0] - start[0], end[1] - start[1], 0.0).normalized()


def _split_distance(start: Point, end: Point) -> float:
    """Distance in xy from start to end."""
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _split_is_stronger(strong: _SplitSegment, weak: _SplitSegment) -> bool:
    """True when segment strong takes a collinear overlap from weak: the boundary first, then the earlier line."""

    if strong.boundary != weak.boundary:
        return strong.boundary

    return strong.source < weak.source


def _split_parameter(segment: _SplitSegment, point: Point) -> float:
    """Distance along a segment from its start to the foot of a point."""
    return (point - segment.start).dot(_split_direction(segment.start, segment.end))


def _split_overlap(
    segments: list[_SplitSegment], strong: int, weak: int, tolerance: float
) -> None:
    """The weak segment of a collinear pair loses the stretch the strong one covers and keeps the rest as new pieces."""

    along = _split_direction(segments[strong].start, segments[strong].end)
    direction = _split_direction(segments[weak].start, segments[weak].end)

    if (
        abs(along.cross(direction)[2]) > Tolerance.ANGULAR
        or abs((segments[weak].start - segments[strong].start).cross(along)[2])
        > tolerance
    ):
        return

    length = _split_distance(segments[weak].start, segments[weak].end)
    low = max(
        0.0,
        min(
            _split_parameter(segments[weak], segments[strong].start),
            _split_parameter(segments[weak], segments[strong].end),
        ),
    )
    high = min(
        length,
        max(
            _split_parameter(segments[weak], segments[strong].start),
            _split_parameter(segments[weak], segments[strong].end),
        ),
    )

    if high - low <= tolerance:
        return

    loser = segments[weak]
    segments[weak].alive = False

    if low > tolerance:
        segments.append(
            _SplitSegment(
                loser.start, loser.start + direction * low, loser.source, loser.boundary
            )
        )

    if length - high > tolerance:
        segments.append(
            _SplitSegment(
                loser.start + direction * high, loser.end, loser.source, loser.boundary
            )
        )


def _split_overlaps(segments: list[_SplitSegment], tolerance: float) -> None:
    """Collinear overlaps resolved over every pair, the weaker segment of each giving way."""

    i = 0

    while i < len(segments):
        j = 0

        while j < len(segments):
            if (
                i != j
                and segments[i].alive
                and segments[j].alive
                and not _split_is_stronger(segments[j], segments[i])
            ):
                _split_overlap(segments, i, j, tolerance)

            j += 1

        i += 1


def _split_crossing(
    segments: list[_SplitSegment],
    first: int,
    second: int,
    tolerance: float,
    stops: list[_SplitStop],
) -> None:
    """The crossing of segments first and second as a stop on both, when they cross within tolerance."""

    along = segments[first].end - segments[first].start
    across = segments[second].end - segments[second].start
    denominator = along.cross(across)[2]

    if abs(denominator) < Tolerance.ABSOLUTE * along.magnitude() * across.magnitude():
        return

    offset = segments[second].start - segments[first].start
    on_first = offset.cross(across)[2] / denominator
    on_second = offset.cross(along)[2] / denominator

    if (
        on_first < -tolerance / along.magnitude()
        or on_first > 1.0 + tolerance / along.magnitude()
        or on_second < -tolerance / across.magnitude()
        or on_second > 1.0 + tolerance / across.magnitude()
    ):
        return

    segments[first].stops.append(
        (min(max(on_first, 0.0), 1.0) * along.magnitude(), len(stops))
    )
    segments[second].stops.append(
        (min(max(on_second, 0.0), 1.0) * across.magnitude(), len(stops))
    )
    order = 1 if segments[first].boundary or segments[second].boundary else 2
    stops.append(
        _SplitStop(segments[first].start + along * min(max(on_first, 0.0), 1.0), order)
    )


def _split_stops(segments: list[_SplitSegment], tolerance: float) -> list[_SplitStop]:
    """The stops of every live segment: its ends, then every crossing with a later one."""

    stops = []

    for segment in segments:
        if not segment.alive:
            continue

        segment.stops.append((0.0, len(stops)))
        stops.append(_SplitStop(segment.start, 0 if segment.boundary else 2))
        segment.stops.append((_split_distance(segment.start, segment.end), len(stops)))
        stops.append(_SplitStop(segment.end, 0 if segment.boundary else 2))

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments[i].alive and segments[j].alive:
                _split_crossing(segments, i, j, tolerance, stops)

    return stops


def _split_welds(
    stops: list[_SplitStop], merge: float
) -> tuple[list[Point], list[int]]:
    """Stops within merge of an earlier stop in priority order welded onto it: the kept points and the kept index of every stop."""

    points = []
    kept = [0] * len(stops)

    for order in range(3):
        for index in range(len(stops)):
            if stops[index].order != order:
                continue

            found = len(points)

            for k in range(len(points)):
                if _split_distance(points[k], stops[index].point) <= merge:
                    found = k
                    break

            if found == len(points):
                points.append(stops[index].point)

            kept[index] = found

    return points, kept


def _split_pieces(
    segments: list[_SplitSegment], canonical: list[int]
) -> tuple[list[tuple[int, int]], list[int]]:
    """Pieces of every live segment between consecutive stops as welded vertex pairs with their source, each pair once."""

    seen = set()
    pairs = []
    sources = []

    for segment in segments:
        segment.stops.sort()

        if not segment.alive:
            continue

        for k in range(len(segment.stops) - 1):
            first = canonical[segment.stops[k][1]]
            second = canonical[segment.stops[k + 1][1]]
            piece = (min(first, second), max(first, second))

            if piece[0] != piece[1] and piece not in seen:
                seen.add(piece)
                pairs.append(piece)
                sources.append(segment.source)

    return pairs, sources


def _split_pruned(
    pairs: list[tuple[int, int]], sources: list[int], vertices: int
) -> tuple[list[tuple[int, int]], list[int]]:
    """Pieces with a dangling end removed until every end is shared."""

    for _ in range(len(pairs) + 1):
        degree = [0] * vertices

        for piece in pairs:
            degree[piece[0]] += 1
            degree[piece[1]] += 1

        kept_pairs = []
        kept_sources = []

        for k in range(len(pairs)):
            if degree[pairs[k][0]] >= 2 and degree[pairs[k][1]] >= 2:
                kept_pairs.append(pairs[k])
                kept_sources.append(sources[k])

        pruned = len(kept_pairs) != len(pairs)
        pairs = kept_pairs
        sources = kept_sources

        if not pruned:
            break

    return pairs, sources
