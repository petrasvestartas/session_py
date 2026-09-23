from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import json
import math
import sys
import uuid
from .tolerance import Tolerance
from .tolerance import TOLERANCE
from .tolerance import SCALE
from .tolerance import TO_DEGREES
from .tolerance import TO_RADIANS

if TYPE_CHECKING:
    from pathlib import Path
    from .point import Point
    from .polyline import Polyline
    from .proto import vector_pb2
    from .xform import Xform


class Vector:
    """A 3D vector with a cached magnitude."""

    __slots__ = ("_guid", "_x", "_y", "_z", "_magnitude", "_has_magnitude", "name")

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Construct from components."""

        self._guid = None  # Lazily minted GUID.
        self._x = x  # X component.
        self._y = y  # Y component.
        self._z = z  # Z component.
        self._magnitude = 0.0  # Cached magnitude.
        self._has_magnitude = False  # Whether the cached magnitude is valid.
        self.name = "my_vector"  # Vector name.

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = Vector(self._x, self._y, self._z)
        result._magnitude = self._magnitude
        result._has_magnitude = self._has_magnitude
        result.name = self.name
        memo[id(self)] = result

        return result

    def duplicate(self) -> Vector:
        """Copy with a new guid and the same data."""
        return copy.deepcopy(self)

    @staticmethod
    def zero() -> Vector:
        """Construct the zero vector."""
        return Vector(0.0, 0.0, 0.0)

    @staticmethod
    def x_axis() -> Vector:
        """Construct the unit vector along x."""
        return Vector(1.0, 0.0, 0.0)

    @staticmethod
    def y_axis() -> Vector:
        """Construct the unit vector along y."""
        return Vector(0.0, 1.0, 0.0)

    @staticmethod
    def z_axis() -> Vector:
        """Construct the unit vector along z."""
        return Vector(0.0, 0.0, 1.0)

    @staticmethod
    def from_points(p0: Point, p1: Point) -> Vector:
        """Construct the vector from p0 to p1."""
        return p1 - p0

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
    def __setitem__(self, index: int, value: float) -> None:
        """Set the component by index (0=x, 1=y, 2=z), dropping the cached magnitude."""

        self._has_magnitude = False

        if index == 0:
            self._x = value
        elif index == 1:
            self._y = value
        elif index == 2:
            self._z = value
        else:
            raise IndexError("Index out of range")

    def __getitem__(self, index: int) -> float:
        """Return the component by index (0=x, 1=y, 2=z)."""

        if index == 0:
            return self._x

        if index == 1:
            return self._y

        if index == 2:
            return self._z

        raise IndexError("Index out of range")

    def __eq__(self, other) -> bool:
        """Compare components within rounding."""

        if not isinstance(other, Vector):
            return False

        return (
            self.name == other.name
            and round(self._x * 1000000.0) == round(other._x * 1000000.0)
            and round(self._y * 1000000.0) == round(other._y * 1000000.0)
            and round(self._z * 1000000.0) == round(other._z * 1000000.0)
        )

    def __ne__(self, other) -> bool:
        """Compare components within rounding."""
        return not self == other

    def __imul__(self, factor: float) -> Vector:
        """Scale in place."""

        self._x *= factor
        self._y *= factor
        self._z *= factor
        self._has_magnitude = False

        return self

    def __itruediv__(self, factor: float) -> Vector:
        """Divide in place."""

        self._x /= factor
        self._y /= factor
        self._z /= factor
        self._has_magnitude = False

        return self

    def __iadd__(self, other: Vector) -> Vector:
        """Add in place."""

        self._x += other[0]
        self._y += other[1]
        self._z += other[2]
        self._has_magnitude = False

        return self

    def __isub__(self, other: Vector) -> Vector:
        """Subtract in place."""

        self._x -= other[0]
        self._y -= other[1]
        self._z -= other[2]
        self._has_magnitude = False

        return self

    def __mul__(self, factor: float) -> Vector:
        """Return a scaled copy."""
        return Vector(self._x * factor, self._y * factor, self._z * factor)

    def __truediv__(self, factor: float) -> Vector:
        """Return a divided copy."""
        return Vector(self._x / factor, self._y / factor, self._z / factor)

    def __add__(self, other: Vector) -> Vector:
        """Return the sum."""
        return Vector(self._x + other[0], self._y + other[1], self._z + other[2])

    def __sub__(self, other: Vector) -> Vector:
        """Return the difference."""
        return Vector(self._x - other[0], self._y - other[1], self._z - other[2])

    def __neg__(self) -> Vector:
        """Return the negation."""
        return Vector(-self._x, -self._y, -self._z)

    def __rmul__(self, factor: float) -> Vector:
        """Return a vector scaled by a factor on the left."""
        return self * factor

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════
    def transform(self, xform: Xform) -> None:
        """Transform in place; only rotation and scale apply, a vector has no position."""

        x = self._x
        y = self._y
        z = self._z
        m = xform.m
        self._x = m[0] * x + m[4] * y + m[8] * z
        self._y = m[1] * x + m[5] * y + m[9] * z
        self._z = m[2] * x + m[6] * y + m[10] * z
        self._has_magnitude = False

    def transformed(self, xform: Xform) -> Vector:
        """Return a transformed copy."""

        result = self.duplicate()
        result.transform(xform)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometry
    # ═══════════════════════════════════════════════════════════════════════════
    def reverse(self) -> None:
        """Negate every component in place."""

        self._x = -self._x
        self._y = -self._y
        self._z = -self._z

    def _compute_magnitude(self) -> float:
        """Return the magnitude scaled to stay finite for large components."""

        ax = abs(self._x)
        ay = abs(self._y)
        az = abs(self._z)
        x_zero = ax < Tolerance.ZERO_TOLERANCE
        y_zero = ay < Tolerance.ZERO_TOLERANCE
        z_zero = az < Tolerance.ZERO_TOLERANCE

        if x_zero and y_zero and z_zero:
            return 0.0

        if x_zero and y_zero:
            return az

        if x_zero and z_zero:
            return ay

        if y_zero and z_zero:
            return ax

        if ay >= ax and ay >= az:
            ax, ay = ay, ax
        elif az >= ax and az >= ay:
            ax, az = az, ax

        if ax > sys.float_info.min:
            ay /= ax
            az /= ax

            return ax * math.sqrt(1.0 + ay * ay + az * az)

        if ax > 0.0 and math.isfinite(ax):
            return ax

        return 0.0

    def magnitude(self) -> float:
        """Return the cached magnitude."""

        if not self._has_magnitude:
            self._magnitude = self._compute_magnitude()
            self._has_magnitude = True

        return self._magnitude

    def magnitude_squared(self) -> float:
        """Return the squared magnitude without the square root."""
        return self._x * self._x + self._y * self._y + self._z * self._z

    def normalize_self(self) -> bool:
        """Set unit length in place; false when the magnitude is zero."""

        d = self._compute_magnitude()

        if d <= 0.0:
            return False

        self._x /= d
        self._y /= d
        self._z /= d
        self._magnitude = 1.0
        self._has_magnitude = True

        return True

    def normalized(self) -> Vector:
        """Return a unit length copy; zero when the magnitude is zero."""

        result = Vector(self._x, self._y, self._z)

        if not result.normalize_self():
            return Vector.zero()

        return result

    def dot(self, other: Vector) -> float:
        """Return the dot product."""
        return self._x * other[0] + self._y * other[1] + self._z * other[2]

    def cross(self, other: Vector) -> Vector:
        """Return the cross product."""
        return Vector(
            self._y * other[2] - self._z * other[1],
            self._z * other[0] - self._x * other[2],
            self._x * other[1] - self._y * other[0],
        )

    def angle(
        self,
        other: Vector,
        sign_by_cross_product: bool = True,
        degrees: bool = True,
        tolerance: float = Tolerance.ZERO_TOLERANCE,
    ) -> float:
        """Return the angle to other, negated when the cross product points down z; zero below tolerance."""

        denominator = self.magnitude() * other.magnitude()

        if denominator < tolerance:
            return 0.0

        cos_angle = max(-1.0, min(1.0, self.dot(other) / denominator))
        angle = math.acos(cos_angle)

        if sign_by_cross_product and self.cross(other)[2] < 0.0:
            angle = -angle

        return angle * TO_DEGREES if degrees else angle

    def projection(
        self, projection_vector: Vector, tolerance: float = Tolerance.ZERO_TOLERANCE
    ) -> tuple[Vector, float, Vector, float]:
        """Return the projection onto projection_vector: (projection, projected length, perpendicular, perpendicular length)."""

        projection_vector_length = projection_vector.magnitude()

        if projection_vector_length < tolerance:
            return Vector(0.0, 0.0, 0.0), 0.0, Vector(0.0, 0.0, 0.0), 0.0

        projection_vector_unit = projection_vector / projection_vector_length
        projected_length = self.dot(projection_vector_unit)
        projected = projection_vector_unit * projected_length
        perpendicular = self - projected
        perpendicular_length = perpendicular.magnitude()

        return projected, projected_length, perpendicular, perpendicular_length

    def is_parallel_to(self, other: Vector) -> int:
        """Return 1 when parallel, -1 when antiparallel, 0 otherwise."""

        cos_tolerance = math.cos(Tolerance.ANGLE_TOLERANCE_DEGREES * TO_RADIANS)
        denominator = self.magnitude() * other.magnitude()

        if denominator <= 0.0:
            return 0

        cos_angle = self.dot(other) / denominator

        if cos_angle >= cos_tolerance:
            return 1

        if cos_angle <= -cos_tolerance:
            return -1

        return 0

    def is_perpendicular_to(self, other: Vector) -> bool:
        """Return whether the dot product is within tolerance of zero."""
        return abs(self.dot(other)) < Tolerance.ZERO_TOLERANCE

    def perpendicular_to(self, v: Vector) -> bool:
        """Set this vector perpendicular to v; false when v is zero."""

        i = 0
        j = 1
        k = 2
        a = v[0]
        b = -v[1]

        if abs(v[1]) > abs(v[0]):
            if abs(v[2]) > abs(v[1]):
                i = 2
                j = 1
                k = 0
                a = v[2]
                b = -v[1]
            elif abs(v[2]) >= abs(v[0]):
                i = 1
                j = 2
                k = 0
                a = v[1]
                b = -v[2]
            else:
                i = 1
                j = 0
                k = 2
                a = v[1]
                b = -v[0]
        elif abs(v[2]) > abs(v[0]):
            i = 2
            j = 0
            k = 1
            a = v[2]
            b = -v[0]
        elif abs(v[2]) > abs(v[1]):
            i = 0
            j = 2
            k = 1
            a = v[0]
            b = -v[2]

        coords = [0.0, 0.0, 0.0]
        coords[i] = b
        coords[j] = a
        coords[k] = 0.0
        self._x = coords[0]
        self._y = coords[1]
        self._z = coords[2]
        self._has_magnitude = False

        return a != 0.0

    def is_zero(self) -> bool:
        """Return whether the magnitude is within tolerance of zero."""
        return self._compute_magnitude() < Tolerance.ZERO_TOLERANCE

    def get_leveled_vector(self, vertical_height: float) -> Vector:
        """Return a copy scaled along its direction so its rise along z equals vertical_height."""

        copy = Vector(self._x, self._y, self._z)

        if copy.normalize_self():
            angle_rad = copy.angle(Vector.z_axis(), False) * TO_RADIANS
            copy *= vertical_height / math.cos(angle_rad)

        return copy

    def coordinate_direction_3angles(
        self, degrees: bool = False
    ) -> tuple[float, float, float]:
        """Return the angles to the x, y and z axes."""

        r = math.sqrt(self._x * self._x + self._y * self._y + self._z * self._z)

        if r == 0.0:
            return (0.0, 0.0, 0.0)

        alpha = math.acos(self._x / r)
        beta = math.acos(self._y / r)
        gamma = math.acos(self._z / r)

        if degrees:
            return (alpha * TO_DEGREES, beta * TO_DEGREES, gamma * TO_DEGREES)

        return (alpha, beta, gamma)

    def coordinate_direction_2angles(
        self, degrees: bool = False
    ) -> tuple[float, float]:
        """Return the polar angle from z and the azimuth from x."""

        r = math.sqrt(self._x * self._x + self._y * self._y + self._z * self._z)

        if r == 0.0:
            return (0.0, 0.0)

        phi = math.acos(self._z / r)
        theta = math.atan2(self._y, self._x)

        if degrees:
            return (phi * TO_DEGREES, theta * TO_DEGREES)

        return (phi, theta)

    @staticmethod
    def angle_between_vector_xy_components(vector: Vector) -> float:
        """Return the angle in degrees of the xy projection from the x-axis."""
        return math.atan2(vector[1], vector[0]) * TO_DEGREES

    @staticmethod
    def sum_of_vectors(vectors: list[Vector]) -> Vector:
        """Return the component-wise sum."""

        sum = Vector(0.0, 0.0, 0.0)

        for vector in vectors:
            sum += vector

        return sum

    @staticmethod
    def average(vectors: list[Vector]) -> Vector:
        """Return the component-wise average; empty input returns zero."""

        if not vectors:
            return Vector.zero()

        return Vector.sum_of_vectors(vectors) / float(len(vectors))

    def scale_up(self) -> None:
        """Scale in place by SCALE."""
        self *= SCALE

    def scale_down(self) -> None:
        """Scale in place by 1 / SCALE."""
        self *= 1.0 / SCALE

    def reflect(self, plane_normal: Vector) -> Vector:
        """Return the reflection through the plane with the given unit normal."""

        d = self.dot(plane_normal)

        return self - plane_normal * (2.0 * d)

    @staticmethod
    def average_normal(points: list[Point]) -> Vector:
        """Return the unit area-weighted normal of a polygon by Newell's method."""

        if not points:
            return Vector.zero()

        closed = (points[-1] - points[0]).magnitude_squared() < 1e-10
        n = len(points) - 1 if closed else len(points)
        normal = Vector(0.0, 0.0, 0.0)

        for i in range(n):
            prev = (i + n - 1) % n
            next = (i + 1) % n
            a = points[i] - points[prev]
            b = points[next] - points[i]
            normal += a.cross(b)

        if not normal.normalize_self():
            return Vector.zero()

        return normal

    @staticmethod
    def average_normal_polyline(polyline: Polyline) -> Vector:
        """Return the unit area-weighted normal of a closed polyline by Newell's method."""
        return Vector.average_normal(polyline.get_points())

    # ═══════════════════════════════════════════════════════════════════════════
    # Triangle laws
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def cosine_law(
        triangle_edge_length_a: float,
        triangle_edge_length_b: float,
        angle_in_between_edges: float,
        degrees: bool = True,
    ) -> float:
        """Return the third side from two sides and the angle between them."""

        to_radians = TO_RADIANS if degrees else 1.0

        return math.sqrt(
            triangle_edge_length_a * triangle_edge_length_a
            + triangle_edge_length_b * triangle_edge_length_b
            - 2.0
            * triangle_edge_length_a
            * triangle_edge_length_b
            * math.cos(angle_in_between_edges * to_radians)
        )

    @staticmethod
    def sine_law_angle(
        triangle_edge_length_a: float,
        angle_in_front_of_a: float,
        triangle_edge_length_b: float,
        degrees: bool = True,
    ) -> float:
        """Return the angle opposite side b from side a, the angle opposite a and side b."""

        to_radians = TO_RADIANS if degrees else 1.0
        to_degrees = TO_DEGREES if degrees else 1.0

        return (
            math.asin(
                triangle_edge_length_b
                * math.sin(angle_in_front_of_a * to_radians)
                / triangle_edge_length_a
            )
            * to_degrees
        )

    @staticmethod
    def sine_law_length(
        triangle_edge_length_a: float,
        angle_in_front_of_a: float,
        angle_in_front_of_b: float,
        degrees: bool = True,
    ) -> float:
        """Return side b from side a and the angles opposite a and b."""

        to_radians = TO_RADIANS if degrees else 1.0

        return (
            triangle_edge_length_a
            * math.sin(angle_in_front_of_b * to_radians)
            / math.sin(angle_in_front_of_a * to_radians)
        )

    @staticmethod
    def angle_from_cosine_law(
        triangle_edge_length_a: float,
        triangle_edge_length_b: float,
        triangle_edge_length_c: float,
        degrees: bool = True,
    ) -> float:
        """Return the angle opposite side c from the three sides."""

        cos_c = (
            triangle_edge_length_a * triangle_edge_length_a
            + triangle_edge_length_b * triangle_edge_length_b
            - triangle_edge_length_c * triangle_edge_length_c
        ) / (2.0 * triangle_edge_length_a * triangle_edge_length_b)
        angle_rad = math.acos(cos_c)

        return angle_rad * TO_DEGREES if degrees else angle_rad

    @staticmethod
    def side_from_sine_law(
        angle_in_front_of_result_side: float,
        angle_in_front_of_known_side: float,
        known_side_length: float,
        degrees: bool = True,
    ) -> float:
        """Return the side opposite the first angle from two angles and the side opposite the second."""

        to_radians = TO_RADIANS if degrees else 1.0

        return (
            known_side_length
            * math.sin(angle_in_front_of_result_side * to_radians)
            / math.sin(angle_in_front_of_known_side * to_radians)
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        return {
            "guid": self.guid,
            "name": self.name,
            "type": "Vector",
            "x": self._x,
            "y": self._y,
            "z": self._z,
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Vector:
        """Deserialize from a JSON object."""

        vector = cls(data["x"], data["y"], data["z"])
        vector.guid = guid if guid is not None else data["guid"]
        vector.name = name if name is not None else data["name"]

        return vector

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> Vector:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: str | Path) -> None:
        """Write to a JSON file."""

        with open(filepath, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filepath: str | Path) -> Vector:
        """Read from a JSON file."""

        with open(filepath) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> vector_pb2.Vector:
        """Convert to the protobuf message."""

        from .proto import vector_pb2

        proto = vector_pb2.Vector()
        proto.name = self.name
        proto.x = self._x
        proto.y = self._y
        proto.z = self._z

        return proto

    @classmethod
    def from_proto(cls, proto: vector_pb2.Vector) -> Vector:
        """Construct from the protobuf message."""

        vector = cls(proto.x, proto.y, proto.z)
        vector.name = proto.name

        return vector

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Vector:
        """Deserialize from protobuf bytes."""

        from .proto import vector_pb2

        proto = vector_pb2.Vector()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filepath: str | Path) -> None:
        """Write to a protobuf file."""

        with open(filepath, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: str | Path) -> Vector:
        """Read from a protobuf file."""

        with open(filepath, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "x, y, z"."""

        prec = Tolerance.ROUNDING

        return f"{TOLERANCE.format_number(self._x, prec)}, {TOLERANCE.format_number(self._y, prec)}, {TOLERANCE.format_number(self._z, prec)}"

    def __repr__(self) -> str:
        """Return "Vector(name, x, y, z, magnitude)"."""

        prec = Tolerance.ROUNDING

        return f"Vector({self.name}, {TOLERANCE.format_number(self._x, prec)}, {TOLERANCE.format_number(self._y, prec)}, {TOLERANCE.format_number(self._z, prec)}, {TOLERANCE.format_number(self.magnitude(), prec)})"
