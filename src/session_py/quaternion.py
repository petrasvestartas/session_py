from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import json
import math
import uuid
from .tolerance import Tolerance
from .tolerance import TOLERANCE
from .tolerance import PI
from .vector import Vector

if TYPE_CHECKING:
    from pathlib import Path
    from .plane import Plane
    from .proto import quaternion_pb2


class Quaternion:
    """A rotation as scalar plus vector part: q = s + xi + yj + zk."""

    __slots__ = ("_guid", "name", "scalar", "vector")

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, scalar: float = 1.0, vector: Vector | None = None):
        """Construct from raw components; vector is (i, j, k), not a rotation axis."""

        self._guid = None  # Lazily minted GUID.
        self.name = "my_quaternion"  # Quaternion name.
        self.scalar = scalar  # Scalar part s.
        self.vector = Vector(0.0, 0.0, 0.0) if vector is None else vector.duplicate()  # Vector part (x, y, z).

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = Quaternion(self.scalar, self.vector)
        result.name = self.name
        memo[id(self)] = result

        return result

    def duplicate(self) -> Quaternion:
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
    def identity() -> Quaternion:
        """Construct the rotation that does nothing: scalar 1, vector 0."""
        return Quaternion(1.0, Vector(0.0, 0.0, 0.0))

    @staticmethod
    def from_components(scalar: float, vector: Vector) -> Quaternion:
        """Construct from raw components; vector is (i, j, k), not a rotation axis."""
        return Quaternion(scalar, vector)

    @staticmethod
    def from_axis_angle(axis: Vector, angle: float) -> Quaternion:
        """Construct the unit quaternion rotating by angle radians around axis."""

        if axis.magnitude() < 1e-10:
            return Quaternion.identity()

        ax = axis.normalized()
        half = angle * 0.5

        return Quaternion(math.cos(half), ax * math.sin(half))

    @staticmethod
    def from_arc(src: Vector, dst: Vector) -> Quaternion:
        """Construct the shortest rotation taking direction src to direction dst."""

        s = src.normalized()
        d = dst.normalized()
        cross = s.cross(d)
        dot_val = s.dot(d)

        if cross.magnitude() < 1e-10:
            if dot_val < 0.0:
                perp = s.cross(Vector(0.0, 0.0, 1.0))

                if perp.magnitude() < 1e-10:
                    perp = s.cross(Vector(0.0, 1.0, 0.0))

                return Quaternion.from_axis_angle(perp.normalized(), PI)

            return Quaternion.identity()

        return Quaternion(1.0 + dot_val, cross).normalized()

    @staticmethod
    def from_euler(x: float, y: float, z: float) -> Quaternion:
        """Construct the rotation from Euler angles in XYZ convention."""

        s1 = math.sin(x * 0.5)
        c1 = math.cos(x * 0.5)
        s2 = math.sin(y * 0.5)
        c2 = math.cos(y * 0.5)
        s3 = math.sin(z * 0.5)
        c3 = math.cos(z * 0.5)

        return Quaternion(
            -s1 * s2 * s3 + c1 * c2 * c3,
            Vector(
                s1 * c2 * c3 + s2 * s3 * c1,
                -s1 * s3 * c2 + s2 * c1 * c3,
                s1 * s2 * c3 + s3 * c1 * c2,
            ),
        )

    @staticmethod
    def from_rotation(plane_a: Plane, plane_b: Plane) -> Quaternion:
        """Construct the rotation mapping the frame of plane_a onto the frame of plane_b."""

        xa = plane_a.x_axis
        ya = plane_a.y_axis
        za = plane_a.z_axis
        xb = plane_b.x_axis
        yb = plane_b.y_axis
        zb = plane_b.z_axis
        m = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        for i in range(3):
            for j in range(3):
                m[i][j] = xb[i] * xa[j] + yb[i] * ya[j] + zb[i] * za[j]

        eps = 1.490116119385e-8
        is_identity = True

        for i in range(3):
            for j in range(3):
                if abs(m[i][j] - (1.0 if i == j else 0.0)) > eps:
                    is_identity = False

        if is_identity:
            return Quaternion.identity()

        i = 2

        if m[0][0] >= m[1][1] and m[0][0] >= m[2][2]:
            i = 0
        elif m[1][1] >= m[0][0] and m[1][1] >= m[2][2]:
            i = 1

        j = (i + 1) % 3
        k = (i + 2) % 3
        s = 1.0 + m[i][i] - m[j][j] - m[k][k]

        if s <= 0.0:
            return Quaternion.identity()

        r = math.sqrt(s)
        s = 0.5 / r

        q = [0.0, 0.0, 0.0]
        q[i] = 0.5 * r
        q[j] = s * (m[i][j] + m[j][i])
        q[k] = s * (m[k][i] + m[i][k])

        return Quaternion(s * (m[k][j] - m[j][k]), Vector(q[0], q[1], q[2]))

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __getitem__(self, index: int) -> float:
        """Return the component by index (0=scalar, 1=x, 2=y, 3=z)."""

        if index == 0:
            return self.scalar

        if 1 <= index <= 3:
            return self.vector[index - 1]

        raise IndexError("Index out of range")

    def __setitem__(self, index: int, value: float) -> None:
        """Set the component by index (0=scalar, 1=x, 2=y, 3=z)."""

        if index == 0:
            self.scalar = value
        elif 1 <= index <= 3:
            self.vector[index - 1] = value
        else:
            raise IndexError("Index out of range")

    def __eq__(self, other) -> bool:
        """Compare name and components to six decimals; guid ignored."""

        if not isinstance(other, Quaternion):
            return False

        return (
            self.name == other.name
            and round(self.scalar * 1000000.0) == round(other.scalar * 1000000.0)
            and round(self.vector[0] * 1000000.0) == round(other.vector[0] * 1000000.0)
            and round(self.vector[1] * 1000000.0) == round(other.vector[1] * 1000000.0)
            and round(self.vector[2] * 1000000.0) == round(other.vector[2] * 1000000.0)
        )

    def __ne__(self, other) -> bool:
        """Compare name and components to six decimals; guid ignored."""
        return not self.__eq__(other)

    def __mul__(self, other):
        """Return the composition (a * b applies b first, then a) or a copy scaled by a number."""

        if isinstance(other, Quaternion):
            return Quaternion(
                self.scalar * other.scalar - self.vector.dot(other.vector),
                other.vector * self.scalar
                + self.vector * other.scalar
                + self.vector.cross(other.vector),
            )

        return Quaternion(self.scalar * other, self.vector * other)

    def __add__(self, other: Quaternion) -> Quaternion:
        """Return the component-wise sum."""
        return Quaternion(self.scalar + other.scalar, self.vector + other.vector)

    def __sub__(self, other: Quaternion) -> Quaternion:
        """Return the component-wise difference."""
        return Quaternion(self.scalar - other.scalar, self.vector - other.vector)

    def __neg__(self) -> Quaternion:
        """Return the negated copy."""
        return Quaternion(-self.scalar, -self.vector)

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometry
    # ═══════════════════════════════════════════════════════════════════════════
    def to_axis_angle(self) -> tuple[Vector, float]:
        """Return the unit axis and angle in radians; (0, 0, 1) and 0 near identity."""

        qn = self.normalized()
        s = max(-1.0, min(1.0, qn.scalar))
        angle = 2.0 * math.acos(s)
        sin_half = math.sqrt(1.0 - s * s)

        if sin_half < 1e-12:
            return (Vector(0.0, 0.0, 1.0), 0.0)

        return (qn.vector / sin_half, angle)

    def rotate_vector(self, vec: Vector) -> Vector:
        """Return a rotated copy of vec: q * v * q^-1."""

        uv = self.vector.cross(vec)
        uuv = self.vector.cross(uv)

        return vec + (uv * self.scalar + uuv) * 2.0

    def get_rotation(self) -> Plane:
        """Return the world XY plane rotated by this quaternion."""

        from .plane import Plane
        from .point import Point

        a = self.scalar
        b = self.vector[0]
        c = self.vector[1]
        d = self.vector[2]
        xaxis = Vector(
            a * a + b * b - c * c - d * d, 2.0 * (a * d + b * c), 2.0 * (b * d - a * c)
        )
        yaxis = Vector(
            2.0 * (b * c - a * d), a * a - b * b + c * c - d * d, 2.0 * (a * b + c * d)
        )
        zaxis = Vector(
            2.0 * (a * c + b * d), 2.0 * (c * d - a * b), a * a - b * b - c * c + d * d
        )

        return Plane.from_frame(Point(0.0, 0.0, 0.0), xaxis, yaxis, zaxis)

    def magnitude(self) -> float:
        """Return the 4D length."""
        return math.sqrt(self.magnitude_squared())

    def magnitude_squared(self) -> float:
        """Return the squared magnitude without the square root."""
        return self.scalar * self.scalar + self.vector.dot(self.vector)

    def normalized(self) -> Quaternion:
        """Return a unit length copy; identity when the magnitude is zero."""

        mag = self.magnitude()

        if mag < 1e-10:
            return Quaternion.identity()

        q = Quaternion(self.scalar / mag, self.vector / mag)
        q.name = self.name

        return q

    def conjugate(self) -> Quaternion:
        """Return (s, -v); the inverse of a unit quaternion."""

        q = Quaternion(self.scalar, -self.vector)
        q.name = self.name

        return q

    def invert(self) -> Quaternion:
        """Return the multiplicative inverse: conjugate over squared magnitude."""

        mag2 = self.magnitude_squared()

        if mag2 < 1e-20:
            return Quaternion.identity()

        q = Quaternion(self.scalar / mag2, self.vector * (-1.0 / mag2))
        q.name = self.name

        return q

    def dot(self, other: Quaternion) -> float:
        """Return the 4D dot product."""
        return self.scalar * other.scalar + self.vector.dot(other.vector)

    def slerp(self, other: Quaternion, amount: float) -> Quaternion:
        """Return the spherical interpolation at constant angular velocity."""

        target = other
        dot_val = self.dot(target)

        if dot_val < 0.0:
            target = -target
            dot_val = -dot_val

        if dot_val > 0.9995:
            return (self + (target - self) * amount).normalized()

        theta = math.acos(max(-1.0, min(1.0, dot_val)))
        scale1 = math.sin(theta * (1.0 - amount))
        scale2 = math.sin(theta * amount)

        return (self * scale1 + target * scale2) * (1.0 / math.sin(theta))

    def nlerp(self, other: Quaternion, amount: float) -> Quaternion:
        """Return the normalized linear interpolation, cheaper than slerp."""
        return (self * (1.0 - amount) + other * amount).normalized()

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to an ordered JSON object."""

        return {
            "guid": self.guid,
            "name": self.name,
            "s": self.scalar,
            "type": "Quaternion",
            "x": self.vector[0],
            "y": self.vector[1],
            "z": self.vector[2],
        }

    @classmethod
    def __jsonload__(cls, data: dict, guid: str | None = None, name: str | None = None) -> Quaternion:
        """Deserialize from a JSON object."""

        q = cls(data["s"], Vector(data["x"], data["y"], data["z"]))
        q.guid = guid or data["guid"]
        q.name = name or data["name"]

        return q

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> Quaternion:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write JSON to a file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Quaternion:
        """Read JSON from a file."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> quaternion_pb2.Quaternion:
        """Convert to the protobuf message."""

        from .proto import quaternion_pb2

        proto = quaternion_pb2.Quaternion()
        proto.a = self.scalar
        proto.b = self.vector[0]
        proto.c = self.vector[1]
        proto.d = self.vector[2]
        proto.name = self.name

        return proto

    @classmethod
    def from_proto(cls, proto: quaternion_pb2.Quaternion) -> Quaternion:
        """Construct from the protobuf message."""

        q = cls(proto.a, Vector(proto.b, proto.c, proto.d))
        q.name = proto.name

        return q

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Quaternion:
        """Deserialize from protobuf bytes."""

        from .proto import quaternion_pb2

        proto = quaternion_pb2.Quaternion()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write protobuf bytes to a file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Quaternion:
        """Read protobuf bytes from a file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "s, x, y, z"."""

        prec = Tolerance.ROUNDING

        return f"{TOLERANCE.format_number(self.scalar, prec)}, {TOLERANCE.format_number(self.vector[0], prec)}, {TOLERANCE.format_number(self.vector[1], prec)}, {TOLERANCE.format_number(self.vector[2], prec)}"

    def __repr__(self) -> str:
        """Return "Quaternion(name, s, x, y, z)"."""

        prec = Tolerance.ROUNDING

        return f"Quaternion({self.name}, {TOLERANCE.format_number(self.scalar, prec)}, {TOLERANCE.format_number(self.vector[0], prec)}, {TOLERANCE.format_number(self.vector[1], prec)}, {TOLERANCE.format_number(self.vector[2], prec)})"
