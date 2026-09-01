from __future__ import annotations
from typing import Union
from typing import Optional
from typing import TYPE_CHECKING
import copy
import uuid
import math
from .tolerance import Tolerance
from .tolerance import PI
from .vector import Vector

if TYPE_CHECKING:
    from pathlib import Path
    from .plane import Plane


class Quaternion:
    """A quaternion for 3D rotations (scalar + vector)."""

    def __init__(self, scalar: float = 1.0, vector: Optional["Vector"] = None):
        """Default constructor (identity quaternion)."""
        self.typ = "Quaternion"
        self._guid = None
        self.name = "my_quaternion"
        self.scalar = scalar
        self.vector = vector if vector is not None else Vector(0.0, 0.0, 0.0)

    @property
    def guid(self) -> str:
        """Lazy GUID accessor."""
        if self._guid is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    @staticmethod
    def identity() -> "Quaternion":
        """Identity quaternion (scalar=1, vector=0)."""
        return Quaternion(1.0, Vector(0.0, 0.0, 0.0))

    @staticmethod
    def from_components(scalar: float, vector: Vector) -> "Quaternion":
        """Create a quaternion from raw scalar (real) and vector (imaginary) components.

        WARNING: The ``vector`` argument is NOT a rotation axis. It is the
        ``(i, j, k)`` coefficients of the quaternion. Most users want
        :meth:`from_axis_angle` instead.

        A quaternion is canonically written as ``q = s + xi + yj + zk`` where
        ``s`` is the scalar (real) part and ``(x, y, z)`` is the vector
        (imaginary) part. Use this constructor only when you have raw
        quaternion components.

        Visually constructing a plane from ``(s, v)`` values
        ----------------------------------------------------
        1. If ``v`` should be the plane's NORMAL (the geometric meaning users
           usually expect), bypass the quaternion entirely::

               p = Plane.from_point_normal(Point(0, 0, 0), v)

        2. If you want the plane produced by the quaternion's rotation
           (i.e. the world XY plane rotated by ``q``), normalize first::

               p = Quaternion.from_components(s, v).normalized().get_rotation()

           The result's normal is the rotation of ``(0, 0, 1)`` by ``q``,
           which equals ``v`` only in the trivial case where the rotation
           axis is already Z.

        3. If you want a quaternion whose rotation produces a plane with
           normal ``v``, use :meth:`from_arc`::

               q = Quaternion.from_arc(Vector(0, 0, 1), v.normalized())
               p = q.get_rotation()   # p.z_axis == v.normalized()

        Parameters
        ----------
        scalar : float
            Real part of the quaternion (the ``s`` in ``q = s + xi + yj + zk``).
        vector : Vector
            Imaginary parts (i, j, k coefficients) — NOT a rotation axis.

        Returns
        -------
        Quaternion
            Quaternion with the given raw components (not normalized).
        """
        return Quaternion(scalar, vector)

    @staticmethod
    def from_axis_angle(axis: "Vector", angle: float) -> "Quaternion":
        """Build a unit quaternion that rotates by ``angle`` radians around ``axis``.

        THE everyday rotation builder. Use this whenever you can describe the
        rotation as "spin by N radians around this direction" - turning a wheel,
        opening a door, orbiting a camera. The result is always unit-length.
        """
        ax = axis.normalized()
        half = angle * 0.5
        return Quaternion(math.cos(half), ax * math.sin(half))

    def to_axis_angle(self) -> tuple[Vector, float]:
        """Extract ``(axis, angle)`` from this quaternion — the inverse of :meth:`from_axis_angle`.

        Geometric meaning of a quaternion ``(s, v)``::

            axis  = v / |v|
            angle = 2 * acos(s / |q|)

        Normalizes internally, so non-unit quaternions are handled correctly.

        Edge case: for the identity quaternion (or any near-identity) the
        axis is undefined; this function returns ``(Vector(0, 0, 1), 0.0)``.

        Example
        -------
        >>> q = Quaternion.from_components(2.0, Vector(1.0, 2.0, 3.0))
        >>> axis, angle = q.to_axis_angle()
        >>> # axis = (1,2,3)/sqrt(14), angle ≈ 2.1617 rad ≈ 123.85°
        >>> # Reconstruct via geometric form:
        >>> q2 = Quaternion.from_axis_angle(axis, angle)  # == q.normalized()

        Returns
        -------
        tuple[Vector, float]
            ``(unit axis, angle in radians)``.
        """
        qn = self.normalized()
        s = max(-1.0, min(1.0, qn.scalar))
        angle = 2.0 * math.acos(s)
        sin_half = math.sqrt(1.0 - s * s)
        if sin_half < 1e-12:
            return (Vector(0.0, 0.0, 1.0), 0.0)
        axis = Vector(qn.vector[0] / sin_half, qn.vector[1] / sin_half, qn.vector[2] / sin_half)
        return (axis, angle)

    @staticmethod
    def from_arc(src: "Vector", dst: "Vector") -> "Quaternion":
        """Build the shortest rotation that maps direction ``src`` to direction ``dst``.

        Use this for "look at" logic (point a camera at a target), aligning a
        model's forward axis with a desired direction, or snapping one face
        normal to another. Both arguments are normalized internally.
        """
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
    def from_euler(x: float, y: float, z: float) -> "Quaternion":
        """Build a quaternion from three Euler angles (XYZ convention).

        Use only at I/O boundaries: importing rotations stored as pitch/yaw/roll
        or accepting user input. AVOID for composition - Euler angles suffer
        from gimbal lock. Store/compose as quaternions, convert to Euler only
        to display or save.
        """
        s1, c1 = math.sin(x * 0.5), math.cos(x * 0.5)
        s2, c2 = math.sin(y * 0.5), math.cos(y * 0.5)
        s3, c3 = math.sin(z * 0.5), math.cos(z * 0.5)
        return Quaternion(
            -s1 * s2 * s3 + c1 * c2 * c3,
            Vector(s1 * c2 * c3 + s2 * s3 * c1,
                   -s1 * s3 * c2 + s2 * c1 * c3,
                    s1 * s2 * c3 + s3 * c1 * c2))

    @staticmethod
    def from_rotation(plane_a: "Plane", plane_b: "Plane") -> "Quaternion":
        """Build the quaternion that maps the basis of ``plane_a`` onto the basis of ``plane_b``.

        Use this to snap one local frame to another - aligning two CAD parts
        by their reference planes, transferring a frame between objects, or
        computing the relative rotation between two coordinate systems.
        (Rhino: Quaternion.Rotation(plane, plane))
        """
        xa, ya, za = plane_a.x_axis, plane_a.y_axis, plane_a.z_axis
        xb, yb, zb = plane_b.x_axis, plane_b.y_axis, plane_b.z_axis
        m = [[0.0]*3 for _ in range(3)]
        m[0][0] = xb[0]*xa[0] + yb[0]*ya[0] + zb[0]*za[0]
        m[0][1] = xb[0]*xa[1] + yb[0]*ya[1] + zb[0]*za[1]
        m[0][2] = xb[0]*xa[2] + yb[0]*ya[2] + zb[0]*za[2]
        m[1][0] = xb[1]*xa[0] + yb[1]*ya[0] + zb[1]*za[0]
        m[1][1] = xb[1]*xa[1] + yb[1]*ya[1] + zb[1]*za[1]
        m[1][2] = xb[1]*xa[2] + yb[1]*ya[2] + zb[1]*za[2]
        m[2][0] = xb[2]*xa[0] + yb[2]*ya[0] + zb[2]*za[0]
        m[2][1] = xb[2]*xa[1] + yb[2]*ya[1] + zb[2]*za[1]
        m[2][2] = xb[2]*xa[2] + yb[2]*ya[2] + zb[2]*za[2]
        is_identity = True
        eps = 1.490116119385e-8
        for i in range(3):
            if not is_identity:
                break
            for j in range(3):
                d = abs(m[i][i] - 1.0) if i == j else abs(m[i][j])
                if d > eps:
                    is_identity = False
                    break
        if is_identity:
            return Quaternion(1.0, Vector(0.0, 0.0, 0.0))
        if m[0][0] >= m[1][1]:
            i = 0 if m[0][0] >= m[2][2] else 2
        else:
            i = 1 if m[1][1] >= m[2][2] else 2
        j = (i + 1) % 3
        k = (i + 2) % 3
        s = 1.0 + m[i][i] - m[j][j] - m[k][k]
        if s <= 0.0:
            return Quaternion(1.0, Vector(0.0, 0.0, 0.0))
        r = math.sqrt(s)
        s = 0.5 / r
        q = [0.0, 0.0, 0.0]
        q[i] = 0.5 * r
        q[j] = s * (m[i][j] + m[j][i])
        q[k] = s * (m[k][i] + m[i][k])
        return Quaternion(s * (m[k][j] - m[j][k]), Vector(q[0], q[1], q[2]))

    def get_rotation(self) -> "Plane":
        """Apply this rotation to the world XY plane and return the resulting Plane.

        Use this to visualize a quaternion as a frame in 3D, or to convert a
        stored quaternion into a Plane for frame-based APIs in the rest of the
        kernel. Inverse: ``from_rotation(xy_plane(), result)``.
        (Rhino: Quaternion.GetRotation(out plane))
        """
        from .plane import Plane
        from .point import Point
        a, b, c, d = self.scalar, self.vector[0], self.vector[1], self.vector[2]
        xaxis = Vector(a*a + b*b - c*c - d*d, 2.0*(a*d + b*c),       2.0*(b*d - a*c))
        yaxis = Vector(2.0*(b*c - a*d),       a*a - b*b + c*c - d*d, 2.0*(a*b + c*d))
        return Plane(Point(0.0, 0.0, 0.0), xaxis, yaxis)

    def duplicate(self) -> "Quaternion":
        """Create a deep copy of this quaternion with a new GUID."""
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def rotate_vector(self, v: "Vector") -> "Vector":
        """Apply this rotation to a 3D vector and return the rotated vector.

        Use this when you have a quaternion orientation and need to know where
        a specific direction points after the rotation - the camera's forward
        axis, a bone's tip, the normal of a rotated face. Math: ``q*v_pure*q^-1``.
        """
        qv = self.vector
        uv = qv.cross(v)
        uuv = qv.cross(uv)
        return v + (uv * self.scalar + uuv) * 2.0

    def magnitude(self) -> float:
        """Euclidean norm."""
        return math.sqrt(self.magnitude_squared())

    def magnitude_squared(self) -> float:
        """Squared magnitude."""
        return self.scalar * self.scalar + self.vector[0] * self.vector[0] + self.vector[1] * self.vector[1] + self.vector[2] * self.vector[2]

    def normalized(self) -> "Quaternion":
        """Return a unit-length copy of this quaternion (divides by magnitude).

        Use periodically after composing many rotations - floating-point drift
        slowly makes a quaternion non-unit, and a non-unit quaternion no longer
        represents a valid rotation.
        """
        mag = self.magnitude()
        if mag > 1e-10:
            q = Quaternion(self.scalar / mag, self.vector / mag)
            q.typ = self.typ
            q.guid = self.guid
            q.name = self.name
            return q
        return Quaternion.identity()

    def conjugate(self) -> "Quaternion":
        """Flip the sign of the vector part: ``(s, v)`` -> ``(s, -v)``.

        For UNIT quaternions this equals the inverse - the opposite rotation.
        Use as the cheap inverse when you KNOW the quaternion is unit-length.
        """
        q = Quaternion(self.scalar, self.vector * -1.0)
        q.typ = self.typ
        q.guid = self.guid
        q.name = self.name
        return q

    def invert(self) -> "Quaternion":
        """True multiplicative inverse: conjugate / magnitude_squared.

        Works for non-unit quaternions too. Use as the safe inverse when the
        quaternion may not be unit-length. ``q * q.invert()`` always equals
        identity.
        """
        mag2 = self.magnitude_squared()
        if mag2 < 1e-20:
            return Quaternion.identity()
        q = Quaternion(self.scalar / mag2, self.vector * (-1.0 / mag2))
        q.typ = self.typ
        q.guid = self.guid
        q.name = self.name
        return q

    def dot(self, other: "Quaternion") -> float:
        """Algebraic 4D dot product (NOT a geometric operation).

        Used inside slerp implementations and as a similarity measure between
        two unit quaternions (1 = same, 0 = 90 deg apart).
        """
        return self.scalar * other.scalar + self.vector.dot(other.vector)

    def slerp(self, other: "Quaternion", amount: float) -> "Quaternion":
        """Spherical Linear intERPolation along the shortest great-circle path on S^3.

        Constant angular velocity. Use for high-quality animation between two
        orientations - camera transitions, character bones, anything where
        smoothness matters more than raw speed.
        """
        dot_val = self.dot(other)
        if dot_val > 0.9995:
            return (self + (other - self) * amount).normalized()
        robust_dot = max(-1.0, min(1.0, dot_val))
        theta = math.acos(robust_dot)
        scale1 = math.sin(theta * (1.0 - amount))
        scale2 = math.sin(theta * amount)
        sin_theta = math.sin(theta)
        return (self * scale1 + other * scale2) * (1.0 / sin_theta)

    def nlerp(self, other: "Quaternion", amount: float) -> "Quaternion":
        """Normalized Linear intERPolation. Cheaper than slerp.

        Angular velocity isn't perfectly uniform. Use in real-time loops where
        every microsecond matters and the visual difference from slerp is
        negligible.
        """
        return (self * (1.0 - amount) + other * amount).normalized()

    def __getitem__(self, index):
        if index == 0:
            return self.scalar
        elif index == 1:
            return self.vector[0]
        elif index == 2:
            return self.vector[1]
        elif index == 3:
            return self.vector[2]
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        if index == 0:
            self.scalar = value
        elif index == 1:
            self.vector[0] = value
        elif index == 2:
            self.vector[1] = value
        elif index == 3:
            self.vector[2] = value
        else:
            raise IndexError("Index out of range")

    def __mul__(self, other):
        """Quaternion multiplication (composition) or scalar multiplication."""
        if isinstance(other, Quaternion):
            scalar = self.scalar * other.scalar - self.vector.dot(other.vector)
            vector = other.vector * self.scalar + self.vector * other.scalar + self.vector.cross(other.vector)
            return Quaternion(scalar, vector)
        if isinstance(other, (int, float)):
            return Quaternion(self.scalar * other, self.vector * other)
        raise TypeError(f"unsupported operand type for *: Quaternion and {type(other)}")

    def __add__(self, other):
        """Component-wise addition."""
        return Quaternion(self.scalar + other.scalar, self.vector + other.vector)

    def __sub__(self, other):
        """Component-wise subtraction."""
        return Quaternion(self.scalar - other.scalar, self.vector - other.vector)

    def __neg__(self):
        """Negation."""
        return Quaternion(-self.scalar, self.vector * -1.0)

    def __eq__(self, other):
        # C++ and Rust take a `Quaternion` and cannot be handed anything else; Python can, and
        # without this guard `q == None` raised AttributeError instead of answering False.
        if not isinstance(other, Quaternion):
            return False
        return (
            self.name == other.name
            and round(self.scalar, Tolerance.ROUNDING) == round(other.scalar, Tolerance.ROUNDING)
            and round(self.vector[0], Tolerance.ROUNDING) == round(other.vector[0], Tolerance.ROUNDING)
            and round(self.vector[1], Tolerance.ROUNDING) == round(other.vector[1], Tolerance.ROUNDING)
            and round(self.vector[2], Tolerance.ROUNDING) == round(other.vector[2], Tolerance.ROUNDING)
        )

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return f"{self.scalar}, {self.vector[0]}, {self.vector[1]}, {self.vector[2]}"

    def __repr__(self):
        return f"Quaternion({self.name}, {self.scalar}, {self.vector[0]}, {self.vector[1]}, {self.vector[2]})"

    def __jsondump__(self):
        """Serialize to JSON dict."""
        return {
            "type": f"{self.__class__.__name__}",
            "guid": self.guid,
            "name": self.name,
            "s": self.scalar,
            "x": self.vector[0],
            "y": self.vector[1],
            "z": self.vector[2],
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from JSON dict."""
        from .file_encoders import file_decode_node
        from .vector import Vector

        if "v" in data:
            vector = file_decode_node(data["v"])
        else:
            vector = Vector(data["x"], data["y"], data["z"])

        q = cls(data["s"], vector)
        q.guid = guid
        q.name = name
        return q

    def file_json_dumps(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Quaternion":
        """Load from JSON string."""
        import json
        data = json.loads(json_string)
        return cls.__jsonload__(data, guid=data.get("guid"), name=data.get("name"))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write JSON to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "Quaternion":
        """Read JSON from file."""
        import json
        with open(filepath) as f:
            data = json.load(f)
        return cls.__jsonload__(data, guid=data.get("guid"), name=data.get("name"))

    def pb_dumps(self) -> bytes:
        """Convert to protobuf binary format."""
        from .proto import quaternion_pb2
        proto = quaternion_pb2.Quaternion()
        proto.a = self.scalar
        proto.b = self.vector[0]
        proto.c = self.vector[1]
        proto.d = self.vector[2]
        proto.name = self.name
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Quaternion":
        """Create Quaternion from protobuf binary data."""
        from .proto import quaternion_pb2
        from .vector import Vector
        proto = quaternion_pb2.Quaternion()
        proto.ParseFromString(data)
        q = cls(proto.a, Vector(proto.b, proto.c, proto.d))
        q.name = proto.name
        return q

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write protobuf to file."""
        data = self.pb_dumps()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "Quaternion":
        """Read protobuf from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)
