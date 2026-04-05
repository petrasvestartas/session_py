import uuid
import math
from .vector import Vector


class Quaternion:
    """A quaternion for 3D rotations (scalar + vector)."""

    def __init__(self, s=1.0, v=None):
        """Default constructor (identity quaternion)."""
        self.typ = "Quaternion"
        self._guid = None
        self.name = "my_quaternion"
        self.s = s
        self.v = v if v is not None else Vector(0.0, 0.0, 0.0)

    @property
    def guid(self) -> str:
        """Lazy GUID accessor."""
        if self._guid is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

    @staticmethod
    def identity():
        """Identity quaternion (s=1, v=0)."""
        return Quaternion(1.0, Vector(0.0, 0.0, 0.0))

    @staticmethod
    def from_sv(s, v):
        """Create from scalar and vector components."""
        return Quaternion(s, v)

    @staticmethod
    def from_axis_angle(axis, angle):
        """Create from axis of rotation and angle."""
        ax = axis.normalized()
        half = angle * 0.5
        return Quaternion(math.cos(half), ax * math.sin(half))

    @staticmethod
    def from_arc(src, dst):
        """Create rotation from source vector to destination vector."""
        s = src.normalized()
        d = dst.normalized()
        cross = s.cross(d)
        dot_val = s.dot(d)
        if cross.magnitude() < 1e-10:
            if dot_val < 0.0:
                perp = s.cross(Vector(0.0, 0.0, 1.0))
                if perp.magnitude() < 1e-10:
                    perp = s.cross(Vector(0.0, 1.0, 0.0))
                return Quaternion.from_axis_angle(perp.normalized(), math.pi)
            return Quaternion.identity()
        return Quaternion(1.0 + dot_val, cross).normalized()

    @staticmethod
    def from_euler(x, y, z):
        """Create from Euler angles (XYZ convention)."""
        s1, c1 = math.sin(x * 0.5), math.cos(x * 0.5)
        s2, c2 = math.sin(y * 0.5), math.cos(y * 0.5)
        s3, c3 = math.sin(z * 0.5), math.cos(z * 0.5)
        return Quaternion(
            -s1 * s2 * s3 + c1 * c2 * c3,
            Vector(s1 * c2 * c3 + s2 * s3 * c1,
                   -s1 * s3 * c2 + s2 * c1 * c3,
                    s1 * s2 * c3 + s3 * c1 * c2))

    def rotate_vector(self, v):
        """Rotate a vector by this quaternion."""
        qv = self.v
        uv = qv.cross(v)
        uuv = qv.cross(uv)
        return v + (uv * self.s + uuv) * 2.0

    def magnitude(self):
        """Euclidean norm."""
        return math.sqrt(self.magnitude_squared())

    def magnitude_squared(self):
        """Squared magnitude."""
        return self.s * self.s + self.v[0] * self.v[0] + self.v[1] * self.v[1] + self.v[2] * self.v[2]

    def normalized(self):
        """Unit quaternion with same direction."""
        mag = self.magnitude()
        if mag > 1e-10:
            q = Quaternion(self.s / mag, self.v / mag)
            q.typ = self.typ
            q.guid = self.guid
            q.name = self.name
            return q
        return Quaternion.identity()

    def conjugate(self):
        """Conjugate (negates vector part)."""
        q = Quaternion(self.s, self.v * -1.0)
        q.typ = self.typ
        q.guid = self.guid
        q.name = self.name
        return q

    def invert(self):
        """Multiplicative inverse."""
        mag2 = self.magnitude_squared()
        if mag2 < 1e-20:
            return Quaternion.identity()
        q = Quaternion(self.s / mag2, self.v * (-1.0 / mag2))
        q.typ = self.typ
        q.guid = self.guid
        q.name = self.name
        return q

    def dot(self, other):
        """Dot product with another quaternion."""
        return self.s * other.s + self.v.dot(other.v)

    def slerp(self, other, amount):
        """Spherical linear interpolation."""
        dot_val = self.dot(other)
        if dot_val > 0.9995:
            return (self + (other - self) * amount).normalized()
        robust_dot = max(-1.0, min(1.0, dot_val))
        theta = math.acos(robust_dot)
        scale1 = math.sin(theta * (1.0 - amount))
        scale2 = math.sin(theta * amount)
        sin_theta = math.sin(theta)
        return (self * scale1 + other * scale2) * (1.0 / sin_theta)

    def nlerp(self, other, amount):
        """Normalized linear interpolation."""
        return (self * (1.0 - amount) + other * amount).normalized()

    def __mul__(self, other):
        """Quaternion multiplication (composition) or scalar multiplication."""
        if isinstance(other, Quaternion):
            s = self.s * other.s - self.v.dot(other.v)
            v = other.v * self.s + self.v * other.s + self.v.cross(other.v)
            return Quaternion(s, v)
        if isinstance(other, (int, float)):
            return Quaternion(self.s * other, self.v * other)
        raise TypeError(f"unsupported operand type for *: Quaternion and {type(other)}")

    def __add__(self, other):
        """Component-wise addition."""
        return Quaternion(self.s + other.s, self.v + other.v)

    def __sub__(self, other):
        """Component-wise subtraction."""
        return Quaternion(self.s - other.s, self.v - other.v)

    def __neg__(self):
        """Negation."""
        return Quaternion(-self.s, self.v * -1.0)

    def __jsondump__(self):
        """Serialize to JSON dict."""
        return {
            "type": f"{self.__class__.__name__}",
            "guid": self.guid,
            "name": self.name,
            "s": self.s,
            "x": self.v[0],
            "y": self.v[1],
            "z": self.v[2],
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from JSON dict."""
        from .encoders import decode_node
        from .vector import Vector

        if "v" in data:
            v = decode_node(data["v"])
        else:
            v = Vector(data["x"], data["y"], data["z"])

        q = cls(data["s"], v)
        q.guid = guid
        q.name = name
        return q
