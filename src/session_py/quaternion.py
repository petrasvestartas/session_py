import uuid
import math
from .vector import Vector


class Quaternion:
    def __init__(self, s=1.0, v=None):
        self.typ = "Quaternion"
        self._guid = None
        self.name = "my_quaternion"
        self.s = s
        self.v = v if v is not None else Vector(0.0, 0.0, 0.0)

    @property
    def guid(self) -> str:
        if self._guid is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

    @staticmethod
    def identity():
        return Quaternion(1.0, Vector(0.0, 0.0, 0.0))

    @staticmethod
    def from_sv(s, v):
        return Quaternion(s, v)

    @staticmethod
    def from_axis_angle(axis, angle):
        ax = axis.normalize()
        half = angle * 0.5
        return Quaternion(math.cos(half), ax * math.sin(half))

    @staticmethod
    def from_arc(src, dst):
        s = src.normalize()
        d = dst.normalize()
        cross = s.cross(d)
        dot_val = s.dot(d)
        if cross.magnitude() < 1e-10:
            if dot_val < 0.0:
                perp = s.cross(Vector(0.0, 0.0, 1.0))
                if perp.magnitude() < 1e-10:
                    perp = s.cross(Vector(0.0, 1.0, 0.0))
                return Quaternion.from_axis_angle(perp.normalize(), math.pi)
            return Quaternion.identity()
        return Quaternion(1.0 + dot_val, cross).normalize()

    @staticmethod
    def from_euler(x, y, z):
        s1, c1 = math.sin(x * 0.5), math.cos(x * 0.5)
        s2, c2 = math.sin(y * 0.5), math.cos(y * 0.5)
        s3, c3 = math.sin(z * 0.5), math.cos(z * 0.5)
        return Quaternion(
            -s1 * s2 * s3 + c1 * c2 * c3,
            Vector(s1 * c2 * c3 + s2 * s3 * c1,
                   -s1 * s3 * c2 + s2 * c1 * c3,
                    s1 * s2 * c3 + s3 * c1 * c2))

    def rotate_vector(self, v):
        qv = self.v
        uv = qv.cross(v)
        uuv = qv.cross(uv)
        return v + (uv * self.s + uuv) * 2.0

    def magnitude(self):
        return math.sqrt(self.magnitude2())

    def magnitude2(self):
        return self.s * self.s + self.v[0] * self.v[0] + self.v[1] * self.v[1] + self.v[2] * self.v[2]

    def normalize(self):
        mag = self.magnitude()
        if mag > 1e-10:
            q = Quaternion(self.s / mag, self.v / mag)
            q.typ = self.typ
            q.guid = self.guid
            q.name = self.name
            return q
        return Quaternion.identity()

    def conjugate(self):
        q = Quaternion(self.s, self.v * -1.0)
        q.typ = self.typ
        q.guid = self.guid
        q.name = self.name
        return q

    def invert(self):
        mag2 = self.magnitude2()
        if mag2 < 1e-20:
            return Quaternion.identity()
        q = Quaternion(self.s / mag2, self.v * (-1.0 / mag2))
        q.typ = self.typ
        q.guid = self.guid
        q.name = self.name
        return q

    def dot(self, other):
        return self.s * other.s + self.v.dot(other.v)

    def slerp(self, other, amount):
        dot_val = self.dot(other)
        if dot_val > 0.9995:
            return (self + (other - self) * amount).normalize()
        robust_dot = max(-1.0, min(1.0, dot_val))
        theta = math.acos(robust_dot)
        scale1 = math.sin(theta * (1.0 - amount))
        scale2 = math.sin(theta * amount)
        sin_theta = math.sin(theta)
        return (self * scale1 + other * scale2) * (1.0 / sin_theta)

    def nlerp(self, other, amount):
        return (self * (1.0 - amount) + other * amount).normalize()

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            s = self.s * other.s - self.v.dot(other.v)
            v = other.v * self.s + self.v * other.s + self.v.cross(other.v)
            return Quaternion(s, v)
        if isinstance(other, (int, float)):
            return Quaternion(self.s * other, self.v * other)
        raise TypeError(f"unsupported operand type for *: Quaternion and {type(other)}")

    def __add__(self, other):
        return Quaternion(self.s + other.s, self.v + other.v)

    def __sub__(self, other):
        return Quaternion(self.s - other.s, self.v - other.v)

    def __neg__(self):
        return Quaternion(-self.s, self.v * -1.0)

    def __jsondump__(self):
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
