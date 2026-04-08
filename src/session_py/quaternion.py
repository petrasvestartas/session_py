import copy
import uuid
import math
from .tolerance import Tolerance
from .tolerance import PI
from .vector import Vector


class Quaternion:
    """A quaternion for 3D rotations (scalar + vector)."""

    def __init__(self, scalar=1.0, vector=None):
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
    def guid(self, value: str):
        self._guid = value

    @staticmethod
    def identity():
        """Identity quaternion (scalar=1, vector=0)."""
        return Quaternion(1.0, Vector(0.0, 0.0, 0.0))

    @staticmethod
    def from_scalar_and_vector(scalar, vector):
        """Create from scalar and vector components."""
        return Quaternion(scalar, vector)

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
                return Quaternion.from_axis_angle(perp.normalized(), PI)
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

    @staticmethod
    def from_rotation(plane_a, plane_b):
        """Create rotation that maps the basis of plane_a onto plane_b (Rhino: Quaternion.Rotation(plane, plane))."""
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

    def get_rotation(self):
        """Apply this quaternion's rotation to the world XY plane and return the resulting plane (Rhino: Quaternion.GetRotation(out plane))."""
        from .plane import Plane
        from .point import Point
        a, b, c, d = self.scalar, self.vector[0], self.vector[1], self.vector[2]
        xaxis = Vector(a*a + b*b - c*c - d*d, 2.0*(a*d + b*c),       2.0*(b*d - a*c))
        yaxis = Vector(2.0*(b*c - a*d),       a*a - b*b + c*c - d*d, 2.0*(a*b + c*d))
        return Plane(Point(0.0, 0.0, 0.0), xaxis, yaxis)

    def duplicate(self):
        """Create a deep copy of this quaternion with a new GUID."""
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def rotate_vector(self, v):
        """Rotate a vector by this quaternion."""
        qv = self.vector
        uv = qv.cross(v)
        uuv = qv.cross(uv)
        return v + (uv * self.scalar + uuv) * 2.0

    def magnitude(self):
        """Euclidean norm."""
        return math.sqrt(self.magnitude_squared())

    def magnitude_squared(self):
        """Squared magnitude."""
        return self.scalar * self.scalar + self.vector[0] * self.vector[0] + self.vector[1] * self.vector[1] + self.vector[2] * self.vector[2]

    def normalized(self):
        """Unit quaternion with same direction."""
        mag = self.magnitude()
        if mag > 1e-10:
            q = Quaternion(self.scalar / mag, self.vector / mag)
            q.typ = self.typ
            q.guid = self.guid
            q.name = self.name
            return q
        return Quaternion.identity()

    def conjugate(self):
        """Conjugate (negates vector part)."""
        q = Quaternion(self.scalar, self.vector * -1.0)
        q.typ = self.typ
        q.guid = self.guid
        q.name = self.name
        return q

    def invert(self):
        """Multiplicative inverse."""
        mag2 = self.magnitude_squared()
        if mag2 < 1e-20:
            return Quaternion.identity()
        q = Quaternion(self.scalar / mag2, self.vector * (-1.0 / mag2))
        q.typ = self.typ
        q.guid = self.guid
        q.name = self.name
        return q

    def dot(self, other):
        """Dot product with another quaternion."""
        return self.scalar * other.scalar + self.vector.dot(other.vector)

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
        from .encoders import decode_node
        from .vector import Vector

        if "v" in data:
            vector = decode_node(data["v"])
        else:
            vector = Vector(data["x"], data["y"], data["z"])

        q = cls(data["s"], vector)
        q.guid = guid
        q.name = name
        return q

    def json_dumps(self):
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, json_string):
        """Load from JSON string."""
        import json
        data = json.loads(json_string)
        return cls.__jsonload__(data, guid=data.get("guid"), name=data.get("name"))

    def json_dump(self, filepath):
        """Write JSON to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        """Read JSON from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data, guid=data.get("guid"), name=data.get("name"))

    def pb_dumps(self):
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
    def pb_loads(cls, data):
        """Create Quaternion from protobuf binary data."""
        from .proto import quaternion_pb2
        from .vector import Vector
        proto = quaternion_pb2.Quaternion()
        proto.ParseFromString(data)
        q = cls(proto.a, Vector(proto.b, proto.c, proto.d))
        q.name = proto.name
        return q

    def pb_dump(self, filepath):
        """Write protobuf to file."""
        data = self.pb_dumps()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath):
        """Read protobuf from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)
