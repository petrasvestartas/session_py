from typing import Union
from typing import TYPE_CHECKING
import uuid
import math
from .color import Color
from .vector import Vector
from .tolerance import Tolerance
import copy

if TYPE_CHECKING:
    from pathlib import Path
    from .xform import Xform


class Point:
    """A 3D point with visual properties.

    Parameters
    ----------
    x : float, optional
        X coordinate. Defaults to 0.0.
    y : float, optional
        Y coordinate. Defaults to 0.0.
    z : float, optional
        Z coordinate. Defaults to 0.0.

    Attributes
    ----------
    name : str
        The name of the point.
    guid : str
        The unique identifier of the point.
    pointcolor : :class:`Color`
        The color of the point.
    width : float
        The width of the point for display.

    """

    __slots__ = ("_guid", "name", "_x", "_y", "_z", "width", "_pointcolor")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, name: str = "my_point"):
        self._guid = None
        self.name = name
        self._x = x
        self._y = y
        self._z = z
        self.width = 1.0
        self._pointcolor = None

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def refresh_guid(self) -> None:
        """Clear the guid so a FRESH one mints lazily on next read — the duplicate/copy enabler."""
        self._guid = None

    @property
    def pointcolor(self) -> "Color":
        if self._pointcolor is None:
            self._pointcolor = Color.black()
        return self._pointcolor

    @pointcolor.setter
    def pointcolor(self, value: "Color") -> None:
        self._pointcolor = value

    ###########################################################################################
    # Operators
    ###########################################################################################

    def __deepcopy__(self, memo):

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # New guid
        result.guid = str(uuid.uuid4())

        # Copy remaining fields
        result.name = copy.deepcopy(self.name, memo)
        result._x = self._x
        result._y = self._y
        result._z = self._z
        result.width = self.width
        result.pointcolor = copy.deepcopy(self.pointcolor, memo)
        return result

    def duplicate(self) -> "Point":
        """Create a deep copy of this point with a new GUID.

        Returns
        -------
        :class:`Point`
            A new Point with identical values but a different GUID.

        """
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    @staticmethod
    def sum(p0: "Point", p1: "Point") -> "Point":
        """Returns a new point that is the sum of two points.

        Parameters
        ----------
        p0 : :class:`Point`
            First point.
        p1 : :class:`Point`
            Second point.

        Returns
        -------
        :class:`Point`
            A new Point with coordinates (p0.x + p1.x, p0.y + p1.y, p0.z + p1.z).

        """
        return Point(p0[0] + p1[0], p0[1] + p1[1], p0[2] + p1[2])

    @staticmethod
    def sub(p0: "Point", p1: "Point") -> "Point":
        """Returns a new point that is the difference of two points.

        Parameters
        ----------
        p0 : :class:`Point`
            First point.
        p1 : :class:`Point`
            Second point.

        Returns
        -------
        :class:`Point`
            A new Point with coordinates (p0.x - p1.x, p0.y - p1.y, p0.z - p1.z).

        """
        return Point(p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2])

    ###########################################################################################
    # Coordinate Properties
    ###########################################################################################

    @property
    def x(self) -> float:
        """Get the X coordinate."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        """Set the X coordinate."""
        self._x = value

    @property
    def y(self) -> float:
        """Get the Y coordinate."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        """Set the Y coordinate."""
        self._y = value

    @property
    def z(self) -> float:
        """Get the Z coordinate."""
        return self._z

    @z.setter
    def z(self, value: float) -> None:
        """Set the Z coordinate."""
        self._z = value

    ###########################################################################################
    # No-copy Operators
    ###########################################################################################

    def __getitem__(self, index):
        if index == 0:
            return self._x
        elif index == 1:
            return self._y
        elif index == 2:
            return self._z
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        if index == 0:
            self._x = value
        elif index == 1:
            self._y = value
        elif index == 2:
            self._z = value
        else:
            raise IndexError("Index out of range")

    def __imul__(self, other):
        self._x *= other
        self._y *= other
        self._z *= other
        return self

    def __itruediv__(self, other):
        self._x /= other
        self._y /= other
        self._z /= other
        return self

    def __iadd__(self, other):
        if isinstance(other, Vector):
            self._x += other[0]
            self._y += other[1]
            self._z += other[2]
        else:
            raise TypeError("Point can only be added with Vector")
        return self

    def __isub__(self, other):
        if isinstance(other, Vector):
            self._x -= other[0]
            self._y -= other[1]
            self._z -= other[2]
        else:
            raise TypeError("Point can only be subtracted with Vector")
        return self

    ###########################################################################################
    # Copy Operators
    ###########################################################################################

    def __mul__(self, other):
        return Point(self[0] * other, self[1] * other, self[2] * other)

    def __truediv__(self, other):
        return Point(self[0] / other, self[1] / other, self[2] / other)

    def __add__(self, other):
        return Point(self[0] + other[0], self[1] + other[1], self[2] + other[2])

    def __sub__(self, other):
        return Vector(self[0] - other[0], self[1] - other[1], self[2] - other[2])

    ###########################################################################################
    # Transformation
    ###########################################################################################

    def transform(self, xform: "Xform") -> None:
        """Apply a transformation to the point coordinates, in place."""
        x, y, z = self[0], self[1], self[2]
        m = xform.m
        w = m[3]*x + m[7]*y + m[11]*z + m[15]
        w_inv = 1.0 / w if abs(w) > 1e-10 else 1.0
        self[0] = (m[0]*x + m[4]*y + m[8]*z + m[12]) * w_inv
        self[1] = (m[1]*x + m[5]*y + m[9]*z + m[13]) * w_inv
        self[2] = (m[2]*x + m[6]*y + m[10]*z + m[14]) * w_inv

    def transformed(self, xform: "Xform") -> "Point":
        """Return a transformed copy of the point, leaving the original unchanged.

        Returns
        -------
        Point
            A new transformed point.
        """

        result = copy.deepcopy(self)
        result.transform(xform)
        return result

    ###########################################################################################
    # Details
    ###########################################################################################

    @staticmethod
    def is_ccw(a: "Point", b: "Point", c: "Point") -> bool:
        """Check if the points are in counter-clockwise order on xy plane.

        Parameters
        ----------
        a : :class:`Point`
            First point.
        b : :class:`Point`
            Second point.
        c : :class:`Point`
            Third point.

        Returns
        -------
        bool
            True if the points are in counter-clockwise order, False otherwise.

        """

        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def mid_point(self, p: "Point") -> "Point":
        """Calculate the mid point between this point and another point.

        Parameters
        ----------
        p : :class:`Point`
            The other point.

        Returns
        -------
        :class:`Point`
            The mid point between this point and the other point.

        """

        return Point((self[0] + p[0]) / 2, (self[1] + p[1]) / 2, (self[2] + p[2]) / 2)

    def distance(self, p: "Point", double_min: float = 1e-12) -> float:
        """Calculate the distance between this point and another point.

        Parameters
        ----------
        p : :class:`Point`
            The other point.
        double_min : float, optional
            The minimum value for the distance. Defaults to 1e-12.

        Returns
        -------
        float
            The distance between this point and the other point.

        """

        x = abs(self[0] - p[0])
        y = abs(self[1] - p[1])
        z = abs(self[2] - p[2])
        length = 0.0

        if y >= x and y >= z:
            length, x, y = x, y, x
        elif z >= x and z >= y:
            length, x, z = x, z, x

        if x > double_min:
            y /= x
            z /= x
            length = x * math.sqrt(1.0 + y * y + z * z)
        # For "almost zero" distances you approximate:
        elif x > 0.0 and math.isfinite(x):
            length = x
        else:
            length = 0.0

        return length

    def squared_distance(self, p: "Point", double_min: float = 1e-12) -> float:
        """Calculate the squared distance between this point and another point.

        Parameters
        ----------
        p : :class:`Point`
            The other point.
        double_min : float, optional
            The minimum value for the distance. Defaults to 1e-12.

        Returns
        -------
        float
            The distance between this point and the other point.

        """

        x = abs(self[0] - p[0])
        y = abs(self[1] - p[1])
        z = abs(self[2] - p[2])
        length = 0.0

        if y >= x and y >= z:
            length, x, y = x, y, x
        elif z >= x and z >= y:
            length, x, z = x, z, x

        if x > double_min:
            y /= x
            z /= x
            length = x * x * (1.0 + y * y + z * z)
        # For "almost zero" distances you approximate:
        elif x > 0.0 and math.isfinite(x):
            length = x * x
        else:
            length = 0.0

        return length

    @staticmethod
    def area(points: list["Point"]) -> float:
        """Calculate the area of a 2d polygon.

        Parameters
        ----------
        points : list[:class:`Point`]
            The points of the polygon.

        Returns
        -------
        float
            The area of the polygon.

        """

        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        return abs(area) / 2.0

    @staticmethod
    def centroid_quad(vertices: list["Point"]) -> "Point":
        """Calculate the centroid of a quadrilateral.

        Parameters
        ----------
        vertices : list[:class:`Point`]
            The vertices of the quadrilateral.

        Returns
        -------
        :class:`Point`
            The centroid of the quadrilateral.

        """

        if len(vertices) != 4:
            raise ValueError("Polygon must have exactly 4 vertices.")

        total_area = 0.0
        centroid_sum = Vector(0, 0, 0)

        for i in range(4):
            p0, p1, p2 = vertices[i], vertices[(i + 1) % 4], vertices[(i + 2) % 4]
            tri_area = (
                abs(
                    p0[0] * (p1[1] - p2[1])
                    + p1[0] * (p2[1] - p0[1])
                    + p2[0] * (p0[1] - p1[1])
                )
                / 2.0
            )
            total_area += tri_area
            tri_centroid = Vector(
                (p0[0] + p1[0] + p2[0]) / 3.0,
                (p0[1] + p1[1] + p2[1]) / 3.0,
                (p0[2] + p1[2] + p2[2]) / 3.0,
            )
            centroid_sum += tri_centroid * tri_area

        result = centroid_sum / total_area
        return Point(result[0], result[1], result[2])

    @staticmethod
    def centroid(points: list["Point"]) -> "Point":
        """Average of an arbitrary list of points.

        Parameters
        ----------
        points : list[:class:`Point`]

        Returns
        -------
        :class:`Point`
        """
        if not points:
            return Point(0, 0, 0)
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
    def dihedral_angle_deg(p: "Point", q: "Point", r: "Point", s: "Point") -> float:
        """Approximate dihedral angle in degrees between half-planes (p,q,r) and (p,q,s).

        Half-plane normals are built via the shared edge ``pq``.

        Parameters
        ----------
        p, q, r, s : :class:`Point`

        Returns
        -------
        float
        """
        pq = Vector(q[0] - p[0], q[1] - p[1], q[2] - p[2])
        pr = Vector(r[0] - p[0], r[1] - p[1], r[2] - p[2])
        ps = Vector(s[0] - p[0], s[1] - p[1], s[2] - p[2])
        n1 = pq.cross(pr)
        n2 = pq.cross(ps)
        m1 = n1.magnitude()
        m2 = n2.magnitude()
        if m1 < Tolerance.ZERO_TOLERANCE or m2 < Tolerance.ZERO_TOLERANCE:
            return 0.0
        cos_t = n1.dot(n2) / (m1 * m2)
        if cos_t > 1.0:
            cos_t = 1.0
        if cos_t < -1.0:
            cos_t = -1.0
        return math.acos(cos_t) * (180.0 / 3.141592653589793)

    ###########################################################################################
    # JSON Serialization
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field.

        Returns
        -------
        dict
            Dictionary with 'type', 'guid', 'name', and object fields.

        """
        # Alphabetical order to match Rust's serde_json
        return {
            "guid": self.guid,
            "name": self.name,
            "pointcolor": self.pointcolor.__jsondump__(),
            "type": f"{self.__class__.__name__}",
            "width": self.width,
            "x": self[0],
            "y": self[1],
            "z": self[2],
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format."""
        from .file_encoders import file_decode_node

        pt = cls(data["x"], data["y"], data["z"])
        pt.width = data.get("width", 1.0)

        # Decode nested color (supports polymorphic dicts and plain values)
        pt.pointcolor = file_decode_node(data.get("pointcolor"))

        # Always assign metadata (per project convention)
        pt.guid = guid if guid is not None else data.get("guid", pt.guid)
        pt.name = name if name is not None else data.get("name", pt.name)

        return pt

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write JSON to file.

        Parameters
        ----------
        filepath : str or Path
            Path to the output file.

        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "Point":
        """Read JSON from file.

        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file.

        Returns
        -------
        :class:`Point`
            The deserialized Point.

        """
        import json
        with open(filepath) as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def file_json_dumps(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Point":
        """Load from JSON string."""
        import json
        return cls.__jsonload__(json.loads(json_string))

    ###########################################################################################
    # Protobuf Serialization
    ###########################################################################################

    def pb_dumps(self) -> bytes:
        """Convert to protobuf binary format.

        Returns
        -------
        bytes
            Serialized protobuf data.

        """
        from .proto import point_pb2
        
        proto = point_pb2.Point()
        proto.guid = self.guid
        proto.name = self.name
        proto.x = self[0]
        proto.y = self[1]
        proto.z = self[2]
        proto.width = self.width
        
        # Set color (no guid in proto schema)
        proto.pointcolor.name = self.pointcolor.name
        proto.pointcolor.r = self.pointcolor[0]
        proto.pointcolor.g = self.pointcolor[1]
        proto.pointcolor.b = self.pointcolor[2]
        proto.pointcolor.a = self.pointcolor[3]
        
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Point":
        """Create Point from protobuf binary data.

        Parameters
        ----------
        data : bytes
            Protobuf-encoded point data.

        Returns
        -------
        :class:`Point`
            The deserialized Point.

        """
        from .proto import point_pb2
        from .color import Color
        
        proto = point_pb2.Point()
        proto.ParseFromString(data)
        
        pt = cls(proto.x, proto.y, proto.z)
        pt.guid = proto.guid
        pt.name = proto.name
        pt.width = proto.width
        
        # Load color (no guid in proto schema)
        pt.pointcolor = Color(
            proto.pointcolor.r,
            proto.pointcolor.g,
            proto.pointcolor.b,
            proto.pointcolor.a
        )
        pt.pointcolor.name = proto.pointcolor.name
        
        return pt

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write protobuf to file.

        Parameters
        ----------
        filepath : str
            Path to the output file.

        """
        data = self.pb_dumps()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "Point":
        """Read protobuf from file.

        Parameters
        ----------
        filepath : str
            Path to the protobuf file.

        Returns
        -------
        :class:`Point`
            The deserialized Point.

        """
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)

    def __str__(self):
        return f"{self[0]}, {self[1]}, {self[2]}"

    def __repr__(self):
        return f"Point({self.name}, {self[0]}, {self[1]}, {self[2]}, {repr(self.pointcolor)}, {self.width})"

    def __eq__(self, other):
        return (
            self.name == other.name
            and round(self[0], Tolerance.ROUNDING) == round(other[0], Tolerance.ROUNDING)
            and round(self[1], Tolerance.ROUNDING) == round(other[1], Tolerance.ROUNDING)
            and round(self[2], Tolerance.ROUNDING) == round(other[2], Tolerance.ROUNDING)
            and round(self.width, Tolerance.ROUNDING) == round(other.width, Tolerance.ROUNDING)
            and self.pointcolor == other.pointcolor
        )

    def __ne__(self, other):
        return not self == other
