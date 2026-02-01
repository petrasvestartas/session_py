import uuid
import copy
from typing import List, Optional

from .color import Color
from .point import Point
from .vector import Vector
from .xform import Xform


class PointCloud:
    """A point cloud with coordinates, normals, and colors stored as flat arrays.

    Internally stores data as flat arrays for efficient serialization:
    - coords: [x0, y0, z0, x1, y1, z1, ...]
    - colors: [r0, g0, b0, a0, r1, g1, b1, a1, ...]
    - normals: [nx0, ny0, nz0, nx1, ny1, nz1, ...]
    """

    def __init__(self, points: Optional[List[Point]] = None,
                 normals: Optional[List[Vector]] = None,
                 colors: Optional[List[Color]] = None):
        """Creates a new PointCloud with default guid and name.

        Args:
            points: Collection of points (converted to flat coords internally).
            normals: Collection of normals (converted to flat array internally).
            colors: Collection of colors (converted to flat array internally).
        """
        self.guid = str(uuid.uuid4())
        self.name = "my_pointcloud"
        self.point_size = 1.0
        self.xform = Xform.identity()

        # Store as flat arrays
        self._coords: List[float] = []
        self._colors: List[int] = []
        self._normals: List[float] = []

        if points is not None:
            for p in points:
                self._coords.extend([p[0], p[1], p[2]])

        if colors is not None:
            for c in colors:
                self._colors.extend([c[0], c[1], c[2], c[3]])

        if normals is not None:
            for n in normals:
                self._normals.extend([n[0], n[1], n[2]])

    @classmethod
    def from_coords(cls, coords: List[float],
                    colors: Optional[List[int]] = None,
                    normals: Optional[List[float]] = None) -> "PointCloud":
        """Create a PointCloud from flat arrays.

        Args:
            coords: Flat array [x0, y0, z0, x1, y1, z1, ...]
            colors: Flat array [r0, g0, b0, a0, r1, g1, b1, a1, ...]
            normals: Flat array [nx0, ny0, nz0, nx1, ny1, nz1, ...]

        Returns:
            New PointCloud instance.
        """
        pc = cls()
        pc._coords = list(coords)
        if colors is not None:
            pc._colors = list(colors)
        if normals is not None:
            pc._normals = list(normals)
        return pc

    ###########################################################################################
    # Point Access (compatibility layer)
    ###########################################################################################

    def point_count(self) -> int:
        """Returns the number of points."""
        return len(self._coords) // 3

    def __len__(self) -> int:
        """Returns the number of points."""
        return self.point_count()

    def is_empty(self) -> bool:
        """Returns true if the point cloud has no points."""
        return self.point_count() == 0

    def get_point(self, index: int) -> Point:
        """Get point at index as Point object."""
        idx = index * 3
        return Point(self._coords[idx], self._coords[idx + 1], self._coords[idx + 2])

    def set_point(self, index: int, point: Point) -> None:
        """Set point at index from Point object."""
        idx = index * 3
        self._coords[idx] = point[0]
        self._coords[idx + 1] = point[1]
        self._coords[idx + 2] = point[2]

    def add_point(self, point: Point) -> None:
        """Add a point to the cloud."""
        self._coords.extend([point[0], point[1], point[2]])

    def get_points(self) -> List[Point]:
        """Returns all points as Point objects."""
        points = []
        for i in range(self.point_count()):
            idx = i * 3
            points.append(Point(self._coords[idx], self._coords[idx + 1], self._coords[idx + 2]))
        return points

    @property
    def points(self) -> List[Point]:
        """Property for backward compatibility - returns list of Point objects."""
        return self.get_points()

    @points.setter
    def points(self, value: List[Point]) -> None:
        """Set points from a list of Point objects."""
        self._coords = []
        for p in value:
            self._coords.extend([p[0], p[1], p[2]])

    ###########################################################################################
    # Color Access
    ###########################################################################################

    def color_count(self) -> int:
        """Returns the number of colors."""
        return len(self._colors) // 4

    def get_color(self, index: int) -> Color:
        """Get color at index as Color object."""
        idx = index * 4
        return Color(self._colors[idx], self._colors[idx + 1],
                     self._colors[idx + 2], self._colors[idx + 3])

    def set_color(self, index: int, color: Color) -> None:
        """Set color at index from Color object."""
        idx = index * 4
        self._colors[idx] = color[0]
        self._colors[idx + 1] = color[1]
        self._colors[idx + 2] = color[2]
        self._colors[idx + 3] = color[3]

    def add_color(self, color: Color) -> None:
        """Add a color to the cloud."""
        self._colors.extend([color[0], color[1], color[2], color[3]])

    def get_colors(self) -> List[Color]:
        """Returns all colors as Color objects."""
        colors = []
        for i in range(self.color_count()):
            idx = i * 4
            colors.append(Color(self._colors[idx], self._colors[idx + 1],
                                self._colors[idx + 2], self._colors[idx + 3]))
        return colors

    @property
    def colors(self) -> List[Color]:
        """Property for backward compatibility."""
        return self.get_colors()

    @colors.setter
    def colors(self, value: List[Color]) -> None:
        """Set colors from a list of Color objects."""
        self._colors = []
        for c in value:
            self._colors.extend([c.r, c.g, c.b, c.a])

    ###########################################################################################
    # Normal Access
    ###########################################################################################

    def normal_count(self) -> int:
        """Returns the number of normals."""
        return len(self._normals) // 3

    def get_normal(self, index: int) -> Vector:
        """Get normal at index as Vector object."""
        idx = index * 3
        return Vector(self._normals[idx], self._normals[idx + 1], self._normals[idx + 2])

    def set_normal(self, index: int, normal: Vector) -> None:
        """Set normal at index from Vector object."""
        idx = index * 3
        self._normals[idx] = normal[0]
        self._normals[idx + 1] = normal[1]
        self._normals[idx + 2] = normal[2]

    def add_normal(self, normal: Vector) -> None:
        """Add a normal to the cloud."""
        self._normals.extend([normal[0], normal[1], normal[2]])

    def get_normals(self) -> List[Vector]:
        """Returns all normals as Vector objects."""
        normals = []
        for i in range(self.normal_count()):
            idx = i * 3
            normals.append(Vector(self._normals[idx], self._normals[idx + 1], self._normals[idx + 2]))
        return normals

    @property
    def normals(self) -> List[Vector]:
        """Property for backward compatibility."""
        return self.get_normals()

    @normals.setter
    def normals(self, value: List[Vector]) -> None:
        """Set normals from a list of Vector objects."""
        self._normals = []
        for n in value:
            self._normals.extend([n[0], n[1], n[2]])

    ###########################################################################################
    # String Representations
    ###########################################################################################

    def __str__(self) -> str:
        """Minimal string representation."""
        return f"{self.point_count()} points"

    def __repr__(self) -> str:
        """Full string representation."""
        return f"PointCloud({self.name}, {self.point_count()} points, {self.color_count()} colors, {self.normal_count()} normals)"

    def str(self) -> str:
        """Minimal string representation."""
        return self.__str__()

    def repr(self) -> str:
        """Full string representation."""
        return self.__repr__()

    ###########################################################################################
    # Duplicate and Equality
    ###########################################################################################

    def duplicate(self) -> "PointCloud":
        """Create a deep copy with a new GUID."""
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def __eq__(self, other) -> bool:
        """Equality comparison (ignores guid)."""
        if not isinstance(other, PointCloud):
            return False
        return (self.name == other.name and
                self._coords == other._coords and
                self._colors == other._colors and
                self._normals == other._normals)

    ###########################################################################################
    # Transform
    ###########################################################################################

    def transform(self) -> None:
        """Apply the stored xform transformation to the point cloud in-place."""
        for i in range(self.point_count()):
            idx = i * 3
            pt = Point(self._coords[idx], self._coords[idx + 1], self._coords[idx + 2])
            self.xform.transform_point(pt)
            self._coords[idx] = pt[0]
            self._coords[idx + 1] = pt[1]
            self._coords[idx + 2] = pt[2]

        for i in range(self.normal_count()):
            idx = i * 3
            n = Vector(self._normals[idx], self._normals[idx + 1], self._normals[idx + 2])
            self.xform.transform_vector(n)
            self._normals[idx] = n[0]
            self._normals[idx + 1] = n[1]
            self._normals[idx + 2] = n[2]

        self.xform = Xform.identity()

    def transformed(self) -> "PointCloud":
        """Return a transformed copy of the point cloud."""
        result = copy.deepcopy(self)
        result.transform()
        return result

    ###########################################################################################
    # No-copy Operators
    ###########################################################################################

    def __iadd__(self, other: Vector) -> "PointCloud":
        """Translate point cloud by vector (in-place)."""
        for i in range(self.point_count()):
            idx = i * 3
            self._coords[idx] += other[0]
            self._coords[idx + 1] += other[1]
            self._coords[idx + 2] += other[2]
        return self

    def __isub__(self, other: Vector) -> "PointCloud":
        """Translate point cloud by negative vector (in-place)."""
        for i in range(self.point_count()):
            idx = i * 3
            self._coords[idx] -= other[0]
            self._coords[idx + 1] -= other[1]
            self._coords[idx + 2] -= other[2]
        return self

    ###########################################################################################
    # Copy Operators
    ###########################################################################################

    def __add__(self, other: Vector) -> "PointCloud":
        """Translate point cloud by vector (copy)."""
        result = self.duplicate()
        result.guid = self.guid  # Keep same guid for copy operators
        result += other
        return result

    def __sub__(self, other: Vector) -> "PointCloud":
        """Translate point cloud by negative vector (copy)."""
        result = self.duplicate()
        result.guid = self.guid  # Keep same guid for copy operators
        result -= other
        return result

    ###########################################################################################
    # JSON Serialization
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field."""
        # Alphabetical order to match Rust's serde_json
        return {
            "colors": self._colors,
            "coords": self._coords,
            "guid": self.guid,
            "name": self.name,
            "normals": self._normals,
            "point_size": self.point_size,
            "type": f"{self.__class__.__name__}",
            "xform": self.xform.__jsondump__(),
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format."""
        from .encoders import decode_node

        pc = cls.from_coords(
            data.get("coords", []),
            data.get("colors", []),
            data.get("normals", [])
        )
        pc.guid = guid if guid is not None else data.get("guid", pc.guid)
        pc.name = name if name is not None else data.get("name", pc.name)

        if "point_size" in data:
            pc.point_size = data["point_size"]
        if "xform" in data:
            pc.xform = decode_node(data["xform"])

        return pc

    def json_dump(self, filepath) -> None:
        """Write JSON to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath) -> "PointCloud":
        """Read JSON from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def json_dumps(self):
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, json_string):
        """Load from JSON string."""
        import json
        return cls.__jsonload__(json.loads(json_string))

    ###########################################################################################
    # Protobuf Serialization
    ###########################################################################################

    def pb_dumps(self) -> bytes:
        """Convert to protobuf binary format."""
        from .proto import pointcloud_pb2

        proto = pointcloud_pb2.PointCloud()
        proto.guid = self.guid
        proto.name = self.name
        proto.coords.extend(self._coords)
        proto.colors.extend(self._colors)
        proto.normals.extend(self._normals)
        proto.point_size = self.point_size

        # Serialize xform
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "PointCloud":
        """Create from protobuf binary format."""
        from .proto import pointcloud_pb2

        proto = pointcloud_pb2.PointCloud()
        proto.ParseFromString(data)

        pc = cls.from_coords(
            list(proto.coords),
            list(proto.colors),
            list(proto.normals)
        )
        pc.guid = proto.guid
        pc.name = proto.name
        pc.point_size = proto.point_size if proto.point_size > 0 else 1.0

        # Deserialize xform
        if proto.HasField("xform"):
            pc.xform.name = proto.xform.name
            for i, val in enumerate(proto.xform.matrix):
                if i < 16:
                    pc.xform.m[i] = val

        return pc

    def pb_dump(self, filepath) -> None:
        """Write protobuf to file."""
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath) -> "PointCloud":
        """Read protobuf from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)
