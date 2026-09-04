from __future__ import annotations
from typing import List
from typing import Optional
from typing import Union
from typing import TYPE_CHECKING
import uuid
import copy

from .color import Color
from .point import Point
from .vector import Vector

if TYPE_CHECKING:
    from pathlib import Path
    from .xform import Xform


class PointCloud:
    """A point cloud with coordinates, normals, and colors stored as flat arrays."""

    def __init__(self, points: list[Point] | None = None,
                 normals: list[Vector] | None = None,
                 colors: list[Color] | None = None):
        """Default constructor (empty cloud) or with points, normals, and colors."""
        self._guid = None
        self.name = "my_pointcloud"
        self.point_size = 1.0

        # Store as flat arrays
        self.coords: list[float] = []
        self._colors: list[int] = []   # flat 0-255, matching Rust/C++ and the proto's uint32
        self._normals: list[float] = []
        # LOD octree over the points, one flat list per SpatialOctree node field. Built by
        # build_lod(), which PERMUTES the three lists above into octree order - so a node is one
        # contiguous (_lod_first, _lod_count) range and the order permutation never has to be
        # stored. Empty means no octree.
        self._lod_min: list[float] = []
        self._lod_size: list[float] = []
        self._lod_spacing: list[float] = []
        self._lod_level: list[int] = []
        self._lod_first: list[int] = []
        self._lod_count: list[int] = []
        self._lod_children: list[int] = []
        # STABLE per-point ids, one per point, parallel to coords. Assigned once by the first
        # build_lod and permuted with the points ever after, so an index that moves does not take
        # the point's identity with it. Empty = no tree yet, so the index IS the id.
        self._point_ids: list[int] = []

        if points is not None:
            for p in points:
                self.coords.extend([p[0], p[1], p[2]])

        if colors is not None:
            for c in colors:
                self._colors.extend([round(c[0] * 255), round(c[1] * 255),
                                     round(c[2] * 255), round(c[3] * 255)])

        if normals is not None:
            for n in normals:
                self._normals.extend([n[0], n[1], n[2]])

    @property
    def guid(self) -> str:
        """Lazy GUID accessor."""
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    @classmethod
    def from_coords(cls, coords: list[float],
                    colors: list[int] | None = None,
                    normals: list[float] | None = None) -> PointCloud:
        """Create from flat arrays of coords, colors, and normals."""
        pc = cls()
        pc.coords = list(coords)
        if colors is not None:
            pc._colors = list(colors)
        if normals is not None:
            pc._normals = list(normals)
        return pc

    # ═══════════════════════════════════════════════════════════════════════════
    # Point Access (compatibility layer)
    # ═══════════════════════════════════════════════════════════════════════════

    def point_count(self) -> int:
        """Returns the number of points."""
        return len(self.coords) // 3

    def __len__(self) -> int:
        """Returns the number of points."""
        return self.point_count()

    def is_empty(self) -> bool:
        """Returns true if the point cloud has no points."""
        return self.point_count() == 0

    def get_point(self, index: int) -> Point:
        """Get point at index as Point object."""
        idx = index * 3
        return Point(self.coords[idx], self.coords[idx + 1], self.coords[idx + 2])

    def set_point(self, index: int, point: Point) -> None:
        """Set point at index from Point object."""
        idx = index * 3
        self.coords[idx] = point[0]
        self.coords[idx + 1] = point[1]
        self.coords[idx + 2] = point[2]

    def add_point(self, point: Point) -> None:
        """Add a point to the cloud."""
        self.coords.extend([point[0], point[1], point[2]])

    def get_points(self) -> list[Point]:
        """Returns all points as Point objects."""
        points = []
        for i in range(self.point_count()):
            idx = i * 3
            points.append(Point(self.coords[idx], self.coords[idx + 1], self.coords[idx + 2]))
        return points

    @property
    def points(self) -> list[Point]:
        """Property for backward compatibility - returns list of Point objects."""
        return self.get_points()

    @points.setter
    def points(self, value: list[Point]) -> None:
        """Set points from a list of Point objects."""
        self.coords = []
        for p in value:
            self.coords.extend([p[0], p[1], p[2]])

    # ═══════════════════════════════════════════════════════════════════════════
    # Color Access
    # ═══════════════════════════════════════════════════════════════════════════

    def color_count(self) -> int:
        """Returns the number of colors."""
        return len(self._colors) // 4

    def get_color(self, index: int) -> Color:
        """Get color at index as Color object."""
        idx = index * 4
        return Color(self._colors[idx] / 255.0, self._colors[idx + 1] / 255.0,
                     self._colors[idx + 2] / 255.0, self._colors[idx + 3] / 255.0)

    def set_color(self, index: int, color: Color) -> None:
        """Set color at index from Color object."""
        idx = index * 4
        self._colors[idx] = round(color[0] * 255)
        self._colors[idx + 1] = round(color[1] * 255)
        self._colors[idx + 2] = round(color[2] * 255)
        self._colors[idx + 3] = round(color[3] * 255)

    def add_color(self, color: Color) -> None:
        """Add a color to the cloud."""
        self._colors.extend([round(color[0] * 255), round(color[1] * 255),
                             round(color[2] * 255), round(color[3] * 255)])

    def get_colors(self) -> list[Color]:
        """Returns all colors as Color objects."""
        colors = []
        for i in range(self.color_count()):
            idx = i * 4
            colors.append(Color(self._colors[idx] / 255.0, self._colors[idx + 1] / 255.0,
                                self._colors[idx + 2] / 255.0, self._colors[idx + 3] / 255.0))
        return colors

    @property
    def colors(self) -> list[int]:
        """The flat colour array itself, [r0, g0, b0, a0, r1, ...] as 0-255 - the same encoding
        the proto carries, and the same accessor Rust and C++ expose. `get_colors()` is the one
        that builds Color objects; walking millions of points cannot afford that per point."""
        return self._colors

    @colors.setter
    def colors(self, value: list[Color]) -> None:
        """Set colors from a list of Color objects."""
        self._colors = []
        for c in value:
            self._colors.extend([round(c.r * 255), round(c.g * 255),
                                 round(c.b * 255), round(c.a * 255)])

    # ═══════════════════════════════════════════════════════════════════════════
    # Normal Access
    # ═══════════════════════════════════════════════════════════════════════════

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

    def get_normals(self) -> list[Vector]:
        """Returns all normals as Vector objects."""
        normals = []
        for i in range(self.normal_count()):
            idx = i * 3
            normals.append(Vector(self._normals[idx], self._normals[idx + 1], self._normals[idx + 2]))
        return normals

    @property
    def normals(self) -> list[Vector]:
        """Property for backward compatibility."""
        return self.get_normals()

    @normals.setter
    def normals(self, value: list[Vector]) -> None:
        """Set normals from a list of Vector objects."""
        self._normals = []
        for n in value:
            self._normals.extend([n[0], n[1], n[2]])

    # ═══════════════════════════════════════════════════════════════════════════
    # LOD Octree
    # ═══════════════════════════════════════════════════════════════════════════

    def build_lod(self, root_spacing: float, leaf_capacity: int) -> None:
        """Build the LOD octree and REORDER the points into octree order.

        Every point's index changes; a node becomes one contiguous range. Expensive - about
        10 s on 14 M points - so it is called once by whoever writes the cloud, never per
        construction.
        """
        from .spatial_octree import SpatialOctree

        tree = SpatialOctree.from_coords(self.coords, root_spacing, leaf_capacity)
        order = tree.order()

        # Identity is minted HERE, before the first permutation, so an id records where a point
        # began. After this the ids travel with the points and the index is free to move.
        if not self._point_ids:
            self._point_ids = list(range(len(self.coords) // 3))

        # Permute the three parallel lists into octree order. This is what lets a node be one
        # (first, count) range, so `order` itself never has to be stored - 4 bytes a point.
        coords: list[float] = []
        colors: list[int] = []
        normals: list[float] = []
        ids: list[int] = []
        has_colors = len(self._colors) == len(order) * 4
        has_normals = len(self._normals) == len(order) * 3
        for idx in order:
            ids.append(self._point_ids[idx])
            coords.extend(self.coords[idx * 3:idx * 3 + 3])
            if has_colors:
                colors.extend(self._colors[idx * 4:idx * 4 + 4])
            if has_normals:
                normals.extend(self._normals[idx * 3:idx * 3 + 3])
        self.coords = coords
        self._point_ids = ids
        if has_colors:
            self._colors = colors
        if has_normals:
            self._normals = normals

        self._lod_min = []
        self._lod_size = []
        self._lod_spacing = []
        self._lod_level = []
        self._lod_first = []
        self._lod_count = []
        self._lod_children = []
        for i in range(tree.node_count()):
            center, size = tree.node_cube(i)
            self._lod_min.extend([center[0] - size * 0.5, center[1] - size * 0.5, center[2] - size * 0.5])
            self._lod_size.append(size)
            self._lod_spacing.append(tree.node_spacing(i))
            self._lod_level.append(tree.node_level(i))
            first, count = tree.node_range(i)
            self._lod_first.append(first)
            self._lod_count.append(count)
            kids = tree.children(i)
            self._lod_children.extend([kids[k] if k < len(kids) else -1 for k in range(8)])

    def has_lod(self) -> bool:
        """True when an octree has been built."""
        return len(self._lod_size) > 0

    def lod_node_count(self) -> int:
        """Number of octree nodes."""
        return len(self._lod_size)

    def lod_cube(self, i: int) -> tuple[Point, float]:
        """Node cube: center and edge length."""
        half = self._lod_size[i] * 0.5
        return (Point(self._lod_min[i * 3] + half, self._lod_min[i * 3 + 1] + half,
                      self._lod_min[i * 3 + 2] + half), self._lod_size[i])

    def lod_spacing(self, i: int) -> float:
        """Grid-accept spacing of a node."""
        return self._lod_spacing[i]

    def lod_level(self, i: int) -> int:
        """Node depth from the root."""
        return self._lod_level[i]

    def lod_range(self, i: int) -> tuple[int, int]:
        """Node point range as (first, count) into the reordered lists."""
        return (self._lod_first[i], self._lod_count[i])

    def lod_children(self, i: int) -> list[int]:
        """Present child node indices, compacted, -1 padding."""
        return self._lod_children[i * 8:i * 8 + 8]

    # ═══════════════════════════════════════════════════════════════════════════
    # Stable Point Ids
    # ═══════════════════════════════════════════════════════════════════════════

    def point_ids(self) -> list[int]:
        """The stable ids, parallel to the points. Empty until a tree is built."""
        return self._point_ids

    def point_id(self, index: int) -> int:
        """The stable id of a point, by its CURRENT index.

        Falls back to the index itself while no tree has been built, which is exactly what the
        id would have been.
        """
        return index if not self._point_ids else self._point_ids[index]

    def index_of_id(self, id: int) -> int:
        """Where a stable id lives NOW, or -1 if this cloud has no such point.

        Linear: a caller resolving many ids should build its own map.
        """
        if not self._point_ids:
            return id if 0 <= id < len(self.coords) // 3 else -1
        try:
            return self._point_ids.index(id)
        except ValueError:
            return -1

    # ═══════════════════════════════════════════════════════════════════════════
    # String Representations
    # ═══════════════════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════════════════
    # Duplicate and Equality
    # ═══════════════════════════════════════════════════════════════════════════

    def duplicate(self) -> PointCloud:
        """Create a deep copy with a new GUID."""
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def __eq__(self, other) -> bool:
        """Equality comparison (ignores guid)."""
        if not isinstance(other, PointCloud):
            return False
        return (self.name == other.name and
                self.coords == other.coords and
                self._colors == other._colors and
                self._normals == other._normals and
                self._lod_first == other._lod_first and
                self._lod_count == other._lod_count and
                self._point_ids == other._point_ids)

    # ═══════════════════════════════════════════════════════════════════════════
    # Transform
    # ═══════════════════════════════════════════════════════════════════════════

    def transform(self, xform: Xform) -> None:
        """Apply a transformation to the point cloud in-place."""
        for i in range(self.point_count()):
            idx = i * 3
            pt = Point(self.coords[idx], self.coords[idx + 1], self.coords[idx + 2])
            pt.transform(xform)
            self.coords[idx] = pt[0]
            self.coords[idx + 1] = pt[1]
            self.coords[idx + 2] = pt[2]

        for i in range(self.normal_count()):
            idx = i * 3
            n = Vector(self._normals[idx], self._normals[idx + 1], self._normals[idx + 2])
            n.transform(xform)
            self._normals[idx] = n[0]
            self._normals[idx + 1] = n[1]
            self._normals[idx + 2] = n[2]

    def transformed(self, xform: Xform) -> PointCloud:
        """Return a transformed copy of the point cloud."""
        result = copy.deepcopy(self)
        result.transform(xform)
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # No-copy Operators
    # ═══════════════════════════════════════════════════════════════════════════

    def __iadd__(self, other: Vector) -> PointCloud:
        """Translate point cloud by vector (in-place)."""
        for i in range(self.point_count()):
            idx = i * 3
            self.coords[idx] += other[0]
            self.coords[idx + 1] += other[1]
            self.coords[idx + 2] += other[2]
        return self

    def __isub__(self, other: Vector) -> PointCloud:
        """Translate point cloud by negative vector (in-place)."""
        for i in range(self.point_count()):
            idx = i * 3
            self.coords[idx] -= other[0]
            self.coords[idx + 1] -= other[1]
            self.coords[idx + 2] -= other[2]
        return self

    # ═══════════════════════════════════════════════════════════════════════════
    # Copy Operators
    # ═══════════════════════════════════════════════════════════════════════════

    def __add__(self, other: Vector) -> PointCloud:
        """Translate point cloud by vector (copy)."""
        result = self.duplicate()
        result.guid = self.guid  # Keep same guid for copy operators
        result += other
        return result

    def __sub__(self, other: Vector) -> PointCloud:
        """Translate point cloud by negative vector (copy)."""
        result = self.duplicate()
        result.guid = self.guid  # Keep same guid for copy operators
        result -= other
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON Serialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field."""
        # Alphabetical order to match Rust's serde_json
        return {
            "colors": self._colors,
            "coords": self.coords,
            "guid": self.guid,
            "lod_children": self._lod_children,
            "lod_count": self._lod_count,
            "lod_first": self._lod_first,
            "lod_level": self._lod_level,
            "lod_min": self._lod_min,
            "lod_size": self._lod_size,
            "lod_spacing": self._lod_spacing,
            "name": self.name,
            "normals": self._normals,
            "point_ids": self._point_ids,
            "point_size": self.point_size,
            "type": f"{self.__class__.__name__}",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format."""
        from .file_encoders import file_decode_node

        pc = cls.from_coords(
            data.get("coords", []),
            data.get("colors", []),
            data.get("normals", [])
        )
        pc.guid = guid if guid is not None else data.get("guid", pc.guid)
        pc.name = name if name is not None else data.get("name", pc.name)

        if "point_size" in data:
            pc.point_size = data["point_size"]
        pc._lod_min = data.get("lod_min", [])
        pc._lod_size = data.get("lod_size", [])
        pc._lod_spacing = data.get("lod_spacing", [])
        pc._lod_level = data.get("lod_level", [])
        pc._lod_first = data.get("lod_first", [])
        pc._lod_count = data.get("lod_count", [])
        pc._lod_children = data.get("lod_children", [])
        pc._point_ids = data.get("point_ids", [])

        return pc

    def file_json_dump(self, filepath: str | Path) -> None:
        """Write JSON to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: str | Path) -> PointCloud:
        """Read JSON from file."""
        import json
        with open(filepath) as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def file_json_dumps(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> PointCloud:
        """Load from JSON string."""
        import json
        return cls.__jsonload__(json.loads(json_string))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf Serialization
    # ═══════════════════════════════════════════════════════════════════════════

    def pb_dumps(self) -> bytes:
        """Convert to protobuf binary format."""
        from .proto import pointcloud_pb2

        proto = pointcloud_pb2.PointCloud()
        proto.guid = self.guid
        proto.name = self.name
        proto.coords.extend(self.coords)
        proto.colors.extend(self._colors)
        proto.normals.extend(self._normals)
        proto.point_size = self.point_size
        proto.lod_min.extend(self._lod_min)
        proto.lod_size.extend(self._lod_size)
        proto.lod_spacing.extend(self._lod_spacing)
        proto.lod_level.extend(self._lod_level)
        proto.lod_first.extend(self._lod_first)
        proto.lod_count.extend(self._lod_count)
        proto.lod_children.extend(self._lod_children)
        proto.point_ids.extend(self._point_ids)

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> PointCloud:
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
        pc._lod_min = list(proto.lod_min)
        pc._lod_size = list(proto.lod_size)
        pc._lod_spacing = list(proto.lod_spacing)
        pc._lod_level = list(proto.lod_level)
        pc._lod_first = list(proto.lod_first)
        pc._lod_count = list(proto.lod_count)
        pc._lod_children = list(proto.lod_children)
        pc._point_ids = list(proto.point_ids)

        return pc

    def pb_dump(self, filepath: str | Path) -> None:
        """Write protobuf to file."""
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: str | Path) -> PointCloud:
        """Read protobuf from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)
