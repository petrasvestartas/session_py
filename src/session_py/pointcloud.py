from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union
import copy
import json
import uuid
from .color import Color
from .point import Point
from .vector import Vector

if TYPE_CHECKING:
    from pathlib import Path
    from .xform import Xform


class PointCloud:
    """A point cloud as flat coordinate, color and normal arrays with an optional LOD octree"""

    __slots__ = (
        "_guid",
        "name",
        "point_size",
        "_coords",
        "_colors",
        "_normals",
        "_lod_min",
        "_lod_size",
        "_lod_spacing",
        "_lod_level",
        "_lod_first",
        "_lod_count",
        "_lod_children",
        "_point_ids",
    )

    def __init__(self, points: list[Point] | None = None, normals: list[Vector] | None = None, colors: list[Color] | None = None):
        self._guid = None
        self.name = "my_pointcloud"
        self.point_size = 1.0
        self._coords: list[float] = []
        self._colors: list[int] = []
        self._normals: list[float] = []
        self._lod_min: list[float] = []
        self._lod_size: list[float] = []
        self._lod_spacing: list[float] = []
        self._lod_level: list[int] = []
        self._lod_first: list[int] = []
        self._lod_count: list[int] = []
        self._lod_children: list[int] = []
        self._point_ids: list[int] = []
        for p in points or []:
            self.add_point(p)
        for n in normals or []:
            self.add_normal(n)
        for c in colors or []:
            self.add_color(c)

    def __deepcopy__(self, memo):
        """Copy (new guid, same data)"""
        result = PointCloud()
        result.name = self.name
        result.point_size = self.point_size
        result._coords = list(self._coords)
        result._colors = list(self._colors)
        result._normals = list(self._normals)
        result._lod_min = list(self._lod_min)
        result._lod_size = list(self._lod_size)
        result._lod_spacing = list(self._lod_spacing)
        result._lod_level = list(self._lod_level)
        result._lod_first = list(self._lod_first)
        result._lod_count = list(self._lod_count)
        result._lod_children = list(self._lod_children)
        result._point_ids = list(self._point_ids)
        memo[id(self)] = result
        return result

    def duplicate(self) -> "PointCloud":
        """Copy (new guid, same data)"""
        return copy.deepcopy(self)

    def has_guid(self) -> bool:
        return self._guid is not None

    @property
    def guid(self) -> str:
        if self._guid is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def refresh_guid(self) -> None:
        """Clear the guid so a fresh one mints lazily on next read"""
        self._guid = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def from_coords(coords: list[float], colors: list[int] | None = None, normals: list[float] | None = None) -> "PointCloud":
        """Cloud from flat arrays: coords [x, y, z, ...], colors [r, g, b, a, ...] as 0-255, normals [nx, ny, nz, ...]"""
        cloud = PointCloud()
        cloud._coords = list(coords)
        cloud._colors = list(colors or [])
        cloud._normals = list(normals or [])
        return cloud

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════

    def __eq__(self, other) -> bool:
        """Same name, arrays, LOD ranges and point ids; guid ignored"""
        if not isinstance(other, PointCloud):
            return False
        return (
            self.name == other.name
            and self._coords == other._coords
            and self._colors == other._colors
            and self._normals == other._normals
            and self._lod_first == other._lod_first
            and self._lod_count == other._lod_count
            and self._point_ids == other._point_ids
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __iadd__(self, other: Vector) -> "PointCloud":
        for i in range(0, len(self._coords), 3):
            self._coords[i] += other[0]
            self._coords[i + 1] += other[1]
            self._coords[i + 2] += other[2]
        return self

    def __isub__(self, other: Vector) -> "PointCloud":
        for i in range(0, len(self._coords), 3):
            self._coords[i] -= other[0]
            self._coords[i + 1] -= other[1]
            self._coords[i + 2] -= other[2]
        return self

    def __add__(self, other: Vector) -> "PointCloud":
        result = copy.deepcopy(self)
        result += other
        return result

    def __sub__(self, other: Vector) -> "PointCloud":
        result = copy.deepcopy(self)
        result -= other
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════

    def transform(self, xform: "Xform") -> None:
        """Transform points and normals in place"""
        for i in range(self.point_count()):
            self.set_point(i, self.get_point(i).transformed(xform))
        for i in range(self.normal_count()):
            self.set_normal(i, self.get_normal(i).transformed(xform))

    def transformed(self, xform: "Xform") -> "PointCloud":
        """Transformed copy"""
        result = copy.deepcopy(self)
        result.transform(xform)
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Points
    # ═══════════════════════════════════════════════════════════════════════════

    def point_count(self) -> int:
        return len(self._coords) // 3

    def __len__(self) -> int:
        return self.point_count()

    def is_empty(self) -> bool:
        return len(self._coords) == 0

    def get_point(self, index: int) -> Point:
        idx = index * 3
        return Point(self._coords[idx], self._coords[idx + 1], self._coords[idx + 2])

    def set_point(self, index: int, point: Point) -> None:
        idx = index * 3
        self._coords[idx] = point[0]
        self._coords[idx + 1] = point[1]
        self._coords[idx + 2] = point[2]

    def add_point(self, point: Point) -> None:
        self._coords.append(point[0])
        self._coords.append(point[1])
        self._coords.append(point[2])

    def get_points(self) -> list[Point]:
        points = []
        for i in range(self.point_count()):
            points.append(self.get_point(i))
        return points

    def coords(self) -> list[float]:
        """The flat coordinate array itself; get_point builds a Point per call"""
        return self._coords

    # ═══════════════════════════════════════════════════════════════════════════
    # Colors
    # ═══════════════════════════════════════════════════════════════════════════

    def color_count(self) -> int:
        return len(self._colors) // 4

    def get_color(self, index: int) -> Color:
        idx = index * 4
        return Color(self._colors[idx] / 255.0, self._colors[idx + 1] / 255.0, self._colors[idx + 2] / 255.0, self._colors[idx + 3] / 255.0)

    def set_color(self, index: int, color: Color) -> None:
        idx = index * 4
        self._colors[idx] = round(color.r * 255.0)
        self._colors[idx + 1] = round(color.g * 255.0)
        self._colors[idx + 2] = round(color.b * 255.0)
        self._colors[idx + 3] = round(color.a * 255.0)

    def add_color(self, color: Color) -> None:
        self._colors.append(round(color.r * 255.0))
        self._colors.append(round(color.g * 255.0))
        self._colors.append(round(color.b * 255.0))
        self._colors.append(round(color.a * 255.0))

    def get_colors(self) -> list[Color]:
        colors = []
        for i in range(self.color_count()):
            colors.append(self.get_color(i))
        return colors

    def colors(self) -> list[int]:
        """The flat 0-255 color array itself, the encoding the proto carries"""
        return self._colors

    # ═══════════════════════════════════════════════════════════════════════════
    # Normals
    # ═══════════════════════════════════════════════════════════════════════════

    def normal_count(self) -> int:
        return len(self._normals) // 3

    def get_normal(self, index: int) -> Vector:
        idx = index * 3
        return Vector(self._normals[idx], self._normals[idx + 1], self._normals[idx + 2])

    def set_normal(self, index: int, normal: Vector) -> None:
        idx = index * 3
        self._normals[idx] = normal[0]
        self._normals[idx + 1] = normal[1]
        self._normals[idx + 2] = normal[2]

    def add_normal(self, normal: Vector) -> None:
        self._normals.append(normal[0])
        self._normals.append(normal[1])
        self._normals.append(normal[2])

    def get_normals(self) -> list[Vector]:
        normals = []
        for i in range(self.normal_count()):
            normals.append(self.get_normal(i))
        return normals

    # ═══════════════════════════════════════════════════════════════════════════
    # LOD octree
    # ═══════════════════════════════════════════════════════════════════════════

    def build_lod(self, root_spacing: float, leaf_capacity: int) -> None:
        """Build the octree and permute the arrays into octree order, so a node is one contiguous range"""
        from .spatial_octree import SpatialOctree

        tree = SpatialOctree.from_coords(self._coords, root_spacing, leaf_capacity)
        order = tree.order()
        if not self._point_ids:
            for i in range(self.point_count()):
                self._point_ids.append(i)

        has_colors = len(self._colors) == len(order) * 4
        has_normals = len(self._normals) == len(order) * 3
        coords: list[float] = []
        colors: list[int] = []
        normals: list[float] = []
        ids: list[int] = []
        for idx in order:
            ids.append(self._point_ids[idx])
            for k in range(3):
                coords.append(self._coords[idx * 3 + k])
            if has_colors:
                for k in range(4):
                    colors.append(self._colors[idx * 4 + k])
            if has_normals:
                for k in range(3):
                    normals.append(self._normals[idx * 3 + k])
        self._coords = coords
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
            cube = tree.node_cube(i)
            range_ = tree.node_range(i)
            kids = tree.children(i)
            for k in range(3):
                self._lod_min.append(cube[0][k] - cube[1] * 0.5)
            self._lod_size.append(cube[1])
            self._lod_spacing.append(tree.node_spacing(i))
            self._lod_level.append(tree.node_level(i))
            self._lod_first.append(range_[0])
            self._lod_count.append(range_[1])
            for k in range(8):
                self._lod_children.append(kids[k] if k < len(kids) else -1)

    def has_lod(self) -> bool:
        return len(self._lod_size) > 0

    def lod_node_count(self) -> int:
        return len(self._lod_size)

    def lod_cube(self, i: int) -> tuple[Point, float]:
        """Node cube center and edge length"""
        half = self._lod_size[i] * 0.5
        return (Point(self._lod_min[i * 3] + half, self._lod_min[i * 3 + 1] + half, self._lod_min[i * 3 + 2] + half), self._lod_size[i])

    def lod_spacing(self, i: int) -> float:
        """Grid-accept spacing of a node"""
        return self._lod_spacing[i]

    def lod_level(self, i: int) -> int:
        """Node depth from the root"""
        return self._lod_level[i]

    def lod_range(self, i: int) -> tuple[int, int]:
        """Node point range as (first, count) into the reordered arrays"""
        return (self._lod_first[i], self._lod_count[i])

    def lod_children(self, i: int) -> list[int]:
        """Present child node indices compacted into 8 slots, -1 unused"""
        return self._lod_children[i * 8 : i * 8 + 8]

    # ═══════════════════════════════════════════════════════════════════════════
    # Point ids
    # ═══════════════════════════════════════════════════════════════════════════

    def point_ids(self) -> list[int]:
        """Stable ids parallel to the points, minted by the first build_lod; empty before that"""
        return self._point_ids

    def point_id(self, index: int) -> int:
        """Stable id of the point at index; the index itself before a tree is built"""
        return index if not self._point_ids else self._point_ids[index]

    def index_of_id(self, id: int) -> int:
        """Current index of a stable id, -1 when the cloud has no such point"""
        if not self._point_ids:
            return id if 0 <= id < self.point_count() else -1
        for i in range(len(self._point_ids)):
            if self._point_ids[i] == id:
                return i
        return -1

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self) -> dict:
        return {
            "colors": self._colors,
            "coords": self._coords,
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
            "type": "PointCloud",
        }

    @classmethod
    def __jsonload__(cls, data: dict, guid: str = None, name: str = None) -> "PointCloud":
        cloud = cls.from_coords(data.get("coords", []), data.get("colors", []), data.get("normals", []))
        cloud.guid = guid if guid is not None else data.get("guid", cloud.guid)
        cloud.name = name if name is not None else data.get("name", cloud.name)
        cloud.point_size = data.get("point_size", 1.0)
        cloud._lod_min = data.get("lod_min", [])
        cloud._lod_size = data.get("lod_size", [])
        cloud._lod_spacing = data.get("lod_spacing", [])
        cloud._lod_level = data.get("lod_level", [])
        cloud._lod_first = data.get("lod_first", [])
        cloud._lod_count = data.get("lod_count", [])
        cloud._lod_children = data.get("lod_children", [])
        cloud._point_ids = data.get("point_ids", [])
        return cloud

    def file_json_dumps(self) -> str:
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "PointCloud":
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        with open(filepath, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "PointCloud":
        with open(filepath) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def pb_dumps(self) -> bytes:
        from .proto import pointcloud_pb2

        proto = pointcloud_pb2.PointCloud()
        if self.has_guid():
            proto.guid = self._guid
        proto.name = self.name
        proto.point_size = self.point_size
        proto.coords.extend(self._coords)
        proto.colors.extend(self._colors)
        proto.normals.extend(self._normals)
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
    def pb_loads(cls, data: bytes) -> "PointCloud":
        from .proto import pointcloud_pb2

        proto = pointcloud_pb2.PointCloud()
        proto.ParseFromString(data)
        cloud = cls.from_coords(list(proto.coords), list(proto.colors), list(proto.normals))
        if proto.guid:
            cloud.guid = proto.guid
        cloud.name = proto.name
        if proto.point_size > 0.0:
            cloud.point_size = proto.point_size
        cloud._lod_min = list(proto.lod_min)
        cloud._lod_size = list(proto.lod_size)
        cloud._lod_spacing = list(proto.lod_spacing)
        cloud._lod_level = list(proto.lod_level)
        cloud._lod_first = list(proto.lod_first)
        cloud._lod_count = list(proto.lod_count)
        cloud._lod_children = list(proto.lod_children)
        cloud._point_ids = list(proto.point_ids)
        return cloud

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        with open(filepath, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "PointCloud":
        with open(filepath, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════

    def __str__(self) -> str:
        """N points"""
        return f"{self.point_count()} points"

    def __repr__(self) -> str:
        """PointCloud(name, N points, N colors, N normals)"""
        return f"PointCloud({self.name}, {self.point_count()} points, {self.color_count()} colors, {self.normal_count()} normals)"
