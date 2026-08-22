from __future__ import annotations
import uuid
import copy
from .element import Element
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .line import Line
    from .mesh import Mesh
    from .plane import Plane
    from .polyline import Polyline
    from .vector import Vector


class ElementColumn(Element):
    def __init__(self, width: float = 0.4, depth: float = 0.4, height: float = 3.0, name: str = "my_column"):
        super().__init__(geometry=None, name=name)
        self._width = width
        self._depth = depth
        self._height = height
        self._geometry = self.compute_element_geometry()

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        self._width = value
        self._geometry = self.compute_element_geometry()
        self.reset()

    @property
    def depth(self) -> float:
        return self._depth

    @depth.setter
    def depth(self, value: float) -> None:
        self._depth = value
        self._geometry = self.compute_element_geometry()
        self.reset()

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        self._height = value
        self._geometry = self.compute_element_geometry()
        self.reset()

    @property
    def center_line(self) -> "Line":
        from .line import Line
        return Line(0, 0, 0, 0, 0, self._height)

    def compute_element_geometry(self) -> "Mesh":
        from .mesh import Mesh
        from .point import Point
        hx = self._width * 0.5
        hy = self._depth * 0.5
        vertices = [
            Point(-hx, -hy, 0),
            Point( hx, -hy, 0),
            Point( hx,  hy, 0),
            Point(-hx,  hy, 0),
            Point(-hx, -hy, self._height),
            Point( hx, -hy, self._height),
            Point( hx,  hy, self._height),
            Point(-hx,  hy, self._height),
        ]
        faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ]
        return Mesh.from_vertices_and_faces(vertices, faces)

    def extend(self, distance: float) -> None:
        self._height += distance * 2
        self._geometry = self.compute_element_geometry()
        self.reset()

    def compute_polylines(self) -> list["Polyline"]:
        from .polyline import Polyline
        from .point import Point
        hx = self._width * 0.5
        hy = self._depth * 0.5
        b = [Point(-hx, -hy, 0), Point(hx, -hy, 0), Point(hx, hy, 0), Point(-hx, hy, 0)]
        t = [Point(-hx, -hy, self._height), Point(hx, -hy, self._height), Point(hx, hy, self._height), Point(-hx, hy, self._height)]
        bottom_pl = Polyline([b[0], b[3], b[2], b[1], Point(b[0][0], b[0][1], b[0][2])])
        top_pl = Polyline([t[0], t[1], t[2], t[3], Point(t[0][0], t[0][1], t[0][2])])
        side0 = Polyline([b[0], b[1], t[1], t[0], Point(b[0][0], b[0][1], b[0][2])])
        side1 = Polyline([b[2], b[3], t[3], t[2], Point(b[2][0], b[2][1], b[2][2])])
        side2 = Polyline([b[0], t[0], t[3], b[3], Point(b[0][0], b[0][1], b[0][2])])
        side3 = Polyline([b[1], b[2], t[2], t[1], Point(b[1][0], b[1][1], b[1][2])])
        return [bottom_pl, top_pl, side0, side1, side2, side3]

    def compute_planes(self) -> list["Plane"]:
        from .plane import Plane
        from .point import Point
        from .vector import Vector
        hx = self._width * 0.5
        hy = self._depth * 0.5
        hz = self._height * 0.5
        return [
            Plane.from_point_normal(Point(0, 0, 0), Vector(0, 0, -1)),
            Plane.from_point_normal(Point(0, 0, self._height), Vector(0, 0, 1)),
            Plane.from_point_normal(Point(0, -hy, hz), Vector(0, -1, 0)),
            Plane.from_point_normal(Point(0, hy, hz), Vector(0, 1, 0)),
            Plane.from_point_normal(Point(-hx, 0, hz), Vector(-1, 0, 0)),
            Plane.from_point_normal(Point(hx, 0, hz), Vector(1, 0, 0)),
        ]

    def compute_edge_vectors(self) -> list["Vector"]:
        from .vector import Vector
        return [
            Vector(1, 0, 0), Vector(0, 1, 0), Vector(-1, 0, 0), Vector(0, -1, 0),
            Vector(1, 0, 0), Vector(0, 1, 0), Vector(-1, 0, 0), Vector(0, -1, 0),
            Vector(0, 0, 1), Vector(0, 0, 1), Vector(0, 0, 1), Vector(0, 0, 1),
        ]

    def compute_axis(self) -> "Line":
        from .line import Line
        return Line(0, 0, 0, 0, 0, self._height)

    ###########################################################################################
    # Operators
    ###########################################################################################

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.guid = str(uuid.uuid4())
        result.name = copy.deepcopy(self.name, memo)
        result._width = self._width
        result._depth = self._depth
        result._height = self._height
        result._geometry = copy.deepcopy(self._geometry, memo)
        result._features = list(self._features)
        result._is_dirty = True
        result._aabb = None
        result._obb = None
        result._collision_mesh = None
        result._point = None
        result._polylines = None
        result._planes = None
        result._edge_vectors = None
        result._axis = None
        return result

    def __eq__(self, other):
        if not isinstance(other, ElementColumn):
            return False
        return (self.name == other.name and
                self._width == other._width and
                self._depth == other._depth and
                self._height == other._height)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"ElementColumn({self.name}, {self._width}, {self._depth}, {self._height})"

    def __repr__(self):
        return f"ElementColumn({self.guid}, {self.name}, {self._width}, {self._depth}, {self._height})"

    ###########################################################################################
    # Serialization - JSON
    ###########################################################################################

    def __jsondump__(self):
        return {
            "depth": self._depth,
            "geometry_data": self._geometry.__jsondump__() if self._geometry else None,
            "geometry_type": type(self._geometry).__name__ if self._geometry else "None",
            "guid": self.guid,
            "height": self._height,
            "name": self.name,
            "type": "ElementColumn",
            "width": self._width,
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        from .file_encoders import file_decode_node
        elem = cls(
            width=data.get("width", 0.4),
            depth=data.get("depth", 0.4),
            height=data.get("height", 3.0),
        )
        elem.guid = guid if guid is not None else data.get("guid", elem.guid)
        elem.name = name if name is not None else data.get("name", elem.name)
        return elem

    ###########################################################################################
    # Serialization - Protobuf (reuse Element proto with parametric fields in geometry_data)
    ###########################################################################################

    def pb_dumps(self) -> bytes:
        from .proto import element_pb2
        proto = element_pb2.Element()
        proto.guid = self.guid
        proto.name = self.name
        proto.geometry_type = "ElementColumn"
        import json
        proto.geometry_data = json.dumps({
            "width": self._width,
            "depth": self._depth,
            "height": self._height,
        }).encode()
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "ElementColumn":
        from .proto import element_pb2
        import json
        proto = element_pb2.Element()
        proto.ParseFromString(data)
        params = json.loads(proto.geometry_data.decode())
        elem = cls(
            width=params["width"],
            depth=params["depth"],
            height=params["height"],
        )
        elem.guid = proto.guid
        elem.name = proto.name
        return elem
