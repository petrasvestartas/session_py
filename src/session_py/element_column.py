import uuid
import copy
from .element import Element
from .xform import Xform


class ColumnElement(Element):
    def __init__(self, width=0.4, depth=0.4, height=3.0, name="my_column"):
        super().__init__(geometry=None, name=name)
        self._width = width
        self._depth = depth
        self._height = height
        self._geometry = self.compute_element_geometry()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._geometry = self.compute_element_geometry()
        self.reset()

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value
        self._geometry = self.compute_element_geometry()
        self.reset()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        self._geometry = self.compute_element_geometry()
        self.reset()

    @property
    def center_line(self):
        from .line import Line
        return Line(0, 0, 0, 0, 0, self._height)

    def compute_element_geometry(self):
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

    def extend(self, distance):
        self._height += distance * 2
        self._geometry = self.compute_element_geometry()
        self.reset()

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
        result._session_transformation = copy.deepcopy(self._session_transformation, memo)
        result._features = list(self._features)
        result._is_dirty = True
        result._aabb = None
        result._obb = None
        result._collision_mesh = None
        result._point = None
        return result

    def __eq__(self, other):
        if not isinstance(other, ColumnElement):
            return False
        return (self.name == other.name and
                self._width == other._width and
                self._depth == other._depth and
                self._height == other._height)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"ColumnElement({self.name}, {self._width}, {self._depth}, {self._height})"

    def __repr__(self):
        return f"ColumnElement({self.guid}, {self.name}, {self._width}, {self._depth}, {self._height})"

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
            "session_transformation": self.session_transformation.__jsondump__(),
            "type": "ColumnElement",
            "width": self._width,
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        from .encoders import decode_node
        elem = cls(
            width=data.get("width", 0.4),
            depth=data.get("depth", 0.4),
            height=data.get("height", 3.0),
        )
        elem.guid = guid if guid is not None else data.get("guid", elem.guid)
        elem.name = name if name is not None else data.get("name", elem.name)
        if "session_transformation" in data:
            elem.session_transformation = decode_node(data["session_transformation"])
        return elem

    ###########################################################################################
    # Serialization - Protobuf (reuse Element proto with parametric fields in geometry_data)
    ###########################################################################################

    def pb_dumps(self):
        from .proto import element_pb2
        proto = element_pb2.Element()
        proto.guid = self.guid
        proto.name = self.name
        proto.geometry_type = "ColumnElement"
        import json
        proto.geometry_data = json.dumps({
            "width": self._width,
            "depth": self._depth,
            "height": self._height,
        }).encode()
        proto.session_transformation.name = self.session_transformation.name
        proto.session_transformation.matrix.extend(self.session_transformation.m)
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
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
        xf = Xform()
        xf.name = proto.session_transformation.name
        xf.m = list(proto.session_transformation.matrix)
        elem.session_transformation = xf
        return elem
