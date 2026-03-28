import uuid
import copy
from .element import Element
from .xform import Xform


class PlateElement(Element):
    def __init__(self, polygon=None, thickness=0.1, name="my_plate"):
        super().__init__(geometry=None, name=name)
        from .point import Point
        if polygon is None:
            polygon = [
                Point(-0.5, -0.5, 0),
                Point( 0.5, -0.5, 0),
                Point( 0.5,  0.5, 0),
                Point(-0.5,  0.5, 0),
            ]
        self._polygon = [Point(p[0], p[1], p[2]) for p in polygon]
        self._thickness = thickness
        self._geometry = self.compute_element_geometry()

    @property
    def polygon(self):
        return self._polygon

    @polygon.setter
    def polygon(self, value):
        from .point import Point
        self._polygon = [Point(p[0], p[1], p[2]) for p in value]
        self._geometry = self.compute_element_geometry()
        self.reset()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value
        self._geometry = self.compute_element_geometry()
        self.reset()

    @staticmethod
    def _polygon_normal(pts):
        from .vector import Vector
        nx, ny, nz = 0.0, 0.0, 0.0
        n = len(pts)
        for i in range(n):
            c = pts[i]
            nx_pt = pts[(i + 1) % n]
            nx += (c[1] - nx_pt[1]) * (c[2] + nx_pt[2])
            ny += (c[2] - nx_pt[2]) * (c[0] + nx_pt[0])
            nz += (c[0] - nx_pt[0]) * (c[1] + nx_pt[1])
        mag = (nx * nx + ny * ny + nz * nz) ** 0.5
        if mag < 1e-12:
            return Vector(0, 0, 1)
        return Vector(nx / mag, ny / mag, nz / mag)

    def compute_element_geometry(self):
        from .mesh import Mesh
        from .point import Point
        normal = self._polygon_normal(self._polygon)
        n = len(self._polygon)
        bottom = []
        top = []
        for p in self._polygon:
            bottom.append(Point(p[0], p[1], p[2]))
            top.append(Point(
                p[0] - normal[0] * self._thickness,
                p[1] - normal[1] * self._thickness,
                p[2] - normal[2] * self._thickness,
            ))
        vertices = bottom + top
        bottom_face = list(range(n - 1, -1, -1))
        top_face = list(range(n, 2 * n))
        faces = [bottom_face, top_face]
        for i in range(n):
            a = i
            b = (i + 1) % n
            c = b + n
            d = a + n
            faces.append([a, b, c, d])
        return Mesh.from_vertices_and_faces(vertices, faces)

    ###########################################################################################
    # Operators
    ###########################################################################################

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.guid = str(uuid.uuid4())
        result.name = copy.deepcopy(self.name, memo)
        result._polygon = copy.deepcopy(self._polygon, memo)
        result._thickness = self._thickness
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
        if not isinstance(other, PlateElement):
            return False
        if self.name != other.name:
            return False
        if self._thickness != other._thickness:
            return False
        if len(self._polygon) != len(other._polygon):
            return False
        for a, b in zip(self._polygon, other._polygon):
            if a[0] != b[0] or a[1] != b[1] or a[2] != b[2]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"PlateElement({self.name}, {len(self._polygon)} pts, {self._thickness})"

    def __repr__(self):
        return f"PlateElement({self.guid}, {self.name}, {len(self._polygon)} pts, {self._thickness})"

    ###########################################################################################
    # Serialization - JSON
    ###########################################################################################

    def __jsondump__(self):
        return {
            "geometry_data": self._geometry.__jsondump__() if self._geometry else None,
            "geometry_type": type(self._geometry).__name__ if self._geometry else "None",
            "guid": self.guid,
            "name": self.name,
            "polygon": [[p[0], p[1], p[2]] for p in self._polygon],
            "session_transformation": self.session_transformation.__jsondump__(),
            "thickness": self._thickness,
            "type": "PlateElement",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        from .encoders import decode_node
        from .point import Point
        polygon = [Point(p[0], p[1], p[2]) for p in data.get("polygon", [])]
        elem = cls(
            polygon=polygon if polygon else None,
            thickness=data.get("thickness", 0.1),
        )
        elem.guid = guid if guid is not None else data.get("guid", elem.guid)
        elem.name = name if name is not None else data.get("name", elem.name)
        if "session_transformation" in data:
            elem.session_transformation = decode_node(data["session_transformation"])
        return elem

    ###########################################################################################
    # Serialization - Protobuf
    ###########################################################################################

    def pb_dumps(self):
        from .proto import element_pb2
        proto = element_pb2.Element()
        proto.guid = self.guid
        proto.name = self.name
        proto.geometry_type = "PlateElement"
        import json
        proto.geometry_data = json.dumps({
            "polygon": [[p[0], p[1], p[2]] for p in self._polygon],
            "thickness": self._thickness,
        }).encode()
        proto.session_transformation.name = self.session_transformation.name
        proto.session_transformation.matrix.extend(self.session_transformation.m)
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        from .proto import element_pb2
        from .point import Point
        import json
        proto = element_pb2.Element()
        proto.ParseFromString(data)
        params = json.loads(proto.geometry_data.decode())
        polygon = [Point(p[0], p[1], p[2]) for p in params["polygon"]]
        elem = cls(
            polygon=polygon,
            thickness=params["thickness"],
        )
        elem.guid = proto.guid
        elem.name = proto.name
        xf = Xform()
        xf.name = proto.session_transformation.name
        xf.m = list(proto.session_transformation.matrix)
        elem.session_transformation = xf
        return elem
