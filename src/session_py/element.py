import uuid
import copy
from .xform import Xform


class Element:
    def __init__(self, geometry=None, name="my_element"):
        self._guid = None
        self.name = name
        self._geometry = geometry
        self._session_transformation = None
        self._features = []
        self._is_dirty = True
        self._aabb = None
        self._obb = None
        self._collision_mesh = None
        self._point = None
        self._polylines = None
        self._planes = None
        self._edge_vectors = None
        self._axis = None

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

    @property
    def geometry(self):
        return self._geometry

    @property
    def session_transformation(self):
        if self._session_transformation is None:
            self._session_transformation = Xform.identity()
        return self._session_transformation

    @session_transformation.setter
    def session_transformation(self, value):
        self._session_transformation = value
        self._is_dirty = True

    @property
    def session_geometry(self):
        if self._geometry is None:
            return None
        geo = copy.deepcopy(self._geometry)
        geo = self.apply_features(geo)
        xf = self.session_transformation
        if not xf.is_identity():
            geo.xform = xf * geo.xform
            geo.transform()
        return geo

    @property
    def aabb(self):
        if self._is_dirty or self._aabb is None:
            self._aabb = self.compute_aabb()
        return self._aabb

    @property
    def obb(self):
        if self._is_dirty or self._obb is None:
            self._obb = self.compute_obb()
        return self._obb

    @property
    def collision_mesh(self):
        if self._is_dirty or self._collision_mesh is None:
            self._collision_mesh = self.compute_collision_mesh()
        return self._collision_mesh

    @property
    def point(self):
        if self._is_dirty or self._point is None:
            self._point = self.compute_point()
        return self._point

    @property
    def polylines(self):
        if self._is_dirty or self._polylines is None:
            self._polylines = self.compute_polylines()
        return self._polylines

    @property
    def planes(self):
        if self._is_dirty or self._planes is None:
            self._planes = self.compute_planes()
        return self._planes

    @property
    def edge_vectors(self):
        if self._is_dirty or self._edge_vectors is None:
            self._edge_vectors = self.compute_edge_vectors()
        return self._edge_vectors

    @property
    def axis(self):
        if self._is_dirty or self._axis is None:
            self._axis = self.compute_axis()
        return self._axis

    @property
    def is_dirty(self):
        return self._is_dirty

    ###########################################################################################
    # Operators
    ###########################################################################################

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.guid = str(uuid.uuid4())
        result.name = copy.deepcopy(self.name, memo)
        result._geometry = copy.deepcopy(self._geometry, memo)
        result._session_transformation = copy.deepcopy(self._session_transformation, memo)
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

    def duplicate(self):
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def __eq__(self, other):
        if not isinstance(other, Element):
            return False
        return self.name == other.name and type(self._geometry) == type(other._geometry)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        geo_type = type(self._geometry).__name__ if self._geometry else "None"
        return f"Element({self.name}, {geo_type})"

    def __repr__(self):
        geo_type = type(self._geometry).__name__ if self._geometry else "None"
        return f"Element({self.guid}, {self.name}, {geo_type})"

    ###########################################################################################
    # Mutators
    ###########################################################################################

    def add_feature(self, feature):
        self._features.append(feature)
        self._is_dirty = True

    def set_geometry(self, geometry):
        self._geometry = geometry
        self._is_dirty = True

    def set_polylines(self, polylines):
        self._polylines = polylines

    def set_planes(self, planes):
        self._planes = planes

    def reset(self):
        self._is_dirty = True
        self._aabb = None
        self._obb = None
        self._collision_mesh = None
        self._point = None
        self._polylines = None
        self._planes = None
        self._edge_vectors = None
        self._axis = None

    ###########################################################################################
    # Computation
    ###########################################################################################

    def compute_aabb(self):
        from .obb import OBB
        from .point import Point
        geo = self.session_geometry
        if geo is None:
            return OBB.from_point(Point(0, 0, 0), 0.0)
        return self._obb_from_geometry(geo, aabb=True)

    def compute_obb(self):
        from .obb import OBB
        from .point import Point
        geo = self.session_geometry
        if geo is None:
            return OBB.from_point(Point(0, 0, 0), 0.0)
        return self._obb_from_geometry(geo, aabb=False)

    def compute_collision_mesh(self):
        from .mesh import Mesh
        geo = self.session_geometry
        if geo is None:
            return Mesh()
        if isinstance(geo, Mesh):
            return geo
        return Mesh()

    def compute_point(self):
        from .point import Point
        from .mesh import Mesh
        from .brep import BRep
        geo = self.session_geometry
        if geo is None:
            return Point(0, 0, 0)
        if isinstance(geo, Mesh):
            verts = list(geo.vertex.values())
            if not verts:
                return Point(0, 0, 0)
            sx = sum(v.x for v in verts)
            sy = sum(v.y for v in verts)
            sz = sum(v.z for v in verts)
            n = len(verts)
            return Point(sx / n, sy / n, sz / n)
        if isinstance(geo, BRep):
            pts = geo.m_vertices
            if not pts:
                return Point(0, 0, 0)
            sx = sum(p[0] for p in pts)
            sy = sum(p[1] for p in pts)
            sz = sum(p[2] for p in pts)
            n = len(pts)
            return Point(sx / n, sy / n, sz / n)
        return Point(0, 0, 0)

    def compute_polylines(self):
        return []

    def compute_planes(self):
        return []

    def compute_edge_vectors(self):
        return []

    def compute_axis(self):
        return None

    def apply_features(self, geometry):
        for feature in self._features:
            geometry = feature(geometry)
        return geometry

    @staticmethod
    def _obb_from_geometry(geo, aabb=True):
        from .obb import OBB
        from .point import Point
        from .mesh import Mesh
        from .brep import BRep
        inflate = 0.0
        if isinstance(geo, Mesh):
            points = [v.position() for v in geo.vertex.values()]
            if not points:
                return OBB.from_point(Point(0, 0, 0), inflate)
            return OBB.from_points(points, inflate)
        if isinstance(geo, BRep):
            if not geo.m_vertices:
                return OBB.from_point(Point(0, 0, 0), inflate)
            return OBB.from_points(geo.m_vertices, inflate)
        return OBB.from_point(Point(0, 0, 0), inflate)

    ###########################################################################################
    # Serialization - JSON
    ###########################################################################################

    def __jsondump__(self):
        geo_data = None
        geo_type = "None"
        if self._geometry is not None:
            geo_type = type(self._geometry).__name__
            geo_data = self._geometry.__jsondump__()
        return {
            "geometry_data": geo_data,
            "geometry_type": geo_type,
            "guid": self.guid,
            "name": self.name,
            "session_transformation": self.session_transformation.__jsondump__(),
            "type": "Element",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        from .encoders import decode_node
        geo_type = data.get("geometry_type", "None")
        geo_data = data.get("geometry_data")
        geometry = None
        if geo_data is not None and geo_type != "None":
            geometry = decode_node(geo_data)
        elem = cls(geometry=geometry)
        elem.guid = guid if guid is not None else data.get("guid", elem.guid)
        elem.name = name if name is not None else data.get("name", elem.name)
        if "session_transformation" in data:
            elem.session_transformation = decode_node(data["session_transformation"])
        return elem

    def json_dumps(self):
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, s):
        import json
        return cls.__jsonload__(json.loads(s))

    def json_dump(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        import json
        with open(filepath, 'r') as f:
            return cls.__jsonload__(json.load(f))

    ###########################################################################################
    # Serialization - Protobuf
    ###########################################################################################

    def pb_dumps(self):
        from .proto import element_pb2
        proto = element_pb2.Element()
        proto.guid = self.guid
        proto.name = self.name
        if self._geometry is not None:
            proto.geometry_type = type(self._geometry).__name__
            proto.geometry_data = self._geometry.pb_dumps()
        else:
            proto.geometry_type = "None"
        proto.session_transformation.name = self.session_transformation.name
        proto.session_transformation.matrix.extend(self.session_transformation.m)
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        from .proto import element_pb2
        from .xform import Xform
        proto = element_pb2.Element()
        proto.ParseFromString(data)
        geometry = None
        if proto.geometry_type and proto.geometry_type != "None" and proto.geometry_data:
            geometry = cls._pb_load_geometry(proto.geometry_type, proto.geometry_data)
        elem = cls(geometry=geometry)
        elem.guid = proto.guid
        elem.name = proto.name
        xf = Xform()
        xf.name = proto.session_transformation.name
        xf.m = list(proto.session_transformation.matrix)
        elem.session_transformation = xf
        return elem

    @staticmethod
    def _pb_load_geometry(geo_type, geo_data):
        from .point import Point
        from .line import Line
        from .plane import Plane
        from .polyline import Polyline
        from .mesh import Mesh
        from .obb import OBB
        from .pointcloud import PointCloud
        from .nurbscurve import NurbsCurve
        from .nurbssurface import NurbsSurface
        from .brep import BRep
        type_map = {
            "Point": Point,
            "Line": Line,
            "Plane": Plane,
            "Polyline": Polyline,
            "Mesh": Mesh,
            "OBB": OBB,
            "PointCloud": PointCloud,
            "NurbsCurve": NurbsCurve,
            "NurbsSurface": NurbsSurface,
            "BRep": BRep,
        }
        if geo_type in ("ElementColumn", "ElementBeam", "ElementPlate"):
            return None
        klass = type_map.get(geo_type)
        if klass is None:
            return None
        return klass.pb_loads(geo_data)

    def pb_dump(self, filepath):
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath):
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())
