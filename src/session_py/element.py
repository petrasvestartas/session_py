from __future__ import annotations
from typing import Union
from collections.abc import Callable
from typing import Optional
from typing import TYPE_CHECKING
import uuid
import copy
from .xform import Xform

if TYPE_CHECKING:
    from pathlib import Path
    from .vector import Vector
    from .brep import BRep
    from .line import Line
    from .mesh import Mesh
    from .obb import OBB
    from .plane import Plane
    from .point import Point
    from .polyline import Polyline


class ElementFeature:
    """One modification applied to a host element - a cut, a drill, a joint pocket.

    The serializable half of what :meth:`Element.add_geometry_op` cannot be: that takes a
    callable, so an operation applied in memory vanishes the moment the Session is written.
    Domains worked around it by adding flat arrays to Element - a joint type code per face -
    which is how timber fields ended up in element.proto and had to be reserved out again.

    The kernel does not know how to APPLY one: ``feature_type`` means something only to the
    package that wrote it. It knows enough to DRAW one, which is what lets a viewer show
    features from a package it has never heard of.
    """

    def __init__(self, feature_type: str = "", face_index: int = -1, outlines=None, name: str = ""):
        self._guid: str | None = None
        self.name = name
        self.feature_type = feature_type
        #: Face of the host this applies to; -1 = the whole element.
        self.face_index = face_index
        self.outlines = list(outlines or [])

    @property
    def guid(self) -> str:
        """Lazily minted, like every other identity in the kernel - a feature nobody names never
        pays for a guid.

        A feature is addressable in its own right: the package that wrote a joint needs to name
        it again later, to update it, to report a clash against it, or to let a viewer select one
        of the forty cuts on a beam. The only other handle is the index in ``features``, and that
        moves the moment an earlier feature is removed.
        """
        if getattr(self, "_guid", None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def __eq__(self, other) -> bool:
        if not isinstance(other, ElementFeature):
            return NotImplemented
        return (
            self.name == other.name
            and self.feature_type == other.feature_type
            and self.face_index == other.face_index
            and self.outlines == other.outlines
        )

    def __repr__(self) -> str:
        return (
            f"ElementFeature({self.feature_type}, face {self.face_index}, "
            f"{len(self.outlines)} outline(s))"
        )


class Element:
    def __init__(self, geometry: Union["Mesh", "BRep"] | None = None, name: str = "my_element"):
        self._guid = None
        self.name = name
        self._geometry = geometry
        # Callables, applied lazily when geometry is computed - NOT serializable. Renamed off
        # "feature" so the serializable `features` below can own that name; two different
        # things wearing one name is what made a joint type code look like it needed its own
        # field on Element.
        self._geometry_ops = []
        self._features: list[ElementFeature] = []
        self._insertion_vectors: list = []
        self._dimensions = None
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
    def guid(self, value: str) -> None:
        self._guid = value

    @property
    def geometry(self) -> Union["Mesh", "BRep"] | None:
        return self._geometry

    @property
    def has_geometry(self) -> bool:
        return self._geometry is not None

    @property
    def geometry_type_name(self) -> str:
        return type(self._geometry).__name__ if self._geometry is not None else "None"

    def session_geometry(self, xform: Xform) -> Union["Mesh", "BRep"] | None:
        """The element's geometry placed by ``xform``. The placement is supplied by the caller -
        an Element no longer stores one; the Session does. Pass identity for local geometry.
        """
        from .mesh import Mesh
        if self._geometry is None:
            return None
        geo = copy.deepcopy(self._geometry)
        if isinstance(geo, Mesh):
            geo = self.apply_geometry_ops(geo)
        if not xform.is_identity():
            geo.transform(xform)
        return geo

    @property
    def aabb(self) -> "OBB":
        if self._is_dirty or self._aabb is None:
            self._aabb = self.compute_aabb()
        return self._aabb

    @property
    def obb(self) -> "OBB":
        if self._is_dirty or self._obb is None:
            self._obb = self.compute_obb()
        return self._obb

    @property
    def collision_mesh(self) -> "Mesh":
        if self._is_dirty or self._collision_mesh is None:
            self._collision_mesh = self.compute_collision_mesh()
        return self._collision_mesh

    @property
    def point(self) -> "Point":
        if self._is_dirty or self._point is None:
            self._point = self.compute_point()
        return self._point

    @property
    def polylines(self) -> list["Polyline"]:
        if self._is_dirty or self._polylines is None:
            self._polylines = self.compute_polylines()
        return self._polylines

    @property
    def planes(self) -> list["Plane"]:
        if self._is_dirty or self._planes is None:
            self._planes = self.compute_planes()
        return self._planes

    @property
    def edge_vectors(self) -> list["Vector"]:
        if self._is_dirty or self._edge_vectors is None:
            self._edge_vectors = self.compute_edge_vectors()
        return self._edge_vectors

    @property
    def axis(self) -> Optional["Line"]:
        if self._is_dirty or self._axis is None:
            self._axis = self.compute_axis()
        return self._axis

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty
    @property
    def cached_aabb(self) -> Optional["OBB"]:
        return self._aabb

    @property
    def cached_obb(self) -> Optional["OBB"]:
        return self._obb

    @property
    def cached_collision_mesh(self) -> Optional["Mesh"]:
        return self._collision_mesh

    @property
    def cached_point(self) -> Optional["Point"]:
        return self._point

    @property
    def geometry_ops_count(self) -> int:
        return len(self._geometry_ops)

    @property
    def features_count(self) -> int:
        return len(self._features)

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
        result._geometry_ops = list(self._geometry_ops)
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

    def duplicate(self) -> "Element":
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def __eq__(self, other):
        if not isinstance(other, Element):
            return False
        return self.name == other.name and self.geometry_type_name == other.geometry_type_name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"Element({self.name}, {self.geometry_type_name})"

    def __repr__(self):
        return f"Element({self.guid}, {self.name}, {self.geometry_type_name})"

    ###########################################################################################
    # Mutators
    ###########################################################################################

    def add_geometry_op(self, op: Callable) -> None:
        """Add an in-memory mesh operation. Not serialized - see :class:`ElementFeature`."""
        self._geometry_ops.append(op)
        self._is_dirty = True

    def add_feature(self, feature: "ElementFeature") -> None:
        """Add a modification carried BY this element, and written with it."""
        self._features.append(feature)

    @property
    def features(self) -> list:
        return self._features

    @features.setter
    def features(self, features) -> None:
        self._features = list(features)

    @property
    def insertion_vectors(self) -> list:
        """Direction(s) the element is inserted along when the assembly is put together.

        General to any assembly: it is what an assembly sequence is ordered by. Plural because
        an element with several jointed faces can admit a different direction per face.
        """
        return self._insertion_vectors

    @insertion_vectors.setter
    def insertion_vectors(self, vectors) -> None:
        self._insertion_vectors = list(vectors)

    @property
    def dimensions(self):
        """NOMINAL extents in this element's own frame - authored intent, NOT a measurement.

        Plate: x/y outline extent, z thickness. Beam: x/y cross-section, z length.

        Deliberately distinct from :attr:`obb`, which MEASURES the geometry that exists. The
        two are allowed to disagree: a thickness drives a loft before there is any geometry to
        measure, so the nominal value has to exist first and outlive what is built from it.
        Read ``obb`` for how big it IS, this for how big it was MEANT to be. ``None`` = never
        authored, which (0, 0, 0) does not mean.
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value) -> None:
        self._dimensions = value

    def place(self, xform: Xform) -> None:
        """Bake a placement into this element's own geometry, invalidating the cached boxes.
        The Session owns the placement, so it hands it in here rather than the Element storing it.
        """
        self._geometry = self.session_geometry(xform)
        self._is_dirty = True

    def set_geometry(self, geometry: Union["Mesh", "BRep"] | None) -> None:
        self._geometry = geometry
        self._is_dirty = True

    def set_polylines(self, polylines: list["Polyline"]) -> None:
        self._polylines = polylines

    def set_planes(self, planes: list["Plane"]) -> None:
        self._planes = planes

    def reset(self) -> None:
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

    def compute_aabb(self) -> "OBB":
        from .obb import OBB
        from .point import Point
        geo = self.session_geometry(Xform.identity())
        if geo is None:
            return OBB.from_point(Point(0, 0, 0), 0.0)
        return self._obb_from_geometry(geo)

    def compute_obb(self) -> "OBB":
        from .obb import OBB
        from .point import Point
        geo = self.session_geometry(Xform.identity())
        if geo is None:
            return OBB.from_point(Point(0, 0, 0), 0.0)
        return self._obb_from_geometry(geo)

    def compute_collision_mesh(self) -> "Mesh":
        from .mesh import Mesh
        geo = self.session_geometry(Xform.identity())
        if geo is None:
            return Mesh()
        if isinstance(geo, Mesh):
            return geo
        return Mesh()

    def compute_point(self) -> "Point":
        from .point import Point
        from .mesh import Mesh
        from .brep import BRep
        geo = self.session_geometry(Xform.identity())
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

    def compute_polylines(self) -> list["Polyline"]:
        return []

    def compute_planes(self) -> list["Plane"]:
        return []

    def compute_edge_vectors(self) -> list["Vector"]:
        return []

    def compute_axis(self) -> Optional["Line"]:
        return None

    def apply_geometry_ops(self, geometry: "Mesh") -> "Mesh":
        for op in self._geometry_ops:
            geometry = op(geometry)
        return geometry

    @staticmethod
    def _obb_from_geometry(geo):
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
        geo_data = self._geometry.__jsondump__() if self._geometry is not None else None
        return {
            "geometry_data": geo_data,
            "geometry_type": self.geometry_type_name,
            "guid": self.guid,
            "name": self.name,
            "type": "Element",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        from .file_encoders import file_decode_node
        geo_type = data.get("geometry_type", "None")
        geo_data = data.get("geometry_data")
        geometry = None
        if geo_data is not None and geo_type != "None":
            geometry = file_decode_node(geo_data)
        elem = cls(geometry=geometry)
        elem.guid = guid if guid is not None else data.get("guid", elem.guid)
        elem.name = name if name is not None else data.get("name", elem.name)
        return elem

    def file_json_dumps(self) -> str:
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, s: str) -> "Element":
        import json
        return cls.__jsonload__(json.loads(s))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "Element":
        import json
        with open(filepath) as f:
            return cls.__jsonload__(json.load(f))

    ###########################################################################################
    # Serialization - Protobuf
    ###########################################################################################

    def pb_dumps(self) -> bytes:
        from .proto import element_pb2
        proto = element_pb2.Element()
        proto.guid = self.guid
        proto.name = self.name
        if self._geometry is not None:
            proto.geometry_type = type(self._geometry).__name__
            proto.geometry_data = self._geometry.pb_dumps()
        else:
            proto.geometry_type = "None"
        # Both empty for a plain Element, and proto3 does not emit empty scalars - so a base
        # element's bytes are unchanged by the registry, keeping the golden files valid.
        proto.element_type = self.element_type_name()
        proto.element_data = self.element_data_dumps()

        for v in self._insertion_vectors:
            proto.insertion_vectors.add().ParseFromString(v.pb_dumps())
        if self._dimensions is not None:
            proto.dimensions.ParseFromString(self._dimensions.pb_dumps())
        for f in self._features:
            pf = proto.features.add()
            pf.guid = f.guid
            pf.name = f.name
            pf.feature_type = f.feature_type
            pf.face_index = f.face_index
            for o in f.outlines:
                pf.outlines.add().ParseFromString(o.pb_dumps())
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Element":
        from .proto import element_pb2
        proto = element_pb2.Element()
        proto.ParseFromString(data)
        geometry = None
        if proto.geometry_type and proto.geometry_type != "None" and proto.geometry_data:
            geometry = cls._pb_load_geometry(proto.geometry_type, proto.geometry_data)
        elem = cls(geometry=geometry)
        elem.guid = proto.guid
        elem.name = proto.name

        from .polyline import Polyline
        from .vector import Vector

        elem._insertion_vectors = [
            Vector.pb_loads(v.SerializeToString()) for v in proto.insertion_vectors
        ]
        # HasField, not a truthiness check: (0, 0, 0) is a legitimate authored value and must
        # not be confused with "never authored", which is what None means here.
        if proto.HasField("dimensions"):
            elem._dimensions = Vector.pb_loads(proto.dimensions.SerializeToString())
        # Assigned, not minted: a feature off the wire is the SAME feature the package wrote,
        # and anything holding its guid must still find it. Empty means the file predates the
        # field, so the lazy mint is left to whoever asks first.
        elem._features = []
        for f in proto.features:
            feature = ElementFeature(
                feature_type=f.feature_type,
                face_index=f.face_index,
                outlines=[Polyline.pb_loads(o.SerializeToString()) for o in f.outlines],
                name=f.name,
            )
            if f.guid:
                feature.guid = f.guid
            elem._features.append(feature)
        return elem

    # ── Polymorphic elements ────────────────────────────────────────────────────────────
    # Port of the C++ registry (session_cpp/src/element.cpp). An Element is a geometry
    # container that knows nothing about domains; a downstream package that needs more
    # registers a factory under its own type name, and the kernel carries `element_type` and
    # `element_data` through without interpreting either.
    #
    # `pb_loads` returns `cls`, so it cannot produce a subclass it was never told about -
    # Objects.pb_loads called it and got a base Element back, silently dropping whatever the
    # package had written. `pb_loads_polymorphic` is the missing half of that round trip.

    #: type name -> factory(data: bytes) -> Element
    _registry: dict = {}

    def element_type_name(self) -> str:
        """This element's own type name, written to ``element_type``.

        The base returns ``""`` so nothing is emitted for a plain Element.
        """
        return ""

    def element_data_dumps(self) -> bytes:
        """This element's own state, written to ``element_data``.

        Opaque to the kernel - the format is the registering package's business.
        """
        return b""

    @classmethod
    def register_type(cls, type_name: str, factory) -> None:
        """Register ``factory`` for ``type_name``; re-registering the same name replaces it.

        ``factory`` takes the full serialized ``session_proto.Element`` bytes - the same
        bytes ``pb_loads`` takes - so it can read the base fields as well as ``element_data``.
        """
        if not type_name or factory is None:
            return
        Element._registry[type_name] = factory

    @staticmethod
    def is_registered(type_name: str) -> bool:
        return type_name in Element._registry

    @staticmethod
    def registered_types() -> list:
        return sorted(Element._registry)

    @classmethod
    def pb_loads_polymorphic(cls, data: bytes) -> "Element":
        """Load an element, preserving its derived type when one is registered.

        Falls back to a base Element when ``element_type`` is empty OR names a type nobody
        registered - an unknown domain type degrades to its geometry rather than failing the
        whole Session, which is what lets a viewer open a file written by a package it does
        not have.
        """
        from .proto import element_pb2
        proto = element_pb2.Element()
        proto.ParseFromString(data)

        factory = Element._registry.get(proto.element_type) if proto.element_type else None
        if factory is not None:
            derived = factory(data)
            if derived is not None:
                return derived
            # A factory returning None is a bug in that package, not a corrupt file - fall
            # through to the base so one bad type cannot take the Session with it.

        return cls.pb_loads(data)

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
        klass = type_map.get(geo_type)
        if klass is None:
            return None
        return klass.pb_loads(geo_data)

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "Element":
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())
