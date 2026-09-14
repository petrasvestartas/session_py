from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import ClassVar
import uuid
import copy
from .xform import Xform

if TYPE_CHECKING:
    from pathlib import Path
    from .brep import BRep
    from .line import Line
    from .mesh import Mesh
    from .obb import OBB
    from .plane import Plane
    from .point import Point
    from .polyline import Polyline
    from .vector import Vector


# ═══════════════════════════════════════════════════════════════════════════
# Hex encoding
# ═══════════════════════════════════════════════════════════════════════════


def _to_hex(data: bytes) -> str:
    """``element_data`` is opaque bytes and JSON has none, so it travels as hex, identically in the three kernels."""
    return data.hex()


def _from_hex(s: str) -> bytes:
    return bytes.fromhex(s) if s else b""


# ═══════════════════════════════════════════════════════════════════════════
# ElementFeature
# ═══════════════════════════════════════════════════════════════════════════


class ElementFeature:
    """One serializable modification of a host element - a cut, a drill, a joint pocket - that the kernel draws but never applies."""

    def __init__(
        self,
        feature_type: str = "",
        face_index: int = -1,
        outlines: list[Polyline] | None = None,
        name: str = "",
    ):
        self._guid: str | None = None
        self.name = name
        self.feature_type = feature_type
        self.face_index = face_index
        self.outlines = list(outlines or [])

    def __deepcopy__(self, memo):
        """A copy is a new feature and mints its own guid."""
        result = ElementFeature(
            self.feature_type,
            self.face_index,
            copy.deepcopy(self.outlines, memo),
            self.name,
        )
        memo[id(self)] = result
        return result

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
        """Clear the guid so a fresh one mints lazily on next read."""
        self._guid = None

    def __eq__(self, other) -> bool:
        """Data equality, not identity: the guid is ignored."""
        if not isinstance(other, ElementFeature):
            return NotImplemented
        return (
            self.name == other.name
            and self.feature_type == other.feature_type
            and self.face_index == other.face_index
            and self.outlines == other.outlines
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # ElementFeature - JSON
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self):
        return {
            "face_index": self.face_index,
            "feature_type": self.feature_type,
            "guid": self.guid,
            "name": self.name,
            "outlines": [o.__jsondump__() for o in self.outlines],
            "type": "ElementFeature",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        from .polyline import Polyline

        f = cls()
        f.face_index = data.get("face_index", -1)
        f.feature_type = data.get("feature_type", "")
        g = guid if guid is not None else data.get("guid", "")
        if g:
            f.guid = g
        f.name = name if name is not None else data.get("name", "")
        for o in data.get("outlines", []):
            f.outlines.append(Polyline.__jsonload__(o))
        return f

    def file_json_dumps(self) -> str:
        import json

        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, s: str) -> ElementFeature:
        import json

        return cls.__jsonload__(json.loads(s))

    def file_json_dump(self, filepath: str | Path) -> None:
        import json

        with open(filepath, "w") as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: str | Path) -> ElementFeature:
        import json

        with open(filepath) as f:
            return cls.__jsonload__(json.load(f))

    # ═══════════════════════════════════════════════════════════════════════════
    # ElementFeature - Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def pb_dumps(self) -> bytes:
        from .proto import element_pb2

        proto = element_pb2.ElementFeature()
        if self.has_guid():
            proto.guid = self._guid
        proto.name = self.name
        proto.feature_type = self.feature_type
        proto.face_index = self.face_index
        for o in self.outlines:
            proto.outlines.add().ParseFromString(o.pb_dumps())
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> ElementFeature:
        from .polyline import Polyline
        from .proto import element_pb2

        proto = element_pb2.ElementFeature()
        proto.ParseFromString(data)
        f = cls()
        if proto.guid:
            f.guid = proto.guid
        f.name = proto.name
        f.feature_type = proto.feature_type
        f.face_index = proto.face_index
        for o in proto.outlines:
            f.outlines.append(Polyline.pb_loads(o.SerializeToString()))
        return f

    def pb_dump(self, filepath: str | Path) -> None:
        with open(filepath, "wb") as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: str | Path) -> ElementFeature:
        with open(filepath, "rb") as f:
            return cls.pb_loads(f.read())

    def __str__(self) -> str:
        return f"ElementFeature({self.feature_type}, face {self.face_index}, {len(self.outlines)} outline(s))"

    def __repr__(self) -> str:
        return self.__str__()


# ═══════════════════════════════════════════════════════════════════════════
# Element
# ═══════════════════════════════════════════════════════════════════════════


class Element:
    def __init__(self, geometry: Mesh | BRep | None = None, name: str = "my_element"):
        self._guid: str | None = None
        self.name = name
        self._geometry = geometry
        self._geometry_ops: list[Callable] = []
        self._features: list[ElementFeature] = []
        self._insertion_vectors: list[Vector] = []
        self._dimensions: Vector | None = None
        self._element_type = ""
        self._element_data = b""
        self.reset()

    def __deepcopy__(self, memo):
        """A copy is a new element and mints its own guid."""
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        result._guid = None
        result.name = self.name
        result._geometry = copy.deepcopy(self._geometry, memo)
        result._geometry_ops = list(self._geometry_ops)
        result._features = copy.deepcopy(self._features, memo)
        result._insertion_vectors = copy.deepcopy(self._insertion_vectors, memo)
        result._dimensions = copy.deepcopy(self._dimensions, memo)
        result._element_type = self._element_type
        result._element_data = self._element_data
        result.reset()
        return result

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

    @property
    def geometry(self) -> Mesh | BRep | None:
        return self._geometry

    @property
    def has_geometry(self) -> bool:
        return self._geometry is not None

    @property
    def geometry_type_name(self) -> str:
        from .brep import BRep
        from .mesh import Mesh

        if isinstance(self._geometry, Mesh):
            return "Mesh"
        if isinstance(self._geometry, BRep):
            return "BRep"
        return "None"

    def session_geometry(self, xform: Xform) -> Mesh | BRep | None:
        """The geometry placed by ``xform``; the Session owns the placement, so pass identity for local geometry."""
        from .mesh import Mesh

        if self._geometry is None:
            return None
        geo = copy.deepcopy(self._geometry)
        if isinstance(geo, Mesh):
            geo = self._apply_geometry_ops(geo)
        if not xform.is_identity():
            geo.transform(xform)
        return geo

    @property
    def aabb(self) -> OBB:
        if self._is_dirty or self._aabb is None:
            self._aabb = self._compute_aabb()
            self._is_dirty = False
        return self._aabb

    @property
    def obb(self) -> OBB:
        if self._is_dirty or self._obb is None:
            self._obb = self._compute_obb()
            self._is_dirty = False
        return self._obb

    @property
    def collision_mesh(self) -> Mesh:
        if self._is_dirty or self._collision_mesh is None:
            self._collision_mesh = self._compute_collision_mesh()
            self._is_dirty = False
        return self._collision_mesh

    @property
    def point(self) -> Point:
        if self._is_dirty or self._point is None:
            self._point = self._compute_point()
            self._is_dirty = False
        return self._point

    @property
    def polylines(self) -> list[Polyline]:
        if self._is_dirty or self._polylines is None:
            self._polylines = self._compute_polylines()
            self._is_dirty = False
        return self._polylines

    @property
    def planes(self) -> list[Plane]:
        if self._is_dirty or self._planes is None:
            self._planes = self._compute_planes()
            self._is_dirty = False
        return self._planes

    @property
    def edge_vectors(self) -> list[Vector]:
        if self._is_dirty or self._edge_vectors is None:
            self._edge_vectors = self._compute_edge_vectors()
            self._is_dirty = False
        return self._edge_vectors

    @property
    def axis(self) -> Line | None:
        if self._is_dirty or self._axis is None:
            self._axis = self._compute_axis()
            self._is_dirty = False
        return self._axis

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty

    @property
    def cached_aabb(self) -> OBB | None:
        return self._aabb

    @property
    def cached_obb(self) -> OBB | None:
        return self._obb

    @property
    def cached_collision_mesh(self) -> Mesh | None:
        return self._collision_mesh

    @property
    def cached_point(self) -> Point | None:
        return self._point

    @property
    def geometry_ops_count(self) -> int:
        return len(self._geometry_ops)

    @property
    def features_count(self) -> int:
        return len(self._features)

    @property
    def features(self) -> list[ElementFeature]:
        """Modifications carried by this element and written with it; ``add_geometry_op`` is the in-memory counterpart that is not."""
        return self._features

    @features.setter
    def features(self, features: list[ElementFeature]) -> None:
        self._features = list(features)

    @property
    def insertion_vectors(self) -> list[Vector]:
        """Direction(s) the element is inserted along when the assembly is put together, one per jointed face."""
        return self._insertion_vectors

    @insertion_vectors.setter
    def insertion_vectors(self, vectors: list[Vector]) -> None:
        self._insertion_vectors = list(vectors)

    @property
    def dimensions(self) -> Vector | None:
        """Nominal extents in the element's own frame (plate: x/y outline, z thickness) - authored intent, not the measured ``obb``; None = never authored."""
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value: Vector | None) -> None:
        self._dimensions = value

    def element_type_name(self) -> str:
        """The derived type name this element was loaded with, written to ``element_type``; a plain Element authored in memory returns ``""``."""
        return self._element_type

    def element_data_dumps(self) -> bytes:
        """The derived type's own state, opaque to the kernel and carried through untouched."""
        return self._element_data

    # ═══════════════════════════════════════════════════════════════════════════
    # Element - Mutators
    # ═══════════════════════════════════════════════════════════════════════════

    def add_geometry_op(self, op: Callable) -> None:
        self._geometry_ops.append(op)
        self.reset()

    def add_feature(self, feature: ElementFeature) -> None:
        self._features.append(feature)

    def place(self, xform: Xform) -> None:
        """Bake a placement into the element's own geometry, invalidating the cached boxes."""
        self._geometry = self.session_geometry(xform)
        self.reset()

    def set_geometry(self, geometry: Mesh | BRep | None) -> None:
        self._geometry = geometry
        self.reset()

    def set_polylines(self, polylines: list[Polyline]) -> None:
        self._polylines = polylines

    def set_planes(self, planes: list[Plane]) -> None:
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

    # ═══════════════════════════════════════════════════════════════════════════
    # Element - Operators
    # ═══════════════════════════════════════════════════════════════════════════

    def __eq__(self, other) -> bool:
        """Data equality, not identity: every field that survives a round trip, guid excluded."""
        if not isinstance(other, Element):
            return False
        return (
            self.name == other.name
            and self.geometry_type_name == other.geometry_type_name
            and self.element_type_name() == other.element_type_name()
            and self.element_data_dumps() == other.element_data_dumps()
            and self._insertion_vectors == other._insertion_vectors
            and self._dimensions == other._dimensions
            and self._features == other._features
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # Element - Computation
    # ═══════════════════════════════════════════════════════════════════════════

    def duplicate(self) -> Element:
        return copy.deepcopy(self)

    def _compute_aabb(self) -> OBB:
        return self._obb_from_geometry(self.session_geometry(Xform.identity()))

    def _compute_obb(self) -> OBB:
        return self._obb_from_geometry(self.session_geometry(Xform.identity()))

    def _compute_collision_mesh(self) -> Mesh:
        from .mesh import Mesh

        geo = self.session_geometry(Xform.identity())
        if isinstance(geo, Mesh):
            return geo
        return Mesh()

    def _compute_point(self) -> Point:
        from .point import Point

        return Point.centroid(
            self._points_from_geometry(self.session_geometry(Xform.identity()))
        )

    def _compute_polylines(self) -> list[Polyline]:
        """A mesh solid is its face outlines; a domain type with its own face order overrides this."""
        from .mesh import Mesh

        if isinstance(self._geometry, Mesh):
            return self._geometry.face_outlines()
        return []

    def _compute_planes(self) -> list[Plane]:
        """One plane per face outline: centroid origin, Newell normal, closing point dropped first."""
        from .plane import Plane
        from .point import Point
        from .vector import Vector

        planes = []
        for outline in self._compute_polylines():
            points = outline.get_points()
            if len(points) > 1 and points[0] == points[-1]:
                points.pop()
            if len(points) < 3:
                continue
            planes.append(
                Plane.from_point_normal(
                    Point.centroid(points), Vector.average_normal(points)
                )
            )
        return planes

    def _compute_edge_vectors(self) -> list[Vector]:
        return []

    def _compute_axis(self) -> Line | None:
        return None

    def _apply_geometry_ops(self, geo: Mesh) -> Mesh:
        for f in self._geometry_ops:
            geo = f(geo)
        return geo

    @staticmethod
    def _points_from_geometry(geo: Mesh | BRep | None) -> list[Point]:
        from .brep import BRep
        from .mesh import Mesh

        points = []
        if isinstance(geo, Mesh):
            for v in geo.vertex.values():
                points.append(v.position())
        if isinstance(geo, BRep):
            points = geo.vertex_points()
        return points

    @staticmethod
    def _obb_from_geometry(geo: Mesh | BRep | None) -> OBB:
        from .obb import OBB
        from .point import Point

        points = Element._points_from_geometry(geo)
        if not points:
            return OBB.from_point(Point(0, 0, 0), 0.0)
        return OBB.from_points(points, 0.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # Element - JSON
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self):
        geo_data = self._geometry.__jsondump__() if self._geometry is not None else None
        dims = self._dimensions.__jsondump__() if self._dimensions is not None else None
        return {
            "dimensions": dims,
            "element_data": _to_hex(self.element_data_dumps()),
            "element_type": self.element_type_name(),
            "features": [f.__jsondump__() for f in self._features],
            "geometry_data": geo_data,
            "geometry_type": self.geometry_type_name,
            "guid": self.guid,
            "insertion_vectors": [v.__jsondump__() for v in self._insertion_vectors],
            "name": self.name,
            "type": "Element",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        from .brep import BRep
        from .mesh import Mesh
        from .vector import Vector

        elem = cls()
        geo_type = data.get("geometry_type", "None")
        geo_data = data.get("geometry_data")
        if geo_type == "Mesh" and geo_data is not None:
            elem._geometry = Mesh.__jsonload__(geo_data)
        if geo_type == "BRep" and geo_data is not None:
            elem._geometry = BRep.__jsonload__(geo_data)
        g = guid if guid is not None else data.get("guid", "")
        if g:
            elem.guid = g
        elem.name = name if name is not None else data.get("name", elem.name)
        dims = data.get("dimensions")
        if dims is not None:
            elem._dimensions = Vector.__jsonload__(dims)
        elem._element_type = data.get("element_type", "")
        elem._element_data = _from_hex(data.get("element_data", ""))
        for f in data.get("features", []):
            elem._features.append(ElementFeature.__jsonload__(f))
        for v in data.get("insertion_vectors", []):
            elem._insertion_vectors.append(Vector.__jsonload__(v))
        return elem

    def file_json_dumps(self) -> str:
        import json

        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, s: str) -> Element:
        import json

        return cls.__jsonload__(json.loads(s))

    def file_json_dump(self, filepath: str | Path) -> None:
        import json

        with open(filepath, "w") as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: str | Path) -> Element:
        import json

        with open(filepath) as f:
            return cls.__jsonload__(json.load(f))

    # ═══════════════════════════════════════════════════════════════════════════
    # Element - Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def pb_dumps(self) -> bytes:
        from .proto import element_pb2

        proto = element_pb2.Element()
        if self.has_guid():
            proto.guid = self._guid
        proto.name = self.name
        proto.geometry_type = self.geometry_type_name
        if self._geometry is not None:
            proto.geometry_data = self._geometry.pb_dumps()
        proto.element_type = self.element_type_name()
        proto.element_data = self.element_data_dumps()
        for v in self._insertion_vectors:
            proto.insertion_vectors.extend([v[0], v[1], v[2]])
        if self._dimensions is not None:
            proto.dimensions.extend(
                [self._dimensions[0], self._dimensions[1], self._dimensions[2]]
            )
        for f in self._features:
            proto.features.add().ParseFromString(f.pb_dumps())
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Element:
        from .brep import BRep
        from .mesh import Mesh
        from .proto import element_pb2
        from .vector import Vector

        proto = element_pb2.Element()
        proto.ParseFromString(data)
        elem = cls()
        if proto.guid:
            elem.guid = proto.guid
        elem.name = proto.name
        has_data = len(proto.geometry_data) > 0
        if proto.geometry_type == "Mesh" and has_data:
            elem._geometry = Mesh.pb_loads(proto.geometry_data)
        if proto.geometry_type == "BRep" and has_data:
            elem._geometry = BRep.pb_loads(proto.geometry_data)
        elem._element_type = proto.element_type
        elem._element_data = proto.element_data
        iv = proto.insertion_vectors
        for i in range(0, len(iv) - 2, 3):
            elem._insertion_vectors.append(Vector(iv[i], iv[i + 1], iv[i + 2]))
        if len(proto.dimensions) == 3:
            elem._dimensions = Vector(
                proto.dimensions[0], proto.dimensions[1], proto.dimensions[2]
            )
        for f in proto.features:
            elem._features.append(ElementFeature.pb_loads(f.SerializeToString()))
        return elem

    def pb_dump(self, filepath: str | Path) -> None:
        with open(filepath, "wb") as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: str | Path) -> Element:
        with open(filepath, "rb") as f:
            return cls.pb_loads(f.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # Element - Polymorphic registry
    # ═══════════════════════════════════════════════════════════════════════════

    _registry: ClassVar[dict[str, Callable]] = {}

    @staticmethod
    def register_type(type_name: str, factory: Callable) -> None:
        """Register ``factory`` for ``type_name``; re-registering the same name replaces it."""
        if not type_name or factory is None:
            return
        Element._registry[type_name] = factory

    @staticmethod
    def is_registered(type_name: str) -> bool:
        return type_name in Element._registry

    @staticmethod
    def registered_types() -> list[str]:
        return sorted(Element._registry)

    @staticmethod
    def _build_registered(type_name: str, data: bytes) -> Element | None:
        """A factory that raises or returns None degrades to the base exactly like an unregistered type."""
        factory = Element._registry.get(type_name) if type_name else None
        if factory is None:
            return None
        try:
            return factory(data)
        except Exception:
            return None

    @classmethod
    def pb_loads_polymorphic(cls, data: bytes) -> Element:
        """Load through the registered factory, degrading to a base Element that still carries ``element_type``/``element_data`` when the type is unknown or the factory fails."""
        from .proto import element_pb2

        proto = element_pb2.Element()
        proto.ParseFromString(data)
        derived = cls._build_registered(proto.element_type, data)
        if derived is not None:
            return derived
        return cls.pb_loads(data)

    @classmethod
    def file_json_loads_polymorphic(cls, s: str) -> Element:
        """The same from JSON, re-encoded to proto bytes so one registration serves both formats."""
        base = cls.file_json_loads(s)
        derived = cls._build_registered(base.element_type_name(), base.pb_dumps())
        if derived is not None:
            return derived
        return base

    def __str__(self) -> str:
        return f"Element({self.name}, {self.geometry_type_name})"

    def __repr__(self) -> str:
        return f"Element({self.guid}, {self.name}, {self.geometry_type_name})"
