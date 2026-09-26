from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import json
import uuid
from .collection import Collection
from .point import Point
from .line import Line
from .plane import Plane
from .obb import OBB
from .polyline import Polyline
from .pointcloud import PointCloud
from .mesh import Mesh
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .brep import BRep
from .element import Element
from .instance_ref import InstanceRef

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import objects_pb2


def _clone_list(items: Collection, memo: dict) -> Collection:
    """One list, duplicated: new list, new objects, same guids."""

    out = Collection()

    for item in items:
        clone = copy.deepcopy(item, memo)

        if item.has_guid():
            clone.guid = item.guid

        out.append(clone)

    return out


def _dump_list(items: Collection) -> list:
    """Serialize every object of a list to JSON."""

    out = []

    for item in items:
        out.append(item.__jsondump__())

    return out


def _load_list(data: dict, key: str) -> Collection:
    """Load every object under key, keeping guids."""

    from .file_encoders import file_decode_node

    out = Collection()

    for item in data.get(key, []):
        out.append(file_decode_node(item))

    return out


def _dump_pb_list(items: Collection, repeated) -> None:
    """Convert every object of a list into a repeated proto field."""

    for item in items:
        repeated.add().CopyFrom(item.to_proto())


def _load_pb_list(repeated, cls) -> Collection:
    """Load every message of a repeated proto field, keeping guids."""

    out = Collection()

    for item in repeated:
        out.append(cls.from_proto(item))

    return out


class Component:
    """A custom domain object stored generically in a Session; every field except type/guid/name lives in extra."""

    def __init__(self):
        """Construct an empty component."""

        self._guid = None  # Lazily minted GUID.
        self.type_name = ""  # Class name, e.g. "FloorBuilder".
        self.name = "my_component"  # Human-readable name.
        self.extra = {}  # All custom fields.

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def has_guid(self) -> bool:
        """Return whether the lazy guid has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the guid, creating it on first access."""

        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        """Set the guid."""
        self._guid = value

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        data = dict(self.extra)
        data["guid"] = self.guid
        data["name"] = self.name
        data["type"] = self.type_name

        return dict(sorted(data.items()))

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Component:
        """Deserialize from a JSON object."""

        component = cls()
        component.guid = (
            guid if guid is not None else data.get("guid", str(uuid.uuid4()))
        )
        component.name = name if name is not None else data.get("name", "my_component")
        component.type_name = data.get("type", "")
        component.extra = dict(data)
        component.extra.pop("guid", None)
        component.extra.pop("name", None)
        component.extra.pop("type", None)

        return component

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> objects_pb2.Component:
        """Convert to the protobuf message."""

        from .proto import objects_pb2

        proto = objects_pb2.Component()
        proto.type_name = self.type_name
        proto.guid = self.guid
        proto.name = self.name
        proto.json_data = json.dumps(self.extra)

        return proto

    @classmethod
    def from_proto(cls, proto: objects_pb2.Component) -> Component:
        """Construct from the protobuf message."""

        component = cls()
        component.type_name = proto.type_name
        component.guid = proto.guid
        component.name = proto.name

        if proto.json_data:
            component.extra = json.loads(proto.json_data)

        return component

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Component:
        """Deserialize from protobuf bytes."""

        from .proto import objects_pb2

        proto = objects_pb2.Component()
        proto.ParseFromString(data)

        return cls.from_proto(proto)


class Objects:
    """A collection of geometry objects."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, name: str = "my_objects"):
        """Construct an empty collection with every list allocated."""

        self._guid = None  # Lazily minted GUID.
        self.name = name  # The name of the collection.
        self.points = Collection()  # Points.
        self.lines = Collection()  # Lines.
        self.planes = Collection()  # Planes.
        self.bboxes = Collection()  # Bounding boxes.
        self.polylines = Collection()  # Polylines.
        self.pointclouds = Collection()  # Point clouds.
        self.meshes = Collection()  # Meshes.
        self.nurbscurves = Collection()  # NURBS curves.
        self.nurbssurfaces = Collection()  # NURBS surfaces.
        self.breps = Collection()  # BReps.
        self.elements = Collection()  # Elements.
        self.components = Collection()  # Components.
        self.instances = Collection()  # Instances, each placing a definition by guid.

    def __deepcopy__(self, memo):
        """Copy every list and every object in it, guids included, so a Session's indexes still match."""

        result = Objects(self.name)

        if self.has_guid():
            result.guid = self.guid

        result.points = _clone_list(self.points, memo)
        result.lines = _clone_list(self.lines, memo)
        result.planes = _clone_list(self.planes, memo)
        result.bboxes = _clone_list(self.bboxes, memo)
        result.polylines = _clone_list(self.polylines, memo)
        result.pointclouds = _clone_list(self.pointclouds, memo)
        result.meshes = _clone_list(self.meshes, memo)
        result.nurbscurves = _clone_list(self.nurbscurves, memo)
        result.nurbssurfaces = _clone_list(self.nurbssurfaces, memo)
        result.breps = _clone_list(self.breps, memo)
        result.elements = _clone_list(self.elements, memo)
        result.components = copy.deepcopy(self.components, memo)
        result.instances = _clone_list(self.instances, memo)
        memo[id(self)] = result

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def has_guid(self) -> bool:
        """Return whether the lazy guid has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the guid, creating it on first access."""

        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        """Set the guid."""
        self._guid = value

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        return {
            "bboxes": _dump_list(self.bboxes),
            "breps": _dump_list(self.breps),
            "components": _dump_list(self.components),
            "elements": _dump_list(self.elements),
            "guid": self.guid,
            "instances": _dump_list(self.instances),
            "lines": _dump_list(self.lines),
            "meshes": _dump_list(self.meshes),
            "name": self.name,
            "nurbscurves": _dump_list(self.nurbscurves),
            "nurbssurfaces": _dump_list(self.nurbssurfaces),
            "planes": _dump_list(self.planes),
            "pointclouds": _dump_list(self.pointclouds),
            "points": _dump_list(self.points),
            "polylines": _dump_list(self.polylines),
            "type": "Objects",
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Objects:
        """Deserialize from a JSON object."""

        objects = cls(name if name is not None else data.get("name", "my_objects"))
        objects.guid = guid if guid is not None else data.get("guid", str(uuid.uuid4()))
        objects.bboxes = _load_list(data, "bboxes")
        objects.breps = _load_list(data, "breps")
        objects.elements = _load_list(data, "elements")
        objects.instances = _load_list(data, "instances")
        objects.lines = _load_list(data, "lines")
        objects.meshes = _load_list(data, "meshes")
        objects.nurbscurves = _load_list(data, "nurbscurves")
        objects.nurbssurfaces = _load_list(data, "nurbssurfaces")
        objects.planes = _load_list(data, "planes")
        objects.pointclouds = _load_list(data, "pointclouds")
        objects.points = _load_list(data, "points")
        objects.polylines = _load_list(data, "polylines")

        for component in data.get("components", []):
            if not isinstance(component, Component):
                component = Component.__jsonload__(component)

            objects.components.append(component)

        return objects

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__(), separators=(",", ":"))

    @classmethod
    def file_json_loads(cls, json_string: str) -> Objects:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=4)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Objects:
        """Read from a JSON file."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> objects_pb2.Objects:
        """Convert to the protobuf message."""

        from .proto import objects_pb2

        proto = objects_pb2.Objects()
        proto.name = self.name

        if self.has_guid():
            proto.guid = self.guid

        _dump_pb_list(self.points, proto.points)
        _dump_pb_list(self.lines, proto.lines)
        _dump_pb_list(self.planes, proto.planes)
        _dump_pb_list(self.bboxes, proto.bboxes)
        _dump_pb_list(self.polylines, proto.polylines)
        _dump_pb_list(self.pointclouds, proto.pointclouds)
        _dump_pb_list(self.meshes, proto.meshes)
        _dump_pb_list(self.nurbscurves, proto.nurbscurves)
        _dump_pb_list(self.nurbssurfaces, proto.nurbssurfaces)
        _dump_pb_list(self.breps, proto.breps)
        _dump_pb_list(self.elements, proto.elements)
        _dump_pb_list(self.components, proto.components)
        _dump_pb_list(self.instances, proto.instances)

        return proto

    @classmethod
    def from_proto(cls, proto: objects_pb2.Objects) -> Objects:
        """Construct from the protobuf message; elements load through the polymorphic registry."""

        objects = cls(proto.name)

        if proto.guid:
            objects.guid = proto.guid

        objects.points = _load_pb_list(proto.points, Point)
        objects.lines = _load_pb_list(proto.lines, Line)
        objects.planes = _load_pb_list(proto.planes, Plane)
        objects.bboxes = _load_pb_list(proto.bboxes, OBB)
        objects.polylines = _load_pb_list(proto.polylines, Polyline)
        objects.pointclouds = _load_pb_list(proto.pointclouds, PointCloud)
        objects.meshes = _load_pb_list(proto.meshes, Mesh)
        objects.nurbscurves = _load_pb_list(proto.nurbscurves, NurbsCurve)
        objects.nurbssurfaces = _load_pb_list(proto.nurbssurfaces, NurbsSurface)
        objects.breps = _load_pb_list(proto.breps, BRep)

        for element in proto.elements:
            objects.elements.append(
                Element.pb_loads_polymorphic(element.SerializeToString())
            )

        objects.components = _load_pb_list(proto.components, Component)
        objects.instances = _load_pb_list(proto.instances, InstanceRef)

        return objects

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Objects:
        """Deserialize from protobuf bytes."""

        from .proto import objects_pb2

        proto = objects_pb2.Objects()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write to a protobuf file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Objects:
        """Read from a protobuf file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "Objects(name=..., guid=..., points=...)"."""
        return f"Objects(name={self.name}, guid={self.guid}, points={len(self.points)})"

    def __repr__(self) -> str:
        """Return "Objects(name=..., guid=..., points=...)"."""
        return str(self)
