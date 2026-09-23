from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    from .proto import objects_pb2
    from pathlib import Path

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
import json
import uuid


class Component:
    """A custom domain object stored generically in a Session; every field except type/guid/name lives in extra."""

    def __init__(
        self, type_name: str = "", name: str = "my_component", extra: dict | None = None
    ):
        """Construct from type name, name and custom fields."""

        self._guid = None
        self.type_name = type_name  # Class name, e.g. "FloorBuilder".
        self.name = name  # Human-readable name.
        self.extra = extra if extra is not None else {}  # All custom fields.

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

    def __jsondump__(self):
        """Serialize to a JSON object."""

        data = dict(self.extra)
        data["type"] = self.type_name
        data["guid"] = self.guid
        data["name"] = self.name

        return data

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from a JSON object."""

        component = cls()
        component.type_name = data.get("type", "")
        component.guid = (
            guid if guid is not None else data.get("guid", str(uuid.uuid4()))
        )
        component.name = name if name is not None else data.get("name", "my_component")
        component.extra = dict(data)
        component.extra.pop("type", None)
        component.extra.pop("guid", None)
        component.extra.pop("name", None)

        return component

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""

        from .proto import objects_pb2

        proto = objects_pb2.Component()
        proto.type_name = self.type_name
        proto.guid = self.guid
        proto.name = self.name
        proto.json_data = json.dumps(self.extra)

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Component":
        """Deserialize from protobuf bytes."""

        from .proto import objects_pb2

        proto = objects_pb2.Component()
        proto.ParseFromString(data)
        component = cls()
        component.type_name = proto.type_name
        component.guid = proto.guid
        component.name = proto.name
        component.extra = json.loads(proto.json_data) if proto.json_data else {}

        return component


class Objects:
    """A collection of geometry objects."""

    def __init__(self, name: str = "my_objects"):
        """Construct an empty collection with every list allocated."""

        self._guid = None
        self.name = name  # The name of the collection.
        self.points: list[Point] = []
        self.lines: list[Line] = []
        self.planes: list[Plane] = []
        self.bboxes: list[OBB] = []
        self.polylines: list[Polyline] = []
        self.pointclouds: list[PointCloud] = []
        self.meshes: list[Mesh] = []
        self.nurbscurves: list[NurbsCurve] = []
        self.nurbssurfaces: list[NurbsSurface] = []
        self.breps: list[BRep] = []
        self.elements: list[Element] = []
        self.components: list[Component] = []

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

    def __str__(self):
        """Return a string representation of the collection."""
        return f"Objects(name={self.name}, guid={self.guid}, points={len(self.points)})"

    def __repr__(self):
        """Return a string representation of the collection for debugging."""
        return str(self)

    # ═══════════════════════════════════════════════════════════════════════════
    # Serialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self):
        """Serialize to a JSON object."""

        return {
            "type": "Objects",
            "guid": self.guid,
            "name": self.name,
            "bboxes": [b.__jsondump__() for b in self.bboxes],
            "breps": [b.__jsondump__() for b in self.breps],
            "components": [c.__jsondump__() for c in self.components],
            "elements": [e.__jsondump__() for e in self.elements],
            "lines": [l.__jsondump__() for l in self.lines],
            "meshes": [m.__jsondump__() for m in self.meshes],
            "nurbscurves": [nc.__jsondump__() for nc in self.nurbscurves],
            "nurbssurfaces": [ns.__jsondump__() for ns in self.nurbssurfaces],
            "planes": [pl.__jsondump__() for pl in self.planes],
            "pointclouds": [pc.__jsondump__() for pc in self.pointclouds],
            "points": [p.__jsondump__() for p in self.points],
            "polylines": [pl.__jsondump__() for pl in self.polylines],
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from a JSON object."""

        from .file_encoders import file_decode_node

        objects = cls(name if name is not None else data.get("name", "my_objects"))
        objects.guid = guid if guid is not None else data.get("guid", str(uuid.uuid4()))
        objects.bboxes = [file_decode_node(b) for b in data.get("bboxes", [])]
        objects.breps = [file_decode_node(b) for b in data.get("breps", [])]
        objects.components = [
            Component.__jsonload__(c) for c in data.get("components", [])
        ]
        objects.elements = [file_decode_node(e) for e in data.get("elements", [])]
        objects.lines = [file_decode_node(l) for l in data.get("lines", [])]
        objects.meshes = [file_decode_node(m) for m in data.get("meshes", [])]
        objects.nurbscurves = [
            file_decode_node(nc) for nc in data.get("nurbscurves", [])
        ]
        objects.nurbssurfaces = [
            file_decode_node(ns) for ns in data.get("nurbssurfaces", [])
        ]
        objects.planes = [file_decode_node(pl) for pl in data.get("planes", [])]
        objects.pointclouds = [
            file_decode_node(pc) for pc in data.get("pointclouds", [])
        ]
        objects.points = [file_decode_node(p) for p in data.get("points", [])]
        objects.polylines = [file_decode_node(pl) for pl in data.get("polylines", [])]

        return objects

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, s: str) -> "Objects":
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(s))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "Objects":
        """Read from a JSON file."""
        with open(filepath) as f:
            return cls.__jsonload__(json.load(f))

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""

        from .proto import objects_pb2

        proto = objects_pb2.Objects()
        proto.name = self.name

        if self.has_guid():
            proto.guid = self._guid

        for p in self.points:
            proto.points.add().ParseFromString(p.pb_dumps())

        for l in self.lines:
            proto.lines.add().ParseFromString(l.pb_dumps())

        for pl in self.planes:
            proto.planes.add().ParseFromString(pl.pb_dumps())

        for b in self.bboxes:
            proto.bboxes.add().ParseFromString(b.pb_dumps())

        for pl in self.polylines:
            proto.polylines.add().CopyFrom(pl.to_proto())

        for pc in self.pointclouds:
            proto.pointclouds.add().ParseFromString(pc.pb_dumps())

        for m in self.meshes:
            m.pb_fill(proto.meshes.add())

        for nc in self.nurbscurves:
            proto.nurbscurves.add().CopyFrom(nc.to_proto())

        for ns in self.nurbssurfaces:
            ns.pb_fill(proto.nurbssurfaces.add())

        for b in self.breps:
            proto.breps.add().ParseFromString(b.pb_dumps())

        for e in self.elements:
            proto.elements.add().ParseFromString(e.pb_dumps())

        for c in self.components:
            proto.components.add().ParseFromString(c.pb_dumps())

        return proto.SerializeToString()

    @classmethod
    def from_proto(cls, proto: "objects_pb2.Objects") -> "Objects":
        """Construct from a decoded proto message."""

        objects = cls(proto.name)

        if proto.guid:
            objects.guid = proto.guid

        for p in proto.points:
            objects.points.append(Point.pb_loads(p.SerializeToString()))

        for l in proto.lines:
            objects.lines.append(Line.pb_loads(l.SerializeToString()))

        for pl in proto.planes:
            objects.planes.append(Plane.pb_loads(pl.SerializeToString()))

        for b in proto.bboxes:
            objects.bboxes.append(OBB.pb_loads(b.SerializeToString()))

        for pl in proto.polylines:
            objects.polylines.append(Polyline.pb_loads(pl.SerializeToString()))

        for pc in proto.pointclouds:
            objects.pointclouds.append(PointCloud.pb_loads(pc.SerializeToString()))

        for m in proto.meshes:
            objects.meshes.append(Mesh.from_proto(m))

        for nc in proto.nurbscurves:
            objects.nurbscurves.append(NurbsCurve.pb_loads(nc.SerializeToString()))

        for ns in proto.nurbssurfaces:
            objects.nurbssurfaces.append(NurbsSurface.pb_loads(ns.SerializeToString()))

        for b in proto.breps:
            objects.breps.append(BRep.pb_loads(b.SerializeToString()))

        for e in proto.elements:
            objects.elements.append(Element.pb_loads_polymorphic(e.SerializeToString()))

        for c in proto.components:
            objects.components.append(Component.pb_loads(c.SerializeToString()))

        return objects

    @classmethod
    def pb_loads(cls, data: bytes) -> "Objects":
        """Deserialize from protobuf bytes."""

        from .proto import objects_pb2

        proto = objects_pb2.Objects()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write to a protobuf file."""
        with open(filepath, "wb") as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "Objects":
        """Read from a protobuf file."""
        with open(filepath, "rb") as f:
            return cls.pb_loads(f.read())
