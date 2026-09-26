from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import json
import uuid
from .color import Color
from .element import ElementFeature
from .tolerance import Tolerance
from .tolerance import TOLERANCE
from .xform import Xform

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import instance_ref_pb2


class InstanceRef:
    """A block reference: places a definition (by guid) at a transform."""

    __slots__ = (
        "_guid",
        "name",
        "definition_guid",
        "xform",
        "color",
        "flags",
        "features",
    )

    FLAG_HIDDEN = 1  # Not drawn.
    FLAG_LOCKED = 2  # Not selectable.
    FLAG_COLOR = 4  # color overrides the definition's.

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, definition_guid: str = "", xform: Xform | None = None):
        """Construct from a definition guid and a placement."""

        self._guid = None  # Lazily minted GUID.
        self.name = "my_instance_ref"  # Instance name.
        self.definition_guid = definition_guid  # Guid of the referenced definition.
        self.xform = Xform.identity()  # Placement outside a Session; inside one Session.xforms places the instance and this stays identity.
        self.color = Color.white()  # Display color, used when flags has FLAG_COLOR.
        self.flags = 0  # FLAG_* bits.
        self.features: list[ElementFeature] = []  # In the definition frame.

        if xform is not None:
            self.xform = xform.duplicate()

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = InstanceRef(self.definition_guid, self.xform)
        result.name = self.name
        result.color = copy.deepcopy(self.color, memo)
        result.flags = self.flags
        result.features = copy.deepcopy(self.features, memo)
        memo[id(self)] = result

        return result

    def duplicate(self) -> InstanceRef:
        """Copy with a new guid and the same data."""
        return copy.deepcopy(self)

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
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def with_name(name: str, definition_guid: str, xform: Xform) -> InstanceRef:
        """Construct from a name, a definition guid and a placement."""

        ref = InstanceRef(definition_guid, xform)
        ref.name = name

        return ref

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __getitem__(self, index: int) -> float:
        """Return the placement matrix entry by index (0..15, column-major)."""

        if index < 0 or index >= 16:
            raise IndexError("Index out of bounds")

        return self.xform.m[index]

    def __setitem__(self, index: int, value: float) -> None:
        """Set the placement matrix entry by index (0..15, column-major)."""

        if index < 0 or index >= 16:
            raise IndexError("Index out of bounds")

        self.xform.m[index] = value

    def __eq__(self, other) -> bool:
        """Compare definition guid, placement, color, flags and features."""

        if not isinstance(other, InstanceRef):
            return False

        return (
            self.definition_guid == other.definition_guid
            and self.xform == other.xform
            and self.color == other.color
            and self.flags == other.flags
            and self.features == other.features
        )

    def __ne__(self, other) -> bool:
        """Compare definition guid, placement, color, flags and features."""
        return not self == other

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════
    def transform(self, t: Xform) -> None:
        """Compose in place: xform = t * xform."""
        self.xform = t * self.xform

    def transformed(self, t: Xform) -> InstanceRef:
        """Return a composed copy."""

        result = self.duplicate()
        result.transform(t)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        features = []

        for feature in self.features:
            features.append(feature.__jsondump__())

        return {
            "color": self.color.__jsondump__(),
            "definition_guid": self.definition_guid,
            "features": features,
            "flags": self.flags,
            "guid": self.guid,
            "name": self.name,
            "type": "InstanceRef",
            "xform": self.xform.__jsondump__(),
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> InstanceRef:
        """Deserialize from a JSON object; missing flags and features default to none."""

        from .file_encoders import file_decode_node

        ref = cls(data["definition_guid"])
        ref.xform = file_decode_node(data["xform"])
        ref.color = file_decode_node(data["color"])
        ref.flags = data.get("flags", 0)
        ref.guid = guid if guid is not None else data["guid"]
        ref.name = name if name is not None else data["name"]

        for feature in data.get("features", []):
            ref.features.append(file_decode_node(feature))

        return ref

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__(), separators=(",", ":"))

    @classmethod
    def file_json_loads(cls, json_string: str) -> InstanceRef:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=4)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> InstanceRef:
        """Read from a JSON file."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> instance_ref_pb2.InstanceRef:
        """Convert to the protobuf message; an identity xform, and color without FLAG_COLOR, are not written."""

        from .proto import instance_ref_pb2

        proto = instance_ref_pb2.InstanceRef()

        if self.has_guid():
            proto.guid = self.guid

        proto.name = self.name
        proto.definition_guid = self.definition_guid

        if not self.xform.is_identity():
            proto.xform.CopyFrom(self.xform.to_proto())

        if self.flags & InstanceRef.FLAG_COLOR:
            proto.color.CopyFrom(self.color.to_proto())

        proto.flags = self.flags

        for feature in self.features:
            proto.features.add().CopyFrom(feature.to_proto())

        return proto

    @classmethod
    def from_proto(cls, proto: instance_ref_pb2.InstanceRef) -> InstanceRef:
        """Construct from the protobuf message; an absent xform is identity and an absent color white."""

        ref = cls()

        if proto.guid:
            ref.guid = proto.guid

        ref.name = proto.name
        ref.definition_guid = proto.definition_guid

        if proto.HasField("xform"):
            ref.xform = Xform.from_proto(proto.xform)

        if proto.HasField("color"):
            ref.color = Color.from_proto(proto.color)

        ref.flags = proto.flags

        for feature in proto.features:
            ref.features.append(ElementFeature.from_proto(feature))

        return ref

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> InstanceRef:
        """Deserialize from protobuf bytes."""

        from .proto import instance_ref_pb2

        proto = instance_ref_pb2.InstanceRef()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write to a protobuf file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> InstanceRef:
        """Read from a protobuf file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "definition_guid @ [tx, ty, tz]"."""

        prec = Tolerance.ROUNDING

        return f"{self.definition_guid} @ [{TOLERANCE.format_number(self.xform.m[12], prec)}, {TOLERANCE.format_number(self.xform.m[13], prec)}, {TOLERANCE.format_number(self.xform.m[14], prec)}]"

    def __repr__(self) -> str:
        """Return "InstanceRef(name, definition_guid, Color(...), flags)"."""
        return f"InstanceRef({self.name}, {self.definition_guid}, {repr(self.color)}, {self.flags})"
