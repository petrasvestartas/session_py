from __future__ import annotations
from typing import Union
from typing import Optional
from typing import TYPE_CHECKING
import uuid
from .color import Color
from .xform import Xform

if TYPE_CHECKING:
    from pathlib import Path


class InstanceRef:
    """A block reference: places a definition (by guid) at a transform.

    The only per-instance data is the placement ``xform``; the geometry lives
    once in the definition the ``definition_guid`` points to. Mirrors the Rhino
    block model.

    Parameters
    ----------
    definition_guid : str, optional
        Guid of the definition this instance places. Defaults to "".
    xform : Xform, optional
        Placement transform. Defaults to identity.

    Attributes
    ----------
    guid : str
        Unique identifier of the instance.
    name : str
        Name of the instance.
    definition_guid : str
        Guid of the definition this instance places.
    xform : Xform
        Placement transform (the only per-instance data).
    color : Color
        Per-instance color override.
    flags : int
        Reserved: selection / cull / visibility.
    """

    def __init__(self, definition_guid: str = "", xform: Optional["Xform"] = None):
        self._guid = None
        self.name = "my_instance_ref"
        self.definition_guid = definition_guid
        self._xform = xform
        self._color = None
        self.flags = 0

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    @property
    def xform(self) -> "Xform":
        if getattr(self, '_xform', None) is None:
            self._xform = Xform.identity()
        return self._xform

    @xform.setter
    def xform(self, value: "Xform") -> None:
        self._xform = value

    @property
    def color(self) -> "Color":
        if self._color is None:
            self._color = Color.white()
        return self._color

    @color.setter
    def color(self, value: "Color") -> None:
        self._color = value

    @classmethod
    def with_name(cls, name: str, definition_guid: str, xform: "Xform") -> "InstanceRef":
        """Create an instance reference with a specific name.

        Returns
        -------
        InstanceRef
            New named instance reference.
        """
        ref = cls(definition_guid, xform)
        ref.name = name
        return ref

    def duplicate(self) -> "InstanceRef":
        """Create a deep copy of this instance with a new GUID.

        Returns
        -------
        :class:`InstanceRef`
            A new InstanceRef with identical values but a different GUID.
        """
        import copy
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def transform(self, t: "Xform") -> None:
        """Compose an extra transform onto the placement (in-place): xform = t * xform."""
        self.xform = t * self.xform

    def transformed(self, t: "Xform") -> "InstanceRef":
        """Return a copy with an extra transform composed onto the placement.

        Returns
        -------
        :class:`InstanceRef`
            A new InstanceRef with the transform composed.
        """
        import copy
        result = copy.deepcopy(self)
        result.transform(t)
        return result

    def __getitem__(self, index):
        """Get placement matrix element by index (0-15, column-major)."""
        return self.xform.m[index]

    def __setitem__(self, index, value):
        """Set placement matrix element by index (0-15, column-major)."""
        if index < 0 or index >= 16:
            raise IndexError("Index out of bounds")
        self.xform.m[index] = value

    def __eq__(self, other):
        if not isinstance(other, InstanceRef):
            return NotImplemented
        return (
            self.definition_guid == other.definition_guid
            and self.xform == other.xform
            and self.color == other.color
            and self.flags == other.flags
        )

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    # ═══════════════════════════════════════════════════════════════════════════
    # Polymorphic JSON Serialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field."""
        # Alphabetical order to match Rust's serde_json
        return {
            "color": self.color.__jsondump__(),
            "definition_guid": self.definition_guid,
            "flags": self.flags,
            "guid": self.guid,
            "name": self.name,
            "type": f"{self.__class__.__name__}",
            "xform": self.xform.__jsondump__(),
        }

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write JSON to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "InstanceRef":
        """Read JSON from file."""
        import json
        with open(filepath) as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def file_json_dumps(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "InstanceRef":
        """Load from JSON string."""
        import json
        return cls.__jsonload__(json.loads(json_string))

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format."""
        from .file_encoders import file_decode_node

        ref = cls(data.get("definition_guid", ""))
        ref.guid = guid if guid is not None else data.get("guid", ref.guid)
        ref.name = name if name is not None else data.get("name", ref.name)
        if "xform" in data:
            ref.xform = file_decode_node(data["xform"])
        if "color" in data:
            ref.color = file_decode_node(data["color"])
        if "flags" in data:
            ref.flags = data["flags"]
        return ref

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf Serialization
    # ═══════════════════════════════════════════════════════════════════════════

    def pb_dumps(self) -> bytes:
        """Convert to protobuf binary format."""
        from .proto import instance_ref_pb2

        proto = instance_ref_pb2.InstanceRef()
        proto.guid = self.guid
        proto.name = self.name
        proto.definition_guid = self.definition_guid
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)
        proto.color.r = self.color.r
        proto.color.g = self.color.g
        proto.color.b = self.color.b
        proto.color.a = self.color.a
        proto.flags = self.flags
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "InstanceRef":
        """Create InstanceRef from protobuf binary data."""
        from .proto import instance_ref_pb2

        proto = instance_ref_pb2.InstanceRef()
        proto.ParseFromString(data)

        ref = cls(proto.definition_guid)
        ref.guid = proto.guid
        ref.name = proto.name
        if proto.HasField('xform'):
            ref.xform = Xform()
            ref.xform.name = proto.xform.name
            ref.xform.m = list(proto.xform.matrix)
        if proto.HasField('color'):
            ref.color = Color(proto.color.r, proto.color.g, proto.color.b, proto.color.a)
        ref.flags = proto.flags
        return ref

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write protobuf to file."""
        data = self.pb_dumps()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "InstanceRef":
        """Read protobuf from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)

    def __str__(self):
        """String representation (definition + placement translation)."""
        return f"{self.definition_guid} @ [{self.xform.m[12]}, {self.xform.m[13]}, {self.xform.m[14]}]"

    def __repr__(self):
        """Detailed representation."""
        return f"InstanceRef({self.name}, {self.definition_guid}, {repr(self.color)}, {self.flags})"
