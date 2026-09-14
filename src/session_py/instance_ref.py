from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union
import copy
import json
import uuid
from .color import Color
from .tolerance import Tolerance
from .tolerance import TOLERANCE
from .xform import Xform

if TYPE_CHECKING:
    from pathlib import Path


class InstanceRef:
    """A block reference: places a definition (by guid) at a transform"""

    __slots__ = ("_guid", "name", "definition_guid", "xform", "color", "flags")

    def __init__(self, definition_guid: str = "", xform: Xform | None = None):
        self._guid = None
        self.name = "my_instance_ref"
        self.definition_guid = definition_guid
        self.xform = Xform.identity() if xform is None else xform.duplicate()
        self.color = Color.white()
        self.flags = 0

    def __deepcopy__(self, memo):
        """Copy (new guid, same data)"""
        result = InstanceRef(self.definition_guid, self.xform)
        result.name = self.name
        result.color = copy.deepcopy(self.color, memo)
        result.flags = self.flags
        memo[id(self)] = result
        return result

    def duplicate(self) -> "InstanceRef":
        """Copy (new guid, same data)"""
        return copy.deepcopy(self)

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

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def with_name(name: str, definition_guid: str, xform: Xform) -> "InstanceRef":
        """Instance with a name, a definition guid and a placement"""
        ref = InstanceRef(definition_guid, xform)
        ref.name = name
        return ref

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════

    def __getitem__(self, index: int) -> float:
        """Placement matrix entry by index (0..15, column-major)"""
        if index < 0 or index >= 16:
            raise IndexError("Index out of bounds")
        return self.xform.m[index]

    def __setitem__(self, index: int, value: float) -> None:
        if index < 0 or index >= 16:
            raise IndexError("Index out of bounds")
        self.xform.m[index] = value

    def __eq__(self, other) -> bool:
        if not isinstance(other, InstanceRef):
            return False
        return (
            self.definition_guid == other.definition_guid
            and self.xform == other.xform
            and self.color == other.color
            and self.flags == other.flags
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════

    def transform(self, t: Xform) -> None:
        """Compose in place: xform = t * xform"""
        self.xform = t * self.xform

    def transformed(self, t: Xform) -> "InstanceRef":
        """Composed copy"""
        result = self.duplicate()
        result.transform(t)
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self) -> dict:
        return {
            "color": self.color.__jsondump__(),
            "definition_guid": self.definition_guid,
            "flags": self.flags,
            "guid": self.guid,
            "name": self.name,
            "type": "InstanceRef",
            "xform": self.xform.__jsondump__(),
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str = None, name: str = None
    ) -> "InstanceRef":
        from .file_encoders import file_decode_node

        ref = cls(data["definition_guid"], file_decode_node(data["xform"]))
        ref.color = file_decode_node(data["color"])
        ref.flags = data["flags"]
        ref.guid = guid if guid is not None else data["guid"]
        ref.name = name if name is not None else data["name"]
        return ref

    def file_json_dumps(self) -> str:
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "InstanceRef":
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        with open(filepath, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "InstanceRef":
        with open(filepath) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def pb_dumps(self) -> bytes:
        from .proto import instance_ref_pb2

        proto = instance_ref_pb2.InstanceRef()
        if self.has_guid():
            proto.guid = self._guid
        proto.name = self.name
        proto.definition_guid = self.definition_guid
        proto.xform.name = self.xform.name
        for i in range(16):
            proto.xform.matrix.append(self.xform.m[i])
        proto.color.name = self.color.name
        proto.color.r = self.color.r
        proto.color.g = self.color.g
        proto.color.b = self.color.b
        proto.color.a = self.color.a
        proto.flags = self.flags
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "InstanceRef":
        from .proto import instance_ref_pb2

        proto = instance_ref_pb2.InstanceRef()
        proto.ParseFromString(data)
        ref = cls()
        if proto.guid:
            ref.guid = proto.guid
        ref.name = proto.name
        ref.definition_guid = proto.definition_guid
        ref.xform.name = proto.xform.name
        for i in range(min(len(proto.xform.matrix), 16)):
            ref.xform.m[i] = proto.xform.matrix[i]
        ref.color.name = proto.color.name
        ref.color.r = proto.color.r
        ref.color.g = proto.color.g
        ref.color.b = proto.color.b
        ref.color.a = proto.color.a
        ref.flags = proto.flags
        return ref

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        with open(filepath, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "InstanceRef":
        with open(filepath, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════

    def __str__(self) -> str:
        """definition_guid @ [tx, ty, tz]"""
        prec = Tolerance.ROUNDING
        return f"{self.definition_guid} @ [{TOLERANCE.format_number(self.xform.m[12], prec)}, {TOLERANCE.format_number(self.xform.m[13], prec)}, {TOLERANCE.format_number(self.xform.m[14], prec)}]"

    def __repr__(self) -> str:
        """InstanceRef(name, definition_guid, Color(...), flags)"""
        return f"InstanceRef({self.name}, {self.definition_guid}, {repr(self.color)}, {self.flags})"
