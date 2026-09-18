from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Optional
from typing import Union
import copy
import json
import uuid

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import color_pb2


class Color:
    """A named color with RGBA components in [0.0, 1.0]"""

    __slots__ = ("_guid", "name", "r", "g", "b", "a")

    def __init__(self, r: float = 1.0, g: float = 1.0, b: float = 1.0, a: float = 1.0, name: str = "my_color"):
        self._guid = None
        self.name = name
        self.r = max(0.0, min(1.0, float(r)))
        self.g = max(0.0, min(1.0, float(g)))
        self.b = max(0.0, min(1.0, float(b)))
        self.a = max(0.0, min(1.0, float(a)))

    def __deepcopy__(self, memo):
        """Copy (new guid, same data)"""
        result = Color(self.r, self.g, self.b, self.a, self.name)
        memo[id(self)] = result

        return result

    def duplicate(self) -> "Color":
        """Copy (new guid, same data)"""
        return copy.deepcopy(self)

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
        self._guid = value

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.r

        if index == 1:
            return self.g

        if index == 2:
            return self.b

        if index == 3:
            return self.a

        raise IndexError("Index out of range")

    def __setitem__(self, index: int, value: float) -> None:
        if index == 0:
            self.r = value
        elif index == 1:
            self.g = value
        elif index == 2:
            self.b = value
        elif index == 3:
            self.a = value
        else:
            raise IndexError("Index out of range")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Color):
            return False

        return (
            self.name == other.name
            and self.r == other.r
            and self.g == other.g
            and self.b == other.b
            and self.a == other.a
        )

    def __ne__(self, other: object) -> bool:
        return not self == other

    # ═══════════════════════════════════════════════════════════════════════════
    # Presets
    # ═══════════════════════════════════════════════════════════════════════════

    @classmethod
    def white(cls) -> "Color":
        """Return opaque white."""
        return cls(1.0, 1.0, 1.0, 1.0, "white")

    @classmethod
    def black(cls) -> "Color":
        """Return opaque black."""
        return cls(0.0, 0.0, 0.0, 1.0, "black")

    @classmethod
    def grey(cls) -> "Color":
        """Return opaque grey."""
        return cls(0.5, 0.5, 0.5, 1.0, "grey")

    @classmethod
    def red(cls) -> "Color":
        """Return opaque red."""
        return cls(1.0, 0.0, 0.0, 1.0, "red")

    @classmethod
    def orange(cls) -> "Color":
        """Return opaque orange."""
        return cls(1.0, 0.5, 0.0, 1.0, "orange")

    @classmethod
    def yellow(cls) -> "Color":
        """Return opaque yellow."""
        return cls(1.0, 1.0, 0.0, 1.0, "yellow")

    @classmethod
    def lime(cls) -> "Color":
        """Return opaque lime."""
        return cls(0.5, 1.0, 0.0, 1.0, "lime")

    @classmethod
    def green(cls) -> "Color":
        """Return opaque green."""
        return cls(0.0, 1.0, 0.0, 1.0, "green")

    @classmethod
    def mint(cls) -> "Color":
        """Return opaque mint."""
        return cls(0.0, 1.0, 0.5, 1.0, "mint")

    @classmethod
    def cyan(cls) -> "Color":
        """Return opaque cyan."""
        return cls(0.0, 1.0, 1.0, 1.0, "cyan")

    @classmethod
    def azure(cls) -> "Color":
        """Return opaque azure."""
        return cls(0.0, 0.5, 1.0, 1.0, "azure")

    @classmethod
    def blue(cls) -> "Color":
        """Return opaque blue."""
        return cls(0.0, 0.0, 1.0, 1.0, "blue")

    @classmethod
    def violet(cls) -> "Color":
        """Return opaque violet."""
        return cls(0.5, 0.0, 1.0, 1.0, "violet")

    @classmethod
    def magenta(cls) -> "Color":
        """Return opaque magenta."""
        return cls(1.0, 0.0, 1.0, 1.0, "magenta")

    @classmethod
    def pink(cls) -> "Color":
        """Return opaque pink."""
        return cls(1.0, 0.0, 0.5, 1.0, "pink")

    @classmethod
    def maroon(cls) -> "Color":
        """Return opaque maroon."""
        return cls(0.5, 0.0, 0.0, 1.0, "maroon")

    @classmethod
    def brown(cls) -> "Color":
        """Return opaque brown."""
        return cls(0.5, 0.25, 0.0, 1.0, "brown")

    @classmethod
    def olive(cls) -> "Color":
        """Return opaque olive."""
        return cls(0.5, 0.5, 0.0, 1.0, "olive")

    @classmethod
    def teal(cls) -> "Color":
        """Return opaque teal."""
        return cls(0.0, 0.5, 0.5, 1.0, "teal")

    @classmethod
    def navy(cls) -> "Color":
        """Return opaque navy."""
        return cls(0.0, 0.0, 0.5, 1.0, "navy")

    @classmethod
    def purple(cls) -> "Color":
        """Return opaque purple."""
        return cls(0.5, 0.0, 0.5, 1.0, "purple")

    @classmethod
    def silver(cls) -> "Color":
        """Return opaque silver."""
        return cls(0.75, 0.75, 0.75, 1.0, "silver")

    @classmethod
    def lightgrey(cls) -> "Color":
        """Return opaque light grey, the default surface color of meshes, breps and surfaces."""
        return cls(0.9, 0.9, 0.9, 1.0, "lightgrey")

    @classmethod
    def palette(cls) -> list["Color"]:
        """The 12 spectral colors in order"""

        return [
            cls.red(),
            cls.orange(),
            cls.yellow(),
            cls.lime(),
            cls.green(),
            cls.mint(),
            cls.cyan(),
            cls.azure(),
            cls.blue(),
            cls.violet(),
            cls.magenta(),
            cls.pink(),
        ]

    # ═══════════════════════════════════════════════════════════════════════════
    # Conversion
    # ═══════════════════════════════════════════════════════════════════════════

    def to_unified_array(self) -> list[float]:
        """Components as [r, g, b, a]"""
        return [self.r, self.g, self.b, self.a]

    @classmethod
    def from_unified_array(cls, arr: list[float]) -> "Color":
        """Color from [r, g, b, a]"""
        return cls(arr[0], arr[1], arr[2], arr[3])

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "g": self.g,
            "guid": self.guid,
            "name": self.name,
            "r": self.r,
            "type": "Color",
        }

    @classmethod
    def __jsonload__(cls, data: dict, guid: Optional[str] = None, name: Optional[str] = None) -> "Color":
        color = cls(data["r"], data["g"], data["b"], data["a"], name or data["name"])
        color.guid = guid or data["guid"]

        return color

    def file_json_dumps(self) -> str:
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Color":
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        with open(filepath, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "Color":
        with open(filepath) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def to_proto(self) -> "color_pb2.Color":
        from .proto import color_pb2

        proto = color_pb2.Color()

        if self.has_guid():
            proto.guid = self.guid

        proto.name = self.name
        proto.r = self.r
        proto.g = self.g
        proto.b = self.b
        proto.a = self.a

        return proto

    @classmethod
    def from_proto(cls, proto: "color_pb2.Color") -> "Color":
        color = cls(proto.r, proto.g, proto.b, proto.a, proto.name)

        if proto.guid:
            color.guid = proto.guid

        return color

    def pb_dumps(self) -> bytes:
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Color":
        from .proto import color_pb2

        proto = color_pb2.Color()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        with open(filepath, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "Color":
        with open(filepath, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════

    def __str__(self) -> str:
        """r, g, b, a"""
        return f"{self.r:.1f}, {self.g:.1f}, {self.b:.1f}, {self.a:.1f}"

    def __repr__(self) -> str:
        """Color(name, r, g, b, a)"""
        return f"Color({self.name}, {self.r:.1f}, {self.g:.1f}, {self.b:.1f}, {self.a:.1f})"
