from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import json
import uuid
from .color import Color
from .point import Point

if TYPE_CHECKING:
    from pathlib import Path
    from .objects import Component
    from .proto import objects_pb2


class Text:
    """Text or TeX with a position, pixel height, colour, outline and projection mode."""

    def __init__(self, text: str = "", at: Point | None = None):
        """Construct text at a position."""

        self._guid = None  # Lazily minted GUID.
        self.name = "my_text"  # Text name.
        self.text = text  # Plain text or TeX source.
        self.at = (
            at if at is not None else Point()
        )  # Position in the text coordinate system.
        self.height = 28.0  # Letter height in CSS pixels.
        self.color = Color.white()  # Text colour.
        self.outline = 0.0  # Glyph outline width in CSS pixels; negative is black.
        self.math = False  # Whether text contains TeX.
        self.spatial = False  # Whether the text belongs to the navigable 3D region.
        self.serif = False  # Whether to use the serif mathematics face.

    @staticmethod
    def from_tex(text: str, at: Point | None = None, spatial: bool = False) -> Text:
        """Construct TeX with flat-panel or fixed-size spatial defaults."""

        result = Text(text, at)
        result.math = True
        result.spatial = spatial
        result.height = 18.0 if spatial else 28.0
        result.outline = -1.5 if spatial else 0.0
        result.serif = not spatial

        return result

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = Text(self.text, copy.deepcopy(self.at, memo))
        result.name = self.name
        result.height = self.height
        result.color = copy.deepcopy(self.color, memo)
        result.outline = self.outline
        result.math = self.math
        result.spatial = self.spatial
        result.serif = self.serif
        memo[id(self)] = result

        return result

    def duplicate(self) -> Text:
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
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __eq__(self, other) -> bool:
        """Compare every text property, ignoring the guid."""

        if not isinstance(other, Text):
            return False

        return (
            self.name == other.name
            and self.text == other.text
            and self.at == other.at
            and self.height == other.height
            and self.color == other.color
            and self.outline == other.outline
            and self.math == other.math
            and self.spatial == other.spatial
            and self.serif == other.serif
        )

    def __ne__(self, other) -> bool:
        """Return whether two texts differ."""
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to an ordered JSON object."""

        return {
            "at": self.at.__jsondump__(),
            "color": self.color.__jsondump__(),
            "guid": self.guid,
            "height": self.height,
            "math": self.math,
            "name": self.name,
            "outline": self.outline,
            "serif": self.serif,
            "spatial": self.spatial,
            "text": self.text,
            "type": "Text",
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Text:
        """Deserialize from a JSON object."""

        result = cls(data.get("text", ""))
        result.guid = guid if guid is not None else data.get("guid", str(uuid.uuid4()))
        result.name = name if name is not None else data.get("name", "my_text")
        result.height = data.get("height", 28.0)
        result.outline = data.get("outline", 0.0)
        result.math = data.get("math", False)
        result.spatial = data.get("spatial", False)
        result.serif = data.get("serif", False)

        if "at" in data:
            result.at = Point.__jsonload__(data["at"])

        if "color" in data:
            result.color = Color.__jsonload__(data["color"])

        return result

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__(), separators=(",", ":"))

    @classmethod
    def file_json_loads(cls, json_string: str) -> Text:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write JSON to a file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=4)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Text:
        """Read JSON from a file."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> objects_pb2.Component:
        """Convert to the generic component protobuf message."""
        return self.to_component().to_proto()

    @classmethod
    def from_proto(cls, proto: objects_pb2.Component) -> Text:
        """Construct from a generic component protobuf message."""

        from .objects import Component

        return cls.from_component(Component.from_proto(proto))

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Text:
        """Deserialize from protobuf bytes."""

        from .proto import objects_pb2

        proto = objects_pb2.Component()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write protobuf bytes to a file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Text:
        """Read protobuf bytes from a file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # Component
    # ═══════════════════════════════════════════════════════════════════════════
    def to_component(self) -> Component:
        """Return a Session component of type Text with this guid and name."""

        from .objects import Component

        component = Component()
        component.type_name = "Text"
        component.guid = self.guid
        component.name = self.name
        component.extra = self.__jsondump__()
        del component.extra["guid"]
        del component.extra["name"]
        del component.extra["type"]

        return component

    @classmethod
    def from_component(cls, component: Component) -> Text:
        """Construct from a component made by to_component."""

        data = dict(component.extra)
        data["guid"] = component.guid
        data["name"] = component.name

        return cls.__jsonload__(data)

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return the text content."""
        return self.text

    def __repr__(self) -> str:
        """Return a representation with every property."""
        return f"Text({json.dumps(self.__jsondump__(), separators=(',', ':'))})"
