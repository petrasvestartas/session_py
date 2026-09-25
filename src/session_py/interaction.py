from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import ClassVar
import copy
import json
import uuid

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import interaction_pb2


# ═══════════════════════════════════════════════════════════════════════════
# Hex encoding
# ═══════════════════════════════════════════════════════════════════════════
def _to_hex(data: bytes) -> str:
    """Encode bytes as hex text, since interaction_data is opaque and JSON carries no bytes."""
    return data.hex()


def _from_hex(s: str) -> bytes:
    """Decode hex text back to bytes."""
    return bytes.fromhex(s) if s else b""


class Interaction(ABC):
    """What joins two elements, stored on their graph edge: an abstract base with a guid, a name and a registry that loads each subclass by its type name."""

    _registry: ClassVar[dict[str, Callable]] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, name: str = ""):
        """Construct from a name; the class is abstract, so only a subclass constructs one."""

        self._guid: str | None = None  # Lazily minted guid.
        self.name = name  # What joins the pair, e.g. "glue"; empty when unnamed.

    def __copy__(self):
        """Copy with the same guid, minting it on self first, so a stored interaction keeps its identity."""

        result = self.__class__.__new__(self.__class__)
        result.__dict__.update(self.__dict__)
        result._guid = self.guid

        return result

    def __deepcopy__(self, memo):
        """Copy with the same guid, a subclass's own attributes included."""

        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result

        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))

        result._guid = self.guid

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

    @abstractmethod
    def interaction_type_name(self) -> str:
        """Return the type name the subclass registered its factory under."""

    @abstractmethod
    def interaction_data_dumps(self) -> bytes:
        """Return the subclass's own state, opaque to the kernel; its factory reads it back."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __eq__(self, other) -> bool:
        """Compare name, type and data; guid ignored."""

        if not isinstance(other, Interaction):
            return False

        return (
            self.name == other.name
            and self.interaction_type_name() == other.interaction_type_name()
            and self.interaction_data_dumps() == other.interaction_data_dumps()
        )

    def __ne__(self, other) -> bool:
        """Compare name, type and data; guid ignored."""
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════════════════
    def clone(self) -> Interaction:
        """Return a polymorphic copy of the same subclass with the same guid."""
        return copy.deepcopy(self)

    # ═══════════════════════════════════════════════════════════════════════════
    # Polymorphic registry
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def register_type(type_name: str, factory: Callable) -> None:
        """Register factory for type_name; re-registering the same name replaces it."""

        if not type_name or factory is None:
            return

        Interaction._registry[type_name] = factory

    @staticmethod
    def is_registered(type_name: str) -> bool:
        """Return whether a factory is registered for type_name."""
        return type_name in Interaction._registry

    @staticmethod
    def _build_registered(
        type_name: str, data: bytes, guid: str, name: str
    ) -> Interaction:
        """The registered subclass built from its data, an InteractionUnknown when the type is unknown or its factory fails, then given the guid and the name."""

        factory = Interaction._registry.get(type_name)
        interaction = None

        if factory is not None:
            try:
                interaction = factory(data)
            except Exception:
                interaction = None

        if interaction is None:
            interaction = InteractionUnknown(type_name, data)

        if guid:
            interaction.guid = guid

        interaction.name = name

        return interaction

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        return {
            "guid": self.guid,
            "interaction_data": _to_hex(self.interaction_data_dumps()),
            "interaction_type": self.interaction_type_name(),
            "name": self.name,
            "type": "Interaction",
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Interaction:
        """Deserialize from a JSON object through the registry; an unregistered type loads as an InteractionUnknown."""

        return Interaction._build_registered(
            data.get("interaction_type", ""),
            _from_hex(data.get("interaction_data", "")),
            guid if guid is not None else data.get("guid", ""),
            name if name is not None else data.get("name", ""),
        )

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__(), separators=(",", ":"))

    @classmethod
    def file_json_loads(cls, json_string: str) -> Interaction:
        """Deserialize from a JSON string through the registry; an unregistered type loads as an InteractionUnknown."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=4)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Interaction:
        """Read from a JSON file through the registry; an unregistered type loads as an InteractionUnknown."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> interaction_pb2.Interaction:
        """Convert to the protobuf message."""

        from .proto import interaction_pb2

        proto = interaction_pb2.Interaction()
        proto.guid = self.guid
        proto.name = self.name
        proto.interaction_type = self.interaction_type_name()
        proto.interaction_data = self.interaction_data_dumps()

        return proto

    @classmethod
    def from_proto(cls, proto: interaction_pb2.Interaction) -> Interaction:
        """Construct from the protobuf message through the registry; an unregistered type loads as an InteractionUnknown."""

        return Interaction._build_registered(
            proto.interaction_type, proto.interaction_data, proto.guid, proto.name
        )

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Interaction:
        """Deserialize from protobuf bytes through the registry; an unregistered type loads as an InteractionUnknown."""

        from .proto import interaction_pb2

        proto = interaction_pb2.Interaction()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write to a protobuf file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Interaction:
        """Read from a protobuf file through the registry; an unregistered type loads as an InteractionUnknown."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "Type(name)"."""
        return f"{self.interaction_type_name()}({self.name})"

    def __repr__(self) -> str:
        """Return "Type(guid, name)"."""
        return f"{self.interaction_type_name()}({self.guid}, {self.name})"


# ═══════════════════════════════════════════════════════════════════════════
# InteractionUnknown
# ═══════════════════════════════════════════════════════════════════════════
class InteractionUnknown(Interaction):
    """An interaction whose type has no registered factory: a load keeps its type name and data so a save writes them back unchanged."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, type_name: str = "", data: bytes = b"", name: str = ""):
        """Construct from a type name, its data and a name."""

        super().__init__(name)
        self.type_name = type_name  # The type name it was written under.
        self.data = data  # Its opaque state.

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def interaction_type_name(self) -> str:
        """Return the type name it was written under."""
        return self.type_name

    def interaction_data_dumps(self) -> bytes:
        """Return its opaque state."""
        return self.data
