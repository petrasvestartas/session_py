from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING
from typing import Optional
from typing import Union
import uuid

if TYPE_CHECKING:
    from pathlib import Path

try:
    from .proto import color_pb2

    _HAS_PROTOBUF = True
except ImportError:
    _HAS_PROTOBUF = False


class Color:
    """A named color with RGBA components clamped to ``[0.0, 1.0]``.

    Parameters
    ----------
    r : float, optional
        Red component.
    g : float, optional
        Green component.
    b : float, optional
        Blue component.
    a : float, optional
        Alpha component.
    name : str, optional
        Color name.

    Attributes
    ----------
    name : str
        Color name.
    r, g, b, a : float
        Clamped color components.

    Examples
    --------
    >>> Color(1.0, 0.5, 0.25).to_unified_array()
    [1.0, 0.5, 0.25, 1.0]
    """

    def __init__(
        self,
        r: float = 1.0,
        g: float = 1.0,
        b: float = 1.0,
        a: float = 1.0,
        name: str = "my_color",
    ) -> None:
        """Construct from RGBA components, each clamped to [0.0, 1.0]"""
        self._guid = None
        self.name = name
        self.r = max(0.0, min(1.0, float(r)))
        self.g = max(0.0, min(1.0, float(g)))
        self.b = max(0.0, min(1.0, float(b)))
        self.a = max(0.0, min(1.0, float(a)))

    def __deepcopy__(self, memo: dict[int, object]) -> "Color":
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
        """Set the guid."""
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
        """Return opaque white.

        Returns
        -------
        Color
            The named white preset.

        Examples
        --------
        >>> Color.white().name
        'white'
        """
        return cls(1.0, 1.0, 1.0, 1.0, "white")

    @classmethod
    def black(cls) -> "Color":
        """Return opaque black.

        Returns
        -------
        Color
            The named black preset.

        Examples
        --------
        >>> Color.black().name
        'black'
        """
        return cls(0.0, 0.0, 0.0, 1.0, "black")

    @classmethod
    def grey(cls) -> "Color":
        """Return opaque grey.

        Returns
        -------
        Color
            The named grey preset.

        Examples
        --------
        >>> Color.grey().name
        'grey'
        """
        return cls(0.5, 0.5, 0.5, 1.0, "grey")

    @classmethod
    def red(cls) -> "Color":
        """Return opaque red.

        Returns
        -------
        Color
            The named red preset.

        Examples
        --------
        >>> Color.red().name
        'red'
        """
        return cls(1.0, 0.0, 0.0, 1.0, "red")

    @classmethod
    def orange(cls) -> "Color":
        """Return opaque orange.

        Returns
        -------
        Color
            The named orange preset.

        Examples
        --------
        >>> Color.orange().name
        'orange'
        """
        return cls(1.0, 0.5, 0.0, 1.0, "orange")

    @classmethod
    def yellow(cls) -> "Color":
        """Return opaque yellow.

        Returns
        -------
        Color
            The named yellow preset.

        Examples
        --------
        >>> Color.yellow().name
        'yellow'
        """
        return cls(1.0, 1.0, 0.0, 1.0, "yellow")

    @classmethod
    def lime(cls) -> "Color":
        """Return opaque lime.

        Returns
        -------
        Color
            The named lime preset.

        Examples
        --------
        >>> Color.lime().name
        'lime'
        """
        return cls(0.5, 1.0, 0.0, 1.0, "lime")

    @classmethod
    def green(cls) -> "Color":
        """Return opaque green.

        Returns
        -------
        Color
            The named green preset.

        Examples
        --------
        >>> Color.green().name
        'green'
        """
        return cls(0.0, 1.0, 0.0, 1.0, "green")

    @classmethod
    def mint(cls) -> "Color":
        """Return opaque mint.

        Returns
        -------
        Color
            The named mint preset.

        Examples
        --------
        >>> Color.mint().name
        'mint'
        """
        return cls(0.0, 1.0, 0.5, 1.0, "mint")

    @classmethod
    def cyan(cls) -> "Color":
        """Return opaque cyan.

        Returns
        -------
        Color
            The named cyan preset.

        Examples
        --------
        >>> Color.cyan().name
        'cyan'
        """
        return cls(0.0, 1.0, 1.0, 1.0, "cyan")

    @classmethod
    def azure(cls) -> "Color":
        """Return opaque azure.

        Returns
        -------
        Color
            The named azure preset.

        Examples
        --------
        >>> Color.azure().name
        'azure'
        """
        return cls(0.0, 0.5, 1.0, 1.0, "azure")

    @classmethod
    def blue(cls) -> "Color":
        """Return opaque blue.

        Returns
        -------
        Color
            The named blue preset.

        Examples
        --------
        >>> Color.blue().name
        'blue'
        """
        return cls(0.0, 0.0, 1.0, 1.0, "blue")

    @classmethod
    def violet(cls) -> "Color":
        """Return opaque violet.

        Returns
        -------
        Color
            The named violet preset.

        Examples
        --------
        >>> Color.violet().name
        'violet'
        """
        return cls(0.5, 0.0, 1.0, 1.0, "violet")

    @classmethod
    def magenta(cls) -> "Color":
        """Return opaque magenta.

        Returns
        -------
        Color
            The named magenta preset.

        Examples
        --------
        >>> Color.magenta().name
        'magenta'
        """
        return cls(1.0, 0.0, 1.0, 1.0, "magenta")

    @classmethod
    def pink(cls) -> "Color":
        """Return opaque pink.

        Returns
        -------
        Color
            The named pink preset.

        Examples
        --------
        >>> Color.pink().name
        'pink'
        """
        return cls(1.0, 0.0, 0.5, 1.0, "pink")

    @classmethod
    def maroon(cls) -> "Color":
        """Return opaque maroon.

        Returns
        -------
        Color
            The named maroon preset.

        Examples
        --------
        >>> Color.maroon().name
        'maroon'
        """
        return cls(0.5, 0.0, 0.0, 1.0, "maroon")

    @classmethod
    def brown(cls) -> "Color":
        """Return opaque brown.

        Returns
        -------
        Color
            The named brown preset.

        Examples
        --------
        >>> Color.brown().name
        'brown'
        """
        return cls(0.5, 0.25, 0.0, 1.0, "brown")

    @classmethod
    def olive(cls) -> "Color":
        """Return opaque olive.

        Returns
        -------
        Color
            The named olive preset.

        Examples
        --------
        >>> Color.olive().name
        'olive'
        """
        return cls(0.5, 0.5, 0.0, 1.0, "olive")

    @classmethod
    def teal(cls) -> "Color":
        """Return opaque teal.

        Returns
        -------
        Color
            The named teal preset.

        Examples
        --------
        >>> Color.teal().name
        'teal'
        """
        return cls(0.0, 0.5, 0.5, 1.0, "teal")

    @classmethod
    def navy(cls) -> "Color":
        """Return opaque navy.

        Returns
        -------
        Color
            The named navy preset.

        Examples
        --------
        >>> Color.navy().name
        'navy'
        """
        return cls(0.0, 0.0, 0.5, 1.0, "navy")

    @classmethod
    def purple(cls) -> "Color":
        """Return opaque purple.

        Returns
        -------
        Color
            The named purple preset.

        Examples
        --------
        >>> Color.purple().name
        'purple'
        """
        return cls(0.5, 0.0, 0.5, 1.0, "purple")

    @classmethod
    def silver(cls) -> "Color":
        """Return opaque silver.

        Returns
        -------
        Color
            The named silver preset.

        Examples
        --------
        >>> Color.silver().name
        'silver'
        """
        return cls(0.75, 0.75, 0.75, 1.0, "silver")

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
    def __jsonload__(
        cls,
        data: dict,
        guid: Optional[str] = None,
        name: Optional[str] = None,
    ) -> "Color":
        color = cls(data["r"], data["g"], data["b"], data["a"], name or data["name"])
        color.guid = guid or data["guid"]
        return color

    def file_json_dumps(self) -> str:
        """Serialize this color to a JSON string.

        Returns
        -------
        str
            JSON representation.
        """
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Color":
        """Deserialize a color from a JSON string.

        Parameters
        ----------
        json_string : str
            JSON representation.

        Returns
        -------
        Color
            Parsed color.
        """
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: Union[str, Path]) -> None:
        """Write this color as JSON to a file.

        Parameters
        ----------
        filepath : Union[str, pathlib.Path]
            Output path.
        """
        with open(filepath, "w") as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, Path]) -> "Color":
        """Read a color from a JSON file.

        Parameters
        ----------
        filepath : Union[str, pathlib.Path]
            Input path.

        Returns
        -------
        Color
            Parsed color.
        """
        with open(filepath) as f:
            return cls.__jsonload__(json.load(f))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def to_proto(self) -> color_pb2.Color:
        """Convert this color to its protobuf message.

        Returns
        -------
        color_pb2.Color
            Protobuf representation.

        Raises
        ------
        ImportError
            If protobuf support is unavailable.
        """
        if not _HAS_PROTOBUF:
            raise ImportError("protobuf not available")
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
    def from_proto(cls, proto: color_pb2.Color) -> "Color":
        """Construct a color from its protobuf message.

        Parameters
        ----------
        proto : color_pb2.Color
            Protobuf representation.

        Returns
        -------
        Color
            Converted color.

        Raises
        ------
        ImportError
            If protobuf support is unavailable.
        """
        if not _HAS_PROTOBUF:
            raise ImportError("protobuf not available")
        color = cls(proto.r, proto.g, proto.b, proto.a, proto.name)
        if proto.guid:
            color.guid = proto.guid
        return color

    def pb_dumps(self) -> bytes:
        """Serialize this color to protobuf bytes.

        Returns
        -------
        bytes
            Encoded protobuf message.
        """
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Color":
        """Deserialize a color from protobuf bytes.

        Parameters
        ----------
        data : bytes
            Encoded protobuf message.

        Returns
        -------
        Color
            Decoded color.

        Raises
        ------
        ImportError
            If protobuf support is unavailable.
        """
        if not _HAS_PROTOBUF:
            raise ImportError("protobuf not available")
        proto = color_pb2.Color()
        proto.ParseFromString(data)
        return cls.from_proto(proto)

    def pb_dump(self, filepath: Union[str, Path]) -> None:
        """Write this color as protobuf bytes to a file.

        Parameters
        ----------
        filepath : Union[str, pathlib.Path]
            Output path.
        """
        data = self.pb_dumps()
        with open(filepath, "wb") as f:
            written = f.write(data)
            if written != len(data):
                raise OSError(f"Failed to write protobuf file: {filepath}")

    @classmethod
    def pb_load(cls, filepath: Union[str, Path]) -> "Color":
        """Read a color from a protobuf file.

        Parameters
        ----------
        filepath : Union[str, pathlib.Path]
            Input path.

        Returns
        -------
        Color
            Decoded color.
        """
        with open(filepath, "rb") as f:
            data = f.read()
        return cls.pb_loads(data)

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════

    def __str__(self) -> str:
        """r, g, b, a"""
        return f"{self.r:.1f}, {self.g:.1f}, {self.b:.1f}, {self.a:.1f}"

    def __repr__(self) -> str:
        """Color(name, r, g, b, a)"""
        return f"Color({self.name}, {self.r:.1f}, {self.g:.1f}, {self.b:.1f}, {self.a:.1f})"
