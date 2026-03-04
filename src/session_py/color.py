import uuid
import copy

# Import protobuf at module level for performance
try:
    from .proto import color_pb2
    _HAS_PROTOBUF = True
except ImportError:
    _HAS_PROTOBUF = False

class Color:
    """An index-based 0-255 color with RGBA values.

    Parameters
    ----------
    r : int, optional
        Red component (0-255). Defaults to 255.
    g : int, optional
        Green component (0-255). Defaults to 255.
    b : int, optional
        Blue component (0-255). Defaults to 255.
    a : int, optional
        Alpha component (0-255). Defaults to 255.
    name : str, optional
        Name of the color. Defaults to "white".

    Attributes
    ----------
    name : str
        The name of the color.
    guid : str
        The unique identifier of the color.
    """

    def __init__(self, r: int, g: int, b: int, a: int, name: str = "my_color"):
        self.guid = str(uuid.uuid4())
        self.name = name
        self._r = int(r)
        self._g = int(g)
        self._b = int(b)
        self._a = int(a)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = int(value)

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, value):
        self._g = int(value)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = int(value)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = int(value)

    ###########################################################################################
    # Operators
    ###########################################################################################

    def __deepcopy__(self, memo):

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # New guid
        result.guid = str(uuid.uuid4())

        # Copy remaining fields
        result.name = copy.deepcopy(self.name, memo)
        result._r = self._r
        result._g = self._g
        result._b = self._b
        result._a = self._a
        return result

    def duplicate(self) -> "Color":
        """Create a deep copy of this color with a new GUID.

        Returns
        -------
        :class:`Color`
            A new Color with identical RGBA values but a different GUID.

        """
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    ###########################################################################################
    # No-copy Operators
    ###########################################################################################

    def __getitem__(self, index):
        if index == 0:
            return self._r
        elif index == 1:
            return self._g
        elif index == 2:
            return self._b
        elif index == 3:
            return self._a
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        if index == 0:
            self._r = value
        elif index == 1:
            self._g = value
        elif index == 2:
            self._b = value
        elif index == 3:
            self._a = value
        else:
            raise IndexError("Index out of range")

    ###########################################################################################
    # Details
    ###########################################################################################

    def to_unified_array(self) -> list[float]:
        """Convert to normalized float array [0-1].

        Returns
        -------
        list[float]
            Array [r, g, b, a] with values normalized to [0.0, 1.0].

        """
        return [self[0] / 255.0, self[1] / 255.0, self[2] / 255.0, self[3] / 255.0]

    @classmethod
    def from_unified_array(cls, arr) -> "Color":
        """Create color from normalized float values [0-1].

        Parameters
        ----------
        arr : list[float]
            Array [r, g, b, a] with values in [0.0, 1.0] range.

        Returns
        -------
        :class:`Color`
            A new Color with values converted to 0-255 range.

        """
        return cls(
            int(arr[0] * 255.0 + 0.5),
            int(arr[1] * 255.0 + 0.5),
            int(arr[2] * 255.0 + 0.5),
            int(arr[3] * 255.0 + 0.5),
        )

    ###########################################################################################
    # Presets
    ###########################################################################################

    @classmethod
    def white(cls) -> "Color":
        """Create a white color."""
        color = cls(255, 255, 255, 255)
        color.name = "white"
        return color

    @classmethod
    def black(cls) -> "Color":
        """Create a black color."""
        color = cls(0, 0, 0, 255)
        color.name = "black"
        return color

    @classmethod
    def grey(cls) -> "Color":
        """Create a grey color."""
        color = cls(128, 128, 128, 255)
        color.name = "grey"
        return color

    @classmethod
    def red(cls) -> "Color":
        """Create a red color."""
        color = cls(255, 0, 0, 255)
        color.name = "red"
        return color

    @classmethod
    def orange(cls) -> "Color":
        """Create an orange color."""
        color = cls(255, 128, 0, 255)
        color.name = "orange"
        return color

    @classmethod
    def yellow(cls) -> "Color":
        """Create a yellow color."""
        color = cls(255, 255, 0, 255)
        color.name = "yellow"
        return color

    @classmethod
    def lime(cls) -> "Color":
        """Create a lime color."""
        color = cls(128, 255, 0, 255)
        color.name = "lime"
        return color

    @classmethod
    def green(cls) -> "Color":
        """Create a green color."""
        color = cls(0, 255, 0, 255)
        color.name = "green"
        return color

    @classmethod
    def mint(cls) -> "Color":
        """Create a mint color."""
        color = cls(0, 255, 128, 255)
        color.name = "mint"
        return color

    @classmethod
    def cyan(cls) -> "Color":
        """Create a cyan color."""
        color = cls(0, 255, 255, 255)
        color.name = "cyan"
        return color

    @classmethod
    def azure(cls) -> "Color":
        """Create an azure color."""
        color = cls(0, 128, 255, 255)
        color.name = "azure"
        return color

    @classmethod
    def blue(cls) -> "Color":
        """Create a blue color."""
        color = cls(0, 0, 255, 255)
        color.name = "blue"
        return color

    @classmethod
    def violet(cls) -> "Color":
        """Create a violet color."""
        color = cls(128, 0, 255, 255)
        color.name = "violet"
        return color

    @classmethod
    def magenta(cls) -> "Color":
        """Create a magenta color."""
        color = cls(255, 0, 255, 255)
        color.name = "magenta"
        return color

    @classmethod
    def pink(cls) -> "Color":
        """Create a pink color."""
        color = cls(255, 0, 128, 255)
        color.name = "pink"
        return color

    @classmethod
    def maroon(cls) -> "Color":
        """Create a maroon color."""
        color = cls(128, 0, 0, 255)
        color.name = "maroon"
        return color

    @classmethod
    def brown(cls) -> "Color":
        """Create a brown color."""
        color = cls(128, 64, 0, 255)
        color.name = "brown"
        return color

    @classmethod
    def olive(cls) -> "Color":
        """Create an olive color."""
        color = cls(128, 128, 0, 255)
        color.name = "olive"
        return color

    @classmethod
    def teal(cls) -> "Color":
        """Create a teal color."""
        color = cls(0, 128, 128, 255)
        color.name = "teal"
        return color

    @classmethod
    def navy(cls) -> "Color":
        """Create a navy color."""
        color = cls(0, 0, 128, 255)
        color.name = "navy"
        return color

    @classmethod
    def purple(cls) -> "Color":
        """Create a purple color."""
        color = cls(128, 0, 128, 255)
        color.name = "purple"
        return color

    @classmethod
    def silver(cls) -> "Color":
        """Create a silver color."""
        color = cls(192, 192, 192, 255)
        color.name = "silver"
        return color

    @classmethod
    def palette(cls):
        """Return a palette of 12 spectral colors in order."""
        return [cls.red(), cls.orange(), cls.yellow(), cls.lime(), cls.green(), cls.mint(), cls.cyan(), cls.azure(), cls.blue(), cls.violet(), cls.magenta(), cls.pink()]

    ###########################################################################################
    # JSON Serialization
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field."""
        # Alphabetical order to match Rust's serde_json
        return {
            "a": self[3],
            "b": self[2],
            "g": self[1],
            "guid": self.guid,
            "name": self.name,
            "r": self[0],
            "type": f"{self.__class__.__name__}",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format."""
        color = cls(data["r"], data["g"], data["b"], data.get("a", 255))
        color.guid = guid if guid is not None else data.get("guid", color.guid)
        color.name = name if name is not None else data.get("name", color.name)
        return color

    def json_dump(self, filepath):
        """Write JSON to file.

        Parameters
        ----------
        filepath : str or Path
            Path to the output file.

        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        """Read JSON from file.

        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file.

        Returns
        -------
        :class:`Color`
            The deserialized Color.

        """
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def json_dumps(self):
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, json_string):
        """Load from JSON string."""
        import json
        return cls.__jsonload__(json.loads(json_string))

    ###########################################################################################
    # Protobuf Serialization
    ###########################################################################################

    def pb_dumps(self):
        """Convert to protobuf binary format.

        Returns
        -------
        bytes
            Serialized protobuf data.

        Raises
        ------
        ImportError
            If protobuf module is not available.

        """
        if not _HAS_PROTOBUF:
            raise ImportError("protobuf not available")
        proto = color_pb2.Color()
        proto.guid = self.guid
        proto.name = self.name
        proto.r = self[0]
        proto.g = self[1]
        proto.b = self[2]
        proto.a = self[3]
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        """Create color from protobuf binary data.

        Parameters
        ----------
        data : bytes
            Protobuf-encoded color data.

        Returns
        -------
        :class:`Color`
            The deserialized Color.

        Raises
        ------
        ImportError
            If protobuf module is not available.

        """
        if not _HAS_PROTOBUF:
            raise ImportError("protobuf not available")
        proto = color_pb2.Color()
        proto.ParseFromString(data)
        
        color = cls(proto.r, proto.g, proto.b, proto.a)
        color.guid = proto.guid
        color.name = proto.name
        return color

    def pb_dump(self, filepath):
        """Write protobuf to file.

        Parameters
        ----------
        filepath : str
            Path to the output file.

        """
        data = self.pb_dumps()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath):
        """Read protobuf from file.

        Parameters
        ----------
        filepath : str
            Path to the protobuf file.

        Returns
        -------
        :class:`Color`
            The deserialized Color.

        """
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)

    def __str__(self) -> str:
        """String representation."""
        return f"{self[0]}, {self[1]}, {self[2]}, {self[3]}"

    def __repr__(self) -> str:
        return f"Color({self.name}, {self[0]}, {self[1]}, {self[2]}, {self[3]})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Color):
            return False
        return (
            self.name == other.name
            and self[0] == other[0]
            and self[1] == other[1]
            and self[2] == other[2]
            and self[3] == other[3]
        )

    def __ne__(self, other) -> bool:
        return not self == other