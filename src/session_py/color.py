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

    _cache: dict = {}

    def __init__(self, r: int, g: int, b: int, a: int, name: str = "my_color"):
        self._guid = None
        self.name = name
        self._r = max(0, min(255, int(r)))
        self._g = max(0, min(255, int(g)))
        self._b = max(0, min(255, int(b)))
        self._a = max(0, min(255, int(a)))

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = max(0, min(255, int(value)))

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, value):
        self._g = max(0, min(255, int(value)))

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = max(0, min(255, int(value)))

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = max(0, min(255, int(value)))

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
        if "white" not in cls._cache:
            c = cls(255, 255, 255, 255); c.name = "white"; cls._cache["white"] = c
        return cls._cache["white"]

    @classmethod
    def black(cls) -> "Color":
        """Create a black color."""
        if "black" not in cls._cache:
            c = cls(0, 0, 0, 255); c.name = "black"; cls._cache["black"] = c
        return cls._cache["black"]

    @classmethod
    def grey(cls) -> "Color":
        """Create a grey color."""
        if "grey" not in cls._cache:
            c = cls(128, 128, 128, 255); c.name = "grey"; cls._cache["grey"] = c
        return cls._cache["grey"]

    @classmethod
    def red(cls) -> "Color":
        """Create a red color."""
        if "red" not in cls._cache:
            c = cls(255, 0, 0, 255); c.name = "red"; cls._cache["red"] = c
        return cls._cache["red"]

    @classmethod
    def orange(cls) -> "Color":
        """Create an orange color."""
        if "orange" not in cls._cache:
            c = cls(255, 128, 0, 255); c.name = "orange"; cls._cache["orange"] = c
        return cls._cache["orange"]

    @classmethod
    def yellow(cls) -> "Color":
        """Create a yellow color."""
        if "yellow" not in cls._cache:
            c = cls(255, 255, 0, 255); c.name = "yellow"; cls._cache["yellow"] = c
        return cls._cache["yellow"]

    @classmethod
    def lime(cls) -> "Color":
        """Create a lime color."""
        if "lime" not in cls._cache:
            c = cls(128, 255, 0, 255); c.name = "lime"; cls._cache["lime"] = c
        return cls._cache["lime"]

    @classmethod
    def green(cls) -> "Color":
        """Create a green color."""
        if "green" not in cls._cache:
            c = cls(0, 255, 0, 255); c.name = "green"; cls._cache["green"] = c
        return cls._cache["green"]

    @classmethod
    def mint(cls) -> "Color":
        """Create a mint color."""
        if "mint" not in cls._cache:
            c = cls(0, 255, 128, 255); c.name = "mint"; cls._cache["mint"] = c
        return cls._cache["mint"]

    @classmethod
    def cyan(cls) -> "Color":
        """Create a cyan color."""
        if "cyan" not in cls._cache:
            c = cls(0, 255, 255, 255); c.name = "cyan"; cls._cache["cyan"] = c
        return cls._cache["cyan"]

    @classmethod
    def azure(cls) -> "Color":
        """Create an azure color."""
        if "azure" not in cls._cache:
            c = cls(0, 128, 255, 255); c.name = "azure"; cls._cache["azure"] = c
        return cls._cache["azure"]

    @classmethod
    def blue(cls) -> "Color":
        """Create a blue color."""
        if "blue" not in cls._cache:
            c = cls(0, 0, 255, 255); c.name = "blue"; cls._cache["blue"] = c
        return cls._cache["blue"]

    @classmethod
    def violet(cls) -> "Color":
        """Create a violet color."""
        if "violet" not in cls._cache:
            c = cls(128, 0, 255, 255); c.name = "violet"; cls._cache["violet"] = c
        return cls._cache["violet"]

    @classmethod
    def magenta(cls) -> "Color":
        """Create a magenta color."""
        if "magenta" not in cls._cache:
            c = cls(255, 0, 255, 255); c.name = "magenta"; cls._cache["magenta"] = c
        return cls._cache["magenta"]

    @classmethod
    def pink(cls) -> "Color":
        """Create a pink color."""
        if "pink" not in cls._cache:
            c = cls(255, 0, 128, 255); c.name = "pink"; cls._cache["pink"] = c
        return cls._cache["pink"]

    @classmethod
    def maroon(cls) -> "Color":
        """Create a maroon color."""
        if "maroon" not in cls._cache:
            c = cls(128, 0, 0, 255); c.name = "maroon"; cls._cache["maroon"] = c
        return cls._cache["maroon"]

    @classmethod
    def brown(cls) -> "Color":
        """Create a brown color."""
        if "brown" not in cls._cache:
            c = cls(128, 64, 0, 255); c.name = "brown"; cls._cache["brown"] = c
        return cls._cache["brown"]

    @classmethod
    def olive(cls) -> "Color":
        """Create an olive color."""
        if "olive" not in cls._cache:
            c = cls(128, 128, 0, 255); c.name = "olive"; cls._cache["olive"] = c
        return cls._cache["olive"]

    @classmethod
    def teal(cls) -> "Color":
        """Create a teal color."""
        if "teal" not in cls._cache:
            c = cls(0, 128, 128, 255); c.name = "teal"; cls._cache["teal"] = c
        return cls._cache["teal"]

    @classmethod
    def navy(cls) -> "Color":
        """Create a navy color."""
        if "navy" not in cls._cache:
            c = cls(0, 0, 128, 255); c.name = "navy"; cls._cache["navy"] = c
        return cls._cache["navy"]

    @classmethod
    def purple(cls) -> "Color":
        """Create a purple color."""
        if "purple" not in cls._cache:
            c = cls(128, 0, 128, 255); c.name = "purple"; cls._cache["purple"] = c
        return cls._cache["purple"]

    @classmethod
    def silver(cls) -> "Color":
        """Create a silver color."""
        if "silver" not in cls._cache:
            c = cls(192, 192, 192, 255); c.name = "silver"; cls._cache["silver"] = c
        return cls._cache["silver"]

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

    def file_json_dump(self, filepath):
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
    def file_json_load(cls, filepath):
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

    def file_json_dumps(self):
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string):
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