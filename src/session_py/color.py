import uuid
import copy

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
        """Duplicate the color."""
        return copy.deepcopy(self)

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

    def to_float_array(self) -> list[float]:
        """Convert to normalized float array [0-1] (matches Rust implementation)."""
        return [self[0] / 255.0, self[1] / 255.0, self[2] / 255.0, self[3] / 255.0]

    @classmethod
    def from_float(cls, r, g, b, a) -> "Color":
        """Create color from normalized float values [0-1]."""
        return cls(r * 255.0, g * 255.0, b * 255.0, a * 255.0)

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

    ###########################################################################################
    # JSON Serialization
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field."""
        return {
            "type": f"{self.__class__.__name__}",
            "guid": self.guid,
            "name": self.name,
            "r": self[0],
            "g": self[1],
            "b": self[2],
            "a": self[3],
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format."""
        color = cls(data["r"], data["g"], data["b"], data.get("a", 255))
        color.guid = guid
        color.name = name
        return color

    ###########################################################################################
    # Protobuf Serialization
    ###########################################################################################

    def to_protobuf(self):
        """Convert to protobuf binary format."""
        from .proto import color_pb2
        
        proto = color_pb2.Color()
        proto.guid = self.guid
        proto.name = self.name
        proto.r = self[0]
        proto.g = self[1]
        proto.b = self[2]
        proto.a = self[3]
        return proto.SerializeToString()

    @classmethod
    def from_protobuf(cls, data):
        """Create color from protobuf binary data."""
        from .proto import color_pb2
        
        proto = color_pb2.Color()
        proto.ParseFromString(data)
        
        color = cls(proto.r, proto.g, proto.b, proto.a)
        color.guid = proto.guid
        color.name = proto.name
        return color

    def protobuf_dump(self, filepath):
        """Write protobuf to file."""
        data = self.to_protobuf()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def protobuf_load(cls, filepath):
        """Read protobuf from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.from_protobuf(data)