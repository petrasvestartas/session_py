from __future__ import annotations
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    from .point import Point
    from .proto import tolerance_pb2
    from .vector import Vector


PI = math.pi  # Circle constant.
TWO_PI = 2.0 * math.pi  # Full turn in radians.
HALF_PI = math.pi / 2.0  # Quarter turn in radians.
TO_DEGREES = 180.0 / math.pi  # Radian-to-degree factor.
TO_RADIANS = math.pi / 180.0  # Degree-to-radian factor.
SCALE = 1e6  # Default coordinate-key scale.


class Tolerance:
    """Tolerance settings for geometric comparisons"""

    PI = math.pi  # Circle constant.
    TWO_PI = 2.0 * math.pi  # Full turn in radians.
    HALF_PI = math.pi / 2.0  # Quarter turn in radians.
    TO_DEGREES = 180.0 / math.pi  # Radian-to-degree factor.
    TO_RADIANS = math.pi / 180.0  # Degree-to-radian factor.

    ABSOLUTE = 1e-9  # Default absolute tolerance.
    RELATIVE = 1e-6  # Default relative tolerance.
    ANGULAR = 1e-6  # Default angular tolerance.
    APPROXIMATION = 1e-3  # Default approximation tolerance.
    PRECISION = 3  # Default decimal precision.
    LINEARDEFLECTION = 1e-3  # Default linear deflection.
    ANGULARDEFLECTION = 1e-1  # Default angular deflection.
    ANGLE_TOLERANCE_DEGREES = 0.11  # Angular tolerance in degrees.
    ZERO_TOLERANCE = 1e-12  # Used heavily by algorithms; do not change.
    ROUNDING = 6  # Default coordinate-key rounding.

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, unit: str = "M"):
        """Construct tolerance with a unit system ("M" or "MM")"""

        self._unit = unit
        self._absolute = None
        self._relative = None
        self._angular = None
        self._approximation = None
        self._precision = None
        self._lineardeflection = None
        self._angulardeflection = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def unit(self) -> str:
        """Current unit system"""
        return self._unit

    def absolute(self) -> float:
        """Absolute tolerance value (or default ABSOLUTE)"""
        return self._absolute if self._absolute is not None else self.ABSOLUTE

    def relative(self) -> float:
        """Relative tolerance value (or default RELATIVE)"""
        return self._relative if self._relative is not None else self.RELATIVE

    def angular(self) -> float:
        """Angular tolerance value in radians (or default ANGULAR)"""
        return self._angular if self._angular is not None else self.ANGULAR

    def approximation(self) -> float:
        """Approximation tolerance (or default APPROXIMATION)"""

        return (
            self._approximation
            if self._approximation is not None
            else self.APPROXIMATION
        )

    def precision(self) -> int:
        """Decimal precision used for formatting (or default PRECISION)"""
        return self._precision if self._precision is not None else self.PRECISION

    def lineardeflection(self) -> float:
        """Linear deflection value (or default LINEARDEFLECTION)"""

        return (
            self._lineardeflection
            if self._lineardeflection is not None
            else self.LINEARDEFLECTION
        )

    def angulardeflection(self) -> float:
        """Angular deflection value (or default ANGULARDEFLECTION)"""

        return (
            self._angulardeflection
            if self._angulardeflection is not None
            else self.ANGULARDEFLECTION
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Mutators
    # ═══════════════════════════════════════════════════════════════════════════
    def reset(self) -> None:
        """Reset all overrides to default constants"""

        self._absolute = None
        self._relative = None
        self._angular = None
        self._approximation = None
        self._precision = None
        self._lineardeflection = None
        self._angulardeflection = None

    def set_unit(self, value: str) -> None:
        """Set current unit system"""

        if value != "M" and value != "MM":
            raise ValueError(f"Invalid unit: {value}")

        self._unit = value

    def set_absolute(self, value: float) -> None:
        """Override absolute tolerance"""
        self._absolute = value

    def set_relative(self, value: float) -> None:
        """Override relative tolerance"""
        self._relative = value

    def set_angular(self, value: float) -> None:
        """Override angular tolerance (radians)"""
        self._angular = value

    def set_approximation(self, value: float) -> None:
        """Override approximation tolerance"""
        self._approximation = value

    def set_precision(self, value: int) -> None:
        """Override decimal precision for formatting"""

        if value == 0:
            raise ValueError("Precision cannot be zero.")

        self._precision = value

    def set_lineardeflection(self, value: float) -> None:
        """Override linear deflection"""
        self._lineardeflection = value

    def set_angulardeflection(self, value: float) -> None:
        """Override angular deflection"""
        self._angulardeflection = value

    @contextmanager
    def temporary(self) -> Iterator[Tolerance]:
        """Context manager that restores tolerance on exit"""

        saved = dict(self.__dict__)

        try:
            yield self
        finally:
            self.__dict__.update(saved)

    # ═══════════════════════════════════════════════════════════════════════════
    # Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    def tolerance(self, truevalue: float, rtol: float, atol: float) -> float:
        """Compute combined tolerance from relative and absolute components"""
        return rtol * abs(truevalue) + atol

    def compare(self, a: float, b: float, rtol: float, atol: float) -> bool:
        """Compare two values within tolerance"""
        return abs(a - b) <= self.tolerance(b, rtol, atol)

    def is_zero(self, a: float) -> bool:
        """Check if value is within zero tolerance"""
        return abs(a) <= self.absolute()

    def is_positive(self, a: float) -> bool:
        """Check if value is positive within tolerance"""
        return a > self.absolute()

    def is_negative(self, a: float) -> bool:
        """Check if value is negative within tolerance"""
        return a < -self.absolute()

    def is_between(self, value: float, minval: float, maxval: float) -> bool:
        """Check if value is within a range with absolute tolerance"""

        atol = self.absolute()

        return minval - atol <= value and value <= maxval + atol

    def is_close(self, a: float, b: float) -> bool:
        """Check closeness between two values using rtol/atol"""
        return self.compare(a, b, self.relative(), self.absolute())

    def is_angle_zero(self, a: float) -> bool:
        """Check if an angle is effectively zero (radians)"""
        return abs(a) <= self.angular()

    def is_angles_close(self, a: float, b: float) -> bool:
        """Check if two angles are close (radians)"""
        return abs(a - b) <= self.angular()

    def is_point_close(self, a: Point, b: Point) -> bool:
        """Check if two 3D points are equal within absolute tolerance"""

        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]

        return dx * dx + dy * dy + dz * dz <= self.absolute() * self.absolute()

    def is_vector_close(self, a: Vector, b: Vector) -> bool:
        """Check if two 3D vectors are equal within absolute tolerance"""

        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]

        return dx * dx + dy * dy + dz * dz <= self.absolute() * self.absolute()

    def is_allclose(self, a: Sequence[float], b: Sequence[float]) -> bool:
        """Check if two lists of values are element-wise close"""

        if len(a) != len(b):
            return False

        rtol = self.relative()
        atol = self.absolute()

        for i in range(len(a)):
            if not self.compare(a[i], b[i], rtol, atol):
                return False

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # Keys and formatting
    # ═══════════════════════════════════════════════════════════════════════════
    def key(self, x: float, y: float, z: float, precision: int = -999) -> str:
        """Create a geometric key string for 3D point with optional precision"""

        prec = precision if precision != -999 else self.precision()

        if prec == 0:
            raise ValueError("Precision cannot be zero.")

        if prec == -1:
            return f"{int(x)},{int(y)},{int(z)}"

        if prec < -1:
            factor = 10.0 ** (-prec - 1)

            return f"{int(Tolerance.round_to(x / factor, 0) * factor)},{int(Tolerance.round_to(y / factor, 0) * factor)},{int(Tolerance.round_to(z / factor, 0) * factor)}"

        threshold = 10.0**-prec * 0.5

        if abs(x) < threshold:
            x = 0.0

        if abs(y) < threshold:
            y = 0.0

        if abs(z) < threshold:
            z = 0.0

        return f"{x:.{prec}f},{y:.{prec}f},{z:.{prec}f}"

    def key_xy(self, x: float, y: float, precision: int = -999) -> str:
        """Create a geometric key string for 2D point with optional precision"""

        prec = precision if precision != -999 else self.precision()

        if prec == 0:
            raise ValueError("Precision cannot be zero.")

        if prec == -1:
            return f"{int(x)},{int(y)}"

        if prec < -1:
            factor = 10.0 ** (-prec - 1)

            return f"{int(Tolerance.round_to(x / factor, 0) * factor)},{int(Tolerance.round_to(y / factor, 0) * factor)}"

        threshold = 10.0**-prec * 0.5

        if abs(x) < threshold:
            x = 0.0

        if abs(y) < threshold:
            y = 0.0

        return f"{x:.{prec}f},{y:.{prec}f}"

    def format_number(self, number: float, precision: int = -999) -> str:
        """Format a number with optional precision override"""

        prec = precision if precision != -999 else self.precision()

        if prec == 0:
            raise ValueError("Precision cannot be zero.")

        if prec == -1:
            return f"{int(Tolerance.round_to(number, 0))}"

        if prec < -1:
            factor = 10.0 ** (-prec - 1)

            return f"{int(Tolerance.round_to(number / factor, 0) * factor)}"

        return f"{number:.{prec}f}"

    def precision_from_tolerance(self, tol: float = -1) -> int:
        """Determine decimal precision from a tolerance value"""

        value = tol if tol >= 0 else self.absolute()

        if value >= 1.0:
            return 0

        text = f"{value:e}"
        pos = text.find("e-")

        if pos == -1:
            return 0

        return int(text[pos + 2 :])

    # ═══════════════════════════════════════════════════════════════════════════
    # Numeric conversion
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def to_radians(degrees: float) -> float:
        """Convert degrees to radians"""
        return degrees * Tolerance.TO_RADIANS

    @staticmethod
    def to_degrees(radians: float) -> float:
        """Convert radians to degrees"""
        return radians * Tolerance.TO_DEGREES

    @staticmethod
    def round_to(value: float, ndigits: int) -> float:
        """Round a value to a given number of decimal places"""

        factor = 10.0**ndigits

        return math.copysign(math.floor(abs(value) * factor + 0.5), value) / factor

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""

        return {
            "absolute": self.absolute(),
            "angular": self.angular(),
            "angulardeflection": self.angulardeflection(),
            "approximation": self.approximation(),
            "lineardeflection": self.lineardeflection(),
            "precision": self.precision(),
            "relative": self.relative(),
            "type": "Tolerance",
            "unit": self.unit(),
        }

    @classmethod
    def __jsonload__(cls, data: dict) -> "Tolerance":
        """Deserialize from a JSON-compatible dictionary."""

        tolerance = cls(data["unit"])
        tolerance.set_absolute(data["absolute"])
        tolerance.set_angular(data["angular"])
        tolerance.set_angulardeflection(data["angulardeflection"])
        tolerance.set_approximation(data["approximation"])
        tolerance.set_lineardeflection(data["lineardeflection"])
        tolerance.set_precision(data["precision"])
        tolerance.set_relative(data["relative"])

        return tolerance

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Tolerance":
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: Union[str, Path]) -> None:
        """Write JSON to a file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filename: Union[str, Path]) -> "Tolerance":
        """Read JSON from a file."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> tolerance_pb2.Tolerance:
        """Convert to the protobuf message."""

        from .proto import tolerance_pb2

        proto = tolerance_pb2.Tolerance()
        proto.unit = self.unit()
        proto.absolute = self.absolute()
        proto.relative = self.relative()
        proto.angular = self.angular()
        proto.approximation = self.approximation()
        proto.precision = self.precision()
        proto.lineardeflection = self.lineardeflection()
        proto.angulardeflection = self.angulardeflection()

        return proto

    @classmethod
    def from_proto(cls, proto: tolerance_pb2.Tolerance) -> "Tolerance":
        """Construct from the protobuf message."""

        tolerance = cls(proto.unit)
        tolerance.set_absolute(proto.absolute)
        tolerance.set_relative(proto.relative)
        tolerance.set_angular(proto.angular)
        tolerance.set_approximation(proto.approximation)
        tolerance.set_precision(proto.precision)
        tolerance.set_lineardeflection(proto.lineardeflection)
        tolerance.set_angulardeflection(proto.angulardeflection)

        return tolerance

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Tolerance":
        """Deserialize from protobuf bytes."""

        from .proto import tolerance_pb2

        proto = tolerance_pb2.Tolerance()
        consumed = proto.ParseFromString(data)

        if consumed != len(data):
            raise ValueError("Failed to parse Tolerance protobuf data")

        return cls.from_proto(proto)

    def pb_dump(self, filename: Union[str, Path]) -> None:
        """Write protobuf bytes to a file."""

        data = self.pb_dumps()

        with open(filename, "wb") as file:
            written = file.write(data)

            if written != len(data):
                raise OSError(f"Failed to write protobuf file: {filename}")

    @classmethod
    def pb_load(cls, filename: Union[str, Path]) -> "Tolerance":
        """Read protobuf bytes from a file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "Tolerance(unit)"."""
        return f"Tolerance({self.unit()})"

    def __repr__(self) -> str:
        """Return a constructor-style representation."""
        return f"Tolerance(unit='{self.unit()}', absolute={self.absolute()}, relative={self.relative()}, angular={self.angular()}, approximation={self.approximation()}, precision={self.precision()}, lineardeflection={self.lineardeflection()}, angulardeflection={self.angulardeflection()})"


TOLERANCE = Tolerance()  # Global tolerance instance.

# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════


def is_finite(x: float) -> bool:
    """Check if a number is finite"""
    return math.isfinite(x)


def unique_from_two_int(a: int, b: int) -> int:
    """Order-independent key from two ints: larger in the high 32 bits"""

    lo = min(a, b)
    hi = max(a, b)

    return (hi << 32) | lo


def wrap_index(index: int, n: int) -> int:
    """Signed modulo into [0, n-1]; 0 when n == 0"""

    if n == 0:
        return 0

    return ((index % n) + n) % n


def triangle_edge_by_angle(edge_length: float, angle_deg: float) -> float:
    """Opposite side of a right triangle: edge_length * tan(angle_deg)"""
    return edge_length * math.tan(Tolerance.to_radians(angle_deg))


def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees"""
    return radians * Tolerance.TO_DEGREES


def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians"""
    return degrees * Tolerance.TO_RADIANS


def count_digits(n: float) -> int:
    """Number of decimal digits of the integer part of |n|; 0 when |n| < 1"""

    value = abs(n)

    if value < 1.0:
        return 0

    return int(math.log10(value)) + 1
