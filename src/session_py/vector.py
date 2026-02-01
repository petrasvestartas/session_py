import uuid
import math
from .tolerance import Tolerance, TO_DEGREES, TO_RADIANS


class Vector:
    """A 3D vector with visual properties.

    Parameters
    ----------
    x : float, optional
        X coordinate. Defaults to 0.0.
    y : float, optional
        Y coordinate. Defaults to 0.0.
    z : float, optional
        Z coordinate. Defaults to 0.0.

    Attributes
    ----------
    guid : str
        The unique identifier of the vector.
    name : str
        The name of the vector.
    _x : float
        The X coordinate (access via [0]).
    _y : float
        The Y coordinate (access via [1]).
    _z : float
        The Z coordinate (access via [2]).

    """

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.guid = str(uuid.uuid4())
        self.name = "my_vector"
        self._x = x
        self._y = y
        self._z = z
        self._magnitude = 0.0
        self._has_magnitude = False

    @property
    def x(self):
        """Get X coordinate."""
        return self._x

    @x.setter
    def x(self, value):
        """Set X coordinate."""
        self._x = value
        self._has_magnitude = False

    @property
    def y(self):
        """Get Y coordinate."""
        return self._y

    @y.setter
    def y(self, value):
        """Set Y coordinate."""
        self._y = value
        self._has_magnitude = False

    @property
    def z(self):
        """Get Z coordinate."""
        return self._z

    @z.setter
    def z(self, value):
        """Set Z coordinate."""
        self._z = value
        self._has_magnitude = False

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # New guid
        result.guid = str(uuid.uuid4())

        # Copy remaining fields
        result.name = self.name
        result._x = self._x
        result._y = self._y
        result._z = self._z
        result._magnitude = self._magnitude
        result._has_magnitude = self._has_magnitude
        return result

    def duplicate(self):
        """Create a deep copy of this vector with a new GUID.

        Returns
        -------
        :class:`Vector`
            A new Vector with identical values but a different GUID.

        """
        import copy
        import uuid
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def __str__(self):
        return f"Vector({self[0]}, {self[1]}, {self[2]})"

    def __repr__(self):
        return f"Vector({self.guid}, {self.name}, {self[0]}, {self[1]}, {self[2]})"

    def str(self):
        """Simple string form: just coordinates formatted to 6 decimals."""
        from .tolerance import Tolerance
        return f"{round(self[0], Tolerance.ROUNDING):.6f}, {round(self[1], Tolerance.ROUNDING):.6f}, {round(self[2], Tolerance.ROUNDING):.6f}"

    def repr(self):
        """Detailed representation with name, coordinates, and magnitude."""
        from .tolerance import Tolerance
        mag = self.magnitude()
        return f"Vector({self.name}, {round(self[0], Tolerance.ROUNDING):.6f}, {round(self[1], Tolerance.ROUNDING):.6f}, {round(self[2], Tolerance.ROUNDING):.6f}, {round(mag, Tolerance.ROUNDING):.6f})"

    def __eq__(self, other):
        return (
            self.name == other.name
            and round(self[0], 6) == round(other[0], 6)
            and round(self[1], 6) == round(other[1], 6)
            and round(self[2], 6) == round(other[2], 6)
        )

    def __ne__(self, other):
        return not self == other

    ###########################################################################################
    # No-copy Operators
    ###########################################################################################

    def __getitem__(self, index):
        """Access coordinate by index (0=x, 1=y, 2=z)."""
        if index == 0:
            return self._x
        elif index == 1:
            return self._y
        elif index == 2:
            return self._z
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        """Set coordinate by index (0=x, 1=y, 2=z). Invalidates length cache."""
        if index == 0:
            self._x = value
        elif index == 1:
            self._y = value
        elif index == 2:
            self._z = value
        else:
            raise IndexError("Index out of range")
        self._has_magnitude = False

    def __imul__(self, other):
        self._x *= other
        self._y *= other
        self._z *= other
        self._has_magnitude = False
        return self

    def __itruediv__(self, other):
        self._x /= other
        self._y /= other
        self._z /= other
        self._has_magnitude = False
        return self

    def __iadd__(self, other):
        self._x += other._x
        self._y += other._y
        self._z += other._z
        self._has_magnitude = False
        return self

    def __isub__(self, other):
        self._x -= other._x
        self._y -= other._y
        self._z -= other._z
        self._has_magnitude = False
        return self

    ###########################################################################################
    # Copy Operators
    ###########################################################################################

    def __mul__(self, other):
        return Vector(self._x * other, self._y * other, self._z * other)

    def __truediv__(self, other):
        return Vector(self._x / other, self._y / other, self._z / other)

    def __add__(self, other):
        return Vector(self._x + other._x, self._y + other._y, self._z + other._z)

    def __sub__(self, other):
        return Vector(self._x - other._x, self._y - other._y, self._z - other._z)

    ###########################################################################################
    # Static Methods
    ###########################################################################################

    @staticmethod
    def zero():
        """Get a zero vector (0, 0, 0).

        Returns
        -------
        :class:`Vector`
            Zero vector (0, 0, 0).

        """
        return Vector(0.0, 0.0, 0.0)

    @staticmethod
    def x_axis():
        """Get unit vector along the x-axis.

        Returns
        -------
        :class:`Vector`
            Unit vector (1, 0, 0).

        """
        return Vector(1.0, 0.0, 0.0)

    @staticmethod
    def y_axis():
        """Get unit vector along the y-axis.

        Returns
        -------
        :class:`Vector`
            Unit vector (0, 1, 0).

        """
        return Vector(0.0, 1.0, 0.0)

    @staticmethod
    def z_axis():
        """Get unit vector along the z-axis.

        Returns
        -------
        :class:`Vector`
            Unit vector (0, 0, 1).

        """
        return Vector(0.0, 0.0, 1.0)

    @staticmethod
    def from_points(p0, p1):
        """Vector from p0 to p1 (p1 - p0).

        Parameters
        ----------
        p0 : :class:`Point`
            Start point.
        p1 : :class:`Point`
            End point.

        Returns
        -------
        :class:`Vector`
            The vector from p0 to p1.

        """
        return Vector(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])

    ###########################################################################################
    # Details
    ###########################################################################################

    def scale(self, factor):
        """Scale the vector by a factor.

        Parameters
        ----------
        factor : float
            The scaling factor to apply to all components.

        """
        self._x *= factor
        self._y *= factor
        self._z *= factor
        self._has_magnitude = False

    def scale_up(self):
        """Scale the vector up by the global scale factor (SCALE)."""
        from .tolerance import SCALE
        self.scale(SCALE)

    def scale_down(self):
        """Scale the vector down by the global scale factor (1/SCALE)."""
        from .tolerance import SCALE
        self.scale(1.0 / SCALE)

    def reverse(self):
        """Reverse the vector (negate all components).

        Returns
        -------
        :class:`Vector`
            Self.

        """
        self._x = -self._x
        self._y = -self._y
        self._z = -self._z
        self._has_magnitude = False
        return self

    def _compute_magnitude(self):
        """Compute the magnitude of the vector without caching.

        Use magnitude() for cached version.
        """
        mag = 0.0

        x = abs(self._x)
        y = abs(self._y)
        z = abs(self._z)

        # Handle two zero case:
        x_zero = x < Tolerance.ZERO_TOLERANCE
        y_zero = y < Tolerance.ZERO_TOLERANCE
        z_zero = z < Tolerance.ZERO_TOLERANCE

        if x_zero and y_zero and z_zero:
            mag = 0.0
            return mag
        elif x_zero and y_zero:
            mag = z
            return mag
        elif x_zero and z_zero:
            mag = y
            return mag
        elif y_zero and z_zero:
            mag = x
            return mag

        # Handle one or none zero case:
        # Sort so that x is the largest component
        if y >= x and y >= z:
            mag = x
            x = y
            y = mag
        elif z >= x and z >= y:
            mag = x
            x = z
            z = mag

        # For small denormalized doubles (positive but smaller
        # than DOUBLE_MIN), some compilers/FPUs set 1.0/x to +INF.
        # Without the DOUBLE_MIN test we end up with
        # microscopic vectors that have infinite magnitude!
        if x > 2.22507385850720200e-308:
            y /= x
            z /= x
            mag = x * math.sqrt(1.0 + y * y + z * z)
        elif x > 0.0 and math.isfinite(x):
            mag = x
        else:
            mag = 0.0

        return mag

    def magnitude(self):
        """Get the cached magnitude of the vector, computing it if necessary.

        Returns
        -------
        float
            The magnitude of the vector.
        """
        if not self._has_magnitude:
            self._magnitude = self._compute_magnitude()
            self._has_magnitude = True

        return self._magnitude

    def magnitude_squared(self):
        """Get the squared magnitude of the vector (avoids sqrt for performance).

        Returns
        -------
        float
            The squared magnitude of the vector.
        """
        return self._x * self._x + self._y * self._y + self._z * self._z

    def normalize_self(self):
        """Normalize the vector in place (make it unit magnitude).

        Returns
        -------
        bool
            True if successful, False if vector has zero magnitude.
        """
        d = self.magnitude()
        if d > 0.0:
            self._x /= d
            self._y /= d
            self._z /= d
            self._magnitude = 1.0
            self._has_magnitude = True
            return True
        return False

    def normalize(self):
        """Return a normalized copy of the vector.

        Returns
        -------
        Vector
            A new vector that is the unit vector of this vector.
        """
        normalized_vector = Vector(self._x, self._y, self._z)
        normalized_vector.normalize_self()
        return normalized_vector

    def normalized(self):
        """Alias for normalize() - Return a normalized copy of the vector.

        Returns
        -------
        Vector
            A new vector that is the unit vector of this vector.
        """
        return self.normalize()

    def dot(self, other):
        """Calculate dot product with another vector.

        Parameters
        ----------
        other : :class:`Vector`
            Other vector.

        Returns
        -------
        float
            Dot product value.

        """
        return self._x * other._x + self._y * other._y + self._z * other._z

    def cross(self, other):
        """Calculate cross product with another vector.

        Parameters
        ----------
        other : :class:`Vector`
            Other vector.

        Returns
        -------
        :class:`Vector`
            Cross product vector (orthogonal to inputs).

        """
        x = self._y * other._z - self._z * other._y
        y = self._z * other._x - self._x * other._z
        z = self._x * other._y - self._y * other._x
        return Vector(x, y, z)

    def is_parallel_to(self, v):
        """Check if this vector is parallel/antiparallel to another.

        Parameters
        ----------
        v : :class:`Vector`
            Other vector.

        Returns
        -------
        int
            1 if parallel, -1 if antiparallel, 0 otherwise.

        """
        ll = self.magnitude() * v.magnitude()

        if ll > 0.0:
            cos_angle = self.dot(v) / ll
            angle_in_radians = Tolerance.ANGLE_TOLERANCE_DEGREES * TO_RADIANS
            cos_tol = math.cos(angle_in_radians)

            if cos_angle >= cos_tol:
                return 1  # Parallel
            elif cos_angle <= -cos_tol:
                return -1  # Antiparallel
            else:
                return 0  # Not parallel
        else:
            return 0  # Not parallel

    def angle(self, other, sign_by_cross_product=False, degrees=True, tolerance=1e-12):
        """Angle between this vector and another.

        Parameters
        ----------
        other : :class:`Vector`
            The other vector.
        sign_by_cross_product : bool, optional
            If True, sign the angle using the z-component of the cross product.
        degrees : bool, optional
            If True (default), return angle in degrees; otherwise radians.
        tolerance : float, optional
            Denominator tolerance to treat near-zero lengths as zero.

        Returns
        -------
        float
            The angle value (degrees if `degrees` else radians).

        """
        dot_product = self.dot(other)
        len0 = self.magnitude()
        len1 = other.magnitude()

        denominator = len0 * len1
        if denominator < tolerance:
            return 0.0

        cos_angle = dot_product / denominator
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]

        angle = math.acos(cos_angle)

        if sign_by_cross_product:
            # Raw cross product z-component for sign check
            cross_z = self._x * other._y - self._y * other._x
            if cross_z < 0:
                angle = -angle

        if degrees:
            return math.degrees(angle)
        return angle

    def projection(self, projection_vector, tolerance=1e-12):
        """Project this vector onto another vector.

        Parameters
        ----------
        projection_vector : :class:`Vector`
            Vector to project onto.
        tolerance : float, optional
            Treat `projection_vector` length below this as zero.

        Returns
        -------
        tuple
            (projection_vector, projected_length, perpendicular_vector, perpendicular_length),
            where projection_vector is :class:`Vector`, projected_length is float,
            perpendicular_vector is :class:`Vector`, and perpendicular_length is float.

        """
        projection_vector_length = projection_vector.magnitude()

        if projection_vector_length < tolerance:
            return Vector(0, 0, 0), 0.0, Vector(0, 0, 0), 0.0

        projection_vector_unit = Vector(
            projection_vector[0] / projection_vector_length,
            projection_vector[1] / projection_vector_length,
            projection_vector[2] / projection_vector_length,
        )

        projected_vector_length = self.dot(projection_vector_unit)
        out_projection_vector = projection_vector_unit * projected_vector_length

        out_perpendicular_vector = self - out_projection_vector
        out_perpendicular_length = out_perpendicular_vector.magnitude()

        return (
            out_projection_vector,
            projected_vector_length,
            out_perpendicular_vector,
            out_perpendicular_length,
        )

    def get_leveled_vector(self, vertical_height):
        """Get a copy scaled by a vertical height along the Z-axis.

        Parameters
        ----------
        vertical_height : float
            Target vertical height.

        Returns
        -------
        :class:`Vector`
            Scaled copy matching the C++ implementation.

        """
        copy = Vector(self._x, self._y, self._z)

        if copy.normalize_self():
            reference_vector = Vector(0, 0, 1)
            angle_deg = copy.angle(
                reference_vector, sign_by_cross_product=False, degrees=True
            )
            angle_rad = math.radians(angle_deg)
            inclined_offset = vertical_height / math.cos(angle_rad)
            copy *= inclined_offset

        return copy

    @staticmethod
    def cosine_law(
        triangle_edge_length_a,
        triangle_edge_length_b,
        angle_in_between_edges,
        degrees=True,
    ):
        """Calculate third side of triangle using the cosine law.

        Parameters
        ----------
        triangle_edge_length_a : float
            Length of side a.
        triangle_edge_length_b : float
            Length of side b.
        angle_in_between_edges : float
            Angle between a and b.
        degrees : bool, optional
            If True, the angle is provided in degrees.

        Returns
        -------
        float
            Length of the third side.

        """
        to_radians = TO_RADIANS if degrees else 1.0
        return math.sqrt(
            triangle_edge_length_a**2
            + triangle_edge_length_b**2
            - 2
            * triangle_edge_length_a
            * triangle_edge_length_b
            * math.cos(angle_in_between_edges * to_radians)
        )

    @staticmethod
    def sine_law_angle(
        triangle_edge_length_a,
        angle_in_front_of_a,
        triangle_edge_length_b,
        degrees=True,
    ):
        """Calculate angle using the sine law.

        Parameters
        ----------
        triangle_edge_length_a : float
            Length of side a.
        angle_in_front_of_a : float
            Angle opposite to side a.
        triangle_edge_length_b : float
            Length of side b.
        degrees : bool, optional
            If True, return angle in degrees.

        Returns
        -------
        float
            Angle opposite to side b (degrees if `degrees`).

        """
        to_radians = TO_RADIANS if degrees else 1.0
        to_degrees = TO_DEGREES if degrees else 1.0
        return (
            math.asin(
                (triangle_edge_length_b * math.sin(angle_in_front_of_a * to_radians))
                / triangle_edge_length_a
            )
            * to_degrees
        )

    @staticmethod
    def sine_law_length(
        triangle_edge_length_a, angle_in_front_of_a, angle_in_front_of_b, degrees=True
    ):
        """Calculate side length using the sine law.

        Parameters
        ----------
        triangle_edge_length_a : float
            Length of side a.
        angle_in_front_of_a : float
            Angle opposite to side a.
        angle_in_front_of_b : float
            Angle opposite to side b.
        degrees : bool, optional
            If True, angles are provided in degrees.

        Returns
        -------
        float
            Length of side b.

        """
        to_radians = TO_RADIANS if degrees else 1.0
        return (
            triangle_edge_length_a * math.sin(angle_in_front_of_b * to_radians)
        ) / math.sin(angle_in_front_of_a * to_radians)

    @staticmethod
    def angle_from_cosine_law(
        triangle_edge_length_a,
        triangle_edge_length_b,
        triangle_edge_length_c,
        degrees=True,
    ):
        """Calculate angle opposite to side c using the cosine law.

        Parameters
        ----------
        triangle_edge_length_a : float
            Length of side a (adjacent to angle C).
        triangle_edge_length_b : float
            Length of side b (adjacent to angle C).
        triangle_edge_length_c : float
            Length of side c (opposite to angle C).
        degrees : bool, optional
            If True, return degrees; otherwise radians.

        Returns
        -------
        float
            Angle opposite to side c.

        """
        a = triangle_edge_length_a
        b = triangle_edge_length_b
        c = triangle_edge_length_c

        # cos(C) = (a² + b² - c²) / (2ab)
        cos_c = (a**2 + b**2 - c**2) / (2.0 * a * b)
        angle_rad = math.acos(cos_c)

        if degrees:
            return angle_rad * TO_DEGREES
        else:
            return angle_rad

    @staticmethod
    def side_from_sine_law(
        angle_in_front_of_result_side,
        angle_in_front_of_known_side,
        known_side_length,
        degrees=True,
    ):
        """Calculate side length using the sine law.

        Given two angles and the side opposite to one of them, calculates
        the side opposite to the other angle: a/sin(A) = b/sin(B)

        Parameters
        ----------
        angle_in_front_of_result_side : float
            Angle opposite to the side we want to find.
        angle_in_front_of_known_side : float
            Angle opposite to the known side.
        known_side_length : float
            Length of the known side.
        degrees : bool, optional
            If True, angles are in degrees; otherwise radians.

        Returns
        -------
        float
            Length of the side opposite to the first angle.

        """
        to_radians = TO_RADIANS if degrees else 1.0
        angle_a = angle_in_front_of_result_side * to_radians
        angle_b = angle_in_front_of_known_side * to_radians

        # a = b·sin(A)/sin(B)
        return (known_side_length * math.sin(angle_a)) / math.sin(angle_b)

    @staticmethod
    def angle_between_vector_xy_components(vector, degrees=True):
        """Angle of vector's XY projection from +X axis (atan2).

        Parameters
        ----------
        vector : :class:`Vector`
            Input vector.
        degrees : bool, optional
            If True, return degrees; otherwise radians.

        Returns
        -------
        float
            Angle in the XY plane.

        """
        to_degrees = TO_DEGREES if degrees else 1.0
        return math.atan2(vector[1], vector[0]) * to_degrees

    @staticmethod
    def sum_of_vectors(vectors):
        """Sum a list of vectors (component-wise).

        Parameters
        ----------
        vectors : list[:class:`Vector`]
            Vectors to sum.

        Returns
        -------
        :class:`Vector`
            The component-wise sum.

        """
        x = y = z = 0.0
        for vector in vectors:
            x += vector._x
            y += vector._y
            z += vector._z
        return Vector(x, y, z)

    @staticmethod
    def average(vectors):
        """Compute the average of a list of vectors.

        Parameters
        ----------
        vectors : list[:class:`Vector`]
            Vectors to average.

        Returns
        -------
        :class:`Vector`
            The component-wise average, or zero vector if empty.

        """
        if not vectors:
            return Vector.zero()
        s = Vector.sum_of_vectors(vectors)
        count = len(vectors)
        return Vector(s._x / count, s._y / count, s._z / count)

    def is_perpendicular_to(self, other):
        """Check if this vector is perpendicular to another.

        Parameters
        ----------
        other : :class:`Vector`
            The other vector.

        Returns
        -------
        bool
            True if perpendicular (dot product within tolerance of zero).

        """
        return abs(self.dot(other)) < Tolerance.ZERO_TOLERANCE

    def is_zero(self):
        """Check if this vector is a zero vector.

        Returns
        -------
        bool
            True if length is within tolerance of zero.

        """
        return self._compute_magnitude() < Tolerance.ZERO_TOLERANCE

    def coordinate_direction_3angles(self, degrees=True):
        """Compute coordinate direction angles (alpha, beta, gamma).

        Parameters
        ----------
        degrees : bool, optional
            Return angles in degrees if True, radians if False.

        Returns
        -------
        tuple
            (alpha, beta, gamma)

        """
        r = math.sqrt(self._x**2 + self._y**2 + self._z**2)

        if r == 0:
            return (0, 0, 0)

        x_proportion = self._x / r
        y_proportion = self._y / r
        z_proportion = self._z / r

        alpha = math.acos(x_proportion)
        beta = math.acos(y_proportion)
        gamma = math.acos(z_proportion)

        if degrees:
            alpha *= TO_DEGREES
            beta *= TO_DEGREES
            gamma *= TO_DEGREES

        return (alpha, beta, gamma)

    def coordinate_direction_2angles(self, degrees=True):
        """Compute coordinate direction angles (phi, theta).

        Parameters
        ----------
        degrees : bool, optional
            Return angles in degrees if True, radians if False.

        Returns
        -------
        tuple
            (phi, theta)

        """
        r = math.sqrt(self._x**2 + self._y**2 + self._z**2)

        if r == 0:
            return (0, 0)

        phi = math.acos(self._z / r)
        theta = math.atan2(self._y, self._x)

        if degrees:
            phi *= TO_DEGREES
            theta *= TO_DEGREES

        return (phi, theta)

    def perpendicular_to(self, v):
        """Set this vector to be perpendicular to `v`.

        Parameters
        ----------
        v : :class:`Vector`
            Reference vector.

        Returns
        -------
        bool
            True on success, False otherwise.

        """
        k = 2

        if abs(v[1]) > abs(v[0]):
            if abs(v[2]) > abs(v[1]):
                # |v.z| > |v.y| > |v.x|
                i, j, k = 2, 1, 0
                a, b = v[2], -v[1]
            elif abs(v[2]) >= abs(v[0]):
                # |v.y| >= |v.z| >= |v.x|
                i, j, k = 1, 2, 0
                a, b = v[1], -v[2]
            else:
                # |v.y| > |v.x| > |v.z|
                i, j, k = 1, 0, 2
                a, b = v[1], -v[0]
        elif abs(v[2]) > abs(v[0]):
            # |v.z| > |v.x| >= |v.y|
            i, j, k = 2, 0, 1
            a, b = v[2], -v[0]
        elif abs(v[2]) > abs(v[1]):
            # |v.x| >= |v.z| > |v.y|
            i, j, k = 0, 2, 1
            a, b = v[0], -v[2]
        else:
            # |v.x| >= |v.y| >= |v.z|
            i, j, k = 0, 1, 2
            a, b = v[0], -v[1]

        coords = [0, 0, 0]
        coords[i] = b
        coords[j] = a
        coords[k] = 0.0

        self._x, self._y, self._z = coords
        self._has_magnitude = False

    ###########################################################################################
    # Polymorphic JSON Serialization (COMPAS-style)
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field."""
        # Alphabetical order to match Rust's serde_json
        return {
            "guid": self.guid,
            "name": self.name,
            "type": f"{self.__class__.__name__}",
            "x": self[0],
            "y": self[1],
            "z": self[2],
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format."""
        vec = cls(data["x"], data["y"], data["z"])
        vec.guid = guid if guid is not None else data.get("guid", vec.guid)
        vec.name = name if name is not None else data.get("name", vec.name)
        return vec

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
        :class:`Vector`
            The deserialized Vector.

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

        """
        from .proto import vector_pb2
        
        proto = vector_pb2.Vector()
        proto.x = self._x
        proto.y = self._y
        proto.z = self._z
        proto.name = self.name
        
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        """Create Vector from protobuf binary data.

        Parameters
        ----------
        data : bytes
            Protobuf-encoded vector data.

        Returns
        -------
        :class:`Vector`
            The deserialized Vector.

        """
        from .proto import vector_pb2
        
        proto = vector_pb2.Vector()
        proto.ParseFromString(data)
        
        v = cls(proto.x, proto.y, proto.z)
        v.name = proto.name
        
        return v

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
        :class:`Vector`
            The deserialized Vector.

        """
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)
