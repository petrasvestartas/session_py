import uuid
from .color import Color
from .xform import Xform
from .point import Point
from .vector import Vector


class Line:
    """A 3D line segment with visual properties.

    Parameters
    ----------
    x0 : float, optional
        X coordinate of start point. Defaults to 0.0.
    y0 : float, optional
        Y coordinate of start point. Defaults to 0.0.
    z0 : float, optional
        Z coordinate of start point. Defaults to 0.0.
    x1 : float, optional
        X coordinate of end point. Defaults to 0.0.
    y1 : float, optional
        Y coordinate of end point. Defaults to 0.0.
    z1 : float, optional
        Z coordinate of end point. Defaults to 1.0.

    Attributes
    ----------
    guid : str
        Unique identifier of the line.
    name : str
        Name of the line.
    linecolor : Color
        Color of the line.
    width : float
        Width of the line for display.
    """

    def __init__(self, x0=0.0, y0=0.0, z0=0.0, x1=0.0, y1=0.0, z1=1.0):
        self._guid = None
        self.name = "my_line"
        self._x0 = x0
        self._y0 = y0
        self._z0 = z0
        self._x1 = x1
        self._y1 = y1
        self._z1 = z1
        self.width = 1.0
        self._linecolor = None
        self._xform = None

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

    @property
    def linecolor(self):
        if self._linecolor is None:
            self._linecolor = Color.black()
        return self._linecolor

    @linecolor.setter
    def linecolor(self, value):
        self._linecolor = value

    @property
    def xform(self):
        if getattr(self, '_xform', None) is None:
            self._xform = Xform.identity()
        return self._xform

    @xform.setter
    def xform(self, value):
        self._xform = value

    def duplicate(self):
        """Create a deep copy of this line with a new GUID.

        Returns
        -------
        :class:`Line`
            A new Line with identical values but a different GUID.

        """
        import copy
        import uuid
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    @classmethod
    def fit_points(cls, points, length=None):
        """Fit a line to a set of points using least squares (PCA).

        Uses Principal Component Analysis to find the best-fit line
        that minimizes perpendicular distances to all points.

        Parameters
        ----------
        points : list of Point
            List of points to fit (minimum 2 points required).
        length : float, optional
            Length of the resulting line. If None, uses the extent
            of points projected onto the line direction.

        Returns
        -------
        Line
            Best-fit line through the points.

        Raises
        ------
        ValueError
            If fewer than 2 points are provided.
        """
        if len(points) < 2:
            raise ValueError("At least 2 points are required for line fitting")

        n = len(points)

        # Compute centroid
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        cz = sum(p[2] for p in points) / n

        # Compute covariance matrix elements
        cxx = cyy = czz = cxy = cxz = cyz = 0.0
        for p in points:
            dx = p[0] - cx
            dy = p[1] - cy
            dz = p[2] - cz
            cxx += dx * dx
            cyy += dy * dy
            czz += dz * dz
            cxy += dx * dy
            cxz += dx * dz
            cyz += dy * dz

        # Use numpy for eigenvalue decomposition
        import numpy as np
        cov = np.array([
            [cxx, cxy, cxz],
            [cxy, cyy, cyz],
            [cxz, cyz, czz]
        ])
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Eigenvector with largest eigenvalue is the line direction
        idx = np.argmax(eigenvalues)
        direction = eigenvectors[:, idx]

        # Determine line extent from projected points
        if length is None:
            t_min = t_max = 0.0
            for p in points:
                dx = p[0] - cx
                dy = p[1] - cy
                dz = p[2] - cz
                t = dx * direction[0] + dy * direction[1] + dz * direction[2]
                t_min = min(t_min, t)
                t_max = max(t_max, t)
            half_len = max(abs(t_min), abs(t_max))
            if half_len < 1e-10:
                half_len = 0.5  # Default if all points are coincident
        else:
            half_len = length / 2.0

        # Create line from centroid +/- direction * half_len
        x0 = cx - direction[0] * half_len
        y0 = cy - direction[1] * half_len
        z0 = cz - direction[2] * half_len
        x1 = cx + direction[0] * half_len
        y1 = cy + direction[1] * half_len
        z1 = cz + direction[2] * half_len

        return cls(x0, y0, z0, x1, y1, z1)

    @classmethod
    def from_points(cls, p1, p2):
        """Create a line from two points.

        Parameters
        ----------
        p1 : Point
            Start point.
        p2 : Point
            End point.

        Returns
        -------
        Line
            New line from p1 to p2.
        """
        return cls(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])

    @classmethod
    def from_point_and_vector(cls, point, vector):
        """Create a line from a point and a vector.

        Parameters
        ----------
        point : Point
            Start point of the line.
        vector : Vector
            Direction and length of the line.

        Returns
        -------
        Line
            New line from point to point + vector.
        """
        return cls(
            point[0], point[1], point[2],
            point[0] + vector[0], point[1] + vector[1], point[2] + vector[2]
        )

    @classmethod
    def from_point_direction_length(cls, point, direction, length):
        """Create a line from a point, direction, and length.

        Parameters
        ----------
        point : Point
            Start point of the line.
        direction : Vector
            Direction of the line (will be normalized).
        length : float
            Length of the line.

        Returns
        -------
        Line
            New line from point in direction with given length.
        """
        d = direction.normalized()
        return cls(
            point[0], point[1], point[2],
            point[0] + d[0] * length, point[1] + d[1] * length, point[2] + d[2] * length
        )

    @classmethod
    def with_name(cls, name, x0, y0, z0, x1, y1, z1):
        """Create a line with a specific name.

        Parameters
        ----------
        name : str
            Name for the line.
        x0, y0, z0 : float
            Start point coordinates.
        x1, y1, z1 : float
            End point coordinates.

        Returns
        -------
        Line
            New named line.
        """
        line = cls(x0, y0, z0, x1, y1, z1)
        line.name = name
        return line

    def length(self):
        """Calculate the length of the line.

        Returns
        -------
        float
            Length of the line.
        """
        dx = self._x1 - self._x0
        dy = self._y1 - self._y0
        dz = self._z1 - self._z0
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def squared_length(self):
        """Calculate the squared length of the line.

        Returns
        -------
        float
            Squared length of the line.
        """
        dx = self._x1 - self._x0
        dy = self._y1 - self._y0
        dz = self._z1 - self._z0
        return dx * dx + dy * dy + dz * dz

    def to_vector(self):
        """Convert line to vector from start to end.

        Returns
        -------
        Vector
            Direction vector of the line.
        """
        return Vector(self._x1 - self._x0, self._y1 - self._y0, self._z1 - self._z0)

    def to_direction(self):
        """Convert line to unit direction vector.

        Returns
        -------
        Vector
            Normalized direction vector from start to end.
        """
        return self.to_vector().normalized()

    def point_at(self, t):
        """Get point at parameter t along the line.

        Parameters
        ----------
        t : float
            Parameter value (0.0 = start, 1.0 = end).

        Returns
        -------
        Point
            Point at parameter t.
        """
        s = 1.0 - t
        return Point(
            s * self._x0 + t * self._x1,
            s * self._y0 + t * self._y1,
            s * self._z0 + t * self._z1,
        )

    def subdivide(self, n):
        """Subdivide line into n points.

        Parameters
        ----------
        n : int
            Number of points (must be >= 2).

        Returns
        -------
        list of Point
            List of n points along the line, including start and end.
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        points = []
        for i in range(n):
            t = i / (n - 1)
            points.append(self.point_at(t))
        return points

    def subdivide_by_distance(self, distance):
        """Subdivide line by approximate distance between points.

        Parameters
        ----------
        distance : float
            Target distance between consecutive points.

        Returns
        -------
        list of Point
            List of points along the line, including start and end.
        """
        if distance <= 0:
            raise ValueError("distance must be positive")
        length = self.length()
        if length < 1e-10:
            return [self.start(), self.end()]
        n = max(2, int(length / distance + 0.5) + 1)
        return self.subdivide(n)

    def start(self):
        """Get start point.

        Returns
        -------
        Point
            Start point of the line.
        """
        return Point(self._x0, self._y0, self._z0)

    def end(self):
        """Get end point.

        Returns
        -------
        Point
            End point of the line.
        """
        return Point(self._x1, self._y1, self._z1)

    def center(self):
        """Get center point (average of start and end).

        Returns
        -------
        Point
            Center point of the line.
        """
        return Point(
            (self._x0 + self._x1) * 0.5,
            (self._y0 + self._y1) * 0.5,
            (self._z0 + self._z1) * 0.5,
        )

    def closest_point(self, point, limited=True):
        """Find the closest point on the line to a given point.

        Parameters
        ----------
        point : Point
            The point to find the closest point to.
        limited : bool, optional
            If True (default), clamp to segment [0,1]. If False, treat as infinite line.

        Returns
        -------
        tuple
            (float, Point) — parameter t and closest point.
        """
        dx = self._x1 - self._x0
        dy = self._y1 - self._y0
        dz = self._z1 - self._z0
        len_sq = dx * dx + dy * dy + dz * dz
        if len_sq < 1e-20:
            return (0.0, self.start())
        t = ((point[0] - self._x0) * dx + (point[1] - self._y0) * dy + (point[2] - self._z0) * dz) / len_sq
        if limited:
            t = max(0.0, min(1.0, t))
        return (t, self.point_at(t))

    @staticmethod
    def get_middle_line(line0_start: Point, line0_end: Point, line1_start: Point, line1_end: Point):
        """Calculate middle line between two line segments.

        Returns
        -------
        tuple
            (start_point, end_point) of the middle line.
        """
        p0 = Point(
            (line0_start.x + line1_start.x) * 0.5,
            (line0_start.y + line1_start.y) * 0.5,
            (line0_start.z + line1_start.z) * 0.5,
        )
        p1 = Point(
            (line0_end.x + line1_end.x) * 0.5,
            (line0_end.y + line1_end.y) * 0.5,
            (line0_end.z + line1_end.z) * 0.5,
        )
        return p0, p1

    def __getitem__(self, index):
        """Get coordinate by index (0-5)."""
        coords = [self._x0, self._y0, self._z0, self._x1, self._y1, self._z1]
        return coords[index]

    def __setitem__(self, index, value):
        """Set coordinate by index (0-5)."""
        if index == 0:
            self._x0 = value
        elif index == 1:
            self._y0 = value
        elif index == 2:
            self._z0 = value
        elif index == 3:
            self._x1 = value
        elif index == 4:
            self._y1 = value
        elif index == 5:
            self._z1 = value
        else:
            raise IndexError("Index out of bounds")

    def __iadd__(self, other):
        """Add vector to line in place."""
        if isinstance(other, Vector):
            self._x0 += other[0]
            self._y0 += other[1]
            self._z0 += other[2]
            self._x1 += other[0]
            self._y1 += other[1]
            self._z1 += other[2]
        return self

    def __isub__(self, other):
        """Subtract vector from line in place."""
        if isinstance(other, Vector):
            self._x0 -= other[0]
            self._y0 -= other[1]
            self._z0 -= other[2]
            self._x1 -= other[0]
            self._y1 -= other[1]
            self._z1 -= other[2]
        return self

    def __imul__(self, factor):
        """Multiply line coordinates by scalar in place."""
        self._x0 *= factor
        self._y0 *= factor
        self._z0 *= factor
        self._x1 *= factor
        self._y1 *= factor
        self._z1 *= factor
        return self

    def __itruediv__(self, factor):
        """Divide line coordinates by scalar in place."""
        self._x0 /= factor
        self._y0 /= factor
        self._z0 /= factor
        self._x1 /= factor
        self._y1 /= factor
        self._z1 /= factor
        return self

    def __add__(self, other):
        """Add vector to line."""
        if isinstance(other, Vector):
            return Line(
                self._x0 + other[0],
                self._y0 + other[1],
                self._z0 + other[2],
                self._x1 + other[0],
                self._y1 + other[1],
                self._z1 + other[2],
            )
        return NotImplemented

    def __sub__(self, other):
        """Subtract vector from line."""
        if isinstance(other, Vector):
            return Line(
                self._x0 - other[0],
                self._y0 - other[1],
                self._z0 - other[2],
                self._x1 - other[0],
                self._y1 - other[1],
                self._z1 - other[2],
            )
        return NotImplemented

    def __mul__(self, factor):
        """Multiply line by scalar."""
        return Line(
            self._x0 * factor,
            self._y0 * factor,
            self._z0 * factor,
            self._x1 * factor,
            self._y1 * factor,
            self._z1 * factor,
        )

    def __truediv__(self, factor):
        """Divide line by scalar."""
        return Line(
            self._x0 / factor,
            self._y0 / factor,
            self._z0 / factor,
            self._x1 / factor,
            self._y1 / factor,
            self._z1 / factor,
        )

    def __neg__(self):
        """Negate line (flip direction)."""
        return Line(self._x1, self._y1, self._z1, self._x0, self._y0, self._z0)

    def transform(self):
        """Apply the stored xform transformation to the line coordinates.

        Transforms the line in-place and resets xform to identity.
        """
        start = Point(self._x0, self._y0, self._z0)
        end = Point(self._x1, self._y1, self._z1)

        start.xform = self.xform
        start.transform()
        end.xform = self.xform
        end.transform()

        self._x0 = start[0]
        self._y0 = start[1]
        self._z0 = start[2]
        self._x1 = end[0]
        self._y1 = end[1]
        self._z1 = end[2]
        self.xform = Xform.identity()

    def transformed(self):
        """Return a transformed copy of the line.

        Returns a new line with the transformation applied.
        The original line and its xform remain unchanged.

        Returns
        -------
        Line
            A new transformed line.
        """
        import copy

        result = copy.deepcopy(self)
        result.transform()
        return result

    def overlap(self, other):
        """Return the overlapping segment between this line and ``other``.

        Returns
        -------
        Optional[Line]
            ``None`` if the segments do not overlap (or overlap at a point).
        """
        from .polyline import Polyline
        result = Polyline.line_line_overlap(self.start(), self.end(), other.start(), other.end())
        if result is None:
            return None
        return Line.from_points(result[0], result[1])

    def overlap_average(self, other):
        """Return the average of the two reciprocal overlaps between this line and ``other``.

        Returns
        -------
        Optional[Line]
            ``None`` if the resulting overlap collapses to a point.
        """
        from .polyline import Polyline
        result = Polyline.line_line_overlap_average(self.start(), self.end(), other.start(), other.end())
        if result is None:
            return None
        out = Line.from_points(result[0], result[1])
        if out.squared_length() <= 0.0:
            return None
        return out

    def extend(self, ext_start, ext_end):
        """Extend this line in place by ``ext_start`` at the start end and ``ext_end`` at the end.

        Mutates the line in place. Mirrors C++ ``Line::extend``.
        """
        s = self.start()
        e = self.end()
        v = e - s
        mag = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5
        if mag < 1e-20:
            return
        ux = v[0] / mag
        uy = v[1] / mag
        uz = v[2] / mag
        new_s = Point(s[0] - ux * ext_start, s[1] - uy * ext_start, s[2] - uz * ext_start)
        new_e = Point(e[0] + ux * ext_end,   e[1] + uy * ext_end,   e[2] + uz * ext_end)
        self._x0 = new_s[0]; self._y0 = new_s[1]; self._z0 = new_s[2]
        self._x1 = new_e[0]; self._y1 = new_e[1]; self._z1 = new_e[2]

    ###########################################################################################
    # Polymorphic JSON Serialization
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field.

        Returns
        -------
        dict
            Dictionary with 'type', 'guid', 'name', and object fields.

        """
        # Alphabetical order to match Rust's serde_json
        return {
            "guid": self.guid,
            "linecolor": self.linecolor.__jsondump__(),
            "name": self.name,
            "type": f"{self.__class__.__name__}",
            "width": self.width,
            "x0": self._x0,
            "x1": self._x1,
            "xform": self.xform.__jsondump__(),
            "y0": self._y0,
            "y1": self._y1,
            "z0": self._z0,
            "z1": self._z1,
        }

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
        :class:`Line`
            The deserialized Line.

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

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format.

        Parameters
        ----------
        data : dict
            Dictionary containing line data.
        guid : str, optional
            GUID for the line.
        name : str, optional
            Name for the line.

        Returns
        -------
        :class:`Line`
            Reconstructed line instance.

        """
        from .file_encoders import file_decode_node

        line = cls(
            data["x0"], data["y0"], data["z0"], data["x1"], data["y1"], data["z1"]
        )
        line.guid = guid if guid is not None else data.get("guid", line.guid)
        line.name = name if name is not None else data.get("name", line.name)

        if "width" in data:
            line.width = data["width"]
        if "linecolor" in data:
            line.linecolor = file_decode_node(data["linecolor"])

        if "xform" in data:
            line.xform = file_decode_node(data["xform"])

        return line

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
        from .proto import line_pb2
        from .proto import point_pb2

        proto = line_pb2.Line()
        proto.guid = self.guid
        proto.name = self.name

        # Set start point
        proto.start.x = self._x0
        proto.start.y = self._y0
        proto.start.z = self._z0
        proto.start.guid = ""
        proto.start.name = ""
        proto.start.width = 1.0

        # Set end point
        proto.end.x = self._x1
        proto.end.y = self._y1
        proto.end.z = self._z1
        proto.end.guid = ""
        proto.end.name = ""
        proto.end.width = 1.0

        # Set xform
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        """Create Line from protobuf binary data.

        Parameters
        ----------
        data : bytes
            Protobuf-encoded line data.

        Returns
        -------
        :class:`Line`
            The deserialized Line.

        """
        from .proto import line_pb2

        proto = line_pb2.Line()
        proto.ParseFromString(data)

        line = cls(
            proto.start.x, proto.start.y, proto.start.z,
            proto.end.x, proto.end.y, proto.end.z
        )
        line.guid = proto.guid
        line.name = proto.name

        # Load xform if present
        if proto.HasField('xform'):
            line.xform = Xform()
            line.xform.name = proto.xform.name
            line.xform.m = list(proto.xform.matrix)

        return line

    def pb_dump(self, filepath):
        """Write protobuf to file.

        Parameters
        ----------
        filepath : str or Path
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
        filepath : str or Path
            Path to the protobuf file.

        Returns
        -------
        :class:`Line`
            The deserialized Line.

        """
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)

    def __str__(self):
        """String representation."""
        return f"Line({self._x0}, {self._y0}, {self._z0}, {self._x1}, {self._y1}, {self._z1})"

    def __repr__(self):
        """Detailed representation."""
        return f"Line({self.name}, {self._x0}, {self._y0}, {self._z0}, {self._x1}, {self._y1}, {self._z1}, {repr(self.linecolor)}, {self.width})"
