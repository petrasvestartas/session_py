import uuid
import math
from .color import Color
from .point import Point
from .vector import Vector
from .tolerance import Tolerance
from .xform import Xform


class Plane:
    """A 3D plane defined by origin and coordinate axes.

    Parameters
    ----------
    origin : Point, optional
        Origin point of the plane. Defaults to Point(0, 0, 0).
    x_axis : Vector, optional
        X-axis direction. Defaults to Vector(1, 0, 0).
    y_axis : Vector, optional
        Y-axis direction. Defaults to Vector(0, 1, 0).
    name : str, optional
        Name of the plane. Defaults to "my_plane".

    Attributes
    ----------
    guid : str
        The unique identifier of the plane.
    name : str
        The name of the plane.
    origin : Point
        The origin point of the plane.
    x_axis : Vector
        The X-axis direction vector.
    y_axis : Vector
        The Y-axis direction vector.
    z_axis : Vector
        The Z-axis direction vector (normal).
    a : float
        Plane equation coefficient (normal x-component).
    b : float
        Plane equation coefficient (normal y-component).
    c : float
        Plane equation coefficient (normal z-component).
    d : float
        Plane equation coefficient (distance from origin).
    """

    def __init__(self, origin=None, x_axis=None, y_axis=None, name="my_plane", width=1.0):
        self._guid = None
        self.name = name
        self.width = width
        self._linecolor = None
        self._xform = None

        if origin is None:
            self._origin = Point(0.0, 0.0, 0.0)
        else:
            self._origin = origin

        if x_axis is None:
            self._x_axis = Vector.x_axis()
        else:
            self._x_axis = x_axis
            self._x_axis.normalize_self()

        if y_axis is None:
            self._y_axis = Vector.y_axis()
        else:
            self._y_axis = y_axis - x_axis * (y_axis.dot(self._x_axis))
            self._y_axis.normalize_self()

        self._z_axis = self._x_axis.cross(self._y_axis)
        self._z_axis.normalize_self()

        self._update_equation()

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

    @property
    def xform(self):
        if getattr(self, '_xform', None) is None:
            self._xform = Xform.identity()
        return self._xform

    @xform.setter
    def xform(self, value):
        self._xform = value

    @property
    def linecolor(self):
        if self._linecolor is None:
            self._linecolor = Color.blue()
        return self._linecolor

    @linecolor.setter
    def linecolor(self, value):
        self._linecolor = value

    def _update_equation(self):
        """Update plane equation coefficients from z_axis and origin."""
        self._a = self._z_axis[0]
        self._b = self._z_axis[1]
        self._c = self._z_axis[2]
        self._d = -(
            self._a * self._origin[0]
            + self._b * self._origin[1]
            + self._c * self._origin[2]
        )

    @property
    def origin(self):
        """Get the origin point."""
        return self._origin

    @property
    def x_axis(self):
        """Get the X-axis vector."""
        return self._x_axis

    @property
    def y_axis(self):
        """Get the Y-axis vector."""
        return self._y_axis

    @property
    def z_axis(self):
        """Get the Z-axis vector (normal)."""
        return self._z_axis

    @property
    def a(self):
        """Get plane equation coefficient a."""
        return self._a

    @property
    def b(self):
        """Get plane equation coefficient b."""
        return self._b

    @property
    def c(self):
        """Get plane equation coefficient c."""
        return self._c

    @property
    def d(self):
        """Get plane equation coefficient d."""
        return self._d

    @staticmethod
    def from_point_normal(point, normal, normalize=True):
        """Create a plane from a point and normal vector.

        Parameters
        ----------
        point : Point
            Point on the plane.
        normal : Vector
            Normal vector of the plane.
        normalize : bool, optional
            Normalize axes. Set False when normal is already unit-length. Defaults to True.

        Returns
        -------
        Plane
            The constructed plane.
        """
        plane = Plane.__new__(Plane)
        plane.guid = str(uuid.uuid4())
        plane.name = "my_plane"
        plane.width = 1.0
        plane._linecolor = None
        plane.xform = Xform.identity()
        plane._origin = point
        plane._z_axis = Vector(normal[0], normal[1], normal[2])
        if normalize:
            plane._z_axis.normalize_self()
        plane._x_axis = Vector()
        plane._x_axis.perpendicular_to(plane._z_axis)
        if normalize:
            plane._x_axis.normalize_self()
        plane._y_axis = plane._z_axis.cross(plane._x_axis)
        if normalize:
            plane._y_axis.normalize_self()
        plane._update_equation()
        return plane

    @staticmethod
    def from_points(points):
        """Create a plane from three or more points.

        Parameters
        ----------
        points : list of Point
            List of at least 3 points.

        Returns
        -------
        Plane
            The constructed plane.
        """
        if len(points) < 3:
            return Plane()

        plane = Plane.__new__(Plane)
        plane.guid = str(uuid.uuid4())
        plane.name = "my_plane"
        plane.width = 1.0
        plane._linecolor = None
        plane.xform = Xform.identity()
        plane._origin = points[0]

        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        plane._z_axis = v1.cross(v2)
        plane._z_axis.normalize_self()

        plane._x_axis = Vector(v1[0], v1[1], v1[2])
        plane._x_axis.normalize_self()
        plane._y_axis = plane._z_axis.cross(plane._x_axis)
        plane._y_axis.normalize_self()

        plane._update_equation()
        return plane

    @staticmethod
    def from_points_pca(points):
        if len(points) < 3:
            return Plane()

        n = len(points)
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        cz = sum(p[2] for p in points) / n

        cxx = cyy = czz = cxy = cxz = cyz = 0.0
        for p in points:
            dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
            cxx += dx * dx; cyy += dy * dy; czz += dz * dz
            cxy += dx * dy; cxz += dx * dz; cyz += dy * dz

        eigvec = [[0.0]*3 for _ in range(3)]
        eigval = [0.0]*3
        cov = [[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]]

        for e in range(3):
            vx, vy, vz = (1.0, 0.0, 0.0) if e == 0 else ((0.0, 1.0, 0.0) if e == 1 else (0.0, 0.0, 1.0))
            for _ in range(100):
                nx = cov[0][0] * vx + cov[0][1] * vy + cov[0][2] * vz
                ny = cov[1][0] * vx + cov[1][1] * vy + cov[1][2] * vz
                nz = cov[2][0] * vx + cov[2][1] * vy + cov[2][2] * vz
                mag = math.sqrt(nx*nx + ny*ny + nz*nz)
                if mag < 1e-15:
                    break
                vx, vy, vz = nx/mag, ny/mag, nz/mag
            eigvec[e] = [vx, vy, vz]
            eigval[e] = (cov[0][0]*vx*vx + cov[1][1]*vy*vy + cov[2][2]*vz*vz
                        + 2*cov[0][1]*vx*vy + 2*cov[0][2]*vx*vz + 2*cov[1][2]*vy*vz)
            for i in range(3):
                for j in range(3):
                    cov[i][j] -= eigval[e] * eigvec[e][i] * eigvec[e][j]

        x_axis = Vector(eigvec[0][0], eigvec[0][1], eigvec[0][2])
        y_axis = Vector(eigvec[1][0], eigvec[1][1], eigvec[1][2])
        z_axis = x_axis.cross(y_axis)
        z_axis.normalize_self()
        y_axis = z_axis.cross(x_axis)
        y_axis.normalize_self()
        x_axis.normalize_self()

        plane = Plane.__new__(Plane)
        plane.guid = str(uuid.uuid4())
        plane.name = "my_plane"
        plane.width = 1.0
        plane._linecolor = None
        plane.xform = Xform.identity()
        plane._origin = Point(cx, cy, cz)
        plane._x_axis = x_axis
        plane._y_axis = y_axis
        plane._z_axis = z_axis
        plane._update_equation()
        return plane

    @staticmethod
    def from_two_points(point1, point2):
        """Create a plane from two points.

        Parameters
        ----------
        point1 : Point
            First point.
        point2 : Point
            Second point.

        Returns
        -------
        Plane
            The constructed plane.
        """
        plane = Plane.__new__(Plane)
        plane.guid = str(uuid.uuid4())
        plane.name = "my_plane"
        plane.width = 1.0
        plane._linecolor = None
        plane.xform = Xform.identity()
        plane._origin = point1

        direction = point2 - point1
        direction.normalize_self()
        plane._z_axis = Vector()
        plane._z_axis.perpendicular_to(direction)
        plane._z_axis.normalize_self()

        plane._x_axis = direction
        plane._y_axis = plane._z_axis.cross(plane._x_axis)
        plane._y_axis.normalize_self()

        plane._update_equation()
        return plane

    @staticmethod
    def xy_plane():
        """Create the XY plane.

        Returns
        -------
        Plane
            XY plane at origin.
        """
        plane = Plane.__new__(Plane)
        plane.guid = str(uuid.uuid4())
        plane.name = "xy_plane"
        plane.width = 1.0
        plane._linecolor = None
        plane.xform = Xform.identity()
        plane._origin = Point(0.0, 0.0, 0.0)
        plane._x_axis = Vector.x_axis()
        plane._y_axis = Vector.y_axis()
        plane._z_axis = Vector.z_axis()
        plane._a = 0.0
        plane._b = 0.0
        plane._c = 1.0
        plane._d = 0.0
        return plane

    @staticmethod
    def yz_plane():
        """Create the YZ plane.

        Returns
        -------
        Plane
            YZ plane at origin.
        """
        plane = Plane.__new__(Plane)
        plane.guid = str(uuid.uuid4())
        plane.name = "yz_plane"
        plane.width = 1.0
        plane._linecolor = None
        plane.xform = Xform.identity()
        plane._origin = Point(0.0, 0.0, 0.0)
        plane._x_axis = Vector.y_axis()
        plane._y_axis = Vector.z_axis()
        plane._z_axis = Vector.x_axis()
        plane._a = 1.0
        plane._b = 0.0
        plane._c = 0.0
        plane._d = 0.0
        return plane

    @staticmethod
    def xz_plane():
        """Create the XZ plane.

        Returns
        -------
        Plane
            XZ plane at origin.
        """
        plane = Plane.__new__(Plane)
        plane.guid = str(uuid.uuid4())
        plane.name = "xz_plane"
        plane.width = 1.0
        plane._linecolor = None
        plane.xform = Xform.identity()
        plane._origin = Point(0.0, 0.0, 0.0)
        plane._x_axis = Vector.x_axis()
        plane._y_axis = Vector(0.0, 0.0, -1.0)
        plane._z_axis = Vector(0.0, 1.0, 0.0)
        plane._a = 0.0
        plane._b = 1.0
        plane._c = 0.0
        plane._d = 0.0
        return plane

    @staticmethod
    def invalid():
        p = object.__new__(Plane)
        p.guid = str(uuid.uuid4())
        p.name = "my_plane"
        p.width = 1.0
        p._linecolor = None
        p.xform = Xform.identity()
        p._origin = Point(0, 0, 0)
        p._x_axis = Vector(0, 0, 0)
        p._y_axis = Vector(0, 0, 0)
        p._z_axis = Vector(0, 0, 0)
        p._a = 0.0
        p._b = 0.0
        p._c = 0.0
        p._d = 0.0
        return p

    def is_valid(self) -> bool:
        return self._x_axis.magnitude() > 1e-14 and self._y_axis.magnitude() > 1e-14 and self._z_axis.magnitude() > 1e-14

    @staticmethod
    def from_frame(origin, x_axis, y_axis, z_axis):
        p = object.__new__(Plane)
        p.guid = str(uuid.uuid4())
        p.name = "my_plane"
        p.width = 1.0
        p._linecolor = None
        p.xform = Xform.identity()
        p._origin = origin
        p._x_axis = x_axis
        p._y_axis = y_axis
        p._z_axis = z_axis
        p._a = z_axis[0]
        p._b = z_axis[1]
        p._c = z_axis[2]
        p._d = -(z_axis[0] * origin[0] + z_axis[1] * origin[1] + z_axis[2] * origin[2])
        return p

    ###########################################################################################
    # Operators
    ###########################################################################################

    def transform(self):
        """Apply the stored xform transformation to the plane.

        Transforms the plane in-place and resets xform to identity.
        """
        self._origin.xform = self.xform
        self._origin.transform()
        self._x_axis.xform = self.xform
        self._x_axis.transform()
        self._y_axis.xform = self.xform
        self._y_axis.transform()
        self._z_axis.xform = self.xform
        self._z_axis.transform()
        self.xform = Xform.identity()

    def transformed(self):
        """Return a transformed copy of the plane."""
        import copy

        result = copy.deepcopy(self)
        result.transform()
        return result

    def duplicate(self):
        """Create a deep copy with a new GUID."""
        import copy

        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    @property
    def str(self):
        """Return minimal string representation."""
        ox = f"{self._origin[0]:.6f}"
        oy = f"{self._origin[1]:.6f}"
        oz = f"{self._origin[2]:.6f}"
        xx = f"{self._x_axis[0]:.6f}"
        xy = f"{self._x_axis[1]:.6f}"
        xz = f"{self._x_axis[2]:.6f}"
        yx = f"{self._y_axis[0]:.6f}"
        yy = f"{self._y_axis[1]:.6f}"
        yz = f"{self._y_axis[2]:.6f}"
        zx = f"{self._z_axis[0]:.6f}"
        zy = f"{self._z_axis[1]:.6f}"
        zz = f"{self._z_axis[2]:.6f}"
        return f"{ox}, {oy}, {oz}\n{xx}, {xy}, {xz}\n{yx}, {yy}, {yz}\n{zx}, {zy}, {zz}"

    def repr(self):
        """Return full string representation."""
        return f"Plane({self.name}, {self._origin[0]}, {self._origin[1]}, {self._origin[2]}, {self._z_axis[0]}, {self._z_axis[1]}, {self._z_axis[2]}, {repr(self.linecolor)})"

    def __str__(self):
        return self.str

    def __repr__(self):
        return self.repr()

    def __eq__(self, other):
        if isinstance(other, Plane):
            return (self.name == other.name and
                    self._origin == other._origin and
                    self._x_axis == other._x_axis and
                    self._y_axis == other._y_axis and
                    self._z_axis == other._z_axis and
                    self.linecolor == other.linecolor)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, index):
        """Get axis by index (0=x, 1=y, 2=z)."""
        if index == 0:
            return self._x_axis
        elif index == 1:
            return self._y_axis
        elif index == 2:
            return self._z_axis
        raise IndexError("Plane index out of range (0-2)")

    ###########################################################################################
    # No-copy Operators
    ###########################################################################################

    def __iadd__(self, other):
        """Translate plane by vector (in-place)."""
        if isinstance(other, Vector):
            self._origin += other
            self._update_equation()
        return self

    def __isub__(self, other):
        """Translate plane by negative vector (in-place)."""
        if isinstance(other, Vector):
            self._origin -= other
            self._update_equation()
        return self

    ###########################################################################################
    # Copy Operators
    ###########################################################################################

    def __add__(self, other):
        """Translate plane by vector (copy)."""
        if isinstance(other, Vector):
            result = Plane.__new__(Plane)
            result.guid = self.guid
            result.name = self.name
            result.width = self.width
            result._linecolor = None
            result.xform = Xform.identity()
            result._origin = self._origin + other
            result._x_axis = Vector(self._x_axis[0], self._x_axis[1], self._x_axis[2])
            result._y_axis = Vector(self._y_axis[0], self._y_axis[1], self._y_axis[2])
            result._z_axis = Vector(self._z_axis[0], self._z_axis[1], self._z_axis[2])
            result._update_equation()
            return result
        return NotImplemented

    def __sub__(self, other):
        """Translate plane by negative vector (copy)."""
        if isinstance(other, Vector):
            result = Plane.__new__(Plane)
            result.guid = self.guid
            result.name = self.name
            result.width = self.width
            result.xform = Xform.identity()
            result._origin = self._origin - other
            result._x_axis = Vector(self._x_axis[0], self._x_axis[1], self._x_axis[2])
            result._y_axis = Vector(self._y_axis[0], self._y_axis[1], self._y_axis[2])
            result._z_axis = Vector(self._z_axis[0], self._z_axis[1], self._z_axis[2])
            result._update_equation()
            return result
        return NotImplemented

    ###########################################################################################
    # Details
    ###########################################################################################

    def reverse(self):
        """Reverse the plane's normal direction."""
        temp = self._x_axis
        self._x_axis = self._y_axis
        self._y_axis = temp
        self._z_axis.reverse()
        self._update_equation()

    def rotate(self, angles_in_radians):
        """Rotate the plane around its normal.

        Parameters
        ----------
        angles_in_radians : float
            Rotation angle in radians.
        """
        cos_angle = math.cos(angles_in_radians)
        sin_angle = math.sin(angles_in_radians)

        new_x = self._x_axis * cos_angle + self._y_axis * sin_angle
        new_y = self._y_axis * cos_angle - self._x_axis * sin_angle

        self._x_axis = new_x
        self._y_axis = new_y
        self._update_equation()

    def is_right_hand(self):
        """Check if the plane follows the right-hand rule.

        Returns
        -------
        bool
            True if x_axis × y_axis = z_axis (right-handed).
        """
        cross = self._x_axis.cross(self._y_axis)
        dot_product = cross.dot(self._z_axis)
        return dot_product > 0.999

    @staticmethod
    def is_same_direction(plane0, plane1, can_be_flipped=True):
        """Check if two planes have the same or flipped normal.

        Parameters
        ----------
        plane0 : Plane
            First plane.
        plane1 : Plane
            Second plane.
        can_be_flipped : bool, optional
            Allow flipped normals. Defaults to True.

        Returns
        -------
        bool
            True if normals are parallel or antiparallel.
        """
        n0 = plane0._z_axis
        n1 = plane1._z_axis

        parallel = n0.is_parallel_to(n1)

        if can_be_flipped:
            return parallel != 0
        else:
            return parallel == -1

    @staticmethod
    def is_same_position(plane0, plane1):
        """Check if two planes are in the same position.

        Parameters
        ----------
        plane0 : Plane
            First plane.
        plane1 : Plane
            Second plane.

        Returns
        -------
        bool
            True if origins are very close.
        """
        dist0 = abs(
            plane0._a * plane1._origin[0]
            + plane0._b * plane1._origin[1]
            + plane0._c * plane1._origin[2]
            + plane0._d
        )

        dist1 = abs(
            plane1._a * plane0._origin[0]
            + plane1._b * plane0._origin[1]
            + plane1._c * plane0._origin[2]
            + plane1._d
        )

        tolerance = Tolerance.APPROXIMATION
        return dist0 < tolerance and dist1 < tolerance

    @staticmethod
    def is_coplanar(plane0, plane1, can_be_flipped=True):
        """Check if two planes are coplanar.

        Parameters
        ----------
        plane0 : Plane
            First plane.
        plane1 : Plane
            Second plane.
        can_be_flipped : bool, optional
            Allow flipped normals. Defaults to True.

        Returns
        -------
        bool
            True if planes are coplanar.
        """
        return Plane.is_same_direction(
            plane0, plane1, can_be_flipped
        ) and Plane.is_same_position(plane0, plane1)

    @staticmethod
    def is_coplanar_from_normals(origin0, normal0, origin1, normal1, can_be_flipped=True, tolerance=-1.0):
        """Check coplanarity from origin+normal without constructing Plane objects."""
        from .vector import Vector
        n0 = Vector(normal0[0], normal0[1], normal0[2])
        n1 = Vector(normal1[0], normal1[1], normal1[2])
        parallel = n0.is_parallel_to(n1)
        if can_be_flipped:
            if parallel == 0:
                return False
        else:
            if parallel != -1:
                return False
        a0, b0, c0 = n0[0], n0[1], n0[2]
        d0 = -(a0 * origin0[0] + b0 * origin0[1] + c0 * origin0[2])
        a1, b1, c1 = n1[0], n1[1], n1[2]
        d1 = -(a1 * origin1[0] + b1 * origin1[1] + c1 * origin1[2])
        from .tolerance import TOLERANCE
        tol = TOLERANCE.approximation if tolerance < 0 else tolerance
        dist0 = abs(a0 * origin1[0] + b0 * origin1[1] + c0 * origin1[2] + d0)
        dist1 = abs(a1 * origin0[0] + b1 * origin0[1] + c1 * origin0[2] + d1)
        return dist0 < tol and dist1 < tol

    def has_on_negative_side(self, p):
        """Sign test using the cached plane equation ``ax + by + cz + d``.

        Returns ``True`` if ``p`` lies on the negative side
        (``a*p[0] + b*p[1] + c*p[2] + d < 0``). Mirrors CGAL's
        ``Plane_3::has_on_negative_side``.

        Parameters
        ----------
        p : :class:`Point`

        Returns
        -------
        bool
        """
        return (self.a * p[0] + self.b * p[1] + self.c * p[2] + self.d) < 0.0

    def translate_by_normal(self, distance):
        """Translate (move) a plane along its normal direction by a specified distance.

        Parameters
        ----------
        distance : float
            Distance to move the plane along its normal (positive = normal direction, negative = opposite).

        Returns
        -------
        Plane
            New plane translated by the specified distance.
        """
        normal = Vector(self._z_axis[0], self._z_axis[1], self._z_axis[2])
        normal.normalize_self()

        new_origin = self._origin + (normal * distance)

        return Plane(new_origin, self._x_axis, self._y_axis)

    def to_polylines(self, scale=1.0):
        from .polyline import Polyline
        s = scale * 0.5
        o = self._origin
        x = self._x_axis
        y = self._y_axis
        z = self._z_axis
        c0 = Point(o[0] - x[0]*s - y[0]*s, o[1] - x[1]*s - y[1]*s, o[2] - x[2]*s - y[2]*s)
        c1 = Point(o[0] + x[0]*s - y[0]*s, o[1] + x[1]*s - y[1]*s, o[2] + x[2]*s - y[2]*s)
        c2 = Point(o[0] + x[0]*s + y[0]*s, o[1] + x[1]*s + y[1]*s, o[2] + x[2]*s + y[2]*s)
        c3 = Point(o[0] - x[0]*s + y[0]*s, o[1] - x[1]*s + y[1]*s, o[2] - x[2]*s + y[2]*s)
        rect = Polyline([c0, c1, c2, c3, c0])
        rect.linecolor = Color(self.linecolor[0], self.linecolor[1], self.linecolor[2], self.linecolor[3])
        origin_pt = Point(o[0], o[1], o[2])
        x_line = Polyline([origin_pt, Point(o[0] + x[0]*s, o[1] + x[1]*s, o[2] + x[2]*s)])
        x_line.linecolor = Color.red()
        y_line = Polyline([origin_pt, Point(o[0] + y[0]*s, o[1] + y[1]*s, o[2] + y[2]*s)])
        y_line.linecolor = Color.green()
        z_line = Polyline([origin_pt, Point(o[0] + z[0]*s, o[1] + z[1]*s, o[2] + z[2]*s)])
        z_line.linecolor = Color.blue()
        return [rect, x_line, y_line, z_line]

    ###########################################################################################
    # Polymorphic JSON Serialization
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field.

        Returns
        -------
        dict
            Dictionary with 'type', 'guid', 'name', and object fields.
            Uses single flat array of 12 numbers for frame:
            [ox, oy, oz, xx, xy, xz, yx, yy, yz, zx, zy, zz]
            Plane equation coefficients (a, b, c, d) are computed on load.

        """
        # Alphabetical order to match Rust's serde_json
        return {
            "linecolor": self.linecolor.__jsondump__(),
            "frame": [
                self._origin[0], self._origin[1], self._origin[2],
                self._x_axis[0], self._x_axis[1], self._x_axis[2],
                self._y_axis[0], self._y_axis[1], self._y_axis[2],
                self._z_axis[0], self._z_axis[1], self._z_axis[2],
            ],
            "guid": self.guid,
            "name": self.name,
            "type": f"{self.__class__.__name__}",
            "width": self.width,
            "xform": self.xform.__jsondump__(),
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format.

        Parameters
        ----------
        data : dict
            Dictionary containing plane data.
        guid : str, optional
            GUID for the plane.
        name : str, optional
            Name for the plane.

        Returns
        -------
        :class:`Plane`
            Reconstructed plane instance.

        """
        from .encoders import decode_node

        # Load frame as flat array of 12 numbers:
        # [ox, oy, oz, xx, xy, xz, yx, yy, yz, zx, zy, zz]
        frame = data["frame"]

        origin = Point(frame[0], frame[1], frame[2])
        x_axis = Vector(frame[3], frame[4], frame[5])
        y_axis = Vector(frame[6], frame[7], frame[8])

        width = data.get("width", 1.0)

        plane = cls(origin, x_axis, y_axis, width=width)
        plane.guid = guid if guid is not None else data.get("guid", plane.guid)
        plane.name = name if name is not None else data.get("name", plane.name)

        # Load linecolor
        if "linecolor" in data:
            plane.linecolor = decode_node(data["linecolor"])

        # Load xform if present
        if "xform" in data:
            plane.xform = decode_node(data["xform"])

        return plane

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
        :class:`Plane`
            The deserialized Plane.

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
        from .proto import plane_pb2

        proto = plane_pb2.Plane()
        proto.guid = self.guid
        proto.name = self.name
        proto.width = self.width

        # Set frame as flat array of 12 numbers:
        # [ox, oy, oz, xx, xy, xz, yx, yy, yz, zx, zy, zz]
        proto.frame.extend([
            self._origin[0], self._origin[1], self._origin[2],
            self._x_axis[0], self._x_axis[1], self._x_axis[2],
            self._y_axis[0], self._y_axis[1], self._y_axis[2],
            self._z_axis[0], self._z_axis[1], self._z_axis[2],
        ])

        # Set linecolor
        proto.linecolor.name = self.linecolor.name
        proto.linecolor.r = self.linecolor[0]
        proto.linecolor.g = self.linecolor[1]
        proto.linecolor.b = self.linecolor[2]
        proto.linecolor.a = self.linecolor[3]

        # Set xform
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        """Create Plane from protobuf binary data.

        Parameters
        ----------
        data : bytes
            Protobuf-encoded plane data.

        Returns
        -------
        :class:`Plane`
            The deserialized Plane.

        """
        from .proto import plane_pb2

        proto = plane_pb2.Plane()
        proto.ParseFromString(data)

        # Load frame as flat array of 12 numbers
        frame = list(proto.frame)
        origin = Point(frame[0], frame[1], frame[2])
        x_axis = Vector(frame[3], frame[4], frame[5])
        y_axis = Vector(frame[6], frame[7], frame[8])

        plane = cls(origin, x_axis, y_axis, width=proto.width if proto.width > 0 else 1.0)
        plane.guid = proto.guid
        plane.name = proto.name

        # Load linecolor
        plane.linecolor = Color(
            proto.linecolor.r,
            proto.linecolor.g,
            proto.linecolor.b,
            proto.linecolor.a
        )
        plane.linecolor.name = proto.linecolor.name

        # Load xform if present
        if proto.HasField('xform'):
            plane.xform = Xform()
            plane.xform.name = proto.xform.name
            plane.xform.m = list(proto.xform.matrix)

        return plane

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
        :class:`Plane`
            The deserialized Plane.

        """
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)
