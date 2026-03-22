import numpy as np
import math
from typing import List, Tuple, Optional, Union
import uuid

from .point import Point
from .vector import Vector
from .plane import Plane
from .tolerance import Tolerance
from .tolerance import PI
from .obb import Obb
from .xform import Xform
from .color import Color
from .nurbscurve import NurbsCurve
from . import knot


class NurbsSurface:
    """A Non-Uniform Rational B-Spline (NURBS) surface.
    
    A NURBS surface is defined by:
    - 2D array of control points (CVs)
    - Two knot vectors (one for each parameter direction)
    - Degrees in both directions (order = degree + 1)
    - Optional weights for rational surfaces
    
    Based on OpenNURBS implementation, adapted for session_py data types.
    
    Parameters
    ----------
    dimension : int, optional
        The dimension of the surface (typically 3 for 3D). Defaults to 3.
    is_rational : bool, optional
        Whether the surface is rational (has weights). Defaults to False.
    order0 : int, optional
        The order in u direction (degree + 1). Defaults to 4 (cubic).
    order1 : int, optional
        The order in v direction (degree + 1). Defaults to 4 (cubic).
    cv_count0 : int, optional
        Number of control vertices in u direction. Defaults to 0.
    cv_count1 : int, optional
        Number of control vertices in v direction. Defaults to 0.
    """
    
    def __init__(self, dimension: int = 3, is_rational: bool = False,
                 order0: int = 4, order1: int = 4,
                 cv_count0: int = 0, cv_count1: int = 0,
                 is_periodic_u: bool = False, is_periodic_v: bool = False,
                 knot_delta_u: float = 1.0, knot_delta_v: float = 1.0):
        """Initialize a NURBS surface."""
        self._guid = None
        self.name = "my_nurbssurface"
        self.width = 1.0
        self.pointcolors = []
        self.facecolors = []
        self.linecolors = []
        self._xform = None

        # Core NURBS data
        self.m_dim = 0
        self.m_is_rat = 0
        self.m_order = [0, 0]
        self.m_cv_count = [0, 0]
        self.m_cv_stride = [0, 0]

        # Data arrays
        self.m_knot = [np.array([], dtype=np.float64), np.array([], dtype=np.float64)]
        self.m_cv = np.array([], dtype=np.float64)
        self.m_mesh = None

        # Create if parameters provided
        if cv_count0 > 0 and cv_count1 > 0:
            self._create_impl(dimension, is_rational, order0, order1, cv_count0, cv_count1)

            # Initialize knot vectors
            if is_periodic_u:
                self.make_periodic_uniform_knot_vector(0, knot_delta_u)
            else:
                self.make_clamped_uniform_knot_vector(0, knot_delta_u)

            if is_periodic_v:
                self.make_periodic_uniform_knot_vector(1, knot_delta_v)
            else:
                self.make_clamped_uniform_knot_vector(1, knot_delta_v)
    
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

    ###########################################################################
    # INITIALIZATION & CREATION
    ###########################################################################

    def initialize(self):
        """Initialize all fields to zero/empty."""
        self._guid = None
        self.name = "my_nurbssurface"
        self.width = 1.0
        self.pointcolors = []
        self.facecolors = []
        self.linecolors = []
        self._xform = None

        self.m_dim = 0
        self.m_is_rat = 0
        self.m_order = [0, 0]
        self.m_cv_count = [0, 0]
        self.m_cv_stride = [0, 0]

        self.m_knot = [np.array([], dtype=np.float64), np.array([], dtype=np.float64)]
        self.m_cv = np.array([], dtype=np.float64)
    
    @staticmethod
    def create_raw(dimension: int, is_rational: bool,
               order0: int, order1: int,
               cv_count0: int, cv_count1: int,
               is_periodic_u: bool = False, is_periodic_v: bool = False,
               knot_delta_u: float = 1.0, knot_delta_v: float = 1.0) -> 'NurbsSurface':
        """Create NURBS surface with specified parameters (static factory method).

        Parameters
        ----------
        dimension : int
            Dimension of the surface (typically 3).
        is_rational : bool
            Whether the surface should be rational.
        order0 : int
            Order in u direction (degree + 1).
        order1 : int
            Order in v direction (degree + 1).
        cv_count0 : int
            Number of control vertices in u direction.
        cv_count1 : int
            Number of control vertices in v direction.
        is_periodic_u : bool, optional
            If True, creates periodic uniform knot vector in u direction. Defaults to False.
        is_periodic_v : bool, optional
            If True, creates periodic uniform knot vector in v direction. Defaults to False.
        knot_delta_u : float, optional
            Knot spacing in u direction. Defaults to 1.0.
        knot_delta_v : float, optional
            Knot spacing in v direction. Defaults to 1.0.

        Returns
        -------
        NurbsSurface or None
            The created surface, or None if parameters are invalid.
        """
        surf = NurbsSurface()
        if surf._create_impl(dimension, is_rational, order0, order1, cv_count0, cv_count1):
            # Initialize knot vectors
            if is_periodic_u:
                surf.make_periodic_uniform_knot_vector(0, knot_delta_u)
            else:
                surf.make_clamped_uniform_knot_vector(0, knot_delta_u)

            if is_periodic_v:
                surf.make_periodic_uniform_knot_vector(1, knot_delta_v)
            else:
                surf.make_clamped_uniform_knot_vector(1, knot_delta_v)

            return surf
        return None

    @staticmethod
    def create(periodic_u: bool, periodic_v: bool,
               degree_u: int, degree_v: int,
               cv_count_u: int, cv_count_v: int,
               points: List['Point']) -> 'NurbsSurface':
        if degree_u < 1 or degree_v < 1:
            raise ValueError(f"NurbsSurface.create: degree must be >= 1, got degree_u={degree_u}, degree_v={degree_v}")
        if cv_count_u < degree_u + 1:
            raise ValueError(f"NurbsSurface.create: cv_count_u ({cv_count_u}) must be >= degree_u+1 ({degree_u + 1})")
        if cv_count_v < degree_v + 1:
            raise ValueError(f"NurbsSurface.create: cv_count_v ({cv_count_v}) must be >= degree_v+1 ({degree_v + 1})")
        expected = cv_count_u * cv_count_v
        if len(points) != expected:
            raise ValueError(f"NurbsSurface.create: expected {expected} points ({cv_count_u}x{cv_count_v}), got {len(points)}")
        order0 = degree_u + 1
        order1 = degree_v + 1
        surf = NurbsSurface.create_raw(3, False, order0, order1, cv_count_u, cv_count_v,
                                       periodic_u, periodic_v, 1.0, 1.0)
        if surf is None:
            return NurbsSurface()
        for i in range(cv_count_u):
            for j in range(cv_count_v):
                surf.set_cv(i, j, points[i * cv_count_v + j])
        return surf

    def _create_impl(self, dimension: int, is_rational: bool,
               order0: int, order1: int,
               cv_count0: int, cv_count1: int) -> bool:
        """Create NURBS surface with specified parameters.
        
        Parameters
        ----------
        dimension : int
            Dimension of the surface (typically 3).
        is_rational : bool
            Whether the surface should be rational.
        order0 : int
            Order in u direction (degree + 1).
        order1 : int
            Order in v direction (degree + 1).
        cv_count0 : int
            Number of control vertices in u direction.
        cv_count1 : int
            Number of control vertices in v direction.
        
        Returns
        -------
        bool
            True if creation successful, False otherwise.
        """
        if dimension < 1 or order0 < 2 or order1 < 2:
            return False
        if cv_count0 < order0 or cv_count1 < order1:
            return False
        
        self.destroy()
        
        self.m_dim = dimension
        self.m_is_rat = 1 if is_rational else 0
        self.m_order = [order0, order1]
        self.m_cv_count = [cv_count0, cv_count1]
        
        # OpenNURBS stride pattern: [1] is CV size, [0] is row stride
        cv_size_val = (dimension + 1) if is_rational else dimension
        self.m_cv_stride[1] = cv_size_val
        self.m_cv_stride[0] = cv_size_val * cv_count1
        
        # Allocate knot vectors
        # OpenNURBS formula: knot_count = order + cv_count - 2
        knot_count0 = order0 + cv_count0 - 2
        knot_count1 = order1 + cv_count1 - 2
        
        self.m_knot[0] = np.zeros(knot_count0, dtype=np.float64)
        self.m_knot[1] = np.zeros(knot_count1, dtype=np.float64)

        # Allocate CV array
        total_cvs = cv_count0 * cv_count1
        cv_array_size = total_cvs * cv_size_val
        self.m_cv = np.zeros(cv_array_size, dtype=np.float64)

        # Initialize weights to 1 if rational
        if is_rational:
            for i in range(cv_count0):
                for j in range(cv_count1):
                    self.set_weight(i, j, 1.0)
        
        return True
    
    def destroy(self):
        """Deallocate all memory and reset to empty state."""
        self.m_knot = [np.array([], dtype=np.float64), np.array([], dtype=np.float64)]
        self.m_cv = np.array([], dtype=np.float64)
        self.initialize()
    
    ###########################################################################
    # BOOLEAN QUERIES
    ###########################################################################

    def is_valid(self) -> bool:
        """Check if NURBS surface is valid.

        Returns
        -------
        bool
            True if surface is valid, False otherwise.
        """
        if self.m_dim < 1:
            return False

        # Check both directions
        for dir in range(2):
            if self.m_order[dir] < 2:
                return False
            if self.m_cv_count[dir] < self.m_order[dir]:
                return False

            # OpenNURBS formula: knot_count = order + cv_count - 2
            knot_count = self.m_order[dir] + self.m_cv_count[dir] - 2
            if len(self.m_knot[dir]) != knot_count:
                return False

            if not self.is_valid_knot_vector(dir):
                return False

            # Check stride is valid (OpenNURBS check)
            cv_size_val = (self.m_dim + 1) if self.m_is_rat else self.m_dim
            if self.m_cv_stride[dir] < cv_size_val:
                return False

        # Check CV array size
        cv_size_val = (self.m_dim + 1) if self.m_is_rat else self.m_dim
        expected_cv_size = self.m_cv_count[0] * self.m_cv_count[1] * cv_size_val
        if len(self.m_cv) < expected_cv_size:
            return False

        return True

    def is_rational(self) -> bool:
        """Check if surface is rational."""
        return self.m_is_rat != 0

    def is_closed(self, dir: int) -> bool:
        """Check if surface is closed in specified direction.

        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).

        Returns
        -------
        bool
            True if closed, False otherwise.
        """
        if dir < 0 or dir >= 2 or not self.is_valid():
            return False

        # Check if first and last rows/columns are coincident
        if dir == 0:
            # Check u direction - compare first and last u CVs
            for j in range(self.m_cv_count[1]):
                pt0 = self.get_cv(0, j)
                pt1 = self.get_cv(self.m_cv_count[0] - 1, j)
                if pt0.distance(pt1) > 1e-12:
                    return False
        else:
            # Check v direction - compare first and last v CVs
            for i in range(self.m_cv_count[0]):
                pt0 = self.get_cv(i, 0)
                pt1 = self.get_cv(i, self.m_cv_count[1] - 1)
                if pt0.distance(pt1) > 1e-12:
                    return False

        return True

    def is_periodic(self, dir: int) -> bool:
        """Check if surface is periodic in specified direction.

        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).

        Returns
        -------
        bool
            True if periodic, False otherwise.
        """
        if dir < 0 or dir >= 2 or not self.is_valid():
            return False

        # Check knot vector periodicity
        degree = self.degree(dir)
        kc = self.knot_count(dir)

        if kc != self.m_order[dir] + self.m_cv_count[dir] - 2:
            return False

        # Check uniform spacing
        delta = self.m_knot[dir][self.m_cv_count[dir] - 1] - self.m_knot[dir][degree]
        if delta <= 0:
            return False

        for i in range(self.m_cv_count[dir] - 1):
            expected = self.m_knot[dir][i + degree] + delta
            if abs(self.m_knot[dir][i + self.m_order[dir] - 1] - expected) > 1e-10:
                return False

        # Check CV periodicity
        i0 = self.m_order[dir] - 2
        i1 = self.m_cv_count[dir] - 1

        for k in range(self.m_cv_count[1 - dir]):
            for check_i in range(i0 + 1):
                if dir == 0:
                    pt0 = self.get_cv(check_i, k)
                    pt1 = self.get_cv(i1 - (i0 - check_i), k)
                else:
                    pt0 = self.get_cv(k, check_i)
                    pt1 = self.get_cv(k, i1 - (i0 - check_i))

                if pt0.distance(pt1) > 1e-12:
                    return False

        return True

    def is_planar(self, plane: Optional[Plane] = None, tolerance: float = Tolerance.ZERO_TOLERANCE) -> bool:
        """Check if surface is planar within tolerance.

        Parameters
        ----------
        plane : Plane, optional
            If provided, will be set to the best-fit plane.
        tolerance : float, optional
            Tolerance for planarity check.

        Returns
        -------
        bool
            True if surface is planar, False otherwise.
        """
        if self.m_cv_count[0] < 2 or self.m_cv_count[1] < 2:
            return False

        p0 = self.get_cv(0, 0)
        normal = Vector(0, 0, 0)
        n_len = 0.0
        found = False
        for i in range(self.m_cv_count[0]):
            if found:
                break
            for j in range(self.m_cv_count[1]):
                if found:
                    break
                for ii in range(i, self.m_cv_count[0]):
                    if found:
                        break
                    jj_start = j + 1 if ii == i else 0
                    for jj in range(jj_start, self.m_cv_count[1]):
                        pa = self.get_cv(i, j)
                        pb = self.get_cv(ii, jj)
                        va = Vector(pa.x - p0.x, pa.y - p0.y, pa.z - p0.z)
                        vb = Vector(pb.x - p0.x, pb.y - p0.y, pb.z - p0.z)
                        normal = va.cross(vb)
                        n_len = normal.magnitude()
                        if n_len >= 1e-14:
                            found = True
                            break
        if n_len < 1e-14:
            return True

        normal = normal / n_len
        test_plane = Plane(p0, normal)

        # Check all CVs against plane
        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                pt = self.get_cv(i, j)
                # Compute distance: |dot(pt - p0, normal)|
                v = Vector(pt.x - p0.x, pt.y - p0.y, pt.z - p0.z)
                dist = abs(v.dot(normal))
                if dist > tolerance:
                    return False

        if plane is not None:
            plane.origin = test_plane.origin
            plane.normal = test_plane.normal

        return True

    def is_singular(self, side: int) -> bool:
        """Check if surface side is singular (collapsed to a point).

        Parameters
        ----------
        side : int
            Side (0=south, 1=east, 2=north, 3=west).

        Returns
        -------
        bool
            True if singular, False otherwise.
        """
        if not self.is_valid():
            return False

        points = []

        if side == 0:  # south (v=0)
            if not self.is_clamped(1, 0):
                return False
            points = [self.get_cv(i, 0) for i in range(self.m_cv_count[0])]
        elif side == 1:  # east (u=max)
            if not self.is_clamped(0, 1):
                return False
            points = [self.get_cv(self.m_cv_count[0] - 1, j) for j in range(self.m_cv_count[1])]
        elif side == 2:  # north (v=max)
            if not self.is_clamped(1, 1):
                return False
            points = [self.get_cv(i, self.m_cv_count[1] - 1) for i in range(self.m_cv_count[0])]
        elif side == 3:  # west (u=0)
            if not self.is_clamped(0, 0):
                return False
            points = [self.get_cv(0, j) for j in range(self.m_cv_count[1])]
        else:
            return False

        # Check if all points are coincident
        if len(points) < 2:
            return False

        p0 = points[0]
        for pt in points[1:]:
            if p0.distance(pt) > 1e-12:
                return False

        return True

    def is_clamped(self, dir: int, end: int = 2) -> bool:
        """Check if surface is clamped in specified direction.

        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        end : int, optional
            Which end to check (0=start, 1=end, 2=both). Defaults to 2.

        Returns
        -------
        bool
            True if clamped, False otherwise.
        """
        if dir < 0 or dir >= 2:
            return False
        if len(self.m_knot[dir]) == 0:
            return False

        # Use knot module function
        return knot.is_clamped(self.m_order[dir], self.m_cv_count[dir], self.m_knot[dir], end)

    def is_duplicate(self, other, ignore_parameterization: bool = False, tolerance: float = None) -> bool:
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE
        if not self.is_valid() or not other.is_valid():
            return False
        if self.m_dim != other.m_dim:
            return False
        if self.m_is_rat != other.m_is_rat:
            return False
        if self.m_order[0] != other.m_order[0] or self.m_order[1] != other.m_order[1]:
            return False
        if self.m_cv_count[0] != other.m_cv_count[0] or self.m_cv_count[1] != other.m_cv_count[1]:
            return False
        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                p1 = self.get_cv(i, j)
                p2 = other.get_cv(i, j)
                if p1.distance(p2) > tolerance:
                    return False
                if self.m_is_rat:
                    if abs(self.weight(i, j) - other.weight(i, j)) > tolerance:
                        return False
        if not ignore_parameterization:
            for dir in range(2):
                for i in range(self.knot_count(dir)):
                    if abs(self.knot(dir, i) - other.knot(dir, i)) > tolerance:
                        return False
        return True

    def is_valid_knot_vector(self, dir: int) -> bool:
        """Check if knot vector is valid in specified direction.

        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).

        Returns
        -------
        bool
            True if knot vector is valid (non-decreasing).
        """
        if dir < 0 or dir >= 2:
            return False
        kc = self.knot_count(dir)
        if len(self.m_knot[dir]) != kc:
            return False

        for i in range(1, kc):
            if self.m_knot[dir][i] < self.m_knot[dir][i-1]:
                return False
        return True

    def __eq__(self, other) -> bool:
        """Check equality with another NurbsSurface (compares all attributes except guid)."""
        if not isinstance(other, NurbsSurface):
            return False

        # Compare metadata (excluding guid)
        if self.name != other.name:
            return False
        if self.width != other.width:
            return False
        if self.pointcolors != other.pointcolors:
            return False
        if self.facecolors != other.facecolors:
            return False
        if self.linecolors != other.linecolors:
            return False
        if self.xform != other.xform:
            return False

        # Compare NURBS structure
        if self.m_dim != other.m_dim:
            return False
        if self.m_is_rat != other.m_is_rat:
            return False
        if self.m_order != other.m_order:
            return False
        if self.m_cv_count != other.m_cv_count:
            return False
        if self.m_cv_stride != other.m_cv_stride:
            return False

        # Compare knot vectors
        for i in range(2):
            if not np.array_equal(self.m_knot[i], other.m_knot[i]):
                return False

        # Compare control vertices
        if not np.array_equal(self.m_cv, other.m_cv):
            return False

        return True

    def __ne__(self, other) -> bool:
        """Check inequality with another NurbsSurface."""
        return not self.__eq__(other)

    def duplicate(self) -> 'NurbsSurface':
        """Create a deep copy of this surface with a new GUID.

        Returns
        -------
        NurbsSurface
            A new surface that is a copy of this one with a different GUID.
        """
        import copy
        import uuid
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result
    
    ###########################################################################
    # ACCESSORS
    ###########################################################################
    
    def dimension(self) -> int:
        """Get dimension of the surface."""
        return self.m_dim

    def order(self, dir: int) -> int:
        """Get order (degree + 1) in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        
        Returns
        -------
        int
            Order in specified direction, or 0 if invalid direction.
        """
        return self.m_order[dir] if 0 <= dir < 2 else 0
    
    def degree(self, dir: int) -> int:
        """Get degree (order - 1) in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        
        Returns
        -------
        int
            Degree in specified direction, or 0 if invalid direction.
        """
        return (self.m_order[dir] - 1) if 0 <= dir < 2 else 0
    
    def cv_count(self, dir: Optional[int] = None) -> int:
        """Get number of control vertices.
        
        Parameters
        ----------
        dir : int, optional
            Direction (0 for u, 1 for v). If None, returns total count.
        
        Returns
        -------
        int
            Number of control vertices.
        """
        if dir is None:
            return self.m_cv_count[0] * self.m_cv_count[1]
        return self.m_cv_count[dir] if 0 <= dir < 2 else 0
    
    def cv_count_dir(self, dir: Optional[int] = None) -> int:
        """Get number of control vertices (alias for cv_count).
        
        Parameters
        ----------
        dir : int, optional
            Direction (0 for u, 1 for v). If None, returns total count.
        
        Returns
        -------
        int
            Number of control vertices.
        """
        return self.cv_count(dir)
    
    def cv_size(self) -> int:
        """Get size of each control vertex.
        
        Returns
        -------
        int
            Dimension + 1 if rational, else dimension.
        """
        return (self.m_dim + 1) if self.m_is_rat else self.m_dim
    
    def knot_count(self, dir: int) -> int:
        """Get knot count in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        
        Returns
        -------
        int
            Number of knots in specified direction.
        """
        if dir < 0 or dir >= 2:
            return 0
        # OpenNURBS formula: knot_count = order + cv_count - 2
        return self.m_order[dir] + self.m_cv_count[dir] - 2
    
    def span_count(self, dir: int) -> int:
        """Get number of spans in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        
        Returns
        -------
        int
            Number of spans in specified direction.
        """
        if dir < 0 or dir >= 2:
            return 0
        return self.m_cv_count[dir] - self.m_order[dir] + 1

    ###########################################################################
    # CONTROL VERTEX ACCESS
    ###########################################################################
    
    def cv(self, i: int, j: int) -> Optional[np.ndarray]:
        """Get pointer to CV data at indices (i, j).
        
        Parameters
        ----------
        i : int
            Index in u direction.
        j : int
            Index in v direction.
        
        Returns
        -------
        np.ndarray or None
            View of CV data, or None if indices invalid.
        """
        if i < 0 or i >= self.m_cv_count[0] or j < 0 or j >= self.m_cv_count[1]:
            return None
        # OpenNURBS pattern: CV(i,j) = m_cv[i*m_cv_stride[0] + j*m_cv_stride[1]]
        index = i * self.m_cv_stride[0] + j * self.m_cv_stride[1]
        cv_size_val = self.cv_size()
        return self.m_cv[index:index + cv_size_val]
    
    def get_cv(self, i: int, j: int) -> Point:
        """Get control point as Point.
        
        Parameters
        ----------
        i : int
            Index in u direction.
        j : int
            Index in v direction.
        
        Returns
        -------
        Point
            Control point at (i, j).
        """
        cv_ptr = self.cv(i, j)
        if cv_ptr is None:
            return Point(0, 0, 0)
        
        if self.m_is_rat:
            w = cv_ptr[self.m_dim]
            if abs(w) < 1e-14:
                return Point(0, 0, 0)
            return Point(cv_ptr[0]/w,
                        cv_ptr[1]/w if self.m_dim > 1 else 0,
                        cv_ptr[2]/w if self.m_dim > 2 else 0)
        
        return Point(cv_ptr[0],
                    cv_ptr[1] if self.m_dim > 1 else 0,
                    cv_ptr[2] if self.m_dim > 2 else 0)
    
    def get_cv_4d(self, i: int, j: int) -> Tuple[bool, float, float, float, float]:
        """Get control point as homogeneous coordinates (x, y, z, w).
        
        Parameters
        ----------
        i : int
            Index in u direction.
        j : int
            Index in v direction.
        
        Returns
        -------
        tuple
            (success, x, y, z, w)
        """
        cv_ptr = self.cv(i, j)
        if cv_ptr is None:
            return (False, 0.0, 0.0, 0.0, 1.0)
        
        x = cv_ptr[0]
        y = cv_ptr[1] if self.m_dim > 1 else 0.0
        z = cv_ptr[2] if self.m_dim > 2 else 0.0
        w = cv_ptr[self.m_dim] if self.m_is_rat else 1.0
        
        return (True, x, y, z, w)
    
    def set_cv(self, i: int, j: int, point: Point) -> bool:
        """Set control point from Point.
        
        Parameters
        ----------
        i : int
            Index in u direction.
        j : int
            Index in v direction.
        point : Point
            Point to set.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        cv_ptr = self.cv(i, j)
        if cv_ptr is None:
            return False

        if self.m_is_rat:
            # For rational surfaces, store homogeneous coordinates (x*w, y*w, z*w, w)
            w = cv_ptr[self.m_dim]  # Get current weight
            if abs(w) < 1e-14:
                w = 1.0
            cv_ptr[0] = point.x * w
            if self.m_dim > 1:
                cv_ptr[1] = point.y * w
            if self.m_dim > 2:
                cv_ptr[2] = point.z * w
        else:
            cv_ptr[0] = point.x
            if self.m_dim > 1:
                cv_ptr[1] = point.y
            if self.m_dim > 2:
                cv_ptr[2] = point.z

        return True
    
    def set_cv_4d(self, i: int, j: int, x: float, y: float, z: float, w: float) -> bool:
        """Set control point from homogeneous coordinates.
        
        Parameters
        ----------
        i : int
            Index in u direction.
        j : int
            Index in v direction.
        x, y, z, w : float
            Homogeneous coordinates.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        cv_ptr = self.cv(i, j)
        if cv_ptr is None:
            return False
        
        cv_ptr[0] = x
        if self.m_dim > 1:
            cv_ptr[1] = y
        if self.m_dim > 2:
            cv_ptr[2] = z
        if self.m_is_rat:
            cv_ptr[self.m_dim] = w
        
        return True
    
    def weight(self, i: int, j: int) -> float:
        """Get weight at control vertex index.
        
        Parameters
        ----------
        i : int
            Index in u direction.
        j : int
            Index in v direction.
        
        Returns
        -------
        float
            Weight value (1.0 if non-rational).
        """
        if not self.m_is_rat:
            return 1.0
        cv_ptr = self.cv(i, j)
        return cv_ptr[self.m_dim] if cv_ptr is not None else 1.0
    
    def set_weight(self, i: int, j: int, w: float) -> bool:
        """Set weight at control vertex index.
        
        Parameters
        ----------
        i : int
            Index in u direction.
        j : int
            Index in v direction.
        w : float
            Weight value.
        
        Returns
        -------
        bool
            True if successful, False if non-rational or invalid indices.
        """
        if not self.m_is_rat:
            return False
        cv_ptr = self.cv(i, j)
        if cv_ptr is None:
            return False

        # Rescale homogeneous coordinates when changing weight
        old_w = cv_ptr[self.m_dim]
        if abs(old_w) < 1e-14:
            old_w = 1.0
        if abs(w) < 1e-14:
            w = 1.0

        scale = w / old_w
        cv_ptr[0] *= scale
        if self.m_dim > 1:
            cv_ptr[1] *= scale
        if self.m_dim > 2:
            cv_ptr[2] *= scale
        cv_ptr[self.m_dim] = w
        return True
    
    ###########################################################################
    # KNOT ACCESS
    ###########################################################################
    
    def knot(self, dir: int, knot_index: int) -> float:
        """Get knot value at index in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        knot_index : int
            Index in knot vector.
        
        Returns
        -------
        float
            Knot value, or 0.0 if invalid.
        """
        if dir < 0 or dir >= 2 or knot_index < 0 or knot_index >= len(self.m_knot[dir]):
            return 0.0
        return self.m_knot[dir][knot_index]
    
    def set_knot(self, dir: int, knot_index: int, knot_value: float) -> bool:
        """Set knot value at index in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        knot_index : int
            Index in knot vector.
        knot_value : float
            Knot value to set.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if dir < 0 or dir >= 2 or knot_index < 0 or knot_index >= len(self.m_knot[dir]):
            return False
        self.m_knot[dir][knot_index] = knot_value
        return True
    
    def knot_multiplicity(self, dir: int, knot_index: int) -> int:
        """Get knot multiplicity at index in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        knot_index : int
            Index in knot vector.
        
        Returns
        -------
        int
            Multiplicity of the knot.
        """
        if dir < 0 or dir >= 2:
            return 0
        
        # Use knot module function
        return knot.multiplicity(self.m_order[dir], self.m_cv_count[dir], self.m_knot[dir], knot_index)
    
    def get_knots(self, dir: int) -> np.ndarray:
        """Get all knot values for specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        
        Returns
        -------
        np.ndarray
            Copy of knot vector.
        """
        if dir < 0 or dir >= 2:
            return np.array([])
        return self.m_knot[dir].copy()

    ###########################################################################
    # DOMAIN & PARAMETERIZATION
    ###########################################################################
    
    def domain(self, dir: int) -> Tuple[float, float]:
        """Get surface domain [start_param, end_param] in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        
        Returns
        -------
        tuple
            (start_param, end_param)
        """
        if not self.is_valid() or dir < 0 or dir >= 2:
            return (0.0, 0.0)
        return (self.m_knot[dir][self.m_order[dir] - 2],
                self.m_knot[dir][self.m_cv_count[dir] - 1])
    
    def set_domain(self, dir: int, t0: float, t1: float) -> bool:
        """Set surface domain in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        t0 : float
            Start parameter.
        t1 : float
            End parameter.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if dir < 0 or dir >= 2 or t0 >= t1:
            return False
        
        old_t0, old_t1 = self.domain(dir)
        if abs(old_t1 - old_t0) < 1e-14:
            return False
        
        scale = (t1 - t0) / (old_t1 - old_t0)
        for i in range(len(self.m_knot[dir])):
            self.m_knot[dir][i] = t0 + (self.m_knot[dir][i] - old_t0) * scale
        
        return True
    
    def get_span_vector(self, dir: int) -> np.ndarray:
        """Get span (distinct knot intervals) values in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        
        Returns
        -------
        np.ndarray
            Array of span values.
        """
        if dir < 0 or dir >= 2:
            return np.array([])
        
        spans = []
        for i in range(len(self.m_knot[dir]) - 1):
            if abs(self.m_knot[dir][i+1] - self.m_knot[dir][i]) > 1e-14:
                spans.append(self.m_knot[dir][i])
        
        if len(self.m_knot[dir]) > 0:
            spans.append(self.m_knot[dir][-1])
        
        return np.array(spans)
    
    ###########################################################################
    # KNOT VECTOR OPERATIONS
    ###########################################################################
    
    def make_clamped_uniform_knot_vector(self, dir: int, delta: float = 1.0) -> bool:
        """Make knot vector a clamped uniform knot vector.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        delta : float, optional
            Spacing between internal knots. Defaults to 1.0.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if dir < 0 or dir >= 2:
            return False
        if self.m_order[dir] < 2 or self.m_cv_count[dir] < self.m_order[dir]:
            return False
        
        # Use knot module function
        result = knot.make_clamped_uniform(self.m_order[dir], self.m_cv_count[dir], delta)
        if result is None:
            return False
        self.m_knot[dir] = result
        return True
    
    def make_periodic_uniform_knot_vector(self, dir: int, delta: float = 1.0) -> bool:
        """Make knot vector a periodic uniform knot vector.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        delta : float, optional
            Spacing between knots. Defaults to 1.0.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if dir < 0 or dir >= 2:
            return False
        if self.m_order[dir] < 2 or self.m_cv_count[dir] < self.m_order[dir]:
            return False
        
        # Use knot module function
        result = knot.make_periodic_uniform(self.m_order[dir], self.m_cv_count[dir], delta)
        if result is None:
            return False
        self.m_knot[dir] = result
        return True

    def _find_span(self, dir: int, t: float) -> int:
        """Find the knot span index containing parameter t.
        
        Implements ON_NurbsSpanIndex algorithm from OpenNURBS.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        t : float
            Parameter value.
        
        Returns
        -------
        int
            Span index in range [0, cv_count-order].
        """
        # Use knot module function
        return knot.find_span(self.m_order[dir], self.m_cv_count[dir], self.m_knot[dir], t)
    
    def _basis_functions(self, dir: int, span: int, t: float) -> np.ndarray:
        """Compute basis functions.
        
        Implements ON_EvaluateNurbsBasis algorithm from OpenNURBS.
        The span parameter is the offset returned by _find_span.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        span : int
            Span index (offset in knot array).
        t : float
            Parameter value.
        
        Returns
        -------
        np.ndarray
            Basis function values.
        """
        order = self.m_order[dir]
        d = order - 1
        knot_base = span + d
        knot = self.m_knot[dir]

        if knot[knot_base - 1] == knot[knot_base]:
            out = np.zeros(order)
            if t <= knot[knot_base]:
                out[0] = 1.0
            else:
                out[order - 1] = 1.0
            return out

        N = np.zeros(order * order)
        N[order * order - 1] = 1.0
        left = np.zeros(d)
        right = np.zeros(d)
        N_idx = order * order - 1
        k_right = knot_base
        k_left = knot_base - 1

        for j in range(d):
            N0_idx = N_idx
            N_idx -= (order + 1)
            left[j] = t - knot[k_left]
            right[j] = knot[k_right] - t
            k_left -= 1
            k_right += 1

            x = 0.0
            for r in range(j + 1):
                a0 = left[j - r]
                a1 = right[r]
                denom = a0 + a1
                y = N[N0_idx + r] / denom if abs(denom) > 0.0 else 0.0
                N[N_idx + r] = x + a1 * y
                x = a0 * y
            N[N_idx + j + 1] = x

        return N[0:order]

    def _basis_functions_derivatives(self, dir: int, span: int, t: float, deriv_order: int) -> list:
        if dir < 0 or dir >= 2:
            return []
        order = self.m_order[dir]
        degree = order - 1
        kv = self.m_knot[dir]
        knot_base = span + degree

        ders = [[0.0] * order for _ in range(deriv_order + 1)]

        if kv[knot_base - 1] == kv[knot_base]:
            return ders

        ndu = [[0.0] * order for _ in range(order)]
        ndu[0][0] = 1.0
        left = [0.0] * (degree + 1)
        right = [0.0] * (degree + 1)

        for j in range(1, degree + 1):
            left[j] = t - kv[knot_base - j]
            right[j] = kv[knot_base + j - 1] - t
            saved = 0.0
            for r in range(j):
                ndu[j][r] = right[r + 1] + left[j - r]
                temp = ndu[r][j - 1] / ndu[j][r]
                ndu[r][j] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            ndu[j][j] = saved

        for j in range(degree + 1):
            ders[0][j] = ndu[j][degree]

        a = [[0.0] * order for _ in range(2)]
        for r in range(degree + 1):
            s1, s2 = 0, 1
            a[0][0] = 1.0
            for k in range(1, deriv_order + 1):
                d = 0.0
                rk = r - k
                pk = degree - k
                if r >= k:
                    a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                    d = a[s2][0] * ndu[rk][pk]
                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else degree - r
                for j in range(j1, j2 + 1):
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
                    d += a[s2][j] * ndu[rk + j][pk]
                if r <= pk:
                    a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
                    d += a[s2][k] * ndu[r][pk]
                ders[k][r] = d
                s1, s2 = s2, s1

        factorial = degree
        for k in range(1, deriv_order + 1):
            for j in range(degree + 1):
                ders[k][j] *= factorial
            factorial *= (degree - k)

        return ders

    ###########################################################################
    # EVALUATION
    ###########################################################################
    
    def point_at(self, u: float, v: float) -> Point:
        """Evaluate point on surface at parameter (u, v).
        
        Parameters
        ----------
        u : float
            Parameter in u direction.
        v : float
            Parameter in v direction.
        
        Returns
        -------
        Point
            Point on surface.
        """
        if not self.is_valid():
            return Point(0, 0, 0)
        
        # Find spans - returns indices in range [0, cv_count-order]
        span_u = self._find_span(0, u)
        span_v = self._find_span(1, v)
        
        # Compute basis functions
        Nu = self._basis_functions(0, span_u, u)
        Nv = self._basis_functions(1, span_v, v)
        
        # Evaluate surface point - OpenNURBS lines 1107-1117
        # CV index starts at span (since span is in range [0, cv_count-order])
        cv_size_val = self.cv_size()
        point = np.zeros(cv_size_val)
        
        for j0 in range(self.m_order[0]):
            cv_i = span_u + j0
            for j1 in range(self.m_order[1]):
                cv_j = span_v + j1
                c = Nu[j0] * Nv[j1]
                cv_data = self.cv(cv_i, cv_j)
                if cv_data is not None:
                    point += c * cv_data
        
        # Handle rational case
        if self.m_is_rat and abs(point[self.m_dim]) > 1e-14:
            w = point[self.m_dim]
            return Point(point[0]/w,
                        point[1]/w if self.m_dim > 1 else 0,
                        point[2]/w if self.m_dim > 2 else 0)
        
        return Point(point[0],
                    point[1] if self.m_dim > 1 else 0,
                    point[2] if self.m_dim > 2 else 0)
    
    def point_at_corner(self, u_end: int, v_end: int) -> Point:
        """Get point at corner (u_end, v_end) where end is 0 or 1.
        
        Parameters
        ----------
        u_end : int
            U corner (0 or 1).
        v_end : int
            V corner (0 or 1).
        
        Returns
        -------
        Point
            Corner point.
        """
        i = 0 if u_end == 0 else self.m_cv_count[0] - 1
        j = 0 if v_end == 0 else self.m_cv_count[1] - 1
        return self.get_cv(i, j)
    
    def normal_at(self, u: float, v: float) -> Vector:
        """Get normal vector at parameter (u, v).
        
        Parameters
        ----------
        u : float
            Parameter in u direction.
        v : float
            Parameter in v direction.
        
        Returns
        -------
        Vector
            Normal vector at (u, v).
        """
        derivs = self.evaluate(u, v, 1)
        if len(derivs) < 3:
            return Vector(0, 0, 1)
        
        du = derivs[1]
        dv = derivs[2]
        normal = dv.cross(du)
        
        mag = normal.magnitude()
        if mag < 1e-14:
            return Vector(0, 0, 1)
        
        return normal / mag
    
    def evaluate(self, u: float, v: float, num_derivs: int = 0) -> List[Vector]:
        if not self.is_valid() or num_derivs < 0:
            return []
        max_derivs = min(num_derivs, 2)
        span_u = self._find_span(0, u)
        span_v = self._find_span(1, v)
        ders_u = self._basis_functions_derivatives(0, span_u, u, max_derivs)
        ders_v = self._basis_functions_derivatives(1, span_v, v, max_derivs)

        cv_size_val = (self.m_dim + 1) if self.m_is_rat else self.m_dim

        # Compute all homogeneous derivatives
        skl_all = []
        for k in range(max_derivs + 1):
            for l in range(max_derivs - k + 1):
                skl = [0.0] * cv_size_val
                for i in range(self.m_order[0]):
                    cv_i = span_u + i
                    for j in range(self.m_order[1]):
                        cv_j = span_v + j
                        coeff = ders_u[k][i] * ders_v[l][j]
                        cv_data = self.cv(cv_i, cv_j)
                        if cv_data is not None:
                            for d in range(cv_size_val):
                                skl[d] += coeff * cv_data[d]
                skl_all.append((k, l, skl))

        if not self.m_is_rat:
            result = []
            for k, l, skl in skl_all:
                result.append(Vector(
                    skl[0],
                    skl[1] if self.m_dim > 1 else 0,
                    skl[2] if self.m_dim > 2 else 0
                ))
            return result

        # Rational: proper quotient rule (NURBS Book A4.2)
        w00 = skl_all[0][2][self.m_dim]
        if abs(w00) < 1e-14:
            return [Vector(0, 0, 0)] * len(skl_all)
        dim = self.m_dim
        pt = Vector(skl_all[0][2][0] / w00,
                    skl_all[0][2][1] / w00 if dim > 1 else 0,
                    skl_all[0][2][2] / w00 if dim > 2 else 0)
        result = [pt]

        # Build lookup for weight derivatives
        wders = {}
        for k, l, skl in skl_all:
            wders[(k, l)] = skl[dim]

        # Cartesian derivatives lookup
        aders = {(0, 0): pt}
        for k, l, skl in skl_all[1:]:
            a = [skl[0], skl[1] if dim > 1 else 0, skl[2] if dim > 2 else 0]
            for i in range(1, k + 1):
                from math import comb
                prev = aders.get((k - i, l))
                if prev is not None:
                    c = comb(k, i) * wders.get((i, 0), 0)
                    a[0] -= c * prev[0]
                    a[1] -= c * prev[1]
                    a[2] -= c * prev[2]
            for j in range(1, l + 1):
                from math import comb
                prev = aders.get((k, l - j))
                if prev is not None:
                    c = comb(l, j) * wders.get((0, j), 0)
                    a[0] -= c * prev[0]
                    a[1] -= c * prev[1]
                    a[2] -= c * prev[2]
            v = Vector(a[0] / w00, a[1] / w00, a[2] / w00)
            aders[(k, l)] = v
            result.append(v)

        return result
    
    ###########################################################################
    # TRANSFORMATION
    ###########################################################################
    
    def transform(self, xform: Optional[Xform] = None) -> bool:
        """Apply transformation to surface (in-place).
        
        Parameters
        ----------
        xform : Xform, optional
            Transformation to apply. If None, uses self.xform.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if xform is None:
            xform = self.xform
        
        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                pt = self.get_cv(i, j)
                if pt is not None:
                    # transform_point modifies in-place
                    xform.transform_point(pt)
                    self.set_cv(i, j, pt)
        
        return True
    
    def transformed(self, xform: Optional[Xform] = None) -> 'NurbsSurface':
        """Get transformed copy of surface.
        
        Parameters
        ----------
        xform : Xform, optional
            Transformation to apply. If None, uses self.xform.
        
        Returns
        -------
        NurbsSurface
            Transformed copy.
        """
        copy = NurbsSurface()
        copy.m_dim = self.m_dim
        copy.m_is_rat = self.m_is_rat
        copy.m_order = self.m_order.copy()
        copy.m_cv_count = self.m_cv_count.copy()
        copy.m_cv_stride = self.m_cv_stride.copy()
        copy.m_knot = [self.m_knot[0].copy(), self.m_knot[1].copy()]
        copy.m_cv = self.m_cv.copy()
        copy.guid = self.guid
        copy.name = self.name
        copy.width = self.width
        copy.pointcolors = list(self.pointcolors)
        copy.facecolors = list(self.facecolors)
        copy.linecolors = list(self.linecolors)
        copy.xform = self.xform
        
        copy.transform(xform)
        return copy
    
    ###########################################################################
    # MODIFICATION OPERATIONS
    ###########################################################################
    
    def reverse(self, dir: int) -> bool:
        """Reverse surface direction.
        
        Parameters
        ----------
        dir : int
            Direction to reverse (0 for u, 1 for v).
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if dir < 0 or dir >= 2:
            return False
        if not self.is_valid():
            return False
        
        # Reverse knot vector using knot module function
        knot.reverse(self.m_order[dir], self.m_cv_count[dir], self.m_knot[dir])
        
        # Reverse control points in specified direction
        if dir == 0:
            # Reverse u direction (reverse rows)
            for i in range(self.m_cv_count[0] // 2):
                for j in range(self.m_cv_count[1]):
                    # Swap CVs
                    cv1 = self.get_cv(i, j)
                    cv2 = self.get_cv(self.m_cv_count[0] - 1 - i, j)
                    self.set_cv(i, j, cv2)
                    self.set_cv(self.m_cv_count[0] - 1 - i, j, cv1)
                    
                    # Swap weights if rational
                    if self.m_is_rat:
                        w1 = self.weight(i, j)
                        w2 = self.weight(self.m_cv_count[0] - 1 - i, j)
                        self.set_weight(i, j, w2)
                        self.set_weight(self.m_cv_count[0] - 1 - i, j, w1)
        else:
            # Reverse v direction (reverse columns)
            for i in range(self.m_cv_count[0]):
                for j in range(self.m_cv_count[1] // 2):
                    # Swap CVs
                    cv1 = self.get_cv(i, j)
                    cv2 = self.get_cv(i, self.m_cv_count[1] - 1 - j)
                    self.set_cv(i, j, cv2)
                    self.set_cv(i, self.m_cv_count[1] - 1 - j, cv1)
                    
                    # Swap weights if rational
                    if self.m_is_rat:
                        w1 = self.weight(i, j)
                        w2 = self.weight(i, self.m_cv_count[1] - 1 - j)
                        self.set_weight(i, j, w2)
                        self.set_weight(i, self.m_cv_count[1] - 1 - j, w1)
        
        return True
    
    def transpose(self) -> bool:
        """Transpose surface (swap u and v parameters).
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if not self.is_valid():
            return False
        
        # Swap orders and counts
        self.m_order[0], self.m_order[1] = self.m_order[1], self.m_order[0]
        self.m_cv_count[0], self.m_cv_count[1] = self.m_cv_count[1], self.m_cv_count[0]

        # Swap knot vectors
        self.m_knot[0], self.m_knot[1] = self.m_knot[1], self.m_knot[0]

        # Rebuild CV array with transposed indices
        cv_size_val = self.cv_size()
        new_cv = np.zeros(len(self.m_cv))
        
        for i in range(self.m_cv_count[1]):
            for j in range(self.m_cv_count[0]):
                old_index = i * self.m_cv_stride[0] + j * self.m_cv_stride[1]
                new_index = j * cv_size_val * self.m_cv_count[1] + i * cv_size_val
                new_cv[new_index:new_index + cv_size_val] = self.m_cv[old_index:old_index + cv_size_val]
        
        self.m_cv = new_cv
        
        # Update strides
        self.m_cv_stride[1] = cv_size_val
        self.m_cv_stride[0] = cv_size_val * self.m_cv_count[1]
        
        return True
    
    def swap_coordinates(self, axis_i: int, axis_j: int) -> bool:
        """Swap two coordinate axes.
        
        Parameters
        ----------
        axis_i : int
            First axis (0=x, 1=y, 2=z).
        axis_j : int
            Second axis (0=x, 1=y, 2=z).
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if axis_i < 0 or axis_i >= self.m_dim or axis_j < 0 or axis_j >= self.m_dim:
            return False
        if axis_i == axis_j:
            return True
        
        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                cv_ptr = self.cv(i, j)
                if cv_ptr is not None:
                    cv_ptr[axis_i], cv_ptr[axis_j] = cv_ptr[axis_j], cv_ptr[axis_i]
        
        return True
    
    def make_rational(self) -> bool:
        """Make surface rational (if not already).
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if self.m_is_rat:
            return True
        
        old_cv_size = self.m_dim
        new_cv_size = self.m_dim + 1
        new_cv = np.zeros(self.m_cv_count[0] * self.m_cv_count[1] * new_cv_size)
        
        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                old_index = i * (old_cv_size * self.m_cv_count[1]) + j * old_cv_size
                new_index = i * (new_cv_size * self.m_cv_count[1]) + j * new_cv_size
                
                # Copy coordinates
                new_cv[new_index:new_index + self.m_dim] = self.m_cv[old_index:old_index + self.m_dim]
                # Set weight to 1.0
                new_cv[new_index + self.m_dim] = 1.0
        
        self.m_cv = new_cv
        self.m_is_rat = 1
        self.m_cv_stride[1] = new_cv_size
        self.m_cv_stride[0] = new_cv_size * self.m_cv_count[1]

        return True

    def make_non_rational(self) -> bool:
        """Convert surface to non-rational (OpenNURBS implementation).
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if not self.m_is_rat:
            return True
        
        # OpenNURBS algorithm: iterate through CVs, divide by weight, pack tightly
        if self.m_order[0] > 0 and self.m_order[1] > 0 and self.m_dim > 0:
            new_cv = np.zeros(self.m_cv_count[0] * self.m_cv_count[1] * self.m_dim, dtype=np.float64)
            new_idx = 0
            
            # Process in optimal order based on stride
            if self.m_cv_stride[0] < self.m_cv_stride[1]:
                # Iterate j (outer), then i (inner)
                for j in range(self.m_cv_count[1]):
                    for i in range(self.m_cv_count[0]):
                        cv_ptr = self.cv(i, j)
                        if cv_ptr is not None and len(cv_ptr) > self.m_dim:
                            w = cv_ptr[self.m_dim]
                            w = 1.0 / w if abs(w) > 1e-14 else 1.0
                            for d in range(self.m_dim):
                                new_cv[new_idx] = w * cv_ptr[d]
                                new_idx += 1
            else:
                # Iterate i (outer), then j (inner)
                for i in range(self.m_cv_count[0]):
                    for j in range(self.m_cv_count[1]):
                        cv_ptr = self.cv(i, j)
                        if cv_ptr is not None and len(cv_ptr) > self.m_dim:
                            w = cv_ptr[self.m_dim]
                            w = 1.0 / w if abs(w) > 1e-14 else 1.0
                            for d in range(self.m_dim):
                                new_cv[new_idx] = w * cv_ptr[d]
                                new_idx += 1
            
            # Update strides for non-rational layout
            self.m_is_rat = 0
            if self.m_cv_stride[0] < self.m_cv_stride[1]:
                self.m_cv_stride[0] = self.m_dim
                self.m_cv_stride[1] = self.m_dim * self.m_cv_count[0]
            else:
                self.m_cv_stride[1] = self.m_dim
                self.m_cv_stride[0] = self.m_dim * self.m_cv_count[1]
            
            self.m_cv = new_cv
        
        return not self.is_rational()
    
    ###########################################################################
    # GEOMETRIC OPERATIONS
    ###########################################################################
    
    def get_bounding_box(self) -> Obb:
        """Get bounding box of surface.

        Returns
        -------
        Obb
            Bounding box containing all control points.
        """
        if not self.is_valid() or self.m_cv_count[0] == 0 or self.m_cv_count[1] == 0:
            return Obb()

        min_pt = self.get_cv(0, 0)
        max_pt = Point(min_pt.x, min_pt.y, min_pt.z)

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                pt = self.get_cv(i, j)
                min_pt = Point(min(min_pt.x, pt.x),
                              min(min_pt.y, pt.y),
                              min(min_pt.z, pt.z))
                max_pt = Point(max(max_pt.x, pt.x),
                              max(max_pt.y, pt.y),
                              max(max_pt.z, pt.z))

        center = Point((min_pt.x + max_pt.x) / 2.0,
                      (min_pt.y + max_pt.y) / 2.0,
                      (min_pt.z + max_pt.z) / 2.0)
        half_size = Vector((max_pt.x - min_pt.x) / 2.0,
                          (max_pt.y - min_pt.y) / 2.0,
                          (max_pt.z - min_pt.z) / 2.0)

        return Obb(center, Vector.x_axis(), Vector.y_axis(), Vector.z_axis(), half_size)
    
    def divide_by_count(self, nu: int, nv: int):
        u0, u1 = self.domain(0)
        v0, v1 = self.domain(1)

        grid = []
        params = []
        for i in range(nu + 1):
            row = []
            param_row = []
            u = u0 + (u1 - u0) * (i / nu) if nu > 0 else u0
            for j in range(nv + 1):
                v = v0 + (v1 - v0) * (j / nv) if nv > 0 else v0
                row.append(self.point_at(u, v))
                param_row.append((u, v))
            grid.append(row)
            params.append(param_row)

        return grid, params

    def divide_by_count_points(self, nu: int, nv: int):
        if not self.is_valid():
            return [], [], []

        u0, u1 = self.domain(0)
        v0, v1 = self.domain(1)

        grid = []
        grid_vector = []
        params = []
        for i in range(nu + 1):
            row = []
            row_vector = []
            param_row = []
            u = u0 + (u1 - u0) * (i / nu) if nu > 0 else u0
            for j in range(nv + 1):
                v = v0 + (v1 - v0) * (j / nv) if nv > 0 else v0
                row.append(self.point_at(u, v))
                row_vector.append(self.normal_at(u, v))
                param_row.append((u, v))
            grid.append(row)
            grid_vector.append(row_vector)
            params.append(param_row)

        return grid, grid_vector, params

    def divide_by_count_planes(self, nu: int, nv: int):
        if not self.is_valid():
            return [], []

        u0, u1 = self.domain(0)
        v0, v1 = self.domain(1)

        grid = []
        params = []
        for i in range(nu + 1):
            row = []
            param_row = []
            u = u0 + (u1 - u0) * (i / nu) if nu > 0 else u0
            for j in range(nv + 1):
                v = v0 + (v1 - v0) * (j / nv) if nv > 0 else v0
                origin = self.point_at(u, v)
                derivs = self.evaluate(u, v, 1)
                su = derivs[2]
                sv = derivs[1]
                x_axis = su.duplicate()
                if x_axis.magnitude() > 1e-14:
                    x_axis.normalize_self()
                y_axis = sv.duplicate()
                if y_axis.magnitude() > 1e-14:
                    y_axis.normalize_self()
                n = self.normal_at(u, v)
                plane = Plane()
                plane._origin = origin
                plane._x_axis = x_axis
                plane._y_axis = y_axis
                plane._z_axis = n
                plane._update_equation()
                row.append(plane)
                param_row.append((u, v))
            grid.append(row)
            params.append(param_row)

        return grid, params

    def _compute_bbox_diagonal(self):
        import math
        minx = miny = minz = 1e30
        maxx = maxy = maxz = -1e30
        for i in range(self.cv_count(0)):
            for j in range(self.cv_count(1)):
                p = self.get_cv(i, j)
                if p[0] < minx: minx = p[0]
                if p[1] < miny: miny = p[1]
                if p[2] < minz: minz = p[2]
                if p[0] > maxx: maxx = p[0]
                if p[1] > maxy: maxy = p[1]
                if p[2] > maxz: maxz = p[2]
        dx, dy, dz = maxx-minx, maxy-miny, maxz-minz
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _span_subs(self, dir, sp, osp, max_angle_deg, bbox_diag):
        import math
        n = len(sp) - 1
        n_other = len(osp) - 1
        subs = [1] * n
        deg_u = self.degree(0)
        deg_v = self.degree(1)
        degree_dir = deg_u if dir == 0 else deg_v
        s_positions = [(osp[k] + osp[k + 1]) * 0.5 for k in range(n_other)]
        for i in range(n):
            t0, t1 = sp[i], sp[i + 1]
            if degree_dir > 1:
                max_angle = 0.0
                for s in s_positions:
                    prev_n = None
                    total_angle = 0.0
                    for k in range(5):
                        t = t0 + k * (t1 - t0) / 4.0
                        if dir == 0:
                            nv = self.normal_at(t, s)
                        else:
                            nv = self.normal_at(s, t)
                        if prev_n is not None:
                            dot = prev_n[0]*nv[0] + prev_n[1]*nv[1] + prev_n[2]*nv[2]
                            dot = max(-1.0, min(1.0, dot))
                            total_angle += math.acos(dot) * 180.0 / PI
                        prev_n = nv
                    if total_angle > max_angle:
                        max_angle = total_angle
                subs[i] = max(1, min(int(math.ceil(max_angle / max_angle_deg)), 24))
            chord_tol = bbox_diag * 0.005
            max_dev = 0.0
            nc = min(n_other, 3)
            for ci in range(nc + 1):
                sv = osp[0] + ci * (osp[-1] - osp[0]) / max(nc, 1)
                if dir == 0:
                    pa = self.point_at(t0, sv)
                    pb = self.point_at(t1, sv)
                else:
                    pa = self.point_at(sv, t0)
                    pb = self.point_at(sv, t1)
                for k in range(1, 4):
                    frac = k / 4.0
                    tm = t0 + frac * (t1 - t0)
                    if dir == 0:
                        pm = self.point_at(tm, sv)
                    else:
                        pm = self.point_at(sv, tm)
                    lx = pa[0] + frac * (pb[0] - pa[0])
                    ly = pa[1] + frac * (pb[1] - pa[1])
                    lz = pa[2] + frac * (pb[2] - pa[2])
                    dx, dy, dz = pm[0] - lx, pm[1] - ly, pm[2] - lz
                    dev = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if dev > max_dev:
                        max_dev = dev
            if max_dev > chord_tol:
                chord_subs = max(2, int(math.ceil(math.sqrt(max_dev / chord_tol))))
                subs[i] = max(subs[i], min(chord_subs, 24))
            if degree_dir > 1:
                subs[i] = max(subs[i], 2)
        return subs

    def mesh(self):
        if self.m_mesh is not None:
            return self.m_mesh
        import math
        from .mesh import Mesh
        from .vector import Vector
        usp = self.get_span_vector(0)
        vsp = self.get_span_vector(1)
        if len(usp) < 2 or len(vsp) < 2:
            return Mesh()
        if self.is_planar(tolerance=1e-6):
            result = Mesh()
            p00 = self.point_at_corner(0, 0)
            p10 = self.point_at_corner(1, 0)
            p11 = self.point_at_corner(1, 1)
            p01 = self.point_at_corner(0, 1)
            d2 = (p00[0]-p01[0])**2 + (p00[1]-p01[1])**2 + (p00[2]-p01[2])**2
            if d2 < 1e-20:
                v0 = result.add_vertex(p00)
                v1 = result.add_vertex(p10)
                v2 = result.add_vertex(p11)
                result.add_face([v0, v1, v2])
                e1 = Vector(p10[0]-p00[0], p10[1]-p00[1], p10[2]-p00[2])
                e2 = Vector(p11[0]-p00[0], p11[1]-p00[1], p11[2]-p00[2])
                normal = e1.cross(e2)
            else:
                v0 = result.add_vertex(p00)
                v1 = result.add_vertex(p10)
                v2 = result.add_vertex(p11)
                v3 = result.add_vertex(p01)
                result.add_face([v0, v1, v2])
                result.add_face([v0, v2, v3])
                derivs = self.evaluate(0.5, 0.5, 1)
                normal = derivs[1].cross(derivs[2]) if len(derivs) >= 3 else Vector(0, 0, 1)
            nlen = normal.magnitude()
            n = normal * (1.0 / nlen) if nlen > 1e-15 else normal
            for vkey in result.vertex:
                result.vertex[vkey].set_normal(n[0], n[1], n[2])
            self.m_mesh = result
            return result
        ns_u = len(usp) - 1
        ns_v = len(vsp) - 1
        max_angle_deg = 20.0
        bbox_diag = self._compute_bbox_diagonal()
        deg_u = self.degree(0)
        deg_v = self.degree(1)
        u_subs = self._span_subs(0, usp, vsp, max_angle_deg, bbox_diag)
        v_subs = self._span_subs(1, vsp, usp, max_angle_deg, bbox_diag)
        total_u = sum(u_subs) + 1
        total_v = sum(v_subs) + 1
        v_mid = (vsp[0] + vsp[-1]) * 0.5
        u_mid = (usp[0] + usp[-1]) * 0.5
        u_len = 0.0
        p0 = self.point_at(usp[0], v_mid)
        n_sample = max(total_u, 10)
        for i in range(1, n_sample + 1):
            u = usp[0] + i * (usp[-1] - usp[0]) / n_sample
            p1 = self.point_at(u, v_mid)
            u_len += math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2 + (p1[2]-p0[2])**2)
            p0 = p1
        v_len = 0.0
        p0 = self.point_at(u_mid, vsp[0])
        n_sample = max(total_v, 10)
        for i in range(1, n_sample + 1):
            v = vsp[0] + i * (vsp[-1] - vsp[0]) / n_sample
            p1 = self.point_at(u_mid, v)
            v_len += math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2 + (p1[2]-p0[2])**2)
            p0 = p1
        if u_len > 1e-14 and v_len > 1e-14 and total_u > 0 and total_v > 0:
            spacing_u = u_len / total_u
            spacing_v = v_len / total_v
            ratio = spacing_u / spacing_v
            if ratio > 2.0 and deg_u > 1:
                scale = math.sqrt(ratio)
                u_subs = [min(int(math.ceil(s * scale)), 24) for s in u_subs]
            elif ratio < 0.5 and deg_v > 1:
                scale = math.sqrt(1.0 / ratio)
                v_subs = [min(int(math.ceil(s * scale)), 24) for s in v_subs]
        if deg_u == 1 and deg_v == 1:
            max_twist = 0.0
            chord_tol = bbox_diag * 0.005 if bbox_diag > 0 else 1e-6
            for i in range(ns_u):
                for j in range(ns_v):
                    u0, u1 = usp[i], usp[i + 1]
                    v0, v1 = vsp[j], vsp[j + 1]
                    pm = self.point_at((u0 + u1) * 0.5, (v0 + v1) * 0.5)
                    p00 = self.point_at(u0, v0)
                    p11 = self.point_at(u1, v1)
                    mx = (p00[0] + p11[0]) * 0.5
                    my = (p00[1] + p11[1]) * 0.5
                    mz = (p00[2] + p11[2]) * 0.5
                    dx, dy, dz = pm[0] - mx, pm[1] - my, pm[2] - mz
                    twist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if twist > max_twist:
                        max_twist = twist
            if max_twist > chord_tol:
                twist_subs = max(4, min(int(math.ceil(2.0 * math.sqrt(max_twist / chord_tol))), 24))
                for i in range(len(u_subs)):
                    u_subs[i] = max(u_subs[i], twist_subs)
                for i in range(len(v_subs)):
                    v_subs[i] = max(v_subs[i], twist_subs)
        us = []
        for i in range(len(usp) - 1):
            for s in range(u_subs[i]):
                us.append(usp[i] + s * (usp[i + 1] - usp[i]) / u_subs[i])
        us.append(usp[-1])
        vs = []
        for i in range(len(vsp) - 1):
            for s in range(v_subs[i]):
                vs.append(vsp[i] + s * (vsp[i + 1] - vsp[i]) / v_subs[i])
        vs.append(vsp[-1])
        closed_u = self.is_closed(0)
        closed_v = self.is_closed(1)
        def fix_closed_gap(params, spans, closed):
            if not closed or len(params) < 3:
                return
            params.pop()
            domain_end = spans[-1]
            wrap_gap = domain_end - params[-1]
            max_gap = max((params[i] - params[i-1] for i in range(1, len(params))), default=0)
            if max_gap > 0 and wrap_gap > max_gap * 1.5:
                extra = int(math.ceil(wrap_gap / max_gap)) - 1
                step = wrap_gap / (extra + 1)
                for e in range(1, extra + 1):
                    params.append(params[-1] + step)
        fix_closed_gap(us, usp, closed_u)
        fix_closed_gap(vs, vsp, closed_v)
        nu, nv = len(us), len(vs)
        sing_v0 = self.is_singular(0)
        sing_v1 = self.is_singular(2)
        j_start = 1 if sing_v0 else 0
        j_end = nv - 1 if sing_v1 else nv
        nv_grid = j_end - j_start
        result = Mesh()
        south_pole = 0
        north_pole = 0
        if sing_v0:
            south_pole = result.add_vertex(self.point_at(us[0], vs[0]))
            result.vertex[south_pole].attributes["u"] = us[0]
            result.vertex[south_pole].attributes["v"] = vs[0]
        if sing_v1:
            north_pole = result.add_vertex(self.point_at(us[0], vs[nv - 1]))
            result.vertex[north_pole].attributes["u"] = us[0]
            result.vertex[north_pole].attributes["v"] = vs[nv - 1]
        vkeys = []
        for i in range(nu):
            for j in range(j_start, j_end):
                vk = result.add_vertex(self.point_at(us[i], vs[j]))
                result.vertex[vk].attributes["u"] = us[i]
                result.vertex[vk].attributes["v"] = vs[j]
                vkeys.append(vk)
        def grid_idx(i, j):
            return vkeys[i * nv_grid + (j - j_start)]
        nu_faces = nu if closed_u else nu - 1
        if sing_v0:
            for i in range(nu_faces):
                i1 = (i + 1) % nu
                result.add_face([south_pole, grid_idx(i1, j_start), grid_idx(i, j_start)])
        nv_interior = nv_grid - 1
        if closed_v and not sing_v0 and not sing_v1:
            nv_interior = nv_grid
        for i in range(nu_faces):
            for jj in range(nv_interior):
                j = jj + j_start
                i1 = (i + 1) % nu
                j1 = ((jj + 1) % nv_grid + j_start) if (closed_v and not sing_v0 and not sing_v1) else (j + 1)
                v00, v10 = grid_idx(i, j), grid_idx(i1, j)
                v01, v11 = grid_idx(i, j1), grid_idx(i1, j1)
                if (i + jj) % 2 == 0:
                    result.add_face([v00, v10, v11])
                    result.add_face([v00, v11, v01])
                else:
                    result.add_face([v00, v10, v01])
                    result.add_face([v10, v11, v01])
        if sing_v1:
            j_last = j_end - 1
            for i in range(nu_faces):
                i1 = (i + 1) % nu
                result.add_face([grid_idx(i, j_last), grid_idx(i1, j_last), north_pole])
        max_vkey = max(result.vertex.keys()) if result.vertex else 0
        vnx = [0.0] * (max_vkey + 1)
        vny = [0.0] * (max_vkey + 1)
        vnz = [0.0] * (max_vkey + 1)
        for fi, vids in result.face.items():
            if len(vids) < 3:
                continue
            p0 = result.vertex[vids[0]]
            p1 = result.vertex[vids[1]]
            p2 = result.vertex[vids[2]]
            e1x, e1y, e1z = p1.x-p0.x, p1.y-p0.y, p1.z-p0.z
            e2x, e2y, e2z = p2.x-p0.x, p2.y-p0.y, p2.z-p0.z
            fnx = e1y*e2z - e1z*e2y
            fny = e1z*e2x - e1x*e2z
            fnz = e1x*e2y - e1y*e2x
            for vi in vids:
                vnx[vi] += fnx
                vny[vi] += fny
                vnz[vi] += fnz
        for vk in result.vertex:
            ln = math.sqrt(vnx[vk]**2 + vny[vk]**2 + vnz[vk]**2)
            if ln > 1e-15:
                vnx[vk] /= ln
                vny[vk] /= ln
                vnz[vk] /= ln
            result.vertex[vk].set_normal(vnx[vk], vny[vk], vnz[vk])
        self.m_mesh = result
        return self.m_mesh

    def mesh_adaptive(self, max_angle: float = 20.0, max_edge_length: float = 0.0,
                      min_edge_length: float = 0.0, max_chord_height: float = 0.0):
        if self.m_mesh is not None:
            return self.m_mesh
        if not self.is_valid():
            from .mesh import Mesh
            return Mesh()
        from .remesh_nurbssurface_adaptive import RemeshNurbssurfaceAdaptive
        mesher = RemeshNurbssurfaceAdaptive(self)
        mesher.set_max_angle(max_angle)
        mesher.set_max_edge_length(max_edge_length)
        mesher.set_min_edge_length(min_edge_length)
        mesher.set_max_chord_height(max_chord_height)
        self.m_mesh = mesher.mesh()
        return self.m_mesh

    ###########################################################################
    # JSON SERIALIZATION
    ###########################################################################
    
    def jsondump(self) -> dict:
        """Convert to JSON dictionary."""
        return self.__jsondump__()
    
    @staticmethod
    def jsonload(data: dict) -> 'NurbsSurface':
        """Load from JSON dictionary."""
        return NurbsSurface.__jsonload__(data)
    
    ###########################################################################
    # STRING REPRESENTATION
    ###########################################################################
    
    def to_string(self) -> str:
        """Get string representation.
        
        Returns
        -------
        str
            String representation of surface.
        """
        return (f"NurbsSurface(name={self.name}, "
                f"degree=({self.degree(0)},{self.degree(1)}), "
                f"cvs=({self.m_cv_count[0]},{self.m_cv_count[1]}))")
    
    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        result = (f"NurbsSurface(\n  name={self.name},\n"
                  f"  degree=({self.degree(0)},{self.degree(1)}),\n"
                  f"  cvs=({self.m_cv_count[0]},{self.m_cv_count[1]}),\n"
                  f"  rational={'true' if self.m_is_rat else 'false'},\n"
                  f"  control_points=[\n")
        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                p = self.get_cv(i, j)
                result += f"    {p[0]:g}, {p[1]:g}, {p[2]:g}\n"
        result += "  ]\n)"
        return result

    @staticmethod
    def create_ruled(curveA, curveB):
        from .primitives import Primitives
        return Primitives.create_ruled(curveA, curveB)

    @staticmethod
    def create_loft(input_curves, degree_v=3):
        from .primitives import Primitives
        return Primitives.create_loft(input_curves, degree_v)

    @staticmethod
    def _merge_knot_vectors(a, b, tol=1e-10):
        from .primitives import Primitives
        return Primitives._merge_knot_vectors(a, b, tol)

    @staticmethod
    def _knot_vectors_equal(a, b, tol=1e-10):
        from .primitives import Primitives
        return Primitives._knot_vectors_equal(a, b, tol)

    @staticmethod
    def _make_curves_compatible(curves):
        from .primitives import Primitives
        Primitives._make_curves_compatible(curves)

    @staticmethod
    def create_planar(curves):
        from .primitives import Primitives
        if isinstance(curves, list):
            if len(curves) == 1:
                return Primitives.create_planar(curves[0])
            if not curves:
                return NurbsSurface()
            return Primitives.create_planar(curves[0])
        return Primitives.create_planar(curves)

    ###########################################################################
    # ADDITIONAL CREATION METHODS
    ###########################################################################
    
    def create_clamped_uniform(self, dimension: int, order0: int, order1: int,
                              cv_count0: int, cv_count1: int,
                              knot_delta0: float = 1.0, knot_delta1: float = 1.0) -> bool:
        """Create clamped uniform NURBS surface.
        
        Parameters
        ----------
        dimension : int
            Dimension of the surface.
        order0, order1 : int
            Orders in u and v directions.
        cv_count0, cv_count1 : int
            Number of CVs in u and v directions.
        knot_delta0, knot_delta1 : float, optional
            Knot spacing.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if not self._create_impl(dimension, False, order0, order1, cv_count0, cv_count1):
            return False
        
        self.make_clamped_uniform_knot_vector(0, knot_delta0)
        self.make_clamped_uniform_knot_vector(1, knot_delta1)
        
        return True
    
    ###########################################################################
    # ADDITIONAL ACCESSORS
    ###########################################################################
    
    def cv_count_total(self) -> int:
        """Get total number of control vertices.
        
        Returns
        -------
        int
            Total number of CVs.
        """
        return self.m_cv_count[0] * self.m_cv_count[1]
    
    ###########################################################################
    # KNOT VECTOR OPERATIONS (ADDITIONAL)
    ###########################################################################
    
    def _to_curve_internal(self, dir: int):
        """Pack surface into a high-dimensional curve along dir."""
        dim = self.m_dim
        if dir == 0:
            n_along = self.cv_count(0)
            n_other = self.cv_count(1)
        else:
            n_along = self.cv_count(1)
            n_other = self.cv_count(0)
        hdim = dim * n_other
        crv = NurbsCurve(hdim, False, self.order(dir), n_along)
        for k in range(self.knot_count(dir)):
            crv.set_knot(k, self.m_knot[dir][k])
        for i in range(n_along):
            cv_data = []
            for j in range(n_other):
                if dir == 0:
                    p = self.get_cv(i, j)
                else:
                    p = self.get_cv(j, i)
                cv_data.extend([p[0], p[1], p[2]])
            for d in range(hdim):
                crv.m_cv[i * crv.m_cv_stride + d] = cv_data[d]
        return crv

    def _from_curve_internal(self, crv, dir: int):
        """Unpack high-dimensional curve back into surface."""
        dim = self.m_dim
        if dir == 0:
            n_other = self.cv_count(1)
            new_n_along = crv.cv_count()
        else:
            n_other = self.cv_count(0)
            new_n_along = crv.cv_count()
        new_order = crv.order()
        if dir == 0:
            new_srf = NurbsSurface.create_raw(dim, False, new_order, self.order(1),
                                               new_n_along, self.cv_count(1))
            for k in range(crv.knot_count()):
                new_srf.set_knot(0, k, crv.knot(k))
            for k in range(self.knot_count(1)):
                new_srf.set_knot(1, k, self.m_knot[1][k])
        else:
            new_srf = NurbsSurface.create_raw(dim, False, self.order(0), new_order,
                                               self.cv_count(0), new_n_along)
            for k in range(self.knot_count(0)):
                new_srf.set_knot(0, k, self.m_knot[0][k])
            for k in range(crv.knot_count()):
                new_srf.set_knot(1, k, crv.knot(k))
        for i in range(new_n_along):
            for j in range(n_other):
                base = i * crv.m_cv_stride + j * dim
                x = crv.m_cv[base]
                y = crv.m_cv[base + 1]
                z = crv.m_cv[base + 2]
                if dir == 0:
                    new_srf.set_cv(i, j, Point(x, y, z))
                else:
                    new_srf.set_cv(j, i, Point(x, y, z))
        self.m_order = new_srf.m_order
        self.m_cv_count = new_srf.m_cv_count
        self.m_knot = new_srf.m_knot
        self.m_cv = new_srf.m_cv
        self.m_cv_stride = new_srf.m_cv_stride
        return True

    def insert_knot(self, dir: int, knot_value: float, knot_multiplicity: int = 1) -> bool:
        if dir < 0 or dir > 1:
            return False
        crv = self._to_curve_internal(dir)
        if crv is None:
            return False
        for _ in range(knot_multiplicity):
            if not crv.insert_knot(knot_value, 1):
                return False
        return self._from_curve_internal(crv, dir)
    
    ###########################################################################
    # MODIFICATION OPERATIONS (ADDITIONAL)
    ###########################################################################
    
    def trim(self, dir: int, domain: Tuple[float, float]) -> bool:
        """Trim surface to sub-domain in specified direction.

        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        domain : tuple
            (start, end) domain.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if dir < 0 or dir > 1 or not self.is_valid():
            return False
        crv = self._to_curve_internal(dir)
        if crv is None:
            return False
        if not crv.trim(domain[0], domain[1]):
            return False
        return self._from_curve_internal(crv, dir)
    
    def split(self, dir: int, c: float) -> Tuple[Optional['NurbsSurface'], Optional['NurbsSurface']]:
        """Split surface at parameter in specified direction.

        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        c : float
            Parameter value to split at.

        Returns
        -------
        tuple
            (west_or_south_side, east_or_north_side) or (None, None) on failure.
        """
        import copy
        if dir < 0 or dir > 1 or not self.is_valid():
            return (None, None)
        t0, t1 = self.domain(dir)
        if c <= t0 or c >= t1:
            return (None, None)
        lo = copy.deepcopy(self)
        hi = copy.deepcopy(self)
        if not lo.trim(dir, (t0, c)) or not hi.trim(dir, (c, t1)):
            return (None, None)
        return (lo, hi)
    
    def clamp_end(self, dir: int, end: int) -> bool:
        """Clamp knot vector end(s) (OpenNURBS implementation).
        
        Sets initial/final (order-2) knot values to match knot[order-2]/knot[cv_count-1].
        
        Parameters
        ----------
        dir : int
            Direction (0 or 1).
        end : int
            Which end to clamp (0=start, 1=end, 2=both).
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if dir < 0 or dir > 1:
            return False
        if not self.is_valid():
            return False
        
        # Use knot module function
        return knot.clamp(self.m_order[dir], self.m_cv_count[dir], self.m_knot[dir], end)
    
    def increase_degree(self, dir: int, desired_degree: int) -> bool:
        if dir < 0 or dir > 1:
            return False
        if desired_degree < self.degree(dir):
            return False
        if desired_degree == self.degree(dir):
            return True
        crv = self._to_curve_internal(dir)
        if crv is None:
            return False
        if not crv.increase_degree(desired_degree):
            return False
        return self._from_curve_internal(crv, dir)
    
    ###########################################################################
    # TRANSFORMATION (OVERLOADS)
    ###########################################################################
    
    def transform_stored(self) -> bool:
        """Apply stored xform transformation (in-place).
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        return self.transform(self.xform)
    
    def transformed_stored(self) -> 'NurbsSurface':
        """Get transformed copy using stored xform.
        
        Returns
        -------
        NurbsSurface
            Transformed copy.
        """
        return self.transformed(self.xform)
    
    ###########################################################################
    # GEOMETRIC OPERATIONS (ADDITIONAL)
    ###########################################################################
    
    def iso_curve(self, dir: int, c: float) -> Optional['NurbsCurve']:
        """Get isoparametric curve at parameter.
        
        Parameters
        ----------
        dir : int
            Direction (0=iso-u curve where v varies, 1=iso-v curve where u varies).
        c : float
            Parameter value.
        
        Returns
        -------
        NurbsCurve or None
            Isoparametric curve, or None on failure.
        """
        from .nurbscurve import NurbsCurve
        
        if (dir != 0 and dir != 1) or not self.is_valid():
            return None
        
        # Create output curve with proper initialization
        nurbs_crv = NurbsCurve()
        if not nurbs_crv.create_curve(self.m_dim, self.m_is_rat != 0, self.m_order[dir], self.m_cv_count[dir]):
            return None
        
        # Copy knot vector for varying direction
        for i in range(nurbs_crv.knot_count()):
            nurbs_crv.set_knot(i, self.knot(dir, i))
        
        # Find span in constant direction
        span_index = self._find_span(1 - dir, c)
        if span_index < 0:
            span_index = 0
        elif span_index > self.m_cv_count[1 - dir] - self.m_order[1 - dir]:
            span_index = self.m_cv_count[1 - dir] - self.m_order[1 - dir]
        
        # Compute basis functions in constant direction
        basis = self._basis_functions(1 - dir, span_index, c)
        
        # Evaluate CVs for isocurve
        for i in range(nurbs_crv.m_cv_count):
            cv_sum = np.zeros(self.cv_size())
            
            for k in range(self.m_order[1 - dir]):
                if dir == 0:
                    # iso-u: v varies, u is constant at c
                    cv_ptr = self.cv(span_index + k, i)
                else:
                    # iso-v: u varies, v is constant at c
                    cv_ptr = self.cv(i, span_index + k)
                
                if cv_ptr is not None:
                    cv_sum += basis[k] * cv_ptr
            
            # Set CV in curve
            if self.m_is_rat and abs(cv_sum[self.m_dim]) > 1e-14:
                w = cv_sum[self.m_dim]
                pt = Point(cv_sum[0]/w,
                          cv_sum[1]/w if self.m_dim > 1 else 0,
                          cv_sum[2]/w if self.m_dim > 2 else 0)
                nurbs_crv.set_cv(i, pt)
                if nurbs_crv.m_is_rat:
                    nurbs_crv.set_weight(i, w)
            else:
                pt = Point(cv_sum[0],
                          cv_sum[1] if self.m_dim > 1 else 0,
                          cv_sum[2] if self.m_dim > 2 else 0)
                nurbs_crv.set_cv(i, pt)
        
        return nurbs_crv

    ###########################################################################
    # ADVANCED OPERATIONS
    ###########################################################################
    
    def zero_cvs(self) -> bool:
        """Zero all control vertices (set weights to 1 if rational).
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                self.set_cv(i, j, Point(0, 0, 0))
                if self.m_is_rat:
                    self.set_weight(i, j, 1.0)
        return True
    
    ###########################################################################
    # JSON SERIALIZATION
    ###########################################################################
    
    def __jsondump__(self) -> dict:
        """Convert to JSON-serializable dict. Field order matches C++ ground truth (alphabetical)."""
        d = {
            'control_points': self.m_cv.tolist(),
            'cv_count_u': self.m_cv_count[0],
            'cv_count_v': self.m_cv_count[1],
            'dimension': self.m_dim,
            'facecolors': [v for c in self.facecolors for v in (c.r, c.g, c.b, c.a)],
            'guid': self.guid,
            'is_rational': bool(self.m_is_rat),
            'knots_u': self.m_knot[0].tolist(),
            'knots_v': self.m_knot[1].tolist(),
            'linecolors': [v for c in self.linecolors for v in (c.r, c.g, c.b, c.a)],
            'name': self.name,
            'order_u': self.m_order[0],
            'order_v': self.m_order[1],
            'pointcolors': [v for c in self.pointcolors for v in (c.r, c.g, c.b, c.a)],
            'type': 'NurbsSurface',
            'width': self.width,
            'xform': self.xform.__jsondump__(),
        }
        if self.m_mesh is not None:
            d['mesh'] = self.m_mesh.__jsondump__()
        return d
    
    @classmethod
    def __jsonload__(cls, data: dict, guid=None, name=None) -> 'NurbsSurface':
        """Create from JSON dict."""
        from .color import Color
        srf = cls()

        dimension = data.get('dimension', 3)
        is_rational = data.get('is_rational', False)
        order_u = data.get('order_u', 4)
        order_v = data.get('order_v', 4)
        cv_count_u = data.get('cv_count_u', 0)
        cv_count_v = data.get('cv_count_v', 0)

        if cv_count_u > 0 and cv_count_v > 0:
            srf._create_impl(dimension, is_rational, order_u, order_v, cv_count_u, cv_count_v)

            if 'knots_u' in data:
                srf.m_knot[0] = np.array(data['knots_u'], dtype=np.float64)
            if 'knots_v' in data:
                srf.m_knot[1] = np.array(data['knots_v'], dtype=np.float64)
            if 'control_points' in data:
                srf.m_cv = np.array(data['control_points'], dtype=np.float64)

        srf.guid = guid if guid is not None else data.get('guid', srf.guid)
        srf.name = name if name is not None else data.get('name', 'my_nurbssurface')
        srf.width = data.get('width', 1.0)

        if 'pointcolors' in data:
            arr = data['pointcolors']
            srf.pointcolors = [Color(arr[i], arr[i+1], arr[i+2], arr[i+3]) for i in range(0, len(arr) - 3, 4)]
        if 'facecolors' in data:
            arr = data['facecolors']
            srf.facecolors = [Color(arr[i], arr[i+1], arr[i+2], arr[i+3]) for i in range(0, len(arr) - 3, 4)]
        if 'linecolors' in data:
            arr = data['linecolors']
            srf.linecolors = [Color(arr[i], arr[i+1], arr[i+2], arr[i+3]) for i in range(0, len(arr) - 3, 4)]
        if 'xform' in data and data['xform'] is not None:
            from .xform import Xform
            srf.xform = Xform.__jsonload__(data['xform'])

        if data.get('mesh'):
            from .mesh import Mesh
            srf.m_mesh = Mesh.__jsonload__(data['mesh'])

        return srf
    
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
    def json_load(cls, filepath) -> 'NurbsSurface':
        """Read JSON from file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file.
        
        Returns
        -------
        NurbsSurface
            The deserialized NurbsSurface.
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

    ###########################################################################
    # PROTOBUF SERIALIZATION
    ###########################################################################

    def pb_dumps(self):
        """Convert to protobuf binary format.

        Returns
        -------
        bytes
            Serialized protobuf data.
        """
        from .proto import nurbssurface_pb2

        proto = nurbssurface_pb2.NurbsSurface()
        proto.guid = self.guid
        proto.name = self.name
        proto.dimension = self.m_dim
        proto.is_rational = bool(self.m_is_rat)
        proto.order_u = self.m_order[0]
        proto.order_v = self.m_order[1]
        proto.cv_count_u = self.m_cv_count[0]
        proto.cv_count_v = self.m_cv_count[1]
        proto.cv_stride_u = self.m_cv_stride[0]
        proto.cv_stride_v = self.m_cv_stride[1]

        # Knot vectors
        proto.knots_u.extend(self.m_knot[0].tolist())
        proto.knots_v.extend(self.m_knot[1].tolist())

        # Control vertices (flat array)
        proto.cvs.extend(self.m_cv.tolist())

        # Visual properties
        proto.width = self.width

        for c in self.pointcolors:
            cp = proto.pointcolors.add()
            cp.r = int(c.r); cp.g = int(c.g); cp.b = int(c.b); cp.a = int(c.a)
        for c in self.facecolors:
            cp = proto.facecolors.add()
            cp.r = int(c.r); cp.g = int(c.g); cp.b = int(c.b); cp.a = int(c.a)
        for c in self.linecolors:
            cp = proto.linecolors.add()
            cp.r = int(c.r); cp.g = int(c.g); cp.b = int(c.b); cp.a = int(c.a)

        # Transform
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)

        # Cached mesh
        if self.m_mesh is not None and self.m_mesh.number_of_vertices() > 0:
            mesh_data = self.m_mesh.pb_dumps()
            proto.cached_mesh.ParseFromString(mesh_data)

        return proto.SerializeToString()

    def pb_fill(self, proto):
        """Fill an existing NurbsSurface proto message directly (avoids serialize/deserialize cycle)."""
        proto.guid = self.guid
        proto.name = self.name
        proto.dimension = self.m_dim
        proto.is_rational = bool(self.m_is_rat)
        proto.order_u = self.m_order[0]
        proto.order_v = self.m_order[1]
        proto.cv_count_u = self.m_cv_count[0]
        proto.cv_count_v = self.m_cv_count[1]
        proto.cv_stride_u = self.m_cv_stride[0]
        proto.cv_stride_v = self.m_cv_stride[1]
        proto.knots_u.extend(self.m_knot[0].tolist())
        proto.knots_v.extend(self.m_knot[1].tolist())
        proto.cvs.extend(self.m_cv.tolist())
        proto.width = self.width
        for c in self.pointcolors:
            cp = proto.pointcolors.add()
            cp.r = int(c.r); cp.g = int(c.g); cp.b = int(c.b); cp.a = int(c.a)
        for c in self.facecolors:
            cp = proto.facecolors.add()
            cp.r = int(c.r); cp.g = int(c.g); cp.b = int(c.b); cp.a = int(c.a)
        for c in self.linecolors:
            cp = proto.linecolors.add()
            cp.r = int(c.r); cp.g = int(c.g); cp.b = int(c.b); cp.a = int(c.a)
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)
        if self.m_mesh is not None and self.m_mesh.number_of_vertices() > 0:
            proto.cached_mesh.ParseFromString(self.m_mesh.pb_dumps())

    @classmethod
    def pb_loads(cls, data):
        """Create NurbsSurface from protobuf binary data.

        Parameters
        ----------
        data : bytes
            Protobuf-encoded surface data.

        Returns
        -------
        NurbsSurface
            The deserialized NurbsSurface.
        """
        from .proto import nurbssurface_pb2
        from .color import Color
        from .xform import Xform
        import numpy as np

        proto = nurbssurface_pb2.NurbsSurface()
        proto.ParseFromString(data)

        # Create surface with correct dimensions
        surface = cls()
        surface._create_impl(
            proto.dimension,
            proto.is_rational,
            proto.order_u,
            proto.order_v,
            proto.cv_count_u,
            proto.cv_count_v
        )

        # Load metadata
        surface.guid = proto.guid
        surface.name = proto.name
        surface.width = proto.width

        # Load knot vectors
        if len(proto.knots_u) == len(surface.m_knot[0]):
            surface.m_knot[0] = np.array(list(proto.knots_u), dtype=np.float64)
        if len(proto.knots_v) == len(surface.m_knot[1]):
            surface.m_knot[1] = np.array(list(proto.knots_v), dtype=np.float64)

        # Load control vertices
        if len(proto.cvs) == len(surface.m_cv):
            surface.m_cv = np.array(list(proto.cvs), dtype=np.float64)

        surface.pointcolors = [Color(c.r, c.g, c.b, c.a) for c in proto.pointcolors]
        surface.facecolors = [Color(c.r, c.g, c.b, c.a) for c in proto.facecolors]
        surface.linecolors = [Color(c.r, c.g, c.b, c.a) for c in proto.linecolors]

        # Load xform
        surface.xform = Xform()
        surface.xform.name = proto.xform.name
        surface.xform.m = list(proto.xform.matrix)

        # Load cached mesh
        if proto.HasField('cached_mesh') and len(proto.cached_mesh.vertices) > 0:
            from .mesh import Mesh
            mesh_data = proto.cached_mesh.SerializeToString()
            surface.m_mesh = Mesh.pb_loads(mesh_data)

        return surface

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
    def pb_load(cls, filepath) -> 'NurbsSurface':
        """Read protobuf from file.

        Parameters
        ----------
        filepath : str or Path
            Path to the protobuf file.

        Returns
        -------
        NurbsSurface
            The deserialized NurbsSurface.
        """
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)
