import numpy as np
import math
from typing import List, Tuple, Optional, Union
import uuid

from .point import Point
from .vector import Vector
from .plane import Plane
from .tolerance import Tolerance
from .tolerance import PI
from .boundingbox import BoundingBox
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
        self.guid = str(uuid.uuid4())
        self.name = "my_nurbssurface"
        self.width = 1.0
        self.surfacecolor = Color.black()
        self.xform = Xform.identity()

        # Core NURBS data
        self.m_dim = 0
        self.m_is_rat = 0
        self.m_order = [0, 0]
        self.m_cv_count = [0, 0]
        self.m_cv_stride = [0, 0]

        # Data arrays
        self.m_knot = [np.array([], dtype=np.float64), np.array([], dtype=np.float64)]
        self.m_cv = np.array([], dtype=np.float64)
        self.m_outer_loop = NurbsCurve()
        self.m_inner_loops = []
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
    
    ###########################################################################
    # INITIALIZATION & CREATION
    ###########################################################################
    
    def initialize(self):
        """Initialize all fields to zero/empty."""
        self.guid = str(uuid.uuid4())
        self.name = "my_nurbssurface"
        self.width = 1.0
        self.surfacecolor = Color.black()
        self.xform = Xform.identity()
        
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
        if cv_count_u < 2 or cv_count_v < 2:
            return NurbsSurface()
        if len(points) != cv_count_u * cv_count_v:
            return NurbsSurface()
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
    # VALIDATION
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

    def __eq__(self, other) -> bool:
        """Check equality with another NurbsSurface (compares all attributes except guid)."""
        if not isinstance(other, NurbsSurface):
            return False

        # Compare metadata (excluding guid)
        if self.name != other.name:
            return False
        if self.width != other.width:
            return False
        if self.surfacecolor != other.surfacecolor:
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
    
    def is_rational(self) -> bool:
        """Check if surface is rational."""
        return self.m_is_rat != 0
    
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
        """Evaluate point and derivatives on surface.
        
        Parameters
        ----------
        u : float
            Parameter in u direction.
        v : float
            Parameter in v direction.
        num_derivs : int, optional
            Number of derivatives to compute. Defaults to 0.
        
        Returns
        -------
        list of Vector
            [point, du, dv, duu, duv, dvv, ...] depending on num_derivs.
        """
        if not self.is_valid():
            return [Vector(0, 0, 0)]
        
        # For now, implement basic evaluation (point only)
        # Full derivative implementation would require basis_functions_derivatives
        pt = self.point_at(u, v)
        result = [Vector(pt.x, pt.y, pt[2])]
        
        if num_derivs > 0:
            # Approximate derivatives with finite differences
            h = 1e-6
            u0, u1 = self.domain(0)
            v0, v1 = self.domain(1)
            
            # du derivative
            if u + h <= u1:
                pt_u = self.point_at(u + h, v)
                du = Vector((pt_u.x - pt.x) / h,
                           (pt_u.y - pt.y) / h,
                           (pt_u.z - pt.z) / h)
            else:
                pt_u = self.point_at(u - h, v)
                du = Vector((pt.x - pt_u.x) / h,
                           (pt.y - pt_u.y) / h,
                           (pt.z - pt_u.z) / h)
            result.append(du)
            
            # dv derivative
            if v + h <= v1:
                pt_v = self.point_at(u, v + h)
                dv = Vector((pt_v.x - pt.x) / h,
                           (pt_v.y - pt.y) / h,
                           (pt_v.z - pt.z) / h)
            else:
                pt_v = self.point_at(u, v - h)
                dv = Vector((pt.x - pt_v.x) / h,
                           (pt.y - pt_v.y) / h,
                           (pt.z - pt_v.z) / h)
            result.append(dv)
        
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
        copy.surfacecolor = self.surfacecolor
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
        
        for i in range(self.m_cv_count[1]):  # Note: swapped
            for j in range(self.m_cv_count[0]):
                old_index = j * self.m_cv_stride[0] + i * self.m_cv_stride[1]
                new_index = i * cv_size_val * self.m_cv_count[0] + j * cv_size_val
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
    
    def get_bounding_box(self) -> BoundingBox:
        """Get bounding box of surface.

        Returns
        -------
        BoundingBox
            Bounding box containing all control points.
        """
        if not self.is_valid() or self.m_cv_count[0] == 0 or self.m_cv_count[1] == 0:
            return BoundingBox()

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

        return BoundingBox(center, Vector.x_axis(), Vector.y_axis(), Vector.z_axis(), half_size)
    
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


    def set_outer_loop(self, loop):
        self.m_outer_loop = loop

    def get_outer_loop(self):
        return self.m_outer_loop

    def is_trimmed(self):
        return self.m_outer_loop.is_valid()

    def clear_outer_loop(self):
        self.m_outer_loop = NurbsCurve()

    def add_inner_loop(self, loop):
        self.m_inner_loops.append(loop)

    def get_inner_loop(self, index):
        return self.m_inner_loops[index]

    def inner_loop_count(self):
        return len(self.m_inner_loops)

    def clear_inner_loops(self):
        self.m_inner_loops = []

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

    def _span_flatness(self, span_u, span_v):
        import math
        deg_u = self.degree(0)
        deg_v = self.degree(1)
        c00 = self.get_cv(span_u, span_v)
        c10 = self.get_cv(span_u + deg_u, span_v)
        c01 = self.get_cv(span_u, span_v + deg_v)
        c11 = self.get_cv(span_u + deg_u, span_v + deg_v)
        max_dist = 0.0
        for a in range(deg_u + 1):
            s = a / deg_u if deg_u > 0 else 0.0
            for b in range(deg_v + 1):
                if (a == 0 or a == deg_u) and (b == 0 or b == deg_v):
                    continue
                t = b / deg_v if deg_v > 0 else 0.0
                cv = self.get_cv(span_u + a, span_v + b)
                bx = (1-s)*(1-t)*c00[0] + s*(1-t)*c10[0] + (1-s)*t*c01[0] + s*t*c11[0]
                by = (1-s)*(1-t)*c00[1] + s*(1-t)*c10[1] + (1-s)*t*c01[1] + s*t*c11[1]
                bz = (1-s)*(1-t)*c00[2] + s*(1-t)*c10[2] + (1-s)*t*c01[2] + s*t*c11[2]
                dx, dy, dz = cv[0]-bx, cv[1]-by, cv[2]-bz
                d = math.sqrt(dx*dx + dy*dy + dz*dz)
                if d > max_dist:
                    max_dist = d
        return max_dist

    def _span_subs(self, dir, sp, osp, flat, max_angle_deg, flat_tol):
        import math
        n = len(sp) - 1
        subs = [2] * n
        s_positions = [
            osp[0] + (osp[-1] - osp[0]) * 0.25,
            (osp[0] + osp[-1]) * 0.5,
            osp[0] + (osp[-1] - osp[0]) * 0.75
        ]
        for i in range(n):
            if flat[i] < flat_tol:
                continue
            t0, t1 = sp[i], sp[i + 1]
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
            subs[i] = max(2, min(int(math.ceil(max_angle / max_angle_deg)), 24))
        return subs

    def mesh(self):
        if self.m_mesh is not None:
            return self.m_mesh
        import math
        from .mesh import Mesh
        usp = self.get_span_vector(0)
        vsp = self.get_span_vector(1)
        if len(usp) < 2 or len(vsp) < 2:
            return Mesh()
        ns_u = len(usp) - 1
        ns_v = len(vsp) - 1
        max_angle_deg = 15.0
        bbox_diag = self._compute_bbox_diagonal()
        flat_tol = bbox_diag * 0.001
        u_flat = [0.0] * ns_u
        v_flat = [0.0] * ns_v
        for i in range(ns_u):
            for j in range(ns_v):
                f = self._span_flatness(i, j)
                if f > u_flat[i]:
                    u_flat[i] = f
                if f > v_flat[j]:
                    v_flat[j] = f
        u_subs = self._span_subs(0, usp, vsp, u_flat, max_angle_deg, flat_tol)
        v_subs = self._span_subs(1, vsp, usp, v_flat, max_angle_deg, flat_tol)
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
            if ratio > 2.0:
                scale = math.sqrt(ratio)
                u_subs = [min(int(math.ceil(s * scale)), 24) for s in u_subs]
            elif ratio < 0.5:
                scale = math.sqrt(1.0 / ratio)
                v_subs = [min(int(math.ceil(s * scale)), 24) for s in v_subs]
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
        if closed_u:
            us.pop()
        if closed_v:
            vs.pop()
        nu, nv = len(us), len(vs)
        result = Mesh()
        vkeys = []
        for i in range(nu):
            for j in range(nv):
                pt = self.point_at(us[i], vs[j])
                vk = result.add_vertex(pt)
                vkeys.append(vk)
        nu_faces = nu if closed_u else nu - 1
        nv_faces = nv if closed_v else nv - 1
        for i in range(nu_faces):
            for j in range(nv_faces):
                i1 = (i + 1) % nu
                j1 = (j + 1) % nv
                v00 = vkeys[i * nv + j]
                v10 = vkeys[i1 * nv + j]
                v01 = vkeys[i * nv + j1]
                v11 = vkeys[i1 * nv + j1]
                if (i + j) % 2 == 0:
                    result.add_face([v00, v10, v11])
                    result.add_face([v00, v11, v01])
                else:
                    result.add_face([v00, v10, v01])
                    result.add_face([v10, v11, v01])
        nv_total = len(result.vertex)
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

    def boundary_curves_3d(self):
        """Return 3D boundary curves evaluated on the surface.

        Returns list of NurbsCurves: [outer_boundary, hole0, hole1, ...]
        The outer_loop and inner_loops are stored in UV parameter space.
        This method evaluates them on the surface to produce 3D curves.
        """
        if not self.is_trimmed():
            return []
        curves = []
        for loop in [self.m_outer_loop] + self.m_inner_loops:
            pts_3d = []
            for i in range(loop.cv_count()):
                uv = loop.get_cv(i)
                pts_3d.append(self.point_at(uv[0], uv[1]))
            curves.append(NurbsCurve.create(True, 1, pts_3d))
        return curves

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
        # Simple implementation: check if all CVs are coplanar
        if self.m_cv_count[0] < 2 or self.m_cv_count[1] < 2:
            return False
        
        # For 2x2 or smaller, all points define a plane (or are collinear)
        if self.m_cv_count[0] <= 2 and self.m_cv_count[1] <= 2:
            return True
        
        # Get three non-collinear points
        p0 = self.get_cv(0, 0)
        p1 = self.get_cv(self.m_cv_count[0] - 1, 0)
        p2 = self.get_cv(0, self.m_cv_count[1] - 1)
        
        # Create plane from these points
        v1 = Vector(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
        v2 = Vector(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z)
        normal = v1.cross(v2)
        
        if normal.magnitude() < 1e-14:
            return False
        
        normal = normal / normal.magnitude()
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
    
    ###########################################################################
    # JSON SERIALIZATION
    ###########################################################################
    
    def jsondump(self) -> dict:
        """Convert to JSON dictionary.
        
        Returns
        -------
        dict
            JSON representation of surface.
        """
        return {
            "guid": self.guid,
            "name": self.name,
            "type": "NurbsSurface",
            "dimension": self.m_dim,
            "is_rational": bool(self.m_is_rat),
            "order_u": self.m_order[0],
            "order_v": self.m_order[1],
            "cv_count_u": self.m_cv_count[0],
            "cv_count_v": self.m_cv_count[1],
            "knots_u": self.m_knot[0].tolist(),
            "knots_v": self.m_knot[1].tolist(),
            "control_points": self.m_cv.tolist(),
            "width": self.width,
            "surfacecolor": self.surfacecolor.jsondump() if hasattr(self.surfacecolor, 'jsondump') else None,
            "xform": self.xform.jsondump() if hasattr(self.xform, 'jsondump') else None
        }
    
    @staticmethod
    def jsonload(data: dict) -> 'NurbsSurface':
        """Load from JSON dictionary.
        
        Parameters
        ----------
        data : dict
            JSON representation.
        
        Returns
        -------
        NurbsSurface
            Loaded surface.
        """
        srf = NurbsSurface()
        
        srf.guid = data.get("guid", str(uuid.uuid4()))
        srf.name = data.get("name", "my_nurbssurface")
        srf.width = data.get("width", 1.0)
        
        dimension = data.get("dimension", 3)
        is_rational = data.get("is_rational", False)
        order_u = data.get("order_u", 4)
        order_v = data.get("order_v", 4)
        cv_count_u = data.get("cv_count_u", 0)
        cv_count_v = data.get("cv_count_v", 0)
        
        if cv_count_u > 0 and cv_count_v > 0:
            srf._create_impl(dimension, is_rational, order_u, order_v, cv_count_u, cv_count_v)
            
            # Load knots
            if "knots_u" in data:
                srf.m_knot[0] = np.array(data["knots_u"], dtype=np.float64)
            if "knots_v" in data:
                srf.m_knot[1] = np.array(data["knots_v"], dtype=np.float64)
            
            # Load control points
            if "control_points" in data:
                srf.m_cv = np.array(data["control_points"], dtype=np.float64)
            
            # Load color and xform if present
            if "surfacecolor" in data and data["surfacecolor"] is not None:
                srf.surfacecolor = Color.jsonload(data["surfacecolor"])
            if "xform" in data and data["xform"] is not None:
                srf.xform = Xform.jsonload(data["xform"])
        
        return srf
    
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
                result += f"    {p[0]}, {p[1]}, {p[2]}\n"
        result += "  ]\n)"
        return result
    
    ###########################################################################
    # ADDITIONAL STATIC FACTORY METHODS
    ###########################################################################
    
    @staticmethod
    def create_ruled(curveA: 'NurbsCurve', curveB: 'NurbsCurve') -> 'NurbsSurface':
        if not curveA.is_valid() or not curveB.is_valid():
            return NurbsSurface()

        cA = curveA.duplicate()
        cB = curveB.duplicate()

        cA.set_domain(0.0, 1.0)
        cB.set_domain(0.0, 1.0)

        if cA.degree() < cB.degree():
            cA.increase_degree(cB.degree())
        elif cB.degree() < cA.degree():
            cB.increase_degree(cA.degree())

        if cA.is_rational() or cB.is_rational():
            cA.make_rational()
            cB.make_rational()

        knots_a = list(cA.get_knots())
        knots_b = list(cB.get_knots())
        tol = 1e-10

        for k in knots_b:
            found = any(abs(ka - k) < tol for ka in knots_a)
            if not found:
                cA.insert_knot(k, 1)

        knots_a = list(cA.get_knots())
        for k in knots_a:
            found = any(abs(kb - k) < tol for kb in knots_b)
            if not found:
                cB.insert_knot(k, 1)

        order_u = cA.order()
        cv_count_u = cA.cv_count()
        is_rat = cA.is_rational()

        surface = NurbsSurface.create_raw(3, is_rat, order_u, 2, cv_count_u, 2)
        if surface is None:
            return NurbsSurface()

        for i in range(cA.knot_count()):
            surface.set_knot(0, i, cA.knot(i))

        surface.set_knot(1, 0, 0.0)
        surface.set_knot(1, 1, 1.0)

        if is_rat:
            for i in range(cv_count_u):
                ok_a, ax, ay, az, aw = cA.get_cv_4d(i)
                surface.set_cv_4d(i, 0, ax, ay, az, aw)
                ok_b, bx, by, bz, bw = cB.get_cv_4d(i)
                surface.set_cv_4d(i, 1, bx, by, bz, bw)
        else:
            for i in range(cv_count_u):
                surface.set_cv(i, 0, cA.get_cv(i))
                surface.set_cv(i, 1, cB.get_cv(i))

        return surface

    @staticmethod
    def create_loft(input_curves, degree_v=3):
        if len(input_curves) < 2:
            return NurbsSurface()
        for c in input_curves:
            if not c.is_valid():
                return NurbsSurface()

        curves = [c.duplicate() for c in input_curves]
        NurbsSurface._make_curves_compatible(curves)
        NurbsSurface._make_curves_compatible(curves)

        n_sections = len(curves)
        cv_count_u = curves[0].cv_count()
        order_u = curves[0].order()
        is_rat = curves[0].is_rational()

        if degree_v >= n_sections:
            degree_v = n_sections - 1
        if degree_v < 1:
            degree_v = 1
        order_v = degree_v + 1

        v_params = [0.0] * n_sections
        for k in range(1, n_sections):
            pk_prev = curves[k - 1].point_at_middle()
            pk_curr = curves[k].point_at_middle()
            dx = pk_curr[0] - pk_prev[0]
            dy = pk_curr[1] - pk_prev[1]
            dz = pk_curr[2] - pk_prev[2]
            v_params[k] = v_params[k - 1] + math.sqrt(dx * dx + dy * dy + dz * dz)

        total_len = v_params[-1]
        if total_len > 1e-14:
            for k in range(n_sections):
                v_params[k] /= total_len
        else:
            for k in range(n_sections):
                v_params[k] = float(k) / (n_sections - 1)

        cv_count_v = n_sections
        knot_count_v = order_v + cv_count_v - 2
        knots_v = [0.0] * knot_count_v

        if degree_v >= n_sections - 1:
            d = degree_v
            for i in range(d):
                knots_v[i] = 0.0
            for i in range(d, knot_count_v):
                knots_v[i] = 1.0
        else:
            for i in range(order_v - 1):
                knots_v[i] = v_params[0]
            for j in range(1, n_sections - order_v + 1):
                s = 0.0
                for i in range(j, j + degree_v):
                    s += v_params[i]
                knots_v[order_v - 2 + j] = s / degree_v
            for i in range(knot_count_v - order_v + 1, knot_count_v):
                knots_v[i] = v_params[n_sections - 1]

        surface = NurbsSurface.create_raw(3, is_rat, order_u, order_v, cv_count_u, cv_count_v)
        if surface is None:
            return NurbsSurface()

        for i in range(surface.knot_count(0)):
            surface.set_knot(0, i, curves[0].knot(i))
        for i in range(len(knots_v)):
            if i < surface.knot_count(1):
                surface.set_knot(1, i, knots_v[i])

        n = n_sections
        N_matrix = [[0.0] * n for _ in range(n)]
        knots_v_arr = np.array(knots_v)

        for k in range(n):
            t = v_params[k]
            t0 = knots_v[order_v - 2]
            t1 = knots_v[knot_count_v - order_v + 1]
            if t < t0:
                t = t0
            if t > t1:
                t = t1

            span = knot.find_span(order_v, cv_count_v, knots_v_arr, t)
            d = order_v - 1
            knot_base = span + d

            if knots_v[knot_base - 1] == knots_v[knot_base]:
                if t <= knots_v[knot_base]:
                    N_matrix[k][span] = 1.0
                else:
                    N_matrix[k][span + order_v - 1] = 1.0
                continue

            Nvals = [0.0] * (order_v * order_v)
            Nvals[order_v * order_v - 1] = 1.0
            left = [0.0] * d
            right = [0.0] * d
            N_idx = order_v * order_v - 1
            k_right = knot_base
            k_left = knot_base - 1

            for j in range(d):
                N0_idx = N_idx
                N_idx -= (order_v + 1)
                left[j] = t - knots_v[k_left]
                right[j] = knots_v[k_right] - t
                k_left -= 1
                k_right += 1

                x = 0.0
                for r in range(j + 1):
                    a0 = left[j - r]
                    a1 = right[r]
                    denom = a0 + a1
                    y = Nvals[N0_idx + r] / denom if denom != 0.0 else 0.0
                    Nvals[N_idx + r] = x + a1 * y
                    x = a0 * y
                Nvals[N_idx + j + 1] = x

            for j in range(order_v):
                col = span + j
                if 0 <= col < n:
                    N_matrix[k][col] = Nvals[j]

        dim = 4 if is_rat else 3
        for i in range(cv_count_u):
            rhs = [[0.0] * dim for _ in range(n)]
            for k in range(n):
                if is_rat:
                    cx, cy, cz, cw = curves[k].get_cv_4d(i)
                    rhs[k] = [cx, cy, cz, cw]
                else:
                    p = curves[k].get_cv(i)
                    rhs[k] = [p[0], p[1], p[2]]

            A = [row[:] for row in N_matrix]
            b = [row[:] for row in rhs]

            for col in range(n):
                max_row = col
                max_val = abs(A[col][col])
                for row in range(col + 1, n):
                    if abs(A[row][col]) > max_val:
                        max_val = abs(A[row][col])
                        max_row = row
                if max_val < 1e-14:
                    continue
                A[col], A[max_row] = A[max_row], A[col]
                b[col], b[max_row] = b[max_row], b[col]
                for row in range(col + 1, n):
                    factor = A[row][col] / A[col][col]
                    for c in range(col, n):
                        A[row][c] -= factor * A[col][c]
                    for d2 in range(dim):
                        b[row][d2] -= factor * b[col][d2]

            Q = [[0.0] * dim for _ in range(n)]
            for row in range(n - 1, -1, -1):
                for d2 in range(dim):
                    Q[row][d2] = b[row][d2]
                    for c in range(row + 1, n):
                        Q[row][d2] -= A[row][c] * Q[c][d2]
                    if abs(A[row][row]) > 1e-14:
                        Q[row][d2] /= A[row][row]

            for j in range(n):
                if is_rat:
                    surface.set_cv_4d(i, j, Q[j][0], Q[j][1], Q[j][2], Q[j][3])
                else:
                    surface.set_cv(i, j, Point(Q[j][0], Q[j][1], Q[j][2]))

        return surface

    @staticmethod
    def _merge_knot_vectors(a, b, tol=1e-10):
        merged = []
        i, j = 0, 0
        while i < len(a) and j < len(b):
            if abs(a[i] - b[j]) < tol:
                merged.append(a[i])
                i += 1
                j += 1
            elif a[i] < b[j]:
                merged.append(a[i])
                i += 1
            else:
                merged.append(b[j])
                j += 1
        while i < len(a):
            merged.append(a[i])
            i += 1
        while j < len(b):
            merged.append(b[j])
            j += 1
        return merged

    @staticmethod
    def _knot_vectors_equal(a, b, tol=1e-10):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if abs(a[i] - b[i]) > tol:
                return False
        return True

    @staticmethod
    def _make_curves_compatible(curves):
        if len(curves) < 2:
            return
        max_deg = max(c.degree() for c in curves)
        for c in curves:
            if c.degree() < max_deg:
                c.increase_degree(max_deg)
        any_rational = any(c.is_rational() for c in curves)
        if any_rational:
            for c in curves:
                c.make_rational()
        already_compatible = True
        for i in range(1, len(curves)):
            if curves[i].cv_count() != curves[0].cv_count():
                already_compatible = False
                break
            if not NurbsSurface._knot_vectors_equal(list(curves[i].get_knots()), list(curves[0].get_knots())):
                already_compatible = False
                break
        if already_compatible:
            return
        for c in curves:
            c.set_domain(0.0, 1.0)
        unified = list(curves[0].get_knots())
        for i in range(1, len(curves)):
            unified = NurbsSurface._merge_knot_vectors(unified, list(curves[i].get_knots()))
        tol = 1e-10
        for c in curves:
            cur_knots = list(c.get_knots())
            for k in unified:
                found = any(abs(ck - k) < tol for ck in cur_knots)
                if not found:
                    c.insert_knot(k, 1)
                    cur_knots = list(c.get_knots())

    @staticmethod
    def create_planar(curves):
        surface = NurbsSurface()
        if not curves:
            return surface

        all_pts = []
        for crv in curves:
            for i in range(crv.cv_count()):
                all_pts.append(crv.get_cv(i))
        if len(all_pts) < 3:
            return surface

        plane = Plane.from_points_pca(all_pts)
        if plane.z_axis.magnitude() < 1e-10:
            return surface

        xax = plane.x_axis
        yax = plane.y_axis
        orig = plane.origin

        min_u = float('inf')
        max_u = float('-inf')
        min_v = float('inf')
        max_v = float('-inf')

        for pt in all_pts:
            dx = pt[0] - orig[0]
            dy = pt[1] - orig[1]
            dz = pt[2] - orig[2]
            u = dx * xax[0] + dy * xax[1] + dz * xax[2]
            v = dx * yax[0] + dy * yax[1] + dz * yax[2]
            min_u = min(min_u, u)
            max_u = max(max_u, u)
            min_v = min(min_v, v)
            max_v = max(max_v, v)

        pad = max(max_u - min_u, max_v - min_v) * 0.05
        if pad < 1e-6:
            pad = 1.0
        min_u -= pad
        max_u += pad
        min_v -= pad
        max_v += pad

        range_u = max_u - min_u
        range_v = max_v - min_v

        surface = NurbsSurface.create_raw(3, False, 2, 2, 2, 2)
        if surface is None:
            return NurbsSurface()
        surface.set_knot(0, 0, 0.0)
        surface.set_knot(0, 1, 1.0)
        surface.set_knot(1, 0, 0.0)
        surface.set_knot(1, 1, 1.0)

        def plane_pt(u, v):
            return Point(
                orig[0] + u * xax[0] + v * yax[0],
                orig[1] + u * xax[1] + v * yax[1],
                orig[2] + u * xax[2] + v * yax[2]
            )

        surface.set_cv(0, 0, plane_pt(min_u, min_v))
        surface.set_cv(0, 1, plane_pt(min_u, max_v))
        surface.set_cv(1, 0, plane_pt(max_u, min_v))
        surface.set_cv(1, 1, plane_pt(max_u, max_v))

        uv_pts = []
        for crv in curves:
            pts3d, params = crv.divide_by_count(50, True)
            for pt in pts3d:
                dx = pt[0] - orig[0]
                dy = pt[1] - orig[1]
                dz = pt[2] - orig[2]
                pu = dx * xax[0] + dy * xax[1] + dz * xax[2]
                pv = dx * yax[0] + dy * yax[1] + dz * yax[2]
                nu = (pu - min_u) / range_u
                nv = (pv - min_v) / range_v
                uv_pts.append(Point(nu, nv, 0.0))

        if len(uv_pts) >= 3:
            loop = NurbsCurve.create(False, 3, uv_pts)
            surface.set_outer_loop(loop)

        return surface

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
    # GEOMETRIC QUERIES
    ###########################################################################
    
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
    
    ###########################################################################
    # NETWORK SURFACE (GORDON SURFACE)
    ###########################################################################

    @staticmethod
    def create_network(u_curves, v_curves):
        from .knot import find_span
        surface = NurbsSurface()
        n_u = len(u_curves)
        n_v = len(v_curves)
        if n_u < 2 or n_v < 2:
            return surface

        u_crvs = [c.duplicate() for c in u_curves]
        v_crvs = [c.duplicate() for c in v_curves]

        def min_dist_sq(crv, pt):
            t0, t1 = crv.domain()
            best = 1e30
            for i in range(51):
                t = t0 + (t1 - t0) * i / 50.0
                p = crv.point_at(t)
                d = (p[0]-pt[0])**2 + (p[1]-pt[1])**2 + (p[2]-pt[2])**2
                if d < best:
                    best = d
            return best

        def find_param(crv, pt):
            t0, t1 = crv.domain()
            best_t, best_d = t0, 1e30
            ns = 200
            for i in range(ns + 1):
                t = t0 + (t1 - t0) * i / ns
                p = crv.point_at(t)
                d = (p[0]-pt[0])**2 + (p[1]-pt[1])**2 + (p[2]-pt[2])**2
                if d < best_d:
                    best_d = d
                    best_t = t
            for _ in range(20):
                derivs = crv.evaluate(best_t, 2)
                dx = derivs[0][0]-pt[0]
                dy = derivs[0][1]-pt[1]
                dz = derivs[0][2]-pt[2]
                f1 = 2.0*(dx*derivs[1][0] + dy*derivs[1][1] + dz*derivs[1][2])
                f2 = 2.0*(derivs[1][0]**2 + derivs[1][1]**2 + derivs[1][2]**2
                          + dx*derivs[2][0] + dy*derivs[2][1] + dz*derivs[2][2])
                if abs(f2) < 1e-14:
                    break
                dt = f1 / f2
                best_t -= dt
                best_t = max(t0, min(t1, best_t))
                if abs(dt) < 1e-14:
                    break
            return best_t

        # Orient v-curves: start near u_curves[0]
        for vc in v_crvs:
            vs = vc.point_at(vc.domain_start())
            ve = vc.point_at(vc.domain_end())
            if min_dist_sq(u_crvs[0], ve) < min_dist_sq(u_crvs[0], vs):
                vc.reverse()

        # Sort v-curves by u-parameter on u_curves[0]
        u_params_raw = [find_param(u_crvs[0], vc.point_at(vc.domain_start())) for vc in v_crvs]
        v_order = sorted(range(n_v), key=lambda j: u_params_raw[j])
        v_crvs = [v_crvs[j] for j in v_order]

        # Orient u-curves: start near first sorted v-curve
        for uc in u_crvs:
            us = uc.point_at(uc.domain_start())
            ue = uc.point_at(uc.domain_end())
            if min_dist_sq(v_crvs[0], ue) < min_dist_sq(v_crvs[0], us):
                uc.reverse()

        # Make curves compatible
        NurbsSurface._make_curves_compatible(u_crvs)
        NurbsSurface._make_curves_compatible(v_crvs)
        for c in u_crvs:
            c.set_domain(0.0, 1.0)
        for c in v_crvs:
            c.set_domain(0.0, 1.0)

        # Find intersection parameters for ALL pairs
        t_u_ij = [[0.0]*n_v for _ in range(n_u)]
        t_v_ij = [[0.0]*n_v for _ in range(n_u)]
        for i in range(n_u):
            for j in range(n_v):
                approx_v = float(i) / (n_u - 1) if n_u > 1 else 0.0
                pv = v_crvs[j].point_at(approx_v)
                t_u_ij[i][j] = find_param(u_crvs[i], pv)
                pu = u_crvs[i].point_at(t_u_ij[i][j])
                t_v_ij[i][j] = find_param(v_crvs[j], pu)

        # Average intersection parameters
        u_params = [sum(t_u_ij[i][j] for i in range(n_u)) / n_u for j in range(n_v)]
        v_params = [sum(t_v_ij[i][j] for j in range(n_v)) / n_v for i in range(n_u)]

        # Helpers
        def gauss_solve(n, dim, A, b):
            for col in range(n):
                mr, mv = col, abs(A[col][col])
                for r in range(col+1, n):
                    if abs(A[r][col]) > mv:
                        mv = abs(A[r][col])
                        mr = r
                if mv < 1e-14:
                    continue
                A[col], A[mr] = A[mr], A[col]
                b[col], b[mr] = b[mr], b[col]
                for r in range(col+1, n):
                    f = A[r][col] / A[col][col]
                    for c in range(col, n):
                        A[r][c] -= f * A[col][c]
                    for d in range(dim):
                        b[r][d] -= f * b[col][d]
            for r in range(n-1, -1, -1):
                for d in range(dim):
                    for c in range(r+1, n):
                        b[r][d] -= A[r][c] * b[c][d]
                    if abs(A[r][r]) > 1e-14:
                        b[r][d] /= A[r][r]

        def build_interp_knots(n_pts, deg, params):
            ord_ = deg + 1
            kc = ord_ + n_pts - 2
            knots = [0.0] * kc
            for i in range(ord_ - 1):
                knots[i] = params[0]
            for j in range(1, n_pts - ord_ + 1):
                s = sum(params[j:j+deg]) / deg
                knots[ord_ - 2 + j] = s
            for i in range(kc - ord_ + 1, kc):
                knots[i] = params[n_pts - 1]
            return knots

        def build_basis_matrix(n, ord_, params, knots):
            N = [[0.0]*n for _ in range(n)]
            for row in range(n):
                t = params[row]
                span = find_span(ord_, n, knots, t)
                d = ord_ - 1
                kb = span + d
                if knots[kb - 1] == knots[kb]:
                    if t <= knots[kb]:
                        N[row][span] = 1.0
                    else:
                        N[row][span + ord_ - 1] = 1.0
                    continue
                Nv = [0.0] * (ord_ * ord_)
                Nv[ord_ * ord_ - 1] = 1.0
                left = [0.0] * d
                right = [0.0] * d
                ni = ord_ * ord_ - 1
                kr, kl = kb, kb - 1
                for jj in range(d):
                    n0 = ni
                    ni -= (ord_ + 1)
                    left[jj] = t - knots[kl]
                    right[jj] = knots[kr] - t
                    kl -= 1
                    kr += 1
                    xv = 0.0
                    for r in range(jj + 1):
                        a0 = left[jj - r]
                        a1 = right[r]
                        den = a0 + a1
                        yv = Nv[n0 + r] / den if den != 0.0 else 0.0
                        Nv[ni + r] = xv + a1 * yv
                        xv = a0 * yv
                    Nv[ni + jj + 1] = xv
                for jj in range(ord_):
                    col = span + jj
                    if 0 <= col < n:
                        N[row][col] = Nv[jj]
            return N

        # Reparametrize curves using monotone Hermite parameter mapping
        def monotone_hermite_eval(xs, ys, t):
            n = len(xs)
            if n < 2: return t
            if t <= xs[0]: return ys[0]
            if t >= xs[n-1]: return ys[n-1]
            delta = [(ys[k+1]-ys[k])/(xs[k+1]-xs[k]) for k in range(n-1)]
            d = [0.0]*n
            d[0] = delta[0]; d[n-1] = delta[n-2]
            for k in range(1, n-1):
                d[k] = 0.0 if delta[k-1]*delta[k] <= 0 else (delta[k-1]+delta[k])/2.0
            for k in range(n-1):
                if abs(delta[k]) < 1e-15: d[k] = 0; d[k+1] = 0; continue
                alpha, beta = d[k]/delta[k], d[k+1]/delta[k]
                if alpha < 0: d[k] = 0; alpha = 0
                if beta < 0: d[k+1] = 0; beta = 0
                r2 = alpha*alpha + beta*beta
                if r2 > 9.0:
                    tau = 3.0 / (r2**0.5)
                    d[k] = tau * alpha * delta[k]
                    d[k+1] = tau * beta * delta[k]
            ki = 0
            for i in range(n-1):
                if t < xs[i+1]: ki = i; break
                if i == n-2: ki = i
            h = xs[ki+1] - xs[ki]
            if h < 1e-15: return ys[ki]
            s = (t - xs[ki]) / h; s2 = s*s; s3 = s2*s
            return (2*s3-3*s2+1)*ys[ki] + (s3-2*s2+s)*h*d[ki] \
                 + (-2*s3+3*s2)*ys[ki+1] + (s3-s2)*h*d[ki+1]

        def reparametrize_curve(crv, target_params, actual_params):
            np_ = len(target_params)
            mx, my = [], []
            has0 = any(abs(target_params[k]) < 1e-10 for k in range(np_))
            has1 = any(abs(target_params[k] - 1.0) < 1e-10 for k in range(np_))
            if not has0: mx.append(0.0); my.append(0.0)
            for k in range(np_): mx.append(target_params[k]); my.append(actual_params[k])
            if not has1: mx.append(1.0); my.append(1.0)
            max_dev = max(abs(my[k]-mx[k]) for k in range(len(mx)))
            if max_dev < 1e-6: return crv
            nu = max(crv.cv_count() * 5, 30)
            sample_set = set()
            for k in range(nu): sample_set.add(float(k) / (nu - 1))
            for k in range(np_): sample_set.add(target_params[k])
            sp = sorted(sample_set)
            ns = len(sp)
            pts = []
            for k in range(ns):
                tm = max(0.0, min(1.0, monotone_hermite_eval(mx, my, sp[k])))
                pts.append(crv.point_at(tm))
            deg = min(3, ns - 1); ord_ = deg + 1
            kn = build_interp_knots(ns, deg, sp)
            N = build_basis_matrix(ns, ord_, sp, kn)
            rhs = [[pts[k][0], pts[k][1], pts[k][2]] for k in range(ns)]
            A_copy = [row[:] for row in N]
            gauss_solve(ns, 3, A_copy, rhs)
            nc = NurbsCurve(3, False, ord_, ns)
            for k in range(min(len(kn), nc.knot_count())):
                nc.set_knot(k, kn[k])
            for k in range(ns):
                nc.set_cv(k, Point(rhs[k][0], rhs[k][1], rhs[k][2]))
            nc.set_domain(0.0, 1.0)
            return nc

        for i in range(n_u):
            tgt = [u_params[j] for j in range(n_v)]
            act = [t_u_ij[i][j] for j in range(n_v)]
            u_crvs[i] = reparametrize_curve(u_crvs[i], tgt, act)
        for j in range(n_v):
            tgt = [v_params[i] for i in range(n_u)]
            act = [t_v_ij[i][j] for i in range(n_u)]
            v_crvs[j] = reparametrize_curve(v_crvs[j], tgt, act)

        NurbsSurface._make_curves_compatible(u_crvs)
        NurbsSurface._make_curves_compatible(v_crvs)
        for c in u_crvs: c.set_domain(0.0, 1.0)
        for c in v_crvs: c.set_domain(0.0, 1.0)

        # Intersection points from u-curves (ensures exact v-curve interpolation at grid points)
        P_ij = [[None]*n_v for _ in range(n_u)]
        for i in range(n_u):
            for j in range(n_v):
                P_ij[i][j] = u_crvs[i].point_at(u_params[j])

        def skin_curves(curves, cross_params, cross_degree):
            nc = len(curves)
            cv_along = curves[0].cv_count()
            order_along = curves[0].order()
            cdeg = min(cross_degree, nc - 1)
            cord = cdeg + 1
            cross_knots = build_interp_knots(nc, cdeg, cross_params)
            N_cross = build_basis_matrix(nc, cord, cross_params, cross_knots)
            srf = NurbsSurface.create_raw(3, False, order_along, cord, cv_along, nc)
            for i in range(srf.knot_count(0)):
                srf.set_knot(0, i, curves[0].knot(i))
            v_kc = len(cross_knots)
            for i in range(min(v_kc, srf.knot_count(1))):
                srf.set_knot(1, i, cross_knots[i])
            for j in range(cv_along):
                A = [row[:] for row in N_cross]
                rhs = [[0.0]*3 for _ in range(nc)]
                for k in range(nc):
                    p = curves[k].get_cv(j)
                    rhs[k] = [p[0], p[1], p[2]]
                gauss_solve(nc, 3, A, rhs)
                for k in range(nc):
                    srf.set_cv(j, k, Point(rhs[k][0], rhs[k][1], rhs[k][2]))
            return srf

        # Step 1: S_profiles = skin u-curves at v_params
        s_profiles = skin_curves(u_crvs, v_params, 3)

        # Step 2: S_guides = skin v-curves at u_params, transpose
        s_guides = skin_curves(v_crvs, u_params, 3)
        s_guides.transpose()

        # Step 3: S_tensor = interpolate intersection point grid
        u_deg = min(3, n_v - 1)
        u_ord = u_deg + 1
        u_knots_t = build_interp_knots(n_v, u_deg, u_params)
        N_u = build_basis_matrix(n_v, u_ord, u_params, u_knots_t)
        row_curves = []
        for i in range(n_u):
            A = [row[:] for row in N_u]
            rhs = [[P_ij[i][j][0], P_ij[i][j][1], P_ij[i][j][2]] for j in range(n_v)]
            gauss_solve(n_v, 3, A, rhs)
            crv = NurbsCurve(3, False, u_ord, n_v)
            for k in range(min(len(u_knots_t), crv.knot_count())):
                crv.set_knot(k, u_knots_t[k])
            for k in range(n_v):
                crv.set_cv(k, Point(rhs[k][0], rhs[k][1], rhs[k][2]))
            row_curves.append(crv)
        s_tensor = skin_curves(row_curves, v_params, 3)

        # Step 4: Make all three surfaces compatible
        for dir_ in range(2):
            s_profiles.set_domain(dir_, 0.0, 1.0)
            s_guides.set_domain(dir_, 0.0, 1.0)
            s_tensor.set_domain(dir_, 0.0, 1.0)

        max_deg_u = max(s_profiles.degree(0), s_guides.degree(0), s_tensor.degree(0))
        max_deg_v = max(s_profiles.degree(1), s_guides.degree(1), s_tensor.degree(1))
        for srf in [s_profiles, s_guides, s_tensor]:
            if srf.degree(0) < max_deg_u:
                srf.increase_degree(0, max_deg_u)
            if srf.degree(1) < max_deg_v:
                srf.increase_degree(1, max_deg_v)

        ktol = 1e-10
        for dir_ in range(2):
            ka = list(s_profiles.get_knots(dir_))
            kb = list(s_guides.get_knots(dir_))
            kc = list(s_tensor.get_knots(dir_))
            unified = NurbsSurface._merge_knot_vectors(
                NurbsSurface._merge_knot_vectors(ka, kb), kc)
            for srf in [s_profiles, s_guides, s_tensor]:
                cur = list(srf.get_knots(dir_))
                for kv in unified:
                    found = any(abs(ck - kv) < ktol for ck in cur)
                    if not found:
                        srf.insert_knot(dir_, kv, 1)
                        cur = list(srf.get_knots(dir_))

        # Step 5: Combine control points
        final_cv_u = s_profiles.cv_count(0)
        final_cv_v = s_profiles.cv_count(1)
        surface = s_profiles.duplicate()
        for i in range(final_cv_u):
            for j in range(final_cv_v):
                pp = s_profiles.get_cv(i, j)
                pg = s_guides.get_cv(i, j)
                pt = s_tensor.get_cv(i, j)
                surface.set_cv(i, j, Point(
                    pp[0] + pg[0] - pt[0],
                    pp[1] + pg[1] - pt[1],
                    pp[2] + pg[2] - pt[2]))
        return surface

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
        # Stub - requires knot insertion and removal
        return False
    
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
        # Stub - requires knot insertion
        return (None, None)
    
    def extend(self, dir: int, domain: Tuple[float, float]) -> bool:
        """Extend surface to include domain in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        domain : tuple
            (start, end) domain to extend to.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        # Stub - requires curve extension algorithms
        return False
    
    def make_non_rational(self) -> bool:
        """Make surface non-rational if all weights are equal.
        
        Returns
        -------
        bool
            True if successful, False if weights are non-uniform.
        """
        if not self.m_is_rat:
            return True
        
        # Check if all weights are 1.0
        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                w = self.weight(i, j)
                if abs(w - 1.0) > 1e-10:
                    return False  # Cannot make non-rational
        
        # Convert to non-rational by removing weights
        old_cv_size = self.m_dim + 1
        new_cv_size = self.m_dim
        new_cv = np.zeros(self.m_cv_count[0] * self.m_cv_count[1] * new_cv_size)
        
        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                old_index = i * self.m_cv_stride[0] + j * self.m_cv_stride[1]
                new_index = i * (new_cv_size * self.m_cv_count[1]) + j * new_cv_size
                
                # Copy coordinates (not weight)
                new_cv[new_index:new_index + self.m_dim] = self.m_cv[old_index:old_index + self.m_dim]
        
        self.m_cv = new_cv
        self.m_is_rat = 0
        self.m_cv_stride[1] = new_cv_size
        self.m_cv_stride[0] = new_cv_size * self.m_cv_count[1]

        return True

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
    
    def area(self, tolerance: float = 1e-6) -> float:
        """Get surface area (approximate).
        
        Parameters
        ----------
        tolerance : float, optional
            Tolerance for approximation.
        
        Returns
        -------
        float
            Approximate surface area.
        """
        # Stub - requires numerical integration
        return 0.0
    
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
                    cv_ptr = self.cv(span_index - self.m_order[1 - dir] + 1 + k, i)
                else:
                    # iso-v: u varies, v is constant at c
                    cv_ptr = self.cv(i, span_index - self.m_order[1 - dir] + 1 + k)
                
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
    
    def closest_point(self, point):
        dom_u = self.domain(0)
        dom_v = self.domain(1)
        nu, nv = 16, 16
        best_dist2 = 1e300
        best_u = (dom_u[0] + dom_u[1]) * 0.5
        best_v = (dom_v[0] + dom_v[1]) * 0.5
        for i in range(nu + 1):
            u = dom_u[0] + (dom_u[1] - dom_u[0]) * i / nu
            for j in range(nv + 1):
                v = dom_v[0] + (dom_v[1] - dom_v[0]) * j / nv
                pt = self.point_at(u, v)
                dx = pt[0] - point[0]
                dy = pt[1] - point[1]
                dz = pt[2] - point[2]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < best_dist2:
                    best_dist2 = d2
                    best_u = u
                    best_v = v
        u, v = best_u, best_v
        for _ in range(20):
            derivs = self.evaluate(u, v, 1)
            if len(derivs) < 3:
                break
            dx = derivs[0][0] - point[0]
            dy = derivs[0][1] - point[1]
            dz = derivs[0][2] - point[2]
            su0, su1, su2 = derivs[1][0], derivs[1][1], derivs[1][2]
            sv0, sv1, sv2 = derivs[2][0], derivs[2][1], derivs[2][2]
            fu = dx * su0 + dy * su1 + dz * su2
            fv = dx * sv0 + dy * sv1 + dz * sv2
            if abs(fu) < 1e-14 and abs(fv) < 1e-14:
                break
            juu = su0 * su0 + su1 * su1 + su2 * su2
            juv = su0 * sv0 + su1 * sv1 + su2 * sv2
            jvv = sv0 * sv0 + sv1 * sv1 + sv2 * sv2
            det = juu * jvv - juv * juv
            if abs(det) < 1e-30:
                break
            du = -(jvv * fu - juv * fv) / det
            dv = -(juu * fv - juv * fu) / det
            u += du
            v += dv
            u = max(dom_u[0], min(dom_u[1], u))
            v = max(dom_v[0], min(dom_v[1], v))
            if du * du + dv * dv < 1e-28:
                break
        return (self.point_at(u, v), u, v)

    def add_hole(self, curve_3d):
        dom = curve_3d.domain()
        sdom_u = self.domain(0)
        sdom_v = self.domain(1)
        range_u = sdom_u[1] - sdom_u[0]
        range_v = sdom_v[1] - sdom_v[0]
        n_samples = max(curve_3d.cv_count() * 4, 32)
        uv_pts = []
        for i in range(n_samples):
            t = dom[0] + (dom[1] - dom[0]) * i / n_samples
            pt3d = curve_3d.point_at(t)
            _, u, v = self.closest_point(pt3d)
            nu = (u - sdom_u[0]) / range_u
            nv = (v - sdom_v[0]) / range_v
            uv_pts.append(Point(nu, nv, 0.0))
        if len(uv_pts) >= 3:
            from .nurbscurve import NurbsCurve
            self.add_inner_loop(NurbsCurve.create(True, 1, uv_pts))

    def add_holes(self, curves_3d):
        for crv in curves_3d:
            self.add_hole(crv)
    
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
    
    def is_duplicate(self, other: 'NurbsSurface', ignore_parameterization: bool,
                    tolerance: float = Tolerance.ZERO_TOLERANCE) -> bool:
        """Check if this surface is duplicate of another.
        
        Parameters
        ----------
        other : NurbsSurface
            Other surface to compare.
        ignore_parameterization : bool
            Whether to ignore parameterization differences.
        tolerance : float, optional
            Tolerance for comparison.
        
        Returns
        -------
        bool
            True if duplicate, False otherwise.
        """
        # Stub - requires comprehensive comparison
        return False
    
    def collapse_side(self, side: int, point: Point) -> bool:
        """Collapse side of surface to a point.
        
        Parameters
        ----------
        side : int
            Side (0=SW, 1=SE, 2=NE, 3=NW).
        point : Point
            Point to collapse to.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        # Stub - requires knot manipulation
        return False
    
    ###########################################################################
    # JSON SERIALIZATION
    ###########################################################################
    
    def __jsondump__(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'guid': self.guid,
            'name': self.name,
            'dimension': self.m_dim,
            'is_rational': bool(self.m_is_rat),
            'order_u': self.m_order[0],
            'order_v': self.m_order[1],
            'cv_count_u': self.m_cv_count[0],
            'cv_count_v': self.m_cv_count[1],
            'knots_u': self.m_knot[0].tolist(),
            'knots_v': self.m_knot[1].tolist(),
            'control_points': self.m_cv.tolist(),
            'width': self.width,
            'surfacecolor': self.surfacecolor.__jsondump__(),
            **(({'outer_loop': self.m_outer_loop.__jsondump__()} if self.is_trimmed() else {})),
            **(({'inner_loops': [l.__jsondump__() for l in self.m_inner_loops]} if self.m_inner_loops else {})),
            **(({'mesh': self.m_mesh.__jsondump__()} if self.m_mesh is not None else {}))
        }
    
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

        if 'surfacecolor' in data:
            srf.surfacecolor = Color.__jsonload__(data['surfacecolor'])

        if 'outer_loop' in data:
            srf.m_outer_loop = NurbsCurve.__jsonload__(data['outer_loop'])
        if 'inner_loops' in data:
            srf.m_inner_loops = [NurbsCurve.__jsonload__(l) for l in data['inner_loops']]
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

        # Surface color
        proto.surfacecolor.name = self.surfacecolor.name
        proto.surfacecolor.r = self.surfacecolor[0]
        proto.surfacecolor.g = self.surfacecolor[1]
        proto.surfacecolor.b = self.surfacecolor[2]
        proto.surfacecolor.a = self.surfacecolor[3]

        # Transform
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)

        # Outer loop
        if self.is_trimmed():
            loop_data = self.m_outer_loop.pb_dumps()
            proto.outer_loop.ParseFromString(loop_data)

        # Inner loops
        for inner in self.m_inner_loops:
            loop_data = inner.pb_dumps()
            il = proto.inner_loops.add()
            il.ParseFromString(loop_data)

        return proto.SerializeToString()

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

        # Load color
        surface.surfacecolor = Color(
            proto.surfacecolor.r,
            proto.surfacecolor.g,
            proto.surfacecolor.b,
            proto.surfacecolor.a
        )
        surface.surfacecolor.name = proto.surfacecolor.name

        # Load xform
        surface.xform = Xform()
        surface.xform.name = proto.xform.name
        surface.xform.m = list(proto.xform.matrix)

        # Load outer loop
        if proto.HasField('outer_loop') and proto.outer_loop.cv_count > 0:
            loop_data = proto.outer_loop.SerializeToString()
            surface.m_outer_loop = NurbsCurve.pb_loads(loop_data)

        # Load inner loops
        for il in proto.inner_loops:
            loop_data = il.SerializeToString()
            surface.m_inner_loops.append(NurbsCurve.pb_loads(loop_data))

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
