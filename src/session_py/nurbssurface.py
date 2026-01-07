import numpy as np
import math
from typing import List, Tuple, Optional, Union
import uuid

from .point import Point
from .vector import Vector
from .plane import Plane
from .tolerance import Tolerance
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
        self.surfacecolor = Color.white()
        self.xform = Xform.identity()

        # Core NURBS data
        self.m_dim = 0
        self.m_is_rat = 0
        self.m_order = [0, 0]
        self.m_cv_count = [0, 0]
        self.m_cv_stride = [0, 0]
        self.m_knot_capacity = [0, 0]
        self.m_cv_capacity = 0

        # Data arrays
        self.m_knot = [np.array([], dtype=np.float64), np.array([], dtype=np.float64)]
        self.m_cv = np.array([], dtype=np.float64)

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
        self.surfacecolor = Color.white()
        self.xform = Xform.identity()
        
        self.m_dim = 0
        self.m_is_rat = 0
        self.m_order = [0, 0]
        self.m_cv_count = [0, 0]
        self.m_cv_stride = [0, 0]
        self.m_knot_capacity = [0, 0]
        self.m_cv_capacity = 0
        
        self.m_knot = [np.array([], dtype=np.float64), np.array([], dtype=np.float64)]
        self.m_cv = np.array([], dtype=np.float64)
    
    @staticmethod
    def create(dimension: int, is_rational: bool,
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
        self.m_knot_capacity = [knot_count0, knot_count1]
        
        # Allocate CV array
        total_cvs = cv_count0 * cv_count1
        cv_array_size = total_cvs * cv_size_val
        self.m_cv = np.zeros(cv_array_size, dtype=np.float64)
        self.m_cv_capacity = cv_array_size
        
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
    
    def cv_capacity(self) -> int:
        """Get CV capacity."""
        return self.m_cv_capacity
    
    def knot_capacity(self, dir: int) -> int:
        """Get knot capacity in specified direction."""
        return self.m_knot_capacity[dir] if 0 <= dir < 2 else 0
    
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
        d = order - 1  # degree
        
        # OpenNURBS shifts knot by (order-2) + span, then by d inside basis
        # Net shift: span + d = span + order - 1
        # But we need to account for the pointer offset behavior
        knot_base = span + d  # This gives us the right position
        knot = self.m_knot[dir]
        
        # Check for degenerate span
        if knot[knot_base - 1] == knot[knot_base]:
            return np.zeros(order)
        
        N = np.zeros(order * order)
        N[order * order - 1] = 1.0
        
        left = np.zeros(d)
        right = np.zeros(d)
        
        # Cox-de Boor recursion  - OpenNURBS lines 702-718
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
                y = N[N0_idx + r] / (a0 + a1)
                N[N_idx + r] = x + a1 * y
                x = a0 * y
            N[N_idx + j + 1] = x
        
        # Return just the final row of basis functions
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
        normal = du.cross(dv)
        
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
        copy.m_knot_capacity = self.m_knot_capacity.copy()
        copy.m_cv_capacity = self.m_cv_capacity
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
        self.m_knot_capacity[0], self.m_knot_capacity[1] = self.m_knot_capacity[1], self.m_knot_capacity[0]
        
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
        self.m_cv_capacity = len(new_cv)
        
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
    
    def change_dimension(self, desired_dimension: int) -> bool:
        """Change dimension of surface.
        
        Parameters
        ----------
        desired_dimension : int
            New dimension.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if desired_dimension == self.m_dim:
            return True
        if desired_dimension < 1:
            return False
        
        old_dim = self.m_dim
        new_dim = desired_dimension
        old_cv_size = (old_dim + 1) if self.m_is_rat else old_dim
        new_cv_size = (new_dim + 1) if self.m_is_rat else new_dim
        
        new_cv = np.zeros(self.m_cv_count[0] * self.m_cv_count[1] * new_cv_size)
        
        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                old_index = i * self.m_cv_stride[0] + j * self.m_cv_stride[1]
                new_index = i * (new_cv_size * self.m_cv_count[1]) + j * new_cv_size
                
                # Copy existing dimensions
                copy_dim = min(old_dim, new_dim)
                new_cv[new_index:new_index + copy_dim] = self.m_cv[old_index:old_index + copy_dim]
                
                # Copy weight if rational
                if self.m_is_rat:
                    new_cv[new_index + new_dim] = self.m_cv[old_index + old_dim]
        
        self.m_cv = new_cv
        self.m_dim = new_dim
        self.m_cv_stride[1] = new_cv_size
        self.m_cv_stride[0] = new_cv_size * self.m_cv_count[1]
        self.m_cv_capacity = len(new_cv)
        
        return True
    
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
    
    def subdivide(self, nu: int, nv: int) -> List[List[Point]]:
        """Subdivide surface into a grid of points.

        Evaluates the surface at regular intervals in both parameter directions
        to create a grid of points.

        Parameters
        ----------
        nu : int
            Number of subdivisions in u direction.
        nv : int
            Number of subdivisions in v direction.

        Returns
        -------
        list of list of Point
            2D grid of points, where grid[i][j] is the point at subdivision (i, j).
            Grid dimensions are (nu+1) x (nv+1).
        """
        
        u0, u1 = self.domain(0)
        v0, v1 = self.domain(1)

        # flat list of points
        points = []

        # mapping from (i, j) → vertex index
        index = lambda i, j: i * (nv + 1) + j

        # generate points
        for i in range(nu + 1):
            u = u0 + (u1 - u0) * (i / nu) if nu > 0 else u0
            for j in range(nv + 1):
                v = v0 + (v1 - v0) * (j / nv) if nv > 0 else v0
                points.append(self.point_at(u, v))

        # generate quad faces using indices
        faces = []
        for i in range(nu):
            for j in range(nv):
                faces.append([
                    index(i,     j),
                    index(i + 1, j),
                    index(i + 1, j + 1),
                    index(i,     j + 1),
                ])

        return points, faces


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
        return (f"NurbsSurface(dim={self.m_dim}, "
                f"order=({self.m_order[0]},{self.m_order[1]}), "
                f"cv_count=({self.m_cv_count[0]},{self.m_cv_count[1]}))")
    
    def __str__(self) -> str:
        """Minimal string representation with topology info."""
        u_status = "closed" if self.is_closed(0) else "open"
        v_status = "closed" if self.is_closed(1) else "open"
        return (f"order=({self.m_order[0]},{self.m_order[1]}), "
                f"cv=({self.m_cv_count[0]},{self.m_cv_count[1]}), "
                f"u={u_status}, v={v_status}")

    def __repr__(self) -> str:
        """Full detailed representation (follows protobuf schema)."""
        knots_u_str = f"[{', '.join(f'{k:.3g}' for k in self.m_knot[0][:5])}{'...' if len(self.m_knot[0]) > 5 else ''}]"
        knots_v_str = f"[{', '.join(f'{k:.3g}' for k in self.m_knot[1][:5])}{'...' if len(self.m_knot[1]) > 5 else ''}]"

        return (f"NurbsSurface(name='{self.name}', dim={self.m_dim}, "
                f"rational={bool(self.m_is_rat)}, "
                f"order_u={self.m_order[0]}, order_v={self.m_order[1]}, "
                f"cv_count_u={self.m_cv_count[0]}, cv_count_v={self.m_cv_count[1]}, "
                f"cv_stride_u={self.m_cv_stride[0]}, cv_stride_v={self.m_cv_stride[1]}, "
                f"knots_u={knots_u_str}, knots_v={knots_v_str}, "
                f"cvs={len(self.m_cv)} values, "
                f"width={self.width}, "
                f"surfacecolor={repr(self.surfacecolor)}, "
                f"xform={repr(self.xform)})")
    
    ###########################################################################
    # ADDITIONAL STATIC FACTORY METHODS
    ###########################################################################
    
    @staticmethod
    def create_ruled(curveA: 'NurbsCurve', curveB: 'NurbsCurve') -> 'NurbsSurface':
        """Create ruled surface from two curves.
        
        Parameters
        ----------
        curveA : NurbsCurve
            First curve.
        curveB : NurbsCurve
            Second curve.
        
        Returns
        -------
        NurbsSurface
            Ruled surface.
        """
        # Stub - complex implementation
        return NurbsSurface()
    
    @staticmethod
    def create_planar(curves: List['NurbsCurve']) -> 'NurbsSurface':
        """Create planar surface from boundary curves.
        
        Parameters
        ----------
        curves : list of NurbsCurve
            Boundary curves.
        
        Returns
        -------
        NurbsSurface
            Planar surface.
        """
        # Stub - complex implementation
        return NurbsSurface()
    
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
    # KNOT VECTOR OPERATIONS (ADDITIONAL)
    ###########################################################################
    
    def insert_knot(self, dir: int, knot_value: float, knot_multiplicity: int = 1) -> bool:
        """Insert knot in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        knot_value : float
            Knot value to insert.
        knot_multiplicity : int, optional
            Number of times to insert. Defaults to 1.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        # Stub - requires Boehm's algorithm
        return False
    
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
        self.m_cv_capacity = len(new_cv)
        
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
        """Increase degree in specified direction.
        
        Parameters
        ----------
        dir : int
            Direction (0 for u, 1 for v).
        desired_degree : int
            Desired degree.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        # Stub - requires degree elevation algorithm
        return False
    
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
    
    def closest_point(self, point: Point) -> Tuple[Point, float, float]:
        """Closest point on surface to test point.
        
        Parameters
        ----------
        point : Point
            Test point.
        
        Returns
        -------
        tuple
            (closest_point, u_param, v_param)
        """
        # Stub - requires iterative closest point algorithm
        u0, u1 = self.domain(0)
        v0, v1 = self.domain(1)
        u_out = (u0 + u1) / 2.0
        v_out = (v0 + v1) / 2.0
        return (self.point_at(u_out, v_out), u_out, v_out)
    
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
            'surfacecolor': {
                'r': self.surfacecolor[0],
                'g': self.surfacecolor[1],
                'b': self.surfacecolor[2],
                'a': self.surfacecolor[3]
            }
        }
    
    @classmethod
    def __jsonload__(cls, data: dict) -> 'NurbsSurface':
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
        
        # Set metadata AFTER _create_impl (which calls destroy/initialize)
        srf.guid = data.get('guid', srf.guid)
        srf.name = data.get('name', 'my_nurbssurface')
        srf.width = data.get('width', 1.0)
        
        if 'surfacecolor' in data:
            c = data['surfacecolor']
            srf.surfacecolor = Color(c.get('r', 255), c.get('g', 255), c.get('b', 255), c.get('a', 255))
        
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
    
    ###########################################################################
    # PROTOBUF SERIALIZATION
    ###########################################################################

    def to_protobuf(self):
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

        return proto.SerializeToString()

    @classmethod
    def from_protobuf(cls, data):
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

        return surface

    def protobuf_dump(self, filepath):
        """Write protobuf to file.

        Parameters
        ----------
        filepath : str or Path
            Path to the output file.
        """
        data = self.to_protobuf()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def protobuf_load(cls, filepath) -> 'NurbsSurface':
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
        return cls.from_protobuf(data)
