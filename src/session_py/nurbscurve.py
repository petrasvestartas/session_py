import numpy as np
import math
from typing import List, Tuple, Optional, Union
import uuid

from .point import Point
from .vector import Vector
from .plane import Plane
from .tolerance import Tolerance
from .xform import Xform
from .color import Color
from . import knot
from .knot import CurveKnotStyle


class NurbsCurve:
    """A Non-Uniform Rational B-Spline (NURBS) curve.

    Parameters
    ----------
    dimension : int, optional
        The dimension of the curve (typically 3 for 3D). Defaults to 3.
    is_rational : bool, optional
        Whether the curve is rational (has weights). Defaults to False.
    order : int, optional
        The order of the curve (degree + 1). Defaults to 4 (cubic).
    cv_count : int, optional
        Number of control vertices. Defaults to 0.

    Attributes
    ----------
    guid : str
        Unique identifier of the curve.
    name : str
        Name of the curve.
    m_dim : int
        Dimension of the curve.
    m_is_rat : int
        1 if rational, 0 if non-rational.
    m_order : int
        Order of the curve (degree + 1).
    m_cv_count : int
        Number of control vertices.
    m_cv_stride : int
        Stride between control vertices in array.
    m_knot : np.ndarray
        Knot vector array.
    m_cv : np.ndarray
        Control vertex data array (homogeneous if rational).
    """
    
    def __init__(self, dimension: int = 3, is_rational: bool = False,
                 order: int = 4, cv_count: int = 0):
        self.guid = str(uuid.uuid4())
        self.name = "my_nurbscurve"
        self.width = 1.0
        self.linecolor = Color.white()
        self.xform = Xform.identity()

        self.m_dim = dimension
        self.m_is_rat = 1 if is_rational else 0
        self.m_order = order
        self.m_cv_count = cv_count
        self.m_cv_stride = (dimension + 1) if is_rational else dimension

        self.m_knot = np.array([], dtype=np.float64)
        self.m_cv = np.array([], dtype=np.float64)

        self._rmf_cache = None
    
    #############################################################################
    # STATIC FACTORY METHODS
    #############################################################################
    
    @staticmethod
    def create(periodic: bool, degree: int, points: List[Point], 
               dimension: int = 3, knot_delta: float = 1.0) -> 'NurbsCurve':
        """Create NURBS curve from points.

        Parameters
        ----------
        periodic : bool
            If True, creates a periodic curve; otherwise clamped.
        degree : int
            The degree of the curve.
        points : list of Point
            Control points for the curve.
        dimension : int, optional
            Dimension of the curve. Defaults to 3.
        knot_delta : float, optional
            Spacing between knots. Defaults to 1.0.

        Returns
        -------
        NurbsCurve
            The created NURBS curve.
        """
        curve = NurbsCurve()
        if periodic:
            curve.create_periodic_uniform(dimension, degree + 1, points, knot_delta)
        else:
            curve.create_clamped_uniform(dimension, degree + 1, points, knot_delta)
        return curve

    #############################################################################
    # INITIALIZATION & CREATION
    #############################################################################
    
    def initialize(self):
        """Initialize all fields to zero/empty.
        
        Returns
        -------
        None
        """
        self.m_dim = 0
        self.m_is_rat = 0
        self.m_order = 0
        self.m_cv_count = 0
        self.m_cv_stride = 0
        self.m_knot = np.array([], dtype=np.float64)
        self.m_cv = np.array([], dtype=np.float64)
    
    def create_curve(self, dimension: int, is_rational: bool, 
                    order: int, cv_count: int) -> bool:
        """Create NURBS curve with specified parameters"""
        if dimension < 1 or order < 2 or cv_count < order:
            return False
        
        self.m_dim = dimension
        self.m_is_rat = 1 if is_rational else 0
        self.m_order = order
        self.m_cv_count = cv_count
        self.m_cv_stride = (dimension + 1) if is_rational else dimension
        
        # Allocate arrays
        knot_count = order + cv_count - 2
        self.m_knot = np.zeros(knot_count, dtype=np.float64)
        self.m_cv = np.zeros(cv_count * self.m_cv_stride, dtype=np.float64)
        
        # Set weights to 1.0 if rational
        if is_rational:
            for i in range(cv_count):
                self.m_cv[i * self.m_cv_stride + dimension] = 1.0
        
        return True
    
    def create_clamped_uniform(self, dimension: int, order: int, 
                              points: List[Point], knot_delta: float = 1.0) -> bool:
        """Create clamped uniform NURBS curve from control points"""
        if not points or len(points) < order:
            return False
        
        if not self.create_curve(dimension, False, order, len(points)):
            return False
        
        # Set control points
        for i, pt in enumerate(points):
            self.set_cv(i, pt)
        
        # Create clamped uniform knot vector
        self._make_clamped_uniform_knot_vector(knot_delta)
        
        return True
    
    def create_periodic_uniform(self, dimension: int, order: int,
                               points: List[Point], knot_delta: float = 1.0) -> bool:
        """Create periodic uniform NURBS curve from control points"""
        if not points or len(points) < order:
            return False

        if not self.create_curve(dimension, False, order, len(points)):
            return False

        # Set control points
        for i, pt in enumerate(points):
            self.set_cv(i, pt)

        # Create periodic uniform knot vector
        self._make_periodic_uniform_knot_vector(knot_delta)

        return True

    def destroy(self):
        """Deallocate all memory and reset to empty state"""
        self.initialize()
    
    #############################################################################
    # VALIDATION
    #############################################################################
    
    def is_valid(self) -> bool:
        """Check if NURBS curve is valid"""
        if self.m_dim < 1:
            return False
        if self.m_order < 2:
            return False
        if self.m_cv_count < self.m_order:
            return False
        if len(self.m_knot) != self.m_order + self.m_cv_count - 2:
            return False
        if len(self.m_cv) < self.m_cv_count * self.m_cv_stride:
            return False
        
        # Check knot vector is non-decreasing
        for i in range(len(self.m_knot) - 1):
            if self.m_knot[i] > self.m_knot[i + 1] + Tolerance.ZERO_TOLERANCE:
                return False
        
        return True
    
    #############################################################################
    # ACCESSORS
    #############################################################################
    
    def dimension(self) -> int:
        return self.m_dim
    
    def is_rational(self) -> bool:
        return self.m_is_rat != 0
    
    def order(self) -> int:
        return self.m_order
    
    def degree(self) -> int:
        return self.m_order - 1
    
    def cv_count(self) -> int:
        return self.m_cv_count
    
    def cv_size(self) -> int:
        """Size of each control vertex"""
        return (self.m_dim + 1) if self.m_is_rat else self.m_dim
    
    def knot_count(self) -> int:
        return self.m_order + self.m_cv_count - 2
    
    def span_count(self) -> int:
        return self.m_cv_count - self.m_order + 1

    #############################################################################
    # CONTROL VERTEX ACCESS
    #############################################################################
    
    def get_cv(self, cv_index: int) -> Optional[Point]:
        """Get control point at index as Point (Euclidean coordinates)"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return None

        idx = cv_index * self.m_cv_stride
        return Point(
            self.m_cv[idx] if self.m_dim > 0 else 0,
            self.m_cv[idx + 1] if self.m_dim > 1 else 0,
            self.m_cv[idx + 2] if self.m_dim > 2 else 0
            )

    def cv(self, cv_index: int) -> Optional[List[float]]:
        """Get raw CV data at index (like C++ double* cv(int))"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return None
        idx = cv_index * self.m_cv_stride
        return list(self.m_cv[idx:idx + self.m_cv_stride])

    def set_cv(self, cv_index: int, point: Point) -> bool:
        """Set control point at index from Point"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return False

        idx = cv_index * self.m_cv_stride
        if self.m_dim > 0:
            self.m_cv[idx] = point.x
        if self.m_dim > 1:
            self.m_cv[idx + 1] = point.y
        if self.m_dim > 2:
            self.m_cv[idx + 2] = point.z

        if self.m_is_rat:
            w = self.m_cv[idx + self.m_dim]
            if self.m_dim > 0:
                self.m_cv[idx] *= w
            if self.m_dim > 1:
                self.m_cv[idx + 1] *= w
            if self.m_dim > 2:
                self.m_cv[idx + 2] *= w

        self._invalidate_rmf_cache()
        return True
    
    def get_cv_4d(self, cv_index: int) -> Optional[Tuple[float, float, float, float]]:
        """Get control point as homogeneous coordinates (x, y, z, w)"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return None
        
        idx = cv_index * self.m_cv_stride
        x = self.m_cv[idx] if self.m_dim > 0 else 0.0
        y = self.m_cv[idx + 1] if self.m_dim > 1 else 0.0
        z = self.m_cv[idx + 2] if self.m_dim > 2 else 0.0
        w = self.m_cv[idx + self.m_dim] if self.m_is_rat else 1.0
        
        return (x, y, z, w)
    
    def set_cv_4d(self, cv_index: int, x: float, y: float, z: float, w: float) -> bool:
        """Set control point from homogeneous coordinates"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return False

        # Make rational if w != 1.0 (matches C++ implementation)
        if not self.m_is_rat and w != 1.0:
            self.make_rational()

        idx = cv_index * self.m_cv_stride
        if self.m_dim > 0:
            self.m_cv[idx] = x
        if self.m_dim > 1:
            self.m_cv[idx + 1] = y
        if self.m_dim > 2:
            self.m_cv[idx + 2] = z
        if self.m_is_rat:
            self.m_cv[idx + self.m_dim] = w

        self._invalidate_rmf_cache()
        return True
    
    def weight(self, cv_index: int) -> float:
        """Get weight at control vertex index"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return 1.0
        
        if not self.m_is_rat:
            return 1.0
        
        idx = cv_index * self.m_cv_stride
        return self.m_cv[idx + self.m_dim]
    
    def set_weight(self, cv_index: int, weight: float) -> bool:
        """Set weight at control vertex index"""
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return False

        if not self.m_is_rat:
            if abs(weight - 1.0) > Tolerance.ZERO_TOLERANCE:
                self.make_rational()

        if self.m_is_rat:
            idx = cv_index * self.m_cv_stride
            self.m_cv[idx + self.m_dim] = weight

        self._invalidate_rmf_cache()
        return True
    
    #############################################################################
    # KNOT ACCESS
    #############################################################################
    
    def knot(self, knot_index: int) -> float:
        """Get knot value at index"""
        if knot_index < 0 or knot_index >= len(self.m_knot):
            return 0.0
        return self.m_knot[knot_index]
    
    def set_knot(self, knot_index: int, knot_value: float) -> bool:
        """Set knot value at index"""
        if knot_index < 0 or knot_index >= len(self.m_knot):
            return False
        self.m_knot[knot_index] = knot_value
        self._invalidate_rmf_cache()
        return True
    
    def knot_multiplicity(self, knot_index: int) -> int:
        """Get knot multiplicity at index"""
        if knot_index < 0 or knot_index >= len(self.m_knot):
            return 0
        
        knot_value = self.m_knot[knot_index]
        mult = 1
        
        # Count after
        for i in range(knot_index + 1, len(self.m_knot)):
            if abs(self.m_knot[i] - knot_value) < Tolerance.ZERO_TOLERANCE:
                mult += 1
            else:
                break
        
        # Count before
        for i in range(knot_index - 1, -1, -1):
            if abs(self.m_knot[i] - knot_value) < Tolerance.ZERO_TOLERANCE:
                mult += 1
            else:
                break
        
        return mult
    
    def get_knots(self) -> np.ndarray:
        """Get all knot values"""
        return self.m_knot.copy()
    
    def knot_array(self) -> np.ndarray:
        """Get pointer to knot array"""
        return self.m_knot
    
    def cv_array(self) -> np.ndarray:
        """Get pointer to CV array"""
        return self.m_cv
    
    def is_valid_knot_vector(self) -> bool:
        """Check if knot vector is valid"""
        if len(self.m_knot) != self.knot_count():
            return False
        
        for i in range(len(self.m_knot) - 1):
            if self.m_knot[i] > self.m_knot[i + 1] + Tolerance.ZERO_TOLERANCE:
                return False
        
        return True
    
    #############################################################################
    # DOMAIN & PARAMETERIZATION
    #############################################################################
    
    def domain(self) -> Tuple[float, float]:
        """Get curve domain [start_param, end_param]"""
        if not self.is_valid():
            return (0.0, 0.0)
        return (self.m_knot[self.m_order - 2], self.m_knot[self.m_cv_count - 1])

    def domain_start(self) -> float:
        """Get start of domain"""
        t0, _ = self.domain()
        return t0

    def domain_end(self) -> float:
        """Get end of domain"""
        _, t1 = self.domain()
        return t1

    def domain_middle(self) -> float:
        """Get middle of domain"""
        t0, t1 = self.domain()
        return (t0 + t1) * 0.5

    def set_domain(self, t0: float, t1: float) -> bool:
        """Set curve domain"""
        if not self.is_valid():
            return False
        if t0 >= t1:
            return False

        old_t0, old_t1 = self.domain()
        if abs(old_t1 - old_t0) < Tolerance.ZERO_TOLERANCE:
            return False

        scale = (t1 - t0) / (old_t1 - old_t0)
        for i in range(len(self.m_knot)):
            self.m_knot[i] = t0 + (self.m_knot[i] - old_t0) * scale

        self._invalidate_rmf_cache()
        return True
    
    def get_span_vector(self) -> List[float]:
        """Get span (distinct knot intervals) values"""
        if not self.is_valid():
            return []
        
        spans = []
        for i in range(self.m_order - 2, self.m_cv_count):
            if i == self.m_order - 2 or abs(self.m_knot[i] - self.m_knot[i-1]) > Tolerance.ZERO_TOLERANCE:
                spans.append(self.m_knot[i])
        
        return spans

    #############################################################################
    # KNOT VECTOR OPERATIONS (CONTINUED)
    #############################################################################
    
    def _make_clamped_uniform_knot_vector(self, delta: float = 1.0) -> bool:
        """Make knot vector a clamped uniform knot vector.
        
        Implementation matches OpenNURBS ON_MakeClampedUniformKnotVector.
        """
        if delta <= 0.0:
            return False
        if self.m_order < 2 or self.m_cv_count < self.m_order:
            return False
        
        # Use knot module function
        result = knot.make_clamped_uniform(self.m_order, self.m_cv_count, delta)
        if result is None:
            return False
        self.m_knot = result
        return True
    
    def _make_periodic_uniform_knot_vector(self, delta: float = 1.0) -> bool:
        """Make knot vector a periodic uniform knot vector"""
        if delta <= 0.0:
            return False
        if self.m_order < 2 or self.m_cv_count < self.m_order:
            return False
        
        # Use knot module function
        result = knot.make_periodic_uniform(self.m_order, self.m_cv_count, delta)
        if result is None:
            return False
        self.m_knot = result
        return True
    
    #############################################################################
    # EVALUATION
    #############################################################################
    
    def point_at(self, t: float) -> Point:
        """Evaluate point at parameter t.
        
        Implementation matches OpenNURBS evaluation approach.
        """
        if not self.is_valid():
            return Point(0, 0, 0)
        
        # Find span (returns index relative to shifted knot array)
        span = self._find_span(t)
        if span < 0:
            return Point(0, 0, 0)
        
        # Evaluate using Cox-de Boor algorithm
        N = self._basis_functions(span, t)
        
        # Compute point
        pt = np.zeros(self.m_dim)
        
        if self.m_is_rat:
            # Rational curve: C(t) = Σ(Ni * wi * Pi) / Σ(Ni * wi)
            w = 0.0
            for i in range(self.m_order):
                cv_idx = span + i
                if cv_idx < 0 or cv_idx >= self.m_cv_count:
                    continue
                idx = cv_idx * self.m_cv_stride
                weight = self.m_cv[idx + self.m_dim]
                w += N[i] * weight
                for j in range(self.m_dim):
                    pt[j] += N[i] * self.m_cv[idx + j] * weight

            if abs(w) > 1e-10:
                pt /= w
        else:
            # Non-rational curve
            # In OpenNURBS, span index directly corresponds to CV starting index
            for i in range(self.m_order):
                cv_idx = span + i
                if cv_idx < 0 or cv_idx >= self.m_cv_count:
                    continue
                idx = cv_idx * self.m_cv_stride
                for j in range(self.m_dim):
                    pt[j] += N[i] * self.m_cv[idx + j]
        
        return Point(pt[0], pt[1] if self.m_dim > 1 else 0, pt[2] if self.m_dim > 2 else 0)
    
    def point_at_start(self) -> Point:
        """Evaluate point at curve start"""
        t0, _ = self.domain()
        return self.point_at(t0)
    
    def point_at_end(self) -> Point:
        """Evaluate point at curve end"""
        _, t1 = self.domain()
        return self.point_at(t1)

    def point_at_middle(self) -> Point:
        """Evaluate point at curve middle"""
        return self.point_at(self.domain_middle())

    def tangent_at(self, t: float) -> Vector:
        """Evaluate tangent vector at parameter t (normalized)"""
        if not self.is_valid():
            return Vector(0, 0, 0)

        t0, t1 = self.domain()
        h = (t1 - t0) * 1e-7

        if t <= t0 + h:
            p1 = self.point_at(t0)
            p2 = self.point_at(t0 + h)
        elif t >= t1 - h:
            p1 = self.point_at(t1 - h)
            p2 = self.point_at(t1)
        else:
            p1 = self.point_at(t - h)
            p2 = self.point_at(t + h)

        tan = Vector(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
        mag = tan.magnitude()
        if mag > 1e-14:
            tan.normalize_self()
        return tan

    def frame_at(self, t: float, normalized: bool = True) -> Optional[Tuple[Point, Vector, Vector, Vector]]:
        """Get Frenet frame at parameter t (tangent, normal, binormal)"""
        if not self.is_valid():
            return None

        t0, t1 = self.domain()
        if normalized:
            if t < 0.0 or t > 1.0:
                return None
            param = t0 + t * (t1 - t0)
        else:
            if t < t0 or t > t1:
                return None
            param = t

        h = (t1 - t0) * 1e-5
        origin = self.point_at(param)

        # Handle endpoints with one-sided differences
        if param <= t0 + h:
            p0 = self.point_at(t0)
            pp = self.point_at(t0 + h)
            pp2 = self.point_at(t0 + 2 * h)
            d1 = Vector(pp[0] - p0[0], pp[1] - p0[1], pp[2] - p0[2])
            d2 = Vector(
                (pp2[0] - 2 * pp[0] + p0[0]) / (h * h),
                (pp2[1] - 2 * pp[1] + p0[1]) / (h * h),
                (pp2[2] - 2 * pp[2] + p0[2]) / (h * h)
            )
        elif param >= t1 - h:
            pm = self.point_at(t1 - h)
            p0 = self.point_at(t1)
            pm2 = self.point_at(t1 - 2 * h)
            d1 = Vector(p0[0] - pm[0], p0[1] - pm[1], p0[2] - pm[2])
            d2 = Vector(
                (p0[0] - 2 * pm[0] + pm2[0]) / (h * h),
                (p0[1] - 2 * pm[1] + pm2[1]) / (h * h),
                (p0[2] - 2 * pm[2] + pm2[2]) / (h * h)
            )
        else:
            # Central difference for interior points
            pm = self.point_at(param - h)
            p0 = self.point_at(param)
            pp = self.point_at(param + h)
            d1 = Vector(
                (pp[0] - pm[0]) / (2 * h),
                (pp[1] - pm[1]) / (2 * h),
                (pp[2] - pm[2]) / (2 * h)
            )
            d2 = Vector(
                (pp[0] - 2 * p0[0] + pm[0]) / (h * h),
                (pp[1] - 2 * p0[1] + pm[1]) / (h * h),
                (pp[2] - 2 * p0[2] + pm[2]) / (h * h)
            )

        d1_mag = d1.magnitude()
        if d1_mag < 1e-14:
            return None

        T = d1.normalized()

        d2_dot_T = d2.dot(T)
        N = Vector(d2[0] - d2_dot_T * T[0], d2[1] - d2_dot_T * T[1], d2[2] - d2_dot_T * T[2])
        n_mag = N.magnitude()

        if n_mag < 1e-14:
            world_z = Vector(0, 0, 1)
            N = T.cross(world_z)
            n_mag = N.magnitude()
            if n_mag < 1e-14:
                world_y = Vector(0, 1, 0)
                N = T.cross(world_y)
                n_mag = N.magnitude()

        if n_mag > 1e-14:
            N = N.normalized()

        B = T.cross(N).normalized()

        return (origin, T, N, B)

    def _invalidate_rmf_cache(self):
        self._rmf_cache = None

    def _ensure_rmf_cache(self):
        if self._rmf_cache is not None:
            return

        num_samples = max(20, self.span_count() * 4)
        t0, t1 = self.domain()
        dt = (t1 - t0) / (num_samples - 1)

        params = []
        quaternions = []
        origins = []

        for i in range(num_samples):
            t = t0 + i * dt
            params.append(t)

            result = self._perpendicular_frame_at_internal(t, False)
            if result:
                o, r, s, tangent = result
                origins.append(o)
                quaternions.append(self._frame_to_quaternion(r, s, tangent))
            else:
                origins.append(Point(0, 0, 0))
                quaternions.append([1.0, 0.0, 0.0, 0.0])

        self._rmf_cache = {'params': params, 'quaternions': quaternions, 'origins': origins}

    @staticmethod
    def _frame_to_quaternion(r: Vector, s: Vector, t: Vector) -> List[float]:
        trace = r.x + s.y + t.z

        if trace > 0:
            big_s = math.sqrt(trace + 1.0) * 2
            return [0.25 * big_s, (s.z - t.y) / big_s, (t.x - r.z) / big_s, (r.y - s.x) / big_s]
        elif r.x > s.y and r.x > t.z:
            big_s = math.sqrt(1.0 + r.x - s.y - t.z) * 2
            return [(s.z - t.y) / big_s, 0.25 * big_s, (s.x + r.y) / big_s, (t.x + r.z) / big_s]
        elif s.y > t.z:
            big_s = math.sqrt(1.0 + s.y - r.x - t.z) * 2
            return [(t.x - r.z) / big_s, (s.x + r.y) / big_s, 0.25 * big_s, (t.y + s.z) / big_s]
        else:
            big_s = math.sqrt(1.0 + t.z - r.x - s.y) * 2
            return [(r.y - s.x) / big_s, (t.x + r.z) / big_s, (t.y + s.z) / big_s, 0.25 * big_s]

    @staticmethod
    def _quaternion_to_frame(q: List[float]) -> Tuple[Vector, Vector, Vector]:
        w, x, y, z = q
        r = Vector(1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y))
        s = Vector(2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x))
        t = Vector(2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y))
        return (r, s, t)

    @staticmethod
    def _slerp(q0: List[float], q1: List[float], u: float) -> List[float]:
        dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3]

        q1_adj = q1
        if dot < 0:
            dot = -dot
            q1_adj = [-q1[0], -q1[1], -q1[2], -q1[3]]

        if dot > 0.9995:
            result = [
                q0[0] + u * (q1_adj[0] - q0[0]),
                q0[1] + u * (q1_adj[1] - q0[1]),
                q0[2] + u * (q1_adj[2] - q0[2]),
                q0[3] + u * (q1_adj[3] - q0[3])
            ]
            norm = math.sqrt(sum(r*r for r in result))
            return [r / norm for r in result]

        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        w0 = math.sin((1 - u) * theta) / sin_theta
        w1 = math.sin(u * theta) / sin_theta

        return [
            w0*q0[0] + w1*q1_adj[0],
            w0*q0[1] + w1*q1_adj[1],
            w0*q0[2] + w1*q1_adj[2],
            w0*q0[3] + w1*q1_adj[3]
        ]

    def _perpendicular_frame_at_internal(self, t: float, normalized: bool) -> Optional[Tuple[Point, Vector, Vector, Vector]]:
        """Internal: compute RMF with Frenet initialization (matches Rhino)"""
        if not self.is_valid():
            return None

        t0, t1 = self.domain()
        param = t0 + t * (t1 - t0) if normalized else t
        if normalized and (t < 0.0 or t > 1.0):
            return None
        if not normalized and (t < t0 or t > t1):
            return None

        # Get initial frame at t0 using Frenet (curvature-based)
        derivs0 = self.evaluate(t0, 2)
        D1_0 = Vector(derivs0[1].x, derivs0[1].y, derivs0[1].z)
        D2_0 = Vector(derivs0[2].x, derivs0[2].y, derivs0[2].z)

        D1_0_mag = D1_0.magnitude()
        if D1_0_mag < 1e-14:
            return None

        tangent0 = D1_0 / D1_0_mag

        # Initial normal from curvature (Frenet)
        D2_dot_D1 = D2_0.dot(D1_0)
        D1_0_mag_sq = D1_0_mag * D1_0_mag
        N0_unnorm = Vector(
            D2_0.x - (D2_dot_D1 / D1_0_mag_sq) * D1_0.x,
            D2_0.y - (D2_dot_D1 / D1_0_mag_sq) * D1_0.y,
            D2_0.z - (D2_dot_D1 / D1_0_mag_sq) * D1_0.z
        )

        N0_mag = N0_unnorm.magnitude()
        if N0_mag < 1e-14:
            world_z = Vector(0, 0, 1)
            N0_unnorm = world_z.cross(tangent0)
            N0_mag = N0_unnorm.magnitude()
            if N0_mag < 1e-14:
                world_y = Vector(0, 1, 0)
                N0_unnorm = world_y.cross(tangent0)
                N0_mag = N0_unnorm.magnitude()
        r0 = N0_unnorm / N0_mag

        origin = self.point_at(param)

        # If at start, return Frenet frame directly
        if abs(param - t0) < 1e-14:
            s0 = tangent0.cross(r0).normalized()
            return (origin, r0, s0, tangent0)

        # Propagate frame using Double Reflection (RMF) algorithm
        num_steps = max(10, int((param - t0) / (t1 - t0) * 100))
        dt = (param - t0) / num_steps

        ri = r0
        ti = t0
        xi = self.point_at(ti)
        tangent_i = tangent0

        for _ in range(num_steps):
            if ti >= param - 1e-14:
                break
            ti_next = min(ti + dt, param)
            xi_next = self.point_at(ti_next)
            tangent_next = self.tangent_at(ti_next).normalized()

            v1 = Vector(xi_next.x - xi.x, xi_next.y - xi.y, xi_next.z - xi.z)
            c1 = v1.dot(v1)
            if c1 < 1e-28:
                ti, xi, tangent_i = ti_next, xi_next, tangent_next
                continue

            ri_dot_v1 = ri.dot(v1)
            r_l = Vector(
                ri.x - 2.0 * ri_dot_v1 / c1 * v1.x,
                ri.y - 2.0 * ri_dot_v1 / c1 * v1.y,
                ri.z - 2.0 * ri_dot_v1 / c1 * v1.z
            )

            ti_dot_v1 = tangent_i.dot(v1)
            t_l = Vector(
                tangent_i.x - 2.0 * ti_dot_v1 / c1 * v1.x,
                tangent_i.y - 2.0 * ti_dot_v1 / c1 * v1.y,
                tangent_i.z - 2.0 * ti_dot_v1 / c1 * v1.z
            )

            v2 = Vector(tangent_next.x - t_l.x, tangent_next.y - t_l.y, tangent_next.z - t_l.z)
            c2 = v2.dot(v2)
            if c2 < 1e-28:
                ri = r_l
            else:
                rl_dot_v2 = r_l.dot(v2)
                ri = Vector(
                    r_l.x - 2.0 * rl_dot_v2 / c2 * v2.x,
                    r_l.y - 2.0 * rl_dot_v2 / c2 * v2.y,
                    r_l.z - 2.0 * rl_dot_v2 / c2 * v2.z
                )

            ri = ri.normalized()
            ti, xi, tangent_i = ti_next, xi_next, tangent_next

        tangent = self.tangent_at(param).normalized()
        ri_dot_t = ri.dot(tangent)
        ri = Vector(ri.x - ri_dot_t * tangent.x, ri.y - ri_dot_t * tangent.y, ri.z - ri_dot_t * tangent.z).normalized()
        s = tangent.cross(ri).normalized()

        return (origin, ri, s, tangent)

    def perpendicular_frame_at(self, t: float, normalized: bool = True) -> Optional[Tuple[Point, Vector, Vector, Vector]]:
        """Get rotation minimizing perpendicular frame at parameter t
        Uses the exact Double Reflection algorithm for accuracy"""
        return self._perpendicular_frame_at_internal(t, normalized)

    def get_perpendicular_frames(self, params: List[float]) -> List[Tuple[Point, Vector, Vector, Vector]]:
        """Get multiple perpendicular frames along the curve"""
        return [f for t in params if (f := self.perpendicular_frame_at(t, True)) is not None]

    def _find_span(self, t: float) -> int:
        """Find knot span index for parameter t using binary search.
        
        Implementation matches OpenNURBS ON_NurbsSpanIndex.
        
        Returns
        -------
        int
            Span index relative to shifted knot array (0-based from domain start)
        """
        if not self.is_valid():
            return -1
        
        # Use knot module function
        return knot.find_span(self.m_order, self.m_cv_count, self.m_knot, t)
    
    def _basis_functions(self, span: int, t: float) -> np.ndarray:
        """Compute non-zero basis functions at parameter t.
        
        Implementation matches OpenNURBS Cox-de Boor algorithm.
        
        Parameters
        ----------
        span : int
            Knot span index from _find_span() (relative to shifted array).
        t : float
            Parameter value.
            
        Returns
        -------
        np.ndarray
            Array of m_order non-zero basis function values.
        """
        N = np.zeros(self.m_order)
        left = np.zeros(self.m_order)
        right = np.zeros(self.m_order)
        
        # Offset knot pointer like OpenNURBS does
        offset = self.m_order - 2 + span
        
        N[0] = 1.0
        
        for j in range(1, self.m_order):
            left[j] = t - self.m_knot[offset + 1 - j]
            right[j] = self.m_knot[offset + j] - t
            saved = 0.0
            
            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            
            N[j] = saved
        
        return N

    def _basis_functions_derivatives(self, span: int, t: float, deriv_order: int) -> np.ndarray:
        """Compute basis function derivatives at parameter t.

        Algorithm A2.3 from "The NURBS Book" (Piegl & Tiller).
        Matches OpenNURBS/Rhino implementation.

        Parameters
        ----------
        span : int
            Knot span index from _find_span().
        t : float
            Parameter value.
        deriv_order : int
            Maximum derivative order.

        Returns
        -------
        np.ndarray
            2D array [deriv_order+1, m_order] of basis function derivatives.
        """
        p = self.degree()
        n_der = min(deriv_order, p)

        ders = np.zeros((n_der + 1, p + 1))
        left = np.zeros(p + 1)
        right = np.zeros(p + 1)
        ndu = np.zeros((p + 1, p + 1))

        # Offset knot pointer like OpenNURBS
        offset = self.m_order - 2 + span

        ndu[0, 0] = 1.0
        for j in range(1, p + 1):
            left[j] = t - self.m_knot[offset + 1 - j]
            right[j] = self.m_knot[offset + j] - t
            saved = 0.0
            for r in range(j):
                # Store knot differences in ndu[j, r] for derivative computation
                ndu[j, r] = right[r + 1] + left[j - r]
                temp = ndu[r, j - 1] / ndu[j, r] if abs(ndu[j, r]) > 1e-14 else 0.0
                ndu[r, j] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            ndu[j, j] = saved

        # Load basis functions
        for j in range(p + 1):
            ders[0, j] = ndu[j, p]

        # Compute derivatives using Eq. 2.10 from The NURBS Book
        a = np.zeros((2, p + 1))
        for r in range(p + 1):
            s1 = 0
            s2 = 1
            a[0, 0] = 1.0

            for k in range(1, n_der + 1):
                d = 0.0
                rk = r - k
                pk = p - k

                if r >= k:
                    a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                    d = a[s2, 0] * ndu[rk, pk]

                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else p - r

                for j in range(j1, j2 + 1):
                    a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                    d += a[s2, j] * ndu[rk + j, pk]

                if r <= pk:
                    a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                    d += a[s2, k] * ndu[r, pk]

                ders[k, r] = d
                s1, s2 = s2, s1

        # Apply factorial scaling: p!/(p-k)! (falling factorial)
        factor = float(p)
        for k in range(1, n_der + 1):
            for j in range(p + 1):
                ders[k, j] *= factor
            factor *= (p - k)

        return ders

    #############################################################################
    # GEOMETRIC QUERIES
    #############################################################################
    
    def is_closed(self) -> bool:
        """Check if curve is closed"""
        if not self.is_valid():
            return False
        
        p_start = self.point_at_start()
        p_end = self.point_at_end()
        return p_start.distance(p_end) < Tolerance.ZERO_TOLERANCE
    
    def is_periodic(self) -> bool:
        """Check if curve is periodic"""
        if not self.is_valid():
            return False
        
        # Check if knots and CVs wrap around
        if not self.is_closed():
            return False
        
        # Check if first order-1 CVs match last order-1 CVs
        for i in range(self.m_order - 1):
            p1 = self.get_cv(i)
            p2 = self.get_cv(self.m_cv_count - self.m_order + 1 + i)
            if p1 and p2 and p1.distance(p2) > Tolerance.ZERO_TOLERANCE:
                return False
        
        return True
    
    def length(self) -> float:
        """Compute curve length using Gauss-Legendre quadrature"""
        if not self.is_valid():
            return 0.0

        GL_X = [
            -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
            -0.4333953941292472, -0.1488743389816312,
             0.1488743389816312,  0.4333953941292472,  0.6794095682990244,
             0.8650633666889845,  0.9739065285171717
        ]
        GL_W = [
            0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
            0.2692667193099963, 0.2955242247147529,
            0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
            0.1494513491505806, 0.0666713443086881
        ]

        total = 0.0
        n_spans = self.span_count()
        SUBDIVISIONS = 4

        for span in range(n_spans):
            span_a = self.m_knot[self.m_order - 2 + span]
            span_b = self.m_knot[self.m_order - 1 + span]
            if span_b <= span_a:
                continue

            span_width = (span_b - span_a) / SUBDIVISIONS
            for sub in range(SUBDIVISIONS):
                a = span_a + sub * span_width
                b = a + span_width
                mid = (a + b) * 0.5
                half = (b - a) * 0.5
                s = 0.0

                for i in range(10):
                    t = mid + half * GL_X[i]
                    derivs = self.evaluate(t, 1)
                    s += GL_W[i] * derivs[1].magnitude()

                total += half * s

        return total
    
    #############################################################################
    # MODIFICATION OPERATIONS
    #############################################################################
    
    def make_rational(self) -> bool:
        """Convert to rational curve"""
        if self.m_is_rat:
            return True
        
        new_stride = self.m_dim + 1
        new_cv = np.zeros(self.m_cv_count * new_stride)
        
        for i in range(self.m_cv_count):
            old_idx = i * self.m_cv_stride
            new_idx = i * new_stride
            
            for j in range(self.m_dim):
                new_cv[new_idx + j] = self.m_cv[old_idx + j]
            new_cv[new_idx + self.m_dim] = 1.0  # Weight
        
        self.m_is_rat = 1
        self.m_cv_stride = new_stride
        self.m_cv = new_cv
        
        return True
    
    def make_non_rational(self, force: bool = False) -> bool:
        """Convert to non-rational curve.

        If force=False (default), fails when weights differ.
        If force=True, sets all weights to 1.0 (changes geometry!).
        """
        if not self.m_is_rat:
            return True

        if force:
            for i in range(self.m_cv_count):
                idx = i * self.m_cv_stride
                self.m_cv[idx + self.m_dim] = 1.0
        else:
            w0 = self.weight(0)
            for i in range(1, self.m_cv_count):
                if abs(self.weight(i) - w0) > Tolerance.ZERO_TOLERANCE:
                    return False

        new_stride = self.m_dim
        new_cv = np.zeros(self.m_cv_count * new_stride)

        for i in range(self.m_cv_count):
            old_idx = i * self.m_cv_stride
            new_idx = i * new_stride
            for j in range(self.m_dim):
                new_cv[new_idx + j] = self.m_cv[old_idx + j]

        self.m_is_rat = 0
        self.m_cv_stride = new_stride
        self.m_cv = new_cv

        return True
    
    def reverse(self) -> bool:
        """Reverse curve direction"""
        if not self.is_valid():
            return False

        t0, t1 = self.domain()
        for i in range(len(self.m_knot)):
            self.m_knot[i] = t0 + t1 - self.m_knot[i]
        self.m_knot = np.flip(self.m_knot).copy()

        cvs = self.cv_size()
        for i in range(self.m_cv_count // 2):
            j = self.m_cv_count - 1 - i
            for k in range(cvs):
                temp = self.m_cv[i * cvs + k]
                self.m_cv[i * cvs + k] = self.m_cv[j * cvs + k]
                self.m_cv[j * cvs + k] = temp

        self._invalidate_rmf_cache()
        return True
    
    #############################################################################
    # CONVERSION METHODS
    #############################################################################
    
    def divide_by_count(self, count: int, include_endpoints: bool = True) -> Tuple[List[Point], List[float]]:
        """Divide curve into uniform arc-length segments.

        Parameters
        ----------
        count : int
            Number of points to generate (must be >= 2).
        include_endpoints : bool, optional
            If True, includes curve endpoints in the result. Defaults to True.

        Returns
        -------
        tuple of (list of Point, list of float)
            The points and their parameters on the curve.
        """
        points = []
        params = []

        if not self.is_valid() or count < 2:
            return points, params

        t0, t1 = self.domain()
        dom_len = t1 - t0
        h = dom_len * 1e-8

        # Gauss-Legendre 5-point nodes and weights for [-1, 1]
        GL_NODES = [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
        GL_WEIGHTS = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]

        def derivative_at(t):
            if t <= t0 + h:
                p1 = self.point_at(t0)
                p2 = self.point_at(t0 + h)
                dt = h
            elif t >= t1 - h:
                p1 = self.point_at(t1 - h)
                p2 = self.point_at(t1)
                dt = h
            else:
                p1 = self.point_at(t - h)
                p2 = self.point_at(t + h)
                dt = 2.0 * h
            return Vector((p2[0] - p1[0]) / dt, (p2[1] - p1[1]) / dt, (p2[2] - p1[2]) / dt)

        def arc_length_gauss(ta, tb):
            mid = (ta + tb) * 0.5
            half = (tb - ta) * 0.5
            total = 0.0
            for i in range(5):
                t = mid + half * GL_NODES[i]
                total += GL_WEIGHTS[i] * derivative_at(t).magnitude()
            return half * total

        # Build arc-length table with high resolution
        n_samples = max(1000, count * 100)
        dt = (t1 - t0) / n_samples

        t_vals = [t0 + i * dt for i in range(n_samples + 1)]
        s_vals = [0.0]

        for i in range(1, n_samples + 1):
            s_vals.append(s_vals[i - 1] + arc_length_gauss(t_vals[i - 1], t_vals[i]))

        total_len = s_vals[n_samples]
        n_segs = (count - 1) if include_endpoints else (count + 1)
        seg_len = total_len / n_segs

        def find_t_at_s(s_target):
            if s_target <= 0.0:
                return t0
            if s_target >= total_len:
                return t1

            # Binary search for bracket
            lo, hi = 0, n_samples
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if s_vals[mid] < s_target:
                    lo = mid
                else:
                    hi = mid

            # Linear interpolation for initial guess
            frac = (s_target - s_vals[lo]) / (s_vals[hi] - s_vals[lo])
            t = t_vals[lo] + frac * (t_vals[hi] - t_vals[lo])

            # Newton-Raphson refinement
            t_lo, t_hi = t_vals[lo], t_vals[hi]
            for _ in range(20):
                s_cur = s_vals[lo] + arc_length_gauss(t_vals[lo], t)
                error = s_cur - s_target

                if abs(error) < 1e-12:
                    break

                speed = derivative_at(t).magnitude()
                if speed < 1e-14:
                    if error > 0:
                        t_hi = t
                        t = (t_lo + t_hi) * 0.5
                    else:
                        t_lo = t
                        t = (t_lo + t_hi) * 0.5
                    continue

                t_new = t - error / speed
                if t_new <= t_lo or t_new >= t_hi:
                    if error > 0:
                        t_hi = t
                        t = (t_lo + t_hi) * 0.5
                    else:
                        t_lo = t
                        t = (t_lo + t_hi) * 0.5
                else:
                    t = t_new

            return t

        for i in range(count):
            if include_endpoints:
                s_target = seg_len * i
            else:
                s_target = seg_len * (i + 1)

            t = find_t_at_s(s_target)
            points.append(self.point_at(t))
            params.append(t)

        return points, params
    
    def _zero_cvs(self) -> bool:
        """Zero all control vertices and set weights to 1 if rational.
        
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        
        self.m_cv.fill(0.0)
        
        if self.m_is_rat:
            for i in range(self.m_cv_count):
                self.m_cv[i * self.m_cv_stride + self.m_dim] = 1.0
        
        return True

    def insert_knot(self, knot_value: float, knot_multiplicity: int = 1) -> bool:
        if not self.is_valid():
            return False

        p = self.degree()
        if knot_multiplicity < 1 or knot_multiplicity > p:
            return False

        d0, d1 = self.domain()
        if knot_value < d0 or knot_value > d1:
            return False

        # Handle end knots
        if knot_value == d0:
            if knot_multiplicity == p:
                return self.clamp_end(0)
            if knot_multiplicity == 1:
                return True
            return False
        if knot_value == d1:
            if knot_multiplicity == p:
                return self.clamp_end(1)
            if knot_multiplicity == 1:
                return True
            return False

        import numpy as np
        import math

        n = self.m_cv_count - 1
        full_knot_count = self.m_cv_count + self.m_order

        for insert_iter in range(knot_multiplicity):
            # Build full knot vector
            U = np.zeros(full_knot_count)
            U[0] = self.m_knot[0]
            for i in range(len(self.m_knot)):
                U[i + 1] = self.m_knot[i]
            U[full_knot_count - 1] = self.m_knot[-1]

            # Count current multiplicity
            tol = (abs(d0) + abs(d1) + abs(d1 - d0)) * math.sqrt(np.finfo(float).eps)
            mult = sum(1 for i in range(full_knot_count) if abs(U[i] - knot_value) <= tol)
            if mult >= p:
                return False

            # Find span
            span = self._find_span(knot_value)
            k = span + self.m_order - 1

            # Single-knot insertion
            m_full = full_knot_count - 1
            new_full_knot_count = full_knot_count + 1
            new_cv_count = self.m_cv_count + 1

            U_new = np.zeros(new_full_knot_count)
            cv_new = np.zeros(new_cv_count * self.m_cv_stride)

            # Copy unaffected knots
            for i in range(k + 1):
                U_new[i] = U[i]
            U_new[k + 1] = knot_value
            for i in range(k + 1, m_full + 1):
                U_new[i + 1] = U[i]

            # Copy unaffected CVs before
            for i in range(k - p + 1):
                src = i * self.m_cv_stride
                dst = i * self.m_cv_stride
                cv_new[dst:dst + self.m_cv_stride] = self.m_cv[src:src + self.m_cv_stride]

            # Copy unaffected CVs after
            for i in range(k + 1, n + 2):
                src = (i - 1) * self.m_cv_stride
                dst = i * self.m_cv_stride
                cv_new[dst:dst + self.m_cv_stride] = self.m_cv[src:src + self.m_cv_stride]

            # Compute new CVs in affected region
            for i in range(k - p + 1, k + 1):
                denom = U[i + p] - U[i]
                alpha = (knot_value - U[i]) / denom if denom != 0.0 else 0.0

                src_prev = (i - 1) * self.m_cv_stride
                src_curr = i * self.m_cv_stride
                dst = i * self.m_cv_stride

                for d in range(self.m_cv_stride):
                    cv_new[dst + d] = (1.0 - alpha) * self.m_cv[src_prev + d] + alpha * self.m_cv[src_curr + d]

            # Update internal state
            self.m_cv_count = new_cv_count
            self.m_cv = cv_new

            new_compressed_knot_count = self.m_order + self.m_cv_count - 2
            self.m_knot = np.array([U_new[i + 1] for i in range(new_compressed_knot_count)])

            full_knot_count = new_full_knot_count
            n = self.m_cv_count - 1

        return True

    def is_clamped(self, end: int = 2) -> bool:
        """Check if knot vector is clamped at ends.
        
        Parameters
        ----------
        end : int, optional
            0 for start, 1 for end, 2 for both. Defaults to 2.
            
        Returns
        -------
        bool
            True if clamped at specified end(s).
        """
        if not self.is_valid():
            return False
        
        # Use knot module function
        return knot.is_clamped(self.m_order, self.m_cv_count, self.m_knot, end)
    
    def greville_abcissa(self, cv_index: int) -> float:
        """Get Greville abcissa for a control point.

        Parameters
        ----------
        cv_index : int
            Index of the control vertex.

        Returns
        -------
        float
            The Greville abcissa parameter value.
        """
        if cv_index < 0 or cv_index >= self.m_cv_count:
            return 0.0

        knot = self.m_knot[cv_index:]
        order = self.m_order

        if order <= 2 or knot[0] == knot[order - 2]:
            return float(knot[0])

        p = order - 1
        k0 = knot[0]
        k = knot[p // 2]
        k1 = knot[p - 1]
        tol = (k1 - k0) * 1.490116119385e-8

        g = sum(knot[i] for i in range(p)) / p

        if abs(2.0 * k - (k0 + k1)) <= tol and abs(g - k) <= (abs(g) * 1.490116119385e-8 + tol):
            g = k

        return float(g)
    
    def get_greville_abcissae(self) -> List[float]:
        """Get all Greville abcissae.
        
        Returns
        -------
        list of float
            Greville parameters for all control vertices.
        """
        return [self.greville_abcissa(i) for i in range(self.m_cv_count)]
    
    def is_linear(self, tolerance: float = None) -> bool:
        """Check if curve is a straight line.
        
        Parameters
        ----------
        tolerance : float, optional
            Maximum deviation from line. Defaults to Tolerance.ZERO_TOLERANCE.
            
        Returns
        -------
        bool
            True if curve is linear within tolerance.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE
        
        if not self.is_valid() or self.m_cv_count < 2:
            return False
        
        p_start = self.point_at_start()
        p_end = self.point_at_end()
        line_length = p_start.distance(p_end)
        
        if line_length < tolerance:
            return True
        
        num_samples = max(20, self.m_cv_count * 2)
        t0, t1 = self.domain()
        dt = (t1 - t0) / num_samples
        
        for i in range(1, num_samples):
            t = t0 + i * dt
            p = self.point_at(t)
            
            v = Vector(p_end.x - p_start.x, p_end.y - p_start.y, p_end.z - p_start.z)
            w = Vector(p.x - p_start.x, p.y - p_start.y, p.z - p_start.z)
            
            c1 = w.dot(v)
            c2 = v.dot(v)
            
            if c2 > Tolerance.ZERO_TOLERANCE:
                b = c1 / c2
                pb = Point(p_start.x + b * v[0], p_start.y + b * v[1], p_start.z + b * v[2])
                dist = p.distance(pb)
                if dist > tolerance:
                    return False
        
        return True
    
    def is_planar(self, tolerance: float = None) -> bool:
        """Check if curve lies in a plane.
        
        Parameters
        ----------
        tolerance : float, optional
            Maximum deviation from plane. Defaults to Tolerance.ZERO_TOLERANCE.
            
        Returns
        -------
        bool
            True if curve is planar within tolerance.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE
        
        if not self.is_valid() or self.m_cv_count < 3:
            return True
        
        p0 = self.get_cv(0)
        p1 = self.get_cv(self.m_cv_count // 2)
        p2 = self.get_cv(self.m_cv_count - 1)
        
        if not (p0 and p1 and p2):
            return False
        
        v1 = Vector(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
        v2 = Vector(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z)
        normal = v1.cross(v2)
        
        if normal.magnitude() < Tolerance.ZERO_TOLERANCE:
            return True
        
        normal = normal.normalize()
        plane = Plane(p0, normal)
        
        for i in range(self.m_cv_count):
            pt = self.get_cv(i)
            if pt:
                v = Vector(pt.x - plane.origin.x, pt.y - plane.origin.y, pt.z - plane.origin.z)
                dist = abs(v.dot(plane.z_axis))
                if dist > tolerance:
                    return False
        
        return True
    
    def closest_point(self, test_point: Point, tolerance: float = None) -> Tuple[Point, float]:
        """Find closest point on curve to test point.
        
        Parameters
        ----------
        test_point : Point
            The point to find the closest curve point to.
        tolerance : float, optional
            Convergence tolerance. Defaults to Tolerance.ZERO_TOLERANCE.
            
        Returns
        -------
        tuple of (Point, float)
            The closest point and its parameter value.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE
        
        if not self.is_valid():
            return Point(0, 0, 0), 0.0
        
        t0, t1 = self.domain()
        num_samples = max(50, self.m_cv_count * 5)
        dt = (t1 - t0) / num_samples
        
        min_dist = float('inf')
        best_t = t0
        
        for i in range(num_samples + 1):
            t = t0 + i * dt
            p = self.point_at(t)
            dist = test_point.distance(p)
            if dist < min_dist:
                min_dist = dist
                best_t = t
        
        t = best_t
        for _ in range(20):
            pt = self.point_at(t)
            tan = self.tangent_at(t)
            
            v = Vector(test_point.x - pt.x, test_point.y - pt.y, test_point.z - pt.z)
            dt_step = v.dot(tan) / max(tan.dot(tan), Tolerance.ZERO_TOLERANCE)
            
            t_new = t + dt_step
            t_new = max(t0, min(t1, t_new))
            
            if abs(t_new - t) < tolerance:
                break
            
            t = t_new
        
        return self.point_at(t), t
    
    def change_dimension(self, desired_dimension: int) -> bool:
        """Change the dimension of the curve.
        
        Parameters
        ----------
        desired_dimension : int
            Target dimension (must be >= 1).
            
        Returns
        -------
        bool
            True if successful.
        """
        if desired_dimension < 1:
            return False
        if desired_dimension == self.m_dim:
            return True
        
        new_stride = (desired_dimension + 1) if self.m_is_rat else desired_dimension
        new_cv = np.zeros(self.m_cv_count * new_stride)
        
        copy_dim = min(self.m_dim, desired_dimension)
        
        for i in range(self.m_cv_count):
            old_idx = i * self.m_cv_stride
            new_idx = i * new_stride
            
            for j in range(copy_dim):
                new_cv[new_idx + j] = self.m_cv[old_idx + j]
            
            if self.m_is_rat:
                new_cv[new_idx + desired_dimension] = self.m_cv[old_idx + self.m_dim]
        
        self.m_dim = desired_dimension
        self.m_cv_stride = new_stride
        self.m_cv = new_cv
        
        return True
    
    def increase_degree(self, desired_degree: int) -> bool:
        """Increase the degree of the curve using degree elevation.
        
        Parameters
        ----------
        desired_degree : int
            Target degree (must be >= current degree).
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        if desired_degree <= self.degree():
            return True
        
        degree_inc = desired_degree - self.degree()
        
        for _ in range(degree_inc):
            old_order = self.m_order
            old_cv_count = self.m_cv_count
            new_order = old_order + 1
            new_cv_count = old_cv_count + old_cv_count - old_order + 1
            
            old_knots = self.m_knot.copy()
            old_cvs = self.m_cv.copy()
            
            new_knot_count = new_order + new_cv_count - 2
            new_knots = np.zeros(new_knot_count)
            new_cvs = np.zeros(new_cv_count * self.cv_size())
            
            new_k = 0
            old_k = 0
            while old_k < len(old_knots):
                knot_value = old_knots[old_k]
                mult = 1
                
                while old_k + mult < len(old_knots) and abs(old_knots[old_k + mult] - knot_value) < Tolerance.ZERO_TOLERANCE:
                    mult += 1
                
                for _ in range(mult + 1):
                    if new_k < new_knot_count:
                        new_knots[new_k] = knot_value
                        new_k += 1
                
                old_k += mult
            
            cvs = self.cv_size()
            for i in range(new_cv_count):
                if i == 0:
                    for j in range(cvs):
                        new_cvs[i * cvs + j] = old_cvs[j]
                elif i >= old_cv_count:
                    for j in range(cvs):
                        new_cvs[i * cvs + j] = old_cvs[(old_cv_count - 1) * cvs + j]
                else:
                    alpha = i / new_order
                    for j in range(cvs):
                        cv_prev = old_cvs[(i - 1) * cvs + j] if i - 1 < old_cv_count else 0.0
                        cv_curr = old_cvs[i * cvs + j] if i < old_cv_count else 0.0
                        new_cvs[i * cvs + j] = alpha * cv_prev + (1.0 - alpha) * cv_curr
            
            self.m_order = new_order
            self.m_cv_count = new_cv_count
            self.m_knot = new_knots
            self.m_cv = new_cvs
        
        return True
    
    def trim(self, t0: float, t1: float) -> bool:
        """Trim curve to a parameter sub-interval.
        
        Parameters
        ----------
        t0 : float
            Start parameter.
        t1 : float
            End parameter.
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid() or t0 >= t1:
            return False
        
        domain_t0, domain_t1 = self.domain()
        if t0 < domain_t0 or t1 > domain_t1:
            return False
        
        num_samples = max(20, self.m_cv_count * 2)
        dt = (t1 - t0) / (num_samples - 1)
        
        points = [self.point_at(t0 + i * dt) for i in range(num_samples)]
        
        self.create_clamped_uniform(self.m_dim, self.m_order, points, 1.0)
        self.set_domain(t0, t1)
        
        return True
    
    def divide_by_length(self, segment_length: float) -> Tuple[List[Point], List[float]]:
        """Divide curve by arc length using Gauss-Legendre quadrature.

        Parameters
        ----------
        segment_length : float
            Target length between points.

        Returns
        -------
        tuple of (list of Point, list of float)
            Points and parameters spaced by segment_length.
        """
        points = []
        params = []

        if not self.is_valid() or segment_length <= 0.0:
            return points, params

        t0, t1 = self.domain()
        dom_len = t1 - t0
        h = dom_len * 1e-8

        # Compute derivative (un-normalized) at parameter t
        def derivative_at(t: float) -> Vector:
            if t <= t0 + h:
                p1 = self.point_at(t0)
                p2 = self.point_at(t0 + h)
                dt = h
            elif t >= t1 - h:
                p1 = self.point_at(t1 - h)
                p2 = self.point_at(t1)
                dt = h
            else:
                p1 = self.point_at(t - h)
                p2 = self.point_at(t + h)
                dt = 2.0 * h
            return Vector((p2.x - p1.x) / dt, (p2.y - p1.y) / dt, (p2.z - p1.z) / dt)

        # 5-point Gauss-Legendre nodes and weights for [-1, 1]
        GL_NODES = [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
        GL_WEIGHTS = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]

        # Arc length via Gauss-Legendre quadrature
        def arc_length_gauss(ta: float, tb: float) -> float:
            mid = (ta + tb) * 0.5
            half = (tb - ta) * 0.5
            total = 0.0
            for i in range(5):
                t = mid + half * GL_NODES[i]
                total += GL_WEIGHTS[i] * derivative_at(t).magnitude()
            return half * total

        # Build arc-length table with high resolution
        curve_len = self.length()
        n_samples = max(1000, int(curve_len / segment_length) * 100)
        dt = (t1 - t0) / n_samples

        t_vals = [0.0] * (n_samples + 1)
        s_vals = [0.0] * (n_samples + 1)

        t_vals[0] = t0
        s_vals[0] = 0.0

        for i in range(1, n_samples + 1):
            t_vals[i] = t0 + i * dt
            s_vals[i] = s_vals[i-1] + arc_length_gauss(t_vals[i-1], t_vals[i])

        total_len = s_vals[n_samples]

        # Find parameter at target arc length with Newton-Raphson refinement
        def find_t_at_s(s_target: float) -> float:
            if s_target <= 0.0:
                return t0
            if s_target >= total_len:
                return t1

            # Binary search for bracket
            lo, hi = 0, n_samples
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if s_vals[mid] < s_target:
                    lo = mid
                else:
                    hi = mid

            # Initial guess: linear interpolation
            frac = (s_target - s_vals[lo]) / (s_vals[hi] - s_vals[lo])
            t = t_vals[lo] + frac * (t_vals[hi] - t_vals[lo])

            # Newton-Raphson refinement
            t_lo, t_hi = t_vals[lo], t_vals[hi]
            for _ in range(20):
                s_cur = s_vals[lo] + arc_length_gauss(t_vals[lo], t)
                error = s_cur - s_target

                if abs(error) < 1e-12:
                    break

                speed = derivative_at(t).magnitude()
                if speed < 1e-14:
                    if error > 0:
                        t_hi = t
                        t = (t_lo + t_hi) * 0.5
                    else:
                        t_lo = t
                        t = (t_lo + t_hi) * 0.5
                    continue

                t_new = t - error / speed
                if t_new <= t_lo or t_new >= t_hi:
                    if error > 0:
                        t_hi = t
                        t = (t_lo + t_hi) * 0.5
                    else:
                        t_lo = t
                        t = (t_lo + t_hi) * 0.5
                else:
                    t = t_new

            return t

        # Add points at each segment_length interval
        s = 0.0
        while s <= total_len + 1e-10:
            t = find_t_at_s(s)
            points.append(self.point_at(t))
            params.append(t)
            s += segment_length

        return points, params
    
    def split(self, t: float) -> Tuple[Optional['NurbsCurve'], Optional['NurbsCurve']]:
        """Split curve at parameter t into left and right parts.
        
        Parameters
        ----------
        t : float
            Parameter value to split at.
            
        Returns
        -------
        tuple of (NurbsCurve, NurbsCurve) or (None, None)
            Left and right curves, or None if invalid.
        """
        if not self.is_valid():
            return None, None
        
        t0, t1 = self.domain()
        if t <= t0 or t >= t1:
            return None, None
        
        left_curve = NurbsCurve()
        right_curve = NurbsCurve()
        
        # Left curve: from t0 to t
        num_samples = max(20, self.m_cv_count)
        dt = (t - t0) / (num_samples - 1)
        left_points = [self.point_at(t0 + i * dt) for i in range(num_samples)]
        left_curve.create_clamped_uniform(self.m_dim, self.m_order, left_points, 1.0)
        left_curve.set_domain(t0, t)
        
        # Right curve: from t to t1
        dt = (t1 - t) / (num_samples - 1)
        right_points = [self.point_at(t + i * dt) for i in range(num_samples)]
        right_curve.create_clamped_uniform(self.m_dim, self.m_order, right_points, 1.0)
        right_curve.set_domain(t, t1)
        
        return left_curve, right_curve
    
    def extend(self, t0: float, t1: float) -> bool:
        """Extend curve to include domain [t0, t1].

        Uses de Boor extrapolation matching C++ implementation.

        Parameters
        ----------
        t0 : float
            New start parameter (can be before current start).
        t1 : float
            New end parameter (can be after current end).

        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid() or self.is_closed():
            return False

        domain_t0, domain_t1 = self.domain()
        cv_dim = self.cv_size()
        changed = False

        # Extend start (t0 < current domain start)
        if t0 < domain_t0:
            self.clamp_end(0)
            # Extrapolate using de Boor algorithm
            self._evaluate_nurbs_de_boor_inplace(cv_dim, self.m_order, 0, 1, t0)
            for i in range(self.m_order - 1):
                self.m_knot[i] = t0
            changed = True

        # Extend end (t1 > current domain end)
        if t1 > domain_t1:
            self.clamp_end(1)
            # Extrapolate using de Boor algorithm
            i0 = self.m_cv_count - self.m_order
            self._evaluate_nurbs_de_boor_inplace(cv_dim, self.m_order, i0, -1, t1)
            kc = self.knot_count()
            for i in range(self.m_cv_count - 1, kc):
                self.m_knot[i] = t1
            changed = True

        return changed

    def _evaluate_nurbs_de_boor_inplace(self, cvdim: int, order: int, cv_start: int, direction: int, t: float):
        """Internal de Boor evaluation for curve extension (modifies CVs in place)."""
        if order < 2:
            return

        stride = self.m_cv_stride
        for i in range(1, order):
            k0 = cv_start + i - 1 if direction > 0 else cv_start + order - i
            k1 = k0 + direction

            a = self.m_knot[cv_start + (order - 1 if direction > 0 else 0)]
            b = self.m_knot[cv_start + (i if direction > 0 else order - 1 - i)]

            if abs(b - a) < 1e-14:
                continue

            s = (t - a) / (b - a)

            for j in range(cvdim):
                cv0_val = self.m_cv[k0 * stride + j]
                cv1_val = self.m_cv[k1 * stride + j]
                self.m_cv[k0 * stride + j] = cv0_val + s * (cv0_val - cv1_val)
    
    def swap_coordinates(self, axis_i: int, axis_j: int) -> bool:
        """Swap two coordinate axes.
        
        Parameters
        ----------
        axis_i : int
            First axis index (0=x, 1=y, 2=z).
        axis_j : int
            Second axis index (0=x, 1=y, 2=z).
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        if axis_i < 0 or axis_i >= self.m_dim or axis_j < 0 or axis_j >= self.m_dim:
            return False
        if axis_i == axis_j:
            return True
        
        for i in range(self.m_cv_count):
            idx = i * self.m_cv_stride
            temp = self.m_cv[idx + axis_i]
            self.m_cv[idx + axis_i] = self.m_cv[idx + axis_j]
            self.m_cv[idx + axis_j] = temp
        
        return True
    
    def set_start_point(self, start_point: Point) -> bool:
        """Force curve to start at specified point.
        
        Parameters
        ----------
        start_point : Point
            New start point.
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        
        return self.set_cv(0, start_point)
    
    def set_end_point(self, end_point: Point) -> bool:
        """Force curve to end at specified point.
        
        Parameters
        ----------
        end_point : Point
            New end point.
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        
        return self.set_cv(self.m_cv_count - 1, end_point)

    #############################################################################
    # TRANSFORMATION
    #############################################################################

    def transform(self, xform: Xform = None) -> bool:
        """Apply transformation to the curve.

        Parameters
        ----------
        xform : Xform, optional
            Transformation to apply. If None, uses stored self.xform.

        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False

        xf = xform if xform is not None else self.xform

        for i in range(self.m_cv_count):
            pt = self.get_cv(i)
            if pt:
                xf.transform_point(pt)
                self.set_cv(i, pt)

        return True

    def transformed(self, xform: Xform = None) -> 'NurbsCurve':
        """Get transformed copy of the curve.

        Parameters
        ----------
        xform : Xform, optional
            Transformation to apply. If None, uses stored self.xform.

        Returns
        -------
        NurbsCurve
            Transformed copy of the curve.
        """
        result = self.duplicate()
        result.xform = self.xform.duplicate()

        xf = xform if xform is not None else self.xform
        result.transform(xf)

        return result
    
    def superfluous_knot(self, end: int) -> float:
        """Get superfluous knot value at end.

        Parameters
        ----------
        end : int
            0 for start, 1 for end.

        Returns
        -------
        float
            The superfluous knot value.
        """
        if not self.is_valid():
            return 0.0

        kc = self.knot_count()
        if end == 0:
            # First superfluous knot: reflect first knot across knot[order-2]
            return 2.0 * self.m_knot[0] - self.m_knot[self.m_order - 2]
        else:
            # Last superfluous knot: reflect last knot across knot[cv_count-order]
            return 2.0 * self.m_knot[kc - 1] - self.m_knot[self.m_cv_count - self.m_order]
    
    def is_in_plane(self, test_plane: Plane, tolerance: float = None) -> bool:
        """Check if curve lies in a specific plane.
        
        Parameters
        ----------
        test_plane : Plane
            The plane to test against.
        tolerance : float, optional
            Maximum deviation. Defaults to Tolerance.ZERO_TOLERANCE.
            
        Returns
        -------
        bool
            True if curve lies in the plane.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE
        
        if not self.is_valid():
            return False
        
        # Check all CVs against plane
        for i in range(self.m_cv_count):
            pt = self.get_cv(i)
            if pt:
                v = Vector(pt.x - test_plane.origin.x, pt.y - test_plane.origin.y, pt.z - test_plane.origin.z)
                dist = abs(v.dot(test_plane.z_axis))
                if dist > tolerance:
                    return False
        
        return True
    
    def is_singular(self) -> bool:
        """Check if entire curve is singular (collapsed to a point).
        
        Returns
        -------
        bool
            True if curve is singular.
        """
        if not self.is_valid():
            return False
        
        p_first = self.point_at_start()
        
        # Check if all sample points are at same location
        t0, t1 = self.domain()
        num_samples = max(10, self.m_cv_count)
        dt = (t1 - t0) / num_samples
        
        for i in range(1, num_samples + 1):
            t = t0 + i * dt
            p = self.point_at(t)
            if p_first.distance(p) > Tolerance.ZERO_TOLERANCE:
                return False
        
        return True
    
    def _clean_knots(self, tolerance: float = 0.0) -> bool:
        """Clean up invalid knots (remove duplicates within tolerance).
        
        Parameters
        ----------
        tolerance : float, optional
            Knot comparison tolerance. Defaults to 0.0.
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        
        if tolerance <= 0.0:
            tolerance = Tolerance.ZERO_TOLERANCE
        
        # Remove knots that are too close together
        cleaned_knots = [self.m_knot[0]]
        for i in range(1, len(self.m_knot)):
            if abs(self.m_knot[i] - cleaned_knots[-1]) > tolerance:
                cleaned_knots.append(self.m_knot[i])
        
        if len(cleaned_knots) != len(self.m_knot):
            self.m_knot = np.array(cleaned_knots)
        
        return True
    
    def clamp_end(self, end: int) -> bool:
        """Clamp ends (add multiplicity to end knots).

        Parameters
        ----------
        end : int
            0 for start, 1 for end, 2 for both.

        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid():
            return False
        if end < 0 or end > 2:
            return False

        # Clamp start
        if end == 0 or end == 2:
            t = self.m_knot[self.m_order - 2]
            for i in range(self.m_order - 2):
                self.m_knot[i] = t

        # Clamp end
        if end == 1 or end == 2:
            t = self.m_knot[self.m_cv_count - 1]
            kc = self.knot_count()
            for i in range(self.m_cv_count, kc):
                self.m_knot[i] = t

        return True
    
    def evaluate(self, t: float, derivative_count: int = 0) -> List[Vector]:
        """Evaluate point and derivatives on curve at parameter t.

        Parameters
        ----------
        t : float
            Parameter value.
        derivative_count : int, optional
            Number of derivatives to compute. Defaults to 0 (point only).

        Returns
        -------
        list of Vector
            [point, 1st_derivative, 2nd_derivative, ...].
        """
        result = []

        if not self.is_valid():
            result.append(Vector(0, 0, 0))
            return result

        # Clamp derivative order to degree
        max_derivs = min(derivative_count, self.degree())

        span = self._find_span(t)
        ders = self._basis_functions_derivatives(span, t, max_derivs)

        # Evaluate non-rational or homogeneous coordinates and derivatives
        p = self.degree()
        Aders = [[0.0, 0.0, 0.0, 0.0] for _ in range(max_derivs + 1)]

        for k in range(max_derivs + 1):
            for j in range(p + 1):
                cv_idx = span + j
                if cv_idx < 0 or cv_idx >= self.m_cv_count:
                    continue
                idx = cv_idx * self.m_cv_stride

                Nx = ders[k, j]
                cx = self.m_cv[idx]
                cy = self.m_cv[idx + 1] if self.m_dim > 1 else 0.0
                cz = self.m_cv[idx + 2] if self.m_dim > 2 else 0.0
                wv = self.m_cv[idx + self.m_dim] if self.m_is_rat else 1.0

                Aders[k][0] += Nx * cx * wv
                Aders[k][1] += Nx * cy * wv
                Aders[k][2] += Nx * cz * wv
                Aders[k][3] += Nx * wv

        # Convert from homogeneous derivatives (Aders) to Cartesian derivatives
        Cders = [[0.0, 0.0, 0.0] for _ in range(max_derivs + 1)]

        if not self.m_is_rat:
            # Non-rational: derivatives are directly Aders (w == 1)
            for k in range(max_derivs + 1):
                Cders[k] = [Aders[k][0], Aders[k][1], Aders[k][2]]
        else:
            # Rational: use standard formula (Piegl & Tiller, Eq. 2.28)
            for k in range(max_derivs + 1):
                w = Aders[0][3]
                inv_w = 1.0 / w if w != 0.0 else 0.0

                # Initialize derivative to homogeneous derivative
                Ck_x = Aders[k][0]
                Ck_y = Aders[k][1]
                Ck_z = Aders[k][2]

                # Subtract contributions of weight derivatives
                for j_idx in range(1, k + 1):
                    # Binomial coefficient: k! / (j! * (k-j)!)
                    coeff = 1.0
                    for ii in range(1, j_idx + 1):
                        coeff = coeff * (k - ii + 1) / ii
                    wj = Aders[j_idx][3]
                    Ck_x -= coeff * wj * Cders[k - j_idx][0]
                    Ck_y -= coeff * wj * Cders[k - j_idx][1]
                    Ck_z -= coeff * wj * Cders[k - j_idx][2]

                Ck_x *= inv_w
                Ck_y *= inv_w
                Ck_z *= inv_w
                Cders[k] = [Ck_x, Ck_y, Ck_z]

        # Fill result vectors (0th derivative = point)
        for k in range(max_derivs + 1):
            result.append(Vector(Cders[k][0], Cders[k][1], Cders[k][2]))

        # If caller requested more derivatives than degree, pad with zeros
        for k in range(max_derivs + 1, derivative_count + 1):
            result.append(Vector(0.0, 0.0, 0.0))

        return result

    def to_string(self) -> str:
        """Convert curve to string representation.
        
        Returns
        -------
        str
            String description of the curve.
        """
        return (f"NurbsCurve(dim={self.m_dim}, rational={bool(self.m_is_rat)}, "
                f"order={self.m_order}, cvs={self.m_cv_count}, "
                f"knots={self.knot_count()}, valid={self.is_valid()})")

    #############################################################################
    # STRING REPRESENTATION
    #############################################################################

    def __str__(self) -> str:
        """String representation."""
        return f"NurbsCurve(name={self.name}, degree={self.degree()}, cvs={self.m_cv_count})"

    def __repr__(self) -> str:
        """Representation string."""
        rational_str = "true" if self.m_is_rat else "false"
        lines = [
            "NurbsCurve(",
            f"  name={self.name},",
            f"  degree={self.degree()},",
            f"  cvs={self.m_cv_count},",
            f"  rational={rational_str},",
            "  control_points=["
        ]
        for i in range(self.m_cv_count):
            p = self.get_cv(i)
            lines.append(f"    {p[0]}, {p[1]}, {p[2]}")
        lines.append("  ]")
        lines.append(")")
        return "\n".join(lines)

    def duplicate(self) -> "NurbsCurve":
        """Create a duplicate with a new GUID.

        Returns
        -------
        NurbsCurve
            A copy of the curve with a new GUID.
        """
        import copy
        import uuid
        new_curve = copy.deepcopy(self)
        new_curve.guid = str(uuid.uuid4())
        return new_curve

    def is_arc(self, tolerance: float = None) -> bool:
        """Check if curve is an arc.
        
        Parameters
        ----------
        tolerance : float, optional
            Tolerance for arc test. Defaults to Tolerance.ZERO_TOLERANCE.
            
        Returns
        -------
        bool
            True if curve is an arc.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE
        
        if not self.is_valid() or not self.is_planar(tolerance):
            return False
        
        # Sample curve and check if all points are equidistant from center
        # This is a simplified test
        return False  # Full implementation would compute center and radius
    
    def is_natural(self, end: int = 2) -> bool:
        """Test if curve has natural end (zero 2nd derivative).

        Parameters
        ----------
        end : int, optional
            0 for start, 1 for end, 2 for both. Defaults to 2.

        Returns
        -------
        bool
            True if has natural end.
        """
        if not self.is_valid():
            return False

        tol_factor = 1e-8
        t0, t1 = self.domain()

        # Check start (pass=0) and/or end (pass=1)
        start_pass = 0 if (end == 0 or end == 2) else 1
        end_pass = 2 if (end == 1 or end == 2) else 1

        for pass_idx in range(start_pass, end_pass):
            t = t0 if pass_idx == 0 else t1

            # Evaluate 2nd derivative
            derivs = self.evaluate(t, 2)
            if len(derivs) < 3:
                return False

            d2 = derivs[2]
            d2_len = d2.magnitude()

            # Get control polygon length for tolerance
            if pass_idx == 0:
                cv0 = self.get_cv(0)
                cv2 = self.get_cv(min(2, self.m_cv_count - 1))
            else:
                cv0 = self.get_cv(self.m_cv_count - 1)
                cv2 = self.get_cv(max(0, self.m_cv_count - 3))

            tol = cv0.distance(cv2) * tol_factor

            if d2_len > tol:
                return False

        return True
    
    def is_polyline(self) -> Tuple[bool, List[Point], List[float]]:
        """Check if curve can be represented as a polyline.
        
        Returns
        -------
        tuple of (bool, list of Point, list of float)
            (is_polyline, points, parameters) or (False, [], []).
        """
        if not self.is_valid():
            return False, [], []
        
        # Check if curve is linear
        if self.is_linear():
            points = [self.point_at_start(), self.point_at_end()]
            t0, t1 = self.domain()
            params = [t0, t1]
            return True, points, params
        
        return False, [], []
    
    def to_polyline_adaptive(self, angle_tolerance: float = 0.1, 
                            min_edge_length: float = 0.0,
                            max_edge_length: float = 0.0) -> Tuple[List[Point], List[float]]:
        """Convert curve to polyline with adaptive sampling (curvature-based).

        Parameters
        ----------
        angle_tolerance : float, optional
            Maximum angle between segments in radians. Defaults to 0.1.
        min_edge_length : float, optional
            Minimum distance between points. Defaults to 0.0 (auto).
        max_edge_length : float, optional
            Maximum distance between points. Defaults to 0.0 (auto).

        Returns
        -------
        tuple of (list of Point, list of float)
            Points and parameters.
        """
        if not self.is_valid():
            return [], []

        if angle_tolerance <= 0.0:
            angle_tolerance = 0.1

        t0, t1 = self.domain()
        curve_len = self.length()

        # Set default edge lengths if not specified (matches C++ implementation)
        if max_edge_length <= 0.0:
            max_edge_length = curve_len / 10.0
        if min_edge_length <= 0.0:
            min_edge_length = curve_len / 1000.0
        if min_edge_length > max_edge_length:
            min_edge_length = max_edge_length * 0.1

        # Collect (param, point) pairs using binary subdivision
        samples = [(t0, self.point_at(t0)), (t1, self.point_at(t1))]

        # Work queue: segments to potentially subdivide (ta, tb)
        work_queue = [(t0, t1)]

        max_iterations = 10000
        iterations = 0

        while work_queue and iterations < max_iterations:
            iterations += 1
            ta, tb = work_queue.pop()

            pa = self.point_at(ta)
            pb = self.point_at(tb)
            chord_length = pa.distance(pb)

            if chord_length < min_edge_length:
                continue

            tm = (ta + tb) * 0.5
            pm = self.point_at(tm)

            # Check deviation: distance from midpoint to chord
            chord = Vector(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z)
            to_mid = Vector(pm.x - pa.x, pm.y - pa.y, pm.z - pa.z)
            chord_len_sq = chord.dot(chord)
            deviation = 0.0

            if chord_len_sq > 1e-20:
                proj = to_mid.dot(chord) / chord_len_sq
                projected = Point(pa.x + proj * chord.x, pa.y + proj * chord.y, pa.z + proj * chord.z)
                deviation = pm.distance(projected)

            # Convert angle tolerance to approximate deviation tolerance
            # For small angles: deviation ≈ chord_length * sin(angle/2) ≈ chord_length * angle/2
            deviation_tolerance = chord_length * angle_tolerance * 0.5

            need_subdivide = (deviation > deviation_tolerance) or (chord_length > max_edge_length)

            if need_subdivide:
                samples.append((tm, pm))
                work_queue.append((ta, tm))
                work_queue.append((tm, tb))

        # Sort by parameter
        samples.sort(key=lambda x: x[0])

        # Extract results
        points = [p for _, p in samples]
        params = [t for t, _ in samples]

        return points, params
    
    def _span_is_linear(self, span_index: int, min_length: float = 0.0,
                       tolerance: float = None) -> bool:
        """Check if span is linear within tolerance.
        
        Parameters
        ----------
        span_index : int
            Index of the span.
        min_length : float, optional
            Minimum length to consider. Defaults to 0.0.
        tolerance : float, optional
            Tolerance for linearity. Defaults to Tolerance.ZERO_TOLERANCE.
            
        Returns
        -------
        bool
            True if span is linear.
        """
        if tolerance is None:
            tolerance = Tolerance.ZERO_TOLERANCE
        
        if not self.is_valid():
            return False
        
        spans = self.get_span_vector()
        if span_index < 0 or span_index >= len(spans) - 1:
            return False
        
        t0 = spans[span_index]
        t1 = spans[span_index + 1]
        
        p0 = self.point_at(t0)
        p1 = self.point_at(t1)
        
        length = p0.distance(p1)
        if length < min_length:
            return False
        
        # Check deviation from line
        num_samples = 5
        dt = (t1 - t0) / (num_samples - 1)
        
        for i in range(1, num_samples - 1):
            t = t0 + i * dt
            p = self.point_at(t)
            
            # Distance from point to line
            v = Vector(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
            w = Vector(p.x - p0.x, p.y - p0.y, p.z - p0.z)
            
            c1 = w.dot(v)
            c2 = v.dot(v)
            
            if c2 > Tolerance.ZERO_TOLERANCE:
                b = c1 / c2
                pb = Point(p0.x + b * v.x, p0.y + b * v.y, p0.z + b * v.z)
                dist = p.distance(pb)
                if dist > tolerance:
                    return False
        
        return True
    
    def _span_is_singular(self, span_index: int) -> bool:
        """Check if span is singular (collapsed to a point).
        
        Parameters
        ----------
        span_index : int
            Index of the span.
            
        Returns
        -------
        bool
            True if span is singular.
        """
        if not self.is_valid():
            return False
        
        spans = self.get_span_vector()
        if span_index < 0 or span_index >= len(spans) - 1:
            return False
        
        t0 = spans[span_index]
        t1 = spans[span_index + 1]
        
        p0 = self.point_at(t0)
        p1 = self.point_at(t1)
        
        return p0.distance(p1) < Tolerance.ZERO_TOLERANCE
    
    def _repair_bad_knots(self, tolerance: float = 0.0, repair: bool = True) -> bool:
        """Repair bad knots (too close, high multiplicity).
        
        Parameters
        ----------
        tolerance : float, optional
            Knot tolerance. Defaults to 0.0.
        repair : bool, optional
            If True, repairs knots; if False, only checks. Defaults to True.
            
        Returns
        -------
        bool
            True if knots are valid or repaired.
        """
        if not self.is_valid():
            return False
        
        if repair:
            return self._clean_knots(tolerance)
        
        # Just check
        for i in range(len(self.m_knot) - 1):
            if self.m_knot[i] > self.m_knot[i + 1] + Tolerance.ZERO_TOLERANCE:
                return False
        
        return True
    
    def change_closed_curve_seam(self, t: float) -> bool:
        """Change seam point of closed periodic curve.
        
        Parameters
        ----------
        t : float
            New seam parameter.
            
        Returns
        -------
        bool
            True if successful.
        """
        if not self.is_valid() or not self.is_closed():
            return False
        
        t0, t1 = self.domain()
        if t < t0 or t > t1:
            return False
        
        # This is a complex operation
        # Would require reparameterization and CV reordering
        return False  # Stub for now
    
    def _get_parameter_tolerance(self, t: float) -> Tuple[float, float]:
        """Get parameter tolerance at point.
        
        Parameters
        ----------
        t : float
            Parameter value.
            
        Returns
        -------
        tuple of (float, float)
            (t_minus, t_plus) tolerance bounds.
        """
        if not self.is_valid():
            return (0.0, 0.0)
        
        # Simple implementation: use small epsilon
        eps = Tolerance.ZERO_TOLERANCE * 10.0
        return (t - eps, t + eps)
    
    ###########################################################################################
    # JSON Serialization
    ###########################################################################################

    def __jsondump__(self):
        """Return a JSON-serializable dictionary representation (matches C++ format)."""
        control_points = []
        for i in range(self.m_cv_count):
            p = self.get_cv(i)
            if p:
                control_points.append([p[0], p[1], p[2]])
            else:
                control_points.append([0.0, 0.0, 0.0])
        return {
            "control_points": control_points,
            "cv_count": int(self.m_cv_count),
            "cv_stride": int(self.m_cv_stride),
            "dimension": int(self.m_dim),
            "guid": self.guid,
            "is_rational": self.m_is_rat != 0,
            "knots": self.m_knot.tolist() if hasattr(self.m_knot, 'tolist') else list(self.m_knot),
            "linecolor": self.linecolor.__jsondump__(),
            "name": self.name,
            "order": int(self.m_order),
            "width": float(self.width),
            "xform": self.xform.__jsondump__(),
        }

    @classmethod
    def __jsonload__(cls, data):
        """Create NurbsCurve from JSON dictionary (accepts C++ format)."""
        curve = cls()
        curve.guid = data.get("guid", curve.guid)
        curve.name = data.get("name", curve.name)
        curve.width = data.get("width", 1.0)
        if "linecolor" in data:
            curve.linecolor = Color.__jsonload__(data["linecolor"])
        if "xform" in data:
            curve.xform = Xform.__jsonload__(data["xform"])
        curve.m_dim = data.get("dimension", 0)
        curve.m_is_rat = 1 if data.get("is_rational", False) else 0
        curve.m_order = data.get("order", 0)
        curve.m_cv_count = data.get("cv_count", 0)
        curve.m_cv_stride = data.get("cv_stride", curve.m_dim + (1 if curve.m_is_rat else 0))
        curve.m_knot = np.array(data.get("knots", []), dtype=np.float64)
        control_points = data.get("control_points", [])
        flat_cv = []
        for cp in control_points:
            flat_cv.extend(cp[:curve.m_cv_stride])
        curve.m_cv = np.array(flat_cv, dtype=np.float64)
        return curve

    def json_dump(self, filepath):
        """Write JSON to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        """Read JSON from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    ###########################################################################################
    # Protobuf Serialization
    ###########################################################################################

    def protobuf_dump(self, filepath):
        """Write protobuf binary to file."""
        try:
            from .proto import nurbscurve_pb2
            from .proto import color_pb2
            from .proto import xform_pb2
            proto = nurbscurve_pb2.NurbsCurve()
            proto.guid = self.guid
            proto.name = self.name
            proto.dimension = int(self.m_dim)
            proto.is_rational = bool(self.m_is_rat)
            proto.order = int(self.m_order)
            proto.cv_count = int(self.m_cv_count)
            proto.cv_stride = int(self.m_cv_stride)
            proto.knots.extend(self.m_knot.tolist() if hasattr(self.m_knot, 'tolist') else list(self.m_knot))
            proto.cvs.extend(self.m_cv.tolist() if hasattr(self.m_cv, 'tolist') else list(self.m_cv))
            proto.width = float(self.width)
            proto.linecolor.guid = self.linecolor.guid
            proto.linecolor.r = int(self.linecolor.r)
            proto.linecolor.g = int(self.linecolor.g)
            proto.linecolor.b = int(self.linecolor.b)
            proto.linecolor.a = int(self.linecolor.a)
            proto.linecolor.name = self.linecolor.name
            proto.xform.guid = self.xform.guid
            proto.xform.name = self.xform.name
            proto.xform.matrix.extend(self.xform.m.flatten().tolist() if hasattr(self.xform.m, 'flatten') else list(self.xform.m))
            with open(filepath, 'wb') as f:
                f.write(proto.SerializeToString())
        except ImportError:
            raise ImportError("protobuf not available - run ./protobuf.sh to install")

    @classmethod
    def protobuf_load(cls, filepath):
        """Read protobuf binary from file."""
        try:
            from .proto import nurbscurve_pb2
            proto = nurbscurve_pb2.NurbsCurve()
            with open(filepath, 'rb') as f:
                proto.ParseFromString(f.read())
            curve = cls()
            curve.guid = proto.guid
            curve.name = proto.name
            curve.m_dim = proto.dimension
            curve.m_is_rat = 1 if proto.is_rational else 0
            curve.m_order = proto.order
            curve.m_cv_count = proto.cv_count
            curve.m_cv_stride = proto.cv_stride
            curve.m_knot = np.array(list(proto.knots), dtype=np.float64)
            curve.m_cv = np.array(list(proto.cvs), dtype=np.float64)
            curve.width = proto.width if proto.width != 0.0 else 1.0
            if proto.HasField('linecolor'):
                curve.linecolor = Color(proto.linecolor.r, proto.linecolor.g,
                                         proto.linecolor.b, proto.linecolor.a)
                curve.linecolor.guid = proto.linecolor.guid
                curve.linecolor.name = proto.linecolor.name
            if proto.HasField('xform'):
                curve.xform = Xform()
                curve.xform.guid = proto.xform.guid
                curve.xform.name = proto.xform.name
                if proto.xform.matrix:
                    curve.xform.m = np.array(list(proto.xform.matrix), dtype=np.float64).reshape(4, 4)
            return curve
        except ImportError:
            raise ImportError("protobuf not available - run ./protobuf.sh to install")
