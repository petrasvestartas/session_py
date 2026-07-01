"""NurbsKnot vector utility functions for NURBS curves and surfaces.

This module provides standalone functions for working with nurbsknot vectors,
following the OpenNURBS pattern (opennurbs_nurbsknot.h/cpp).

These functions operate on numpy arrays and can be used independently
or called by NurbsCurve and NurbsSurface classes.
"""

import numpy as np
from typing import Tuple, Optional, List
from enum import IntEnum
import math


class CurveNurbsKnotStyle(IntEnum):
    """NurbsKnot spacing style for interpolated curves (matches Rhino's CurveNurbsKnotStyle)."""
    Uniform = 0              # Parameter spacing = 1.0
    Chord = 1                # Chord-length parameterization
    ChordSquareRoot = 2      # Centripetal (sqrt chord) parameterization
    UniformPeriodic = 3      # Periodic + uniform
    ChordPeriodic = 4        # Periodic + chord
    ChordSquareRootPeriodic = 5  # Periodic + centripetal


class CurveInterpStyle(IntEnum):
    """End-tangent (boundary) condition for cubic interpolation.

    Both styles share chord-length parameters and clamped knots; they differ only
    in how the start/end tangents (and hence the 2nd/penultimate control points)
    are estimated:
      - Rhino: normalized Bessel tangents (matches Rhino / OpenNURBS).
      - Occt:  un-normalized derivative of the cubic Lagrange polynomial through the
               first/last 4 points (matches OCCT GeomAPI_Interpolate::BuildTangents).
    """
    Rhino = 0
    Occt = 1


def nurbsknot_count(order: int, cv_count: int) -> int:
    """Compute the number of nurbsknots in a nurbsknot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1), must be >= 2.
    cv_count : int
        Number of control vertices, must be >= order.
    
    Returns
    -------
    int
        Number of nurbsknots: order + cv_count - 2
    
    Notes
    -----
    OpenNURBS uses a nurbsknot vector convention that omits the first and last
    "superfluous" nurbsknots that would be present in a standard B-spline nurbsknot vector.
    """
    return order + cv_count - 2


def domain_tolerance(a: float, b: float) -> float:
    """Compute tolerance associated with a domain interval.
    
    Parameters
    ----------
    a : float
        Start of domain.
    b : float
        End of domain.
    
    Returns
    -------
    float
        Tolerance value.
    """
    if a == b:
        return 0.0
    SQRT_EPSILON = 1.4901161193847656e-08  # sqrt of machine epsilon
    EPSILON = 2.220446049250313e-16
    tol = (abs(a) + abs(b) + abs(a - b)) * SQRT_EPSILON
    if tol < EPSILON:
        tol = EPSILON
    return tol


def make_clamped_uniform(order: int, cv_count: int, delta: float = 1.0) -> Optional[np.ndarray]:
    """Create a clamped uniform nurbsknot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1), must be >= 2.
    cv_count : int
        Number of control vertices, must be >= order.
    delta : float, optional
        Spacing between interior nurbsknots. Default is 1.0.
    
    Returns
    -------
    np.ndarray or None
        Clamped uniform nurbsknot vector with domain [0, (cv_count - order + 1) * delta],
        or None if parameters are invalid.
    
    Notes
    -----
    Implements ON_MakeClampedUniformNurbsKnotVector from OpenNURBS.
    The resulting nurbsknot vector has:
    - First (order-1) nurbsknots clamped to 0
    - Interior nurbsknots uniformly spaced by delta
    - Last (order-1) nurbsknots clamped to the final value
    """
    if order < 2 or cv_count < order or delta <= 0.0:
        return None
    
    kc = nurbsknot_count(order, cv_count)
    nurbsknot = np.zeros(kc, dtype=np.float64)
    
    # Fill interior nurbsknots: from index (order-2) to (cv_count-1)
    k = 0.0
    for i in range(order - 2, cv_count):
        nurbsknot[i] = k
        k += delta
    
    # Clamp both ends
    clamp(order, cv_count, nurbsknot, 2)
    
    return nurbsknot


def make_periodic_uniform(order: int, cv_count: int, delta: float = 1.0) -> Optional[np.ndarray]:
    """Create a periodic uniform nurbsknot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1), must be >= 2.
    cv_count : int
        Number of control vertices, must be >= order.
    delta : float, optional
        Spacing between nurbsknots. Default is 1.0.
    
    Returns
    -------
    np.ndarray or None
        Periodic uniform nurbsknot vector, or None if parameters are invalid.
    
    Notes
    -----
    Implements ON_MakePeriodicUniformNurbsKnotVector from OpenNURBS.
    """
    if order < 2 or cv_count < order or delta <= 0.0:
        return None
    
    kc = nurbsknot_count(order, cv_count)
    nurbsknot = np.zeros(kc, dtype=np.float64)
    
    k = 0.0
    for i in range(kc):
        nurbsknot[i] = k
        k += delta
    
    return nurbsknot


def clamp(order: int, cv_count: int, nurbsknot: np.ndarray, end: int = 2) -> bool:
    """Clamp the ends of a nurbsknot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector to clamp (modified in place).
    end : int, optional
        Which end to clamp: 0 = left, 1 = right, 2 = both. Default is 2.
    
    Returns
    -------
    bool
        True if successful.
    
    Notes
    -----
    Implements ON_ClampNurbsKnotVector from OpenNURBS.
    Sets the first/last (order-2) nurbsknots equal to the boundary values.
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return False
    
    # Clamp left end
    if end == 0 or end == 2:
        clamp_value = nurbsknot[order - 2]
        for i in range(order - 2):
            nurbsknot[i] = clamp_value
    
    # Clamp right end
    if end == 1 or end == 2:
        clamp_value = nurbsknot[cv_count - 1]
        for i in range(cv_count, kc):
            nurbsknot[i] = clamp_value
    
    return True


def is_valid(order: int, cv_count: int, nurbsknot: np.ndarray) -> bool:
    """Check if a nurbsknot vector is valid.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector to validate.
    
    Returns
    -------
    bool
        True if the nurbsknot vector is valid.
    
    Notes
    -----
    A valid nurbsknot vector must:
    - Have correct length (order + cv_count - 2)
    - Be non-decreasing
    - Have nurbsknot[i] < nurbsknot[i + order - 1] for all valid i (no degenerate spans)
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return False
    
    # Check non-decreasing
    for i in range(1, kc):
        if nurbsknot[i] < nurbsknot[i - 1]:
            return False
    
    # Check no degenerate spans (nurbsknot[i] < nurbsknot[i + order - 1])
    for i in range(kc - order + 1):
        if nurbsknot[i] >= nurbsknot[i + order - 1]:
            return False
    
    return True


def is_clamped(order: int, cv_count: int, nurbsknot: np.ndarray, end: int = 2) -> bool:
    """Check if a nurbsknot vector is clamped.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector to check.
    end : int, optional
        Which end to check: 0 = left, 1 = right, 2 = both. Default is 2.
    
    Returns
    -------
    bool
        True if the nurbsknot vector is clamped at the specified end(s).
    
    Notes
    -----
    A clamped nurbsknot vector has (order-1) equal nurbsknots at each end.
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return False
    
    mult = order - 1
    tol = 1e-10
    
    # Check left end
    if end == 0 or end == 2:
        if mult > kc:
            return False
        start_value = nurbsknot[0]
        for i in range(1, mult):
            if abs(nurbsknot[i] - start_value) > tol:
                return False
    
    # Check right end
    if end == 1 or end == 2:
        if mult > kc:
            return False
        end_value = nurbsknot[-1]
        for i in range(1, mult):
            if abs(nurbsknot[kc - 1 - i] - end_value) > tol:
                return False
    
    return True


def is_periodic(order: int, cv_count: int, nurbsknot: np.ndarray) -> bool:
    """Check if a nurbsknot vector is periodic.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector to check.
    
    Returns
    -------
    bool
        True if the nurbsknot vector is periodic.
    
    Notes
    -----
    A periodic nurbsknot vector has uniform spacing throughout.
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc or kc < 2:
        return False
    
    delta = nurbsknot[1] - nurbsknot[0]
    if delta <= 0:
        return False
    
    tol = 1e-10
    for i in range(2, kc):
        if abs((nurbsknot[i] - nurbsknot[i - 1]) - delta) > tol:
            return False
    
    return True


def is_uniform(order: int, cv_count: int, nurbsknot: np.ndarray) -> bool:
    """Check if a nurbsknot vector is uniform (interior nurbsknots evenly spaced).
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector to check.
    
    Returns
    -------
    bool
        True if the interior nurbsknots are uniformly spaced.
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return False
    
    # Check interior nurbsknots (from order-2 to cv_count-1)
    if cv_count <= order:
        return True  # No interior nurbsknots
    
    start_idx = order - 2
    end_idx = cv_count - 1
    
    if end_idx <= start_idx:
        return True
    
    delta = nurbsknot[start_idx + 1] - nurbsknot[start_idx]
    if delta <= 0:
        return False
    
    tol = 1e-10
    for i in range(start_idx + 2, end_idx + 1):
        if abs((nurbsknot[i] - nurbsknot[i - 1]) - delta) > tol:
            return False
    
    return True


def get_domain(order: int, cv_count: int, nurbsknot: np.ndarray) -> Tuple[float, float]:
    """Get the domain of a nurbsknot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector.
    
    Returns
    -------
    tuple of float
        (t0, t1) domain interval.
    
    Notes
    -----
    Domain is [nurbsknot[order-2], nurbsknot[cv_count-1]].
    """
    if order < 2 or cv_count < order or len(nurbsknot) < nurbsknot_count(order, cv_count):
        return (0.0, 0.0)
    
    return (nurbsknot[order - 2], nurbsknot[cv_count - 1])


def set_domain(order: int, cv_count: int, nurbsknot: np.ndarray, 
               t0: float, t1: float) -> bool:
    """Set the domain of a nurbsknot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector (modified in place).
    t0 : float
        New domain start.
    t1 : float
        New domain end.
    
    Returns
    -------
    bool
        True if successful.
    
    Notes
    -----
    Implements ON_SetNurbsKnotVectorDomain from OpenNURBS.
    Rescales the nurbsknot vector to the new domain.
    """
    if order < 2 or cv_count < order or t0 >= t1:
        return False
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return False
    
    old_t0, old_t1 = get_domain(order, cv_count, nurbsknot)
    if old_t1 <= old_t0:
        return False
    
    scale = (t1 - t0) / (old_t1 - old_t0)
    for i in range(kc):
        nurbsknot[i] = t0 + (nurbsknot[i] - old_t0) * scale
    
    return True


def reverse(order: int, cv_count: int, nurbsknot: np.ndarray) -> bool:
    """Reverse a nurbsknot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector (modified in place).
    
    Returns
    -------
    bool
        True if successful.
    
    Notes
    -----
    Implements ON_ReverseNurbsKnotVector from OpenNURBS.
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return False
    
    # Reverse the array
    nurbsknot[:] = nurbsknot[::-1]
    
    # Negate and shift to maintain same domain direction
    t0 = nurbsknot[0]
    t1 = nurbsknot[-1]
    for i in range(kc):
        nurbsknot[i] = t0 + t1 - nurbsknot[i]
    
    return True


def multiplicity(order: int, cv_count: int, nurbsknot: np.ndarray, 
                 nurbsknot_index: int) -> int:
    """Get the multiplicity of a nurbsknot.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector.
    nurbsknot_index : int
        Index of the nurbsknot to check.
    
    Returns
    -------
    int
        Multiplicity of the nurbsknot at the given index.
    
    Notes
    -----
    Implements ON_NurbsKnotMultiplicity from OpenNURBS.
    """
    if order < 2 or cv_count < order:
        return 0
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc or nurbsknot_index < 0 or nurbsknot_index >= kc:
        return 0
    
    nurbsknot_value = nurbsknot[nurbsknot_index]
    mult = 1
    tol = 1e-14
    
    # Count preceding equal nurbsknots
    i = nurbsknot_index - 1
    while i >= 0 and abs(nurbsknot[i] - nurbsknot_value) < tol:
        mult += 1
        i -= 1
    
    # Count following equal nurbsknots
    i = nurbsknot_index + 1
    while i < kc and abs(nurbsknot[i] - nurbsknot_value) < tol:
        mult += 1
        i += 1
    
    return mult


def span_count(order: int, cv_count: int, nurbsknot: np.ndarray) -> int:
    """Get the number of spans in a nurbsknot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector.
    
    Returns
    -------
    int
        Number of non-empty spans.
    
    Notes
    -----
    Implements ON_NurbsKnotVectorSpanCount from OpenNURBS.
    """
    if order < 2 or cv_count < order:
        return 0
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return 0
    
    count = 0
    d = order - 1  # degree
    
    for i in range(cv_count - order + 1):
        if nurbsknot[i + d - 1] < nurbsknot[i + d]:
            count += 1
    
    return count


def get_span_vector(order: int, cv_count: int, nurbsknot: np.ndarray) -> np.ndarray:
    """Get the span breakpoints of a nurbsknot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector.
    
    Returns
    -------
    np.ndarray
        Array of unique nurbsknot values that define span boundaries.
    """
    if order < 2 or cv_count < order:
        return np.array([])
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return np.array([])
    
    spans = []
    tol = 1e-14
    
    for i in range(kc - 1):
        if abs(nurbsknot[i + 1] - nurbsknot[i]) > tol:
            spans.append(nurbsknot[i])
    
    if kc > 0:
        spans.append(nurbsknot[-1])
    
    return np.array(spans)


def find_span(order: int, cv_count: int, nurbsknot: np.ndarray, 
              t: float, side: int = 0, hint: int = 0) -> int:
    """Find the nurbsknot span containing parameter t.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector.
    t : float
        Parameter value to locate.
    side : int, optional
        When t is at a nurbsknot: 0 = from above (default), -1 = from below, 1 = from above.
    hint : int, optional
        Search hint (not used in this implementation).
    
    Returns
    -------
    int
        Span index in range [0, cv_count - order].
    
    Notes
    -----
    Implements ON_NurbsSpanIndex from OpenNURBS.
    Returns the index i such that nurbsknot[i+order-2] <= t < nurbsknot[i+order-1],
    or the appropriate boundary span if t is outside the domain.
    """
    if order < 2 or cv_count < order:
        return 0
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return 0
    
    # Shift by (order - 2) as in OpenNURBS
    nurbsknot_offset = order - 2
    span_len = cv_count - order + 2
    
    # Handle boundary cases
    if t <= nurbsknot[nurbsknot_offset]:
        return 0
    if t >= nurbsknot[nurbsknot_offset + span_len - 1]:
        return span_len - 2
    
    # Binary search
    low = 0
    high = span_len - 1
    
    while high > low + 1:
        mid = (low + high) // 2
        if t < nurbsknot[nurbsknot_offset + mid]:
            high = mid
        else:
            low = mid
    
    return low


def superfluous_nurbsknot(order: int, cv_count: int, nurbsknot: np.ndarray, 
                     end: int) -> float:
    """Get the superfluous nurbsknot value at the specified end.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector.
    end : int
        0 = first superfluous nurbsknot, 1 = last superfluous nurbsknot.
    
    Returns
    -------
    float
        Superfluous nurbsknot value.
    
    Notes
    -----
    Implements ON_SuperfluousNurbsKnot from OpenNURBS.
    The "superfluous" nurbsknots are the ones that would be at the very start
    and end of a standard B-spline nurbsknot vector but are omitted in OpenNURBS.
    """
    if order < 2 or cv_count < order:
        return 0.0
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return 0.0
    
    if end == 0:
        # First superfluous nurbsknot
        return 2.0 * nurbsknot[0] - nurbsknot[order - 2]
    else:
        # Last superfluous nurbsknot
        return 2.0 * nurbsknot[-1] - nurbsknot[cv_count - order]


def greville_abcissa(order: int, nurbsknot: np.ndarray) -> float:
    """Compute a single Greville abscissa.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    nurbsknot : np.ndarray
        Array of (order - 1) nurbsknot values.
    
    Returns
    -------
    float
        Greville abscissa (average of the nurbsknots).
    
    Notes
    -----
    Implements ON_GrevilleAbcissa from OpenNURBS.
    """
    if order < 2 or len(nurbsknot) < order - 1:
        return 0.0
    
    d = order - 1  # degree
    return sum(nurbsknot[:d]) / d


def get_greville_abcissae(order: int, cv_count: int, nurbsknot: np.ndarray,
                          periodic: bool = False) -> np.ndarray:
    """Get all Greville abscissae for a nurbsknot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    nurbsknot : np.ndarray
        NurbsKnot vector.
    periodic : bool, optional
        True for periodic curves. Default is False.
    
    Returns
    -------
    np.ndarray
        Array of Greville abscissae.
    
    Notes
    -----
    Implements ON_GetGrevilleAbcissae from OpenNURBS.
    For non-periodic: returns cv_count values.
    For periodic: returns cv_count - order + 1 values.
    """
    if order < 2 or cv_count < order:
        return np.array([])
    
    kc = nurbsknot_count(order, cv_count)
    if len(nurbsknot) != kc:
        return np.array([])
    
    d = order - 1  # degree

    if periodic:
        count = cv_count - order + 1
    else:
        count = cv_count

    g = np.zeros(count, dtype=np.float64)

    for i in range(count):
        g[i] = sum(nurbsknot[i:i + d]) / d

    return g


def solve_tridiagonal(dim: int, n: int, lower: List[float], diag: List[float],
                      upper: List[float], rhs: List[float]) -> Optional[List[float]]:
    """Solve tridiagonal linear system using Thomas algorithm.

    Parameters
    ----------
    dim : int
        Dimension of each variable (e.g., 3 for 3D points).
    n : int
        Number of equations.
    lower : list
        Lower diagonal coefficients (length n, first element unused).
    diag : list
        Main diagonal coefficients (length n).
    upper : list
        Upper diagonal coefficients (length n, last element unused).
    rhs : list
        Right-hand side values (length n * dim).

    Returns
    -------
    list or None
        Solution vector (length n * dim), or None if singular.
    """
    if n < 1 or dim < 1:
        return None

    eps = 1e-14
    c_star = [0.0] * n
    d_star = [0.0] * (n * dim)
    solution = [0.0] * (n * dim)

    if abs(diag[0]) < eps:
        return None

    c_star[0] = upper[0] / diag[0]
    for d in range(dim):
        d_star[d] = rhs[d] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c_star[i-1]
        if abs(denom) < eps:
            return None

        c_star[i] = upper[i] / denom if i < n-1 else 0.0
        for d in range(dim):
            d_star[i * dim + d] = (rhs[i * dim + d] - lower[i] * d_star[(i-1) * dim + d]) / denom

    for d in range(dim):
        solution[(n-1) * dim + d] = d_star[(n-1) * dim + d]

    for i in range(n - 2, -1, -1):
        for d in range(dim):
            solution[i * dim + d] = d_star[i * dim + d] - c_star[i] * solution[(i+1) * dim + d]

    return solution


def compute_parameters(points: np.ndarray, style: CurveNurbsKnotStyle) -> np.ndarray:
    """Compute parameters for interpolation based on nurbsknot style.

    Parameters
    ----------
    points : np.ndarray
        Input points, shape (n, dim).
    style : CurveNurbsKnotStyle
        NurbsKnot style (Uniform, Chord, or ChordSquareRoot).

    Returns
    -------
    np.ndarray
        Parameter values for each point.
    """
    n = len(points)
    params = np.zeros(n)
    if n < 2:
        return params

    base_style = int(style) % 3  # 0=Uniform, 1=Chord, 2=ChordSquareRoot

    for i in range(1, n):
        diff = points[i] - points[i-1]
        dist = math.sqrt(np.dot(diff, diff))

        if base_style == 0:  # Uniform
            delta = 1.0
        elif base_style == 1:  # Chord
            delta = dist
        else:  # ChordSquareRoot (centripetal)
            delta = math.sqrt(dist)

        params[i] = params[i-1] + delta

    return params


def build_interp_nurbsknots(params: np.ndarray, degree: int) -> np.ndarray:
    """Build clamped nurbsknot vector from parameters for interpolation.

    Parameters
    ----------
    params : np.ndarray
        Parameter values for each input point.
    degree : int
        Curve degree.

    Returns
    -------
    np.ndarray
        NurbsKnot vector for interpolated curve.
    """
    n = len(params)
    if n < 2 or degree < 1:
        return np.array([])

    order = degree + 1
    cv_count = n + 2  # Natural end conditions add 2 CVs
    kc = nurbsknot_count(order, cv_count)

    nurbsknots = np.zeros(kc)
    t_max = params[-1]

    # Clamped start: first (order-1) nurbsknots = 0
    for i in range(order - 1):
        nurbsknots[i] = 0.0

    # Interior nurbsknots from parameters (skip first and last)
    for i in range(1, n - 1):
        nurbsknots[order - 2 + i] = params[i]

    # Clamped end: last (order-1) nurbsknots = t_max
    for i in range(order - 1):
        nurbsknots[kc - 1 - i] = t_max

    return nurbsknots


def eval_basis(order: int, nurbsknot: np.ndarray, span: int, t: float) -> List[float]:
    """Evaluate B-spline basis functions at parameter t (Cox-de Boor).

    Parameters
    ----------
    order : int
        Order of the B-spline (degree + 1).
    nurbsknot : np.ndarray
        Full nurbsknot vector.
    span : int
        Span index (from find_span).
    t : float
        Parameter value.

    Returns
    -------
    list
        Vector of 'order' basis function values.
    """
    basis = [0.0] * order
    left = [0.0] * order
    right = [0.0] * order

    k_offset = order - 2 + span
    basis[0] = 1.0

    for j in range(1, order):
        left[j] = t - nurbsknot[k_offset + 1 - j]
        right[j] = nurbsknot[k_offset + j] - t
        saved = 0.0

        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = basis[r] / denom if denom != 0.0 else 0.0
            basis[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        basis[j] = saved

    return basis


def build_fitted_nurbsknots(params, num_cvs: int, degree: int):
    m = len(params)
    n_interior = num_cvs - degree - 1
    order = degree + 1
    kc = nurbsknot_count(order, num_cvs)
    nurbsknots = [0.0] * kc

    for i in range(degree):
        nurbsknots[i] = params[0]

    d = float(m) / (num_cvs - degree)
    for j in range(1, n_interior + 1):
        i = int(j * d)
        alpha = j * d - i
        nurbsknots[degree - 1 + j] = (1.0 - alpha) * params[i - 1] + alpha * params[i]

    for i in range(num_cvs - 1, kc):
        nurbsknots[i] = params[m - 1]

    return nurbsknots


def build_fitted_nurbsknots_adaptive(params, points, point_count, dim, num_cvs, degree, scale=3.0):
    m = point_count
    if m < 3 or points is None:
        return build_fitted_nurbsknots(params, num_cvs, degree)

    turn = [0.0] * m
    for i in range(1, m - 1):
        dot, len1sq, len2sq = 0.0, 0.0, 0.0
        for d in range(dim):
            a = points[i*dim+d] - points[(i-1)*dim+d]
            b = points[(i+1)*dim+d] - points[i*dim+d]
            dot += a * b; len1sq += a * a; len2sq += b * b
        len1, len2 = math.sqrt(len1sq), math.sqrt(len2sq)
        if len1 > 1e-14 and len2 > 1e-14:
            c = max(-1.0, min(1.0, dot / (len1 * len2)))
            turn[i] = math.acos(c)

    cum = [0.0] * m
    for i in range(m - 1):
        chord = params[i+1] - params[i]
        if chord < 1e-14: chord = 1e-14
        cum[i+1] = cum[i] + chord * (1.0 + scale * (turn[i] + turn[i+1]) * 0.5)
    total = cum[m-1]

    n_interior = num_cvs - degree - 1
    order = degree + 1
    kc = nurbsknot_count(order, num_cvs)
    nurbsknots = [0.0] * kc
    for i in range(degree): nurbsknots[i] = params[0]

    for j in range(1, n_interior + 1):
        target = total * j / (n_interior + 1)
        lo, hi = 0, m - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid+1] < target: lo = mid + 1
            else: hi = mid
        frac = (target - cum[lo]) / (cum[lo+1] - cum[lo]) if cum[lo+1] > cum[lo] else 0.0
        nurbsknots[degree - 1 + j] = params[lo] + frac * (params[lo+1] - params[lo])

    for i in range(num_cvs - 1, kc): nurbsknots[i] = params[m-1]
    return nurbsknots


def build_fitted_nurbsknots_periodic_adaptive(params, points, n, dim, num_cvs, degree, scale=3.0):
    cv_count = num_cvs + degree
    order = degree + 1
    kc = cv_count + order - 2
    T = params[n]

    if n < 3 or points is None:
        delta = T / num_cvs
        return [(i - degree + 1) * delta for i in range(kc)]

    turn = [0.0] * n
    for i in range(n):
        prev, nxt = (i - 1 + n) % n, (i + 1) % n
        dot, len1sq, len2sq = 0.0, 0.0, 0.0
        for d in range(dim):
            a = points[i*dim+d] - points[prev*dim+d]
            b = points[nxt*dim+d] - points[i*dim+d]
            dot += a * b; len1sq += a * a; len2sq += b * b
        len1, len2 = math.sqrt(len1sq), math.sqrt(len2sq)
        if len1 > 1e-14 and len2 > 1e-14:
            c = max(-1.0, min(1.0, dot / (len1 * len2)))
            turn[i] = math.acos(c)

    cum = [0.0] * (n + 1)
    for i in range(n):
        chord = params[i+1] - params[i]
        if chord < 1e-14: chord = 1e-14
        nxt = (i + 1) % n
        cum[i+1] = cum[i] + chord * (1.0 + scale * (turn[i] + turn[nxt]) * 0.5)
    total = cum[n]

    base = [0.0] * num_cvs
    for j in range(num_cvs):
        target = total * j / num_cvs
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid+1] < target: lo = mid + 1
            else: hi = mid
        frac = (target - cum[lo]) / (cum[lo+1] - cum[lo]) if cum[lo+1] > cum[lo] else 0.0
        base[j] = params[lo] + frac * (params[lo+1] - params[lo])

    intervals = [0.0] * num_cvs
    for j in range(num_cvs - 1): intervals[j] = base[j+1] - base[j]
    intervals[num_cvs - 1] = T - base[num_cvs - 1]

    nurbsknots = [0.0] * kc
    nurbsknots[degree - 1] = 0.0
    for i in range(1, degree):
        nurbsknots[degree - 1 - i] = nurbsknots[degree - i] - intervals[num_cvs - i]
    for i in range(kc - degree):
        nurbsknots[degree + i] = nurbsknots[degree - 1 + i] + intervals[i % num_cvs]

    return nurbsknots


def solve_banded_spd(dim: int, n: int, half_bw: int, band, rhs):
    bw1 = half_bw + 1

    for i in range(n):
        for j in range(max(0, i - half_bw), i + 1):
            s = 0.0
            for k in range(max(0, i - half_bw), j):
                s += band[i * bw1 + (i - k)] * band[j * bw1 + (j - k)]
            if i == j:
                val = band[i * bw1] - s
                if val <= 1e-30:
                    return False
                band[i * bw1] = math.sqrt(val)
            else:
                band[i * bw1 + (i - j)] = (band[i * bw1 + (i - j)] - s) / band[j * bw1]

    for i in range(n):
        for d in range(dim):
            s = 0.0
            for k in range(max(0, i - half_bw), i):
                s += band[i * bw1 + (i - k)] * rhs[k * dim + d]
            rhs[i * dim + d] = (rhs[i * dim + d] - s) / band[i * bw1]

    for i in range(n - 1, -1, -1):
        for d in range(dim):
            s = 0.0
            for k in range(i + 1, min(n, i + half_bw + 1)):
                s += band[k * bw1 + (k - i)] * rhs[k * dim + d]
            rhs[i * dim + d] = (rhs[i * dim + d] - s) / band[i * bw1]

    return True
