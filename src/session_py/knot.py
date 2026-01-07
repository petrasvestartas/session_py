"""Knot vector utility functions for NURBS curves and surfaces.

This module provides standalone functions for working with knot vectors,
following the OpenNURBS pattern (opennurbs_knot.h/cpp).

These functions operate on numpy arrays and can be used independently
or called by NurbsCurve and NurbsSurface classes.
"""

import numpy as np
from typing import Tuple, Optional


def knot_count(order: int, cv_count: int) -> int:
    """Compute the number of knots in a knot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1), must be >= 2.
    cv_count : int
        Number of control vertices, must be >= order.
    
    Returns
    -------
    int
        Number of knots: order + cv_count - 2
    
    Notes
    -----
    OpenNURBS uses a knot vector convention that omits the first and last
    "superfluous" knots that would be present in a standard B-spline knot vector.
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
    """Create a clamped uniform knot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1), must be >= 2.
    cv_count : int
        Number of control vertices, must be >= order.
    delta : float, optional
        Spacing between interior knots. Default is 1.0.
    
    Returns
    -------
    np.ndarray or None
        Clamped uniform knot vector with domain [0, (cv_count - order + 1) * delta],
        or None if parameters are invalid.
    
    Notes
    -----
    Implements ON_MakeClampedUniformKnotVector from OpenNURBS.
    The resulting knot vector has:
    - First (order-1) knots clamped to 0
    - Interior knots uniformly spaced by delta
    - Last (order-1) knots clamped to the final value
    """
    if order < 2 or cv_count < order or delta <= 0.0:
        return None
    
    kc = knot_count(order, cv_count)
    knot = np.zeros(kc, dtype=np.float64)
    
    # Fill interior knots: from index (order-2) to (cv_count-1)
    k = 0.0
    for i in range(order - 2, cv_count):
        knot[i] = k
        k += delta
    
    # Clamp both ends
    clamp(order, cv_count, knot, 2)
    
    return knot


def make_periodic_uniform(order: int, cv_count: int, delta: float = 1.0) -> Optional[np.ndarray]:
    """Create a periodic uniform knot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1), must be >= 2.
    cv_count : int
        Number of control vertices, must be >= order.
    delta : float, optional
        Spacing between knots. Default is 1.0.
    
    Returns
    -------
    np.ndarray or None
        Periodic uniform knot vector, or None if parameters are invalid.
    
    Notes
    -----
    Implements ON_MakePeriodicUniformKnotVector from OpenNURBS.
    """
    if order < 2 or cv_count < order or delta <= 0.0:
        return None
    
    kc = knot_count(order, cv_count)
    knot = np.zeros(kc, dtype=np.float64)
    
    k = 0.0
    for i in range(kc):
        knot[i] = k
        k += delta
    
    return knot


def clamp(order: int, cv_count: int, knot: np.ndarray, end: int = 2) -> bool:
    """Clamp the ends of a knot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector to clamp (modified in place).
    end : int, optional
        Which end to clamp: 0 = left, 1 = right, 2 = both. Default is 2.
    
    Returns
    -------
    bool
        True if successful.
    
    Notes
    -----
    Implements ON_ClampKnotVector from OpenNURBS.
    Sets the first/last (order-2) knots equal to the boundary values.
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return False
    
    # Clamp left end
    if end == 0 or end == 2:
        clamp_value = knot[order - 2]
        for i in range(order - 2):
            knot[i] = clamp_value
    
    # Clamp right end
    if end == 1 or end == 2:
        clamp_value = knot[cv_count - 1]
        for i in range(cv_count, kc):
            knot[i] = clamp_value
    
    return True


def is_valid(order: int, cv_count: int, knot: np.ndarray) -> bool:
    """Check if a knot vector is valid.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector to validate.
    
    Returns
    -------
    bool
        True if the knot vector is valid.
    
    Notes
    -----
    A valid knot vector must:
    - Have correct length (order + cv_count - 2)
    - Be non-decreasing
    - Have knot[i] < knot[i + order - 1] for all valid i (no degenerate spans)
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return False
    
    # Check non-decreasing
    for i in range(1, kc):
        if knot[i] < knot[i - 1]:
            return False
    
    # Check no degenerate spans (knot[i] < knot[i + order - 1])
    for i in range(kc - order + 1):
        if knot[i] >= knot[i + order - 1]:
            return False
    
    return True


def is_clamped(order: int, cv_count: int, knot: np.ndarray, end: int = 2) -> bool:
    """Check if a knot vector is clamped.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector to check.
    end : int, optional
        Which end to check: 0 = left, 1 = right, 2 = both. Default is 2.
    
    Returns
    -------
    bool
        True if the knot vector is clamped at the specified end(s).
    
    Notes
    -----
    A clamped knot vector has (order-1) equal knots at each end.
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return False
    
    mult = order - 1
    tol = 1e-10
    
    # Check left end
    if end == 0 or end == 2:
        if mult > kc:
            return False
        start_value = knot[0]
        for i in range(1, mult):
            if abs(knot[i] - start_value) > tol:
                return False
    
    # Check right end
    if end == 1 or end == 2:
        if mult > kc:
            return False
        end_value = knot[-1]
        for i in range(1, mult):
            if abs(knot[kc - 1 - i] - end_value) > tol:
                return False
    
    return True


def is_periodic(order: int, cv_count: int, knot: np.ndarray) -> bool:
    """Check if a knot vector is periodic.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector to check.
    
    Returns
    -------
    bool
        True if the knot vector is periodic.
    
    Notes
    -----
    A periodic knot vector has uniform spacing throughout.
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc or kc < 2:
        return False
    
    delta = knot[1] - knot[0]
    if delta <= 0:
        return False
    
    tol = 1e-10
    for i in range(2, kc):
        if abs((knot[i] - knot[i - 1]) - delta) > tol:
            return False
    
    return True


def is_uniform(order: int, cv_count: int, knot: np.ndarray) -> bool:
    """Check if a knot vector is uniform (interior knots evenly spaced).
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector to check.
    
    Returns
    -------
    bool
        True if the interior knots are uniformly spaced.
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return False
    
    # Check interior knots (from order-2 to cv_count-1)
    if cv_count <= order:
        return True  # No interior knots
    
    start_idx = order - 2
    end_idx = cv_count - 1
    
    if end_idx <= start_idx:
        return True
    
    delta = knot[start_idx + 1] - knot[start_idx]
    if delta <= 0:
        return False
    
    tol = 1e-10
    for i in range(start_idx + 2, end_idx + 1):
        if abs((knot[i] - knot[i - 1]) - delta) > tol:
            return False
    
    return True


def get_domain(order: int, cv_count: int, knot: np.ndarray) -> Tuple[float, float]:
    """Get the domain of a knot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector.
    
    Returns
    -------
    tuple of float
        (t0, t1) domain interval.
    
    Notes
    -----
    Domain is [knot[order-2], knot[cv_count-1]].
    """
    if order < 2 or cv_count < order or len(knot) < knot_count(order, cv_count):
        return (0.0, 0.0)
    
    return (knot[order - 2], knot[cv_count - 1])


def set_domain(order: int, cv_count: int, knot: np.ndarray, 
               t0: float, t1: float) -> bool:
    """Set the domain of a knot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector (modified in place).
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
    Implements ON_SetKnotVectorDomain from OpenNURBS.
    Rescales the knot vector to the new domain.
    """
    if order < 2 or cv_count < order or t0 >= t1:
        return False
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return False
    
    old_t0, old_t1 = get_domain(order, cv_count, knot)
    if old_t1 <= old_t0:
        return False
    
    scale = (t1 - t0) / (old_t1 - old_t0)
    for i in range(kc):
        knot[i] = t0 + (knot[i] - old_t0) * scale
    
    return True


def reverse(order: int, cv_count: int, knot: np.ndarray) -> bool:
    """Reverse a knot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector (modified in place).
    
    Returns
    -------
    bool
        True if successful.
    
    Notes
    -----
    Implements ON_ReverseKnotVector from OpenNURBS.
    """
    if order < 2 or cv_count < order:
        return False
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return False
    
    # Reverse the array
    knot[:] = knot[::-1]
    
    # Negate and shift to maintain same domain direction
    t0 = knot[0]
    t1 = knot[-1]
    for i in range(kc):
        knot[i] = t0 + t1 - knot[i]
    
    return True


def multiplicity(order: int, cv_count: int, knot: np.ndarray, 
                 knot_index: int) -> int:
    """Get the multiplicity of a knot.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector.
    knot_index : int
        Index of the knot to check.
    
    Returns
    -------
    int
        Multiplicity of the knot at the given index.
    
    Notes
    -----
    Implements ON_KnotMultiplicity from OpenNURBS.
    """
    if order < 2 or cv_count < order:
        return 0
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc or knot_index < 0 or knot_index >= kc:
        return 0
    
    knot_value = knot[knot_index]
    mult = 1
    tol = 1e-14
    
    # Count preceding equal knots
    i = knot_index - 1
    while i >= 0 and abs(knot[i] - knot_value) < tol:
        mult += 1
        i -= 1
    
    # Count following equal knots
    i = knot_index + 1
    while i < kc and abs(knot[i] - knot_value) < tol:
        mult += 1
        i += 1
    
    return mult


def span_count(order: int, cv_count: int, knot: np.ndarray) -> int:
    """Get the number of spans in a knot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector.
    
    Returns
    -------
    int
        Number of non-empty spans.
    
    Notes
    -----
    Implements ON_KnotVectorSpanCount from OpenNURBS.
    """
    if order < 2 or cv_count < order:
        return 0
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return 0
    
    count = 0
    d = order - 1  # degree
    
    for i in range(cv_count - order + 1):
        if knot[i + d - 1] < knot[i + d]:
            count += 1
    
    return count


def get_span_vector(order: int, cv_count: int, knot: np.ndarray) -> np.ndarray:
    """Get the span breakpoints of a knot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector.
    
    Returns
    -------
    np.ndarray
        Array of unique knot values that define span boundaries.
    """
    if order < 2 or cv_count < order:
        return np.array([])
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return np.array([])
    
    spans = []
    tol = 1e-14
    
    for i in range(kc - 1):
        if abs(knot[i + 1] - knot[i]) > tol:
            spans.append(knot[i])
    
    if kc > 0:
        spans.append(knot[-1])
    
    return np.array(spans)


def find_span(order: int, cv_count: int, knot: np.ndarray, 
              t: float, side: int = 0, hint: int = 0) -> int:
    """Find the knot span containing parameter t.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector.
    t : float
        Parameter value to locate.
    side : int, optional
        When t is at a knot: 0 = from above (default), -1 = from below, 1 = from above.
    hint : int, optional
        Search hint (not used in this implementation).
    
    Returns
    -------
    int
        Span index in range [0, cv_count - order].
    
    Notes
    -----
    Implements ON_NurbsSpanIndex from OpenNURBS.
    Returns the index i such that knot[i+order-2] <= t < knot[i+order-1],
    or the appropriate boundary span if t is outside the domain.
    """
    if order < 2 or cv_count < order:
        return 0
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return 0
    
    # Shift by (order - 2) as in OpenNURBS
    knot_offset = order - 2
    span_len = cv_count - order + 2
    
    # Handle boundary cases
    if t <= knot[knot_offset]:
        return 0
    if t >= knot[knot_offset + span_len - 1]:
        return span_len - 2
    
    # Binary search
    low = 0
    high = span_len - 1
    
    while high > low + 1:
        mid = (low + high) // 2
        if t < knot[knot_offset + mid]:
            high = mid
        else:
            low = mid
    
    return low


def superfluous_knot(order: int, cv_count: int, knot: np.ndarray, 
                     end: int) -> float:
    """Get the superfluous knot value at the specified end.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector.
    end : int
        0 = first superfluous knot, 1 = last superfluous knot.
    
    Returns
    -------
    float
        Superfluous knot value.
    
    Notes
    -----
    Implements ON_SuperfluousKnot from OpenNURBS.
    The "superfluous" knots are the ones that would be at the very start
    and end of a standard B-spline knot vector but are omitted in OpenNURBS.
    """
    if order < 2 or cv_count < order:
        return 0.0
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return 0.0
    
    if end == 0:
        # First superfluous knot
        return 2.0 * knot[0] - knot[order - 2]
    else:
        # Last superfluous knot
        return 2.0 * knot[-1] - knot[cv_count - order]


def greville_abcissa(order: int, knot: np.ndarray) -> float:
    """Compute a single Greville abscissa.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    knot : np.ndarray
        Array of (order - 1) knot values.
    
    Returns
    -------
    float
        Greville abscissa (average of the knots).
    
    Notes
    -----
    Implements ON_GrevilleAbcissa from OpenNURBS.
    """
    if order < 2 or len(knot) < order - 1:
        return 0.0
    
    d = order - 1  # degree
    return sum(knot[:d]) / d


def get_greville_abcissae(order: int, cv_count: int, knot: np.ndarray,
                          periodic: bool = False) -> np.ndarray:
    """Get all Greville abscissae for a knot vector.
    
    Parameters
    ----------
    order : int
        Order of the NURBS (degree + 1).
    cv_count : int
        Number of control vertices.
    knot : np.ndarray
        Knot vector.
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
    
    kc = knot_count(order, cv_count)
    if len(knot) != kc:
        return np.array([])
    
    d = order - 1  # degree
    
    if periodic:
        count = cv_count - order + 1
    else:
        count = cv_count
    
    g = np.zeros(count, dtype=np.float64)
    
    for i in range(count):
        g[i] = sum(knot[i:i + d]) / d
    
    return g
