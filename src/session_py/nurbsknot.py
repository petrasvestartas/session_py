from __future__ import annotations
import math
import sys

from collections.abc import MutableSequence
from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from .tolerance import Tolerance


KNOT_TOLERANCE = Tolerance.ABSOLUTE / 10.0
PIVOT_TOLERANCE = Tolerance.ZERO_TOLERANCE / 100.0
POSITIVE_DEFINITE_TOLERANCE = (
    Tolerance.ABSOLUTE * Tolerance.ABSOLUTE * Tolerance.ZERO_TOLERANCE
)


def _are_finite(values: Sequence[float] | None, count: int) -> bool:

    if values is None or count < 0 or len(values) < count:
        return False

    for i in range(count):
        if not math.isfinite(values[i]):
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Knot styles
# ═══════════════════════════════════════════════════════════════════════════
class CurveNurbsKnotStyle(IntEnum):
    """Parameter spacing for interpolated curves."""

    Uniform = 0  # Equal parameter spacing.
    Chord = 1  # Spacing proportional to chord length.
    ChordSquareRoot = 2  # Spacing proportional to the square root of chord length.
    UniformPeriodic = 3  # Equal spacing for a periodic curve.
    ChordPeriodic = 4  # Chord-length spacing for a periodic curve.
    ChordSquareRootPeriodic = 5  # Square-root chord spacing for a periodic curve.


class CurveInterpStyle(IntEnum):
    """End-tangent estimate for cubic curve interpolation."""

    Rhino = 0  # Bessel end tangents matching Rhino.
    Occt = 1  # Cubic Lagrange end tangents matching OCCT.


# ═══════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════
def nurbsknot_count(order: int, cv_count: int) -> int:
    """Return the number of nurbsknots for an order and control-point count."""

    if order < 2 or cv_count < order:
        return 0

    count = order + cv_count - 2

    return count if count <= sys.maxsize else 0


def domain_tolerance(a: float, b: float) -> float:
    """Return the floating-point tolerance associated with the domain interval [a, b]."""

    if a == b:
        return 0.0

    epsilon = sys.float_info.epsilon
    tol = (abs(a) + abs(b) + abs(a - b)) * math.sqrt(epsilon)

    return max(tol, epsilon)


def compute_clamped_uniform(
    order: int, cv_count: int, delta: float = 1.0
) -> np.ndarray:
    """Return a clamped uniform nurbsknot vector, or an empty vector for invalid arguments."""

    if order < 2 or cv_count < order or not math.isfinite(delta) or delta <= 0.0:
        return np.array([], dtype=np.float64)

    kc = nurbsknot_count(order, cv_count)

    if kc == 0:
        return np.array([], dtype=np.float64)

    nurbsknot = np.zeros(kc, dtype=np.float64)
    k = 0.0

    for i in range(order - 2, cv_count):
        nurbsknot[i] = k
        k += delta

    if not clamp(order, cv_count, nurbsknot, 2):
        return np.array([], dtype=np.float64)

    return nurbsknot


def compute_periodic_uniform(
    order: int, cv_count: int, delta: float = 1.0
) -> np.ndarray:
    """Return a periodic uniform nurbsknot vector, or an empty vector for invalid arguments."""

    if order < 2 or cv_count < order or not math.isfinite(delta) or delta <= 0.0:
        return np.array([], dtype=np.float64)

    kc = nurbsknot_count(order, cv_count)

    if kc == 0:
        return np.array([], dtype=np.float64)

    nurbsknot = np.zeros(kc, dtype=np.float64)
    k = 0.0

    for i in range(kc):
        nurbsknot[i] = k
        k += delta

    return nurbsknot


def clamp(
    order: int, cv_count: int, nurbsknot: MutableSequence[float], end: int = 2
) -> bool:
    """Clamp the selected ends in place, where end is 0 for left, 1 for right, or 2 for both."""

    if order < 2 or cv_count < order or end < 0 or end > 2:
        return False

    kc = nurbsknot_count(order, cv_count)

    if kc == 0 or len(nurbsknot) != kc or not _are_finite(nurbsknot, kc):
        return False

    if end == 0 or end == 2:
        clamp_value = nurbsknot[order - 2]

        for i in range(order - 2):
            nurbsknot[i] = clamp_value

    if end == 1 or end == 2:
        clamp_value = nurbsknot[cv_count - 1]

        for i in range(cv_count, kc):
            nurbsknot[i] = clamp_value

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Queries
# ═══════════════════════════════════════════════════════════════════════════
def is_valid(order: int, cv_count: int, nurbsknot: Sequence[float]) -> bool:
    """Return whether the vector has the required length, finite values, and valid spans."""

    if order < 2 or cv_count < order:
        return False

    kc = nurbsknot_count(order, cv_count)

    if kc == 0 or len(nurbsknot) != kc or not _are_finite(nurbsknot, kc):
        return False

    for i in range(1, kc):
        if nurbsknot[i] < nurbsknot[i - 1]:
            return False

    for i in range(kc - order + 1):
        if nurbsknot[i] >= nurbsknot[i + order - 1]:
            return False

    return True


def is_clamped(
    order: int, cv_count: int, nurbsknot: Sequence[float], end: int = 2
) -> bool:
    """Return whether the selected ends contain order - 1 equal nurbsknots."""

    if order < 2 or cv_count < order or end < 0 or end > 2:
        return False

    kc = nurbsknot_count(order, cv_count)

    if kc == 0 or len(nurbsknot) != kc or not _are_finite(nurbsknot, kc):
        return False

    mult = order - 1
    tol = KNOT_TOLERANCE

    if end == 0 or end == 2:
        if mult > kc:
            return False

        start_value = nurbsknot[0]

        for i in range(1, mult):
            if abs(nurbsknot[i] - start_value) > tol:
                return False

    if end == 1 or end == 2:
        if mult > kc:
            return False

        end_value = nurbsknot[kc - 1]

        for i in range(1, mult):
            if abs(nurbsknot[kc - 1 - i] - end_value) > tol:
                return False

    return True


def is_periodic(order: int, cv_count: int, nurbsknot: Sequence[float]) -> bool:
    """Return whether the nurbsknot vector has finite, positive, uniform spacing."""

    if order < 2 or cv_count < order:
        return False

    kc = nurbsknot_count(order, cv_count)

    if kc < 2 or len(nurbsknot) != kc or not _are_finite(nurbsknot, kc):
        return False

    delta = nurbsknot[1] - nurbsknot[0]

    if delta <= 0:
        return False

    tol = KNOT_TOLERANCE

    for i in range(2, kc):
        if abs((nurbsknot[i] - nurbsknot[i - 1]) - delta) > tol:
            return False

    return True


def get_domain(
    order: int, cv_count: int, nurbsknot: Sequence[float]
) -> tuple[float, float]:
    """Return the finite domain endpoints, or (0, 0) when the required entries cannot be read."""

    if order < 2 or cv_count < order:
        return (0.0, 0.0)

    kc = nurbsknot_count(order, cv_count)

    if kc == 0 or len(nurbsknot) < kc:
        return (0.0, 0.0)

    start = nurbsknot[order - 2]
    end = nurbsknot[cv_count - 1]

    if not math.isfinite(start) or not math.isfinite(end):
        return (0.0, 0.0)

    return (start, end)


def set_domain(
    order: int, cv_count: int, nurbsknot: MutableSequence[float], t0: float, t1: float
) -> bool:
    """Rescale the nurbsknot vector in place to the finite domain [t0, t1]."""

    if (
        order < 2
        or cv_count < order
        or not math.isfinite(t0)
        or not math.isfinite(t1)
        or t0 >= t1
    ):
        return False

    kc = nurbsknot_count(order, cv_count)

    if kc == 0 or len(nurbsknot) != kc or not _are_finite(nurbsknot, kc):
        return False

    domain = get_domain(order, cv_count, nurbsknot)
    old_t0 = domain[0]
    old_t1 = domain[1]

    if old_t1 <= old_t0:
        return False

    scale = (t1 - t0) / (old_t1 - old_t0)

    for i in range(kc):
        nurbsknot[i] = t0 + (nurbsknot[i] - old_t0) * scale

    return True


def reverse(order: int, cv_count: int, nurbsknot: MutableSequence[float]) -> bool:
    """Reverse a finite nurbsknot vector in place while preserving its domain."""

    if order < 2 or cv_count < order:
        return False

    kc = nurbsknot_count(order, cv_count)

    if kc == 0 or len(nurbsknot) != kc or not _are_finite(nurbsknot, kc):
        return False

    nurbsknot[:] = nurbsknot[::-1]

    t0 = nurbsknot[0]
    t1 = nurbsknot[kc - 1]

    for i in range(kc):
        nurbsknot[i] = t0 + t1 - nurbsknot[i]

    return True


def multiplicity(
    order: int, cv_count: int, nurbsknot: Sequence[float], nurbsknot_index: int
) -> int:
    """Return the multiplicity at nurbsknot_index, or zero for invalid arguments."""

    if order < 2 or cv_count < order:
        return 0

    kc = nurbsknot_count(order, cv_count)

    if (
        kc == 0
        or len(nurbsknot) != kc
        or nurbsknot_index < 0
        or nurbsknot_index >= kc
        or not _are_finite(nurbsknot, kc)
    ):
        return 0

    nurbsknot_value = nurbsknot[nurbsknot_index]
    tol = PIVOT_TOLERANCE
    mult = 1
    i = nurbsknot_index - 1

    while i >= 0 and abs(nurbsknot[i] - nurbsknot_value) < tol:
        mult += 1
        i -= 1

    i = nurbsknot_index + 1

    while i < kc and abs(nurbsknot[i] - nurbsknot_value) < tol:
        mult += 1
        i += 1

    return mult


def span_count(order: int, cv_count: int, nurbsknot: Sequence[float]) -> int:
    """Return the number of non-empty spans, or zero for invalid arguments."""

    if order < 2 or cv_count < order:
        return 0

    kc = nurbsknot_count(order, cv_count)

    if kc == 0 or len(nurbsknot) != kc or not _are_finite(nurbsknot, kc):
        return 0

    d = order - 1
    count = 0

    for i in range(cv_count - order + 1):
        if nurbsknot[i + d - 1] < nurbsknot[i + d]:
            count += 1

    return count


def find_span(
    order: int,
    cv_count: int,
    nurbsknot: Sequence[float],
    t: float,
    side: int = 0,
    hint: int = 0,
) -> int:
    """Return the index of the span containing finite parameter t in a valid nondecreasing nurbsknot vector."""

    del side, hint

    if order < 2 or cv_count < order or not math.isfinite(t):
        return 0

    kc = nurbsknot_count(order, cv_count)

    if kc == 0 or len(nurbsknot) != kc:
        return 0

    nurbsknot_offset = order - 2
    span_len = cv_count - order + 2
    start = nurbsknot[nurbsknot_offset]
    end = nurbsknot[nurbsknot_offset + span_len - 1]

    if not math.isfinite(start) or not math.isfinite(end):
        return 0

    if t <= start:
        return 0

    if t >= end:
        return span_len - 2

    low = 0
    high = span_len - 1

    while high > low + 1:
        mid = low + (high - low) // 2
        mid_value = nurbsknot[nurbsknot_offset + mid]

        if not math.isfinite(mid_value):
            return 0

        if t < mid_value:
            high = mid
        else:
            low = mid

    return low


def get_greville_abcissae(
    order: int, cv_count: int, nurbsknot: Sequence[float], periodic: bool = False
) -> np.ndarray:
    """Return the Greville abscissae, or an empty vector for invalid arguments."""

    if order < 2 or cv_count < order:
        return np.array([], dtype=np.float64)

    kc = nurbsknot_count(order, cv_count)

    if kc == 0 or len(nurbsknot) != kc or not _are_finite(nurbsknot, kc):
        return np.array([], dtype=np.float64)

    d = order - 1
    count = cv_count - order + 1 if periodic else cv_count
    g = np.zeros(count, dtype=np.float64)

    for i in range(count):
        sum = 0.0

        for j in range(d):
            sum += nurbsknot[i + j]

        g[i] = sum / d

    return g


# ═══════════════════════════════════════════════════════════════════════════
# Interpolation
# ═══════════════════════════════════════════════════════════════════════════
def solve_tridiagonal(
    dim: int,
    n: int,
    lower: Sequence[float],
    diag: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> list[float] | None:
    """Solve a finite tridiagonal system with the Thomas algorithm, returning None if invalid or singular."""

    if n < 1 or dim < 1:
        return None

    rhs_count = n * dim

    if len(lower) < n or len(diag) < n or len(upper) < n or len(rhs) < rhs_count:
        return None

    if (
        not _are_finite(lower, n)
        or not _are_finite(diag, n)
        or not _are_finite(upper, n)
        or not _are_finite(rhs, rhs_count)
    ):
        return None

    eps = PIVOT_TOLERANCE
    c_star = [0.0] * n
    d_star = [0.0] * rhs_count
    solution = [0.0] * rhs_count

    if abs(diag[0]) < eps:
        return None

    c_star[0] = upper[0] / diag[0]

    for d in range(dim):
        d_star[d] = rhs[d] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c_star[i - 1]

        if abs(denom) < eps:
            return None

        c_star[i] = upper[i] / denom if i < n - 1 else 0.0

        for d in range(dim):
            index = i * dim + d
            previous = (i - 1) * dim + d
            d_star[index] = (rhs[index] - lower[i] * d_star[previous]) / denom

    for d in range(dim):
        solution[(n - 1) * dim + d] = d_star[(n - 1) * dim + d]

    for i in range(n - 2, -1, -1):
        for d in range(dim):
            index = i * dim + d
            next = (i + 1) * dim + d
            solution[index] = d_star[index] - c_star[i] * solution[next]

    return solution


def compute_parameters(
    points: Sequence[float] | None,
    point_count: int,
    dim: int,
    style: CurveNurbsKnotStyle,
) -> np.ndarray:
    """Return one parameter per point from a flat point_count by dim coordinate array."""

    if point_count < 1 or dim < 1:
        return np.array([], dtype=np.float64)

    value_count = point_count * dim

    if not _are_finite(points, value_count):
        return np.array([], dtype=np.float64)

    params = np.zeros(point_count, dtype=np.float64)

    if point_count < 2:
        return params

    base_style = int(style) % 3

    for i in range(1, point_count):
        dist = 0.0

        for d in range(dim):
            index = i * dim + d
            previous = (i - 1) * dim + d
            diff = points[index] - points[previous]
            dist += diff * diff

        dist = math.sqrt(dist)
        delta = dist

        if base_style == 0:
            delta = 1.0
        elif base_style == 2:
            delta = math.sqrt(dist)

        params[i] = params[i - 1] + delta

    return params


def build_interp_nurbsknots(params: Sequence[float], degree: int) -> np.ndarray:
    """Return a clamped interpolation nurbsknot vector with natural end conditions."""

    n = len(params)

    if n < 2 or degree < 1 or not _are_finite(params, n):
        return np.array([], dtype=np.float64)

    order = degree + 1
    cv_count = n + 2
    kc = nurbsknot_count(order, cv_count)

    if kc == 0:
        return np.array([], dtype=np.float64)

    t_max = params[n - 1]
    nurbsknots = np.zeros(kc, dtype=np.float64)

    for i in range(1, n - 1):
        nurbsknots[order - 2 + i] = params[i]

    for i in range(order - 1):
        nurbsknots[kc - 1 - i] = t_max

    return nurbsknots


def eval_basis(
    order: int, nurbsknot: Sequence[float], span: int, t: float
) -> list[float]:
    """Return the order non-zero B-spline basis values at t by Cox-de Boor evaluation over a finite span window."""

    if order < 1 or span < 0 or not math.isfinite(t):
        return []

    if order == 1:
        return [1.0]

    first = span
    width = 2 * order - 2
    end = first + width

    if len(nurbsknot) < end:
        return []

    for i in range(first, end):
        if not math.isfinite(nurbsknot[i]):
            return []

    basis = [0.0] * order
    left = [0.0] * order
    right = [0.0] * order
    k_offset = first + order - 2
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


# ═══════════════════════════════════════════════════════════════════════════
# Fitting
# ═══════════════════════════════════════════════════════════════════════════
def _build_fitted_nurbsknots(
    params: Sequence[float], num_cvs: int, degree: int
) -> list[float]:

    m = len(params)
    n_interior = num_cvs - degree - 1
    order = degree + 1
    kc = nurbsknot_count(order, num_cvs)

    if kc == 0:
        return []

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


def _turn_angle(
    points: Sequence[float], dim: int, prev: int, i: int, next: int
) -> float:

    dot = 0.0
    len1sq = 0.0
    len2sq = 0.0

    for d in range(dim):
        index = i * dim + d
        a = points[index] - points[prev * dim + d]
        b = points[next * dim + d] - points[index]
        dot += a * b
        len1sq += a * a
        len2sq += b * b

    len1 = math.sqrt(len1sq)
    len2 = math.sqrt(len2sq)

    if len1 <= PIVOT_TOLERANCE or len2 <= PIVOT_TOLERANCE:
        return 0.0

    return math.acos(max(-1.0, min(1.0, dot / (len1 * len2))))


def _locate_target(
    params: Sequence[float], cum: Sequence[float], last: int, target: float
) -> float:

    lo = 0
    hi = last

    while lo < hi:
        mid = lo + (hi - lo) // 2

        if cum[mid + 1] < target:
            lo = mid + 1
        else:
            hi = mid

    frac = (
        (target - cum[lo]) / (cum[lo + 1] - cum[lo]) if cum[lo + 1] > cum[lo] else 0.0
    )

    return params[lo] + frac * (params[lo + 1] - params[lo])


def build_fitted_nurbsknots_adaptive(
    params: Sequence[float],
    points: Sequence[float] | None,
    point_count: int,
    dim: int,
    num_cvs: int,
    degree: int,
    scale: float = 3.0,
) -> list[float]:
    """Return a clamped fitting vector with denser nurbsknots where the points turn."""

    m = point_count

    if (
        m < 2
        or dim < 1
        or num_cvs <= degree
        or degree < 1
        or not math.isfinite(scale)
        or len(params) < m
        or not _are_finite(params, m)
    ):
        return []

    if m < 3 or points is None:
        if m < num_cvs - degree:
            return []

        return _build_fitted_nurbsknots(params, num_cvs, degree)

    if not _are_finite(points, m * dim):
        return []

    turn = [0.0] * m

    for i in range(1, m - 1):
        turn[i] = _turn_angle(points, dim, i - 1, i, i + 1)

    cum = [0.0] * m

    for i in range(m - 1):
        chord = max(params[i + 1] - params[i], PIVOT_TOLERANCE)
        cum[i + 1] = cum[i] + chord * (1.0 + scale * (turn[i] + turn[i + 1]) * 0.5)

    total = cum[m - 1]
    n_interior = num_cvs - degree - 1
    order = degree + 1
    kc = nurbsknot_count(order, num_cvs)

    if kc == 0:
        return []

    nurbsknots = [0.0] * kc

    for i in range(degree):
        nurbsknots[i] = params[0]

    for j in range(1, n_interior + 1):
        nurbsknots[degree - 1 + j] = _locate_target(
            params, cum, m - 2, total * j / (n_interior + 1)
        )

    for i in range(num_cvs - 1, kc):
        nurbsknots[i] = params[m - 1]

    return nurbsknots


def _periodic_weighted_lengths(
    params: Sequence[float],
    points: Sequence[float],
    n: int,
    dim: int,
    scale: float,
) -> list[float]:
    """Cumulative chord lengths around closed points, weighted up where the points turn."""

    turn = [0.0] * n

    for i in range(n):
        turn[i] = _turn_angle(points, dim, n - 1 if i == 0 else i - 1, i, (i + 1) % n)

    cum = [0.0] * (n + 1)

    for i in range(n):
        chord = max(params[i + 1] - params[i], PIVOT_TOLERANCE)
        cum[i + 1] = cum[i] + chord * (
            1.0 + scale * (turn[i] + turn[(i + 1) % n]) * 0.5
        )

    return cum


def _periodic_intervals(
    params: Sequence[float],
    cum: Sequence[float],
    n: int,
    num_cvs: int,
    period: float,
) -> list[float]:
    """Periodic nurbsknot intervals that split the weighted length into equal shares."""

    total = cum[n]
    base = [0.0] * num_cvs

    for j in range(num_cvs):
        base[j] = _locate_target(params, cum, n - 1, total * j / num_cvs)

    intervals = [0.0] * num_cvs

    for j in range(num_cvs - 1):
        intervals[j] = base[j + 1] - base[j]

    intervals[num_cvs - 1] = period - base[num_cvs - 1]

    return intervals


def build_fitted_nurbsknots_periodic_adaptive(
    params: Sequence[float],
    points: Sequence[float] | None,
    n: int,
    dim: int,
    num_cvs: int,
    degree: int,
    scale: float = 3.0,
) -> list[float]:
    """Return a periodic fitting vector with denser nurbsknots where the closed points turn."""

    if (
        n < 0
        or degree < 1
        or degree - 1 > num_cvs
        or not math.isfinite(scale)
        or len(params) <= n
        or not _are_finite(params, n + 1)
    ):
        return []

    cv_count = num_cvs + degree
    order = degree + 1
    kc = nurbsknot_count(order, cv_count)

    if kc == 0:
        return []

    period = params[n]
    nurbsknots = [0.0] * kc

    if not math.isfinite(period) or period <= 0.0:
        return []

    if n < 3 or points is None:
        delta = period / num_cvs

        for i in range(kc):
            nurbsknots[i] = (i - degree + 1) * delta

        return nurbsknots

    if dim < 1 or not _are_finite(points, n * dim):
        return []

    cum = _periodic_weighted_lengths(params, points, n, dim, scale)
    intervals = _periodic_intervals(params, cum, n, num_cvs, period)

    for i in range(1, degree):
        nurbsknots[degree - 1 - i] = nurbsknots[degree - i] - intervals[num_cvs - i]

    for i in range(kc - degree):
        nurbsknots[degree + i] = nurbsknots[degree - 1 + i] + intervals[i % num_cvs]

    return nurbsknots


def _banded_cholesky(n: int, half_bw: int, band: MutableSequence[float]) -> bool:
    """Cholesky factor of a banded symmetric matrix in place; false when it is not positive definite."""

    bw1 = half_bw + 1

    for i in range(n):
        for j in range(max(0, i - half_bw), i + 1):
            sum = 0.0

            for k in range(max(0, i - half_bw), j):
                i_index = i * bw1 + i - k
                j_index = j * bw1 + j - k
                sum += band[i_index] * band[j_index]

            index = i * bw1 + i - j

            if i == j:
                val = band[index] - sum

                if val <= POSITIVE_DEFINITE_TOLERANCE:
                    return False

                band[index] = math.sqrt(val)
            else:
                band[index] = (band[index] - sum) / band[j * bw1]

    return True


def _banded_forward_substitute(
    dim: int,
    n: int,
    half_bw: int,
    band: Sequence[float],
    rhs: MutableSequence[float],
) -> None:
    """Forward substitution with the lower banded Cholesky factor, in place on rhs."""

    bw1 = half_bw + 1

    for i in range(n):
        for d in range(dim):
            sum = 0.0

            for k in range(max(0, i - half_bw), i):
                sum += band[i * bw1 + i - k] * rhs[k * dim + d]

            index = i * dim + d
            rhs[index] = (rhs[index] - sum) / band[i * bw1]


def _banded_back_substitute(
    dim: int,
    n: int,
    half_bw: int,
    band: Sequence[float],
    rhs: MutableSequence[float],
) -> None:
    """Back substitution with the transposed banded Cholesky factor, in place on rhs."""

    bw1 = half_bw + 1

    for i in range(n - 1, -1, -1):
        for d in range(dim):
            sum = 0.0
            upper = min(n, i + bw1)

            for k in range(i + 1, upper):
                sum += band[k * bw1 + k - i] * rhs[k * dim + d]

            index = i * dim + d
            rhs[index] = (rhs[index] - sum) / band[i * bw1]


def solve_banded_spd(
    dim: int,
    n: int,
    half_bw: int,
    band: MutableSequence[float],
    rhs: MutableSequence[float],
) -> bool:
    """Solve a finite banded symmetric positive-definite system in place with Cholesky factorization."""

    if dim < 1 or n < 1 or half_bw < 0:
        return False

    band_count = n * (half_bw + 1)
    rhs_count = n * dim

    if not _are_finite(band, band_count) or not _are_finite(rhs, rhs_count):
        return False

    if not _banded_cholesky(n, half_bw, band):
        return False

    _banded_forward_substitute(dim, n, half_bw, band, rhs)
    _banded_back_substitute(dim, n, half_bw, band, rhs)

    return True
