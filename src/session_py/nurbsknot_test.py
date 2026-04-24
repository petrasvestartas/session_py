"""Tests for nurbsknot module using mini_test framework."""

from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("NurbsKnot", "Make Clamped Uniform")
def test_make_clamped_uniform():
    from session_py import nurbsknot

    # 0 0 0 1 2 2 2
    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.make_clamped_uniform(order, cv_count)
    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]))


@MINI_TEST("NurbsKnot", "Make Periodic Uniform")
def test_make_periodic_uniform():
    from session_py import nurbsknot

    # 0 1 2 3 4 5 6
    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.make_periodic_uniform(order, cv_count)
    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@MINI_TEST("NurbsKnot", "Is Clamped")
def test_is_clamped():
    from session_py import nurbsknot

    # 0 0 0 1 2 2 2
    # 0 1 2 3 4 5 6
    order = 4
    cv_count = 5
    nurbsknots_periodic = nurbsknot.make_periodic_uniform(order, cv_count)
    nurbsknots_clamped = nurbsknot.make_clamped_uniform(order, cv_count)
    is_not_clamped = nurbsknot.is_clamped(order, cv_count, nurbsknots_periodic)
    is_clamped = nurbsknot.is_clamped(order, cv_count, nurbsknots_clamped)
    MINI_CHECK(not is_not_clamped and is_clamped)


@MINI_TEST("NurbsKnot", "Reverse")
def test_reverse():
    from session_py import nurbsknot

    # Symmetric nurbsknot vector -> reverse gives back the same (palindrome)
    # 0 0 0 1 2 2 2
    order = 4
    cv_count = 5
    nurbsknots_sym = nurbsknot.make_clamped_uniform(order, cv_count)
    nurbsknot.reverse(order, cv_count, nurbsknots_sym)
    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots_sym, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]))

    # Asymmetric nurbsknot vector -> extra nurbsknot at 0.5 shifts to 1.5 after reverse
    # 0 0 0 0.5 1 2 2 2 -> 0 0 0 1 1.5 2 2 2
    nurbsknots_asym = [0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 2.0, 2.0]
    nurbsknot.reverse(4, 6, nurbsknots_asym)
    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots_asym, [0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.0, 2.0]))


@MINI_TEST("NurbsKnot", "Find Span")
def test_find_span():
    from session_py import nurbsknot

    # 0 0 0 1 2 2 2
    order = 4
    cv_count = 5
    nurbsknots_clamped = nurbsknot.make_clamped_uniform(order, cv_count)
    #   - 0.5 falls in span [0, 1] -> index 0
    #   - 1.5 falls in span [1, 2] -> index 1
    spancount0 = nurbsknot.find_span(order, cv_count, nurbsknots_clamped, 0.5)
    spancount1 = nurbsknot.find_span(order, cv_count, nurbsknots_clamped, 1.5)
    MINI_CHECK(spancount0 == 0 and spancount1 == 1)


@MINI_TEST("NurbsKnot", "Solve Tridiagonal")
def test_solve_tridiagonal():
    from session_py import nurbsknot

    # Thomas algorithm -- an O(n) solver for tridiagonal linear systems
    #   | 2 1 | |x0|   |3|
    #   | 1 2 | |x1| = |3|
    #   -> solution: x0 = 1, x1 = 1
    lo = [0, 1]
    di = [2, 2]
    up = [1, 0]
    rh = [3, 3]
    sol = nurbsknot.solve_tridiagonal(1, 2, lo, di, up, rh)
    MINI_CHECK(TOLERANCE.is_allclose(sol, [1.0, 1.0]))


@MINI_TEST("NurbsKnot", "Compute Parameters")
def test_compute_parameters():
    from session_py import nurbsknot
    import numpy as np

    pts = np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0]], dtype=float)
    # Chord-length parameterization: since all gaps are 1.0, params = {0, 1, 2, 3}
    t = nurbsknot.compute_parameters(pts, nurbsknot.CurveNurbsKnotStyle.Chord)
    MINI_CHECK(TOLERANCE.is_allclose(t, [0.0, 1.0, 2.0, 3.0]))


@MINI_TEST("NurbsKnot", "Build Interpolation NurbsKnots")
def test_build_interp_nurbsknots():
    from session_py import nurbsknot

    params = [0.0, 1.0, 2.0, 3.0]
    degree = 3
    # cv_count = n + 2 = 6 (natural end conditions add 2 CVs)
    # kc = order + cv_count - 2 = 4 + 6 - 2 = 8
    #   [0, 0, 0,  |  1, 2,  |  3, 3, 3]
    #   <-clamp->    interior    <-clamp->
    nurbsknots = nurbsknot.build_interp_nurbsknots(params, degree)
    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]))


@MINI_TEST("NurbsKnot", "Evaluation Basis")
def test_eval_basis():
    from session_py import nurbsknot

    # Cox-de Boor recursive evaluation of B-spline basis functions
    # At parameter t, exactly 'order' basis functions are non-zero
    # Partition of unity: they always sum to 1.0
    # Used to evaluate NURBS curves/surfaces: C(t) = sum(N_i(t) * P_i)
    # 0 0 0 1 2 2 2
    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.make_clamped_uniform(order, cv_count)
    span = nurbsknot.find_span(order, cv_count, nurbsknots, 0.5)
    basis = nurbsknot.eval_basis(order, nurbsknots, span, 0.5)
    MINI_CHECK(TOLERANCE.is_allclose(basis, [0.125, 0.59375, 0.25, 0.03125]))


@MINI_TEST("NurbsKnot", "Build Fitted NurbsKnots Adaptive")
def test_build_fitted_nurbsknots_adaptive():
    from session_py import nurbsknot

    # Builds nurbsknot vectors for least-squares fitting
    # Concentrates nurbsknots where curvature is high (sharp turns)
    # For collinear points (zero curvature), interior nurbsknots are evenly distributed
    pts = [0,0,0, 1,0,0, 2,0,0, 3,0,0, 4,0,0]
    params = [0.0, 1.0, 2.0, 3.0, 4.0]
    nurbsknots = nurbsknot.build_fitted_nurbsknots_adaptive(params, pts, 5, 3, 5, 3)
    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots, [0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0]))


@MINI_TEST("NurbsKnot", "Build Fitted NurbsKnots Periodic Adaptive")
def test_build_fitted_nurbsknots_periodic_adaptive():
    from session_py import nurbsknot

    # Periodic version for closed curves -- nurbsknots wrap around
    # For a regular square (equal turns, equal chords), nurbsknots are uniformly spaced
    pts = [0,0,0, 1,0,0, 1,1,0, 0,1,0]
    params = [0.0, 1.0, 2.0, 3.0, 4.0]
    nurbsknots = nurbsknot.build_fitted_nurbsknots_periodic_adaptive(params, pts, 4, 3, 4, 3)
    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@MINI_TEST("NurbsKnot", "Solve Banded SPD")
def test_solve_banded_spd():
    from session_py import nurbsknot

    # Cholesky solver for banded symmetric positive-definite systems
    #   | 4 2 0 |       |8 |       |1|
    #   | 2 5 1 | * x = |13| -> x = |2|
    #   | 0 1 3 |       |5 |       |1|
    band = [4, 0, 5, 2, 3, 1]
    rhs = [8, 13, 5]
    nurbsknot.solve_banded_spd(1, 3, 1, band, rhs)
    MINI_CHECK(TOLERANCE.is_allclose(rhs, [1.0, 2.0, 1.0]))


if __name__ == "__main__":
    run_all("python")
