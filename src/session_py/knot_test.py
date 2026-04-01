"""Tests for knot module using mini_test framework."""

from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Knot", "Make Clamped Uniform")
def test_make_clamped_uniform():
    from session_py import knot

    # 0 0 0 1 2 2 2
    order = 4
    cv_count = 5
    knots = knot.make_clamped_uniform(order, cv_count)
    MINI_CHECK(TOLERANCE.is_allclose(knots, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]))


@MINI_TEST("Knot", "Make Periodic Uniform")
def test_make_periodic_uniform():
    from session_py import knot

    # 0 1 2 3 4 5 6
    order = 4
    cv_count = 5
    knots = knot.make_periodic_uniform(order, cv_count)
    MINI_CHECK(TOLERANCE.is_allclose(knots, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@MINI_TEST("Knot", "Is Clamped")
def test_is_clamped():
    from session_py import knot

    # 0 0 0 1 2 2 2
    # 0 1 2 3 4 5 6
    order = 4
    cv_count = 5
    knots_periodic = knot.make_periodic_uniform(order, cv_count)
    knots_clamped = knot.make_clamped_uniform(order, cv_count)
    is_not_clamped = knot.is_clamped(order, cv_count, knots_periodic)
    is_clamped = knot.is_clamped(order, cv_count, knots_clamped)
    MINI_CHECK(not is_not_clamped and is_clamped)


@MINI_TEST("Knot", "Reverse")
def test_reverse():
    from session_py import knot

    # Symmetric knot vector -> reverse gives back the same (palindrome)
    # 0 0 0 1 2 2 2
    order = 4
    cv_count = 5
    knots_sym = knot.make_clamped_uniform(order, cv_count)
    knot.reverse(order, cv_count, knots_sym)
    MINI_CHECK(TOLERANCE.is_allclose(knots_sym, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]))

    # Asymmetric knot vector -> extra knot at 0.5 shifts to 1.5 after reverse
    # 0 0 0 0.5 1 2 2 2 -> 0 0 0 1 1.5 2 2 2
    knots_asym = [0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 2.0, 2.0]
    knot.reverse(4, 6, knots_asym)
    MINI_CHECK(TOLERANCE.is_allclose(knots_asym, [0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.0, 2.0]))


@MINI_TEST("Knot", "Find Span")
def test_find_span():
    from session_py import knot

    # 0 0 0 1 2 2 2
    order = 4
    cv_count = 5
    knots_clamped = knot.make_clamped_uniform(order, cv_count)
    #   - 0.5 falls in span [0, 1] -> index 0
    #   - 1.5 falls in span [1, 2] -> index 1
    spancount0 = knot.find_span(order, cv_count, knots_clamped, 0.5)
    spancount1 = knot.find_span(order, cv_count, knots_clamped, 1.5)
    MINI_CHECK(spancount0 == 0 and spancount1 == 1)


@MINI_TEST("Knot", "Solve Tridiagonal")
def test_solve_tridiagonal():
    from session_py import knot

    # Thomas algorithm -- an O(n) solver for tridiagonal linear systems
    #   | 2 1 | |x0|   |3|
    #   | 1 2 | |x1| = |3|
    #   -> solution: x0 = 1, x1 = 1
    lo = [0, 1]
    di = [2, 2]
    up = [1, 0]
    rh = [3, 3]
    sol = knot.solve_tridiagonal(1, 2, lo, di, up, rh)
    MINI_CHECK(TOLERANCE.is_allclose(sol, [1.0, 1.0]))


@MINI_TEST("Knot", "Compute Parameters")
def test_compute_parameters():
    from session_py import knot
    import numpy as np

    pts = np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0]], dtype=float)
    # Chord-length parameterization: since all gaps are 1.0, params = {0, 1, 2, 3}
    t = knot.compute_parameters(pts, knot.CurveKnotStyle.Chord)
    MINI_CHECK(TOLERANCE.is_allclose(t, [0.0, 1.0, 2.0, 3.0]))


@MINI_TEST("Knot", "Build Interpolation Knots")
def test_build_interp_knots():
    from session_py import knot

    params = [0.0, 1.0, 2.0, 3.0]
    degree = 3
    # cv_count = n + 2 = 6 (natural end conditions add 2 CVs)
    # kc = order + cv_count - 2 = 4 + 6 - 2 = 8
    #   [0, 0, 0,  |  1, 2,  |  3, 3, 3]
    #   <-clamp->    interior    <-clamp->
    knots = knot.build_interp_knots(params, degree)
    MINI_CHECK(TOLERANCE.is_allclose(knots, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]))


@MINI_TEST("Knot", "Evaluation Basis")
def test_eval_basis():
    from session_py import knot

    # Cox-de Boor recursive evaluation of B-spline basis functions
    # At parameter t, exactly 'order' basis functions are non-zero
    # Partition of unity: they always sum to 1.0
    # Used to evaluate NURBS curves/surfaces: C(t) = sum(N_i(t) * P_i)
    # 0 0 0 1 2 2 2
    order = 4
    cv_count = 5
    knots = knot.make_clamped_uniform(order, cv_count)
    span = knot.find_span(order, cv_count, knots, 0.5)
    basis = knot.eval_basis(order, knots, span, 0.5)
    MINI_CHECK(TOLERANCE.is_allclose(basis, [0.125, 0.59375, 0.25, 0.03125]))


@MINI_TEST("Knot", "Build Fitted Knots Adaptive")
def test_build_fitted_knots_adaptive():
    from session_py import knot

    # Builds knot vectors for least-squares fitting
    # Concentrates knots where curvature is high (sharp turns)
    # For collinear points (zero curvature), interior knots are evenly distributed
    pts = [0,0,0, 1,0,0, 2,0,0, 3,0,0, 4,0,0]
    params = [0.0, 1.0, 2.0, 3.0, 4.0]
    knots = knot.build_fitted_knots_adaptive(params, pts, 5, 3, 5, 3)
    MINI_CHECK(TOLERANCE.is_allclose(knots, [0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0]))


@MINI_TEST("Knot", "Build Fitted Knots Periodic Adaptive")
def test_build_fitted_knots_periodic_adaptive():
    from session_py import knot

    # Periodic version for closed curves -- knots wrap around
    # For a regular square (equal turns, equal chords), knots are uniformly spaced
    pts = [0,0,0, 1,0,0, 1,1,0, 0,1,0]
    params = [0.0, 1.0, 2.0, 3.0, 4.0]
    knots = knot.build_fitted_knots_periodic_adaptive(params, pts, 4, 3, 4, 3)
    MINI_CHECK(TOLERANCE.is_allclose(knots, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@MINI_TEST("Knot", "Solve Banded SPD")
def test_solve_banded_spd():
    from session_py import knot

    # Cholesky solver for banded symmetric positive-definite systems
    #   | 4 2 0 |       |8 |       |1|
    #   | 2 5 1 | * x = |13| -> x = |2|
    #   | 0 1 3 |       |5 |       |1|
    band = [4, 0, 5, 2, 3, 1]
    rhs = [8, 13, 5]
    knot.solve_banded_spd(1, 3, 1, band, rhs)
    MINI_CHECK(TOLERANCE.is_allclose(rhs, [1.0, 2.0, 1.0]))


if __name__ == "__main__":
    run_all("python")
