from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import Tolerance
import math
import sys


@MINI_TEST("NurbsKnot", "Nurbsknot Count")
def test_nurbsknot_count():
    from session_py import nurbsknot

    MINI_CHECK(nurbsknot.nurbsknot_count(4, 5) == 7)
    MINI_CHECK(nurbsknot.nurbsknot_count(0, 0) == 0)
    MINI_CHECK(nurbsknot.nurbsknot_count(4, 3) == 0)
    MINI_CHECK(nurbsknot.nurbsknot_count(2, sys.maxsize) == sys.maxsize)
    MINI_CHECK(nurbsknot.nurbsknot_count(sys.maxsize, sys.maxsize) == 0)


@MINI_TEST("NurbsKnot", "Domain Tolerance")
def test_domain_tolerance():
    from session_py import nurbsknot

    MINI_CHECK(nurbsknot.domain_tolerance(1.0, 1.0) == 0.0)
    MINI_CHECK(
        TOLERANCE.is_close(nurbsknot.domain_tolerance(0.0, 1.0), 2.980232238769531e-08)
    )
    MINI_CHECK(
        nurbsknot.domain_tolerance(0.0, float.fromhex("0x0.0000000000001p-1022"))
        == sys.float_info.epsilon
    )


@MINI_TEST("NurbsKnot", "Compute Clamped Uniform")
def test_compute_clamped_uniform():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.compute_clamped_uniform(order, cv_count)

    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]))
    MINI_CHECK(len(nurbsknot.compute_clamped_uniform(1, cv_count)) == 0)
    MINI_CHECK(
        len(nurbsknot.compute_clamped_uniform(order, cv_count, float("nan"))) == 0
    )
    MINI_CHECK(len(nurbsknot.compute_clamped_uniform(sys.maxsize, sys.maxsize)) == 0)


@MINI_TEST("NurbsKnot", "Compute Periodic Uniform")
def test_compute_periodic_uniform():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.compute_periodic_uniform(order, cv_count)

    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    MINI_CHECK(len(nurbsknot.compute_periodic_uniform(order, cv_count, 0.0)) == 0)
    MINI_CHECK(
        len(nurbsknot.compute_periodic_uniform(order, cv_count, float("inf"))) == 0
    )


@MINI_TEST("NurbsKnot", "Clamp")
def test_clamp():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots = [9.0, 9.0, 0.0, 1.0, 2.0, 9.0, 9.0]
    ok = nurbsknot.clamp(order, cv_count, nurbsknots)

    MINI_CHECK(ok)
    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]))

    left = [9.0, 9.0, 0.0, 1.0, 2.0, 8.0, 9.0]
    right = [9.0, 8.0, 0.0, 1.0, 2.0, 9.0, 9.0]

    MINI_CHECK(nurbsknot.clamp(order, cv_count, left, 0))
    MINI_CHECK(nurbsknot.clamp(order, cv_count, right, 1))
    MINI_CHECK(TOLERANCE.is_allclose(left, [0.0, 0.0, 0.0, 1.0, 2.0, 8.0, 9.0]))
    MINI_CHECK(TOLERANCE.is_allclose(right, [9.0, 8.0, 0.0, 1.0, 2.0, 2.0, 2.0]))
    MINI_CHECK(not nurbsknot.clamp(order, cv_count, right, 3))


@MINI_TEST("NurbsKnot", "Is Valid")
def test_is_valid():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots_clamped = nurbsknot.compute_clamped_uniform(order, cv_count)
    nurbsknots_flat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    nurbsknots_nan = list(nurbsknots_clamped)
    nurbsknots_nan[3] = float("nan")

    MINI_CHECK(nurbsknot.is_valid(order, cv_count, nurbsknots_clamped))
    MINI_CHECK(not nurbsknot.is_valid(order, cv_count, nurbsknots_flat))
    MINI_CHECK(not nurbsknot.is_valid(order, cv_count, nurbsknots_nan))
    MINI_CHECK(not nurbsknot.is_valid(order, cv_count, [0.0, 1.0]))


@MINI_TEST("NurbsKnot", "Is Clamped")
def test_is_clamped():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots_periodic = nurbsknot.compute_periodic_uniform(order, cv_count)
    nurbsknots_clamped = nurbsknot.compute_clamped_uniform(order, cv_count)
    is_not_clamped = nurbsknot.is_clamped(order, cv_count, nurbsknots_periodic)
    is_clamped = nurbsknot.is_clamped(order, cv_count, nurbsknots_clamped)

    MINI_CHECK(not is_not_clamped and is_clamped)
    MINI_CHECK(nurbsknot.is_clamped(order, cv_count, nurbsknots_clamped, 0))
    MINI_CHECK(nurbsknot.is_clamped(order, cv_count, nurbsknots_clamped, 1))
    MINI_CHECK(not nurbsknot.is_clamped(order, cv_count, nurbsknots_clamped, 3))


@MINI_TEST("NurbsKnot", "Is Periodic")
def test_is_periodic():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots_periodic = nurbsknot.compute_periodic_uniform(order, cv_count)
    nurbsknots_clamped = nurbsknot.compute_clamped_uniform(order, cv_count)

    MINI_CHECK(nurbsknot.is_periodic(order, cv_count, nurbsknots_periodic))
    MINI_CHECK(not nurbsknot.is_periodic(order, cv_count, nurbsknots_clamped))

    nurbsknots_periodic[3] = float("nan")

    MINI_CHECK(not nurbsknot.is_periodic(order, cv_count, nurbsknots_periodic))


@MINI_TEST("NurbsKnot", "Get Domain")
def test_get_domain():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.compute_clamped_uniform(order, cv_count)
    domain = nurbsknot.get_domain(order, cv_count, nurbsknots)

    MINI_CHECK(TOLERANCE.is_close(domain[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(domain[1], 2.0))

    nurbsknots[3] = float("nan")

    MINI_CHECK(nurbsknot.get_domain(order, cv_count, nurbsknots) == domain)

    nurbsknots[2] = float("nan")

    MINI_CHECK(nurbsknot.get_domain(order, cv_count, nurbsknots) == (0.0, 0.0))


@MINI_TEST("NurbsKnot", "Set Domain")
def test_set_domain():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.compute_clamped_uniform(order, cv_count)
    ok = nurbsknot.set_domain(order, cv_count, nurbsknots, 0.0, 1.0)

    MINI_CHECK(ok)
    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]))
    MINI_CHECK(not nurbsknot.set_domain(order, cv_count, nurbsknots, 1.0, 1.0))
    MINI_CHECK(not nurbsknot.set_domain(order, cv_count, nurbsknots, 0.0, float("nan")))


@MINI_TEST("NurbsKnot", "Reverse")
def test_reverse():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots_sym = nurbsknot.compute_clamped_uniform(order, cv_count)

    MINI_CHECK(nurbsknot.reverse(order, cv_count, nurbsknots_sym))
    MINI_CHECK(
        TOLERANCE.is_allclose(nurbsknots_sym, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    )

    nurbsknots_asym = [0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 2.0, 2.0]

    MINI_CHECK(nurbsknot.reverse(4, 6, nurbsknots_asym))
    MINI_CHECK(
        TOLERANCE.is_allclose(nurbsknots_asym, [0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.0, 2.0])
    )

    nurbsknots_asym[3] = float("inf")

    MINI_CHECK(not nurbsknot.reverse(4, 6, nurbsknots_asym))


@MINI_TEST("NurbsKnot", "Multiplicity")
def test_multiplicity():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.compute_clamped_uniform(order, cv_count)

    MINI_CHECK(nurbsknot.multiplicity(order, cv_count, nurbsknots, 0) == 3)
    MINI_CHECK(nurbsknot.multiplicity(order, cv_count, nurbsknots, 3) == 1)
    MINI_CHECK(nurbsknot.multiplicity(order, cv_count, nurbsknots, 7) == 0)

    nurbsknots[3] = float("nan")

    MINI_CHECK(nurbsknot.multiplicity(order, cv_count, nurbsknots, 3) == 0)


@MINI_TEST("NurbsKnot", "Span Count")
def test_span_count():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.compute_clamped_uniform(order, cv_count)

    MINI_CHECK(nurbsknot.span_count(order, cv_count, nurbsknots) == 2)

    nurbsknots[3] = float("nan")

    MINI_CHECK(nurbsknot.span_count(order, cv_count, nurbsknots) == 0)


@MINI_TEST("NurbsKnot", "Find Span")
def test_find_span():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots_clamped = nurbsknot.compute_clamped_uniform(order, cv_count)
    spancount0 = nurbsknot.find_span(order, cv_count, nurbsknots_clamped, 0.5)
    spancount1 = nurbsknot.find_span(order, cv_count, nurbsknots_clamped, 1.5)

    MINI_CHECK(spancount0 == 0 and spancount1 == 1)
    MINI_CHECK(nurbsknot.find_span(order, cv_count, nurbsknots_clamped, -1.0) == 0)
    MINI_CHECK(nurbsknot.find_span(order, cv_count, nurbsknots_clamped, 3.0) == 1)
    MINI_CHECK(
        nurbsknot.find_span(order, cv_count, nurbsknots_clamped, 0.5, -1, 42) == 0
    )
    MINI_CHECK(
        nurbsknot.find_span(order, cv_count, nurbsknots_clamped, float("nan")) == 0
    )


@MINI_TEST("NurbsKnot", "Get Greville Abcissae")
def test_get_greville_abcissae():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.compute_clamped_uniform(order, cv_count)
    greville = nurbsknot.get_greville_abcissae(order, cv_count, nurbsknots)
    periodic = nurbsknot.get_greville_abcissae(order, cv_count, nurbsknots, True)

    MINI_CHECK(TOLERANCE.is_allclose(greville, [0.0, 1.0 / 3.0, 1.0, 5.0 / 3.0, 2.0]))
    MINI_CHECK(TOLERANCE.is_allclose(periodic, [0.0, 1.0 / 3.0]))

    nurbsknots[2] = float("inf")

    MINI_CHECK(len(nurbsknot.get_greville_abcissae(order, cv_count, nurbsknots)) == 0)


@MINI_TEST("NurbsKnot", "Solve Tridiagonal")
def test_solve_tridiagonal():
    from session_py import nurbsknot

    lo = [0.0, 1.0]
    di = [2.0, 2.0]
    up = [1.0, 0.0]
    rh = [3.0, 3.0]
    sol = nurbsknot.solve_tridiagonal(1, 2, lo, di, up, rh)

    MINI_CHECK(sol is not None)
    MINI_CHECK(TOLERANCE.is_allclose(sol, [1.0, 1.0]))

    rh2 = [3.0, 0.0, 3.0, 3.0]
    sol = nurbsknot.solve_tridiagonal(2, 2, lo, di, up, rh2)

    MINI_CHECK(sol is not None)
    MINI_CHECK(TOLERANCE.is_allclose(sol, [1.0, -1.0, 1.0, 2.0]))

    singular = [0.0, 2.0]

    MINI_CHECK(nurbsknot.solve_tridiagonal(1, 2, lo, singular, up, rh) is None)
    MINI_CHECK(nurbsknot.solve_tridiagonal(sys.maxsize, 2, lo, di, up, rh) is None)


@MINI_TEST("NurbsKnot", "Compute Parameters")
def test_compute_parameters():
    from session_py import nurbsknot

    pts = [0.0, 0.0, 4.0, 0.0, 4.0, 9.0]
    uniform = nurbsknot.compute_parameters(
        pts, 3, 2, nurbsknot.CurveNurbsKnotStyle.Uniform
    )
    chord = nurbsknot.compute_parameters(pts, 3, 2, nurbsknot.CurveNurbsKnotStyle.Chord)
    root = nurbsknot.compute_parameters(
        pts, 3, 2, nurbsknot.CurveNurbsKnotStyle.ChordSquareRoot
    )
    periodic = nurbsknot.compute_parameters(
        pts, 3, 2, nurbsknot.CurveNurbsKnotStyle.ChordPeriodic
    )

    MINI_CHECK(TOLERANCE.is_allclose(uniform, [0.0, 1.0, 2.0]))
    MINI_CHECK(TOLERANCE.is_allclose(chord, [0.0, 4.0, 13.0]))
    MINI_CHECK(TOLERANCE.is_allclose(root, [0.0, 2.0, 5.0]))
    MINI_CHECK(TOLERANCE.is_allclose(periodic, chord))
    MINI_CHECK(
        nurbsknot.CurveInterpStyle.Rhino.value == 0
        and nurbsknot.CurveInterpStyle.Occt.value == 1
    )
    MINI_CHECK(
        len(
            nurbsknot.compute_parameters(
                None, 3, 2, nurbsknot.CurveNurbsKnotStyle.Chord
            )
        )
        == 0
    )


@MINI_TEST("NurbsKnot", "Build Interp Nurbsknots")
def test_build_interp_nurbsknots():
    from session_py import nurbsknot

    params = [0.0, 1.0, 2.0, 3.0]
    degree = 3
    nurbsknots = nurbsknot.build_interp_nurbsknots(params, degree)

    MINI_CHECK(
        TOLERANCE.is_allclose(nurbsknots, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    )

    params[2] = float("nan")

    MINI_CHECK(len(nurbsknot.build_interp_nurbsknots(params, degree)) == 0)
    MINI_CHECK(len(nurbsknot.build_interp_nurbsknots([0.0, 1.0, 2.0], 5)) == 0)


@MINI_TEST("NurbsKnot", "Eval Basis")
def test_eval_basis():
    from session_py import nurbsknot

    order = 4
    cv_count = 5
    nurbsknots = nurbsknot.compute_clamped_uniform(order, cv_count)
    span = nurbsknot.find_span(order, cv_count, nurbsknots, 0.5)
    basis = nurbsknot.eval_basis(order, nurbsknots, span, 0.5)
    nan = float("nan")

    MINI_CHECK(TOLERANCE.is_allclose(basis, [0.125, 0.59375, 0.25, 0.03125]))
    MINI_CHECK(TOLERANCE.is_allclose(nurbsknot.eval_basis(1, [], 0, 0.5), [1.0]))
    MINI_CHECK(len(nurbsknot.eval_basis(0, [], 0, 0.5)) == 0)
    MINI_CHECK(len(nurbsknot.eval_basis(order, [0.0], span, 0.5)) == 0)
    MINI_CHECK(
        TOLERANCE.is_allclose(
            nurbsknot.eval_basis(3, [nan, -1.0, 0.0, 1.0, 2.0, 3.0], 2, 1.5),
            [0.125, 0.75, 0.125],
        )
    )
    MINI_CHECK(
        len(nurbsknot.eval_basis(3, [-2.0, -1.0, nan, 1.0, 2.0, 3.0], 2, 1.5)) == 0
    )


@MINI_TEST("NurbsKnot", "Build Fitted Nurbsknots Adaptive")
def test_build_fitted_nurbsknots_adaptive():
    from session_py import nurbsknot

    pts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0]
    params = nurbsknot.compute_parameters(
        pts, 5, 3, nurbsknot.CurveNurbsKnotStyle.Chord
    )
    nurbsknots = nurbsknot.build_fitted_nurbsknots_adaptive(params, pts, 5, 3, 5, 3)
    fallback = nurbsknot.build_fitted_nurbsknots_adaptive(params, None, 5, 3, 5, 3)
    dense = nurbsknot.build_fitted_nurbsknots_adaptive(
        [0.0, 1.0, 2.0], pts, 3, 3, 5, 1, 1.0
    )

    MINI_CHECK(TOLERANCE.is_allclose(nurbsknots, [0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0]))
    MINI_CHECK(TOLERANCE.is_allclose(fallback, [0.0, 0.0, 0.0, 1.5, 4.0, 4.0, 4.0]))
    MINI_CHECK(
        len(nurbsknot.build_fitted_nurbsknots_adaptive(params, pts, 5, 3, 3, 3)) == 0
    )
    MINI_CHECK(
        len(
            nurbsknot.build_fitted_nurbsknots_adaptive(
                [0.0, 1.0], None, 2, 3, 4, 1, 1.0
            )
        )
        == 0
    )
    MINI_CHECK(TOLERANCE.is_allclose(dense, [0.0, 0.5, 1.0, 1.5, 2.0]))


@MINI_TEST("NurbsKnot", "Build Fitted Nurbsknots Periodic Adaptive")
def test_build_fitted_nurbsknots_periodic_adaptive():
    from session_py import nurbsknot

    pts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    params = [0.0, 1.0, 2.0, 3.0, 4.0]
    nurbsknots = nurbsknot.build_fitted_nurbsknots_periodic_adaptive(
        params, pts, 4, 3, 4, 3
    )
    fallback = nurbsknot.build_fitted_nurbsknots_periodic_adaptive(
        [0.0, 1.0, 2.0], None, 2, 3, 4, 3
    )
    boundary = nurbsknot.build_fitted_nurbsknots_periodic_adaptive(
        [0.0, 1.0, 2.0, 3.0], pts, 3, 3, 1, 2, 1.0
    )

    MINI_CHECK(
        TOLERANCE.is_allclose(
            nurbsknots, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
    )
    MINI_CHECK(
        TOLERANCE.is_allclose(fallback, [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    )
    MINI_CHECK(
        len(
            nurbsknot.build_fitted_nurbsknots_periodic_adaptive([0.0], None, 0, 3, 4, 3)
        )
        == 0
    )
    MINI_CHECK(
        len(
            nurbsknot.build_fitted_nurbsknots_periodic_adaptive(
                [0.0, 1.0, 2.0, 3.0], pts, 3, 3, 1, 3, 1.0
            )
        )
        == 0
    )
    MINI_CHECK(TOLERANCE.is_allclose(boundary, [-3.0, 0.0, 3.0, 6.0]))


@MINI_TEST("NurbsKnot", "Solve Banded SPD")
def test_solve_banded_spd():
    from session_py import nurbsknot

    band = [4.0, 0.0, 5.0, 2.0, 3.0, 1.0]
    rhs = [8.0, 13.0, 5.0]

    MINI_CHECK(nurbsknot.solve_banded_spd(1, 3, 1, band, rhs))
    MINI_CHECK(TOLERANCE.is_allclose(rhs, [1.0, 2.0, 1.0]))

    singular = [0.0, 0.0]
    value = [1.0]

    MINI_CHECK(not nurbsknot.solve_banded_spd(1, 1, 1, singular, value))
    MINI_CHECK(not nurbsknot.solve_banded_spd(1, 2, 1, singular, value))

    cutoff_value = Tolerance.ABSOLUTE * Tolerance.ABSOLUTE * Tolerance.ZERO_TOLERANCE
    cutoff = [cutoff_value]
    value = [1.0]

    MINI_CHECK(not nurbsknot.solve_banded_spd(1, 1, 0, cutoff, value))

    cutoff = [math.nextafter(cutoff_value, math.inf)]
    value = [1.0]

    MINI_CHECK(nurbsknot.solve_banded_spd(1, 1, 0, cutoff, value))
    MINI_CHECK(not nurbsknot.solve_banded_spd(sys.maxsize, 2, 1, singular, value))


if __name__ == "__main__":
    run_all("python")
