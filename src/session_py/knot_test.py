"""Tests for knot module using mini_test framework."""

from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("Knot", "knot_count")
def test_knot_count():
    from session_py import knot

    # Calculate knot counts for various order/cv_count combinations
    count1 = knot.knot_count(2, 2)
    count2 = knot.knot_count(3, 3)
    count3 = knot.knot_count(4, 4)
    count4 = knot.knot_count(4, 5)
    count5 = knot.knot_count(3, 4)

    MINI_CHECK(count1 == 2)
    MINI_CHECK(count2 == 4)
    MINI_CHECK(count3 == 6)
    MINI_CHECK(count4 == 7)
    MINI_CHECK(count5 == 5)


@MINI_TEST("Knot", "make_clamped_uniform")
def test_make_clamped_uniform():
    from session_py import knot

    # Basic clamped uniform knot vector
    k = knot.make_clamped_uniform(4, 4, 1.0)
    k_len = len(k)
    k0, k1, k2 = k[0], k[1], k[2]
    k3, k4, k5 = k[3], k[4], k[5]

    # With custom delta
    k2_vec = knot.make_clamped_uniform(3, 4, 2.5)
    k2_len = len(k2_vec)
    t0, t1 = knot.get_domain(3, 4, k2_vec)

    # Invalid params
    k_invalid1 = knot.make_clamped_uniform(1, 2, 1.0)
    k_invalid2 = knot.make_clamped_uniform(4, 3, 1.0)

    MINI_CHECK(k is not None)
    MINI_CHECK(k_len == 6)
    MINI_CHECK(k0 == 0.0 and k1 == 0.0 and k2 == 0.0)
    MINI_CHECK(k3 == 1.0 and k4 == 1.0 and k5 == 1.0)
    MINI_CHECK(k2_len == 5)
    MINI_CHECK(t0 == 0.0 and t1 == 5.0)
    MINI_CHECK(k_invalid1 is None)
    MINI_CHECK(k_invalid2 is None)


@MINI_TEST("Knot", "make_periodic_uniform")
def test_make_periodic_uniform():
    from session_py import knot
    import numpy as np

    # Create periodic uniform knot vector
    k = knot.make_periodic_uniform(3, 4, 1.0)
    k_len = len(k)
    k0, k1, k2, k3, k4 = k[0], k[1], k[2], k[3], k[4]

    MINI_CHECK(k is not None)
    MINI_CHECK(k_len == 5)
    MINI_CHECK(k0 == 0.0 and k1 == 1.0 and k2 == 2.0 and k3 == 3.0 and k4 == 4.0)


@MINI_TEST("Knot", "clamp")
def test_clamp():
    from session_py import knot
    import numpy as np

    # Clamp a periodic knot vector
    k = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    clamp_result = knot.clamp(4, 4, k, 2)
    first_clamped = k[0] == k[1] and k[1] == k[2]
    last_clamped = k[3] == k[4] and k[4] == k[5]

    MINI_CHECK(clamp_result)
    MINI_CHECK(first_clamped)
    MINI_CHECK(last_clamped)


@MINI_TEST("Knot", "is_valid")
def test_is_valid():
    from session_py import knot
    import numpy as np

    # Valid clamped knot vector
    k = knot.make_clamped_uniform(4, 4, 1.0)
    valid = knot.is_valid(4, 4, k)

    # Invalid - decreasing values
    k_invalid = np.array([0.0, 0.0, 1.0, 0.5, 1.0, 1.0])
    invalid = knot.is_valid(4, 4, k_invalid)

    MINI_CHECK(valid)
    MINI_CHECK(not invalid)


@MINI_TEST("Knot", "is_clamped")
def test_is_clamped():
    from session_py import knot

    # Clamped knot vector
    k = knot.make_clamped_uniform(4, 4, 1.0)
    clamped_both = knot.is_clamped(4, 4, k, 2)
    clamped_start = knot.is_clamped(4, 4, k, 0)
    clamped_end = knot.is_clamped(4, 4, k, 1)

    # Periodic knot vector (not clamped)
    k2 = knot.make_periodic_uniform(4, 5, 1.0)
    not_clamped = knot.is_clamped(4, 5, k2, 2)

    MINI_CHECK(clamped_both)
    MINI_CHECK(clamped_start)
    MINI_CHECK(clamped_end)
    MINI_CHECK(not not_clamped)


@MINI_TEST("Knot", "is_periodic")
def test_is_periodic():
    from session_py import knot

    # Periodic knot vector
    k = knot.make_periodic_uniform(3, 4, 1.0)
    periodic = knot.is_periodic(3, 4, k)

    # Clamped knot vector (not periodic)
    k2 = knot.make_clamped_uniform(4, 4, 1.0)
    not_periodic = knot.is_periodic(4, 4, k2)

    MINI_CHECK(periodic)
    MINI_CHECK(not not_periodic)


@MINI_TEST("Knot", "get_domain")
def test_get_domain():
    from session_py import knot

    # Get domain of clamped knot vector
    k = knot.make_clamped_uniform(4, 4, 1.0)
    t0, t1 = knot.get_domain(4, 4, k)

    MINI_CHECK(t0 == 0.0)
    MINI_CHECK(t1 == 1.0)


@MINI_TEST("Knot", "set_domain")
def test_set_domain():
    from session_py import knot

    # Create knot vector and set domain
    k = knot.make_clamped_uniform(4, 4, 1.0)
    set_result = knot.set_domain(4, 4, k, 5.0, 10.0)
    t0, t1 = knot.get_domain(4, 4, k)
    t0_close = abs(t0 - 5.0) < 1e-10
    t1_close = abs(t1 - 10.0) < 1e-10

    MINI_CHECK(set_result)
    MINI_CHECK(t0_close)
    MINI_CHECK(t1_close)


@MINI_TEST("Knot", "reverse")
def test_reverse():
    from session_py import knot

    # Reverse knot vector
    k = knot.make_clamped_uniform(4, 4, 1.0)
    t0_orig, t1_orig = knot.get_domain(4, 4, k)
    reverse_result = knot.reverse(4, 4, k)
    t0, t1 = knot.get_domain(4, 4, k)
    t0_preserved = abs(t0 - t0_orig) < 1e-10
    t1_preserved = abs(t1 - t1_orig) < 1e-10

    MINI_CHECK(reverse_result)
    MINI_CHECK(t0_preserved)
    MINI_CHECK(t1_preserved)


@MINI_TEST("Knot", "multiplicity")
def test_multiplicity():
    from session_py import knot

    # Check multiplicity at clamped ends
    k = knot.make_clamped_uniform(4, 4, 1.0)
    mult_first = knot.multiplicity(4, 4, k, 0)
    mult_last = knot.multiplicity(4, 4, k, 5)

    MINI_CHECK(mult_first == 3)
    MINI_CHECK(mult_last == 3)


@MINI_TEST("Knot", "span_count")
def test_span_count():
    from session_py import knot

    # Single Bezier span
    k = knot.make_clamped_uniform(4, 4, 1.0)
    span1 = knot.span_count(4, 4, k)

    # Multiple spans
    k2 = knot.make_clamped_uniform(3, 5, 1.0)
    span2 = knot.span_count(3, 5, k2)

    MINI_CHECK(span1 == 1)
    MINI_CHECK(span2 == 3)


@MINI_TEST("Knot", "find_span")
def test_find_span():
    from session_py import knot

    # Find span in single-span knot vector
    k = knot.make_clamped_uniform(4, 4, 1.0)
    span_0 = knot.find_span(4, 4, k, 0.0)
    span_mid = knot.find_span(4, 4, k, 0.5)
    span_1 = knot.find_span(4, 4, k, 1.0)

    # Find span in multi-span knot vector
    k2 = knot.make_clamped_uniform(3, 5, 1.0)
    span2_0 = knot.find_span(3, 5, k2, 0.0)
    span2_mid = knot.find_span(3, 5, k2, 1.5)
    span2_end = knot.find_span(3, 5, k2, 2.5)

    MINI_CHECK(span_0 == 0 and span_mid == 0 and span_1 == 0)
    MINI_CHECK(span2_0 == 0 and span2_mid == 1 and span2_end == 2)


@MINI_TEST("Knot", "greville_abcissae")
def test_greville_abcissae():
    from session_py import knot

    # Get Greville abcissae (control point parameter values)
    k = knot.make_clamped_uniform(3, 4, 1.0)
    g = knot.get_greville_abcissae(3, 4, k)
    g_len = len(g)

    MINI_CHECK(g_len == 4)


@MINI_TEST("Knot", "domain_tolerance")
def test_domain_tolerance():
    from session_py import knot

    # Calculate domain tolerance
    tol_same = knot.domain_tolerance(1.0, 1.0)
    tol_diff = knot.domain_tolerance(0.0, 1.0)

    MINI_CHECK(tol_same == 0.0)
    MINI_CHECK(tol_diff > 0.0)


if __name__ == "__main__":
    run_all("python")
