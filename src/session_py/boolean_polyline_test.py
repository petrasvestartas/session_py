from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("Polyline", "Boolean Op")
def test_polyline_boolean_op():
    from session_py import Polyline
    from session_py import Point

    P = Point
    sq_a = Polyline([
        P(-1.0, -1.0, 0.0),
        P( 1.0, -1.0, 0.0),
        P( 1.0,  1.0, 0.0),
        P(-1.0,  1.0, 0.0),
    ])
    sq_b = Polyline([
        P(0.0, 0.0, 0.0),
        P(2.0, 0.0, 0.0),
        P(2.0, 2.0, 0.0),
        P(0.0, 2.0, 0.0),
    ])
    sq_inside = Polyline([
        P(-0.5, -0.5, 0.0),
        P( 0.5, -0.5, 0.0),
        P( 0.5,  0.5, 0.0),
        P(-0.5,  0.5, 0.0),
    ])
    sq_disjoint = Polyline([
        P(5.0, 5.0, 0.0),
        P(6.0, 5.0, 0.0),
        P(6.0, 6.0, 0.0),
        P(5.0, 6.0, 0.0),
    ])

    isect = Polyline.boolean_op(sq_a, sq_b, 0)
    uni   = Polyline.boolean_op(sq_a, sq_b, 1)
    diff  = Polyline.boolean_op(sq_a, sq_b, 2)

    MINI_CHECK(len(isect) == 1)
    MINI_CHECK(isect[0].point_count() == 4)
    MINI_CHECK(len(uni) == 1)
    MINI_CHECK(uni[0].point_count() == 8)
    MINI_CHECK(len(diff) == 1)
    MINI_CHECK(diff[0].point_count() == 6)

    isect_in = Polyline.boolean_op(sq_a, sq_inside, 0)
    uni_in   = Polyline.boolean_op(sq_a, sq_inside, 1)
    diff_in  = Polyline.boolean_op(sq_a, sq_inside, 2)

    MINI_CHECK(len(isect_in) == 1)
    MINI_CHECK(isect_in[0].point_count() == 4)
    MINI_CHECK(len(uni_in) == 1)
    MINI_CHECK(uni_in[0].point_count() == 4)
    MINI_CHECK(len(diff_in) == 1)
    MINI_CHECK(diff_in[0].point_count() == 4)

    isect_dis = Polyline.boolean_op(sq_a, sq_disjoint, 0)
    uni_dis   = Polyline.boolean_op(sq_a, sq_disjoint, 1)
    diff_dis  = Polyline.boolean_op(sq_a, sq_disjoint, 2)

    MINI_CHECK(len(isect_dis) == 0)
    MINI_CHECK(len(uni_dis) == 2)
    MINI_CHECK(len(diff_dis) == 1)


if __name__ == "__main__":
    run_all("python")
