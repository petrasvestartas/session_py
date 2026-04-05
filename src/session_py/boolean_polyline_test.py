from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all

import math

PI2 = 6.283185307179586476


@MINI_TEST("Boolean Polyline", "Overlapping Squares")
def test_boolean_polyline_overlapping_squares():
    from session_py import Polyline
    from session_py import Point

    P = Point
    a = Polyline([P(-1,-1,0), P(1,-1,0), P(1,1,0), P(-1,1,0), P(-1,-1,0)])
    b = Polyline([P(0,0,0), P(2,0,0), P(2,2,0), P(0,2,0), P(0,0,0)])
    isect = Polyline.boolean_op(a, b, 0)
    uni   = Polyline.boolean_op(a, b, 1)
    diff  = Polyline.boolean_op(a, b, 2)
    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Circle Vs Rectangle")
def test_boolean_polyline_circle_vs_rectangle():
    from session_py import Polyline
    from session_py import Point

    pts = [Point(5.0 + 1.5 * math.cos(PI2 * i / 64), 1.5 * math.sin(PI2 * i / 64), 0.0) for i in range(64)]
    pts.append(pts[0])
    circle = Polyline(pts)
    rect = Polyline([Point(4,-0.5,0), Point(7,-0.5,0), Point(7,0.5,0), Point(4,0.5,0), Point(4,-0.5,0)])
    isect = Polyline.boolean_op(circle, rect, 0)
    uni   = Polyline.boolean_op(circle, rect, 1)
    diff  = Polyline.boolean_op(circle, rect, 2)
    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Star Vs Circle")
def test_boolean_polyline_star_vs_circle():
    from session_py import Polyline
    from session_py import Point

    star_pts = [Point(10.0 + (2.0 if i % 2 == 0 else 0.8) * math.cos(PI2 * i / 10), (2.0 if i % 2 == 0 else 0.8) * math.sin(PI2 * i / 10), 0.0) for i in range(10)]
    star_pts.append(star_pts[0])
    star = Polyline(star_pts)
    circ_pts = [Point(10.5 + 1.2 * math.cos(PI2 * i / 32), 0.5 + 1.2 * math.sin(PI2 * i / 32), 0.0) for i in range(32)]
    circ_pts.append(circ_pts[0])
    circle = Polyline(circ_pts)
    isect = Polyline.boolean_op(star, circle, 0)
    uni   = Polyline.boolean_op(star, circle, 1)
    diff  = Polyline.boolean_op(star, circle, 2)
    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "L Shape Vs Rectangle")
def test_boolean_polyline_l_shape_vs_rectangle():
    from session_py import Polyline
    from session_py import Point

    P = Point
    l_shape = Polyline([P(15,-1,0), P(18,-1,0), P(18,0,0), P(16,0,0), P(16,2,0), P(15,2,0), P(15,-1,0)])
    rect = Polyline([P(15.5,-0.5,0), P(18.5,-0.5,0), P(18.5,1.5,0), P(15.5,1.5,0), P(15.5,-0.5,0)])
    isect = Polyline.boolean_op(l_shape, rect, 0)
    uni   = Polyline.boolean_op(l_shape, rect, 1)
    diff  = Polyline.boolean_op(l_shape, rect, 2)
    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Two Large Circles")
def test_boolean_polyline_two_large_circles():
    from session_py import Polyline
    from session_py import Point

    pts_a = [Point(22.0 + 2.0 * math.cos(PI2 * i / 256), 2.0 * math.sin(PI2 * i / 256), 0.0) for i in range(256)]
    pts_a.append(pts_a[0])
    pts_b = [Point(23.0 + 2.0 * math.cos(PI2 * i / 256), 0.5 + 2.0 * math.sin(PI2 * i / 256), 0.0) for i in range(256)]
    pts_b.append(pts_b[0])
    ca = Polyline(pts_a)
    cb = Polyline(pts_b)
    isect = Polyline.boolean_op(ca, cb, 0)
    uni   = Polyline.boolean_op(ca, cb, 1)
    diff  = Polyline.boolean_op(ca, cb, 2)
    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Diamond Vs Triangle")
def test_boolean_polyline_diamond_vs_triangle():
    from session_py import Polyline
    from session_py import Point

    P = Point
    diamond = Polyline([P(28,0,0), P(30,-2,0), P(32,0,0), P(30,2,0), P(28,0,0)])
    tri = Polyline([P(29,-2,0), P(33,0,0), P(29,2,0), P(29,-2,0)])
    isect = Polyline.boolean_op(diamond, tri, 0)
    uni   = Polyline.boolean_op(diamond, tri, 1)
    diff  = Polyline.boolean_op(diamond, tri, 2)
    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Star Vs Star")
def test_boolean_polyline_star_vs_star():
    from session_py import Polyline
    from session_py import Point

    pts_a = [Point(36.0 + (2.5 if i % 2 == 0 else 1.0) * math.cos(PI2 * i / 12), (2.5 if i % 2 == 0 else 1.0) * math.sin(PI2 * i / 12), 0.0) for i in range(12)]
    pts_a.append(pts_a[0])
    pts_b = [Point(37.0 + (2.0 if i % 2 == 0 else 0.8) * math.cos(PI2 * i / 10), 0.5 + (2.0 if i % 2 == 0 else 0.8) * math.sin(PI2 * i / 10), 0.0) for i in range(10)]
    pts_b.append(pts_b[0])
    sa = Polyline(pts_a)
    sb = Polyline(pts_b)
    isect = Polyline.boolean_op(sa, sb, 0)
    uni   = Polyline.boolean_op(sa, sb, 1)
    diff  = Polyline.boolean_op(sa, sb, 2)
    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Cross Shape")
def test_boolean_polyline_cross_shape():
    from session_py import Polyline
    from session_py import Point

    P = Point
    narrow = Polyline([P(42,-2,0), P(44,-2,0), P(44,2,0), P(42,2,0), P(42,-2,0)])
    wide = Polyline([P(40,-0.5,0), P(46,-0.5,0), P(46,0.5,0), P(40,0.5,0), P(40,-0.5,0)])
    isect = Polyline.boolean_op(narrow, wide, 0)
    uni   = Polyline.boolean_op(narrow, wide, 1)
    diff  = Polyline.boolean_op(narrow, wide, 2)
    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Concave Arrow Vs Circle")
def test_boolean_polyline_concave_arrow_vs_circle():
    from session_py import Polyline
    from session_py import Point

    P = Point
    arrow = Polyline([P(49,0,0), P(52,2,0), P(51,0.5,0), P(53,0.5,0), P(53,-0.5,0), P(51,-0.5,0), P(52,-2,0), P(49,0,0)])
    pts = [Point(51.5 + 1.5 * math.cos(PI2 * i / 48), 1.5 * math.sin(PI2 * i / 48), 0.0) for i in range(48)]
    pts.append(pts[0])
    circle = Polyline(pts)
    isect = Polyline.boolean_op(arrow, circle, 0)
    uni   = Polyline.boolean_op(arrow, circle, 1)
    diff  = Polyline.boolean_op(arrow, circle, 2)
    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Two Large Circles 1000")
def test_boolean_polyline_two_large_circles_1000():
    from session_py import Polyline
    from session_py import Point

    pts_a = [Point(58.0 + 3.0 * math.cos(PI2 * i / 1000), 3.0 * math.sin(PI2 * i / 1000), 0.0) for i in range(1000)]
    pts_a.append(pts_a[0])
    pts_b = [Point(59.5 + 3.0 * math.cos(PI2 * i / 1000), 3.0 * math.sin(PI2 * i / 1000), 0.0) for i in range(1000)]
    pts_b.append(pts_b[0])
    ca = Polyline(pts_a)
    cb = Polyline(pts_b)
    isect = Polyline.boolean_op(ca, cb, 0)
    uni   = Polyline.boolean_op(ca, cb, 1)
    diff  = Polyline.boolean_op(ca, cb, 2)
    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


if __name__ == "__main__":
    run_all("python")
