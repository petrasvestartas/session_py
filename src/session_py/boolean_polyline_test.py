from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
import math

PI2 = 6.283185307179586476


@MINI_TEST("Boolean Polyline", "Overlapping Squares")
def test_boolean_polyline_overlapping_squares():
    from session_py import BooleanPolyline
    from session_py import Point
    from session_py import Polyline

    a = Polyline(
        [
            Point(-1, -1, 0),
            Point(1, -1, 0),
            Point(1, 1, 0),
            Point(-1, 1, 0),
            Point(-1, -1, 0),
        ]
    )
    b = Polyline(
        [
            Point(0, 0, 0),
            Point(2, 0, 0),
            Point(2, 2, 0),
            Point(0, 2, 0),
            Point(0, 0, 0),
        ]
    )
    isect = Polyline.boolean_op(a, b, 0)
    uni = Polyline.boolean_op(a, b, 1)
    diff = Polyline.boolean_op(a, b, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)

    far_a = Polyline(
        [
            Point(10, -1, 0),
            Point(12, -1, 0),
            Point(12, 1, 0),
            Point(10, 1, 0),
            Point(10, -1, 0),
        ]
    )
    far_b = Polyline(
        [
            Point(14, -1, 0),
            Point(16, -1, 0),
            Point(16, 1, 0),
            Point(14, 1, 0),
            Point(14, -1, 0),
        ]
    )

    MINI_CHECK(BooleanPolyline.compute_count(far_a, far_b, 0) == 0)
    MINI_CHECK(
        BooleanPolyline.compute_count(far_a, far_b, 1)
        == far_a.point_count() + far_b.point_count()
    )
    MINI_CHECK(BooleanPolyline.compute_count(far_a, far_b, 2) == far_a.point_count())


@MINI_TEST("Boolean Polyline", "Circle Vs Rectangle")
def test_boolean_polyline_circle_vs_rectangle():
    from session_py import Point
    from session_py import Polyline

    pts = []

    for i in range(64):
        a = PI2 * i / 64
        pts.append(Point(5.0 + 1.5 * math.cos(a), 1.5 * math.sin(a), 0.0))

    pts.append(pts[0])
    circle = Polyline(pts)
    rect = Polyline(
        [
            Point(4, -0.5, 0),
            Point(7, -0.5, 0),
            Point(7, 0.5, 0),
            Point(4, 0.5, 0),
            Point(4, -0.5, 0),
        ]
    )
    isect = Polyline.boolean_op(circle, rect, 0)
    uni = Polyline.boolean_op(circle, rect, 1)
    diff = Polyline.boolean_op(circle, rect, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Star Vs Circle")
def test_boolean_polyline_star_vs_circle():
    from session_py import Point
    from session_py import Polyline

    star_pts = []

    for i in range(10):
        a = PI2 * i / 10
        r = 2.0 if i % 2 == 0 else 0.8
        star_pts.append(Point(10.0 + r * math.cos(a), r * math.sin(a), 0.0))

    star_pts.append(star_pts[0])
    star = Polyline(star_pts)
    circ_pts = []

    for i in range(32):
        a = PI2 * i / 32
        circ_pts.append(Point(10.5 + 1.2 * math.cos(a), 0.5 + 1.2 * math.sin(a), 0.0))

    circ_pts.append(circ_pts[0])
    circle = Polyline(circ_pts)
    isect = Polyline.boolean_op(star, circle, 0)
    uni = Polyline.boolean_op(star, circle, 1)
    diff = Polyline.boolean_op(star, circle, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "L Shape Vs Rectangle")
def test_boolean_polyline_l_shape_vs_rectangle():
    from session_py import Point
    from session_py import Polyline

    l_shape = Polyline(
        [
            Point(15, -1, 0),
            Point(18, -1, 0),
            Point(18, 0, 0),
            Point(16, 0, 0),
            Point(16, 2, 0),
            Point(15, 2, 0),
            Point(15, -1, 0),
        ]
    )
    rect = Polyline(
        [
            Point(15.5, -0.5, 0),
            Point(18.5, -0.5, 0),
            Point(18.5, 1.5, 0),
            Point(15.5, 1.5, 0),
            Point(15.5, -0.5, 0),
        ]
    )
    isect = Polyline.boolean_op(l_shape, rect, 0)
    uni = Polyline.boolean_op(l_shape, rect, 1)
    diff = Polyline.boolean_op(l_shape, rect, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Two Large Circles")
def test_boolean_polyline_two_large_circles():
    from session_py import Point
    from session_py import Polyline

    pts_a = []
    pts_b = []

    for i in range(256):
        a = PI2 * i / 256
        pts_a.append(Point(22.0 + 2.0 * math.cos(a), 2.0 * math.sin(a), 0.0))
        pts_b.append(Point(23.0 + 2.0 * math.cos(a), 0.5 + 2.0 * math.sin(a), 0.0))

    pts_a.append(pts_a[0])
    pts_b.append(pts_b[0])
    ca = Polyline(pts_a)
    cb = Polyline(pts_b)
    isect = Polyline.boolean_op(ca, cb, 0)
    uni = Polyline.boolean_op(ca, cb, 1)
    diff = Polyline.boolean_op(ca, cb, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Diamond Vs Triangle")
def test_boolean_polyline_diamond_vs_triangle():
    from session_py import Point
    from session_py import Polyline

    diamond = Polyline(
        [
            Point(28, 0, 0),
            Point(30, -2, 0),
            Point(32, 0, 0),
            Point(30, 2, 0),
            Point(28, 0, 0),
        ]
    )
    tri = Polyline(
        [
            Point(29, -2, 0),
            Point(33, 0, 0),
            Point(29, 2, 0),
            Point(29, -2, 0),
        ]
    )
    isect = Polyline.boolean_op(diamond, tri, 0)
    uni = Polyline.boolean_op(diamond, tri, 1)
    diff = Polyline.boolean_op(diamond, tri, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Star Vs Star")
def test_boolean_polyline_star_vs_star():
    from session_py import Point
    from session_py import Polyline

    pts_a = []

    for i in range(12):
        a = PI2 * i / 12
        r = 2.5 if i % 2 == 0 else 1.0
        pts_a.append(Point(36.0 + r * math.cos(a), r * math.sin(a), 0.0))

    pts_a.append(pts_a[0])
    pts_b = []

    for i in range(10):
        a = PI2 * i / 10
        r = 2.0 if i % 2 == 0 else 0.8
        pts_b.append(Point(37.0 + r * math.cos(a), 0.5 + r * math.sin(a), 0.0))

    pts_b.append(pts_b[0])
    sa = Polyline(pts_a)
    sb = Polyline(pts_b)
    isect = Polyline.boolean_op(sa, sb, 0)
    uni = Polyline.boolean_op(sa, sb, 1)
    diff = Polyline.boolean_op(sa, sb, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Cross Shape")
def test_boolean_polyline_cross_shape():
    from session_py import Point
    from session_py import Polyline

    narrow = Polyline(
        [
            Point(42, -2, 0),
            Point(44, -2, 0),
            Point(44, 2, 0),
            Point(42, 2, 0),
            Point(42, -2, 0),
        ]
    )
    wide = Polyline(
        [
            Point(40, -0.5, 0),
            Point(46, -0.5, 0),
            Point(46, 0.5, 0),
            Point(40, 0.5, 0),
            Point(40, -0.5, 0),
        ]
    )
    isect = Polyline.boolean_op(narrow, wide, 0)
    uni = Polyline.boolean_op(narrow, wide, 1)
    diff = Polyline.boolean_op(narrow, wide, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Concave Arrow Vs Circle")
def test_boolean_polyline_concave_arrow_vs_circle():
    from session_py import Point
    from session_py import Polyline

    arrow = Polyline(
        [
            Point(49, 0, 0),
            Point(52, 2, 0),
            Point(51, 0.5, 0),
            Point(53, 0.5, 0),
            Point(53, -0.5, 0),
            Point(51, -0.5, 0),
            Point(52, -2, 0),
            Point(49, 0, 0),
        ]
    )
    pts = []

    for i in range(48):
        a = PI2 * i / 48
        pts.append(Point(51.5 + 1.5 * math.cos(a), 1.5 * math.sin(a), 0.0))

    pts.append(pts[0])
    circle = Polyline(pts)
    isect = Polyline.boolean_op(arrow, circle, 0)
    uni = Polyline.boolean_op(arrow, circle, 1)
    diff = Polyline.boolean_op(arrow, circle, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Two Large Circles 1000")
def test_boolean_polyline_two_large_circles_1000():
    from session_py import Point
    from session_py import Polyline

    pts_a = []
    pts_b = []

    for i in range(1000):
        a = PI2 * i / 1000
        pts_a.append(Point(58.0 + 3.0 * math.cos(a), 3.0 * math.sin(a), 0.0))
        pts_b.append(Point(59.5 + 3.0 * math.cos(a), 3.0 * math.sin(a), 0.0))

    pts_a.append(pts_a[0])
    pts_b.append(pts_b[0])
    ca = Polyline(pts_a)
    cb = Polyline(pts_b)
    isect = Polyline.boolean_op(ca, cb, 0)
    uni = Polyline.boolean_op(ca, cb, 1)
    diff = Polyline.boolean_op(ca, cb, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Large Coords Auto Scale")
def test_boolean_polyline_large_coords_auto_scale():
    from session_py import Point
    from session_py import Polyline

    a = Polyline(
        [
            Point(64e6, 1e6, 0),
            Point(64e6 + 2e6, 1e6, 0),
            Point(64e6 + 2e6, 1e6 + 2e6, 0),
            Point(64e6, 1e6 + 2e6, 0),
            Point(64e6, 1e6, 0),
        ]
    )
    b = Polyline(
        [
            Point(64e6 + 1e6, 1e6 + 1e6, 0),
            Point(64e6 + 3e6, 1e6 + 1e6, 0),
            Point(64e6 + 3e6, 1e6 + 3e6, 0),
            Point(64e6 + 1e6, 1e6 + 3e6, 0),
            Point(64e6 + 1e6, 1e6 + 1e6, 0),
        ]
    )
    isect = Polyline.boolean_op(a, b, 0)
    uni = Polyline.boolean_op(a, b, 1)
    diff = Polyline.boolean_op(a, b, 2)

    MINI_CHECK(len(isect) >= 1)
    MINI_CHECK(isect[0].point_count() > 0)
    MINI_CHECK(len(uni) >= 1)
    MINI_CHECK(uni[0].point_count() > 0)
    MINI_CHECK(len(diff) >= 1)
    MINI_CHECK(diff[0].point_count() > 0)


@MINI_TEST("Boolean Polyline", "Regions")
def test_boolean_polyline_regions():
    from session_py import BooleanPolyline
    from session_py import Plane
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    plane = Plane.xy_plane()
    outer = Polyline.rectangle(
        Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 10.0, 10.0
    )
    inner = Polyline.rectangle(
        Point(3.0, 3.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 4.0, 4.0
    )
    left = Polyline.rectangle(
        Point(0.0, -1.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 5.0, 12.0
    )
    apart = Polyline.rectangle(
        Point(20.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 10.0, 10.0
    )
    frame = BooleanPolyline.compute_regions([outer], [inner], 2)
    half = BooleanPolyline.compute_regions(frame, [left], 0)
    both = BooleanPolyline.compute_regions([outer], [apart], 1)
    clockwise = 0

    for ring in frame:
        clockwise += 1 if ring.is_clockwise(plane) else 0

    MINI_CHECK(len(frame) == 2)
    MINI_CHECK(frame[0].is_closed())
    MINI_CHECK(clockwise == 1)
    MINI_CHECK(len(half) == 1)
    MINI_CHECK(half[0].point_count() == 9)
    MINI_CHECK(not half[0].is_clockwise(plane))
    MINI_CHECK(len(both) == 2)


@MINI_TEST("Boolean Polyline", "Regions Orientation")
def test_boolean_polyline_regions_orientation():
    from session_py import BooleanPolyline
    from session_py import Plane
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    plane = Plane.xy_plane()
    outer = Polyline.rectangle(
        Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 10.0, 10.0
    )
    inner = Polyline.rectangle(
        Point(3.0, 3.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 4.0, 4.0
    )
    frame = BooleanPolyline.compute_regions([outer, inner], [], 1)
    turned = BooleanPolyline.compute_regions([outer.reversed(), inner], [], 1)

    MINI_CHECK(len(frame) == 2)
    MINI_CHECK(len(turned) == 2)

    for ring in frame:
        MINI_CHECK(
            ring.is_clockwise(plane)
            == (ring.get_point(0)[0] > 1.0 and ring.get_point(0)[0] < 9.0)
        )

    for ring in turned:
        MINI_CHECK(
            ring.is_clockwise(plane)
            == (ring.get_point(0)[0] > 1.0 and ring.get_point(0)[0] < 9.0)
        )


@MINI_TEST("Boolean Polyline", "Adjacent Rectangles")
def test_boolean_polyline_adjacent_rectangles():
    from session_py import BooleanPolyline
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    a = Polyline.rectangle(
        Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 1.0, 1.0
    )
    b = Polyline.rectangle(
        Point(1.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 1.0, 1.0
    )
    isect = BooleanPolyline.compute(a, b, 0)
    uni = BooleanPolyline.compute(a, b, 1)
    diff = BooleanPolyline.compute(a, b, 2)

    MINI_CHECK(len(isect) == 0)
    MINI_CHECK(len(uni) == 1)
    MINI_CHECK(uni[0].point_count() == 4)
    MINI_CHECK(uni[0].center() == Point(1.0, 0.5, 0.0))
    MINI_CHECK(len(diff) == 1)
    MINI_CHECK(diff[0].point_count() == 4)
    MINI_CHECK(diff[0].center() == Point(0.5, 0.5, 0.0))


@MINI_TEST("Boolean Polyline", "Partial Shared Edge")
def test_boolean_polyline_partial_shared_edge():
    from session_py import BooleanPolyline
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    a = Polyline.rectangle(
        Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 2.0, 2.0
    )
    b = Polyline.rectangle(
        Point(2.0, 1.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 1.0, 2.0
    )
    isect = BooleanPolyline.compute(a, b, 0)
    uni = BooleanPolyline.compute(a, b, 1)
    diff = BooleanPolyline.compute(a, b, 2)

    MINI_CHECK(len(isect) == 0)
    MINI_CHECK(len(uni) == 1)
    MINI_CHECK(uni[0].point_count() == 8)
    MINI_CHECK(uni[0].center() == Point(1.75, 1.5, 0.0))
    MINI_CHECK(len(diff) == 1)
    MINI_CHECK(diff[0].point_count() == 4)
    MINI_CHECK(diff[0].center() == Point(1.0, 1.0, 0.0))


@MINI_TEST("Boolean Polyline", "Collinear Overlap")
def test_boolean_polyline_collinear_overlap():
    from session_py import BooleanPolyline
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    a = Polyline.rectangle(
        Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 2.0, 1.0
    )
    b = Polyline.rectangle(
        Point(1.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 2.0, 1.0
    )
    corner = Polyline.rectangle(
        Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 1.0, 1.0
    )
    isect = BooleanPolyline.compute(a, b, 0)
    uni = BooleanPolyline.compute(a, b, 1)
    diff = BooleanPolyline.compute(a, b, 2)
    notch = BooleanPolyline.compute(a, corner, 2)

    MINI_CHECK(len(isect) == 1)
    MINI_CHECK(isect[0].point_count() == 4)
    MINI_CHECK(isect[0].center() == Point(1.5, 0.5, 0.0))
    MINI_CHECK(len(uni) == 1)
    MINI_CHECK(uni[0].point_count() == 4)
    MINI_CHECK(uni[0].center() == Point(1.5, 0.5, 0.0))
    MINI_CHECK(len(diff) == 1)
    MINI_CHECK(diff[0].point_count() == 4)
    MINI_CHECK(diff[0].center() == Point(0.5, 0.5, 0.0))
    MINI_CHECK(len(notch) == 1)
    MINI_CHECK(notch[0].point_count() == 4)
    MINI_CHECK(notch[0].center() == Point(1.5, 0.5, 0.0))


@MINI_TEST("Boolean Polyline", "T Junction")
def test_boolean_polyline_t_junction():
    from session_py import BooleanPolyline
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    a = Polyline.rectangle(
        Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 4.0, 1.0
    )
    b = Polyline.rectangle(
        Point(1.0, 1.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), 1.0, 2.0
    )
    isect = BooleanPolyline.compute(a, b, 0)
    uni = BooleanPolyline.compute(a, b, 1)
    diff = BooleanPolyline.compute(a, b, 2)

    MINI_CHECK(len(isect) == 0)
    MINI_CHECK(len(uni) == 1)
    MINI_CHECK(uni[0].point_count() == 8)
    MINI_CHECK(uni[0].center() == Point(1.75, 1.25, 0.0))
    MINI_CHECK(len(diff) == 1)
    MINI_CHECK(diff[0].point_count() == 4)
    MINI_CHECK(diff[0].center() == Point(2.0, 0.5, 0.0))


@MINI_TEST("Boolean Polyline Open", "Horizontal Line Vs Unit Square")
def test_boolean_polyline_open_horizontal_line_vs_unit_square():
    from session_py import BooleanPolyline
    from session_py import Point
    from session_py import Polyline

    open_line = Polyline(
        [
            Point(-2, 0, 0),
            Point(2, 0, 0),
        ]
    )
    sq = Polyline(
        [
            Point(-1, -1, 0),
            Point(1, -1, 0),
            Point(1, 1, 0),
            Point(-1, 1, 0),
            Point(-1, -1, 0),
        ]
    )
    out = BooleanPolyline.clip_open_against_closed(open_line, sq)

    MINI_CHECK(len(out) == 1)
    MINI_CHECK(out[0].point_count() == 2)

    p0 = out[0].get_point(0)
    p1 = out[0].get_point(1)

    MINI_CHECK(abs(abs(p0[0]) - 1.0) < 1e-6)
    MINI_CHECK(abs(abs(p1[0]) - 1.0) < 1e-6)
    MINI_CHECK(abs(p0[1]) < 1e-6)
    MINI_CHECK(abs(p1[1]) < 1e-6)


@MINI_TEST("Boolean Polyline Open", "Diagonal Line Vs Unit Square")
def test_boolean_polyline_open_diagonal_line_vs_unit_square():
    from session_py import BooleanPolyline
    from session_py import Point
    from session_py import Polyline

    open_line = Polyline(
        [
            Point(-2, -2, 0),
            Point(2, 2, 0),
        ]
    )
    sq = Polyline(
        [
            Point(-1, -1, 0),
            Point(1, -1, 0),
            Point(1, 1, 0),
            Point(-1, 1, 0),
            Point(-1, -1, 0),
        ]
    )
    out = BooleanPolyline.clip_open_against_closed(open_line, sq)

    MINI_CHECK(len(out) == 1)
    MINI_CHECK(out[0].point_count() == 2)

    p0 = out[0].get_point(0)
    p1 = out[0].get_point(1)

    MINI_CHECK(abs(abs(p0[0]) - 1.0) < 1e-6)
    MINI_CHECK(abs(abs(p1[0]) - 1.0) < 1e-6)


@MINI_TEST("Boolean Polyline Open", "Interior Open Path Passes Through")
def test_boolean_polyline_open_interior_open_path_passes_through():
    from session_py import BooleanPolyline
    from session_py import Point
    from session_py import Polyline

    open_path = Polyline(
        [
            Point(-2, 0, 0),
            Point(0, 0.2, 0),
            Point(2, 0, 0),
        ]
    )
    sq = Polyline(
        [
            Point(-1, -1, 0),
            Point(1, -1, 0),
            Point(1, 1, 0),
            Point(-1, 1, 0),
            Point(-1, -1, 0),
        ]
    )
    out = BooleanPolyline.clip_open_against_closed(open_path, sq)

    MINI_CHECK(len(out) == 1)
    MINI_CHECK(out[0].point_count() >= 3)


if __name__ == "__main__":
    run_all("python")
