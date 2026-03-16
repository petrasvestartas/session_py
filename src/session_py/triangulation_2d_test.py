from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
import math


def _triangle_area_2d(pts, t):
    ax, ay = pts[t[0]][0], pts[t[0]][1]
    bx, by = pts[t[1]][0], pts[t[1]][1]
    cx, cy = pts[t[2]][0], pts[t[2]][1]
    return abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax)) * 0.5

def _total_area(pts, tris):
    return sum(_triangle_area_2d(pts, t) for t in tris)


@MINI_TEST("Triangulation2D", "Triangle")
def test_triangle():
    from session_py import Point
    from session_py.triangulation_2d import triangulate
    boundary = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(0, 1, 0),
        Point(0, 0, 0),
    ]
    tris = triangulate(boundary)
    MINI_CHECK(len(tris) == 1)
    MINI_CHECK(tris[0][0] >= 0)
    MINI_CHECK(tris[0][1] >= 0)
    MINI_CHECK(tris[0][2] >= 0)


@MINI_TEST("Triangulation2D", "Square")
def test_square():
    from session_py import Point
    from session_py.triangulation_2d import triangulate
    boundary = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
        Point(0, 0, 0),
    ]
    tris = triangulate(boundary)
    pts = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    a = _total_area(pts, tris)
    MINI_CHECK(len(tris) == 2)
    MINI_CHECK(abs(a - 1.0) < 1e-10)


@MINI_TEST("Triangulation2D", "Convex Polygon")
def test_convex_polygon():
    from session_py import Point
    from session_py.triangulation_2d import triangulate
    r = 1.0
    hex_pts = [Point(r * math.cos(i * TOLERANCE.PI / 3.0), r * math.sin(i * TOLERANCE.PI / 3.0), 0) for i in range(6)]
    hex_pts.append(hex_pts[0])
    tris = triangulate(hex_pts)
    MINI_CHECK(len(tris) == 4)


@MINI_TEST("Triangulation2D", "Concave Polygon")
def test_concave_polygon():
    from session_py import Point
    from session_py.triangulation_2d import triangulate
    boundary = [
        Point(0, 0, 0), Point(2, 0, 0), Point(2, 2, 0),
        Point(1, 1, 0), Point(0, 2, 0), Point(0, 0, 0),
    ]
    tris = triangulate(boundary)
    pts = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(1, 1, 0),
        Point(0, 2, 0),
    ]
    a = _total_area(pts, tris)
    expected = 3.0
    MINI_CHECK(len(tris) == 3)
    MINI_CHECK(abs(a - expected) < 1e-10)


@MINI_TEST("Triangulation2D", "Polygon With Hole")
def test_polygon_with_hole():
    from session_py import Point
    from session_py.triangulation_2d import triangulate
    boundary = [
        Point(0, 0, 0),
        Point(4, 0, 0),
        Point(4, 4, 0),
        Point(0, 4, 0),
        Point(0, 0, 0),
    ]
    hole = [
        Point(1, 1, 0),
        Point(3, 1, 0),
        Point(3, 3, 0),
        Point(1, 3, 0),
        Point(1, 1, 0),
    ]
    tris = triangulate(boundary, [hole])
    all_pts = [Point(0, 0, 0), Point(4, 0, 0), Point(4, 4, 0), Point(0, 4, 0),
               Point(1, 1, 0), Point(3, 1, 0), Point(3, 3, 0), Point(1, 3, 0)]
    a = _total_area(all_pts, tris)
    expected = 4.0*4.0 - 2.0*2.0
    MINI_CHECK(len(tris) >= 8)
    MINI_CHECK(abs(a - expected) < 1e-10)


@MINI_TEST("Triangulation2D", "Polygon With Multiple Holes")
def test_polygon_with_multiple_holes():
    from session_py import Point
    from session_py.triangulation_2d import triangulate
    boundary = [
        Point(0, 0, 0),
        Point(10, 0, 0),
        Point(10, 8, 0),
        Point(0, 8, 0),
        Point(0, 0, 0),
    ]
    hole0 = [
        Point(1, 1, 0),
        Point(3, 1, 0),
        Point(3, 3, 0),
        Point(1, 3, 0),
        Point(1, 1, 0),
    ]
    hole1 = [
        Point(5, 4, 0),
        Point(9, 4, 0),
        Point(9, 7, 0),
        Point(5, 7, 0),
        Point(5, 4, 0),
    ]
    tris = triangulate(boundary, [hole0, hole1])
    all_pts = [Point(0, 0, 0), Point(10, 0, 0), Point(10, 8, 0), Point(0, 8, 0),
               Point(1, 1, 0), Point(3, 1, 0), Point(3, 3, 0), Point(1, 3, 0),
               Point(5, 4, 0), Point(9, 4, 0), Point(9, 7, 0), Point(5, 7, 0)]
    a = _total_area(all_pts, tris)
    expected = 10.0*8.0 - 2.0*2.0 - 4.0*3.0
    MINI_CHECK(len(tris) >= 10)
    MINI_CHECK(abs(a - expected) < 1e-6)


@MINI_TEST("Triangulation2D", "Winding Correction")
def test_winding_correction():
    from session_py import Point
    from session_py.triangulation_2d import triangulate
    boundary = [
        Point(0, 1, 0),
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    tris = triangulate(boundary)
    pts = [
        Point(0, 1, 0),
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
    ]
    a = _total_area(pts, tris)
    MINI_CHECK(len(tris) == 2)
    MINI_CHECK(abs(a - 1.0) < 1e-10)


if __name__ == "__main__":
    run_all("python")
