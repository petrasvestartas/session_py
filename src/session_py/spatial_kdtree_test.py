from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("SpatialKDTree", "Constructor")
def test_kdtree_constructor():
    from session_py import Point
    from session_py import SpatialKDTree

    pts = [
        Point(0.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
        Point(10.0, 0.0, 0.0),
    ]

    tree = SpatialKDTree(pts)
    hit = tree.nearest(Point(2.0, 0.0, 0.0))

    MINI_CHECK(hit[0] == 1)
    MINI_CHECK(TOLERANCE.is_close(hit[1], 1.0))


@MINI_TEST("SpatialKDTree", "Nearest")
def test_kdtree_nearest():
    from session_py import Point
    from session_py import SpatialKDTree

    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
        Point(4.0, 0.0, 0.0),
    ]

    tree = SpatialKDTree(pts)
    query = Point(1.1, 0.0, 0.0)
    hit = tree.nearest(query)

    MINI_CHECK(hit[0] == 1)
    MINI_CHECK(TOLERANCE.is_close(hit[1], 0.1))


@MINI_TEST("SpatialKDTree", "Nearest K")
def test_kdtree_nearest_k():
    from session_py import Point
    from session_py import SpatialKDTree

    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
        Point(4.0, 0.0, 0.0),
    ]

    tree = SpatialKDTree(pts)
    query = Point(1.5, 0.0, 0.0)
    result = tree.nearest_k(query, 3)

    MINI_CHECK(len(result) == 3)
    MINI_CHECK(TOLERANCE.is_close(result[0][1], 0.5))
    MINI_CHECK(TOLERANCE.is_close(result[1][1], 0.5))
    MINI_CHECK(TOLERANCE.is_close(result[2][1], 1.5))


@MINI_TEST("SpatialKDTree", "Radius Search")
def test_kdtree_radius_search():
    from session_py import Point
    from session_py import SpatialKDTree

    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(5.0, 0.0, 0.0),
    ]

    tree = SpatialKDTree(pts)
    query = Point(0.5, 0.0, 0.0)
    result = tree.radius_search(query, 1.1)

    MINI_CHECK(len(result) == 2)
    MINI_CHECK(TOLERANCE.is_close(result[0][1], 0.5))
    MINI_CHECK(TOLERANCE.is_close(result[1][1], 0.5))


if __name__ == "__main__":
    run_all("python")
