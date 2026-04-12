import math
from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("KDTree", "Nearest")
def test_kdtree_nearest():
    from session_py import KDTree, Point

    # 5 known points on a line: 0, 1, 2, 3, 4
    # Query at 1.1 — nearest should be index 1 (point at x=1), distance 0.1
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
        Point(4.0, 0.0, 0.0),
    ]
    tree = KDTree(pts)
    query = Point(1.1, 0.0, 0.0)
    idx, dist = tree.nearest(query)

    MINI_CHECK(idx == 1)
    MINI_CHECK(TOLERANCE.is_close(dist, 0.1))


@MINI_TEST("KDTree", "Nearest K")
def test_kdtree_nearest_k():
    from session_py import KDTree, Point

    # 5 points on X axis: 0, 1, 2, 3, 4
    # Query at 1.5 — 3 nearest are: x=1 (d=0.5), x=2 (d=0.5), x=3 (d=1.5)
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
        Point(4.0, 0.0, 0.0),
    ]
    tree = KDTree(pts)
    query = Point(1.5, 0.0, 0.0)
    result = tree.nearest_k(query, 3)

    MINI_CHECK(len(result) == 3)
    MINI_CHECK(TOLERANCE.is_close(result[0][1], 0.5))
    MINI_CHECK(TOLERANCE.is_close(result[1][1], 0.5))
    MINI_CHECK(TOLERANCE.is_close(result[2][1], 1.5))


@MINI_TEST("KDTree", "Radius Search")
def test_kdtree_radius_search():
    from session_py import KDTree, Point

    # 4 points: 0, 1, 2, 5 on X axis
    # Query at 0.5, radius 1.1 — finds x=0 (d=0.5) and x=1 (d=0.5)
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(5.0, 0.0, 0.0),
    ]
    tree = KDTree(pts)
    query = Point(0.5, 0.0, 0.0)
    result = tree.radius_search(query, 1.1)

    MINI_CHECK(len(result) == 2)
    MINI_CHECK(TOLERANCE.is_close(result[0][1], 0.5))
    MINI_CHECK(TOLERANCE.is_close(result[1][1], 0.5))


if __name__ == "__main__":
    run_all("python")
