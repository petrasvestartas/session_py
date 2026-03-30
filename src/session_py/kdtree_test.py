import math
from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("KDTree", "Nearest")
def test_kdtree_nearest():
    from session_py import KDTree, Point
    import random
    rng = random.Random(42)
    pts = []
    for _ in range(100):
        pts.append(Point(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0)))
    tree = KDTree(pts)
    query = Point(0.0, 0.0, 0.0)
    idx, dist = tree.nearest(query)
    brute_idx = min(range(len(pts)), key=lambda i: (pts[i][0]**2 + pts[i][1]**2 + pts[i][2]**2))
    brute_dist = math.sqrt(pts[brute_idx][0]**2 + pts[brute_idx][1]**2 + pts[brute_idx][2]**2)

    MINI_CHECK(idx == brute_idx)
    MINI_CHECK(TOLERANCE.is_close(dist, brute_dist))


@MINI_TEST("KDTree", "NearestK")
def test_kdtree_nearest_k():
    from session_py import KDTree, Point
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


@MINI_TEST("KDTree", "RadiusSearch")
def test_kdtree_radius_search():
    from session_py import KDTree, Point
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


@MINI_TEST("KDTree", "SinglePoint")
def test_kdtree_single_point():
    from session_py import KDTree, Point
    pts = [Point(3.0, 4.0, 0.0)]
    tree = KDTree(pts)
    query = Point(0.0, 0.0, 0.0)
    idx, dist = tree.nearest(query)

    MINI_CHECK(idx == 0)
    MINI_CHECK(TOLERANCE.is_close(dist, 5.0))


@MINI_TEST("KDTree", "NearestBruteForce")
def test_kdtree_nearest_brute_force():
    from session_py import KDTree, Point
    import random
    rng = random.Random(7)
    pts = []
    for _ in range(50):
        pts.append(Point(rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0)))
    tree = KDTree(pts)
    queries = [Point(rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0)) for _ in range(10)]
    all_match = True
    for q in queries:
        idx, dist = tree.nearest(q)
        brute = min(range(len(pts)), key=lambda i: (pts[i][0]-q[0])**2+(pts[i][1]-q[1])**2+(pts[i][2]-q[2])**2)
        brute_d = math.sqrt((pts[brute][0]-q[0])**2+(pts[brute][1]-q[1])**2+(pts[brute][2]-q[2])**2)
        if not TOLERANCE.is_close(dist, brute_d):
            all_match = False

    MINI_CHECK(all_match)


if __name__ == "__main__":
    run_all("python")
