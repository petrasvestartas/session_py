import math
from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import PI


@MINI_TEST("ConvexHull", "Hull 2d")
def test_convex_hull_hull_2d():
    from session_py import ConvexHull
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.5, 0.5, 0.0),
        Point(0.3, 0.3, 0.0),
    ]
    hull = ConvexHull.hull_2d(points)

    MINI_CHECK(len(hull) == 4)


@MINI_TEST("ConvexHull", "Hull 2d Collinear")
def test_convex_hull_hull_2d_collinear():
    from session_py import ConvexHull
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
        Point(1.5, 1.0, 0.0),
    ]
    hull = ConvexHull.hull_2d(points)

    MINI_CHECK(len(hull) >= 3)


@MINI_TEST("ConvexHull", "Hull 2d Circle")
def test_convex_hull_hull_2d_circle():
    from session_py import ConvexHull
    from session_py import Point

    n = 12
    points = []

    for i in range(n):
        angle = 2.0 * PI * i / n
        points.append(Point(math.cos(angle), math.sin(angle), 0.0))

    points.append(Point(0.0, 0.0, 0.0))

    hull = ConvexHull.hull_2d(points)

    MINI_CHECK(len(hull) == n)


@MINI_TEST("ConvexHull", "Hull 3d")
def test_convex_hull_hull_3d():
    from session_py import ConvexHull
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 1.0),
        Point(0.25, 0.25, 0.25),
    ]
    mesh = ConvexHull.hull_3d(points)

    MINI_CHECK(mesh.number_of_vertices() == 4)
    MINI_CHECK(mesh.number_of_faces() == 4)

    degenerate = ConvexHull.hull_3d(
        [Point(0, 0, 0), Point(1, 0, 0), Point(2, 0, 0), Point(3, 0, 0)]
    )

    MINI_CHECK(degenerate.number_of_vertices() == 4)
    MINI_CHECK(degenerate.number_of_faces() == 0)


@MINI_TEST("ConvexHull", "Hull 3d Cube")
def test_convex_hull_hull_3d_cube():
    from session_py import ConvexHull
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 1.0),
        Point(1.0, 0.0, 1.0),
        Point(1.0, 1.0, 1.0),
        Point(0.0, 1.0, 1.0),
        Point(0.5, 0.5, 0.5),
    ]
    mesh = ConvexHull.hull_3d(points)

    MINI_CHECK(mesh.number_of_vertices() == 8)
    MINI_CHECK(mesh.number_of_faces() == 12)


if __name__ == "__main__":
    run_all("python")
