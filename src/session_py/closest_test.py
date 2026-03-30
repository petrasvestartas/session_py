from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Closest", "Line Point")
def test_closest_line_point():
    from session_py import Closest
    from session_py import Line
    from session_py import Point

    l = Line(0.0, 0.0, 0.0, 10.0, 0.0, 0.0)

    cp1, t1, d1 = Closest.line_point(l, Point(5.0, 5.0, 0.0))

    MINI_CHECK(TOLERANCE.is_close(cp1[0], 5.0))
    MINI_CHECK(TOLERANCE.is_close(cp1[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(t1, 0.5))
    MINI_CHECK(TOLERANCE.is_close(d1, 5.0))

    cp2, t2, d2 = Closest.line_point(l, Point(-5.0, 0.0, 0.0))
    MINI_CHECK(TOLERANCE.is_close(cp2[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(t2, 0.0))
    MINI_CHECK(TOLERANCE.is_close(d2, 5.0))

    cp3, t3, d3 = Closest.line_point(l, Point(15.0, 0.0, 0.0))
    MINI_CHECK(TOLERANCE.is_close(cp3[0], 10.0))
    MINI_CHECK(TOLERANCE.is_close(t3, 1.0))
    MINI_CHECK(TOLERANCE.is_close(d3, 5.0))


@MINI_TEST("Closest", "Polyline Point")
def test_closest_polyline_point():
    from session_py import Closest
    from session_py import Polyline
    from session_py import Point

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(10.0, 0.0, 0.0),
        Point(10.0, 10.0, 0.0),
    ])

    cp1, t1, d1 = Closest.polyline_point(pl, Point(5.0, 5.0, 0.0))

    MINI_CHECK(TOLERANCE.is_close(d1, 5.0))

    cp2, t2, d2 = Closest.polyline_point(pl, Point(10.0, 5.0, 0.0))
    MINI_CHECK(TOLERANCE.is_close(cp2[0], 10.0))
    MINI_CHECK(TOLERANCE.is_close(cp2[1], 5.0))
    MINI_CHECK(TOLERANCE.is_close(d2, 0.0))


@MINI_TEST("Closest", "Curve Point")
def test_closest_curve_point():
    from session_py import Closest
    from session_py import NurbsCurve
    from session_py import Point

    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 2.0, 0.0),
        Point(3.0, 2.0, 0.0),
        Point(4.0, 0.0, 0.0),
    ]
    crv = NurbsCurve.create(False, 3, pts)

    t, dist = Closest.curve_point(crv, Point(2.0, 3.0, 0.0))

    MINI_CHECK(dist < 1.6)
    cp = crv.point_at(t)
    MINI_CHECK(TOLERANCE.is_close(cp.distance(Point(2.0, 3.0, 0.0)), dist))

    t2, dist2 = Closest.curve_point(crv, Point(0.0, 0.0, 0.0))
    MINI_CHECK(dist2 < 0.01)


@MINI_TEST("Closest", "Surface Point")
def test_closest_surface_point():
    from session_py import Closest
    from session_py import NurbsSurface
    from session_py import Point

    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(1.0, 1.0, 1.0),
        Point(2.0, 1.0, 1.0),
        Point(3.0, 1.0, 0.0),
        Point(0.0, 2.0, 0.0),
        Point(1.0, 2.0, 1.0),
        Point(2.0, 2.0, 1.0),
        Point(3.0, 2.0, 0.0),
        Point(0.0, 3.0, 0.0),
        Point(1.0, 3.0, 0.0),
        Point(2.0, 3.0, 0.0),
        Point(3.0, 3.0, 0.0),
    ]
    srf = NurbsSurface.create(False, False, 3, 3, 4, 4, pts)

    u, v, dist = Closest.surface_point(srf, Point(1.5, 1.5, 2.0))

    MINI_CHECK(dist < 1.5)
    cp = srf.point_at(u, v)
    MINI_CHECK(TOLERANCE.is_close(cp.distance(Point(1.5, 1.5, 2.0)), dist))

    u2, v2, dist2 = Closest.surface_point(srf, Point(0.0, 0.0, 0.0))
    MINI_CHECK(dist2 < 0.01)


@MINI_TEST("Closest", "Mesh Point")
def test_closest_mesh_point():
    from session_py import Closest
    from session_py import Primitives
    from session_py import Point

    m = Primitives.cube(2.0)

    cp1, fk1, d1 = Closest.mesh_point(m, Point(0.0, 0.0, 2.0))

    MINI_CHECK(TOLERANCE.is_close(cp1[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(d1, 1.0))

    cp2, fk2, d2 = Closest.mesh_point(m, Point(1.0, 1.0, 1.0))
    MINI_CHECK(TOLERANCE.is_close(d2, 0.0))


@MINI_TEST("Closest", "Mesh Point AABB")
def test_closest_mesh_point_aabb():
    from session_py import Closest
    from session_py import Primitives
    from session_py import Point

    m = Primitives.cube(2.0)

    cp1, fk1, d1 = Closest.mesh_point_aabb(m, Point(0.0, 0.0, 2.0))

    MINI_CHECK(TOLERANCE.is_close(cp1[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(d1, 1.0))

    cp2, fk2, d2 = Closest.mesh_point_aabb(m, Point(1.0, 1.0, 1.0))
    MINI_CHECK(TOLERANCE.is_close(d2, 0.0))


@MINI_TEST("Closest", "Pointcloud Point")
def test_closest_pointcloud_point():
    from session_py import Closest
    from session_py import PointCloud
    from session_py import Point

    pc = PointCloud([
        Point(0.0, 0.0, 0.0),
        Point(5.0, 0.0, 0.0),
        Point(10.0, 0.0, 0.0),
        Point(10.0, 10.0, 0.0),
    ])

    cp1, i1, d1 = Closest.pointcloud_point(pc, Point(4.0, 0.0, 0.0))

    MINI_CHECK(TOLERANCE.is_close(cp1[0], 5.0))
    MINI_CHECK(i1 == 1)
    MINI_CHECK(TOLERANCE.is_close(d1, 1.0))

    cp2, i2, d2 = Closest.pointcloud_point(pc, Point(10.0, 10.0, 0.0))
    MINI_CHECK(TOLERANCE.is_close(d2, 0.0))
    MINI_CHECK(i2 == 3)


if __name__ == "__main__":
    run_all("python")
