from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("AABB", "Constructor")
def test_aabb_constructor():
    import math
    from session_py import AABB
    from session_py import Point

    a = AABB(0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
    empty = AABB()
    inside = AABB(0.5, 0.0, 0.0, 0.5, 0.5, 0.5)
    outside = AABB(10.0, 0.0, 0.0, 0.5, 0.5, 0.5)

    astr = str(a)
    arepr = repr(a)
    corners = a.get_corners()
    edges = a.get_edges()

    b = AABB(5.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    grown = AABB(0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
    grown.union_with(b)

    c = AABB.merge(
        AABB(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), AABB(4.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    )

    MINI_CHECK(empty == AABB(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    MINI_CHECK(a == AABB(0.0, 0.0, 0.0, 1.0, 2.0, 3.0))
    MINI_CHECK(a != empty)
    MINI_CHECK(astr == "0.000000, 0.000000, 0.000000, 1.000000, 2.000000, 3.000000")
    MINI_CHECK(
        arepr == "AABB(0.000000, 0.000000, 0.000000, 1.000000, 2.000000, 3.000000)"
    )
    MINI_CHECK(TOLERANCE.is_close(a.area(), 88.0))
    MINI_CHECK(a.center() == Point(0.0, 0.0, 0.0))
    MINI_CHECK(TOLERANCE.is_close(a.diagonal(), 2.0 * math.sqrt(14.0)))
    MINI_CHECK(a.is_valid())
    MINI_CHECK(TOLERANCE.is_close(a.volume(), 48.0))
    MINI_CHECK(a.closest_point(Point(0.0, 0.0, 0.0)) == Point(0.0, 0.0, 0.0))
    MINI_CHECK(a.closest_point(Point(10.0, 0.0, 0.0)) == Point(1.0, 0.0, 0.0))
    MINI_CHECK(a.contains(Point(0.0, 0.0, 0.0)))
    MINI_CHECK(not a.contains(Point(10.0, 0.0, 0.0)))
    MINI_CHECK(a.corner(False, False, False) == Point(-1.0, -2.0, -3.0))
    MINI_CHECK(a.corner(True, True, True) == Point(1.0, 2.0, 3.0))
    MINI_CHECK(len(corners) == 8)
    MINI_CHECK(len(edges) == 12)
    MINI_CHECK(a.point_at(1.0, 0.0, 0.0) == Point(1.0, 0.0, 0.0))
    MINI_CHECK(a.point_at(0.0, 0.0, 0.0) == Point(0.0, 0.0, 0.0))
    MINI_CHECK(a.intersects(inside))
    MINI_CHECK(not a.intersects(outside))
    MINI_CHECK(grown.min_point() == Point(-1.0, -2.0, -3.0))
    MINI_CHECK(grown.max_point() == Point(6.0, 2.0, 3.0))
    MINI_CHECK(c.min_point() == Point(-1.0, -1.0, -1.0))
    MINI_CHECK(c.max_point() == Point(5.0, 1.0, 1.0))


@MINI_TEST("AABB", "Empty")
def test_aabb_empty():
    from session_py import AABB
    from session_py import Point

    a = AABB.empty()
    empty_valid = a.is_valid()
    empty_diagonal = a.diagonal()

    a.union_with(AABB.empty())
    merged_valid = a.is_valid()

    a.union_with_point(1.0, 2.0, 3.0)
    point_valid = a.is_valid()
    point_min = a.min_point()
    point_max = a.max_point()

    a.union_with_point(-1.0, 0.0, 5.0)
    grown_min = a.min_point()
    grown_max = a.max_point()

    a.union_with(AABB.empty())
    b = AABB.merge(AABB.empty(), AABB(4.0, 0.0, 0.0, 1.0, 1.0, 1.0))

    MINI_CHECK(not empty_valid)
    MINI_CHECK(TOLERANCE.is_close(empty_diagonal, 0.0))
    MINI_CHECK(not merged_valid)
    MINI_CHECK(point_valid)
    MINI_CHECK(point_min == Point(1.0, 2.0, 3.0))
    MINI_CHECK(point_max == Point(1.0, 2.0, 3.0))
    MINI_CHECK(grown_min == Point(-1.0, 0.0, 3.0))
    MINI_CHECK(grown_max == Point(1.0, 2.0, 5.0))
    MINI_CHECK(a.max_point() == Point(1.0, 2.0, 5.0))
    MINI_CHECK(b.min_point() == Point(3.0, -1.0, -1.0))
    MINI_CHECK(b.max_point() == Point(5.0, 1.0, 1.0))


@MINI_TEST("AABB", "Transform")
def test_aabb_transform():
    from session_py import AABB
    from session_py import Point
    from session_py import Xform

    a = AABB(0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
    moved = a.transformed(Xform.translation(1.0, 2.0, 3.0))
    turned = a.transformed(Xform.rotation_z(90.0, True))
    empty = AABB.empty().transformed(Xform.translation(1.0, 0.0, 0.0))
    a.transform(Xform.scale_xyz(2.0, 2.0, 2.0))

    MINI_CHECK(moved.min_point() == Point(0.0, 0.0, 0.0))
    MINI_CHECK(moved.max_point() == Point(2.0, 4.0, 6.0))
    MINI_CHECK(turned.min_point() == Point(-2.0, -1.0, -3.0))
    MINI_CHECK(turned.max_point() == Point(2.0, 1.0, 3.0))
    MINI_CHECK(a.max_point() == Point(2.0, 4.0, 6.0))
    MINI_CHECK(not empty.is_valid())


@MINI_TEST("AABB", "From Geometry")
def test_aabb_from_geometry():
    from session_py import AABB
    from session_py import Color
    from session_py import Line
    from session_py import NurbsCurve
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import PointCloud
    from session_py import Polyline
    from session_py import Primitives
    from session_py import Vector

    a_pt = AABB.from_point(Point(1.0, 2.0, 3.0), 0.5)

    a_pts = AABB.from_points(
        [
            Point(0.0, 0.0, 0.0),
            Point(3.0, 4.0, 5.0),
        ],
        0.0,
    )

    a_negative = AABB.from_points(
        [
            Point(-5.0, -4.0, -3.0),
            Point(-1.0, -2.0, -1.0),
        ],
        0.0,
    )

    ln = Line(0.0, 0.0, 0.0, 4.0, 0.0, 0.0)
    a_line = AABB.from_line(ln, 1.0)

    pl = Polyline(
        [
            Point(0.0, 0.0, 0.0),
            Point(2.0, 0.0, 0.0),
            Point(2.0, 2.0, 0.0),
        ]
    )
    a_pl = AABB.from_polyline(pl, 0.0)

    cube = Primitives.cube(2.0)
    a_mesh = AABB.from_mesh(cube, 0.0)

    pc = PointCloud(
        [
            Point(0.0, 0.0, 0.0),
            Point(4.0, 2.0, 6.0),
        ],
        [
            Vector(0.0, 0.0, 1.0),
            Vector(0.0, 0.0, 1.0),
        ],
        [
            Color(255, 0, 0, 255),
            Color(0, 255, 0, 255),
        ],
    )
    a_pc = AABB.from_pointcloud(pc, 0.0)

    curve = NurbsCurve.create(
        False,
        2,
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(2.0, 0.0, 0.0),
            Point(3.0, 0.0, 0.0),
        ],
    )
    a_nc = AABB.from_nurbscurve(curve, 0.5, False)

    surf = NurbsSurface.create(
        False,
        False,
        1,
        1,
        2,
        2,
        [
            Point(0.0, 0.0, 0.0),
            Point(2.0, 0.0, 0.0),
            Point(0.0, 2.0, 0.0),
            Point(2.0, 2.0, 2.0),
        ],
    )
    a_ns = AABB.from_nurbssurface(surf, 0.0)

    MINI_CHECK(a_pt.center() == Point(1.0, 2.0, 3.0))
    MINI_CHECK(TOLERANCE.is_close(a_pt.hx, 0.5))
    MINI_CHECK(a_pts.min_point() == Point(0.0, 0.0, 0.0))
    MINI_CHECK(a_pts.max_point() == Point(3.0, 4.0, 5.0))
    MINI_CHECK(a_negative.min_point() == Point(-5.0, -4.0, -3.0))
    MINI_CHECK(a_negative.max_point() == Point(-1.0, -2.0, -1.0))
    MINI_CHECK(a_line.min_point() == Point(-1.0, -1.0, -1.0))
    MINI_CHECK(a_line.max_point() == Point(5.0, 1.0, 1.0))
    MINI_CHECK(a_pl.min_point() == Point(0.0, 0.0, 0.0))
    MINI_CHECK(a_pl.max_point() == Point(2.0, 2.0, 0.0))
    MINI_CHECK(a_mesh.min_point() == Point(-1.0, -1.0, -1.0))
    MINI_CHECK(a_mesh.max_point() == Point(1.0, 1.0, 1.0))
    MINI_CHECK(a_pc.min_point() == Point(0.0, 0.0, 0.0))
    MINI_CHECK(a_pc.max_point() == Point(4.0, 2.0, 6.0))
    MINI_CHECK(a_nc.is_valid())
    MINI_CHECK(a_nc.contains(Point(1.5, 0.0, 0.0)))
    MINI_CHECK(a_ns.is_valid())
    MINI_CHECK(TOLERANCE.is_close(a_ns.volume(), 8.0))


@MINI_TEST("AABB", "From Nurbscurve Tight")
def test_aabb_from_nurbscurve_tight():
    from session_py import AABB
    from session_py import NurbsCurve
    from session_py import Point

    bulge = NurbsCurve.create(
        False,
        2,
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 2.0, 0.0),
            Point(2.0, 1.0, 0.0),
        ],
    )
    arch = NurbsCurve.create(
        False,
        2,
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 2.0, 0.0),
            Point(2.0, 0.0, 0.0),
        ],
    )
    hull = AABB.from_nurbscurve(bulge, 0.0, False)
    tight = AABB.from_nurbscurve(bulge, 0.0, True)
    boundary = AABB.from_nurbscurve(arch, 0.0, True)

    MINI_CHECK(TOLERANCE.is_close(hull.max_point()[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(tight.max_point()[1], 4.0 / 3.0))
    MINI_CHECK(TOLERANCE.is_close(boundary.hy, 0.5))
    MINI_CHECK(TOLERANCE.is_close(boundary.cy, 0.5))


if __name__ == "__main__":
    run_all(language="python")
