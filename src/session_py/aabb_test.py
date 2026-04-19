from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("AABBTree", "Constructor")
def test_aabbtree_constructor():
    from session_py import AABB
    from session_py import Closest

    # AABBTree: O(n log n) build, O(log n) cull — prune candidates before exact test
    box0 = AABB(0.0, 0.0, 0.0, 0.5, 0.5, 0.5)
    box1 = AABB(5.0, 0.0, 0.0, 0.5, 0.5, 0.5)
    box2 = AABB(10.0, 0.0, 0.0, 0.5, 0.5, 0.5)
    pairs = Closest.boxes_closest([box0, box1, box2], 0.0)

    MINI_CHECK(len(pairs) == 0)

    pairs_near = Closest.boxes_closest([box0, AABB(1.0, 0.0, 0.0, 0.5, 0.5, 0.5)], 0.0)

    MINI_CHECK(len(pairs_near) == 1)
    MINI_CHECK(pairs_near[0][0] == 0)
    MINI_CHECK(pairs_near[0][1] == 1)


@MINI_TEST("AABBTree", "Build Empty")
def test_aabbtree_build_empty():
    from session_py import Closest
    from session_py import Mesh
    from session_py import Point

    m = Mesh()
    cp, fk, d = Closest.mesh_point_aabb(m, Point(0.0, 0.0, 0.0))

    MINI_CHECK(d == float('inf'))


@MINI_TEST("AABBTree", "Build Single")
def test_aabbtree_build_single():
    from session_py import Closest
    from session_py import Mesh
    from session_py import Point

    m = Mesh()
    vk0 = m.add_vertex(Point(0.0, 0.0, 0.0))
    vk1 = m.add_vertex(Point(1.0, 0.0, 0.0))
    vk2 = m.add_vertex(Point(0.0, 1.0, 0.0))
    m.add_face([vk0, vk1, vk2])
    cp, fk, d = Closest.mesh_point_aabb(m, Point(0.0, 0.0, 1.0))

    MINI_CHECK(d > 0.0)
    MINI_CHECK(TOLERANCE.is_close(d, 1.0))


@MINI_TEST("AABBTree", "Build Multiple")
def test_aabbtree_build_multiple():
    from session_py import Closest
    from session_py import Mesh
    from session_py import Point

    m = Mesh()
    vk0 = m.add_vertex(Point(0.0, 0.0, 0.0))
    vk1 = m.add_vertex(Point(1.0, 0.0, 0.0))
    vk2 = m.add_vertex(Point(0.0, 1.0, 0.0))
    vk3 = m.add_vertex(Point(5.0, 0.0, 0.0))
    vk4 = m.add_vertex(Point(6.0, 0.0, 0.0))
    vk5 = m.add_vertex(Point(5.0, 1.0, 0.0))
    vk6 = m.add_vertex(Point(10.0, 0.0, 0.0))
    vk7 = m.add_vertex(Point(11.0, 0.0, 0.0))
    vk8 = m.add_vertex(Point(10.0, 1.0, 0.0))
    m.add_face([vk0, vk1, vk2])
    m.add_face([vk3, vk4, vk5])
    m.add_face([vk6, vk7, vk8])
    cp, fk, d = Closest.mesh_point_aabb(m, Point(0.5, 0.0, 0.0))

    MINI_CHECK(d < 0.5)


@MINI_TEST("AABBTree", "Node Count")
def test_aabbtree_node_count():
    from session_py import Closest
    from session_py import Mesh
    from session_py import Point

    m = Mesh()
    vkeys = []
    for i in range(100):
        vkeys.append(m.add_vertex(Point(float(i), 0.0, 0.0)))
        vkeys.append(m.add_vertex(Point(float(i) + 0.5, 0.5, 0.0)))
        vkeys.append(m.add_vertex(Point(float(i), 0.5, 0.0)))
    for i in range(100):
        m.add_face([vkeys[i*3], vkeys[i*3+1], vkeys[i*3+2]])
    cp, fk, d = Closest.mesh_point_aabb(m, Point(50.0, 0.0, 0.0))

    MINI_CHECK(d < 0.5)


@MINI_TEST("AABBTree", "Mesh Point Aabb")
def test_aabbtree_mesh_point_aabb():
    from session_py import Closest
    from session_py import Primitives
    from session_py import Point

    m = Primitives.cube(2.0)
    cp1, fk1, d1 = Closest.mesh_point_aabb(m, Point(0.0, 0.0, 2.0))

    MINI_CHECK(TOLERANCE.is_close(cp1[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(d1, 1.0))
    cp2, fk2, d2 = Closest.mesh_point_aabb(m, Point(1.0, 1.0, 1.0))
    MINI_CHECK(TOLERANCE.is_close(d2, 0.0))


@MINI_TEST("AABBTree", "Mesh Point Aabb Matches Bvh")
def test_aabbtree_mesh_point_aabb_matches_bvh():
    from session_py import Closest
    from session_py import Primitives
    from session_py import Point

    m = Primitives.cube(2.0)
    tp = Point(0.3, 0.7, 1.5)
    cp_bvh, fk_bvh, d_bvh = Closest.mesh_point(m, tp)
    cp_aabb, fk_aabb, d_aabb = Closest.mesh_point_aabb(m, tp)

    MINI_CHECK(TOLERANCE.is_close(d_bvh, d_aabb))
    MINI_CHECK(TOLERANCE.is_close(cp_bvh[0], cp_aabb[0]))
    MINI_CHECK(TOLERANCE.is_close(cp_bvh[1], cp_aabb[1]))
    MINI_CHECK(TOLERANCE.is_close(cp_bvh[2], cp_aabb[2]))


@MINI_TEST("Aabb", "Constructor")
def test_aabb_constructor():
    import math
    from session_py import AABB
    from session_py import Point

    # AABB(0,0,0, 1,2,3) — dims 2×4×6
    a = AABB(0.0, 0.0, 0.0, 1.0, 2.0, 3.0)

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
    MINI_CHECK(len(a.get_corners()) == 8)
    MINI_CHECK(len(a.get_edges()) == 12)
    MINI_CHECK(a.point_at(1.0, 0.0, 0.0) == Point(1.0, 0.0, 0.0))
    MINI_CHECK(a.point_at(0.0, 0.0, 0.0) == Point(0.0, 0.0, 0.0))
    MINI_CHECK(a.intersects(AABB(0.5, 0.0, 0.0, 0.5, 0.5, 0.5)))
    MINI_CHECK(not a.intersects(AABB(10.0, 0.0, 0.0, 0.5, 0.5, 0.5)))
    b = AABB(5.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    a = a.union_with(b)
    MINI_CHECK(a.min_point() == Point(-1.0, -2.0, -3.0))
    MINI_CHECK(a.max_point() == Point(6.0, 2.0, 3.0))
    c = AABB.merge(AABB(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), AABB(4.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    MINI_CHECK(c.min_point() == Point(-1.0, -1.0, -1.0))
    MINI_CHECK(c.max_point() == Point(5.0, 1.0, 1.0))


@MINI_TEST("Aabb", "From Geometry")
def test_aabb_from_geometry():
    from session_py import AABB
    from session_py import Color
    from session_py import Line
    from session_py import Mesh
    from session_py import NurbsCurve
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import PointCloud
    from session_py import Polyline
    from session_py import Primitives
    from session_py import Vector

    a_pt = AABB.from_point(Point(1.0, 2.0, 3.0), 0.5)

    MINI_CHECK(a_pt.center() == Point(1.0, 2.0, 3.0))
    MINI_CHECK(TOLERANCE.is_close(a_pt.hx, 0.5))

    a_pts = AABB.from_points([
        Point(0.0, 0.0, 0.0),
        Point(3.0, 4.0, 5.0),
    ], 0.0)

    MINI_CHECK(a_pts.min_point() == Point(0.0, 0.0, 0.0))
    MINI_CHECK(a_pts.max_point() == Point(3.0, 4.0, 5.0))

    ln = Line(0.0, 0.0, 0.0, 4.0, 0.0, 0.0)
    a_line = AABB.from_line(ln, 1.0)

    MINI_CHECK(a_line.min_point() == Point(-1.0, -1.0, -1.0))
    MINI_CHECK(a_line.max_point() == Point(5.0, 1.0, 1.0))

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(2.0, 2.0, 0.0),
    ])
    a_pl = AABB.from_polyline(pl, 0.0)

    MINI_CHECK(a_pl.min_point() == Point(0.0, 0.0, 0.0))
    MINI_CHECK(a_pl.max_point() == Point(2.0, 2.0, 0.0))

    cube = Primitives.cube(2.0)
    a_mesh = AABB.from_mesh(cube, 0.0)

    MINI_CHECK(a_mesh.min_point() == Point(-1.0, -1.0, -1.0))
    MINI_CHECK(a_mesh.max_point() == Point(1.0, 1.0, 1.0))

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
        ]
    )
    a_pc = AABB.from_pointcloud(pc, 0.0)

    MINI_CHECK(a_pc.min_point() == Point(0.0, 0.0, 0.0))
    MINI_CHECK(a_pc.max_point() == Point(4.0, 2.0, 6.0))

    curve = NurbsCurve.create(False, 2, [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
    ])
    a_nc = AABB.from_nurbscurve(curve, 0.5, False)

    MINI_CHECK(a_nc.is_valid())
    MINI_CHECK(a_nc.contains(Point(1.5, 0.0, 0.0)))

    surf = NurbsSurface.create(False, False, 1, 1, 2, 2, [
        Point(0.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(0.0, 2.0, 0.0),
        Point(2.0, 2.0, 2.0),
    ])
    a_ns = AABB.from_nurbssurface(surf, 0.0)

    MINI_CHECK(a_ns.is_valid())
    MINI_CHECK(TOLERANCE.is_close(a_ns.volume(), 8.0))


@MINI_TEST("AABBTree", "Query Aabb")
def test_aabbtree_query_aabb():
    from session_py import Closest
    from session_py import Mesh
    from session_py import Point

    m = Mesh()
    vk0 = m.add_vertex(Point(0.0, 0.0, 0.0))
    vk1 = m.add_vertex(Point(1.0, 0.0, 0.0))
    vk2 = m.add_vertex(Point(0.5, 1.0, 0.0))
    vk3 = m.add_vertex(Point(20.0, 0.0, 0.0))
    vk4 = m.add_vertex(Point(21.0, 0.0, 0.0))
    vk5 = m.add_vertex(Point(20.5, 1.0, 0.0))
    m.add_face([vk0, vk1, vk2])
    m.add_face([vk3, vk4, vk5])
    cp_near, fk_near, d_near = Closest.mesh_point_aabb(m, Point(0.5, 0.25, 0.0))
    cp_far, fk_far, d_far = Closest.mesh_point_aabb(m, Point(20.5, 0.25, 0.0))

    MINI_CHECK(TOLERANCE.is_close(d_near, 0.0))
    MINI_CHECK(TOLERANCE.is_close(d_far, 0.0))
    MINI_CHECK(fk_near != fk_far)
    MINI_CHECK(cp_near[0] < 2.0)
    MINI_CHECK(cp_far[0] > 15.0)


if __name__ == "__main__":
    run_all(language="python")
