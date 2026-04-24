from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("SpatialAABBTree", "Constructor")
def test_spatial_aabbtree_constructor():
    from session_py import AABB
    from session_py import Closest

    # SpatialAABBTree: O(n log n) build, O(log n) cull — prune candidates before exact test
    box0 = AABB(0.0, 0.0, 0.0, 0.5, 0.5, 0.5)
    box1 = AABB(5.0, 0.0, 0.0, 0.5, 0.5, 0.5)
    box2 = AABB(10.0, 0.0, 0.0, 0.5, 0.5, 0.5)
    pairs = Closest.boxes_closest([box0, box1, box2], 0.0)

    MINI_CHECK(len(pairs) == 0)

    pairs_near = Closest.boxes_closest([box0, AABB(1.0, 0.0, 0.0, 0.5, 0.5, 0.5)], 0.0)

    MINI_CHECK(len(pairs_near) == 1)
    MINI_CHECK(pairs_near[0][0] == 0)
    MINI_CHECK(pairs_near[0][1] == 1)


@MINI_TEST("SpatialAABBTree", "Build Empty")
def test_spatial_aabbtree_build_empty():
    from session_py import Closest
    from session_py import Mesh
    from session_py import Point

    m = Mesh()
    cp, fk, d = Closest.mesh_point_aabb(m, Point(0.0, 0.0, 0.0))

    MINI_CHECK(d == float('inf'))


@MINI_TEST("SpatialAABBTree", "Build Single")
def test_spatial_aabbtree_build_single():
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


@MINI_TEST("SpatialAABBTree", "Build Multiple")
def test_spatial_aabbtree_build_multiple():
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


@MINI_TEST("SpatialAABBTree", "Node Count")
def test_spatial_aabbtree_node_count():
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


@MINI_TEST("SpatialAABBTree", "Mesh Point Aabb")
def test_spatial_aabbtree_mesh_point_aabb():
    from session_py import Closest
    from session_py import Primitives
    from session_py import Point

    m = Primitives.cube(2.0)
    cp1, fk1, d1 = Closest.mesh_point_aabb(m, Point(0.0, 0.0, 2.0))

    MINI_CHECK(TOLERANCE.is_close(cp1[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(d1, 1.0))
    cp2, fk2, d2 = Closest.mesh_point_aabb(m, Point(1.0, 1.0, 1.0))
    MINI_CHECK(TOLERANCE.is_close(d2, 0.0))


@MINI_TEST("SpatialAABBTree", "Mesh Point Aabb Matches Bvh")
def test_spatial_aabbtree_mesh_point_aabb_matches_bvh():
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


@MINI_TEST("SpatialAABBTree", "Query Aabb")
def test_spatial_aabbtree_query_aabb():
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
