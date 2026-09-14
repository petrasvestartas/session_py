from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("SpatialAABBTree", "Constructor")
def test_spatial_aabbtree_constructor():
    from session_py import AABB
    from session_py import Closest

    boxes = [
        AABB(0.0, 0.0, 0.0, 0.5, 0.5, 0.5),
        AABB(5.0, 0.0, 0.0, 0.5, 0.5, 0.5),
        AABB(10.0, 0.0, 0.0, 0.5, 0.5, 0.5),
    ]
    pairs = Closest.boxes_closest(boxes, 0.0)

    MINI_CHECK(len(pairs) == 0)

    boxes_near = [
        AABB(0.0, 0.0, 0.0, 0.5, 0.5, 0.5),
        AABB(1.0, 0.0, 0.0, 0.5, 0.5, 0.5),
    ]
    pairs_near = Closest.boxes_closest(boxes_near, 0.0)

    MINI_CHECK(len(pairs_near) == 1)
    MINI_CHECK(pairs_near[0][0] == 0)
    MINI_CHECK(pairs_near[0][1] == 1)


@MINI_TEST("SpatialAABBTree", "Build Empty")
def test_spatial_aabbtree_build_empty():
    from session_py import SpatialAABBTree

    tree = SpatialAABBTree()
    tree.build([])

    MINI_CHECK(tree.empty())


@MINI_TEST("SpatialAABBTree", "Build Single")
def test_spatial_aabbtree_build_single():
    from session_py import AABB
    from session_py import SpatialAABBTree

    aabb = AABB(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    tree = SpatialAABBTree()
    tree.build([aabb])

    MINI_CHECK(tree.size() == 1)
    MINI_CHECK(tree.nodes[0].object_id == 0)


@MINI_TEST("SpatialAABBTree", "Build Multiple")
def test_spatial_aabbtree_build_multiple():
    from session_py import AABB
    from session_py import SpatialAABBTree

    aabbs = [
        AABB(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        AABB(5.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        AABB(10.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    ]
    tree = SpatialAABBTree()
    tree.build(aabbs)

    MINI_CHECK(tree.size() == 5)
    MINI_CHECK(tree.nodes[0].object_id == -1)


@MINI_TEST("SpatialAABBTree", "Node Count")
def test_spatial_aabbtree_node_count():
    from session_py import AABB
    from session_py import SpatialAABBTree

    aabbs = []
    for i in range(100):
        aabbs.append(AABB(float(i), 0.0, 0.0, 0.5, 0.5, 0.5))
    tree = SpatialAABBTree()
    tree.build(aabbs)

    MINI_CHECK(tree.size() == 199)


@MINI_TEST("SpatialAABBTree", "Mesh Point Aabb")
def test_spatial_aabbtree_mesh_point_aabb():
    from session_py import Closest
    from session_py import Point
    from session_py import Primitives

    m = Primitives.cube(2.0)
    cp1, fk1, d1 = Closest.mesh_point_aabb(m, Point(0.0, 0.0, 2.0))

    MINI_CHECK(TOLERANCE.is_close(cp1[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(d1, 1.0))

    cp2, fk2, d2 = Closest.mesh_point_aabb(m, Point(1.0, 1.0, 1.0))

    MINI_CHECK(TOLERANCE.is_close(d2, 0.0))


@MINI_TEST("SpatialAABBTree", "Mesh Point Aabb Matches Bvh")
def test_spatial_aabbtree_mesh_point_aabb_matches_bvh():
    from session_py import Closest
    from session_py import Point
    from session_py import Primitives

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
    from session_py import AABB
    from session_py import SpatialAABBTree

    aabbs = [
        AABB(0.0, 0.0, 0.0, 0.5, 0.5, 0.5),
        AABB(5.0, 0.0, 0.0, 0.5, 0.5, 0.5),
        AABB(10.0, 0.0, 0.0, 0.5, 0.5, 0.5),
    ]
    tree = SpatialAABBTree()
    tree.build(aabbs)
    hits = tree.query_aabb(AABB(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))

    MINI_CHECK(len(hits) == 1)
    MINI_CHECK(hits[0] == 0)

    none = tree.query_aabb(AABB(20.0, 0.0, 0.0, 0.5, 0.5, 0.5))

    MINI_CHECK(len(none) == 0)

    all_hits = tree.query_aabb(AABB(5.0, 0.0, 0.0, 10.0, 1.0, 1.0))

    MINI_CHECK(len(all_hits) == 3)


if __name__ == "__main__":
    run_all(language="python")
