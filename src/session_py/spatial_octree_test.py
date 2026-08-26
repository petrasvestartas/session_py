from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("SpatialOctree", "Constructor")
def test_octree_constructor():
    from session_py import SpatialOctree
    from session_py import Point

    # SpatialOctree: per-node spacing-limited subsamples for LOD point rendering
    # leaf_capacity 16 > 9 points: the root absorbs everything, one node
    pts = [Point(float(x), 0.0, 0.0) for x in range(9)]
    tree = SpatialOctree(pts, 4.0, 16)

    MINI_CHECK(tree.node_count() == 1)
    MINI_CHECK(tree.node_range(0) == (0, 9))
    MINI_CHECK(tree.order() == [0, 1, 2, 3, 4, 5, 6, 7, 8])


@MINI_TEST("SpatialOctree", "Node Count")
def test_octree_node_count():
    from session_py import SpatialOctree
    from session_py import Point

    # 9 points on X: root cube size 8, spacing 4 -> 2 cells, first-wins accepts x=0
    # and x=4; the 7 leftovers split into two octants -> two leaf children
    pts = [Point(float(x), 0.0, 0.0) for x in range(9)]
    tree = SpatialOctree(pts, 4.0, 4)

    MINI_CHECK(tree.node_count() == 3)


@MINI_TEST("SpatialOctree", "Node Cube")
def test_octree_node_cube():
    from session_py import SpatialOctree
    from session_py import Point

    # Root cube: aabb (0..8, 0, 0) grown to a cube -> center (4,0,0), size 8.
    # Child in octant 6 (x<cx, y>=cy, z>=cz): min (0,0,0), size 4 -> center (2,2,2)
    pts = [Point(float(x), 0.0, 0.0) for x in range(9)]
    tree = SpatialOctree(pts, 4.0, 4)
    center, size = tree.node_cube(0)
    child_center, child_size = tree.node_cube(1)

    MINI_CHECK(TOLERANCE.is_close(center[0], 4.0) and TOLERANCE.is_close(center[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(size, 8.0))
    MINI_CHECK(TOLERANCE.is_close(child_center[0], 2.0) and TOLERANCE.is_close(child_center[2], 2.0))
    MINI_CHECK(TOLERANCE.is_close(child_size, 4.0))


@MINI_TEST("SpatialOctree", "Node Level")
def test_octree_node_level():
    from session_py import SpatialOctree
    from session_py import Point

    pts = [Point(float(x), 0.0, 0.0) for x in range(9)]
    tree = SpatialOctree(pts, 4.0, 4)

    MINI_CHECK(tree.node_level(0) == 0)
    MINI_CHECK(tree.node_level(1) == 1)
    MINI_CHECK(tree.node_level(2) == 1)


@MINI_TEST("SpatialOctree", "Node Spacing")
def test_octree_node_spacing():
    from session_py import SpatialOctree
    from session_py import Point

    # Spacing halves per level, like Potree
    pts = [Point(float(x), 0.0, 0.0) for x in range(9)]
    tree = SpatialOctree(pts, 4.0, 4)

    MINI_CHECK(TOLERANCE.is_close(tree.node_spacing(0), 4.0))
    MINI_CHECK(TOLERANCE.is_close(tree.node_spacing(1), 2.0))
    MINI_CHECK(TOLERANCE.is_close(tree.node_spacing(2), 2.0))


@MINI_TEST("SpatialOctree", "Node Range")
def test_octree_node_range():
    from session_py import SpatialOctree
    from session_py import Point

    # Every node's points are contiguous in order(): root [0..2), children after
    pts = [Point(float(x), 0.0, 0.0) for x in range(9)]
    tree = SpatialOctree(pts, 4.0, 4)

    MINI_CHECK(tree.node_range(0) == (0, 2))
    MINI_CHECK(tree.node_range(1) == (2, 3))
    MINI_CHECK(tree.node_range(2) == (5, 4))


@MINI_TEST("SpatialOctree", "Children")
def test_octree_children():
    from session_py import SpatialOctree
    from session_py import Point

    pts = [Point(float(x), 0.0, 0.0) for x in range(9)]
    tree = SpatialOctree(pts, 4.0, 4)

    MINI_CHECK(tree.children(0) == [1, 2])
    MINI_CHECK(tree.children(1) == [])


@MINI_TEST("SpatialOctree", "Order")
def test_octree_order():
    from session_py import SpatialOctree
    from session_py import Point

    # Root's grid accepts x=0 and x=4 (first point wins its cell); the octant
    # leaves absorb the rest in input order
    pts = [Point(float(x), 0.0, 0.0) for x in range(9)]
    tree = SpatialOctree(pts, 4.0, 4)

    MINI_CHECK(tree.order() == [0, 4, 1, 2, 3, 5, 6, 7, 8])


@MINI_TEST("SpatialOctree", "From Coords")
def test_octree_from_coords():
    from session_py import SpatialOctree

    # The flat-array constructor is the renderer's path (no per-point Point allocs)
    coords = []
    for x in range(9):
        coords.extend([float(x), 0.0, 0.0])
    tree = SpatialOctree.from_coords(coords, 4.0, 4)

    MINI_CHECK(tree.node_count() == 3)
    MINI_CHECK(tree.order() == [0, 4, 1, 2, 3, 5, 6, 7, 8])


if __name__ == "__main__":
    run_all("python")
