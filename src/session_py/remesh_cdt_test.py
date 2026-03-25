from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("RemeshCDT", "Triangle")
def test_remesh_cdt_triangle():
    from session_py.remesh_cdt import cdt_triangulate

    tris = cdt_triangulate([(0, 0), (1, 0), (0, 1)], None)

    MINI_CHECK(len(tris) == 1)
    MINI_CHECK(tris[0][0] != tris[0][1])
    MINI_CHECK(tris[0][1] != tris[0][2])
    MINI_CHECK(tris[0][0] != tris[0][2])


@MINI_TEST("RemeshCDT", "Square")
def test_remesh_cdt_square():
    from session_py.remesh_cdt import cdt_triangulate

    tris = cdt_triangulate([(0, 0), (1, 0), (1, 1), (0, 1)], None)

    MINI_CHECK(len(tris) == 2)


@MINI_TEST("RemeshCDT", "Convex Polygon")
def test_remesh_cdt_convex_polygon():
    from session_py.remesh_cdt import cdt_triangulate
    import math

    hex_pts = [(math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)) for i in range(6)]
    tris = cdt_triangulate(hex_pts, None)

    MINI_CHECK(len(tris) == 4)


@MINI_TEST("RemeshCDT", "Polygon With Hole")
def test_remesh_cdt_polygon_with_hole():
    from session_py.remesh_cdt import cdt_triangulate
    from session_py.tolerance import TOLERANCE

    border = [(0, 0), (4, 0), (4, 4), (0, 4)]
    hole = [(1, 1), (1, 3), (3, 3), (3, 1)]
    tris = cdt_triangulate(border, [hole])

    flat = [(0,0),(4,0),(4,4),(0,4),(1,1),(1,3),(3,3),(3,1)]
    area = sum(abs((flat[b][0]-flat[a][0])*(flat[c][1]-flat[a][1]) - (flat[c][0]-flat[a][0])*(flat[b][1]-flat[a][1])) * 0.5
               for a, b, c in tris)

    MINI_CHECK(len(tris) > 0)
    MINI_CHECK(TOLERANCE.is_close(area, 4.0*4.0 - 2.0*2.0))


if __name__ == "__main__":
    run_all(language="python")
