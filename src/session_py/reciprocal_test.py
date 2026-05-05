from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("Reciprocal", "from_mesh")
def test_reciprocal_from_mesh():
    from session_py import Mesh
    from session_py import Point
    from session_py.reciprocal import Reciprocal

    pts = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(2, 0, 0),
        Point(0, 1, 0),
        Point(1, 1, 0),
        Point(2, 1, 0),
        Point(0, 2, 0),
        Point(1, 2, 0),
        Point(2, 2, 0),
    ]
    faces = [
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7],
    ]
    mesh = Mesh.from_vertices_and_faces(pts, faces)
    r = Reciprocal.from_mesh(mesh, 0.7, 1.4, True, 1.0)
    ne = mesh.number_of_edges()
    MINI_CHECK(len(r.center) == ne)
    MINI_CHECK(len(r.top) == ne)
    MINI_CHECK(len(r.bottom) == ne)
    MINI_CHECK(len(r.lineplanes) == ne)
    MINI_CHECK(len(r.endplanes) == ne)
    MINI_CHECK(r.center[0].length() > 0.0)
    MINI_CHECK(r.lineplanes[0].is_valid())


if __name__ == "__main__":
    run_all("python")
