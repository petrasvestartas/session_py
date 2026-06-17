from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("MeshOffset", "from_mesh")
def test_mesh_offset_from_mesh():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point
    pts = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    mesh = Mesh.from_vertices_and_faces(pts, [[0, 1, 2, 3]])
    result = MeshOffset.from_mesh(mesh, 1.0)
    import copy as _copy
    copy = _copy.copy(result)
    MINI_CHECK(result.is_valid())
    MINI_CHECK(result == copy)
    MINI_CHECK(not (result != copy))
    MINI_CHECK(result.number_of_vertices() == 8)
    MINI_CHECK(result.number_of_faces() == 6)


@MINI_TEST("MeshOffset", "from_mesh_grid")
def test_mesh_offset_from_mesh_grid():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point
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
    result = MeshOffset.from_mesh(mesh, 2.0)
    MINI_CHECK(result.is_valid())
    MINI_CHECK(result.number_of_vertices() == 18)
    MINI_CHECK(result.number_of_faces() == 16)


@MINI_TEST("MeshOffset", "from_mesh_layers")
def test_mesh_offset_from_mesh_layers():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point
    pts = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    mesh = Mesh.from_vertices_and_faces(pts, [[0, 1, 2, 3]])
    layers = MeshOffset.from_mesh_layers(mesh, 1.0)
    MINI_CHECK(layers.bottom.is_valid())
    MINI_CHECK(layers.top.is_valid())
    MINI_CHECK(layers.sides.is_valid())
    MINI_CHECK(layers.bottom.number_of_vertices() == 4)
    MINI_CHECK(layers.bottom.number_of_faces() == 1)
    MINI_CHECK(layers.top.number_of_vertices() == 4)
    MINI_CHECK(layers.top.number_of_faces() == 1)
    MINI_CHECK(layers.sides.number_of_faces() == 4)


@MINI_TEST("MeshOffset", "file_json_dump")
def test_mesh_offset_file_json_dump():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point
    from pathlib import Path
    pts = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    mesh = Mesh.from_vertices_and_faces(pts, [[0, 1, 2, 3]])
    result = MeshOffset.from_mesh(mesh, 1.0)
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh_offset.json"
    result.file_json_dump(fname)
    loaded = Mesh.file_json_load(fname)
    MINI_CHECK(loaded.is_valid())
    MINI_CHECK(loaded.number_of_vertices() == result.number_of_vertices())
    MINI_CHECK(loaded.number_of_faces() == result.number_of_faces())


@MINI_TEST("MeshOffset", "file_json_load")
def test_mesh_offset_file_json_load():
    from session_py import Mesh
    from pathlib import Path
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh_offset.json"
    loaded = Mesh.file_json_load(fname)
    MINI_CHECK(loaded.is_valid())
    MINI_CHECK(loaded.number_of_vertices() == 8)
    MINI_CHECK(loaded.number_of_faces() == 6)


@MINI_TEST("MeshOffset", "to_proto")
def test_mesh_offset_to_proto():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point
    from pathlib import Path
    pts = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    mesh = Mesh.from_vertices_and_faces(pts, [[0, 1, 2, 3]])
    result = MeshOffset.from_mesh(mesh, 1.0)
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh_offset.bin"
    result.pb_dump(fname)
    loaded = Mesh.pb_load(fname)
    MINI_CHECK(loaded.is_valid())
    MINI_CHECK(loaded.number_of_vertices() == result.number_of_vertices())
    MINI_CHECK(loaded.number_of_faces() == result.number_of_faces())


@MINI_TEST("MeshOffset", "from_proto")
def test_mesh_offset_from_proto():
    from session_py import Mesh
    from pathlib import Path
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh_offset.bin"
    loaded = Mesh.pb_load(fname)
    MINI_CHECK(loaded.is_valid())
    MINI_CHECK(loaded.number_of_vertices() == 8)
    MINI_CHECK(loaded.number_of_faces() == 6)


if __name__ == "__main__":
    run_all("python")
