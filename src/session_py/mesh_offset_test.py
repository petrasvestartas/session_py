from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("MeshOffset", "From Mesh")
def test_mesh_offset_from_mesh():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point
    import copy as copier

    points = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    mesh = Mesh.from_vertices_and_faces(points, [[0, 1, 2, 3]])
    result = MeshOffset.from_mesh(mesh, 1.0)
    copy = copier.copy(result)
    MINI_CHECK(result.is_valid())
    MINI_CHECK(result.is_closed())
    MINI_CHECK(result == copy)
    MINI_CHECK(not (result != copy))
    MINI_CHECK(result.number_of_vertices() == 8)
    MINI_CHECK(result.number_of_faces() == 6)


@MINI_TEST("MeshOffset", "From Mesh Grid")
def test_mesh_offset_from_mesh_grid():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point

    points = [
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
    mesh = Mesh.from_vertices_and_faces(points, faces)
    result = MeshOffset.from_mesh(mesh, 2.0)
    MINI_CHECK(result.is_valid())
    MINI_CHECK(result.is_closed())
    MINI_CHECK(result.number_of_vertices() == 18)
    MINI_CHECK(result.number_of_faces() == 16)


@MINI_TEST("MeshOffset", "From Mesh Layers")
def test_mesh_offset_from_mesh_layers():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point

    points = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    mesh = Mesh.from_vertices_and_faces(points, [[0, 1, 2, 3]])
    layers = MeshOffset.from_mesh_layers(mesh, 1.0)
    MINI_CHECK(layers.bottom.is_valid())
    MINI_CHECK(layers.top.is_valid())
    MINI_CHECK(layers.sides.is_valid())
    MINI_CHECK(layers.bottom.number_of_vertices() == 4)
    MINI_CHECK(layers.bottom.number_of_faces() == 1)
    MINI_CHECK(layers.top.number_of_vertices() == 4)
    MINI_CHECK(layers.top.number_of_faces() == 1)
    MINI_CHECK(layers.sides.number_of_faces() == 4)


@MINI_TEST("MeshOffset", "Offset Planes")
def test_mesh_offset_offset_planes():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point

    points = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    mesh = Mesh.from_vertices_and_faces(points, [[0, 1, 2, 3]])
    planes = MeshOffset.offset_planes(mesh, 1.0)
    MINI_CHECK(len(planes) == 1)
    plane = planes[0]
    MINI_CHECK(TOLERANCE.is_close(plane.a, 0.0))
    MINI_CHECK(TOLERANCE.is_close(plane.b, 0.0))
    MINI_CHECK(TOLERANCE.is_close(plane.c, 1.0))
    MINI_CHECK(TOLERANCE.is_close(plane.d, -1.0))
    MINI_CHECK(TOLERANCE.is_close(plane.origin[2], 1.0))


@MINI_TEST("MeshOffset", "Offset Vertices")
def test_mesh_offset_offset_vertices():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point

    points = [
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
    mesh = Mesh.from_vertices_and_faces(points, faces)
    planes = MeshOffset.offset_planes(mesh, 2.0)
    offsets = MeshOffset.offset_vertices(mesh, planes)
    MINI_CHECK(len(planes) == 4)
    MINI_CHECK(len(offsets) == 9)

    for vkey in range(9):
        MINI_CHECK(TOLERANCE.is_close(offsets[vkey][0], points[vkey][0]))
        MINI_CHECK(TOLERANCE.is_close(offsets[vkey][1], points[vkey][1]))
        MINI_CHECK(TOLERANCE.is_close(offsets[vkey][2], 2.0))


@MINI_TEST("MeshOffset", "Json Roundtrip")
def test_mesh_offset_json_roundtrip():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point
    from pathlib import Path

    points = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    mesh = Mesh.from_vertices_and_faces(points, [[0, 1, 2, 3]])
    result = MeshOffset.from_mesh(mesh, 1.0)
    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_mesh_offset.json"
    )
    result.file_json_dump(filename)
    loaded = Mesh.file_json_load(filename)
    MINI_CHECK(loaded == result)
    MINI_CHECK(loaded.number_of_vertices() == 8)
    MINI_CHECK(loaded.number_of_faces() == 6)


@MINI_TEST("MeshOffset", "Protobuf Roundtrip")
def test_mesh_offset_protobuf_roundtrip():
    from session_py import MeshOffset
    from session_py import Mesh
    from session_py import Point
    from pathlib import Path

    points = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    mesh = Mesh.from_vertices_and_faces(points, [[0, 1, 2, 3]])
    result = MeshOffset.from_mesh(mesh, 1.0)
    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_mesh_offset.bin"
    )
    result.pb_dump(filename)
    loaded = Mesh.pb_load(filename)
    MINI_CHECK(loaded == result)
    MINI_CHECK(loaded.number_of_vertices() == 8)
    MINI_CHECK(loaded.number_of_faces() == 6)


if __name__ == "__main__":
    run_all("python")
