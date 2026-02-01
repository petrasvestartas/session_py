from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
import math


@MINI_TEST("Mesh", "constructor")
def test_mesh_constructor():
    from session_py import Mesh

    mesh = Mesh()

    num_vertices = mesh.number_of_vertices()
    num_faces = mesh.number_of_faces()
    num_edges = mesh.number_of_edges()
    is_empty = mesh.is_empty()
    euler = mesh.euler()

    MINI_CHECK(num_vertices == 0)
    MINI_CHECK(num_faces == 0)
    MINI_CHECK(num_edges == 0)
    MINI_CHECK(is_empty)
    MINI_CHECK(euler == 0)
    MINI_CHECK(mesh.name == "my_mesh")
    MINI_CHECK(mesh.guid)


@MINI_TEST("Mesh", "add_vertex")
def test_mesh_add_vertex():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(1.0, 2.0, 3.0))

    MINI_CHECK(mesh.number_of_vertices() == 1)
    MINI_CHECK(not mesh.is_empty())

    pos = mesh.vertex_position(v0)
    MINI_CHECK(pos[0] == 1.0 and pos[1] == 2.0 and pos[2] == 3.0)

    v1 = mesh.add_vertex(Point(4.0, 5.0, 6.0), 42)
    MINI_CHECK(v1 == 42)
    MINI_CHECK(mesh.number_of_vertices() == 2)


@MINI_TEST("Mesh", "add_face")
def test_mesh_add_face():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))

    f = mesh.add_face([v0, v1, v2])
    MINI_CHECK(f is not None)
    MINI_CHECK(mesh.number_of_faces() == 1)
    MINI_CHECK(mesh.number_of_edges() == 3)
    MINI_CHECK(mesh.euler() == 1)

    invalid1 = mesh.add_face([v0, v1])
    MINI_CHECK(invalid1 is None)

    invalid2 = mesh.add_face([v0, v1, 999])
    MINI_CHECK(invalid2 is None)

    invalid3 = mesh.add_face([v0, v1, v0])
    MINI_CHECK(invalid3 is None)


@MINI_TEST("Mesh", "face_vertices")
def test_mesh_face_vertices():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))

    f = mesh.add_face([v0, v1, v2])
    vertices = mesh.face_vertices(f)

    MINI_CHECK(len(vertices) == 3)
    MINI_CHECK(vertices[0] == v0)
    MINI_CHECK(vertices[1] == v1)
    MINI_CHECK(vertices[2] == v2)


@MINI_TEST("Mesh", "vertex_neighbors")
def test_mesh_vertex_neighbors():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))

    mesh.add_face([v0, v1, v2])

    neighbors = mesh.vertex_neighbors(v0)
    MINI_CHECK(len(neighbors) == 2)
    MINI_CHECK(v1 in neighbors)
    MINI_CHECK(v2 in neighbors)


@MINI_TEST("Mesh", "vertex_faces")
def test_mesh_vertex_faces():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    v3 = mesh.add_vertex(Point(1.0, 1.0, 0.0))

    f1 = mesh.add_face([v0, v1, v2])
    f2 = mesh.add_face([v1, v3, v2])

    faces = mesh.vertex_faces(v1)
    MINI_CHECK(len(faces) == 2)
    MINI_CHECK(f1 in faces)
    MINI_CHECK(f2 in faces)


@MINI_TEST("Mesh", "is_vertex_on_boundary")
def test_mesh_is_vertex_on_boundary():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))

    mesh.add_face([v0, v1, v2])

    MINI_CHECK(mesh.is_vertex_on_boundary(v0))
    MINI_CHECK(mesh.is_vertex_on_boundary(v1))
    MINI_CHECK(mesh.is_vertex_on_boundary(v2))


@MINI_TEST("Mesh", "face_normal")
def test_mesh_face_normal():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))

    f = mesh.add_face([v0, v1, v2])
    normal = mesh.face_normal(f)

    MINI_CHECK(TOLERANCE.is_close(normal[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(normal[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(normal[1], 0.0))


@MINI_TEST("Mesh", "face_area")
def test_mesh_face_area():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))

    f = mesh.add_face([v0, v1, v2])
    area = mesh.face_area(f)

    MINI_CHECK(TOLERANCE.is_close(area, 0.5))


@MINI_TEST("Mesh", "from_polygons")
def test_mesh_from_polygons():
    from session_py import Mesh
    from session_py import Point

    triangle = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]

    mesh = Mesh.from_polygons([triangle])
    MINI_CHECK(mesh.number_of_vertices() == 3)
    MINI_CHECK(mesh.number_of_faces() == 1)
    MINI_CHECK(mesh.number_of_edges() == 3)

    tri1 = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]
    tri2 = [
        Point(1.0, 0.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(1.0, 1.0, 0.0),
    ]

    mesh2 = Mesh.from_polygons([tri1, tri2])
    MINI_CHECK(mesh2.number_of_vertices() == 4)
    MINI_CHECK(mesh2.number_of_faces() == 2)


@MINI_TEST("Mesh", "clear")
def test_mesh_clear():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    mesh.add_face([v0, v1, v2])

    MINI_CHECK(not mesh.is_empty())

    mesh.clear()

    MINI_CHECK(mesh.is_empty())
    MINI_CHECK(mesh.number_of_vertices() == 0)
    MINI_CHECK(mesh.number_of_faces() == 0)


@MINI_TEST("Mesh", "transformation")
def test_mesh_transformation():
    from session_py import Mesh
    from session_py import Point
    from session_py import Xform

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    mesh.add_face([v0, v1, v2])

    mesh.xform = Xform.translation(10.0, 20.0, 30.0)
    mesh_transformed = mesh.transformed()
    mesh.transform()

    pos0 = mesh.vertex_position(v0)
    MINI_CHECK(pos0[0] == 10.0)
    MINI_CHECK(pos0[1] == 20.0)
    MINI_CHECK(pos0[2] == 30.0)
    MINI_CHECK(mesh.xform == Xform.identity())
    MINI_CHECK(mesh_transformed.xform == Xform.identity())


@MINI_TEST("Mesh", "json_roundtrip")
def test_mesh_json_roundtrip():
    from session_py import Mesh
    from session_py import Point
    from pathlib import Path

    mesh = Mesh()
    mesh.name = "test_mesh"
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    mesh.add_face([v0, v1, v2])

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   json_dumps()    │ str          │ to JSON string
    #   json_loads(s)   │ str          │ from JSON string
    #   json_dump(path) │ file         │ write to file
    #   json_load(path) │ file         │ read from file

    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh.json"
    mesh.json_dump(filename)
    loaded = Mesh.json_load(filename)

    MINI_CHECK(loaded.name == mesh.name)
    MINI_CHECK(loaded.number_of_vertices() == mesh.number_of_vertices())
    MINI_CHECK(loaded.number_of_faces() == mesh.number_of_faces())


@MINI_TEST("Mesh", "protobuf_roundtrip")
def test_mesh_protobuf_roundtrip():
    from session_py import Mesh
    from session_py import Point
    from pathlib import Path

    mesh = Mesh()
    mesh.name = "test_mesh_proto"
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    mesh.add_face([v0, v1, v2])

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh.bin"
    mesh.pb_dump(filename)
    loaded = Mesh.pb_load(filename)

    MINI_CHECK(loaded.name == mesh.name)
    MINI_CHECK(loaded.number_of_vertices() == mesh.number_of_vertices())
    MINI_CHECK(loaded.number_of_faces() == mesh.number_of_faces())
    MINI_CHECK(loaded.guid == mesh.guid)


@MINI_TEST("Mesh", "vertex_position")
def test_mesh_vertex_position():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(1.0, 2.0, 3.0))

    pos = mesh.vertex_position(v0)
    MINI_CHECK(pos[0] == 1.0)
    MINI_CHECK(pos[1] == 2.0)
    MINI_CHECK(pos[2] == 3.0)
    MINI_CHECK(mesh.vertex_position(999) is None)


@MINI_TEST("Mesh", "vertex_normal")
def test_mesh_vertex_normal():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    v3 = mesh.add_vertex(Point(1.0, 1.0, 0.0))

    mesh.add_face([v0, v1, v3])
    mesh.add_face([v0, v3, v2])

    normal = mesh.vertex_normal(v0)
    MINI_CHECK(abs(normal[2]) == 1.0)


@MINI_TEST("Mesh", "to_vertices_and_faces")
def test_mesh_to_vertices_and_faces():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))

    mesh.add_face([v0, v1, v2])

    vertices, faces = mesh.to_vertices_and_faces()

    MINI_CHECK(len(vertices) == 3)
    MINI_CHECK(len(faces) == 1)
    MINI_CHECK(len(faces[0]) == 3)


if __name__ == "__main__":
    run_all(language="python")
