from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
import math


@MINI_TEST("Mesh", "Constructor")
def test_mesh_constructor():
    from session_py import Mesh
    from session_py import Point

    sides = 6

    # Create hexagon vertices in XY plane
    vertices = []
    for i in range(sides):
        angle = 2.0 * TOLERANCE.PI * i / sides
        x = 1.0 * math.cos(angle)
        y = 1.0 * math.sin(angle)
        vertices.append(Point(x, y, 0.0))

    # Add center point as last vertex
    vertices.append(Point(0.0, 0.0, 0.0))
    faces = [
        [0, 1, 6],
        [1, 2, 6],
        [2, 3, 6],
        [3, 4, 6],
        [4, 5, 6],
        [5, 0, 6],
    ]

    mesh = Mesh.from_vertices_and_faces(vertices, faces)

    num_vertices = mesh.number_of_vertices()
    num_faces = mesh.number_of_faces()
    num_edges = mesh.number_of_edges()
    is_empty = mesh.is_empty()
    euler = mesh.euler()

    # String representations
    sstr = str(mesh)
    srepr = repr(mesh)

    # Copy (new guid)
    import copy
    mcopy = copy.copy(mesh)

    MINI_CHECK(num_vertices == 7)
    MINI_CHECK(num_faces == 6)
    MINI_CHECK(num_edges == 12)
    MINI_CHECK(not is_empty)
    MINI_CHECK(euler == 1)
    MINI_CHECK(mesh.name == "my_mesh")
    MINI_CHECK(mesh.guid)
    MINI_CHECK("Mesh" in sstr)
    MINI_CHECK("name=my_mesh" in srepr)
    MINI_CHECK(mcopy.guid != mesh.guid)
    MINI_CHECK(mcopy == mesh)
    MINI_CHECK(not (mcopy != mesh))


@MINI_TEST("Mesh", "Add_vertex")
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


@MINI_TEST("Mesh", "Add_face")
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


@MINI_TEST("Mesh", "Face_vertices")
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


@MINI_TEST("Mesh", "Vertex_neighbors")
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


@MINI_TEST("Mesh", "Vertex_faces")
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


@MINI_TEST("Mesh", "Is_vertex_on_boundary")
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


@MINI_TEST("Mesh", "Face_normal")
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


@MINI_TEST("Mesh", "Face_area")
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


@MINI_TEST("Mesh", "From_polylines")
def test_mesh_from_polylines():
    from session_py import Mesh
    from session_py import Point

    triangle = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]

    mesh = Mesh.from_polylines([triangle])
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

    mesh2 = Mesh.from_polylines([tri1, tri2])
    MINI_CHECK(mesh2.number_of_vertices() == 4)
    MINI_CHECK(mesh2.number_of_faces() == 2)


@MINI_TEST("Mesh", "Clear")
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


@MINI_TEST("Mesh", "Transformation")
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


@MINI_TEST("Mesh", "Json_roundtrip")
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


@MINI_TEST("Mesh", "Protobuf_roundtrip")
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


@MINI_TEST("Mesh", "Vertex_position")
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


@MINI_TEST("Mesh", "Vertex_normal")
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


@MINI_TEST("Mesh", "To_vertices_and_faces")
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


@MINI_TEST("Mesh", "From_lines")
def test_mesh_from_lines():
    from session_py import Mesh
    from session_py import Line
    from session_py import Point

    # Grid of unit segments forming 4 quads (3x3 grid)
    lines = []
    for i in range(3):
        for j in range(2):
            lines.append(Line.from_points(Point(i,j,0), Point(i,j+1,0)))
            lines.append(Line.from_points(Point(j,i,0), Point(j+1,i,0)))
    mesh = Mesh.from_lines(lines, True)
    MINI_CHECK(mesh.number_of_vertices() == 9)
    MINI_CHECK(mesh.number_of_faces() == 4)


if __name__ == "__main__":
    run_all(language="python")
