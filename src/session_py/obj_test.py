from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
import os
from pathlib import Path


@MINI_TEST("OBJ", "Read Bunny")
def test_read_bunny():
    bunny_path = Path(__file__).resolve().parents[3] / "session_data" / "bunny.obj"
    if not bunny_path.exists():
        return
    from session_py import Mesh
    from session_py.obj import read_obj
    mesh = read_obj(str(bunny_path))

    MINI_CHECK(mesh.number_of_vertices() == 2503)
    MINI_CHECK(mesh.number_of_faces() == 4968)
    vertices, faces = mesh.to_vertices_and_faces()
    MINI_CHECK(len(vertices) == 2503)
    MINI_CHECK(len(faces) == 4968)
    has_non_zero = any(v[0] != 0.0 or v[1] != 0.0 or v[2] != 0.0 for v in vertices)
    MINI_CHECK(has_non_zero)
    MINI_CHECK(all(len(f) >= 3 for f in faces))


@MINI_TEST("OBJ", "Write Read Roundtrip")
def test_write_read_roundtrip():
    from session_py import Mesh, Point
    from session_py.obj import read_obj, write_obj
    original_mesh = Mesh()
    v0 = original_mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = original_mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = original_mesh.add_vertex(Point(0.0, 1.0, 0.0))
    v3 = original_mesh.add_vertex(Point(0.0, 0.0, 1.0))
    original_mesh.add_face([v0, v1, v2])
    original_mesh.add_face([v0, v1, v3])

    MINI_CHECK(original_mesh.number_of_vertices() == 4)
    MINI_CHECK(original_mesh.number_of_faces() == 2)
    temp_file = str(Path(__file__).resolve().parents[2] / "serialization" / "test_temp_roundtrip.obj")
    write_obj(original_mesh, temp_file)
    MINI_CHECK(os.path.exists(temp_file))
    loaded_mesh = read_obj(temp_file)
    MINI_CHECK(loaded_mesh.number_of_vertices() == original_mesh.number_of_vertices())
    MINI_CHECK(loaded_mesh.number_of_faces() == original_mesh.number_of_faces())
    os.remove(temp_file)


if __name__ == "__main__":
    run_all("python")
