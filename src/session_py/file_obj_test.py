from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
import os
from pathlib import Path


@MINI_TEST("FileObj", "Read Bunny")
def test_read_bunny():
    # load Stanford Bunny (real-world OBJ: 2503 vertices, 4968 faces)
    bunny_path = Path(__file__).resolve().parents[3] / "session_data" / "bunny.obj"
    if not bunny_path.exists():
        return
    from session_py import Mesh
    from session_py.file_obj import read_file_obj
    mesh = read_file_obj(str(bunny_path))

    MINI_CHECK(mesh.number_of_vertices() == 2503)
    MINI_CHECK(mesh.number_of_faces() == 4968)
    vertices, faces = mesh.to_vertices_and_faces()
    MINI_CHECK(len(vertices) == 2503)
    MINI_CHECK(len(faces) == 4968)
    has_non_zero = any(v[0] != 0.0 or v[1] != 0.0 or v[2] != 0.0 for v in vertices)
    MINI_CHECK(has_non_zero)
    MINI_CHECK(all(len(f) >= 3 for f in faces))


@MINI_TEST("FileObj", "Write Read Roundtrip")
def test_write_read_roundtrip():
    # build a small mesh (4 verts, 2 faces), write to OBJ, read back, compare counts
    from session_py import Mesh, Point
    from session_py.file_obj import read_file_obj, write_file_obj
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
    write_file_obj(original_mesh, temp_file)
    MINI_CHECK(os.path.exists(temp_file))
    loaded_mesh = read_file_obj(temp_file)
    MINI_CHECK(loaded_mesh.number_of_vertices() == original_mesh.number_of_vertices())
    MINI_CHECK(loaded_mesh.number_of_faces() == original_mesh.number_of_faces())
    os.remove(temp_file)


@MINI_TEST("FileObj", "String Roundtrip")
def test_string_roundtrip():
    from session_py import Mesh, Point
    from session_py.file_obj import read_file_obj_from_str, write_file_obj_to_string
    from session_py.tolerance import TOLERANCE
    original_mesh = Mesh()
    v0 = original_mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = original_mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = original_mesh.add_vertex(Point(0.0, 1.0, 0.0))
    v3 = original_mesh.add_vertex(Point(0.0, 0.0, 1.0))
    original_mesh.add_face([v0, v1, v2])
    original_mesh.add_face([v0, v1, v3])
    s = write_file_obj_to_string(original_mesh)
    loaded_mesh = read_file_obj_from_str(s)

    MINI_CHECK(loaded_mesh.number_of_vertices() == original_mesh.number_of_vertices())
    MINI_CHECK(loaded_mesh.number_of_faces() == original_mesh.number_of_faces())
    MINI_CHECK(TOLERANCE.is_close(loaded_mesh.area(), original_mesh.area()))


if __name__ == "__main__":
    run_all("python")
