from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("FileObj", "Read Bunny")
def test_read_bunny():
    from session_py import read_file_obj
    from pathlib import Path

    bunny_path = Path(__file__).resolve().parents[2] / "session_data" / "bunny.obj"

    MINI_CHECK(bunny_path.exists())

    mesh = read_file_obj(str(bunny_path))
    indexed = mesh.to_vertices_and_faces()
    vertices = indexed[0]
    faces = indexed[1]
    has_non_zero = False

    for v in vertices:
        if v[0] != 0.0 or v[1] != 0.0 or v[2] != 0.0:
            has_non_zero = True

    all_polygons = True

    for f in faces:
        if len(f) < 3:
            all_polygons = False

    MINI_CHECK(mesh.number_of_vertices() == 2503)
    MINI_CHECK(mesh.number_of_faces() == 4968)
    MINI_CHECK(len(vertices) == 2503)
    MINI_CHECK(len(faces) == 4968)
    MINI_CHECK(has_non_zero)
    MINI_CHECK(all_polygons)


@MINI_TEST("FileObj", "Write Read Roundtrip")
def test_write_read_roundtrip():
    from session_py import Mesh
    from session_py import Point
    from session_py import read_file_obj
    from session_py import write_file_obj
    from pathlib import Path
    import os

    os.makedirs(Path(__file__).resolve().parents[2] / "serialization", exist_ok=True)

    original = Mesh()
    v0 = original.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = original.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = original.add_vertex(Point(0.0, 1.0, 0.0))
    v3 = original.add_vertex(Point(0.0, 0.0, 1.0))
    original.add_face([v0, v1, v2])
    original.add_face([v0, v1, v3])

    filepath = str(
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_temp_roundtrip.obj"
    )
    write_file_obj(original, filepath)
    exists = os.path.exists(filepath)
    loaded = read_file_obj(filepath)

    MINI_CHECK(original.number_of_vertices() == 4)
    MINI_CHECK(original.number_of_faces() == 2)
    MINI_CHECK(exists)
    MINI_CHECK(loaded.number_of_vertices() == original.number_of_vertices())
    MINI_CHECK(loaded.number_of_faces() == original.number_of_faces())

    os.remove(filepath)


@MINI_TEST("FileObj", "String Roundtrip")
def test_string_roundtrip():
    from session_py import Mesh
    from session_py import Point
    from session_py import read_file_obj_from_str
    from session_py import write_file_obj_to_string

    original = Mesh()
    v0 = original.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = original.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = original.add_vertex(Point(0.0, 1.0, 0.0))
    v3 = original.add_vertex(Point(0.0, 0.0, 1.0))
    original.add_face([v0, v1, v2])
    original.add_face([v0, v1, v3])

    content = write_file_obj_to_string(original)
    loaded = read_file_obj_from_str(content)

    MINI_CHECK(loaded.number_of_vertices() == original.number_of_vertices())
    MINI_CHECK(loaded.number_of_faces() == original.number_of_faces())
    MINI_CHECK(TOLERANCE.is_close(loaded.area(), original.area()))


if __name__ == "__main__":
    run_all("python")
