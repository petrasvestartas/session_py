from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("IoXyz", "Read Bunny")
def test_read_bunny():
    from session_py import read_xyz
    from pathlib import Path

    bunny_path = Path(__file__).resolve().parents[3] / "session_data" / "bunny.xyz"

    if not bunny_path.exists():
        return

    cloud = read_xyz(str(bunny_path))
    points = cloud.get_points()
    has_non_zero = False

    for p in points:
        if p[0] != 0.0 or p[1] != 0.0 or p[2] != 0.0:
            has_non_zero = True

    MINI_CHECK(cloud.point_count() == 397)
    MINI_CHECK(len(points) == 397)
    MINI_CHECK(has_non_zero)


@MINI_TEST("IoXyz", "Write Read Roundtrip")
def test_write_read_roundtrip():
    from session_py import Point
    from session_py import PointCloud
    from session_py import read_xyz
    from session_py import write_xyz
    from pathlib import Path
    import os

    os.makedirs(Path(__file__).resolve().parents[2] / "serialization", exist_ok=True)

    original = PointCloud()
    original.add_point(Point(0.0, 0.0, 0.0))
    original.add_point(Point(1.0, 0.0, 0.0))
    original.add_point(Point(0.0, 1.0, 0.0))
    original.add_point(Point(0.0, 0.0, 1.0))

    filepath = str(
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_temp_roundtrip.xyz"
    )
    write_xyz(original, filepath)
    exists = os.path.exists(filepath)
    loaded = read_xyz(filepath)

    MINI_CHECK(original.point_count() == 4)
    MINI_CHECK(exists)
    MINI_CHECK(loaded.point_count() == original.point_count())

    os.remove(filepath)


@MINI_TEST("IoXyz", "String Roundtrip")
def test_string_roundtrip():
    from session_py import Point
    from session_py import PointCloud
    from session_py import read_xyz_from_str
    from session_py import write_xyz_to_string

    original = PointCloud()
    original.add_point(Point(0.0, 0.0, 0.0))
    original.add_point(Point(1.0, 0.0, 0.0))
    original.add_point(Point(0.0, 1.0, 0.0))
    original.add_point(Point(0.0, 0.0, 1.0))

    content = write_xyz_to_string(original)
    loaded = read_xyz_from_str(content)

    MINI_CHECK(loaded.point_count() == original.point_count())
    MINI_CHECK(TOLERANCE.is_close(loaded.get_points()[1][0], 1.0))


@MINI_TEST("IoXyz", "Write Exact Text")
def test_write_exact_text():
    from session_py import Point
    from session_py import PointCloud
    from session_py import read_xyz
    from session_py import write_xyz
    from session_py import write_xyz_to_string
    from pathlib import Path
    import os

    os.makedirs(Path(__file__).resolve().parents[2] / "serialization", exist_ok=True)

    original = PointCloud()
    original.add_point(Point(1.0, 2.5, -3.0))
    original.add_point(Point(0.1, 1e-05, 1e16))
    original.add_point(Point(123456.789, -0.0, 1.0 / 3.0))

    filepath = str(
        Path(__file__).resolve().parents[2] / "serialization" / "test_temp_exact.xyz"
    )
    write_xyz(original, filepath)
    with open(filepath, newline="") as file:
        text = file.read()
    loaded = read_xyz(filepath)

    MINI_CHECK(text == "1 2.5 -3\n0.1 1e-05 1e+16\n123456.789 -0 0.3333333333333333\n")
    MINI_CHECK(write_xyz_to_string(loaded) == text)
    MINI_CHECK(loaded.get_points()[2][2] == 1.0 / 3.0)

    os.remove(filepath)


@MINI_TEST("IoXyz", "File Errors")
def test_file_errors():
    from session_py import PointCloud
    from session_py import read_xyz
    from session_py import write_xyz

    cloud = PointCloud()
    read_failed = False
    write_failed = False

    try:
        read_xyz("./serialization/test_temp_missing.xyz")
    except OSError:
        read_failed = True

    try:
        write_xyz(cloud, "")
    except OSError:
        write_failed = True

    MINI_CHECK(read_failed)
    MINI_CHECK(write_failed)


if __name__ == "__main__":
    run_all("python")
