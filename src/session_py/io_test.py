from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
import os
from pathlib import Path


@MINI_TEST("Io", "Read Bunny")
def test_read_bunny():
    # load Stanford Bunny (real-world XYZ point cloud: 397 points)
    bunny_path = Path(__file__).resolve().parents[3] / "session_data" / "bunny.xyz"
    if not bunny_path.exists():
        return
    from session_py import read_xyz
    cloud = read_xyz(str(bunny_path))

    MINI_CHECK(cloud.point_count() == 397)
    points = cloud.get_points()
    MINI_CHECK(len(points) == 397)
    has_non_zero = any(p[0] != 0.0 or p[1] != 0.0 or p[2] != 0.0 for p in points)
    MINI_CHECK(has_non_zero)


@MINI_TEST("Io", "Write Read Roundtrip")
def test_write_read_roundtrip():
    # build a small cloud (4 points), write to XYZ, read back, compare counts
    from session_py import Point
    from session_py import PointCloud
    from session_py import read_xyz
    from session_py import write_xyz
    original = PointCloud()
    original.add_point(Point(0.0, 0.0, 0.0))
    original.add_point(Point(1.0, 0.0, 0.0))
    original.add_point(Point(0.0, 1.0, 0.0))
    original.add_point(Point(0.0, 0.0, 1.0))

    MINI_CHECK(original.point_count() == 4)
    temp_file = str(Path(__file__).resolve().parents[2] / "serialization" / "test_temp_roundtrip.xyz")
    write_xyz(original, temp_file)
    MINI_CHECK(os.path.exists(temp_file))
    loaded = read_xyz(temp_file)
    MINI_CHECK(loaded.point_count() == original.point_count())
    os.remove(temp_file)


@MINI_TEST("Io", "String Roundtrip")
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
    s = write_xyz_to_string(original)
    loaded = read_xyz_from_str(s)

    MINI_CHECK(loaded.point_count() == original.point_count())
    MINI_CHECK(TOLERANCE.is_close(loaded.get_points()[1][0], 1.0))


if __name__ == "__main__":
    run_all("python")
