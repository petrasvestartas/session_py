from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("PointCloud", "Constructor")
def test_pointcloud_constructor():
    from session_py import PointCloud
    from session_py import Point
    from session_py import Vector
    from session_py import Color

    # Constructor with points, normals, colors
    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(1.0, 0.0, 0.0)
    p2 = Point(0.0, 1.0, 0.0)
    n0 = Vector(0.0, 0.0, 1.0)
    n1 = Vector(0.0, 0.0, 1.0)
    n2 = Vector(0.0, 0.0, 1.0)
    c0 = Color(255, 0, 0, 255)
    c1 = Color(0, 255, 0, 255)
    c2 = Color(0, 0, 255, 255)
    pc = PointCloud([p0, p1, p2], [n0, n1, n2], [c0, c1, c2])

    # Basic properties
    point_count = len(pc)
    color_count = pc.color_count()
    normal_count = pc.normal_count()
    is_empty = pc.is_empty()

    # Minimal and Full String Representation
    pcstr = str(pc)
    pcrepr = repr(pc)

    # Copy (duplicates everything except guid)
    pccopy = pc.duplicate()

    # Get point/color/normal at index
    pt0 = pc.get_point(0)
    col0 = pc.get_color(0)
    norm0 = pc.get_normal(0)

    # Add points, colors, normals to empty cloud
    pc2 = PointCloud()
    pc2.add_point(Point(1.0, 2.0, 3.0))
    pc2.add_color(Color(128, 64, 32, 255))
    pc2.add_normal(Vector(1.0, 0.0, 0.0))

    # Set point/color/normal at index
    pc2.set_point(0, Point(4.0, 5.0, 6.0))
    pc2.set_color(0, Color(200, 100, 50, 255))
    pc2.set_normal(0, Vector(0.0, 1.0, 0.0))

    # Translate with Vector offset
    pc3 = PointCloud([Point(1.0, 2.0, 3.0)])
    offset = Vector(10.0, 20.0, 30.0)
    pc_iadd = PointCloud([Point(1.0, 2.0, 3.0)])
    pc_iadd += offset
    pc_isub = PointCloud([Point(1.0, 2.0, 3.0)])
    pc_isub -= offset
    pc_add = pc3 + offset
    pc_sub = pc3 - offset

    # Create from flat arrays
    coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    colors_arr = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]
    normals_arr = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    pc4 = PointCloud.from_coords(coords, colors_arr, normals_arr)

    MINI_CHECK(pc.name == "my_pointcloud" and pc.guid != "" and point_count == 3)
    MINI_CHECK(color_count == 3 and normal_count == 3 and not is_empty)
    MINI_CHECK("3 points" in pcstr)
    MINI_CHECK("PointCloud(my_pointcloud" in pcrepr)
    MINI_CHECK(pccopy == pc and pccopy.guid != pc.guid)
    MINI_CHECK(TOLERANCE.is_close(pt0[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt0[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt0[2], 0.0))
    MINI_CHECK(col0[0] == 255 and col0[1] == 0 and col0[2] == 0 and col0[3] == 255)
    MINI_CHECK(TOLERANCE.is_close(norm0[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(norm0[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(norm0[2], 1.0))
    MINI_CHECK(len(pc2) == 1)
    MINI_CHECK(pc2.color_count() == 1)
    MINI_CHECK(pc2.normal_count() == 1)
    MINI_CHECK(TOLERANCE.is_close(pc2.get_point(0)[0], 4.0))
    MINI_CHECK(TOLERANCE.is_close(pc2.get_point(0)[1], 5.0))
    MINI_CHECK(TOLERANCE.is_close(pc2.get_point(0)[2], 6.0))
    MINI_CHECK(pc2.get_color(0)[0] == 200)
    MINI_CHECK(pc2.get_color(0)[1] == 100)
    MINI_CHECK(pc2.get_color(0)[2] == 50)
    MINI_CHECK(pc2.get_color(0)[3] == 255)
    MINI_CHECK(TOLERANCE.is_close(pc2.get_normal(0)[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pc2.get_normal(0)[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pc2.get_normal(0)[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pc_iadd.get_point(0)[0], 11.0))
    MINI_CHECK(TOLERANCE.is_close(pc_iadd.get_point(0)[1], 22.0))
    MINI_CHECK(TOLERANCE.is_close(pc_iadd.get_point(0)[2], 33.0))
    MINI_CHECK(TOLERANCE.is_close(pc_isub.get_point(0)[0], -9.0))
    MINI_CHECK(TOLERANCE.is_close(pc_isub.get_point(0)[1], -18.0))
    MINI_CHECK(TOLERANCE.is_close(pc_isub.get_point(0)[2], -27.0))
    MINI_CHECK(TOLERANCE.is_close(pc_add.get_point(0)[0], 11.0))
    MINI_CHECK(TOLERANCE.is_close(pc_add.get_point(0)[1], 22.0))
    MINI_CHECK(TOLERANCE.is_close(pc_add.get_point(0)[2], 33.0))
    MINI_CHECK(TOLERANCE.is_close(pc_sub.get_point(0)[0], -9.0))
    MINI_CHECK(TOLERANCE.is_close(pc_sub.get_point(0)[1], -18.0))
    MINI_CHECK(TOLERANCE.is_close(pc_sub.get_point(0)[2], -27.0))
    MINI_CHECK(TOLERANCE.is_close(pc3.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pc3.get_point(0)[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pc3.get_point(0)[2], 3.0))
    MINI_CHECK(len(pc4) == 3)
    MINI_CHECK(pc4.color_count() == 3)
    MINI_CHECK(pc4.normal_count() == 3)
    MINI_CHECK(TOLERANCE.is_close(pc4.get_point(1)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pc4.get_point(1)[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pc4.get_point(1)[2], 0.0))
    MINI_CHECK(pc4.get_color(1)[0] == 0)
    MINI_CHECK(pc4.get_color(1)[1] == 255)
    MINI_CHECK(pc4.get_color(1)[2] == 0)
    MINI_CHECK(pc4.get_color(1)[3] == 255)
    MINI_CHECK(TOLERANCE.is_close(pc4.get_normal(1)[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pc4.get_normal(1)[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pc4.get_normal(1)[2], 1.0))


@MINI_TEST("PointCloud", "Transform")
def test_pointcloud_transform():
    from session_py import PointCloud
    from session_py import Point
    from session_py import Xform

    # Transform - in-place transformation
    pc = PointCloud([Point(1.0, 2.0, 3.0)])
    pc.xform = Xform.translation(10.0, 20.0, 30.0)
    pc.transform()

    # Transformed - returns new cloud
    pc2 = PointCloud([Point(1.0, 2.0, 3.0)])
    pc2.xform = Xform.translation(10.0, 20.0, 30.0)
    pc3 = pc2.transformed()

    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[0], 11.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[1], 22.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[2], 33.0))
    MINI_CHECK(TOLERANCE.is_close(pc3.get_point(0)[0], 11.0))
    MINI_CHECK(TOLERANCE.is_close(pc3.get_point(0)[1], 22.0))
    MINI_CHECK(TOLERANCE.is_close(pc3.get_point(0)[2], 33.0))
    MINI_CHECK(TOLERANCE.is_close(pc2.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pc2.get_point(0)[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pc2.get_point(0)[2], 3.0))


@MINI_TEST("PointCloud", "Json Roundtrip")
def test_pointcloud_json_roundtrip():
    from session_py import PointCloud
    from session_py import Point
    from session_py import Vector
    from session_py import Color
    from pathlib import Path

    pc = PointCloud(
        [
            Point(1.0, 2.0, 3.0),
            Point(4.0, 5.0, 6.0),
        ],
        [
            Vector(0.0, 0.0, 1.0),
            Vector(0.0, 0.0, 1.0),
        ],
        [
            Color(255, 0, 0, 255),
            Color(0, 255, 0, 255),
        ]
    )
    pc.name = "test_pointcloud"

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   json_dumps()    │ str          │ to JSON string
    #   json_loads(s)   │ str          │ from JSON string
    #   json_dump(path) │ file         │ write to file
    #   json_load(path) │ file         │ read from file

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_pointcloud.json"
    pc.json_dump(fname)
    loaded = PointCloud.json_load(fname)

    MINI_CHECK(loaded.name == "test_pointcloud")
    MINI_CHECK(len(loaded) == 2)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[2], 3.0))
    MINI_CHECK(loaded.get_color(0)[0] == 255)
    MINI_CHECK(loaded.get_color(0)[1] == 0)
    MINI_CHECK(loaded.get_color(0)[2] == 0)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_normal(0)[2], 1.0))


@MINI_TEST("PointCloud", "Protobuf Roundtrip")
def test_pointcloud_protobuf_roundtrip():
    from session_py import PointCloud
    from session_py import Point
    from session_py import Vector
    from session_py import Color
    from pathlib import Path

    pc = PointCloud(
        [
            Point(1.0, 2.0, 3.0),
            Point(4.0, 5.0, 6.0),
        ],
        [
            Vector(0.0, 0.0, 1.0),
            Vector(0.0, 0.0, 1.0),
        ],
        [
            Color(255, 0, 0, 255),
            Color(0, 255, 0, 255),
        ]
    )
    pc.name = "test_pointcloud"

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_pointcloud.bin"
    pc.pb_dump(fname)
    loaded = PointCloud.pb_load(fname)

    MINI_CHECK(loaded.name == "test_pointcloud")
    MINI_CHECK(len(loaded) == 2)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[2], 3.0))
    MINI_CHECK(loaded.get_color(0)[0] == 255)
    MINI_CHECK(loaded.get_color(0)[1] == 0)
    MINI_CHECK(loaded.get_color(0)[2] == 0)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_normal(0)[2], 1.0))


if __name__ == "__main__":
    run_all("python")
