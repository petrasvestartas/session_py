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

    # Default constructor (empty cloud)
    pc0 = PointCloud()

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

    # Python-only: coords is a public flat List[float] [x0,y0,z0, x1,y1,z1, ...]
    # C++/Rust store coords privately; use get_point()/get_points() for cross-language code
    coords = pc.coords

    # Minimal and Full String Representation
    pcstr = str(pc)
    pcrepr = repr(pc)

    # Copy (duplicates everything except guid)
    pccopy = pc.duplicate()
    pcother = PointCloud()

    # Copy operators
    offset = Vector(10.0, 20.0, 30.0)
    pc_iadd = PointCloud([Point(1.0, 2.0, 3.0)])
    pc_iadd += offset
    pc_isub = PointCloud([Point(1.0, 2.0, 3.0)])
    pc_isub -= offset
    pc3 = PointCloud([Point(1.0, 2.0, 3.0)])
    pc_add = pc3 + offset
    pc_sub = pc3 - offset

    MINI_CHECK(pc0.name == "my_pointcloud")
    MINI_CHECK(pc0.guid)
    MINI_CHECK(pc0.is_empty())
    MINI_CHECK(len(pc) == 3)
    MINI_CHECK(pcstr == "3 points")
    MINI_CHECK(pcrepr == "PointCloud(my_pointcloud, 3 points, 3 colors, 3 normals)")
    MINI_CHECK(pccopy == pc and pccopy.guid != pc.guid)
    MINI_CHECK(len(coords) == 9 and TOLERANCE.is_close(coords[3], 1.0))
    MINI_CHECK(pcother != pc)
    MINI_CHECK(TOLERANCE.is_close(pc_iadd.get_point(0)[0], 11.0) and TOLERANCE.is_close(pc_iadd.get_point(0)[1], 22.0) and TOLERANCE.is_close(pc_iadd.get_point(0)[2], 33.0))
    MINI_CHECK(TOLERANCE.is_close(pc_isub.get_point(0)[0], -9.0) and TOLERANCE.is_close(pc_isub.get_point(0)[1], -18.0) and TOLERANCE.is_close(pc_isub.get_point(0)[2], -27.0))
    MINI_CHECK(TOLERANCE.is_close(pc_add.get_point(0)[0], 11.0) and TOLERANCE.is_close(pc_add.get_point(0)[1], 22.0) and TOLERANCE.is_close(pc_add.get_point(0)[2], 33.0))
    MINI_CHECK(TOLERANCE.is_close(pc_sub.get_point(0)[0], -9.0) and TOLERANCE.is_close(pc_sub.get_point(0)[1], -18.0) and TOLERANCE.is_close(pc_sub.get_point(0)[2], -27.0))


@MINI_TEST("PointCloud", "From Coords")
def test_pointcloud_from_coords():
    from session_py import PointCloud

    coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    colors = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]
    normals = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    pc = PointCloud.from_coords(coords, colors, normals)

    MINI_CHECK(len(pc) == 3 and pc.color_count() == 3 and pc.normal_count() == 3)
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(1)[0], 1.0))
    MINI_CHECK(pc.get_color(1)[1] == 1.0)
    MINI_CHECK(TOLERANCE.is_close(pc.get_normal(1)[2], 1.0))


@MINI_TEST("PointCloud", "Point Count")
def test_pointcloud_point_count():
    from session_py import PointCloud
    from session_py import Point

    pc = PointCloud([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.0, 1.0, 0.0)])

    MINI_CHECK(pc.point_count() == 3)


@MINI_TEST("PointCloud", "Len")
def test_pointcloud_len():
    from session_py import PointCloud
    from session_py import Point

    pc = PointCloud([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)])

    MINI_CHECK(len(pc) == 2)


@MINI_TEST("PointCloud", "Is Empty")
def test_pointcloud_is_empty():
    from session_py import PointCloud
    from session_py import Point

    pc0 = PointCloud()
    pc1 = PointCloud([Point(0.0, 0.0, 0.0)])

    MINI_CHECK(pc0.is_empty())
    MINI_CHECK(not pc1.is_empty())


@MINI_TEST("PointCloud", "Get Point")
def test_pointcloud_get_point():
    from session_py import PointCloud
    from session_py import Point

    pc = PointCloud([Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)])
    pt = pc.get_point(1)

    MINI_CHECK(TOLERANCE.is_close(pt[0], 4.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 5.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 6.0))


@MINI_TEST("PointCloud", "Set Point")
def test_pointcloud_set_point():
    from session_py import PointCloud
    from session_py import Point

    pc = PointCloud([Point(0.0, 0.0, 0.0)])
    pc.set_point(0, Point(4.0, 5.0, 6.0))

    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[0], 4.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[1], 5.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[2], 6.0))


@MINI_TEST("PointCloud", "Add Point")
def test_pointcloud_add_point():
    from session_py import PointCloud
    from session_py import Point

    pc = PointCloud()
    pc.add_point(Point(1.0, 2.0, 3.0))

    MINI_CHECK(len(pc) == 1)
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[2], 3.0))


@MINI_TEST("PointCloud", "Get Points")
def test_pointcloud_get_points():
    from session_py import PointCloud
    from session_py import Point

    pc = PointCloud([Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)])
    # get_points() allocates N Point objects from flat coords — use pc.coords for fast iteration
    points = pc.get_points()

    # Python-only: direct flat-array access avoids Point allocation overhead
    c = pc.coords
    x1 = c[3]

    MINI_CHECK(len(points) == 2)
    MINI_CHECK(TOLERANCE.is_close(points[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(points[1][2], 6.0))
    MINI_CHECK(TOLERANCE.is_close(x1, 4.0))


@MINI_TEST("PointCloud", "Color Count")
def test_pointcloud_color_count():
    from session_py import PointCloud
    from session_py import Color

    pc = PointCloud(colors=[Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)])

    MINI_CHECK(pc.color_count() == 2)


@MINI_TEST("PointCloud", "Get Color")
def test_pointcloud_get_color():
    from session_py import PointCloud
    from session_py import Color

    pc = PointCloud(colors=[Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)])
    c = pc.get_color(1)

    MINI_CHECK(c[0] == 0.0 and c[1] == 1.0 and c[2] == 0.0 and c[3] == 1.0)


@MINI_TEST("PointCloud", "Set Color")
def test_pointcloud_set_color():
    from session_py import PointCloud
    from session_py import Color

    pc = PointCloud(colors=[Color(0.0, 0.0, 0.0, 0.0)])
    pc.set_color(0, Color(1.0, 0.0, 0.0, 1.0))

    MINI_CHECK(pc.get_color(0)[0] == 1.0 and pc.get_color(0)[1] == 0.0 and pc.get_color(0)[2] == 0.0 and pc.get_color(0)[3] == 1.0)


@MINI_TEST("PointCloud", "Add Color")
def test_pointcloud_add_color():
    from session_py import PointCloud
    from session_py import Color

    pc = PointCloud()
    pc.add_color(Color(1.0, 0.0, 0.0, 1.0))

    MINI_CHECK(pc.color_count() == 1)
    MINI_CHECK(pc.get_color(0)[0] == 1.0 and pc.get_color(0)[1] == 0.0 and pc.get_color(0)[2] == 0.0)


@MINI_TEST("PointCloud", "Coords")
def test_pointcloud_coords():
    from session_py import PointCloud
    from session_py import Point

    pc = PointCloud([Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)])
    coords = pc.coords

    MINI_CHECK(len(coords) == 6)
    MINI_CHECK(TOLERANCE.is_close(coords[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(coords[5], 6.0))


@MINI_TEST("PointCloud", "Colors")
def test_pointcloud_colors():
    from session_py import PointCloud
    from session_py import Color

    pc = PointCloud(colors=[Color(1.0, 0.0, 0.0, 1.0)])
    colors = pc.colors

    MINI_CHECK(len(colors) == 4)
    MINI_CHECK(colors[0] == 255 and colors[1] == 0 and colors[2] == 0 and colors[3] == 255)


@MINI_TEST("PointCloud", "Get Colors")
def test_pointcloud_get_colors():
    from session_py import PointCloud
    from session_py import Color

    pc = PointCloud(colors=[Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)])
    colors = pc.get_colors()

    MINI_CHECK(len(colors) == 2)
    MINI_CHECK(colors[0][0] == 1.0)
    MINI_CHECK(colors[1][1] == 1.0)


@MINI_TEST("PointCloud", "Normal Count")
def test_pointcloud_normal_count():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud(normals=[Vector(0.0, 0.0, 1.0), Vector(0.0, 0.0, 1.0)])

    MINI_CHECK(pc.normal_count() == 2)


@MINI_TEST("PointCloud", "Get Normal")
def test_pointcloud_get_normal():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud(normals=[Vector(0.0, 0.0, 1.0), Vector(1.0, 0.0, 0.0)])
    n = pc.get_normal(1)

    MINI_CHECK(TOLERANCE.is_close(n[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(n[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(n[2], 0.0))


@MINI_TEST("PointCloud", "Set Normal")
def test_pointcloud_set_normal():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud(normals=[Vector(0.0, 0.0, 1.0)])
    pc.set_normal(0, Vector(0.0, 1.0, 0.0))

    MINI_CHECK(TOLERANCE.is_close(pc.get_normal(0)[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_normal(0)[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_normal(0)[2], 0.0))


@MINI_TEST("PointCloud", "Add Normal")
def test_pointcloud_add_normal():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud()
    pc.add_normal(Vector(1.0, 0.0, 0.0))

    MINI_CHECK(pc.normal_count() == 1)
    MINI_CHECK(TOLERANCE.is_close(pc.get_normal(0)[0], 1.0))


@MINI_TEST("PointCloud", "Get Normals")
def test_pointcloud_get_normals():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud(normals=[Vector(0.0, 0.0, 1.0), Vector(1.0, 0.0, 0.0)])
    normals = pc.get_normals()

    MINI_CHECK(len(normals) == 2)
    MINI_CHECK(TOLERANCE.is_close(normals[0][2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(normals[1][0], 1.0))


@MINI_TEST("PointCloud", "Transform")
def test_pointcloud_transform():
    from session_py import PointCloud
    from session_py import Point
    from session_py import Xform

    pc = PointCloud([Point(1.0, 2.0, 3.0)])
    pc_xf = Xform.translation(10.0, 20.0, 30.0)
    pc.transform(pc_xf)

    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[0], 11.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[1], 22.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[2], 33.0))


@MINI_TEST("PointCloud", "Transformed")
def test_pointcloud_transformed():
    from session_py import PointCloud
    from session_py import Point
    from session_py import Xform

    pc = PointCloud([Point(1.0, 2.0, 3.0)])
    pc_xf = Xform.translation(10.0, 20.0, 30.0)
    pc2 = pc.transformed(pc_xf)

    MINI_CHECK(TOLERANCE.is_close(pc2.get_point(0)[0], 11.0))
    MINI_CHECK(TOLERANCE.is_close(pc2.get_point(0)[1], 22.0))
    MINI_CHECK(TOLERANCE.is_close(pc2.get_point(0)[2], 33.0))
    MINI_CHECK(TOLERANCE.is_close(pc.get_point(0)[0], 1.0))


@MINI_TEST("PointCloud", "Json Roundtrip")
def test_pointcloud_json_roundtrip():
    from session_py import PointCloud
    from session_py import Point
    from session_py import Vector
    from session_py import Color
    from pathlib import Path

    pc = PointCloud(
        [Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)],
        [Vector(0.0, 0.0, 1.0), Vector(0.0, 0.0, 1.0)],
        [Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)]
    )
    pc.name = "test_pointcloud"

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   file_json_dumps()    │ str          │ to JSON string
    #   file_json_loads(s)   │ str          │ from JSON string
    #   file_json_dump(path) │ file         │ write to file
    #   file_json_load(path) │ file         │ read from file

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_pointcloud.json"
    pc.file_json_dump(fname)
    loaded = PointCloud.file_json_load(fname)

    MINI_CHECK(loaded.name == "test_pointcloud")
    MINI_CHECK(len(loaded) == 2)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[0], 1.0))
    MINI_CHECK(loaded.get_color(0)[0] == 1.0)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_normal(0)[2], 1.0))


@MINI_TEST("PointCloud", "Protobuf Roundtrip")
def test_pointcloud_protobuf_roundtrip():
    from session_py import PointCloud
    from session_py import Point
    from session_py import Vector
    from session_py import Color
    from pathlib import Path

    pc = PointCloud(
        [Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)],
        [Vector(0.0, 0.0, 1.0), Vector(0.0, 0.0, 1.0)],
        [Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)]
    )
    pc.name = "test_pointcloud"

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_pointcloud.bin"
    pc.pb_dump(fname)
    loaded = PointCloud.pb_load(fname)

    MINI_CHECK(loaded.name == "test_pointcloud")
    MINI_CHECK(len(loaded) == 2)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_color(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_normal(0)[2], 1.0))


if __name__ == "__main__":
    run_all("python")
