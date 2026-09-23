from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("PointCloud", "Constructor")
def test_pointcloud_constructor():
    from session_py import Color
    from session_py import Point
    from session_py import PointCloud
    from session_py import Vector

    pc0 = PointCloud()

    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(1.0, 0.0, 0.0)
    p2 = Point(0.0, 1.0, 0.0)
    n0 = Vector(0.0, 0.0, 1.0)
    n1 = Vector(0.0, 0.0, 1.0)
    n2 = Vector(0.0, 0.0, 1.0)
    c0 = Color(1.0, 0.0, 0.0, 1.0)
    c1 = Color(0.0, 1.0, 0.0, 1.0)
    c2 = Color(0.0, 0.0, 1.0, 1.0)
    pc = PointCloud([p0, p1, p2], [n0, n1, n2], [c0, c1, c2])

    pcstr = str(pc)
    pcrepr = repr(pc)

    pccopy = pc.duplicate()
    pcother = PointCloud()

    offset = Vector(10.0, 20.0, 30.0)
    pc3 = PointCloud([Point(1.0, 2.0, 3.0)], [], [])

    pc_iadd = pc3.duplicate()
    pc_iadd += offset

    pc_isub = pc3.duplicate()
    pc_isub -= offset

    pc_add = pc3 + offset
    pc_sub = pc3 - offset

    MINI_CHECK(pc0.name == "my_pointcloud")
    MINI_CHECK(pc0.guid != "")
    MINI_CHECK(pc0.is_empty())
    MINI_CHECK(len(pc) == 3)
    MINI_CHECK(pcstr == "3 points")
    MINI_CHECK(pcrepr == "PointCloud(my_pointcloud, 3 points, 3 colors, 3 normals)")
    MINI_CHECK(pccopy == pc and pccopy.guid != pc.guid)
    MINI_CHECK(pcother != pc)
    MINI_CHECK(pc_iadd.get_point(0) == Point(11.0, 22.0, 33.0))
    MINI_CHECK(pc_isub.get_point(0) == Point(-9.0, -18.0, -27.0))
    MINI_CHECK(pc_add.get_point(0) == Point(11.0, 22.0, 33.0))
    MINI_CHECK(pc_sub.get_point(0) == Point(-9.0, -18.0, -27.0))


@MINI_TEST("PointCloud", "From Coords")
def test_pointcloud_from_coords():
    from session_py import Color
    from session_py import Point
    from session_py import PointCloud
    from session_py import Vector

    coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    colors = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]
    normals = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    pc = PointCloud.from_coords(coords, colors, normals)

    MINI_CHECK(len(pc) == 3 and pc.color_count() == 3 and pc.normal_count() == 3)
    MINI_CHECK(pc.get_point(1) == Point(1.0, 0.0, 0.0))
    MINI_CHECK(pc.get_color(1) == Color(0.0, 1.0, 0.0, 1.0))
    MINI_CHECK(pc.get_normal(1) == Vector(0.0, 0.0, 1.0))


@MINI_TEST("PointCloud", "Transform")
def test_pointcloud_transform():
    from session_py import Point
    from session_py import PointCloud
    from session_py import Vector
    from session_py import Xform

    pc = PointCloud([Point(1.0, 2.0, 3.0)], [Vector(1.0, 0.0, 0.0)], [])
    xform = Xform.translation(10.0, 20.0, 30.0)
    pc.transform(xform)

    MINI_CHECK(pc.get_point(0) == Point(11.0, 22.0, 33.0))
    MINI_CHECK(pc.get_normal(0) == Vector(1.0, 0.0, 0.0))


@MINI_TEST("PointCloud", "Transformed")
def test_pointcloud_transformed():
    from session_py import Point
    from session_py import PointCloud
    from session_py import Xform

    pc = PointCloud([Point(1.0, 2.0, 3.0)], [], [])
    xform = Xform.translation(10.0, 20.0, 30.0)
    moved = pc.transformed(xform)

    MINI_CHECK(moved.get_point(0) == Point(11.0, 22.0, 33.0))
    MINI_CHECK(pc.get_point(0) == Point(1.0, 2.0, 3.0))


@MINI_TEST("PointCloud", "Point Count")
def test_pointcloud_point_count():
    from session_py import Point
    from session_py import PointCloud

    pc = PointCloud(
        [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.0, 1.0, 0.0)], [], []
    )

    MINI_CHECK(pc.point_count() == 3)


@MINI_TEST("PointCloud", "Len")
def test_pointcloud_len():
    from session_py import Point
    from session_py import PointCloud

    pc = PointCloud([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)], [], [])

    MINI_CHECK(len(pc) == 2)


@MINI_TEST("PointCloud", "Is Empty")
def test_pointcloud_is_empty():
    from session_py import Point
    from session_py import PointCloud

    pc0 = PointCloud()
    pc1 = PointCloud([Point(0.0, 0.0, 0.0)], [], [])

    MINI_CHECK(pc0.is_empty())
    MINI_CHECK(not pc1.is_empty())


@MINI_TEST("PointCloud", "Get Point")
def test_pointcloud_get_point():
    from session_py import Point
    from session_py import PointCloud

    pc = PointCloud([Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)], [], [])
    point = pc.get_point(1)

    MINI_CHECK(point == Point(4.0, 5.0, 6.0))


@MINI_TEST("PointCloud", "Set Point")
def test_pointcloud_set_point():
    from session_py import Point
    from session_py import PointCloud

    pc = PointCloud([Point(0.0, 0.0, 0.0)], [], [])
    pc.set_point(0, Point(4.0, 5.0, 6.0))

    MINI_CHECK(pc.get_point(0) == Point(4.0, 5.0, 6.0))


@MINI_TEST("PointCloud", "Add Point")
def test_pointcloud_add_point():
    from session_py import Point
    from session_py import PointCloud

    pc = PointCloud()
    pc.add_point(Point(1.0, 2.0, 3.0))

    MINI_CHECK(len(pc) == 1)
    MINI_CHECK(pc.get_point(0) == Point(1.0, 2.0, 3.0))


@MINI_TEST("PointCloud", "Get Points")
def test_pointcloud_get_points():
    from session_py import Point
    from session_py import PointCloud

    pc = PointCloud([Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)], [], [])
    points = pc.get_points()

    MINI_CHECK(len(points) == 2)
    MINI_CHECK(points[0] == Point(1.0, 2.0, 3.0))
    MINI_CHECK(points[1] == Point(4.0, 5.0, 6.0))


@MINI_TEST("PointCloud", "Coords")
def test_pointcloud_coords():
    from session_py import Point
    from session_py import PointCloud

    pc = PointCloud([Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)], [], [])
    coords = pc.coords()

    MINI_CHECK(len(coords) == 6)
    MINI_CHECK(coords[0] == 1.0 and coords[5] == 6.0)


@MINI_TEST("PointCloud", "Color Count")
def test_pointcloud_color_count():
    from session_py import Color
    from session_py import PointCloud

    pc = PointCloud([], [], [Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)])

    MINI_CHECK(pc.color_count() == 2)


@MINI_TEST("PointCloud", "Get Color")
def test_pointcloud_get_color():
    from session_py import Color
    from session_py import PointCloud

    pc = PointCloud([], [], [Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)])
    color = pc.get_color(1)

    MINI_CHECK(color == Color(0.0, 1.0, 0.0, 1.0))


@MINI_TEST("PointCloud", "Set Color")
def test_pointcloud_set_color():
    from session_py import Color
    from session_py import PointCloud

    pc = PointCloud([], [], [Color(0.0, 0.0, 0.0, 0.0)])
    pc.set_color(0, Color(1.0, 0.0, 0.0, 1.0))

    MINI_CHECK(pc.get_color(0) == Color(1.0, 0.0, 0.0, 1.0))


@MINI_TEST("PointCloud", "Add Color")
def test_pointcloud_add_color():
    from session_py import Color
    from session_py import PointCloud

    pc = PointCloud()
    pc.add_color(Color(1.0, 0.0, 1.0, 1.0))

    MINI_CHECK(pc.color_count() == 1)
    MINI_CHECK(pc.get_color(0) == Color(1.0, 0.0, 1.0, 1.0))


@MINI_TEST("PointCloud", "Get Colors")
def test_pointcloud_get_colors():
    from session_py import Color
    from session_py import PointCloud

    pc = PointCloud([], [], [Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)])
    colors = pc.get_colors()

    MINI_CHECK(len(colors) == 2)
    MINI_CHECK(colors[0] == Color(1.0, 0.0, 0.0, 1.0))
    MINI_CHECK(colors[1] == Color(0.0, 1.0, 0.0, 1.0))


@MINI_TEST("PointCloud", "Colors")
def test_pointcloud_colors():
    from session_py import Color
    from session_py import PointCloud

    pc = PointCloud([], [], [Color(1.0, 0.0, 0.0, 1.0)])
    colors = pc.colors()

    MINI_CHECK(len(colors) == 4)
    MINI_CHECK(
        colors[0] == 255 and colors[1] == 0 and colors[2] == 0 and colors[3] == 255
    )


@MINI_TEST("PointCloud", "Normal Count")
def test_pointcloud_normal_count():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud([], [Vector(0.0, 0.0, 1.0), Vector(0.0, 0.0, 1.0)], [])

    MINI_CHECK(pc.normal_count() == 2)


@MINI_TEST("PointCloud", "Get Normal")
def test_pointcloud_get_normal():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud([], [Vector(0.0, 0.0, 1.0), Vector(1.0, 0.0, 0.0)], [])
    normal = pc.get_normal(1)

    MINI_CHECK(normal == Vector(1.0, 0.0, 0.0))


@MINI_TEST("PointCloud", "Set Normal")
def test_pointcloud_set_normal():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud([], [Vector(0.0, 0.0, 1.0)], [])
    pc.set_normal(0, Vector(0.0, 1.0, 0.0))

    MINI_CHECK(pc.get_normal(0) == Vector(0.0, 1.0, 0.0))


@MINI_TEST("PointCloud", "Add Normal")
def test_pointcloud_add_normal():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud()
    pc.add_normal(Vector(1.0, 0.0, 0.0))

    MINI_CHECK(pc.normal_count() == 1)
    MINI_CHECK(pc.get_normal(0) == Vector(1.0, 0.0, 0.0))


@MINI_TEST("PointCloud", "Get Normals")
def test_pointcloud_get_normals():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud([], [Vector(0.0, 0.0, 1.0), Vector(1.0, 0.0, 0.0)], [])
    normals = pc.get_normals()

    MINI_CHECK(len(normals) == 2)
    MINI_CHECK(normals[0] == Vector(0.0, 0.0, 1.0))
    MINI_CHECK(normals[1] == Vector(1.0, 0.0, 0.0))


@MINI_TEST("PointCloud", "Normals")
def test_pointcloud_normals():
    from session_py import PointCloud
    from session_py import Vector

    pc = PointCloud([], [Vector(0.0, 0.0, 1.0), Vector(1.0, 0.0, 0.0)], [])
    normals = pc.normals()

    MINI_CHECK(len(normals) == 6)
    MINI_CHECK(normals[2] == 1.0 and normals[3] == 1.0)


@MINI_TEST("PointCloud", "Build Lod")
def test_pointcloud_build_lod():
    from session_py import Point
    from session_py import PointCloud

    coords = [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    pc = PointCloud.from_coords(coords, [], [])
    pc.build_lod(1.0, 2)

    cube = pc.lod_cube(0)
    span = pc.lod_range(0)
    children = pc.lod_children(0)

    MINI_CHECK(pc.has_lod())
    MINI_CHECK(pc.lod_node_count() == 8)
    MINI_CHECK(cube[0] == Point(0.5, 0.5, 0.5) and cube[1] == 1.0)
    MINI_CHECK(pc.lod_spacing(0) == 1.0 and pc.lod_level(1) == 1)
    MINI_CHECK(span[0] == 0 and span[1] == 1)
    MINI_CHECK(children[0] == 1 and children[7] == -1)
    MINI_CHECK(len(pc.coords()) == 24)


@MINI_TEST("PointCloud", "Point Ids")
def test_pointcloud_point_ids():
    from session_py import PointCloud

    coords = [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    pc = PointCloud.from_coords(coords, [], [])
    before = pc.get_point(5)
    pc.build_lod(1.0, 2)

    index = pc.index_of_id(5)

    MINI_CHECK(len(pc.point_ids()) == 8)
    MINI_CHECK(index >= 0)
    MINI_CHECK(pc.point_id(index) == 5)
    MINI_CHECK(pc.get_point(index) == before)


@MINI_TEST("PointCloud", "Json Roundtrip")
def test_pointcloud_json_roundtrip():
    from session_py import Color
    from session_py import Point
    from session_py import PointCloud
    from session_py import Vector
    from pathlib import Path

    pc = PointCloud(
        [Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)],
        [Vector(0.0, 0.0, 1.0), Vector(0.0, 0.0, 1.0)],
        [Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)],
    )
    pc.name = "test_pointcloud"

    guid = pc.guid
    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_pointcloud.json"
    )
    pc.file_json_dump(filename)

    loaded = PointCloud.file_json_load(filename)
    parsed = PointCloud.file_json_loads(pc.file_json_dumps())

    MINI_CHECK(loaded == pc)
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(parsed == pc)
    MINI_CHECK(parsed.guid == guid)


@MINI_TEST("PointCloud", "Protobuf Roundtrip")
def test_pointcloud_protobuf_roundtrip():
    from session_py import Color
    from session_py import Point
    from session_py import PointCloud
    from session_py import Vector
    from pathlib import Path

    fresh = PointCloud()
    fresh_proto = fresh.to_proto()
    pc = PointCloud(
        [Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)],
        [Vector(0.0, 0.0, 1.0), Vector(0.0, 0.0, 1.0)],
        [Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)],
    )
    pc.name = "test_pointcloud"

    guid = pc.guid
    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_pointcloud.bin"
    )
    pc.pb_dump(filename)

    loaded = PointCloud.pb_load(filename)
    parsed = PointCloud.pb_loads(pc.pb_dumps())
    converted = PointCloud.from_proto(pc.to_proto())

    MINI_CHECK(not fresh.has_guid())
    MINI_CHECK(fresh_proto.guid == "")
    MINI_CHECK(loaded == pc)
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(parsed == pc)
    MINI_CHECK(parsed.guid == guid)
    MINI_CHECK(converted == pc)
    MINI_CHECK(converted.guid == guid)


if __name__ == "__main__":
    run_all("python")
