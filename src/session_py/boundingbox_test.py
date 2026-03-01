from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("BoundingBox", "Constructor")
def test_boundingbox_constructor():
    from session_py import BoundingBox
    from session_py import Point
    from session_py import Vector

    # from_point
    bb1 = BoundingBox.from_point(Point(5.0, 5.0, 5.0), 2.0)
    MINI_CHECK(TOLERANCE.is_close(bb1.center[0], 5.0))
    MINI_CHECK(TOLERANCE.is_close(bb1.half_size[0], 2.0))

    # from_points (AABB)
    pts = [Point(0.0, 0.0, 0.0), Point(2.0, 3.0, 4.0)]
    bb2 = BoundingBox.from_points(pts)
    mn = bb2.min_point()
    mx = bb2.max_point()
    MINI_CHECK(TOLERANCE.is_close(mn[0], 0.0) and TOLERANCE.is_close(mn[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(mx[0], 2.0) and TOLERANCE.is_close(mx[2], 4.0))

    # OBB constructor
    obb = BoundingBox(
        center=Point(0.0, 0.0, 0.0),
        x_axis=Vector(1.0, 0.0, 0.0),
        y_axis=Vector(0.0, 1.0, 0.0),
        z_axis=Vector(0.0, 0.0, 1.0),
        half_size=Vector(1.0, 2.0, 3.0)
    )
    MINI_CHECK(TOLERANCE.is_close(obb.half_size[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(obb.half_size[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(obb.half_size[2], 3.0))

    # aabb
    bb_aabb = bb2.aabb()
    MINI_CHECK(TOLERANCE.is_close(bb_aabb.min_point()[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(bb_aabb.max_point()[2], 4.0))

    # corners
    corners = bb2.corners()
    MINI_CHECK(len(corners) == 8)

    # point_at: center + x*x_axis + y*y_axis + z*z_axis (raw OBB offsets)
    p_center = bb2.point_at(0.0, 0.0, 0.0)
    hx, hy, hz = bb2.half_size[0], bb2.half_size[1], bb2.half_size[2]
    p_max_pt = bb2.point_at(hx, hy, hz)
    MINI_CHECK(TOLERANCE.is_close(p_center[0], 1.0) and TOLERANCE.is_close(p_center[2], 2.0))
    MINI_CHECK(TOLERANCE.is_close(p_max_pt[0], 2.0) and TOLERANCE.is_close(p_max_pt[2], 4.0))

    # inflate
    bb3 = BoundingBox.from_points([Point(0.0, 0.0, 0.0), Point(2.0, 2.0, 2.0)])
    bb3.inflate(1.0)
    MINI_CHECK(TOLERANCE.is_close(bb3.min_point()[0], -1.0))
    MINI_CHECK(TOLERANCE.is_close(bb3.max_point()[0], 3.0))

    # guid and name
    MINI_CHECK(bb1.guid)
    bb1.name = "test_bbox"
    MINI_CHECK(bb1.name == "test_bbox")


@MINI_TEST("BoundingBox", "Collision")
def test_boundingbox_collision():
    from session_py import BoundingBox
    from session_py import Point

    bb1 = BoundingBox.from_point(Point(0.0, 0.0, 0.0), 1.0)
    bb2 = BoundingBox.from_point(Point(1.5, 0.0, 0.0), 1.0)
    bb3 = BoundingBox.from_point(Point(5.0, 5.0, 5.0), 0.5)

    MINI_CHECK(bb1.collides_with(bb2))
    MINI_CHECK(not bb1.collides_with(bb3))


@MINI_TEST("BoundingBox", "Transformation")
def test_boundingbox_transformation():
    from session_py import BoundingBox
    from session_py import Point
    from session_py import Xform

    pts = [Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 0.0)]
    bb = BoundingBox.from_points(pts)
    bb.xform = Xform.translation(0.0, 0.0, 5.0)

    bbt = bb.transformed()
    MINI_CHECK(TOLERANCE.is_close(bbt.center[2], 5.0))

    bb.transform()
    MINI_CHECK(bb.xform == Xform.identity())
    MINI_CHECK(TOLERANCE.is_close(bb.center[2], 5.0))


@MINI_TEST("BoundingBox", "Json Roundtrip")
def test_boundingbox_json_roundtrip():
    from session_py import BoundingBox
    from session_py import Point
    from pathlib import Path

    bb = BoundingBox.from_point(Point(1.0, 2.0, 3.0), 5.0)
    bb.name = "test_bbox"

    # JSON object
    d = bb.__jsondump__()
    loaded_j = BoundingBox.__jsonload__(d)
    MINI_CHECK(loaded_j.name == "test_bbox")
    MINI_CHECK(TOLERANCE.is_close(loaded_j.center[0], 1.0))

    # String
    s = bb.json_dumps()
    loaded_s = BoundingBox.json_loads(s)
    MINI_CHECK(loaded_s.name == "test_bbox")
    MINI_CHECK(TOLERANCE.is_close(loaded_s.half_size[0], 5.0))

    # File
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_boundingbox.json"
    bb.json_dump(fname)
    loaded = BoundingBox.json_load(fname)
    MINI_CHECK(loaded.name == "test_bbox")
    MINI_CHECK(TOLERANCE.is_close(loaded.center[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.half_size[0], 5.0))


@MINI_TEST("BoundingBox", "Protobuf Roundtrip")
def test_boundingbox_protobuf_roundtrip():
    from session_py import BoundingBox
    from session_py import Point
    from pathlib import Path

    bb = BoundingBox.from_point(Point(1.0, 2.0, 3.0), 5.0)
    bb.name = "test_bbox_proto"

    # Bytes
    b = bb.pb_dumps()
    loaded_s = BoundingBox.pb_loads(b)
    MINI_CHECK(loaded_s.name == "test_bbox_proto")
    MINI_CHECK(loaded_s.guid == bb.guid)
    MINI_CHECK(TOLERANCE.is_close(loaded_s.center[0], 1.0))

    # File
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_boundingbox.bin"
    bb.pb_dump(fname)
    loaded = BoundingBox.pb_load(fname)
    MINI_CHECK(loaded.name == "test_bbox_proto")
    MINI_CHECK(loaded.guid == bb.guid)
    MINI_CHECK(TOLERANCE.is_close(loaded.center[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.half_size[0], 5.0))


if __name__ == "__main__":
    run_all(language="python")
