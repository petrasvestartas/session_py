from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("Plane", "Constructor")
def test_plane_constructor():
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    # Default constructor - XY plane at origin
    pl = Plane()

    # Origin and axes
    origin = pl.origin
    x_axis = pl.x_axis
    y_axis = pl.y_axis
    z_axis = pl.z_axis

    # Plane equation coefficients (ax + by + cz + d = 0)
    a = pl.a
    b = pl.b
    c = pl.c
    d = pl.d

    # Index access for axes
    ax0 = pl[0]
    ax1 = pl[1]
    ax2 = pl[2]

    # Minimal and Full String Representation
    plstr = str(pl)
    plrepr = repr(pl)

    # Copy (duplicates everything except guid)
    plcopy = pl.duplicate()

    # From point and normal
    p = Point(0.0, 0.0, 5.0)
    n = Vector(0.0, 0.0, 1.0)
    pl_pn = Plane.from_point_normal(p, n)

    # From three points
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]
    pl_pts = Plane.from_points(pts)

    # From two points
    p1 = Point(0.0, 0.0, 0.0)
    p2 = Point(1.0, 0.0, 0.0)
    pl_2pts = Plane.from_two_points(p1, p2)

    # Standard planes
    xy = Plane.xy_plane()
    yz = Plane.yz_plane()
    xz = Plane.xz_plane()

    # Translation operators
    offset = Vector(1.0, 2.0, 3.0)

    # In-place add
    pl_iadd = Plane.xy_plane()
    pl_iadd += offset

    # In-place subtract
    pl_isub = Plane.xy_plane()
    pl_isub -= offset

    # Copy add/subtract
    pl_base = Plane.xy_plane()
    pl_add = pl_base + offset
    pl_sub = pl_base - offset

    MINI_CHECK(pl.name == "my_plane" and pl.guid != "")
    MINI_CHECK(TOLERANCE.is_close(origin[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(origin[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(origin[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(x_axis[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(x_axis[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(x_axis[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(y_axis[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(y_axis[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(y_axis[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(z_axis[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(z_axis[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(z_axis[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(a, 0.0))
    MINI_CHECK(TOLERANCE.is_close(b, 0.0))
    MINI_CHECK(TOLERANCE.is_close(c, 1.0))
    MINI_CHECK(TOLERANCE.is_close(d, 0.0))
    MINI_CHECK(TOLERANCE.is_close(ax0[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(ax1[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(ax2[2], 1.0))
    MINI_CHECK(plstr == "0.000000, 0.000000, 0.000000\n1.000000, 0.000000, 0.000000\n0.000000, 1.000000, 0.000000\n0.000000, 0.000000, 1.000000")
    MINI_CHECK(plrepr == "Plane(my_plane, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, Color(blue, 0, 0, 255, 255))")
    MINI_CHECK(plcopy == pl and plcopy.guid != pl.guid)
    MINI_CHECK(TOLERANCE.is_close(pl_pn.origin[2], 5.0))
    MINI_CHECK(TOLERANCE.is_close(pl_pn.z_axis[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl_pn.d, -5.0))
    MINI_CHECK(TOLERANCE.is_close(pl_pts.c, 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl_pts.d, 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl_2pts.origin[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl_2pts.x_axis[0], 1.0))
    MINI_CHECK(xy.name == "xy_plane" and TOLERANCE.is_close(xy.c, 1.0))
    MINI_CHECK(yz.name == "yz_plane" and TOLERANCE.is_close(yz.a, 1.0))
    MINI_CHECK(xz.name == "xz_plane" and TOLERANCE.is_close(xz.b, 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl_iadd.origin[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl_iadd.origin[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pl_iadd.origin[2], 3.0))
    MINI_CHECK(TOLERANCE.is_close(pl_isub.origin[0], -1.0))
    MINI_CHECK(TOLERANCE.is_close(pl_isub.origin[1], -2.0))
    MINI_CHECK(TOLERANCE.is_close(pl_isub.origin[2], -3.0))
    MINI_CHECK(TOLERANCE.is_close(pl_add.origin[2], 3.0))
    MINI_CHECK(TOLERANCE.is_close(pl_base.origin[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl_sub.origin[2], -3.0))


@MINI_TEST("Plane", "Reverse")
def test_plane_reverse():
    from session_py import Plane

    # Reverse flips normal and swaps x/y axes
    pl = Plane.xy_plane()
    pl.reverse()

    MINI_CHECK(TOLERANCE.is_close(pl.x_axis[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl.x_axis[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl.x_axis[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl.y_axis[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl.y_axis[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl.y_axis[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl.c, -1.0))


@MINI_TEST("Plane", "Rotate")
def test_plane_rotate():
    from session_py import Plane

    # Rotate plane 90 degrees around its normal
    pl = Plane.xy_plane()
    pl.rotate(PI / 2.0)

    MINI_CHECK(TOLERANCE.is_close(pl.x_axis[1], 1.0))


@MINI_TEST("Plane", "Is Right Hand")
def test_plane_is_right_hand():
    from session_py import Plane

    # All standard planes should be right-handed
    xy = Plane.xy_plane()
    yz = Plane.yz_plane()
    xz = Plane.xz_plane()
    default_pl = Plane()

    xy_rh = xy.is_right_hand()
    yz_rh = yz.is_right_hand()
    xz_rh = xz.is_right_hand()
    default_rh = default_pl.is_right_hand()

    # After reverse, should still be right-handed
    default_pl.reverse()
    reversed_rh = default_pl.is_right_hand()

    # After rotate, should still be right-handed
    default_pl.rotate(PI / 4.0)
    rotated_rh = default_pl.is_right_hand()

    MINI_CHECK(xy_rh)
    MINI_CHECK(yz_rh)
    MINI_CHECK(xz_rh)
    MINI_CHECK(default_rh)
    MINI_CHECK(reversed_rh)
    MINI_CHECK(rotated_rh)


@MINI_TEST("Plane", "Is Coplanar")
def test_plane_is_coplanar():
    from session_py import Plane
    from session_py import Vector

    # Same direction (parallel planes)
    p1 = Plane.xy_plane()
    p2 = Plane.xy_plane()
    same_dir = Plane.is_same_direction(p1, p2, True)

    # Flipped direction
    p3 = Plane.xy_plane()
    p3.reverse()
    same_dir_flipped = Plane.is_same_direction(p1, p3, True)
    same_dir_strict = Plane.is_same_direction(p1, p3, False)

    # Same position
    p4 = Plane.xy_plane()
    same_pos = Plane.is_same_position(p1, p4)
    p4 += Vector(0.0, 0.0, 1.0)
    diff_pos = Plane.is_same_position(p1, p4)

    # Coplanar
    p5 = Plane.xy_plane()
    p6 = Plane.xy_plane()
    coplanar = Plane.is_coplanar(p5, p6, True)
    p6.reverse()
    coplanar_reversed = Plane.is_coplanar(p5, p6, True)
    p6 += Vector(0.0, 0.0, 1.0)
    not_coplanar = Plane.is_coplanar(p5, p6, True)

    MINI_CHECK(same_dir)
    MINI_CHECK(same_dir_flipped)
    MINI_CHECK(same_dir_strict)  # can_be_flipped=False means opposite normals required; reversed IS opposite
    MINI_CHECK(same_pos)
    MINI_CHECK(not diff_pos)
    MINI_CHECK(coplanar)
    MINI_CHECK(coplanar_reversed)
    MINI_CHECK(not not_coplanar)


@MINI_TEST("Plane", "Transform")
def test_plane_transform():
    from session_py import Plane
    from session_py import Xform

    # Transform - in-place transformation
    pl = Plane.xy_plane()
    pl.xform = Xform.translation(1.0, 2.0, 3.0)
    pl.transform()

    # Transformed - returns new plane
    pl2 = Plane.xy_plane()
    pl2.xform = Xform.translation(1.0, 2.0, 3.0)
    pl3 = pl2.transformed()

    MINI_CHECK(TOLERANCE.is_close(pl.origin[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl.origin[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pl.origin[2], 3.0))
    MINI_CHECK(TOLERANCE.is_close(pl3.origin[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl3.origin[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pl3.origin[2], 3.0))
    MINI_CHECK(TOLERANCE.is_close(pl2.origin[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl2.origin[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl2.origin[2], 0.0))


@MINI_TEST("Plane", "Json Roundtrip")
def test_plane_json_roundtrip():
    from session_py import Plane
    from pathlib import Path

    pl = Plane.xy_plane()
    pl.name = "test_plane"

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   json_dumps()    │ str          │ to JSON string
    #   json_loads(s)   │ str          │ from JSON string
    #   json_dump(path) │ file         │ write to file
    #   json_load(path) │ file         │ read from file

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_plane.json"
    pl.json_dump(fname)
    loaded = Plane.json_load(fname)

    MINI_CHECK(loaded.name == "test_plane")
    MINI_CHECK(TOLERANCE.is_close(loaded.c, 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.d, 0.0))


@MINI_TEST("Plane", "Protobuf Roundtrip")
def test_plane_protobuf_roundtrip():
    from session_py import Plane
    from pathlib import Path

    pl = Plane.xy_plane()
    pl.name = "test_plane"

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    # pb_dump(fname) / pb_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_plane.bin"
    pl.pb_dump(fname)
    loaded = Plane.pb_load(fname)

    MINI_CHECK(loaded.name == "test_plane")
    MINI_CHECK(TOLERANCE.is_close(loaded.c, 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.d, 0.0))


if __name__ == "__main__":
    run_all("python")
