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
    pl_iadd = Plane.xy_plane()
    pl_iadd += offset
    pl_isub = Plane.xy_plane()
    pl_isub -= offset
    pl_base = Plane.xy_plane()
    pl_add = pl_base + offset
    pl_sub = pl_base - offset

    MINI_CHECK(pl.name == "my_plane" and pl.guid != "")
    MINI_CHECK(TOLERANCE.is_close(origin[0], 0.0) and TOLERANCE.is_close(origin[1], 0.0) and TOLERANCE.is_close(origin[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(x_axis[0], 1.0) and TOLERANCE.is_close(y_axis[1], 1.0) and TOLERANCE.is_close(z_axis[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(a, 0.0) and TOLERANCE.is_close(b, 0.0) and TOLERANCE.is_close(c, 1.0) and TOLERANCE.is_close(d, 0.0))
    MINI_CHECK(TOLERANCE.is_close(ax0[0], 1.0) and TOLERANCE.is_close(ax1[1], 1.0) and TOLERANCE.is_close(ax2[2], 1.0))
    MINI_CHECK(plstr == "0.000000, 0.000000, 0.000000\n1.000000, 0.000000, 0.000000\n0.000000, 1.000000, 0.000000\n0.000000, 0.000000, 1.000000")
    MINI_CHECK(plrepr == "Plane(my_plane, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, Color(blue, 0.0, 0.0, 1.0, 1.0))")
    MINI_CHECK(plcopy == pl and plcopy.guid != pl.guid)
    MINI_CHECK(TOLERANCE.is_close(pl_pn.origin[2], 5.0) and TOLERANCE.is_close(pl_pn.z_axis[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl_pts.c, 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl_2pts.x_axis[0], 1.0))
    MINI_CHECK(xy.name == "xy_plane" and yz.name == "yz_plane" and xz.name == "xz_plane")
    MINI_CHECK(TOLERANCE.is_close(pl_iadd.origin[0], 1.0) and TOLERANCE.is_close(pl_iadd.origin[1], 2.0) and TOLERANCE.is_close(pl_iadd.origin[2], 3.0))
    MINI_CHECK(TOLERANCE.is_close(pl_isub.origin[0], -1.0) and TOLERANCE.is_close(pl_isub.origin[2], -3.0))
    MINI_CHECK(TOLERANCE.is_close(pl_add.origin[2], 3.0))
    MINI_CHECK(TOLERANCE.is_close(pl_sub.origin[2], -3.0))


@MINI_TEST("Plane", "Is Valid")
def test_plane_is_valid():
    from session_py import Plane

    pl = Plane.xy_plane()
    invalid = Plane.invalid()

    MINI_CHECK(pl.is_valid())
    MINI_CHECK(not invalid.is_valid())


@MINI_TEST("Plane", "Reverse")
def test_plane_reverse():
    from session_py import Plane

    pl = Plane.xy_plane()
    pl.reverse()

    MINI_CHECK(TOLERANCE.is_close(pl.x_axis[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl.x_axis[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl.y_axis[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl.y_axis[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pl.c, -1.0))


@MINI_TEST("Plane", "Rotate")
def test_plane_rotate():
    from session_py import Plane

    pl = Plane.xy_plane()
    pl.rotate(PI / 2.0)

    MINI_CHECK(TOLERANCE.is_close(pl.x_axis[1], 1.0))


@MINI_TEST("Plane", "Is Right Hand")
def test_plane_is_right_hand():
    from session_py import Plane

    xy = Plane.xy_plane()
    yz = Plane.yz_plane()
    xz = Plane.xz_plane()

    MINI_CHECK(xy.is_right_hand())
    MINI_CHECK(yz.is_right_hand())
    MINI_CHECK(xz.is_right_hand())


@MINI_TEST("Plane", "Is Same Direction")
def test_plane_is_same_direction():
    from session_py import Plane

    p1 = Plane.xy_plane()
    p2 = Plane.xy_plane()
    p3 = Plane.xy_plane()
    p3.reverse()

    MINI_CHECK(Plane.is_same_direction(p1, p2, True))
    MINI_CHECK(Plane.is_same_direction(p1, p3, True))
    MINI_CHECK(Plane.is_same_direction(p1, p3, False))


@MINI_TEST("Plane", "Is Same Position")
def test_plane_is_same_position():
    from session_py import Plane
    from session_py import Vector

    p1 = Plane.xy_plane()
    p2 = Plane.xy_plane()
    p2 += Vector(0.0, 0.0, 1.0)

    MINI_CHECK(Plane.is_same_position(p1, Plane.xy_plane()))
    MINI_CHECK(not Plane.is_same_position(p1, p2))


@MINI_TEST("Plane", "Is Coplanar")
def test_plane_is_coplanar():
    from session_py import Plane
    from session_py import Vector

    p1 = Plane.xy_plane()
    p2 = Plane.xy_plane()
    p3 = Plane.xy_plane()
    p3 += Vector(0.0, 0.0, 1.0)

    MINI_CHECK(Plane.is_coplanar(p1, p2, True))
    MINI_CHECK(not Plane.is_coplanar(p1, p3, True))


@MINI_TEST("Plane", "Translate By Normal")
def test_plane_translate_by_normal():
    from session_py import Plane

    pl = Plane.xy_plane()
    moved = pl.translate_by_normal(5.0)

    MINI_CHECK(TOLERANCE.is_close(moved.origin[2], 5.0))
    MINI_CHECK(TOLERANCE.is_close(pl.origin[2], 0.0))


@MINI_TEST("Plane", "Base1 Base2")
def test_plane_base1_base2():
    from session_py import Plane

    xy = Plane.xy_plane()
    b1 = xy.base1()
    b2 = xy.base2()
    MINI_CHECK(TOLERANCE.is_close(abs(b1.dot(xy.z_axis)), 0.0))
    MINI_CHECK(TOLERANCE.is_close(abs(b2.dot(xy.z_axis)), 0.0))
    MINI_CHECK(TOLERANCE.is_close(b1.dot(b2), 0.0))
    MINI_CHECK(TOLERANCE.is_close(b1.magnitude(), 1.0))
    MINI_CHECK(TOLERANCE.is_close(b2.magnitude(), 1.0))


@MINI_TEST("Plane", "Transform")
def test_plane_transform():
    from session_py import Plane
    from session_py import Xform

    pl = Plane.xy_plane()
    pl_xf = Xform.translation(1.0, 2.0, 3.0)
    pl.transform(pl_xf)

    MINI_CHECK(TOLERANCE.is_close(pl.origin[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl.origin[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pl.origin[2], 3.0))


@MINI_TEST("Plane", "Transformed")
def test_plane_transformed():
    from session_py import Plane
    from session_py import Xform

    pl = Plane.xy_plane()
    pl_xf = Xform.translation(1.0, 2.0, 3.0)
    pl2 = pl.transformed(pl_xf)

    MINI_CHECK(TOLERANCE.is_close(pl2.origin[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pl2.origin[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pl2.origin[2], 3.0))
    MINI_CHECK(TOLERANCE.is_close(pl.origin[0], 0.0))


@MINI_TEST("Plane", "Json Roundtrip")
def test_plane_json_roundtrip():
    from session_py import Plane
    from pathlib import Path

    pl = Plane.xy_plane()
    pl.name = "test_plane"

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   file_json_dumps()    │ str          │ to JSON string
    #   file_json_loads(s)   │ str          │ from JSON string
    #   file_json_dump(path) │ file         │ write to file
    #   file_json_load(path) │ file         │ read from file

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_plane.json"
    pl.file_json_dump(fname)
    loaded = Plane.file_json_load(fname)

    MINI_CHECK(loaded.name == "test_plane")
    MINI_CHECK(TOLERANCE.is_close(loaded.c, 1.0))


@MINI_TEST("Plane", "Protobuf Roundtrip")
def test_plane_protobuf_roundtrip():
    from session_py import Plane
    from pathlib import Path

    pl = Plane.xy_plane()
    pl.name = "test_plane"

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_plane.bin"
    pl.pb_dump(fname)
    loaded = Plane.pb_load(fname)

    MINI_CHECK(loaded.name == "test_plane")
    MINI_CHECK(TOLERANCE.is_close(loaded.c, 1.0))


@MINI_TEST("Plane", "Has On Negative Side")
def test_plane_has_on_negative_side():
    from session_py import Plane
    from session_py import Point
    pl = Plane.xy_plane()
    above = Point(0.0, 0.0, 1.0)
    below = Point(0.0, 0.0, -1.0)
    MINI_CHECK(pl.has_on_negative_side(below))
    MINI_CHECK(not pl.has_on_negative_side(above))


if __name__ == "__main__":
    run_all("python")
