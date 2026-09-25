from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Point", "Constructor")
def test_point_constructor():
    from session_py import Color
    from session_py import Point
    from session_py import Vector

    p = Point(1.0, 2.0, 3.0)

    p[0] = 10.0
    p[1] = 20.0
    p[2] = 30.0

    x = p[0]
    y = p[1]
    z = p[2]

    pstr = str(p)
    prepr = repr(p)

    pcopy = p.duplicate()
    pother = Point(1.0, 2.0, 3.0)

    pmult = p.duplicate()
    pmult *= 2.0

    pdiv = p.duplicate()
    pdiv /= 2.0

    padd = p.duplicate()
    padd += Vector(1.0, 1.0, 1.0)

    psub = p.duplicate()
    psub -= Vector(1.0, 1.0, 1.0)

    result_mul = p * 2.0
    result_div = p / 2.0
    result_add = p + Vector(1.0, 1.0, 1.0)
    result_sub = p - Vector(1.0, 1.0, 1.0)
    result_diff = p - pother

    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(4.0, 5.0, 6.0)
    psum = Point.sum(p1, p2)
    pdif = p2 - p1

    pguid = Point(1.0, 2.0, 3.0)
    minted = pguid.guid
    pguid.guid = "custom_guid"

    MINI_CHECK(p.name == "my_point")
    MINI_CHECK(p[0] == 10.0 and p[1] == 20.0 and p[2] == 30.0)
    MINI_CHECK(p.width == 1.0)
    MINI_CHECK(p.pointcolor == Color.black())
    MINI_CHECK(p.guid != "")
    MINI_CHECK(x == 10.0 and y == 20.0 and z == 30.0)
    MINI_CHECK(pstr == "10.000000, 20.000000, 30.000000")
    MINI_CHECK(prepr == "Point(my_point, 10.000000, 20.000000, 30.000000, Color(black, 0.0, 0.0, 0.0, 1.0), 1.000000)")
    MINI_CHECK(pcopy == p and pcopy.guid != p.guid)
    MINI_CHECK(pother != p)
    MINI_CHECK(pmult[0] == 20.0 and pmult[1] == 40.0 and pmult[2] == 60.0)
    MINI_CHECK(pdiv[0] == 5.0 and pdiv[1] == 10.0 and pdiv[2] == 15.0)
    MINI_CHECK(padd[0] == 11.0 and padd[1] == 21.0 and padd[2] == 31.0)
    MINI_CHECK(psub[0] == 9.0 and psub[1] == 19.0 and psub[2] == 29.0)
    MINI_CHECK(result_mul[0] == 20.0 and result_mul[1] == 40.0 and result_mul[2] == 60.0)
    MINI_CHECK(result_div[0] == 5.0 and result_div[1] == 10.0 and result_div[2] == 15.0)
    MINI_CHECK(result_add[0] == 11.0 and result_add[1] == 21.0 and result_add[2] == 31.0)
    MINI_CHECK(result_sub[0] == 9.0 and result_sub[1] == 19.0 and result_sub[2] == 29.0)
    MINI_CHECK(result_diff[0] == 9.0 and result_diff[1] == 18.0 and result_diff[2] == 27.0)
    MINI_CHECK(psum[0] == 5.0 and psum[1] == 7.0 and psum[2] == 9.0)
    MINI_CHECK(pdif[0] == 3.0 and pdif[1] == 3.0 and pdif[2] == 3.0)
    MINI_CHECK(pguid.guid != minted and pguid.guid == "custom_guid")


@MINI_TEST("Point", "Transformation")
def test_point_transformation():
    from session_py import Point
    from session_py import Xform

    p = Point(1.0, 2.0, 3.0)
    xform = Xform.translation(1.0, 2.0, 3.0)
    moved = p.transformed(xform)
    p.transform(xform)

    MINI_CHECK(moved[0] == 2.0 and moved[1] == 4.0 and moved[2] == 6.0)
    MINI_CHECK(p[0] == 2.0 and p[1] == 4.0 and p[2] == 6.0)


@MINI_TEST("Point", "Json Roundtrip")
def test_point_json_roundtrip():
    from pathlib import Path
    from session_py import Color
    from session_py import Point

    p = Point(1.5, 2.5, 3.5, "test_point")
    p.width = 2.0
    p.pointcolor = Color(1.0, 0.5, 0.25, 1.0)

    guid = p.guid
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_point.json"
    p.file_json_dump(filename)

    loaded = Point.file_json_load(filename)
    parsed = Point.file_json_loads(p.file_json_dumps())

    MINI_CHECK(loaded.name == "test_point")
    MINI_CHECK(loaded[0] == 1.5 and loaded[1] == 2.5 and loaded[2] == 3.5)
    MINI_CHECK(loaded.width == 2.0)
    MINI_CHECK(loaded.pointcolor[0] == 1.0)
    MINI_CHECK(loaded.pointcolor[1] == 0.5)
    MINI_CHECK(loaded.pointcolor[2] == 0.25)
    MINI_CHECK(loaded.pointcolor[3] == 1.0)
    MINI_CHECK(parsed == p)
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(parsed.guid == guid)


@MINI_TEST("Point", "Protobuf Roundtrip")
def test_point_protobuf_roundtrip():
    from pathlib import Path
    from session_py import Color
    from session_py import Point

    fresh = Point()
    fresh_proto = fresh.to_proto()
    p = Point(1.5, 2.5, 3.5, "test_point")
    p.width = 2.0
    p.pointcolor = Color(1.0, 0.5, 0.25, 1.0)

    guid = p.guid
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_point.bin"
    p.pb_dump(filename)

    loaded = Point.pb_load(filename)
    parsed = Point.pb_loads(p.pb_dumps())
    converted = Point.from_proto(p.to_proto())

    MINI_CHECK(not fresh.has_guid())
    MINI_CHECK(fresh_proto.guid == "")
    MINI_CHECK(loaded.name == "test_point")
    MINI_CHECK(loaded[0] == 1.5 and loaded[1] == 2.5 and loaded[2] == 3.5)
    MINI_CHECK(loaded.width == 2.0)
    MINI_CHECK(loaded.pointcolor[0] == 1.0)
    MINI_CHECK(loaded.pointcolor[1] == 0.5)
    MINI_CHECK(loaded.pointcolor[2] == 0.25)
    MINI_CHECK(loaded.pointcolor[3] == 1.0)
    MINI_CHECK(parsed == p)
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(parsed.guid == guid)
    MINI_CHECK(converted == p)
    MINI_CHECK(converted.guid == guid)


@MINI_TEST("Point", "Is Ccw")
def test_point_is_ccw():
    from session_py import Point

    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(1.0, 0.0, 0.0)
    p2 = Point(0.05, 1.0, 0.0)
    ccw = Point.is_ccw(p0, p1, p2)
    cw = Point.is_ccw(p2, p1, p0)

    MINI_CHECK(ccw)
    MINI_CHECK(not cw)


@MINI_TEST("Point", "Mid Point")
def test_point_mid_point():
    from session_py import Point

    p0 = Point(0.0, 2.0, 1.0)
    p1 = Point(1.0, 5.0, 3.0)
    mid = Point.mid_point(p0, p1)

    MINI_CHECK(mid[0] == 0.5 and mid[1] == 3.5 and mid[2] == 2.0)


@MINI_TEST("Point", "Distance")
def test_point_distance():
    from session_py import Point

    p0 = Point(0.0, 2.0, 1.0)
    p1 = Point(1.0, 5.0, 3.0)
    d = Point.distance(p0, p1)

    MINI_CHECK(TOLERANCE.is_close(d, 3.741657))


@MINI_TEST("Point", "Squared Distance")
def test_point_squared_distance():
    from session_py import Point

    p0 = Point(0.0, 2.0, 1.0)
    p1 = Point(1.0, 5.0, 3.0)
    d = Point.squared_distance(p0, p1)

    MINI_CHECK(TOLERANCE.is_close(d, 14.0))


@MINI_TEST("Point", "Interpolate")
def test_point_interpolate():
    from session_py import Point

    a = Point(0.0, 0.0, 0.0)
    b = Point(4.0, 8.0, 12.0)
    half = Point.lerp(a, b, 0.5)
    inner = Point.interpolate(a, b, 3)
    both = Point.interpolate(a, b, 3, 1)
    start = Point.interpolate(a, b, 3, 2)

    MINI_CHECK(half[0] == 2.0 and half[1] == 4.0 and half[2] == 6.0)
    MINI_CHECK(len(inner) == 3)
    MINI_CHECK(inner[0][0] == 1.0 and inner[1][0] == 2.0 and inner[2][0] == 3.0)
    MINI_CHECK(len(both) == 5)
    MINI_CHECK(both[0][0] == 0.0 and both[4][0] == 4.0)
    MINI_CHECK(len(start) == 4)
    MINI_CHECK(start[0][0] == 0.0 and start[3][0] == 3.0)


@MINI_TEST("Point", "Area")
def test_point_area():
    from session_py import Point

    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(2.0, 0.0, 0.0)
    p2 = Point(2.0, 2.0, 0.0)
    p3 = Point(0.0, 2.0, 0.0)
    area = Point.area([p0, p1, p2, p3])

    MINI_CHECK(area == 4.0)


@MINI_TEST("Point", "Centroid Quad")
def test_point_centroid_quad():
    from session_py import Point

    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(2.0, 0.0, 1.0)
    p2 = Point(2.0, 2.0, 2.0)
    p3 = Point(0.0, 2.0, 1.0)
    centroid = Point.centroid_quad([p0, p1, p2, p3])

    MINI_CHECK(TOLERANCE.is_close(centroid[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(centroid[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(centroid[2], 1.0))


@MINI_TEST("Point", "Centroid")
def test_point_centroid():
    from session_py import Point

    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(2.0, 0.0, 0.0)
    p2 = Point(2.0, 2.0, 0.0)
    p3 = Point(0.0, 2.0, 0.0)
    centroid = Point.centroid([p0, p1, p2, p3])

    MINI_CHECK(TOLERANCE.is_close(centroid[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(centroid[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(centroid[2], 0.0))


@MINI_TEST("Point", "Dihedral Angle Deg")
def test_point_dihedral_angle_deg():
    from session_py import Point

    p = Point(0.0, 0.0, 0.0)
    q = Point(1.0, 0.0, 0.0)
    r = Point(0.0, 1.0, 0.0)
    s = Point(0.0, 0.0, 1.0)
    angle = Point.dihedral_angle_deg(p, q, r, s)

    MINI_CHECK(TOLERANCE.is_close(angle, 90.0))


if __name__ == "__main__":
    run_all(language="python")
