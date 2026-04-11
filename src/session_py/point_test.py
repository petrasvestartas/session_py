from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE

# /home/petras/code/code_session/uvsession/bin/python -m session_py.point_test


@MINI_TEST("Point", "Constructor")
def test_point_constructor():
    from session_py import Point
    from session_py import Vector
    from session_py import Color

    # Constructor
    p = Point(1.0, 2.0, 3.0)

    # Setters
    p[0] = 10.0
    p[1] = 20.0
    p[2] = 30.0

    # Getters
    x = p[0]
    y = p[1]
    z = p[2]

    # Minimal and Full String Representation
    pstr = str(p)
    prepr = repr(p) 

    # Copy (duplicates everything except guid)
    pcopy = p.duplicate()
    pother = Point(1.0, 2.0, 3.0)

    # No-copy operators
    pmult = p.duplicate()
    pmult *= 2.0
    pdiv = p.duplicate()
    pdiv /= 2.0
    padd = p.duplicate()
    padd += Vector(1.0, 1.0, 1.0)
    psub = p.duplicate()
    psub -= Vector(1.0, 1.0, 1.0)

    # Copy operators
    result_mul = p * 2.0
    result_div = p / 2.0
    result_add = p + Vector(1.0, 1.0, 1.0)
    diff_point = p - Vector(1.0, 1.0, 1.0)

    # Static sum and sub methods
    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(4.0, 5.0, 6.0)
    psum = Point.sum(p1, p2)
    pdif = Point.sub(p2, p1)

    MINI_CHECK(p.name == "my_point")
    MINI_CHECK(p[0] == 10.0)
    MINI_CHECK(p[1] == 20.0)
    MINI_CHECK(p[2] == 30.0)
    MINI_CHECK(p.width == 1.0)
    MINI_CHECK(p.pointcolor == Color.blue())
    MINI_CHECK(p.guid)
    MINI_CHECK(x == 10.0 and y == 20.0 and z == 30.0)
    MINI_CHECK(pstr == "10.0, 20.0, 30.0")
    MINI_CHECK(prepr == "Point(my_point, 10.0, 20.0, 30.0, Color(blue, 0, 0, 255, 255), 1.0)")
    MINI_CHECK(pcopy == p and pcopy.guid != p.guid)
    MINI_CHECK(pother != p)
    MINI_CHECK(pmult[0] == 20.0 and pmult[1] == 40.0 and pmult[2] == 60.0)
    MINI_CHECK(pdiv[0] == 5.0 and pdiv[1] == 10.0 and pdiv[2] == 15.0)
    MINI_CHECK(padd[0] == 11.0 and padd[1] == 21.0 and padd[2] == 31.0)
    MINI_CHECK(psub[0] == 9.0 and psub[1] == 19.0 and psub[2] == 29.0)
    MINI_CHECK(result_mul[0] == 20.0 and result_mul[1] == 40.0 and result_mul[2] == 60.0)
    MINI_CHECK(result_div[0] == 5.0 and result_div[1] == 10.0 and result_div[2] == 15.0)
    MINI_CHECK(result_add[0] == 11.0 and result_add[1] == 21.0 and result_add[2] == 31.0)
    MINI_CHECK(diff_point[0] == 9.0 and diff_point[1] == 19.0 and diff_point[2] == 29.0)
    MINI_CHECK(psum[0] == 5.0 and psum[1] == 7.0 and psum[2] == 9.0)
    MINI_CHECK(pdif[0] == 3.0 and pdif[1] == 3.0 and pdif[2] == 3.0)


@MINI_TEST("Point", "Transformation")
def test_transformation():
    from session_py import Point
    from session_py import Xform
    
    p = Point(1.0, 2.0, 3.0)
    p.xform = Xform.translation(1.0, 2.0, 3.0)
    p_transformed = p.transformed() # Make a copy
    p.transform() # After the call, "xform" is reset

    MINI_CHECK(p_transformed[0] == 2.0 and p_transformed[1] == 4.0 and p_transformed[2] == 6.0)
    MINI_CHECK(p[0] == 2.0 and p[1] == 4.0 and p[2] == 6.0)
    MINI_CHECK(p.xform == Xform.identity())


@MINI_TEST("Point", "Json Roundtrip")
def test_point_json_roundtrip():
    from session_py import Point
    from session_py import Color
    from pathlib import Path

    p = Point(1.5, 2.5, 3.5)
    p.name = "test_point"
    p.width = 2.0
    p.pointcolor = Color(255, 128, 64, 255)

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   json_dumps()    │ str          │ to JSON string
    #   json_loads(s)   │ str          │ from JSON string
    #   json_dump(path) │ file         │ write to file
    #   json_load(path) │ file         │ read from file

    # json_dump(fname) / json_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_point.json"
    p.json_dump(fname)
    loaded = Point.json_load(fname)

    MINI_CHECK(isinstance(loaded, Point))
    MINI_CHECK(loaded.name == p.name)
    MINI_CHECK(loaded[0] == p[0])
    MINI_CHECK(loaded[1] == p[1])
    MINI_CHECK(loaded[2] == p[2])
    MINI_CHECK(loaded.width == p.width)
    MINI_CHECK(loaded.pointcolor[0] == 255)
    MINI_CHECK(loaded.pointcolor[1] == 128)
    MINI_CHECK(loaded.pointcolor[2] == 64)
    MINI_CHECK(loaded.pointcolor[3] == 255)


@MINI_TEST("Point", "Protobuf Roundtrip")
def test_point_protobuf_roundtrip():
    from session_py import Point
    from session_py import Color
    from pathlib import Path

    p = Point(1.5, 2.5, 3.5)
    p.name = "test_point"
    p.width = 2.0
    p.pointcolor = Color(255, 128, 64, 255)

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_point.bin"
    p.pb_dump(path)
    loaded = Point.pb_load(path)

    MINI_CHECK(isinstance(loaded, Point))
    MINI_CHECK(loaded.name == p.name)
    MINI_CHECK(loaded[0] == p[0])
    MINI_CHECK(loaded[1] == p[1])
    MINI_CHECK(loaded[2] == p[2])
    MINI_CHECK(loaded.width == p.width)
    MINI_CHECK(loaded.pointcolor[0] == 255)
    MINI_CHECK(loaded.pointcolor[1] == 128)
    MINI_CHECK(loaded.pointcolor[2] == 64)
    MINI_CHECK(loaded.pointcolor[3] == 255)


@MINI_TEST("Point", "Is Ccw")
def test_is_ccw():
    from session_py import Point
    
    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(1.0, 0.0, 0.0)
    p2 = Point(0.05, 1.0, 0.0)

    # Points must be oriented to xy plane.
    is_counter_clock_wise = Point.is_ccw(p0, p1, p2)
    is_clock_wise = Point.is_ccw(p2, p1, p0)
    
    MINI_CHECK(is_counter_clock_wise)
    MINI_CHECK(not is_clock_wise)


@MINI_TEST("Point", "Mid Point")
def test_mid_point():
    from session_py import Point
    
    p0 = Point(0.0, 2.0, 1.0)
    p1 = Point(1.0, 5.0, 3.0)
    mid = Point.mid_point(p0, p1)
    
    MINI_CHECK(mid[0] == 0.5 and mid[1] == 3.5 and mid[2] == 2.0)


@MINI_TEST("Point", "Distance")
def test_distance():
    from session_py import Point

    p0 = Point(0.0, 2.0, 1.0)
    p1 = Point(1.0, 5.0, 3.0)
    d = Point.distance(p0, p1)

    MINI_CHECK(TOLERANCE.is_close(d, 3.741657))


@MINI_TEST("Point", "Squared Distance")
def test_squared_distance():
    from session_py import Point

    p0 = Point(0.0, 2.0, 1.0)
    p1 = Point(1.0, 5.0, 3.0)
    d = Point.squared_distance(p0, p1)

    MINI_CHECK(TOLERANCE.is_close(d, 14.0))


@MINI_TEST("Point", "Area")
def test_area():
    from session_py import Point
    
    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(2.0, 0.0, 0.0)
    p2 = Point(2.0, 2.0, 0.0)
    p3 = Point(0.0, 2.0, 0.0)
    area = Point.area([p0, p1, p2, p3])
    
    MINI_CHECK(area == 4.0)


@MINI_TEST("Point", "Centroid Quad")
def test_centroid_quad():
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
def test_centroid():
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
def test_dihedral_angle_deg():
    from session_py import Point

    p = Point(0.0, 0.0, 0.0)
    q = Point(1.0, 0.0, 0.0)
    r = Point(0.0, 1.0, 0.0)
    s = Point(0.0, 0.0, 1.0)
    angle = Point.dihedral_angle_deg(p, q, r, s)

    MINI_CHECK(TOLERANCE.is_close(angle, 90.0))


if __name__ == "__main__":
    run_all(language="python")