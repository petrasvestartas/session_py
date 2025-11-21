from .mini_test import MINI_TEST, MINI_CHECK, run_all

# /home/petras/code/code_session/uvsession/bin/python -m session_py.point_test


@MINI_TEST("Point", "constructor")
def test_point_constructor():
    from session_py.point import Point
    from session_py.vector import Vector
    from session_py.color import Color
    import copy

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

    # String  representation
    pstr = str(p)
    prepr = repr(p)

    # Copy (duplicates everything except guid)
    pcopy = copy.deepcopy(p)
    pother = Point(1.0, 2.0, 3.0)

    # No-copy operators
    pmult = copy.deepcopy(p)
    pmult *= 2.0
    pdiv = copy.deepcopy(p)
    pdiv /= 2.0
    padd = copy.deepcopy(p)
    padd += Vector(1.0, 1.0, 1.0)
    psub = copy.deepcopy(p)
    psub -= Vector(1.0, 1.0, 1.0)

    # Copy operators
    result_mul = p * 2.0
    result_div = p / 2.0
    result_add = p + Vector(1.0, 1.0, 1.0) # Works with point too
    diff_point = p - Vector(1.0, 1.0, 1.0) # Works with point too

    MINI_CHECK(
        p.name == "my_point" and
        p[0] == 10.0 and 
        p[1] == 20.0 and 
        p[2] == 30.0 and
        p.width == 1.0 and
        p.pointcolor == Color.blue() and
        p.guid)
    
    MINI_CHECK(x == 10.0 and y == 20.0 and z == 30.0)
    
    MINI_CHECK(pstr == "10.0, 20.0, 30.0")
    MINI_CHECK(prepr == "Point(my_point, 10.0, 20.0, 30.0, Color(0, 0, 255, 255), 1.0)")
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


@MINI_TEST("Point", "transformation")
def test_transformation():

    from session_py.point import Point
    from session_py.xform import Xform
    
    p = Point(1.0, 2.0, 3.0)
    p.xform = Xform.translation(1.0, 2.0, 3.0)

    p_transformed = p.transformed() # Make a copy
    p.transform() # After transform, xform is reset to identity

    MINI_CHECK(p_transformed[0] == 2.0 and p_transformed[1] == 4.0 and p_transformed[2] == 6.0)
    MINI_CHECK(p[0] == 2.0 and p[1] == 4.0 and p[2] == 6.0)
    MINI_CHECK(p.xform == Xform.identity())


@MINI_TEST("Point", "is_ccw")
def test_is_ccw():

    from session_py.point import Point
    
    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(1.0, 0.0, 0.0)
    p2 = Point(0.05, 1.0, 0.0)

    # Points must be oriented to xy plane.
    is_counter_clock_wise = Point.is_ccw(p0, p1, p2)
    is_clock_wise = Point.is_ccw(p2, p1, p0)
    
    MINI_CHECK(is_counter_clock_wise)
    MINI_CHECK(not is_clock_wise)


@MINI_TEST("Point", "mid_point")
def test_mid_point():

    from session_py.point import Point
    
    p0 = Point(0.0, 2.0, 1.0)
    p1 = Point(1.0, 5.0, 3.0)
    mid = Point.mid_point(p0, p1)
    
    MINI_CHECK(mid[0] == 0.5 and mid[1] == 3.5 and mid[2] == 2.0)


@MINI_TEST("Point", "distance")
def test_distance():

    from session_py.point import Point
    from session_py.tolerance import Tolerance
    
    p0 = Point(0.0, 2.0, 1.0)
    p1 = Point(1.0, 5.0, 3.0)
    d = round(Point.distance(p0, p1), Tolerance.ROUNDING)
    
    MINI_CHECK(d == 3.741657)


@MINI_TEST("Point", "squared_distance")
def test_squared_distance():

    from session_py.point import Point
    from session_py.tolerance import Tolerance
    
    p0 = Point(0.0, 2.0, 1.0)
    p1 = Point(1.0, 5.0, 3.0)
    d = round(Point.squared_distance(p0, p1), Tolerance.ROUNDING)
    
    MINI_CHECK(d == 14.0)


@MINI_TEST("Point", "area")
def test_area():

    from session_py.point import Point
    
    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(2.0, 0.0, 0.0)
    p2 = Point(2.0, 2.0, 0.0)
    p3 = Point(0.0, 2.0, 0.0)
    area = Point.area([p0, p1, p2, p3])
    
    MINI_CHECK(area == 4.0)


@MINI_TEST("Point", "centroid_quad")
def test_centroid_quad():

    from session_py.point import Point
    from session_py.tolerance import Tolerance
    
    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(2.0, 0.0, 1.0)
    p2 = Point(2.0, 2.0, 2.0)
    p3 = Point(0.0, 2.0, 1.0)
    centroid = Point.centroid_quad([p0, p1, p2, p3])
    x = round(centroid[0], Tolerance.ROUNDING)
    y = round(centroid[1], Tolerance.ROUNDING)
    z = round(centroid[2], Tolerance.ROUNDING)
    
    MINI_CHECK(x == 1.0 and y == 1.0 and z == 1.0)


@MINI_TEST("Point", "json_roundtrip")
def test_point_json_roundtrip():

    from session_py.point import Point
    from session_py.color import Color
    from pathlib import Path
    from session_py.encoders import json_dump, json_load

    p = Point(1.5, 2.5, 3.5)
    p.name = "test_point"
    p.width = 2.0
    p.pointcolor = Color(255, 128, 64, 255)

    path = Path(__file__).resolve().parents[2] / "test_point.json"
    json_dump(p, path)
    loaded = json_load(path)

    MINI_CHECK(isinstance(loaded, Point))
    MINI_CHECK(loaded.name == p.name)
    MINI_CHECK(loaded[0] == p[0])
    MINI_CHECK(loaded[1] == p[1])
    MINI_CHECK(loaded[2] == p[2])
    MINI_CHECK(loaded.width == p.width)
    MINI_CHECK(loaded.pointcolor.r == 255)
    MINI_CHECK(loaded.pointcolor.g == 128)
    MINI_CHECK(loaded.pointcolor.b == 64)
    MINI_CHECK(loaded.pointcolor.a == 255)

if __name__ == "__main__":
    run_all(language="python")