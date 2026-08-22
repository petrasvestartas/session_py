from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Line", "Constructor")
def test_line_constructor():
    from session_py import Line
    from session_py import Point
    from session_py import Vector
    from session_py import Color

    # Constructor
    l = Line(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    # Setters
    l[0] = 10.0
    l[1] = 20.0
    l[2] = 30.0
    l[3] = 40.0
    l[4] = 50.0
    l[5] = 60.0

    # Getters
    x0 = l[0]
    y0 = l[1]
    z0 = l[2]
    x1 = l[3]
    y1 = l[4]
    z1 = l[5]

    # Minimal and Full String Representation
    lstr = str(l)
    lrepr = repr(l)

    # Copy (duplicate everything but guid)
    lcopy = l.duplicate()
    lother = Line(10.0, 20.0, 30.0, 40.0, 50.0, 60.0)

    # No-copy operators
    lmult = l.duplicate()
    lmult *= 2.0
    ldiv = l.duplicate()
    ldiv /= 2.0
    ladd = l.duplicate()
    ladd += Vector(1.0, 1.0, 1.0)
    lsub = l.duplicate()
    lsub -= Vector(1.0, 1.0, 1.0)

    # Copy operators
    rmul = l * 2.0
    rdiv = l / 2.0
    radd = l + Vector(1.0, 1.0, 1.0)
    rdif = l - Vector(1.0, 1.0, 1.0)

    # Negation (flip start and end)
    lneg = Line(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    neg = -lneg

    # From points constructor
    p0 = Point(1.0, 2.0, 3.0)
    p1 = Point(4.0, 5.0, 6.0)
    l2p = Line.from_points(p0, p1)

    # from_point_and_vector constructor
    pv = Point(1.0, 2.0, 3.0)
    vv = Vector(3.0, 4.0, 5.0)
    l_pv = Line.from_point_and_vector(pv, vv)

    # from_point_direction_length constructor
    pd = Point(0.0, 0.0, 0.0)
    dd = Vector(1.0, 0.0, 0.0)
    l_pdl = Line.from_point_direction_length(pd, dd, 5.0)

    # Line with custom color and width
    lc = Line(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    lc.linecolor = Color(1.0, 0.0, 0.0, 1.0, "red")
    lc.width = 2.5

    # with_name constructor
    lwn = Line.with_name("custom", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    # get_middle_line
    ms, me = Line.get_middle_line(
        Point(0.0, 0.0, 0.0), Point(2.0, 0.0, 0.0),
        Point(0.0, 2.0, 0.0), Point(2.0, 2.0, 0.0))

    MINI_CHECK(l.name == "my_line" and l[0] == 10.0 and l[1] == 20.0 and l[2] == 30.0 and l.guid)
    MINI_CHECK(x0 == 10.0 and y0 == 20.0 and z0 == 30.0)
    MINI_CHECK(x1 == 40.0 and y1 == 50.0 and z1 == 60.0)
    MINI_CHECK("10.0" in lstr and "20.0" in lstr and "60.0" in lstr)
    MINI_CHECK("my_line" in lrepr and "10.0" in lrepr and "Color" in lrepr)
    MINI_CHECK(lcopy.guid != l.guid)
    MINI_CHECK(lmult[0] == 20.0 and lmult[3] == 80.0)
    MINI_CHECK(ldiv[0] == 5.0 and ldiv[3] == 20.0)
    MINI_CHECK(ladd[0] == 11.0 and ladd[3] == 41.0)
    MINI_CHECK(lsub[0] == 9.0 and lsub[3] == 39.0)
    MINI_CHECK(rmul[0] == 20.0 and rmul[3] == 80.0)
    MINI_CHECK(rdiv[0] == 5.0 and rdiv[3] == 20.0)
    MINI_CHECK(radd[0] == 11.0 and radd[3] == 41.0)
    MINI_CHECK(rdif[0] == 9.0 and rdif[3] == 39.0)
    MINI_CHECK(neg[0] == 4.0 and neg[1] == 5.0 and neg[2] == 6.0)
    MINI_CHECK(neg[3] == 1.0 and neg[4] == 2.0 and neg[5] == 3.0)
    MINI_CHECK(l2p[0] == 1.0 and l2p[3] == 4.0)
    MINI_CHECK(l_pv[0] == 1.0 and l_pv[1] == 2.0 and l_pv[2] == 3.0)
    MINI_CHECK(l_pv[3] == 4.0 and l_pv[4] == 6.0 and l_pv[5] == 8.0)
    MINI_CHECK(l_pdl[0] == 0.0 and l_pdl[3] == 5.0)
    MINI_CHECK(lc.linecolor[0] == 1.0 and lc.linecolor[1] == 0.0 and lc.width == 2.5)
    MINI_CHECK(lwn.name == "custom" and lwn[3] == 1.0)
    MINI_CHECK(TOLERANCE.is_close(ms[1], 1.0) and TOLERANCE.is_close(me[1], 1.0))


@MINI_TEST("Line", "Transformation")
def test_line_transformation():
    from session_py import Line
    from session_py import Xform

    l = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    l_xf = Xform.translation(10.0, 0.0, 0.0)
    l_transformed = l.transformed(l_xf)  # Make a copy
    l.transform(l_xf)

    MINI_CHECK(l_transformed[0] == 10.0 and l_transformed[3] == 11.0)
    MINI_CHECK(l[0] == 10.0 and l[3] == 11.0)


@MINI_TEST("Line", "Json Roundtrip")
def test_line_json_roundtrip():
    from session_py import Line
    from pathlib import Path

    l = Line(42.1, 84.2, 126.3, 168.4, 210.5, 252.6)
    l.name = "test_line"
    l.dash = [3.0, 2.0]

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   file_json_dumps()    │ str          │ to JSON string
    #   file_json_loads(s)   │ str          │ from JSON string
    #   file_json_dump(path) │ file         │ write to file
    #   file_json_load(path) │ file         │ read from file

    # JSON object
    d = l.__jsondump__()
    loaded_j = Line.__jsonload__(d)

    MINI_CHECK(loaded_j.name == "test_line")

    # String
    s = l.file_json_dumps()
    loaded_s = Line.file_json_loads(s)
    MINI_CHECK(loaded_s.name == "test_line")
    MINI_CHECK(TOLERANCE.is_close(loaded_s[0], 42.1))

    # File
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_line.json"
    l.file_json_dump(fname)
    loaded = Line.file_json_load(fname)

    MINI_CHECK(loaded.name == "test_line")
    MINI_CHECK(TOLERANCE.is_close(loaded[0], 42.1))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], 84.2))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], 126.3))
    MINI_CHECK(TOLERANCE.is_close(loaded[3], 168.4))
    MINI_CHECK(TOLERANCE.is_close(loaded[4], 210.5))
    MINI_CHECK(TOLERANCE.is_close(loaded[5], 252.6))
    MINI_CHECK(loaded.dash == [3.0, 2.0])


@MINI_TEST("Line", "Protobuf Roundtrip")
def test_line_protobuf_roundtrip():
    from session_py import Line
    from pathlib import Path

    l = Line(42.1, 84.2, 126.3, 168.4, 210.5, 252.6)
    l.name = "test_line"
    l.dash = [3.0, 2.0]

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    # Bytes
    b = l.pb_dumps()
    loaded_s = Line.pb_loads(b)

    MINI_CHECK(loaded_s.name == "test_line")
    MINI_CHECK(TOLERANCE.is_close(loaded_s[0], 42.1))
    MINI_CHECK(loaded_s.guid == l.guid)

    # File
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_line.bin"
    l.pb_dump(fname)
    loaded = Line.pb_load(fname)

    MINI_CHECK(loaded.name == "test_line")
    MINI_CHECK(TOLERANCE.is_close(loaded[0], 42.1))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], 84.2))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], 126.3))
    MINI_CHECK(TOLERANCE.is_close(loaded[3], 168.4))
    MINI_CHECK(TOLERANCE.is_close(loaded[4], 210.5))
    MINI_CHECK(TOLERANCE.is_close(loaded[5], 252.6))
    MINI_CHECK(loaded.dash == [3.0, 2.0])
    MINI_CHECK(loaded.guid == l.guid)


@MINI_TEST("Line", "Length")
def test_line_length():
    from session_py import Line

    l = Line(0.0, 0.0, 0.0, 3.0, 4.0, 0.0)
    ln = l.length()
    lsq = l.squared_length()

    MINI_CHECK(TOLERANCE.is_close(ln, 5.0))
    MINI_CHECK(TOLERANCE.is_close(lsq, 25.0))


@MINI_TEST("Line", "To Vector")
def test_line_to_vector():
    from session_py import Line

    l = Line(1.0, 2.0, 3.0, 4.0, 6.0, 9.0)
    v = l.to_vector()

    MINI_CHECK(v[0] == 3.0 and v[1] == 4.0 and v[2] == 6.0)


@MINI_TEST("Line", "To Direction")
def test_line_to_direction():
    from session_py import Line

    l = Line(0.0, 0.0, 0.0, 3.0, 4.0, 0.0)
    d = l.to_direction()

    MINI_CHECK(TOLERANCE.is_close(d[0], 0.6))
    MINI_CHECK(TOLERANCE.is_close(d[1], 0.8))
    MINI_CHECK(TOLERANCE.is_close(d[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(d.magnitude(), 1.0))


@MINI_TEST("Line", "Point At")
def test_line_point_at():
    from session_py import Line

    l = Line(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
    ps = l.point_at(0.0)
    pm = l.point_at(0.5)
    pe = l.point_at(1.0)

    MINI_CHECK(ps[0] == 0.0 and ps[1] == 0.0 and ps[2] == 0.0)
    MINI_CHECK(pm[0] == 5.0 and pm[1] == 5.0 and pm[2] == 5.0)
    MINI_CHECK(pe[0] == 10.0 and pe[1] == 10.0 and pe[2] == 10.0)


@MINI_TEST("Line", "Closest Point")
def test_line_closest_point():
    from session_py import Line
    from session_py import Point

    l = Line(0.0, 0.0, 0.0, 10.0, 0.0, 0.0)
    p1 = Point(5.0, 5.0, 0.0)
    p2 = Point(-5.0, 0.0, 0.0)
    p3 = Point(15.0, 0.0, 0.0)
    t1, cp1 = l.closest_point(p1)
    t2, cp2 = l.closest_point(p2)
    t3, cp3 = l.closest_point(p3)

    MINI_CHECK(cp1[0] == 5.0 and cp1[1] == 0.0 and cp1[2] == 0.0)
    MINI_CHECK(cp2[0] == 0.0 and cp2[1] == 0.0 and cp2[2] == 0.0)
    MINI_CHECK(cp3[0] == 10.0 and cp3[1] == 0.0 and cp3[2] == 0.0)
    MINI_CHECK(TOLERANCE.is_close(t1, 0.5))
    MINI_CHECK(TOLERANCE.is_close(t2, 0.0))
    MINI_CHECK(TOLERANCE.is_close(t3, 1.0))


@MINI_TEST("Line", "Start End Center")
def test_line_start_end_center():
    from session_py import Line

    l = Line(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    start = l.start()
    end = l.end()
    center = l.center()

    MINI_CHECK(start[0] == 1.0 and start[1] == 2.0 and start[2] == 3.0)
    MINI_CHECK(end[0] == 4.0 and end[1] == 5.0 and end[2] == 6.0)
    MINI_CHECK(center[0] == 2.5 and center[1] == 3.5 and center[2] == 4.5)


@MINI_TEST("Line", "Fit Points")
def test_line_fit_points():
    from session_py import Line
    from session_py import Point

    fit_pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.5),
        Point(2.0, 2.0, 1.0),
        Point(3.0, 3.0, 1.5),
    ]
    l_fit = Line.fit_points(fit_pts)

    MINI_CHECK(l_fit.length() > 0.0)


@MINI_TEST("Line", "Subdivide")
def test_line_subdivide():
    from session_py import Line

    l = Line(0.0, 0.0, 0.0, 10.0, 0.0, 0.0)

    # subdivide by count
    pts = l.subdivide(3)

    # subdivide_by_distance
    pts_dist = l.subdivide_by_distance(2.5)

    MINI_CHECK(len(pts) == 3)
    MINI_CHECK(pts[0][0] == 0.0)
    MINI_CHECK(pts[1][0] == 5.0)
    MINI_CHECK(pts[2][0] == 10.0)
    MINI_CHECK(len(pts_dist) == 5)
    MINI_CHECK(TOLERANCE.is_close(pts_dist[0][0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pts_dist[1][0], 2.5))
    MINI_CHECK(TOLERANCE.is_close(pts_dist[4][0], 10.0))


@MINI_TEST("Line", "Overlap")
def test_line_overlap():
    from session_py import Line
    from session_py import Point
    l0 = Line.from_points(Point(0.0, 0.0, 0.0), Point(10.0, 0.0, 0.0))
    l1 = Line.from_points(Point(5.0, 0.0, 0.0), Point(15.0, 0.0, 0.0))
    out = l0.overlap(l1)
    MINI_CHECK(out is not None)
    MINI_CHECK(TOLERANCE.is_close(out.start()[0], 5.0))
    MINI_CHECK(TOLERANCE.is_close(out.end()[0], 10.0))


@MINI_TEST("Line", "Overlap Average")
def test_line_overlap_average():
    from session_py import Line
    from session_py import Point
    l0 = Line.from_points(Point(0.0, 0.0, 0.0), Point(10.0, 0.0, 0.0))
    l1 = Line.from_points(Point(5.0, 0.0, 0.0), Point(15.0, 0.0, 0.0))
    out = l0.overlap_average(l1)
    MINI_CHECK(out is not None)
    MINI_CHECK(TOLERANCE.is_close(out.start()[0], 5.0))
    MINI_CHECK(TOLERANCE.is_close(out.end()[0], 10.0))


@MINI_TEST("Line", "Extend")
def test_line_extend():
    from session_py import Line
    from session_py import Point
    l = Line.from_points(Point(0.0, 0.0, 0.0), Point(10.0, 0.0, 0.0))
    l.extend(1.0, 2.0)
    MINI_CHECK(TOLERANCE.is_close(l.start()[0], -1.0))
    MINI_CHECK(TOLERANCE.is_close(l.end()[0], 12.0))


if __name__ == "__main__":
    run_all("python")
