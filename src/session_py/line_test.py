import math
from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Line", "Constructor")
def test_line_constructor():
    from session_py import Color
    from session_py import Line
    from session_py import Point
    from session_py import Vector

    line = Line(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    line[0] = 10.0
    line[1] = 20.0
    line[2] = 30.0
    line[3] = 40.0
    line[4] = 50.0
    line[5] = 60.0

    x0 = line[0]
    y0 = line[1]
    z0 = line[2]
    x1 = line[3]
    y1 = line[4]
    z1 = line[5]

    lstr = str(line)
    lrepr = repr(line)

    lcopy = line.duplicate()
    lother = Line(10.0, 20.0, 30.0, 40.0, 50.0, 60.0)

    lmult = line.duplicate()
    lmult *= 2.0
    ldiv = line.duplicate()
    ldiv /= 2.0
    ladd = line.duplicate()
    ladd += Vector(1.0, 1.0, 1.0)
    lsub = line.duplicate()
    lsub -= Vector(1.0, 1.0, 1.0)

    rmul = line * 2.0
    rdiv = line / 2.0
    radd = line + Vector(1.0, 1.0, 1.0)
    rdif = line - Vector(1.0, 1.0, 1.0)

    lneg = Line(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    neg = -lneg

    p0 = Point(1.0, 2.0, 3.0)
    p1 = Point(4.0, 5.0, 6.0)
    l2p = Line.from_points(p0, p1)

    pv = Point(1.0, 2.0, 3.0)
    vv = Vector(3.0, 4.0, 5.0)
    l_pv = Line.from_point_and_vector(pv, vv)

    pd = Point(0.0, 0.0, 0.0)
    dd = Vector(1.0, 0.0, 0.0)
    l_pdl = Line.from_point_direction_length(pd, dd, 5.0)

    lc = Line(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    lc.linecolor = Color(1.0, 0.0, 0.0, 1.0, "red")
    lc.width = 2.5

    lwn = Line.with_name("custom", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    ms, me = Line.get_middle_line(
        Point(0.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(0.0, 2.0, 0.0),
        Point(2.0, 2.0, 0.0),
    )

    MINI_CHECK(line.name == "my_line")
    MINI_CHECK(line[0] == 10.0 and line[1] == 20.0 and line[2] == 30.0)
    MINI_CHECK(line.width == 1.0)
    MINI_CHECK(line.linecolor == Color.black())
    MINI_CHECK(line.guid != "")
    MINI_CHECK(
        x0 == 10.0
        and y0 == 20.0
        and z0 == 30.0
        and x1 == 40.0
        and y1 == 50.0
        and z1 == 60.0
    )
    MINI_CHECK(
        lstr == "10.000000, 20.000000, 30.000000, 40.000000, 50.000000, 60.000000"
    )
    MINI_CHECK(
        lrepr
        == "Line(my_line, 10.000000, 20.000000, 30.000000, 40.000000, 50.000000, 60.000000, Color(black, 0.0, 0.0, 0.0, 1.0), 1.000000)"
    )
    MINI_CHECK(lcopy == line and lcopy.guid != line.guid)
    MINI_CHECK(lother == line and lneg != line)
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

    line = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    xform = Xform.translation(10.0, 0.0, 0.0)
    moved = line.transformed(xform)
    line.transform(xform)

    MINI_CHECK(moved[0] == 10.0 and moved[3] == 11.0)
    MINI_CHECK(line[0] == 10.0 and line[3] == 11.0)


@MINI_TEST("Line", "Json Roundtrip")
def test_line_json_roundtrip():
    from pathlib import Path
    from session_py import Line

    line = Line(42.1, 84.2, 126.3, 168.4, 210.5, 252.6)
    line.name = "test_line"
    line.dash = [3.0, 2.0]

    j = line.__jsondump__()
    loaded_j = Line.__jsonload__(j)

    s = line.file_json_dumps()
    loaded_s = Line.file_json_loads(s)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_line.json"
    line.file_json_dump(fname)
    loaded = Line.file_json_load(fname)

    MINI_CHECK(loaded_j.name == "test_line")
    MINI_CHECK(TOLERANCE.is_close(loaded_j[0], 42.1))
    MINI_CHECK(loaded_s.name == "test_line")
    MINI_CHECK(TOLERANCE.is_close(loaded_s[0], 42.1))
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
    from pathlib import Path
    from session_py import Line

    line = Line(42.1, 84.2, 126.3, 168.4, 210.5, 252.6)
    line.name = "test_line"
    line.dash = [3.0, 2.0]

    guid = line.guid
    s = line.pb_dumps()
    loaded_s = Line.pb_loads(s)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_line.bin"
    line.pb_dump(fname)
    loaded = Line.pb_load(fname)
    converted = Line.from_proto(line.to_proto())

    MINI_CHECK(loaded_s.name == "test_line")
    MINI_CHECK(TOLERANCE.is_close(loaded_s[0], 42.1))
    MINI_CHECK(loaded_s.guid == guid)
    MINI_CHECK(loaded.name == "test_line")
    MINI_CHECK(TOLERANCE.is_close(loaded[0], 42.1))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], 84.2))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], 126.3))
    MINI_CHECK(TOLERANCE.is_close(loaded[3], 168.4))
    MINI_CHECK(TOLERANCE.is_close(loaded[4], 210.5))
    MINI_CHECK(TOLERANCE.is_close(loaded[5], 252.6))
    MINI_CHECK(loaded.dash == [3.0, 2.0])
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(converted == line)
    MINI_CHECK(converted.guid == guid)


@MINI_TEST("Line", "Length")
def test_line_length():
    from session_py import Line

    line = Line(0.0, 0.0, 0.0, 3.0, 4.0, 0.0)
    ln = line.length()
    lsq = line.squared_length()

    MINI_CHECK(TOLERANCE.is_close(ln, 5.0))
    MINI_CHECK(TOLERANCE.is_close(lsq, 25.0))


@MINI_TEST("Line", "To Vector")
def test_line_to_vector():
    from session_py import Line

    line = Line(1.0, 2.0, 3.0, 4.0, 6.0, 9.0)
    v = line.to_vector()

    MINI_CHECK(v[0] == 3.0 and v[1] == 4.0 and v[2] == 6.0)


@MINI_TEST("Line", "To Direction")
def test_line_to_direction():
    from session_py import Line

    line = Line(0.0, 0.0, 0.0, 3.0, 4.0, 0.0)
    d = line.to_direction()

    MINI_CHECK(TOLERANCE.is_close(d[0], 0.6))
    MINI_CHECK(TOLERANCE.is_close(d[1], 0.8))
    MINI_CHECK(TOLERANCE.is_close(d[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(d.magnitude(), 1.0))


@MINI_TEST("Line", "Point At")
def test_line_point_at():
    from session_py import Line

    line = Line(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
    ps = line.point_at(0.0)
    pm = line.point_at(0.5)
    pe = line.point_at(1.0)

    MINI_CHECK(ps[0] == 0.0 and ps[1] == 0.0 and ps[2] == 0.0)
    MINI_CHECK(pm[0] == 5.0 and pm[1] == 5.0 and pm[2] == 5.0)
    MINI_CHECK(pe[0] == 10.0 and pe[1] == 10.0 and pe[2] == 10.0)


@MINI_TEST("Line", "Closest Point")
def test_line_closest_point():
    from session_py import Line
    from session_py import Point

    line = Line(0.0, 0.0, 0.0, 10.0, 0.0, 0.0)
    p1 = Point(5.0, 5.0, 0.0)
    p2 = Point(-5.0, 0.0, 0.0)
    p3 = Point(15.0, 0.0, 0.0)
    t1, cp1 = line.closest_point(p1)
    t2, cp2 = line.closest_point(p2)
    t3, cp3 = line.closest_point(p3)

    MINI_CHECK(cp1[0] == 5.0 and cp1[1] == 0.0 and cp1[2] == 0.0)
    MINI_CHECK(cp2[0] == 0.0 and cp2[1] == 0.0 and cp2[2] == 0.0)
    MINI_CHECK(cp3[0] == 10.0 and cp3[1] == 0.0 and cp3[2] == 0.0)
    MINI_CHECK(TOLERANCE.is_close(t1, 0.5))
    MINI_CHECK(TOLERANCE.is_close(t2, 0.0))
    MINI_CHECK(TOLERANCE.is_close(t3, 1.0))


@MINI_TEST("Line", "Closest Point Unlimited")
def test_line_closest_point_unlimited():
    from session_py import Line
    from session_py import Point

    line = Line(0.0, 0.0, 0.0, 10.0, 0.0, 0.0)
    before = line.closest_point(Point(-5.0, 2.0, 0.0), False)
    after = line.closest_point(Point(15.0, 3.0, 0.0), False)

    MINI_CHECK(TOLERANCE.is_close(before[0], -0.5))
    MINI_CHECK(TOLERANCE.is_point_close(before[1], Point(-5.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_close(after[0], 1.5))
    MINI_CHECK(TOLERANCE.is_point_close(after[1], Point(15.0, 0.0, 0.0)))


@MINI_TEST("Line", "Start End Center")
def test_line_start_end_center():
    from session_py import Line

    line = Line(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    start = line.start()
    end = line.end()
    center = line.center()

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

    l_vertical = Line.fit_points(
        [
            Point(0.0, 0.0, 0.0),
            Point(0.0, 1.0, 0.0),
            Point(0.0, 2.0, 0.0),
            Point(0.0, 3.0, 0.0),
        ]
    )

    MINI_CHECK(abs(l_vertical.to_direction()[1]) > 0.99)

    l_skew = Line.fit_points(
        [
            Point(3.0, 0.0, 0.0),
            Point(-3.0, 0.0, 0.0),
            Point(0.0, 2.4, 2.4),
            Point(0.0, -2.4, -2.4),
        ]
    )
    skew = l_skew.to_direction()

    MINI_CHECK(TOLERANCE.is_close(skew[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(abs(skew[1]), math.sqrt(0.5)))
    MINI_CHECK(TOLERANCE.is_close(skew[1], skew[2]))


@MINI_TEST("Line", "Fit Points Uneven")
def test_line_fit_points_uneven():
    from session_py import Line
    from session_py import Point

    line = Line.fit_points(
        [
            Point(0.0, 0.0, 0.0),
            Point(0.0, 1.0, 0.0),
            Point(0.0, 9.0, 0.0),
        ]
    )

    MINI_CHECK(TOLERANCE.is_close(line.length(), 9.0))
    MINI_CHECK(TOLERANCE.is_point_close(line.start(), Point(0.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(line.end(), Point(0.0, 9.0, 0.0)))


@MINI_TEST("Line", "Subdivide")
def test_line_subdivide():
    from session_py import Line

    line = Line(0.0, 0.0, 0.0, 10.0, 0.0, 0.0)
    pts = line.subdivide(3)
    pts_dist = line.subdivide_by_distance(2.5)

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

    line = Line.from_points(Point(0.0, 0.0, 0.0), Point(10.0, 0.0, 0.0))
    line.extend(1.0, 2.0)

    MINI_CHECK(TOLERANCE.is_close(line.start()[0], -1.0))
    MINI_CHECK(TOLERANCE.is_close(line.end()[0], 12.0))


@MINI_TEST("Line", "Extend Keeps Properties")
def test_line_extend_keeps_properties():
    from session_py import Color
    from session_py import Line
    from session_py import Point

    line = Line.from_points(Point(0.0, 0.0, 0.0), Point(10.0, 0.0, 0.0))
    line.name = "beam"
    line.width = 3.0
    line.dash = [2.0, 1.0]
    line.linecolor = Color.red()
    guid = line.guid
    line.extend(1.0, 2.0)

    MINI_CHECK(line.name == "beam")
    MINI_CHECK(line.width == 3.0)
    MINI_CHECK(line.dash == [2.0, 1.0])
    MINI_CHECK(line.linecolor == Color.red())
    MINI_CHECK(line.guid == guid)


@MINI_TEST("Line", "Split At Crossings")
def test_line_split_at_crossings():
    from session_py import Line
    from session_py import Point

    lines = [
        Line.from_points(Point(-2.0, 5.0, 0.0), Point(12.0, 5.0, 0.0)),
        Line.from_points(Point(5.0, 0.0, 0.0), Point(5.0, 10.0, 0.0)),
        Line.from_points(Point(2.0, 0.0, 0.0), Point(8.0, 0.0, 0.0)),
        Line.from_points(Point(0.3, 0.3, 0.0), Point(5.0, 5.0, 0.0)),
    ]
    boundary = [
        Line.from_points(Point(0.0, 0.0, 0.0), Point(10.0, 0.0, 0.0)),
        Line.from_points(Point(10.0, 0.0, 0.0), Point(10.0, 10.0, 0.0)),
        Line.from_points(Point(10.0, 10.0, 0.0), Point(0.0, 10.0, 0.0)),
        Line.from_points(Point(0.0, 10.0, 0.0), Point(0.0, 0.0, 0.0)),
    ]
    split = Line.split_at_crossings(lines, boundary, 0.01, 0.5)
    overlapped = 0

    for i in range(len(split[1])):
        overlapped += 1 if split[1][i] == 2 else 0

    MINI_CHECK(len(split[0]) == 13)
    MINI_CHECK(split[1][0] == 0)
    MINI_CHECK(split[1][2] == 1)
    MINI_CHECK(split[1][4] == 3)
    MINI_CHECK(split[1][5] == 4)
    MINI_CHECK(overlapped == 0)
    MINI_CHECK(TOLERANCE.is_close(split[0][4].start()[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(split[0][4].start()[1], 0.0))


@MINI_TEST("Line", "Split At Crossings Zero Length")
def test_line_split_at_crossings_zero_length():
    from session_py import Line
    from session_py import Point

    lines = [
        Line.from_points(Point(-2.0, 5.0, 0.0), Point(12.0, 5.0, 0.0)),
        Line.from_points(Point(5.0, 5.0, 0.0), Point(5.0, 5.0, 0.0)),
    ]
    boundary = [
        Line.from_points(Point(0.0, 0.0, 0.0), Point(10.0, 0.0, 0.0)),
        Line.from_points(Point(10.0, 0.0, 0.0), Point(10.0, 10.0, 0.0)),
        Line.from_points(Point(10.0, 10.0, 0.0), Point(0.0, 10.0, 0.0)),
        Line.from_points(Point(0.0, 10.0, 0.0), Point(0.0, 0.0, 0.0)),
    ]
    split = Line.split_at_crossings(lines, boundary, 0.01, 0.5)
    zero = 0

    for i in range(len(split[1])):
        zero += 1 if split[1][i] == 1 else 0

    MINI_CHECK(len(split[0]) == 7)
    MINI_CHECK(zero == 0)


if __name__ == "__main__":
    run_all("python")
