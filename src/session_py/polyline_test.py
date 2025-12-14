from .mini_test import MINI_TEST, MINI_CHECK, run_all
from .tolerance import TOLERANCE


@MINI_TEST("Polyline", "constructor")
def test_polyline_constructor():
    from session_py import Polyline
    from session_py import Point
    from session_py import Vector
    from session_py import Color

    # Constructor with points
    p0 = Point(0.0, 0.0, 0.0)
    p1 = Point(1.0, 0.0, 0.0)
    p2 = Point(1.0, 1.0, 0.0)
    p3 = Point(0.0, 1.0, 0.0)
    pl = Polyline([p0, p1, p2, p3])

    # Basic properties
    point_count = len(pl)
    segment_count = pl.segment_count()
    is_empty = pl.is_empty()

    # Get point
    pt = pl.get_point(1)

    # Minimal and Full String Representation
    plstr = str(pl)
    plrepr = repr(pl)

    # Copy (duplicates everything except guid)
    plcopy = pl.duplicate()
    plother = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0)])

    # Translation operators
    pl2 = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0)])
    v = Vector(1.0, 1.0, 1.0)
    pl_add = pl2 + v
    pl_sub = pl2 - v

    # Polyline with custom color and width
    plc = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0)])
    plc.linecolor = Color(255, 0, 0, 255, "red")
    plc.width = 2.5

    MINI_CHECK(pl.name == "my_polyline" and pl.guid != "" and point_count == 4)
    MINI_CHECK(segment_count == 3 and is_empty == False)
    MINI_CHECK(pt[0] == 1.0 and pt[1] == 0.0 and pt[2] == 0.0)
    MINI_CHECK("(0.0, 0.0, 0.0)" in plstr)
    MINI_CHECK("Polyline(my_polyline" in plrepr and "4 points" in plrepr)
    MINI_CHECK(plcopy == plother)
    MINI_CHECK(plcopy.guid != pl.guid)
    MINI_CHECK(pl_add.get_point(0)[0] == 1.0 and pl_add.get_point(0)[1] == 1.0)
    MINI_CHECK(pl_sub.get_point(0)[0] == -1.0 and pl_sub.get_point(0)[1] == -1.0)
    MINI_CHECK(plc.linecolor[0] == 255 and plc.linecolor[1] == 0 and plc.width == 2.5)


@MINI_TEST("Polyline", "transformation")
def test_polyline_transformation():
    from session_py import Polyline
    from session_py import Point
    from session_py import Xform

    pl = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0)])
    pl.xform = Xform.translation(10.0, 0.0, 0.0)
    pl_transformed = pl.transformed()
    pl.transform()

    MINI_CHECK(pl_transformed.get_point(0)[0] == 10.0 and pl_transformed.get_point(1)[0] == 11.0)
    MINI_CHECK(pl.get_point(0)[0] == 10.0 and pl.get_point(1)[0] == 11.0)
    MINI_CHECK(pl.xform == Xform.identity())


@MINI_TEST("Polyline", "json_roundtrip")
def test_polyline_json_roundtrip():
    from session_py import Polyline
    from session_py import Point
    from pathlib import Path

    pl = Polyline([Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0), Point(7.0, 8.0, 9.0), Point(10.0, 11.0, 12.0)])
    pl.name = "test_polyline"

    # json_dump(fname) / json_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "test_polyline.json"
    pl.json_dump(fname)
    loaded = Polyline.json_load(fname)

    MINI_CHECK(loaded.name == "test_polyline")
    MINI_CHECK(len(loaded) == 4)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(1)[1], 5.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(2)[2], 9.0))


@MINI_TEST("Polyline", "protobuf_roundtrip")
def test_polyline_protobuf_roundtrip():
    from session_py import Polyline
    from session_py import Point
    from pathlib import Path

    pl = Polyline([Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0), Point(7.0, 8.0, 9.0), Point(10.0, 11.0, 12.0)])
    pl.name = "test_polyline"

    # protobuf_dump(fname) / protobuf_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "test_polyline.bin"
    pl.protobuf_dump(fname)
    loaded = Polyline.protobuf_load(fname)

    MINI_CHECK(loaded.name == "test_polyline")
    MINI_CHECK(len(loaded) == 4)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(1)[1], 5.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(2)[2], 9.0))


@MINI_TEST("Polyline", "length")
def test_polyline_length():
    from session_py import Polyline
    from session_py import Point

    # L-shaped polyline: 1 unit right, 1 unit up, 1 unit left = 3 units total
    pl = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0)])
    ln = pl.length()
    mag_sq = pl.magnitude_squared()

    MINI_CHECK(TOLERANCE.is_close(ln, 3.0))
    MINI_CHECK(TOLERANCE.is_close(mag_sq, 3.0))


@MINI_TEST("Polyline", "center")
def test_polyline_center():
    from session_py import Polyline
    from session_py import Point

    # Square polyline
    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(2.0, 2.0, 0.0),
        Point(0.0, 2.0, 0.0)
    ])
    c = pl.center()
    cv = pl.center_vec()

    MINI_CHECK(TOLERANCE.is_close(c[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(c[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(c[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(cv[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(cv[1], 1.0))


@MINI_TEST("Polyline", "is_closed")
def test_polyline_is_closed():
    from session_py import Polyline
    from session_py import Point

    # Open polyline
    open_pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0)
    ])
    is_open = open_pl.is_closed()

    # Closed polyline (first and last point same)
    closed_pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 0.0, 0.0)
    ])
    is_closed = closed_pl.is_closed()

    MINI_CHECK(is_open == False)
    MINI_CHECK(is_closed == True)


@MINI_TEST("Polyline", "reverse")
def test_polyline_reverse():
    from session_py import Polyline
    from session_py import Point

    pl = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(2.0, 0.0, 0.0), Point(3.0, 0.0, 0.0)])

    # Test reversed() returns new polyline
    rev = pl.reversed()
    orig_first = pl.get_point(0)[0]
    rev_first = rev.get_point(0)[0]

    # Test reverse() in place
    pl.reverse()
    in_place_first = pl.get_point(0)[0]

    MINI_CHECK(orig_first == 0.0)
    MINI_CHECK(rev_first == 3.0)
    MINI_CHECK(in_place_first == 3.0)


@MINI_TEST("Polyline", "closest_point")
def test_polyline_closest_point():
    from session_py import Polyline
    from session_py import Point

    pl = Polyline([Point(0.0, 0.0, 0.0), Point(2.0, 0.0, 0.0), Point(2.0, 2.0, 0.0), Point(0.0, 2.0, 0.0)])
    test_pt = Point(1.0, 1.0, 0.0)
    distance, edge_id, closest = pl.closest_distance_and_point(test_pt)

    MINI_CHECK(edge_id == 0)
    MINI_CHECK(TOLERANCE.is_close(closest[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(closest[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(distance, 1.0))


if __name__ == "__main__":
    run_all("python")
