from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Polyline", "Constructor")
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

    # Index operator
    pt_idx = pl[1]
    pl_copy = pl.duplicate()
    pl_copy[0] = Point(5.0, 6.0, 7.0)

    # Minimal and Full String Representation
    plstr = str(pl)
    plrepr = repr(pl)

    # Copy (duplicates everything except guid)
    plcopy = pl.duplicate()
    plother = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ])

    # No-copy operators
    plmult = pl.duplicate()
    plmult *= 2.0
    pldiv = pl.duplicate()
    pldiv /= 2.0
    pladd = pl.duplicate()
    pladd += Vector(1.0, 1.0, 1.0)
    plsub = pl.duplicate()
    plsub -= Vector(1.0, 1.0, 1.0)

    # Copy operators
    rmul = pl * 2.0
    rdiv = pl / 2.0
    radd = pl + Vector(1.0, 1.0, 1.0)
    rdif = pl - Vector(1.0, 1.0, 1.0)

    # Negation (reverse point order)
    plneg = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
    ])
    neg = -plneg

    # Polyline with custom color and width
    plc = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ])
    plc.linecolor = Color(255, 0, 0, 255, "red")
    plc.width = 2.5

    MINI_CHECK(pl.name == "my_polyline" and pl.guid != "" and point_count == 4)
    MINI_CHECK(segment_count == 3 and not is_empty)
    MINI_CHECK(pt[0] == 1.0 and pt[1] == 0.0 and pt[2] == 0.0)
    MINI_CHECK(pt_idx[0] == 1.0 and pl_copy[0][0] == 5.0 and pl_copy[0][1] == 6.0)
    MINI_CHECK("(0.0, 0.0, 0.0)" in plstr)
    MINI_CHECK("Polyline(my_polyline" in plrepr and "4 points" in plrepr)
    MINI_CHECK(plcopy == plother)
    MINI_CHECK(plcopy.guid != pl.guid)
    MINI_CHECK(plmult.get_point(1)[0] == 2.0)
    MINI_CHECK(pldiv.get_point(1)[0] == 0.5)
    MINI_CHECK(pladd.get_point(0)[0] == 1.0 and pladd.get_point(0)[1] == 1.0)
    MINI_CHECK(plsub.get_point(0)[0] == -1.0 and plsub.get_point(0)[1] == -1.0)
    MINI_CHECK(rmul.get_point(1)[0] == 2.0)
    MINI_CHECK(rdiv.get_point(1)[0] == 0.5)
    MINI_CHECK(radd.get_point(0)[0] == 1.0 and radd.get_point(0)[1] == 1.0)
    MINI_CHECK(rdif.get_point(0)[0] == -1.0 and rdif.get_point(0)[1] == -1.0)
    MINI_CHECK(neg.get_point(0)[0] == 3.0 and neg.get_point(3)[0] == 0.0)
    MINI_CHECK(plc.linecolor[0] == 255 and plc.linecolor[1] == 0 and plc.width == 2.5)


@MINI_TEST("Polyline", "Transformation")
def test_polyline_transformation():
    from session_py import Polyline
    from session_py import Point
    from session_py import Xform

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ])
    pl.xform = Xform.translation(10.0, 0.0, 0.0)
    pl_transformed = pl.transformed()
    pl.transform()

    MINI_CHECK(pl_transformed.get_point(0)[0] == 10.0 and pl_transformed.get_point(1)[0] == 11.0)
    MINI_CHECK(pl.get_point(0)[0] == 10.0 and pl.get_point(1)[0] == 11.0)
    MINI_CHECK(pl.xform == Xform.identity())


@MINI_TEST("Polyline", "Json Roundtrip")
def test_polyline_json_roundtrip():
    from session_py import Polyline
    from session_py import Point
    from pathlib import Path

    pl = Polyline([
        Point(1.0, 2.0, 3.0),
        Point(4.0, 5.0, 6.0),
        Point(7.0, 8.0, 9.0),
        Point(10.0, 11.0, 12.0),
    ])
    pl.name = "test_polyline"

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   json_dumps()    │ str          │ to JSON string
    #   json_loads(s)   │ str          │ from JSON string
    #   json_dump(path) │ file         │ write to file
    #   json_load(path) │ file         │ read from file

    # json_dump(fname) / json_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_polyline.json"
    pl.json_dump(fname)
    loaded = Polyline.json_load(fname)

    MINI_CHECK(loaded.name == "test_polyline")
    MINI_CHECK(len(loaded) == 4)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(1)[1], 5.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(2)[2], 9.0))


@MINI_TEST("Polyline", "Protobuf Roundtrip")
def test_polyline_protobuf_roundtrip():
    from session_py import Polyline
    from session_py import Point
    from pathlib import Path

    pl = Polyline([
        Point(1.0, 2.0, 3.0),
        Point(4.0, 5.0, 6.0),
        Point(7.0, 8.0, 9.0),
        Point(10.0, 11.0, 12.0),
    ])
    pl.name = "test_polyline"

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    # pb_dump(fname) / pb_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_polyline.bin"
    pl.pb_dump(fname)
    loaded = Polyline.pb_load(fname)

    MINI_CHECK(loaded.name == "test_polyline")
    MINI_CHECK(len(loaded) == 4)
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(1)[1], 5.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.get_point(2)[2], 9.0))


@MINI_TEST("Polyline", "Length")
def test_polyline_length():
    from session_py import Polyline
    from session_py import Point

    # L-shaped polyline: 1 unit right, 1 unit up, 1 unit left = 3 units total
    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ])
    ln = pl.length()
    mag_sq = pl.magnitude_squared()

    MINI_CHECK(TOLERANCE.is_close(ln, 3.0))
    MINI_CHECK(TOLERANCE.is_close(mag_sq, 3.0))


@MINI_TEST("Polyline", "Center")
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

    MINI_CHECK(TOLERANCE.is_close(c[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(c[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(c[2], 0.0))


@MINI_TEST("Polyline", "Is Closed")
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

    MINI_CHECK(not is_open)
    MINI_CHECK(is_closed)


@MINI_TEST("Polyline", "Closed")
def test_polyline_closed():
    from session_py import Polyline
    from session_py import Point

    open_pl = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0)])
    closed_from_open = open_pl.closed()

    closed_pl = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0), Point(0.0, 1.0, 0.0), Point(0.0, 0.0, 0.0)])
    closed_from_closed = closed_pl.closed()

    MINI_CHECK(closed_from_open.point_count() == 5)
    MINI_CHECK(closed_from_open.is_closed())
    MINI_CHECK(closed_from_closed.point_count() == 5)
    MINI_CHECK(closed_from_closed.is_closed())


@MINI_TEST("Polyline", "Reverse")
def test_polyline_reverse():
    from session_py import Polyline
    from session_py import Point

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
    ])

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


@MINI_TEST("Polyline", "Closest Point")
def test_polyline_closest_point():
    from session_py import Polyline
    from session_py import Point

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(2.0, 2.0, 0.0),
        Point(0.0, 2.0, 0.0),
    ])
    test_pt = Point(1.0, 1.0, 0.0)
    distance, edge_id, closest = pl.closest_distance_and_point(test_pt)

    MINI_CHECK(edge_id == 0)
    MINI_CHECK(TOLERANCE.is_close(closest[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(closest[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(distance, 1.0))


@MINI_TEST("Polyline", "Extend Segment")
def test_polyline_extend_segment():
    from session_py import Polyline
    from session_py import Point

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
    ])
    pl.extend_segment(0, 0.5, 0.5)
    first = pl.get_point(0)[0]
    second = pl.get_point(1)[0]

    MINI_CHECK(TOLERANCE.is_close(first, -0.5))
    MINI_CHECK(TOLERANCE.is_close(second, 1.5))


@MINI_TEST("Polyline", "Extend Segment Equally")
def test_polyline_extend_segment_equally():
    from session_py import Polyline
    from session_py import Point

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
    ])
    pl.extend_segment_equally(0, 0.5)
    first = pl.get_point(0)[0]
    second = pl.get_point(1)[0]

    MINI_CHECK(TOLERANCE.is_close(first, -0.5))
    MINI_CHECK(TOLERANCE.is_close(second, 1.5))


@MINI_TEST("Polyline", "Get Points")
def test_polyline_get_points():
    from session_py import Polyline
    from session_py import Point

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ])
    points = pl.get_points()

    MINI_CHECK(len(points) == 4)
    MINI_CHECK(TOLERANCE.is_close(points[0][0], 0.0) and TOLERANCE.is_close(points[0][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(points[1][0], 1.0) and TOLERANCE.is_close(points[1][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(points[2][0], 1.0) and TOLERANCE.is_close(points[2][1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(points[3][0], 0.0) and TOLERANCE.is_close(points[3][1], 1.0))


@MINI_TEST("Polyline", "Get Lines")
def test_polyline_get_lines():
    from session_py import Polyline
    from session_py import Point
    from session_py import Line

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ])
    lines = pl.get_lines()
    lines_prop = pl.lines

    MINI_CHECK(len(lines) == 3)
    MINI_CHECK(len(lines_prop) == 3)
    MINI_CHECK(TOLERANCE.is_close(lines[0][0], 0.0) and TOLERANCE.is_close(lines[0][3], 1.0))
    MINI_CHECK(TOLERANCE.is_close(lines[1][0], 1.0) and TOLERANCE.is_close(lines[1][4], 1.0))
    MINI_CHECK(TOLERANCE.is_close(lines[2][0], 1.0) and TOLERANCE.is_close(lines[2][3], 0.0))


@MINI_TEST("Polyline", "Shift")
def test_polyline_shift():
    from session_py import Polyline
    from session_py import Point

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
    ])
    pl.shift(1)
    first_after_shift = pl.get_point(0)[0]
    pl.shift(-1)
    first_after_unshift = pl.get_point(0)[0]

    MINI_CHECK(TOLERANCE.is_close(first_after_shift, 1.0))
    MINI_CHECK(TOLERANCE.is_close(first_after_unshift, 0.0))


@MINI_TEST("Polyline", "Point At")
def test_polyline_point_at():
    from session_py import Polyline
    from session_py import Point

    start = Point(0.0, 0.0, 0.0)
    end = Point(2.0, 0.0, 0.0)
    mid = Polyline.point_at(start, end, 0.5)
    quarter = Polyline.point_at(start, end, 0.25)

    MINI_CHECK(TOLERANCE.is_close(mid[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(quarter[0], 0.5))


@MINI_TEST("Polyline", "Is Clockwise")
def test_polyline_is_clockwise():
    from session_py import Polyline
    from session_py import Point
    from session_py import Plane

    # Clockwise square (when viewed from +Z)
    cw_pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(1.0, 0.0, 0.0),
    ])
    # Counter-clockwise square
    ccw_pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ])
    plane = Plane()

    MINI_CHECK(cw_pl.is_clockwise(plane) == True)
    MINI_CHECK(ccw_pl.is_clockwise(plane) == False)


@MINI_TEST("Polyline", "Convex Corners")
def test_polyline_convex_corners():
    from session_py import Polyline
    from session_py import Point

    # L-shaped polyline with one concave corner
    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ])
    corners = pl.get_convex_corners()

    MINI_CHECK(len(corners) == 4)
    MINI_CHECK(isinstance(corners[0], bool))


@MINI_TEST("Polyline", "Tween")
def test_polyline_tween():
    from session_py import Polyline
    from session_py import Point

    pl0 = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ])
    pl1 = Polyline([
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
        Point(3.0, 1.0, 0.0),
        Point(2.0, 1.0, 0.0),
    ])
    tweened = Polyline.tween_two_polylines(pl0, pl1, 0.5)

    MINI_CHECK(TOLERANCE.is_close(tweened.get_point(0)[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(tweened.get_point(1)[0], 2.0))


@MINI_TEST("Polyline", "Average Plane")
def test_polyline_average_plane():
    from session_py import Polyline
    from session_py import Point

    pl = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(2.0, 2.0, 0.0),
        Point(0.0, 2.0, 0.0),
    ])
    origin, x_axis, y_axis, z_axis = pl.get_average_plane()
    fast_origin, fast_plane = pl.get_fast_plane()
    avg_normal = pl._average_normal()

    MINI_CHECK(TOLERANCE.is_close(origin[0], 1.0) and TOLERANCE.is_close(origin[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(abs(z_axis[2]), 1.0))
    MINI_CHECK(fast_plane is not None)
    MINI_CHECK(TOLERANCE.is_close(abs(avg_normal[2]), 1.0))


@MINI_TEST("Polyline", "Interpolate Points")
def test_polyline_interpolate_points():
    from session_py import Polyline
    from session_py import Point

    a = Point(0.0, 0.0, 0.0)
    b = Point(4.0, 0.0, 0.0)

    # kind 0: no endpoints — 3 interior points at t=0.25, 0.5, 0.75
    pts0 = Polyline.interpolate_points(a, b, 3, 0)
    # kind 1: both endpoints — 5 points
    pts1 = Polyline.interpolate_points(a, b, 3, 1)
    # kind 2: start only — 4 points (from + 3 interior)
    pts2 = Polyline.interpolate_points(a, b, 3, 2)

    MINI_CHECK(len(pts0) == 3)
    MINI_CHECK(TOLERANCE.is_close(pts0[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pts0[1][0], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pts0[2][0], 3.0))
    MINI_CHECK(len(pts1) == 5)
    MINI_CHECK(TOLERANCE.is_close(pts1[0][0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pts1[4][0], 4.0))
    MINI_CHECK(len(pts2) == 4)
    MINI_CHECK(TOLERANCE.is_close(pts2[0][0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pts2[3][0], 3.0))


@MINI_TEST("Polyline", "Quick Hull")
def test_polyline_quick_hull():
    from session_py import Polyline
    from session_py import Point

    # Square with one interior point — hull should be the 4 corners
    polygon = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.5, 0.5, 0.0),
    ])
    hull = Polyline.quick_hull(polygon)

    MINI_CHECK(hull.point_count() == 4)


@MINI_TEST("Polyline", "Bounding Rectangle")
def test_polyline_bounding_rectangle():
    from session_py import Polyline
    from session_py import Point

    # Axis-aligned 4x3 rectangle
    polygon = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(4.0, 0.0, 0.0),
        Point(4.0, 3.0, 0.0),
        Point(0.0, 3.0, 0.0),
    ])
    rect = Polyline.bounding_rectangle(polygon)

    MINI_CHECK(rect is not None)
    MINI_CHECK(rect.point_count() == 5)
    for p in rect.get_points():
        MINI_CHECK(abs(p[2]) < 1e-6)


@MINI_TEST("Polyline", "Grid Of Points In Polygon")
def test_polyline_grid_of_points():
    from session_py import Polyline
    from session_py import Point

    # 4x4 square polygon
    polygon = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(4.0, 0.0, 0.0),
        Point(4.0, 4.0, 0.0),
        Point(0.0, 4.0, 0.0),
    ])
    pts = Polyline.grid_of_points_in_polygon(polygon, 0.0, 1.0, 100)

    MINI_CHECK(len(pts) > 0)
    for p in pts:
        MINI_CHECK(p[0] > 0.0 and p[0] < 4.0)
        MINI_CHECK(p[1] > 0.0 and p[1] < 4.0)


@MINI_TEST("Polyline", "Simplify Points")
def test_polyline_simplify_points():
    import math
    from session_py import Polyline, Point
    pts = []
    for i in range(100):
        x = float(i)
        y = math.sin(float(i) * 0.1) * 0.001
        z = 0.0
        pts.append(Point(x, y, z))
    result_tight = Polyline.simplify_points(pts, 0.0001)
    result_loose = Polyline.simplify_points(pts, 0.01)
    result_very_loose = Polyline.simplify_points(pts, 1.0)

    MINI_CHECK(len(result_tight) <= len(pts))
    MINI_CHECK(len(result_loose) <= len(result_tight))
    MINI_CHECK(len(result_very_loose) <= len(result_loose))
    MINI_CHECK(result_tight[0][0] == pts[0][0])
    MINI_CHECK(result_tight[-1][0] == pts[-1][0])


@MINI_TEST("Polyline", "Simplify")
def test_polyline_simplify():
    from session_py import Polyline, Point
    pts = []
    for i in range(20):
        x = float(i)
        y = 0.0
        z = 0.0
        pts.append(Point(x, y, z))
    pl = Polyline(pts)
    result = pl.simplify(0.001)

    MINI_CHECK(len(result) == 2)
    MINI_CHECK(TOLERANCE.is_close(result.get_point(0)[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(result.get_point(1)[0], 19.0))


@MINI_TEST("Polyline", "Simplify Collinear")
def test_polyline_simplify_collinear():
    from session_py import Polyline, Point
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 0.0, 0.0),
        Point(4.0, 0.0, 0.0),
    ]
    result = Polyline.simplify_points(pts, 0.001)

    MINI_CHECK(len(result) == 2)
    MINI_CHECK(TOLERANCE.is_close(result[0][0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(result[-1][0], 4.0))


@MINI_TEST("Polyline", "Simplify Zigzag")
def test_polyline_simplify_zigzag():
    from session_py import Polyline, Point
    pts = []
    for i in range(10):
        x = float(i)
        y = 1.0 if (i % 2 == 1) else 0.0
        z = 0.0
        pts.append(Point(x, y, z))
    result_tight = Polyline.simplify_points(pts, 0.1)
    result_loose = Polyline.simplify_points(pts, 2.0)

    MINI_CHECK(len(result_tight) == 10)
    MINI_CHECK(len(result_loose) < len(result_tight))


@MINI_TEST("Polyline", "Simplify Two Points")
def test_polyline_simplify_two_points():
    from session_py import Polyline, Point
    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 1.0),
    ]
    result = Polyline.simplify_points(pts, 0.001)

    MINI_CHECK(len(result) == 2)


if __name__ == "__main__":
    run_all("python")
