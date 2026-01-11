from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("NurbsCurve", "constructor")
def test_nurbscurve_constructor():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0, 0, 0),
        Point(1, 1, 0),
        Point(2, 0, 0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    # Minimal and Full String Representation
    cstr = str(curve)
    crepr = repr(curve)

    # Copy (duplicates everything except guid)
    ccopy = curve.duplicate()
    cother = NurbsCurve.create(periodic=False, degree=2, points=points)

    MINI_CHECK(curve.is_valid() == True)
    MINI_CHECK(curve.cv_count() == 3)
    MINI_CHECK(curve.degree() == 2)
    MINI_CHECK(curve.order() == 3)
    MINI_CHECK(curve.name == "nurbscurve")
    MINI_CHECK(curve.guid)
    MINI_CHECK(cstr == "degree=2, cvs=3")
    MINI_CHECK(crepr == "NurbsCurve(nurbscurve, dim=3, order=3, cvs=3, rational=false)")
    MINI_CHECK(ccopy.cv_count() == curve.cv_count())
    MINI_CHECK(ccopy.guid != curve.guid)


@MINI_TEST("NurbsCurve", "is_valid")
def test_nurbscurve_is_valid():
    from session_py import NurbsCurve
    from session_py import Point

    curve_invalid = NurbsCurve()

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]
    curve_valid = NurbsCurve.create(periodic=False, degree=2, points=points)

    MINI_CHECK(curve_invalid.is_valid() == False)
    MINI_CHECK(curve_valid.is_valid() == True)


@MINI_TEST("NurbsCurve", "get_cv")
def test_nurbscurve_get_cv():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    cv0 = curve.get_cv(0)
    cv1 = curve.get_cv(1)
    cv2 = curve.get_cv(2)

    MINI_CHECK(abs(cv0[0] - 0.0) < 0.01)
    MINI_CHECK(abs(cv1[0] - 1.0) < 0.01)
    MINI_CHECK(abs(cv2[0] - 2.0) < 0.01)


@MINI_TEST("NurbsCurve", "set_cv")
def test_nurbscurve_set_cv():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    curve.set_cv(1, Point(1.5, 2.0, 0.0))
    cv1 = curve.get_cv(1)

    MINI_CHECK(abs(cv1[0] - 1.5) < 0.01)
    MINI_CHECK(abs(cv1[1] - 2.0) < 0.01)


@MINI_TEST("NurbsCurve", "point_at")
def test_nurbscurve_point_at():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    t0, t1 = curve.domain()
    t_mid = (t0 + t1) / 2.0
    pt_mid = curve.point_at(t_mid)

    MINI_CHECK(pt_mid[0] > 0.0)
    MINI_CHECK(pt_mid[0] < 2.0)


@MINI_TEST("NurbsCurve", "point_at_start")
def test_nurbscurve_point_at_start():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    pt_start = curve.point_at_start()

    MINI_CHECK(abs(pt_start[0] - 0.0) < 0.01)
    MINI_CHECK(abs(pt_start[1] - 0.0) < 0.01)


@MINI_TEST("NurbsCurve", "point_at_end")
def test_nurbscurve_point_at_end():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    pt_end = curve.point_at_end()

    MINI_CHECK(abs(pt_end[0] - 2.0) < 0.01)
    MINI_CHECK(abs(pt_end[1] - 0.0) < 0.01)


@MINI_TEST("NurbsCurve", "domain")
def test_nurbscurve_domain():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    t0, t1 = curve.domain()

    MINI_CHECK(t0 < t1)


@MINI_TEST("NurbsCurve", "is_closed")
def test_nurbscurve_is_closed():
    from session_py import NurbsCurve
    from session_py import Point

    points_open = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]
    curve_open = NurbsCurve.create(periodic=False, degree=2, points=points_open)

    points_closed = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 0.0)
    ]
    curve_closed = NurbsCurve.create(periodic=False, degree=3, points=points_closed)

    MINI_CHECK(curve_open.is_closed() == False)
    MINI_CHECK(curve_closed.is_closed() == True)


@MINI_TEST("NurbsCurve", "length")
def test_nurbscurve_length():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=1, points=points)
    length = curve.length()

    MINI_CHECK(abs(length - 1.0) < 0.01)


@MINI_TEST("NurbsCurve", "reverse")
def test_nurbscurve_reverse():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    pt_start_before = curve.point_at_start()
    pt_end_before = curve.point_at_end()
    curve.reverse()
    pt_start_after = curve.point_at_start()
    pt_end_after = curve.point_at_end()

    MINI_CHECK(abs(pt_start_before[0] - pt_end_after[0]) < 0.01)
    MINI_CHECK(abs(pt_end_before[0] - pt_start_after[0]) < 0.01)


@MINI_TEST("NurbsCurve", "make_rational")
def test_nurbscurve_make_rational():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    MINI_CHECK(curve.is_rational() == False)
    curve.make_rational()
    MINI_CHECK(curve.is_rational() == True)


@MINI_TEST("NurbsCurve", "tangent_at")
def test_nurbscurve_tangent_at():
    from session_py import NurbsCurve
    from session_py import Point
    import math

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    t0, t1 = curve.domain()
    t_mid = (t0 + t1) / 2.0
    tangent = curve.tangent_at(t_mid)
    mag = math.sqrt(tangent[0]*tangent[0] + tangent[1]*tangent[1] + tangent[2]*tangent[2])

    MINI_CHECK(mag > 0.5)


@MINI_TEST("NurbsCurve", "knot_count")
def test_nurbscurve_knot_count():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    knot_count = curve.knot_count()

    MINI_CHECK(knot_count == curve.order() + curve.cv_count() - 2)


@MINI_TEST("NurbsCurve", "cv_size")
def test_nurbscurve_cv_size():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    MINI_CHECK(curve.cv_size() == 3)
    curve.make_rational()
    MINI_CHECK(curve.cv_size() == 4)


@MINI_TEST("NurbsCurve", "weight")
def test_nurbscurve_weight():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    curve.make_rational()
    w = curve.weight(0)
    curve.set_weight(1, 2.0)
    w1 = curve.weight(1)

    MINI_CHECK(abs(w - 1.0) < 0.01)
    MINI_CHECK(abs(w1 - 2.0) < 0.01)


@MINI_TEST("NurbsCurve", "is_linear")
def test_nurbscurve_is_linear():
    from session_py import NurbsCurve
    from session_py import Point

    points_linear = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0)
    ]
    curve_linear = NurbsCurve.create(periodic=False, degree=1, points=points_linear)

    points_curved = [
        Point(0.0, 0.0, 0.0),
        Point(0.5, 1.0, 0.0),
        Point(1.0, 0.0, 0.0)
    ]
    curve_curved = NurbsCurve.create(periodic=False, degree=2, points=points_curved)

    MINI_CHECK(curve_linear.is_linear() == True)
    MINI_CHECK(curve_curved.is_linear() == False)


@MINI_TEST("NurbsCurve", "json_roundtrip")
def test_nurbscurve_json_roundtrip():
    from session_py import NurbsCurve
    from session_py import Point
    from pathlib import Path

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbscurve.json"
    curve.json_dump(fname)
    loaded = NurbsCurve.json_load(fname)

    MINI_CHECK(loaded.is_valid() == True)
    MINI_CHECK(loaded.cv_count() == 3)
    MINI_CHECK(loaded.degree() == 2)
    MINI_CHECK(loaded.order() == 3)


@MINI_TEST("NurbsCurve", "protobuf_roundtrip")
def test_nurbscurve_protobuf_roundtrip():
    from session_py import NurbsCurve
    from session_py import Point
    from pathlib import Path

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbscurve.bin"
    curve.protobuf_dump(path)
    loaded = NurbsCurve.protobuf_load(path)

    MINI_CHECK(loaded.is_valid() == True)
    MINI_CHECK(loaded.cv_count() == 3)
    MINI_CHECK(loaded.degree() == 2)
    MINI_CHECK(loaded.order() == 3)


@MINI_TEST("NurbsCurve", "degree")
def test_nurbscurve_degree():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    MINI_CHECK(curve.degree() == 2)
    MINI_CHECK(curve.order() == 3)


@MINI_TEST("NurbsCurve", "dimension")
def test_nurbscurve_dimension():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    MINI_CHECK(curve.dimension() == 3)


@MINI_TEST("NurbsCurve", "is_rational")
def test_nurbscurve_is_rational():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    MINI_CHECK(curve.is_rational() == False)
    curve.make_rational()
    MINI_CHECK(curve.is_rational() == True)


@MINI_TEST("NurbsCurve", "set_weight")
def test_nurbscurve_set_weight():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    curve.make_rational()

    MINI_CHECK(abs(curve.weight(1) - 1.0) < 0.01)
    curve.set_weight(1, 2.0)
    MINI_CHECK(abs(curve.weight(1) - 2.0) < 0.01)


@MINI_TEST("NurbsCurve", "knot")
def test_nurbscurve_knot():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    knot0 = curve.knot(0)
    knot1 = curve.knot(1)

    MINI_CHECK(abs(knot0 - 0.0) < 0.01)
    MINI_CHECK(knot1 >= knot0)


@MINI_TEST("NurbsCurve", "set_knot")
def test_nurbscurve_set_knot():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    result = curve.set_knot(0, 0.5)

    MINI_CHECK(result == True)
    MINI_CHECK(abs(curve.knot(0) - 0.5) < 0.01)


@MINI_TEST("NurbsCurve", "set_domain")
def test_nurbscurve_set_domain():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    result = curve.set_domain(0.0, 10.0)
    t0, t1 = curve.domain()

    MINI_CHECK(result == True)
    MINI_CHECK(abs(t0 - 0.0) < 0.01)
    MINI_CHECK(abs(t1 - 10.0) < 0.01)


@MINI_TEST("NurbsCurve", "span_count")
def test_nurbscurve_span_count():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    spans = curve.span_count()

    MINI_CHECK(spans == 1)


@MINI_TEST("NurbsCurve", "get_span_vector")
def test_nurbscurve_get_span_vector():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 1.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    spans = curve.get_span_vector()

    MINI_CHECK(len(spans) >= 2)


@MINI_TEST("NurbsCurve", "evaluate")
def test_nurbscurve_evaluate():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    t0, t1 = curve.domain()
    t_mid = (t0 + t1) / 2.0
    result = curve.evaluate(t_mid, 1)

    MINI_CHECK(len(result) >= 1)


@MINI_TEST("NurbsCurve", "is_periodic")
def test_nurbscurve_is_periodic():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    MINI_CHECK(curve.is_periodic() == False)


@MINI_TEST("NurbsCurve", "make_non_rational")
def test_nurbscurve_make_non_rational():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    curve.make_rational()

    MINI_CHECK(curve.is_rational() == True)
    curve.make_non_rational()
    MINI_CHECK(curve.is_rational() == False)


@MINI_TEST("NurbsCurve", "divide_by_count")
def test_nurbscurve_divide_by_count():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    pts, params = curve.divide_by_count(5, include_endpoints=True)

    MINI_CHECK(len(pts) == 5)
    MINI_CHECK(len(params) == 5)


@MINI_TEST("NurbsCurve", "intersect_plane")
def test_nurbscurve_intersect_plane():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Plane

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    plane = Plane.xy_plane()
    intersections = curve.intersect_plane(plane)

    MINI_CHECK(isinstance(intersections, list))


@MINI_TEST("NurbsCurve", "create_interpolated")
def test_nurbscurve_create_interpolated():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Tolerance

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 2.0, 0.0),
        Point(3.0, 1.0, 0.0),
        Point(4.0, 3.0, 0.0),
        Point(6.0, 0.0, 0.0)
    ]

    c = NurbsCurve.create_interpolated(points, degree=3, closed=False, knot_style=1)
    t0, t1 = c.domain()

    p0 = c.point_at(t0)
    p1 = c.point_at(t1)

    tol = Tolerance()
    MINI_CHECK(tol.is_close(p0.x, 0.0))
    MINI_CHECK(tol.is_close(p0.y, 0.0))
    MINI_CHECK(tol.is_close(p1.x, 6.0))
    MINI_CHECK(tol.is_close(p1.y, 0.0))


if __name__ == "__main__":
    run_all(language="python")
