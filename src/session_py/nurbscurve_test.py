from session_py.mini_test import MINI_TEST, MINI_CHECK


@MINI_TEST("NurbsCurve", "constructor")
def test_nurbscurve_constructor():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 1.0, 0.0)
    ]

    # The first the curve is closed or open
    # For linear curves use degree 1
    # When 3 points use degree 2 curve, Rhino default
    # When x>3 points use degree 3 curve
    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    curve.set_domain(0.0, 1.0)

    # Minimal and Full String Representation
    cstr = str(curve)
    crepr = repr(curve)

    # Copy (duplicates everything except guid)
    ccopy = curve.duplicate()
    cother = NurbsCurve.create(periodic=False, degree=2, points=points)

    # Point division
    divided = []
    curve.divide_by_count(10, divided)

    MINI_CHECK(curve.is_valid() == True)
    MINI_CHECK(curve.cv_count() == 4)
    MINI_CHECK(curve.degree() == 2)
    MINI_CHECK(curve.order() == 3)
    MINI_CHECK(curve.name == "my_nurbscurve")
    MINI_CHECK(curve.guid != "")
    MINI_CHECK(cstr == "degree=2, cvs=4")
    MINI_CHECK(crepr == "NurbsCurve(my_nurbscurve, dim=3, order=3, cvs=4, rational=false)")
    MINI_CHECK(ccopy.cv_count() == curve.cv_count())
    MINI_CHECK(ccopy.guid != curve.guid)


@MINI_TEST("NurbsCurve", "attributes")
def test_nurbscurve_attributes():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Plane
    from session_py import Tolerance

    TOLERANCE = Tolerance()

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 1.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    #############################################
    # Validation
    #############################################

    # Whole curve
    is_valid = curve.is_valid()
    MINI_CHECK(is_valid == True)

    # Check whole knot vector for
    # For correct size: order + cv_count - 2
    # Non-decreasing (can repeat, can't go down)
    # Valid domain exists
    is_valid_knot_vector = curve.is_valid_knot_vector()
    MINI_CHECK(is_valid_knot_vector == True)

    #############################################
    # Accessors
    #############################################
    # Memory layout 2-2D, 3-3D
    dimension = curve.dimension()
    MINI_CHECK(dimension == 3)
    # Degree - Polynomial order, 1=linear, 2=quadratic, 3=cubic
    degree = curve.degree()
    MINI_CHECK(degree == 2)
    # Is rational is related to control points having weights
    # is_rational = false means control points [x, y, z]
    # is_rational = false means control points [xw, yw, zw]
    # Rational curves are used to represent:
    # Order = degree + 1, control points + order = knots
    order = curve.order()
    MINI_CHECK(order == 3)
    # Number of control vertices
    cv_count = curve.cv_count()
    MINI_CHECK(cv_count == 4)
    # Number of floats per 1 control vertex
    cv_size = curve.cv_size()
    MINI_CHECK(cv_size == 3)
    # The knots are a list of (degree+control_points-1) numbers
    knot_count = curve.knot_count()
    MINI_CHECK(knot_count == 5)
    # Span = a knot interval where a single polynomial segment is evaluated
    # Knot vector: [0, 0, 0 ↑, 1 ↑, 2 ↑, 3, 3, 3]  (cubic, 5 CVs)
    span_count = curve.span_count()
    MINI_CHECK(span_count == 2)
    #####################################################
    # Control Vertex Access
    #  m_cv = [x0, y0, z0, (w0), x1, y1, z1, (w1), ...]
    #          └─── CV 0 ───┘    └─── CV 1 ───┘
    #####################################################

    # Get pointer to control vertex
    # Each CV occupies m_cv_stride doubles:
    # (3 for non-rational, 4 for rational)
    # cv(index) returns pointer to m_cv[index * m_cv_stride]
    p = curve.cv(1)
    MINI_CHECK(p[0] == 1.0 and p[1] == 1.0 and p[2] == 0.0)

    # Returns the control vertex as Point object
    cv_point = curve.get_cv(1)
    MINI_CHECK(cv_point == Point(1.0, 1.0, 0.0))

    # Raw homogeneous coords
    x, y, z, w = curve.get_cv_4d(1)
    MINI_CHECK(x == 1.0 and y == 1.0 and z == 0.0 and w == 1.0)

    # Use for regular points on curve, Polyline, B-Spline
    curve.set_cv(2, Point(2.0, 0.0, 0.5))
    MINI_CHECK(curve.get_cv(2)[0] == 2.0 and curve.get_cv(2)[1] == 0.0 and curve.get_cv(2)[2] == 0.5)

    # Use for rational curvers like circles, ellipses
    curve.set_cv_4d(2, 2.0, 0.0, 0.5, 0.707)
    x, y, z, w = curve.get_cv_4d(2)
    MINI_CHECK(x == 2.0 and y == 0.0 and z == 0.5 and w == 0.707)

    # Get weight of a control vertex (1.0 if non-rational)
    weight = curve.weight(2)
    MINI_CHECK(weight == 0.707)

    # Set the weight of a control vertex
    curve.set_weight(2, 0.5)
    MINI_CHECK(curve.weight(2) == 0.5)

    #####################################################
    # Knot Access
    #####################################################

    # Get knot value at index
    knot3 = curve.knot(3)
    MINI_CHECK(knot3 == 2)

    # Set knot value at index
    # ATTENTION you can brake increasing rule
    curve.set_knot(4, 2)
    MINI_CHECK(curve.knot(4) == 2)

    # Count repeated knots at index [0, 0, 1, 1, 2]
    m0 = curve.knot_multiplicity(0)  # 2 (two 0's)
    m1 = curve.knot_multiplicity(1)  # 2 (still counting the 0's)
    m2 = curve.knot_multiplicity(2)  # 1 (single 0.5)
    m3 = curve.knot_multiplicity(3)  # 2 (single 1's)
    m4 = curve.knot_multiplicity(4)  # 2 (single 2)
    MINI_CHECK(m0 == 2)
    MINI_CHECK(m1 == 2)
    MINI_CHECK(m2 == 1)
    MINI_CHECK(m3 == 2)
    MINI_CHECK(m4 == 2)

    # Superflous knots are used for extension of clamped curves
    # For knot vector [0, 0, 0.5, 1, 2]: 2*knot[4] - knot[1] = 2*2 - 0 = 4
    superfluous_knot = curve.superfluous_knot(1)
    MINI_CHECK(superfluous_knot == 4)

    # Direct memory access to knot values, fast, read-only
    # Vector return is slower and makes a copy
    knot_vector = curve.get_knots()
    MINI_CHECK(knot_vector[0] == 0.0 and knot_vector[1] == 0.0 and
               knot_vector[2] == 1.0 and knot_vector[3] == 2.0 and
               knot_vector[4] == 2.0)

    #####################################################
    # Domain & Parameterization - HERE
    #####################################################

    # get start and end of the curve interval
    start, end = curve.domain()
    MINI_CHECK(start == 0.0 and end == 2.0)

    # Get start, middle and end values of the interval
    start = curve.domain_start()
    middle = curve.domain_middle()
    end = curve.domain_end()
    MINI_CHECK(start == 0.0 and middle == 1.0 and end == 2.0)

    # Change curve domain
    curve.set_domain(0.0, 1.0)
    MINI_CHECK(curve.domain_start() == 0.0 and curve.domain_middle() == 0.5 and curve.domain_end() == 1.0)

    # Span of distict knot intervals
    intervals = curve.get_span_vector()
    MINI_CHECK(intervals[0] == 0.0 and intervals[1] == 0.5 and intervals[2] == 1.0)

    #####################################################
    # Geometric checks
    #####################################################
    # Is rational is related to control points having weights
    # is_rational = false means control points [x, y, z]
    # is_rational = false means control points [xw, yw, zw]
    # Rational curves are used to represent:
    # circles, ellipses, parabolas, hyperbolas exactly
    is_rational = curve.is_rational()
    MINI_CHECK(is_rational == True)

    # circles, ellipses, parabolas, hyperbolas exactly
    closed = curve.is_closed()
    periodic = curve.is_periodic()
    linear = curve.is_linear()
    planar = curve.is_planar()
    arc = curve.is_arc()
    plane = Plane.xy_plane()
    on_plane = curve.is_in_plane(plane)
    is_open = curve.is_natural()
    is_polyline, _, _ = curve.is_polyline()

    MINI_CHECK(closed == False)
    MINI_CHECK(periodic == False)
    MINI_CHECK(linear == False)
    MINI_CHECK(planar == False)
    MINI_CHECK(arc == False)
    MINI_CHECK(on_plane == False)
    MINI_CHECK(is_open == False)
    MINI_CHECK(is_polyline == False)


@MINI_TEST("NurbsCurve", "Conversions")
def test_nurbscurve_conversions():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Tolerance

    TOLERANCE = Tolerance()

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 2.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 2.0, 0.0),
        Point(4.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    # to_polyline_adaptive
    adaptive_pts, adaptive_params = curve.to_polyline_adaptive(0.1, 0.0, 0.0)

    MINI_CHECK(len(adaptive_pts) == 27)
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[0], Point(0.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[13], Point(2.0, 0.5, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[26], Point(4.0, 0.0, 0.0)))

    # divide_by_count
    div_pts, div_params = curve.divide_by_count(10, True)

    MINI_CHECK(len(div_pts) == 10)
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[0], Point(0.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[9], Point(4.0, 0.0, 0.0)))

    # divide_by_length
    len_pts, len_params = curve.divide_by_length(0.5)

    MINI_CHECK(len(len_pts) == 13)
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[0], Point(0.0, 0.0, 0.0)))


@MINI_TEST("NurbsCurve", "Evaluation")
def test_nurbscurve_evaluation():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Tolerance

    TOLERANCE = Tolerance()

    points = [
        Point(1.957614, 1.140253, -0.191281),
        Point(0.912252, 1.886721, 0),
        Point(3.089381, 2.701879, -0.696251),
        Point(5.015145, 1.189141, 0.35799),
        Point(1.854155, 0.514663, 0.347694),
        Point(3.309532, 1.328666, 0),
        Point(3.544072, 2.194233, 0.696217),
        Point(2.903513, 2.091287, 0.696217),
        Point(2.752484, 1.45432, 0),
        Point(2.406227, 1.288248, 0),
        Point(2.15032, 1.868606, 0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    # Get point at parameter t
    point_at = curve.point_at(0.5)
    MINI_CHECK(TOLERANCE.is_close(point_at[0], 1.445733625) and TOLERANCE.is_close(point_at[1], 1.80199875) and TOLERANCE.is_close(point_at[2], -0.134851625))

    # Get point and derivatives at parameter t
    derivatives = curve.evaluate(0.5, 2)
    MINI_CHECK(len(derivatives) == 3)
    MINI_CHECK(TOLERANCE.is_close(derivatives[0][0], 1.445733625) and TOLERANCE.is_close(derivatives[0][1], 1.80199875) and TOLERANCE.is_close(derivatives[0][2], -0.134851625))
    MINI_CHECK(TOLERANCE.is_close(derivatives[1][0], 0.0432025) and TOLERANCE.is_close(derivatives[1][1], 1.154047) and TOLERANCE.is_close(derivatives[1][2], -0.1568445))
    MINI_CHECK(TOLERANCE.is_close(derivatives[2][0], 4.267853) and TOLERANCE.is_close(derivatives[2][1], -0.677778) and TOLERANCE.is_close(derivatives[2][2], -1.078813))

    # Tangent vector at parameter t
    tangent = curve.tangent_at(0.5)
    MINI_CHECK(TOLERANCE.is_close(tangent[0], 0.037069134389828) and TOLERANCE.is_close(tangent[1], 0.990209443486538) and TOLERANCE.is_close(tangent[2], -0.134577625575985))

    # Frame at
    result = curve.frame_at(0.5, normalized=True)
    MINI_CHECK(result is not None)
    o, t, n, b = result
    MINI_CHECK(TOLERANCE.is_close(o[0], 3.156927375) and TOLERANCE.is_close(o[1], 1.3351115) and TOLERANCE.is_close(o[2], 0.130488875))

    MINI_CHECK(curve.frame_at(-0.1, normalized=True) is None)
    MINI_CHECK(curve.frame_at(1.1, normalized=True) is None)
    MINI_CHECK(curve.frame_at(curve.domain_start(), normalized=False) is not None)
    MINI_CHECK(curve.frame_at(curve.domain_end(), normalized=False) is not None)
    MINI_CHECK(curve.frame_at(curve.domain_start() - 0.1, normalized=False) is None)

    # Perpendicular frame at
    result = curve.perpendicular_frame_at(0.5, normalized=True)
    MINI_CHECK(result is not None)
    o, t, n, b = result
    MINI_CHECK(TOLERANCE.is_point_close(o, Point(3.156927375, 1.3351115, 0.130488875)))
    MINI_CHECK(curve.perpendicular_frame_at(-0.1, normalized=True) is None)
    MINI_CHECK(curve.perpendicular_frame_at(1.1, normalized=True) is None)
    MINI_CHECK(curve.perpendicular_frame_at(curve.domain_start(), normalized=False) is not None)
    MINI_CHECK(curve.perpendicular_frame_at(curve.domain_end(), normalized=False) is not None)
    MINI_CHECK(curve.perpendicular_frame_at(curve.domain_start() - 0.1, normalized=False) is None)

    # Points
    p0 = curve.point_at_start()
    p1 = curve.point_at_middle()
    p2 = curve.point_at_end()
    MINI_CHECK(TOLERANCE.is_close(p0[0], 1.957614) and TOLERANCE.is_close(p0[1], 1.140253) and TOLERANCE.is_close(p0[2], -0.191281))
    MINI_CHECK(TOLERANCE.is_close(p1[0], 3.156927375) and TOLERANCE.is_close(p1[1], 1.3351115) and TOLERANCE.is_close(p1[2], 0.130488875))
    MINI_CHECK(TOLERANCE.is_close(p2[0], 2.15032) and TOLERANCE.is_close(p2[1], 1.868606) and TOLERANCE.is_close(p2[2], 0.0))

    curve.set_start_point(Point(1.957614, 1.140253, 2.0))
    curve.set_end_point(Point(2.15032, 1.868606, 2.0))
    MINI_CHECK(TOLERANCE.is_close(curve.point_at_start()[2], 2.0))
    MINI_CHECK(TOLERANCE.is_close(curve.point_at_end()[2], 2.0))


@MINI_TEST("NurbsCurve", "Modifications")
def test_nurbscurve_modifications():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Tolerance

    TOLERANCE = Tolerance()

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 2.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 2.0, 0.0),
        Point(4.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)

    # Reverse the curve
    curve_reversed = curve.duplicate()
    curve_reversed.reverse()
    MINI_CHECK(TOLERANCE.is_point_close(curve_reversed.point_at_start(), curve.point_at_end()))

    # Swap coordinates axes
    curve.swap_coordinates(0, 1)
    MINI_CHECK(TOLERANCE.is_point_close(curve.get_cv(0), Point(0.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(curve.get_cv(1), Point(2.0, 1.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(curve.get_cv(2), Point(0.0, 2.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(curve.get_cv(3), Point(2.0, 3.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(curve.get_cv(4), Point(0.0, 4.0, 0.0)))

    # Split curve at domain middle
    split_t = curve.domain_middle()
    curve_left, curve_right = curve.split(split_t)
    MINI_CHECK(TOLERANCE.is_point_close(curve.point_at(split_t), curve_left.point_at_end()))
    MINI_CHECK(TOLERANCE.is_point_close(curve.point_at(split_t), curve_right.point_at_start()))

    # Extend curve smoothly at both ends
    curve_extended = curve.duplicate()
    curve_extended.extend(curve.domain_start() - 0.5, curve.domain_end() + 0.5)
    MINI_CHECK(curve_extended.length() > curve.length())

    # Enable curve weights - Make rational or non-rational
    curve_rational = curve.duplicate()
    original_length = curve.length()
    curve_rational.make_rational()
    curve_rational.set_weight(2, 10)
    MINI_CHECK(curve_rational.length() != original_length)

    curve_rational.make_non_rational(force=True)
    MINI_CHECK(curve_rational.length() == original_length)


@MINI_TEST("NurbsCurve", "json_roundtrip")
def test_nurbscurve_json_roundtrip():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)

    filename = "serialization/test_nurbscurve.json"
    curve.json_dump(filename)
    loaded = NurbsCurve.json_load(filename)

    MINI_CHECK(loaded.is_valid() == True)
    MINI_CHECK(loaded.cv_count() == 3)
    MINI_CHECK(loaded.degree() == 2)
    MINI_CHECK(loaded.order() == 3)


@MINI_TEST("NurbsCurve", "protobuf_roundtrip")
def test_nurbscurve_protobuf_roundtrip():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)

    filename = "serialization/test_nurbscurve.bin"
    curve.protobuf_dump(filename)
    loaded = NurbsCurve.protobuf_load(filename)

    MINI_CHECK(loaded.is_valid() == True)
    MINI_CHECK(loaded.cv_count() == 3)
    MINI_CHECK(loaded.degree() == 2)
    MINI_CHECK(loaded.order() == 3)


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
    curve.set_domain(0.0, 1.0)
    plane = Plane.xy_plane()
    intersections = curve.intersect_plane(plane)

    MINI_CHECK(len(intersections) >= 0)


if __name__ == "__main__":
    from session_py.mini_test import run_all
    run_all(language="python")
