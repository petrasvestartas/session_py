from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


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
    curve.set_domain(0.0, 1.0)

    # Minimal and Full String Representation
    cstr = str(curve)
    crepr = repr(curve)

    # Copy (duplicates everything except guid)
    ccopy = curve.duplicate()
    cother = NurbsCurve.create(periodic=False, degree=2, points=points)

    # Point division
    divided, _ = curve.divide_by_count(10)

    MINI_CHECK(curve.is_valid() == True)
    MINI_CHECK(curve.cv_count() == 4)
    MINI_CHECK(curve.degree() == 2)
    MINI_CHECK(curve.order() == 3)
    MINI_CHECK(curve.name == "my_nurbscurve")
    MINI_CHECK(curve.guid)
    MINI_CHECK(cstr == "degree=2, cvs=4")
    MINI_CHECK(crepr == "NurbsCurve(my_nurbscurve, dim=3, order=3, cvs=4, rational=false)")
    MINI_CHECK(ccopy.cv_count() == curve.cv_count())
    MINI_CHECK(ccopy.guid != curve.guid)


@MINI_TEST("NurbsCurve", "attributes")
def test_nurbscurve_attributes():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Plane

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
    # Knot vector: [0, 0, 0 ^, 1 ^, 2 ^, 3, 3, 3]  (cubic, 5 CVs)
    span_count = curve.span_count()
    MINI_CHECK(span_count == 2)
    #####################################################
    # Control Vertex Access
    #  m_cv = [x0, y0, z0, (w0), x1, y1, z1, (w1), ...]
    #          --- CV 0 ---      --- CV 1 ---
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
    knots = curve.knot_array()
    k0 = knots[0]
    knot_vector = curve.get_knots()
    MINI_CHECK(k0 == 0.0)
    MINI_CHECK(knot_vector[0] == 0.0 and knot_vector[1] == 0.0 and
               knot_vector[2] == 1.0 and knot_vector[3] == 2.0 and
               knot_vector[4] == 2.0)

    # Control vertex array access
    cvs = curve.cv_array()
    cx0 = cvs[0]
    MINI_CHECK(cx0 == 0.0)

    #####################################################
    # Domain & Parameterization - HERE
    #####################################################

    # get start and end of the curve interval
    interval = curve.domain()
    start = interval[0]
    end = interval[1]
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
    adaptive_pts, adaptive_params = curve.to_polyline_adaptive(angle_tolerance=0.1, min_edge_length=0.0, max_edge_length=0.0)

    MINI_CHECK(len(adaptive_pts) == 27)
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[0], Point(0.000000000000000, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[1], Point(0.183105468750000, 0.348632812500000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[2], Point(0.357421875000000, 0.644531250000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[3], Point(0.679687500000000, 1.078125000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[4], Point(0.966796875000000, 1.300781250000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[5], Point(1.097167968750000, 1.333007812500000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[6], Point(1.159057617187500, 1.329345703125000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[7], Point(1.218750000000000, 1.312500000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[8], Point(1.331542968750000, 1.239257812500000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[9], Point(1.435546875000000, 1.113281250000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[10], Point(1.625000000000000, 0.781250000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[11], Point(1.812500000000000, 0.570312500000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[12], Point(1.906250000000000, 0.517578125000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[13], Point(2.000000000000000, 0.500000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[14], Point(2.093750000000000, 0.517578125000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[15], Point(2.187500000000000, 0.570312500000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[16], Point(2.375000000000000, 0.781250000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[17], Point(2.564453125000000, 1.113281250000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[18], Point(2.668457031250000, 1.239257812500000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[19], Point(2.781250000000000, 1.312500000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[20], Point(2.840942382812500, 1.329345703125000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[21], Point(2.902832031250000, 1.333007812500000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[22], Point(3.033203125000000, 1.300781250000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[23], Point(3.320312500000000, 1.078125000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[24], Point(3.642578125000000, 0.644531250000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[25], Point(3.816894531250000, 0.348632812500000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[26], Point(4.000000000000000, 0.000000000000000, 0.000000000000000)))

    # divide_by_count
    div_pts, div_params = curve.divide_by_count(10, include_endpoints=True)

    MINI_CHECK(len(div_pts) == 10)
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[0], Point(0.000000000000000, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[1], Point(0.328571015882635, 0.598213506310667, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[2], Point(0.740744941524856, 1.140321234797829, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[3], Point(1.338523997492639, 1.232716041998164, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[4], Point(1.712929663130383, 0.664818756620870, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[5], Point(2.287070327006695, 0.664818745295462, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[6], Point(2.661475993133979, 1.232716033043460, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[7], Point(3.259255052521522, 1.140321240507253, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[8], Point(3.671428981912368, 0.598213509892612, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[9], Point(4.000000000000000, 0.000000000000000, 0.000000000000000)))

    # divide_by_length
    len_pts, len_params = curve.divide_by_length(0.5)

    MINI_CHECK(len(len_pts) == 13)
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[0], Point(0.000000000000000, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[1], Point(0.235272731384047, 0.441110443734231, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[2], Point(0.504276692145966, 0.862299318703470, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[3], Point(0.843085062978891, 1.227533014827472, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[4], Point(1.302050970444518, 1.264156212040698, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[5], Point(1.579813544869556, 0.853113314150178, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[6], Point(1.928691287815458, 0.510169864866836, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[7], Point(2.340857741884085, 0.732368000404634, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[8], Point(2.597735401548903, 1.160594587288875, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[9], Point(3.032790392631424, 1.300960469420597, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[10], Point(3.407806728972739, 0.976991467650206, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[11], Point(3.691337413616094, 0.565615072909225, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[12], Point(3.934494402948975, 0.128829830906625, 0.000000000000000)))


@MINI_TEST("NurbsCurve", "frame_at")
def test_nurbscurve_frame_at():
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

    # normalized=true (default): t in [0,1] mapped to domain
    o, t, n, b = curve.frame_at(0.5, normalized=True)

    MINI_CHECK(TOLERANCE.is_close(o[0], 3.156927375000000) and TOLERANCE.is_close(o[1], 1.335111500000000) and TOLERANCE.is_close(o[2], 0.130488875000000))
    MINI_CHECK(TOLERANCE.is_close(t[0], 0.701806140304030) and TOLERANCE.is_close(t[1], 0.697509131556264) and TOLERANCE.is_close(t[2], 0.144738221721788))
    MINI_CHECK(TOLERANCE.is_close(n[0], -0.513930504714161) and TOLERANCE.is_close(n[1], 0.355053088776962) and TOLERANCE.is_close(n[2], 0.780905077761815))
    MINI_CHECK(TOLERANCE.is_close(b[0], 0.493298669931115) and TOLERANCE.is_close(b[1], -0.622429365908747) and TOLERANCE.is_close(b[2], 0.607649657861031))

    MINI_CHECK(curve.frame_at(-0.1, normalized=True) is None)
    MINI_CHECK(curve.frame_at(1.1, normalized=True) is None)
    MINI_CHECK(curve.frame_at(curve.domain_start(), normalized=False) is not None)
    MINI_CHECK(curve.frame_at(curve.domain_end(), normalized=False) is not None)
    MINI_CHECK(curve.frame_at(curve.domain_start() - 0.1, normalized=False) is None)


@MINI_TEST("NurbsCurve", "perpendicular_frame_at")
def test_nurbscurve_perpendicular_frame_at():
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

    # RMF with Frenet initialization (matches Rhino)
    o, t, n, b = curve.perpendicular_frame_at(0.5, normalized=True)
    MINI_CHECK(TOLERANCE.is_point_close(o, Point(3.156927, 1.335111, 0.130489)))
    MINI_CHECK(TOLERANCE.is_close(t[0], 0.632708) and TOLERANCE.is_close(t[1], -0.703687) and TOLERANCE.is_close(t[2], 0.323272))
    MINI_CHECK(TOLERANCE.is_close(n[0], 0.327335) and TOLERANCE.is_close(n[1], -0.135297) and TOLERANCE.is_close(n[2], -0.935172))
    MINI_CHECK(TOLERANCE.is_close(b[0], 0.701806) and TOLERANCE.is_close(b[1], 0.697509) and TOLERANCE.is_close(b[2], 0.144738))
    MINI_CHECK(curve.perpendicular_frame_at(-0.1, normalized=True) is None)
    MINI_CHECK(curve.perpendicular_frame_at(1.1, normalized=True) is None)
    MINI_CHECK(curve.perpendicular_frame_at(curve.domain_start(), normalized=False) is not None)
    MINI_CHECK(curve.perpendicular_frame_at(curve.domain_end(), normalized=False) is not None)
    MINI_CHECK(curve.perpendicular_frame_at(curve.domain_start() - 0.1, normalized=False) is None)



@MINI_TEST("NurbsCurve", "is_valid")
def test_nurbscurve_is_valid():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0)
    ]

    curve_0 = NurbsCurve.create(periodic=False, degree=2, points=points)
    curve_1 = NurbsCurve()
    is_valid_0 = curve_0.is_valid()
    is_valid_1 = curve_1.is_valid()

    MINI_CHECK(is_valid_0 == True)
    MINI_CHECK(is_valid_1 == False)


@MINI_TEST("NurbsCurve", "control vertices")
def test_nurbscurve_control_vertices():
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

    # Get all knots
    for i in range(curve.cv_count()):
        pass
        #k = curve.knot(i)

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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)

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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
    t0, t1 = curve.domain()
    t_mid = (t0 + t1) / 2.0
    tangent = curve.tangent_at(t_mid)

    MINI_CHECK(math.isfinite(tangent[0]))
    MINI_CHECK(math.isfinite(tangent[1]))


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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)

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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)

    MINI_CHECK(curve.degree() == 2)
    MINI_CHECK(curve.order() == 3)



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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)

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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)

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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
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
    curve.set_domain(0.0, 1.0)
    curve.set_domain(0.0, 1.0)
    divided_points, params = curve.divide_by_count(5, include_endpoints=True)

    MINI_CHECK(len(divided_points) == 5)


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
    run_all(language="python")
