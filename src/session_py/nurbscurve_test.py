import math
from pathlib import Path
from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("NurbsCurve", "Constructor")
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
    curve = NurbsCurve.create(False, 2, points)
    curve.set_domain(0.0, 1.0)

    # Minimal and Full String Representation
    cstr = str(curve)
    crepr = repr(curve)

    # Copy (duplicates everything except guid)
    ccopy = curve.duplicate()
    cother = NurbsCurve.create(False, 2, points)

    # Point division
    divided, _ = curve.divide_by_count(10, True)

    MINI_CHECK(curve.is_valid() == True)
    MINI_CHECK(curve.cv_count() == 4)
    MINI_CHECK(curve.degree() == 2)
    MINI_CHECK(curve.order() == 3)
    MINI_CHECK(curve.name == "my_nurbscurve")
    MINI_CHECK(curve.guid != "")
    MINI_CHECK(cstr == "NurbsCurve(name=my_nurbscurve, degree=2, cvs=4)")
    MINI_CHECK("name=my_nurbscurve" in crepr)
    MINI_CHECK(ccopy.cv_count() == curve.cv_count())
    MINI_CHECK(ccopy.guid != curve.guid)


@MINI_TEST("NurbsCurve", "Create Interpolated")
def test_nurbscurve_create_interpolated():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.nurbsknot import CurveNurbsKnotStyle
    from session_py.nurbsknot import CurveInterpStyle

    points = [
        Point(14, 9, 0),
        Point(21, 22, 0),
        Point(26, 10, 0),
        Point(35, 19, 0),
        Point(41, 13, 0),
    ]

    c = NurbsCurve.create_interpolated(points, CurveNurbsKnotStyle.Chord)

    MINI_CHECK(c.is_valid())
    MINI_CHECK(c.degree() == 3)
    MINI_CHECK(c.order() == 4)
    MINI_CHECK(c.cv_count() == 7)
    MINI_CHECK(c.is_rational() == False)

    d0, d1 = c.domain()
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(d0), points[0]))
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(d1), points[4]))
    MINI_CHECK(TOLERANCE.is_point_close(c.get_cv(0), points[0]))
    MINI_CHECK(TOLERANCE.is_point_close(c.get_cv(6), points[4]))

    # Rhino parity: interior CVs match Rhino CreateInterpolatedCurve (Chord)
    # bit-for-bit (validated by the OCCT/Rhino harness in validation/).
    MINI_CHECK(TOLERANCE.is_point_close(c.get_cv(1), Point(15.342776949, 13.734888836, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(c.get_cv(3), Point(24.678472471, 0.354555126, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(c.get_cv(5), Point(39.626394361, 15.472490151, 0.0)))

    # OCCT parity: with CurveInterpStyle.Occt the control points match
    # OCCT GeomAPI_Interpolate exactly (oracle: validation/compare_interp.py).
    co = NurbsCurve.create_interpolated(points, CurveNurbsKnotStyle.Chord, CurveInterpStyle.Occt)
    MINI_CHECK(co.cv_count() == 7)
    MINI_CHECK(TOLERANCE.is_point_close(co.get_cv(0), points[0]))
    MINI_CHECK(TOLERANCE.is_point_close(co.get_cv(6), points[4]))
    MINI_CHECK(TOLERANCE.is_point_close(co.get_cv(1), Point(17.3526678158, 24.4472657919, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(co.get_cv(3), Point(24.7854378511, 2.1457823679, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(co.get_cv(5), Point(39.1865250566, 18.5349257754, 0.0)))

    # Periodic closed curve
    closed_pts = [
        Point(4, 20, 0),
        Point(-2, 20, 0),
        Point(-2, 25, 0),
        Point(-3, 28, 0),
        Point(-10, 28, 0),
        Point(-10, 21, 0),
        Point(-13, 16, 0),
        Point(-8, 14, 0),
        Point(-6, 11, 0),
        Point(0, 15, 0),
    ]

    cp = NurbsCurve.create_interpolated(closed_pts, CurveNurbsKnotStyle.ChordPeriodic)

    MINI_CHECK(cp.is_valid())
    MINI_CHECK(cp.degree() == 3)
    MINI_CHECK(cp.cv_count() == 13)
    MINI_CHECK(cp.is_closed())

    # A periodic curve wraps the first (order - 1) points, so fewer points than the
    # order have nothing to wrap and must not be read past the end of the input.
    too_few = NurbsCurve.create(True, 3, [Point(0, 0, 0), Point(1, 0, 0)])
    MINI_CHECK(not too_few.is_valid())


@MINI_TEST("NurbsCurve", "Create From Parameters")
def test_nurbscurve_create_from_parameters():
    from session_py import NurbsCurve
    from session_py import Point

    # Mirrors compas_occt OCCNurbsCurve.from_parameters / from_points / from_circle.
    # Validated bit-for-bit against OCCT (validation/compare_curve_eval.py).

    # from_points: 4 control points, clamped cubic (knots [0,1] mults [4,4]).
    p4 = [Point(0, 0, 0), Point(3, 6, 0), Point(6, -3, 3), Point(10, 0, 0)]
    c = NurbsCurve.create_from_parameters(p4, [1.0, 1.0, 1.0, 1.0], [0.0, 1.0], [4, 4], 3)
    MINI_CHECK(c.is_valid())
    MINI_CHECK(c.degree() == 3)
    MINI_CHECK(c.cv_count() == 4)
    MINI_CHECK(not c.is_rational())
    d0, d1 = c.domain()
    MINI_CHECK(abs(d0 - 0.0) < 1e-12 and abs(d1 - 1.0) < 1e-12)
    MINI_CHECK(TOLERANCE.is_point_close(c.get_cv(0), Point(0, 0, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(c.get_cv(3), Point(10, 0, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(0.5), Point(4.625, 1.125, 1.125)))

    # from_circle (radius 1): degree-2 rational, 9 poles, exact unit circle.
    w = 0.5 * math.sqrt(2.0)
    cpts = [Point(0, -1, 0), Point(-1, -1, 0), Point(-1, 0, 0), Point(-1, 1, 0), Point(0, 1, 0),
            Point(1, 1, 0), Point(1, 0, 0), Point(1, -1, 0), Point(0, -1, 0)]
    circle = NurbsCurve.create_from_parameters(
        cpts, [1, w, 1, w, 1, w, 1, w, 1], [0.0, 0.25, 0.5, 0.75, 1.0], [3, 2, 2, 2, 3], 2)
    MINI_CHECK(circle.is_valid())
    MINI_CHECK(circle.degree() == 2)
    MINI_CHECK(circle.cv_count() == 9)
    MINI_CHECK(circle.is_rational())
    MINI_CHECK(TOLERANCE.is_point_close(circle.point_at(0.5), Point(0, 1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(circle.point_at(0.125), Point(-w, -w, 0)))
    for k in range(17):
        pp = circle.point_at(k / 16.0)
        MINI_CHECK(abs(math.sqrt(pp[0] * pp[0] + pp[1] * pp[1]) - 1.0) < 1e-9)


@MINI_TEST("NurbsCurve", "Create Fitted")
def test_nurbscurve_create_fitted():
    from session_py import NurbsCurve
    from session_py import Point

    # Open: 21 points on sine wave → fit with 8 CVs
    pts = [Point(i * 2.0 * PI / 20.0, 3.0 * math.sin(i * 2.0 * PI / 20.0), 0.0) for i in range(21)]

    c = NurbsCurve.create_fitted(pts, 8, 3, False)

    MINI_CHECK(c.is_valid())
    MINI_CHECK(c.degree() == 3)
    MINI_CHECK(c.cv_count() == 8)
    d0, d1 = c.domain()
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(d0), pts[0]))
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(d1), pts[20]))

    # Periodic: 24 points on circle → fit with 10 free CVs
    cpts = [Point(math.cos(i * 2.0 * PI / 24.0), math.sin(i * 2.0 * PI / 24.0), 0.0) for i in range(24)]

    cp = NurbsCurve.create_fitted(cpts, 10, 3, True)

    MINI_CHECK(cp.is_valid())
    MINI_CHECK(cp.is_closed())
    MINI_CHECK(cp.cv_count() == 13)


@MINI_TEST("NurbsCurve", "Join")
def test_nurbscurve_join():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Primitives

    arc1 = Primitives.arc(Point(-1.0, 0.0, 0.0), Point(0.0, 1.0, 0.0), Point(1.0, 0.0, 0.0))
    arc2 = Primitives.arc(Point(1.0, 0.0, 0.0), Point(1.5, -1.0, 0.0), Point(1.0, -2.0, 0.0))
    pts = [Point(1.0, -2.0, 0.0), Point(-1.0, 0.0, 0.0)]
    line = NurbsCurve.create(False, 1, pts)
    arc2.reverse()

    joined = NurbsCurve.join([line, arc1, arc2])

    MINI_CHECK(len(joined) == 1)
    MINI_CHECK(joined[0].is_valid())
    MINI_CHECK(joined[0].is_closed())
    MINI_CHECK(joined[0].degree() == 2)
    MINI_CHECK(joined[0].cv_count() == 7)

    l1 = NurbsCurve.create(False, 1, [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)])
    l2 = NurbsCurve.create(False, 1, [Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0)])
    l3 = NurbsCurve.create(False, 1, [Point(9.0, 9.0, 0.0), Point(8.0, 8.0, 0.0)])

    separate = NurbsCurve.join([l1, l3, l2])

    MINI_CHECK(len(separate) == 2)
    MINI_CHECK(separate[0].cv_count() == 3)
    MINI_CHECK(abs(separate[0].length() - 2.0) < 1e-9)


@MINI_TEST("NurbsCurve", "Attributes")
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

    curve = NurbsCurve.create(False, 2, points)

    # ═══════════════════════════════════════════════════════════════════════════
    # Boolean Queries
    # ═══════════════════════════════════════════════════════════════════════════

    # Whole curve
    is_valid = curve.is_valid()
    MINI_CHECK(is_valid == True)

    # Check whole nurbsknot vector for
    # For correct size: order + cv_count - 2
    # Non-decreasing (can repeat, can't go down)
    # Valid domain exists
    is_valid_nurbsknot_vector = curve.is_valid_nurbsknot_vector()
    MINI_CHECK(is_valid_nurbsknot_vector == True)

    # Check if the curve is clamped at start, end, or both
    is_clamped_start = curve.is_clamped(0)
    is_clamped_end = curve.is_clamped(1)
    is_clamped_both = curve.is_clamped(2)
    MINI_CHECK(is_clamped_start == True)
    MINI_CHECK(is_clamped_end == True)
    MINI_CHECK(is_clamped_both == True)

    # Is rational is related to control points having weights
    # is_rational = false means control points [x, y, z]
    # is_rational = false means control points [xw, yw, zw]
    # Rational curves are used to represent:
    # circles, ellipses, parabolas, hyperbolas exactly
    is_rational = curve.is_rational()
    closed = curve.is_closed()
    periodic = curve.is_periodic()
    linear = curve.is_linear()
    planar = curve.is_planar()
    arc = curve.is_arc()
    plane = Plane.xy_plane()
    on_plane = curve.is_in_plane(plane)
    is_open = curve.is_natural()
    is_polyline, _, _ = curve.is_polyline()
    is_singular = curve.is_singular()
    is_duplicate = curve.is_duplicate(curve, False)
    is_continuous = curve.is_continuous(1, curve.domain_middle())

    MINI_CHECK(is_rational == False)
    MINI_CHECK(closed == False)
    MINI_CHECK(periodic == False)
    MINI_CHECK(linear == False)
    MINI_CHECK(planar == True)
    MINI_CHECK(arc == False)
    MINI_CHECK(on_plane == True)
    MINI_CHECK(is_open == False)
    MINI_CHECK(is_polyline == False)
    MINI_CHECK(is_singular == False)
    MINI_CHECK(is_duplicate == True)
    MINI_CHECK(is_continuous == True)

    # ═══════════════════════════════════════════════════════════════════════════
    # NurbsKnot Operations
    # ═══════════════════════════════════════════════════════════════════════════

    # Insert nurbsknot into curve
    # Useful for splitting curves at a parameter
    # Increase local control without changing shape
    copy_curve = curve.duplicate()
    before_pt = copy_curve.point_at(1.5)
    copy_curve.insert_nurbsknot(1.5, 1)
    MINI_CHECK(TOLERANCE.is_point_close(before_pt, copy_curve.point_at(1.5)))

    # A repeated interior nurbsknot ends a span early: the length must still cover
    # every span past it.
    kinked = curve.duplicate()
    kinked_length = kinked.length()
    kinked.insert_nurbsknot(1.5, 2)
    MINI_CHECK(TOLERANCE.is_close(kinked.length(), kinked_length))

    # Useful for controlling curve by cv on lying on it
    greville0 = curve.greville_abcissa(0)
    MINI_CHECK(TOLERANCE.is_close(greville0, 0.0))

    greville = curve.get_greville_abcissae()
    MINI_CHECK(len(greville) == 4)
    MINI_CHECK(TOLERANCE.is_close(greville[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(greville[1], 0.879872167739067))
    MINI_CHECK(TOLERANCE.is_close(greville[2], 2.639616503217201))
    MINI_CHECK(TOLERANCE.is_close(greville[3], 3.519488670956267))

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
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
    # Order = degree + 1, control points + order = nurbsknots
    order = curve.order()
    MINI_CHECK(order == 3)
    # Number of control vertices
    cv_count = curve.cv_count()
    MINI_CHECK(cv_count == 4)
    # Number of floats per 1 control vertex
    cv_size = curve.cv_size()
    MINI_CHECK(cv_size == 3)
    # The nurbsknots are a list of (degree+control_points-1) numbers
    nurbsknot_count = curve.nurbsknot_count()
    MINI_CHECK(nurbsknot_count == 5)
    # Span = a nurbsknot interval where a single polynomial segment is evaluated
    # NurbsKnot vector: [0, 0, 0 ↑, 1 ↑, 2 ↑, 3, 3, 3]  (cubic, 5 CVs)
    span_count = curve.span_count()
    MINI_CHECK(span_count == 2)
    # ═══════════════════════════════════════════════════════════════════════════
    # Control Vertex Access
    #  m_cv = [x0, y0, z0, (w0), x1, y1, z1, (w1), ...]
    #          └─── CV 0 ───┘    └─── CV 1 ───┘
    # ═══════════════════════════════════════════════════════════════════════════

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
    MINI_CHECK(curve.get_cv(2)[0] == 2.0)
    MINI_CHECK(curve.get_cv(2)[1] == 0.0)
    MINI_CHECK(curve.get_cv(2)[2] == 0.5)

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

    # set_cv takes a euclidean point: it must read back unchanged on a rational curve,
    # where a stale weight would scale it.
    curve.set_cv(2, Point(7.0, 8.0, 9.0))
    MINI_CHECK(TOLERANCE.is_point_close(curve.get_cv(2), Point(7.0, 8.0, 9.0)))

    # ═══════════════════════════════════════════════════════════════════════════
    # NurbsKnot Access
    # ═══════════════════════════════════════════════════════════════════════════

    # Get nurbsknot value at index
    nurbsknot3 = curve.nurbsknot(3)
    MINI_CHECK(TOLERANCE.is_close(nurbsknot3, 3.519488670956267))

    # Set nurbsknot value at index
    # ATTENTION you can brake increasing rule
    end_nurbsknot = curve.nurbsknot(4)
    curve.set_nurbsknot(4, end_nurbsknot)
    MINI_CHECK(TOLERANCE.is_close(curve.nurbsknot(4), end_nurbsknot))

    # Count repeated nurbsknots at index [0, 0, 1, 1, 2]
    m0 = curve.nurbsknot_multiplicity(0)  # 2 (two 0's)
    m1 = curve.nurbsknot_multiplicity(1)  # 2 (still counting the 0's)
    m2 = curve.nurbsknot_multiplicity(2)  # 1 (single 0.5)
    m3 = curve.nurbsknot_multiplicity(3)  # 2 (single 1's)
    m4 = curve.nurbsknot_multiplicity(4)  # 2 (single 2)
    MINI_CHECK(m0 == 2)
    MINI_CHECK(m1 == 2)
    MINI_CHECK(m2 == 1)
    MINI_CHECK(m3 == 2)
    MINI_CHECK(m4 == 2)

    # Superflous nurbsknots are used for extension of clamped curves
    superfluous_nurbsknot = curve.superfluous_nurbsknot(1)
    MINI_CHECK(TOLERANCE.is_close(superfluous_nurbsknot, 7.038977341912535))

    # Direct memory access to nurbsknot values, fast, read-only
    # Vector return is slower and makes a copy
    nurbsknots = curve.nurbsknot_array()
    k0 = nurbsknots[0]
    nurbsknot_vector = curve.get_nurbsknots()
    MINI_CHECK(k0 == 0.0)
    MINI_CHECK(TOLERANCE.is_close(nurbsknot_vector[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(nurbsknot_vector[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(nurbsknot_vector[2], 1.759744335478134))
    MINI_CHECK(TOLERANCE.is_close(nurbsknot_vector[3], 3.519488670956267))
    MINI_CHECK(TOLERANCE.is_close(nurbsknot_vector[4], 3.519488670956267))

    # Control vertex array access
    cvs = curve.cv_array()
    cx0 = cvs[0]
    MINI_CHECK(cx0 == 0.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # Domain & Parameterization - HERE
    # ═══════════════════════════════════════════════════════════════════════════

    # get start and end of the curve interval
    start, end = curve.domain()
    MINI_CHECK(TOLERANCE.is_close(start, 0.0) and TOLERANCE.is_close(end, 3.519488670956267))

    # Get start, middle and end values of the interval
    start = curve.domain_start()
    middle = curve.domain_middle()
    end = curve.domain_end()
    MINI_CHECK(TOLERANCE.is_close(start, 0.0))
    MINI_CHECK(TOLERANCE.is_close(middle, 1.759744335478134))
    MINI_CHECK(TOLERANCE.is_close(end, 3.519488670956267))

    # Change curve domain
    curve.set_domain(0.0, 1.0)
    MINI_CHECK(curve.domain_start() == 0.0)
    MINI_CHECK(curve.domain_middle() == 0.5)
    MINI_CHECK(curve.domain_end() == 1.0)

    # Span of distict nurbsknot intervals
    intervals = curve.get_span_vector()
    MINI_CHECK(TOLERANCE.is_close(intervals[0], 0.0) and TOLERANCE.is_close(intervals[1], 0.5) and TOLERANCE.is_close(intervals[2], 1.0))

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometric checks
    # ═══════════════════════════════════════════════════════════════════════════

    found, t_out = curve.get_next_discontinuity(2, curve.domain_start(), curve.domain_end())
    MINI_CHECK(found == True and TOLERANCE.is_close(t_out, 0.5))


@MINI_TEST("NurbsCurve", "Conversions")
def test_nurbscurve_conversions():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 2.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 2.0, 0.0),
        Point(4.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(False, 2, points)

    # to_polyline_adaptive
    adaptive_pts, adaptive_params = curve.to_polyline_adaptive(0.1, 0.0, 0.0)

    MINI_CHECK(len(adaptive_pts) == 27)
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[0], Point(0.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[13], Point(2.0, 0.5, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(adaptive_pts[26], Point(4.0, 0.0, 0.0)))

    # A closed curve has a zero-length start-to-end chord: the subdivision must still
    # sample it instead of returning the degenerate two-point polyline.
    from session_py import Primitives
    circle = Primitives.circle(0.0, 0.0, 0.0, 2.0)
    circle_pts, circle_params = circle.to_polyline_adaptive(0.1, 0.0, 0.0)
    MINI_CHECK(len(circle_pts) == 25)

    # divide_by_count
    div_pts, div_params = curve.divide_by_count(10, True)

    MINI_CHECK(len(div_pts) == 10)
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[0], Point(0.000000000000000, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[1], Point(0.328571016773017, 0.598213507757063, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[2], Point(0.740744944144815, 1.140321237310326, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[3], Point(1.338524001477341, 1.232716038191446, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[4], Point(1.712929668000343, 0.664818751028787, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[5], Point(2.287070333148604, 0.664818752348101, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[6], Point(2.661475999779531, 1.232716039392177, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[7], Point(3.259255057037078, 1.140321236176910, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[8], Point(3.671428983538974, 0.598213507250245, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(div_pts[9], Point(4.000000000000000, 0.000000000000000, 0.000000000000000)))

    # divide_by_length
    len_pts, len_params = curve.divide_by_length(0.5)

    MINI_CHECK(len(len_pts) == 13)
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[0], Point(0.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[6], Point(1.928691288503169, 0.510169864670676, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(len_pts[12], Point(3.934494396222682, 0.128829843907475, 0.0)))


@MINI_TEST("NurbsCurve", "Evaluation")
def test_nurbscurve_evaluation():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Vector
    from session_py import Plane

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

    curve = NurbsCurve.create(False, 2, points)

    # Length
    MINI_CHECK(TOLERANCE.is_close(curve.length(), 11.3010276326))

    # Get point at parameter t
    point_at = curve.point_at(0.5)
    MINI_CHECK(TOLERANCE.is_close(point_at[0], 1.463452399002842))
    MINI_CHECK(TOLERANCE.is_close(point_at[1], 1.680997287875395))
    MINI_CHECK(TOLERANCE.is_close(point_at[2], -0.124474565996108))

    # Get point and derivatives at parameter t
    derivatives = curve.evaluate(0.5, 2)
    MINI_CHECK(len(derivatives) == 3)
    MINI_CHECK(TOLERANCE.is_close(derivatives[0][0], 1.463452399002842))
    MINI_CHECK(TOLERANCE.is_close(derivatives[0][1], 1.680997287875395))
    MINI_CHECK(TOLERANCE.is_close(derivatives[0][2], -0.124474565996108))
    MINI_CHECK(TOLERANCE.is_close(derivatives[1][0], -0.311619416021204))
    MINI_CHECK(TOLERANCE.is_close(derivatives[1][1], 0.974021205471335))
    MINI_CHECK(TOLERANCE.is_close(derivatives[1][2], -0.037441955449586))
    MINI_CHECK(TOLERANCE.is_close(derivatives[2][0], 2.706815143892446))
    MINI_CHECK(TOLERANCE.is_close(derivatives[2][1], -0.429869481117820))
    MINI_CHECK(TOLERANCE.is_close(derivatives[2][2], -0.684219293829483))

    # Tangent vector at parameter t
    tangent = curve.tangent_at(0.5)
    MINI_CHECK(TOLERANCE.is_close(tangent[0], -0.304511941745027))
    MINI_CHECK(TOLERANCE.is_close(tangent[1], 0.951805546117607))
    MINI_CHECK(TOLERANCE.is_close(tangent[2], -0.036587972264639))

    # normalized=true (default): t in [0,1] mapped to domain
    f = curve.plane_at(0.5, True)
    MINI_CHECK(TOLERANCE.is_close(f.origin[0], 3.156927375))
    MINI_CHECK(TOLERANCE.is_close(f.origin[1], 1.3351115))
    MINI_CHECK(TOLERANCE.is_close(f.origin[2], 0.130488875))
    MINI_CHECK(TOLERANCE.is_close(f.x_axis[0], 0.701806140304030))
    MINI_CHECK(TOLERANCE.is_close(f.x_axis[1], 0.697509131556264))
    MINI_CHECK(TOLERANCE.is_close(f.x_axis[2], 0.144738221721788))
    MINI_CHECK(TOLERANCE.is_close(f.y_axis[0], -0.513930504714161))
    MINI_CHECK(TOLERANCE.is_close(f.y_axis[1], 0.355053088776962))
    MINI_CHECK(TOLERANCE.is_close(f.y_axis[2], 0.780905077761815))
    MINI_CHECK(TOLERANCE.is_close(f.z_axis[0], 0.493298669931115))
    MINI_CHECK(TOLERANCE.is_close(f.z_axis[1], -0.622429365908747))
    MINI_CHECK(TOLERANCE.is_close(f.z_axis[2], 0.607649657861031))

    MINI_CHECK(curve.plane_at(-0.1, True).is_valid() == False)
    MINI_CHECK(curve.plane_at(1.1, True).is_valid() == False)
    MINI_CHECK(curve.plane_at(curve.domain_start(), False).is_valid() == True)
    MINI_CHECK(curve.plane_at(curve.domain_end(), False).is_valid() == True)
    MINI_CHECK(curve.plane_at(curve.domain_start() - 0.1, False).is_valid() == False)

    # Perpendicular frame at (RMF with Frenet initialization, matches Rhino)
    pf = curve.perpendicular_plane_at(0.5, True)
    MINI_CHECK(TOLERANCE.is_point_close(pf.origin, Point(3.156927375, 1.3351115, 0.130488875)))
    MINI_CHECK(TOLERANCE.is_vector_close(pf.x_axis, Vector(0.632703652329189, -0.703685357647999, 0.323284713157168)))
    MINI_CHECK(TOLERANCE.is_vector_close(pf.y_axis, Vector(0.327344206830723, -0.135306795251661, -0.935167279909370)))
    MINI_CHECK(TOLERANCE.is_vector_close(pf.z_axis, Vector(0.701806140314880, 0.697509131546342, 0.144738221716994)))
    MINI_CHECK(curve.perpendicular_plane_at(-0.1, True).is_valid() == False)
    MINI_CHECK(curve.perpendicular_plane_at(1.1, True).is_valid() == False)
    MINI_CHECK(curve.perpendicular_plane_at(curve.domain_start(), False).is_valid() == True)
    MINI_CHECK(curve.perpendicular_plane_at(curve.domain_end(), False).is_valid() == True)
    MINI_CHECK(curve.perpendicular_plane_at(curve.domain_start() - 0.1, False).is_valid() == False)

    # Get multiple rotation minimization frames along the curve (matches Rhino)
    frames = curve.get_perpendicular_planes(4)
    MINI_CHECK(len(frames) == 5)
    # Frame 0 (start)
    MINI_CHECK(TOLERANCE.is_point_close(frames[0].origin, Point(1.957614, 1.140253, -0.191281)))
    MINI_CHECK(TOLERANCE.is_vector_close(frames[0].x_axis, Vector(0.532767753269467, 0.809398954921174, -0.247046256496055)))
    MINI_CHECK(TOLERANCE.is_vector_close(frames[0].y_axis, Vector(-0.261213903019039, -0.120386647366337, -0.957744408496052)))
    MINI_CHECK(TOLERANCE.is_vector_close(frames[0].z_axis, Vector(-0.804938393882267, 0.574787253606414, 0.147288136473484)))
    # Frame 2 (middle)
    MINI_CHECK(TOLERANCE.is_point_close(frames[2].origin, Point(3.676077075808618, 0.909845354074582, 0.350126131660904)))
    MINI_CHECK(TOLERANCE.is_vector_close(frames[2].x_axis, Vector(-0.188216728828592, 0.616420980974357, -0.764591156896073)))
    MINI_CHECK(TOLERANCE.is_vector_close(frames[2].y_axis, Vector(0.183061410483993, -0.742842969436200, -0.643950963001702)))
    MINI_CHECK(TOLERANCE.is_vector_close(frames[2].z_axis, Vector(-0.964916049706230, -0.261169479407185, 0.026972579511507)))
    # Frame 4 (end)
    MINI_CHECK(TOLERANCE.is_point_close(frames[4].origin, Point(2.150320000000000, 1.868606000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_vector_close(frames[4].x_axis, Vector(0.183261707646767, 0.080808692310795, 0.979737261594868)))
    MINI_CHECK(TOLERANCE.is_vector_close(frames[4].y_axis, Vector(0.896455027441244, 0.395289116385372, -0.200287039627106)))
    MINI_CHECK(TOLERANCE.is_vector_close(frames[4].z_axis, Vector(-0.403464410184726, 0.914995338629816, 0.000000000000000)))

    # Points
    p0 = curve.point_at_start()
    p1 = curve.point_at_middle()
    p2 = curve.point_at_end()
    MINI_CHECK(TOLERANCE.is_close(p0[0], 1.957614))
    MINI_CHECK(TOLERANCE.is_close(p0[1], 1.140253))
    MINI_CHECK(TOLERANCE.is_close(p0[2], -0.191281))
    MINI_CHECK(TOLERANCE.is_close(p1[0], 3.156927375))
    MINI_CHECK(TOLERANCE.is_close(p1[1], 1.3351115))
    MINI_CHECK(TOLERANCE.is_close(p1[2], 0.130488875))
    MINI_CHECK(TOLERANCE.is_close(p2[0], 2.15032))
    MINI_CHECK(TOLERANCE.is_close(p2[1], 1.868606))
    MINI_CHECK(TOLERANCE.is_close(p2[2], 0.0))

    curve.set_start_point(Point(1.957614, 1.140253, 2.0))
    curve.set_end_point(Point(2.15032, 1.868606, 2.0))
    MINI_CHECK(TOLERANCE.is_close(curve.point_at_start()[2], 2.0))
    MINI_CHECK(TOLERANCE.is_close(curve.point_at_end()[2], 2.0))


@MINI_TEST("NurbsCurve", "Modifications")
def test_nurbscurve_modifications():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 2.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 2.0, 0.0),
        Point(4.0, 0.0, 0.0)
    ]

    curve = NurbsCurve.create(False, 2, points)

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

    # Trim curve at domain parameter
    ct = curve.duplicate()
    a = ct.domain_start() + (ct.domain_end() - ct.domain_start()) / 3.0
    b = ct.domain_start() + 2.0 * (ct.domain_end() - ct.domain_start()) / 3.0
    ct.trim(a, b)
    MINI_CHECK(ct.length() < curve.length())

    # Split curve at domain middle
    split_t = curve.domain_middle()
    curve_left, curve_right = curve.split(split_t)
    MINI_CHECK(TOLERANCE.is_point_close(curve.point_at(split_t), curve_left.point_at_end()))
    MINI_CHECK(TOLERANCE.is_point_close(curve.point_at(split_t), curve_right.point_at_start()))
    # Each piece keeps the parameterization and the geometry of the original curve.
    MINI_CHECK(TOLERANCE.is_point_close(curve_left.point_at_middle(),
                                        curve.point_at((curve.domain_start() + split_t) * 0.5)))

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

    curve_rational.make_non_rational(True)
    MINI_CHECK(curve_rational.length() == original_length)

    # Uniform non-unit weights are removable without moving the curve: the CVs must be
    # divided by the weight, not copied in homogeneous form.
    curve_uniform_w = curve.duplicate()
    curve_uniform_w.make_rational()
    for i in range(curve_uniform_w.cv_count()):
        curve_uniform_w.set_weight(i, 2.0)
    uniform_w_mid = curve_uniform_w.point_at_middle()
    MINI_CHECK(curve_uniform_w.make_non_rational(False))
    MINI_CHECK(TOLERANCE.is_point_close(curve_uniform_w.point_at_middle(), uniform_w_mid))

    # Clamp ends - create unclamped curve manually
    points_open = points
    curve_open = NurbsCurve(3, False, 3, 5)  # dim=3, non-rational, order=3 (deg 2), 5 CVs

    for i in range(5):
        curve_open.set_cv(i, points_open[i])

    for i in range(curve_open.nurbsknot_count()):
        curve_open.set_nurbsknot(i, i * 1.0)

    # Now clamp, making 2 nurbsknots at the ends the same
    curve_open.clamp_end(2)
    nurbsknots = curve_open.get_nurbsknots()
    MINI_CHECK(TOLERANCE.is_close(nurbsknots[0], nurbsknots[1]))
    MINI_CHECK(TOLERANCE.is_close(nurbsknots[-2], nurbsknots[-1]))

    # Increase degree without change the shape
    raised = curve.duplicate()
    raised.increase_degree(3)
    MINI_CHECK(curve.degree() != raised.degree())
    MINI_CHECK(TOLERANCE.is_point_close(curve.point_at_middle(), raised.point_at_middle()))

    # Change closed curve seam
    closed_pts = [
        Point(1.0, 0.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(-1.0, 0.0, 0.0),
        Point(0.0, -1.0, 0.0)
    ]
    c = NurbsCurve.create(True, 2, closed_pts)
    expected_start = c.point_at(c.domain_middle())
    c.change_closed_curve_seam(c.domain_middle())
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at_start(), expected_start))


@MINI_TEST("NurbsCurve", "Transformations")
def test_nurbscurve_transformations():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Xform

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 2.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 2.0, 0.0),
        Point(4.0, 0.0, 0.0)
    ]

    # transform(xform) - in place
    curve1 = NurbsCurve.create(False, 2, points)
    curve1_xf = Xform.translation(0.0, 0.0, 1.0)
    curve1.transform(curve1_xf)
    MINI_CHECK(curve1.cv(0)[2] == 1.0)

    # transform(xform) - Apply custom xform (in-place)
    curve2 = NurbsCurve.create(False, 2, points)
    x = Xform.translation(0.0, 0.0, 1.0)
    curve2.transform(x)
    MINI_CHECK(curve2.cv(0)[2] == 1.0)

    # transformed(xform) - returns a copy
    curve3 = NurbsCurve.create(False, 2, points)
    curve3_xf = Xform.translation(0.0, 0.0, 10.0)
    curve3_transformed = curve3.transformed(curve3_xf)
    MINI_CHECK(curve3_transformed.cv(0)[2] == 10.0)

    # transformed(xform) - Get copy with custom xform
    curve4 = NurbsCurve.create(False, 2, points)
    x = Xform.translation(0.0, 0.0, 10.0)
    curve4_transformed = curve4.transformed(x)
    MINI_CHECK(curve4_transformed.cv(0)[2] == 10.0)


@MINI_TEST("NurbsCurve", "Json Roundtrip")
def test_nurbscurve_json_roundtrip():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 2.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 2.0, 0.0),
        Point(4.0, 0.0, 0.0)
    ]
    curve = NurbsCurve.create(False, 2, points)

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   file_json_dumps()    │ str          │ to JSON string
    #   file_json_loads(s)   │ str          │ from JSON string
    #   file_json_dump(path) │ file         │ write to file
    #   file_json_load(path) │ file         │ read from file

    # JSON object
    json_obj = curve.__jsondump__()
    loaded_json = NurbsCurve.__jsonload__(json_obj)

    # String
    json_string = curve.file_json_dumps()
    loaded_json_string = NurbsCurve.file_json_loads(json_string)

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbscurve.json"
    curve.file_json_dump(filename)
    loaded_from_file = NurbsCurve.file_json_load(filename)

    MINI_CHECK(loaded_json == curve)
    MINI_CHECK(loaded_json_string == curve)
    MINI_CHECK(loaded_from_file == curve)

    # A rational curve survives the round trip only if the weights ride along: its
    # control points are dumped in homogeneous form.
    rational = curve.duplicate()
    rational.make_rational()
    rational.set_weight(1, 0.5)
    loaded_rational = NurbsCurve.file_json_loads(rational.file_json_dumps())
    MINI_CHECK(loaded_rational == rational)


@MINI_TEST("NurbsCurve", "Protobuf Roundtrip")
def test_nurbscurve_protobuf_roundtrip():
    from session_py import NurbsCurve
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 2.0, 0.0),
        Point(2.0, 0.0, 0.0),
        Point(3.0, 2.0, 0.0),
        Point(4.0, 0.0, 0.0)
    ]
    curve = NurbsCurve.create(False, 2, points)

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    # String
    proto_string = curve.pb_dumps()
    loaded_proto_string = NurbsCurve.pb_loads(proto_string)

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbscurve.bin"
    curve.pb_dump(filename)
    loaded = NurbsCurve.pb_load(filename)

    MINI_CHECK(loaded_proto_string == curve)
    MINI_CHECK(loaded == curve)


@MINI_TEST("NurbsCurve", "Curvature")
def test_nurbscurve_curvature():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Primitives

    # A circle of radius R is an exact rational NURBS with constant curvature 1/R.
    r = 2.0
    circle = Primitives.circle(0.0, 0.0, 0.0, r)
    t0, t1 = circle.domain()
    for i in range(9):
        t = t0 + (t1 - t0) * i / 8.0
        MINI_CHECK(abs(circle.curvature_at(t) - 1.0 / r) < 1e-6)
    # The exact rational circle integrates to its exact circumference.
    MINI_CHECK(abs(circle.length() - 2.0 * PI * r) < 1e-9)

    # Closest point: an outside point projects radially onto the circle.
    cp = circle.closest_point(Point(5, 0, 0))
    MINI_CHECK(abs(cp[0] - 2.0) < 1e-5 and abs(cp[1]) < 1e-5 and abs(cp[2]) < 1e-5)

    # Project onto a 3D interpolated curve (curve_closest_point.py). Reference from
    # OCCT GeomAPI_ProjectPointOnCurve (validation/compare_curve_ops.py).
    from session_py.nurbsknot import CurveNurbsKnotStyle
    from session_py.nurbsknot import CurveInterpStyle
    ipts = [Point(0, 0, 0), Point(3, 0, 2), Point(6, 0, -3), Point(8, 0, 0)]
    ic = NurbsCurve.create_interpolated(ipts, CurveNurbsKnotStyle.Chord, CurveInterpStyle.Occt)
    pc = ic.closest_point(Point(2, -1, 0))
    MINI_CHECK(TOLERANCE.is_point_close(pc, Point(0.5808155659, 0.0, 0.9672315271)))

    # Curve-curve closest (curve_closest_parameters_curve.py). Reference from
    # OCCT GeomAPI_ExtremaCurveCurve (u=0.475768, v=0.336691).
    c0 = NurbsCurve.create_from_parameters(
        [Point(0, 0, 0), Point(3, 6, 0), Point(6, -3, 3), Point(10, 0, 0)], [1, 1, 1, 1], [0, 1], [4, 4], 3)
    c1 = NurbsCurve.create_from_parameters(
        [Point(6, -3, 0), Point(3, 1, 0), Point(6, 6, 3), Point(3, 12, 0)], [1, 1, 1, 1], [0, 1], [4, 4], 3)
    (u, v), d = c0.closest_parameters_curve(c1, return_distance=True)
    MINI_CHECK(abs(u - 0.4757682937) < 1e-6 and abs(v - 0.3366914716) < 1e-6)
    (pa, pb), _ = c0.closest_points_curve(c1, return_distance=True)
    MINI_CHECK(TOLERANCE.is_point_close(pa, Point(4.389607399, 1.285537564, 1.067964425)))
    MINI_CHECK(TOLERANCE.is_point_close(pb, Point(4.552264625, 1.380381100, 0.676740741)))

    # A straight line has zero curvature.
    line_pts = [Point(0, 0, 0), Point(1, 0, 0), Point(2, 0, 0), Point(3, 0, 0)]
    line = NurbsCurve.create(False, 1, line_pts)
    MINI_CHECK(line.curvature_at(line.domain_middle()) < 1e-9)


if __name__ == "__main__":
    from .mini_test import run_all
    run_all(language="python")
