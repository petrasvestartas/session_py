from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("NurbsSurface", "Constructor")
def test_nurbssurface_constructor():
    from session_py import NurbsSurface
    from session_py import Point

    points = [
        # i=0
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        # i=1
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        # i=2
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        # i=3
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]

    s = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    # Get mesh
    m = s.mesh()

    # Point division matching Rhino's 4x6 grid
    p, v, uv = s.divide_by_count_points(4, 6)

    # Minimal and Full String Representation
    sstr = str(s)
    srepr = repr(s)

    # Copy (duplicates everything except guid)
    scopy = s.duplicate()
    sother = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    MINI_CHECK(s.is_valid() == True)
    MINI_CHECK(s.cv_count_dir(0) == 4)
    MINI_CHECK(s.cv_count_dir(1) == 4)
    MINI_CHECK(s.cv_count_dir(None) == 16)
    MINI_CHECK(s.degree(0) == 3)
    MINI_CHECK(s.degree(1) == 3)
    MINI_CHECK(s.order(0) == 4)
    MINI_CHECK(s.order(1) == 4)
    MINI_CHECK(s.dimension() == 3)
    MINI_CHECK(not s.is_rational())
    MINI_CHECK(s.nurbsknot_count(0) == 6)
    MINI_CHECK(s.nurbsknot_count(1) == 6)
    MINI_CHECK(s.name == "my_nurbssurface")
    MINI_CHECK(s.guid)
    MINI_CHECK(sstr == "NurbsSurface(name=my_nurbssurface, degree=(3,3), cvs=(4,4))")
    MINI_CHECK(srepr == "NurbsSurface(\n  name=my_nurbssurface,\n  degree=(3,3),\n  cvs=(4,4),\n  rational=false,\n  control_points=[\n    0, 0, 0\n    -1, 0.75, 2\n    -1, 4.25, 2\n    0, 5, 0\n    0.75, -1, 2\n    1.25, 1.25, 4\n    1.25, 3.75, 4\n    0.75, 6, 2\n    4.25, -1, 2\n    3.75, 1.25, 4\n    3.75, 3.75, 4\n    4.25, 6, 2\n    5, 0, 0\n    6, 0.75, 2\n    6, 4.25, 2\n    5, 5, 0\n  ]\n)")
    MINI_CHECK(scopy.cv_count_dir(None) == s.cv_count_dir(None))
    MINI_CHECK(scopy.guid != s.guid)
    MINI_CHECK(TOLERANCE.is_point_close(p[0][0], Point(0.000000000000000, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[0][1], Point(-0.416666666666667, 0.578703703703704, 0.833333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[0][2], Point(-0.666666666666667, 1.462962962962963, 1.333333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[0][3], Point(-0.750000000000000, 2.500000000000000, 1.500000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[0][4], Point(-0.666666666666667, 3.537037037037037, 1.333333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[0][5], Point(-0.416666666666667, 4.421296296296297, 0.833333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[0][6], Point(0.000000000000000, 5.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[1][0], Point(0.992187500000000, -0.562500000000000, 1.125000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[1][1], Point(0.881510416666667, 0.333912037037037, 1.958333333333334)))
    MINI_CHECK(TOLERANCE.is_point_close(p[1][2], Point(0.815104166666667, 1.379629629629630, 2.458333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[1][3], Point(0.792968750000000, 2.500000000000000, 2.625000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[1][4], Point(0.815104166666667, 3.620370370370370, 2.458333333333334)))
    MINI_CHECK(TOLERANCE.is_point_close(p[1][5], Point(0.881510416666667, 4.666087962962964, 1.958333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[1][6], Point(0.992187500000000, 5.562500000000000, 1.125000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[2][0], Point(2.500000000000000, -0.750000000000000, 1.500000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[2][1], Point(2.500000000000000, 0.252314814814815, 2.333333333333334)))
    MINI_CHECK(TOLERANCE.is_point_close(p[2][2], Point(2.500000000000000, 1.351851851851852, 2.833333333333334)))
    MINI_CHECK(TOLERANCE.is_point_close(p[2][3], Point(2.500000000000000, 2.500000000000000, 3.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[2][4], Point(2.500000000000000, 3.648148148148148, 2.833333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[2][5], Point(2.500000000000000, 4.747685185185186, 2.333333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[2][6], Point(2.500000000000000, 5.750000000000000, 1.500000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[3][0], Point(4.007812500000000, -0.562500000000000, 1.125000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[3][1], Point(4.118489583333334, 0.333912037037037, 1.958333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[3][2], Point(4.184895833333334, 1.379629629629630, 2.458333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[3][3], Point(4.207031250000000, 2.500000000000000, 2.625000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[3][4], Point(4.184895833333333, 3.620370370370370, 2.458333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[3][5], Point(4.118489583333333, 4.666087962962964, 1.958333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[3][6], Point(4.007812500000000, 5.562500000000000, 1.125000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[4][0], Point(5.000000000000000, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[4][1], Point(5.416666666666668, 0.578703703703704, 0.833333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[4][2], Point(5.666666666666668, 1.462962962962963, 1.333333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[4][3], Point(5.750000000000000, 2.500000000000000, 1.500000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(p[4][4], Point(5.666666666666666, 3.537037037037037, 1.333333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[4][5], Point(5.416666666666667, 4.421296296296297, 0.833333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(p[4][6], Point(5.000000000000000, 5.000000000000000, 0.000000000000000)))


@MINI_TEST("NurbsSurface", "Create From Parameters")
def test_nurbssurface_create_from_parameters():
    from session_py import NurbsSurface
    from session_py import Point

    # Mirrors compas_occt OCCNurbsSurface.from_parameters / from_points (surface_from_points.py).
    # Validated pointwise against OCCT (validation/compare_surface_eval.py).
    grid = [
        [Point(0, 0, 0), Point(1, 0, 0), Point(2, 0, 0), Point(3, 0, 0)],
        [Point(0, 1, 0), Point(1, 1, 2), Point(2, 1, 2), Point(3, 1, 0)],
        [Point(0, 2, 0), Point(1, 2, 2), Point(2, 2, 2), Point(3, 2, 0)],
        [Point(0, 3, 0), Point(1, 3, 0), Point(2, 3, 0), Point(3, 3, 0)],
    ]
    w = [[1.0] * 4 for _ in range(4)]
    s = NurbsSurface.create_from_parameters(grid, w, [0, 1], [0, 1], [4, 4], [4, 4], 3, 3)
    MINI_CHECK(s.is_valid())
    MINI_CHECK(s.degree(0) == 3 and s.degree(1) == 3)
    MINI_CHECK(s.cv_count(0) == 4 and s.cv_count(1) == 4)
    MINI_CHECK(not s.is_rational())
    u0, u1 = s.domain(0)
    v0, v1 = s.domain(1)
    MINI_CHECK(abs(u0) < 1e-12 and abs(u1 - 1.0) < 1e-12)
    MINI_CHECK(abs(v0) < 1e-12 and abs(v1 - 1.0) < 1e-12)
    MINI_CHECK(TOLERANCE.is_point_close(s.point_at(0.0, 0.0), Point(0, 0, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(s.point_at(1.0, 1.0), Point(3, 3, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(s.point_at(0.5, 0.5), Point(1.5, 1.5, 1.125)))
    MINI_CHECK(TOLERANCE.is_point_close(s.point_at(0.37, 0.41), Point(1.11, 1.23, 1.01496402)))

    # frame_at (surface_frames.py): origin == point_at, z-axis == normal_at.
    # Normal validated vs OCCT D1uxD1v (validation/compare_surface_eval.py).
    fr = s.frame_at(0.3, 0.4)
    MINI_CHECK(TOLERANCE.is_point_close(fr.origin, s.point_at(0.3, 0.4)))
    n = s.normal_at(0.3, 0.4)
    za = fr.z_axis
    MINI_CHECK(abs(za[0] - n[0]) < 1e-9 and abs(za[1] - n[1]) < 1e-9 and abs(za[2] - n[2]) < 1e-9)

    # intersections_with_line (surface_intersections_with_line.py): a vertical line
    # through (1.5, 1.5) hits the surface once at (1.5, 1.5, 1.125). Validated vs
    # OCCT GeomAPI_IntCS (validation harness, dev <= 1.6e-16).
    from session_py import Line
    hits = s.intersections_with_line(Line(1.5, 1.5, -5, 1.5, 1.5, 5))
    MINI_CHECK(len(hits) == 1)
    MINI_CHECK(TOLERANCE.is_point_close(hits[0], Point(1.5, 1.5, 1.125)))


@MINI_TEST("NurbsSurface", "Booleans Queries")
def test_booleans_queries():
    from session_py import NurbsSurface
    from session_py import Plane
    from session_py import Primitives

    s = Primitives.sphere_surface(0, 0, 0, 5.0)

    # Validity surface and nurbsknots
    is_valid = s.is_valid()
    are_nurbsknots_valid = s.is_valid_nurbsknot_vector(0) and s.is_valid_nurbsknot_vector(1)

    # Are control points weights enabled?
    is_rational = s.is_rational()

    # Sphere has one seam that is closed, but two poles
    is_closed = s.is_closed(0) == True and s.is_closed(1) == False

    # sphere cannot be truly periodic because it has poles
    is_periodic = s.is_periodic(0) and s.is_periodic(1)

    # Planarity
    plane = Plane.xy_plane()
    is_planar = s.is_planar(plane)

    # Surface is collapsed to a point
    is_point = s.is_singular(0) and s.is_singular(1) and s.is_singular(2) and s.is_singular(3)

    # Most surfaces are clamped except periodic surfaces
    is_clamped = s.is_clamped(0, 2) and s.is_clamped(1, 2)

    MINI_CHECK(is_valid)
    MINI_CHECK(are_nurbsknots_valid)
    MINI_CHECK(is_rational)
    MINI_CHECK(is_closed)
    MINI_CHECK(not is_periodic)
    MINI_CHECK(not is_planar)
    MINI_CHECK(not is_point)
    MINI_CHECK(is_clamped)


@MINI_TEST("NurbsSurface", "Attributes")
def test_nurbssurface_attributes():
    from session_py import NurbsSurface
    from session_py import Point

    points = [
        # i=0
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        # i=1
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        # i=2
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        # i=3
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]

    s = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    # Check the dimentions of a surface
    # Mostly 3d
    # But 2d can be used for: scalar field over parameter space e.g. czrvatzre map, distance field
    # Planar geometry: texture coordinates
    dimensions = s.dimension()

    # Degree types 1 - linear, 2 - quadratic, 3 - cubic
    order_u = s.order(0)
    order_v = s.order(1)

    # Control vertex count
    cv_count_u = s.cv_count_dir(0)
    cv_count_v = s.cv_count_dir(1)
    cv_count = s.cv_count_dir(None)
    cv_size = s.cv_size()

    # Number of nurbsknots
    k_count_0 = s.nurbsknot_count(0)
    k_count_1 = s.nurbsknot_count(1)

    # Span count
    s_count_0 = s.span_count(0)
    s_count_1 = s.span_count(1)

    MINI_CHECK(dimensions == 3)
    MINI_CHECK(order_u == 4)
    MINI_CHECK(order_v == 4)
    MINI_CHECK(cv_count_u)
    MINI_CHECK(cv_count_v)
    MINI_CHECK(cv_count)
    MINI_CHECK(cv_size)
    MINI_CHECK(k_count_0)
    MINI_CHECK(k_count_1)
    MINI_CHECK(s_count_0)
    MINI_CHECK(s_count_1)


@MINI_TEST("NurbsSurface", "Control Vertices Access")
def test_control_vertices_access():
    from session_py import NurbsSurface
    from session_py import Point

    points = [
        # i=0
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        # i=1
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        # i=2
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        # i=3
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]

    s = NurbsSurface.create(False, False, 3, 3, 4, 4, points)
    s.make_rational()

    # Raw CV access - cv() returns view of internal storage
    cv_arr = s.cv(0, 0)

    MINI_CHECK(cv_arr[2] == 0)
    cv_arr[2] = 10.0
    MINI_CHECK(cv_arr[2] == 10)

    # Point and Weight
    # NOTE
    # point is (Xw, Yw, Zw, w)
    # cv pointer is (X, Y, Z)
    cv = s.get_cv(0, 0)
    MINI_CHECK(cv == Point(0, 0, 10))
    ok, x, y, z, w = s.get_cv_4d(0, 0)
    MINI_CHECK(x == 0 and y == 0 and z == 10 and w == 1)

    s.set_cv(0, 0, Point(0, 0, 5))
    MINI_CHECK(s.get_cv(0, 0) == Point(0, 0, 5))
    s.set_cv_4d(0, 0, 0, 0, 4, 0.5)
    MINI_CHECK(s.get_cv(0, 0) == Point(0, 0, 8))
    MINI_CHECK(s.cv(0, 0)[2] == 4)
    MINI_CHECK(s.weight(0, 0) == 0.5)

    w = s.weight(0, 0)
    s.set_weight(0, 0, 1)
    MINI_CHECK(s.weight(0, 0) == 1)


@MINI_TEST("NurbsSurface", "NurbsKnot Access")
def test_nurbsknot_access():
    from session_py import NurbsSurface
    from session_py import Point

    points = [
        # i=0
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        # i=1
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        # i=2
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        # i=3
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]

    s = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    # Get nurbsknot vectors and individual nurbsknot
    nurbsknots_u = s.get_nurbsknots(0)
    for i in range(s.nurbsknot_count(0)):
        nurbsknot = s.nurbsknot(0, i)
        MINI_CHECK(nurbsknot == nurbsknots_u[i])

    nurbsknots_v = s.get_nurbsknots(1)
    for i in range(s.nurbsknot_count(1)):
        nurbsknot = s.nurbsknot(1, i)
        MINI_CHECK(nurbsknot == nurbsknots_v[i])

    # Set nurbsknots
    is_set = s.set_nurbsknot(0, 2, 0.5)
    MINI_CHECK(is_set)
    MINI_CHECK(s.nurbsknot(0, 2) == 0.5)
    is_set = s.set_nurbsknot(0, 2, 0.0)
    MINI_CHECK(is_set)

    # Verify start multiplicity
    mult_u_start = s.nurbsknot_multiplicity(0, 0)
    mult_v_start = s.nurbsknot_multiplicity(1, 0)
    MINI_CHECK(mult_u_start == 3)
    MINI_CHECK(mult_v_start == 3)

    s.insert_nurbsknot(0, 0.1, 2)
    MINI_CHECK(s.nurbsknot_count(0) == 8)
    MINI_CHECK(s.nurbsknot(0, 3) == 0.1)
    MINI_CHECK(s.nurbsknot_multiplicity(0, 3) == 2)


@MINI_TEST("NurbsSurface", "Domain")
def test_domain():
    from session_py import NurbsSurface
    from session_py import Point

    points = [
        # i=0
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        # i=1
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        # i=2
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        # i=3
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]

    s = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    # Get domain 0 - 1
    domain_u = s.domain(0)
    domain_v = s.domain(1)

    MINI_CHECK(TOLERANCE.is_close(domain_u[0], 0))
    MINI_CHECK(TOLERANCE.is_close(domain_u[1], 1))
    MINI_CHECK(TOLERANCE.is_close(domain_v[0], 0))
    MINI_CHECK(TOLERANCE.is_close(domain_v[1], 1))

    # Set Domain
    is_set_u = s.set_domain(0, -1.1, 2.3)
    is_set_v = s.set_domain(1, -5.1, 1.3)
    MINI_CHECK(is_set_u)
    MINI_CHECK(TOLERANCE.is_close(s.domain(1)[0], -5.1))
    MINI_CHECK(is_set_v)
    MINI_CHECK(TOLERANCE.is_close(s.domain(1)[1], 1.3))

    # Get sorted list of distinct nurbsknot values
    span_vector = s.get_span_vector(0)
    first_item = span_vector[0]
    last_item = span_vector[-1]
    MINI_CHECK(TOLERANCE.is_close(first_item, -1.1))
    MINI_CHECK(TOLERANCE.is_close(last_item, 2.3))


@MINI_TEST("NurbsSurface", "Division")
def test_division():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Vector

    points = [
        # i=0
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        # i=1
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        # i=2
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        # i=3
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]

    s = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    # points, normals, uv
    division_points, vectors, uvs0 = s.divide_by_count_points(3, 3)

    # planes, uv
    planes, uvs1 = s.divide_by_count_planes(3, 3)

    MINI_CHECK(TOLERANCE.is_point_close(division_points[0][0], Point(0, 0, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[0][1], Point(-0.666666666666667, 1.46296296296296, 1.33333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[0][2], Point(-0.666666666666667, 3.53703703703704, 1.33333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[0][3], Point(0, 5, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[1][0], Point(1.46296296296296, -0.666666666666667, 1.33333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[1][1], Point(1.3641975308642, 1.3641975308642, 2.66666666666667)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[1][2], Point(1.3641975308642, 3.6358024691358, 2.66666666666667)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[1][3], Point(1.46296296296296, 5.66666666666667, 1.33333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[2][0], Point(3.53703703703704, -0.666666666666667, 1.33333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[2][1], Point(3.6358024691358, 1.3641975308642, 2.66666666666667)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[2][2], Point(3.6358024691358, 3.6358024691358, 2.66666666666667)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[2][3], Point(3.53703703703704, 5.66666666666667, 1.33333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[3][0], Point(5, 0, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[3][1], Point(5.66666666666667, 1.46296296296296, 1.33333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[3][2], Point(5.66666666666667, 3.53703703703704, 1.33333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(division_points[3][3], Point(5, 5, 0)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[0][0], Vector(-0.704360725060499, -0.704360725060499, -0.0880450906325624)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[0][1], Vector(-0.722897836195991, -0.327787263130091, 0.608255068661856)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[0][2], Vector(-0.722897836195991, 0.327787263130091, 0.608255068661856)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[0][3], Vector(-0.704360725060499, 0.704360725060499, -0.0880450906325624)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[1][0], Vector(-0.327787263130091, -0.722897836195991, 0.608255068661856)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[1][1], Vector(-0.280457757277237, -0.280457757277237, 0.917979788865771)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[1][2], Vector(-0.280457757277237, 0.280457757277237, 0.917979788865771)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[1][3], Vector(-0.327787263130091, 0.722897836195991, 0.608255068661856)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[2][0], Vector(0.327787263130091, -0.722897836195991, 0.608255068661856)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[2][1], Vector(0.280457757277237, -0.280457757277237, 0.917979788865771)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[2][2], Vector(0.280457757277237, 0.280457757277237, 0.917979788865771)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[2][3], Vector(0.327787263130091, 0.722897836195991, 0.608255068661856)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[3][0], Vector(0.704360725060499, -0.704360725060499, -0.0880450906325624)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[3][1], Vector(0.722897836195991, -0.327787263130091, 0.608255068661856)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[3][2], Vector(0.722897836195991, 0.327787263130091, 0.608255068661856)))
    MINI_CHECK(TOLERANCE.is_vector_close(vectors[3][3], Vector(0.704360725060499, 0.704360725060499, -0.0880450906325624)))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][0][0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][0][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][1][0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][1][1], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][2][0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][2][1], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][3][0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][3][1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][0][0], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][0][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][1][0], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][1][1], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][2][0], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][2][1], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][3][0], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][3][1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][0][0], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][0][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][1][0], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][1][1], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][2][0], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][2][1], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][3][0], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][3][1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][0][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][1][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][1][1], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][2][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][2][1], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][3][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][3][1], 1.0))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[0][0].x_axis, Vector(0.317999364001908, -0.423999152002544, 0.847998304005088)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[0][1].x_axis, Vector(0.657483781160109, -0.0556600026378928, 0.751410035611553)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[0][2].x_axis, Vector(0.657483781160109, 0.055660002637893, 0.751410035611553)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[0][3].x_axis, Vector(0.317999364001908, 0.423999152002544, 0.847998304005088)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[1][0].x_axis, Vector(0.93542594448836, -0.158100159631836, 0.316200319263671)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[1][1].x_axis, Vector(0.957938608304167, -0.0211991946512679, 0.286189127792116)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[1][2].x_axis, Vector(0.957938608304167, 0.0211991946512677, 0.286189127792116)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[1][3].x_axis, Vector(0.93542594448836, 0.158100159631835, 0.316200319263671)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[2][0].x_axis, Vector(0.93542594448836, 0.158100159631835, -0.316200319263671)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[2][1].x_axis, Vector(0.957938608304167, 0.0211991946512679, -0.286189127792116)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[2][2].x_axis, Vector(0.957938608304167, -0.021199194651268, -0.286189127792116)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[2][3].x_axis, Vector(0.93542594448836, -0.158100159631836, -0.316200319263671)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[3][0].x_axis, Vector(0.317999364001908, 0.423999152002544, -0.847998304005088)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[3][1].x_axis, Vector(0.657483781160109, 0.0556600026378928, -0.751410035611553)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[3][2].x_axis, Vector(0.657483781160109, -0.0556600026378928, -0.751410035611553)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[3][3].x_axis, Vector(0.317999364001908, -0.423999152002544, -0.847998304005088)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[0][0].y_axis, Vector(-0.423999152002544, 0.317999364001908, 0.847998304005088)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[0][1].y_axis, Vector(-0.158100159631836, 0.93542594448836, 0.316200319263671)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[0][2].y_axis, Vector(0.158100159631835, 0.93542594448836, -0.316200319263671)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[0][3].y_axis, Vector(0.423999152002544, 0.317999364001908, -0.847998304005088)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[1][0].y_axis, Vector(-0.0556600026378928, 0.657483781160109, 0.751410035611553)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[1][1].y_axis, Vector(-0.0211991946512679, 0.957938608304167, 0.286189127792116)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[1][2].y_axis, Vector(0.0211991946512679, 0.957938608304167, -0.286189127792116)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[1][3].y_axis, Vector(0.0556600026378928, 0.657483781160109, -0.751410035611553)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[2][0].y_axis, Vector(0.0556600026378928, 0.657483781160109, 0.751410035611553)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[2][1].y_axis, Vector(0.0211991946512678, 0.957938608304167, 0.286189127792116)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[2][2].y_axis, Vector(-0.0211991946512678, 0.957938608304167, -0.286189127792116)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[2][3].y_axis, Vector(-0.0556600026378928, 0.657483781160109, -0.751410035611553)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[3][0].y_axis, Vector(0.423999152002544, 0.317999364001908, 0.847998304005088)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[3][1].y_axis, Vector(0.158100159631835, 0.93542594448836, 0.316200319263671)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[3][2].y_axis, Vector(-0.158100159631836, 0.93542594448836, -0.316200319263671)))
    MINI_CHECK(TOLERANCE.is_vector_close(planes[3][3].y_axis, Vector(-0.423999152002544, 0.317999364001908, -0.847998304005088)))


@MINI_TEST("NurbsSurface", "Evaluation")
def test_evaluation():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Vector

    points = [
        # i=0
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        # i=1
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        # i=2
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        # i=3
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]

    s = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    u = 0.5
    v = 0.5

    # point_at(u, v) - returns Point
    p1 = s.point_at(u, v)

    MINI_CHECK(TOLERANCE.is_point_close(p1, Point(2.5, 2.5, 3.0)))

    # normal_at(u, v) - returns Vector
    n1 = s.normal_at(u, v)
    MINI_CHECK(TOLERANCE.is_vector_close(n1, Vector(0, 0, 1)))

    # evaluate(u, v, num_derivs) - returns vector of derivatives
    derivs = s.evaluate(u, v, 1)
    MINI_CHECK(TOLERANCE.is_vector_close(derivs[0], Vector(2.5, 2.5, 3.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(derivs[1], Vector(0.0, 6.9375, 0.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(derivs[2], Vector(6.9375, 0.0, 0.0)))

    # point_at_corner(u_end, v_end) - corner point
    p_corner = s.point_at_corner(1, 1)
    MINI_CHECK(TOLERANCE.is_point_close(p_corner, Point(5.0, 5.0, 0.0)))

    # get isocurve - returns NurbsCurve
    iso_u = s.iso_curve(0, v)
    iso_v = s.iso_curve(1, u)
    MINI_CHECK(TOLERANCE.is_point_close(iso_u.point_at(0.5), Point(2.5, 2.5, 3.0)))
    MINI_CHECK(TOLERANCE.is_point_close(iso_v.point_at(0.5), Point(2.5, 2.5, 3.0)))


@MINI_TEST("NurbsSurface", "Modification")
def test_modification():
    import copy
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Vector

    points = [
        # i=0
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        # i=1
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        # i=2
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        # i=3
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]

    s = NurbsSurface.create(False, False, 3, 3, 4, 4, points)


    # Reverse one direction
    s_rev = copy.deepcopy(s)
    s_rev.reverse(0)

    MINI_CHECK(s_rev.point_at_corner(0, 0) == s.point_at_corner(1, 0))
    MINI_CHECK(s_rev.normal_at(0.5, 0.5) == s.normal_at(0.5, 0.5) * -1)

    # Swap u and v direction
    s_tr = copy.deepcopy(s)
    s_tr.transpose()
    MINI_CHECK(s.point_at(0, 0.5) == s_tr.point_at(0.5, 0))

    # Swap coordinates - swap x and z
    s_swap = copy.deepcopy(s)
    s_swap.swap_coordinates(0, 2)
    MINI_CHECK(s.point_at(0.5, 0.5)[0] == s_swap.point_at(0.5, 0.5)[2])
    MINI_CHECK(s.point_at(0.5, 0.5)[2] == s_swap.point_at(0.5, 0.5)[0])

    # Trim surface, domain changed but parametrization preserved
    s_trim = copy.deepcopy(s)
    s_trim.trim(0, (0.25, 0.75))
    MINI_CHECK(TOLERANCE.is_close(s_trim.domain(0)[0], 0.25))
    MINI_CHECK(TOLERANCE.is_close(s_trim.domain(0)[1], 0.75))
    MINI_CHECK(TOLERANCE.is_point_close(s.point_at(0.25, 0.5), s_trim.point_at(0.25, 0.5)))

    # Split surface into 4 quadrants, check shared corner point is the same
    west, east = s.split(0, 0.5)
    ww, we = west.split(1, (west.domain(1)[0] + west.domain(1)[1]) / 2.0)
    ew, ee = east.split(1, (east.domain(1)[0] + east.domain(1)[1]) / 2.0)
    center = s.point_at(0.5, 0.5)
    MINI_CHECK(TOLERANCE.is_point_close(ww.point_at_corner(1, 1), center))
    MINI_CHECK(TOLERANCE.is_point_close(we.point_at_corner(1, 0), center))
    MINI_CHECK(TOLERANCE.is_point_close(ew.point_at_corner(0, 1), center))
    MINI_CHECK(TOLERANCE.is_point_close(ee.point_at_corner(0, 0), center))

    # Make rational and change weight
    s_rat = copy.deepcopy(s)
    s_rat.make_rational()
    s_rat.set_weight(2, 2, 3.0)
    MINI_CHECK(s.point_at(0.5, 0.5) != s_rat.point_at(0.5, 0.5))
    s_rat.make_non_rational()
    MINI_CHECK(s.point_at(0.5, 0.5) == s_rat.point_at(0.5, 0.5))

    # Increase degree
    s_deg = copy.deepcopy(s)
    s_deg.increase_degree(0, 6)
    s_deg.increase_degree(1, 6)
    MINI_CHECK(s.cv_count(0) == 4)
    MINI_CHECK(s.cv_count(1) == 4)
    MINI_CHECK(s_deg.cv_count(0) == 7)
    MINI_CHECK(s_deg.cv_count(1) == 7)


@MINI_TEST("NurbsSurface", "Transformations")
def test_transformations():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Xform

    points = [
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]

    # Variant 1: transform(xform) - in place
    surface1 = NurbsSurface.create(False, False, 3, 3, 4, 4, points)
    surface1_xf = Xform.translation(0.0, 0.0, 1.0)
    surface1.transform(surface1_xf)

    MINI_CHECK(surface1.cv(0, 0)[2] == 1.0)

    # Variant 2: transform(xform) - in place, matrix built separately
    surface2 = NurbsSurface.create(False, False, 3, 3, 4, 4, points)
    x = Xform.translation(0.0, 0.0, 1.0)
    surface2.transform(x)
    MINI_CHECK(surface2.cv(0, 0)[2] == 1.0)

    # Variant 3: transformed(xform) - returns a copy
    surface3 = NurbsSurface.create(False, False, 3, 3, 4, 4, points)
    surface3_xf = Xform.translation(0.0, 0.0, 10.0)
    surface3_transformed = surface3.transformed(surface3_xf)
    MINI_CHECK(surface3_transformed.cv(0, 0)[2] == 10.0)

    # Variant 4: transformed(xform) - returns a copy, matrix built separately
    surface4 = NurbsSurface.create(False, False, 3, 3, 4, 4, points)
    x = Xform.translation(0.0, 0.0, 10.0)
    surface4_transformed = surface4.transformed(x)
    MINI_CHECK(surface4_transformed.cv(0, 0)[2] == 10.0)


@MINI_TEST("NurbsSurface", "Meshing")
def test_meshing():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Mesh
    from session_py import Primitives
    from session_py import Vector
    from session_py import Point

    # 1. Sphere — two poles, closed U, rational
    sphere = Primitives.sphere_surface(0, 0, 0, 3.0)
    mesh_sphere = sphere.mesh()
    mesh_sphere_adaptive = sphere.mesh_adaptive(45.0)

    MINI_CHECK(mesh_sphere.is_valid())
    MINI_CHECK(mesh_sphere_adaptive.is_valid())

    # 2. Cone — singular apex (pole), closed U
    cone = Primitives.cone_surface(0, 12, 0, 2.0, 6.0)
    mesh_cone = cone.mesh()
    mesh_cone_adaptive = cone.mesh_adaptive(45.0)
    MINI_CHECK(mesh_cone.is_valid())
    MINI_CHECK(mesh_cone_adaptive.is_valid())

    # 3. Torus — doubly closed (U and V), rational
    torus = Primitives.torus_surface(0, 24, 0, 4.0, 1.5)
    mesh_torus = torus.mesh()
    mesh_torus_adaptive = torus.mesh_adaptive(45.0)
    MINI_CHECK(mesh_torus.is_valid())
    MINI_CHECK(mesh_torus_adaptive.is_valid())

    # 4. Loft — varying radius circles, closed U, multi-span V
    loft = Primitives.create_loft([
        Primitives.circle(0, 38, 0, 2.0),
        Primitives.circle(0, 38, 2, 1.0),
        Primitives.circle(0, 38, 4, 1.5),
        Primitives.circle(0, 38, 6, 0.8)], 3)
    mesh_loft = loft.mesh()
    mesh_loft_adaptive = loft.mesh_adaptive(45.0)
    MINI_CHECK(mesh_loft.is_valid())
    MINI_CHECK(mesh_loft_adaptive.is_valid())

    # 5. Extrusion (circle) — closed U, linear V, rational
    ext_dir = Vector(0, 0, 5)
    cylinder = Primitives.create_extrusion(Primitives.circle(0, 52, 0, 3.0), ext_dir)
    mesh_cylinder = cylinder.mesh()
    mesh_cylinder_adaptive = cylinder.mesh_adaptive(45.0)
    MINI_CHECK(mesh_cylinder.is_valid())
    MINI_CHECK(mesh_cylinder_adaptive.is_valid())

    # 6. Ruled — bilinear (degree 1x1), tests twist subdivision
    ra = NurbsCurve.create(False, 1, [
        Point(0, 64, 0),
        Point(5, 64, 5),
    ])
    rb = NurbsCurve.create(False, 1, [
        Point(0, 69, 5),
        Point(5, 69, 0),
    ])
    hypar = Primitives.create_ruled(ra, rb)
    mesh_hypar = hypar.mesh()
    mesh_hypar_adaptive = hypar.mesh_adaptive(45.0)
    MINI_CHECK(mesh_hypar.is_valid())
    MINI_CHECK(mesh_hypar_adaptive.is_valid())

    # 7. Sweep1 — circle along curved rail
    profile = Primitives.circle(0, 0, 0, 1.0)
    rail = NurbsCurve.create(False, 2, [
        Point(0, 76, 0),
        Point(0, 81, 0),
        Point(2, 85, 0),
    ])
    sweep1 = Primitives.create_sweep1(rail, profile)
    mesh_sweep1 = sweep1.mesh()
    mesh_sweep1_adaptive = sweep1.mesh_adaptive(45.0)
    MINI_CHECK(mesh_sweep1.is_valid())
    MINI_CHECK(mesh_sweep1_adaptive.is_valid())

    # 8. Sweep2 — two rails + cross sections
    r1 = NurbsCurve.create(False, 2, [
        Point(0, 89, 0),
        Point(1, 93, 0),
        Point(2, 94, 0),
    ])
    r2 = NurbsCurve.create(False, 2, [
        Point(4, 89, 0),
        Point(4, 93, 0),
        Point(3, 94, 0),
    ])
    sh1 = NurbsCurve.create(False, 2, [
        Point(0, 89, 0),
        Point(2, 89, 2),
        Point(4, 89, 0),
    ])
    sh2 = NurbsCurve.create(False, 2, [
        Point(2, 94, 0),
        Point(2.5, 94, 1.5),
        Point(3, 94, 0),
    ])
    sweep2 = Primitives.create_sweep2(r1, r2, [sh1, sh2])
    mesh_sweep2 = sweep2.mesh()
    mesh_sweep2_adaptive = sweep2.mesh_adaptive(45.0)
    MINI_CHECK(mesh_sweep2.is_valid())
    MINI_CHECK(mesh_sweep2_adaptive.is_valid())

    # 9. Edge surface (Coons patch) — 4 boundary curves
    south = NurbsCurve.create(False, 3, [
        Point(1, 104, 0),
        Point(1, 106, 3),
        Point(1, 109, 3),
        Point(1, 111, 0),
    ])
    west  = NurbsCurve.create(False, 2, [
        Point(10, 104, 0),
        Point(5.5, 104, 3.5),
        Point(1, 104, 0),
    ])
    north = NurbsCurve.create(False, 3, [
        Point(10, 104, 0),
        Point(10, 106, 3),
        Point(10, 109, 3),
        Point(10, 111, 0),
    ])
    east  = NurbsCurve.create(False, 2, [
        Point(10, 111, 0),
        Point(5.5, 111, 3.5),
        Point(1, 111, 0),
    ])
    arched = Primitives.create_edge(south, west, north, east)
    mesh_arched = arched.mesh()
    mesh_arched_adaptive = arched.mesh_adaptive(45.0)
    MINI_CHECK(mesh_arched.is_valid())
    MINI_CHECK(mesh_arched_adaptive.is_valid())

    # 10. Wave — multi-span freeform (13x13 CVs, 10 spans)
    wave = Primitives.wave_surface(5.0, 1.5)
    mesh_wave = wave.mesh()
    mesh_wave_adaptive = wave.mesh_adaptive(45.0)
    MINI_CHECK(mesh_wave.is_valid())
    MINI_CHECK(mesh_wave_adaptive.is_valid())

    # 11. Planar — mesh() early exit: 2 triangles
    planar = NurbsCurve.create(False, 1, [
        Point(0, 132, 0),
        Point(6, 132, 0),
        Point(6, 136, 0),
        Point(0, 136, 0),
        Point(0, 132, 0),
    ])
    pln = Primitives.create_planar(planar)
    mesh_planar = pln.mesh()
    mesh_planar_adaptive = pln.mesh_adaptive(45.0)
    MINI_CHECK(mesh_planar.is_valid())
    MINI_CHECK(mesh_planar_adaptive.is_valid())


@MINI_TEST("NurbsSurface", "Split By Plane")
def test_nurbssurface_split_by_plane():
    from session_py import Plane
    from session_py import Point
    from session_py import Vector
    from session_py import Primitives

    cyl = Primitives.cylinder_surface(0.0, 0.0, 0.0, 1.0, 4.0)
    plane = Plane.from_point_normal(Point(0.0, 0.0, 2.0), Vector(0.3, 0.0, 1.0))

    parts = cyl.split_by_plane(plane)

    MINI_CHECK(len(parts) == 2)
    for ts in parts:
        MINI_CHECK(ts.is_trimmed())
        m = ts.mesh_q(20.0, 0.005)
        MINI_CHECK(m.number_of_faces() > 0)

    sphere = Primitives.sphere_surface(0.0, 0.0, 0.0, 1.0)
    plane2 = Plane.from_point_normal(Point(0.0, 0.0, 0.3), Vector(0.0, 0.0, 1.0))

    caps = sphere.split_by_plane(plane2)

    MINI_CHECK(len(caps) == 2)


@MINI_TEST("NurbsSurface", "Split By Curves")
def test_nurbssurface_split_by_curves():
    import math
    from session_py import Closest
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Primitives

    wave = Primitives.wave_surface(10.0, 1.0)
    lift_pts = []
    for i in range(21):
        x = 10.0 * i / 20.0
        y = 5.0 + 2.0 * math.sin(x)
        u, v, d = Closest.surface_point(wave, Point(x, y, 0.0))
        lift_pts.append(wave.point_at(u, v))
    crv = NurbsCurve.create_interpolated(lift_pts)

    parts = wave.split_by_curves([crv])

    MINI_CHECK(len(parts) == 2)
    MINI_CHECK(parts[0].is_trimmed())
    MINI_CHECK(parts[1].is_trimmed())

    off = NurbsCurve.create(False, 1, [Point(50.0, 50.0, 50.0), Point(60.0, 60.0, 60.0)])

    whole = wave.split_by_curves([off])

    MINI_CHECK(len(whole) == 1)


@MINI_TEST("NurbsSurface", "Split By Line")
def test_nurbssurface_split_by_line():
    from session_py import Line
    from session_py import Point
    from session_py import Primitives

    wave = Primitives.wave_surface(10.0, 1.0)
    line = Line.from_points(Point(-1.0, 5.0, 0.0), Point(11.0, 5.0, 0.0))

    parts = wave.split_by_line(line)

    MINI_CHECK(len(parts) == 2)
    MINI_CHECK(parts[0].is_trimmed())
    MINI_CHECK(parts[1].is_trimmed())


@MINI_TEST("NurbsSurface", "Split By Surface")
def test_nurbssurface_split_by_surface():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Primitives

    cyl = Primitives.cylinder_surface(0.0, 0.0, -2.0, 1.0, 4.0)
    flat = NurbsSurface.create(False, False, 1, 1, 2, 2, [
        Point(-3.0, -3.0, 0.0),
        Point(-3.0, 3.0, 0.0),
        Point(3.0, -3.0, 0.0),
        Point(3.0, 3.0, 0.0),
    ])

    parts = cyl.split_by_surface(flat)

    MINI_CHECK(len(parts) == 2)
    for ts in parts:
        MINI_CHECK(ts.is_trimmed())
        m = ts.mesh_q(20.0, 0.005)
        MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("NurbsSurface", "Split By Brep")
def test_nurbssurface_split_by_brep():
    from session_py import NurbsSurface
    from session_py import BRep
    from session_py import Point

    flat = NurbsSurface.create(False, False, 1, 1, 2, 2, [
        Point(-3.0, -3.0, 0.0),
        Point(-3.0, 3.0, 0.0),
        Point(3.0, -3.0, 0.0),
        Point(3.0, 3.0, 0.0),
    ])
    cutter = BRep.create_box(2.0, 2.0, 2.0)

    parts = flat.split_by_brep(cutter)

    MINI_CHECK(len(parts) == 2)
    for ts in parts:
        MINI_CHECK(ts.is_trimmed())
        m = ts.mesh_q(20.0, 0.005)
        MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("NurbsSurface", "Json Roundtrip")
def test_json_roundtrip():
    from session_py import NurbsSurface
    from session_py import Point
    from pathlib import Path

    points = [
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]
    surface = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    # JSON object
    json_obj = surface.__jsondump__()
    loaded_json = NurbsSurface.__jsonload__(json_obj)

    # String
    json_string = surface.file_json_dumps()
    loaded_json_string = NurbsSurface.file_json_loads(json_string)

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbssurface.json"
    surface.file_json_dump(filename)
    loaded_from_file = NurbsSurface.file_json_load(filename)

    MINI_CHECK(loaded_json == surface)
    MINI_CHECK(loaded_json_string == surface)
    MINI_CHECK(loaded_from_file == surface)


@MINI_TEST("NurbsSurface", "Protobuf Roundtrip")
def test_protobuf_roundtrip():
    from session_py import NurbsSurface
    from session_py import Point
    from pathlib import Path

    points = [
        Point(0.0, 0.0, 0.0),
        Point(-1.0, 0.75, 2.0),
        Point(-1.0, 4.25, 2.0),
        Point(0.0, 5.0, 0.0),
        Point(0.75, -1.0, 2.0),
        Point(1.25, 1.25, 4.0),
        Point(1.25, 3.75, 4.0),
        Point(0.75, 6.0, 2.0),
        Point(4.25, -1.0, 2.0),
        Point(3.75, 1.25, 4.0),
        Point(3.75, 3.75, 4.0),
        Point(4.25, 6.0, 2.0),
        Point(5.0, 0.0, 0.0),
        Point(6.0, 0.75, 2.0),
        Point(6.0, 4.25, 2.0),
        Point(5.0, 5.0, 0.0),
    ]
    surface = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    # String
    proto_string = surface.pb_dumps()
    loaded_proto_string = NurbsSurface.pb_loads(proto_string)

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbssurface.bin"
    surface.pb_dump(filename)
    loaded = NurbsSurface.pb_load(filename)

    MINI_CHECK(loaded_proto_string == surface)
    MINI_CHECK(loaded == surface)


@MINI_TEST("NurbsSurface", "ClosestPoint")
def test_nurbssurface_closest_point():
    from session_py import Primitives
    from session_py import Point
    # Sphere radius 2 at origin: closest surface point to an outside point is radial.
    sphere = Primitives.sphere_surface(0, 0, 0, 2.0)
    cp = sphere.closest_point(Point(5, 0, 0))
    MINI_CHECK(abs(cp[0] - 2.0) < 1e-4 and abs(cp[1]) < 1e-4 and abs(cp[2]) < 1e-4)
    # Curvature: sphere radius R has Gaussian K = 1/R^2, |mean| = 1/R.
    u0, u1 = sphere.domain(0)
    v0, v1 = sphere.domain(1)
    um = u0 + 0.37 * (u1 - u0)
    vm = v0 + 0.41 * (v1 - v0)
    MINI_CHECK(abs(sphere.gaussian_curvature(um, vm) - 0.25) < 1e-3)
    MINI_CHECK(abs(abs(sphere.mean_curvature(um, vm)) - 0.5) < 1e-3)


if __name__ == "__main__":
    run_all(language="python")
