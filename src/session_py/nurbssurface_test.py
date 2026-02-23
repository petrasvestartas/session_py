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
    MINI_CHECK(s.knot_count(0) == 6)
    MINI_CHECK(s.knot_count(1) == 6)
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


@MINI_TEST("NurbsSurface", "Booleans Queries")
def test_booleans_queries():
    from session_py import NurbsSurface
    from session_py import Plane
    from session_py import Primitives

    s = Primitives.sphere_surface(0, 0, 0, 5.0)

    # Validity surface and knots
    is_valid = s.is_valid()
    are_knots_valid = s.is_valid_knot_vector(0) and s.is_valid_knot_vector(1)

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
    MINI_CHECK(are_knots_valid)
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

    # Number of knots
    k_count_0 = s.knot_count(0)
    k_count_1 = s.knot_count(1)

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
    MINI_CHECK(s.get_cv(0, 0) == Point(0, 0, 8) and s.cv(0, 0)[2] == 4 and s.weight(0, 0) == 0.5)

    w = s.weight(0, 0)
    s.set_weight(0, 0, 1)
    MINI_CHECK(s.weight(0, 0) == 1)


@MINI_TEST("NurbsSurface", "Knot Access")
def test_knot_access():
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

    # Get knot vectors and individual knot
    knots_u = s.get_knots(0)
    for i in range(s.knot_count(0)):
        knot = s.knot(0, i)
        MINI_CHECK(knot == knots_u[i])

    knots_v = s.get_knots(1)
    for i in range(s.knot_count(1)):
        knot = s.knot(1, i)
        MINI_CHECK(knot == knots_v[i])

    # Set knots
    is_set = s.set_knot(0, 2, 0.5)
    MINI_CHECK(s.knot(0, 2) == 0.5)
    is_set = s.set_knot(0, 2, 0.0)

    # Verify start multiplicity
    mult_u_start = s.knot_multiplicity(0, 0)
    mult_v_start = s.knot_multiplicity(1, 0)
    MINI_CHECK(mult_u_start == 3)
    MINI_CHECK(mult_v_start == 3)

    s.insert_knot(0, 0.1, 2)
    MINI_CHECK(s.knot_count(0) == 8)
    MINI_CHECK(s.knot(0, 3) == 0.1)
    MINI_CHECK(s.knot_multiplicity(0, 3) == 2)


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

    # Set Domain
    is_set_u = s.set_domain(0, -1.1, 2.3)
    is_set_v = s.set_domain(1, -5.1, 1.3)
    MINI_CHECK(is_set_u and TOLERANCE.is_close(s.domain(1)[0], -5.1))
    MINI_CHECK(is_set_v and TOLERANCE.is_close(s.domain(1)[1], 1.3))

    # Get sorted list of distinct knot values
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
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][0][0], 0.0) and TOLERANCE.is_close(uvs0[0][0][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][1][0], 0.0) and TOLERANCE.is_close(uvs0[0][1][1], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][2][0], 0.0) and TOLERANCE.is_close(uvs0[0][2][1], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[0][3][0], 0.0) and TOLERANCE.is_close(uvs0[0][3][1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][0][0], 0.333333333333333) and TOLERANCE.is_close(uvs0[1][0][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][1][0], 0.333333333333333) and TOLERANCE.is_close(uvs0[1][1][1], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][2][0], 0.333333333333333) and TOLERANCE.is_close(uvs0[1][2][1], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[1][3][0], 0.333333333333333) and TOLERANCE.is_close(uvs0[1][3][1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][0][0], 0.666666666666667) and TOLERANCE.is_close(uvs0[2][0][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][1][0], 0.666666666666667) and TOLERANCE.is_close(uvs0[2][1][1], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][2][0], 0.666666666666667) and TOLERANCE.is_close(uvs0[2][2][1], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[2][3][0], 0.666666666666667) and TOLERANCE.is_close(uvs0[2][3][1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][0][0], 1.0) and TOLERANCE.is_close(uvs0[3][0][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][1][0], 1.0) and TOLERANCE.is_close(uvs0[3][1][1], 0.333333333333333))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][2][0], 1.0) and TOLERANCE.is_close(uvs0[3][2][1], 0.666666666666667))
    MINI_CHECK(TOLERANCE.is_close(uvs0[3][3][0], 1.0) and TOLERANCE.is_close(uvs0[3][3][1], 1.0))
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
    MINI_CHECK(TOLERANCE.is_close(s_trim.domain(0)[0], 0.25) and TOLERANCE.is_close(s_trim.domain(0)[1], 0.75))
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
    MINI_CHECK(s.cv_count(0) == 4 and s.cv_count(1) == 4)
    MINI_CHECK(s_deg.cv_count(0) == 7 and s_deg.cv_count(1) == 7)


@MINI_TEST("NurbsSurface", "Isocurve")
def test_isocurve():
    from session_py import NurbsSurface
    from session_py import Point

    # Create surface
    points = [Point(float(i), float(j), 0.0) for i in range(3) for j in range(3)]
    surf = NurbsSurface.create(False, False, 2, 2, 3, 3, points)

    # Extract iso-u curve (v varies)
    u0, u1 = surf.domain(0)
    u_mid = (u0 + u1) / 2.0
    iso_u = surf.iso_curve(0, u_mid)

    # Extract iso-v curve (u varies)
    v0, v1 = surf.domain(1)
    v_mid = (v0 + v1) / 2.0
    iso_v = surf.iso_curve(1, v_mid)

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(iso_u is not None)
    MINI_CHECK(iso_u.is_valid())
    MINI_CHECK(iso_v is not None)
    MINI_CHECK(iso_v.is_valid())


@MINI_TEST("NurbsSurface", "Transformation")
def test_transformation():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Xform

    # Create simple surface
    points = [
        Point(0.0, 0.0, 0.0), Point(0.0, 1.0, 0.0),
        Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0),
    ]
    surf = NurbsSurface.create(False, False, 1, 1, 2, 2, points)

    # Apply translation
    xf = Xform.translation(1.0, 2.0, 3.0)
    surf.transform(xf)

    # Check transformed CV
    pt = surf.get_cv(0, 0)

    MINI_CHECK(TOLERANCE.is_close(pt[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 3.0))


@MINI_TEST("NurbsSurface", "Json_roundtrip")
def test_json_roundtrip():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Color
    from pathlib import Path

    points = [Point(float(i), float(j), 0.0) for i in range(3) for j in range(3)]
    surf = NurbsSurface.create(False, False, 2, 2, 3, 3, points)
    surf.name = "test_nurbssurface"
    surf.width = 2.0
    surf.facecolors = [Color(255, 128, 64, 255)]
    surf.pointcolors = [Color(0, 255, 0, 255)]
    surf.linecolors = [Color(0, 0, 255, 255)]

    # JSON object
    json_obj = surf.__jsondump__()
    loaded_json = NurbsSurface.__jsonload__(json_obj)

    # String
    json_string = surf.json_dumps()
    loaded_json_string = NurbsSurface.json_loads(json_string)

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbssurface.json"
    surf.json_dump(filename)
    loaded_from_file = NurbsSurface.json_load(filename)

    MINI_CHECK(loaded_json == surf)
    MINI_CHECK(loaded_json_string == surf)
    MINI_CHECK(loaded_from_file == surf)


@MINI_TEST("NurbsSurface", "Protobuf_roundtrip")
def test_protobuf_roundtrip():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Color
    from pathlib import Path

    points = [Point(float(i), float(j), 0.0) for i in range(3) for j in range(3)]
    surf = NurbsSurface.create(False, False, 2, 2, 3, 3, points)
    surf.name = "test_nurbssurface"
    surf.width = 2.0
    surf.facecolors = [Color(255, 128, 64, 255)]
    surf.pointcolors = [Color(0, 255, 0, 255)]
    surf.linecolors = [Color(0, 0, 255, 255)]

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    # String
    proto_string = surf.pb_dumps()
    loaded_proto_string = NurbsSurface.pb_loads(proto_string)

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbssurface.bin"
    surf.pb_dump(filename)
    loaded = NurbsSurface.pb_load(filename)

    MINI_CHECK(loaded_proto_string == surf)
    MINI_CHECK(loaded == surf)


@MINI_TEST("NurbsSurface", "Advanced_accessors")
def test_advanced_accessors():
    from session_py import NurbsSurface
    from session_py import Point

    # Create rational surface for testing get_cv_4d/set_cv_4d
    points = [Point(0.0, 0.0, 0.0)] * 9
    surf = NurbsSurface.create(False, False, 2, 2, 3, 3, points)
    surf.make_rational()

    # Test set_cv_4d with homogeneous coordinates
    x, y, z, w = 2.0, 3.0, 4.0, 2.0

    # Set CV using set_cv_4d
    surf.set_cv_4d(1, 1, x, y, z, w)

    # Get CV and verify using get_cv_4d
    ok, rx, ry, rz, rw = surf.get_cv_4d(1, 1)

    # Also test get_cv
    pt = surf.get_cv(1, 1)
    retrieved_w = surf.weight(1, 1)

    # Test knot_multiplicity
    mult = surf.knot_count(0)
    first_knot_mult = 0
    if mult > 0:
        first_val = surf.knot(0, 0)
        count = 1
        for i in range(1, mult):
            val = surf.knot(0, i)
            if abs(val - first_val) < 1e-10:
                count += 1
            else:
                break
        first_knot_mult = count

    MINI_CHECK(surf.is_rational())
    MINI_CHECK(TOLERANCE.is_close(rx, x))
    MINI_CHECK(TOLERANCE.is_close(ry, y))
    MINI_CHECK(TOLERANCE.is_close(rz, z))
    MINI_CHECK(TOLERANCE.is_close(rw, w))
    # get_cv returns Euclidean coordinates, so it divides homogeneous coords by w
    MINI_CHECK(TOLERANCE.is_close(pt[0], x/w))
    MINI_CHECK(TOLERANCE.is_close(pt[1], y/w))
    MINI_CHECK(TOLERANCE.is_close(pt[2], z/w))
    MINI_CHECK(TOLERANCE.is_close(retrieved_w, w))
    MINI_CHECK(first_knot_mult > 0)


@MINI_TEST("NurbsSurface", "Singularity")
def test_singularity():
    from session_py import NurbsSurface
    from session_py import Point

    # Create a simple surface with all CVs at different points (non-singular)
    points = [
        Point(0.0, 0.0, 0.0), Point(0.0, 1.0, 0.0),
        Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0),
    ]
    surf = NurbsSurface.create(False, False, 1, 1, 2, 2, points)

    # Test is_singular for each side
    is_singular_south = surf.is_singular(0)
    is_singular_east = surf.is_singular(1)
    is_singular_north = surf.is_singular(2)
    is_singular_west = surf.is_singular(3)

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(not is_singular_south)
    MINI_CHECK(not is_singular_east)
    MINI_CHECK(not is_singular_north)
    MINI_CHECK(not is_singular_west)


@MINI_TEST("NurbsSurface", "Domain_operations")
def test_domain_operations():
    from session_py import NurbsSurface
    from session_py import Point

    points = [Point(0.0, 0.0, 0.0)] * 9
    surf = NurbsSurface.create(False, False, 2, 2, 3, 3, points)

    # Get initial domain
    dom_u = surf.domain(0)
    dom_v = surf.domain(1)

    # Set new domain
    surf.set_domain(0, 0.0, 10.0)
    surf.set_domain(1, 5.0, 15.0)

    new_dom_u = surf.domain(0)
    new_dom_v = surf.domain(1)

    # Get span vectors
    span_u = surf.get_span_vector(0)
    span_v = surf.get_span_vector(1)

    MINI_CHECK(dom_u[0] == 0.0 and dom_u[1] > 0.0)
    MINI_CHECK(dom_v[0] == 0.0 and dom_v[1] > 0.0)
    MINI_CHECK(TOLERANCE.is_close(new_dom_u[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(new_dom_u[1], 10.0))
    MINI_CHECK(TOLERANCE.is_close(new_dom_v[0], 5.0))
    MINI_CHECK(TOLERANCE.is_close(new_dom_v[1], 15.0))
    MINI_CHECK(len(span_u) > 0)
    MINI_CHECK(len(span_v) > 0)


@MINI_TEST("NurbsSurface", "Corner_points")
def test_corner_points():
    from session_py import NurbsSurface
    from session_py import Point

    points = [
        Point(0.0, 0.0, 0.0), Point(0.0, 10.0, 0.0),
        Point(10.0, 0.0, 0.0), Point(10.0, 10.0, 0.0),
    ]
    surf = NurbsSurface.create(False, False, 1, 1, 2, 2, points)

    # Get corner points
    p00 = surf.point_at_corner(0, 0)
    p10 = surf.point_at_corner(1, 0)
    p01 = surf.point_at_corner(0, 1)
    p11 = surf.point_at_corner(1, 1)

    MINI_CHECK(TOLERANCE.is_close(p00[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(p10[0], 10.0))
    MINI_CHECK(TOLERANCE.is_close(p01[1], 10.0))
    MINI_CHECK(TOLERANCE.is_close(p11[0], 10.0) and TOLERANCE.is_close(p11[1], 10.0))


@MINI_TEST("NurbsSurface", "Swap_coordinates")
def test_swap_coordinates():
    from session_py import NurbsSurface
    from session_py import Point

    points = [
        Point(1.0, 2.0, 3.0), Point(0.0, 0.0, 0.0),
        Point(0.0, 0.0, 0.0), Point(0.0, 0.0, 0.0),
    ]
    surf = NurbsSurface.create(False, False, 1, 1, 2, 2, points)

    # Swap X and Y
    surf.swap_coordinates(0, 1)

    pt = surf.get_cv(0, 0)

    MINI_CHECK(TOLERANCE.is_close(pt[0], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 3.0))


@MINI_TEST("NurbsSurface", "Get_knots")
def test_get_knots():
    from session_py import NurbsSurface
    from session_py import Point

    points = [Point(float(i), float(j), 0.0) for i in range(4) for j in range(3)]
    surf = NurbsSurface.create(False, False, 3, 2, 4, 3, points)

    knots_u = surf.get_knots(0)
    knots_v = surf.get_knots(1)

    MINI_CHECK(len(knots_u) == surf.knot_count(0))
    MINI_CHECK(len(knots_v) == surf.knot_count(1))
    MINI_CHECK(len(knots_u) > 0)
    MINI_CHECK(len(knots_v) > 0)


@MINI_TEST("NurbsSurface", "Make_non_rational")
def test_make_non_rational():
    from session_py import NurbsSurface
    from session_py import Point

    # Create surface, then make rational with all weights = 1
    points = [Point(float(i), float(j), 0.0) for i in range(3) for j in range(3)]
    surf = NurbsSurface.create(False, False, 2, 2, 3, 3, points)
    surf.make_rational()

    # Set all weights to 1.0
    for i in range(3):
        for j in range(3):
            surf.set_weight(i, j, 1.0)

    was_rational = surf.is_rational()
    surf.make_non_rational()
    is_rational_after = surf.is_rational()

    MINI_CHECK(was_rational)
    MINI_CHECK(not is_rational_after)


@MINI_TEST("NurbsSurface", "Create_clamped_uniform")
def test_create_clamped_uniform():
    from session_py import NurbsSurface

    surf = NurbsSurface()
    surf.create_clamped_uniform(3, 4, 3, 4, 4, 1.0, 2.0)

    dom_u = surf.domain(0)
    dom_v = surf.domain(1)

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(surf.dimension() == 3)
    MINI_CHECK(surf.order(0) == 4)
    MINI_CHECK(surf.order(1) == 3)
    MINI_CHECK(surf.cv_count_dir(0) == 4)
    MINI_CHECK(surf.cv_count_dir(1) == 4)
    MINI_CHECK(surf.is_clamped(0, 0) and surf.is_clamped(0, 1))
    MINI_CHECK(surf.is_clamped(1, 0) and surf.is_clamped(1, 1))


@MINI_TEST("NurbsSurface", "Knot_multiplicity")
def test_knot_multiplicity():
    from session_py import NurbsSurface
    from session_py import Point

    points = [Point(float(i), float(j), 0.0) for i in range(4) for j in range(4)]
    surf = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    # Check first knot multiplicity (should be equal to degree for clamped)
    mult_u_start = surf.knot_multiplicity(0, 0)
    mult_v_start = surf.knot_multiplicity(1, 0)

    # Check last knot multiplicity
    last_u = surf.knot_count(0) - 1
    last_v = surf.knot_count(1) - 1
    mult_u_end = surf.knot_multiplicity(0, last_u)
    mult_v_end = surf.knot_multiplicity(1, last_v)

    MINI_CHECK(mult_u_start >= surf.degree(0))
    MINI_CHECK(mult_v_start >= surf.degree(1))
    MINI_CHECK(mult_u_end >= surf.degree(0))
    MINI_CHECK(mult_v_end >= surf.degree(1))


if __name__ == "__main__":
    run_all(language="python")
