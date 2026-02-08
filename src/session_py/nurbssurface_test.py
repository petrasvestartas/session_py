from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("NurbsSurface", "constructor")
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

    # Minimal and Full String Representation
    sstr = str(s)
    srepr = repr(s)

    # Copy (duplicates everything except guid)
    scopy = s.duplicate()
    sother = NurbsSurface.create(False, False, 3, 3, 4, 4, points)

    # Point division matching Rhino's 4x6 grid
    v, uv = s.divide_by_count(4, 6)

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
    MINI_CHECK("name=my_nurbssurface" in srepr)
    MINI_CHECK(scopy.cv_count_dir(None) == s.cv_count_dir(None))
    MINI_CHECK(scopy.guid != s.guid)

    MINI_CHECK(TOLERANCE.is_point_close(v[0][0], Point(0.000000000000000, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[0][1], Point(-0.416666666666667, 0.578703703703704, 0.833333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[0][2], Point(-0.666666666666667, 1.462962962962963, 1.333333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[0][3], Point(-0.750000000000000, 2.500000000000000, 1.500000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[0][4], Point(-0.666666666666667, 3.537037037037037, 1.333333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[0][5], Point(-0.416666666666667, 4.421296296296297, 0.833333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[0][6], Point(0.000000000000000, 5.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[1][0], Point(0.992187500000000, -0.562500000000000, 1.125000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[1][1], Point(0.881510416666667, 0.333912037037037, 1.958333333333334)))
    MINI_CHECK(TOLERANCE.is_point_close(v[1][2], Point(0.815104166666667, 1.379629629629630, 2.458333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[1][3], Point(0.792968750000000, 2.500000000000000, 2.625000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[1][4], Point(0.815104166666667, 3.620370370370370, 2.458333333333334)))
    MINI_CHECK(TOLERANCE.is_point_close(v[1][5], Point(0.881510416666667, 4.666087962962964, 1.958333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[1][6], Point(0.992187500000000, 5.562500000000000, 1.125000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[2][0], Point(2.500000000000000, -0.750000000000000, 1.500000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[2][1], Point(2.500000000000000, 0.252314814814815, 2.333333333333334)))
    MINI_CHECK(TOLERANCE.is_point_close(v[2][2], Point(2.500000000000000, 1.351851851851852, 2.833333333333334)))
    MINI_CHECK(TOLERANCE.is_point_close(v[2][3], Point(2.500000000000000, 2.500000000000000, 3.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[2][4], Point(2.500000000000000, 3.648148148148148, 2.833333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[2][5], Point(2.500000000000000, 4.747685185185186, 2.333333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[2][6], Point(2.500000000000000, 5.750000000000000, 1.500000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[3][0], Point(4.007812500000000, -0.562500000000000, 1.125000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[3][1], Point(4.118489583333334, 0.333912037037037, 1.958333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[3][2], Point(4.184895833333334, 1.379629629629630, 2.458333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[3][3], Point(4.207031250000000, 2.500000000000000, 2.625000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[3][4], Point(4.184895833333333, 3.620370370370370, 2.458333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[3][5], Point(4.118489583333333, 4.666087962962964, 1.958333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[3][6], Point(4.007812500000000, 5.562500000000000, 1.125000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[4][0], Point(5.000000000000000, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[4][1], Point(5.416666666666668, 0.578703703703704, 0.833333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[4][2], Point(5.666666666666668, 1.462962962962963, 1.333333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[4][3], Point(5.750000000000000, 2.500000000000000, 1.500000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(v[4][4], Point(5.666666666666666, 3.537037037037037, 1.333333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[4][5], Point(5.416666666666667, 4.421296296296297, 0.833333333333333)))
    MINI_CHECK(TOLERANCE.is_point_close(v[4][6], Point(5.000000000000000, 5.000000000000000, 0.000000000000000)))


@MINI_TEST("NurbsSurface", "create_operations")
def test_create_operations():
    from session_py import NurbsSurface
    from session_py import Point

    # Create a simple 2x2 bilinear surface
    surf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)

    # Set corner control points
    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    surf.set_cv(1, 1, Point(1.0, 1.0, 0.0))

    # Check knot vectors
    u0, u1 = surf.domain(0)
    v0, v1 = surf.domain(1)

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(u0 == 0.0)
    MINI_CHECK(v0 == 0.0)


@MINI_TEST("NurbsSurface", "accessors")
def test_accessors():
    from session_py import NurbsSurface

    surf = NurbsSurface.create_raw(3, False, 4, 3, 5, 4, False, False, 1.0, 1.0)

    # Test knot access
    knot_val = surf.knot(0, 2)

    # Test set knot
    surf.set_knot(0, 2, 5.0)
    new_val = surf.knot(0, 2)

    MINI_CHECK(surf.dimension() == 3)
    MINI_CHECK(not surf.is_rational())
    MINI_CHECK(surf.order(0) == 4)
    MINI_CHECK(surf.order(1) == 3)
    MINI_CHECK(surf.degree(0) == 3)
    MINI_CHECK(surf.degree(1) == 2)
    MINI_CHECK(surf.cv_count_dir(0) == 5)
    MINI_CHECK(surf.cv_count_dir(1) == 4)
    MINI_CHECK(surf.cv_count_dir(None) == 20)
    MINI_CHECK(surf.cv_size() == 3)
    MINI_CHECK(surf.knot_count(0) == 7)
    MINI_CHECK(surf.knot_count(1) == 5)
    MINI_CHECK(surf.span_count(0) == 2)
    MINI_CHECK(surf.span_count(1) == 2)
    MINI_CHECK(new_val == 5.0)


@MINI_TEST("NurbsSurface", "knot_operations")
def test_knot_operations():
    from session_py import NurbsSurface

    surf = NurbsSurface.create_raw(3, False, 4, 4, 4, 4, False, False, 1.0, 1.0)

    # Verify domain
    u0, u1 = surf.domain(0)
    v0, v1 = surf.domain(1)

    MINI_CHECK(u0 == 0.0)
    MINI_CHECK(u1 > u0)
    MINI_CHECK(v0 == 0.0)
    MINI_CHECK(v1 > v0)
    MINI_CHECK(surf.is_clamped(0, 0))
    MINI_CHECK(surf.is_clamped(1, 0))


@MINI_TEST("NurbsSurface", "rational_operations")
def test_rational_operations():
    from session_py import NurbsSurface
    from session_py import Point

    # Create non-rational surface
    surf = NurbsSurface.create_raw(3, False, 3, 3, 3, 3, False, False, 1.0, 1.0)

    # Make it rational
    surf.make_rational()

    # Set a control point and weight
    surf.set_cv(1, 1, Point(1.0, 2.0, 3.0))
    surf.set_weight(1, 1, 2.0)

    # Verify weight
    w = surf.weight(1, 1)

    MINI_CHECK(surf.is_rational())
    MINI_CHECK(surf.cv_size() == 4)
    MINI_CHECK(w == 2.0)


@MINI_TEST("NurbsSurface", "evaluation")
def test_evaluation():
    from session_py import NurbsSurface
    from session_py import Point

    # Create simple bilinear surface
    surf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)

    # Set corner control points
    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    surf.set_cv(1, 1, Point(1.0, 1.0, 0.0))

    # Evaluate at corner
    u0, u1 = surf.domain(0)
    v0, v1 = surf.domain(1)

    pt_corner = surf.point_at(u0, v0)

    # Evaluate at center (should be center of unit square)
    u_mid = (u0 + u1) / 2.0
    v_mid = (v0 + v1) / 2.0
    pt_mid = surf.point_at(u_mid, v_mid)

    # Test derivatives
    derivs = surf.evaluate(u_mid, v_mid, 1)

    # Test normal (for flat plane in XY, normal should point in +Z)
    normal = surf.normal_at(u_mid, v_mid)

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(TOLERANCE.is_close(pt_corner[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt_corner[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt_corner[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt_mid[0], 0.5))
    MINI_CHECK(TOLERANCE.is_close(pt_mid[1], 0.5))
    MINI_CHECK(len(derivs) == 3)
    MINI_CHECK(TOLERANCE.is_close(abs(normal[2]), 1.0))


@MINI_TEST("NurbsSurface", "geometric_queries")
def test_geometric_queries():
    from session_py import NurbsSurface
    from session_py import Point

    # Create and setup surface
    surf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)

    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    surf.set_cv(1, 1, Point(1.0, 1.0, 0.0))

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(not surf.is_closed(0))
    MINI_CHECK(not surf.is_closed(1))
    MINI_CHECK(not surf.is_periodic(0))
    MINI_CHECK(not surf.is_periodic(1))
    MINI_CHECK(surf.is_clamped(0, 0))
    MINI_CHECK(surf.is_clamped(1, 0))
    MINI_CHECK(surf.is_planar(None, 1e-6))


@MINI_TEST("NurbsSurface", "modification")
def test_modification():
    from session_py import NurbsSurface
    from session_py import Point

    surf = NurbsSurface.create_raw(3, False, 2, 2, 3, 2, False, False, 1.0, 1.0)

    # Set some CVs
    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(2, 0, Point(2.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    surf.set_cv(2, 1, Point(2.0, 1.0, 0.0))

    cv_before = surf.get_cv(0, 0)

    # Test reverse in u direction
    surf.reverse(0)
    cv_after = surf.get_cv(2, 0)

    # Reverse back
    surf.reverse(0)

    # Test transpose
    order_u_before = surf.order(0)
    order_v_before = surf.order(1)
    surf.transpose()

    MINI_CHECK(cv_after[0] == cv_before[0])
    MINI_CHECK(surf.order(0) == order_v_before)
    MINI_CHECK(surf.order(1) == order_u_before)


@MINI_TEST("NurbsSurface", "isocurve")
def test_isocurve():
    from session_py import NurbsSurface
    from session_py import Point

    # Create surface
    surf = NurbsSurface.create_raw(3, False, 3, 3, 3, 3, False, False, 1.0, 1.0)

    # Set up a grid of control points
    for i in range(3):
        for j in range(3):
            surf.set_cv(i, j, Point(float(i), float(j), 0.0))

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


@MINI_TEST("NurbsSurface", "transformation")
def test_transformation():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Xform

    # Create simple surface
    surf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)

    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    surf.set_cv(1, 1, Point(1.0, 1.0, 0.0))

    # Apply translation
    xf = Xform.translation(1.0, 2.0, 3.0)
    surf.transform(xf)

    # Check transformed CV
    pt = surf.get_cv(0, 0)

    MINI_CHECK(TOLERANCE.is_close(pt[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 3.0))


@MINI_TEST("NurbsSurface", "json_roundtrip")
def test_json_roundtrip():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Color
    from pathlib import Path

    surf = NurbsSurface.create_raw(3, False, 3, 3, 3, 3, False, False, 1.0, 1.0)
    surf.name = "test_nurbssurface"
    surf.width = 2.0
    surf.surfacecolor = Color(255, 128, 64, 255)

    for i in range(3):
        for j in range(3):
            surf.set_cv(i, j, Point(float(i), float(j), 0.0))

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   json_dumps()    │ str          │ to JSON string
    #   json_loads(s)   │ str          │ from JSON string
    #   json_dump(path) │ file         │ write to file
    #   json_load(path) │ file         │ read from file

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


@MINI_TEST("NurbsSurface", "protobuf_roundtrip")
def test_protobuf_roundtrip():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Color
    from pathlib import Path

    surf = NurbsSurface.create_raw(3, False, 3, 3, 3, 3, False, False, 1.0, 1.0)
    surf.name = "test_nurbssurface"
    surf.width = 2.0
    surf.surfacecolor = Color(255, 128, 64, 255)

    for i in range(3):
        for j in range(3):
            surf.set_cv(i, j, Point(float(i), float(j), 0.0))

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


@MINI_TEST("NurbsSurface", "advanced_accessors")
def test_advanced_accessors():
    from session_py import NurbsSurface
    from session_py import Point

    # Create rational surface for testing get_cv_4d/set_cv_4d
    surf = NurbsSurface.create_raw(3, True, 3, 3, 3, 3, False, False, 1.0, 1.0)

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


@MINI_TEST("NurbsSurface", "clamp_operations")
def test_clamp_operations():
    from session_py import NurbsSurface
    from session_py import Point

    surf = NurbsSurface.create_raw(3, False, 4, 4, 4, 4, False, False, 1.0, 1.0)

    # Set up control points
    for i in range(4):
        for j in range(4):
            surf.set_cv(i, j, Point(float(i), float(j), 0.0))

    # Test clamp_end
    was_clamped_before = surf.is_clamped(0, 2)
    surf.clamp_end(0, 2)
    is_clamped_after = surf.is_clamped(0, 2)

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(is_clamped_after)


@MINI_TEST("NurbsSurface", "singularity")
def test_singularity():
    from session_py import NurbsSurface
    from session_py import Point

    # Create a simple surface
    surf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)

    # Set all CVs to different points (non-singular)
    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    surf.set_cv(1, 1, Point(1.0, 1.0, 0.0))

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


@MINI_TEST("NurbsSurface", "bounding_box")
def test_bounding_box():
    from session_py import NurbsSurface
    from session_py import Point

    surf = NurbsSurface.create_raw(3, False, 2, 2, 3, 3, False, False, 1.0, 1.0)

    # Set CVs in a known range
    for i in range(3):
        for j in range(3):
            surf.set_cv(i, j, Point(float(i), float(j), 0.0))

    # Get bounding box
    bbox = surf.get_bounding_box()

    MINI_CHECK(surf.is_valid())


@MINI_TEST("NurbsSurface", "domain_operations")
def test_domain_operations():
    from session_py import NurbsSurface

    surf = NurbsSurface.create_raw(3, False, 3, 3, 3, 3, False, False, 1.0, 1.0)

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


@MINI_TEST("NurbsSurface", "corner_points")
def test_corner_points():
    from session_py import NurbsSurface
    from session_py import Point

    surf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)

    # Set corner control points
    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(1, 0, Point(10.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 10.0, 0.0))
    surf.set_cv(1, 1, Point(10.0, 10.0, 0.0))

    # Get corner points
    p00 = surf.point_at_corner(0, 0)
    p10 = surf.point_at_corner(1, 0)
    p01 = surf.point_at_corner(0, 1)
    p11 = surf.point_at_corner(1, 1)

    MINI_CHECK(TOLERANCE.is_close(p00[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(p10[0], 10.0))
    MINI_CHECK(TOLERANCE.is_close(p01[1], 10.0))
    MINI_CHECK(TOLERANCE.is_close(p11[0], 10.0) and TOLERANCE.is_close(p11[1], 10.0))


@MINI_TEST("NurbsSurface", "swap_coordinates")
def test_swap_coordinates():
    from session_py import NurbsSurface
    from session_py import Point

    surf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)

    # Set a control point with distinct coordinates
    surf.set_cv(0, 0, Point(1.0, 2.0, 3.0))

    # Swap X and Y
    surf.swap_coordinates(0, 1)

    pt = surf.get_cv(0, 0)

    MINI_CHECK(TOLERANCE.is_close(pt[0], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 3.0))


@MINI_TEST("NurbsSurface", "zero_cvs")
def test_zero_cvs():
    from session_py import NurbsSurface
    from session_py import Point

    surf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)

    # Set non-zero control points
    surf.set_cv(0, 0, Point(1.0, 2.0, 3.0))
    surf.set_cv(1, 1, Point(4.0, 5.0, 6.0))

    # Zero all CVs
    surf.zero_cvs()

    pt0 = surf.get_cv(0, 0)
    pt1 = surf.get_cv(1, 1)

    MINI_CHECK(TOLERANCE.is_close(pt0[0], 0.0) and
               TOLERANCE.is_close(pt0[1], 0.0) and
               TOLERANCE.is_close(pt0[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt1[0], 0.0) and
               TOLERANCE.is_close(pt1[1], 0.0) and
               TOLERANCE.is_close(pt1[2], 0.0))


@MINI_TEST("NurbsSurface", "get_knots")
def test_get_knots():
    from session_py import NurbsSurface

    surf = NurbsSurface.create_raw(3, False, 4, 3, 4, 3, False, False, 1.0, 2.0)

    knots_u = surf.get_knots(0)
    knots_v = surf.get_knots(1)

    MINI_CHECK(len(knots_u) == surf.knot_count(0))
    MINI_CHECK(len(knots_v) == surf.knot_count(1))
    MINI_CHECK(len(knots_u) > 0)
    MINI_CHECK(len(knots_v) > 0)


@MINI_TEST("NurbsSurface", "make_non_rational")
def test_make_non_rational():
    from session_py import NurbsSurface
    from session_py import Point

    # Create rational surface with all weights = 1
    surf = NurbsSurface.create_raw(3, True, 3, 3, 3, 3, False, False, 1.0, 1.0)

    # Set all weights to 1.0
    for i in range(3):
        for j in range(3):
            surf.set_cv(i, j, Point(float(i), float(j), 0.0))
            surf.set_weight(i, j, 1.0)

    was_rational = surf.is_rational()
    surf.make_non_rational()
    is_rational_after = surf.is_rational()

    MINI_CHECK(was_rational)
    MINI_CHECK(not is_rational_after)


@MINI_TEST("NurbsSurface", "create_clamped_uniform")
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


@MINI_TEST("NurbsSurface", "knot_multiplicity")
def test_knot_multiplicity():
    from session_py import NurbsSurface

    surf = NurbsSurface.create_raw(3, False, 4, 4, 4, 4, False, False, 1.0, 1.0)

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


@MINI_TEST("NurbsSurface", "sphere")
def test_sphere():
    import math
    from session_py import NurbsSurface
    from session_py import Point

    radius = 2.0
    w = math.sqrt(2.0) / 2.0  # 0.707107
    pi = PI

    surf = NurbsSurface.create_raw(3, True, 3, 3, 9, 5, False, False, 1.0, 1.0)
    surf.name = "unit_sphere"

    # U-knots: periodic around equator with multiplicity 2
    u_knots = [0, 0, pi * 0.5, pi * 0.5, pi, pi, pi * 1.5, pi * 1.5, pi * 2.0, pi * 2.0]
    for i in range(10):
        surf.set_knot(0, i, u_knots[i])

    # V-knots: from south pole to north pole
    v_knots = [-pi * 0.5, -pi * 0.5, 0, 0, pi * 0.5, pi * 0.5]
    for i in range(6):
        surf.set_knot(1, i, v_knots[i])

    # Set up control points for sphere (9 around, 5 latitude levels)
    # Latitude levels: south pole, -45deg, equator, +45deg, north pole
    lat_weights = [w, 0.5, w, 0.5, w]
    lat_z = [-radius, -radius * w, 0.0, radius * w, radius]
    lat_r = [0.0, radius * w, radius, radius * w, 0.0]

    for j in range(5):
        r = lat_r[j]
        z = lat_z[j]
        # 9 points around (0, 45, 90, 135, 180, 225, 270, 315, 360=0)
        angles = [0, pi * 0.25, pi * 0.5, pi * 0.75, pi, pi * 1.25, pi * 1.5, pi * 1.75, pi * 2.0]
        for i in range(9):
            x = r * math.cos(angles[i])
            y = r * math.sin(angles[i])
            surf.set_cv(i, j, Point(x, y, z))
            # Weight: w for cardinal directions (0, 90, 180, 270), 0.5 for diagonals at non-pole latitudes
            weight = w if i % 2 == 0 else lat_weights[j]
            if j == 0 or j == 4:
                weight = w  # poles
            surf.set_weight(i, j, weight)

    # Verify sphere properties
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(surf.is_rational())
    MINI_CHECK(surf.degree(0) == 2)
    MINI_CHECK(surf.degree(1) == 2)
    MINI_CHECK(surf.cv_count_dir(0) == 9)
    MINI_CHECK(surf.cv_count_dir(1) == 5)

    # Check point on equator at angle 0 (should be at (radius, 0, 0))
    pt = surf.point_at(0.0, 0.0)
    MINI_CHECK(TOLERANCE.is_close(pt[0], radius))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 0.0))

    # Check north pole
    north = surf.point_at(0.0, pi * 0.5)
    MINI_CHECK(TOLERANCE.is_close(north[2], radius))


@MINI_TEST("NurbsSurface", "cylinder")
def test_cylinder():
    import math
    from session_py import NurbsSurface
    from session_py import Point

    radius = 1.5
    height = 3.0
    w = math.sqrt(2.0) / 2.0
    pi = PI

    surf = NurbsSurface.create_raw(3, True, 3, 2, 9, 2, False, False, 1.0, 1.0)
    surf.name = "unit_cylinder"

    # U-knots: periodic for circle
    u_knots = [0, 0, pi * 0.5, pi * 0.5, pi, pi, pi * 1.5, pi * 1.5, pi * 2.0, pi * 2.0]
    for i in range(10):
        surf.set_knot(0, i, u_knots[i])

    # V-knots: linear from bottom to top
    surf.set_knot(1, 0, 0.0)
    surf.set_knot(1, 1, height)

    # Set up control points for cylinder (9 around, 2 heights)
    angles = [0, pi * 0.25, pi * 0.5, pi * 0.75, pi, pi * 1.25, pi * 1.5, pi * 1.75, pi * 2.0]

    for j in range(2):
        z = 0.0 if j == 0 else height
        for i in range(9):
            x = radius * math.cos(angles[i])
            y = radius * math.sin(angles[i])
            # For diagonal points (45, 135, etc), radius needs to be scaled
            if i % 2 == 1:
                x = radius * math.sqrt(2.0) * math.cos(angles[i])
                y = radius * math.sqrt(2.0) * math.sin(angles[i])
            surf.set_cv(i, j, Point(x, y, z))
            weight = 1.0 if i % 2 == 0 else w
            surf.set_weight(i, j, weight)

    # Verify cylinder properties
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(surf.is_rational())
    MINI_CHECK(surf.degree(0) == 2)
    MINI_CHECK(surf.degree(1) == 1)
    MINI_CHECK(surf.cv_count_dir(0) == 9)
    MINI_CHECK(surf.cv_count_dir(1) == 2)

    # Check point on bottom circle at angle 0
    pt_bottom = surf.point_at(0.0, 0.0)
    MINI_CHECK(TOLERANCE.is_close(pt_bottom[0], radius))
    MINI_CHECK(TOLERANCE.is_close(pt_bottom[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt_bottom[2], 0.0))

    # Check point on top circle at angle 0
    pt_top = surf.point_at(0.0, height)
    MINI_CHECK(TOLERANCE.is_close(pt_top[0], radius))
    MINI_CHECK(TOLERANCE.is_close(pt_top[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt_top[2], height))

    # Check midpoint at angle PI/2
    pt_mid = surf.point_at(pi * 0.5, height * 0.5)
    MINI_CHECK(TOLERANCE.is_close(pt_mid[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt_mid[1], radius))
    MINI_CHECK(TOLERANCE.is_close(pt_mid[2], height * 0.5))


@MINI_TEST("NurbsSurface", "torus")
def test_torus():
    import math
    from session_py import NurbsSurface
    from session_py import Point

    major_radius = 3.0
    minor_radius = 1.0
    w = math.sqrt(2.0) / 2.0
    pi = PI

    surf = NurbsSurface.create_raw(3, True, 3, 3, 9, 9, False, False, 1.0, 1.0)
    surf.name = "unit_torus"

    # Both U and V knots: periodic for circles
    knots = [0, 0, pi * 0.5, pi * 0.5, pi, pi, pi * 1.5, pi * 1.5, pi * 2.0, pi * 2.0]
    for i in range(10):
        surf.set_knot(0, i, knots[i])
        surf.set_knot(1, i, knots[i])

    # Set up control points for torus
    # U: major angle (around torus), V: minor angle (around tube)
    angles = [0, pi * 0.25, pi * 0.5, pi * 0.75, pi, pi * 1.25, pi * 1.5, pi * 1.75, pi * 2.0]

    for i in range(9):
        major_angle = angles[i]
        cos_ma = math.cos(major_angle)
        sin_ma = math.sin(major_angle)
        major_scale = 1.0 if i % 2 == 0 else math.sqrt(2.0)

        for j in range(9):
            minor_angle = angles[j]
            cos_mi = math.cos(minor_angle)
            sin_mi = math.sin(minor_angle)
            minor_scale = 1.0 if j % 2 == 0 else math.sqrt(2.0)

            r = major_radius + minor_radius * minor_scale * cos_mi
            x = r * major_scale * cos_ma
            y = r * major_scale * sin_ma
            z = minor_radius * minor_scale * sin_mi

            surf.set_cv(i, j, Point(x, y, z))

            # Weight is product of major and minor weights
            w_major = 1.0 if i % 2 == 0 else w
            w_minor = 1.0 if j % 2 == 0 else w
            surf.set_weight(i, j, w_major * w_minor)

    # Verify torus properties
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(surf.is_rational())
    MINI_CHECK(surf.degree(0) == 2)
    MINI_CHECK(surf.degree(1) == 2)
    MINI_CHECK(surf.cv_count_dir(0) == 9)
    MINI_CHECK(surf.cv_count_dir(1) == 9)

    # Check point at (0, 0) - should be on outer edge of torus
    pt = surf.point_at(0.0, 0.0)
    MINI_CHECK(TOLERANCE.is_close(pt[0], major_radius + minor_radius))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 0.0))

    # Check point at (PI, 0) - should be on opposite side of torus
    pt_opp = surf.point_at(pi, 0.0)
    MINI_CHECK(TOLERANCE.is_close(pt_opp[0], -(major_radius + minor_radius)))
    MINI_CHECK(TOLERANCE.is_close(pt_opp[1], 0.0))


@MINI_TEST("NurbsSurface", "cone")
def test_cone():
    import math
    from session_py import NurbsSurface
    from session_py import Point

    radius = 2.0
    height = 4.0
    w = math.sqrt(2.0) / 2.0
    pi = PI

    surf = NurbsSurface.create_raw(3, True, 3, 2, 9, 2, False, False, 1.0, 1.0)
    surf.name = "unit_cone"

    # U-knots: periodic for circle
    u_knots = [0, 0, pi * 0.5, pi * 0.5, pi, pi, pi * 1.5, pi * 1.5, pi * 2.0, pi * 2.0]
    for i in range(10):
        surf.set_knot(0, i, u_knots[i])

    # V-knots: linear from apex to base
    surf.set_knot(1, 0, 0.0)
    surf.set_knot(1, 1, height)

    # Set up control points for cone (9 around, 2 heights: apex and base)
    angles = [0, pi * 0.25, pi * 0.5, pi * 0.75, pi, pi * 1.25, pi * 1.5, pi * 1.75, pi * 2.0]

    # Apex (j=0) - all points collapse to the apex
    for i in range(9):
        surf.set_cv(i, 0, Point(0.0, 0.0, height))
        weight = 1.0 if i % 2 == 0 else w
        surf.set_weight(i, 0, weight)

    # Base (j=1) - circle at z=0
    for i in range(9):
        x = radius * math.cos(angles[i])
        y = radius * math.sin(angles[i])
        if i % 2 == 1:
            x = radius * math.sqrt(2.0) * math.cos(angles[i])
            y = radius * math.sqrt(2.0) * math.sin(angles[i])
        surf.set_cv(i, 1, Point(x, y, 0.0))
        weight = 1.0 if i % 2 == 0 else w
        surf.set_weight(i, 1, weight)

    # Verify cone properties
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(surf.is_rational())
    MINI_CHECK(surf.degree(0) == 2)
    MINI_CHECK(surf.degree(1) == 1)
    MINI_CHECK(surf.cv_count_dir(0) == 9)
    MINI_CHECK(surf.cv_count_dir(1) == 2)

    # Check apex
    apex = surf.point_at(0.0, 0.0)
    MINI_CHECK(TOLERANCE.is_close(apex[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(apex[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(apex[2], height))

    # Check point on base circle at angle 0
    base = surf.point_at(0.0, height)
    MINI_CHECK(TOLERANCE.is_close(base[0], radius))
    MINI_CHECK(TOLERANCE.is_close(base[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(base[2], 0.0))

    # Check midpoint - should be at half radius, half height
    mid = surf.point_at(0.0, height * 0.5)
    MINI_CHECK(TOLERANCE.is_close(mid[0], radius * 0.5))
    MINI_CHECK(TOLERANCE.is_close(mid[2], height * 0.5))


@MINI_TEST("NurbsSurface", "create_planar")
def test_nurbssurface_create_planar():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point

    pts = [Point(0,0,0), Point(3,1,0), Point(5,0.5,0), Point(6,3,0), Point(4,5,0), Point(1,4,0), Point(0,0,0)]
    boundary = NurbsCurve.create(False, 3, pts)
    surf = NurbsSurface.create_planar([boundary])

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(surf.is_planar())
    MINI_CHECK(surf.degree(0) == 1)
    MINI_CHECK(surf.degree(1) == 1)
    MINI_CHECK(surf.cv_count_dir(0) == 2)
    MINI_CHECK(surf.cv_count_dir(1) == 2)
    MINI_CHECK(surf.is_trimmed())
    MINI_CHECK(surf.get_outer_loop().is_valid())

    pd, puv = surf.divide_by_count(4, 4)
    MINI_CHECK(len(pd) == 5)
    MINI_CHECK(len(pd[0]) == 5)


@MINI_TEST("NurbsSurface", "constructor_ruled_surface")
def test_nurbssurface_constructor_ruled_surface():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Vector

    pts_a = [Point(3,0,0), Point(-2,0,5)]
    pts_b = [Point(3,5,5), Point(-2,5,0)]
    crvA = NurbsCurve.create(False, 1, pts_a)
    crvB = NurbsCurve.create(False, 1, pts_b)
    ruled = NurbsSurface.create_ruled(crvA, crvB)

    MINI_CHECK(ruled.is_valid())
    MINI_CHECK(ruled.degree(0) == 1)
    MINI_CHECK(ruled.degree(1) == 1)
    MINI_CHECK(ruled.cv_count_dir(0) == 2)
    MINI_CHECK(ruled.cv_count_dir(1) == 2)

    rd, ruv = ruled.divide_by_count(4, 4)
    MINI_CHECK(len(rd) == 5)
    MINI_CHECK(len(rd[0]) == 5)

    pts = []
    for i in range(len(rd)):
        for j in range(len(rd[i])):
            pts.append(rd[i][j])

    normals = []
    for i in range(len(ruv)):
        for j in range(len(ruv[i])):
            normals.append(ruled.normal_at(ruv[i][j][0], ruv[i][j][1]))

    uvs = []
    for i in range(len(ruv)):
        for j in range(len(ruv[i])):
            uvs.append(ruv[i][j])

    MINI_CHECK(TOLERANCE.is_point_close(pts[0],  Point( 3.00, 0.00, 0.00)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[1],  Point( 3.00, 1.25, 1.25)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[2],  Point( 3.00, 2.50, 2.50)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[3],  Point( 3.00, 3.75, 3.75)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[4],  Point( 3.00, 5.00, 5.00)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[5],  Point( 1.75, 0.00, 1.25)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[6],  Point( 1.75, 1.25, 1.875)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[7],  Point( 1.75, 2.50, 2.50)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[8],  Point( 1.75, 3.75, 3.125)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[9],  Point( 1.75, 5.00, 3.75)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[10], Point( 0.50, 0.00, 2.50)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[11], Point( 0.50, 1.25, 2.50)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[12], Point( 0.50, 2.50, 2.50)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[13], Point( 0.50, 3.75, 2.50)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[14], Point( 0.50, 5.00, 2.50)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[15], Point(-0.75, 0.00, 3.75)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[16], Point(-0.75, 1.25, 3.125)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[17], Point(-0.75, 2.50, 2.50)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[18], Point(-0.75, 3.75, 1.875)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[19], Point(-0.75, 5.00, 1.25)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[20], Point(-2.00, 0.00, 5.00)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[21], Point(-2.00, 1.25, 3.75)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[22], Point(-2.00, 2.50, 2.50)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[23], Point(-2.00, 3.75, 1.25)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[24], Point(-2.00, 5.00, 0.00)))

    MINI_CHECK(TOLERANCE.is_vector_close(normals[0],  Vector( 0.577350269189626, -0.577350269189626,  0.577350269189626)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[1],  Vector( 1.0/3.0, -2.0/3.0, 2.0/3.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[2],  Vector( 0.0, -0.707106781186547,  0.707106781186547)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[3],  Vector(-1.0/3.0, -2.0/3.0, 2.0/3.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[4],  Vector(-0.577350269189626, -0.577350269189626,  0.577350269189626)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[5],  Vector( 2.0/3.0, -1.0/3.0, 2.0/3.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[6],  Vector( 0.408248290463863, -0.408248290463863,  0.816496580927726)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[7],  Vector( 0.0, -0.447213595499958,  0.894427190999916)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[8],  Vector(-0.408248290463863, -0.408248290463863,  0.816496580927726)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[9],  Vector(-2.0/3.0, -1.0/3.0, 2.0/3.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[10], Vector( 0.707106781186547,  0.0,  0.707106781186547)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[11], Vector( 0.447213595499958,  0.0,  0.894427190999916)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[12], Vector( 0.0, 0.0, 1.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[13], Vector(-0.447213595499958,  0.0,  0.894427190999916)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[14], Vector(-0.707106781186547,  0.0,  0.707106781186547)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[15], Vector( 2.0/3.0, 1.0/3.0, 2.0/3.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[16], Vector( 0.408248290463863,  0.408248290463863,  0.816496580927726)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[17], Vector( 0.0, 0.447213595499958,  0.894427190999916)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[18], Vector(-0.408248290463863,  0.408248290463863,  0.816496580927726)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[19], Vector(-2.0/3.0, 1.0/3.0, 2.0/3.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[20], Vector( 0.577350269189626,  0.577350269189626,  0.577350269189626)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[21], Vector( 1.0/3.0, 2.0/3.0, 2.0/3.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[22], Vector( 0.0, 0.707106781186547,  0.707106781186547)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[23], Vector(-1.0/3.0, 2.0/3.0, 2.0/3.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(normals[24], Vector(-0.577350269189626,  0.577350269189626,  0.577350269189626)))

    MINI_CHECK(TOLERANCE.is_close(uvs[0][0],  0.00) and TOLERANCE.is_close(uvs[0][1],  0.00))
    MINI_CHECK(TOLERANCE.is_close(uvs[1][0],  0.00) and TOLERANCE.is_close(uvs[1][1],  0.25))
    MINI_CHECK(TOLERANCE.is_close(uvs[4][0],  0.00) and TOLERANCE.is_close(uvs[4][1],  1.00))
    MINI_CHECK(TOLERANCE.is_close(uvs[6][0],  0.25) and TOLERANCE.is_close(uvs[6][1],  0.25))
    MINI_CHECK(TOLERANCE.is_close(uvs[12][0], 0.50) and TOLERANCE.is_close(uvs[12][1], 0.50))
    MINI_CHECK(TOLERANCE.is_close(uvs[24][0], 1.00) and TOLERANCE.is_close(uvs[24][1], 1.00))

if __name__ == "__main__":
    run_all(language="python")
