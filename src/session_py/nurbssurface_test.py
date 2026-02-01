from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("NurbsSurface", "constructor")
def test_nurbssurface_constructor():
    from session_py import NurbsSurface
    from session_py import Color
    from session_py import Point

    s = NurbsSurface.create(3, False, 3, 3, 4, 4, knot_delta_u=2.5, knot_delta_v=2.5)
    cvs = [
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

    # Setters
    idx = 0
    for i in range(s.cv_count_dir(0)):
        for j in range(s.cv_count_dir(1)):
            s.set_cv(i, j, cvs[idx])
            idx += 1

    # Getters
    control_point = s.get_cv(2, 1) # 3.75, 1.25, 4.0
    point = s.point_at(2.5, 2.5) # 2.5, 2.5, 4.0

    # String representation
    str_repr = str(s)

    # Duplicate for comparison
    s_copy = s.duplicate()

    # Quad faces as coordinates (surface subdivision)
    divisions_u = 5
    divisions_v = 5
    v, _ = s.subdivide(divisions_u, divisions_v)

    MINI_CHECK(s.name == "my_nurbssurface")
    MINI_CHECK(s.width == 1.0)
    MINI_CHECK(s.surfacecolor == Color.black())
    MINI_CHECK(s.guid)
    MINI_CHECK(s.m_dim == 3)
    MINI_CHECK(not s.m_is_rat)

    MINI_CHECK(s.dimension() == 3)
    MINI_CHECK(not s.is_rational())
    MINI_CHECK(s.order(0) == 3)
    MINI_CHECK(s.order(1) == 3)
    MINI_CHECK(s.degree(0) == 2)
    MINI_CHECK(s.degree(1) == 2)
    MINI_CHECK(s.cv_count_dir(0) == 4)
    MINI_CHECK(s.cv_count_dir(1) == 4)
    MINI_CHECK(s.cv_count_dir(None) == 16)
    MINI_CHECK(s.knot_count(0) == 5)
    MINI_CHECK(s.knot_count(1) == 5)

    MINI_CHECK(control_point[0] == 3.75 and control_point[1] == 1.25 and control_point[2] == 4.0)
    MINI_CHECK(point[0] == 2.5 and point[1] == 2.5 and point[2] == 4.0)

    MINI_CHECK(str_repr == "NurbsSurface(dim=3, order=(3,3), cv_count=(4,4))")

    MINI_CHECK(s_copy == s)
    MINI_CHECK(s_copy.name == s.name)
    MINI_CHECK(s_copy.width == s.width)
    MINI_CHECK(s_copy.surfacecolor == s.surfacecolor)
    MINI_CHECK(s_copy.guid != s.guid)

    # Helper function for tolerance-based point comparison
    def close_pt(a, x, y, z):
        return TOLERANCE.is_close(a[0], x) and TOLERANCE.is_close(a[1], y) and TOLERANCE.is_close(a[2], z)
    MINI_CHECK(close_pt(v[0], 0.0, 0.0, 0.0))
    MINI_CHECK(close_pt(v[1], -0.64, 0.76, 1.28))
    MINI_CHECK(close_pt(v[2], -0.96, 1.84, 1.92))
    MINI_CHECK(close_pt(v[3], -0.96, 3.16, 1.92))
    MINI_CHECK(close_pt(v[4], -0.64, 4.24, 1.28))
    MINI_CHECK(close_pt(v[5], 0.0, 5.0, 0.0))

    MINI_CHECK(close_pt(v[6], 0.76, -0.64, 1.28))
    MINI_CHECK(close_pt(v[7], 0.6832, 0.6832, 2.56))
    MINI_CHECK(close_pt(v[8], 0.6448, 1.9168, 3.2))
    MINI_CHECK(close_pt(v[9], 0.6448, 3.0832, 3.2))
    MINI_CHECK(close_pt(v[10], 0.6832, 4.3168, 2.56))
    MINI_CHECK(close_pt(v[11], 0.76, 5.64, 1.28))

    MINI_CHECK(close_pt(v[12], 1.84, -0.96, 1.92))
    MINI_CHECK(close_pt(v[13], 1.9168, 0.6448, 3.2))
    MINI_CHECK(close_pt(v[14], 1.9552, 1.9552, 3.84))
    MINI_CHECK(close_pt(v[15], 1.9552, 3.0448, 3.84))
    MINI_CHECK(close_pt(v[16], 1.9168, 4.3552, 3.2))
    MINI_CHECK(close_pt(v[17], 1.84, 5.96, 1.92))

    MINI_CHECK(close_pt(v[18], 3.16, -0.96, 1.92))
    MINI_CHECK(close_pt(v[19], 3.0832, 0.6448, 3.2))
    MINI_CHECK(close_pt(v[20], 3.0448, 1.9552, 3.84))
    MINI_CHECK(close_pt(v[21], 3.0448, 3.0448, 3.84))
    MINI_CHECK(close_pt(v[22], 3.0832, 4.3552, 3.2))
    MINI_CHECK(close_pt(v[23], 3.16, 5.96, 1.92))

    MINI_CHECK(close_pt(v[24], 4.24, -0.64, 1.28))
    MINI_CHECK(close_pt(v[25], 4.3168, 0.6832, 2.56))
    MINI_CHECK(close_pt(v[26], 4.3552, 1.9168, 3.2))
    MINI_CHECK(close_pt(v[27], 4.3552, 3.0832, 3.2))
    MINI_CHECK(close_pt(v[28], 4.3168, 4.3168, 2.56))
    MINI_CHECK(close_pt(v[29], 4.24, 5.64, 1.28))

    MINI_CHECK(close_pt(v[30], 5.0, 0.0, 0.0))
    MINI_CHECK(close_pt(v[31], 5.64, 0.76, 1.28))
    MINI_CHECK(close_pt(v[32], 5.96, 1.84, 1.92))
    MINI_CHECK(close_pt(v[33], 5.96, 3.16, 1.92))
    MINI_CHECK(close_pt(v[34], 5.64, 4.24, 1.28))
    MINI_CHECK(close_pt(v[35], 5.0, 5.0, 0.0))





@MINI_TEST("NurbsSurface", "create_operations")
def test_create_operations():
    from session_py import NurbsSurface
    from session_py import Point

    # Create a simple 2x2 bilinear surface
    surf = NurbsSurface.create(3, False, 2, 2, 2, 2)

    # Set up clamped uniform knot vectors
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

    # Set corner control points
    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    surf.set_cv(1, 1, Point(1.0, 1.0, 0.0))

    # Check knot vectors
    u0, u1 = surf.domain(0)
    v0, v1 = surf.domain(1)

    MINI_CHECK(surf is not None)
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(u0 == 0.0)
    MINI_CHECK(v0 == 0.0)


@MINI_TEST("NurbsSurface", "accessors")
def test_accessors():
    from session_py import NurbsSurface
    from session_py import Point

    surf = NurbsSurface.create(3, False, 4, 3, 5, 4)

    # Test knot access
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    knot_val = surf.knot(0, 2)

    # Test set knot
    surf.set_knot(0, 2, 5.0)
    new_val = surf.knot(0, 2)

    MINI_CHECK(surf is not None)
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

    surf = NurbsSurface.create(3, False, 4, 4, 4, 4)

    # Make clamped uniform knot vector
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

    # Verify domain
    u0, u1 = surf.domain(0)
    v0, v1 = surf.domain(1)

    MINI_CHECK(surf is not None)
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
    surf = NurbsSurface.create(3, False, 3, 3, 3, 3)

    # Make it rational
    surf.make_rational()

    # Set a control point and weight
    surf.set_cv(1, 1, Point(1.0, 2.0, 3.0))
    surf.set_weight(1, 1, 2.0)

    # Verify weight
    w = surf.weight(1, 1)

    # Get CV
    pt = surf.get_cv(1, 1)

    MINI_CHECK(surf is not None)
    MINI_CHECK(surf.is_rational())
    MINI_CHECK(surf.cv_size() == 4)
    MINI_CHECK(w == 2.0)
    MINI_CHECK(pt is not None)


@MINI_TEST("NurbsSurface", "evaluation")
def test_evaluation():
    from session_py import NurbsSurface
    from session_py import Point

    # Create simple bilinear surface (2x2 control points, order 2x2)
    surf = NurbsSurface.create(3, False, 2, 2, 2, 2)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

    # Set corner control points to unit square in XY plane
    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    surf.set_cv(1, 1, Point(1.0, 1.0, 0.0))

    # Evaluate at domain bounds
    u0, u1 = surf.domain(0)
    v0, v1 = surf.domain(1)

    pt_corner = surf.point_at(u0, v0)
    pt_mid = surf.point_at((u0 + u1) / 2.0, (v0 + v1) / 2.0)
    derivs = surf.evaluate((u0 + u1) / 2.0, (v0 + v1) / 2.0, 1)
    normal = surf.normal_at((u0 + u1) / 2.0, (v0 + v1) / 2.0)

    MINI_CHECK(surf is not None)
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(pt_corner is not None)
    MINI_CHECK(TOLERANCE.is_close(pt_corner[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt_corner[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt_corner[2], 0.0))
    MINI_CHECK(pt_mid is not None)
    MINI_CHECK(TOLERANCE.is_close(pt_mid[0], 0.5))
    MINI_CHECK(TOLERANCE.is_close(pt_mid[1], 0.5))
    MINI_CHECK(len(derivs) == 3)
    MINI_CHECK(TOLERANCE.is_close(abs(normal[2]), 1.0))


@MINI_TEST("NurbsSurface", "geometric_queries")
def test_geometric_queries():
    from session_py import NurbsSurface
    from session_py import Point

    # Create and setup surface
    surf = NurbsSurface.create(3, False, 2, 2, 2, 2)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    surf.set_cv(1, 1, Point(1.0, 1.0, 0.0))

    MINI_CHECK(surf is not None)
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(not surf.is_closed(0))
    MINI_CHECK(not surf.is_closed(1))
    MINI_CHECK(not surf.is_periodic(0))
    MINI_CHECK(not surf.is_periodic(1))
    MINI_CHECK(surf.is_clamped(0, 0))
    MINI_CHECK(surf.is_clamped(1, 0))
    MINI_CHECK(surf.is_planar(1e-6))


@MINI_TEST("NurbsSurface", "modification")
def test_modification():
    from session_py import NurbsSurface
    from session_py import Point

    surf = NurbsSurface.create(3, False, 2, 2, 3, 2)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

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

    MINI_CHECK(surf is not None)
    MINI_CHECK(cv_after[0] == cv_before[0])
    MINI_CHECK(surf.order(0) == order_v_before)
    MINI_CHECK(surf.order(1) == order_u_before)


@MINI_TEST("NurbsSurface", "isocurve")
def test_isocurve():
    from session_py import NurbsSurface
    from session_py import Point

    # Create surface
    surf = NurbsSurface.create(3, False, 3, 3, 3, 3)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

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

    MINI_CHECK(surf is not None)
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(iso_u is not None)
    MINI_CHECK(iso_v is not None)


@MINI_TEST("NurbsSurface", "transformation")
def test_transformation():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Xform

    # Create simple surface
    surf = NurbsSurface.create(3, False, 2, 2, 2, 2)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

    surf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    surf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    surf.set_cv(1, 1, Point(1.0, 1.0, 0.0))

    # Apply translation
    xf = Xform.translation(1.0, 2.0, 3.0)
    surf.transform(xf)

    # Check transformed CV
    pt = surf.get_cv(0, 0)

    MINI_CHECK(surf is not None)
    MINI_CHECK(pt is not None)
    MINI_CHECK(TOLERANCE.is_close(pt[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 3.0))


@MINI_TEST("NurbsSurface", "json_roundtrip")
def test_json_roundtrip():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Color
    from pathlib import Path

    # Create and setup surface
    surf = NurbsSurface.create(3, False, 3, 3, 3, 3)
    surf.name = "test_nurbssurface"
    surf.width = 2.0
    surf.surfacecolor = Color(255, 128, 64, 255)

    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

    # Set some CVs
    for i in range(3):
        for j in range(3):
            surf.set_cv(i, j, Point(float(i), float(j), 0.0))

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   json_dumps()    │ str          │ to JSON string
    #   json_loads(s)   │ str          │ from JSON string
    #   json_dump(path) │ file         │ write to file
    #   json_load(path) │ file         │ read from file

    # Serialize to JSON
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbssurface.json"
    surf.json_dump(fname)
    loaded = NurbsSurface.json_load(fname)

    MINI_CHECK(surf is not None)
    MINI_CHECK(loaded.name == surf.name)
    MINI_CHECK(loaded.width == surf.width)
    MINI_CHECK(loaded.m_dim == surf.m_dim)
    MINI_CHECK(loaded.m_is_rat == surf.m_is_rat)
    MINI_CHECK(loaded.m_order[0] == surf.m_order[0])
    MINI_CHECK(loaded.m_order[1] == surf.m_order[1])
    MINI_CHECK(loaded.m_cv_count[0] == surf.m_cv_count[0])
    MINI_CHECK(loaded.m_cv_count[1] == surf.m_cv_count[1])
    MINI_CHECK(loaded.surfacecolor[0] == 255)
    MINI_CHECK(loaded.surfacecolor[1] == 128)
    MINI_CHECK(loaded.surfacecolor[2] == 64)


@MINI_TEST("NurbsSurface", "protobuf_roundtrip")
def test_protobuf_roundtrip():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Color
    from pathlib import Path

    # Create and setup surface
    surf = NurbsSurface.create(3, False, 3, 3, 3, 3)
    surf.name = "test_nurbssurface"
    surf.width = 2.0
    surf.surfacecolor = Color(255, 128, 64, 255)

    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

    # Set some CVs
    for i in range(3):
        for j in range(3):
            surf.set_cv(i, j, Point(float(i), float(j), 0.0))

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbssurface.bin"
    surf.pb_dump(path)
    loaded = NurbsSurface.pb_load(path)

    MINI_CHECK(surf is not None)
    MINI_CHECK(loaded.name == surf.name)
    MINI_CHECK(loaded.width == surf.width)
    MINI_CHECK(loaded.m_dim == surf.m_dim)
    MINI_CHECK(loaded.m_is_rat == surf.m_is_rat)
    MINI_CHECK(loaded.m_order[0] == surf.m_order[0])
    MINI_CHECK(loaded.m_order[1] == surf.m_order[1])
    MINI_CHECK(loaded.m_cv_count[0] == surf.m_cv_count[0])
    MINI_CHECK(loaded.m_cv_count[1] == surf.m_cv_count[1])
    MINI_CHECK(loaded.surfacecolor[0] == 255)
    MINI_CHECK(loaded.surfacecolor[1] == 128)
    MINI_CHECK(loaded.surfacecolor[2] == 64)


@MINI_TEST("NurbsSurface", "advanced_accessors")
def test_advanced_accessors():
    from session_py import NurbsSurface
    from session_py import Point

    # Create rational surface for testing get_cv_4d/set_cv_4d
    surf = NurbsSurface.create(3, True, 3, 3, 3, 3)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

    # Test set_cv_4d with homogeneous coordinates
    x, y, z, w = 2.0, 3.0, 4.0, 2.0

    # Set CV using set_cv first
    surf.set_cv(1, 1, Point(x, y, z))
    surf.set_weight(1, 1, w)

    # Get CV and verify
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

    MINI_CHECK(surf is not None)
    MINI_CHECK(surf.is_rational())
    MINI_CHECK(pt[0] == x and pt[1] == y and pt[2] == z)
    MINI_CHECK(retrieved_w == w)
    MINI_CHECK(first_knot_mult > 0)


@MINI_TEST("NurbsSurface", "clamp_operations")
def test_clamp_operations():
    from session_py import NurbsSurface
    from session_py import Point

    surf = NurbsSurface.create(3, False, 4, 4, 4, 4)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

    # Set up control points
    for i in range(4):
        for j in range(4):
            surf.set_cv(i, j, Point(float(i), float(j), 0.0))

    # Test clamp_end
    was_clamped_before = surf.is_clamped(0, 2)
    surf.clamp_end(0, 2)
    is_clamped_after = surf.is_clamped(0, 2)

    MINI_CHECK(surf is not None)
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(is_clamped_after)


@MINI_TEST("NurbsSurface", "singularity")
def test_singularity():
    from session_py import NurbsSurface
    from session_py import Point

    # Create a simple surface
    surf = NurbsSurface.create(3, False, 2, 2, 2, 2)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

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

    MINI_CHECK(surf is not None)
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(not is_singular_south)
    MINI_CHECK(not is_singular_east)
    MINI_CHECK(not is_singular_north)
    MINI_CHECK(not is_singular_west)


@MINI_TEST("NurbsSurface", "bounding_box")
def test_bounding_box():
    from session_py import NurbsSurface
    from session_py import Point

    surf = NurbsSurface.create(3, False, 2, 2, 3, 3)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

    # Set CVs in a known range
    for i in range(3):
        for j in range(3):
            surf.set_cv(i, j, Point(float(i), float(j), 0.0))

    # Get bounding box
    bbox = surf.get_bounding_box()

    MINI_CHECK(surf is not None)
    MINI_CHECK(surf.is_valid())
    MINI_CHECK(bbox is not None)


@MINI_TEST("NurbsSurface", "domain_operations")
def test_domain_operations():
    from session_py import NurbsSurface
    from .tolerance import TOLERANCE

    surf = NurbsSurface.create(3, False, 3, 3, 3, 3)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

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
    from .tolerance import TOLERANCE

    surf = NurbsSurface.create(3, False, 2, 2, 2, 2)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

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
    from .tolerance import TOLERANCE

    surf = NurbsSurface.create(3, False, 2, 2, 2, 2)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

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
    from .tolerance import TOLERANCE

    surf = NurbsSurface.create(3, False, 2, 2, 2, 2)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

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

    surf = NurbsSurface.create(3, False, 4, 3, 4, 3)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 2.0)

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
    surf = NurbsSurface.create(3, True, 3, 3, 3, 3)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

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

    surf = NurbsSurface.create(3, False, 4, 4, 4, 4)
    surf.make_clamped_uniform_knot_vector(0, 1.0)
    surf.make_clamped_uniform_knot_vector(1, 1.0)

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
    w = math.sqrt(2.0) / 2.0
    pi = math.pi

    surf = NurbsSurface.create(3, True, 3, 3, 9, 5)
    surf.name = "unit_sphere"

    u_knots = [0, 0, pi * 0.5, pi * 0.5, pi, pi, pi * 1.5, pi * 1.5, pi * 2.0, pi * 2.0]
    for i in range(10):
        surf.set_knot(0, i, u_knots[i])

    v_knots = [-pi * 0.5, -pi * 0.5, 0, 0, pi * 0.5, pi * 0.5]
    for i in range(6):
        surf.set_knot(1, i, v_knots[i])

    lat_weights = [w, 0.5, w, 0.5, w]
    lat_z = [-radius, -radius * w, 0.0, radius * w, radius]
    lat_r = [0.0, radius * w, radius, radius * w, 0.0]

    for j in range(5):
        r = lat_r[j]
        z = lat_z[j]
        angles = [0, pi * 0.25, pi * 0.5, pi * 0.75, pi, pi * 1.25, pi * 1.5, pi * 1.75, pi * 2.0]
        for i in range(9):
            x = r * math.cos(angles[i])
            y = r * math.sin(angles[i])
            surf.set_cv(i, j, Point(x, y, z))
            weight = w if i % 2 == 0 else lat_weights[j]
            if j == 0 or j == 4:
                weight = w
            surf.set_weight(i, j, weight)

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(surf.is_rational())
    MINI_CHECK(surf.degree(0) == 2)
    MINI_CHECK(surf.degree(1) == 2)
    MINI_CHECK(surf.cv_count_dir(0) == 9)
    MINI_CHECK(surf.cv_count_dir(1) == 5)

    pt = surf.point_at(0.0, 0.0)
    MINI_CHECK(TOLERANCE.is_close(pt[0], radius))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 0.0))

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
    pi = math.pi

    surf = NurbsSurface.create(3, True, 3, 2, 9, 2)
    surf.name = "unit_cylinder"

    u_knots = [0, 0, pi * 0.5, pi * 0.5, pi, pi, pi * 1.5, pi * 1.5, pi * 2.0, pi * 2.0]
    for i in range(10):
        surf.set_knot(0, i, u_knots[i])

    surf.set_knot(1, 0, 0.0)
    surf.set_knot(1, 1, height)

    angles = [0, pi * 0.25, pi * 0.5, pi * 0.75, pi, pi * 1.25, pi * 1.5, pi * 1.75, pi * 2.0]

    for j in range(2):
        z = 0.0 if j == 0 else height
        for i in range(9):
            if i % 2 == 1:
                x = radius * math.sqrt(2.0) * math.cos(angles[i])
                y = radius * math.sqrt(2.0) * math.sin(angles[i])
            else:
                x = radius * math.cos(angles[i])
                y = radius * math.sin(angles[i])
            surf.set_cv(i, j, Point(x, y, z))
            weight = 1.0 if i % 2 == 0 else w
            surf.set_weight(i, j, weight)

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(surf.is_rational())
    MINI_CHECK(surf.degree(0) == 2)
    MINI_CHECK(surf.degree(1) == 1)
    MINI_CHECK(surf.cv_count_dir(0) == 9)
    MINI_CHECK(surf.cv_count_dir(1) == 2)

    pt_bottom = surf.point_at(0.0, 0.0)
    MINI_CHECK(TOLERANCE.is_close(pt_bottom[0], radius))
    MINI_CHECK(TOLERANCE.is_close(pt_bottom[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt_bottom[2], 0.0))

    pt_top = surf.point_at(0.0, height)
    MINI_CHECK(TOLERANCE.is_close(pt_top[0], radius))
    MINI_CHECK(TOLERANCE.is_close(pt_top[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt_top[2], height))

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
    pi = math.pi

    surf = NurbsSurface.create(3, True, 3, 3, 9, 9)
    surf.name = "unit_torus"

    knots = [0, 0, pi * 0.5, pi * 0.5, pi, pi, pi * 1.5, pi * 1.5, pi * 2.0, pi * 2.0]
    for i in range(10):
        surf.set_knot(0, i, knots[i])
        surf.set_knot(1, i, knots[i])

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

            w_major = 1.0 if i % 2 == 0 else w
            w_minor = 1.0 if j % 2 == 0 else w
            surf.set_weight(i, j, w_major * w_minor)

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(surf.is_rational())
    MINI_CHECK(surf.degree(0) == 2)
    MINI_CHECK(surf.degree(1) == 2)
    MINI_CHECK(surf.cv_count_dir(0) == 9)
    MINI_CHECK(surf.cv_count_dir(1) == 9)

    pt = surf.point_at(0.0, 0.0)
    MINI_CHECK(TOLERANCE.is_close(pt[0], major_radius + minor_radius))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 0.0))

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
    pi = math.pi

    surf = NurbsSurface.create(3, True, 3, 2, 9, 2)
    surf.name = "unit_cone"

    u_knots = [0, 0, pi * 0.5, pi * 0.5, pi, pi, pi * 1.5, pi * 1.5, pi * 2.0, pi * 2.0]
    for i in range(10):
        surf.set_knot(0, i, u_knots[i])

    surf.set_knot(1, 0, 0.0)
    surf.set_knot(1, 1, height)

    angles = [0, pi * 0.25, pi * 0.5, pi * 0.75, pi, pi * 1.25, pi * 1.5, pi * 1.75, pi * 2.0]

    for i in range(9):
        surf.set_cv(i, 0, Point(0.0, 0.0, height))
        weight = 1.0 if i % 2 == 0 else w
        surf.set_weight(i, 0, weight)

    for i in range(9):
        if i % 2 == 1:
            x = radius * math.sqrt(2.0) * math.cos(angles[i])
            y = radius * math.sqrt(2.0) * math.sin(angles[i])
        else:
            x = radius * math.cos(angles[i])
            y = radius * math.sin(angles[i])
        surf.set_cv(i, 1, Point(x, y, 0.0))
        weight = 1.0 if i % 2 == 0 else w
        surf.set_weight(i, 1, weight)

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(surf.is_rational())
    MINI_CHECK(surf.degree(0) == 2)
    MINI_CHECK(surf.degree(1) == 1)
    MINI_CHECK(surf.cv_count_dir(0) == 9)
    MINI_CHECK(surf.cv_count_dir(1) == 2)

    apex = surf.point_at(0.0, 0.0)
    MINI_CHECK(TOLERANCE.is_close(apex[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(apex[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(apex[2], height))

    base = surf.point_at(0.0, height)
    MINI_CHECK(TOLERANCE.is_close(base[0], radius))
    MINI_CHECK(TOLERANCE.is_close(base[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(base[2], 0.0))

    mid = surf.point_at(0.0, height * 0.5)
    MINI_CHECK(TOLERANCE.is_close(mid[0], radius * 0.5))
    MINI_CHECK(TOLERANCE.is_close(mid[2], height * 0.5))


if __name__ == "__main__":
    run_all(language="python")
