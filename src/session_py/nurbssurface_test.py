from .nurbssurface import NurbsSurface
from .point import Point
from .vector import Vector


def test_nurbssurface_constructor():
    """Test basic NurbsSurface construction"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    assert srf.dimension() == 3
    assert srf.is_rational() == False
    assert srf.order(0) == 4
    assert srf.order(1) == 4
    assert srf.cv_count(0) == 5
    assert srf.cv_count(1) == 5
    assert srf.degree(0) == 3
    assert srf.degree(1) == 3


def test_nurbssurface_validation():
    """Test surface validation before and after setup"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    # Valid after creation (has knot arrays allocated)
    assert srf.is_valid() == True
    
    # Still valid after setting knots
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    assert srf.is_valid() == True


def test_nurbssurface_domain():
    """Test domain queries"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    
    # With 5 CVs and order 4, clamped uniform knot vector with delta=1.0
    # creates knots at indices 2,3,4 with values 0.0, 1.0, 2.0
    # Domain is [first_interior, last_interior] = [0.0, 2.0]
    assert u0 == 0.0
    assert u1 == 2.0
    assert v0 == 0.0
    assert v1 == 2.0


def test_nurbssurface_cv_access():
    """Test control vertex access"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    # Set a CV
    assert srf.set_cv(2, 2, Point(2, 3, 4)) == True
    
    # Get it back
    cv = srf.get_cv(2, 2)
    assert cv.x == 2.0
    assert cv.y == 3.0
    assert cv.z == 4.0


def test_nurbssurface_make_rational():
    """Test conversion to rational"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    assert srf.is_rational() == False
    assert srf.make_rational() == True
    assert srf.is_rational() == True


def test_nurbssurface_point_evaluation():
    """Test surface point evaluation"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    # Set CVs in a grid (non-zero to avoid divide by zero)
    for i in range(5):
        for j in range(5):
            srf.set_cv(i, j, Point(i * 1.0, j * 1.0, 0.1))
    
    # Evaluate at mid-domain (avoid boundary issues)
    pt = srf.point_at(0.5, 0.5)
    # Just check it doesn't crash
    assert pt is not None


def test_nurbssurface_reverse():
    """Test reversing surface direction"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    for i in range(5):
        for j in range(5):
            srf.set_cv(i, j, Point(i * 1.0, j * 1.0, 0))
    
    assert srf.reverse(0) == True
    assert srf.is_valid() == True


def test_nurbssurface_transpose():
    """Test transposing u and v directions"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    for i in range(5):
        for j in range(5):
            srf.set_cv(i, j, Point(i * 1.0, j * 1.0, 0))
    
    assert srf.transpose() == True
    assert srf.is_valid() == True


def test_nurbssurface_geometric_queries():
    """Test geometric property queries"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    for i in range(5):
        for j in range(5):
            srf.set_cv(i, j, Point(i * 1.0, j * 1.0, 0))
    
    assert srf.is_closed(0) == False
    assert srf.is_periodic(0) == False
    # Planar surface (all z=0)
    assert srf.is_planar(None, 0.01) == True


def test_nurbssurface_string_representation():
    """Test string conversion"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    s = str(srf)
    assert "NurbsSurface" in s
    assert "dim=3" in s
    assert "order=(4,4)" in s
    assert "cv_count=(5,5)" in s


###########################################################################################
# JSON
###########################################################################################

def test_nurbssurface_make_non_rational():
    """Test make_non_rational (OpenNURBS implementation)"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    for i in range(5):
        for j in range(5):
            srf.set_cv(i, j, Point(i * 1.0, j * 1.0, 0.1))
    
    # Make rational first
    assert srf.make_rational() == True
    assert srf.is_rational() == True
    
    # Then make non-rational
    assert srf.make_non_rational() == True
    assert srf.is_rational() == False


def test_nurbssurface_clamp_end():
    """Test clamp_end (OpenNURBS implementation)"""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    # Clamp start of u direction
    assert srf.clamp_end(0, 0) == True
    
    # Clamp end of v direction
    assert srf.clamp_end(1, 1) == True
    
    # Clamp both ends of u direction
    assert srf.clamp_end(0, 2) == True


# JSON serialization test skipped - not yet implemented in encoders

# def test_nurbssurface_json_roundtrip():
#     """Test JSON serialization roundtrip"""
#     from pathlib import Path
#     from session_py.encoders import json_dump, json_load
#     
#     srf = NurbsSurface(3, False, 4, 4, 5, 5)
#     srf.name = "test_surface"
#     srf.make_clamped_uniform_knot_vector(0, 1.0)
#     srf.make_clamped_uniform_knot_vector(1, 1.0)
#     
#     # Set some CVs
#     for i in range(5):
#         for j in range(5):
#             srf.set_cv(i, j, Point(i * 0.5, j * 0.5, 0))
#     
#     path = Path(__file__).resolve().parents[2] / "test_nurbssurface.json"
#     json_dump(srf, path)
#     loaded = json_load(path)
#     
#     assert isinstance(loaded, NurbsSurface)
#     assert loaded.dimension() == 3
#     assert loaded.order(0) == 4
#     assert loaded.order(1) == 4
#     assert loaded.cv_count(0) == 5
#     assert loaded.cv_count(1) == 5
#     assert loaded.name == "test_surface"


def test_nurbssurface_10x10_grid_evaluation():
    """Test 10x10 grid evaluation with mesh generation and coordinate verification."""
    from .mesh import Mesh
    
    # Create NURBS surface (3D, order 4, 5x5 control points)
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    
    # Setup knot vectors
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    # Set control points (corners inverted, middle lifted)
    srf.set_cv(0, 0, Point(0.0, 0.0, -2.5))
    srf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    srf.set_cv(0, 2, Point(0.0, 2.0, 0.0))
    srf.set_cv(0, 3, Point(0.0, 3.0, 0.0))
    srf.set_cv(0, 4, Point(0.0, 4.0, -2.5))
    
    srf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    srf.set_cv(1, 1, Point(1.0, 1.0, 0.0))
    srf.set_cv(1, 2, Point(1.0, 2.0, 5.0))
    srf.set_cv(1, 3, Point(1.0, 3.0, 0.0))
    srf.set_cv(1, 4, Point(1.0, 4.0, 0.0))
    
    srf.set_cv(2, 0, Point(2.0, 0.0, 0.0))
    srf.set_cv(2, 1, Point(2.0, 1.0, 0.0))
    srf.set_cv(2, 2, Point(2.0, 2.0, 0.0))
    srf.set_cv(2, 3, Point(2.0, 3.0, 0.0))
    srf.set_cv(2, 4, Point(2.0, 4.0, 0.0))
    
    srf.set_cv(3, 0, Point(3.0, 0.0, 0.0))
    srf.set_cv(3, 1, Point(3.0, 1.0, 0.0))
    srf.set_cv(3, 2, Point(3.0, 2.0, 0.0))
    srf.set_cv(3, 3, Point(3.0, 3.0, 0.0))
    srf.set_cv(3, 4, Point(3.0, 4.0, 0.0))
    
    srf.set_cv(4, 0, Point(4.0, 0.0, -2.5))
    srf.set_cv(4, 1, Point(4.0, 1.0, 0.0))
    srf.set_cv(4, 2, Point(4.0, 2.0, 0.0))
    srf.set_cv(4, 3, Point(4.0, 3.0, 0.0))
    srf.set_cv(4, 4, Point(4.0, 4.0, -2.5))
    
    # Expected coordinates from our corrected OpenNURBS implementation (rounded to 3 decimals)
    expected = [
        (0.000, 0.000, -2.500), (0.000, 0.598, -1.176), (0.000, 1.081, -0.429), (0.000, 1.481, -0.093), (0.000, 1.833, -0.003),
        (0.000, 2.167, -0.003), (0.000, 2.519, -0.093), (0.000, 2.919, -0.429), (0.000, 3.402, -1.176), (0.000, 4.000, -2.500),
        (0.598, 0.000, -1.176), (0.598, 0.598, -0.407), (0.598, 1.081, 0.282), (0.598, 1.481, 0.815), (0.598, 1.833, 1.118),
        (0.598, 2.167, 1.118), (0.598, 2.519, 0.815), (0.598, 2.919, 0.282), (0.598, 3.402, -0.407), (0.598, 4.000, -1.176),
        (1.081, 0.000, -0.429), (1.081, 0.598, -0.013), (1.081, 1.081, 0.550), (1.081, 1.481, 1.092), (1.081, 1.833, 1.443),
        (1.081, 2.167, 1.443), (1.081, 2.519, 1.092), (1.081, 2.919, 0.550), (1.081, 3.402, -0.013), (1.081, 4.000, -0.429),
        (1.481, 0.000, -0.093), (1.481, 0.598, 0.120), (1.481, 1.081, 0.525), (1.481, 1.481, 0.957), (1.481, 1.833, 1.252),
        (1.481, 2.167, 1.252), (1.481, 2.519, 0.957), (1.481, 2.919, 0.525), (1.481, 3.402, 0.120), (1.481, 4.000, -0.093),
        (1.833, 0.000, -0.003), (1.833, 0.598, 0.106), (1.833, 1.081, 0.354), (1.833, 1.481, 0.630), (1.833, 1.833, 0.821),
        (1.833, 2.167, 0.821), (1.833, 2.519, 0.630), (1.833, 2.919, 0.354), (1.833, 3.402, 0.106), (1.833, 4.000, -0.003),
        (2.167, 0.000, -0.003), (2.167, 0.598, 0.054), (2.167, 1.081, 0.182), (2.167, 1.481, 0.325), (2.167, 1.833, 0.424),
        (2.167, 2.167, 0.424), (2.167, 2.519, 0.325), (2.167, 2.919, 0.182), (2.167, 3.402, 0.054), (2.167, 4.000, -0.003),
        (2.519, 0.000, -0.093), (2.519, 0.598, -0.020), (2.519, 1.081, 0.061), (2.519, 1.481, 0.134), (2.519, 1.833, 0.179),
        (2.519, 2.167, 0.179), (2.519, 2.519, 0.134), (2.519, 2.919, 0.061), (2.519, 3.402, -0.020), (2.519, 4.000, -0.093),
        (2.919, 0.000, -0.429), (2.919, 0.598, -0.195), (2.919, 1.081, -0.051), (2.919, 1.481, 0.025), (2.919, 1.833, 0.052),
        (2.919, 2.167, 0.052), (2.919, 2.519, 0.025), (2.919, 2.919, -0.051), (2.919, 3.402, -0.195), (2.919, 4.000, -0.429),
        (3.402, 0.000, -1.176), (3.402, 0.598, -0.553), (3.402, 1.081, -0.199), (3.402, 1.481, -0.038), (3.402, 1.833, 0.005),
        (3.402, 2.167, 0.005), (3.402, 2.519, -0.038), (3.402, 2.919, -0.199), (3.402, 3.402, -0.553), (3.402, 4.000, -1.176),
        (4.000, 0.000, -2.500), (4.000, 0.598, -1.176), (4.000, 1.081, -0.429), (4.000, 1.481, -0.093), (4.000, 1.833, -0.003),
        (4.000, 2.167, -0.003), (4.000, 2.519, -0.093), (4.000, 2.919, -0.429), (4.000, 3.402, -1.176), (4.000, 4.000, -2.500),
    ]
    
    # Evaluate 10x10 grid
    grid_size = 10
    u_min, u_max = srf.domain(0)
    v_min, v_max = srf.domain(1)
    
    # Create mesh
    mesh = Mesh()
    vertex_keys = []
    
    for i in range(grid_size):
        row_keys = []
        for j in range(grid_size):
            u = u_min + (u_max - u_min) * i / (grid_size - 1)
            v = v_min + (v_max - v_min) * j / (grid_size - 1)
            pt = srf.point_at(u, v)
            key = mesh.add_vertex(pt)
            row_keys.append(key)
        vertex_keys.append(row_keys)
    
    # Verify mesh vertices
    assert mesh.number_of_vertices() == 100
    
    # Check all coordinates (rounded to 3 decimals)
    # Collect all vertex keys in order
    all_keys = [key for row in vertex_keys for key in row]
    for idx, (exp_x, exp_y, exp_z) in enumerate(expected):
        vd = mesh.vertex[all_keys[idx]]
        assert round(vd.x, 3) == exp_x, f"Vertex {idx} x: {round(vd.x, 3)} != {exp_x}"
        assert round(vd.y, 3) == exp_y, f"Vertex {idx} y: {round(vd.y, 3)} != {exp_y}"
        assert round(vd.z, 3) == exp_z, f"Vertex {idx} z: {round(vd.z, 3)} != {exp_z}"
    
    # Create quad faces
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            mesh.add_face([vertex_keys[i][j], vertex_keys[i+1][j], 
                          vertex_keys[i+1][j+1], vertex_keys[i][j+1]])
    
    # Verify mesh topology
    assert mesh.number_of_faces() == 81
    assert mesh.number_of_edges() == 180


def test_nurbssurface_isocurve_extraction():
    """Test isocurve extraction at surface mid-parameters."""
    # Create NURBS surface (3D, order 4, 5x5 control points)
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    
    # Setup knot vectors
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)
    
    # Set control points
    srf.set_cv(0, 0, Point(0.0, 0.0, -2.5))
    srf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
    srf.set_cv(0, 2, Point(0.0, 2.0, 0.0))
    srf.set_cv(0, 3, Point(0.0, 3.0, 0.0))
    srf.set_cv(0, 4, Point(0.0, 4.0, -2.5))
    
    srf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    srf.set_cv(1, 1, Point(1.0, 1.0, 0.0))
    srf.set_cv(1, 2, Point(1.0, 2.0, 5.0))
    srf.set_cv(1, 3, Point(1.0, 3.0, 0.0))
    srf.set_cv(1, 4, Point(1.0, 4.0, 0.0))
    
    srf.set_cv(2, 0, Point(2.0, 0.0, 0.0))
    srf.set_cv(2, 1, Point(2.0, 1.0, 0.0))
    srf.set_cv(2, 2, Point(2.0, 2.0, 0.0))
    srf.set_cv(2, 3, Point(2.0, 3.0, 0.0))
    srf.set_cv(2, 4, Point(2.0, 4.0, 0.0))
    
    srf.set_cv(3, 0, Point(3.0, 0.0, 0.0))
    srf.set_cv(3, 1, Point(3.0, 1.0, 0.0))
    srf.set_cv(3, 2, Point(3.0, 2.0, 0.0))
    srf.set_cv(3, 3, Point(3.0, 3.0, 0.0))
    srf.set_cv(3, 4, Point(3.0, 4.0, 0.0))
    
    srf.set_cv(4, 0, Point(4.0, 0.0, -2.5))
    srf.set_cv(4, 1, Point(4.0, 1.0, 0.0))
    srf.set_cv(4, 2, Point(4.0, 2.0, 0.0))
    srf.set_cv(4, 3, Point(4.0, 3.0, 0.0))
    srf.set_cv(4, 4, Point(4.0, 4.0, -2.5))
    
    u_min, u_max = srf.domain(0)
    v_min, v_max = srf.domain(1)
    u_param = 0.5 * (u_min + u_max)
    v_param = 0.5 * (v_min + v_max)
    divisions = 50
    
    # Extract iso-v curve (u varies, v constant at mid)
    iso_v_pts = []
    for i in range(divisions):
        u = u_min + (u_max - u_min) * i / (divisions - 1)
        pt = srf.point_at(u, v_param)
        iso_v_pts.append(pt)
    
    # Extract iso-u curve (v varies, u constant at mid)
    iso_u_pts = []
    for j in range(divisions):
        v = v_min + (v_max - v_min) * j / (divisions - 1)
        pt = srf.point_at(u_param, v)
        iso_u_pts.append(pt)
    
    # Expected iso-v coordinates (u varies, v=2.0)
    expected_iso_v = [
        (0.000, 2.000, 0.000), (0.120, 2.000, 0.288), (0.235, 2.000, 0.540), (0.346, 2.000, 0.758), (0.452, 2.000, 0.944),
        (0.554, 2.000, 1.099), (0.652, 2.000, 1.226), (0.746, 2.000, 1.327), (0.837, 2.000, 1.402), (0.924, 2.000, 1.454),
        (1.009, 2.000, 1.485), (1.090, 2.000, 1.496), (1.168, 2.000, 1.489), (1.244, 2.000, 1.466), (1.318, 2.000, 1.429),
        (1.389, 2.000, 1.379), (1.459, 2.000, 1.318), (1.526, 2.000, 1.249), (1.593, 2.000, 1.173), (1.658, 2.000, 1.091),
        (1.721, 2.000, 1.006), (1.784, 2.000, 0.918), (1.846, 2.000, 0.831), (1.908, 2.000, 0.746), (1.969, 2.000, 0.664),
        (2.031, 2.000, 0.588), (2.092, 2.000, 0.517), (2.154, 2.000, 0.453), (2.216, 2.000, 0.394), (2.279, 2.000, 0.340),
        (2.342, 2.000, 0.292), (2.407, 2.000, 0.248), (2.474, 2.000, 0.209), (2.541, 2.000, 0.174), (2.611, 2.000, 0.143),
        (2.682, 2.000, 0.117), (2.756, 2.000, 0.093), (2.832, 2.000, 0.073), (2.910, 2.000, 0.057), (2.991, 2.000, 0.042),
        (3.076, 2.000, 0.031), (3.163, 2.000, 0.022), (3.254, 2.000, 0.015), (3.348, 2.000, 0.009), (3.446, 2.000, 0.005),
        (3.548, 2.000, 0.003), (3.654, 2.000, 0.001), (3.765, 2.000, 0.000), (3.880, 2.000, 0.000), (4.000, 2.000, 0.000),
    ]
    
    # Expected iso-u coordinates (v varies, u=2.0)
    expected_iso_u = [
        (2.000, 0.000, 0.000), (2.000, 0.120, 0.003), (2.000, 0.235, 0.012), (2.000, 0.346, 0.026), (2.000, 0.452, 0.045),
        (2.000, 0.554, 0.067), (2.000, 0.652, 0.094), (2.000, 0.746, 0.124), (2.000, 0.837, 0.156), (2.000, 0.924, 0.191),
        (2.000, 1.009, 0.227), (2.000, 1.090, 0.265), (2.000, 1.168, 0.303), (2.000, 1.244, 0.341), (2.000, 1.318, 0.379),
        (2.000, 1.389, 0.416), (2.000, 1.459, 0.452), (2.000, 1.526, 0.485), (2.000, 1.593, 0.516), (2.000, 1.658, 0.545),
        (2.000, 1.721, 0.569), (2.000, 1.784, 0.590), (2.000, 1.846, 0.607), (2.000, 1.908, 0.618), (2.000, 1.969, 0.624),
        (2.000, 2.031, 0.624), (2.000, 2.092, 0.618), (2.000, 2.154, 0.607), (2.000, 2.216, 0.590), (2.000, 2.279, 0.569),
        (2.000, 2.342, 0.545), (2.000, 2.407, 0.516), (2.000, 2.474, 0.485), (2.000, 2.541, 0.452), (2.000, 2.611, 0.416),
        (2.000, 2.682, 0.379), (2.000, 2.756, 0.341), (2.000, 2.832, 0.303), (2.000, 2.910, 0.265), (2.000, 2.991, 0.227),
        (2.000, 3.076, 0.191), (2.000, 3.163, 0.156), (2.000, 3.254, 0.124), (2.000, 3.348, 0.094), (2.000, 3.446, 0.067),
        (2.000, 3.548, 0.045), (2.000, 3.654, 0.026), (2.000, 3.765, 0.012), (2.000, 3.880, 0.003), (2.000, 4.000, 0.000),
    ]
    
    # Verify iso-v points
    assert len(iso_v_pts) == 50
    for i, (exp_x, exp_y, exp_z) in enumerate(expected_iso_v):
        pt = iso_v_pts[i]
        assert round(pt.x, 3) == exp_x, f"Iso-v point {i} x: {round(pt.x, 3)} != {exp_x}"
        assert round(pt.y, 3) == exp_y, f"Iso-v point {i} y: {round(pt.y, 3)} != {exp_y}"
        assert round(pt.z, 3) == exp_z, f"Iso-v point {i} z: {round(pt.z, 3)} != {exp_z}"
    
    # Verify iso-u points
    assert len(iso_u_pts) == 50
    for i, (exp_x, exp_y, exp_z) in enumerate(expected_iso_u):
        pt = iso_u_pts[i]
        assert round(pt.x, 3) == exp_x, f"Iso-u point {i} x: {round(pt.x, 3)} != {exp_x}"
        assert round(pt.y, 3) == exp_y, f"Iso-u point {i} y: {round(pt.y, 3)} != {exp_y}"
        assert round(pt.z, 3) == exp_z, f"Iso-u point {i} z: {round(pt.z, 3)} != {exp_z}"


def test_nurbssurface_center_frame_numeric_3dec():
    """Short test: center point value at (u_mid, v_mid) rounded to 3 decimals."""
    srf = NurbsSurface(3, False, 4, 4, 5, 5)
    srf.make_clamped_uniform_knot_vector(0, 1.0)
    srf.make_clamped_uniform_knot_vector(1, 1.0)

    srf.set_cv(0, 0, Point(0.0, 0.0, -2.5))
    srf.set_cv(0, 1, Point(0.0, 1.0,  0.0))
    srf.set_cv(0, 2, Point(0.0, 2.0,  0.0))
    srf.set_cv(0, 3, Point(0.0, 3.0,  0.0))
    srf.set_cv(0, 4, Point(0.0, 4.0, -2.5))

    srf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
    srf.set_cv(1, 1, Point(1.0, 1.0, 0.0))
    srf.set_cv(1, 2, Point(1.0, 2.0, 5.0))
    srf.set_cv(1, 3, Point(1.0, 3.0, 0.0))
    srf.set_cv(1, 4, Point(1.0, 4.0, 0.0))

    srf.set_cv(2, 0, Point(2.0, 0.0, 0.0))
    srf.set_cv(2, 1, Point(2.0, 1.0, 0.0))
    srf.set_cv(2, 2, Point(2.0, 2.0, 0.0))
    srf.set_cv(2, 3, Point(2.0, 3.0, 0.0))
    srf.set_cv(2, 4, Point(2.0, 4.0, 0.0))

    srf.set_cv(3, 0, Point(3.0, 0.0, 0.0))
    srf.set_cv(3, 1, Point(3.0, 1.0, 0.0))
    srf.set_cv(3, 2, Point(3.0, 2.0, 0.0))
    srf.set_cv(3, 3, Point(3.0, 3.0, 0.0))
    srf.set_cv(3, 4, Point(3.0, 4.0, 0.0))

    srf.set_cv(4, 0, Point(4.0, 0.0, -2.5))
    srf.set_cv(4, 1, Point(4.0, 1.0, 0.0))
    srf.set_cv(4, 2, Point(4.0, 2.0, 0.0))
    srf.set_cv(4, 3, Point(4.0, 3.0, 0.0))
    srf.set_cv(4, 4, Point(4.0, 4.0, -2.5))

    u_min, u_max = srf.domain(0)
    v_min, v_max = srf.domain(1)
    u_mid = 0.5 * (u_min + u_max)
    v_mid = 0.5 * (v_min + v_max)

    pt = srf.point_at(u_mid, v_mid)
    assert round(pt.x, 3) == 2.000
    assert round(pt.y, 3) == 2.000
    assert round(pt.z, 3) == 0.625
