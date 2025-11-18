from .nurbscurve import NurbsCurve
from .point import Point
from .vector import Vector
from .plane import Plane


def test_nurbscurve_constructor():
    """Test basic NurbsCurve construction"""
    curve = NurbsCurve(3, False, 4, 5)
    assert curve.dimension() == 3
    assert curve.is_rational() == False
    assert curve.order() == 4
    assert curve.degree() == 3
    assert curve.cv_count() == 5


def test_nurbscurve_create():
    """Test creating curve from points"""
    points = [
        Point(0, 0, 0),
        Point(1, 1, 0),
        Point(2, 0, 0)
    ]
    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    assert curve.is_valid() == True
    assert curve.cv_count() == 3


def test_nurbscurve_parabola():
    """Test parabola curve creation (from user's code example)"""
    # Create parabola points
    start_point = Point(0, 0, 0)
    end_point = Point(10, 0, 0)
    rise = 2.0
    
    points = [
        Point(start_point.x, start_point.y, start_point.z),
        Point((start_point.x + end_point.x) * 0.5, 
              (start_point.y + end_point.y) * 0.5, 
              (start_point.z + end_point.z) * 0.5 + rise),
        Point(end_point.x, end_point.y, end_point.z),
    ]
    
    # Create parabolic curve
    curve = NurbsCurve.create(periodic=False, degree=len(points)-1, points=points)
    assert curve.is_valid() == True
    assert curve.cv_count() == 3
    
    # Create planes perpendicular to X-axis
    axis_for_plane_intersection = Vector(1, 0, 0)
    divisions = 7
    half_xysize = 5.0
    step = half_xysize / (divisions - 1)
    
    planes = []
    for i in range(divisions):
        translation = axis_for_plane_intersection * step * i
        plane_origin = Point(
            start_point.x + translation.x,
            start_point.y + translation.y,
            start_point.z + translation.z
        )
        planes.append(Plane.from_point_normal(plane_origin, axis_for_plane_intersection))
    
    assert len(planes) == divisions
    
    # Intersect curve with each plane
    sampled_points = []
    for plane in planes:
        intersection_points = curve.intersect_plane_points(plane)
        if intersection_points:
            sampled_points.append(intersection_points[0])
    
    # Should have intersections
    assert len(sampled_points) > 0
    # First point should be near start
    assert abs(sampled_points[0].x - start_point.x) < 0.1
    # Middle should have positive z (rise)
    if len(sampled_points) >= 4:
        mid_idx = len(sampled_points) // 2
        assert sampled_points[mid_idx].z > 0


def test_nurbscurve_point_evaluation():
    """Test curve point evaluation"""
    points = [Point(0, 0, 0), Point(1, 1, 0), Point(2, 0, 0)]
    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    
    # Evaluate at start
    pt_start = curve.point_at_start()
    assert abs(pt_start.x - 0.0) < 0.01
    
    # Evaluate at end  
    pt_end = curve.point_at_end()
    assert abs(pt_end.x - 2.0) < 0.01


def test_nurbscurve_closed():
    """Test closed curve"""
    points = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
        Point(0, 0, 0)  # Close the loop
    ]
    curve = NurbsCurve.create(periodic=False, degree=3, points=points)
    assert curve.is_closed() == True


def test_nurbscurve_length():
    """Test curve length calculation"""
    points = [Point(0, 0, 0), Point(1, 0, 0)]
    curve = NurbsCurve.create(periodic=False, degree=1, points=points)
    length = curve.length()
    assert abs(length - 1.0) < 0.01


def test_nurbscurve_reverse():
    """Test reversing curve direction"""
    points = [Point(0, 0, 0), Point(1, 1, 0), Point(2, 0, 0)]
    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    
    pt_start_before = curve.point_at_start()
    pt_end_before = curve.point_at_end()
    
    curve.reverse()
    
    pt_start_after = curve.point_at_start()
    pt_end_after = curve.point_at_end()
    
    # Start and end should be swapped
    assert abs(pt_start_before.x - pt_end_after.x) < 0.01
    assert abs(pt_end_before.x - pt_start_after.x) < 0.01


def test_nurbscurve_make_rational():
    """Test conversion to rational"""
    points = [Point(0, 0, 0), Point(1, 1, 0), Point(2, 0, 0)]
    curve = NurbsCurve.create(periodic=False, degree=2, points=points)
    
    assert curve.is_rational() == False
    curve.make_rational()
    assert curve.is_rational() == True


def test_nurbscurve_string_representation():
    """Test string conversion"""
    curve = NurbsCurve(3, False, 4, 5)
    s = str(curve)
    assert "NurbsCurve" in s


def test_nurbscurve_frames_3d():
    """Short test: compute normal and Frenet frames on a non-planar 3D curve."""
    import math
    # Build a clearly 3D curve (wavy helix)
    ctrl = []
    for k in range(8):
        t = k / 7.0 * 2.0 * math.pi
        r = 1.5 + 0.3 * math.cos(3.0 * t)
        x = r * math.cos(t)
        y = r * math.sin(t)
        z = 0.6 * t
        ctrl.append(Point(x, y, z))

    crv = NurbsCurve.create(periodic=False, degree=3, points=ctrl)
    t0, t1 = crv.domain()
    t = 0.5 * (t0 + t1)

    # Normal plane (plane normal = tangent)
    T = crv.tangent_at(t)
    assert abs(T.magnitude() - 1.0) < 1e-6
    fallback = Vector(0, 0, 1) if abs(T.z) < 0.9 else Vector(0, 1, 0)
    e1 = (T.cross(fallback)).normalize()
    e2 = T.cross(e1)
    assert abs(e1.magnitude() - 1.0) < 1e-6
    assert abs(e2.magnitude() - 1.0) < 1e-6
    assert abs(e1.dot(T)) < 1e-6
    assert abs(e2.dot(T)) < 1e-6
    assert abs(e1.dot(e2)) < 1e-6

    # Frenet frame (T, N, B)
    ders = crv.evaluate(t, 2)
    d1 = ders[1]
    d2 = ders[2]
    T_f = d1.normalize()
    proj = d2.dot(T_f)
    N_raw = Vector(d2.x - T_f.x * proj, d2.y - T_f.y * proj, d2.z - T_f.z * proj)
    assert N_raw.magnitude() > 1e-8
    N = N_raw.normalize()
    B = T_f.cross(N)
    assert abs(T_f.magnitude() - 1.0) < 1e-6
    assert abs(N.magnitude() - 1.0) < 1e-6
    assert abs(B.magnitude() - 1.0) < 1e-6
    assert abs(T_f.dot(N)) < 1e-6
    assert abs(T_f.dot(B)) < 1e-6
    assert abs(N.dot(B)) < 1e-6
    # Right-handed check
    rhs = T_f.cross(N)
    assert rhs.dot(B) > 0.999
