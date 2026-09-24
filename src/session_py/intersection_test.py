from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
import math


def lies_on_curve(curve3d, pcurve, surface):
    """Worst 3D distance from the pcurve lifted onto the surface to the section curve."""

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    dense = []

    for j in range(129):
        dense.append(curve3d.point_at(j / 128.0))

    worst = 0.0

    for i in range(33):
        q = pcurve.point_at(i / 32.0)
        s = surface.point_at(min(max(q[0], u0), u1), min(max(q[1], v0), v1))
        best = dense[0].distance(s)

        for p in dense:
            best = min(best, p.distance(s))

        worst = max(worst, best)

    return worst


def on_both(c3, da, db):
    """Worst distance of the section curve from either analytic surface."""

    worst = 0.0

    for i in range(65):
        p = c3.point_at(i / 64.0)
        worst = max(worst, max(da(p), db(p)))

    return worst


def distance_sphere(p):
    """Distance from the unit-radius-2 sphere at the origin."""
    return abs(math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - 2.0)


def distance_cylinder(p):
    """Distance from the radius-0.3 cylinder on the z axis at x = 1.3."""
    return abs(math.sqrt((p[0] - 1.3) * (p[0] - 1.3) + p[1] * p[1]) - 0.3)


def distance_sphere2(p):
    """Distance from the radius-2 sphere at x = 2."""
    return abs(math.sqrt((p[0] - 2.0) * (p[0] - 2.0) + p[1] * p[1] + p[2] * p[2]) - 2.0)


def distance_torus(p):
    """Distance from the torus of radii 2 and 0.5 at the origin."""
    ring = math.sqrt(p[0] * p[0] + p[1] * p[1]) - 2.0

    return abs(math.sqrt(ring * ring + p[2] * p[2]) - 0.5)


def distance_flat(p):
    """Distance from the xy plane."""
    return abs(p[2])


def bilinear(p00, p01, p10, p11):
    """Planar degree-1 surface through four corner points."""
    from session_py import NurbsSurface

    return NurbsSurface.create(False, False, 1, 1, 2, 2, [p00, p01, p10, p11])


def lifted_distance(pcurve, surface, d):
    """Worst distance of the pcurve lifted onto the surface from a reference shape."""

    t0, t1 = pcurve.domain()
    worst = 0.0

    for i in range(33):
        uv = pcurve.point_at(t0 + (t1 - t0) * i / 32.0)
        worst = max(worst, d(surface.point_at(uv[0], uv[1])))

    return worst


def distance_cone(p):
    """Distance from the cone of base radius 1.5 at z = 0 and apex at z = 3."""
    return abs(math.sqrt(p[0] * p[0] + p[1] * p[1]) - (3.0 - p[2]) * 0.5)


def distance_flat_half(p):
    """Distance from the plane z = 0.5."""
    return abs(p[2] - 0.5)


def distance_wall(p):
    """Distance from the plane x = 0.2."""
    return abs(p[0] - 0.2)


def distance_unit_cylinder(p):
    """Distance from the radius-1 cylinder on the z axis."""
    return abs(math.sqrt(p[0] * p[0] + p[1] * p[1]) - 1.0)


def distance_x_cylinder(p):
    """Distance from the radius-1 cylinder on the x axis."""
    return abs(math.sqrt(p[1] * p[1] + p[2] * p[2]) - 1.0)


def distance_wide_cylinder(p):
    """Distance from the radius-2.2 cylinder on the z axis."""
    return abs(math.sqrt(p[0] * p[0] + p[1] * p[1]) - 2.2)


def distance_high_torus(p):
    """Distance from the torus of radii 1 and 0.3 at z = 1."""
    ring = math.sqrt(p[0] * p[0] + p[1] * p[1]) - 1.0

    return abs(math.sqrt(ring * ring + (p[2] - 1.0) * (p[2] - 1.0)) - 0.3)


def distance_wide_torus(p):
    """Distance from the torus of radii 2.3 and 0.5 at z = 0.3."""
    ring = math.sqrt(p[0] * p[0] + p[1] * p[1]) - 2.3

    return abs(math.sqrt(ring * ring + (p[2] - 0.3) * (p[2] - 0.3)) - 0.5)


def distance_square(p):
    """Distance from the square of half size 1.6 at z = 0.5."""
    return max(
        abs(p[2] - 0.5), max(max(0.0, abs(p[0]) - 1.6), max(0.0, abs(p[1]) - 1.6))
    )


def distance_slanted(p):
    """Distance from the plane through the slanted cutter."""
    return abs(-2.0 * p[0] + p[1] + 10.0 * p[2] - 3.0) / math.sqrt(105.0)


@MINI_TEST("Intersection", "Line Line")
def test_intersection_line_line():
    from session_py import intersection
    from session_py import Line
    from session_py import Tolerance

    line0 = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    line1 = Line(0.5, -1.0, 0.0, 0.5, 1.0, 0.0)
    output = intersection.line_line(line0, line1, Tolerance.APPROXIMATION)

    MINI_CHECK(output is not None)
    MINI_CHECK(TOLERANCE.is_close(output[0], 0.5))
    MINI_CHECK(TOLERANCE.is_close(output[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(output[2], 0.0))


@MINI_TEST("Intersection", "Line Line Parallel")
def test_intersection_line_line_parallel():
    from session_py import intersection
    from session_py import Line
    from session_py import Tolerance

    line0 = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    line1 = Line(0.0, 1.0, 0.0, 1.0, 1.0, 0.0)
    output = intersection.line_line(line0, line1, Tolerance.APPROXIMATION)

    MINI_CHECK(output is None)


@MINI_TEST("Intersection", "Line Line Parameters")
def test_intersection_line_line_parameters():
    from session_py import intersection
    from session_py import Line
    from session_py import Tolerance

    line0 = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    line1 = Line(0.5, -1.0, 0.0, 0.5, 1.0, 0.0)
    result = intersection.line_line_parameters(line0, line1, Tolerance.APPROXIMATION)

    MINI_CHECK(result is not None)

    t0, t1 = result

    MINI_CHECK(TOLERANCE.is_close(t0, 0.5))
    MINI_CHECK(TOLERANCE.is_close(t1, 0.5))


@MINI_TEST("Intersection", "Line Line Parameters Endpoints")
def test_intersection_line_line_parameters_endpoints():
    from session_py import intersection
    from session_py import Line
    from session_py import Tolerance

    line0 = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    line1 = Line(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    result = intersection.line_line_parameters(line0, line1, Tolerance.APPROXIMATION)

    MINI_CHECK(result is not None)

    t0, t1 = result

    MINI_CHECK(TOLERANCE.is_close(t0, 0.0))
    MINI_CHECK(TOLERANCE.is_close(t1, 0.0))


@MINI_TEST("Intersection", "Line Line Parameters Infinite")
def test_intersection_line_line_parameters_infinite():
    from session_py import intersection
    from session_py import Line
    from session_py import Tolerance

    line0 = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    line1 = Line(2.0, -1.0, 0.0, 2.0, 1.0, 0.0)
    result = intersection.line_line_parameters(
        line0, line1, Tolerance.APPROXIMATION, False
    )

    MINI_CHECK(result is not None)

    t0, t1 = result

    MINI_CHECK(TOLERANCE.is_close(t0, 2.0))


@MINI_TEST("Intersection", "Plane Plane")
def test_intersection_plane_plane():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    p0 = Point(0.0, 0.0, 0.0)
    n0 = Vector(0.0, 0.0, 1.0)
    plane0 = Plane.from_point_normal(p0, n0)

    p1 = Point(0.0, 0.0, 0.0)
    n1 = Vector(0.0, 1.0, 0.0)
    plane1 = Plane.from_point_normal(p1, n1)

    output = intersection.plane_plane(plane0, plane1)

    MINI_CHECK(output is not None)

    line_dir = output.to_vector()

    MINI_CHECK(abs(abs(line_dir[0]) - 1.0) < 1e-4)
    MINI_CHECK(abs(line_dir[1]) < 1e-4)
    MINI_CHECK(abs(line_dir[2]) < 1e-4)


@MINI_TEST("Intersection", "Plane Plane Complex")
def test_intersection_plane_plane_complex():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    plane_origin_0 = Point(213.787107, 513.797811, -24.743845)
    plane_xaxis_0 = Vector(0.907673, -0.258819, 0.330366)
    plane_yaxis_0 = Vector(0.272094, 0.96225, 0.006285)
    pl0 = Plane(plane_origin_0, plane_xaxis_0, plane_yaxis_0)

    plane_origin_1 = Point(247.17924, 499.115486, 59.619568)
    plane_xaxis_1 = Vector(0.552465, 0.816035, 0.16991)
    plane_yaxis_1 = Vector(0.172987, 0.087156, -0.98106)
    pl1 = Plane(plane_origin_1, plane_xaxis_1, plane_yaxis_1)

    intersection_line = intersection.plane_plane(pl0, pl1)

    MINI_CHECK(intersection_line is not None)

    start = intersection_line.start()
    end = intersection_line.end()

    MINI_CHECK(abs(start[0] - 252.4632) < 0.01)
    MINI_CHECK(abs(start[1] - 495.32248) < 0.01)
    MINI_CHECK(abs(start[2] - (-10.002656)) < 0.01)

    MINI_CHECK(abs(end[0] - 253.01033) < 0.01)
    MINI_CHECK(abs(end[1] - 496.1218) < 0.01)
    MINI_CHECK(abs(end[2] - (-9.888727)) < 0.01)


@MINI_TEST("Intersection", "Plane Plane To Line Canonical")
def test_intersection_plane_plane_to_line_canonical():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    p0 = Point(0.0, 0.0, 2.0)
    n0 = Vector(0.0, 0.0, 1.0)
    plane0 = Plane.from_point_normal(p0, n0)

    p1 = Point(3.0, 0.0, 0.0)
    n1 = Vector(1.0, 0.0, 0.0)
    plane1 = Plane.from_point_normal(p1, n1)

    output = intersection.plane_plane_to_line_canonical(plane0, plane1)

    MINI_CHECK(output is not None)
    MINI_CHECK(TOLERANCE.is_close(output.start()[0], 3.0))
    MINI_CHECK(TOLERANCE.is_close(output.start()[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(output.start()[2], 2.0))
    MINI_CHECK(TOLERANCE.is_close(output.end()[1], -1.0))

    p2 = Point(0.0, 0.0, 5.0)
    plane2 = Plane.from_point_normal(p2, n0)

    MINI_CHECK(intersection.plane_plane_to_line_canonical(plane0, plane2) is None)


@MINI_TEST("Intersection", "Line Plane")
def test_intersection_line_plane():
    from session_py import intersection
    from session_py import Line
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    p = Point(0.0, 0.0, 1.0)
    n = Vector(0.0, 0.0, 1.0)
    plane = Plane.from_point_normal(p, n)

    line = Line(0.0, 0.0, 0.0, 0.0, 0.0, 2.0)

    output = intersection.line_plane(line, plane, True)

    MINI_CHECK(output is not None)
    MINI_CHECK(TOLERANCE.is_close(output[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(output[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(output[2], 1.0))


@MINI_TEST("Intersection", "Line Plane Parallel")
def test_intersection_line_plane_parallel():
    from session_py import intersection
    from session_py import Line
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    p = Point(0.0, 0.0, 1.0)
    n = Vector(0.0, 0.0, 1.0)
    plane = Plane.from_point_normal(p, n)

    line = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    output = intersection.line_plane(line, plane, True)

    MINI_CHECK(output is None)


@MINI_TEST("Intersection", "Line Plane Real World")
def test_intersection_line_plane_real_world():
    from session_py import intersection
    from session_py import Line
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    l0 = Line(500.000, -573.576, -819.152, 500.000, 573.576, 819.152)

    plane_origin_0 = Point(213.787107, 513.797811, -24.743845)
    plane_xaxis_0 = Vector(0.907673, -0.258819, 0.330366)
    plane_yaxis_0 = Vector(0.272094, 0.96225, 0.006285)
    pl0 = Plane(plane_origin_0, plane_xaxis_0, plane_yaxis_0)

    lp = intersection.line_plane(l0, pl0)

    MINI_CHECK(lp is not None)
    MINI_CHECK(abs(lp[0] - 500.0) < 0.1)
    MINI_CHECK(abs(lp[1] - 77.7531) < 0.01)
    MINI_CHECK(abs(lp[2] - 111.043) < 0.01)


@MINI_TEST("Intersection", "Plane Plane Plane")
def test_intersection_plane_plane_plane():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    plane_origin_0 = Point(213.787107, 513.797811, -24.743845)
    plane_xaxis_0 = Vector(0.907673, -0.258819, 0.330366)
    plane_yaxis_0 = Vector(0.272094, 0.96225, 0.006285)
    pl0 = Plane(plane_origin_0, plane_xaxis_0, plane_yaxis_0)

    plane_origin_1 = Point(247.17924, 499.115486, 59.619568)
    plane_xaxis_1 = Vector(0.552465, 0.816035, 0.16991)
    plane_yaxis_1 = Vector(0.172987, 0.087156, -0.98106)
    pl1 = Plane(plane_origin_1, plane_xaxis_1, plane_yaxis_1)

    plane_origin_2 = Point(221.399816, 605.893667, -54.000116)
    plane_xaxis_2 = Vector(0.903451, -0.360516, -0.231957)
    plane_yaxis_2 = Vector(0.172742, -0.189057, 0.966653)
    pl2 = Plane(plane_origin_2, plane_xaxis_2, plane_yaxis_2)

    output = intersection.plane_plane_plane(pl0, pl1, pl2)

    MINI_CHECK(output is not None)
    MINI_CHECK(abs(output[0] - 300.5) < 0.1)
    MINI_CHECK(abs(output[1] - 565.5) < 0.1)
    MINI_CHECK(abs(output[2] - 0.0) < 0.1)


@MINI_TEST("Intersection", "Plane Plane Plane Parallel")
def test_intersection_plane_plane_plane_parallel():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    p0 = Point(0.0, 0.0, 0.0)
    n0 = Vector(0.0, 0.0, 1.0)
    plane0 = Plane.from_point_normal(p0, n0)

    p1 = Point(0.0, 0.0, 1.0)
    n1 = Vector(0.0, 0.0, 1.0)
    plane1 = Plane.from_point_normal(p1, n1)

    p2 = Point(0.0, 0.0, 0.0)
    n2 = Vector(1.0, 0.0, 0.0)
    plane2 = Plane.from_point_normal(p2, n2)

    output = intersection.plane_plane_plane(plane0, plane1, plane2)

    MINI_CHECK(output is None)


@MINI_TEST("Intersection", "Ray Box")
def test_intersection_ray_box():
    from session_py import intersection
    from session_py import OBB
    from session_py import Point
    from session_py import Vector

    center = Point(0.0, 0.0, 0.0)
    x_axis = Vector(1.0, 0.0, 0.0)
    y_axis = Vector(0.0, 1.0, 0.0)
    z_axis = Vector(0.0, 0.0, 1.0)
    half_size = Vector(1.0, 1.0, 1.0)
    box = OBB(center, x_axis, y_axis, z_axis, half_size)

    origin = Point(-5.0, 0.0, 0.0)
    direction = Vector(1.0, 0.0, 0.0)

    result, tmin, tmax = intersection.ray_box_parameters(
        origin, direction, box, 0.0, 100.0
    )

    MINI_CHECK(result)
    MINI_CHECK(abs(tmin - 4.0) < 1e-4)
    MINI_CHECK(abs(tmax - 6.0) < 1e-4)


@MINI_TEST("Intersection", "Ray Box Miss")
def test_intersection_ray_box_miss():
    from session_py import intersection
    from session_py import OBB
    from session_py import Point
    from session_py import Vector

    center = Point(0.0, 0.0, 0.0)
    x_axis = Vector(1.0, 0.0, 0.0)
    y_axis = Vector(0.0, 1.0, 0.0)
    z_axis = Vector(0.0, 0.0, 1.0)
    half_size = Vector(1.0, 1.0, 1.0)
    box = OBB(center, x_axis, y_axis, z_axis, half_size)

    origin = Point(-5.0, 5.0, 0.0)
    direction = Vector(1.0, 0.0, 0.0)

    result, _tmin, _tmax = intersection.ray_box_parameters(
        origin, direction, box, 0.0, 100.0
    )

    MINI_CHECK(not result)


@MINI_TEST("Intersection", "Ray Sphere")
def test_intersection_ray_sphere():
    from session_py import intersection
    from session_py import Point
    from session_py import Vector

    origin = Point(-5.0, 0.0, 0.0)
    direction = Vector(1.0, 0.0, 0.0)
    center = Point(0.0, 0.0, 0.0)
    radius = 2.0

    hits, t0, t1 = intersection.ray_sphere_parameters(origin, direction, center, radius)

    MINI_CHECK(hits == 2)
    MINI_CHECK(abs(t0 - 3.0) < 1e-4)
    MINI_CHECK(abs(t1 - 7.0) < 1e-4)


@MINI_TEST("Intersection", "Ray Sphere Tangent")
def test_intersection_ray_sphere_tangent():
    from session_py import intersection
    from session_py import Point
    from session_py import Vector

    origin = Point(-5.0, 2.0, 0.0)
    direction = Vector(1.0, 0.0, 0.0)
    center = Point(0.0, 0.0, 0.0)
    radius = 2.0

    hits, t0, _t1 = intersection.ray_sphere_parameters(
        origin, direction, center, radius
    )

    MINI_CHECK(hits == 1)
    MINI_CHECK(abs(t0 - 5.0) < 1e-4)


@MINI_TEST("Intersection", "Ray Sphere Miss")
def test_intersection_ray_sphere_miss():
    from session_py import intersection
    from session_py import Point
    from session_py import Vector

    origin = Point(-5.0, 5.0, 0.0)
    direction = Vector(1.0, 0.0, 0.0)
    center = Point(0.0, 0.0, 0.0)
    radius = 2.0

    hits, _t0, _t1 = intersection.ray_sphere_parameters(
        origin, direction, center, radius
    )

    MINI_CHECK(hits == 0)


@MINI_TEST("Intersection", "Ray Triangle")
def test_intersection_ray_triangle():
    from session_py import intersection
    from session_py import Point
    from session_py import Vector

    origin = Point(0.5, 0.5, -1.0)
    direction = Vector(0.0, 0.0, 1.0)

    v0 = Point(0.0, 0.0, 0.0)
    v1 = Point(1.0, 0.0, 0.0)
    v2 = Point(0.0, 1.0, 0.0)

    result, t, _u, _v, parallel = intersection.ray_triangle_parameters(
        origin, direction, v0, v1, v2, 1e-6
    )

    MINI_CHECK(result)
    MINI_CHECK(not parallel)
    MINI_CHECK(abs(t - 1.0) < 1e-4)


@MINI_TEST("Intersection", "Ray Triangle Miss")
def test_intersection_ray_triangle_miss():
    from session_py import intersection
    from session_py import Point
    from session_py import Vector

    origin = Point(2.0, 2.0, -1.0)
    direction = Vector(0.0, 0.0, 1.0)

    v0 = Point(0.0, 0.0, 0.0)
    v1 = Point(1.0, 0.0, 0.0)
    v2 = Point(0.0, 1.0, 0.0)

    result, _t, _u, _v, _parallel = intersection.ray_triangle_parameters(
        origin, direction, v0, v1, v2, 1e-6
    )

    MINI_CHECK(not result)


@MINI_TEST("Intersection", "Ray Triangle Parallel")
def test_intersection_ray_triangle_parallel():
    from session_py import intersection
    from session_py import Point
    from session_py import Vector

    origin = Point(0.5, 0.5, -1.0)
    direction = Vector(1.0, 0.0, 0.0)

    v0 = Point(0.0, 0.0, 0.0)
    v1 = Point(1.0, 0.0, 0.0)
    v2 = Point(0.0, 1.0, 0.0)

    result, _t, _u, _v, parallel = intersection.ray_triangle_parameters(
        origin, direction, v0, v1, v2, 1e-6
    )

    MINI_CHECK(not result)
    MINI_CHECK(parallel)


@MINI_TEST("Intersection", "Ray Mesh")
def test_intersection_ray_mesh():
    from session_py import intersection
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    polygons = [
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0.0),
        ],
        [
            Point(0.0, 0.0, 1.0),
            Point(1.0, 0.0, 1.0),
            Point(1.0, 1.0, 1.0),
            Point(0.0, 1.0, 1.0),
        ],
    ]

    mesh = Mesh.from_polylines(polygons)

    origin = Point(0.5, 0.5, -1.0)
    direction = Vector(0.0, 0.0, 1.0)

    result, hits = intersection.ray_mesh_hits(origin, direction, mesh, True)

    MINI_CHECK(result)
    MINI_CHECK(len(hits) >= 1)
    MINI_CHECK(abs(hits[0].t - 1.0) < 1e-3)


@MINI_TEST("Intersection", "Ray Mesh First")
def test_intersection_ray_mesh_first():
    from session_py import intersection
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    polygons = [
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0.0),
        ],
        [
            Point(0.0, 0.0, 1.0),
            Point(1.0, 0.0, 1.0),
            Point(1.0, 1.0, 1.0),
            Point(0.0, 1.0, 1.0),
        ],
    ]

    mesh = Mesh.from_polylines(polygons)

    origin = Point(0.5, 0.5, -1.0)
    direction = Vector(0.0, 0.0, 1.0)

    result, hits = intersection.ray_mesh_hits(origin, direction, mesh, False)

    MINI_CHECK(result)
    MINI_CHECK(len(hits) == 1)


@MINI_TEST("Intersection", "Ray Mesh Miss")
def test_intersection_ray_mesh_miss():
    from session_py import intersection
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    polygons = [
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0.0),
        ],
    ]

    mesh = Mesh.from_polylines(polygons)

    origin = Point(5.0, 5.0, -1.0)
    direction = Vector(0.0, 0.0, 1.0)

    result, hits = intersection.ray_mesh_hits(origin, direction, mesh, True)

    MINI_CHECK(not result)
    MINI_CHECK(len(hits) == 0)


@MINI_TEST("Intersection", "Ray Mesh Bvh")
def test_intersection_ray_mesh_bvh():
    from session_py import intersection
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    polygons = [
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0.0),
        ],
        [
            Point(0.0, 0.0, 1.0),
            Point(1.0, 0.0, 1.0),
            Point(1.0, 1.0, 1.0),
            Point(0.0, 1.0, 1.0),
        ],
    ]

    mesh = Mesh.from_polylines(polygons)

    origin = Point(0.5, 0.5, -1.0)
    direction = Vector(0.0, 0.0, 1.0)

    result, hits = intersection.ray_mesh_bvh_hits(origin, direction, mesh, True)

    MINI_CHECK(result)
    MINI_CHECK(len(hits) >= 1)
    MINI_CHECK(abs(hits[0].t - 1.0) < 1e-3)


@MINI_TEST("Intersection", "Ray Mesh Bvh First")
def test_intersection_ray_mesh_bvh_first():
    from session_py import intersection
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    polygons = [
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0.0),
        ],
        [
            Point(0.0, 0.0, 1.0),
            Point(1.0, 0.0, 1.0),
            Point(1.0, 1.0, 1.0),
            Point(0.0, 1.0, 1.0),
        ],
    ]

    mesh = Mesh.from_polylines(polygons)

    origin = Point(0.5, 0.5, -1.0)
    direction = Vector(0.0, 0.0, 1.0)

    result, hits = intersection.ray_mesh_bvh_hits(origin, direction, mesh, False)

    MINI_CHECK(result)
    MINI_CHECK(len(hits) == 1)


@MINI_TEST("Intersection", "Ray Mesh Bvh Miss")
def test_intersection_ray_mesh_bvh_miss():
    from session_py import intersection
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    polygons = [
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0.0),
        ],
    ]

    mesh = Mesh.from_polylines(polygons)

    origin = Point(5.0, 5.0, -1.0)
    direction = Vector(0.0, 0.0, 1.0)

    result, hits = intersection.ray_mesh_bvh_hits(origin, direction, mesh, True)

    MINI_CHECK(not result)
    MINI_CHECK(len(hits) == 0)


@MINI_TEST("Intersection", "Ray Mesh Bvh Vs Naive")
def test_intersection_ray_mesh_bvh_vs_naive():
    from session_py import intersection
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    polygons = []

    for i in range(10):
        for j in range(10):
            x = float(i)
            y = float(j)

            polygons.append(
                [
                    Point(x, y, 0.0),
                    Point(x + 1.0, y, 0.0),
                    Point(x + 1.0, y + 1.0, 0.0),
                    Point(x, y + 1.0, 0.0),
                ]
            )

    mesh = Mesh.from_polylines(polygons)

    origin = Point(5.5, 5.5, -1.0)
    direction = Vector(0.0, 0.0, 1.0)

    result_naive, hits_naive = intersection.ray_mesh_hits(origin, direction, mesh, True)

    result_bvh, hits_bvh = intersection.ray_mesh_bvh_hits(origin, direction, mesh, True)

    MINI_CHECK(result_naive == result_bvh)
    MINI_CHECK(len(hits_naive) == len(hits_bvh))

    if hits_naive:
        MINI_CHECK(abs(hits_naive[0].t - hits_bvh[0].t) < 1e-4)
        MINI_CHECK(hits_naive[0].face_index == hits_bvh[0].face_index)


@MINI_TEST("Intersection", "Ray Box Real World")
def test_intersection_ray_box_real_world():
    from session_py import intersection
    from session_py import OBB
    from session_py import Line
    from session_py import Point

    l0 = Line(500.0, -573.576, -819.152, 500.0, 573.576, 819.152)
    min_pt = Point(214.0, 192.0, 484.0)
    max_pt = Point(694.0, 567.0, 796.0)
    box = OBB.from_points([min_pt, max_pt])
    points = intersection.ray_box(l0, box, 0.0, 1000.0)

    MINI_CHECK(points is not None)
    MINI_CHECK(len(points) == 2)
    MINI_CHECK(abs(points[0][0] - 500.0) < 0.1)
    MINI_CHECK(abs(points[0][1] - 338.9) < 0.1)
    MINI_CHECK(abs(points[0][2] - 484.0) < 0.1)
    MINI_CHECK(abs(points[1][0] - 500.0) < 0.1)
    MINI_CHECK(abs(points[1][1] - 557.365) < 0.1)
    MINI_CHECK(abs(points[1][2] - 796.0) < 0.1)


@MINI_TEST("Intersection", "Ray Sphere Real World")
def test_intersection_ray_sphere_real_world():
    from session_py import intersection
    from session_py import Line
    from session_py import Point

    l0 = Line(500.0, -573.576, -819.152, 500.0, 573.576, 819.152)
    sphere_center = Point(457.0, 192.0, 207.0)
    radius = 265.0
    points = intersection.ray_sphere(l0, sphere_center, radius)

    MINI_CHECK(points is not None)
    MINI_CHECK(len(points) == 2)
    MINI_CHECK(abs(points[0][0] - 500.0) < 0.1)
    MINI_CHECK(abs(points[0][1] - 12.08) < 0.1)
    MINI_CHECK(abs(points[0][2] - 17.25) < 0.1)
    MINI_CHECK(abs(points[1][0] - 500.0) < 0.1)
    MINI_CHECK(abs(points[1][1] - 308.77) < 0.1)
    MINI_CHECK(abs(points[1][2] - 440.97) < 0.1)


@MINI_TEST("Intersection", "Ray Triangle Real World")
def test_intersection_ray_triangle_real_world():
    from session_py import intersection
    from session_py import Line
    from session_py import Point
    from session_py import Tolerance

    l0 = Line(500.0, -573.576, -819.152, 500.0, 573.576, 819.152)
    p1 = Point(214.0, 567.0, 484.0)
    p2 = Point(214.0, 192.0, 796.0)
    p3 = Point(694.0, 192.0, 484.0)
    result = intersection.ray_triangle(l0, p1, p2, p3, Tolerance.APPROXIMATION)

    MINI_CHECK(result is not None)
    MINI_CHECK(abs(result[0] - 500.0) < 0.1)
    MINI_CHECK(abs(result[1] - 340.616) < 0.01)
    MINI_CHECK(abs(result[2] - 486.451) < 0.01)


@MINI_TEST("Intersection", "Curve Plane")
def test_intersection_curve_plane():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Primitives
    from session_py import Vector

    circle = Primitives.circle(0.0, 0.0, 0.0, 2.0)
    origin = Point(1.0, 0.0, 0.0)
    normal = Vector(1.0, 0.0, 0.0)
    plane = Plane.from_point_normal(origin, normal)
    params = intersection.curve_plane(circle, plane)
    points = intersection.curve_plane_points(circle, plane)

    MINI_CHECK(len(params) == 2)
    MINI_CHECK(len(points) == 2)

    for p in points:
        MINI_CHECK(abs(p[0] - 1.0) < 1e-9)
        MINI_CHECK(abs(abs(p[1]) - math.sqrt(3.0)) < 1e-9)

    offset = 1.98 / math.sqrt(2.0)
    diagonal = Plane.from_point_normal(
        Point(offset, offset, 0.0), Vector(1.0, 1.0, 0.0)
    )
    hidden = intersection.curve_plane(circle, diagonal)

    MINI_CHECK(len(hidden) == 2)


@MINI_TEST("Intersection", "Curve Plane Bezier Clipping")
def test_intersection_curve_plane_bezier_clipping():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Primitives
    from session_py import Vector

    circle = Primitives.circle(0.0, 0.0, 0.0, 2.0)
    origin = Point(1.0, 0.0, 0.0)
    normal = Vector(1.0, 0.0, 0.0)
    plane = Plane.from_point_normal(origin, normal)
    params = intersection.curve_plane_bezier_clipping(circle, plane)

    MINI_CHECK(len(params) == 2)

    for t in params:
        MINI_CHECK(abs(circle.point_at(t)[0] - 1.0) < 1e-9)

    unit = Primitives.circle(0.0, 0.0, 0.0, 1.0)
    tilted = Plane.from_point_normal(Point(0.0, 0.0, 0.2), Vector(0.3, 0.1, 1.0))
    roots = intersection.curve_plane_bezier_clipping(unit, tilted)

    MINI_CHECK(len(roots) == 2)


@MINI_TEST("Intersection", "Curve Plane Algebraic")
def test_intersection_curve_plane_algebraic():
    from session_py import intersection
    from session_py import NurbsCurve
    from session_py import Plane
    from session_py import Point
    from session_py import Primitives
    from session_py import Vector

    curve = NurbsCurve.create(
        False,
        3,
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 2.0, 0.0),
            Point(2.0, 2.0, 0.0),
            Point(3.0, 0.0, 0.0),
        ],
    )

    origin = Point(1.0, 0.0, 0.0)
    normal = Vector(1.0, 0.0, 0.0)
    plane = Plane.from_point_normal(origin, normal)
    params = intersection.curve_plane_algebraic(curve, plane)

    MINI_CHECK(len(params) == 1)
    MINI_CHECK(abs(curve.point_at(params[0])[0] - 1.0) < 1e-9)
    MINI_CHECK(abs(curve.point_at(params[0])[1] - 4.0 / 3.0) < 1e-9)

    circle = Primitives.circle(0.0, 0.0, 0.0, 2.0)
    circle_params = intersection.curve_plane_algebraic(circle, plane)

    MINI_CHECK(len(circle_params) == 2)


@MINI_TEST("Intersection", "Curve Plane Production")
def test_intersection_curve_plane_production():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Primitives
    from session_py import Vector

    circle = Primitives.circle(0.0, 0.0, 0.0, 2.0)
    origin = Point(1.0, 0.0, 0.0)
    normal = Vector(1.0, 0.0, 0.0)
    plane = Plane.from_point_normal(origin, normal)
    params = intersection.curve_plane_production(circle, plane)

    MINI_CHECK(len(params) == 2)

    for t in params:
        MINI_CHECK(abs(circle.point_at(t)[0] - 1.0) < 1e-9)


@MINI_TEST("Intersection", "Curve Closest Point")
def test_intersection_curve_closest_point():
    from session_py import intersection
    from session_py import Point
    from session_py import Primitives

    circle = Primitives.circle(0.0, 0.0, 0.0, 2.0)
    test_point = Point(3.0, 0.0, 0.0)
    result = intersection.curve_closest_point(circle, test_point)
    closest = circle.point_at(result[0])

    MINI_CHECK(abs(result[1] - 1.0) < 1e-6)
    MINI_CHECK(abs(closest[0] - 2.0) < 1e-6)
    MINI_CHECK(abs(closest[1]) < 1e-6)


@MINI_TEST("Intersection", "Surface Plane")
def test_intersection_surface_plane():
    from session_py import intersection
    from session_py import NurbsSurface
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    pts = [
        Point(0.0, 0.0, 0.0),
        Point(0.0, 10.0, 0.0),
        Point(10.0, 0.0, 10.0),
        Point(10.0, 10.0, 10.0),
    ]
    srf = NurbsSurface.create(False, False, 1, 1, 2, 2, pts)
    plane = Plane.from_point_normal(Point(0.0, 0.0, 5.0), Vector(0.0, 0.0, 1.0))
    curves = intersection.surface_plane(srf, plane)

    MINI_CHECK(len(curves) == 1)
    MINI_CHECK(curves[0].is_valid())

    t0, t1 = curves[0].domain()

    for i in range(11):
        t = t0 + (t1 - t0) * i / 10.0
        p = curves[0].point_at(t)
        MINI_CHECK(abs(p[0] - 5.0) < 0.5)
        MINI_CHECK(abs(p[2] - 5.0) < 0.5)


@MINI_TEST("Intersection", "Surface Plane Curved")
def test_intersection_surface_plane_curved():
    from session_py import intersection
    from session_py import NurbsSurface
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    pts = []

    for i in range(4):
        for j in range(4):
            x = i * 10.0
            y = j * 10.0
            z = 10.0 if (i == 1 or i == 2) and (j == 1 or j == 2) else 0.0
            pts.append(Point(x, y, z))

    srf = NurbsSurface.create(False, False, 3, 3, 4, 4, pts)
    plane = Plane.from_point_normal(Point(0.0, 0.0, 3.0), Vector(0.0, 0.0, 1.0))
    curves = intersection.surface_plane(srf, plane)

    MINI_CHECK(len(curves) >= 1)
    MINI_CHECK(curves[0].is_valid())
    MINI_CHECK(curves[0].degree() == 3)

    t0, t1 = curves[0].domain()

    for i in range(11):
        t = t0 + (t1 - t0) * i / 10.0
        p = curves[0].point_at(t)
        MINI_CHECK(abs(p[2] - 3.0) < 1.0)


@MINI_TEST("Intersection", "Surface Plane Miss")
def test_intersection_surface_plane_miss():
    from session_py import intersection
    from session_py import NurbsSurface
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    pts = [
        Point(0.0, 0.0, 0.0),
        Point(0.0, 10.0, 0.0),
        Point(10.0, 0.0, 0.0),
        Point(10.0, 10.0, 0.0),
    ]
    srf = NurbsSurface.create(False, False, 1, 1, 2, 2, pts)
    plane = Plane.from_point_normal(Point(0.0, 0.0, 5.0), Vector(0.0, 0.0, 1.0))
    curves = intersection.surface_plane(srf, plane)

    MINI_CHECK(len(curves) == 0)


@MINI_TEST("Intersection", "Surface Plane UV")
def test_intersection_surface_plane_uv():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Vector
    from session_py import Primitives

    cyl = Primitives.cylinder_surface(0.0, 0.0, 0.0, 1.0, 4.0)
    plane = Plane.from_point_normal(Point(0.0, 0.0, 2.0), Vector(0.3, 0.0, 1.0))
    pairs = intersection.surface_plane_uv(cyl, plane)

    MINI_CHECK(len(pairs) == 1)

    curve3 = pairs[0][0]
    pcurve = pairs[0][1]

    MINI_CHECK(curve3.is_valid())
    MINI_CHECK(pcurve.is_valid())
    MINI_CHECK(curve3.is_closed())

    u0, u1 = cyl.domain(0)

    MINI_CHECK(
        abs(pcurve.point_at(0.0)[0] - u1) < 1e-9
        or abs(pcurve.point_at(0.0)[0] - u0) < 1e-9
    )
    MINI_CHECK(
        abs(pcurve.point_at(1.0)[0] - u1) < 1e-9
        or abs(pcurve.point_at(1.0)[0] - u0) < 1e-9
    )

    pn = plane.z_axis
    po = plane.origin
    max_off = 0.0

    for i in range(17):
        p2 = pcurve.point_at(i / 16.0)
        s = cyl.point_at(p2[0], p2[1])
        off = abs(
            (s[0] - po[0]) * pn[0] + (s[1] - po[1]) * pn[1] + (s[2] - po[2]) * pn[2]
        )
        max_off = max(max_off, off)

    MINI_CHECK(max_off < 0.05)

    torus = Primitives.torus_surface(0.0, 0.0, 0.0, 2.0, 0.5)
    plane2 = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0))
    pairs2 = intersection.surface_plane_uv(torus, plane2)

    MINI_CHECK(len(pairs2) == 2)

    tu0, tu1 = torus.domain(0)
    tv0, tv1 = torus.domain(1)
    inside = True

    for pair in pairs2:
        for i in range(17):
            p2 = pair[1].point_at(i / 16.0)

            if (
                p2[0] < tu0 - 1e-6
                or p2[0] > tu1 + 1e-6
                or p2[1] < tv0 - 1e-6
                or p2[1] > tv1 + 1e-6
            ):
                inside = False

    MINI_CHECK(inside)


@MINI_TEST("Intersection", "Surface Surface")
def test_intersection_surface_surface():
    from session_py import intersection
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Primitives

    flat = NurbsSurface.create(
        False,
        False,
        1,
        1,
        2,
        2,
        [
            Point(-3.0, -3.0, 0.5),
            Point(-3.0, 3.0, 0.5),
            Point(3.0, -3.0, 0.5),
            Point(3.0, 3.0, 0.5),
        ],
    )
    cyl = Primitives.cylinder_surface(0.0, 0.0, -2.0, 1.0, 4.0)
    flat_triples = intersection.surface_surface(flat, cyl)

    MINI_CHECK(len(flat_triples) == 1)

    c3, pa, pb = flat_triples[0]

    MINI_CHECK(c3.is_valid() and pa.is_valid() and pb.is_valid())
    MINI_CHECK(c3.is_closed())
    MINI_CHECK(lies_on_curve(c3, pa, flat) < 0.05)
    MINI_CHECK(lies_on_curve(c3, pb, cyl) < 0.05)

    sphere = Primitives.sphere_surface(0.0, 0.0, 0.0, 2.0)
    cyl2 = Primitives.cylinder_surface(1.3, 0.0, -3.0, 0.3, 6.0)
    triples = intersection.surface_surface(sphere, cyl2)

    MINI_CHECK(len(triples) >= 2)

    clean = 0

    for c3, pa, pb in triples:
        MINI_CHECK(c3.is_valid() and pa.is_valid() and pb.is_valid())

        if lies_on_curve(c3, pa, sphere) < 0.05 and lies_on_curve(c3, pb, cyl2) < 0.05:
            clean += 1

    MINI_CHECK(clean >= 2)

    sphere2 = Primitives.sphere_surface(0.0, 0.0, 0.0, 2.0)

    flat04 = NurbsSurface.create(
        False,
        False,
        1,
        1,
        2,
        2,
        [
            Point(-3.0, -3.0, 0.4),
            Point(-3.0, 3.0, 0.4),
            Point(3.0, -3.0, 0.4),
            Point(3.0, 3.0, 0.4),
        ],
    )

    ex_triples = intersection.surface_surface(sphere2, flat04)

    MINI_CHECK(len(ex_triples) == 1)

    ex_c3 = ex_triples[0][0]
    expected_r = math.sqrt(3.84)
    max_dev = 0.0

    for j in range(257):
        p = ex_c3.point_at(j / 256.0)
        rr = math.sqrt(p[0] * p[0] + p[1] * p[1])
        max_dev = max(max_dev, abs(rr - expected_r))
        max_dev = max(max_dev, abs(p[2] - 0.4))

    MINI_CHECK(max_dev < 1e-9)


@MINI_TEST("Intersection", "Surface Surface Accuracy")
def test_intersection_surface_surface_accuracy():
    from session_py import intersection
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Primitives

    sphere = Primitives.sphere_surface(0.0, 0.0, 0.0, 2.0)
    cyl = Primitives.cylinder_surface(1.3, 0.0, -3.0, 0.3, 6.0)
    tr = intersection.surface_surface(sphere, cyl)

    MINI_CHECK(len(tr) >= 2)

    for c3, pa, pb in tr:
        MINI_CHECK(on_both(c3, distance_sphere, distance_cylinder) < 1e-5)

    sphere2 = Primitives.sphere_surface(2.0, 0.0, 0.0, 2.0)
    tr2 = intersection.surface_surface(sphere, sphere2)

    MINI_CHECK(len(tr2) >= 1)

    for c3, pa, pb in tr2:
        MINI_CHECK(on_both(c3, distance_sphere, distance_sphere2) < 1e-6)

    torus = Primitives.torus_surface(0.0, 0.0, 0.0, 2.0, 0.5)
    flat = NurbsSurface.create(
        False,
        False,
        1,
        1,
        2,
        2,
        [
            Point(-9.0, -9.0, 0.0),
            Point(-9.0, 9.0, 0.0),
            Point(9.0, -9.0, 0.0),
            Point(9.0, 9.0, 0.0),
        ],
    )
    tr3 = intersection.surface_surface(torus, flat)

    MINI_CHECK(len(tr3) == 2)

    for c3, pa, pb in tr3:
        MINI_CHECK(on_both(c3, distance_torus, distance_flat) < 1e-6)


@MINI_TEST("Intersection", "Surface Surface Planes")
def test_intersection_surface_surface_planes():
    from session_py import intersection
    from session_py import Point

    flat = bilinear(
        Point(-3.0, -3.0, 0.5),
        Point(-3.0, 3.0, 0.5),
        Point(3.0, -3.0, 0.5),
        Point(3.0, 3.0, 0.5),
    )
    wall = bilinear(
        Point(0.2, -3.0, -3.0),
        Point(0.2, -3.0, 3.0),
        Point(0.2, 3.0, -3.0),
        Point(0.2, 3.0, 3.0),
    )
    far = bilinear(
        Point(5.0, -3.0, -3.0),
        Point(5.0, -3.0, 3.0),
        Point(5.0, 3.0, -3.0),
        Point(5.0, 3.0, 3.0),
    )
    tr = intersection.surface_surface(flat, wall)

    MINI_CHECK(len(tr) == 1)

    c3 = tr[0][0]
    start = c3.point_at_start()
    end = c3.point_at_end()

    MINI_CHECK(
        TOLERANCE.is_close(start[0], 0.2)
        and TOLERANCE.is_close(start[1], -3.0)
        and TOLERANCE.is_close(start[2], 0.5)
    )
    MINI_CHECK(
        TOLERANCE.is_close(end[0], 0.2)
        and TOLERANCE.is_close(end[1], 3.0)
        and TOLERANCE.is_close(end[2], 0.5)
    )
    MINI_CHECK(lies_on_curve(c3, tr[0][1], flat) < 1e-9)
    MINI_CHECK(lies_on_curve(c3, tr[0][2], wall) < 1e-9)
    MINI_CHECK(len(intersection.surface_surface(flat, far)) == 0)


@MINI_TEST("Intersection", "Surface Surface Plane Cone")
def test_intersection_surface_surface_plane_cone():
    from session_py import intersection
    from session_py import Point
    from session_py import Primitives

    cone = Primitives.cone_surface(0.0, 0.0, 0.0, 1.5, 3.0)
    flat = bilinear(
        Point(-3.0, -3.0, 0.5),
        Point(-3.0, 3.0, 0.5),
        Point(3.0, -3.0, 0.5),
        Point(3.0, 3.0, 0.5),
    )
    steep = bilinear(
        Point(1.1, -3.0, -3.0),
        Point(1.1, 3.0, -3.0),
        Point(-0.3, -3.0, 4.0),
        Point(-0.3, 3.0, 4.0),
    )
    slant = bilinear(
        Point(-3.0, -3.0, 8.5),
        Point(-3.0, 3.0, 8.5),
        Point(3.0, -3.0, -3.5),
        Point(3.0, 3.0, -3.5),
    )
    axial = bilinear(
        Point(0.0, -3.0, -3.0),
        Point(0.0, -3.0, 4.0),
        Point(0.0, 3.0, -3.0),
        Point(0.0, 3.0, 4.0),
    )
    circle = intersection.surface_surface(cone, flat)

    MINI_CHECK(len(circle) == 1)
    MINI_CHECK(on_both(circle[0][0], distance_cone, distance_flat_half) < 1e-9)
    MINI_CHECK(lies_on_curve(circle[0][0], circle[0][1], cone) < 1e-9)

    hyperbola = intersection.surface_surface(cone, steep)

    MINI_CHECK(len(hyperbola) == 1)
    MINI_CHECK(hyperbola[0][0].degree() == 2)
    MINI_CHECK(on_both(hyperbola[0][0], distance_cone, distance_cone) < 1e-9)

    parabola = intersection.surface_surface(cone, slant)

    MINI_CHECK(len(parabola) == 1)
    MINI_CHECK(on_both(parabola[0][0], distance_cone, distance_cone) < 1e-6)

    lines = intersection.surface_surface(cone, axial)

    MINI_CHECK(len(lines) == 2)

    for line in lines:
        apex = line[0].point_at_start()

        MINI_CHECK(
            TOLERANCE.is_close(apex[0], 0.0)
            and TOLERANCE.is_close(apex[1], 0.0)
            and TOLERANCE.is_close(apex[2], 3.0)
        )
        MINI_CHECK(on_both(line[0], distance_cone, distance_cone) < 1e-9)


@MINI_TEST("Intersection", "Surface Surface Plane Torus")
def test_intersection_surface_surface_plane_torus():
    from session_py import intersection
    from session_py import Point
    from session_py import Primitives

    torus = Primitives.torus_surface(0.0, 0.0, 0.0, 2.0, 0.5)
    wall = bilinear(
        Point(0.2, -3.0, -3.0),
        Point(0.2, -3.0, 3.0),
        Point(0.2, 3.0, -3.0),
        Point(0.2, 3.0, 3.0),
    )
    tr = intersection.surface_surface(torus, wall)

    MINI_CHECK(len(tr) == 2)

    for t in tr:
        MINI_CHECK(t[0].is_closed())
        MINI_CHECK(on_both(t[0], distance_torus, distance_wall) < 1e-4)
        MINI_CHECK(lies_on_curve(t[0], t[2], wall) < 1e-4)


@MINI_TEST("Intersection", "Surface Surface Cylinders")
def test_intersection_surface_surface_cylinders():
    from session_py import intersection
    from session_py import Primitives
    from session_py import Tolerance
    from session_py import Vector
    from session_py import Xform

    cyl = Primitives.cylinder_surface(0.0, 0.0, -2.0, 1.0, 4.0)
    beside = Primitives.cylinder_surface(1.5, 0.0, -2.0, 1.0, 4.0)
    across = Primitives.cylinder_surface(0.0, 0.0, -2.0, 1.0, 4.0).transformed(
        Xform.rotation(Vector(0.0, 1.0, 0.0), Tolerance.HALF_PI)
    )
    lines = intersection.surface_surface(cyl, beside)

    MINI_CHECK(len(lines) == 2)

    for line in lines:
        start = line[0].point_at_start()
        end = line[0].point_at_end()

        MINI_CHECK(
            TOLERANCE.is_close(start[0], 0.75) and TOLERANCE.is_close(end[0], 0.75)
        )
        MINI_CHECK(
            TOLERANCE.is_close(abs(start[1]), math.sqrt(0.4375))
            and TOLERANCE.is_close(start[1], end[1])
        )
        MINI_CHECK(lies_on_curve(line[0], line[1], cyl) < 1e-9)

    ellipses = intersection.surface_surface(cyl, across)

    MINI_CHECK(len(ellipses) == 2)

    for ellipse in ellipses:
        MINI_CHECK(ellipse[0].is_closed())
        MINI_CHECK(
            on_both(ellipse[0], distance_unit_cylinder, distance_x_cylinder) < 1e-9
        )


@MINI_TEST("Intersection", "Surface Surface Coaxial Quadrics")
def test_intersection_surface_surface_coaxial_quadrics():
    from session_py import intersection
    from session_py import Primitives

    sphere = Primitives.sphere_surface(0.0, 0.0, 0.0, 2.0)
    cyl = Primitives.cylinder_surface(0.0, 0.0, -2.0, 1.0, 4.0)
    cone = Primitives.cone_surface(0.0, 0.0, 0.0, 1.5, 3.0)
    sphere_cyl = intersection.surface_surface(sphere, cyl)

    MINI_CHECK(len(sphere_cyl) == 2)

    for t in sphere_cyl:
        MINI_CHECK(TOLERANCE.is_close(abs(t[0].point_at_start()[2]), math.sqrt(3.0)))
        MINI_CHECK(on_both(t[0], distance_sphere, distance_unit_cylinder) < 1e-9)
        MINI_CHECK(lies_on_curve(t[0], t[1], sphere) < 1e-9)
        MINI_CHECK(lies_on_curve(t[0], t[2], cyl) < 1e-9)

    cyl_cone = intersection.surface_surface(cyl, cone)

    MINI_CHECK(len(cyl_cone) == 1)
    MINI_CHECK(TOLERANCE.is_close(cyl_cone[0][0].point_at_start()[2], 1.0))
    MINI_CHECK(on_both(cyl_cone[0][0], distance_unit_cylinder, distance_cone) < 1e-9)
    MINI_CHECK(lies_on_curve(cyl_cone[0][0], cyl_cone[0][2], cone) < 1e-9)

    cone_sphere = intersection.surface_surface(cone, sphere)

    MINI_CHECK(len(cone_sphere) == 2)

    for t in cone_sphere:
        MINI_CHECK(on_both(t[0], distance_cone, distance_sphere) < 1e-9)


@MINI_TEST("Intersection", "Surface Surface Coaxial Tori")
def test_intersection_surface_surface_coaxial_tori():
    from session_py import intersection
    from session_py import Primitives

    torus = Primitives.torus_surface(0.0, 0.0, 0.0, 2.0, 0.5)
    wide_cyl = Primitives.cylinder_surface(0.0, 0.0, -2.0, 2.2, 4.0)
    cone = Primitives.cone_surface(0.0, 0.0, 0.0, 1.5, 3.0)
    high_torus = Primitives.torus_surface(0.0, 0.0, 1.0, 1.0, 0.3)
    sphere = Primitives.sphere_surface(0.0, 0.0, 0.0, 2.0)
    wide_torus = Primitives.torus_surface(0.0, 0.0, 0.3, 2.3, 0.5)
    cyl_torus = intersection.surface_surface(wide_cyl, torus)

    MINI_CHECK(len(cyl_torus) == 2)

    for t in cyl_torus:
        MINI_CHECK(TOLERANCE.is_close(abs(t[0].point_at_start()[2]), math.sqrt(0.21)))
        MINI_CHECK(on_both(t[0], distance_wide_cylinder, distance_torus) < 1e-9)
        MINI_CHECK(lies_on_curve(t[0], t[2], torus) < 1e-9)

    cone_torus = intersection.surface_surface(cone, high_torus)

    MINI_CHECK(len(cone_torus) == 2)

    for t in cone_torus:
        MINI_CHECK(on_both(t[0], distance_cone, distance_high_torus) < 1e-9)
        MINI_CHECK(lies_on_curve(t[0], t[2], high_torus) < 1e-9)

    sphere_torus = intersection.surface_surface(sphere, torus)

    MINI_CHECK(len(sphere_torus) == 2)

    for t in sphere_torus:
        MINI_CHECK(on_both(t[0], distance_sphere, distance_torus) < 1e-9)

    torus_torus = intersection.surface_surface(torus, wide_torus)

    MINI_CHECK(len(torus_torus) == 2)

    for t in torus_torus:
        MINI_CHECK(on_both(t[0], distance_torus, distance_wide_torus) < 1e-9)
        MINI_CHECK(lies_on_curve(t[0], t[1], torus) < 1e-9)


@MINI_TEST("Intersection", "Cut Curves On Surface")
def test_intersection_cut_curves_on_surface():
    from session_py import intersection
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import Primitives

    flat = NurbsSurface.create(
        False,
        False,
        1,
        1,
        2,
        2,
        [
            Point(-3.0, -3.0, 0.0),
            Point(-3.0, 3.0, 0.0),
            Point(3.0, -3.0, 0.0),
            Point(3.0, 3.0, 0.0),
        ],
    )
    cyl = Primitives.cylinder_surface(0.0, 0.0, -2.0, 1.0, 4.0)
    pcurves = intersection.cut_curves_on_surface(flat, cyl)

    MINI_CHECK(len(pcurves) == 1)
    MINI_CHECK(pcurves[0].is_valid())

    max_off = 0.0

    for i in range(17):
        uv = pcurves[0].point_at(i / 16.0)
        p = flat.point_at(uv[0], uv[1])
        max_off = max(max_off, abs(math.sqrt(p[0] * p[0] + p[1] * p[1]) - 1.0))

    MINI_CHECK(max_off < 1e-3)


@MINI_TEST("Intersection", "Cut Curves On Surface Pullbacks")
def test_intersection_cut_curves_on_surface_pullbacks():
    from session_py import intersection
    from session_py import Point
    from session_py import Primitives

    sphere = Primitives.sphere_surface(0.0, 0.0, 0.0, 2.0)
    cone = Primitives.cone_surface(0.0, 0.0, 0.0, 1.5, 3.0)
    wall = bilinear(
        Point(0.2, -3.0, -3.0),
        Point(0.2, -3.0, 3.0),
        Point(0.2, 3.0, -3.0),
        Point(0.2, 3.0, 3.0),
    )
    square = bilinear(
        Point(-1.6, -1.6, 0.5),
        Point(-1.6, 1.6, 0.5),
        Point(1.6, -1.6, 0.5),
        Point(1.6, 1.6, 0.5),
    )
    sphere_cuts = intersection.cut_curves_on_surface(sphere, wall)

    MINI_CHECK(len(sphere_cuts) == 3)

    for pc in sphere_cuts:
        MINI_CHECK(lifted_distance(pc, sphere, distance_wall) < 5e-3)

    cone_cuts = intersection.cut_curves_on_surface(cone, wall)

    MINI_CHECK(len(cone_cuts) == 2)

    for pc in cone_cuts:
        MINI_CHECK(lifted_distance(pc, cone, distance_wall) < 1e-3)

    square_cuts = intersection.cut_curves_on_surface(sphere, square)

    MINI_CHECK(len(square_cuts) == 4)

    for pc in square_cuts:
        MINI_CHECK(lifted_distance(pc, sphere, distance_square) < 2e-3)


@MINI_TEST("Intersection", "Cut Curves On Surface Torus")
def test_intersection_cut_curves_on_surface_torus():
    from session_py import intersection
    from session_py import Point
    from session_py import Primitives

    torus = Primitives.torus_surface(0.0, 0.0, 0.0, 2.0, 0.5)
    wall = bilinear(
        Point(0.2, -3.0, -3.0),
        Point(0.2, -3.0, 3.0),
        Point(0.2, 3.0, -3.0),
        Point(0.2, 3.0, 3.0),
    )
    cuts = intersection.cut_curves_on_surface(torus, wall)

    MINI_CHECK(len(cuts) == 4)

    for pc in cuts:
        MINI_CHECK(lifted_distance(pc, torus, distance_wall) < 1e-5)


@MINI_TEST("Intersection", "Cut Curves Slanted Cutter")
def test_intersection_cut_curves_slanted_cutter():
    from session_py import intersection
    from session_py import Point
    from session_py import Primitives

    cone = Primitives.cone_surface(0.0, 0.0, 0.0, 1.5, 3.0)
    slanted = bilinear(
        Point(-3.0, -3.0, 0.0),
        Point(-3.0, 3.0, -0.6),
        Point(3.0, -3.0, 1.2),
        Point(3.0, 3.0, 0.6),
    )
    cuts = intersection.cut_curves_on_surface(cone, slanted)

    MINI_CHECK(len(cuts) == 2)

    domain = cuts[0].domain()
    uv = cuts[0].point_at((domain[0] + domain[1]) * 0.5)
    p = cone.point_at(uv[0], uv[1])

    MINI_CHECK(distance_cone(p) < 1e-3)
    MINI_CHECK(distance_slanted(p) < 1e-3)


@MINI_TEST("Intersection", "Cut Curves Plane Trapezoid")
def test_intersection_cut_curves_plane_trapezoid():
    from session_py import intersection
    from session_py import Point

    trapezoid = bilinear(
        Point(-3.0, -3.0, 0.0),
        Point(-1.0, 3.0, 0.0),
        Point(3.0, -3.0, 0.0),
        Point(7.0, 3.0, 0.0),
    )
    wall = bilinear(
        Point(6.0, -5.0, -1.0),
        Point(6.0, 5.0, -1.0),
        Point(6.0, -5.0, 1.0),
        Point(6.0, 5.0, 1.0),
    )
    target_cuts = intersection.cut_curves_on_surface(trapezoid, wall)
    cutter_cuts = intersection.cut_curves_on_surface(wall, trapezoid)

    MINI_CHECK(len(target_cuts) == 1)
    MINI_CHECK(len(cutter_cuts) == 1)

    target_domain = target_cuts[0].domain()
    target_uv0 = target_cuts[0].point_at(target_domain[0])
    target_uv1 = target_cuts[0].point_at(target_domain[1])
    target_p0 = trapezoid.point_at(target_uv0[0], target_uv0[1])
    target_p1 = trapezoid.point_at(target_uv1[0], target_uv1[1])

    MINI_CHECK(abs(target_p0[0] - 6.0) < 1e-3)
    MINI_CHECK(abs(target_p1[0] - 6.0) < 1e-3)
    MINI_CHECK(abs(min(target_p0[1], target_p1[1]) - 1.5) < 1e-3)
    MINI_CHECK(abs(max(target_p0[1], target_p1[1]) - 3.0) < 1e-3)

    cutter_domain = cutter_cuts[0].domain()
    cutter_uv0 = cutter_cuts[0].point_at(cutter_domain[0])
    cutter_uv1 = cutter_cuts[0].point_at(cutter_domain[1])
    cutter_p0 = wall.point_at(cutter_uv0[0], cutter_uv0[1])
    cutter_p1 = wall.point_at(cutter_uv1[0], cutter_uv1[1])

    MINI_CHECK(abs(min(cutter_p0[1], cutter_p1[1]) - 1.5) < 1e-3)
    MINI_CHECK(abs(max(cutter_p0[1], cutter_p1[1]) - 3.0) < 1e-3)


@MINI_TEST("Intersection", "Remap")
def test_intersection_remap():
    from session_py import intersection

    MINI_CHECK(abs(intersection.remap(5.0, 0.0, 10.0, 0.0, 1.0) - 0.5) < 1e-9)
    MINI_CHECK(abs(intersection.remap(0.0, 0.0, 10.0, 0.0, 1.0) - 0.0) < 1e-9)
    MINI_CHECK(abs(intersection.remap(10.0, 0.0, 10.0, 0.0, 1.0) - 1.0) < 1e-9)


@MINI_TEST("Intersection", "Closest Point On Segment")
def test_intersection_closest_point_on_segment():
    from session_py import intersection
    from session_py import Line
    from session_py import Point

    seg = Line(0.0, 0.0, 0.0, 4.0, 0.0, 0.0)
    pt = Point(2.0, 3.0, 0.0)
    cp, t = intersection.closest_point_on_segment(pt, seg)

    MINI_CHECK(abs(cp[0] - 2.0) < 1e-9)
    MINI_CHECK(abs(cp[1] - 0.0) < 1e-9)
    MINI_CHECK(abs(t - 0.5) < 1e-9)

    pt2 = Point(-2.0, 1.0, 0.0)
    cp2, t2 = intersection.closest_point_on_segment(pt2, seg)

    MINI_CHECK(abs(cp2[0] - 0.0) < 1e-9)
    MINI_CHECK(abs(t2 - 0.0) < 1e-9)


@MINI_TEST("Intersection", "Plane Plane Plane Check Parallel")
def test_intersection_plane_plane_plane_check():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    p0 = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0))
    p1 = Plane.from_point_normal(Point(0.0, 0.0, 1.0), Vector(0.0, 0.0, 1.0))
    p2 = Plane.from_point_normal(Point(0.0, 0.0, 2.0), Vector(0.0, 0.0, 1.0))

    MINI_CHECK(intersection.plane_plane_plane_check(p0, p1, p2, 0.1) is None)

    px = Plane.from_point_normal(Point(1.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0))
    py = Plane.from_point_normal(Point(0.0, 2.0, 0.0), Vector(0.0, 1.0, 0.0))
    pz = Plane.from_point_normal(Point(0.0, 0.0, 3.0), Vector(0.0, 0.0, 1.0))
    pt = intersection.plane_plane_plane_check(px, py, pz, 0.1)

    MINI_CHECK(pt is not None)
    MINI_CHECK(abs(pt[0] - 1.0) < 1e-6)
    MINI_CHECK(abs(pt[1] - 2.0) < 1e-6)
    MINI_CHECK(abs(pt[2] - 3.0) < 1e-6)


@MINI_TEST("Intersection", "Plane 4 Planes Closed")
def test_intersection_plane_4planes():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    main = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0))
    planes = [
        Plane.from_point_normal(Point(-1.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0)),
        Plane.from_point_normal(Point(0.0, -1.0, 0.0), Vector(0.0, 1.0, 0.0)),
        Plane.from_point_normal(Point(1.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0)),
        Plane.from_point_normal(Point(0.0, 1.0, 0.0), Vector(0.0, 1.0, 0.0)),
    ]
    result = intersection.plane_4planes(main, planes)

    MINI_CHECK(result is not None)
    MINI_CHECK(result.point_count() == 5)

    for i in range(result.point_count()):
        p = result.get_point(i)
        MINI_CHECK(abs(p[2]) < 1e-6)

    first = result.get_point(0)
    last = result.get_point(4)

    MINI_CHECK(abs(first[0] - last[0]) < 1e-6)
    MINI_CHECK(abs(first[1] - last[1]) < 1e-6)


@MINI_TEST("Intersection", "Plane 4 Planes Open")
def test_intersection_plane_4planes_open():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    main = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0))
    planes = [
        Plane.from_point_normal(Point(-1.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0)),
        Plane.from_point_normal(Point(0.0, -1.0, 0.0), Vector(0.0, 1.0, 0.0)),
        Plane.from_point_normal(Point(1.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0)),
        Plane.from_point_normal(Point(0.0, 1.0, 0.0), Vector(0.0, 1.0, 0.0)),
    ]
    result = intersection.plane_4planes_open(main, planes)

    MINI_CHECK(result is not None)
    MINI_CHECK(result.point_count() == 4)


@MINI_TEST("Intersection", "Plane 4 Lines")
def test_intersection_plane_4lines():
    from session_py import intersection
    from session_py import Line
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    plane = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0))
    l0 = Line(-1.0, -1.0, -1.0, -1.0, 1.0, 1.0)
    l1 = Line(1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    l2 = Line(-1.0, -1.0, -1.0, 1.0, -1.0, 1.0)
    l3 = Line(-1.0, 1.0, -1.0, 1.0, 1.0, 1.0)
    result = intersection.plane_4lines(plane, l0, l1, l2, l3)

    MINI_CHECK(result is not None)
    MINI_CHECK(result.point_count() == 5)

    for i in range(result.point_count()):
        p = result.get_point(i)
        MINI_CHECK(abs(p[2]) < 1e-6)


@MINI_TEST("Intersection", "Line Two Planes")
def test_intersection_line_two_planes():
    from session_py import intersection
    from session_py import Line
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    line = Line(0.0, 0.0, -5.0, 0.0, 0.0, 5.0)
    o0 = Point(0.0, 0.0, -1.0)
    o1 = Point(0.0, 0.0, 2.0)
    n = Vector(0.0, 0.0, 1.0)
    plane0 = Plane.from_point_normal(o0, n)
    plane1 = Plane.from_point_normal(o1, n)
    output = intersection.line_two_planes(line, plane0, plane1)

    MINI_CHECK(output is not None)
    MINI_CHECK(TOLERANCE.is_close(output.start()[2], -1.0))
    MINI_CHECK(TOLERANCE.is_close(output.end()[2], 2.0))


@MINI_TEST("Intersection", "Scale Vector To Distance Of 2 Planes")
def test_intersection_scale_vector_to_distance_of_2planes():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    p0 = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0))
    p1 = Plane.from_point_normal(Point(0.0, 0.0, 3.0), Vector(0.0, 0.0, 1.0))
    direction = Vector(0.0, 0.0, 1.0)
    result = intersection.scale_vector_to_distance_of_2planes(direction, p0, p1)

    MINI_CHECK(result is not None)
    MINI_CHECK(abs(result[2] - 3.0) < 1e-6)


@MINI_TEST("Intersection", "Polyline Plane")
def test_intersection_polyline_plane():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    poly = Polyline(
        [
            Point(-1.0, -1.0, 0.0),
            Point(1.0, -1.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(-1.0, 1.0, 0.0),
            Point(-1.0, -1.0, 0.0),
        ]
    )
    plane = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0))
    result = intersection.polyline_plane(poly, plane)

    MINI_CHECK(result is not None)

    pts, indices = result

    MINI_CHECK(len(pts) == 2)

    for p in pts:
        MINI_CHECK(abs(p[0]) < 1e-9)


@MINI_TEST("Intersection", "Line Line 3D")
def test_intersection_line_line_3d():
    from session_py import intersection
    from session_py import Line

    cutter = Line(0.0, 1.0, 0.0, 2.0, 1.0, 0.0)
    seg = Line(1.0, 0.0, 0.0, 1.0, 2.0, 0.0)
    result = intersection.line_line_3d(cutter, seg)

    MINI_CHECK(result is not None)
    MINI_CHECK(abs(result[0] - 1.0) < 1e-6)
    MINI_CHECK(abs(result[1] - 1.0) < 1e-6)
    MINI_CHECK(abs(result[2] - 0.0) < 1e-6)

    par0 = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    par1 = Line(0.0, 1.0, 0.0, 1.0, 1.0, 0.0)

    MINI_CHECK(intersection.line_line_3d(par0, par1) is None)


@MINI_TEST("Intersection", "Polyline Boolean")
def test_intersection_polyline_boolean():
    from session_py import intersection
    from session_py import Point
    from session_py import Polyline

    a = Polyline(
        [
            Point(0.0, 0.0, 0.0),
            Point(2.0, 0.0, 0.0),
            Point(2.0, 2.0, 0.0),
            Point(0.0, 2.0, 0.0),
            Point(0.0, 0.0, 0.0),
        ]
    )
    b = Polyline(
        [
            Point(1.0, 1.0, 0.0),
            Point(3.0, 1.0, 0.0),
            Point(3.0, 3.0, 0.0),
            Point(1.0, 3.0, 0.0),
            Point(1.0, 1.0, 0.0),
        ]
    )
    intersected = intersection.polyline_boolean(a, b, 0)
    united = intersection.polyline_boolean(a, b, 1)
    difference = intersection.polyline_boolean(a, b, 2)

    MINI_CHECK(len(intersected) == 1)
    MINI_CHECK(len(united) == 1)
    MINI_CHECK(len(difference) == 1)

    for i in range(intersected[0].point_count()):
        p = intersected[0].get_point(i)

        MINI_CHECK(p[0] > 1.0 - 1e-9 and p[0] < 2.0 + 1e-9)
        MINI_CHECK(p[1] > 1.0 - 1e-9 and p[1] < 2.0 + 1e-9)


@MINI_TEST("Intersection", "Offset In 3D")
def test_intersection_offset_in_3d():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Polyline

    square = Polyline(
        [
            Point(0.0, 0.0, 0.0),
            Point(2.0, 0.0, 0.0),
            Point(2.0, 2.0, 0.0),
            Point(0.0, 2.0, 0.0),
            Point(0.0, 0.0, 0.0),
        ]
    )
    plane = Plane.xy_plane()
    ok = intersection.offset_in_3d(square, plane, 0.5)

    MINI_CHECK(ok)
    MINI_CHECK(TOLERANCE.is_close(square.get_point(0)[0], -0.5))
    MINI_CHECK(TOLERANCE.is_close(square.get_point(0)[1], -0.5))

    for i in range(square.point_count()):
        p = square.get_point(i)

        MINI_CHECK(TOLERANCE.is_close(abs(p[0] - 1.0), 1.5))
        MINI_CHECK(TOLERANCE.is_close(abs(p[1] - 1.0), 1.5))


@MINI_TEST("Intersection", "Polyline Boolean 2D In Plane")
def test_intersection_polyline_boolean_2d_in_plane():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Polyline

    a = Polyline(
        [
            Point(0.0, 0.0, 1.0),
            Point(2.0, 0.0, 1.0),
            Point(2.0, 2.0, 1.0),
            Point(0.0, 2.0, 1.0),
            Point(0.0, 0.0, 1.0),
        ]
    )
    b = Polyline(
        [
            Point(1.0, 1.0, 1.0),
            Point(3.0, 1.0, 1.0),
            Point(3.0, 3.0, 1.0),
            Point(1.0, 3.0, 1.0),
            Point(1.0, 1.0, 1.0),
        ]
    )
    plane = Plane.xy_plane()
    result = intersection.polyline_boolean_2d_in_plane(a, b, plane, 0)

    MINI_CHECK(result is not None)
    MINI_CHECK(result.point_count() >= 4)

    for i in range(result.point_count()):
        p = result.get_point(i)

        MINI_CHECK(p[0] > 1.0 - 1e-9 and p[0] < 2.0 + 1e-9)
        MINI_CHECK(p[1] > 1.0 - 1e-9 and p[1] < 2.0 + 1e-9)
        MINI_CHECK(TOLERANCE.is_close(p[2], 1.0))

    tiny = intersection.polyline_boolean_2d_in_plane(a, b, plane, 0, False, 2.0)

    MINI_CHECK(tiny is None)


@MINI_TEST("Intersection", "Polyline Plane To Line")
def test_intersection_polyline_plane_to_line():
    from session_py import Plane
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector
    from session_py.intersection import polyline_plane_to_line

    poly = Polyline(
        [
            Point(0.0, 0.0, 0.0),
            Point(4.0, 0.0, 0.0),
            Point(4.0, 4.0, 0.0),
            Point(0.0, 4.0, 0.0),
            Point(0.0, 0.0, 0.0),
        ]
    )
    pln = Plane.from_point_normal(Point(0.0, 2.0, 0.0), Vector(0.0, 1.0, 0.0))
    out = polyline_plane_to_line(poly, pln, Point(0.0, 0.0, 0.0))

    MINI_CHECK(out is not None)
    MINI_CHECK(TOLERANCE.is_close(out.start()[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(out.end()[0], 4.0))


@MINI_TEST("Intersection", "Quad From Line Top Bottom Planes")
def test_intersection_quad_from_line_top_bottom_planes():
    from session_py import Line
    from session_py import Plane
    from session_py import Point
    from session_py import Vector
    from session_py.intersection import quad_from_line_top_bottom_planes

    face = Plane.xy_plane()
    line = Line(0.0, 0.0, 0.0, 10.0, 0.0, 0.0)
    plane0 = Plane.from_point_normal(Point(0.0, -2.0, 0.0), Vector(0.0, 1.0, 0.0))
    plane1 = Plane.from_point_normal(Point(0.0, 2.0, 0.0), Vector(0.0, 1.0, 0.0))
    out = quad_from_line_top_bottom_planes(face, line, plane0, plane1)

    MINI_CHECK(out is not None)
    MINI_CHECK(out.point_count() == 5)
    MINI_CHECK(TOLERANCE.is_close(abs(out.get_point(0)[1]), 2.0))
    MINI_CHECK(TOLERANCE.is_close(abs(out.get_point(2)[1]), 2.0))
    MINI_CHECK(TOLERANCE.is_close(out.get_point(2)[0], 10.0))


@MINI_TEST("Intersection", "Orthogonal Vector Between Two Plane Pairs")
def test_intersection_orthogonal_vector_between_two_plane_pairs():
    from session_py import Plane
    from session_py import Point
    from session_py import Vector
    from session_py.intersection import orthogonal_vector_between_two_plane_pairs

    pp00 = Plane.xy_plane()
    pp10 = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0))
    pp11 = Plane.from_point_normal(Point(4.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0))
    out = orthogonal_vector_between_two_plane_pairs(pp00, pp10, pp11)

    MINI_CHECK(out is not None)

    mag = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]) ** 0.5

    MINI_CHECK(TOLERANCE.is_close(mag, 4.0))
    MINI_CHECK(TOLERANCE.is_close(out[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(out[2], 0.0))


@MINI_TEST("Intersection", "Closed And Open Paths 2D")
def test_intersection_closed_and_open_paths_2d():
    from session_py import Plane
    from session_py import Point
    from session_py import Polyline
    from session_py.intersection import closed_and_open_paths_2d

    plate = Polyline(
        [
            Point(0.0, 0.0, 0.0),
            Point(10.0, 0.0, 0.0),
            Point(10.0, 10.0, 0.0),
            Point(0.0, 10.0, 0.0),
            Point(0.0, 0.0, 0.0),
        ]
    )
    joint = Polyline(
        [
            Point(-2.0, 5.0, 0.0),
            Point(12.0, 5.0, 0.0),
        ]
    )
    pln = Plane.xy_plane()
    result = closed_and_open_paths_2d(plate, joint, pln)

    MINI_CHECK(result is not None)

    out, (t0, t1) = result

    MINI_CHECK(out.point_count() == 2)
    MINI_CHECK(TOLERANCE.is_close(out.get_point(0)[1], 5.0))
    MINI_CHECK(TOLERANCE.is_close(out.get_point(1)[1], 5.0))

    t_lo = min(t0, t1)
    t_hi = max(t0, t1)

    MINI_CHECK(TOLERANCE.is_close(t_lo, 1.5))
    MINI_CHECK(TOLERANCE.is_close(t_hi, 3.5))


@MINI_TEST("Intersection", "Face To Face")
def test_intersection_face_to_face():
    from session_py import intersection
    from session_py import Plane
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    polylines = [
        [
            Polyline(
                [
                    Point(0.0, 0.0, 0.0),
                    Point(2.0, 0.0, 0.0),
                    Point(2.0, 1.0, 0.0),
                    Point(0.0, 1.0, 0.0),
                    Point(0.0, 0.0, 0.0),
                ]
            ),
            Polyline(
                [
                    Point(0.0, 0.0, 1.0),
                    Point(2.0, 0.0, 1.0),
                    Point(2.0, 1.0, 1.0),
                    Point(0.0, 1.0, 1.0),
                    Point(0.0, 0.0, 1.0),
                ]
            ),
        ],
        [
            Polyline(
                [
                    Point(1.0, 0.5, 1.0),
                    Point(3.0, 0.5, 1.0),
                    Point(3.0, 1.5, 1.0),
                    Point(1.0, 1.5, 1.0),
                    Point(1.0, 0.5, 1.0),
                ]
            ),
            Polyline(
                [
                    Point(1.0, 0.5, 2.0),
                    Point(3.0, 0.5, 2.0),
                    Point(3.0, 1.5, 2.0),
                    Point(1.0, 1.5, 2.0),
                    Point(1.0, 0.5, 2.0),
                ]
            ),
        ],
    ]
    o00 = Point(1.0, 0.5, 0.0)
    o01 = Point(1.0, 0.5, 1.0)
    o10 = Point(2.0, 1.0, 1.0)
    o11 = Point(2.0, 1.0, 2.0)
    down = Vector(0.0, 0.0, -1.0)
    up = Vector(0.0, 0.0, 1.0)

    planes = [
        [
            Plane.from_point_normal(o00, down),
            Plane.from_point_normal(o01, up),
        ],
        [
            Plane.from_point_normal(o10, down),
            Plane.from_point_normal(o11, up),
        ],
    ]

    adjacency = [0, 1, -1, -1]
    contacts = intersection.face_to_face(adjacency, polylines, planes, 0.01)

    MINI_CHECK(len(contacts) == 1)
    MINI_CHECK(contacts[0][0] == 0)
    MINI_CHECK(contacts[0][1] == 1)
    MINI_CHECK(contacts[0][2] == 1)
    MINI_CHECK(contacts[0][3] == 0)
    MINI_CHECK(contacts[0][4] == 2)
    MINI_CHECK(contacts[0][5].is_closed())

    for i in range(contacts[0][5].point_count()):
        p = contacts[0][5].get_point(i)

        MINI_CHECK(p[0] > 1.0 - 1e-9 and p[0] < 2.0 + 1e-9)
        MINI_CHECK(p[1] > 0.5 - 1e-9 and p[1] < 1.0 + 1e-9)
        MINI_CHECK(TOLERANCE.is_close(p[2], 1.0))


@MINI_TEST("Intersection", "Adjacency Search")
def test_intersection_adjacency_search():
    from session_py import intersection
    from session_py import Element
    from session_py import Mesh
    from session_py import Point

    a = Element(
        Mesh.from_polylines(
            [
                [
                    Point(0.0, 0.0, 0.0),
                    Point(1.0, 0.0, 0.0),
                    Point(1.0, 1.0, 0.0),
                    Point(0.0, 1.0, 0.0),
                ],
            ]
        )
    )
    b = Element(
        Mesh.from_polylines(
            [
                [
                    Point(1.0, 0.0, 0.0),
                    Point(2.0, 0.0, 0.0),
                    Point(2.0, 1.0, 0.0),
                    Point(1.0, 1.0, 0.0),
                ],
            ]
        )
    )
    c = Element(
        Mesh.from_polylines(
            [
                [
                    Point(5.0, 0.0, 0.0),
                    Point(6.0, 0.0, 0.0),
                    Point(6.0, 1.0, 0.0),
                    Point(5.0, 1.0, 0.0),
                ],
            ]
        )
    )
    elements = [a, b, c]
    adjacency = intersection.adjacency_search(elements, 0.01)

    MINI_CHECK(len(adjacency) == 4)
    MINI_CHECK(adjacency[0] == 0)
    MINI_CHECK(adjacency[1] == 1)
    MINI_CHECK(adjacency[2] == -1)
    MINI_CHECK(adjacency[3] == -1)


@MINI_TEST("Intersection", "Line Line Classified")
def test_intersection_line_line_classified():
    from session_py import Line
    from session_py.intersection import line_line_classified

    s0 = Line(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    s1 = Line(0.0, -1.0, 0.0, 0.0, 1.0, 0.0)
    result = line_line_classified(s0, s1, 1, 1, 0, 0, 0.5)

    MINI_CHECK(result is not None)

    p0, p1, v0, v1, normal, type0, type1, is_parallel = result

    MINI_CHECK(not is_parallel)
    MINI_CHECK(abs(p0[0]) < 1e-6)
    MINI_CHECK(abs(p0[1]) < 1e-6)
    MINI_CHECK(abs(p1[0]) < 1e-6)
    MINI_CHECK(abs(p1[1]) < 1e-6)
    MINI_CHECK(abs(abs(normal[2]) - 1.0) < 1e-6)

    e0 = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    e1 = Line(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    result2 = line_line_classified(e0, e1, 1, 1, 0, 0, 0.5)

    MINI_CHECK(result2 is not None)

    p0, p1, v0, v1, normal, type0, type1, is_parallel = result2

    MINI_CHECK(not type0)
    MINI_CHECK(not type1)
    MINI_CHECK(abs(p0[0]) < 1e-6)
    MINI_CHECK(abs(p0[1]) < 1e-6)

    q0 = Line(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    q1 = Line(0.0, 1.0, 0.0, 2.0, 1.0, 0.0)
    result3 = line_line_classified(q0, q1, 1, 1, 0, 0, 0.5)

    MINI_CHECK(result3 is not None)

    p0, p1, v0, v1, normal, type0, type1, is_parallel = result3

    MINI_CHECK(is_parallel)
    MINI_CHECK(not type0)
    MINI_CHECK(not type1)


if __name__ == "__main__":
    run_all("python")
