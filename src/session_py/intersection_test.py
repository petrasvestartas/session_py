from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


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
    result = intersection.line_line_parameters(line0, line1, Tolerance.APPROXIMATION, False)

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
    from session_py import Line
    from session_py import Point

    box = OBB.from_points([
        Point(-1.0, -1.0, -1.0),
        Point(1.0, 1.0, 1.0),
    ])
    line = Line(-5.0, 0.0, 0.0, -4.0, 0.0, 0.0)
    points = intersection.ray_box(line, box, 0.0, 100.0)

    MINI_CHECK(points is not None)
    MINI_CHECK(abs(points[0][0] - (-1.0)) < 1e-4)
    MINI_CHECK(abs(points[1][0] - 1.0) < 1e-4)


@MINI_TEST("Intersection", "Ray Box Miss")
def test_intersection_ray_box_miss():
    from session_py import intersection
    from session_py import OBB
    from session_py import Line
    from session_py import Point

    box = OBB.from_points([
        Point(-1.0, -1.0, -1.0),
        Point(1.0, 1.0, 1.0),
    ])
    line = Line(-5.0, 5.0, 0.0, -4.0, 5.0, 0.0)
    points = intersection.ray_box(line, box, 0.0, 100.0)

    MINI_CHECK(points is None)


@MINI_TEST("Intersection", "Ray Sphere")
def test_intersection_ray_sphere():
    from session_py import intersection
    from session_py import Line
    from session_py import Point

    line = Line(-5.0, 0.0, 0.0, -4.0, 0.0, 0.0)
    center = Point(0.0, 0.0, 0.0)
    radius = 2.0
    points = intersection.ray_sphere(line, center, radius)

    MINI_CHECK(points is not None)
    MINI_CHECK(len(points) == 2)
    MINI_CHECK(abs(points[0][0] - (-2.0)) < 1e-4)
    MINI_CHECK(abs(points[1][0] - 2.0) < 1e-4)


@MINI_TEST("Intersection", "Ray Sphere Tangent")
def test_intersection_ray_sphere_tangent():
    from session_py import intersection
    from session_py import Line
    from session_py import Point

    line = Line(-5.0, 2.0, 0.0, -4.0, 2.0, 0.0)
    center = Point(0.0, 0.0, 0.0)
    radius = 2.0
    points = intersection.ray_sphere(line, center, radius)

    MINI_CHECK(points is not None)
    MINI_CHECK(len(points) == 1)
    MINI_CHECK(abs(points[0][0] - 0.0) < 1e-4)


@MINI_TEST("Intersection", "Ray Sphere Miss")
def test_intersection_ray_sphere_miss():
    from session_py import intersection
    from session_py import Line
    from session_py import Point

    line = Line(-5.0, 5.0, 0.0, -4.0, 5.0, 0.0)
    center = Point(0.0, 0.0, 0.0)
    radius = 2.0
    points = intersection.ray_sphere(line, center, radius)

    MINI_CHECK(points is None)


@MINI_TEST("Intersection", "Ray Triangle")
def test_intersection_ray_triangle():
    from session_py import intersection
    from session_py import Line
    from session_py import Point

    line = Line(0.5, 0.5, -1.0, 0.5, 0.5, 0.0)
    v0 = Point(0.0, 0.0, 0.0)
    v1 = Point(1.0, 0.0, 0.0)
    v2 = Point(0.0, 1.0, 0.0)
    result = intersection.ray_triangle(line, v0, v1, v2, 1e-6)

    MINI_CHECK(result is not None)
    MINI_CHECK(abs(result[2] - 0.0) < 1e-4)


@MINI_TEST("Intersection", "Ray Triangle Miss")
def test_intersection_ray_triangle_miss():
    from session_py import intersection
    from session_py import Line
    from session_py import Point

    line = Line(2.0, 2.0, -1.0, 2.0, 2.0, 0.0)
    v0 = Point(0.0, 0.0, 0.0)
    v1 = Point(1.0, 0.0, 0.0)
    v2 = Point(0.0, 1.0, 0.0)
    result = intersection.ray_triangle(line, v0, v1, v2, 1e-6)

    MINI_CHECK(result is None)


@MINI_TEST("Intersection", "Ray Triangle Parallel")
def test_intersection_ray_triangle_parallel():
    from session_py import intersection
    from session_py import Line
    from session_py import Point

    line = Line(0.5, 0.5, -1.0, 1.5, 0.5, -1.0)
    v0 = Point(0.0, 0.0, 0.0)
    v1 = Point(1.0, 0.0, 0.0)
    v2 = Point(0.0, 1.0, 0.0)
    result = intersection.ray_triangle(line, v0, v1, v2, 1e-6)

    MINI_CHECK(result is None)


@MINI_TEST("Intersection", "Ray Mesh")
def test_intersection_ray_mesh():
    from session_py import intersection
    from session_py import Line
    from session_py import Mesh
    from session_py import Point

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
    line = Line(0.5, 0.5, -1.0, 0.5, 0.5, 0.0)
    hits = intersection.ray_mesh(line, mesh, 1e-6, True)

    MINI_CHECK(hits is not None)
    MINI_CHECK(len(hits) >= 1)
    MINI_CHECK(abs(hits[0][2] - 0.0) < 1e-3)


@MINI_TEST("Intersection", "Ray Mesh First")
def test_intersection_ray_mesh_first():
    from session_py import intersection
    from session_py import Line
    from session_py import Mesh
    from session_py import Point

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
    line = Line(0.5, 0.5, -1.0, 0.5, 0.5, 0.0)
    hits = intersection.ray_mesh(line, mesh, 1e-6, False)

    MINI_CHECK(hits is not None)
    MINI_CHECK(len(hits) == 1)


@MINI_TEST("Intersection", "Ray Mesh Miss")
def test_intersection_ray_mesh_miss():
    from session_py import intersection
    from session_py import Line
    from session_py import Mesh
    from session_py import Point

    polygons = [
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0.0),
        ],
    ]
    mesh = Mesh.from_polylines(polygons)
    line = Line(5.0, 5.0, -1.0, 5.0, 5.0, 0.0)
    hits = intersection.ray_mesh(line, mesh, 1e-6, True)

    MINI_CHECK(hits is None)


@MINI_TEST("Intersection", "Ray Mesh Bvh")
def test_intersection_ray_mesh_bvh():
    from session_py import intersection
    from session_py import Line
    from session_py import Mesh
    from session_py import Point

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
    line = Line(0.5, 0.5, -1.0, 0.5, 0.5, 0.0)
    hits = intersection.ray_mesh_bvh(line, mesh, 1e-6, True)

    MINI_CHECK(hits is not None)
    MINI_CHECK(len(hits) >= 1)
    MINI_CHECK(abs(hits[0][2] - 0.0) < 1e-3)


@MINI_TEST("Intersection", "Ray Mesh Bvh First")
def test_intersection_ray_mesh_bvh_first():
    from session_py import intersection
    from session_py import Line
    from session_py import Mesh
    from session_py import Point

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
    line = Line(0.5, 0.5, -1.0, 0.5, 0.5, 0.0)
    hits = intersection.ray_mesh_bvh(line, mesh, 1e-6, False)

    MINI_CHECK(hits is not None)
    MINI_CHECK(len(hits) == 1)


@MINI_TEST("Intersection", "Ray Mesh Bvh Miss")
def test_intersection_ray_mesh_bvh_miss():
    from session_py import intersection
    from session_py import Line
    from session_py import Mesh
    from session_py import Point

    polygons = [
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0.0),
        ],
    ]
    mesh = Mesh.from_polylines(polygons)
    line = Line(5.0, 5.0, -1.0, 5.0, 5.0, 0.0)
    hits = intersection.ray_mesh_bvh(line, mesh, 1e-6, True)

    MINI_CHECK(hits is None)


@MINI_TEST("Intersection", "Ray Mesh Bvh Vs Naive")
def test_intersection_ray_mesh_bvh_vs_naive():
    from session_py import intersection
    from session_py import Line
    from session_py import Mesh
    from session_py import Point

    polygons = []
    for i in range(10):
        for j in range(10):
            x = float(i)
            y = float(j)
            polygons.append([
                Point(x, y, 0.0),
                Point(x + 1.0, y, 0.0),
                Point(x + 1.0, y + 1.0, 0.0),
                Point(x, y + 1.0, 0.0),
            ])
    mesh = Mesh.from_polylines(polygons)
    line = Line(5.5, 5.5, -1.0, 5.5, 5.5, 0.0)
    hits_naive = intersection.ray_mesh(line, mesh, 1e-6, True)
    hits_bvh = intersection.ray_mesh_bvh(line, mesh, 1e-6, True)
    result_naive = hits_naive is not None
    result_bvh = hits_bvh is not None

    MINI_CHECK(result_naive == result_bvh)
    naive_count = len(hits_naive) if hits_naive else 0
    bvh_count = len(hits_bvh) if hits_bvh else 0

    MINI_CHECK(naive_count == bvh_count)
    if hits_naive and hits_bvh:
        MINI_CHECK(abs(hits_naive[0][2] - hits_bvh[0][2]) < 1e-4)


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

    # Three parallel planes → None
    p0 = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0))
    p1 = Plane.from_point_normal(Point(0.0, 0.0, 1.0), Vector(0.0, 0.0, 1.0))
    p2 = Plane.from_point_normal(Point(0.0, 0.0, 2.0), Vector(0.0, 0.0, 1.0))

    MINI_CHECK(intersection.plane_plane_plane_check(p0, p1, p2) is None)

    # Three valid planes
    px = Plane.from_point_normal(Point(1.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0))
    py = Plane.from_point_normal(Point(0.0, 2.0, 0.0), Vector(0.0, 1.0, 0.0))
    pz = Plane.from_point_normal(Point(0.0, 0.0, 3.0), Vector(0.0, 0.0, 1.0))
    pt = intersection.plane_plane_plane_check(px, py, pz)

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
    # Cycle left→bottom→right→top so adjacent pairs are non-parallel
    planes = [
        Plane.from_point_normal(Point(-1.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0)),  # x=-1
        Plane.from_point_normal(Point(0.0, -1.0, 0.0), Vector(0.0, 1.0, 0.0)),  # y=-1
        Plane.from_point_normal(Point(1.0, 0.0, 0.0),  Vector(1.0, 0.0, 0.0)),  # x= 1
        Plane.from_point_normal(Point(0.0, 1.0, 0.0),  Vector(0.0, 1.0, 0.0)),  # y= 1
    ]
    result = intersection.plane_4planes(main, planes)

    MINI_CHECK(result is not None)
    MINI_CHECK(result.point_count() == 5)
    for i in range(result.point_count()):
        p = result.get_point(i)
        MINI_CHECK(abs(p[2]) < 1e-6)
    # First == last (closed)
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
        Plane.from_point_normal(Point(1.0, 0.0, 0.0),  Vector(1.0, 0.0, 0.0)),
        Plane.from_point_normal(Point(0.0, 1.0, 0.0),  Vector(0.0, 1.0, 0.0)),
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
    l0 = Line(-1.0, -1.0, -1.0, -1.0,  1.0, 1.0)
    l1 = Line( 1.0, -1.0, -1.0,  1.0,  1.0, 1.0)
    l2 = Line(-1.0, -1.0, -1.0,  1.0, -1.0, 1.0)
    l3 = Line(-1.0,  1.0, -1.0,  1.0,  1.0, 1.0)
    result = intersection.plane_4lines(plane, l0, l1, l2, l3)

    MINI_CHECK(result is not None)
    MINI_CHECK(result.point_count() == 5)
    for i in range(result.point_count()):
        p = result.get_point(i)
        MINI_CHECK(abs(p[2]) < 1e-6)


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

    poly = Polyline([
        Point(-1.0, -1.0, 0.0),
        Point(1.0,  -1.0, 0.0),
        Point(1.0,   1.0, 0.0),
        Point(-1.0,  1.0, 0.0),
        Point(-1.0, -1.0, 0.0),  # closed
    ])
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
    from session_py import Point

    # Two lines that cross at (1, 1, 0)
    cutter = Line(0.0, 1.0, 0.0, 2.0, 1.0, 0.0)
    seg    = Line(1.0, 0.0, 0.0, 1.0, 2.0, 0.0)
    result = intersection.line_line_3d(cutter, seg)

    MINI_CHECK(result is not None)
    MINI_CHECK(abs(result[0] - 1.0) < 1e-6)
    MINI_CHECK(abs(result[1] - 1.0) < 1e-6)
    MINI_CHECK(abs(result[2] - 0.0) < 1e-6)

    # Parallel lines → None
    par0 = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    par1 = Line(0.0, 1.0, 0.0, 1.0, 1.0, 0.0)

    MINI_CHECK(intersection.line_line_3d(par0, par1) is None)


@MINI_TEST("Intersection", "Polyline Plane To Line")
def test_intersection_polyline_plane_to_line():
    from session_py import Plane, Point, Polyline, Vector
    from session_py.intersection import polyline_plane_to_line
    poly = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(4.0, 0.0, 0.0),
        Point(4.0, 4.0, 0.0),
        Point(0.0, 4.0, 0.0),
        Point(0.0, 0.0, 0.0),
    ])
    pln = Plane.from_point_normal(Point(0.0, 2.0, 0.0), Vector(0.0, 1.0, 0.0))
    out = polyline_plane_to_line(poly, pln, Point(0.0, 0.0, 0.0))
    MINI_CHECK(out is not None)
    MINI_CHECK(TOLERANCE.is_close(out.start()[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(out.end()[0], 4.0))


@MINI_TEST("Intersection", "Quad From Line Top Bottom Planes")
def test_intersection_quad_from_line_top_bottom_planes():
    from session_py import Line, Plane, Point, Vector
    from session_py.intersection import quad_from_line_top_bottom_planes
    face = Plane.xy_plane()
    line = Line(0.0, 0.0, 0.0, 10.0, 0.0, 0.0)
    plane0 = Plane.from_point_normal(Point(0.0, -2.0, 0.0), Vector(0.0, 1.0, 0.0))
    plane1 = Plane.from_point_normal(Point(0.0,  2.0, 0.0), Vector(0.0, 1.0, 0.0))
    out = quad_from_line_top_bottom_planes(face, line, plane0, plane1)
    MINI_CHECK(out is not None)
    MINI_CHECK(out.point_count() == 5)
    MINI_CHECK(TOLERANCE.is_close(abs(out.get_point(0)[1]), 2.0))
    MINI_CHECK(TOLERANCE.is_close(abs(out.get_point(2)[1]), 2.0))
    MINI_CHECK(TOLERANCE.is_close(out.get_point(2)[0], 10.0))


@MINI_TEST("Intersection", "Orthogonal Vector Between Two Plane Pairs")
def test_intersection_orthogonal_vector_between_two_plane_pairs():
    from session_py import Plane, Point, Vector
    from session_py.intersection import orthogonal_vector_between_two_plane_pairs
    pp00 = Plane.xy_plane()
    pp10 = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0))
    pp11 = Plane.from_point_normal(Point(4.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0))
    out = orthogonal_vector_between_two_plane_pairs(pp00, pp10, pp11)
    MINI_CHECK(out is not None)
    mag = (out[0]*out[0] + out[1]*out[1] + out[2]*out[2]) ** 0.5
    MINI_CHECK(TOLERANCE.is_close(mag, 4.0))
    MINI_CHECK(TOLERANCE.is_close(out[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(out[2], 0.0))


@MINI_TEST("Intersection", "Closed And Open Paths 2D")
def test_intersection_closed_and_open_paths_2d():
    from session_py import Plane, Point, Polyline
    from session_py.intersection import closed_and_open_paths_2d
    plate = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(10.0, 0.0, 0.0),
        Point(10.0, 10.0, 0.0),
        Point(0.0, 10.0, 0.0),
        Point(0.0, 0.0, 0.0),
    ])
    joint = Polyline([
        Point(-2.0, 5.0, 0.0),
        Point(12.0, 5.0, 0.0),
    ])
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


if __name__ == "__main__":
    run_all("python")
