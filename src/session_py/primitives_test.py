from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI
import math


@MINI_TEST("Primitives", "Mesh_arrow")
def test_mesh_arrow():
    from session_py import Primitives
    from session_py import Line

    line = Line(0.0, 0.0, 0.0, 0.0, 0.0, 8.0)
    m = Primitives.arrow_mesh(line, 1.0)

    MINI_CHECK(m.number_of_vertices() == 29)
    MINI_CHECK(m.number_of_faces() == 28)


@MINI_TEST("Primitives", "Mesh_cylinder")
def test_mesh_cylinder():
    from session_py import Primitives
    from session_py import Line

    line = Line(0.0, 0.0, 0.0, 0.0, 0.0, 8.0)
    m = Primitives.cylinder_mesh(line, 1.0)

    MINI_CHECK(m.number_of_vertices() == 20)
    MINI_CHECK(m.number_of_faces() == 20)


@MINI_TEST("Primitives", "Nurbscurve_polyline")
def test_nurbscurve_polyline():
    from session_py import NurbsCurve
    from session_py import Point

    c = NurbsCurve.create(False, 1, [
        Point(0,0,0), Point(1,2,0), Point(2,0,0), Point(3,2,0), Point(4,0,0)])

    MINI_CHECK(c.cv_count() == 5)
    MINI_CHECK(c.order() == 2)
    MINI_CHECK(c.degree() == 1)
    MINI_CHECK(c.is_rational() == False)
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(c.domain_start()), Point(0,0,0)))
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(c.domain_end()), Point(4,0,0)))


@MINI_TEST("Primitives", "Nurbscurve_circle")
def test_nurbscurve_circle():
    from session_py import Primitives

    c = Primitives.circle(0.0, 0.0, 0.0, 1.0)

    MINI_CHECK(c.cv_count() == 9)
    MINI_CHECK(c.order() == 3)
    MINI_CHECK(c.is_rational() == True)


@MINI_TEST("Primitives", "Nurbscurve_ellipse")
def test_nurbscurve_ellipse():
    from session_py import Primitives

    c = Primitives.ellipse(0.0, 0.0, 0.0, 2.0, 1.0)

    MINI_CHECK(c.cv_count() == 9)
    MINI_CHECK(c.order() == 3)
    MINI_CHECK(c.is_rational() == True)


@MINI_TEST("Primitives", "Nurbscurve_arc")
def test_nurbscurve_arc():
    from session_py import Primitives
    from session_py import Point

    start = Point(0.0, 0.0, 0.0)
    mid = Point(1.0, 1.0, 0.0)
    end = Point(2.0, 0.0, 0.0)
    c = Primitives.arc(start, mid, end)

    MINI_CHECK(c.cv_count() == 3)
    MINI_CHECK(c.order() == 3)
    MINI_CHECK(c.is_rational() == True)


@MINI_TEST("Primitives", "Nurbscurve_parabola")
def test_nurbscurve_parabola():
    from session_py import Primitives
    from session_py import Point

    p0 = Point(-1.0, 1.0, 0.0)
    p1 = Point(0.0, 0.0, 0.0)
    p2 = Point(1.0, 1.0, 0.0)
    c = Primitives.parabola(p0, p1, p2)

    MINI_CHECK(c.cv_count() == 3)
    MINI_CHECK(c.order() == 3)
    MINI_CHECK(c.is_rational() == False)


@MINI_TEST("Primitives", "Nurbscurve_hyperbola")
def test_nurbscurve_hyperbola():
    from session_py import Primitives
    from session_py import Point

    center = Point(0.0, 0.0, 0.0)
    c = Primitives.hyperbola(center, 1.0, 1.0, 1.0)

    MINI_CHECK(c.cv_count() >= 4)
    MINI_CHECK(c.order() == 4)
    MINI_CHECK(c.is_rational() == False)


@MINI_TEST("Primitives", "Nurbscurve_spiral")
def test_nurbscurve_spiral():
    from session_py import Primitives

    c = Primitives.spiral(1.0, 2.0, 1.0, 5.0)

    MINI_CHECK(c.cv_count() >= 4)
    MINI_CHECK(c.order() == 4)
    MINI_CHECK(c.is_rational() == False)


@MINI_TEST("Primitives", "Nurbssurface_cylinder")
def test_nurbssurface_cylinder():
    from session_py import Primitives


    s = Primitives.cylinder_surface(0.0, 0.0, 0.0, 1.0, 5.0)

    MINI_CHECK(s.is_valid())
    MINI_CHECK(s.is_rational())
    MINI_CHECK(s.cv_count_dir(0) == 9)
    MINI_CHECK(s.cv_count_dir(1) == 2)
    MINI_CHECK(s.order(0) == 3)
    MINI_CHECK(s.order(1) == 2)

    p00 = s.point_at(0.0, 0.0)
    MINI_CHECK(abs(p00[0] - 1.0) < 1e-10)
    MINI_CHECK(abs(p00[1] - 0.0) < 1e-10)
    MINI_CHECK(abs(p00[2] - 0.0) < 1e-10)

    p01 = s.point_at(0.0, 1.0)
    MINI_CHECK(abs(p01[0] - 1.0) < 1e-10)
    MINI_CHECK(abs(p01[1] - 0.0) < 1e-10)
    MINI_CHECK(abs(p01[2] - 5.0) < 1e-10)

    pmid = s.point_at(1.0, 0.5)
    MINI_CHECK(abs(pmid[0] - 0.0) < 1e-10)
    MINI_CHECK(abs(pmid[1] - 1.0) < 1e-10)
    MINI_CHECK(abs(pmid[2] - 2.5) < 1e-10)


@MINI_TEST("Primitives", "Nurbssurface_cone")
def test_nurbssurface_cone():
    from session_py import Primitives


    s = Primitives.cone_surface(0.0, 0.0, 0.0, 1.0, 5.0)

    MINI_CHECK(s.is_valid())
    MINI_CHECK(s.is_rational())
    MINI_CHECK(s.cv_count_dir(0) == 9)
    MINI_CHECK(s.cv_count_dir(1) == 2)
    MINI_CHECK(s.order(0) == 3)
    MINI_CHECK(s.order(1) == 2)

    pbase = s.point_at(0.0, 0.0)
    MINI_CHECK(abs(pbase[0] - 1.0) < 1e-10)
    MINI_CHECK(abs(pbase[1] - 0.0) < 1e-10)
    MINI_CHECK(abs(pbase[2] - 0.0) < 1e-10)

    papex = s.point_at(0.0, 1.0)
    MINI_CHECK(abs(papex[0] - 0.0) < 1e-10)
    MINI_CHECK(abs(papex[1] - 0.0) < 1e-10)
    MINI_CHECK(abs(papex[2] - 5.0) < 1e-10)

    pmid = s.point_at(0.0, 0.5)
    MINI_CHECK(abs(pmid[0] - 0.5) < 1e-10)
    MINI_CHECK(abs(pmid[1] - 0.0) < 1e-10)
    MINI_CHECK(abs(pmid[2] - 2.5) < 1e-10)


@MINI_TEST("Primitives", "Nurbssurface_sphere")
def test_nurbssurface_sphere():
    from session_py import Primitives

    s = Primitives.sphere_surface(0.0, 0.0, 0.0, 2.0)

    MINI_CHECK(s.is_valid())
    MINI_CHECK(s.is_rational())
    MINI_CHECK(s.cv_count_dir(0) == 9)
    MINI_CHECK(s.cv_count_dir(1) == 5)
    MINI_CHECK(s.order(0) == 3)
    MINI_CHECK(s.order(1) == 3)

    p00 = s.point_at(0.0, 0.0)
    MINI_CHECK(abs(p00[0] - 0.0) < 1e-10)
    MINI_CHECK(abs(p00[1] - 0.0) < 1e-10)
    MINI_CHECK(abs(p00[2] - (-2.0)) < 1e-10)

    p_top = s.point_at(0.0, 2.0)
    MINI_CHECK(abs(p_top[0] - 0.0) < 1e-10)
    MINI_CHECK(abs(p_top[1] - 0.0) < 1e-10)
    MINI_CHECK(abs(p_top[2] - 2.0) < 1e-10)

    p_eq = s.point_at(0.0, 1.0)
    MINI_CHECK(abs(p_eq[0] - 2.0) < 1e-10)
    MINI_CHECK(abs(p_eq[1] - 0.0) < 1e-10)
    MINI_CHECK(abs(p_eq[2] - 0.0) < 1e-10)

    p_eq2 = s.point_at(1.0, 1.0)
    MINI_CHECK(abs(p_eq2[0] - 0.0) < 1e-10)
    MINI_CHECK(abs(p_eq2[1] - 2.0) < 1e-10)
    MINI_CHECK(abs(p_eq2[2] - 0.0) < 1e-10)


@MINI_TEST("Primitives", "Nurbssurface_quad_sphere")
def test_nurbssurface_quad_sphere():
    from session_py import Primitives
    import math

    R = 5.0
    faces = Primitives.quad_sphere(0.0, 0.0, 0.0, R)

    MINI_CHECK(len(faces) == 6)
    for f in range(6):
        MINI_CHECK(faces[f].is_valid())
        MINI_CHECK(faces[f].is_rational())
        MINI_CHECK(faces[f].order(0) == 3)
        MINI_CHECK(faces[f].order(1) == 3)
        MINI_CHECK(faces[f].cv_count_dir(0) == 3)
        MINI_CHECK(faces[f].cv_count_dir(1) == 3)

    max_err = 0.0
    for f in range(6):
        for i in range(5):
            u = i / 4.0
            for j in range(5):
                v = j / 4.0
                p = faces[f].point_at(u, v)
                dist = math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
                err = abs(dist - R)
                if err > max_err:
                    max_err = err
    MINI_CHECK(max_err < 0.02 * R)


@MINI_TEST("Primitives", "Nurbssurface_torus")
def test_nurbssurface_torus():
    from session_py import Primitives


    s = Primitives.torus_surface(0.0, 0.0, 0.0, 3.0, 1.0)

    MINI_CHECK(s.is_valid())
    MINI_CHECK(s.is_rational())
    MINI_CHECK(s.cv_count_dir(0) == 9)
    MINI_CHECK(s.cv_count_dir(1) == 9)
    MINI_CHECK(s.order(0) == 3)
    MINI_CHECK(s.order(1) == 3)

    p00 = s.point_at(0.0, 0.0)
    MINI_CHECK(abs(p00[0] - 4.0) < 1e-10)
    MINI_CHECK(abs(p00[1] - 0.0) < 1e-10)
    MINI_CHECK(abs(p00[2] - 0.0) < 1e-10)

    p10 = s.point_at(1.0, 0.0)
    MINI_CHECK(abs(p10[0] - 0.0) < 1e-10)
    MINI_CHECK(abs(p10[1] - 4.0) < 1e-10)
    MINI_CHECK(abs(p10[2] - 0.0) < 1e-10)

    p_top = s.point_at(0.0, 1.0)
    MINI_CHECK(abs(p_top[0] - 3.0) < 1e-10)
    MINI_CHECK(abs(p_top[1] - 0.0) < 1e-10)
    MINI_CHECK(abs(p_top[2] - 1.0) < 1e-10)


@MINI_TEST("Primitives", "Nurbssurface_ruled")
def test_nurbssurface_ruled():
    from session_py import Primitives
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Vector

    pts_a = [Point(3.0, 0.0, 0.0), Point(-2.0, 0.0, 5.0)]
    pts_b = [Point(3.0, 5.0, 5.0), Point(-2.0, 5.0, 0.0)]
    crv_a = NurbsCurve.create(False, 1, pts_a)
    crv_b = NurbsCurve.create(False, 1, pts_b)
    srf = Primitives.create_ruled(crv_a, crv_b)
    _m = srf.mesh()

    MINI_CHECK(srf.is_valid())
    MINI_CHECK(srf.degree(0) == 1)
    MINI_CHECK(srf.degree(1) == 1)
    MINI_CHECK(srf.cv_count_dir(0) == 2)
    MINI_CHECK(srf.cv_count_dir(1) == 2)

    rd, ruv = srf.divide_by_count(4, 4)
    MINI_CHECK(len(rd) == 5)
    MINI_CHECK(len(rd[0]) == 5)

    pts = []
    for i in range(len(rd)):
        for j in range(len(rd[i])):
            pts.append(rd[i][j])

    normals = []
    for i in range(len(ruv)):
        for j in range(len(ruv[i])):
            normals.append(srf.normal_at(ruv[i][j][0], ruv[i][j][1]))

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


@MINI_TEST("Primitives", "Nurbssurface_planar")
def test_nurbssurface_planar():
    from session_py import Primitives
    from session_py import NurbsCurve
    from session_py import Point

    c1 = math.cos(0.7); s1 = math.sin(0.7)
    c2 = math.cos(0.96); s2 = math.sin(0.96)
    c3 = math.cos(0.52); s3 = math.sin(0.52)
    c4 = math.cos(1.13); s4 = math.sin(1.13)

    ca = NurbsCurve.create(False, 1, [
        Point(0.0, 0.0, 0.0), Point(4.0, 0.0, 0.0),
        Point(4.0, 3.0*c1, 3.0*s1), Point(0.0, 3.0*c1, 3.0*s1),
        Point(0.0, 0.0, 0.0)])
    s_quad = Primitives.create_planar(ca)
    m_quad = s_quad.mesh()

    cb1 = NurbsCurve.create(False, 1, [
        Point(8.0, 0.0, 0.0), Point(8.0+5.0*c2, 0.0, 5.0*s2),
        Point(8.0+2.0*c2, 3.0, 2.0*s2), Point(8.0, 0.0, 0.0)])
    s_triangle = Primitives.create_planar(cb1)
    m_triangle = s_triangle.mesh()

    ox = 18.0
    cb2 = NurbsCurve.create(False, 1, [
        Point(ox+0.0*c3, 0.0*s3, 0.0),
        Point(ox+4.0*c3, 4.0*s3, 0.0),
        Point(ox+5.0*c3-2.0*s3, 5.0*s3+2.0*c3, 0.0),
        Point(ox+3.0*c3-4.0*s3, 3.0*s3+4.0*c3, 0.0),
        Point(ox-1.0*c3-3.0*s3, -1.0*s3+3.0*c3, 0.0),
        Point(ox+0.0*c3, 0.0*s3, 0.0)])
    s_polygon = Primitives.create_planar(cb2)
    m_polygon = s_polygon.mesh()

    cc = NurbsCurve.create(False, 3, [
        Point(26.0, 0.0, 0.0),
        Point(29.0, 1.0*c4, 1.0*s4),
        Point(31.0, 0.5*c4, 0.5*s4),
        Point(32.0, 3.0*c4, 3.0*s4),
        Point(30.0, 5.0*c4, 5.0*s4),
        Point(27.0, 4.0*c4, 4.0*s4),
        Point(26.0, 0.0, 0.0)])
    s_nurbs = Primitives.create_planar(cc)
    m_nurbs = s_nurbs.mesh()

    MINI_CHECK(s_quad.is_valid())
    MINI_CHECK(s_quad.is_planar())
    MINI_CHECK(s_quad.cv_count_dir(0) == 2)
    MINI_CHECK(s_quad.cv_count_dir(1) == 2)
    MINI_CHECK(m_quad.number_of_vertices() == 4)
    MINI_CHECK(m_quad.number_of_faces() == 2)
    MINI_CHECK(TOLERANCE.is_point_close(s_quad.get_cv(0, 0), Point(0.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_quad.get_cv(0, 1), Point(0.0, 2.294526561853465, 1.932653061713073)))
    MINI_CHECK(TOLERANCE.is_point_close(s_quad.get_cv(1, 0), Point(4.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_quad.get_cv(1, 1), Point(4.0, 2.294526561853465, 1.932653061713073)))

    MINI_CHECK(s_triangle.is_valid())
    MINI_CHECK(s_triangle.is_planar())
    MINI_CHECK(s_triangle.cv_count_dir(0) == 2)
    MINI_CHECK(s_triangle.cv_count_dir(1) == 2)
    MINI_CHECK(m_triangle.number_of_vertices() == 3)
    MINI_CHECK(m_triangle.number_of_faces() == 1)
    MINI_CHECK(TOLERANCE.is_point_close(s_triangle.get_cv(0, 0), Point(8.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_triangle.get_cv(0, 1), Point(8.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_triangle.get_cv(1, 0), Point(10.867599930362283, 0.0, 4.095957841504991)))
    MINI_CHECK(TOLERANCE.is_point_close(s_triangle.get_cv(1, 1), Point(9.147039972144913, 3.0, 1.638383136601997)))

    MINI_CHECK(s_polygon.is_valid())
    MINI_CHECK(s_polygon.is_planar())
    MINI_CHECK(s_polygon.cv_count_dir(0) == 2)
    MINI_CHECK(s_polygon.cv_count_dir(1) == 2)
    MINI_CHECK(m_polygon.number_of_vertices() == 4)
    MINI_CHECK(m_polygon.number_of_faces() == 2)
    MINI_CHECK(TOLERANCE.is_point_close(s_polygon.get_cv(0, 0), Point(19.673777861921977, 6.364048611360808, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_polygon.get_cv(0, 1), Point(22.915428262469927, 2.987233669553135, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_polygon.get_cv(1, 0), Point(15.247175891573059, 2.114631246911942, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_polygon.get_cv(1, 1), Point(18.488826292121008, -1.262183694895731, 0.0)))

    MINI_CHECK(s_nurbs.is_valid())
    MINI_CHECK(s_nurbs.is_planar())
    MINI_CHECK(s_nurbs.cv_count_dir(0) == 2)
    MINI_CHECK(s_nurbs.cv_count_dir(1) == 2)
    MINI_CHECK(m_nurbs.number_of_vertices() == 4)
    MINI_CHECK(m_nurbs.number_of_faces() == 2)
    MINI_CHECK(TOLERANCE.is_point_close(s_nurbs.get_cv(0, 0), Point(26.652846559932474, -0.727774577493594, -1.542700265577809)))
    MINI_CHECK(TOLERANCE.is_point_close(s_nurbs.get_cv(0, 1), Point(24.347485651711366, 0.916607409071279, 1.942978687541882)))
    MINI_CHECK(TOLERANCE.is_point_close(s_nurbs.get_cv(1, 0), Point(32.606791655643732, 0.791738725121784, 1.678288276735475)))
    MINI_CHECK(TOLERANCE.is_point_close(s_nurbs.get_cv(1, 1), Point(30.301430747422629, 2.436120711686657, 5.163967229855166)))


@MINI_TEST("Primitives", "Nurbssurface_extrusion")
def test_nurbssurface_extrusion():
    from session_py import Primitives
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Vector

    direction = Vector(0.0, 1.0, 5.0)

    c1 = NurbsCurve.create(False, 1, [Point(13.0, 0.0, 0.0), Point(18.0, 0.0, 0.0)])
    s_line = Primitives.create_extrusion(c1, direction)
    m_line = s_line.mesh()

    c2 = Primitives.circle(24.0, 0.0, 0.0, 3.0)
    s_circle = Primitives.create_extrusion(c2, direction)
    m_circle = s_circle.mesh()

    c3 = NurbsCurve.create(False, 2, [Point(30.0, 0.0, 0.0), Point(33.0, 5.0, 0.0), Point(37.0, 0.0, 0.0)])
    s_arc = Primitives.create_extrusion(c3, direction)
    m_arc = s_arc.mesh()

    c4 = NurbsCurve.create(False, 1, [Point(40.0, 3.0, 0.0), Point(45.0, 0.0, 0.0), Point(50.0, 3.0, 0.0), Point(55.0, 0.0, 0.0)])
    s_wavy = Primitives.create_extrusion(c4, direction)
    m_wavy = s_wavy.mesh()

    MINI_CHECK(s_line.is_valid())
    MINI_CHECK(s_line.degree(0) == 1 and s_line.degree(1) == 1)
    MINI_CHECK(s_line.cv_count_dir(0) == 2 and s_line.cv_count_dir(1) == 2)
    MINI_CHECK(m_line.number_of_vertices() == 4)
    MINI_CHECK(m_line.number_of_faces() == 2)
    MINI_CHECK(TOLERANCE.is_point_close(s_line.get_cv(0, 0), Point(13.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_line.get_cv(0, 1), Point(13.0, 1.0, 5.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_line.get_cv(1, 0), Point(18.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_line.get_cv(1, 1), Point(18.0, 1.0, 5.0)))

    MINI_CHECK(s_circle.is_valid())
    MINI_CHECK(s_circle.degree(0) == 2 and s_circle.degree(1) == 1)
    MINI_CHECK(s_circle.is_rational())
    MINI_CHECK(s_circle.is_closed(0) == True and s_circle.is_closed(1) == False)
    MINI_CHECK(s_circle.cv_count_dir(0) == 9 and s_circle.cv_count_dir(1) == 2)
    MINI_CHECK(m_circle.number_of_vertices() == 40)
    MINI_CHECK(m_circle.number_of_faces() == 40)
    MINI_CHECK(TOLERANCE.is_point_close(s_circle.get_cv(0, 0), Point(27.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_circle.get_cv(0, 1), Point(27.0, 1.0, 5.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_circle.get_cv(4, 0), Point(21.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_circle.get_cv(4, 1), Point(21.0, 1.0, 5.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_circle.get_cv(8, 0), Point(27.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_circle.get_cv(8, 1), Point(27.0, 1.0, 5.0)))

    MINI_CHECK(s_arc.is_valid())
    MINI_CHECK(s_arc.degree(0) == 2 and s_arc.degree(1) == 1)
    MINI_CHECK(s_arc.cv_count_dir(0) == 3 and s_arc.cv_count_dir(1) == 2)
    MINI_CHECK(m_arc.number_of_vertices() == 16)
    MINI_CHECK(m_arc.number_of_faces() == 14)
    MINI_CHECK(TOLERANCE.is_point_close(s_arc.get_cv(0, 0), Point(30.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_arc.get_cv(0, 1), Point(30.0, 1.0, 5.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_arc.get_cv(1, 0), Point(33.0, 5.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_arc.get_cv(1, 1), Point(33.0, 6.0, 5.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_arc.get_cv(2, 0), Point(37.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_arc.get_cv(2, 1), Point(37.0, 1.0, 5.0)))

    MINI_CHECK(s_wavy.is_valid())
    MINI_CHECK(s_wavy.degree(0) == 1 and s_wavy.degree(1) == 1)
    MINI_CHECK(s_wavy.cv_count_dir(0) == 4 and s_wavy.cv_count_dir(1) == 2)
    MINI_CHECK(m_wavy.number_of_vertices() == 8)
    MINI_CHECK(m_wavy.number_of_faces() == 6)
    MINI_CHECK(TOLERANCE.is_point_close(s_wavy.get_cv(0, 0), Point(40.0, 3.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_wavy.get_cv(0, 1), Point(40.0, 4.0, 5.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_wavy.get_cv(1, 0), Point(45.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_wavy.get_cv(1, 1), Point(45.0, 1.0, 5.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_wavy.get_cv(3, 0), Point(55.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_wavy.get_cv(3, 1), Point(55.0, 1.0, 5.0)))


@MINI_TEST("Primitives", "Nurbssurface_loft")
def test_nurbssurface_loft():
    from session_py import Primitives
    from session_py import NurbsCurve
    from session_py import Point

    c1 = Primitives.circle(0.0, 0.0, 0.0, 2.0)
    c2 = Primitives.circle(0.0, 0.0, 2.0, 1.0)
    c3 = Primitives.circle(0.0, 0.0, 4.0, 1.5)
    c4 = Primitives.circle(0.0, 0.0, 6.0, 0.8)

    srf = Primitives.create_loft([c1, c2, c3, c4], 3)

    MINI_CHECK(srf.is_valid())
    MINI_CHECK(srf.cv_count_dir(0) == 9)
    MINI_CHECK(srf.cv_count_dir(1) == 4)
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(0, 0), Point(2.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(0, 1), Point(-0.677194251158421, 0.0, 1.75222035185728)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(0, 2), Point(3.00619893067415, 0.0, 4.08030037218547)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(0, 3), Point(0.8, 0.0, 6.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(1, 0), Point(2.0, 2.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(1, 1), Point(-0.677194251158421, -0.677194251158421, 1.75222035185728)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(1, 2), Point(3.00619893067414, 3.00619893067414, 4.08030037218547)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(1, 3), Point(0.8, 0.8, 6.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(2, 0), Point(0.0, 2.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(2, 1), Point(0.0, -0.677194251158421, 1.75222035185728)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(2, 2), Point(0.0, 3.00619893067415, 4.08030037218547)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(2, 3), Point(0.0, 0.8, 6.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(3, 0), Point(-2.0, 2.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(3, 1), Point(0.677194251158421, -0.677194251158421, 1.75222035185728)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(3, 2), Point(-3.00619893067414, 3.00619893067414, 4.08030037218547)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(3, 3), Point(-0.8, 0.8, 6.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(4, 0), Point(-2.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(4, 1), Point(0.677194251158421, 0.0, 1.75222035185728)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(4, 2), Point(-3.00619893067415, 0.0, 4.08030037218547)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(4, 3), Point(-0.8, 0.0, 6.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(5, 0), Point(-2.0, -2.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(5, 1), Point(0.677194251158421, 0.677194251158421, 1.75222035185728)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(5, 2), Point(-3.00619893067414, -3.00619893067414, 4.08030037218547)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(5, 3), Point(-0.8, -0.8, 6.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(6, 0), Point(0.0, -2.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(6, 1), Point(0.0, 0.677194251158421, 1.75222035185728)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(6, 2), Point(0.0, -3.00619893067415, 4.08030037218547)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(6, 3), Point(0.0, -0.8, 6.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(7, 0), Point(2.0, -2.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(7, 1), Point(-0.677194251158421, 0.677194251158421, 1.75222035185728)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(7, 2), Point(3.00619893067414, -3.00619893067414, 4.08030037218547)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(7, 3), Point(0.8, -0.8, 6.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(8, 0), Point(2.0, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(8, 1), Point(-0.677194251158421, 0.0, 1.75222035185728)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(8, 2), Point(3.00619893067415, 0.0, 4.08030037218547)))
    MINI_CHECK(TOLERANCE.is_point_close(srf.get_cv(8, 3), Point(0.8, 0.0, 6.0)))

    open_pts = [
        [Point(10.0, -12.0, 0.0), Point(10.0, -10.0, 3.0), Point(10.0, -7.0, 3.0), Point(10.0, -5.0, 0.0)],
        [Point(5.5, -12.0, 3.5), Point(5.5, -10.0, 1.5), Point(5.5, -7.0, 1.5), Point(5.5, -5.0, 3.5)],
        [Point(1.0, -12.0, 0.0), Point(1.0, -10.0, 3.0), Point(1.0, -7.0, 3.0), Point(1.0, -5.0, 0.0)],
    ]
    open_curves = [
        NurbsCurve.create(False, 3, open_pts[0]),
        NurbsCurve.create(False, 3, open_pts[1]),
        NurbsCurve.create(False, 3, open_pts[2]),
    ]
    open_srf = Primitives.create_loft(open_curves, 3)

    MINI_CHECK(open_srf.is_valid())
    MINI_CHECK(open_srf.cv_count_dir(0) == 4)
    MINI_CHECK(open_srf.cv_count_dir(1) == 3)

    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(0, 0), Point(10.0, -12.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(0, 1), Point(5.5, -12.0, 7.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(0, 2), Point(1.0, -12.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(1, 0), Point(10.0, -10.0, 3.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(1, 1), Point(5.5, -10.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(1, 2), Point(1.0, -10.0, 3.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(2, 0), Point(10.0, -7.0, 3.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(2, 1), Point(5.5, -7.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(2, 2), Point(1.0, -7.0, 3.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(3, 0), Point(10.0, -5.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(3, 1), Point(5.5, -5.0, 7.0)))
    MINI_CHECK(TOLERANCE.is_point_close(open_srf.get_cv(3, 2), Point(1.0, -5.0, 0.0)))


@MINI_TEST("Primitives", "Nurbssurface_revolve")
def test_nurbssurface_revolve():
    from session_py import Primitives
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Vector

    pa = NurbsCurve.create(False, 3, [
        Point(1.5, 0.0, 0.0), Point(1.5, 0.0, 0.3), Point(0.3, 0.0, 0.5),
        Point(0.3, 0.0, 2.5), Point(0.2, 0.0, 3.0), Point(2.0, 0.0, 4.5), Point(1.8, 0.0, 5.0)])
    s_vase = Primitives.create_revolve(pa, Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0), 2.0 * PI)
    m_vase = s_vase.mesh()

    w = math.sqrt(2.0) / 2.0
    cw = [1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0]
    ca = [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0]
    sa = [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0]
    ck = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    rr = 5.0; r = 1.5; tcx = 14.0
    pb = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=9)
    for i in range(10):
        pb.set_knot(i, ck[i])
    for i in range(9):
        pb.set_cv_4d(i, (tcx + rr + r * ca[i]) * cw[i], 0.0, r * sa[i] * cw[i], cw[i])
    s_torus = Primitives.create_revolve(pb, Point(tcx, 0.0, 0.0), Vector(0.0, 0.0, 1.0), 2.0 * PI)
    m_torus = s_torus.mesh()

    pc = NurbsCurve.create(False, 1, [Point(29.0, 0.0, -0.5), Point(29.0, 0.0, 0.5)])
    s_elbow = Primitives.create_revolve(pc, Point(26.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0), PI / 2.0)
    m_elbow = s_elbow.mesh()

    sr = 2.0; scx = 36.0
    pd = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=5)
    sk = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    for i in range(6):
        pd.set_knot(i, sk[i])
    spx = [0.0, sr, sr, sr, 0.0]; spz = [-sr, -sr, 0.0, sr, sr]; spw = [1.0, w, 1.0, w, 1.0]
    for i in range(5):
        pd.set_cv_4d(i, (scx + spx[i]) * spw[i], 0.0, spz[i] * spw[i], spw[i])
    s_sphere = Primitives.create_revolve(pd, Point(scx, 0.0, 0.0), Vector(0.0, 0.0, 1.0), 2.0 * PI)
    m_sphere = s_sphere.mesh()

    pe = NurbsCurve.create(False, 1, [Point(44.0, 0.0, 3.0), Point(46.0, 0.0, 0.0)])
    s_cone = Primitives.create_revolve(pe, Point(44.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0), 2.0 * PI)
    m_cone = s_cone.mesh()

    MINI_CHECK(s_vase.is_valid())
    MINI_CHECK(s_vase.is_closed(0) == True)
    MINI_CHECK(s_vase.is_closed(1) == False)
    MINI_CHECK(s_vase.cv_count_dir(0) == 9)
    MINI_CHECK(s_vase.cv_count_dir(1) == 7)
    MINI_CHECK(m_vase.number_of_vertices() == 660)
    MINI_CHECK(m_vase.number_of_faces() == 1280)
    MINI_CHECK(TOLERANCE.is_point_close(s_vase.get_cv(0, 0), Point(1.5, 0.0, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_vase.get_cv(0, 6), Point(1.8, 0.0, 5.0)))

    MINI_CHECK(s_torus.is_valid())
    MINI_CHECK(s_torus.is_closed(0) == True)
    MINI_CHECK(s_torus.is_closed(1) == True)
    MINI_CHECK(s_torus.cv_count_dir(0) == 9)
    MINI_CHECK(s_torus.cv_count_dir(1) == 9)
    MINI_CHECK(m_torus.number_of_vertices() == 640)
    MINI_CHECK(m_torus.number_of_faces() == 1280)
    MINI_CHECK(TOLERANCE.is_point_close(s_torus.get_cv(0, 0), Point(20.5, 0.0, 0.0)))

    MINI_CHECK(s_elbow.is_valid())
    MINI_CHECK(s_elbow.is_closed(0) == False)
    MINI_CHECK(s_elbow.is_closed(1) == False)
    MINI_CHECK(s_elbow.cv_count_dir(0) == 3)
    MINI_CHECK(s_elbow.cv_count_dir(1) == 2)
    MINI_CHECK(m_elbow.number_of_vertices() == 16)
    MINI_CHECK(m_elbow.number_of_faces() == 14)
    MINI_CHECK(TOLERANCE.is_point_close(s_elbow.get_cv(0, 0), Point(29.0, 0.0, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(s_elbow.get_cv(0, 1), Point(29.0, 0.0, 0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(s_elbow.get_cv(2, 0), Point(26.0, 3.0, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(s_elbow.get_cv(2, 1), Point(26.0, 3.0, 0.5)))

    MINI_CHECK(s_sphere.is_valid())
    MINI_CHECK(s_sphere.is_closed(0) == True)
    MINI_CHECK(s_sphere.is_closed(1) == False)
    MINI_CHECK(s_sphere.is_singular(0) == True)
    MINI_CHECK(s_sphere.is_singular(2) == True)
    MINI_CHECK(s_sphere.cv_count_dir(0) == 9)
    MINI_CHECK(s_sphere.cv_count_dir(1) == 5)
    MINI_CHECK(m_sphere.number_of_vertices() > 0)
    MINI_CHECK(m_sphere.number_of_faces() > 0)
    MINI_CHECK(TOLERANCE.is_point_close(s_sphere.get_cv(0, 0), Point(36.0, 0.0, -2.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sphere.get_cv(0, 4), Point(36.0, 0.0, 2.0)))

    MINI_CHECK(s_cone.is_valid())
    MINI_CHECK(s_cone.is_closed(0) == True)
    MINI_CHECK(s_cone.is_closed(1) == False)
    MINI_CHECK(s_cone.is_singular(0) == True)
    MINI_CHECK(s_cone.is_singular(2) == False)
    MINI_CHECK(s_cone.cv_count_dir(0) == 9)
    MINI_CHECK(s_cone.cv_count_dir(1) == 2)
    MINI_CHECK(m_cone.number_of_vertices() > 0)
    MINI_CHECK(m_cone.number_of_faces() > 0)
    MINI_CHECK(TOLERANCE.is_point_close(s_cone.get_cv(0, 0), Point(44.0, 0.0, 3.0)))
    MINI_CHECK(TOLERANCE.is_point_close(s_cone.get_cv(0, 1), Point(46.0, 0.0, 0.0)))


@MINI_TEST("Primitives", "Nurbssurface_sweep")
def test_nurbssurface_sweep():
    from session_py import Primitives
    from session_py import NurbsCurve
    from session_py import Point

    rail = NurbsCurve.create(False, 2, [Point(0.0, 0.0, 0.0), Point(0.0, 5.0, 0.0), Point(2.0, 9.0, 0.0)])
    profile = Primitives.circle(0.0, 0.0, 0.0, 1.0)
    s_sweep1 = Primitives.create_sweep1(rail, profile)
    m_sweep1 = s_sweep1.mesh()

    rail1 = NurbsCurve.create(False, 2, [Point(6.0, -1.0, 0.0), Point(7.0, 3.0, 0.0), Point(8.0, 4.0, 0.0)])
    rail2 = NurbsCurve.create(False, 2, [Point(10.0, -1.0, 0.0), Point(10.0, 3.0, 0.0), Point(9.0, 4.0, 0.0)])
    shape1 = NurbsCurve.create(False, 2, [Point(6.0, -1.0, 0.0), Point(8.0, -1.0, 2.0), Point(10.0, -1.0, 0.0)])
    shape2 = NurbsCurve.create(False, 2, [Point(8.0, 4.0, 0.0), Point(8.5, 4.0, 1.5), Point(9.0, 4.0, 0.0)])
    s_sweep2 = Primitives.create_sweep2(rail1, rail2, [shape1, shape2])
    m_sweep2 = s_sweep2.mesh()

    MINI_CHECK(s_sweep1.is_valid())
    MINI_CHECK(s_sweep1.is_rational())
    MINI_CHECK(s_sweep1.cv_count_dir(0) == 9)
    MINI_CHECK(s_sweep1.cv_count_dir(1) == 6)
    MINI_CHECK(m_sweep1.number_of_vertices() > 0)
    MINI_CHECK(m_sweep1.number_of_faces() > 0)
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep1.get_cv(0, 0), Point(0.888888888888889, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep1.get_cv(0, 1), Point(0.888635792881381, 1.202714517481950, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep1.get_cv(0, 2), Point(1.024939251342349, 2.995270183326687, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep1.get_cv(0, 3), Point(1.646130147308625, 5.890456645823520, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep1.get_cv(0, 4), Point(2.268126490080245, 7.550043751484516, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep1.get_cv(0, 5), Point(2.795046402150731, 8.602476824301650, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep1.get_cv(1, 0), Point(0.888888888888889, 0.000000000000000, -1.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep1.get_cv(4, 0), Point(-1.111111111111111, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep1.get_cv(8, 0), Point(0.888888888888889, 0.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep1.get_cv(8, 5), Point(2.795046402150731, 8.602476824301650, 0.000000000000000)))

    MINI_CHECK(s_sweep2.is_valid())
    MINI_CHECK(s_sweep2.cv_count_dir(0) == 3)
    MINI_CHECK(s_sweep2.cv_count_dir(1) == 6)
    MINI_CHECK(m_sweep2.number_of_vertices() > 0)
    MINI_CHECK(m_sweep2.number_of_faces() > 0)
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep2.get_cv(0, 0), Point(6.000000000000000, -1.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep2.get_cv(0, 5), Point(8.000000000000000, 4.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep2.get_cv(2, 0), Point(9.999999999999998, -1.000000000000000, 0.000000000000000)))
    MINI_CHECK(TOLERANCE.is_point_close(s_sweep2.get_cv(2, 5), Point(8.999999999999998, 4.000000000000000, 0.000000000000000)))


@MINI_TEST("Primitives", "Nurbssurface_edge")
def test_nurbssurface_edge():
    from session_py import Primitives
    from session_py import NurbsCurve
    from session_py import Point

    pts_south = [Point(1.0, 20.569076, 0.0), Point(1.0, 22.569076, 3.0), Point(1.0, 25.569076, 3.0), Point(1.0, 27.569076, 0.0)]
    pts_west  = [Point(10.0, 20.569076, 0.0), Point(5.5, 20.569076, 3.5), Point(1.0, 20.569076, 0.0)]
    pts_north = [Point(10.0, 20.569076, 0.0), Point(10.0, 22.569076, 3.0), Point(10.0, 25.569076, 3.0), Point(10.0, 27.569076, 0.0)]
    pts_east  = [Point(10.0, 27.569076, 0.0), Point(5.5, 27.569076, 3.5), Point(1.0, 27.569076, 0.0)]

    south = NurbsCurve.create(False, 3, pts_south)
    west  = NurbsCurve.create(False, 2, pts_west)
    north = NurbsCurve.create(False, 3, pts_north)
    east  = NurbsCurve.create(False, 2, pts_east)

    surf = Primitives.create_edge(south, west, north, east)
    m = surf.mesh()

    MINI_CHECK(surf.is_valid())
    MINI_CHECK(m.number_of_faces() > 0)
    MINI_CHECK(surf.degree(0) == 2)
    MINI_CHECK(surf.degree(1) == 3)
    MINI_CHECK(surf.cv_count_dir(0) == 3)
    MINI_CHECK(surf.cv_count_dir(1) == 4)

    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(0, 0), Point(1.0, 20.569076, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(0, 1), Point(1.0, 22.569076, 3.0)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(0, 2), Point(1.0, 25.569076, 3.0)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(0, 3), Point(1.0, 27.569076, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(1, 0), Point(5.5, 20.569076, 3.5)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(1, 1), Point(5.5, 22.569076, 6.5)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(1, 2), Point(5.5, 25.569076, 6.5)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(1, 3), Point(5.5, 27.569076, 3.5)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(2, 0), Point(10.0, 20.569076, 0.0)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(2, 1), Point(10.0, 22.569076, 3.0)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(2, 2), Point(10.0, 25.569076, 3.0)))
    MINI_CHECK(TOLERANCE.is_point_close(surf.get_cv(2, 3), Point(10.0, 27.569076, 0.0)))


###########################################################################################
# Surface-to-mesh subdivision
###########################################################################################

@MINI_TEST("Primitives", "Mesh_quad_mesh")
def test_mesh_quad_mesh():
    from session_py import Primitives

    cyl = Primitives.cylinder_surface(0, 0, 0, 1.0, 5.0)
    m = Primitives.quad_mesh(cyl, 8, 4)
    MINI_CHECK(m.number_of_vertices() == 40)
    MINI_CHECK(m.number_of_faces() == 32)
    MINI_CHECK(m.is_valid())

    sph = Primitives.sphere_surface(0, 0, 0, 3.0)
    m2 = Primitives.quad_mesh(sph, 8, 4)
    MINI_CHECK(m2.number_of_vertices() == 26)
    MINI_CHECK(m2.number_of_faces() == 32)
    MINI_CHECK(m2.is_valid())


@MINI_TEST("Primitives", "Mesh_diamond_mesh")
def test_mesh_diamond_mesh():
    from session_py import Primitives

    cyl = Primitives.cylinder_surface(0, 0, 0, 1.0, 5.0)
    m = Primitives.diamond_mesh(cyl, 8, 4)
    MINI_CHECK(m.number_of_vertices() == 40)
    MINI_CHECK(m.number_of_faces() == 20)
    MINI_CHECK(m.is_valid())

    sph = Primitives.sphere_surface(0, 0, 0, 3.0)
    m2 = Primitives.diamond_mesh(sph, 8, 4)
    MINI_CHECK(m2.number_of_vertices() == 26)
    MINI_CHECK(m2.number_of_faces() == 12)
    MINI_CHECK(m2.is_valid())


@MINI_TEST("Primitives", "Mesh_hex_mesh")
def test_mesh_hex_mesh():
    from session_py import Primitives

    cyl = Primitives.cylinder_surface(0, 0, 0, 1.0, 5.0)
    m = Primitives.hex_mesh(cyl, 6, 4, 1.0/3.0)
    MINI_CHECK(m.number_of_vertices() == 78)
    MINI_CHECK(m.number_of_faces() == 15)
    MINI_CHECK(m.is_valid())

    sph = Primitives.sphere_surface(0, 0, 0, 3.0)
    m2 = Primitives.hex_mesh(sph, 6, 4, 1.0/3.0)
    MINI_CHECK(m2.number_of_vertices() == 68)
    MINI_CHECK(m2.number_of_faces() == 15)
    MINI_CHECK(m2.is_valid())



@MINI_TEST("Primitives", "Mesh_cone_subdivisions")
def test_mesh_cone_subdivisions():
    from session_py import Primitives

    cone = Primitives.cone_surface(0, 0, 0, 3.0, 5.0)

    m1 = Primitives.quad_mesh(cone, 8, 4)
    MINI_CHECK(m1.number_of_vertices() == 33)
    MINI_CHECK(m1.number_of_faces() == 32)
    MINI_CHECK(m1.is_valid())

    m2 = Primitives.diamond_mesh(cone, 8, 4)
    MINI_CHECK(m2.number_of_vertices() == 33)
    MINI_CHECK(m2.number_of_faces() == 16)
    MINI_CHECK(m2.is_valid())

    m3 = Primitives.hex_mesh(cone, 6, 4, 1.0/3.0)
    MINI_CHECK(m3.number_of_vertices() == 73)
    MINI_CHECK(m3.number_of_faces() == 15)
    MINI_CHECK(m3.is_valid())


@MINI_TEST("Primitives", "Nurbscurve_interpolated")
def test_nurbscurve_interpolated():
    from session_py import Primitives
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import knot

    points = [
        Point(14, 9, 0), Point(15.342777, 13.734889, 0), Point(21.897914, 32.239195, 0),
        Point(24.678472, 0.354555, 0), Point(33.813678, 24.76858, 0),
        Point(39.626394, 15.47249, 0), Point(41, 13, 0)
    ]

    c = Primitives.create_interpolated(points, knot.CurveKnotStyle.Chord)

    MINI_CHECK(c.is_valid())
    MINI_CHECK(c.degree() == 3)
    MINI_CHECK(c.order() == 4)
    MINI_CHECK(c.cv_count() == 9)
    MINI_CHECK(c.is_rational() == False)

    d0, d1 = c.domain()
    knots = c.get_knots()
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(d0), points[0]))
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(knots[3]), points[1]))
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(knots[4]), points[2]))
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(knots[5]), points[3]))
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(knots[6]), points[4]))
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(knots[7]), points[5]))
    MINI_CHECK(TOLERANCE.is_point_close(c.point_at(d1), points[6]))

    MINI_CHECK(TOLERANCE.is_point_close(c.get_cv(0), points[0]))
    MINI_CHECK(TOLERANCE.is_point_close(c.get_cv(8), points[6]))

    pts4 = [Point(0,0,0), Point(1,2,0), Point(3,1,0), Point(5,3,0)]
    c4 = Primitives.create_interpolated(pts4, knot.CurveKnotStyle.Chord)
    MINI_CHECK(c4.is_valid())
    MINI_CHECK(c4.degree() == 3)
    MINI_CHECK(c4.cv_count() == 6)
    d4_0, d4_1 = c4.domain()
    MINI_CHECK(TOLERANCE.is_point_close(c4.point_at(d4_0), pts4[0]))
    MINI_CHECK(TOLERANCE.is_point_close(c4.point_at(d4_1), pts4[3]))


if __name__ == "__main__":
    run_all(language="python")
