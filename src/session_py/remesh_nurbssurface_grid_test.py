from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import Tolerance


@MINI_TEST("RemeshNurbsSurfaceGrid", "Singular Planar Normal")
def test_remesh_nurbssurface_grid_singular_planar_normal():
    """U-collapsed corners retain their fan normal on every nondegenerate triangle."""
    from session_py import NurbsSurface, Point, RemeshNurbsSurfaceGrid
    surface = NurbsSurface.create(False,False,1,1,2,2,[
        Point(0,0,1),Point(0,0,1),Point(-1,0,0),Point(1,0,0)])
    mesh = RemeshNurbsSurfaceGrid.from_u_v_q(surface,0,0,5.0,0.001)
    apex = False
    for face in mesh.face.values():
        a,b,c = [mesh.vertex[key] for key in face]
        if abs((b.x-a.x)*(c.z-a.z)-(b.z-a.z)*(c.x-a.x)) <= 1e-14:
            continue
        for vertex in (a,b,c):
            normal = vertex.normal()
            MINI_CHECK(abs(normal[0]) < 1e-12 and abs(normal[2]) < 1e-12)
            MINI_CHECK(abs(abs(normal[1])-1.0) < 1e-12)
            apex = apex or vertex.z == 1.0
    MINI_CHECK(apex)


@MINI_TEST("RemeshNurbsSurfaceGrid", "Analytic Normals")
def test_remesh_nurbssurface_grid_analytic_normals():
    from session_py import RemeshNurbsSurfaceGrid
    from session_py import Primitives

    surfaces = [
        Primitives.sphere_surface(0.0, 0.0, 0.0, 1.0),
        Primitives.cylinder_surface(0.0, 0.0, 0.0, 1.0, 5.0),
        Primitives.cone_surface(0.0, 0.0, 0.0, 1.0, 5.0),
    ]
    for index, s in enumerate(surfaces):
        m = RemeshNurbsSurfaceGrid.from_u_v_q(s, 0, 0, 30.0, 0.01)
        for vd in m.vertex.values():
            n = vd.normal()
            length = n[0]*n[0] + n[1]*n[1] + n[2]*n[2]
            MINI_CHECK(abs(length - 1.0) < Tolerance.ZERO_TOLERANCE)
            if index < 2:
                z = vd.z if index == 0 else 0.0
                dot = vd.x*n[0] + vd.y*n[1] + z*n[2]
                MINI_CHECK(abs(dot - 1.0) < Tolerance.ZERO_TOLERANCE)


@MINI_TEST("RemeshNurbsSurfaceGrid", "Sphere")
def test_remesh_nurbs_surface_grid_sphere():
    from session_py import RemeshNurbsSurfaceGrid
    from session_py import Primitives

    s = Primitives.sphere_surface(0, 0, 0, 1.0)
    m = RemeshNurbsSurfaceGrid.from_u_v(s, 0, 0)

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 191)
    MINI_CHECK(m.number_of_faces() == 378)


@MINI_TEST("RemeshNurbsSurfaceGrid", "Torus")
def test_remesh_nurbs_surface_grid_torus():
    from session_py import RemeshNurbsSurfaceGrid
    from session_py import Primitives

    s = Primitives.torus_surface(0, 0, 0, 3.0, 1.0)
    m = RemeshNurbsSurfaceGrid.from_u_v(s, 0, 0)

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 693)
    MINI_CHECK(m.number_of_faces() == 1386)


@MINI_TEST("RemeshNurbsSurfaceGrid", "Cylinder")
def test_remesh_nurbs_surface_grid_cylinder():
    from session_py import RemeshNurbsSurfaceGrid
    from session_py import Primitives

    s = Primitives.cylinder_surface(0, 0, 0, 1.0, 5.0)
    m = RemeshNurbsSurfaceGrid.from_u_v(s, 0, 0)

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 42)
    MINI_CHECK(m.number_of_faces() == 42)


@MINI_TEST("RemeshNurbsSurfaceGrid", "Cone")
def test_remesh_nurbs_surface_grid_cone():
    from session_py import RemeshNurbsSurfaceGrid
    from session_py import Primitives

    s = Primitives.cone_surface(0, 0, 0, 1.0, 5.0)
    m = RemeshNurbsSurfaceGrid.from_u_v(s, 0, 0)

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 22)
    MINI_CHECK(m.number_of_faces() == 21)


@MINI_TEST("RemeshNurbsSurfaceGrid", "Doubly Curved")
def test_remesh_nurbs_surface_grid_doubly_curved():
    from session_py import RemeshNurbsSurfaceGrid
    from session_py import Primitives

    s = Primitives.wave_surface(1.0, 0.5)
    m = RemeshNurbsSurfaceGrid.from_u_v(s, 0, 0)

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 961)
    MINI_CHECK(m.number_of_faces() == 1800)


@MINI_TEST("RemeshNurbsSurfaceGrid", "Grid Target")
def test_remesh_nurbs_surface_grid_grid_target():
    from session_py import RemeshNurbsSurfaceGrid
    from session_py import Primitives

    s = Primitives.wave_surface(1.0, 0.5)
    m_lo = RemeshNurbsSurfaceGrid.from_u_v(s, 8, 8)
    m_hi = RemeshNurbsSurfaceGrid.from_u_v(s, 32, 32)

    MINI_CHECK(m_lo.is_valid())
    MINI_CHECK(m_lo.number_of_vertices() == 64)
    MINI_CHECK(m_hi.is_valid())
    MINI_CHECK(m_hi.number_of_vertices() > m_lo.number_of_vertices())


@MINI_TEST("RemeshNurbsSurfaceGrid", "Flat Quad")
def test_remesh_nurbs_surface_grid_flat_quad():
    from session_py import RemeshNurbsSurfaceGrid
    from session_py import NurbsSurface
    from session_py import Point

    s = NurbsSurface.create(False, False, 1, 1, 2, 2, [
        Point(0, 0, 0),
        Point(0, 4, 0),
        Point(4, 0, 0),
        Point(4, 4, 0),
    ])
    m = RemeshNurbsSurfaceGrid.from_u_v(s, 0, 0)

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 4)
    MINI_CHECK(m.number_of_faces() == 2)


@MINI_TEST("RemeshNurbsSurfaceGrid", "Flat Triangle")
def test_remesh_nurbs_surface_grid_flat_triangle():
    from session_py import RemeshNurbsSurfaceGrid
    from session_py import NurbsSurface
    from session_py import Point

    s = NurbsSurface.create(False, False, 1, 1, 2, 2, [
        Point(0, 0, 0),
        Point(2, 4, 0),
        Point(4, 0, 0),
        Point(2, 4, 0),
    ])
    m = RemeshNurbsSurfaceGrid.from_u_v(s, 0, 0)

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 3)
    MINI_CHECK(m.number_of_faces() == 1)


@MINI_TEST("RemeshNurbsSurfaceGrid", "Double-Curved Triangle")
def test_remesh_nurbs_surface_grid_double_curved_triangle():
    from session_py import RemeshNurbsSurfaceGrid
    from session_py import NurbsSurface
    from session_py import Point

    s = NurbsSurface.create(False, False, 2, 2, 3, 3, [
        Point(0, 0, 0),
        Point(2, 0, 3),
        Point(4, 0, 0),
        Point(0, 2, 2),
        Point(2, 2, 5),
        Point(4, 2, 2),
        Point(2, 4, 0),
        Point(2, 4, 0),
        Point(2, 4, 0),
    ])
    m = RemeshNurbsSurfaceGrid.from_u_v(s, 0, 0)

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 64)
    MINI_CHECK(m.number_of_faces() == 98)



@MINI_TEST("RemeshNurbsSurfaceGrid", "Crease Normals")
def test_remesh_nurbssurface_grid_crease_normals():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import RemeshNurbsSurfaceGrid
    import math
    s = NurbsSurface.create(False, False, 1, 1, 3, 2, [
        Point(0.0, 0.0, 0.0), Point(0.0, 1.0, 0.0),
        Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0),
        Point(2.0, 0.0, 1.0), Point(2.0, 1.0, 1.0),
    ])
    m = RemeshNurbsSurfaceGrid.from_u_v(s, 0, 0)
    MINI_CHECK(len(m.vertex) == 8)
    MINI_CHECK(len(m.face) == 4)
    flat = tilted = 0
    for vd in m.vertex.values():
        if vd.x != 1.0:
            continue
        n = vd.normal()
        if abs(n[0]) < Tolerance.ZERO_TOLERANCE:
            flat += 1
        if abs(n[0] + math.sqrt(0.5)) < Tolerance.ZERO_TOLERANCE:
            tilted += 1
    MINI_CHECK(flat == 2 and tilted == 2)


if __name__ == "__main__":
    run_all(language="python")
