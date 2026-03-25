from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


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
        Point(0, 0, 0), Point(0, 4, 0),
        Point(4, 0, 0), Point(4, 4, 0),
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
        Point(0, 0, 0), Point(2, 4, 0),
        Point(4, 0, 0), Point(2, 4, 0),
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


if __name__ == "__main__":
    run_all(language="python")
