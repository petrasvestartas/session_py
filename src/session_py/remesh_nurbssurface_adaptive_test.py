from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("RemeshNurbsSurfaceAdaptive", "Constructor")
def test_remesh_nurbssurface_adaptive_constructor():
    from session_py import RemeshNurbsSurfaceAdaptive
    from session_py import Primitives

    s = Primitives.sphere_surface(0, 0, 0, 1.0)
    ta = RemeshNurbsSurfaceAdaptive(s)

    MINI_CHECK(ta.get_max_angle() == 20.0)
    MINI_CHECK(ta.get_max_edge_length() == 0.0)
    MINI_CHECK(ta.get_min_edge_length() == 0.0)
    MINI_CHECK(ta.get_max_chord_height() == 0.0)


@MINI_TEST("RemeshNurbsSurfaceAdaptive", "Parameters")
def test_remesh_nurbssurface_adaptive_parameters():
    from session_py import RemeshNurbsSurfaceAdaptive
    from session_py import Primitives

    s = Primitives.sphere_surface(0, 0, 0, 1.0)
    ta = RemeshNurbsSurfaceAdaptive(s)
    ta.set_max_angle(15.0) \
      .set_max_edge_length(2.0) \
      .set_min_edge_length(0.1) \
      .set_max_chord_height(0.05)

    MINI_CHECK(ta.get_max_angle() == 15.0)
    MINI_CHECK(ta.get_max_edge_length() == 2.0)
    MINI_CHECK(ta.get_min_edge_length() == 0.1)
    MINI_CHECK(ta.get_max_chord_height() == 0.05)


@MINI_TEST("RemeshNurbsSurfaceAdaptive", "Mesh")
def test_remesh_nurbssurface_adaptive_mesh():
    from session_py import RemeshNurbsSurfaceAdaptive
    from session_py import Primitives

    s = Primitives.sphere_surface(0, 0, 0, 1.0)
    ta = RemeshNurbsSurfaceAdaptive(s)
    m = ta.mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 418)


@MINI_TEST("RemeshNurbsSurfaceAdaptive", "Torus")
def test_remesh_nurbssurface_adaptive_torus():
    from session_py import RemeshNurbsSurfaceAdaptive
    from session_py import Primitives

    s = Primitives.torus_surface(0, 0, 0, 3.0, 1.0)
    m = RemeshNurbsSurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 1024)


@MINI_TEST("RemeshNurbsSurfaceAdaptive", "Cylinder")
def test_remesh_nurbssurface_adaptive_cylinder():
    from session_py import RemeshNurbsSurfaceAdaptive
    from session_py import Primitives

    s = Primitives.cylinder_surface(0, 0, 0, 1.0, 5.0)
    m = RemeshNurbsSurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 64)


@MINI_TEST("RemeshNurbsSurfaceAdaptive", "Cone")
def test_remesh_nurbssurface_adaptive_cone():
    from session_py import RemeshNurbsSurfaceAdaptive
    from session_py import Primitives

    s = Primitives.cone_surface(0, 0, 0, 1.0, 5.0)
    m = RemeshNurbsSurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 33)


@MINI_TEST("RemeshNurbsSurfaceAdaptive", "Doubly Curved")
def test_remesh_nurbssurface_adaptive_doubly_curved():
    from session_py import RemeshNurbsSurfaceAdaptive
    from session_py import Primitives

    s = Primitives.wave_surface(1.0, 0.5)
    m = RemeshNurbsSurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 1175)


@MINI_TEST("RemeshNurbsSurfaceAdaptive", "Flat")
def test_remesh_nurbssurface_adaptive_flat():
    from session_py import RemeshNurbsSurfaceAdaptive
    from session_py import Primitives

    s = Primitives.wave_surface(1.0, 0.0)
    m = RemeshNurbsSurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 169)


@MINI_TEST("RemeshNurbsSurfaceAdaptive", "Singular Triangle")
def test_remesh_nurbssurface_adaptive_singular_triangle():
    from session_py import RemeshNurbsSurfaceAdaptive
    from session_py import NurbsSurface
    from session_py import Point

    s = NurbsSurface.create(False, False, 2, 1, 3, 2, [
        Point(0, 0, 0),
        Point(2, 0, 3),
        Point(4, 0, 0),
        Point(2, 4, 0),
        Point(2, 4, 0),
        Point(2, 4, 0),
    ])
    m = RemeshNurbsSurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 83)


@MINI_TEST("RemeshNurbsSurfaceAdaptive", "Double-Curved Triangle")
def test_remesh_nurbssurface_adaptive_double_curved_triangle():
    from session_py import RemeshNurbsSurfaceAdaptive
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
    m = RemeshNurbsSurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 91)


if __name__ == "__main__":
    run_all(language="python")
