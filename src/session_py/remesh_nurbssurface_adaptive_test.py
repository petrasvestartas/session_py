from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("RemeshNurbssurfaceAdaptive", "Constructor")
def test_remesh_nurbssurface_adaptive_constructor():
    from session_py import RemeshNurbssurfaceAdaptive
    from session_py import Primitives

    s = Primitives.sphere_surface(0, 0, 0, 1.0)
    ta = RemeshNurbssurfaceAdaptive(s)

    MINI_CHECK(ta.get_max_angle() == 20.0)
    MINI_CHECK(ta.get_max_edge_length() == 0.0)
    MINI_CHECK(ta.get_min_edge_length() == 0.0)
    MINI_CHECK(ta.get_max_chord_height() == 0.0)


@MINI_TEST("RemeshNurbssurfaceAdaptive", "Parameters")
def test_remesh_nurbssurface_adaptive_parameters():
    from session_py import RemeshNurbssurfaceAdaptive
    from session_py import Primitives

    s = Primitives.sphere_surface(0, 0, 0, 1.0)
    ta = RemeshNurbssurfaceAdaptive(s)
    ta.set_max_angle(15.0) \
      .set_max_edge_length(2.0) \
      .set_min_edge_length(0.1) \
      .set_max_chord_height(0.05)

    MINI_CHECK(ta.get_max_angle() == 15.0)
    MINI_CHECK(ta.get_max_edge_length() == 2.0)
    MINI_CHECK(ta.get_min_edge_length() == 0.1)
    MINI_CHECK(ta.get_max_chord_height() == 0.05)


@MINI_TEST("RemeshNurbssurfaceAdaptive", "Mesh")
def test_remesh_nurbssurface_adaptive_mesh():
    from session_py import RemeshNurbssurfaceAdaptive
    from session_py import Primitives

    s = Primitives.sphere_surface(0, 0, 0, 1.0)
    ta = RemeshNurbssurfaceAdaptive(s)
    m = ta.mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() > 0)
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("RemeshNurbssurfaceAdaptive", "Torus")
def test_remesh_nurbssurface_adaptive_torus():
    from session_py import RemeshNurbssurfaceAdaptive
    from session_py import Primitives

    s = Primitives.torus_surface(0, 0, 0, 3.0, 1.0)
    m = RemeshNurbssurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() > 0)
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("RemeshNurbssurfaceAdaptive", "Cylinder")
def test_remesh_nurbssurface_adaptive_cylinder():
    from session_py import RemeshNurbssurfaceAdaptive
    from session_py import Primitives

    s = Primitives.cylinder_surface(0, 0, 0, 1.0, 5.0)
    m = RemeshNurbssurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() > 0)
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("RemeshNurbssurfaceAdaptive", "Cone")
def test_remesh_nurbssurface_adaptive_cone():
    from session_py import RemeshNurbssurfaceAdaptive
    from session_py import Primitives

    s = Primitives.cone_surface(0, 0, 0, 1.0, 5.0)
    m = RemeshNurbssurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() > 0)
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("RemeshNurbssurfaceAdaptive", "Doubly Curved")
def test_remesh_nurbssurface_adaptive_doubly_curved():
    from session_py import RemeshNurbssurfaceAdaptive
    from session_py import Primitives

    s = Primitives.wave_surface(1.0, 0.5)
    m = RemeshNurbssurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() > 0)
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("RemeshNurbssurfaceAdaptive", "Flat")
def test_remesh_nurbssurface_adaptive_flat():
    from session_py import RemeshNurbssurfaceAdaptive
    from session_py import Primitives

    s = Primitives.wave_surface(1.0, 0.0)
    m = RemeshNurbssurfaceAdaptive(s).mesh()

    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() > 0)
    MINI_CHECK(m.number_of_faces() > 0)


if __name__ == "__main__":
    run_all(language="python")
