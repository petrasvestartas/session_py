from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("MeshIso", "Eval Gyroid")
def test_mesh_iso_eval_gyroid():
    from session_py import MeshIso
    from session_py import TpmsType

    MINI_CHECK(TOLERANCE.is_close(MeshIso.eval(TpmsType.GYROID, PI / 2.0, 0.0, 0.0, 1.0), 1.0))
    MINI_CHECK(TOLERANCE.is_close(MeshIso.eval(TpmsType.SCHWARZ_P, 0.0, 0.0, 0.0, 1.0), 3.0))


@MINI_TEST("MeshIso", "Eval SchwarzP")
def test_mesh_iso_eval_schwarz_p():
    from session_py import MeshIso
    from session_py import TpmsType

    MINI_CHECK(TOLERANCE.is_close(MeshIso.eval(TpmsType.SCHWARZ_P, PI, PI, PI, 1.0), -3.0))


@MINI_TEST("MeshIso", "Eval Diamond")
def test_mesh_iso_eval_diamond():
    from session_py import MeshIso
    from session_py import TpmsType

    MINI_CHECK(TOLERANCE.is_close(MeshIso.eval(TpmsType.DIAMOND, 0.0, 0.0, 0.0, 1.0), 0.0))


@MINI_TEST("MeshIso", "From Tpms Gyroid Solid")
def test_mesh_iso_from_tpms_gyroid_solid():
    from session_py import MeshIso
    from session_py import TpmsType
    from session_py import TpmsMode
    from session_py import BoundingBox
    from session_py import Point

    box = BoundingBox.from_points([Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)])
    m = MeshIso.from_tpms(TpmsType.GYROID, box, 10, 10, 10, 0.0, 1.0, TpmsMode.SOLID)
    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() > 0)
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("MeshIso", "From Tpms Diamond Sheet")
def test_mesh_iso_from_tpms_diamond_sheet():
    from session_py import MeshIso
    from session_py import TpmsType
    from session_py import TpmsMode
    from session_py import BoundingBox
    from session_py import Point

    box = BoundingBox.from_points([Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)])
    m = MeshIso.from_tpms(TpmsType.DIAMOND, box, 10, 10, 10, 0.0, 1.0, TpmsMode.SHEET, 0.1)
    MINI_CHECK(m.is_valid())


@MINI_TEST("MeshIso", "From Tpms Neovius Shell")
def test_mesh_iso_from_tpms_neovius_shell():
    from session_py import MeshIso
    from session_py import TpmsType
    from session_py import TpmsMode
    from session_py import BoundingBox
    from session_py import Point

    box = BoundingBox.from_points([Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)])
    m = MeshIso.from_tpms(TpmsType.NEOVIUS, box, 10, 10, 10, 0.0, 1.0, TpmsMode.SHELL, 0.1)
    MINI_CHECK(m.is_valid())


@MINI_TEST("MeshIso", "SDF Sphere")
def test_mesh_iso_sdf_sphere():
    from session_py import MeshIso

    MINI_CHECK(TOLERANCE.is_close(MeshIso.sdf_sphere(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0), 0.0))
    MINI_CHECK(TOLERANCE.is_close(MeshIso.sdf_sphere(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), -1.0))
    MINI_CHECK(TOLERANCE.is_close(MeshIso.sdf_sphere(0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0), 1.0))


@MINI_TEST("MeshIso", "Smooth Union")
def test_mesh_iso_smooth_union():
    from session_py import MeshIso

    MINI_CHECK(MeshIso.smooth_union(-1.0, 1.0, 8.0) < 0.0)
    MINI_CHECK(MeshIso.smooth_union(1.0, 1.0, 8.0) < 1.0)


@MINI_TEST("MeshIso", "From Function")
def test_mesh_iso_from_function():
    from session_py import MeshIso
    from session_py import BoundingBox
    from session_py import Point

    box = BoundingBox.from_points([Point(-2.0, -2.0, -2.0), Point(2.0, 2.0, 2.0)])
    def fn(x, y, z):
        return MeshIso.sdf_sphere(0.0, 0.0, 0.0, 1.0, x, y, z)
    m = MeshIso.from_function(fn, box, 10, 10, 10, 0.0)
    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() > 0)


if __name__ == "__main__":
    run_all(language="python")
