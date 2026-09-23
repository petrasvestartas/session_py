from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import PI
import math
import os
from pathlib import Path


def _serialization_path(name: str) -> str:
    """Path of a file in the serialization folder, created when missing."""

    folder = Path(__file__).resolve().parents[2] / "serialization"
    folder.mkdir(parents=True, exist_ok=True)

    return str(folder / name)


@MINI_TEST("FileStep", "NurbsCurve Round Trip")
def test_nurbscurve_round_trip():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import file_step

    path = _serialization_path("test_step_nurbscurve.step")
    pts = [
        Point(0, 0, 0),
        Point(1, 2, 0),
        Point(2, 2, 0),
        Point(3, 0, 0),
    ]
    nc = NurbsCurve.create(False, 3, pts)

    MINI_CHECK(nc.is_valid())
    MINI_CHECK(nc.degree() == 3)
    MINI_CHECK(nc.cv_count() == 4)

    file_step.write_file_step_nurbscurves([nc], path)

    MINI_CHECK(os.path.exists(path))

    curves = file_step.read_file_step_nurbscurves(path)

    MINI_CHECK(len(curves) >= 1)

    back = curves[0]

    MINI_CHECK(back.is_valid())
    MINI_CHECK(back.degree() == 3)
    MINI_CHECK(back.cv_count() == 4)
    MINI_CHECK(back.m_is_rat == 0)

    kn_orig = nc.get_nurbsknots()
    kn_back = back.get_nurbsknots()

    MINI_CHECK(len(kn_orig) == len(kn_back))

    for i in range(min(len(kn_orig), len(kn_back))):
        MINI_CHECK(abs(kn_orig[i] - kn_back[i]) < 1e-10)

    for i in range(4):
        p_orig = nc.get_cv(i)
        p_back = back.get_cv(i)

        MINI_CHECK(abs(p_orig[0] - p_back[0]) < 1e-10)
        MINI_CHECK(abs(p_orig[1] - p_back[1]) < 1e-10)
        MINI_CHECK(abs(p_orig[2] - p_back[2]) < 1e-10)

    os.remove(path)


@MINI_TEST("FileStep", "NurbsCurve Rational Round Trip")
def test_nurbscurve_rational_round_trip():
    import numpy as np
    from session_py import NurbsCurve
    from session_py import file_step

    path = _serialization_path("test_step_nurbscurve_rat.step")
    nc = NurbsCurve(3, True, 3, 3)
    w_mid = math.cos(PI / 4.0)
    nc.m_nurbsknot = np.array([0.0, 0.0, 1.0, 1.0])
    cv = nc.m_cv
    cv[0] = 1.0
    cv[1] = 0.0
    cv[2] = 0.0
    cv[3] = 1.0
    cv[4] = w_mid * 1.0
    cv[5] = w_mid * 1.0
    cv[6] = 0.0
    cv[7] = w_mid
    cv[8] = 0.0
    cv[9] = 1.0
    cv[10] = 0.0
    cv[11] = 1.0

    MINI_CHECK(nc.is_valid())
    MINI_CHECK(nc.degree() == 2)
    MINI_CHECK(nc.cv_count() == 3)
    MINI_CHECK(nc.m_is_rat == 1)

    file_step.write_file_step_nurbscurves([nc], path)

    MINI_CHECK(os.path.exists(path))

    curves = file_step.read_file_step_nurbscurves(path)

    MINI_CHECK(len(curves) >= 1)

    back = curves[0]

    MINI_CHECK(back.is_valid())
    MINI_CHECK(back.degree() == 2)
    MINI_CHECK(back.cv_count() == 3)
    MINI_CHECK(back.m_is_rat == 1)

    cv_back = back.m_cv
    s = back.m_cv_stride

    for i in range(3):
        w_orig = cv[i * 4 + 3]
        w_back = cv_back[i * s + 3]

        MINI_CHECK(abs(w_orig - w_back) < 1e-10)

        if abs(w_orig) > 1e-12 and abs(w_back) > 1e-12:
            MINI_CHECK(
                abs(cv[i * 4 + 0] / w_orig - cv_back[i * s + 0] / w_back) < 1e-10
            )
            MINI_CHECK(
                abs(cv[i * 4 + 1] / w_orig - cv_back[i * s + 1] / w_back) < 1e-10
            )

    os.remove(path)


@MINI_TEST("FileStep", "NurbsSurface Round Trip")
def test_nurbssurface_round_trip():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py import file_step

    path = _serialization_path("test_step_nurbssurface.step")
    pts = []

    for u in range(4):
        for v in range(4):
            pts.append(Point(float(u), float(v), math.sin(u + v) * 0.5))

    srf = NurbsSurface.create(False, False, 3, 3, 4, 4, pts)

    MINI_CHECK(srf.is_valid())
    MINI_CHECK(srf.degree(0) == 3)
    MINI_CHECK(srf.degree(1) == 3)
    MINI_CHECK(srf.cv_count(0) == 4)
    MINI_CHECK(srf.cv_count(1) == 4)

    file_step.write_file_step_nurbssurfaces([srf], path)

    MINI_CHECK(os.path.exists(path))

    surfaces = file_step.read_file_step_nurbssurfaces(path)

    MINI_CHECK(len(surfaces) >= 1)

    back = surfaces[0]

    MINI_CHECK(back.is_valid())
    MINI_CHECK(back.degree(0) == 3)
    MINI_CHECK(back.degree(1) == 3)
    MINI_CHECK(back.cv_count(0) == 4)
    MINI_CHECK(back.cv_count(1) == 4)
    MINI_CHECK(back.m_is_rat == 0)

    ku_orig = srf.m_nurbsknot[0]
    kv_orig = srf.m_nurbsknot[1]
    ku_back = back.m_nurbsknot[0]
    kv_back = back.m_nurbsknot[1]

    MINI_CHECK(len(ku_orig) == len(ku_back))
    MINI_CHECK(len(kv_orig) == len(kv_back))

    for i in range(min(len(ku_orig), len(ku_back))):
        MINI_CHECK(abs(ku_orig[i] - ku_back[i]) < 1e-10)

    for u in range(4):
        for v in range(4):
            p_orig = srf.get_cv(u, v)
            p_back = back.get_cv(u, v)

            MINI_CHECK(abs(p_orig[0] - p_back[0]) < 1e-10)
            MINI_CHECK(abs(p_orig[1] - p_back[1]) < 1e-10)
            MINI_CHECK(abs(p_orig[2] - p_back[2]) < 1e-10)

    os.remove(path)


@MINI_TEST("FileStep", "NurbsSurface Rational Round Trip")
def test_nurbssurface_rational_round_trip():
    import numpy as np
    from session_py import NurbsSurface
    from session_py import file_step

    path = _serialization_path("test_step_nurbssurface_rat.step")
    srf = NurbsSurface(3, True, 3, 3, 3, 3)
    srf.m_nurbsknot[0] = np.array([0.0, 0.0, 1.0, 1.0])
    srf.m_nurbsknot[1] = np.array([0.0, 0.0, 1.0, 1.0])

    w = 0.8

    for u in range(3):
        for v in range(3):
            x = float(u)
            y = float(v)
            z = math.sin(u + v) * 0.3
            srf.set_cv_4d(u, v, w * x, w * y, w * z, w)

    MINI_CHECK(srf.is_valid())
    MINI_CHECK(srf.m_is_rat == 1)

    file_step.write_file_step_nurbssurfaces([srf], path)

    MINI_CHECK(os.path.exists(path))

    surfaces = file_step.read_file_step_nurbssurfaces(path)

    MINI_CHECK(len(surfaces) >= 1)

    back = surfaces[0]

    MINI_CHECK(back.is_valid())
    MINI_CHECK(back.degree(0) == 2)
    MINI_CHECK(back.degree(1) == 2)
    MINI_CHECK(back.cv_count(0) == 3)
    MINI_CHECK(back.cv_count(1) == 3)
    MINI_CHECK(back.m_is_rat == 1)

    for u in range(3):
        for v in range(3):
            _, x1, y1, z1, w1 = srf.get_cv_4d(u, v)
            _, x2, y2, z2, w2 = back.get_cv_4d(u, v)

            MINI_CHECK(abs(w1 - w2) < 1e-10)

            if abs(w1) > 1e-12 and abs(w2) > 1e-12:
                MINI_CHECK(abs(x1 / w1 - x2 / w2) < 1e-10)
                MINI_CHECK(abs(y1 / w1 - y2 / w2) < 1e-10)

    os.remove(path)


@MINI_TEST("FileStep", "NurbsSurfaceTrimmed Round Trip")
def test_nurbssurface_trimmed_round_trip():
    import numpy as np
    from session_py import NurbsCurve
    from session_py import NurbsSurface
    from session_py import NurbsSurfaceTrimmed
    from session_py import Point
    from session_py import file_step

    path = _serialization_path("test_step_nurbssurface_trimmed.step")
    pts = []

    for u in range(4):
        for v in range(4):
            pts.append(Point(float(u), float(v), 0.0))

    srf = NurbsSurface.create(False, False, 3, 3, 4, 4, pts)
    loop_pts = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
        Point(0, 0, 0),
    ]
    outer = NurbsCurve(2, False, 2, 5)
    outer.m_nurbsknot = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    cv = outer.m_cv

    for i in range(5):
        cv[i * 2 + 0] = loop_pts[i][0]
        cv[i * 2 + 1] = loop_pts[i][1]

    MINI_CHECK(outer.is_valid())

    trimmed = NurbsSurfaceTrimmed.create(srf, outer)

    MINI_CHECK(trimmed.m_surface.is_valid())

    file_step.write_file_step_nurbssurfaces_trimmed([trimmed], path)

    MINI_CHECK(os.path.exists(path))

    surfaces = file_step.read_file_step_nurbssurfaces(path)

    MINI_CHECK(len(surfaces) >= 1)

    back_srf = surfaces[0]

    MINI_CHECK(back_srf.is_valid())
    MINI_CHECK(back_srf.degree(0) == 3)
    MINI_CHECK(back_srf.degree(1) == 3)
    MINI_CHECK(back_srf.cv_count(0) == 4)
    MINI_CHECK(back_srf.cv_count(1) == 4)

    ncurves = file_step.read_file_step_nurbscurves(path)

    MINI_CHECK(len(ncurves) >= 1)

    os.remove(path)


@MINI_TEST("FileStep", "BRep Read Schoring")
def test_brep_read_schoring():
    from session_py import file_step

    step_path = (
        Path(__file__).resolve().parents[3]
        / "session_data"
        / "elements"
        / "schoring_foot_0.step"
    )

    if not step_path.exists():
        return

    breps = file_step.read_file_step_breps(str(step_path))

    MINI_CHECK(len(breps) == 3)

    total_faces = 0
    total_edges = 0
    total_verts = 0

    for b in breps:
        total_faces += b.face_count()
        total_edges += b.edge_count()
        total_verts += b.vertex_count()

    MINI_CHECK(total_faces == 38)
    MINI_CHECK(total_edges == 103)
    MINI_CHECK(total_verts == 74)

    for b in breps:
        MINI_CHECK(b.is_valid())
        MINI_CHECK(len(b.m_surfaces) == b.face_count())
        MINI_CHECK(len(b.m_curves_3d) == b.edge_count())
        MINI_CHECK(b.shell_count() == 1 and b.solid_count() == 1)

        for e in b.m_edges:
            MINI_CHECK(len(e.pcurves) > 0)

    pts = file_step.read_file_step_points(str(step_path))

    MINI_CHECK(len(pts) == 350)


@MINI_TEST("FileStep", "BRep Round Trip")
def test_brep_round_trip():
    from session_py import BRep
    from session_py import file_step

    path = _serialization_path("test_brep_roundtrip.step")
    cyl = BRep.create_cylinder(1.0, 2.0)
    cyl.name = "cylinder"

    file_step.write_file_step_brep(cyl, path)

    breps = file_step.read_file_step_breps(path)

    MINI_CHECK(len(breps) == 1)
    MINI_CHECK(breps[0].is_valid())
    MINI_CHECK(breps[0].face_count() == 3)
    MINI_CHECK(breps[0].edge_count() == 3)
    MINI_CHECK(breps[0].vertex_count() == 2)
    MINI_CHECK(breps[0].is_solid())
    MINI_CHECK(abs(breps[0].volume() - cyl.volume()) < 0.05 * cyl.volume())

    os.remove(path)


if __name__ == "__main__":
    run_all(language="python")
