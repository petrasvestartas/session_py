from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("NurbsSurfaceTrimmed", "Constructor")
def test_nurbssurface_trimmed_constructor():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed

    # Create a bilinear surface
    srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)
    srf.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    srf.set_cv(1, 0, Point(6.0, 0.0, 0.0))
    srf.set_cv(0, 1, Point(0.0, 6.0, 0.0))
    srf.set_cv(1, 1, Point(6.0, 6.0, 0.0))

    # Outer trim loop (rectangle in UV space)
    outer = NurbsCurve.create(True, 1, [
        Point(0.1, 0.1, 0.0), Point(0.9, 0.1, 0.0),
        Point(0.9, 0.9, 0.0), Point(0.1, 0.9, 0.0),
    ])

    ts = NurbsSurfaceTrimmed.create(srf, outer)

    # String representations
    sstr = str(ts)
    srepr = repr(ts)

    # Copy (new guid)
    tscopy = ts.duplicate()

    MINI_CHECK(ts.is_valid())
    MINI_CHECK(ts.is_trimmed())
    MINI_CHECK(ts.name == "my_nurbssurface_trimmed")
    MINI_CHECK(ts.guid)
    MINI_CHECK("NurbsSurfaceTrimmed" in sstr)
    MINI_CHECK("name=my_nurbssurface_trimmed" in srepr)
    MINI_CHECK(tscopy.is_valid())
    MINI_CHECK(tscopy.guid != ts.guid)
    MINI_CHECK(tscopy == ts)


@MINI_TEST("NurbsSurfaceTrimmed", "Constructor Planar")
def test_nurbssurface_trimmed_constructor_planar():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed
    import math

    # Planar curve boundary
    pts = [
        Point(0, 0, 0), Point(3, 1, 0), Point(5, 0.5, 0),
        Point(6, 3, 0), Point(4, 5, 0), Point(1, 4, 0),
    ]
    bnd = NurbsCurve.create(True, 3, pts)
    ts = NurbsSurfaceTrimmed.create_planar(bnd)

    # Rotated planar
    pts = [
        Point(0, 0, 0),
        Point(3, 1, -2),
        Point(5, 2, -3),
        Point(4, 4, 0),
        Point(1, 3, 2),
    ]
    bnd = NurbsCurve.create(True, 3, pts)
    ts = NurbsSurfaceTrimmed.create_planar(bnd)

    # Triangle
    bnd = NurbsCurve.create(True, 1, [Point(0, 0, 0), Point(6, 3, 3), Point(2, 5, 1)])
    ts = NurbsSurfaceTrimmed.create_planar(bnd)

    # Trapezoid
    bnd = NurbsCurve.create(True, 1, [
        Point(0, 0, 6), Point(5, 0, 6), Point(4, 4, 2), Point(1, 4, 2),
    ])
    ts = NurbsSurfaceTrimmed.create_planar(bnd)

    # Rectangle with a hole
    bnd = NurbsCurve.create(True, 1, [
        Point(0, 0, 0), Point(6, 0, 0), Point(6, 6, 0), Point(0, 6, 0),
    ])
    ts = NurbsSurfaceTrimmed.create_planar(bnd)
    ts.add_hole(NurbsCurve.create(True, 1, [
        Point(2, 2, 0), Point(4, 2, 0), Point(4, 4, 0), Point(2, 4, 0),
    ]))

    # Hexagon with 2 holes
    R = 4.0
    pts = []
    for k in range(6):
        a = k * PI / 3.0
        pts.append(Point(R * math.cos(a), R * math.sin(a), R * math.cos(a) * 0.5))
    bnd = NurbsCurve.create(True, 1, pts)
    ts = NurbsSurfaceTrimmed.create_planar(bnd)
    ts.add_holes([
        NurbsCurve.create(True, 1, [
            Point(1.5, 0.5, 0.75), Point(2.5, 0.5, 1.25), Point(2.0, 1.5, 1.0),
        ]),
        NurbsCurve.create(True, 1, [
            Point(-2, -0.5, -1), Point(-1, -0.5, -0.5), Point(-1, -1.5, -0.5), Point(-2, -1.5, -1),
        ]),
    ])


@MINI_TEST("NurbsSurfaceTrimmed", "Constructor Hole")
def test_nurbssurface_trimmed_constructor_hole():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed
    import math

    # Create surface with bump
    n = 8
    pts = []
    for i in range(n):
        for j in range(n):
            x = float(i)
            y = float(j)
            r2 = (x - 1.5) * (x - 1.5) + (y - 1.5) * (y - 1.5)
            z = 5.0 * math.exp(-r2 / 1.0) + 0.3 * math.sin(PI * x / 7) * math.sin(PI * y / 7)
            pts.append(Point(x, y, z))

    srf = NurbsSurface.create(False, False, 3, 3, n, n, pts)

    # Create outer loop (full boundary in UV)
    outer = NurbsCurve.create(True, 1, [
        Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0),
    ])

    ts = NurbsSurfaceTrimmed.create(srf, outer)

    # Add hole as UV curve directly
    hole = NurbsCurve.create(True, 1, [
        Point(0.4, 0.4, 0.0), Point(0.6, 0.4, 0.0),
        Point(0.6, 0.6, 0.0), Point(0.4, 0.6, 0.0),
    ])
    ts.add_inner_loop(hole)

    MINI_CHECK(ts.is_valid())
    MINI_CHECK(ts.is_trimmed())
    MINI_CHECK(ts.inner_loop_count() == 1)


@MINI_TEST("NurbsSurfaceTrimmed", "Accessors")
def test_nurbssurface_trimmed_accessors():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed

    srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(5, 0, 0))
    srf.set_cv(0, 1, Point(0, 5, 0))
    srf.set_cv(1, 1, Point(5, 5, 0))

    outer = NurbsCurve.create(True, 1, [
        Point(0.1, 0.1, 0), Point(0.9, 0.1, 0),
        Point(0.9, 0.9, 0), Point(0.1, 0.9, 0),
    ])

    ts = NurbsSurfaceTrimmed.create(srf, outer)
    ts.name = "test_accessors"
    ts.width = 2.5

    got_srf = ts.surface()
    got_loop = ts.get_outer_loop()

    MINI_CHECK(ts.is_valid())
    MINI_CHECK(ts.is_trimmed())
    MINI_CHECK(ts.name == "test_accessors")
    MINI_CHECK(ts.width == 2.5)
    MINI_CHECK(got_srf.is_valid())
    MINI_CHECK(got_loop.is_valid())
    MINI_CHECK(ts.inner_loop_count() == 0)


@MINI_TEST("NurbsSurfaceTrimmed", "Add Inner Loop")
def test_nurbssurface_trimmed_add_inner_loop():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed

    srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(10, 0, 0))
    srf.set_cv(0, 1, Point(0, 10, 0))
    srf.set_cv(1, 1, Point(10, 10, 0))

    outer = NurbsCurve.create(True, 1, [
        Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0),
    ])

    ts = NurbsSurfaceTrimmed.create(srf, outer)

    # Add inner loops (holes in UV)
    hole1 = NurbsCurve.create(True, 1, [
        Point(0.2, 0.2, 0), Point(0.4, 0.2, 0),
        Point(0.4, 0.4, 0), Point(0.2, 0.4, 0),
    ])
    hole2 = NurbsCurve.create(True, 1, [
        Point(0.6, 0.6, 0), Point(0.8, 0.6, 0),
        Point(0.8, 0.8, 0), Point(0.6, 0.8, 0),
    ])

    ts.add_inner_loop(hole1)
    ts.add_inner_loop(hole2)

    got = ts.get_inner_loop(0)

    MINI_CHECK(ts.inner_loop_count() == 2)
    MINI_CHECK(got.is_valid())

    ts.clear_inner_loops()
    MINI_CHECK(ts.inner_loop_count() == 0)


@MINI_TEST("NurbsSurfaceTrimmed", "Point At")
def test_nurbssurface_trimmed_point_at():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed

    srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(4, 0, 0))
    srf.set_cv(0, 1, Point(0, 4, 0))
    srf.set_cv(1, 1, Point(4, 4, 0))

    outer = NurbsCurve.create(True, 1, [
        Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0),
    ])

    ts = NurbsSurfaceTrimmed.create(srf, outer)

    u0, u1 = ts.surface().domain(0)
    v0, v1 = ts.surface().domain(1)
    u_mid = (u0 + u1) / 2.0
    v_mid = (v0 + v1) / 2.0

    pt = ts.point_at(u_mid, v_mid)
    nm = ts.normal_at(u_mid, v_mid)

    MINI_CHECK(TOLERANCE.is_close(pt[0], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(abs(nm[2]), 1.0))


@MINI_TEST("NurbsSurfaceTrimmed", "Mesh")
def test_nurbssurface_trimmed_mesh():
    import math
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed

    srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(6, 0, 0))
    srf.set_cv(0, 1, Point(0, 6, 0))
    srf.set_cv(1, 1, Point(6, 6, 0))
    outer = NurbsCurve.create(True, 1, [
        Point(0.05, 0.05, 0), Point(0.95, 0.05, 0),
        Point(0.95, 0.95, 0), Point(0.05, 0.95, 0),
    ])
    ts = NurbsSurfaceTrimmed.create(srf, outer)
    m = ts.mesh()
    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_vertices() >= 4)
    MINI_CHECK(m.number_of_faces() >= 2)
    for vd in m.vertex.values():
        nx = vd.attributes.get("nx", 0.0)
        ny = vd.attributes.get("ny", 0.0)
        nz = vd.attributes.get("nz", 0.0)
        MINI_CHECK(math.sqrt(nx*nx + ny*ny + nz*nz) > 0.5)

    bnd = NurbsCurve.create(True, 1, [
        Point(0, 0, 0), Point(6, 0, 0), Point(6, 6, 0), Point(0, 6, 0),
    ])
    ts_hole = NurbsSurfaceTrimmed.create_planar(bnd)
    ts_hole.add_hole(NurbsCurve.create(True, 1, [
        Point(2, 2, 0), Point(4, 2, 0), Point(4, 4, 0), Point(2, 4, 0),
    ]))
    mh = ts_hole.mesh()
    MINI_CHECK(not mh.is_empty())
    MINI_CHECK(mh.number_of_faces() >= 2)

    cw = math.sqrt(2.0) / 2.0
    ccx = [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0]
    ccy = [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0]
    cwt = [1.0, cw, 1.0, cw, 1.0, cw, 1.0, cw, 1.0]
    import numpy as _np
    circle_loop = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=9)
    circle_loop.m_nurbsknot = _np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], dtype=_np.float64)
    for i in range(9):
        circle_loop.set_cv_4d(i, (0.5 + 0.5 * ccx[i]) * cwt[i], (0.5 + 0.5 * ccy[i]) * cwt[i], 0.0, cwt[i])
    ts_circ = NurbsSurfaceTrimmed.create(srf, circle_loop)
    mc = ts_circ.mesh()
    MINI_CHECK(not mc.is_empty())
    MINI_CHECK(mc.number_of_vertices() >= 30)
    MINI_CHECK(mc.number_of_faces() >= 30)
    for vd in mc.vertex.values():
        nx = vd.attributes.get("nx", 0.0)
        ny = vd.attributes.get("ny", 0.0)
        nz = vd.attributes.get("nz", 0.0)
        MINI_CHECK(math.sqrt(nx*nx + ny*ny + nz*nz) > 0.5)

    n = 8
    pts = []
    for i in range(n):
        for j in range(n):
            x, y = float(i), float(j)
            r2 = (x - 1.5)**2 + (y - 1.5)**2
            z = 5.0 * math.exp(-r2) + 0.3 * math.sin(math.pi*x/7.0) * math.sin(math.pi*y/7.0)
            pts.append(Point(x, y, z))
    bump_srf = NurbsSurface.create(False, False, 3, 3, n, n, pts)
    bump_outer = NurbsCurve.create(True, 1, [
        Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0),
    ])
    ts_bump = NurbsSurfaceTrimmed.create(bump_srf, bump_outer)
    mb = ts_bump.mesh()
    MINI_CHECK(not mb.is_empty())
    MINI_CHECK(mb.number_of_vertices() >= 20)
    MINI_CHECK(mb.number_of_faces() >= 30)
    for vd in mb.vertex.values():
        nx = vd.attributes.get("nx", 0.0)
        ny = vd.attributes.get("ny", 0.0)
        nz = vd.attributes.get("nz", 0.0)
        MINI_CHECK(math.sqrt(nx*nx + ny*ny + nz*nz) > 0.5)


@MINI_TEST("NurbsSurfaceTrimmed", "Split By UV Curves")
def test_nurbssurface_trimmed_split_by_uv_curves():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed
    from session_py.primitives import Primitives

    srf = Primitives.wave_surface(10.0, 1.0)
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    pts = [Point(u0 + (u1-u0)*0.4, v0, 0.0), Point(u0 + (u1-u0)*0.6, v1, 0.0)]
    line = NurbsCurve.create(False, 1, pts)

    parts = NurbsSurfaceTrimmed.split_by_uv_curves(srf, [line])

    MINI_CHECK(len(parts) == 2)
    MINI_CHECK(parts[0].is_trimmed())
    MINI_CHECK(parts[1].is_trimmed())

    circle = Primitives.circle((u0+u1)*0.5, (v0+v1)*0.5, 0.0, (u1-u0)*0.2)

    ring = NurbsSurfaceTrimmed.split_by_uv_curves(srf, [circle])

    MINI_CHECK(len(ring) == 2)
    MINI_CHECK(ring[0].inner_loop_count() + ring[1].inner_loop_count() == 1)

    dangling = NurbsCurve.create(False, 1, [Point(3.0, 3.0, 0.0), Point(5.0, 5.0, 0.0)])

    whole = NurbsSurfaceTrimmed.split_by_uv_curves(srf, [dangling])

    MINI_CHECK(len(whole) == 1)


@MINI_TEST("NurbsSurfaceTrimmed", "Transformation")
def test_nurbssurface_trimmed_transformation():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Xform
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed

    srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(1, 0, 0))
    srf.set_cv(0, 1, Point(0, 1, 0))
    srf.set_cv(1, 1, Point(1, 1, 0))

    outer = NurbsCurve.create(True, 1, [
        Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0),
    ])

    ts = NurbsSurfaceTrimmed.create(srf, outer)
    ts_xf = Xform.translation(10.0, 20.0, 30.0)
    ts2 = ts.transformed(ts_xf)

    u0, u1 = ts2.surface().domain(0)
    v0, v1 = ts2.surface().domain(1)
    pt = ts2.point_at(u0, v0)

    MINI_CHECK(TOLERANCE.is_close(pt[0], 10.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 20.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 30.0))


@MINI_TEST("NurbsSurfaceTrimmed", "Json Roundtrip")
def test_nurbssurface_trimmed_json_roundtrip():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Color
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed
    from pathlib import Path

    srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(5, 0, 0))
    srf.set_cv(0, 1, Point(0, 5, 0))
    srf.set_cv(1, 1, Point(5, 5, 0))

    outer = NurbsCurve.create(True, 1, [
        Point(0.1, 0.1, 0), Point(0.9, 0.1, 0),
        Point(0.9, 0.9, 0), Point(0.1, 0.9, 0),
    ])

    ts = NurbsSurfaceTrimmed.create(srf, outer)
    ts.name = "test_nurbssurface_trimmed"
    ts.width = 2.0
    ts.surfacecolor = Color(255, 128, 64, 255)

    # JSON object
    json_obj = ts.__jsondump__()
    loaded_json = NurbsSurfaceTrimmed.__jsonload__(json_obj)

    # String
    json_string = ts.file_json_dumps()
    loaded_json_string = NurbsSurfaceTrimmed.file_json_loads(json_string)

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbssurface_trimmed.json"
    ts.file_json_dump(filename)
    loaded_from_file = NurbsSurfaceTrimmed.file_json_load(filename)

    MINI_CHECK(loaded_json == ts)
    MINI_CHECK(loaded_json_string == ts)
    MINI_CHECK(loaded_from_file == ts)


@MINI_TEST("NurbsSurfaceTrimmed", "Protobuf Roundtrip")
def test_nurbssurface_trimmed_protobuf_roundtrip():
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Color
    from session_py.nurbssurface_trimmed import NurbsSurfaceTrimmed
    from pathlib import Path

    srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(5, 0, 0))
    srf.set_cv(0, 1, Point(0, 5, 0))
    srf.set_cv(1, 1, Point(5, 5, 0))

    outer = NurbsCurve.create(True, 1, [
        Point(0.1, 0.1, 0), Point(0.9, 0.1, 0),
        Point(0.9, 0.9, 0), Point(0.1, 0.9, 0),
    ])

    ts = NurbsSurfaceTrimmed.create(srf, outer)
    ts.name = "test_nurbssurface_trimmed"
    ts.width = 2.0
    ts.surfacecolor = Color(255, 128, 64, 255)

    # String
    proto_string = ts.pb_dumps()
    loaded_proto_string = NurbsSurfaceTrimmed.pb_loads(proto_string)

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_nurbssurface_trimmed.bin"
    ts.pb_dump(filename)
    loaded = NurbsSurfaceTrimmed.pb_load(filename)

    MINI_CHECK(loaded_proto_string == ts)
    MINI_CHECK(loaded == ts)


if __name__ == "__main__":
    run_all("python")
