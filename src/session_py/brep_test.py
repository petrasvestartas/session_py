from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("BRep", "Constructor")
def test_brep_constructor():
    from session_py import BRep
    from session_py import Point

    b = BRep()

    sstr = str(b)
    srepr = repr(b)

    bcopy = b.duplicate()

    MINI_CHECK(not b.is_valid())
    MINI_CHECK(b.face_count() == 0)
    MINI_CHECK(b.name == "my_brep")
    MINI_CHECK(len(b.guid) > 0)
    MINI_CHECK("BRep" in sstr)
    MINI_CHECK("name=my_brep" in srepr)
    MINI_CHECK(bcopy.guid != b.guid)
    MINI_CHECK(bcopy == b)
    MINI_CHECK(not (bcopy != b))


@MINI_TEST("BRep", "Create Box")
def test_brep_create_box():
    from session_py import BRep
    from session_py import Point

    b = BRep.create_box(2.0, 3.0, 4.0)

    MINI_CHECK(b.is_valid())
    MINI_CHECK(b.face_count() == 6)
    MINI_CHECK(b.edge_count() == 12)
    MINI_CHECK(b.vertex_count() == 8)
    MINI_CHECK(b.is_solid())
    MINI_CHECK(b.name == "box")


@MINI_TEST("BRep", "Accessors")
def test_brep_accessors():
    from session_py import BRep
    from session_py import Point
    from session_py import NurbsSurface

    b = BRep.create_box(2.0, 3.0, 4.0)

    fc = b.face_count()
    ec = b.edge_count()
    vc = b.vertex_count()

    MINI_CHECK(fc == 6)
    MINI_CHECK(ec == 12)
    MINI_CHECK(vc == 8)
    MINI_CHECK(len(b.m_surfaces) == 6)
    MINI_CHECK(len(b.m_loops) == 6)
    MINI_CHECK(len(b.m_trims) == 24)


@MINI_TEST("BRep", "Add Face")
def test_brep_add_face():
    from session_py import BRep
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point

    b = BRep()
    srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(1, 0, 0))
    srf.set_cv(0, 1, Point(0, 1, 0))
    srf.set_cv(1, 1, Point(1, 1, 0))

    si = b.add_surface(srf)
    fi = b.add_face(si, False)
    li = b.add_loop(fi, 0)

    trim = NurbsCurve.create(False, 1, [
        Point(0, 0, 0),
        Point(1, 0, 0),
    ])
    ci = b.add_curve_2d(trim)
    b.add_trim(ci, -1, li, False, 0)

    MINI_CHECK(b.face_count() == 1)
    MINI_CHECK(len(b.m_surfaces) == 1)
    MINI_CHECK(len(b.m_loops) == 1)
    MINI_CHECK(len(b.m_trims) == 1)


@MINI_TEST("BRep", "Mesh")
def test_brep_mesh():
    from session_py import BRep
    from session_py import Mesh

    b = BRep.create_box(2.0, 3.0, 4.0)
    m = b.mesh()

    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_vertices() > 0)
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("BRep", "Point At")
def test_brep_point_at():
    from session_py import BRep
    from session_py import Point

    b = BRep.create_box(2.0, 3.0, 4.0)
    pt = b.point_at(0, 0.5, 0.5)

    MINI_CHECK(abs(pt[2] + 2.0) < 0.01 or abs(pt[2] - 2.0) < 0.01
            or abs(pt[1] + 1.5) < 0.01 or abs(pt[1] - 1.5) < 0.01
            or abs(pt[0] + 1.0) < 0.01 or abs(pt[0] - 1.0) < 0.01)


@MINI_TEST("BRep", "Is Solid")
def test_brep_is_solid():
    from session_py import BRep
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point

    b = BRep.create_box(2.0, 3.0, 4.0)

    single = BRep()
    srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2, False, False, 1.0, 1.0)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(1, 0, 0))
    srf.set_cv(0, 1, Point(0, 1, 0))
    srf.set_cv(1, 1, Point(1, 1, 0))
    si = single.add_surface(srf)
    single.add_face(si, False)
    single.add_vertex(Point(0, 0, 0))

    MINI_CHECK(b.is_solid())
    MINI_CHECK(not single.is_solid())


@MINI_TEST("BRep", "Transformation")
def test_brep_transformation():
    from session_py import BRep
    from session_py import Point
    from session_py import Xform

    b = BRep.create_box(2.0, 3.0, 4.0)
    b.xform = Xform.translation(10.0, 20.0, 30.0)
    moved = b.transformed()

    pt = moved.point_at(0, 0.0, 0.0)
    pt_orig = b.point_at(0, 0.0, 0.0)

    MINI_CHECK(abs(pt[0] - pt_orig[0] - 10.0) < 0.01)
    MINI_CHECK(abs(pt[1] - pt_orig[1] - 20.0) < 0.01)
    MINI_CHECK(abs(pt[2] - pt_orig[2] - 30.0) < 0.01)


@MINI_TEST("BRep", "Json Roundtrip")
def test_json_roundtrip():
    from session_py import BRep
    from session_py import Color
    from pathlib import Path

    b = BRep.create_box(2.0, 3.0, 4.0)
    b.name = "test_brep"
    b.width = 2.0
    b.surfacecolor = Color(255, 128, 64, 255)

    # JSON object
    json_obj = b.__jsondump__()
    loaded_json = BRep.__jsonload__(json_obj)

    # String
    json_string = b.file_json_dumps()
    loaded_json_string = BRep.file_json_loads(json_string)

    # File
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_brep.json"
    fname.parent.mkdir(parents=True, exist_ok=True)
    b.file_json_dump(str(fname))
    loaded_from_file = BRep.file_json_load(str(fname))

    MINI_CHECK(loaded_json == b)
    MINI_CHECK(loaded_json_string == b)
    MINI_CHECK(loaded_from_file == b)


@MINI_TEST("BRep", "Create Cylinder")
def test_brep_create_cylinder():
    from session_py import BRep
    from session_py import Mesh

    cyl = BRep.create_cylinder(1.0, 2.0)
    m = cyl.mesh()

    MINI_CHECK(cyl.is_valid())
    MINI_CHECK(cyl.face_count() == 3)
    MINI_CHECK(cyl.is_solid())
    MINI_CHECK(cyl.name == "cylinder")
    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_vertices() > 0)


@MINI_TEST("BRep", "Create Sphere")
def test_brep_create_sphere():
    from session_py import BRep
    from session_py import Mesh

    sph = BRep.create_sphere(2.0)
    m = sph.mesh()

    MINI_CHECK(sph.is_valid())
    MINI_CHECK(sph.face_count() == 1)
    MINI_CHECK(sph.is_solid())
    MINI_CHECK(sph.name == "sphere")
    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_vertices() > 0)


@MINI_TEST("BRep", "From Polylines")
def test_brep_from_polylines():
    from session_py import BRep
    from session_py import Polyline
    from session_py import Mesh
    from session_py import Point

    hx, hy, hz = 1.0, 1.5, 2.0
    c = [
        Point(-hx, -hy, -hz),
        Point(hx, -hy, -hz),
        Point(hx, hy, -hz),
        Point(-hx, hy, -hz),
        Point(-hx, -hy, hz),
        Point(hx, -hy, hz),
        Point(hx, hy, hz),
        Point(-hx, hy, hz),
    ]

    bottom = Polyline([
        c[0],
        c[3],
        c[2],
        c[1],
        c[0],
    ])
    top = Polyline([
        c[4],
        c[5],
        c[6],
        c[7],
        c[4],
    ])
    front = Polyline([
        c[0],
        c[1],
        c[5],
        c[4],
        c[0],
    ])
    right = Polyline([
        c[1],
        c[2],
        c[6],
        c[5],
        c[1],
    ])
    back = Polyline([
        c[2],
        c[3],
        c[7],
        c[6],
        c[2],
    ])
    left = Polyline([
        c[3],
        c[0],
        c[4],
        c[7],
        c[3],
    ])

    b = BRep.from_polylines([bottom, top, front, right, back, left])
    m = b.mesh()

    MINI_CHECK(b.is_valid())
    MINI_CHECK(b.face_count() == 6)
    MINI_CHECK(b.edge_count() == 12)
    MINI_CHECK(b.vertex_count() == 8)
    MINI_CHECK(b.is_solid())
    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("BRep", "From Nurbscurves")
def test_brep_from_nurbscurves():
    from session_py import BRep
    from session_py import NurbsCurve
    from session_py import Mesh
    from session_py import Point

    hx, hy, hz = 1.0, 1.5, 2.0
    c = [
        Point(-hx, -hy, -hz),
        Point(hx, -hy, -hz),
        Point(hx, hy, -hz),
        Point(-hx, hy, -hz),
        Point(-hx, -hy, hz),
        Point(hx, -hy, hz),
        Point(hx, hy, hz),
        Point(-hx, hy, hz),
    ]

    bottom = NurbsCurve.create(False, 1, [
        c[0],
        c[3],
        c[2],
        c[1],
        c[0],
    ])
    top = NurbsCurve.create(False, 1, [
        c[4],
        c[5],
        c[6],
        c[7],
        c[4],
    ])
    front = NurbsCurve.create(False, 1, [
        c[0],
        c[1],
        c[5],
        c[4],
        c[0],
    ])
    right = NurbsCurve.create(False, 1, [
        c[1],
        c[2],
        c[6],
        c[5],
        c[1],
    ])
    back = NurbsCurve.create(False, 1, [
        c[2],
        c[3],
        c[7],
        c[6],
        c[2],
    ])
    left = NurbsCurve.create(False, 1, [
        c[3],
        c[0],
        c[4],
        c[7],
        c[3],
    ])

    b = BRep.from_nurbscurves([bottom, top, front, right, back, left])
    m = b.mesh()

    MINI_CHECK(b.is_valid())
    MINI_CHECK(b.face_count() == 6)
    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("BRep", "From Nurbscurves Holes")
def test_brep_from_nurbscurves_holes():
    from session_py import BRep
    from session_py import NurbsCurve
    from session_py import Mesh
    from session_py import Point
    from session_py import Primitives

    outer = NurbsCurve.create(False, 1, [
        Point(-5, -5, 0),
        Point(5, -5, 0),
        Point(5, 5, 0),
        Point(-5, 5, 0),
        Point(-5, -5, 0),
    ])
    hole = Primitives.circle(0.0, 0.0, 0.0, 2.0)

    b = BRep.from_nurbscurves([outer], [[hole]])
    m = b.mesh()

    MINI_CHECK(b.is_valid())
    MINI_CHECK(b.face_count() == 1)
    MINI_CHECK(len(b.m_loops) == 2)
    MINI_CHECK(b.m_loops[0].type == 0)
    MINI_CHECK(b.m_loops[1].type == 1)
    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("BRep", "Create Block With Hole")
def test_brep_create_block_with_hole():
    from session_py.brep import BRep
    from session_py import Mesh

    bh = BRep.create_block_with_hole(8.0, 6.0, 4.0, 1.5)
    m = bh.mesh()

    MINI_CHECK(bh.is_valid())
    MINI_CHECK(bh.face_count() == 7)
    MINI_CHECK(bh.name == "block_with_hole")
    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_vertices() > 0)
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("BRep", "Mesh Orientation")
def test_brep_mesh_orientation():
    from session_py.brep import BRep
    from session_py import Mesh

    # Reversed faces must flip winding; the bug inflated volume() past the solid box.
    bh = BRep.create_block_with_hole(8.0, 6.0, 4.0, 1.5)
    vol = bh.mesh().volume()

    MINI_CHECK(vol > 60.0)
    MINI_CHECK(vol < 175.0)


@MINI_TEST("BRep", "Protobuf Roundtrip")
def test_protobuf_roundtrip():
    from session_py import BRep
    from session_py import Color
    from pathlib import Path

    b = BRep.create_box(2.0, 3.0, 4.0)
    b.name = "test_brep"
    b.width = 2.0
    b.surfacecolor = Color(255, 128, 64, 255)

    # String
    proto_data = b.pb_dumps()
    loaded_proto = BRep.pb_loads(proto_data)

    # File
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_brep.bin"
    fname.parent.mkdir(parents=True, exist_ok=True)
    b.pb_dump(str(fname))
    loaded = BRep.pb_load(str(fname))

    MINI_CHECK(loaded_proto == b)
    MINI_CHECK(loaded == b)


if __name__ == "__main__":
    run_all("python")
