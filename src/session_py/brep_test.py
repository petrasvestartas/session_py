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


@MINI_TEST("BRep", "Create Cone")
def test_brep_create_cone():
    import math
    from session_py import BRep

    PI = math.pi
    cone = BRep.create_cone(1.0, 2.0)   # base r=1 at z=0, apex z=2
    m = cone.mesh()

    MINI_CHECK(cone.is_valid())
    MINI_CHECK(cone.face_count() == 2)           # side + base cap
    MINI_CHECK(cone.is_solid())
    MINI_CHECK(cone.name == "cone")
    MINI_CHECK(not m.is_empty())
    # V = (1/3) pi r^2 h
    MINI_CHECK(abs(cone.volume() - (PI * 1.0 * 2.0 / 3.0)) / (PI * 2.0 / 3.0) < 1e-4)


@MINI_TEST("BRep", "Create Torus")
def test_brep_create_torus():
    import math
    from session_py import BRep

    PI = math.pi
    tor = BRep.create_torus(2.0, 0.5)   # major R=2, minor r=0.5
    m = tor.mesh()

    MINI_CHECK(tor.is_valid())
    MINI_CHECK(tor.face_count() == 1)
    MINI_CHECK(tor.is_solid())
    MINI_CHECK(tor.name == "torus")
    MINI_CHECK(not m.is_empty())
    # V = 2 pi^2 R r^2
    ref = 2.0 * PI * PI * 2.0 * 0.25
    MINI_CHECK(abs(tor.volume() - ref) / ref < 1e-3)


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


@MINI_TEST("BRep", "Split By Plane")
def test_brep_split_by_plane():
    from session_py import BRep
    from session_py import Plane
    from session_py import Point
    from session_py import Vector
    from session_py.brep import BRepLoopType

    box = BRep.create_box(2.0, 2.0, 2.0)
    plane = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0))
    split = box.split_by_plane(plane)
    box_area = box.mesh().area()
    split_area = split.mesh().area()
    inner = 0
    for face in split.m_faces:
        for li in face.loop_indices:
            if split.m_loops[li].type == BRepLoopType.Inner:
                inner += 1

    MINI_CHECK(split.face_count() == 10)
    MINI_CHECK(abs(split_area - box_area) < box_area * 0.01)
    MINI_CHECK(not split.mesh().is_empty())
    MINI_CHECK(inner == 0)

    cylinder = BRep.create_cylinder(1.0, 4.0)
    mid = Plane.from_point_normal(Point(0.0, 0.0, 1.0), Vector(0.0, 0.0, 1.0))
    cut = cylinder.split_by_plane(mid)

    MINI_CHECK(cut.face_count() == 4)
    MINI_CHECK(abs(cut.mesh().area() - cylinder.mesh().area()) < cylinder.mesh().area() * 0.02)


@MINI_TEST("BRep", "Split By Plane Pieces")
def test_brep_split_by_plane_pieces():
    from session_py import BRep
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    box = BRep.create_box(2.0, 2.0, 2.0)
    plane = Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0))
    pieces = box.split_by_plane_pieces(plane)
    total = 0.0
    for piece in pieces:
        total += piece.mesh().area()

    MINI_CHECK(len(pieces) == 2)
    MINI_CHECK(pieces[0].face_count() == 5)
    MINI_CHECK(pieces[1].face_count() == 5)
    MINI_CHECK(abs(total - box.mesh().area()) < box.mesh().area() * 0.01)

    far = Plane.from_point_normal(Point(0.0, 0.0, 5.0), Vector(0.0, 0.0, 1.0))
    whole = box.split_by_plane_pieces(far)

    MINI_CHECK(len(whole) == 1)
    MINI_CHECK(whole[0].face_count() == 6)


@MINI_TEST("BRep", "Split By Line")
def test_brep_split_by_line():
    from session_py import BRep
    from session_py import Line
    from session_py import Point

    box = BRep.create_box(2.0, 2.0, 2.0)
    line = Line.from_points(Point(0.0, -2.0, 1.0), Point(0.0, 2.0, 1.0))
    split = box.split_by_line(line)
    box_area = box.mesh().area()
    split_area = split.mesh().area()

    MINI_CHECK(split.face_count() == 7)
    MINI_CHECK(abs(split_area - box_area) < box_area * 0.01)
    MINI_CHECK(not split.mesh().is_empty())


@MINI_TEST("BRep", "Split By Brep")
def test_brep_split_by_brep():
    from session_py import BRep

    target = BRep.create_box(4.0, 4.0, 2.0)
    cutter = BRep.create_box(2.0, 2.0, 6.0)
    split = target.split_by_brep(cutter)
    target_area = target.mesh().area()
    split_area = split.mesh().area()

    MINI_CHECK(split.face_count() == 8)
    MINI_CHECK(abs(split_area - target_area) < target_area * 0.01)
    MINI_CHECK(not split.mesh().is_empty())


@MINI_TEST("BRep", "Volume")
def test_brep_volume():
    import math
    from .brep import BRep
    # Exact divergence-theorem volume: matches OCCT BRepGProp to machine precision.
    vbox = BRep.create_box(2, 3, 4).volume()
    vcyl = BRep.create_cylinder(1, 4).volume()
    vsph = BRep.create_sphere(2).volume()
    MINI_CHECK(abs(vbox - 24.0) < 1e-9)
    MINI_CHECK(abs(vcyl - 4 * math.pi) < 4 * math.pi * 1e-9)
    MINI_CHECK(abs(vsph - (4.0 / 3.0) * math.pi * 8) < (4.0 / 3.0) * math.pi * 8 * 1e-9)
    MINI_CHECK(BRep.create_box(2, 3, 4).face_count() == 6)
    MINI_CHECK(BRep.create_cylinder(1, 4).face_count() == 3)
    MINI_CHECK(BRep.create_sphere(2).face_count() == 1)


def test_brep_block_with_hole_volume():
    # Box(4) with a through cylindrical hole (r=1): the annular top/bottom faces have an inner
    # loop, so volume()'s face-interior sample must land on the MATERIAL (not in the hole) for
    # the outward-sign probe. OCCT BRepGProp: 64 - pi*r^2*h.
    import math
    from .brep import BRep
    bh = BRep.create_block_with_hole(4, 4, 4, 1.0)
    ref = 64.0 - math.pi * 1.0 * 1.0 * 4.0
    MINI_CHECK(bh.face_count() == 7)
    MINI_CHECK(abs(bh.volume() - ref) / ref < 1e-6)
    MINI_CHECK(bh.is_solid())


def test_brep_boolean_example_brep_booleans():
    # Reproduces docs/examples/breps/brep_booleans.py: Box(2) + Cylinder(r=0.7, h=3, centred).
    # OCCT (oracle): fuse 9.539380400258997/10, cut 4.921239199482002/7, common 3.078760800517997/3.
    from .brep import BRep
    from .xform import Xform
    box = BRep.create_box(2, 2, 2)
    cyl = BRep.create_cylinder(0.7, 3.0)
    cyl.xform = Xform.translation(0, 0, -1.5)
    cyl = cyl.transformed()
    fus = box.boolean_union(cyl)
    cut = box.boolean_difference(cyl)
    com = box.boolean_intersection(cyl)
    MINI_CHECK(fus.face_count() == 10)
    MINI_CHECK(cut.face_count() == 7)
    MINI_CHECK(com.face_count() == 3)
    MINI_CHECK(abs(fus.volume() - 9.539380400258997) / 9.539380400258997 < 1e-6)
    MINI_CHECK(abs(cut.volume() - 4.921239199482002) / 4.921239199482002 < 1e-6)
    MINI_CHECK(abs(com.volume() - 3.078760800517997) / 3.078760800517997 < 1e-6)
    MINI_CHECK(fus.is_solid())
    MINI_CHECK(cut.is_solid())
    MINI_CHECK(com.is_solid())


def test_brep_boolean_offcenter_cylinder():
    # Cylinder through a box, shifted +0.5 in x. OCCT (oracle): cut 64-4pi/7, common 4pi/3, fuse 64+2pi/10.
    import math
    from .brep import BRep
    from .xform import Xform
    PI = math.pi
    box = BRep.create_box(4, 4, 4)
    cyl = BRep.create_cylinder(1.0, 6.0)
    cyl.xform = Xform.translation(0.5, 0, -3)
    cyl = cyl.transformed()
    cut = box.boolean_difference(cyl)
    com = box.boolean_intersection(cyl)
    fus = box.boolean_union(cyl)
    MINI_CHECK(cut.face_count() == 7)
    MINI_CHECK(com.face_count() == 3)
    MINI_CHECK(fus.face_count() == 10)
    MINI_CHECK(abs(cut.volume() - (64 - 4 * PI)) / (64 - 4 * PI) < 1e-6)
    MINI_CHECK(abs(com.volume() - (4 * PI)) / (4 * PI) < 1e-6)
    MINI_CHECK(abs(fus.volume() - (64 + 2 * PI)) / (64 + 2 * PI) < 1e-6)
    MINI_CHECK(cut.is_solid())
    MINI_CHECK(com.is_solid())
    MINI_CHECK(fus.is_solid())


def test_brep_boolean_contained_box():
    # B (vol 8) fully inside A (vol 64). OCCT (oracle): cut 56/12, common 8/6, fuse 64/6.
    from .brep import BRep
    ba = BRep.create_box(4, 4, 4)
    bb = BRep.create_box(2, 2, 2)
    cut = ba.boolean_difference(bb)
    com = ba.boolean_intersection(bb)
    fus = ba.boolean_union(bb)
    MINI_CHECK(abs(cut.volume() - 56.0) < 1e-6)
    MINI_CHECK(abs(com.volume() - 8.0) < 1e-6)
    MINI_CHECK(abs(fus.volume() - 64.0) < 1e-6)
    MINI_CHECK(cut.face_count() == 12)
    MINI_CHECK(com.face_count() == 6)
    MINI_CHECK(fus.face_count() == 6)
    MINI_CHECK(cut.is_solid())
    MINI_CHECK(com.is_solid())
    MINI_CHECK(fus.is_solid())


def test_brep_boolean_contained_sphere():
    # Sphere (r=1.5) fully inside box(4): no surface intersection (no seam-straddling),
    # exercises robust volume() over a full periodic sphere + degenerate-pole-edge is_solid().
    # OCCT: cut 64-(4/3)pi r^3 / 7, common (4/3)pi r^3 / 1, fuse 64 / 6, all watertight.
    import math
    from .brep import BRep
    sv = (4.0 / 3.0) * math.pi * 1.5 ** 3
    box = BRep.create_box(4, 4, 4)
    sph = BRep.create_sphere(1.5)
    cut = box.boolean_difference(sph)
    com = box.boolean_intersection(sph)
    fus = box.boolean_union(sph)
    MINI_CHECK(abs(cut.volume() - (64.0 - sv)) / (64.0 - sv) < 1e-6)
    MINI_CHECK(abs(com.volume() - sv) / sv < 1e-6)
    MINI_CHECK(abs(fus.volume() - 64.0) < 1e-6)
    MINI_CHECK(cut.face_count() == 7)
    MINI_CHECK(com.face_count() == 1)
    MINI_CHECK(fus.face_count() == 6)
    MINI_CHECK(cut.is_solid())
    MINI_CHECK(com.is_solid())
    MINI_CHECK(fus.is_solid())


def test_brep_boolean_box_box():
    # Pure-planar partial overlap. A=[-2,2]^3 (64); B=[1,3]x[-1,1]^2 (8); overlap 4.
    # OCCT (oracle): cut 60/11, common 4/6, fuse 68/11.
    from .brep import BRep
    from .xform import Xform
    ba = BRep.create_box(4, 4, 4)
    bb = BRep.create_box(2, 2, 2)
    bb.xform = Xform.translation(2, 0, 0)
    bb = bb.transformed()
    cut = ba.boolean_difference(bb)
    com = ba.boolean_intersection(bb)
    fus = ba.boolean_union(bb)
    MINI_CHECK(abs(cut.volume() - 60.0) < 1e-6)
    MINI_CHECK(abs(com.volume() - 4.0) < 1e-6)
    MINI_CHECK(abs(fus.volume() - 68.0) < 1e-6)
    MINI_CHECK(cut.is_solid())
    MINI_CHECK(com.is_solid())
    MINI_CHECK(fus.is_solid())


def test_brep_boolean_sphere_split():
    # Box(4) - Sphere(2.5): the sphere pokes through every box face. The +x cap straddles the
    # periodic u-seam; the analytic sphere pull-back (replicating OCCT ProjLib_Sphere's per-point
    # inverse U=atan2 -> EXACT seam crossings) cuts it on BOTH sides, so it splits into two
    # half-caps -> 8 sphere regions (OCCT keeps the cap as a single seam-spanning face = 7).
    # Box-sphere boolean matches OCCT face counts (cut 7, common 7) and volumes to <0.3%
    # (cut 9.5457 / common 54.4543) via two fixes: (1) analytic_sphere_pullback maps longitude->u
    # through the TRUE rational-NURBS parametrization (was a linear approx distorting the cut
    # circle ~2% in flux); (2) volume() integrates sphere cap-cut faces by the analytic boundary
    # integral flux = C.A - R^2*closed_integral(h dtheta) instead of a masked Gauss.
    from .brep import BRep
    sph = BRep.create_sphere(2.5)
    box = BRep.create_box(4, 4, 4)
    B2 = sph.split_by_brep(box)
    MINI_CHECK(B2.face_count() == 8)
    bcut = box.boolean_difference(sph)
    bcom = box.boolean_intersection(sph)
    MINI_CHECK(bcut.is_solid())   # NOW WATERTIGHT via shared-section-edge co-refinement
    MINI_CHECK(bcom.is_solid())
    MINI_CHECK(bcut.face_count() == 7)
    MINI_CHECK(bcom.face_count() == 7)
    MINI_CHECK(abs(bcut.volume() - 9.545724580842144) / 9.545724580842144 < 0.01)
    MINI_CHECK(abs(bcom.volume() - 54.45427562996632) / 54.45427562996632 < 0.002)
    MINI_CHECK(abs(bcut.volume() + bcom.volume() - 64.0) < 1e-4)  # partition box exactly


if __name__ == "__main__":
    run_all("python")
