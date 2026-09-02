from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


def _edges_manifold(b) -> bool:
    # Every non-degenerated edge of a solid is used by exactly two faces with opposite
    # composed orientations (the manifold contract BRepCheck enforces).
    for ei in range(b.edge_count()):
        if b.m_edges[ei].degenerated:
            continue
        uses = b.edge_faces(ei)
        if len(uses) != 2:
            return False
        if uses[0].orientation == uses[1].orientation:
            return False
    return True


@MINI_TEST("BRep", "Constructor")
def test_brep_constructor():
    from session_py import BRep
    from session_py import Point

    b = BRep()

    # String representations
    sstr = str(b)
    srepr = repr(b)

    # Copy (new guid)
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

    box = BRep.create_box(2.0, 3.0, 4.0)

    MINI_CHECK(box.is_valid())
    MINI_CHECK(box.face_count() == 6)
    MINI_CHECK(box.edge_count() == 12)
    MINI_CHECK(box.vertex_count() == 8)
    MINI_CHECK(box.is_solid())
    MINI_CHECK(box.name == "box")


@MINI_TEST("BRep", "Accessors")
def test_brep_accessors():
    from session_py import BRep
    from session_py import Point

    box = BRep.create_box(2.0, 3.0, 4.0)

    vc = box.vertex_count()
    ec = box.edge_count()
    wc = box.wire_count()
    fc = box.face_count()
    sc = box.shell_count()
    oc = box.solid_count()
    pts = box.vertex_points()

    MINI_CHECK(vc == 8)
    MINI_CHECK(ec == 12)
    MINI_CHECK(wc == 6)
    MINI_CHECK(fc == 6)
    MINI_CHECK(sc == 1)
    MINI_CHECK(oc == 1)
    MINI_CHECK(len(pts) == 8)
    MINI_CHECK(abs(pts[0][0] + 1.0) < 1e-9)
    MINI_CHECK(len(box.m_surfaces) == 6)
    MINI_CHECK(len(box.m_curves_3d) == 12)
    MINI_CHECK(len(box.m_curves_2d) == 24)


@MINI_TEST("BRep", "Add Face")
def test_brep_add_face():
    from session_py import BRep
    from session_py import BRepOrientation
    from session_py import BRepRef
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Mesh

    b = BRep()
    srf = NurbsSurface(3, False, 2, 2, 2, 2)
    srf.set_cv(0, 0, Point(0, 0, 0)); srf.set_cv(1, 0, Point(1, 0, 0))
    srf.set_cv(0, 1, Point(0, 1, 0)); srf.set_cv(1, 1, Point(1, 1, 0))
    si = b.add_surface(srf)

    corners = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    refs = []
    for i in range(4):
        b.add_vertex(corners[i])
    for i in range(4):
        j = (i + 1) % 4
        ci = b.add_curve_3d(NurbsCurve.create(False, 1, [corners[i], corners[j]]))
        ei = b.add_edge(ci, i, j)
        c2 = b.add_curve_2d(NurbsCurve.create(False, 1, [corners[i], corners[j]]))
        b.add_pcurve(ei, si, c2)
        refs.append(BRepRef(ei, BRepOrientation.Forward))
    wi = b.add_wire(refs)
    fi = b.add_face(si, [BRepRef(wi, BRepOrientation.Forward)])
    m = b.mesh()

    MINI_CHECK(b.is_valid())
    MINI_CHECK(fi == 0)
    MINI_CHECK(b.face_count() == 1)
    MINI_CHECK(b.wire_count() == 1)
    MINI_CHECK(b.edge_count() == 4)
    MINI_CHECK(b.vertex_count() == 4)
    MINI_CHECK(len(b.m_edges[0].pcurves) == 1)
    MINI_CHECK(b.pcurve_index(0, 0, BRepOrientation.Forward) == 0)
    MINI_CHECK(not b.is_solid())
    MINI_CHECK(not m.is_empty())


@MINI_TEST("BRep", "Mesh")
def test_brep_mesh():
    from session_py import BRep
    from session_py import Mesh

    box = BRep.create_box(2.0, 3.0, 4.0)
    m = box.mesh()
    fm = box.face_meshes()

    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_vertices() > 0)
    MINI_CHECK(m.number_of_faces() > 0)
    MINI_CHECK(len(fm) == 6)
    MINI_CHECK(not fm[0].is_empty())


@MINI_TEST("BRep", "Point At")
def test_brep_point_at():
    from session_py import BRep
    from session_py import Point
    from session_py import Vector

    box = BRep.create_box(2.0, 3.0, 4.0)
    pt = box.point_at(0, 0.5, 0.5)
    n = box.normal_at(0, 0.5, 0.5)
    n_top = box.normal_at(1, 0.5, 0.5)

    MINI_CHECK(abs(pt[2] + 2.0) < 1e-9)
    MINI_CHECK(abs(pt[0]) < 1e-9)
    MINI_CHECK(abs(pt[1]) < 1e-9)
    MINI_CHECK(n[2] < -0.99)
    MINI_CHECK(n_top[2] > 0.99)


@MINI_TEST("BRep", "Is Solid")
def test_brep_is_solid():
    from session_py import BRep
    from session_py import Point
    from session_py import Polyline

    box = BRep.create_box(2.0, 3.0, 4.0)
    cyl = BRep.create_cylinder(1.0, 2.0)
    sph = BRep.create_sphere(1.0)
    cone = BRep.create_cone(1.0, 2.0)
    pyr = BRep.create_pyramid(2.0, 1.0)
    tor = BRep.create_torus(2.0, 0.5)
    blk = BRep.create_block_with_hole(4.0, 4.0, 2.0, 1.0)

    quad = Polyline([
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
        Point(0, 0, 0),
    ])
    sheet = BRep.from_polylines([quad])

    MINI_CHECK(box.is_solid() and _edges_manifold(box))
    MINI_CHECK(cyl.is_solid() and _edges_manifold(cyl))
    MINI_CHECK(sph.is_solid() and _edges_manifold(sph))
    MINI_CHECK(cone.is_solid() and _edges_manifold(cone))
    MINI_CHECK(pyr.is_solid() and _edges_manifold(pyr))
    MINI_CHECK(tor.is_solid() and _edges_manifold(tor))
    MINI_CHECK(blk.is_solid() and _edges_manifold(blk))
    MINI_CHECK(not sheet.is_solid())
    MINI_CHECK(sheet.solid_count() == 0)


@MINI_TEST("BRep", "Is Closed")
def test_brep_is_closed():
    from session_py import BRep

    box = BRep.create_box(2.0, 3.0, 4.0)
    open_shell = box.duplicate()
    open_shell.m_shells[0].faces.pop()

    MINI_CHECK(box.is_closed(0))
    MINI_CHECK(not box.is_closed(1))
    MINI_CHECK(not open_shell.is_closed(0))
    MINI_CHECK(not open_shell.is_solid())


@MINI_TEST("BRep", "Wire Edges")
def test_brep_wire_edges():
    from session_py import BRep
    from session_py import BRepOrientation
    from session_py import BRepRef
    from session_py.brep import brep_compose
    from session_py.brep import brep_reverse

    box = BRep.create_box(2.0, 3.0, 4.0)
    fwd = BRepRef(0, BRepOrientation.Forward)
    rev = BRepRef(0, BRepOrientation.Reversed)
    a = box.wire_edges(fwd)
    c = box.wire_edges(rev)

    MINI_CHECK(len(a) == 4)
    MINI_CHECK(len(c) == 4)
    MINI_CHECK(a[0].index == c[3].index)
    MINI_CHECK(a[0].orientation == brep_reverse(c[3].orientation))
    MINI_CHECK(brep_compose(BRepOrientation.Reversed, BRepOrientation.Reversed) == BRepOrientation.Forward)
    MINI_CHECK(brep_compose(BRepOrientation.Forward, BRepOrientation.Reversed) == BRepOrientation.Reversed)
    MINI_CHECK(brep_compose(BRepOrientation.Internal, BRepOrientation.Reversed) == BRepOrientation.Internal)


@MINI_TEST("BRep", "Edge Faces")
def test_brep_edge_faces():
    from session_py import BRep
    from session_py import BRepOrientation

    cyl = BRep.create_cylinder(1.0, 2.0)
    bot = cyl.edge_faces(0)
    seam = cyl.edge_faces(2)
    pc_f = cyl.pcurve_index(2, 0, BRepOrientation.Forward)
    pc_r = cyl.pcurve_index(2, 0, BRepOrientation.Reversed)

    MINI_CHECK(len(bot) == 2)
    MINI_CHECK(bot[0].index == 0 and bot[1].index == 1)
    MINI_CHECK(bot[0].orientation != bot[1].orientation)
    MINI_CHECK(len(seam) == 2)
    MINI_CHECK(seam[0].index == 0 and seam[1].index == 0)
    MINI_CHECK(pc_f >= 0 and pc_r >= 0 and pc_f != pc_r)
    MINI_CHECK(cyl.pcurve_index(2, 1, BRepOrientation.Forward) == -1)
    MINI_CHECK(cyl.face_orientation(0) == BRepOrientation.Forward)


@MINI_TEST("BRep", "Update Tolerances")
def test_brep_update_tolerances():
    from session_py import BRep
    from session_py import Point

    box = BRep.create_box(2.0, 3.0, 4.0)
    worst = box.update_tolerances()
    bent = box.duplicate()
    bent.m_vertices[0].point = Point(-1.0, -1.5, -2.01)
    worst_bent = bent.update_tolerances()
    prims = [BRep.create_cylinder(1.0, 2.0), BRep.create_sphere(1.0), BRep.create_cone(1.0, 2.0),
             BRep.create_pyramid(2.0, 1.0), BRep.create_torus(2.0, 0.5), BRep.create_block_with_hole(4.0, 4.0, 2.0, 1.0)]
    worst_prims = max(p.update_tolerances() for p in prims)

    MINI_CHECK(worst < 1e-9)
    MINI_CHECK(box.m_edges[0].tolerance < 1e-9)
    MINI_CHECK(abs(worst_bent - 0.01) < 1e-9)
    MINI_CHECK(abs(bent.m_vertices[0].tolerance - 0.01) < 1e-9)
    MINI_CHECK(bent.m_vertices[6].tolerance < 1e-9)
    MINI_CHECK(worst_prims < 1e-6)


@MINI_TEST("BRep", "Transformation")
def test_brep_transformation():
    from session_py import BRep
    from session_py import Point
    from session_py import Xform

    box = BRep.create_box(2.0, 3.0, 4.0)
    box_xf = Xform.translation(10.0, 20.0, 30.0)
    moved = box.transformed(box_xf)

    pt = moved.point_at(0, 0.0, 0.0)
    pt_orig = box.point_at(0, 0.0, 0.0)

    MINI_CHECK(abs(pt[0] - pt_orig[0] - 10.0) < 0.01)
    MINI_CHECK(abs(pt[1] - pt_orig[1] - 20.0) < 0.01)
    MINI_CHECK(abs(pt[2] - pt_orig[2] - 30.0) < 0.01)
    MINI_CHECK(abs(moved.m_vertices[0].point[0] - box.m_vertices[0].point[0] - 10.0) < 0.01)


@MINI_TEST("BRep", "Transform Roundtrip")
def test_brep_transform_roundtrip():
    from session_py import BRep
    from session_py import Point
    from session_py import Vector
    from session_py import Xform

    axis = Vector(0.3, 0.5, 0.81)
    rot = Xform.rotation(axis, 37.0, True)
    tr = Xform.translation(10.0, -5.0, 3.0)
    box = BRep.create_box(2.0, 3.0, 4.0)
    moved = box.transformed(rot).transformed(tr)

    match = True
    for i in range(len(box.m_vertices)):
        expect = tr.transform_point(rot.transform_point(box.m_vertices[i].point))
        if moved.m_vertices[i].point.distance(expect) > 1e-9:
            match = False

    back = moved.transformed(tr.inverse()).transformed(rot.inverse())

    restored = True
    for i in range(len(box.m_vertices)):
        if back.m_vertices[i].point.distance(box.m_vertices[i].point) > 1e-9:
            restored = False

    MINI_CHECK(match)
    MINI_CHECK(restored)
    MINI_CHECK(back.is_solid())
    MINI_CHECK(back.update_tolerances() < 1e-9)


@MINI_TEST("BRep", "Json Roundtrip")
def test_json_roundtrip():
    from session_py import BRep
    from session_py import BRepOrientation
    from session_py import Color
    from pathlib import Path

    box = BRep.create_cylinder(1.0, 2.0)
    box.name = "test_brep"
    box.width = 2.0
    box.surfacecolor = Color(255, 128, 64, 255)

    # JSON object
    json_obj = box.__jsondump__()
    loaded_json = BRep.__jsonload__(json_obj)

    # String
    json_string = box.file_json_dumps()
    loaded_json_string = BRep.file_json_loads(json_string)

    # File
    filename = Path(__file__).parent.parent.parent / "serialization" / "test_brep.json"
    box.file_json_dump(filename)
    loaded_from_file = BRep.file_json_load(filename)

    MINI_CHECK(loaded_json == box)
    MINI_CHECK(loaded_json_string == box)
    MINI_CHECK(loaded_from_file == box)
    MINI_CHECK(loaded_from_file.is_solid())
    MINI_CHECK(loaded_from_file.m_edges[2].pcurves[0].curve_2d_index_2 >= 0)
    MINI_CHECK(loaded_from_file.m_wires[0].edges[2].orientation == BRepOrientation.Reversed)


@MINI_TEST("BRep", "Create Cylinder")
def test_brep_create_cylinder():
    from session_py import BRep
    from session_py import Mesh

    cyl = BRep.create_cylinder(1.0, 2.0)
    m = cyl.mesh()

    MINI_CHECK(cyl.is_valid())
    MINI_CHECK(cyl.face_count() == 3)
    MINI_CHECK(cyl.edge_count() == 3)
    MINI_CHECK(cyl.vertex_count() == 2)
    MINI_CHECK(cyl.is_solid())
    MINI_CHECK(cyl.name == "cylinder")
    MINI_CHECK(not m.is_empty())


@MINI_TEST("BRep", "Create Sphere")
def test_brep_create_sphere():
    from session_py import BRep
    from session_py import Mesh

    sph = BRep.create_sphere(1.0)
    m = sph.mesh()

    MINI_CHECK(sph.is_valid())
    MINI_CHECK(sph.face_count() == 1)
    MINI_CHECK(sph.edge_count() == 3)
    MINI_CHECK(sph.vertex_count() == 2)
    MINI_CHECK(sph.m_edges[1].degenerated and sph.m_edges[2].degenerated)
    MINI_CHECK(sph.is_solid())
    MINI_CHECK(sph.name == "sphere")
    MINI_CHECK(not m.is_empty())


@MINI_TEST("BRep", "Create Cone")
def test_brep_create_cone():
    from session_py import BRep
    from session_py import Mesh

    cone = BRep.create_cone(1.0, 2.0)
    m = cone.mesh()

    MINI_CHECK(cone.is_valid())
    MINI_CHECK(cone.face_count() == 2)
    MINI_CHECK(cone.edge_count() == 3)
    MINI_CHECK(cone.vertex_count() == 2)
    MINI_CHECK(cone.is_solid())
    MINI_CHECK(cone.name == "cone")
    MINI_CHECK(not m.is_empty())


@MINI_TEST("BRep", "Create Pyramid")
def test_brep_create_pyramid():
    from session_py import BRep
    from session_py import Mesh

    pyr = BRep.create_pyramid(2.0, 1.0)
    m = pyr.mesh()

    MINI_CHECK(pyr.is_valid())
    MINI_CHECK(pyr.face_count() == 5)
    MINI_CHECK(pyr.edge_count() == 12)
    MINI_CHECK(pyr.vertex_count() == 5)
    MINI_CHECK(pyr.is_solid())
    MINI_CHECK(pyr.name == "pyramid")
    MINI_CHECK(not m.is_empty())


@MINI_TEST("BRep", "Create Torus")
def test_brep_create_torus():
    from session_py import BRep
    from session_py import Mesh

    tor = BRep.create_torus(2.0, 0.5)
    m = tor.mesh()

    MINI_CHECK(tor.is_valid())
    MINI_CHECK(tor.face_count() == 1)
    MINI_CHECK(tor.edge_count() == 2)
    MINI_CHECK(tor.vertex_count() == 1)
    MINI_CHECK(tor.is_solid())
    MINI_CHECK(tor.name == "torus")
    MINI_CHECK(not m.is_empty())


@MINI_TEST("BRep", "Create Block With Hole")
def test_brep_create_block_with_hole():
    from session_py import BRep
    from session_py import BRepOrientation
    from session_py import Mesh

    bh = BRep.create_block_with_hole(8.0, 6.0, 4.0, 1.5)
    m = bh.mesh()

    MINI_CHECK(bh.is_valid())
    MINI_CHECK(bh.face_count() == 7)
    MINI_CHECK(bh.edge_count() == 15)
    MINI_CHECK(bh.vertex_count() == 10)
    MINI_CHECK(len(bh.m_faces[6].wires) == 2)
    MINI_CHECK(bh.face_orientation(4) == BRepOrientation.Reversed)
    MINI_CHECK(bh.is_solid())
    MINI_CHECK(bh.name == "block_with_hole")
    MINI_CHECK(not m.is_empty())


@MINI_TEST("BRep", "From Polylines")
def test_brep_from_polylines():
    from session_py import BRep
    from session_py import Point
    from session_py import Polyline
    from session_py import Mesh

    hx, hy, hz = 1.0, 1.5, 2.0
    c = [
        Point(-hx, -hy, -hz),
        Point( hx, -hy, -hz),
        Point( hx,  hy, -hz),
        Point(-hx,  hy, -hz),
        Point(-hx, -hy,  hz),
        Point( hx, -hy,  hz),
        Point( hx,  hy,  hz),
        Point(-hx,  hy,  hz),
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
    MINI_CHECK(b.shell_count() == 1)
    MINI_CHECK(b.is_solid() and _edges_manifold(b))
    MINI_CHECK(abs(b.volume() - 24.0) < 1e-6)
    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("BRep", "From Nurbscurves")
def test_brep_from_nurbscurves():
    from session_py import BRep
    from session_py import Point
    from session_py import NurbsCurve
    from session_py import Mesh

    hx, hy, hz = 1.0, 1.5, 2.0
    c = [
        Point(-hx, -hy, -hz),
        Point( hx, -hy, -hz),
        Point( hx,  hy, -hz),
        Point(-hx,  hy, -hz),
        Point(-hx, -hy,  hz),
        Point( hx, -hy,  hz),
        Point( hx,  hy,  hz),
        Point(-hx,  hy,  hz),
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
    MINI_CHECK(b.edge_count() == 6)
    MINI_CHECK(b.vertex_count() == 5)
    MINI_CHECK(not b.is_solid())
    MINI_CHECK(not m.is_empty())
    MINI_CHECK(m.number_of_faces() > 0)


@MINI_TEST("BRep", "From Nurbscurves Holes")
def test_brep_from_nurbscurves_holes():
    from session_py import BRep
    from session_py import Point
    from session_py import NurbsCurve
    from session_py import Mesh
    from session_py import Primitives
    from session_py.tolerance import PI

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
    MINI_CHECK(b.wire_count() == 2)
    MINI_CHECK(len(b.m_faces[0].wires) == 2)
    MINI_CHECK(b.m_faces[0].wires[1].index == 1)
    MINI_CHECK(not m.is_empty())
    MINI_CHECK(abs(m.area() - (100.0 - PI * 4.0)) < 0.5)


@MINI_TEST("BRep", "Mesh Orientation")
def test_brep_mesh_orientation():
    from session_py import BRep
    from session_py import Mesh
    from session_py.tolerance import PI

    # Reversed faces must flip winding; an unflipped bore inflates the volume.
    bh = BRep.create_block_with_hole(8.0, 6.0, 4.0, 1.5)
    vol = bh.mesh().volume()
    ref = 8.0 * 6.0 * 4.0 - PI * 1.5 * 1.5 * 4.0

    MINI_CHECK(abs(vol - ref) / ref < 0.02)


@MINI_TEST("BRep", "Protobuf Roundtrip")
def test_protobuf_roundtrip():
    from session_py import BRep
    from session_py import BRepOrientation
    from session_py import Color
    from pathlib import Path

    box = BRep.create_cylinder(1.0, 2.0)
    box.name = "test_brep"
    box.width = 2.0
    box.surfacecolor = Color(255, 128, 64, 255)

    # String
    proto_string = box.pb_dumps()
    loaded_proto_string = BRep.pb_loads(proto_string)

    # File
    filename = Path(__file__).parent.parent.parent / "serialization" / "test_brep.bin"
    box.pb_dump(filename)
    loaded = BRep.pb_load(filename)

    MINI_CHECK(loaded_proto_string == box)
    MINI_CHECK(loaded == box)
    MINI_CHECK(loaded.is_solid())
    MINI_CHECK(loaded.m_edges[2].pcurves[0].curve_2d_index_2 >= 0)
    MINI_CHECK(loaded.m_wires[0].edges[2].orientation == BRepOrientation.Reversed)


@MINI_TEST("BRep", "Volume")
def test_brep_volume():
    from session_py import BRep
    from session_py.tolerance import PI

    box = BRep.create_box(2, 3, 4)          # 2x3x4 -> 24
    cyl = BRep.create_cylinder(1.0, 4.0)    # pi r^2 h = 4 pi
    sph = BRep.create_sphere(2.0)           # 4/3 pi r^3
    vbox, vcyl, vsph = box.volume(), cyl.volume(), sph.volume()

    # Tessellated volume: the default grid density is 2-4% under the analytic value.
    MINI_CHECK(abs(vbox - 24.0) < 1e-9)
    MINI_CHECK(abs(vcyl - 4 * PI) / (4 * PI) < 0.05)
    MINI_CHECK(abs(vsph - (4.0 / 3.0) * PI * 8) / ((4.0 / 3.0) * PI * 8) < 0.05)


if __name__ == "__main__":
    run_all()
