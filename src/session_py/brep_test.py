from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


def _edges_manifold(b) -> bool:
    """Every non-degenerated edge of a solid is used by exactly two faces with opposite composed orientations"""

    for ei in range(b.edge_count()):
        if b.m_edges[ei].degenerated:
            continue

        uses = b.edge_faces(ei)

        if len(uses) != 2:
            return False

        if uses[0].orientation == uses[1].orientation:
            return False

    return True


def _boundary_points(mesh) -> list[tuple[float, float, float]]:
    """Sorted positions of the mesh vertices on the v = 0 side"""

    points = []

    for v in mesh.vertex.values():
        if v.attributes["v"] == 0.0:
            points.append((v.position()[0], v.position()[1], v.position()[2]))

    points.sort()

    return points


def _build_quad_face(b) -> int:
    """Unit planar quad face with straight edges and pcurves; returns the face index"""

    from session_py import BRepOrientation
    from session_py import BRepRef
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point

    srf = NurbsSurface(3, False, 2, 2, 2, 2)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(1, 0, 0))
    srf.set_cv(0, 1, Point(0, 1, 0))
    srf.set_cv(1, 1, Point(1, 1, 0))
    si = b.add_surface(srf)
    corners = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]

    for i in range(4):
        b.add_vertex(corners[i])

    refs = []

    for i in range(4):
        j = (i + 1) % 4
        ci = b.add_curve_3d(NurbsCurve.create(False, 1, [corners[i], corners[j]]))
        ei = b.add_edge(ci, i, j)
        c2 = b.add_curve_2d(NurbsCurve.create(False, 1, [corners[i], corners[j]]))
        b.add_pcurve(ei, si, c2)
        refs.append(BRepRef(ei, BRepOrientation.Forward))

    wi = b.add_wire(refs)

    return b.add_face(si, [BRepRef(wi, BRepOrientation.Forward)])


@MINI_TEST("BRep", "Shared Grid Boundary")
def test_brep_shared_grid_boundary():
    from session_py import BRep
    from session_py import BRepOrientation
    from session_py import BRepRef
    from session_py import NurbsSurface
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import RemeshNurbsSurfaceGrid
    from session_py.tolerance import PI
    import math
    import sys

    b = BRep()
    surfaces = []

    for face in range(2):
        points = []

        for i in range(3):
            for j in range(2):
                z = 0.0 if i != 1 else (0.5 if j == 0 or face == 0 else 4.0)
                points.append(Point(i * 0.5, j * (1.0 if face == 0 else -1.0), z))

        surface = NurbsSurface.create(False, False, 2, 1, 3, 2, points)
        si = b.add_surface(surface)
        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        vertices = []

        for side in range(4):
            vertices.append(
                b.add_vertex(surface.point_at(corners[side][0], corners[side][1]))
            )

        edges = []

        for side in range(4):
            a = corners[side]
            z = corners[(side + 1) % 4]
            edge = 0

            if side != 0 or face != 1:
                dir = 0 if a[0] != z[0] else 1
                curve = surface.iso_curve(dir, a[1 - dir])

                if a[dir] > z[dir]:
                    curve.reverse()

                edge = b.add_edge(
                    b.add_curve_3d(curve), vertices[side], vertices[(side + 1) % 4]
                )

            pc = NurbsCurve.create(
                False, 1, [Point(a[0], a[1], 0), Point(z[0], z[1], 0)]
            )
            b.add_pcurve(edge, si, b.add_curve_2d(pc))
            edges.append(BRepRef(edge, BRepOrientation.Forward))

        b.add_face(si, [BRepRef(b.add_wire(edges), BRepOrientation.Forward)], 1e-8)
        surfaces.append(surface)

    original = []

    for s in surfaces:
        original.append(RemeshNurbsSurfaceGrid.from_u_v_q(s, 0, 0, 20.0, 0.005))

    MINI_CHECK(
        len(_boundary_points(original[0])) == 7
        and len(_boundary_points(original[1])) == 11
    )

    meshes = b.face_meshes_q(True, 20.0, 0.005)
    first = _boundary_points(meshes[0])
    second = _boundary_points(meshes[1])

    MINI_CHECK(first == second and len(first) == 7)
    MINI_CHECK(len(meshes[0].face) == len(original[0].face) and len(meshes[1].face) > 0)

    maximum = 0.0

    for i in range(len(first) - 1):
        a = first[i]
        z = first[i + 1]
        actual = surfaces[0].point_at((a[0] + z[0]) * 0.5, 0.0)
        sag = 0.0

        for d in range(3):
            sag += (actual[d] - (a[d] + z[d]) * 0.5) ** 2

        maximum = max(maximum, math.sqrt(sag))

    MINI_CHECK(maximum <= 0.005 * 1.5)

    refined = RemeshNurbsSurfaceGrid.from_u_v_q(surfaces[0], 0, 0, 5.0, 0.001)
    meshes = b.face_meshes_q(True, 5.0, 0.001)
    first = _boundary_points(meshes[0])

    MINI_CHECK(
        first == _boundary_points(meshes[1])
        and len(meshes[0].face) > 0
        and len(meshes[1].face) > 0
    )

    for point in _boundary_points(refined):
        MINI_CHECK(point in first)

    cosine = math.cos(5.0 * PI / 180.0)

    for i in range(len(first) - 1):
        a = surfaces[0].normal_at(first[i][0], 0.0)
        z = surfaces[0].normal_at(first[i + 1][0], 0.0)
        MINI_CHECK(a.dot(z) >= cosine - 64.0 * sys.float_info.epsilon)


@MINI_TEST("BRep", "Constructor")
def test_brep_constructor():
    from session_py import BRep

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

    b = BRep()
    srf = NurbsSurface(3, False, 2, 2, 2, 2)
    srf.set_cv(0, 0, Point(0, 0, 0))
    srf.set_cv(1, 0, Point(1, 0, 0))
    srf.set_cv(0, 1, Point(0, 1, 0))
    srf.set_cv(1, 1, Point(1, 1, 0))
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

    quad = Polyline(
        [
            Point(0, 0, 0),
            Point(1, 0, 0),
            Point(1, 1, 0),
            Point(0, 1, 0),
            Point(0, 0, 0),
        ]
    )
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
    MINI_CHECK(
        brep_compose(BRepOrientation.Reversed, BRepOrientation.Reversed)
        == BRepOrientation.Forward
    )
    MINI_CHECK(
        brep_compose(BRepOrientation.Forward, BRepOrientation.Reversed)
        == BRepOrientation.Reversed
    )
    MINI_CHECK(
        brep_compose(BRepOrientation.Internal, BRepOrientation.Reversed)
        == BRepOrientation.Internal
    )


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
    worst_prims = 0.0

    for p in [
        BRep.create_cylinder(1.0, 2.0),
        BRep.create_sphere(1.0),
        BRep.create_cone(1.0, 2.0),
        BRep.create_pyramid(2.0, 1.0),
        BRep.create_torus(2.0, 0.5),
        BRep.create_block_with_hole(4.0, 4.0, 2.0, 1.0),
    ]:
        worst_prims = max(worst_prims, p.update_tolerances())

    MINI_CHECK(worst < 1e-9)
    MINI_CHECK(box.m_edges[0].tolerance < 1e-9)
    MINI_CHECK(abs(worst_bent - 0.01) < 1e-9)
    MINI_CHECK(abs(bent.m_vertices[0].tolerance - 0.01) < 1e-9)
    MINI_CHECK(bent.m_vertices[6].tolerance < 1e-9)
    MINI_CHECK(worst_prims < 1e-6)


@MINI_TEST("BRep", "Transformation")
def test_brep_transformation():
    from session_py import BRep
    from session_py import Xform

    box = BRep.create_box(2.0, 3.0, 4.0)
    box_xf = Xform.translation(10.0, 20.0, 30.0)
    moved = box.transformed(box_xf)

    pt = moved.point_at(0, 0.0, 0.0)
    pt_orig = box.point_at(0, 0.0, 0.0)

    MINI_CHECK(abs(pt[0] - pt_orig[0] - 10.0) < 0.01)
    MINI_CHECK(abs(pt[1] - pt_orig[1] - 20.0) < 0.01)
    MINI_CHECK(abs(pt[2] - pt_orig[2] - 30.0) < 0.01)
    MINI_CHECK(
        abs(moved.m_vertices[0].point[0] - box.m_vertices[0].point[0] - 10.0) < 0.01
    )


@MINI_TEST("BRep", "Transform Roundtrip")
def test_brep_transform_roundtrip():
    from session_py import BRep
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


@MINI_TEST("BRep", "Cut By Plane")
def test_brep_cut_by_plane():
    from session_py import BRep
    from session_py import Plane
    from session_py import Point
    from session_py import Vector
    from session_py import Xform

    box = BRep.create_box(2.0, 2.0, 2.0)
    half = box.cut_by_plane(
        Plane.from_point_normal(Point(0.0, 0.0, 0.0), Vector(0.0, 0.0, 1.0))
    )

    far = Xform.translation(100000.0, 200000.0, 30000.0) * Xform.rotation(
        Vector(1.0, 2.0, 3.0), 40.0, True
    )
    beam = BRep.create_box(200.0, 100.0, 600.0).transformed(far)
    piece = beam.cut_by_plane(
        Plane.from_point_normal(
            far.transform_point(Point(0.0, 0.0, 0.0)),
            far.transform_vector(Vector(0.0, 0.0, 1.0)),
        )
    )

    MINI_CHECK(half.is_solid())
    MINI_CHECK(half.vertex_count() == 8)
    MINI_CHECK(half.edge_count() == 12)
    MINI_CHECK(half.face_count() == 6)
    MINI_CHECK(abs(half.volume() - 4.0) < 1e-9)
    MINI_CHECK(piece.is_solid())
    MINI_CHECK(piece.face_count() == 6)
    MINI_CHECK(abs(piece.volume() - 6000000.0) < 0.01)


@MINI_TEST("BRep", "Json Roundtrip")
def test_brep_json_roundtrip():
    from session_py import BRep
    from session_py import BRepOrientation
    from session_py import Color
    from pathlib import Path

    box = BRep.create_cylinder(1.0, 2.0)
    box.name = "test_brep"
    box.width = 2.0
    box.surfacecolor = Color(255, 128, 64, 255)

    json_obj = box.__jsondump__()
    loaded_json = BRep.__jsonload__(json_obj)

    json_string = box.file_json_dumps()
    loaded_json_string = BRep.file_json_loads(json_string)

    filename = Path(__file__).parent.parent.parent / "serialization" / "test_brep.json"
    box.file_json_dump(filename)
    loaded_from_file = BRep.file_json_load(filename)

    MINI_CHECK(loaded_json == box)
    MINI_CHECK(loaded_json_string == box)
    MINI_CHECK(loaded_from_file == box)
    MINI_CHECK(loaded_from_file.is_solid())
    MINI_CHECK(loaded_from_file.m_edges[2].pcurves[0].curve_2d_index_2 >= 0)
    MINI_CHECK(
        loaded_from_file.m_wires[0].edges[2].orientation == BRepOrientation.Reversed
    )


@MINI_TEST("BRep", "Create Cylinder")
def test_brep_create_cylinder():
    from session_py import BRep

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

    hx = 1.0
    hy = 1.5
    hz = 2.0
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

    bottom = Polyline(
        [
            c[0],
            c[3],
            c[2],
            c[1],
            c[0],
        ]
    )
    top = Polyline(
        [
            c[4],
            c[5],
            c[6],
            c[7],
            c[4],
        ]
    )
    front = Polyline(
        [
            c[0],
            c[1],
            c[5],
            c[4],
            c[0],
        ]
    )
    right = Polyline(
        [
            c[1],
            c[2],
            c[6],
            c[5],
            c[1],
        ]
    )
    back = Polyline(
        [
            c[2],
            c[3],
            c[7],
            c[6],
            c[2],
        ]
    )
    left = Polyline(
        [
            c[3],
            c[0],
            c[4],
            c[7],
            c[3],
        ]
    )

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

    hx = 1.0
    hy = 1.5
    hz = 2.0
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

    bottom = NurbsCurve.create(
        False,
        1,
        [
            c[0],
            c[3],
            c[2],
            c[1],
            c[0],
        ],
    )
    top = NurbsCurve.create(
        False,
        1,
        [
            c[4],
            c[5],
            c[6],
            c[7],
            c[4],
        ],
    )
    front = NurbsCurve.create(
        False,
        1,
        [
            c[0],
            c[1],
            c[5],
            c[4],
            c[0],
        ],
    )
    right = NurbsCurve.create(
        False,
        1,
        [
            c[1],
            c[2],
            c[6],
            c[5],
            c[1],
        ],
    )
    back = NurbsCurve.create(
        False,
        1,
        [
            c[2],
            c[3],
            c[7],
            c[6],
            c[2],
        ],
    )
    left = NurbsCurve.create(
        False,
        1,
        [
            c[3],
            c[0],
            c[4],
            c[7],
            c[3],
        ],
    )

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
    from session_py import Primitives
    from session_py.tolerance import PI

    outer = NurbsCurve.create(
        False,
        1,
        [
            Point(-5, -5, 0),
            Point(5, -5, 0),
            Point(5, 5, 0),
            Point(-5, 5, 0),
            Point(-5, -5, 0),
        ],
    )
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


def _ring(pts, z):
    """Closed square ring at height z through the (x, y) corners"""

    from session_py import Point
    from session_py import Polyline

    v = []

    for p in pts:
        v.append(Point(p[0], p[1], z))

    v.append(Point(pts[0][0], pts[0][1], z))

    return Polyline(v)


def _box_with_square_hole():
    """4 x 4 x 2 box with a 1 x 1 through-hole along z: bottom and top with a hole, four outer and four inner side quads"""

    from session_py import Point
    from session_py import Polyline

    outer = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    inner = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    faces = [_ring(outer, 0.0), _ring(outer, 2.0)]
    holes = [[_ring(inner, 0.0)], [_ring(inner, 2.0)]]

    for loop in (outer, inner):
        for i in range(4):
            a = loop[i]
            b = loop[(i + 1) % 4]
            faces.append(
                Polyline(
                    [
                        Point(a[0], a[1], 0.0),
                        Point(b[0], b[1], 0.0),
                        Point(b[0], b[1], 2.0),
                        Point(a[0], a[1], 2.0),
                        Point(a[0], a[1], 0.0),
                    ]
                )
            )
            holes.append([])

    return faces, holes


@MINI_TEST("BRep", "From Polylines Holes")
def test_brep_from_polylines_holes():
    from session_py import BRep

    faces, holes = _box_with_square_hole()
    b = BRep.from_polylines(faces, holes)

    MINI_CHECK(b.face_count() == 10)
    MINI_CHECK(len(b.m_faces[0].wires) == 2)
    MINI_CHECK(b.is_solid())
    MINI_CHECK(abs(b.mesh().volume() - 30.0) < 1e-6)


@MINI_TEST("BRep", "Planar Fast Path")
def test_brep_planar_fast_path():
    from session_py import BRep

    faces, holes = _box_with_square_hole()
    b = BRep.from_polylines(faces, holes)
    fm = b.face_meshes()
    total = 0.0

    for m in fm:
        total += m.area()

    tagged = 0

    for vd in fm[0].vertex.values():
        for key in vd.attributes.keys():
            if str(key).startswith("brep_edge/"):
                tagged += 1
                break

    MINI_CHECK(len(fm) == 10)
    MINI_CHECK(len(fm[0].vertex) == 8 and len(fm[0].face) == 8)
    MINI_CHECK(len(fm[2].vertex) == 4 and len(fm[2].face) == 2)
    MINI_CHECK(abs(total - 70.0) < 1e-6)
    MINI_CHECK(tagged == 8)


@MINI_TEST("BRep", "Mesh Orientation")
def test_brep_mesh_orientation():
    from session_py import BRep
    from session_py.tolerance import PI

    bh = BRep.create_block_with_hole(8.0, 6.0, 4.0, 1.5)
    vol = bh.mesh().volume()
    ref = 8.0 * 6.0 * 4.0 - PI * 1.5 * 1.5 * 4.0

    MINI_CHECK(abs(vol - ref) / ref < 0.02)


@MINI_TEST("BRep", "Protobuf Roundtrip")
def test_brep_protobuf_roundtrip():
    from session_py import BRep
    from session_py import BRepOrientation
    from session_py import Color
    from pathlib import Path

    box = BRep.create_cylinder(1.0, 2.0)
    box.name = "test_brep"
    box.width = 2.0
    box.surfacecolor = Color(255, 128, 64, 255)

    loaded_proto = BRep.from_proto(box.to_proto())

    proto_string = box.pb_dumps()
    loaded_proto_string = BRep.pb_loads(proto_string)

    filename = Path(__file__).parent.parent.parent / "serialization" / "test_brep.bin"
    box.pb_dump(filename)
    loaded = BRep.pb_load(filename)

    MINI_CHECK(loaded_proto == box)
    MINI_CHECK(loaded_proto_string == box)
    MINI_CHECK(loaded == box)
    MINI_CHECK(loaded.is_solid())
    MINI_CHECK(loaded.m_edges[2].pcurves[0].curve_2d_index_2 >= 0)
    MINI_CHECK(loaded.m_wires[0].edges[2].orientation == BRepOrientation.Reversed)


@MINI_TEST("BRep", "Volume")
def test_brep_volume():
    from session_py import BRep
    from session_py.tolerance import PI

    box = BRep.create_box(2, 3, 4)
    cyl = BRep.create_cylinder(1.0, 4.0)
    sph = BRep.create_sphere(2.0)
    vbox = box.volume()
    vcyl = cyl.volume()
    vsph = sph.volume()

    MINI_CHECK(abs(vbox - 24.0) < 1e-9)
    MINI_CHECK(abs(vcyl - 4 * PI) / (4 * PI) < 0.05)
    MINI_CHECK(abs(vsph - (4.0 / 3.0) * PI * 8) / ((4.0 / 3.0) * PI * 8) < 0.05)


@MINI_TEST("BRep", "Face Polylines Box")
def test_brep_face_polylines_box():
    from session_py import BRep

    b = BRep.create_box(2.0, 2.0, 2.0)
    pls = b.face_polylines()
    pls_planes = b.face_planes()

    MINI_CHECK(len(pls) == 6)
    MINI_CHECK(len(pls_planes) == len(pls))

    seen = [[False, False], [False, False], [False, False]]

    for fi in range(len(pls)):
        p = pls[fi]
        pl = pls_planes[fi]
        MINI_CHECK(p.point_count() == 5)
        MINI_CHECK(p.get_point(0) == p.get_point(4))

        axis = -1
        sign = 0.0

        for a in range(3):
            constant = True

            for i in range(1, p.point_count()):
                if abs(p.get_point(i)[a] - p.get_point(0)[a]) > 1e-9:
                    constant = False
                    break

            if constant and abs(abs(p.get_point(0)[a]) - 1.0) < 1e-9:
                axis = a
                sign = 1.0 if p.get_point(0)[a] > 0 else -1.0
                break

        MINI_CHECK(axis >= 0)
        MINI_CHECK(abs(pl.origin[axis] - sign) < 1e-9)

        n = pl.z_axis
        MINI_CHECK(abs(abs(n[axis]) - 1.0) < 1e-6)

        for a2 in range(3):
            if a2 != axis:
                MINI_CHECK(abs(n[a2]) < 1e-6)

        normal_sign = 1 if n[axis] > 0.0 else 0
        MINI_CHECK(not seen[axis][normal_sign])
        seen[axis][normal_sign] = True

    for a in range(3):
        for s in range(2):
            MINI_CHECK(seen[a][s])


@MINI_TEST("BRep", "Face Polylines Cylinder Caps Only")
def test_brep_face_polylines_cylinder_caps_only():
    from session_py import BRep

    b = BRep.create_cylinder(1.0, 4.0)

    MINI_CHECK(b.face_count() == 3)
    MINI_CHECK(len(b.face_polylines()) == 2)
    MINI_CHECK(len(b.face_planes()) == 2)


@MINI_TEST("BRep", "Face Polylines Ignores Holes")
def test_brep_face_polylines_ignores_holes():
    from session_py import BRep
    import math

    b = BRep.create_block_with_hole(4.0, 4.0, 2.0, 1.0)
    pls = b.face_polylines()

    MINI_CHECK(len(pls) == 6)
    MINI_CHECK(len(pls) == len(b.face_planes()))

    for p in pls:
        MINI_CHECK(p.point_count() == 5)

        for i in range(p.point_count()):
            pt = p.get_point(i)
            on_bounds = (
                abs(abs(pt[0]) - 2.0) < 1e-6
                or abs(abs(pt[1]) - 2.0) < 1e-6
                or abs(abs(pt[2]) - 1.0) < 1e-6
            )
            MINI_CHECK(on_bounds)
            radius_to_z_axis = math.sqrt(pt[0] * pt[0] + pt[1] * pt[1])
            MINI_CHECK(radius_to_z_axis > 1.0 + 1e-6)


@MINI_TEST("BRep", "Face Polylines No Planar Faces")
def test_brep_face_polylines_no_planar_faces():
    from session_py import BRep

    b = BRep.create_sphere(1.0)

    MINI_CHECK(len(b.face_polylines()) == 0)
    MINI_CHECK(len(b.face_planes()) == 0)


@MINI_TEST("BRep", "Face Planes Reversed Flip")
def test_brep_face_planes_reversed_flip():
    from session_py import BRep
    from session_py import BRepOrientation
    from session_py import BRepRef

    b = BRep()
    fi_forward = _build_quad_face(b)
    b.add_shell([BRepRef(fi_forward, BRepOrientation.Forward)])
    fi_reversed = _build_quad_face(b)
    b.add_shell([BRepRef(fi_reversed, BRepOrientation.Reversed)])

    MINI_CHECK(b.face_orientation(fi_forward) == BRepOrientation.Forward)
    MINI_CHECK(b.face_orientation(fi_reversed) == BRepOrientation.Reversed)

    planes = b.face_planes()
    MINI_CHECK(len(planes) == 2)

    n_forward = planes[0].z_axis
    n_reversed = planes[1].z_axis
    MINI_CHECK(abs(n_forward[0] + n_reversed[0]) < 1e-9)
    MINI_CHECK(abs(n_forward[1] + n_reversed[1]) < 1e-9)
    MINI_CHECK(abs(n_forward[2] + n_reversed[2]) < 1e-9)


@MINI_TEST("BRep", "Face Planes Point Outward")
def test_brep_face_planes_point_outward():
    from session_py import BRep

    box = BRep.create_box(2.0, 2.0, 2.0)
    box_planes = box.face_planes()
    MINI_CHECK(len(box_planes) == 6)

    for pl in box_planes:
        o = pl.origin
        n = pl.z_axis
        d = o[0] * n[0] + o[1] * n[1] + o[2] * n[2]
        MINI_CHECK(d > 0.0)

    cyl = BRep.create_cylinder(1.0, 4.0)
    cyl_planes = cyl.face_planes()
    MINI_CHECK(len(cyl_planes) == 2)

    for pl in cyl_planes:
        o = pl.origin
        n = pl.z_axis
        mid_z = 2.0
        MINI_CHECK((o[2] - mid_z) * n[2] > 0.0)


@MINI_TEST("BRep", "Face Planes Outward Block With Hole")
def test_brep_face_planes_outward_block_with_hole():
    from session_py import BRep
    from session_py import Point

    b = BRep.create_block_with_hole(4.0, 4.0, 2.0, 1.0)
    MINI_CHECK(b.is_solid())

    solid_centroid = Point.centroid(b.vertex_points())
    planes = b.face_planes()
    MINI_CHECK(len(planes) == 6)

    for pl in planes:
        o = pl.origin
        n = pl.z_axis
        d = (
            (o[0] - solid_centroid[0]) * n[0]
            + (o[1] - solid_centroid[1]) * n[1]
            + (o[2] - solid_centroid[2]) * n[2]
        )
        MINI_CHECK(d > 0.0)


@MINI_TEST("BRep", "Face Planes Outward Under Mirrored Winding")
def test_brep_face_planes_outward_under_mirrored_winding():
    from session_py import BRep
    from session_py import BRepOrientation
    from session_py import Point
    from session_py import Xform

    mirror = Xform.scale_xyz(-1.0, 1.0, 1.0)
    b = BRep.create_box(2.0, 2.0, 2.0).transformed(mirror)
    MINI_CHECK(b.is_solid())

    pls = b.face_polylines()
    planes = b.face_planes()
    MINI_CHECK(len(pls) == 6)
    MINI_CHECK(len(planes) == 6)

    solid_centroid = Point.centroid(b.vertex_points())

    for fi in range(len(pls)):
        o = planes[fi].origin
        n = planes[fi].z_axis
        d = (
            (o[0] - solid_centroid[0]) * n[0]
            + (o[1] - solid_centroid[1]) * n[1]
            + (o[2] - solid_centroid[2]) * n[2]
        )
        MINI_CHECK(d > 0.0)

        expected = []

        for er in b.wire_edges(b.m_faces[fi].wires[0]):
            edge = b.m_edges[er.index]
            reversed_ = er.orientation == BRepOrientation.Reversed
            start = edge.end_vertex if reversed_ else edge.start_vertex
            expected.append(b.m_vertices[start].point)

        expected.append(expected[0])

        actual = pls[fi].get_points()
        MINI_CHECK(len(actual) == len(expected))

        for k in range(len(actual)):
            MINI_CHECK(actual[k] == expected[k])


if __name__ == "__main__":
    run_all()
