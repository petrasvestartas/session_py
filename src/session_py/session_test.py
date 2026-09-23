from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("Session", "Constructor")
def test_session_constructor():
    from session_py import Session

    session = Session()
    named = Session("my_named_session")

    MINI_CHECK(session.name == "my_session")
    MINI_CHECK(bool(session.guid))
    MINI_CHECK(named.name == "my_named_session")


@MINI_TEST("Session", "Copy")
def test_session_copy():
    from session_py import Session
    from session_py import Point
    from session_py import Element
    from session_py import Xform
    from copy import deepcopy

    session = Session("original")
    point = Point(1.0, 2.0, 3.0)
    element = Element("plate")
    group = session.add_group("Group")
    session.add_point(point, group)
    session.add_element(element, group)
    session.add_edge(point.guid, element.guid, "touching")
    session.set_xform(point.guid, Xform.translation(1.0, 0.0, 0.0))
    guid = session.guid

    copy = deepcopy(session)

    MINI_CHECK(copy.name == session.name)
    MINI_CHECK(copy.guid == guid)
    MINI_CHECK(len(copy.objects.points) == 1)
    MINI_CHECK(len(copy.objects.elements) == 1)
    MINI_CHECK(len(copy.lookup) == len(session.lookup))
    MINI_CHECK(copy.graph.number_of_edges() == 1)
    MINI_CHECK(len(copy.xforms) == 1)
    MINI_CHECK(
        len(copy.tree.root.descendants()) == len(session.tree.root.descendants())
    )

    MINI_CHECK(copy.objects.points is not session.objects.points)
    MINI_CHECK(copy.objects.elements is not session.objects.elements)
    MINI_CHECK(copy.tree.root is not session.tree.root)
    MINI_CHECK(copy.objects.points[0] is not session.objects.points[0])
    MINI_CHECK(copy.objects.points[0].guid == point.guid)
    MINI_CHECK(copy.objects.elements[0].guid == element.guid)
    MINI_CHECK(point.guid in copy.lookup)

    copy.objects.points.clear()
    copied_nodes = copy.tree.nodes
    MINI_CHECK(len(copied_nodes) > 1)
    copy.tree.remove(copied_nodes[1])
    MINI_CHECK(len(session.objects.points) == 1)
    MINI_CHECK(len(copy.objects.points) == 0)
    MINI_CHECK(len(session.tree.root.descendants()) > 0)
    MINI_CHECK(len(copy.tree.root.descendants()) == 0)


@MINI_TEST("Session", "Add Point")
def test_session_add_point():
    from session_py import Session
    from session_py import Point

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    session.add_point(point)

    MINI_CHECK(len(session.objects.points) == 1)
    MINI_CHECK(point.guid in session.lookup)
    MINI_CHECK(session.graph.has_node(point.guid))


@MINI_TEST("Session", "Add Line")
def test_session_add_line():
    from session_py import Session
    from session_py import Line
    from session_py import Point

    session = Session()
    line = Line(Point(0, 0, 0), Point(1, 0, 0))
    session.add_line(line)

    MINI_CHECK(len(session.objects.lines) == 1)
    MINI_CHECK(line.guid in session.lookup)


@MINI_TEST("Session", "Add Plane")
def test_session_add_plane():
    from session_py import Session
    from session_py import Plane

    session = Session()
    plane = Plane.xy_plane()
    session.add_plane(plane)

    MINI_CHECK(len(session.objects.planes) == 1)
    MINI_CHECK(plane.guid in session.lookup)


@MINI_TEST("Session", "Add OBB")
def test_session_add_obb():
    from session_py import Session
    from session_py import OBB
    from session_py import Point
    from session_py import Vector

    session = Session()
    obb = OBB(
        center=Point(0.0, 0.0, 0.0),
        x_axis=Vector(1.0, 0.0, 0.0),
        y_axis=Vector(0.0, 1.0, 0.0),
        z_axis=Vector(0.0, 0.0, 1.0),
        half_size=Vector(1.0, 1.0, 1.0),
    )
    session.add_obb(obb)

    MINI_CHECK(len(session.objects.bboxes) == 1)
    MINI_CHECK(obb.guid in session.lookup)


@MINI_TEST("Session", "Add Polyline")
def test_session_add_polyline():
    from session_py import Session
    from session_py import Polyline
    from session_py import Point

    session = Session()
    pl = Polyline([Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0)])
    session.add_polyline(pl)

    MINI_CHECK(len(session.objects.polylines) == 1)
    MINI_CHECK(pl.guid in session.lookup)


@MINI_TEST("Session", "Select By Type")
def test_session_select_by_type():
    from session_py import Session
    from session_py import Polyline
    from session_py import Point
    from session_py import Mesh

    session = Session()
    g0 = session.add_group("g0")
    g1 = session.add_group("g1")
    g2 = session.add_group("g2")

    session.add_polyline(Polyline([Point(0, 0, 0), Point(1, 0, 0)]), g0)
    session.add_polyline(Polyline([Point(0, 1, 0), Point(1, 1, 0)]), g0)
    session.add_polyline(Polyline([Point(0, 2, 0), Point(1, 2, 0)]), g1)
    session.add_point(Point(9, 9, 9), g2)

    groups = session.select_by_type(Polyline)

    MINI_CHECK(len(groups) == 2)
    MINI_CHECK(len(groups[0]) == 2)
    MINI_CHECK(len(groups[1]) == 1)
    MINI_CHECK(TOLERANCE.is_close(groups[1][0].get_point(0)[1], 2.0))

    MINI_CHECK(len(session.select_by_type(Mesh)) == 0)


@MINI_TEST("Session", "Add Pointcloud")
def test_session_add_pointcloud():
    from session_py import Session
    from session_py import PointCloud
    from session_py import Point

    session = Session()
    pc = PointCloud([Point(0, 0, 0), Point(1, 0, 0)])
    session.add_pointcloud(pc)

    MINI_CHECK(len(session.objects.pointclouds) == 1)
    MINI_CHECK(pc.guid in session.lookup)


@MINI_TEST("Session", "Add Mesh")
def test_session_add_mesh():
    from session_py import Session
    from session_py import Mesh
    from session_py import Point

    session = Session()
    mesh = Mesh()
    mesh.add_vertex(Point(0, 0, 0), 0)
    mesh.add_vertex(Point(1, 0, 0), 1)
    mesh.add_vertex(Point(0, 1, 0), 2)
    mesh.add_face([0, 1, 2])
    session.add_mesh(mesh)

    MINI_CHECK(len(session.objects.meshes) == 1)
    MINI_CHECK(mesh.guid in session.lookup)


@MINI_TEST("Session", "Add Nurbscurve")
def test_session_add_nurbscurve():
    from session_py import Session
    from session_py import NurbsCurve
    from session_py import Point

    session = Session()
    pts = [Point(0, 0, 0), Point(1, 1, 0), Point(2, 0, 0), Point(3, 1, 0)]
    nc = NurbsCurve.create(False, 2, pts)
    session.add_nurbscurve(nc)

    MINI_CHECK(len(session.objects.nurbscurves) == 1)
    MINI_CHECK(nc.guid in session.lookup)


@MINI_TEST("Session", "Add Nurbssurface")
def test_session_add_nurbssurface():
    from session_py import Session
    from session_py import NurbsSurface
    from session_py import Point

    session = Session()
    pts = [
        Point(0, 0, 0),
        Point(0, 1, 0),
        Point(0, 2, 0),
        Point(0, 3, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(1, 2, 0),
        Point(1, 3, 0),
        Point(2, 0, 0),
        Point(2, 1, 0),
        Point(2, 2, 0),
        Point(2, 3, 0),
        Point(3, 0, 0),
        Point(3, 1, 0),
        Point(3, 2, 0),
        Point(3, 3, 0),
    ]
    ns = NurbsSurface.create(False, False, 3, 3, 4, 4, pts)
    session.add_nurbssurface(ns)

    MINI_CHECK(len(session.objects.nurbssurfaces) == 1)
    MINI_CHECK(ns.guid in session.lookup)


@MINI_TEST("Session", "Add Brep")
def test_session_add_brep():
    from session_py import Session
    from session_py import BRep

    session = Session()
    brep = BRep.create_box(1.0, 1.0, 1.0)
    session.add_brep(brep)

    MINI_CHECK(len(session.objects.breps) == 1)
    MINI_CHECK(brep.guid in session.lookup)


@MINI_TEST("Session", "Add Element")
def test_session_add_element():
    from session_py import Session

    session = Session()
    from session_py import Element

    plate = Element("p1")
    session.add_element(plate)

    MINI_CHECK(len(session.objects.elements) == 1)
    MINI_CHECK(plate.guid in session.lookup)
    MINI_CHECK(session.graph.has_node(plate.guid))


@MINI_TEST("Session", "Add Empty Geometry")
def test_session_add_empty_geometry():
    from session_py import Session
    from session_py import Point
    from session_py import Polyline
    from session_py import PointCloud
    from session_py import Mesh
    from session_py import NurbsCurve
    from session_py import NurbsSurface
    from session_py import BRep

    session = Session()
    group = session.add_group("empty")

    MINI_CHECK(session.add_point(None, group) is None)
    MINI_CHECK(session.add_polyline(Polyline([Point(0, 0, 0)]), group) is None)
    MINI_CHECK(session.add_pointcloud(PointCloud([], [], []), group) is None)
    MINI_CHECK(session.add_mesh(Mesh(), group) is None)
    MINI_CHECK(session.add_nurbscurve(NurbsCurve(), group) is None)
    MINI_CHECK(session.add_nurbssurface(NurbsSurface(), group) is None)
    MINI_CHECK(session.add_brep(BRep(), group) is None)

    vertices_only = Mesh()
    vertices_only.add_vertex(Point(0, 0, 0), 0)
    MINI_CHECK(session.add_mesh(vertices_only, group) is None)

    session.add(session.add_mesh(Mesh(), group), group)

    MINI_CHECK(len(session.lookup) == 0)
    MINI_CHECK(len(session.order()) == 0)
    MINI_CHECK(len(group.children) == 0)


@MINI_TEST("Session", "Add Group")
def test_session_add_group():
    from session_py import Session

    session = Session()
    group = session.add_group("my_group")

    MINI_CHECK(group is not None)
    MINI_CHECK(group.name == "my_group")


@MINI_TEST("Session", "Add Edge")
def test_session_add_edge():
    from session_py import Session
    from session_py import Point

    session = Session()
    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(4.0, 5.0, 6.0)
    session.add_point(p1)
    session.add_point(p2)
    session.add_edge(p1.guid, p2.guid, "connection")

    MINI_CHECK(session.graph.has_edge((p1.guid, p2.guid)))


@MINI_TEST("Session", "Add Hierarchy")
def test_session_add_hierarchy():
    from session_py import Session
    from session_py import Point

    session = Session()
    p1 = Point(0, 0, 0)
    p2 = Point(1, 0, 0)
    n1 = session.add_point(p1)
    n2 = session.add_point(p2)
    session.add(n1)
    session.add(n2)
    ok = session.add_hierarchy(n1.guid, n2.guid)

    MINI_CHECK(ok)


@MINI_TEST("Session", "Get Children")
def test_session_get_children():
    from session_py import Session
    from session_py import Point

    session = Session()
    p1 = Point(0, 0, 0)
    p2 = Point(1, 0, 0)
    n1 = session.add_point(p1)
    n2 = session.add_point(p2)
    session.add(n1)
    session.add(n2)
    session.add_hierarchy(n1.guid, n2.guid)

    children = session.get_children(n1.guid)

    MINI_CHECK(len(children) == 1)
    MINI_CHECK(children[0] == n2.guid)


@MINI_TEST("Session", "Add Relationship")
def test_session_add_relationship():
    from session_py import Session
    from session_py import Point

    session = Session()
    p1 = Point(0, 0, 0)
    p2 = Point(1, 0, 0)
    session.add_point(p1)
    session.add_point(p2)
    session.add_relationship(p1.guid, p2.guid, "connects_to")

    MINI_CHECK(session.graph.has_edge((p1.guid, p2.guid)))


@MINI_TEST("Session", "Get Neighbours")
def test_session_get_neighbours():
    from session_py import Session
    from session_py import Point

    session = Session()
    p1 = Point(0, 0, 0)
    p2 = Point(1, 0, 0)
    session.add_point(p1)
    session.add_point(p2)
    session.add_edge(p1.guid, p2.guid, "connection")

    neighbours = session.get_neighbours(p1.guid)

    MINI_CHECK(len(neighbours) == 1)
    MINI_CHECK(neighbours[0] == p2.guid)


@MINI_TEST("Session", "Get Collisions")
def test_session_get_collisions():
    from session_py import Session
    from session_py import OBB
    from session_py import Point
    from session_py import Vector

    session = Session()
    obb1 = OBB(
        center=Point(0.0, 0.0, 0.0),
        x_axis=Vector(1.0, 0.0, 0.0),
        y_axis=Vector(0.0, 1.0, 0.0),
        z_axis=Vector(0.0, 0.0, 1.0),
        half_size=Vector(2.0, 2.0, 2.0),
    )
    obb2 = OBB(
        center=Point(1.0, 0.0, 0.0),
        x_axis=Vector(1.0, 0.0, 0.0),
        y_axis=Vector(0.0, 1.0, 0.0),
        z_axis=Vector(0.0, 0.0, 1.0),
        half_size=Vector(2.0, 2.0, 2.0),
    )
    session.add_obb(obb1)
    session.add_obb(obb2)
    pairs = session.get_collisions()

    MINI_CHECK(len(pairs) >= 1)


@MINI_TEST("Session", "Ray Cast")
def test_session_ray_cast():
    from session_py import Session
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    session = Session()
    mesh = Mesh()
    mesh.add_vertex(Point(-1.0, -1.0, 0.0), 0)
    mesh.add_vertex(Point(1.0, -1.0, 0.0), 1)
    mesh.add_vertex(Point(0.0, 1.0, 0.0), 2)
    mesh.add_face([0, 1, 2])
    session.add_mesh(mesh)
    hits = session.ray_cast(Point(0.0, 0.0, 2.0), Vector(0.0, 0.0, -1.0))

    MINI_CHECK(len(hits) >= 1)

    from session_py import Xform

    placed = Mesh()
    placed.add_vertex(Point(-1.0, -1.0, 0.0), 0)
    placed.add_vertex(Point(1.0, -1.0, 0.0), 1)
    placed.add_vertex(Point(0.0, 1.0, 0.0), 2)
    placed.add_face([0, 1, 2])
    placed_guid = placed.guid
    session.add_mesh(placed)
    session.set_xform(placed_guid, Xform.translation(100.0, 0.0, 0.0))
    hits2 = session.ray_cast(Point(100.0, 0.0, 2.0), Vector(0.0, 0.0, -1.0))

    MINI_CHECK(len(hits2) >= 1)
    MINI_CHECK(TOLERANCE.is_close(hits2[0].hit_point[0], 100.0))


@MINI_TEST("Session", "Get Object")
def test_session_get_object():
    from session_py import Session
    from session_py import Point

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    session.add_point(point)
    retrieved = session.get_object(point.guid)

    MINI_CHECK(retrieved is not None)
    MINI_CHECK(retrieved.guid == point.guid)


@MINI_TEST("Session", "Remove Object")
def test_session_remove_object():
    from session_py import Session
    from session_py import Point
    from pathlib import Path

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    session.add_point(point)
    removed = session.remove_object(point.guid)

    from session_py import Element

    plate = Element("p1")
    eguid = plate.guid
    session.add_element(plate)
    eremoved = session.remove_object(eguid)

    fname = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_session_remove.bin"
    )
    session.pb_dump(fname)
    loaded = Session.pb_load(fname)

    MINI_CHECK(removed)
    MINI_CHECK(point.guid not in session.lookup)
    MINI_CHECK(eremoved)
    MINI_CHECK(len(session.objects.elements) == 0)
    MINI_CHECK(eguid not in loaded.lookup)


@MINI_TEST("Session", "Get Geometry")
def test_session_get_geometry():
    from session_py import Session
    from session_py import Point

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    session.add_point(point)

    geom = session.get_geometry()

    MINI_CHECK(len(geom.points) == 1)


@MINI_TEST("Session", "Get Geometry Is Pure")
def test_session_get_geometry_is_pure():
    from session_py import Session
    from session_py import Point
    from session_py import Xform

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    guid = point.guid
    session.add_point(point)
    session.set_xform(guid, Xform.translation(10.0, 0.0, 0.0))

    first = session.get_geometry().points[0]
    second = session.get_geometry().points[0]

    MINI_CHECK(TOLERANCE.is_close(first[0], 11.0))
    MINI_CHECK(TOLERANCE.is_close(second[0], 11.0))
    MINI_CHECK(TOLERANCE.is_close(point[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(session.objects.points[0][0], 1.0))


@MINI_TEST("Session", "Json Roundtrip")
def test_session_json_roundtrip():
    from session_py import Session
    from session_py import Point
    from pathlib import Path

    session = Session()
    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(4.0, 5.0, 6.0)
    session.add_point(p1)
    session.add_point(p2)
    session.add_edge(p1.guid, p2.guid, "connection")

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_session.json"
    session.file_json_dump(fname)
    loaded = Session.file_json_load(fname)

    MINI_CHECK(loaded.name == session.name)
    MINI_CHECK(len(loaded.lookup) == len(session.lookup))
    MINI_CHECK(loaded.graph.number_of_vertices() == session.graph.number_of_vertices())


@MINI_TEST("Session", "Protobuf Roundtrip")
def test_session_protobuf_roundtrip():
    from session_py import Session
    from session_py import Point
    from pathlib import Path

    session = Session()
    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(4.0, 5.0, 6.0)
    session.add_point(p1)
    session.add_point(p2)
    session.add_edge(p1.guid, p2.guid, "connection")

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_session.bin"
    session.pb_dump(fname)
    loaded = Session.pb_load(fname)

    MINI_CHECK(loaded.name == session.name)
    MINI_CHECK(len(loaded.lookup) == len(session.lookup))


@MINI_TEST("Session", "Lookup Mutation Roundtrip")
def test_session_lookup_mutation_roundtrip():
    from session_py import Session
    from session_py import Line
    from pathlib import Path

    session = Session()
    line = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    guid = line.guid
    session.add_line(line)

    session.lookup[guid].width = 5.0

    fname = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_session_lookup.bin"
    )
    session.pb_dump(fname)
    loaded = Session.pb_load(fname)

    MINI_CHECK(loaded.objects.lines[0].width == 5.0)
    MINI_CHECK(loaded.lookup[guid].width == 5.0)


@MINI_TEST("Session", "Order")
def test_session_order():
    from session_py import Session
    from session_py import Line
    from session_py import Point
    from pathlib import Path

    session = Session()
    line = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    point = Point(1.0, 2.0, 3.0)
    line_guid = line.guid
    point_guid = point.guid
    session.add_line(line)
    session.add_point(point)

    order = session.order()

    fname = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_session_order.bin"
    )
    session.pb_dump(fname)
    loaded = Session.pb_load(fname)

    MINI_CHECK(len(order) == 2)
    MINI_CHECK(order[0] == point_guid)
    MINI_CHECK(order[1] == line_guid)
    MINI_CHECK(loaded.order() == order)


@MINI_TEST("Session", "Set Xform")
def test_session_set_xform():
    from session_py import Session
    from session_py import Point
    from session_py import Xform

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    guid = point.guid
    session.add_point(point)

    shift = Xform.translation(5.0, 0.0, 0.0)
    session.set_xform(guid, shift)

    MINI_CHECK(session.xform(guid) == shift)
    MINI_CHECK(session.world_xform(guid) == shift)
    MINI_CHECK(session.world_xforms()[guid] == shift)
    MINI_CHECK(session.xform("missing") == Xform.identity())
    MINI_CHECK(session.remove_xform(guid))
    MINI_CHECK(session.xform(guid) == Xform.identity())


@MINI_TEST("Session", "World Xform Hierarchy")
def test_session_world_xform_hierarchy():
    from session_py import Session
    from session_py import Point
    from session_py import Xform

    session = Session()
    a = Point(0.0, 0.0, 0.0)
    b = Point(0.0, 0.0, 0.0)
    c = Point(0.0, 0.0, 0.0)
    a_guid = a.guid
    b_guid = b.guid
    c_guid = c.guid
    a_node = session.add_point(a)
    b_node = session.add_point(b)
    c_node = session.add_point(c)

    session.add(a_node)
    session.add(b_node, a_node)
    session.add(c_node, b_node)

    a_xform = Xform.rotation_z(PI / 2.0)
    b_xform = Xform.translation(2.0, 0.0, 0.0)
    c_xform = Xform.rotation_z(PI / 2.0)
    session.set_xform(a_guid, a_xform)
    session.set_xform(b_guid, b_xform)
    session.set_xform(c_guid, c_xform)

    world = session.world_xforms()

    MINI_CHECK(session.world_xform(a_guid) == a_xform)
    MINI_CHECK(session.world_xform(b_guid) == a_xform * b_xform)
    MINI_CHECK(session.world_xform(c_guid) == a_xform * b_xform * c_xform)
    MINI_CHECK(world[c_guid] == session.world_xform(c_guid))


@MINI_TEST("Session", "Xform Roundtrip")
def test_session_xform_roundtrip():
    from session_py import Session
    from session_py import Point
    from session_py import Xform
    from pathlib import Path

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    guid = point.guid
    session.add_point(point)
    session.set_xform(guid, Xform.translation(7.0, 8.0, 9.0))

    fname = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_session_xform.bin"
    )
    session.pb_dump(fname)
    loaded = Session.pb_load(fname)
    json_loaded = Session.file_json_loads(session.file_json_dumps())

    MINI_CHECK(loaded.xform(guid) == session.xform(guid))
    MINI_CHECK(len(loaded.xforms) == 1)
    MINI_CHECK(json_loaded.xform(guid) == session.xform(guid))
    MINI_CHECK(len(json_loaded.xforms) == 1)


def create_box(center, size):
    """A cube mesh of the given size centred on center."""

    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    h = size * 0.5
    verts = [
        Point(center[0] - h, center[1] - h, center[2] - h),
        Point(center[0] + h, center[1] - h, center[2] - h),
        Point(center[0] + h, center[1] + h, center[2] - h),
        Point(center[0] - h, center[1] + h, center[2] - h),
        Point(center[0] - h, center[1] - h, center[2] + h),
        Point(center[0] + h, center[1] - h, center[2] + h),
        Point(center[0] + h, center[1] + h, center[2] + h),
        Point(center[0] - h, center[1] + h, center[2] + h),
    ]

    for i in range(len(verts)):
        mesh.add_vertex(verts[i], i)

    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [2, 6, 7, 3],
        [0, 3, 7, 4],
        [1, 5, 6, 2],
    ]

    for f in faces:
        mesh.add_face(f)

    return mesh


@MINI_TEST("Session", "Tree Transformation Hierarchy")
def test_session_tree_transformation_hierarchy():
    from session_py import Session
    from session_py import Point
    from session_py import Vector
    from session_py import Xform
    from session_py import Plane

    scene = Session("tree_transformation_test")

    box1 = create_box(Point(0, 0, 0), 2.0)
    box1_guid = box1.guid
    box1_node = scene.add_mesh(box1)
    box2 = create_box(Point(0, 0, 0), 2.0)
    box2_guid = box2.guid
    box2_node = scene.add_mesh(box2)
    box3 = create_box(Point(0, 0, 0), 2.0)
    box3_guid = box3.guid
    box3_node = scene.add_mesh(box3)

    scene.add(box1_node)
    scene.add(box2_node, box1_node)
    scene.add(box3_node, box2_node)

    plane_from = Plane(Point(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
    plane_to = Plane(Point(0, 0, 1.0), Vector(1, 0, 0), Vector(0, 1, 0))
    xy_to_top = Xform.plane_to_plane(plane_from, plane_to)
    scene.set_xform(box1_guid, Xform.rotation_z(PI / 1.5) * xy_to_top)
    scene.set_xform(
        box2_guid, Xform.translation(2.0, 0, 0) * Xform.rotation_z(PI / 6.0)
    )
    scene.set_xform(box3_guid, Xform.translation(2.0, 0, 0))

    world3 = scene.world_xform(box3_guid)
    expected = world3.transform_point(Point(-1.0, -1.0, -1.0))
    transformed = scene.get_geometry()
    baked = transformed.meshes[2].vertex_point(0)

    MINI_CHECK(len(transformed.meshes) == 3)
    MINI_CHECK(TOLERANCE.is_close(baked[0], expected[0]))
    MINI_CHECK(TOLERANCE.is_close(baked[1], expected[1]))
    MINI_CHECK(TOLERANCE.is_close(baked[2], expected[2]))


@MINI_TEST("Session", "Add Component")
def test_session_add_component():
    from session_py import Session
    from session_py import Component

    session = Session()

    c = Component()
    c.type_name = "FloorBuilder"
    c.name = "floor_builder"
    c.extra = {"size": 3000, "height": 650}
    guid = c.guid

    session.add_component(c)

    MINI_CHECK(len(session.objects.components) == 1)
    MINI_CHECK(guid in session.component_lookup)
    MINI_CHECK(session.graph.has_node(guid))


@MINI_TEST("Session", "Component Json Roundtrip")
def test_session_component_json_roundtrip():
    from session_py import Session
    from session_py import Component
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = Session()
    c = Component()
    c.type_name = "FloorBuilder"
    c.name = "floor_builder"
    c.extra = {"size": 3000, "height": 650, "rise": 453}
    guid = c.guid
    original.add_component(c)

    filename = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_session_component.json"
    )
    file_json_dump(original, filename)
    loaded = file_json_load(filename)

    MINI_CHECK(len(loaded.objects.components) == 1)
    MINI_CHECK(loaded.objects.components[0].type_name == "FloorBuilder")
    MINI_CHECK(loaded.objects.components[0].extra["size"] == 3000)
    MINI_CHECK(loaded.objects.components[0].guid == guid)


@MINI_TEST("Session", "Document Workflow")
def test_session_document_workflow():
    from session_py import Session
    from session_py import Point
    from session_py import Xform
    from pathlib import Path

    session = Session()
    a = Point(1.0, 0.0, 0.0)
    b = Point(2.0, 0.0, 0.0)
    c = Point(3.0, 0.0, 0.0)
    a_guid = a.guid
    b_guid = b.guid
    c_guid = c.guid
    session.add_point(a)
    session.add_point(b)
    session.add_point(c)

    session.replace(b_guid, Point(20.0, 0.0, 0.0))
    session.remove_object(c_guid)
    shift = Xform.translation(0.0, 5.0, 0.0)
    session.set_xform(a_guid, shift)

    fname = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_session_document.bin"
    )
    session.pb_dump(fname)
    loaded = Session.pb_load(fname)

    MINI_CHECK(len(loaded.lookup) == 2)
    MINI_CHECK(a_guid in loaded.lookup)
    MINI_CHECK(b_guid in loaded.lookup)
    MINI_CHECK(c_guid not in loaded.lookup)
    MINI_CHECK(TOLERANCE.is_close(loaded.lookup[b_guid][0], 20.0))
    MINI_CHECK(loaded.xform(a_guid) == shift)
    MINI_CHECK(loaded.history.depth() == 0)


@MINI_TEST("Session", "Undo Remove")
def test_session_undo_remove():
    from session_py import Session
    from session_py import Point
    from session_py import Xform

    session = Session()
    group = session.add_group("g")
    a = Point(1.0, 0.0, 0.0)
    b = Point(2.0, 0.0, 0.0)
    c = Point(3.0, 0.0, 0.0)
    a_guid = a.guid
    b_guid = b.guid
    c_guid = c.guid
    session.add_point(a, group)
    b_node = session.add_point(b, group)
    session.add_point(c, b_node)
    session.add_edge(a_guid, b_guid, "connection")
    shift = Xform.translation(0.0, 5.0, 0.0)
    session.set_xform(b_guid, shift)

    session.begin("remove")
    session.remove_object(b_guid)
    session.commit()
    gone = b_guid not in session.lookup and len(group.children) == 1
    session.undo()

    MINI_CHECK(gone)
    MINI_CHECK(b_guid in session.lookup)
    MINI_CHECK(session.objects.points[1].guid == b_guid)
    MINI_CHECK(group.children[1].name == b_guid)
    MINI_CHECK(group.children[1].children[0].name == c_guid)
    MINI_CHECK(session.graph.has_edge((a_guid, b_guid)))
    MINI_CHECK(session.graph.edge_label(a_guid, b_guid) == "connection")
    MINI_CHECK(session.xform(b_guid) == shift)

    session.redo()

    MINI_CHECK(b_guid not in session.lookup)
    MINI_CHECK(len(session.objects.points) == 2)
    MINI_CHECK(len(group.children) == 1)
    MINI_CHECK(not session.graph.has_edge((a_guid, b_guid)))
    MINI_CHECK(session.xform(b_guid) == Xform.identity())


@MINI_TEST("Session", "Undo Add")
def test_session_undo_add():
    from session_py import Session
    from session_py import Point

    session = Session()
    group = session.add_group("g")
    session.add_point(Point(0.0, 0.0, 0.0), group)
    point = Point(1.0, 2.0, 3.0)
    guid = point.guid

    session.begin("add")
    session.add_point(point, group)
    session.commit()
    session.undo()
    gone = guid not in session.lookup and len(session.objects.points) == 1
    session.redo()

    MINI_CHECK(gone)
    MINI_CHECK(guid in session.lookup)
    MINI_CHECK(session.objects.points[1].guid == guid)
    MINI_CHECK(group.children[1].name == guid)
    MINI_CHECK(session.graph.has_node(guid))
    MINI_CHECK(TOLERANCE.is_close(session.lookup[guid][2], 3.0))


@MINI_TEST("Session", "Undo Replace")
def test_session_undo_replace():
    from session_py import Session
    from session_py import Point

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    guid = point.guid
    session.add_point(point)

    session.begin("replace")
    session.replace(guid, Point(9.0, 9.0, 9.0))
    session.commit()
    replaced = session.lookup[guid][0]
    session.undo()
    restored = session.lookup[guid][0]
    session.redo()

    MINI_CHECK(TOLERANCE.is_close(replaced, 9.0))
    MINI_CHECK(TOLERANCE.is_close(restored, 1.0))
    MINI_CHECK(TOLERANCE.is_close(session.lookup[guid][0], 9.0))
    MINI_CHECK(session.objects.points[0].guid == guid)
    MINI_CHECK(len(session.objects.points) == 1)


@MINI_TEST("Session", "Undo Xform")
def test_session_undo_xform():
    from session_py import Session
    from session_py import Point
    from session_py import Xform

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    guid = point.guid
    session.add_point(point)
    shift = Xform.translation(5.0, 0.0, 0.0)

    session.begin("move")
    session.set_xform(guid, shift)
    session.commit()
    session.undo()
    cleared = session.xform(guid) == Xform.identity()
    session.redo()

    session.begin("reset")
    session.remove_xform(guid)
    session.commit()
    session.undo()

    MINI_CHECK(cleared)
    MINI_CHECK(session.xform(guid) == shift)
    MINI_CHECK(len(session.xforms) == 1)


@MINI_TEST("Session", "History Purged On Save")
def test_session_history_purged_on_save():
    from session_py import Session
    from session_py import Point

    session = Session()

    session.begin("add")
    session.add_point(Point(0.0, 0.0, 0.0))
    session.commit()
    before_pb = session.history.depth()
    session.pb_dumps()
    after_pb = session.history.depth()

    session.begin("add")
    session.add_point(Point(1.0, 0.0, 0.0))
    session.commit()
    before_json = session.history.depth()
    session.file_json_dumps()

    MINI_CHECK(before_pb == 1)
    MINI_CHECK(after_pb == 0)
    MINI_CHECK(before_json == 1)
    MINI_CHECK(session.history.depth() == 0)
    MINI_CHECK(not session.undo())
    MINI_CHECK(len(session.objects.points) == 2)


@MINI_TEST("Session", "History Capacity")
def test_session_history_capacity():
    from session_py import Session
    from session_py import Point

    session = Session()

    for i in range(70):
        session.begin("add")
        session.add_point(Point(float(i), 0.0, 0.0))
        session.commit()

    depth = session.history.depth()

    while session.undo():
        pass

    MINI_CHECK(depth == 64)
    MINI_CHECK(not session.history.can_undo())
    MINI_CHECK(len(session.objects.points) == 6)
    MINI_CHECK(TOLERANCE.is_close(session.objects.points[5][0], 5.0))


@MINI_TEST("Session", "Str Hierarchy")
def test_session_str_hierarchy():
    from session_py import Session
    from session_py import Point

    session = Session("blocks")
    group = session.add_group("Group")
    session.add_point(Point(0.0, 0.0, 0.0), group)
    session.add_point(Point(1.0, 0.0, 0.0), group)
    text = str(session)

    MINI_CHECK("Spatial Hierarchy" in text)
    MINI_CHECK("Element Interactions" in text)
    MINI_CHECK("\u2514\u2500\u2500 " in text)
    MINI_CHECK("<Tree with " in text)
    MINI_CHECK("<Graph with " in text)
    MINI_CHECK(repr(session).startswith("Session(name=blocks"))


@MINI_TEST("Session", "Add Definition")
def test_session_add_definition():
    from session_py import Session
    from session_py import Point
    from session_py import Xform

    session = Session()
    box = create_box(Point(0, 0, 0), 2.0)
    guid = session.add_definition(box)
    again = session.add_definition(box)
    session.set_xform(guid, Xform.translation(1.0, 0.0, 0.0))
    point = Point(1.0, 2.0, 3.0)
    session.add_point(point)
    taken = session.add_definition(point)

    MINI_CHECK(guid == box.guid)
    MINI_CHECK(again == guid)
    MINI_CHECK(taken == "")
    MINI_CHECK(len(session.definitions.meshes) == 1)
    MINI_CHECK(guid in session.definition_lookup)
    MINI_CHECK(guid not in session.lookup)
    MINI_CHECK(len(session.order()) == 1)
    MINI_CHECK(not session.graph.has_node(guid))
    MINI_CHECK(session.tree.get_node_by_name(guid) is None)
    MINI_CHECK(len(session.xforms) == 0)


@MINI_TEST("Session", "Add Instance")
def test_session_add_instance():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Xform

    session = Session()
    group = session.add_group("bay")
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    instance = InstanceRef(definition, Xform.translation(0.0, 0.0, 3.0))
    instance.name = "column"
    guid = instance.guid
    node = session.add_instance(instance, Xform.translation(10.0, 0.0, 0.0), group)
    orphan = session.add_instance(InstanceRef("missing", Xform.identity()))

    MINI_CHECK(node.name == guid)
    MINI_CHECK(group.children[0].name == guid)
    MINI_CHECK(session.graph.node_label(guid) == "instance_column")
    MINI_CHECK(len(session.objects.instances) == 1)
    MINI_CHECK(session.instance_lookup[guid].xform == Xform.identity())
    MINI_CHECK(session.xform(guid) == Xform.translation(10.0, 0.0, 3.0))
    MINI_CHECK(orphan is None)
    MINI_CHECK(len(session.order()) == 0)


@MINI_TEST("Session", "Definition Of")
def test_session_definition_of():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Xform

    session = Session()
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    instance = InstanceRef(definition, Xform.identity())
    guid = instance.guid
    session.add_instance(instance)
    found = session.definition_of(guid)

    MINI_CHECK(found is not None)
    MINI_CHECK(found.guid == definition)
    MINI_CHECK(session.definition_of(definition) is None)
    MINI_CHECK(session.definition_of("missing") is None)


@MINI_TEST("Session", "Instances Of")
def test_session_instances_of():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Xform

    session = Session()
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    first = InstanceRef(definition, Xform.identity())
    second = InstanceRef(definition, Xform.identity())
    first_guid = first.guid
    second_guid = second.guid
    session.add_instance(first)
    session.add_instance(second)
    guids = session.instances_of(definition)

    MINI_CHECK(len(guids) == 2)
    MINI_CHECK(guids[0] == first_guid)
    MINI_CHECK(guids[1] == second_guid)
    MINI_CHECK(len(session.instances_of("missing")) == 0)


@MINI_TEST("Session", "World Geometry")
def test_session_world_geometry():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Xform

    session = Session()
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    instance = InstanceRef(definition, Xform.identity())
    instance.name = "box"
    guid = instance.guid
    session.add_instance(instance, Xform.translation(10.0, 0.0, 0.0))
    point = Point(1.0, 2.0, 3.0)
    session.add_point(point)
    session.set_xform(point.guid, Xform.translation(0.0, 0.0, 5.0))

    mesh = session.world_geometry(guid)
    moved = session.world_geometry(point.guid)
    local = session.definition_lookup[definition]

    MINI_CHECK(mesh.guid == guid)
    MINI_CHECK(mesh.name == "box")
    MINI_CHECK(TOLERANCE.is_close(mesh.vertex_point(0)[0], 9.0))
    MINI_CHECK(TOLERANCE.is_close(local.vertex_point(0)[0], -1.0))
    MINI_CHECK(TOLERANCE.is_close(moved[2], 8.0))
    MINI_CHECK(TOLERANCE.is_close(point[2], 3.0))
    MINI_CHECK(session.world_geometry("missing") is None)


@MINI_TEST("Session", "Get Geometry Resolves Instances")
def test_session_get_geometry_resolves_instances():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Xform

    session = Session()
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    group = session.add_group("row")
    session.set_xform("row", Xform.translation(0.0, 5.0, 0.0))
    session.add_instance(
        InstanceRef(definition, Xform.identity()),
        Xform.translation(10.0, 0.0, 0.0),
        group,
    )
    session.add_instance(
        InstanceRef(definition, Xform.identity()),
        Xform.translation(20.0, 0.0, 0.0),
        group,
    )

    geometry = session.get_geometry()
    corner = geometry.meshes[1].vertex_point(0)

    MINI_CHECK(len(geometry.instances) == 0)
    MINI_CHECK(len(geometry.meshes) == 2)
    MINI_CHECK(TOLERANCE.is_close(corner[0], 19.0))
    MINI_CHECK(TOLERANCE.is_close(corner[1], 4.0))
    MINI_CHECK(len(session.objects.instances) == 2)
    MINI_CHECK(len(session.objects.meshes) == 0)


@MINI_TEST("Session", "Replace Definition")
def test_session_replace_definition():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Xform

    session = Session()
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    first = InstanceRef(definition, Xform.identity())
    second = InstanceRef(definition, Xform.identity())
    second_guid = second.guid
    session.add_instance(first, Xform.translation(10.0, 0.0, 0.0))
    session.add_instance(second, Xform.translation(20.0, 0.0, 0.0))

    replaced = session.replace_definition(definition, create_box(Point(0, 0, 0), 4.0))
    missing = session.replace_definition("missing", create_box(Point(0, 0, 0), 4.0))
    mesh = session.world_geometry(second_guid)

    MINI_CHECK(replaced)
    MINI_CHECK(not missing)
    MINI_CHECK(len(session.definitions.meshes) == 1)
    MINI_CHECK(session.definitions.meshes[0].guid == definition)
    MINI_CHECK(TOLERANCE.is_close(mesh.vertex_point(0)[0], 18.0))


@MINI_TEST("Session", "Remove Definition")
def test_session_remove_definition():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Xform

    session = Session()
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    instance = InstanceRef(definition, Xform.identity())
    guid = instance.guid
    session.add_instance(instance)

    refused = not session.remove_definition(definition)
    session.remove_object(guid)
    removed = session.remove_definition(definition)

    MINI_CHECK(refused)
    MINI_CHECK(removed)
    MINI_CHECK(len(session.definitions.meshes) == 0)
    MINI_CHECK(len(session.definition_lookup) == 0)
    MINI_CHECK(len(session.objects.instances) == 0)
    MINI_CHECK(not session.remove_definition("missing"))


@MINI_TEST("Session", "To Instance")
def test_session_to_instance():
    from session_py import Session
    from session_py import Point
    from session_py import Xform

    session = Session()
    group = session.add_group("bay")
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    point = Point(0.0, 0.0, 0.0)
    box = create_box(Point(5.0, 0.0, 0.0), 2.0)
    box.name = "column"
    guid = box.guid
    session.add_point(point, group)
    session.add_mesh(box, group)
    session.add_edge(point.guid, guid, "contact")
    session.set_xform(guid, Xform.translation(0.0, 0.0, 1.0))

    before = session.world_geometry(guid).vertex_point(0)
    converted = session.to_instance(guid, definition, Xform.translation(5.0, 0.0, 0.0))
    after = session.world_geometry(guid).vertex_point(0)

    MINI_CHECK(converted)
    MINI_CHECK(len(session.objects.meshes) == 0)
    MINI_CHECK(session.instance_lookup[guid].name == "column")
    MINI_CHECK(session.instance_lookup[guid].definition_guid == definition)
    MINI_CHECK(group.children[1].name == guid)
    MINI_CHECK(session.graph.has_edge((point.guid, guid)))
    MINI_CHECK(session.graph.node_label(guid) == "instance_column")
    MINI_CHECK(TOLERANCE.is_close(before[0], after[0]))
    MINI_CHECK(TOLERANCE.is_close(before[2], after[2]))
    MINI_CHECK(not session.to_instance(guid, definition, Xform.identity()))


@MINI_TEST("Session", "Explode")
def test_session_explode():
    from session_py import Session
    from session_py import Element
    from session_py import ElementFeature
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Polyline
    from session_py import Xform

    session = Session()
    definition = session.add_definition(
        Element(create_box(Point(0, 0, 0), 2.0), "plate")
    )
    instance = InstanceRef(definition, Xform.identity())
    instance.name = "deck"
    instance.features.append(
        ElementFeature("contact", 0, [Polyline([Point(0, 0, 0), Point(1, 0, 0)])])
    )
    guid = instance.guid
    feature = instance.features[0].guid
    point = Point(0.0, 0.0, 0.0)
    session.add_point(point)
    session.add_instance(instance, Xform.translation(10.0, 0.0, 0.0))
    session.add_edge(point.guid, guid, "contact")

    exploded = session.explode(guid)
    element = session.get_object(guid)

    MINI_CHECK(exploded)
    MINI_CHECK(len(session.objects.instances) == 0)
    MINI_CHECK(element.name == "deck")
    MINI_CHECK(len(element.features) == 1)
    MINI_CHECK(element.features[0].guid == feature)
    MINI_CHECK(session.xform(guid) == Xform.translation(10.0, 0.0, 0.0))
    MINI_CHECK(session.graph.has_edge((point.guid, guid)))
    MINI_CHECK(session.graph.node_label(guid) == "element_deck")
    MINI_CHECK(len(session.definitions.elements) == 1)
    MINI_CHECK(not session.explode(guid))


@MINI_TEST("Session", "Undo Instance")
def test_session_undo_instance():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Xform

    session = Session()
    group = session.add_group("bay")
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    point = Point(0.0, 0.0, 0.0)
    box = create_box(Point(0, 0, 0), 2.0)
    guid = box.guid
    session.add_point(point, group)
    session.add_mesh(box, group)
    session.add_edge(point.guid, guid, "contact")
    edge = session.graph.edges[point.guid][guid].guid
    order = session.order()
    tree = str(session.tree)
    label = session.graph.node_label(guid)

    session.begin("to instance")
    session.to_instance(guid, definition, Xform.identity())
    session.commit()
    session.begin("explode")
    session.explode(guid)
    session.commit()
    session.undo()
    instanced = guid in session.instance_lookup and str(session.tree) == tree
    session.undo()

    instance = InstanceRef(definition, Xform.identity())
    added = instance.guid
    session.begin("add")
    session.add_instance(instance, Xform.translation(1.0, 0.0, 0.0), group)
    session.commit()
    session.undo()
    gone = len(session.instance_lookup) == 0 and len(session.xforms) == 0
    session.redo()

    MINI_CHECK(instanced)
    MINI_CHECK(gone)
    MINI_CHECK(session.objects.meshes[0].guid == guid)
    MINI_CHECK(session.graph.has_edge((point.guid, guid)))
    MINI_CHECK(session.graph.edges[guid][point.guid].guid == edge)
    MINI_CHECK(session.graph.node_label(guid) == label)
    MINI_CHECK(session.order() == order)
    MINI_CHECK(session.xform(added) == Xform.translation(1.0, 0.0, 0.0))
    MINI_CHECK(group.children[2].name == added)


@MINI_TEST("Session", "Instance Json Roundtrip")
def test_session_instance_json_roundtrip():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Xform
    from pathlib import Path

    session = Session()
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    instance = InstanceRef(definition, Xform.identity())
    guid = instance.guid
    session.add_instance(instance, Xform.translation(10.0, 0.0, 0.0))

    fname = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_session_instance.json"
    )
    session.file_json_dump(fname)
    loaded = Session.file_json_load(fname)
    data = session.__jsondump__()
    data["objects"]["instances"][0]["xform"] = Xform.translation(
        0.0, 0.0, 1.0
    ).__jsondump__()
    folded = Session.__jsonload__(data)

    MINI_CHECK(len(loaded.definitions.meshes) == 1)
    MINI_CHECK(guid in loaded.instance_lookup)
    MINI_CHECK(loaded.definition_of(guid) is not None)
    MINI_CHECK(loaded.xform(guid) == Xform.translation(10.0, 0.0, 0.0))
    MINI_CHECK(folded.xform(guid) == Xform.translation(10.0, 0.0, 1.0))
    MINI_CHECK(folded.instance_lookup[guid].xform == Xform.identity())


@MINI_TEST("Session", "Instance Protobuf Roundtrip")
def test_session_instance_protobuf_roundtrip():
    from session_py import Session
    from session_py import ElementFeature
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Polyline
    from session_py import Xform
    from session_py.proto import session_pb2
    from pathlib import Path

    session = Session()
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    instance = InstanceRef(definition, Xform.identity())
    instance.features.append(
        ElementFeature("contact", 0, [Polyline([Point(0, 0, 0), Point(1, 0, 0)])])
    )
    guid = instance.guid
    feature = instance.features[0].guid
    session.add_instance(instance, Xform.translation(10.0, 0.0, 0.0))

    fname = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_session_instance.bin"
    )
    session.pb_dump(fname)
    loaded = Session.pb_load(fname)
    plain = session_pb2.Session()
    plain.ParseFromString(Session().pb_dumps())

    MINI_CHECK(len(loaded.definitions.meshes) == 1)
    MINI_CHECK(len(loaded.instance_lookup[guid].features) == 1)
    MINI_CHECK(loaded.instance_lookup[guid].features[0].guid == feature)
    MINI_CHECK(loaded.definition_of(guid) is not None)
    MINI_CHECK(loaded.xform(guid) == Xform.translation(10.0, 0.0, 0.0))
    MINI_CHECK(not plain.HasField("definitions"))


@MINI_TEST("Session", "Get Collisions Instances")
def test_session_get_collisions_instances():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Xform

    session = Session()
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    first = InstanceRef(definition, Xform.identity())
    second = InstanceRef(definition, Xform.identity())
    third = InstanceRef(definition, Xform.identity())
    first_guid = first.guid
    second_guid = second.guid
    third_guid = third.guid
    session.add_instance(first)
    session.add_instance(second, Xform.translation(1.0, 0.0, 0.0))
    session.add_instance(third, Xform.translation(100.0, 0.0, 0.0))

    pairs = session.get_collisions()

    MINI_CHECK(len(pairs) == 1)
    MINI_CHECK(session.graph.has_edge((first_guid, second_guid)))
    MINI_CHECK(not session.graph.has_edge((first_guid, third_guid)))


@MINI_TEST("Session", "Ray Cast Instance")
def test_session_ray_cast_instance():
    from session_py import Session
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Vector
    from session_py import Xform

    session = Session()
    definition = session.add_definition(create_box(Point(0, 0, 0), 2.0))
    instance = InstanceRef(definition, Xform.identity())
    guid = instance.guid
    session.add_instance(instance, Xform.translation(100.0, 0.0, 0.0))

    hits = session.ray_cast(Point(100.0, 0.0, 5.0), Vector(0.0, 0.0, -1.0))

    MINI_CHECK(len(hits) == 1)
    MINI_CHECK(hits[0].guid == guid)
    MINI_CHECK(TOLERANCE.is_close(hits[0].hit_point[0], 100.0))
    MINI_CHECK(TOLERANCE.is_close(hits[0].hit_point[2], 1.0))


if __name__ == "__main__":
    run_all(language="python")
