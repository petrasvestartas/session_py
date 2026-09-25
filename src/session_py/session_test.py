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
    element = Element(name="plate")
    group = session.add_group("Group")
    session.add_point(point, group)
    session.add_element(element, group)
    session.add_edge(point.guid, element.guid, "touching")
    session.set_xform(point.guid, Xform.translation(1.0, 0.0, 0.0))
    guid = session.guid

    copy = deepcopy(session)

    MINI_CHECK(copy.name == session.name)
    MINI_CHECK(copy.guid == guid)
    MINI_CHECK(copy.history.depth() == 0)
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
    from session_py import Element

    session = Session()
    plate = Element(name="p1")
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


def _named_interaction_class():
    """A test-only subclass: a named interaction with no state of its own."""

    from session_py import Interaction

    class NamedInteraction(Interaction):
        def interaction_type_name(self):
            """Return the registered type name."""
            return "NamedInteraction"

        def interaction_data_dumps(self):
            """Return no state."""
            return b""

    return NamedInteraction


def _named_interaction(data):
    """Build a NamedInteraction from its data."""
    return _named_interaction_class()()


@MINI_TEST("Session", "Add Interaction")
def test_session_add_interaction():
    from session_py import Session
    from session_py import Element

    NamedInteraction = _named_interaction_class()

    session = Session()
    a = Element(name="a")
    b = Element(name="b")
    absent = Element(name="absent")
    session.add_element(a)
    session.add_element(b)
    session.add_edge(a.guid, b.guid, "authored")
    glue = session.add_interaction(a, b, NamedInteraction("glue"))
    id = session.graph.edges[a.guid][b.guid].guid
    screw = session.add_interaction(b, a, NamedInteraction("screw"))

    MINI_CHECK(len(session.interactions) == 1)
    MINI_CHECK(len(session.interactions[id]) == 2)
    MINI_CHECK(session.interactions[id][0].guid == glue.guid)
    MINI_CHECK(session.interactions[id][1].guid == screw.guid)
    MINI_CHECK(session.graph.number_of_edges() == 1)
    MINI_CHECK(session.graph.edges[b.guid][a.guid].guid == id)
    MINI_CHECK(session.graph.edges[a.guid][b.guid].attribute == "authored")

    missing_rejected = False
    self_rejected = False

    try:
        session.add_interaction(a, absent, NamedInteraction())
    except ValueError:
        missing_rejected = True

    try:
        session.add_interaction(a, a, NamedInteraction())
    except ValueError:
        self_rejected = True

    MINI_CHECK(missing_rejected)
    MINI_CHECK(self_rejected)
    MINI_CHECK(session.graph.number_of_edges() == 1)


@MINI_TEST("Session", "Get Interaction")
def test_session_get_interaction():
    import copy
    from session_py import Session
    from session_py import Element
    from session_py import Interaction

    NamedInteraction = _named_interaction_class()
    Interaction.register_type("NamedInteraction", _named_interaction)

    session = Session()
    a = Element(name="a")
    b = Element(name="b")
    c = Element(name="c")
    session.add_element(a)
    session.add_element(b)
    session.add_element(c)
    session.add_edge(a.guid, c.guid, "authored")
    before = session.get_interaction(a, b)
    bare = session.get_interaction(a, c)
    glue = session.add_interaction(a, b, NamedInteraction("glue"))
    id = session.graph.edges[a.guid][b.guid].guid
    duplicate = copy.deepcopy(session)
    duplicate.interactions[id][0].name = "screw"
    loaded_b = Session.pb_loads(session.pb_dumps())
    loaded_j = Session.file_json_loads(session.file_json_dumps())

    MINI_CHECK(len(before) == 0)
    MINI_CHECK(len(bare) == 0)
    MINI_CHECK(session.get_interaction(a, b)[0].guid == glue.guid)
    MINI_CHECK(session.get_interaction(b, a)[0].guid == glue.guid)
    MINI_CHECK(session.get_interaction(a, b)[0].name == "glue")
    MINI_CHECK(duplicate.get_interaction(b, a)[0].name == "screw")
    MINI_CHECK(duplicate.get_interaction(b, a)[0].guid == glue.guid)
    MINI_CHECK(loaded_b.get_interaction(b, a)[0] == glue)
    MINI_CHECK(loaded_b.get_interaction(b, a)[0].guid == glue.guid)
    MINI_CHECK(loaded_j.get_interaction(b, a)[0] == glue)


@MINI_TEST("Session", "Has Interaction")
def test_session_has_interaction():
    from session_py import Session
    from session_py import Element
    from session_py import Interaction

    NamedInteraction = _named_interaction_class()
    Interaction.register_type("NamedInteraction", _named_interaction)

    session = Session()
    a = Element(name="a")
    b = Element(name="b")
    absent = Element(name="absent")
    session.add_element(a)
    session.add_element(b)
    before = session.has_interaction(a, b)
    session.add_interaction(a, b, NamedInteraction())
    loaded = Session.pb_loads(session.pb_dumps())

    MINI_CHECK(not before)
    MINI_CHECK(session.has_interaction(a, b))
    MINI_CHECK(session.has_interaction(b, a))
    MINI_CHECK(not session.has_interaction(a, absent))
    MINI_CHECK(loaded.has_interaction(b, a))


@MINI_TEST("Session", "Remove Interaction")
def test_session_remove_interaction():
    from session_py import Session
    from session_py import Element

    NamedInteraction = _named_interaction_class()

    session = Session()
    a = Element(name="a")
    b = Element(name="b")
    c = Element(name="c")
    session.add_element(a)
    session.add_element(b)
    session.add_element(c)
    session.add_interaction(a, b, NamedInteraction("glue"))
    session.add_interaction(a, c, NamedInteraction())
    session.remove_interaction(b, a)
    session.remove_interaction(b, a)

    MINI_CHECK(not session.has_interaction(a, b))
    MINI_CHECK(session.has_interaction(a, c))
    MINI_CHECK(len(session.get_interaction(a, b)) == 0)
    MINI_CHECK(len(session.interactions) == 1)
    MINI_CHECK(session.graph.number_of_edges() == 1)
    MINI_CHECK(session.graph.has_node(b.guid))


@MINI_TEST("Session", "Undo Remove Interaction")
def test_session_undo_remove_interaction():
    from session_py import Session
    from session_py import Element

    NamedInteraction = _named_interaction_class()

    session = Session()
    a = Element(name="a")
    b = Element(name="b")
    session.add_element(a)
    session.add_element(b)
    glue = session.add_interaction(a, b, NamedInteraction("glue"))
    id = session.graph.edges[a.guid][b.guid].guid

    session.begin("remove")
    session.remove_object(b.guid)
    session.commit()
    dropped = id not in session.interactions
    session.undo()

    MINI_CHECK(dropped)
    MINI_CHECK(session.graph.edges[a.guid][b.guid].guid == id)
    MINI_CHECK(len(session.get_interaction(a, b)) == 1)
    MINI_CHECK(session.get_interaction(a, b)[0].guid == glue.guid)
    MINI_CHECK(session.get_interaction(a, b)[0].name == "glue")

    session.redo()

    MINI_CHECK(id not in session.interactions)


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
    from session_py import Xform

    session = Session()
    mesh = Mesh()
    mesh.add_vertex(Point(-1.0, -1.0, 0.0), 0)
    mesh.add_vertex(Point(1.0, -1.0, 0.0), 1)
    mesh.add_vertex(Point(0.0, 1.0, 0.0), 2)
    mesh.add_face([0, 1, 2])
    session.add_mesh(mesh)
    hits = session.ray_cast(Point(0.0, 0.0, 2.0), Vector(0.0, 0.0, -1.0))

    MINI_CHECK(len(hits) >= 1)

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
    from session_py import Element
    from pathlib import Path

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    session.add_point(point)
    removed = session.remove_object(point.guid)

    plate = Element(name="p1")
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
    MINI_CHECK(session.objects.elements.number_of_slots() == 0)
    MINI_CHECK(session.number_of_dead() == 0)
    MINI_CHECK(not session.graph.has_node(eguid))
    MINI_CHECK(eguid not in loaded.lookup)
    MINI_CHECK(loaded.objects.points.number_of_slots() == 0)


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
    converted = Session.from_proto(session.to_proto())

    MINI_CHECK(loaded.name == session.name)
    MINI_CHECK(len(loaded.lookup) == len(session.lookup))
    MINI_CHECK(len(converted.lookup) == len(session.lookup))
    MINI_CHECK(converted.graph.has_edge((p1.guid, p2.guid)))


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
    MINI_CHECK(len(session.objects.points) == 2)
    MINI_CHECK(session.objects.points.number_of_slots() == 2)


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
    stored = session.objects.points[1]

    session.begin("remove")
    session.remove_object(b_guid)
    session.commit()
    gone = b_guid not in session.lookup and len(group.children) == 1
    session.undo()

    MINI_CHECK(gone)
    MINI_CHECK(b_guid in session.lookup)
    MINI_CHECK(session.objects.points[1].guid == b_guid)
    MINI_CHECK(session.objects.points[1] is stored)
    MINI_CHECK(group.children[1].name == b_guid)
    MINI_CHECK(group.children[1] is b_node)
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
    MINI_CHECK(session.objects.points.number_of_slots() == 2)
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
    MINI_CHECK(session.objects.points.number_of_slots() == 1)


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
    MINI_CHECK(session.history.undo_stack[0].ops[0].kind == "xform")


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
    MINI_CHECK(session.number_of_dead() == 0)
    MINI_CHECK(not session.undo())
    MINI_CHECK(len(session.objects.points) == 2)


@MINI_TEST("Session", "Purge On Save")
def test_session_purge_on_save():
    from session_py import Point
    from session_py import Session

    session = Session()
    guids = []

    for i in range(5):
        node = session.add_point(Point(float(i), 0.0, 0.0))
        guids.append(node.name)

    for i in [1, 3]:
        session.begin("remove")
        session.remove_object(guids[i])
        session.commit()

    data = session.pb_dumps()
    loaded = Session.pb_loads(data)
    indices = []

    for vertex in session.graph.get_vertices():
        indices.append(vertex.index)

    indices.sort()

    MINI_CHECK(session.history.depth() == 0)
    MINI_CHECK(session.number_of_dead() == 0)
    MINI_CHECK(session.objects.points.number_of_slots() == 3)
    MINI_CHECK(indices == [0, 1, 2])
    MINI_CHECK(loaded.order() == [guids[0], guids[2], guids[4]])
    MINI_CHECK(loaded.pb_dumps() == data)


@MINI_TEST("Session", "Purge Unreachable")
def test_session_purge_unreachable():
    from session_py import Point
    from session_py import Session
    from session_py.session import PURGE_WORK

    session = Session()
    guids = []

    for i in range(70):
        node = session.add_point(Point(float(i), 0.0, 0.0))
        guids.append(node.name)

    for guid in guids:
        session.begin("remove")
        session.remove_object(guid)
        session.commit()

    due = session.purge_due()

    while session.purge_step(PURGE_WORK):
        pass

    dead = session.number_of_dead()
    undone = 0

    while session.undo():
        undone += 1

    MINI_CHECK(due)
    MINI_CHECK(dead == 64)
    MINI_CHECK(undone == 64)
    MINI_CHECK(len(session.objects.points) == 64)


@MINI_TEST("Session", "Purge Step")
def test_session_purge_step():
    from session_py import Point
    from session_py import Session
    from session_py import Xform

    session = Session()
    guids = []

    for i in range(10_000):
        node = session.add_point(Point(float(i), 0.0, 0.0))
        guids.append(node.name)

    session.begin("remove")

    for guid in guids[::2]:
        session.remove_object(guid)

    session.commit()

    for i in range(64):
        session.begin("move")
        session.set_xform(guids[1], Xform.translation(float(i), 0.0, 0.0))
        session.commit()

    expected = guids[1::2]
    first = session.purge_step(64)
    ordered = session.order() == expected
    calls = 1

    while session.purge_step(64):
        calls += 1

        if calls == 10:
            session.begin("remove")
            session.remove_object(guids[3])
            session.commit()
            expected.remove(guids[3])

        if calls == 20:
            session.undo()
            expected = guids[1::2]

        if calls == 30:
            node = session.add_point(Point(0.0, 1.0, 0.0))
            expected.append(node.name)

        ordered &= session.order() == expected

    MINI_CHECK(first)
    MINI_CHECK(calls > 30)
    MINI_CHECK(ordered)
    MINI_CHECK(session.order() == expected)
    MINI_CHECK(session.number_of_dead() == 0)
    MINI_CHECK(session.objects.points.number_of_slots() == len(session.objects.points))
    MINI_CHECK(len(session.tree.root.children) == len(expected))


@MINI_TEST("Session", "Checkpoint Keeps History")
def test_session_checkpoint_keeps_history():
    import copy
    import sys
    from session_py import Color
    from session_py import Point
    from session_py import Session
    from session_py import Xform

    session = Session()
    group = session.add_group("group")
    a = Point(0.0, 0.0, 0.0)
    b = Point(1.0, 0.0, 0.0)
    c = Point(2.0, 0.0, 0.0)
    a_guid = a.guid
    b_guid = b.guid
    c_guid = c.guid
    session.add_point(a, group)
    session.add_point(b, group)
    session.add_point(c)
    session.set_node_color(group, Color(1.0, 0.0, 0.0, 1.0))
    session.set_xform("group", Xform.translation(0.0, 0.0, 1.0))
    session.set_xform(c_guid, Xform.translation(5.0, 0.0, 0.0))
    session.add_edge(a_guid, c_guid, "touch")
    whole = session.checkpoint(sys.maxsize)
    duplicate = copy.deepcopy(session).pb_dumps()

    session.begin("remove")
    session.remove_object(b_guid)
    session.commit()
    data = None
    calls = 0

    while data is None:
        data = session.checkpoint(16)
        calls += 1

    loaded = Session.pb_loads(data)

    MINI_CHECK(whole == duplicate)
    MINI_CHECK(calls > 1)
    MINI_CHECK(data == session.to_proto().SerializeToString(deterministic=True))
    MINI_CHECK(session.history.can_undo())
    MINI_CHECK(b_guid not in loaded.lookup)
    MINI_CHECK(loaded.order() == session.order())
    MINI_CHECK(session.undo())
    MINI_CHECK(b_guid in session.lookup)


@MINI_TEST("Session", "Checkpoint Restarts On Edit")
def test_session_checkpoint_restarts_on_edit():
    from session_py import Point
    from session_py import Session

    session = Session()
    guids = []

    for i in range(20):
        node = session.add_point(Point(float(i), 0.0, 0.0))
        guids.append(node.name)

    first = session.checkpoint(10)
    session.begin("remove")
    session.remove_object(guids[5])
    session.commit()
    data = None

    while data is None:
        data = session.checkpoint(10)

    loaded = Session.pb_loads(data)

    MINI_CHECK(first is None)
    MINI_CHECK(data == session.to_proto().SerializeToString(deterministic=True))
    MINI_CHECK(len(loaded.objects.points) == 19)
    MINI_CHECK(guids[5] not in loaded.lookup)
    MINI_CHECK(session.history.can_undo())


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
    MINI_CHECK(session.objects.points.number_of_slots() == 70)
    MINI_CHECK(session.history.dropped == 6)
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
    MINI_CHECK(session.definitions.meshes.number_of_dead() == 1)
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
    session.begin("to instance")
    converted = session.to_instance(guid, definition, Xform.translation(5.0, 0.0, 0.0))
    session.commit()
    after = session.world_geometry(guid).vertex_point(0)

    MINI_CHECK(converted)
    MINI_CHECK(len(session.objects.meshes) == 0)
    MINI_CHECK(session.objects.meshes.number_of_slots() == 1)
    MINI_CHECK(session.history.can_undo())
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
    MINI_CHECK(session.objects.instances.number_of_dead() == 1)
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
    MINI_CHECK(session.objects.meshes.number_of_slots() == 2)
    MINI_CHECK(len(session.objects.instances) == 1)
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


@MINI_TEST("Session", "Get Node")
def test_session_get_node():
    from session_py import Point
    from session_py import Session
    from session_py import Tree
    from session_py import TreeNode

    session = Session()
    node = session.add_point(Point(0.0, 0.0, 0.0))
    guid = node.name
    child = session.add_point(Point(1.0, 0.0, 0.0), node)
    child_guid = child.name
    found = session.get_node(guid)
    session.begin("remove")
    session.remove_object(guid)
    session.commit()
    removed = session.get_node(guid)
    orphaned = session.get_node(child_guid)
    session.undo()
    restored = session.get_node(guid)
    reattached = session.get_node(child_guid)
    indexed = session.node_lookup.get(child_guid) is child
    tree = Tree("swapped")
    tree.add(TreeNode("root"))
    root = tree.root
    swapped = TreeNode(guid)
    tree.add(swapped, root)
    session.tree = tree
    searched = session.get_node(guid)
    session.reindex()

    MINI_CHECK(found is node)
    MINI_CHECK(removed is None)
    MINI_CHECK(orphaned is child)
    MINI_CHECK(restored is node)
    MINI_CHECK(reattached is child and indexed)
    MINI_CHECK(searched is swapped)
    MINI_CHECK(session.node_lookup[guid] is swapped)
    MINI_CHECK(session.get_node("missing") is None)


@MINI_TEST("Session", "Remove Keeps Slot")
def test_session_remove_keeps_slot():
    from session_py import Point
    from session_py import Session
    from session_py import Xform

    session = Session()
    g = session.add_group("g")
    a = Point(1.0, 0.0, 0.0)
    b = Point(2.0, 0.0, 0.0)
    c = Point(3.0, 0.0, 0.0)
    a_guid = a.guid
    b_guid = b.guid
    c_guid = c.guid
    session.add_point(a, g)
    b_node = session.add_point(b, g)
    session.add_point(c, g)
    session.set_xform(b_guid, Xform.translation(0.0, 1.0, 0.0))
    stored = session.objects.points[1]

    session.begin("remove")
    session.remove_object(b_guid)
    session.commit()
    pinned = session.history.undo_stack[0].ops[0].tomb.node is b_node

    MINI_CHECK(len(session.objects.points) == 2)
    MINI_CHECK(session.objects.points.number_of_slots() == 3)
    MINI_CHECK(session.objects.points.get_item(1) is stored)
    MINI_CHECK(b_guid not in session.lookup)
    MINI_CHECK(b_guid not in session.xforms)
    MINI_CHECK(not session.graph.has_node(b_guid))
    MINI_CHECK(session.get_node(b_guid) is None)
    MINI_CHECK(pinned)

    session.undo()
    names = [n.name for n in g.children]

    MINI_CHECK(session.objects.points.get_item(1) is stored)
    MINI_CHECK(session.objects.points.get_slot(b_guid) == 1)
    MINI_CHECK(session.order() == [a_guid, b_guid, c_guid])
    MINI_CHECK(names == [a_guid, b_guid, c_guid])
    MINI_CHECK(g.children[1] is b_node)
    MINI_CHECK(b_node.at() == 1)


@MINI_TEST("Session", "Redo Add Keeps Node")
def test_session_redo_add_keeps_node():
    from session_py import Point
    from session_py import Session

    session = Session()
    a = Point(1.0, 0.0, 0.0)
    a_guid = a.guid

    session.begin("add")
    node = session.add_point(a)
    session.commit()
    node_guid = node.guid
    stored = session.objects.points[0]
    session.undo()
    gone = session.get_node(a_guid) is None and len(session.objects.points) == 0
    session.redo()
    found = session.get_node(a_guid)

    MINI_CHECK(gone)
    MINI_CHECK(found is node)
    MINI_CHECK(node.guid == node_guid)
    MINI_CHECK(session.objects.points[0] is stored)
    MINI_CHECK(session.objects.points.get_slot(a_guid) == 0)
    MINI_CHECK(len(session.tree.root.children) == 1)


@MINI_TEST("Session", "Replace Shares Geometry")
def test_session_replace_shares_geometry():
    from session_py import Point
    from session_py import Session

    session = Session()
    a = Point(1.0, 0.0, 0.0)
    a_guid = a.guid
    session.add_point(a)
    original = session.objects.points[0]
    p2 = Point(9.0, 9.0, 9.0)
    p2.guid = a_guid

    session.begin("replace")
    session.replace(a_guid, p2)
    session.commit()
    op = session.history.undo_stack[0].ops[0]
    shared = op.kind == "replace" and op.before is original and op.after is p2
    swapped = session.objects.points[0] is p2
    session.undo()
    restored = (
        session.objects.points[0] is original
        and session.objects.points.get_slot(a_guid) == 0
    )
    session.redo()

    MINI_CHECK(shared)
    MINI_CHECK(swapped)
    MINI_CHECK(restored)
    MINI_CHECK(session.objects.points[0] is p2)
    MINI_CHECK(session.lookup[a_guid] is p2)


@MINI_TEST("Session", "Replace Across Types")
def test_session_replace_across_types():
    from session_py import Line
    from session_py import Point
    from session_py import Session

    session = Session()
    a = Point(1.0, 0.0, 0.0)
    a_guid = a.guid
    node = session.add_point(a)
    line = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    line.name = "edge"

    session.begin("replace")
    replaced = session.replace(a_guid, line)
    session.commit()
    label = session.graph.node_label(a_guid)
    same_node = session.get_node(a_guid) is node
    counts = (len(session.objects.points), len(session.objects.lines))
    session.undo()
    undone = (len(session.objects.points), len(session.objects.lines))
    restored = session.graph.node_label(a_guid)
    session.redo()

    MINI_CHECK(replaced)
    MINI_CHECK(counts == (0, 1))
    MINI_CHECK(same_node)
    MINI_CHECK(label == "line_edge")
    MINI_CHECK(undone == (1, 0))
    MINI_CHECK(restored == "point_my_point")
    MINI_CHECK(len(session.objects.lines) == 1)
    MINI_CHECK(len(session.objects.points) == 0)
    MINI_CHECK(isinstance(session.lookup[a_guid], Line))
    MINI_CHECK(session.get_node(a_guid) is node)


@MINI_TEST("Session", "Undo Restores Graph")
def test_session_undo_restores_graph():
    from session_py import Point
    from session_py import Session

    session = Session()
    a = Point(0.0, 0.0, 0.0)
    b = Point(1.0, 0.0, 0.0)
    c = Point(2.0, 0.0, 0.0)
    a_guid = a.guid
    b_guid = b.guid
    c_guid = c.guid
    session.add_point(a)
    session.add_point(b)
    session.add_point(c)
    session.graph.set_vertex_attribute(a_guid, "mass", 2.0)
    session.add_edge(a_guid, b_guid, "joint")
    session.graph.set_edge_attribute((a_guid, b_guid), "load", 1.5)
    session.add_edge(a_guid, c_guid, "contact")
    session.add_edge(b_guid, c_guid, "contact")
    before = session.graph.file_json_dumps()

    session.begin("remove")
    session.remove_object(a_guid)
    session.commit()
    taken = not session.graph.has_node(a_guid) and session.graph.number_of_edges() == 1
    session.undo()
    after = session.graph.file_json_dumps()
    session.redo()

    MINI_CHECK(taken)
    MINI_CHECK(before == after)
    MINI_CHECK(not session.graph.has_node(a_guid))
    MINI_CHECK(session.graph.has_edge((b_guid, c_guid)))
    MINI_CHECK(session.graph.number_of_vertices() == 2)

    c_node = session.get_node(c_guid)
    session.tree.remove(c_node)
    session.remove_object(c_guid)

    MINI_CHECK(not session.graph.has_node(c_guid))


@MINI_TEST("Session", "Undo Restores Interactions")
def test_session_undo_restores_interactions():
    from session_py import Element
    from session_py import Session

    NamedInteraction = _named_interaction_class()

    session = Session()
    session.add_element(Element(name="a"))
    session.add_element(Element(name="b"))
    session.add_element(Element(name="c"))
    a = session.objects.elements[0]
    b = session.objects.elements[1]
    c = session.objects.elements[2]
    glue = session.add_interaction(a, b, NamedInteraction("glue")).guid
    nail = session.add_interaction(a, c, NamedInteraction("nail")).guid
    ab = session.graph.edges[a.guid][b.guid].guid
    ac = session.graph.edges[a.guid][c.guid].guid

    session.begin("remove")
    session.remove_object(a.guid)
    session.commit()
    parked = len(session.interactions) == 0
    session.undo()
    session.redo()
    session.undo()

    MINI_CHECK(parked)
    MINI_CHECK(len(session.interactions) == 2)
    MINI_CHECK(session.graph.edges[a.guid][b.guid].guid == ab)
    MINI_CHECK(session.graph.edges[a.guid][c.guid].guid == ac)
    MINI_CHECK(session.get_interaction(a, b)[0].guid == glue)
    MINI_CHECK(session.get_interaction(a, c)[0].guid == nail)
    MINI_CHECK(session.get_interaction(a, c)[0].name == "nail")


@MINI_TEST("Session", "Dead Guid Reused")
def test_session_dead_guid_reused():
    from session_py import Point
    from session_py import Session

    session = Session()
    x = Point(1.0, 0.0, 0.0)
    x_guid = x.guid
    session.add_point(x)
    again = Point(9.0, 0.0, 0.0)
    again.guid = x_guid

    session.begin("reuse")
    session.remove_object(x_guid)
    session.add_point(again)
    session.commit()
    first = session.lookup[x_guid][0]
    slots = (
        session.objects.points.is_dead(0),
        session.objects.points.get_slot(x_guid),
    )
    count = len(session.objects.points)
    session.undo()
    second = session.lookup[x_guid][0]
    undone = (
        session.objects.points.is_dead(1),
        session.objects.points.get_slot(x_guid),
    )
    still = len(session.objects.points)
    session.redo()

    MINI_CHECK(TOLERANCE.is_close(first, 9.0))
    MINI_CHECK(slots == (True, 1))
    MINI_CHECK(count == 1)
    MINI_CHECK(TOLERANCE.is_close(second, 1.0))
    MINI_CHECK(undone == (True, 0))
    MINI_CHECK(still == 1)
    MINI_CHECK(session.objects.points.get_slot(x_guid) == 1)
    MINI_CHECK(len(session.objects.points) == 1)


@MINI_TEST("Session", "Remove Keeps Lookup Edit")
def test_session_remove_keeps_lookup_edit():
    from session_py import Point
    from session_py import Session
    import json

    session = Session()
    a = Point(1.0, 0.0, 0.0)
    a_guid = a.guid
    session.add_point(a)
    edited = Point(7.0, 0.0, 0.0)
    edited.guid = a_guid
    session.lookup[a_guid] = edited

    session.begin("remove")
    session.remove_object(a_guid)
    session.commit()
    session.undo()
    text = json.dumps(session.__jsondump__())

    MINI_CHECK(TOLERANCE.is_close(session.lookup[a_guid][0], 7.0))
    MINI_CHECK(TOLERANCE.is_close(session.objects.points[0][0], 7.0))
    MINI_CHECK("7.0" in text)


@MINI_TEST("Session", "Tree Ops")
def test_session_tree_ops():
    from session_py import Color
    from session_py import Point
    from session_py import Session
    from session_py import TreeNode
    from session_py import Xform

    session = Session()
    session.add_group("A")
    root = session.tree.root
    snapshots = [str(session.tree)]

    session.begin("group")
    node = session.add_group("L")
    session.commit()
    snapshots.append(str(session.tree))

    session.begin("rename")
    renamed = session.rename_node(node, "M")
    session.commit()
    snapshots.append(str(session.tree))

    session.begin("colour")
    coloured = session.set_node_color(node, Color(1.0, 0.0, 0.0, 1.0))
    session.commit()
    snapshots.append(str(session.tree))

    session.begin("remove")
    removed = session.remove_group(node)
    session.commit()
    dead = node.is_dead() and len(root.children) == 1
    restored = []

    for i in range(3, -1, -1):
        session.undo()
        restored.append(str(session.tree) == snapshots[i])

    absent = node.is_dead() and session.tree.get_node_by_name("L") is None
    redone = []

    for i in range(1, 5):
        session.redo()
        redone.append(i == 4 or str(session.tree) == snapshots[i])

    MINI_CHECK(renamed and coloured and removed)
    MINI_CHECK(dead)
    MINI_CHECK(restored == [True, True, True, True])
    MINI_CHECK(absent)
    MINI_CHECK(redone == [True, True, True, True])
    MINI_CHECK(node.is_dead())
    MINI_CHECK(node.name == "M")
    MINI_CHECK(node.at() == 1)
    MINI_CHECK(node.color is not None)
    MINI_CHECK(session.tree.root is root)
    MINI_CHECK(session.history.undo_stack[3].ops[0].kind == "tree")

    point = Point(0.0, 0.0, 0.0)
    guid = point.guid
    held = session.add_point(point)
    session.tree.remove(held)
    session.set_xform(guid, Xform.translation(1.0, 0.0, 0.0))
    session.begin("adopt")
    session.add(TreeNode(guid))
    session.commit()
    session.undo()

    MINI_CHECK(session.get_node(guid) is None)
    MINI_CHECK(guid in session.xforms)


@MINI_TEST("Session", "Move Node")
def test_session_move_node():
    from session_py import Point
    from session_py import Session

    session = Session()
    g1 = session.add_group("g1")
    g2 = session.add_group("g2")
    session.add_point(Point(0.0, 0.0, 0.0), g1)
    x = session.add_point(Point(1.0, 0.0, 0.0), g1)
    y = session.add_point(Point(2.0, 0.0, 0.0), x)
    count = len(session.tree.nodes)

    session.begin("move")
    session.add(x, g2)
    session.commit()
    queued = g1.is_queued()
    moved = (
        len(g1.children) == 1
        and len(g2.children) > 0
        and g2.children[-1] is x
        and y.parent is x
    )
    walked = len(session.tree.nodes)
    session.undo()
    back = len(g1.children) > 1 and g1.children[1] is x and len(g2.children) == 0
    text = str(session.tree)
    session.redo()

    MINI_CHECK(moved)
    MINI_CHECK(queued)
    MINI_CHECK(g2.is_queued())
    MINI_CHECK(walked == count)
    MINI_CHECK(back)
    MINI_CHECK(len(session.tree.nodes) == count)
    MINI_CHECK("TreeNode(, " not in text)
    MINI_CHECK(len(g2.children) == 1)
    MINI_CHECK(len(g1.children) == 1)
    MINI_CHECK(x.parent is g2)


@MINI_TEST("Session", "Deleted Parent Orphans Children")
def test_session_deleted_parent_orphans_children():
    from session_py import Element
    from session_py import Point
    from session_py import Polyline
    from session_py import Session
    from session_py import TreeNode
    from session_py import Xform

    session = Session()
    element = Element(name="E")
    e_guid = element.guid
    e_node = session.add_element(element)
    attributes = TreeNode("attributes")
    session.add(attributes, e_node)
    q = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)])
    q_guid = q.guid
    q_node = session.add_polyline(q, attributes)
    session.set_xform(e_guid, Xform.translation(1.0, 0.0, 0.0))
    session.set_xform(q_guid, Xform.translation(0.0, 1.0, 0.0))
    composed = session.world_xform(q_guid)

    session.begin("remove")
    session.remove_object(e_guid)
    session.commit()
    names = [n.name for n in session.tree.nodes]

    MINI_CHECK(q_guid in session.lookup)
    MINI_CHECK(session.get_node(q_guid) is q_node)
    MINI_CHECK(q_node.parent is attributes)
    MINI_CHECK(attributes.parent is e_node)
    MINI_CHECK(e_node.parent is None)
    MINI_CHECK(session.world_xform(q_guid) == session.xform(q_guid))
    MINI_CHECK(q_guid in session.world_xforms())
    MINI_CHECK(names == ["my_session"])

    session.undo()

    MINI_CHECK(session.world_xform(q_guid) == composed)
    MINI_CHECK(len(session.tree.nodes) == 4)


@MINI_TEST("Session", "Copy Drops Dead")
def test_session_copy_drops_dead():
    from session_py import Point
    from session_py import Session
    from copy import deepcopy

    session = Session()
    g = session.add_group("g")
    a = Point(1.0, 0.0, 0.0)
    b = Point(2.0, 0.0, 0.0)
    b_guid = b.guid
    session.add_point(a, g)
    session.add_point(b, g)

    session.begin("remove")
    session.remove_object(b_guid)
    session.commit()
    copy = deepcopy(session)

    MINI_CHECK(copy.objects.points.number_of_slots() == len(copy.objects.points))
    MINI_CHECK(copy.objects.points.number_of_dead() == 0)
    MINI_CHECK(len(copy.objects.points) == 1)
    MINI_CHECK(copy.history.depth() == 0)
    MINI_CHECK(copy.order() == session.order())
    MINI_CHECK(str(copy.tree) == str(session.tree))
    MINI_CHECK(copy.graph.number_of_vertices() == 1)
    MINI_CHECK(session.undo())
    MINI_CHECK(len(session.objects.points) == 2)
    MINI_CHECK(len(copy.objects.points) == 1)


@MINI_TEST("Session", "Live Views")
def test_session_live_views():
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Session
    from session_py import Vector
    from session_py import Xform
    import json

    session = Session()
    g = session.add_group("gone_group")
    kept = session.add_group("kept")
    definition = session.add_definition(create_box(Point(0.0, 0.0, 0.0), 2.0))
    point = Point(0.0, 0.0, 0.0)
    point_guid = point.guid
    mesh = create_box(Point(5.0, 0.0, 0.0), 2.0)
    mesh_guid = mesh.guid
    instance = InstanceRef(definition, Xform.identity())
    instance_guid = instance.guid
    session.add_point(point, g)
    session.add_mesh(mesh, g)
    session.add_instance(instance, Xform.translation(10.0, 0.0, 0.0), g)
    session.add_point(Point(20.0, 0.0, 0.0), kept)
    session.set_xform(point_guid, Xform.translation(0.0, 1.0, 0.0))
    session.set_xform(mesh_guid, Xform.translation(0.0, 1.0, 0.0))
    session.set_xform("gone_group", Xform.translation(0.0, 0.0, 1.0))

    session.begin("remove")
    session.remove_object(point_guid)
    session.remove_object(mesh_guid)
    session.remove_object(instance_guid)
    session.remove_group(g)
    session.commit()
    gone = [point_guid, mesh_guid, instance_guid, "gone_group"]
    world = session.world_xforms()
    groups = [n.name for n in session.tree.root.children]
    geometry = session.get_geometry()
    collisions = session.get_collisions()
    hits = session.ray_cast(Point(5.0, 0.0, -10.0), Vector(0.0, 0.0, 1.0), 0.01)
    text_json = json.dumps(session.__jsondump__())
    text = str(session) + repr(session)

    MINI_CHECK(all(guid not in gone for guid in session.order()))
    MINI_CHECK(all(name not in world for name in gone))
    MINI_CHECK(len(session.select_by_type(Point)) == 1)
    MINI_CHECK(groups == ["kept"])
    MINI_CHECK(len(geometry.points) == 1 and len(geometry.meshes) == 0)
    MINI_CHECK(len(session.instances_of(definition)) == 0)
    MINI_CHECK(len(collisions) == 0)
    MINI_CHECK(len(hits) == 0)
    MINI_CHECK(all(name not in text_json for name in gone))
    MINI_CHECK(all(name not in text for name in gone))
    MINI_CHECK(session.undo())
    MINI_CHECK(len(session.order()) == 3)


@MINI_TEST("Session", "Unrecorded Remove")
def test_session_unrecorded_remove():
    from session_py import Point
    from session_py import Session

    session = Session()
    a = Point(1.0, 0.0, 0.0)
    a_guid = a.guid
    session.add_point(a)
    session.add_point(Point(2.0, 0.0, 0.0))
    removed = session.remove_object(a_guid)

    MINI_CHECK(removed)
    MINI_CHECK(a_guid not in session.lookup)
    MINI_CHECK(len(session.order()) == 1)
    MINI_CHECK(len(session.tree.nodes) == 2)
    MINI_CHECK(session.objects.points.number_of_dead() == 1)
    MINI_CHECK(session.objects.points.get_tomb(0) is None)
    MINI_CHECK(session.history.dropped == 1)
    MINI_CHECK(not session.undo())


@MINI_TEST("Session", "Remove Twin Keeps Slot")
def test_session_remove_twin_keeps_slot():
    from session_py import Point
    from session_py import Session

    session = Session()
    x = Point(1.0, 0.0, 0.0)
    guid = x.guid
    y = Point(2.0, 0.0, 0.0)
    y.guid = guid
    session.begin("add")
    session.add_point(x)
    session.commit()
    session.add_point(y)
    session.undo()

    MINI_CHECK(session.objects.points.get_slot(guid) == 1)
    MINI_CHECK(session.objects.points.is_dead(0))
    MINI_CHECK(len(session.objects.points) == 1)
    MINI_CHECK(guid in session.lookup)

    removed = session.remove_object(guid)
    data = session.pb_dumps()
    loaded = Session.pb_loads(data)

    MINI_CHECK(removed)
    MINI_CHECK(guid not in session.lookup)
    MINI_CHECK(len(session.objects.points) == 0)
    MINI_CHECK(len(loaded.objects.points) == 0)
    MINI_CHECK(len(loaded.lookup) == 0)


@MINI_TEST("Session", "Redo Twin Keeps Slot")
def test_session_redo_twin_keeps_slot():
    from session_py import Point
    from session_py import Session

    session = Session()
    x = Point(1.0, 0.0, 0.0)
    guid = x.guid
    y = Point(2.0, 0.0, 0.0)
    y.guid = guid
    session.begin("add")
    session.add_point(x)
    session.commit()
    session.add_point(y)
    session.undo()
    session.redo()
    held = 0.0

    if isinstance(session.lookup.get(guid), Point):
        held = session.lookup[guid][0]

    MINI_CHECK(len(session.objects.points) == 1)
    MINI_CHECK(session.objects.points.get_slot(guid) == 1)
    MINI_CHECK(session.objects.points.is_dead(0))
    MINI_CHECK(held == 2.0)
    MINI_CHECK(session.graph.has_node(guid))
    MINI_CHECK(
        session.get_node(guid) is not None and not session.get_node(guid).is_dead()
    )

    session.undo()
    session.redo()

    MINI_CHECK(len(session.objects.points) == 1)
    MINI_CHECK(session.objects.points.get_slot(guid) == 1)

    removed = session.remove_object(guid)
    undone = session.undo()
    data = session.pb_dumps()
    loaded = Session.pb_loads(data)

    MINI_CHECK(removed)
    MINI_CHECK(undone)
    MINI_CHECK(guid not in session.lookup)
    MINI_CHECK(not session.graph.has_node(guid))
    MINI_CHECK(len(session.objects.points) == 0)
    MINI_CHECK(len(loaded.objects.points) == 0)
    MINI_CHECK(len(loaded.lookup) == 0)
    MINI_CHECK(len(loaded.tree.nodes) == 1)


@MINI_TEST("Session", "Purge Clears History")
def test_session_purge_clears_history():
    from session_py import Point
    from session_py import Session

    session = Session()
    a = Point(0.0, 0.0, 0.0)
    b = Point(1.0, 0.0, 0.0)
    c = Point(2.0, 0.0, 0.0)
    a_guid = a.guid
    b_guid = b.guid
    c_guid = c.guid
    session.add_point(a)
    session.add_point(b)
    session.add_point(c)
    session.begin("remove")
    session.remove_object(b_guid)
    session.commit()
    session.purge()
    undone = session.undo()
    indices = []

    for vertex in session.graph.get_vertices():
        indices.append(vertex.index)

    indices.sort()

    MINI_CHECK(not undone)
    MINI_CHECK(session.history.depth() == 0)
    MINI_CHECK(session.order() == [a_guid, c_guid])
    MINI_CHECK(indices == [0, 1])
    MINI_CHECK(session.objects.points.number_of_slots() == 2)
    MINI_CHECK(session.number_of_dead() == 0)
    MINI_CHECK(len(session.tree.nodes) == 3)


@MINI_TEST("Session", "Checkpoint After Purge Steps")
def test_session_checkpoint_after_purge_steps():
    from session_py import Point
    from session_py import Session
    from session_py import Xform
    from session_py.history import CAPACITY
    from session_py.session import PURGE_WORK

    n = 40_000
    bulk = 20_000
    session = Session()
    group = session.add_group("flat")
    guids = []

    for i in range(n):
        node = session.add_point(Point(float(i), 0.0, 0.0), group)
        guids.append(node.name)

    session.begin("remove")

    for guid in guids[:bulk]:
        session.remove_object(guid)

    session.commit()

    for step in range(CAPACITY):
        session.begin("move")
        session.set_xform(guids[n - 1], Xform.translation(float(step), 0.0, 0.0))
        session.commit()

    steps = 0

    while session.purge_step(PURGE_WORK):
        steps += 1

    data = None

    while data is None:
        data = session.checkpoint(PURGE_WORK)

    loaded = Session.pb_loads(data)

    MINI_CHECK(steps > 1)
    MINI_CHECK(len(loaded.objects.points) == n - bulk)
    MINI_CHECK(data == session.to_proto().SerializeToString(deterministic=True))
    MINI_CHECK(session.number_of_dead() == 0)
    MINI_CHECK(session.history.depth() == CAPACITY)


@MINI_TEST("Session", "Steady State Bounds")
def test_session_steady_state_bounds():
    from session_py import Point
    from session_py import Session
    from session_py.history import CAPACITY
    from session_py.session import PURGE_WORK

    n = 2_000
    cycles = 1_000
    session = Session()
    group = session.add_group("flat")
    guids = []

    for i in range(n):
        node = session.add_point(Point(float(i), 0.0, 0.0), group)
        guids.append(node.name)

    for cycle, guid in enumerate(guids[:cycles]):
        session.begin("remove")
        session.remove_object(guid)
        session.commit()
        session.undo()
        session.redo()
        session.begin("add")
        session.add_point(Point(float(cycle), 1.0, 0.0), group)
        session.commit()
        session.purge_step(PURGE_WORK)

    bound = 2 * CAPACITY + 2 * (n // PURGE_WORK + 1)
    points = session.objects.points

    MINI_CHECK(len(points) == n)
    MINI_CHECK(session.history.bytes <= session.history.budget)
    MINI_CHECK(session.number_of_dead() <= bound)
    MINI_CHECK(points.number_of_slots() <= len(points) + bound)
    MINI_CHECK(len(group.children) <= len(points) + bound)


@MINI_TEST("Session", "History Budget Bounds")
def test_session_history_budget_bounds():
    from session_py import Mesh
    from session_py import Point
    from session_py import Session
    from session_py.history import CAPACITY

    side = 30
    vertices = []
    faces = []

    for at in range(side * side):
        vertices.append(Point(float(at // side), float(at % side), 0.0))

    for cell in range((side - 1) * (side - 1)):
        at = cell // (side - 1) * side + cell % (side - 1)
        faces.append([at, at + side, at + side + 1, at + 1])

    session = Session()
    session.history.budget = 1 << 20
    guids = []

    for _ in range(200):
        mesh = Mesh.from_vertices_and_faces(vertices, faces)
        guids.append(mesh.guid)
        session.add_mesh(mesh)

    bounded = True

    for guid in guids:
        session.begin("remove")
        session.remove_object(guid)
        session.commit()
        newest = session.history.undo_stack[session.history.depth() - 1].bytes
        bounded &= session.history.bytes <= session.history.budget + newest

    MINI_CHECK(bounded)
    MINI_CHECK(session.history.depth() < CAPACITY)
    MINI_CHECK(session.history.depth() > 1)
    MINI_CHECK(len(session.objects.meshes) == 0)


@MINI_TEST("Session", "Purge Keeps Replaced Tomb")
def test_session_purge_keeps_replaced_tomb():
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Session
    from session_py.session import PURGE_WORK

    session = Session()
    definition = session.add_definition(Point(1.0, 2.0, 3.0))
    first = InstanceRef(definition)
    second = InstanceRef(definition)
    session.add_instance(first)
    session.begin("add")
    session.add_instance(second)
    session.commit()
    session.begin("explode")
    session.explode(second.guid)
    session.commit()
    session.remove_object(first.guid)

    while session.purge_step(PURGE_WORK):
        pass

    slots = session.objects.instances.number_of_slots()
    unexploded = session.undo()
    instances = len(session.objects.instances)
    unadded = session.undo()

    MINI_CHECK(slots == 1)
    MINI_CHECK(unexploded)
    MINI_CHECK(instances == 1)
    MINI_CHECK(unadded)
    MINI_CHECK(len(session.objects.instances) == 0)
    MINI_CHECK(len(session.objects.points) == 0)
    MINI_CHECK(second.guid not in session.instance_lookup)
    MINI_CHECK(session.redo())
    MINI_CHECK(session.redo())
    MINI_CHECK(len(session.objects.points) == 1)


@MINI_TEST("Session", "Checkpoint Twin Xform")
def test_session_checkpoint_twin_xform():
    from session_py import InstanceRef
    from session_py import Point
    from session_py import Session
    from session_py import Xform

    session = Session()
    definition = session.add_definition(Point(1.0, 2.0, 3.0))
    x = Point(0.0, 0.0, 0.0)
    y = Point(1.0, 0.0, 0.0)
    y.guid = x.guid
    instance = InstanceRef(definition)
    session.add_point(x)
    session.add_point(y)
    session.add_instance(instance, Xform.translation(0.0, 1.0, 0.0))
    session.set_xform(x.guid, Xform.translation(1.0, 0.0, 0.0))
    data = session.checkpoint(1)

    while data is None:
        data = session.checkpoint(1)

    loaded = Session.pb_loads(data)

    MINI_CHECK(data == session.to_proto().SerializeToString(deterministic=True))
    MINI_CHECK(loaded.xform(x.guid) == session.xform(x.guid))
    MINI_CHECK(loaded.xform(instance.guid) == session.xform(instance.guid))


if __name__ == "__main__":
    run_all(language="python")
