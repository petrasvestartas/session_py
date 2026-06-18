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
    line = Line(Point(0,0,0), Point(1,0,0))
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
        half_size=Vector(1.0, 1.0, 1.0)
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
    pl = Polyline([Point(0,0,0), Point(1,0,0), Point(1,1,0)])
    session.add_polyline(pl)

    MINI_CHECK(len(session.objects.polylines) == 1)
    MINI_CHECK(pl.guid in session.lookup)


@MINI_TEST("Session", "Add Pointcloud")
def test_session_add_pointcloud():
    from session_py import Session
    from session_py import PointCloud
    from session_py import Point

    session = Session()
    pc = PointCloud([Point(0,0,0), Point(1,0,0)])
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
    mesh.add_vertex(Point(0,0,0), 0)
    mesh.add_vertex(Point(1,0,0), 1)
    mesh.add_vertex(Point(0,1,0), 2)
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
    pts = [Point(0,0,0), Point(1,1,0), Point(2,0,0), Point(3,1,0)]
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
        Point(0,0,0), Point(0,1,0), Point(0,2,0), Point(0,3,0),
        Point(1,0,0), Point(1,1,0), Point(1,2,0), Point(1,3,0),
        Point(2,0,0), Point(2,1,0), Point(2,2,0), Point(2,3,0),
        Point(3,0,0), Point(3,1,0), Point(3,2,0), Point(3,3,0),
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
    from session_py import ElementPlate
    from session_py import Point

    session = Session()
    polygon = [Point(0,0,0), Point(2,0,0), Point(2,2,0), Point(0,2,0)]
    plate = ElementPlate(polygon=polygon, thickness=0.2, name="p1")
    session.add_element(plate)

    MINI_CHECK(len(session.objects.elements) == 1)
    MINI_CHECK(plate.guid in session.lookup)
    MINI_CHECK(session.graph.has_node(plate.guid))


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
    p1 = Point(0,0,0)
    p2 = Point(1,0,0)
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
    p1 = Point(0,0,0)
    p2 = Point(1,0,0)
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
    p1 = Point(0,0,0)
    p2 = Point(1,0,0)
    session.add_point(p1)
    session.add_point(p2)
    session.add_relationship(p1.guid, p2.guid, "connects_to")

    MINI_CHECK(session.graph.has_edge((p1.guid, p2.guid)))


@MINI_TEST("Session", "Get Neighbours")
def test_session_get_neighbours():
    from session_py import Session
    from session_py import Point

    session = Session()
    p1 = Point(0,0,0)
    p2 = Point(1,0,0)
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
        half_size=Vector(2.0, 2.0, 2.0)
    )
    obb2 = OBB(
        center=Point(1.0, 0.0, 0.0),
        x_axis=Vector(1.0, 0.0, 0.0),
        y_axis=Vector(0.0, 1.0, 0.0),
        z_axis=Vector(0.0, 0.0, 1.0),
        half_size=Vector(2.0, 2.0, 2.0)
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

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    session.add_point(point)
    removed = session.remove_object(point.guid)

    MINI_CHECK(removed)
    MINI_CHECK(point.guid not in session.lookup)


@MINI_TEST("Session", "Get Geometry")
def test_session_get_geometry():
    from session_py import Session
    from session_py import Point

    session = Session()
    point = Point(1.0, 2.0, 3.0)
    session.add_point(point)

    geom = session.get_geometry()

    MINI_CHECK(len(geom.points) == 1)


@MINI_TEST("Session", "Compute Face To Face")
def test_session_compute_face_to_face():
    from session_py import Session
    from session_py import ElementPlate
    from session_py import Point

    session = Session()
    p1 = ElementPlate(polygon=[Point(0,0,0), Point(1,0,0), Point(1,1,0), Point(0,1,0)], thickness=0.2, name="p1")
    p2 = ElementPlate(polygon=[Point(0,0,-0.2), Point(1,0,-0.2), Point(1,1,-0.2), Point(0,1,-0.2)], thickness=0.2, name="p2")
    session.add_element(p1)
    session.add_element(p2)
    session.compute_face_to_face(5.0, 0.001)

    MINI_CHECK(len(session.objects.elements) == 2)
    MINI_CHECK(session.graph.has_edge((p1.guid, p2.guid)))


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

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   file_json_dumps()    │ str          │ to JSON string
    #   file_json_loads(s)   │ str          │ from JSON string
    #   file_json_dump(path) │ file         │ write to file
    #   file_json_load(path) │ file         │ read from file

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


@MINI_TEST("Session", "Tree Transformation Hierarchy")
def test_session_tree_transformation_hierarchy():
    from session_py import Session
    from session_py import Point
    from session_py import Vector
    from session_py import Mesh
    from session_py import Xform
    from session_py import Plane

    scene = Session("tree_transformation_test")

    def create_box(center, size):
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
        for i, v in enumerate(verts):
            mesh.add_vertex(v, i)
        faces = [
            [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
            [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2],
        ]
        for f in faces:
            mesh.add_face(f)
        return mesh

    box1 = create_box(Point(0, 0, 0), 2.0)
    box1_node = scene.add_mesh(box1)
    box2 = create_box(Point(0, 0, 0), 2.0)
    box2_node = scene.add_mesh(box2)
    box3 = create_box(Point(0, 0, 0), 2.0)
    box3_node = scene.add_mesh(box3)

    scene.add(box1_node)
    scene.add(box2_node, box1_node)
    scene.add(box3_node, box2_node)

    plane_from = Plane(Point(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
    plane_to = Plane(Point(0, 0, 1.0), Vector(1, 0, 0), Vector(0, 1, 0))
    xy_to_top = Xform.plane_to_plane(plane_from, plane_to)
    box1.xform = Xform.rotation_z(PI / 1.5) * xy_to_top
    box2.xform = Xform.translation(2.0, 0, 0) * Xform.rotation_z(PI / 6.0)
    box3.xform = Xform.translation(2.0, 0, 0)

    transformed = scene.get_geometry()

    MINI_CHECK(len(transformed.meshes) == 3)
    m1 = transformed.meshes[0]
    v0 = m1.vertex[0]
    MINI_CHECK(abs(v0[0] - 1.36603) < 1e-4)
    MINI_CHECK(abs(v0[1] - (-0.366025)) < 1e-4)
    MINI_CHECK(abs(v0[2] - 0.0) < 1e-4)


@MINI_TEST("Session", "Add Component")
def test_session_add_component():
    import uuid
    from session_py import Session
    from session_py.file_encoders import file_register_class

    class Box:
        def __init__(self, width=1.0, height=2.0):
            self._guid = str(uuid.uuid4())
            self.name = "my_box"
            self.width = width
            self.height = height
        @property
        def guid(self): return self._guid
        def __jsondump__(self):
            return {"type": "Box", "guid": self.guid, "name": self.name,
                    "width": self.width, "height": self.height}
        @classmethod
        def __jsonload__(cls, data, guid=None, name=None):
            obj = cls(data["width"], data["height"])
            obj._guid = guid or data.get("guid", obj._guid)
            obj.name  = name or data.get("name", obj.name)
            return obj

    file_register_class("Box", Box)

    session = Session()
    box = Box(width=3.0, height=5.0)
    guid = box.guid

    session.add_component(box)

    MINI_CHECK(len(session.objects.components) == 1)
    MINI_CHECK(session.lookup[guid] is box)
    MINI_CHECK(session.graph.has_node(guid))


@MINI_TEST("Session", "Component Json Roundtrip")
def test_session_component_json_roundtrip():
    import uuid
    from session_py import Session
    from session_py.file_encoders import file_register_class
    from pathlib import Path

    class Box:
        def __init__(self, width=1.0, height=2.0):
            self._guid = str(uuid.uuid4())
            self.name = "my_box"
            self.width = width
            self.height = height
        @property
        def guid(self): return self._guid
        def __jsondump__(self):
            return {"type": "Box", "guid": self.guid, "name": self.name,
                    "width": self.width, "height": self.height}
        @classmethod
        def __jsonload__(cls, data, guid=None, name=None):
            obj = cls(data["width"], data["height"])
            obj._guid = guid or data.get("guid", obj._guid)
            obj.name  = name or data.get("name", obj.name)
            return obj

    file_register_class("Box", Box)

    original = Session()
    box = Box(width=3.0, height=5.0)
    guid = box.guid
    original.add_component(box)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_session_component.json"
    original.file_json_dump(fname)
    loaded = Session.file_json_load(fname)

    MINI_CHECK(len(loaded.objects.components) == 1)
    MINI_CHECK(loaded.objects.components[0].width == 3.0)
    MINI_CHECK(loaded.objects.components[0].guid == guid)


if __name__ == "__main__":
    run_all(language="python")
