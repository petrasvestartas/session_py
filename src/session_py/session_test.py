from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Session", "Constructor")
def test_session_constructor():
    from session_py import Session
    session = Session()
    MINI_CHECK(session.name == "my_session")
    MINI_CHECK(bool(session.guid))


@MINI_TEST("Session", "Jsondump")
def test_session_jsondump():
    from session_py import Session
    from session_py import Point
    from session_py.encoders import json_dump
    from pathlib import Path

    session = Session()
    point1 = Point(1.0, 2.0, 3.0)
    point2 = Point(4.0, 5.0, 6.0)
    session.add_point(point1)
    session.add_point(point2)
    session.add_edge(point1.guid, point2.guid, "connection")
    data = session.__jsondump__()
    MINI_CHECK(data["name"] == "my_session")
    MINI_CHECK("guid" in data)
    MINI_CHECK(len(data["objects"]["points"]) == 2)
    MINI_CHECK(len(data["graph"]["vertices"]) == 2)
    MINI_CHECK(len(data["graph"]["edges"]) == 1)
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_session.json"
    json_dump(session, fname)


@MINI_TEST("Session", "Jsonload")
def test_session_jsonload():
    from session_py import Session
    from session_py import Point

    session = Session()
    point1 = Point(1.0, 2.0, 3.0)
    point2 = Point(4.0, 5.0, 6.0)
    session.add_point(point1)
    session.add_point(point2)
    session.add_edge(point1.guid, point2.guid, "connection")
    data = session.__jsondump__()
    session2 = Session.__jsonload__(data)
    MINI_CHECK(session2.name == "my_session")
    MINI_CHECK(len(session2.lookup) == 2)
    MINI_CHECK(session2.graph.number_of_vertices() == 2)


@MINI_TEST("Session", "File Io")
def test_session_file_io():
    from session_py import Session
    from session_py import Point
    from session_py.encoders import json_dump
    from session_py.encoders import json_load
    from pathlib import Path
    import os

    session = Session()
    point1 = Point(1.0, 2.0, 3.0)
    point2 = Point(4.0, 5.0, 6.0)
    session.add_point(point1)
    session.add_point(point2)
    session.add_edge(point1.guid, point2.guid, "connection")
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_session_roundtrip.json"
    json_dump(session, fname)
    loaded_session = json_load(fname)
    MINI_CHECK(loaded_session.name == session.name)
    MINI_CHECK(len(loaded_session.lookup) == len(session.lookup))
    MINI_CHECK(loaded_session.graph.number_of_vertices() == session.graph.number_of_vertices())
    os.remove(fname)


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


@MINI_TEST("Session", "Add Edge")
def test_session_add_edge():
    from session_py import Session
    from session_py import Point

    session = Session()
    point1 = Point(1.0, 2.0, 3.0)
    point2 = Point(4.0, 5.0, 6.0)
    session.add_point(point1)
    session.add_point(point2)
    session.add_edge(point1.guid, point2.guid, "connection")
    MINI_CHECK(session.graph.has_edge((point1.guid, point2.guid)))


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


@MINI_TEST("Session", "File Io Comprehensive")
def test_session_file_io_comprehensive():
    from session_py import Session
    from session_py import Point
    from session_py.encoders import json_dump
    from session_py.encoders import json_load
    from pathlib import Path
    import os

    session = Session("./serialization/test_session")
    point1 = Point(1.0, 2.0, 3.0)
    point2 = Point(4.0, 5.0, 6.0)
    session.add_point(point1)
    session.add_point(point2)
    session.add_edge(point1.guid, point2.guid, "./serialization/test_connection")
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_session_comprehensive.json"
    json_dump(session, fname)
    loaded_session = json_load(fname)
    MINI_CHECK(loaded_session.name == session.name)
    MINI_CHECK(len(loaded_session.objects.points) == len(session.objects.points))
    MINI_CHECK(loaded_session.graph.number_of_vertices() == session.graph.number_of_vertices())
    MINI_CHECK(loaded_session.graph.number_of_edges() == session.graph.number_of_edges())
    os.remove(fname)


@MINI_TEST("Session", "Tree Transformation Hierarchy")
def test_session_tree_transformation_hierarchy():
    from session_py import Session
    from session_py import Point
    from session_py import Vector
    from session_py import Mesh
    from session_py import Xform
    from session_py import Plane
    import math

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

    box1 = create_box(Point(0, 0, 0), 2.0)
    box1_node = scene.add_mesh(box1)
    box2 = create_box(Point(0, 0, 0), 2.0)
    box2_node = scene.add_mesh(box2)
    box3 = create_box(Point(0, 0, 0), 2.0)
    box3_node = scene.add_mesh(box3)

    scene.add(box1_node)
    scene.add(box2_node, box1_node)
    scene.add(box3_node, box2_node)

    box1_top = Point(0, 0, 1.0)
    normal = Vector(0, 0, 1)
    x = Vector(1, 0, 0)
    y = Vector(0, 1, 0)
    plane_from = Plane(Point(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
    plane_to = Plane(box1_top, x, y)
    xy_to_top = Xform.plane_to_plane(plane_from, plane_to)
    box1.xform = Xform.rotation_z(math.pi / 1.5) * xy_to_top
    box2.xform = Xform.translation(2.0, 0, 0) * Xform.rotation_z(math.pi / 6.0)
    box3.xform = Xform.translation(2.0, 0, 0)

    transformed = scene.get_geometry()
    MINI_CHECK(len(transformed.meshes) == 3)

    expected_box1 = [
        [1.36603, -0.366025, 0], [0.366025, 1.36603, 0],
        [-1.36603, 0.366025, 0], [-0.366025, -1.36603, 0],
        [1.36603, -0.366025, 2], [0.366025, 1.36603, 2],
        [-1.36603, 0.366025, 2], [-0.366025, -1.36603, 2],
    ]
    expected_box2 = [
        [0.366025, 2.09808, 0], [-1.36603, 3.09808, 0],
        [-2.36603, 1.36603, 0], [-0.633975, 0.366025, 0],
        [0.366025, 2.09808, 2], [-1.36603, 3.09808, 2],
        [-2.36603, 1.36603, 2], [-0.633975, 0.366025, 2],
    ]
    expected_box3 = [
        [-1.36603, 3.09808, 0], [-3.09808, 4.09808, 0],
        [-4.09808, 2.36603, 0], [-2.36603, 1.36603, 0],
        [-1.36603, 3.09808, 2], [-3.09808, 4.09808, 2],
        [-4.09808, 2.36603, 2], [-2.36603, 1.36603, 2],
    ]
    expected_faces = [
        [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
        [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2],
    ]

    m1 = transformed.meshes[0]
    for i in range(8):
        v = m1.vertex[i]
        MINI_CHECK(abs(v[0] - expected_box1[i][0]) < 1e-4)
        MINI_CHECK(abs(v[1] - expected_box1[i][1]) < 1e-4)
        MINI_CHECK(abs(v[2] - expected_box1[i][2]) < 1e-4)

    m2 = transformed.meshes[1]
    for i in range(8):
        v = m2.vertex[i]
        MINI_CHECK(abs(v[0] - expected_box2[i][0]) < 1e-4)
        MINI_CHECK(abs(v[1] - expected_box2[i][1]) < 1e-4)
        MINI_CHECK(abs(v[2] - expected_box2[i][2]) < 1e-4)

    m3 = transformed.meshes[2]
    for i in range(8):
        v = m3.vertex[i]
        MINI_CHECK(abs(v[0] - expected_box3[i][0]) < 1e-4)
        MINI_CHECK(abs(v[1] - expected_box3[i][1]) < 1e-4)
        MINI_CHECK(abs(v[2] - expected_box3[i][2]) < 1e-4)

    for mesh in [m1, m2, m3]:
        MINI_CHECK(len(mesh.face) == 6)
        face_idx = 0
        for key, face in mesh.face.items():
            MINI_CHECK(len(face) == len(expected_faces[face_idx]))
            for i in range(len(face)):
                MINI_CHECK(face[i] == expected_faces[face_idx][i])
            face_idx += 1


if __name__ == "__main__":
    run_all(language="python")
