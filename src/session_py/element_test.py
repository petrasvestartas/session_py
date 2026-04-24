from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Element", "Constructor")
def test_element_constructor():
    from session_py import Mesh
    from session_py import BRep
    from session_py import Element

    # Constructor
    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m, name="test_element")

    # Getters
    geo = e.geometry
    name = e.name
    guid = e.guid
    dirty = e.is_dirty

    # String
    estr = str(e)
    erepr = repr(e)

    # Copy
    ecopy = e.duplicate()

    # Equality
    e2 = Element(geometry=Mesh(), name="test_element")
    e3 = Element(geometry=BRep(), name="other")

    MINI_CHECK(name == "test_element")
    MINI_CHECK(guid is not None and len(guid) > 0)
    MINI_CHECK(dirty)
    MINI_CHECK(isinstance(geo, Mesh))
    MINI_CHECK(estr == "Element(test_element, Mesh)")
    MINI_CHECK(erepr == f"Element({guid}, test_element, Mesh)")
    MINI_CHECK(ecopy == e and ecopy.guid != e.guid)
    MINI_CHECK(e == e2)
    MINI_CHECK(e != e3)


@MINI_TEST("Element", "Session Transformation")
def test_session_transformation():
    from session_py import Mesh
    from session_py import Xform
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m)
    xf = Xform.translation(10.0, 20.0, 30.0)
    e.session_transformation = xf

    MINI_CHECK(e.is_dirty)
    MINI_CHECK(e.session_transformation == xf)


@MINI_TEST("Element", "Add Feature")
def test_add_feature():
    from session_py import Mesh
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m)

    def my_feature(geo):
        return geo

    e.add_feature(my_feature)

    MINI_CHECK(e.is_dirty)
    MINI_CHECK(len(e._features) == 1)


@MINI_TEST("Element", "AABB")
def test_aabb():
    from session_py import Mesh
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m)
    aabb = e.aabb

    MINI_CHECK(aabb is not None)
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[0], 0.5))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[1], 0.5))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[2], 0.0))


@MINI_TEST("Element", "OBB")
def test_obb():
    from session_py import Mesh
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m)
    obb = e.obb

    MINI_CHECK(obb is not None)
    MINI_CHECK(TOLERANCE.is_close(obb.half_size[0], 0.5))
    MINI_CHECK(TOLERANCE.is_close(obb.half_size[1], 0.5))


@MINI_TEST("Element", "Session Geometry")
def test_session_geometry():
    from session_py import Mesh
    from session_py import Xform
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m)
    e.session_transformation = Xform.translation(10.0, 0.0, 0.0)
    sg = e.session_geometry

    MINI_CHECK(isinstance(sg, Mesh))
    verts = list(sg.vertex.values())
    MINI_CHECK(TOLERANCE.is_close(verts[0].x, 10.0))
    MINI_CHECK(TOLERANCE.is_close(verts[1].x, 11.0))
    MINI_CHECK(e.geometry is not sg)


@MINI_TEST("Element", "Reset")
def test_reset():
    from session_py import Mesh
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m)
    _ = e.aabb
    _ = e.point
    e.reset()

    MINI_CHECK(e.is_dirty)
    MINI_CHECK(e._aabb is None)
    MINI_CHECK(e._obb is None)
    MINI_CHECK(e._collision_mesh is None)
    MINI_CHECK(e._point is None)


@MINI_TEST("Element", "Compute Point")
def test_compute_point():
    from session_py import Mesh
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m)
    pt = e.point

    MINI_CHECK(TOLERANCE.is_close(pt[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 0.0))


@MINI_TEST("Element", "Brep Aabb")
def test_brep_aabb():
    from session_py import BRep
    from session_py import Element

    b = BRep.create_box(2.0, 3.0, 4.0)
    e = Element(geometry=b, name="brep_element")
    aabb = e.aabb
    pt = e.point

    MINI_CHECK(aabb is not None)
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[1], 1.5))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[2], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pt[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 0.0))


@MINI_TEST("Element", "Json Roundtrip")
def test_json_roundtrip():
    from session_py import Mesh
    from session_py import Xform
    from session_py import Element
    from pathlib import Path

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m, name="json_test")
    e.session_transformation = Xform.translation(1.0, 2.0, 3.0)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_element.json"
    e.file_json_dump(fname)
    loaded = Element.file_json_load(fname)

    MINI_CHECK(isinstance(loaded, Element))
    MINI_CHECK(loaded.name == "json_test")
    MINI_CHECK(isinstance(loaded.geometry, Mesh))
    MINI_CHECK(len(loaded.geometry.vertex) == 4)


@MINI_TEST("Element", "Protobuf Roundtrip")
def test_protobuf_roundtrip():
    from session_py import BRep
    from session_py import Xform
    from session_py import Element
    from pathlib import Path

    b = BRep.create_box(2.0, 3.0, 4.0)
    e = Element(geometry=b, name="proto_test")
    e.session_transformation = Xform.translation(1.0, 2.0, 3.0)

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_element.bin"
    e.pb_dump(path)
    loaded = Element.pb_load(path)

    MINI_CHECK(isinstance(loaded, Element))
    MINI_CHECK(loaded.name == "proto_test")
    MINI_CHECK(isinstance(loaded.geometry, BRep))
    MINI_CHECK(loaded.geometry.face_count() == 6)
    MINI_CHECK(loaded.geometry.vertex_count() == 8)


@MINI_TEST("Element", "Polylines")
def test_polylines():
    from session_py import Mesh
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m)

    MINI_CHECK(len(e.polylines) == 0)
    MINI_CHECK(len(e.planes) == 0)
    MINI_CHECK(len(e.edge_vectors) == 0)
    MINI_CHECK(e.axis is None)


if __name__ == "__main__":
    run_all(language="python")
