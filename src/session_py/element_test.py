from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


# ═══════════════════════════════════════════════════════════════════════════
# Element
# ═══════════════════════════════════════════════════════════════════════════


@MINI_TEST("Element", "Constructor")
def test_element_constructor():
    from session_py import Mesh
    from session_py import BRep
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(m, "test_element")

    geo = e.geometry
    name = e.name
    guid = e.guid
    dirty = e.is_dirty

    estr = str(e)
    erepr = repr(e)

    ecopy = e.duplicate()

    e2 = Element(Mesh(), "test_element")
    e3 = Element(BRep(), "other")

    MINI_CHECK(name == "test_element")
    MINI_CHECK(len(guid) > 0)
    MINI_CHECK(dirty)
    MINI_CHECK(isinstance(geo, Mesh))
    MINI_CHECK(estr == "Element(test_element, Mesh)")
    MINI_CHECK(erepr == f"Element({guid}, test_element, Mesh)")
    MINI_CHECK(ecopy == e and ecopy.guid != e.guid)
    MINI_CHECK(e == e2)
    MINI_CHECK(e != e3)


@MINI_TEST("Element", "Place")
def test_place():
    from session_py import Mesh
    from session_py import Xform
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(m)
    xf = Xform.translation(10.0, 20.0, 30.0)
    e.place(xf)

    MINI_CHECK(e.is_dirty)
    min_x = float("inf")
    for v in e.geometry.vertex.values():
        min_x = min(min_x, v.x)
    MINI_CHECK(min_x > 9.0)


@MINI_TEST("Element", "Add Geometry Op")
def test_add_geometry_op():
    from session_py import Mesh
    from session_py import BRep
    from session_py import Xform
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(m)

    def my_feature(geo):
        return geo

    e.add_geometry_op(my_feature)

    eb = Element(BRep.create_box(1.0, 1.0, 1.0), "brep_feature")
    eb.add_geometry_op(lambda geo: Mesh())
    sg = eb.session_geometry(Xform.identity())

    MINI_CHECK(e.is_dirty)
    MINI_CHECK(e.geometry_ops_count == 1)
    MINI_CHECK(isinstance(sg, BRep))


@MINI_TEST("Element", "AABB")
def test_aabb():
    from session_py import Mesh
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(m)
    aabb = e.aabb

    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[0], 0.5))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[1], 0.5))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[2], 0.0))
    MINI_CHECK(not e.is_dirty)

    e.add_geometry_op(lambda geo: geo)
    MINI_CHECK(e.is_dirty)
    MINI_CHECK(e.cached_aabb is None)


@MINI_TEST("Element", "OBB")
def test_obb():
    from session_py import Mesh
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(m)
    obb = e.obb

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
    e = Element(m)
    e_xf = Xform.translation(10.0, 0.0, 0.0)
    sg = e.session_geometry(e_xf)

    MINI_CHECK(isinstance(sg, Mesh))
    mesh = sg
    MINI_CHECK(TOLERANCE.is_close(mesh.vertex[0].x, 10.0))
    MINI_CHECK(TOLERANCE.is_close(mesh.vertex[1].x, 11.0))
    MINI_CHECK(e.geometry is not mesh)


@MINI_TEST("Element", "Reset")
def test_reset():
    from session_py import Mesh
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(m)
    _ = e.aabb
    _2 = e.point
    e.reset()

    MINI_CHECK(e.is_dirty)
    MINI_CHECK(e.cached_aabb is None)
    MINI_CHECK(e.cached_obb is None)
    MINI_CHECK(e.cached_collision_mesh is None)
    MINI_CHECK(e.cached_point is None)


@MINI_TEST("Element", "Compute Point")
def test_compute_point():
    from session_py import Mesh
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(m)
    pt = e.point

    MINI_CHECK(TOLERANCE.is_close(pt[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 0.0))


@MINI_TEST("Element", "Brep Aabb")
def test_brep_aabb():
    from session_py import BRep
    from session_py import Element

    b = BRep.create_box(2.0, 3.0, 4.0)
    e = Element(b, "brep_element")
    aabb = e.aabb
    pt = e.point

    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[1], 1.5))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[2], 2.0))
    MINI_CHECK(TOLERANCE.is_close(pt[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 0.0))


@MINI_TEST("Element", "Json Roundtrip")
def test_json_roundtrip():
    from session_py import Mesh
    from session_py import Element
    from pathlib import Path

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(m, "json_test")

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_element.json"
    e.file_json_dump(fname)
    loaded = Element.file_json_load(fname)

    MINI_CHECK(loaded.name == "json_test")
    MINI_CHECK(isinstance(loaded.geometry, Mesh))
    MINI_CHECK(len(loaded.geometry.vertex) == 4)


@MINI_TEST("Element", "Protobuf Roundtrip")
def test_protobuf_roundtrip():
    from session_py import BRep
    from session_py import Element
    from pathlib import Path

    b = BRep.create_box(2.0, 3.0, 4.0)
    e = Element(b, "proto_test")

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_element.bin"
    e.pb_dump(path)
    loaded = Element.pb_load(path)

    MINI_CHECK(loaded.name == "proto_test")
    MINI_CHECK(isinstance(loaded.geometry, BRep))
    MINI_CHECK(loaded.geometry.face_count() == 6)
    MINI_CHECK(loaded.geometry.vertex_count() == 8)


# ═══════════════════════════════════════════════════════════════════════════
# Element - Polylines
# ═══════════════════════════════════════════════════════════════════════════


@MINI_TEST("Element", "Polylines")
def test_polylines():
    from session_py import Mesh
    from session_py import Element
    from session_py import Point

    m = Mesh.from_vertices_and_faces(
        [Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0)],
        [[0, 1, 2, 3]],
    )
    e = Element(m, "test_element")

    MINI_CHECK(len(e.polylines) == 1)
    MINI_CHECK(e.polylines[0].point_count() == 5)
    MINI_CHECK(e.polylines[0].get_point(0) == Point(0, 0, 0))
    MINI_CHECK(e.polylines[0].get_point(4) == Point(0, 0, 0))
    MINI_CHECK(len(e.planes) == 1)
    MINI_CHECK(e.planes[0].origin == Point(0.5, 0.5, 0.0))
    normal = e.planes[0].z_axis
    MINI_CHECK(abs(normal[0]) < 1e-12 and abs(normal[1]) < 1e-12 and normal[2] > 0.0)
    MINI_CHECK(len(e.edge_vectors) == 0)
    MINI_CHECK(e.axis is None)


@MINI_TEST("Element", "Polylines Empty Without Mesh")
def test_polylines_empty_without_mesh():
    from session_py import Element

    MINI_CHECK(len(Element(name="no_geometry").polylines) == 0)
    MINI_CHECK(len(Element(name="no_geometry").planes) == 0)


# ═══════════════════════════════════════════════════════════════════════════
# Element - Polymorphic registry
# ═══════════════════════════════════════════════════════════════════════════


def _test_plate_class():
    """Stand-in for a domain element: carries state the kernel knows nothing about."""
    from session_py import Element

    class TestPlate(Element):
        def __init__(self, geometry=None, name="my_element", thickness=0.0, codes=None):
            super().__init__(geometry, name)
            self.thickness = thickness
            self.codes = list(codes or [])

        def element_type_name(self):
            return "TestPlate"

        def element_data_dumps(self):
            out = str(self.thickness)
            for c in self.codes:
                out += "," + str(c)
            return out.encode()

        @staticmethod
        def register_with_kernel():
            def factory(data):
                from session_py.proto import element_pb2

                base = Element.pb_loads(data)
                proto = element_pb2.Element()
                proto.ParseFromString(data)

                plate = TestPlate(base.geometry, base.name)
                plate.guid = proto.guid

                parts = proto.element_data.decode().split(",")
                plate.thickness = float(parts[0])
                for c in parts[1:]:
                    plate.codes.append(int(c))
                return plate

            Element.register_type("TestPlate", factory)

    return TestPlate


def _unit_quad():
    from session_py import Mesh

    return Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )


@MINI_TEST("Element", "Registry Round Trip")
def test_registry_round_trip():
    from session_py import Element
    from session_py import Mesh

    TestPlate = _test_plate_class()
    TestPlate.register_with_kernel()
    MINI_CHECK(Element.is_registered("TestPlate"))

    plate = TestPlate(_unit_quad(), "plate_0", 12.5, [30, 11, 20])
    guid = plate.guid
    loaded = Element.pb_loads_polymorphic(plate.pb_dumps())

    MINI_CHECK(isinstance(loaded, TestPlate))
    MINI_CHECK(loaded.element_type_name() == "TestPlate")

    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(loaded.name == "plate_0")
    MINI_CHECK(isinstance(loaded.geometry, Mesh))
    MINI_CHECK(abs(loaded.thickness - 12.5) < 1e-9)
    MINI_CHECK(len(loaded.codes) == 3)
    MINI_CHECK(
        loaded.codes[0] == 30 and loaded.codes[1] == 11 and loaded.codes[2] == 20
    )


@MINI_TEST("Element", "Registry Unknown Type Degrades")
def test_registry_unknown_type_degrades():
    from session_py import Element
    from session_py import Mesh
    from session_py.proto import element_pb2

    MINI_CHECK(not Element.is_registered("NeverRegistered"))

    proto = element_pb2.Element()
    proto.ParseFromString(Element(_unit_quad(), "mystery").pb_dumps())
    proto.element_type = "NeverRegistered"
    proto.element_data = b"whatever this package meant"

    loaded = Element.pb_loads_polymorphic(proto.SerializeToString())
    MINI_CHECK(loaded is not None)
    MINI_CHECK(loaded.name == "mystery")
    MINI_CHECK(isinstance(loaded.geometry, Mesh))


@MINI_TEST("Element", "Features Round Trip")
def test_features_round_trip():
    from session_py import Element
    from session_py import ElementFeature
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    e = Element(_unit_quad(), "plate_0")
    e.insertion_vectors = [Vector(0, 0, 1), Vector(1, 0, 0)]
    e.dimensions = Vector(120.0, 80.0, 12.5)
    e.add_feature(
        ElementFeature(
            "cut",
            2,
            [
                Polyline(
                    [Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 0, 0)]
                )
            ],
            "notch",
        )
    )
    feature_guid = e.features[0].guid

    loaded = Element.pb_loads(e.pb_dumps())

    MINI_CHECK(len(loaded.insertion_vectors) == 2)
    MINI_CHECK(loaded.insertion_vectors[0] == Vector(0, 0, 1))
    MINI_CHECK(loaded.dimensions is not None)
    MINI_CHECK(abs(loaded.dimensions[2] - 12.5) < 1e-9)
    MINI_CHECK(len(loaded.features) == 1)
    MINI_CHECK(loaded.features[0].feature_type == "cut")
    MINI_CHECK(loaded.features[0].face_index == 2)
    MINI_CHECK(loaded.features[0].name == "notch")
    MINI_CHECK(len(loaded.features[0].outlines) == 1)
    MINI_CHECK(loaded.features[0].guid == feature_guid)


@MINI_TEST("Element", "Dimensions Are Nominal Not Measured")
def test_dimensions_are_nominal_not_measured():
    from session_py import Element
    from session_py import Vector

    e = Element(_unit_quad(), "plate")
    MINI_CHECK(e.dimensions is None)

    e.dimensions = Vector(120.0, 80.0, 12.5)
    measured = e.obb

    MINI_CHECK(abs(e.dimensions[0] - 120.0) < 1e-9)
    MINI_CHECK(measured.half_size[0] < 1.0)


@MINI_TEST("Element", "Registry Leaves Base Bytes Unchanged")
def test_registry_leaves_base_bytes_unchanged():
    from session_py import Element
    from session_py.proto import element_pb2

    e = Element(_unit_quad(), "plain")
    proto = element_pb2.Element()
    proto.ParseFromString(e.pb_dumps())

    MINI_CHECK(proto.element_type == "")
    MINI_CHECK(proto.element_data == b"")
    MINI_CHECK(e.element_type_name() == "")


@MINI_TEST("Element", "Registry Json Round Trip")
def test_registry_json_round_trip():
    from session_py import Element

    TestPlate = _test_plate_class()
    TestPlate.register_with_kernel()

    plate = TestPlate(_unit_quad(), "plate_json", 9.5, [7, 8])
    loaded = Element.file_json_loads_polymorphic(plate.file_json_dumps())

    MINI_CHECK(isinstance(loaded, TestPlate))
    MINI_CHECK(loaded.name == "plate_json")
    MINI_CHECK(loaded.guid == plate.guid)
    MINI_CHECK(abs(loaded.thickness - 9.5) < 1e-9)
    MINI_CHECK(len(loaded.codes) == 2)
    MINI_CHECK(loaded.codes[0] == 7 and loaded.codes[1] == 8)


@MINI_TEST("Element", "Throwing Factory Degrades To Base")
def test_throwing_factory_degrades_to_base():
    from session_py import Element
    from session_py import Mesh
    from session_py.proto import element_pb2

    def explode(data):
        raise RuntimeError("this package is broken")

    Element.register_type("Exploding", explode)

    proto = element_pb2.Element()
    proto.ParseFromString(Element(_unit_quad(), "victim").pb_dumps())
    proto.element_type = "Exploding"

    loaded = Element.pb_loads_polymorphic(proto.SerializeToString())
    MINI_CHECK(loaded is not None)
    MINI_CHECK(loaded.name == "victim")
    MINI_CHECK(isinstance(loaded.geometry, Mesh))


@MINI_TEST("Element", "Unknown Type Survives Resave")
def test_unknown_type_survives_resave():
    from session_py import Element
    from session_py.proto import element_pb2

    proto = element_pb2.Element()
    proto.ParseFromString(Element(_unit_quad(), "plate").pb_dumps())
    proto.element_type = "wood::Plate"
    proto.element_data = b"the package's own bytes"
    original = proto.SerializeToString()

    loaded = Element.pb_loads(original)
    MINI_CHECK(loaded.element_type_name() == "wood::Plate")
    MINI_CHECK(loaded.element_data_dumps() == b"the package's own bytes")

    resaved = element_pb2.Element()
    resaved.ParseFromString(loaded.pb_dumps())
    MINI_CHECK(resaved.element_type == "wood::Plate")
    MINI_CHECK(resaved.element_data == b"the package's own bytes")


@MINI_TEST("Element", "Duplicate Keeps Every Field")
def test_duplicate_keeps_every_field():
    from session_py import Element
    from session_py import ElementFeature
    from session_py import Vector

    e = Element(_unit_quad(), "original")
    e.insertion_vectors = [Vector(0, 0, 1)]
    e.dimensions = Vector(120.0, 80.0, 12.5)
    e.add_feature(ElementFeature("cut", 2, [], "notch"))

    copy = e.duplicate()

    MINI_CHECK(copy == e)
    MINI_CHECK(copy.guid != e.guid)
    MINI_CHECK(len(copy.insertion_vectors) == 1)
    MINI_CHECK(copy.dimensions is not None)
    MINI_CHECK(len(copy.features) == 1)


@MINI_TEST("Element", "Equality Compares Carried Fields")
def test_equality_compares_carried_fields():
    from session_py import Element
    from session_py import Vector

    a = Element(_unit_quad(), "same")
    b = Element(_unit_quad(), "same")
    MINI_CHECK(a == b)

    b.dimensions = Vector(1, 2, 3)
    MINI_CHECK(a != b)


# ═══════════════════════════════════════════════════════════════════════════
# ElementFeature
# ═══════════════════════════════════════════════════════════════════════════


@MINI_TEST("ElementFeature", "Constructor")
def test_element_feature_constructor():
    from session_py import ElementFeature
    from session_py import Point
    from session_py import Polyline

    outline = Polyline([
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 0, 0),
    ])
    f = ElementFeature("cut", 2, [outline], "notch")

    MINI_CHECK(f.feature_type == "cut")
    MINI_CHECK(f.face_index == 2)
    MINI_CHECK(f.name == "notch")
    MINI_CHECK(len(f.outlines) == 1)

    same = ElementFeature("cut", 2, [outline], "notch")
    MINI_CHECK(f == same)
    MINI_CHECK(not (f != same))
    MINI_CHECK(f.guid != same.guid)

    other = ElementFeature("drill", 2, [outline], "notch")
    MINI_CHECK(f != other)

    MINI_CHECK(str(f) == "ElementFeature(cut, face 2, 1 outline(s))")
    MINI_CHECK(repr(f) == str(f))

    empty = ElementFeature()
    MINI_CHECK(empty.face_index == -1)
    MINI_CHECK(len(empty.outlines) == 0)


@MINI_TEST("ElementFeature", "Json Roundtrip")
def test_element_feature_json_roundtrip():
    from session_py import ElementFeature
    from session_py import Point
    from session_py import Polyline
    from pathlib import Path

    f = ElementFeature(
        "cut",
        2,
        [Polyline([Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 0, 0)])],
        "notch",
    )
    feature_guid = f.guid

    fname = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_element_feature.json"
    )
    f.file_json_dump(fname)
    loaded = ElementFeature.file_json_load(fname)

    MINI_CHECK(loaded == f)
    MINI_CHECK(len(loaded.outlines) == 1)
    MINI_CHECK(loaded.guid == feature_guid)


@MINI_TEST("ElementFeature", "Protobuf Roundtrip")
def test_element_feature_protobuf_roundtrip():
    from session_py import ElementFeature
    from session_py import Point
    from session_py import Polyline
    from pathlib import Path

    f = ElementFeature(
        "drill",
        5,
        [Polyline([Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 0, 0)])],
        "hole",
    )
    feature_guid = f.guid

    path = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_element_feature.bin"
    )
    f.pb_dump(path)
    loaded = ElementFeature.pb_load(path)

    MINI_CHECK(loaded == f)
    MINI_CHECK(loaded.feature_type == "drill")
    MINI_CHECK(loaded.face_index == 5)
    MINI_CHECK(len(loaded.outlines) == 1)
    MINI_CHECK(loaded.guid == feature_guid)


if __name__ == "__main__":
    run_all(language="python")
