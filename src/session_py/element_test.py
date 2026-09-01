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


@MINI_TEST("Element", "Place")
def test_place():
    from session_py import Mesh
    from session_py import Xform
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m)
    xf = Xform.translation(10.0, 20.0, 30.0)
    e.place(xf)

    MINI_CHECK(e.is_dirty)
    min_x = min(v.x for v in e.geometry.vertex.values())
    MINI_CHECK(min_x > 9.0)


@MINI_TEST("Element", "Add Geometry Op")
def test_add_feature():
    from session_py import Mesh
    from session_py import BRep
    from session_py import Xform
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m)

    def my_feature(geo):
        return geo

    e.add_geometry_op(my_feature)

    # Features are Mesh -> Mesh, so BRep geometry passes through untouched
    eb = Element(geometry=BRep.create_box(1.0, 1.0, 1.0), name="brep_feature")
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
    e_xf = Xform.translation(10.0, 0.0, 0.0)
    sg = e.session_geometry(e_xf)

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
    from session_py import Element
    from pathlib import Path

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(geometry=m, name="json_test")

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
    from session_py import Element
    from pathlib import Path

    b = BRep.create_box(2.0, 3.0, 4.0)
    e = Element(geometry=b, name="proto_test")

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


# ── Element - polymorphic registry ──────────────────────────────────────────────────────
# Mirrors session_cpp/src/element_test.cpp. The contract a downstream package depends on: a
# registered type survives a round trip, and an UNregistered one degrades to a base Element
# with its geometry intact rather than failing the load.


def _unit_quad():
    from session_py import Mesh

    return Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]]
    )


def _test_plate_class():
    """Stand-in for a domain element: carries state the kernel knows nothing about."""
    from session_py import Element

    class TestPlate(Element):
        def __init__(self, geometry=None, thickness=0.0, codes=None):
            super().__init__(geometry=geometry)
            self.thickness = thickness
            self.codes = list(codes or [])

        def element_type_name(self):
            return "TestPlate"

        # Deliberately a trivial hand-rolled encoding: the kernel never parses this, so the
        # format is the package's own business - which is the property under test.
        def element_data_dumps(self):
            return ",".join([str(self.thickness)] + [str(c) for c in self.codes]).encode()

        @staticmethod
        def factory(data):
            from session_py.proto import element_pb2

            proto = element_pb2.Element()
            proto.ParseFromString(data)
            parts = proto.element_data.decode().split(",")

            plate = TestPlate(thickness=float(parts[0]), codes=[int(c) for c in parts[1:]])
            base = Element.pb_loads(data)
            plate._geometry = base.geometry
            plate.guid = proto.guid
            plate.name = proto.name
            return plate

    return TestPlate


@MINI_TEST("Element", "RegistryRoundTrip")
def test_registry_round_trip():
    from session_py import Element
    from session_py import Mesh

    TestPlate = _test_plate_class()
    Element.register_type("TestPlate", TestPlate.factory)
    MINI_CHECK(Element.is_registered("TestPlate"))

    plate = TestPlate(geometry=_unit_quad(), thickness=12.5, codes=[30, 11, 20])
    plate.name = "plate_0"
    loaded = Element.pb_loads_polymorphic(plate.pb_dumps())

    # The derived type came back, not a sliced base.
    MINI_CHECK(isinstance(loaded, TestPlate))
    MINI_CHECK(loaded.element_type_name() == "TestPlate")

    # Identity, base state and domain state all survived.
    MINI_CHECK(loaded.guid == plate.guid)
    MINI_CHECK(loaded.name == "plate_0")
    MINI_CHECK(isinstance(loaded.geometry, Mesh))
    MINI_CHECK(TOLERANCE.is_close(loaded.thickness, 12.5))
    MINI_CHECK(loaded.codes == [30, 11, 20])


@MINI_TEST("Element", "RegistryUnknownTypeDegrades")
def test_registry_unknown_type_degrades():
    from session_py import Element
    from session_py import Mesh
    from session_py.proto import element_pb2

    # A file written by a package this interpreter does not have. The element must still
    # load, keeping its geometry - a viewer opens the file, it just does not know what it is.
    MINI_CHECK(not Element.is_registered("NeverRegistered"))

    proto = element_pb2.Element()
    proto.ParseFromString(Element(geometry=_unit_quad()).pb_dumps())
    proto.element_type = "NeverRegistered"
    proto.element_data = b"whatever this package meant"

    loaded = Element.pb_loads_polymorphic(proto.SerializeToString())
    MINI_CHECK(loaded is not None)
    MINI_CHECK(isinstance(loaded.geometry, Mesh))


@MINI_TEST("Element", "FeaturesRoundTrip")
def test_features_round_trip():
    from session_py import Element
    from session_py import ElementFeature
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    # insertion_vectors / dimensions / features are the general shape that replaced the
    # per-domain arrays (joint_types and friends) that used to sit on this message. All three
    # must survive a round trip or a domain is right back to inventing its own fields.
    e = Element(geometry=_unit_quad(), name="plate_0")
    e.insertion_vectors = [Vector(0, 0, 1), Vector(1, 0, 0)]
    e.dimensions = Vector(120.0, 80.0, 12.5)
    e.add_feature(
        ElementFeature(
            "cut", 2,
            [Polyline([Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 0, 0)])],
            "notch",
        )
    )
    feature_guid = e.features[0].guid

    loaded = Element.pb_loads(e.pb_dumps())

    MINI_CHECK(len(loaded.insertion_vectors) == 2)
    MINI_CHECK(loaded.insertion_vectors[0] == Vector(0, 0, 1))
    MINI_CHECK(loaded.dimensions is not None)
    # z is the thickness - the whole reason this is a vector rather than one float.
    MINI_CHECK(TOLERANCE.is_close(loaded.dimensions[2], 12.5))
    MINI_CHECK(len(loaded.features) == 1)
    MINI_CHECK(loaded.features[0].feature_type == "cut")
    MINI_CHECK(loaded.features[0].face_index == 2)
    MINI_CHECK(loaded.features[0].name == "notch")
    MINI_CHECK(len(loaded.features[0].outlines) == 1)
    # The guid is the feature's handle: a package that wrote a joint has to find it again, and
    # the index in `features` moves the moment an earlier feature is removed.
    MINI_CHECK(loaded.features[0].guid == feature_guid)


@MINI_TEST("Element", "DimensionsAreNominalNotMeasured")
def test_dimensions_are_nominal_not_measured():
    from session_py import Element
    from session_py import Vector

    # dimensions is AUTHORED intent; obb MEASURES what exists. They are allowed to disagree,
    # and this pins that they are genuinely independent - a nominal thickness set before any
    # geometry is built must not be overwritten by whatever the geometry turns out to be.
    e = Element(geometry=_unit_quad(), name="plate")
    MINI_CHECK(e.dimensions is None)  # never authored

    e.dimensions = Vector(120.0, 80.0, 12.5)  # nominal, nothing like the unit quad
    measured = e.obb

    MINI_CHECK(TOLERANCE.is_close(e.dimensions[0], 120.0))
    MINI_CHECK(measured.half_size[0] < 1.0)  # the geometry is still a unit quad


@MINI_TEST("Element", "RegistryLeavesBaseBytesUnchanged")
def test_registry_leaves_base_bytes_unchanged():
    from session_py import Element
    from session_py.proto import element_pb2

    # proto3 omits empty scalars, so adding element_type/element_data must not have changed
    # one byte of a plain Element - the cross-language golden files depend on it.
    e = Element(geometry=_unit_quad())
    proto = element_pb2.Element()
    proto.ParseFromString(e.pb_dumps())

    MINI_CHECK(proto.element_type == "")
    MINI_CHECK(proto.element_data == b"")
    MINI_CHECK(e.element_type_name() == "")


@MINI_TEST("Element", "UnknownTypeSurvivesResave")
def test_unknown_type_survives_resave():
    from session_py import Element
    from session_py.proto import element_pb2

    # The whole point of element_type/element_data: a viewer WITHOUT the wood package opens a
    # wood file, edits something else, and saves. If the kernel does not carry these two
    # through, that save silently destroys the payload - the geometry still looks right, so
    # nothing announces the loss. This is the test that would have caught it.
    proto = element_pb2.Element()
    proto.ParseFromString(Element(geometry=_unit_quad(), name="plate").pb_dumps())
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


@MINI_TEST("Element", "DuplicateKeepsEveryField")
def test_duplicate_keeps_every_field():
    from session_py import Element
    from session_py import Vector
    from session_py.element import ElementFeature

    # A copy that drops fields is the same silent data loss as a save that drops them, and a
    # duplicate is what an assembly does to place the same part twice.
    e = Element(geometry=_unit_quad(), name="original")
    e.insertion_vectors = [Vector(0, 0, 1)]
    e.dimensions = Vector(120.0, 80.0, 12.5)
    e.add_feature(ElementFeature("cut", 2, [], "notch"))

    copy = e.duplicate()

    MINI_CHECK(copy == e)  # every carried field compares equal
    MINI_CHECK(copy.guid != e.guid)  # but it is a different object
    MINI_CHECK(len(copy.insertion_vectors) == 1)
    MINI_CHECK(copy.dimensions is not None)
    MINI_CHECK(len(copy.features) == 1)


@MINI_TEST("Element", "EqualityComparesCarriedFields")
def test_equality_compares_carried_fields():
    from session_py import Element
    from session_py import Vector

    # Equality that looks at name and geometry only makes every round-trip test above vacuous:
    # it would pass while the loader dropped all five of the other fields.
    a = Element(geometry=_unit_quad(), name="same")
    b = Element(geometry=_unit_quad(), name="same")
    MINI_CHECK(a == b)

    b.dimensions = Vector(1, 2, 3)
    MINI_CHECK(a != b)


###############################################################################################
# ElementFeature
###############################################################################################

@MINI_TEST("ElementFeature", "Constructor")
def test_element_feature_constructor():
    from session_py import Point
    from session_py import Polyline
    from session_py.element import ElementFeature

    outline = Polyline([Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 0, 0)])
    f = ElementFeature("cut", 2, [outline], "notch")

    MINI_CHECK(f.feature_type == "cut")
    MINI_CHECK(f.face_index == 2)
    MINI_CHECK(f.name == "notch")
    MINI_CHECK(len(f.outlines) == 1)

    same = ElementFeature("cut", 2, [outline], "notch")
    MINI_CHECK(f == same)
    MINI_CHECK(not (f != same))
    # Data equality, not identity - the two guids differ and the features are still equal.
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
    from session_py import Point
    from session_py import Polyline
    from session_py.element import ElementFeature

    f = ElementFeature("cut", 2,
                       [Polyline([Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 0, 0)])],
                       "notch")
    feature_guid = f.guid

    fname = "serialization/test_element_feature.json"
    f.file_json_dump(fname)
    loaded = ElementFeature.file_json_load(fname)

    MINI_CHECK(loaded == f)
    MINI_CHECK(len(loaded.outlines) == 1)
    # Read back, not re-minted: whoever holds the guid must still find this feature.
    MINI_CHECK(loaded.guid == feature_guid)


@MINI_TEST("ElementFeature", "Protobuf Roundtrip")
def test_element_feature_protobuf_roundtrip():
    from session_py import Point
    from session_py import Polyline
    from session_py.element import ElementFeature

    f = ElementFeature("drill", 5,
                       [Polyline([Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 0, 0)])],
                       "hole")
    feature_guid = f.guid

    path = "serialization/test_element_feature.bin"
    f.pb_dump(path)
    loaded = ElementFeature.pb_load(path)

    MINI_CHECK(loaded == f)
    MINI_CHECK(loaded.feature_type == "drill")
    MINI_CHECK(loaded.face_index == 5)
    MINI_CHECK(len(loaded.outlines) == 1)
    MINI_CHECK(loaded.guid == feature_guid)


if __name__ == "__main__":
    run_all(language="python")
