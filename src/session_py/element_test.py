from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


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
        min_x = min(min_x, v.position()[0])

    MINI_CHECK(min_x > 9.0)


@MINI_TEST("Element", "Place Moves Features")
def test_place_moves_features():
    from session_py import Mesh
    from session_py import Element
    from session_py import ElementFeature
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector
    from session_py import Xform

    m = Mesh.from_vertices_and_faces(
        [Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0)],
        [[0, 1, 2, 3]],
    )
    e = Element(m)
    e.add_feature(
        ElementFeature("contact", 0, [Polyline([Point(0, 0, 0), Point(1, 0, 0)])])
    )
    e.set_insertion_vectors([Vector(1, 0, 0)])
    guid = e.features[0].guid
    e.place(Xform.translation(0.0, 0.0, 5.0) * Xform.rotation_z(PI / 2.0))

    moved = e.features[0].outlines[0].get_point(1)
    turned = e.insertion_vectors[0]

    MINI_CHECK(
        TOLERANCE.is_close(moved[0], 0.0)
        and TOLERANCE.is_close(moved[1], 1.0)
        and TOLERANCE.is_close(moved[2], 5.0)
    )
    MINI_CHECK(
        TOLERANCE.is_close(turned[0], 0.0)
        and TOLERANCE.is_close(turned[1], 1.0)
        and TOLERANCE.is_close(turned[2], 0.0)
    )
    MINI_CHECK(e.features[0].guid == guid)


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

    def my_feature(geo):
        return geo

    def empty_mesh(geo):
        return Mesh()

    e = Element(m)
    e.add_geometry_op(my_feature)

    eb = Element(BRep.create_box(1.0, 1.0, 1.0), "brep_feature")
    eb.add_geometry_op(empty_mesh)
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

    def my_feature(geo):
        return geo

    e.add_geometry_op(my_feature)

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


@MINI_TEST("Element", "Session Geometry Mesh")
def test_session_geometry_mesh():
    from session_py import Mesh
    from session_py import Xform
    from session_py import Element

    m = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
    )
    e = Element(m)
    e_xf = Xform.translation(10.0, 0.0, 0.0)
    mesh = e.session_geometry_mesh(e_xf)

    MINI_CHECK(TOLERANCE.is_close(mesh.vertex[0].position()[0], 10.0))
    MINI_CHECK(TOLERANCE.is_close(mesh.vertex[1].position()[0], 11.0))
    MINI_CHECK(e.geometry_mesh() is not mesh)


@MINI_TEST("Element", "Element Geometry Mesh")
def test_element_geometry_mesh():
    from session_py import Mesh
    from session_py import BRep
    from session_py import Element

    mesh = Mesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    element = Element(mesh)
    brep = Element(BRep.create_box(1.0, 1.0, 1.0))

    MINI_CHECK(element.element_geometry_mesh() == mesh)
    MINI_CHECK(element.element_geometry_mesh() is element.element_geometry_mesh())
    MINI_CHECK(brep.element_geometry_mesh().number_of_faces() == 0)
    MINI_CHECK(brep.element_geometry_brep().face_count() == 6)


@MINI_TEST("Element", "Element Geometry Brep")
def test_element_geometry_brep():
    from session_py import Mesh
    from session_py import BRep
    from session_py import Element

    brep = BRep.create_box(1.0, 1.0, 1.0)
    element = Element(brep)
    mesh = Element(Mesh())

    MINI_CHECK(element.element_geometry_brep().vertex_points() == brep.vertex_points())
    MINI_CHECK(element.element_geometry_brep() is element.element_geometry_brep())
    MINI_CHECK(mesh.element_geometry_brep().face_count() == 0)
    MINI_CHECK(mesh.geometry_type_name == "Mesh")


@MINI_TEST("Element", "Model Geometry Mesh")
def test_model_geometry_mesh():
    from session_py import Mesh
    from session_py import Xform
    from session_py import Element

    mesh = Mesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    element = Element(mesh)
    operations = 0

    def shift(geometry):
        nonlocal operations
        operations += 1
        geometry.transform(Xform.translation(10.0, 0.0, 0.0))
        return geometry

    element.add_geometry_op(shift)
    model = element.model_geometry_mesh()
    guid = model.guid

    MINI_CHECK(TOLERANCE.is_close(model.vertex[0].position()[0], 10.0))
    MINI_CHECK(
        TOLERANCE.is_close(element.element_geometry_mesh().vertex[0].position()[0], 0.0)
    )
    MINI_CHECK(element.model_geometry_mesh().guid == guid)
    MINI_CHECK(element.model_geometry_brep().face_count() == 0)
    MINI_CHECK(element.model_geometry_mesh().guid == guid)
    MINI_CHECK(operations == 1)

    element.invalidate_geometry()
    MINI_CHECK(element.model_geometry_mesh().guid != guid)
    MINI_CHECK(operations == 2)

    element.set_geometry(mesh.transformed(Xform.translation(5.0, 0.0, 0.0)))
    MINI_CHECK(
        TOLERANCE.is_close(element.model_geometry_mesh().vertex[0].position()[0], 15.0)
    )
    MINI_CHECK(operations == 3)


@MINI_TEST("Element", "Model Geometry Brep")
def test_model_geometry_brep():
    from session_py import BRep
    from session_py import Xform
    from session_py import Element

    element = Element(BRep.create_box(1.0, 1.0, 1.0))
    model = element.model_geometry_brep()
    guid = model.guid
    points = model.vertex_points()

    MINI_CHECK(points == element.element_geometry_brep().vertex_points())
    MINI_CHECK(element.model_geometry_brep().guid == guid)
    MINI_CHECK(element.model_geometry_mesh().number_of_faces() == 0)
    MINI_CHECK(element.model_geometry_brep().guid == guid)

    element.invalidate_geometry()
    MINI_CHECK(element.model_geometry_brep().guid != guid)

    element.place(Xform.translation(10.0, 0.0, 0.0))
    MINI_CHECK(element.model_geometry_brep().vertex_points() != points)
    MINI_CHECK(
        element.model_geometry_brep().vertex_points()
        == element.element_geometry_brep().vertex_points()
    )


@MINI_TEST("Element", "Geometry Mesh")
def test_geometry_mesh():
    from session_py import Mesh
    from session_py import BRep
    from session_py import Element

    mesh = Mesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    element = Element(mesh)
    empty = Element()
    brep = Element(BRep.create_box(1.0, 1.0, 1.0))

    MINI_CHECK(element.geometry_mesh().number_of_faces() == 1)
    MINI_CHECK(element.geometry_mesh() is element.geometry_mesh())
    MINI_CHECK(empty.geometry_mesh().number_of_faces() == 0)
    MINI_CHECK(brep.geometry_mesh().number_of_faces() == 0)
    MINI_CHECK(brep.geometry_type_name == "BRep")


@MINI_TEST("Element", "Geometry Brep")
def test_geometry_brep():
    from session_py import Mesh
    from session_py import BRep
    from session_py import Element

    element = Element(BRep.create_box(1.0, 1.0, 1.0))
    empty = Element()
    mesh = Element(Mesh())

    MINI_CHECK(element.geometry_brep().face_count() == 6)
    MINI_CHECK(element.geometry_brep() is element.geometry_brep())
    MINI_CHECK(empty.geometry_brep().face_count() == 0)
    MINI_CHECK(mesh.geometry_brep().face_count() == 0)
    MINI_CHECK(mesh.geometry_type_name == "Mesh")


@MINI_TEST("Element", "Session Geometry Brep")
def test_session_geometry_brep():
    from session_py import BRep
    from session_py import Xform
    from session_py import Element

    element = Element(BRep.create_box(1.0, 1.0, 1.0))
    xform = Xform.translation(10.0, 20.0, 30.0)
    placed = element.session_geometry_brep(xform)
    expected = element.geometry_brep().transformed(xform)

    MINI_CHECK(placed.vertex_points() == expected.vertex_points())
    MINI_CHECK(placed.vertex_points() != element.geometry_brep().vertex_points())
    MINI_CHECK(Element().session_geometry_brep(xform).face_count() == 0)


@MINI_TEST("Element", "Compute Geometry Mesh")
def test_compute_geometry_mesh():
    from session_py import Mesh
    from session_py import Element

    mesh = Mesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    element = Element(mesh)
    before = element.geometry_synced()
    element.compute_geometry_mesh()
    element.compute_geometry_mesh()

    MINI_CHECK(not before)
    MINI_CHECK(element.geometry_synced())
    MINI_CHECK(element.geometry_type_name == "Mesh")

    element.invalidate_geometry()
    stale = element.geometry_synced()
    faces = element.geometry_mesh().number_of_faces()

    MINI_CHECK(not stale)
    MINI_CHECK(faces == 1)
    MINI_CHECK(element.geometry_synced())


@MINI_TEST("Element", "Compute Geometry Brep")
def test_compute_geometry_brep():
    from session_py import BRep
    from session_py import Element

    element = Element(BRep.create_box(1.0, 1.0, 1.0))
    element.compute_geometry_brep()
    element.compute_geometry_brep()

    MINI_CHECK(element.geometry_synced())
    MINI_CHECK(element.geometry_type_name == "BRep")

    element.invalidate_geometry()
    loaded = Element.pb_loads(element.pb_dumps())

    MINI_CHECK(element.geometry_synced())
    MINI_CHECK(loaded.geometry_type_name == "BRep")
    MINI_CHECK(loaded.geometry_brep().face_count() == 6)


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

    MINI_CHECK(
        TOLERANCE.is_close(normal[0], 0.0)
        and TOLERANCE.is_close(normal[1], 0.0)
        and normal[2] > 0.0
    )
    MINI_CHECK(len(e.edge_vectors) == 0)
    MINI_CHECK(e.axis is None)


@MINI_TEST("Element", "Set Polylines Sticks")
def test_set_polylines_sticks():
    from session_py import Mesh
    from session_py import Element
    from session_py import Plane
    from session_py import Point
    from session_py import Polyline

    m = Mesh.from_vertices_and_faces(
        [Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0)],
        [[0, 1, 2, 3]],
    )
    e = Element(m)
    e.set_polylines([Polyline([Point(0, 0, 0), Point(2, 0, 0)])])
    e.set_planes([Plane.xy_plane()])

    MINI_CHECK(len(e.polylines) == 1)
    MINI_CHECK(e.polylines[0].point_count() == 2)
    MINI_CHECK(e.planes[0].origin == Point(0, 0, 0))


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
            """Construct from geometry, name, thickness and codes."""

            super().__init__(geometry, name)
            self.thickness = thickness
            self.codes = list(codes or [])

        def element_type_name(self):
            """Return the registered type name."""
            return "TestPlate"

        def element_data_dumps(self):
            """Return the thickness and codes as comma-separated text."""

            out = str(self.thickness)

            for c in self.codes:
                out += "," + str(c)

            return out.encode()

        @staticmethod
        def factory(data):
            """Build a plate from full serialized session_proto.Element bytes."""

            base = Element.pb_loads(data)
            plate = TestPlate(base.geometry, base.name)
            plate.guid = base.guid

            parts = base.element_data_dumps().decode().split(",")
            plate.thickness = float(parts[0])

            for c in parts[1:]:
                plate.codes.append(int(c))

            return plate

        @staticmethod
        def register_with_kernel():
            """Register the factory under "TestPlate"."""
            Element.register_type("TestPlate", TestPlate.factory)

    return TestPlate


def _unit_quad():
    """Return a unit square mesh in the xy plane."""

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
    copy = plate.duplicate()

    MINI_CHECK(isinstance(loaded, TestPlate))
    MINI_CHECK(loaded.element_type_name() == "TestPlate")

    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(loaded.name == "plate_0")
    MINI_CHECK(isinstance(loaded.geometry, Mesh))
    MINI_CHECK(TOLERANCE.is_close(loaded.thickness, 12.5))
    MINI_CHECK(len(loaded.codes) == 3)
    MINI_CHECK(
        loaded.codes[0] == 30 and loaded.codes[1] == 11 and loaded.codes[2] == 20
    )
    MINI_CHECK(copy.codes == plate.codes)


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
    e.set_insertion_vectors([Vector(0, 0, 1), Vector(1, 0, 0)])
    e.set_dimensions(Vector(120.0, 80.0, 12.5))
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
    MINI_CHECK(TOLERANCE.is_close(loaded.dimensions[2], 12.5))
    MINI_CHECK(len(loaded.features) == 1)
    MINI_CHECK(loaded.features[0].feature_type == "cut")
    MINI_CHECK(loaded.features[0].face_index == 2)
    MINI_CHECK(loaded.features[0].name == "notch")
    MINI_CHECK(len(loaded.features[0].outlines) == 1)
    MINI_CHECK(loaded.features[0].visible)
    MINI_CHECK(loaded.features[0].guid == feature_guid)


@MINI_TEST("Element", "Dimensions Are Nominal Not Measured")
def test_dimensions_are_nominal_not_measured():
    from session_py import Element
    from session_py import Vector

    e = Element(_unit_quad(), "plate")

    MINI_CHECK(e.dimensions is None)

    e.set_dimensions(Vector(120.0, 80.0, 12.5))
    measured = e.obb

    MINI_CHECK(TOLERANCE.is_close(e.dimensions[0], 120.0))
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
    MINI_CHECK(TOLERANCE.is_close(loaded.thickness, 9.5))
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
    e.set_insertion_vectors([Vector(0, 0, 1)])
    e.set_dimensions(Vector(120.0, 80.0, 12.5))
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

    b.set_dimensions(Vector(1, 2, 3))

    MINI_CHECK(a != b)


# ═══════════════════════════════════════════════════════════════════════════
# ElementFeature
# ═══════════════════════════════════════════════════════════════════════════


@MINI_TEST("ElementFeature", "Constructor")
def test_element_feature_constructor():
    from session_py import ElementFeature
    from session_py import Point
    from session_py import Polyline

    outline = Polyline(
        [
            Point(0, 0, 0),
            Point(1, 0, 0),
            Point(1, 1, 0),
            Point(0, 0, 0),
        ]
    )
    f = ElementFeature("cut", 2, [outline], "notch")

    MINI_CHECK(f.feature_type == "cut")
    MINI_CHECK(f.face_index == 2)
    MINI_CHECK(f.name == "notch")
    MINI_CHECK(len(f.outlines) == 1)
    MINI_CHECK(f.visible)

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
    f.visible = False

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
    MINI_CHECK(not loaded.visible)
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
    f.visible = False

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
    MINI_CHECK(not loaded.visible)
    MINI_CHECK(loaded.guid == feature_guid)


if __name__ == "__main__":
    run_all(language="python")
