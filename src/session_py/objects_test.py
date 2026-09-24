from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("Objects", "Constructor")
def test_objects_constructor():
    from session_py import Objects

    objects = Objects()
    named = Objects("custom_objects")

    MINI_CHECK(objects.name == "my_objects")
    MINI_CHECK(len(objects.guid) > 0)
    MINI_CHECK(len(objects.points) == 0)
    MINI_CHECK(len(objects.instances) == 0)
    MINI_CHECK(
        str(objects) == "Objects(name=my_objects, guid=" + objects.guid + ", points=0)"
    )
    MINI_CHECK(repr(objects) == str(objects))
    MINI_CHECK(named.name == "custom_objects")


@MINI_TEST("Objects", "Json Roundtrip")
def test_objects_json_roundtrip():
    from session_py import InstanceRef
    from session_py import Line
    from session_py import Mesh
    from session_py import Objects
    from session_py import Plane
    from session_py import Point
    from session_py import Xform
    from pathlib import Path

    original = Objects()
    original.points.append(Point(1.0, 2.0, 3.0))
    original.points.append(Point(4.0, 5.0, 6.0))
    original.lines.append(Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    original.planes.append(Plane.xy_plane())
    original.meshes.append(Mesh.create_box(1.0, 1.0, 1.0))
    instance = InstanceRef("def-abc", Xform.identity())
    guid = instance.guid
    original.instances.append(instance)

    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_objects.json"
    )
    original.file_json_dump(filename)
    loaded = Objects.file_json_load(filename)
    parsed = Objects.file_json_loads(original.file_json_dumps())

    MINI_CHECK(loaded.guid == original.guid)
    MINI_CHECK(parsed.guid == original.guid)
    MINI_CHECK(len(loaded.points) == 2)
    MINI_CHECK(loaded.points[1][0] == 4.0)
    MINI_CHECK(len(loaded.lines) == 1)
    MINI_CHECK(loaded.lines[0].end()[0] == 1.0)
    MINI_CHECK(len(loaded.planes) == 1)
    MINI_CHECK(loaded.planes[0].z_axis[2] == 1.0)
    MINI_CHECK(len(loaded.meshes) == 1)
    MINI_CHECK(loaded.meshes[0].number_of_faces() == 6)
    MINI_CHECK(len(loaded.instances) == 1)
    MINI_CHECK(loaded.instances[0].guid == guid)
    MINI_CHECK(loaded.instances[0].definition_guid == "def-abc")


@MINI_TEST("Objects", "Protobuf Roundtrip")
def test_objects_protobuf_roundtrip():
    from session_py import InstanceRef
    from session_py import Line
    from session_py import Mesh
    from session_py import Objects
    from session_py import Plane
    from session_py import Point
    from session_py import Xform
    from pathlib import Path

    original = Objects()
    original.points.append(Point(1.0, 2.0, 3.0))
    original.points.append(Point(4.0, 5.0, 6.0))
    original.lines.append(Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    original.planes.append(Plane.xy_plane())
    original.meshes.append(Mesh.create_box(1.0, 1.0, 1.0))
    instance = InstanceRef("def-abc", Xform.identity())
    guid = instance.guid
    original.instances.append(instance)

    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_objects.bin"
    )
    original.pb_dump(filename)
    loaded = Objects.pb_load(filename)
    parsed = Objects.pb_loads(original.pb_dumps())

    MINI_CHECK(len(parsed.points) == 2)
    MINI_CHECK(len(loaded.points) == 2)
    MINI_CHECK(loaded.points[1][0] == 4.0)
    MINI_CHECK(len(loaded.lines) == 1)
    MINI_CHECK(loaded.lines[0].end()[0] == 1.0)
    MINI_CHECK(len(loaded.planes) == 1)
    MINI_CHECK(loaded.planes[0].z_axis[2] == 1.0)
    MINI_CHECK(len(loaded.meshes) == 1)
    MINI_CHECK(loaded.meshes[0].number_of_faces() == 6)
    MINI_CHECK(len(loaded.instances) == 1)
    MINI_CHECK(loaded.instances[0].guid == guid)
    MINI_CHECK(loaded.instances[0].definition_guid == "def-abc")


@MINI_TEST("Objects", "Component Constructor")
def test_objects_component_constructor():
    from session_py import Component

    component = Component()
    component.type_name = "FloorBuilder"
    component.name = "floor_builder"
    component.extra = {"size": 3000, "height": 650}

    MINI_CHECK(Component().name == "my_component")
    MINI_CHECK(component.type_name == "FloorBuilder")
    MINI_CHECK(component.name == "floor_builder")
    MINI_CHECK(len(component.guid) > 0)
    MINI_CHECK(component.extra["size"] == 3000)


@MINI_TEST("Objects", "Component Json Roundtrip")
def test_objects_component_json_roundtrip():
    from session_py import Component

    original = Component()
    original.type_name = "FloorBuilder"
    original.name = "floor_builder"
    original.extra = {"size": 3000, "height": 650, "rise": 453}
    guid = original.guid

    data = original.__jsondump__()

    MINI_CHECK(data["type"] == "FloorBuilder")
    MINI_CHECK(data["guid"] == guid)
    MINI_CHECK(data["size"] == 3000)
    MINI_CHECK(data["height"] == 650)

    loaded = Component.__jsonload__(data)

    MINI_CHECK(loaded.type_name == "FloorBuilder")
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(loaded.extra["size"] == 3000)
    MINI_CHECK(loaded.extra["rise"] == 453)


@MINI_TEST("Objects", "Objects Component Json Roundtrip")
def test_objects_objects_component_json_roundtrip():
    from session_py import Component
    from session_py import Objects
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = Objects()
    component = Component()
    component.type_name = "FloorBuilder"
    component.name = "floor_builder"
    component.extra = {"size": 3000, "height": 650}
    guid = component.guid
    original.components.append(component)

    filename = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_objects_component.json"
    )
    file_json_dump(original, filename)
    loaded = file_json_load(filename)

    MINI_CHECK(len(loaded.components) == 1)
    MINI_CHECK(loaded.components[0].type_name == "FloorBuilder")
    MINI_CHECK(loaded.components[0].extra["size"] == 3000)
    MINI_CHECK(loaded.components[0].guid == guid)


@MINI_TEST("Objects", "Component Protobuf Roundtrip")
def test_objects_component_protobuf_roundtrip():
    from session_py import Component
    from session_py import Objects
    from pathlib import Path

    original = Objects()
    component = Component()
    component.type_name = "FloorBuilder"
    component.name = "floor_builder"
    component.extra = {"size": 3000, "height": 650}
    guid = component.guid
    original.components.append(component)

    filename = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_objects_component.bin"
    )
    original.pb_dump(filename)
    loaded = Objects.pb_load(filename)

    MINI_CHECK(len(loaded.components) == 1)
    MINI_CHECK(loaded.components[0].type_name == "FloorBuilder")
    MINI_CHECK(loaded.components[0].guid == guid)
    MINI_CHECK(loaded.components[0].extra["size"] == 3000)


if __name__ == "__main__":
    run_all(language="python")
