from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Objects", "Constructor")
def test_objects_constructor():
    from session_py import Objects

    obj = Objects()
    named = Objects("custom_objects")

    MINI_CHECK(obj.name == "my_objects")
    MINI_CHECK(len(obj.guid) > 0)
    MINI_CHECK(len(str(obj)) > 0)
    MINI_CHECK(named.name == "custom_objects")


@MINI_TEST("Objects", "Json Roundtrip")
def test_objects_json_roundtrip():
    from session_py import Objects
    from session_py import Point
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = Objects()
    point1 = Point(1.0, 2.0, 3.0)
    point2 = Point(4.0, 5.0, 6.0)
    original.points.append(point1)
    original.points.append(point2)

    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_objects.json"
    )
    file_json_dump(original, filename)
    loaded = file_json_load(filename)

    MINI_CHECK(len(loaded.points) == len(original.points))


@MINI_TEST("Objects", "Protobuf Roundtrip")
def test_objects_protobuf_roundtrip():
    from session_py import Objects
    from session_py import Point
    from pathlib import Path

    original = Objects()
    point1 = Point(1.0, 2.0, 3.0)
    point2 = Point(4.0, 5.0, 6.0)
    original.points.append(point1)
    original.points.append(point2)

    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_objects.bin"
    )
    original.pb_dump(filename)
    loaded = Objects.pb_load(filename)

    MINI_CHECK(len(loaded.points) == len(original.points))


@MINI_TEST("Objects", "Component Constructor")
def test_objects_component_constructor():
    from session_py import Component

    c = Component()
    c.type_name = "FloorBuilder"
    c.name = "floor_builder"
    c.extra = {"size": 3000, "height": 650}

    MINI_CHECK(c.type_name == "FloorBuilder")
    MINI_CHECK(c.name == "floor_builder")
    MINI_CHECK(len(c.guid) > 0)
    MINI_CHECK(c.extra["size"] == 3000)


@MINI_TEST("Objects", "Component Json Roundtrip")
def test_objects_component_json_roundtrip():
    from session_py import Component

    original = Component()
    original.type_name = "FloorBuilder"
    original.name = "floor_builder"
    original.extra = {"size": 3000, "height": 650, "rise": 453}
    original_guid = original.guid

    j = original.__jsondump__()
    MINI_CHECK(j["type"] == "FloorBuilder")
    MINI_CHECK(j["guid"] == original_guid)
    MINI_CHECK(j["size"] == 3000)
    MINI_CHECK(j["height"] == 650)

    loaded = Component.__jsonload__(j)
    MINI_CHECK(loaded.type_name == "FloorBuilder")
    MINI_CHECK(loaded.guid == original_guid)
    MINI_CHECK(loaded.extra["size"] == 3000)
    MINI_CHECK(loaded.extra["rise"] == 453)


@MINI_TEST("Objects", "Objects Component Json Roundtrip")
def test_objects_objects_component_json_roundtrip():
    from session_py import Objects
    from session_py import Component
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = Objects()
    c = Component()
    c.type_name = "FloorBuilder"
    c.name = "floor_builder"
    c.extra = {"size": 3000, "height": 650}
    expected_guid = c.guid
    original.components.append(c)

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
    MINI_CHECK(loaded.components[0].guid == expected_guid)


@MINI_TEST("Objects", "Component Protobuf Roundtrip")
def test_objects_component_protobuf_roundtrip():
    from session_py import Objects
    from session_py import Component
    from pathlib import Path

    original = Objects()
    c = Component()
    c.type_name = "FloorBuilder"
    c.name = "floor_builder"
    c.extra = {"size": 3000, "height": 650}
    expected_guid = c.guid
    original.components.append(c)

    filename = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_objects_component.bin"
    )
    original.pb_dump(filename)
    loaded = Objects.pb_load(filename)

    MINI_CHECK(len(loaded.components) == 1)
    MINI_CHECK(loaded.components[0].type_name == "FloorBuilder")
    MINI_CHECK(loaded.components[0].guid == expected_guid)
    MINI_CHECK(loaded.components[0].extra["size"] == 3000)


if __name__ == "__main__":
    run_all(language="python")
