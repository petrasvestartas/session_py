from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Objects", "Constructor")
def test_objects_constructor():
    from session_py import Objects

    obj = Objects()

    MINI_CHECK(obj.name == "my_objects")
    MINI_CHECK(len(obj.guid) > 0)
    MINI_CHECK(len(str(obj)) > 0)


@MINI_TEST("Objects", "Json Roundtrip")
def test_objects_json_roundtrip():
    from session_py import Objects
    from session_py import Point
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = Objects()
    original.points.append(Point(1.0, 2.0, 3.0))
    original.points.append(Point(4.0, 5.0, 6.0))

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_objects.json"
    file_json_dump(original, fname)
    loaded = file_json_load(fname)

    MINI_CHECK(len(loaded.points) == len(original.points))


@MINI_TEST("Objects", "Protobuf Roundtrip")
def test_objects_protobuf_roundtrip():
    from session_py import Objects
    from session_py import Point
    from pathlib import Path

    original = Objects()
    original.points.append(Point(1.0, 2.0, 3.0))
    original.points.append(Point(4.0, 5.0, 6.0))

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_objects.bin"
    original.pb_dump(fname)
    loaded = Objects.pb_load(fname)

    MINI_CHECK(len(loaded.points) == len(original.points))


@MINI_TEST("Objects", "Component Constructor")
def test_objects_component_constructor():
    # A Component is any object with guid + name + __jsondump__/__jsonload__.
    # Here we use a minimal inline class to keep the test self-contained.
    import uuid
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

    box = Box(width=3.0, height=5.0)
    MINI_CHECK(len(box.guid) > 0)
    MINI_CHECK(box.name == "my_box")
    MINI_CHECK(box.width == 3.0)


@MINI_TEST("Objects", "Component Json Roundtrip")
def test_objects_component_json_roundtrip():
    import uuid
    from session_py import Objects
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

    original = Objects()
    box = Box(width=3.0, height=5.0)
    original.components.append(box)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_objects_component.json"
    original.file_json_dump(str(fname))
    loaded = Objects.file_json_load(str(fname))

    MINI_CHECK(len(loaded.components) == 1)
    MINI_CHECK(isinstance(loaded.components[0], Box))
    MINI_CHECK(loaded.components[0].width == 3.0)
    MINI_CHECK(loaded.components[0].height == 5.0)
    MINI_CHECK(loaded.components[0].guid == box.guid)


@MINI_TEST("Session", "Add Component")
def test_session_add_component():
    import uuid
    from session_py.session import Session
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


if __name__ == "__main__":
    run_all(language="python")
