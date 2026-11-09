from __future__ import annotations
from pathlib import Path
from .mini_test import MINI_CHECK
from .mini_test import MINI_TEST


@MINI_TEST("Text", "Constructor")
def test_text_constructor():
    from session_py import Point
    from session_py import Text

    text = Text("n_0", Point(1.0, 2.0, 3.0))
    text.name = "n0"
    text.math = True
    text.spatial = True
    text.outline = -1.5
    copy = text.duplicate()
    other = Text()
    spatial = Text.from_tex("n_0", Point(1.0, 2.0, 3.0), True)
    flat = Text.from_tex("x_0", Point(0.0, 0.0, 0.0))
    string = str(text)
    representation = repr(text)

    MINI_CHECK(text.name == "n0" and bool(text.guid))
    MINI_CHECK(text.at == Point(1.0, 2.0, 3.0))
    MINI_CHECK(text.height == 28.0 and text.color.r == 1.0)
    MINI_CHECK(text.math and text.spatial and not text.serif)
    MINI_CHECK(copy == text and copy.guid != text.guid)
    MINI_CHECK(other != text)
    MINI_CHECK(spatial.math and spatial.height == 18.0 and spatial.outline == -1.5)
    MINI_CHECK(flat.math and not flat.spatial and flat.height == 28.0 and flat.serif)
    MINI_CHECK(string == "n_0")
    MINI_CHECK("Text(" in representation)


@MINI_TEST("Text", "Json")
def test_text_json():
    from session_py import Point
    from session_py import Text

    text = Text("q=\\frac{f}{l}", Point(4.0, 5.0, 6.0))
    text.math = True
    text.serif = True
    filename = Path("text_test.json")
    loaded = Text.file_json_loads(text.file_json_dumps())
    text.file_json_dump(filename)
    from_file = Text.file_json_load(filename)
    filename.unlink()

    MINI_CHECK(loaded == text)
    MINI_CHECK(from_file == text)


@MINI_TEST("Text", "Protobuf")
def test_text_protobuf():
    from session_py import Point
    from session_py import Text

    text = Text("x_0", Point(7.0, 8.0, 9.0))
    text.math = True
    text.spatial = True
    text.outline = -1.5
    filename = Path("text_test.pb")
    loaded = Text.pb_loads(text.pb_dumps())
    text.pb_dump(filename)
    from_file = Text.pb_load(filename)
    filename.unlink()

    MINI_CHECK(loaded == text)
    MINI_CHECK(from_file == text)


@MINI_TEST("Text", "Component")
def test_text_component():
    from session_py import Point
    from session_py import Session
    from session_py import Text

    text = Text("C[0,1]", Point(10.0, 11.0, 0.0))
    text.math = True
    component = text.to_component()
    loaded = Text.from_component(component)
    session = Session("text")
    session.add_text(text)
    restored = Session.pb_loads(session.pb_dumps())

    MINI_CHECK(component.type_name == "Text")
    MINI_CHECK(component.guid == text.guid)
    MINI_CHECK(loaded == text)
    MINI_CHECK(len(session.objects.components) == 1)
    MINI_CHECK(session.objects.components[0].type_name == "Text")
    MINI_CHECK(Text.from_component(restored.objects.components[0]) == text)
