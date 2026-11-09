import base64
import json
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import unquote
from urllib.parse import urlsplit

from .mini_test import MINI_CHECK
from .mini_test import MINI_TEST


@MINI_TEST("Viewer", "Cable net and deck server")
def test_viewer_serves_cable_net_and_presentation():
    from session_py import Point
    from session_py import Session
    from session_py import Text
    from session_py import Xform
    from session_py.viewer import serve

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "dist"
        root.mkdir()
        (root / "index.html").write_text("viewer", encoding="utf-8")
        scenes = Path(temporary) / "scenes"
        running = serve([[0, 0, 0], [1, 2, 3]], [(0, 1)], directory=scenes, viewer=root)
        try:
            with urllib.request.urlopen(running.url) as response:
                MINI_CHECK(response.read() == b"viewer")
            scene_path = unquote(running.url.split("scene=")[1].split("&")[0])
            with urllib.request.urlopen(
                f"http://127.0.0.1:{running.server.server_address[1]}/{scene_path}"
            ) as response:
                scene = json.loads(response.read())
            MINI_CHECK(scene["view"] == "iso")
            with urllib.request.urlopen(
                f"http://127.0.0.1:{running.server.server_address[1]}/{scene['items'][0]['file']}"
            ) as response:
                session = Session.pb_loads(response.read())
            MINI_CHECK(len(session.objects.points) == 2)
            MINI_CHECK(len(session.objects.lines) == 1)
            MINI_CHECK(session.objects.points[1][1] == 2.0)
        finally:
            running.close()

        scene = Session("Topic\nA caption")
        point = Point(0, 0, 0)
        point.name = "n0"
        point.guid = "z-point"
        scene.add_point(point)
        scene.graph.set_vertex_attribute(point.guid, "spatial", 1.0)
        scene.graph.set_vertex_attribute(point.guid, "preserve_spatial", 1.0)
        scene.graph.set_vertex_attribute(point.guid, "pickable", 0.0)
        scene.animate_xform(point.guid, Xform.translation(1, 0, 0), 900, 150)
        text = Text("x_0", Point(100, 100, 0))
        text.name = "label"
        text.guid = "a-text"
        text.math = True
        scene.add_text(text)
        scene.add_relationship(point.guid, text.guid, "highlight")
        other = Point(1, 0, 0)
        other.name = "other"
        other.guid = "0-other"
        scene.add_point(other)
        scene.add_relationship(text.guid, other.guid, "morph")
        (scenes / "steps").mkdir()
        (scenes / "steps" / "stale.pb").write_bytes(b"stale")
        running = serve([[scene]], directory=scenes, viewer=root)
        try:
            MINI_CHECK("presentation=" in running.url)
            route = unquote(running.url.split("presentation=")[1])
            with urllib.request.urlopen(
                f"http://127.0.0.1:{running.server.server_address[1]}/{route}"
            ) as response:
                document = json.loads(response.read())
            step = document["topics"][0]["steps"][0]
            MINI_CHECK(step["name"] == "Topic" and step["caption"] == "A caption")
            MINI_CHECK(document["topics"][0]["preserve_spatial"])
            MINI_CHECK(document["links"] == [["n0", "label"]])
            MINI_CHECK(document["spatial"] == ["n0"])
            MINI_CHECK(document["non_pickable"] == ["n0"])
            MINI_CHECK(
                step["effects"][0]
                == {
                    "name": "n0",
                    "duration_ms": 900.0,
                    "delay_ms": 150.0,
                    "kind": "move",
                    "to": [1.0, 0.0, 0.0],
                }
            )
            MINI_CHECK(
                step["effects"][1]["kind"] == "morph"
                and step["effects"][1]["from"] == "label"
                and step["effects"][1]["name"] == "other"
            )
            filename = step["items"][0]["file"]
            with urllib.request.urlopen(
                f"http://127.0.0.1:{running.server.server_address[1]}/sessions/{route.split('/')[1]}/{filename}"
            ) as response:
                restored = Session.pb_loads(response.read())
            MINI_CHECK(restored.objects.components[0].type_name == "Text")
            MINI_CHECK(Text.from_component(restored.objects.components[0]) == text)
            MINI_CHECK(not (scenes / "steps" / "stale.pb").exists())
        finally:
            running.close()


@MINI_TEST("Viewer", "Browser URL carries its own scene")
def test_viewer_browser_url_carries_scene():
    from session_py import Session
    from session_py.viewer import _browser_url

    address = urlsplit(
        _browser_url([[0, 0, 0], [1, 2, 3]], [(0, 1)], "https://example.test/viewer/")
    )
    MINI_CHECK(address.scheme == "https")
    MINI_CHECK(address.path == "/viewer/")
    MINI_CHECK(
        parse_qs(address.query)
        == {"scene": ["notebook.yaml"], "data": ["off"], "live": ["off"]}
    )
    MINI_CHECK(address.fragment.startswith("session="))
    session = Session.pb_loads(
        base64.b64decode(address.fragment.removeprefix("session="), validate=True)
    )
    MINI_CHECK(len(session.objects.points) == 2)
    MINI_CHECK(len(session.objects.lines) == 1)
    MINI_CHECK(session.objects.points[1][2] == 3.0)


@MINI_TEST("Viewer", "Large matrices do not overflow the optional color mask")
def test_viewer_large_matrix_color_mask():
    from session_py import Matrix
    from session_py import Session
    from session_py.viewer import _colored

    scene = Session("matrices")
    small = Matrix.from_rows([[1, 0], [0, 2]])
    small.name = "small"
    scene.add_matrix(small)
    large = Matrix.from_rows([[1 if row == col else 0 for col in range(9)] for row in range(9)])
    large.name = "large"
    scene.add_matrix(large)
    MINI_CHECK(_colored(scene) == {"small": [0, 3]})


@MINI_TEST("Viewer", "Matrix equations emit arithmetic and cell flashes")
def test_viewer_matrix_equation():
    from session_py import Session
    from session_py.viewer import MatrixEquation
    from session_py.viewer import _effects
    from session_py.viewer import _object_names

    scene = Session("Multiplication")
    equation = MatrixEquation(scene, 900, 350, cell=50)
    a = equation.matrix("A", [[1, 2], [3, 4]], r"\mathbf{A}")
    equation.symbol(r"\times", "times")
    b = equation.matrix("B", [[5], [6]], r"\mathbf{B}")
    equation.symbol("=", "equals")
    result = equation.matrix("C", [[17], [39]], r"\mathbf{C}")
    note = equation.product(a, b, result, 0, 0, (1200, 600), 300)
    names = _object_names([[scene]])
    effects, _ = _effects(scene, names)

    MINI_CHECK(scene.xform(a.guid).m[12] == 900)
    MINI_CHECK(scene.xform(b.guid).m[12] == 1056)
    MINI_CHECK(scene.xform(result.guid).m[12] == 1162)
    MINI_CHECK(scene.component_lookup[note.guid].name == "C-0-0-calculation")
    from session_py import Text

    MINI_CHECK(Text.from_component(scene.component_lookup[note.guid]).text == "1\\times5+2\\times6=17")
    title_guid = next(guid for guid, component in scene.component_lookup.items() if component.name == "A-title")
    MINI_CHECK(Text.from_component(scene.component_lookup[title_guid]).at[1] == 264.0)
    flashes = []
    for effect in effects:
        if effect["kind"] == "highlight":
            flashes.append(effect)
    names = set()
    for effect in flashes:
        names.add(effect["name"])
    MINI_CHECK(len(flashes) == 5)
    MINI_CHECK(names == {"A[0,0]", "A[0,1]", "B[0,0]", "B[1,0]", "C[0,0]"})
    result_flash = next(effect for effect in flashes if effect["name"] == "C[0,0]")
    MINI_CHECK(result_flash["delay_ms"] == 1300)
