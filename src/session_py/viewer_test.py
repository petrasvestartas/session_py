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


@MINI_TEST("Viewer", "Cable net server")
def test_viewer_serves_cable_net():
    from session_py import Session
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
