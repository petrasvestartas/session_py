from __future__ import annotations

import base64
import json
import shutil
import sys
import tempfile
import threading
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from .line import Line
from .point import Point
from .session import Session


BROWSER_VIEWER = "https://brg-teaching.github.io/Teaching/CAS/website/session-viewer/"


class _Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        for prefix, directory in self.server.session_routes.items():
            if path == prefix or path.startswith(prefix):
                relative = path[len(prefix) :]
                previous = self.directory
                self.directory = str(directory)
                self.path = relative
                try:
                    super().do_GET()
                finally:
                    self.directory = previous
                return
        super().do_GET()

    def log_message(self, format, *args):
        pass


class Viewer:
    """A running local viewer; close it to stop the server and remove its temporary scenes."""

    def __init__(self, server, thread, url: str, temporary: Path | None):
        self.server = server
        self.thread = thread
        self.url = url
        self.temporary = temporary
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        if self.temporary is not None:
            shutil.rmtree(self.temporary, ignore_errors=True)


def _dist(viewer) -> Path:
    if viewer is None:
        root = Path(__file__).resolve().parents[3] / "session_viewer" / "dist"
    else:
        root = Path(viewer).expanduser().resolve()
    if not (root / "index.html").is_file():
        raise FileNotFoundError(
            f"session_viewer is not built at {root}; run 'trunk build' in session_viewer or pass viewer=<dist directory>"
        )
    return root


def _cable_session(data, edges) -> Session:
    if isinstance(data, Session):
        return data
    if edges is None:
        raise TypeError("show(nodes, edges) needs edge index pairs")
    session = Session("cable_net")
    nodes = [point if isinstance(point, Point) else Point(*point) for point in data]
    for index, point in enumerate(nodes):
        point.name = f"n{index}"
        point.width = 6.0
        session.add_point(point, None)
    for index, (start, end) in enumerate(edges):
        line = Line.from_points(nodes[start], nodes[end])
        line.name = f"e{index}"
        line.width = 2.0
        session.add_line(line, None)
    return session


def _browser_url(data, edges, viewer) -> str:
    root = str(viewer or BROWSER_VIEWER).rstrip("/")
    address = urlsplit(root)
    if address.scheme == "http" and address.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("Use HTTPS for the browser viewer, or HTTP on localhost")
    if address.scheme not in {"http", "https"}:
        raise ValueError(
            "In JupyterLite, viewer must be the URL of the static session_viewer build"
        )
    session = _cable_session(data, edges)
    query = urlencode({"scene": "notebook.yaml", "data": "off", "live": "off"})
    payload = base64.b64encode(session.pb_dumps()).decode("ascii")
    return f"{root}/?{query}#session={payload}"


async def supports_webgpu() -> bool:
    if sys.platform != "emscripten":
        return False
    try:
        from js import navigator

        gpu = getattr(navigator, "gpu", None)
        if not gpu:
            return False
        adapter = await gpu.requestAdapter()
        return bool(adapter)
    except Exception:
        return False


def serve(data, edges=None, *, directory=None, viewer=None, port: int = 0) -> Viewer:
    """Serve one Session, or nodes and edges, on 127.0.0.1."""
    root = _dist(viewer)
    temporary = None
    if directory is None:
        temporary = Path(tempfile.mkdtemp(prefix="session-viewer-"))
        directory = temporary
    else:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    prefix = f"/sessions/{identifier}/"
    session = _cable_session(data, edges)
    session.pb_dump(directory / "net.pb")
    scene = {
        "name": session.name,
        "items": [{"file": f"sessions/{identifier}/net.pb", "at": [0, 0, 0]}],
        "view": "iso",
        "grid": False,
    }
    (directory / "scene.yaml").write_text(
        json.dumps(scene, indent=2) + "\n", encoding="utf-8"
    )
    query = urlencode({"scene": f"sessions/{identifier}/scene.yaml", "data": "off"})

    handler = partial(_Handler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.session_routes = {prefix: directory.resolve()}
    thread = threading.Thread(
        target=server.serve_forever, name="session-viewer", daemon=True
    )
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/?{query}"
    return Viewer(server, thread, url, temporary)


def show(
    data, edges=None, *, directory=None, viewer=None, port: int = 0, height: int = 640
):
    """Display one Session, or nodes and edges, in a Jupyter iframe."""
    if sys.platform == "emscripten":
        from IPython.display import IFrame

        return IFrame(
            src=_browser_url(data, edges, viewer), width="100%", height=height
        )

    from IPython.display import IFrame

    running = serve(data, edges, directory=directory, viewer=viewer, port=port)
    return IFrame(src=running.url, width="100%", height=height)
