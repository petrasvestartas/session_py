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

from .color import Color
from .line import Line
from .matrix import Matrix
from .point import Point
from .polyline import Polyline
from .session import Session
from .text import Text
from .xform import Xform


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


def _number(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


class MatrixEquation:
    """Arrange kernel matrices and animate their arithmetic with session graph effects."""

    def __init__(
        self,
        session: Session,
        left: float,
        center_y: float,
        cell: float = 58.0,
        gap: float = 28.0,
    ) -> None:
        self.session = session
        self.x = left
        self.center_y = center_y
        self.cell = cell
        self.gap = gap
        self.matrices = {}

    def matrix(
        self,
        name: str,
        values: list[list[float]],
        title: str | None = None,
        cell: float | None = None,
    ) -> Matrix:
        """Append a matrix whose rounded cells and values are typeset by the viewer."""
        matrix = Matrix.from_rows(values)
        matrix.name = name
        matrix.cell = self.cell if cell is None else cell
        matrix.mincolor = Color(37 / 255, 43 / 255, 53 / 255)
        matrix.maxcolor = matrix.mincolor.duplicate()
        matrix.zerocolor = matrix.mincolor.duplicate()
        matrix.textcolor = Color(232 / 255, 237 / 255, 243 / 255)
        matrix.linecolor = matrix.textcolor.duplicate()
        self.session.add_matrix(matrix, None)
        y = self.center_y - (matrix.rows - 1) * matrix.cell * 0.5
        screen = Xform()
        screen.m[5] = -1.0
        screen.m[13] = 760.0
        placement = Xform.translation(self.x, y, 0.0)
        self.session.set_xform(matrix.guid, screen * placement)
        if title is not None:
            label = Text.from_tex(
                title,
                Point(self.x + (matrix.cols - 1) * matrix.cell * 0.5, y - matrix.cell * 0.5 - 36.0),
            )
            label.name = f"{name}-title"
            label.height = 28.0
            label.color = matrix.textcolor.duplicate()
            self.session.add_text(label)
        self.x += matrix.cols * matrix.cell + self.gap
        self.matrices[name] = matrix
        return matrix

    def symbol(self, tex: str, name: str) -> Text:
        """Append an operator exactly between the adjacent matrix boundaries."""
        label = Text.from_tex(tex, Point(self.x - self.cell * 0.5, self.center_y))
        label.name = name
        label.height = 34.0
        label.color = Color(232 / 255, 237 / 255, 243 / 255)
        self.session.add_text(label)
        self.x += self.gap
        return label

    def morph(
        self,
        source: Text,
        target: Text,
        duration: float = 1200.0,
        delay: float = 0.0,
    ) -> None:
        """Transform one equation into another with an ordinary morph relationship."""
        edge = (source.guid, target.guid)
        self.session.add_relationship(*edge, "morph")
        self.session.graph.set_edge_attribute(edge, "duration_ms", duration)
        self.session.graph.set_edge_attribute(edge, "delay_ms", delay)

    def product(
        self,
        a: Matrix,
        b: Matrix,
        result: Matrix,
        row: int,
        col: int,
        at: tuple[float, float],
        delay: float = 0.0,
    ) -> Text:
        """Reveal one dot product while flashing its operands and result cell."""
        if a.cols != b.rows or result.rows != a.rows or result.cols != b.cols:
            raise ValueError("Matrix dimensions are incompatible for multiplication")
        formula = ""
        for k in range(a.cols):
            sign = "-" if a[row, k] * b[k, col] < 0 else "+"
            if k == 0:
                sign = "-" if sign == "-" else ""
            term = f"{_number(abs(a[row, k]))}\\times{_number(abs(b[k, col]))}"
            formula += f"{sign}{term}" if k == 0 else f"{sign}{term}"
        shown = sum(
            float(_number(a[row, k])) * float(_number(b[k, col]))
            for k in range(a.cols)
        )
        expected = float(_number(result[row, col]))
        relation = "\\approx" if abs(shown - expected) > 0.005 else "="
        formula += f"{relation}{_number(result[row, col])}"
        note = Text.from_tex(formula, Point(*at))
        note.name = f"{result.name}-{row}-{col}-calculation"
        note.height = 30.0
        note.color = Color(232 / 255, 237 / 255, 243 / 255)
        self.session.add_text(note)
        self.session.graph.set_vertex_attribute(note.guid, "grow", 500.0)
        self.session.graph.set_vertex_attribute(note.guid, "delay_ms", delay)
        for target, cells in (
            (a, [f"{row},{k}" for k in range(a.cols)]),
            (b, [f"{k},{col}" for k in range(a.cols)]),
        ):
            edge = (note.guid, target.guid)
            self.session.add_relationship(*edge, f"flash:{';'.join(cells)}")
            self.session.graph.set_edge_attribute(edge, "duration_ms", 550.0)
            self.session.graph.set_edge_attribute(edge, "delay_ms", delay)
        edge = (note.guid, result.guid)
        self.session.add_relationship(*edge, f"flash:{row},{col}")
        self.session.graph.set_edge_attribute(edge, "duration_ms", 800.0)
        self.session.graph.set_edge_attribute(edge, "delay_ms", delay + a.cols * 500.0)
        return note

    def sum(
        self,
        a: Matrix,
        b: Matrix,
        result: Matrix,
        row: int,
        col: int,
        at: tuple[float, float],
        delay: float = 0.0,
    ) -> Text:
        """Reveal one entrywise sum while flashing both operands and its result."""
        if a.rows != b.rows or a.cols != b.cols or result.rows != a.rows or result.cols != a.cols:
            raise ValueError("Matrix dimensions must match for addition")
        sign = "-" if b[row, col] < 0 else "+"
        second = _number(abs(b[row, col]))
        formula = f"{_number(a[row, col])}{sign}{second}={_number(result[row, col])}"
        note = Text.from_tex(formula, Point(*at))
        note.name = f"{result.name}-sum-{row}-{col}"
        note.height = 30.0
        note.color = Color(232 / 255, 237 / 255, 243 / 255)
        self.session.add_text(note)
        self.session.graph.set_vertex_attribute(note.guid, "grow", 500.0)
        self.session.graph.set_vertex_attribute(note.guid, "delay_ms", delay)
        for order, target in enumerate((a, b, result)):
            edge = (note.guid, target.guid)
            self.session.add_relationship(*edge, f"flash:{row},{col}")
            self.session.graph.set_edge_attribute(edge, "duration_ms", 600.0)
            self.session.graph.set_edge_attribute(edge, "delay_ms", delay + order * 450.0)
        return note


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


def _is_deck(data) -> bool:
    if not isinstance(data, list) or not data:
        return False
    if all(isinstance(item, Session) for item in data):
        return True
    return all(
        isinstance(topic, (list, tuple))
        and topic
        and all(isinstance(session, Session) for session in topic)
        for topic in data
    )


def _object_names(topics) -> dict[str, str]:
    names = {}
    for topic in topics:
        for session in topic:
            for item in session.lookup.values():
                names[item.guid] = item.name
            for item in session.component_lookup.values():
                names[item.guid] = item.name
            for vertex in session.graph.get_vertices():
                names.setdefault(vertex.name, vertex.name)
    return names


def _rgba(color) -> list[int]:
    return [round(channel * 255) for channel in (color.r, color.g, color.b, color.a)]


def _position(session: Session, guid: str) -> list[float]:
    item = session.lookup.get(guid)
    if isinstance(item, Point):
        return list(item)
    if isinstance(item, Line):
        return [(item[axis] + item[axis + 3]) * 0.5 for axis in range(3)]
    if isinstance(item, Polyline):
        return list(item.get_point(0))
    component = session.component_lookup.get(guid)
    if component is not None and component.type_name == "Text":
        return list(Text.from_component(component).at)
    raise TypeError("move supports points, lines, polylines and text")


def _xform_effect(session: Session, guid: str, names: dict[str, str]) -> dict:
    attributes = session.graph.vertices[guid].attributes
    transform = session.world_xform(guid)
    matrix = transform.m
    effect = {
        "name": names[guid],
        "duration_ms": attributes["xform_ms"],
        "delay_ms": attributes.get("delay_ms", 0.0),
    }
    identity = [1.0 if index in (0, 5, 10) else 0.0 for index in range(12)]
    if all(abs(matrix[index] - identity[index]) < 1e-9 for index in range(12)):
        at = _position(session, guid)
        effect["kind"] = "move"
        effect["to"] = [at[axis] + matrix[12 + axis] for axis in range(3)]
        return effect
    factor = sum(matrix[axis] * matrix[axis] for axis in range(3)) ** 0.5
    uniform = factor > 0.0 and abs(factor - 1.0) > 1e-9
    for index in range(12):
        expected = factor if index in (0, 5, 10) else 0.0
        if abs(matrix[index] - expected) > 1e-9:
            uniform = False
    if not uniform or abs(matrix[15] - 1.0) > 1e-9:
        raise ValueError("animate_xform supports translation and uniform scaling")
    effect["kind"] = "scale"
    effect["factor"] = factor
    effect["about"] = [matrix[12 + axis] / (1.0 - factor) for axis in range(3)]
    return effect


def _effects(session: Session, names: dict[str, str]) -> tuple[list[dict], list[str]]:
    effects = []
    highlight = []
    for vertex in session.graph.get_vertices():
        attributes = vertex.attributes
        if "xform_ms" in attributes:
            effects.append(_xform_effect(session, vertex.name, names))
        for kind in ("grow", "pulse", "cells"):
            if kind in attributes:
                effects.append(
                    {
                        "name": names[vertex.name],
                        "kind": kind,
                        "duration_ms": attributes[kind],
                        "delay_ms": attributes.get("delay_ms", 0.0),
                    }
                )
        if "highlight" in attributes:
            highlight.append(names[vertex.name])
    for first, second in session.graph.get_edges():
        edge = session.graph.edges[first][second]
        first, second = edge.v0, edge.v1
        if edge.attribute == "morph":
            effects.append(
                {
                    "name": names[second],
                    "from": names[first],
                    "kind": "morph",
                    "duration_ms": edge.attributes.get("duration_ms", 650.0),
                    "delay_ms": edge.attributes.get("delay_ms", 0.0),
                }
            )
        kind, _, selector = edge.attribute.partition(":")
        if kind == "flash" and selector:
            for order, cell in enumerate(selector.split(";")):
                target = names[second]
                if "," in cell:
                    target = f"{target}[{cell}]"
                else:
                    target = f"{target}/{cell}"
                effects.append(
                    {
                        "name": target,
                        "kind": "highlight",
                        "duration_ms": edge.attributes.get("duration_ms", 650.0),
                        "delay_ms": edge.attributes.get("delay_ms", 0.0) + order * 500.0,
                    }
                )
    return effects, highlight


def _links(session: Session, names: dict[str, str]) -> list[list[str]]:
    groups = []
    for first, second in session.graph.get_edges():
        edge = session.graph.edges[first][second]
        first, second = edge.v0, edge.v1
        kind, _, selector = edge.attribute.partition(":")
        if kind != "highlight":
            continue
        target = names[second]
        if selector:
            target = (
                f"{target}[{selector}]" if "," in selector else f"{target}/{selector}"
            )
        groups.append([names[first], target])
    return groups


def _spatial(session: Session, names: dict[str, str]) -> tuple[list[str], list[str]]:
    spatial = []
    non_pickable = []
    for guid, vertex in session.graph.vertices.items():
        if guid not in names:
            continue
        if vertex.attributes.get("spatial", 0.0) != 0.0:
            spatial.append(names[guid])
        if vertex.attributes.get("pickable", 1.0) == 0.0:
            non_pickable.append(names[guid])
    return spatial, non_pickable


def _gradients(session: Session) -> dict[str, dict]:
    gradients = {}
    for line in session.objects.lines:
        attributes = session.graph.vertices[line.guid].attributes
        if all(f"gradient_{channel}" in attributes for channel in "rgba"):
            gradients[line.name] = {
                "from": _rgba(line.linecolor),
                "to": [
                    round(attributes[f"gradient_{channel}"] * 255) for channel in "rgba"
                ],
            }
    return gradients


def _colored(session: Session) -> dict[str, list[int]]:
    colored = {}
    for component in session.objects.components:
        if component.type_name != "Matrix":
            continue
        attributes = session.graph.vertices[component.guid].attributes
        cells = [
            int(key[5:])
            for key, value in attributes.items()
            if key.startswith("color") and key[5:].isdigit() and value != 0.0
        ]
        if not cells:
            data = component.extra.get("data", [])
            cells = [index for index, value in enumerate(data) if value != 0.0]
        if any(cell >= 64 for cell in cells):
            continue
        colored[component.name] = sorted(cells)
    return colored


def _write_deck(data, directory: Path) -> None:
    topics = [data] if all(isinstance(item, Session) for item in data) else data
    names = _object_names(topics)
    document = {
        "name": "Session presentation",
        "width": 1600,
        "height": 760,
        "dark": True,
        "stroke_scale": 1.0,
        "head_scale": 0.7,
        "spatial_split": 0.56,
        "topics": [],
        "links": [],
        "spatial": [],
        "non_pickable": [],
        "gradients": {},
        "colored": {},
    }
    steps_directory = directory / "steps"
    steps_directory.mkdir(parents=True, exist_ok=True)
    for stale in steps_directory.glob("*.pb"):
        stale.unlink()
    for ti, topic in enumerate(topics):
        title = topic[0].name.partition("\n")[0]
        steps = []
        for si, session in enumerate(topic):
            name, _, caption = session.name.partition("\n")
            filename = f"steps/{ti:02d}_{si:02d}.pb"
            session.pb_dump(directory / filename)
            effects, highlight = _effects(session, names)
            spatial, non_pickable = _spatial(session, names)
            steps.append(
                {
                    "name": name,
                    "caption": caption,
                    "items": [{"file": filename, "name": name, "at": [0, 0, 0]}],
                    "effects": effects,
                    "highlight": highlight,
                }
            )
            document["spatial"].extend(spatial)
            document["non_pickable"].extend(non_pickable)
            document["gradients"].update(_gradients(session))
            document["colored"].update(_colored(session))
            for group in _links(session, names):
                if group not in document["links"]:
                    document["links"].append(group)
        preserve_spatial = any(
            vertex.attributes.get("preserve_spatial", 0.0) != 0.0
            for session in topic
            for vertex in session.graph.get_vertices()
        )
        document["topics"].append(
            {
                "title": title,
                "preserve_spatial": preserve_spatial,
                "steps": steps,
            }
        )
    document["spatial"] = list(dict.fromkeys(document["spatial"]))
    document["non_pickable"] = list(dict.fromkeys(document["non_pickable"]))
    path = directory / "presentation.yaml"
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _browser_url(data, edges, viewer) -> str:
    if _is_deck(data):
        raise NotImplementedError(
            "Presentations need a local Python kernel and viewer server"
        )
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
    """Serve session topics, one Session, or nodes and edges on 127.0.0.1."""
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
    if _is_deck(data):
        _write_deck(data, directory)
        query = urlencode({"presentation": f"sessions/{identifier}/presentation.yaml"})
    else:
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
    """Display session topics, one Session, or nodes and edges in a Jupyter iframe."""
    if sys.platform == "emscripten":
        from IPython.display import IFrame

        return IFrame(
            src=_browser_url(data, edges, viewer), width="100%", height=height
        )

    from IPython.display import IFrame

    running = serve(data, edges, directory=directory, viewer=viewer, port=port)
    return IFrame(src=running.url, width="100%", height=height)
