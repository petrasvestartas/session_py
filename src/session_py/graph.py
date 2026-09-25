from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable
import json
import uuid

if TYPE_CHECKING:
    from pathlib import Path

    from .proto import graph_pb2


# ═══════════════════════════════════════════════════════════════════════════
# Vertex
# ═══════════════════════════════════════════════════════════════════════════


class Vertex:
    """A graph vertex with a name, attribute string and integer index."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, name: str = "my_vertex", attribute: str = ""):
        """Construct from name and attribute."""

        self._guid = None  # Lazily minted GUID.
        self.name = name  # Vertex name, also the key in Graph.vertices.
        self.attribute = attribute  # Vertex attribute data as string.
        self.attributes = {}  # Name -> value, overriding the graph defaults.
        self.index = -1  # Integer index of the vertex, assigned by Graph.

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def has_guid(self) -> bool:
        """Return whether the lazy GUID has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the GUID, creating it on first access."""

        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON-ready dict."""

        return {
            "attribute": self.attribute,
            "attributes": self.attributes,
            "guid": self.guid,
            "index": self.index,
            "name": self.name,
            "type": "Vertex",
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Vertex:
        """Deserialize from a JSON dict."""

        vertex = cls(name or data["name"], data["attribute"])
        vertex.guid = guid or data["guid"]
        vertex.index = data["index"]

        if "attributes" in data:
            vertex.attributes = dict(data["attributes"])

        return vertex

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "Vertex(guid, name, attribute, index)"."""
        return f"Vertex({self.guid}, {self.name}, {self.attribute}, {self.index})"

    def __repr__(self) -> str:
        """Return "Vertex(guid, name, attribute, index)"."""
        return self.__str__()


# ═══════════════════════════════════════════════════════════════════════════
# Edge
# ═══════════════════════════════════════════════════════════════════════════


class Edge:
    """A graph edge connecting two vertices by name."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, v0: str = "", v1: str = "", attribute: str = ""):
        """Construct from endpoints and attribute."""

        self._guid = None  # Lazily minted GUID.
        self.name = "my_edge"  # Edge name.
        self.v0 = v0  # First vertex name.
        self.v1 = v1  # Second vertex name.
        self.attribute = attribute  # Edge attribute data as string.
        self.attributes = {}  # Name -> value, overriding the graph defaults.
        self.index = -1  # Integer index of the edge, assigned by Graph.

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def has_guid(self) -> bool:
        """Return whether the lazy GUID has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the GUID, creating it on first access."""

        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def vertices(self) -> tuple[str, str]:
        """Return the (v0, v1) tuple."""
        return (self.v0, self.v1)

    def connects(self, vertex_id: str) -> bool:
        """Return whether this edge touches the given vertex."""
        return self.v0 == vertex_id or self.v1 == vertex_id

    def other_vertex(self, vertex_id: str) -> str:
        """Return the other endpoint given one endpoint, empty if not connected."""

        if self.v0 == vertex_id:
            return self.v1

        if self.v1 == vertex_id:
            return self.v0

        return ""

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON-ready dict."""

        return {
            "attribute": self.attribute,
            "attributes": self.attributes,
            "guid": self.guid,
            "index": self.index,
            "name": self.name,
            "type": "Edge",
            "v0": self.v0,
            "v1": self.v1,
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Edge:
        """Deserialize from a JSON dict."""

        edge = cls(data["v0"], data["v1"], data["attribute"])
        edge.name = name or data["name"]
        edge.guid = guid or data["guid"]
        edge.index = data["index"]

        if "attributes" in data:
            edge.attributes = dict(data["attributes"])

        return edge

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "Edge(guid, name, v0, v1, attribute)"."""
        return f"Edge({self.guid}, {self.name}, {self.v0}, {self.v1}, {self.attribute})"

    def __repr__(self) -> str:
        """Return "Edge(guid, name, v0, v1, attribute)"."""
        return self.__str__()


# ═══════════════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════════════


class Graph:
    """An undirected graph with string vertices, string labels and double attributes."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, name: str = "my_graph"):
        """Construct from name."""

        self._guid = None  # Lazily minted GUID.
        self.vertices = {}  # name -> Vertex.
        self.name = name  # Graph name.
        self.vertex_count = 0  # Next available vertex index.
        self.edge_count = 0  # Next available edge index.
        self.edges = {}  # node_name -> {neighbor_name -> Edge}, every edge stored in both directions.
        self.default_vertex_attributes = {}  # Vertex attribute defaults.
        self.default_edge_attributes = {}  # Edge attribute defaults.

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def has_guid(self) -> bool:
        """Return whether the lazy GUID has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the GUID, creating it on first access."""

        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    # ═══════════════════════════════════════════════════════════════════════════
    # Details
    # ═══════════════════════════════════════════════════════════════════════════
    def has_node(self, key: str) -> bool:
        """Return whether a node with the given key exists."""
        return key in self.vertices

    def has_edge(self, key: tuple[str, str]) -> bool:
        """Return whether an edge between the given endpoints exists."""

        if key[0] not in self.edges:
            return False

        return key[1] in self.edges[key[0]]

    def add_node(self, key: str, attribute: str = "") -> str:
        """Add a node and return its key."""

        if self.has_node(key):
            return self.vertices[key].name

        vertex = Vertex(key, attribute)
        vertex.index = self.vertex_count

        self.vertices[key] = vertex
        self.vertex_count += 1

        return vertex.name

    def add_edge(self, u: str, v: str, attribute: str = "") -> tuple[str, str]:
        """Add an edge between u and v, creating missing nodes, and return (u, v)."""

        if not self.has_node(u):
            self.add_node(u)

        if not self.has_node(v):
            self.add_node(v)

        if self.has_edge((u, v)):
            self.edges[u][v].attribute = attribute
            self.edges[v][u] = self.edges[u][v]

            return (u, v)

        edge = Edge(u, v, attribute)
        edge.index = self.edge_count

        self.edges.setdefault(u, {})[v] = edge
        self.edges.setdefault(v, {})[u] = edge
        self.edge_count += 1

        return (u, v)

    def remove_node(self, key: str) -> None:
        """Remove a node and all its edges."""

        if not self.has_node(key):
            raise KeyError(f"Node {key} not in graph")

        if key in self.edges:
            for neighbor in self.edges[key]:
                self.edges[neighbor].pop(key, None)

            del self.edges[key]

        del self.vertices[key]
        self._reassign_indices()
        self._reassign_edge_indices()

    def remove_edge(self, edge: tuple[str, str]) -> None:
        """Remove an edge, keeping its nodes."""

        if not self.has_edge(edge):
            return

        u = edge[0]
        v = edge[1]

        del self.edges[u][v]
        del self.edges[v][u]
        self._reassign_edge_indices()

    def _reassign_indices(self) -> None:
        """Renumber vertex indices 0, 1, 2, ... keeping their relative order."""

        vertices = []

        for vertex_name in self.vertices:
            vertices.append((self.vertices[vertex_name].index, vertex_name))

        vertices.sort()

        for i in range(len(vertices)):
            self.vertices[vertices[i][1]].index = i

        self.vertex_count = len(vertices)

    def _reassign_edge_indices(self) -> None:
        """Renumber edge indices 0, 1, 2, ... keeping their relative order."""

        edges = []

        for u in self.edges:
            for v in self.edges[u]:
                if u < v:
                    edges.append((self.edges[u][v].index, u, v))

        edges.sort()

        for i in range(len(edges)):
            u = edges[i][1]
            v = edges[i][2]

            self.edges[u][v].index = i
            self.edges[v][u].index = i

        self.edge_count = len(edges)

    def get_vertices(self) -> list[Vertex]:
        """Return all vertices in the graph."""

        result = []

        for vertex_name in sorted(self.vertices):
            result.append(self.vertices[vertex_name])

        return result

    def get_edges(self) -> list[tuple[str, str]]:
        """Return all edges in the graph as (u, v) tuples, each once."""

        result = []

        for u in sorted(self.edges):
            for v in sorted(self.edges[u]):
                if u < v:
                    result.append((u, v))

        return result

    def neighbors(self, node: str) -> list[str]:
        """Return all neighbors of a node."""

        if not self.has_node(node):
            raise KeyError(f"Node {node} not in graph")

        if node not in self.edges:
            return []

        return sorted(self.edges[node])

    def edges_of(self, node: str) -> list[tuple[str, str, bool]]:
        """Return incident edges as (other, attribute, forward); forward when node is the edge's v0."""

        result = []

        if node not in self.edges:
            return result

        for other in sorted(self.edges[node]):
            edge = self.edges[node][other]
            result.append((other, edge.attribute, edge.v0 == node))

        return result

    def number_of_vertices(self) -> int:
        """Return the number of vertices in the graph."""
        return len(self.vertices)

    def number_of_edges(self) -> int:
        """Return the number of edges in the graph."""

        count = 0

        for u in self.edges:
            for v in self.edges[u]:
                if u < v:
                    count += 1

        return count

    def clear(self) -> None:
        """Remove all vertices and edges."""

        self.vertices.clear()
        self.edges.clear()
        self.vertex_count = 0
        self.edge_count = 0

    def node_label(self, node: str, value: str = "") -> str:
        """Get or set a node label (sets if value is non-empty)."""

        if not self.has_node(node):
            raise KeyError(f"Node {node} not in graph")

        if value == "":
            return self.vertices[node].attribute

        self.vertices[node].attribute = value

        return value

    def edge_label(self, u: str, v: str, value: str = "") -> str:
        """Get or set an edge label (sets if value is non-empty)."""

        if not self.has_edge((u, v)):
            raise KeyError(f"Edge ({u}, {v}) not in graph")

        if value == "":
            return self.edges[u][v].attribute

        self.edges[u][v].attribute = value
        self.edges[v][u].attribute = value

        return value

    # ═══════════════════════════════════════════════════════════════════════════
    # Attribute API
    # ═══════════════════════════════════════════════════════════════════════════
    def update_default_vertex_attributes(self, attrs: dict[str, float]) -> None:
        """Merge attrs into the default vertex attributes."""

        for name, value in attrs.items():
            self.default_vertex_attributes[name] = value

    def update_default_edge_attributes(self, attrs: dict[str, float]) -> None:
        """Merge attrs into the default edge attributes."""

        for name, value in attrs.items():
            self.default_edge_attributes[name] = value

    def vertex_attribute(self, key: str, name: str) -> float | None:
        """Return the attribute of a vertex, falling back to the default; None when neither exists."""

        vertex = self.vertices.get(key)

        if vertex is None:
            return None

        if name in vertex.attributes:
            return vertex.attributes[name]

        return self.default_vertex_attributes.get(name)

    def set_vertex_attribute(self, key: str, name: str, value: float) -> None:
        """Store an attribute on a vertex."""

        if self.has_node(key):
            self.vertices[key].attributes[name] = value

    def edge_attribute(self, edge: tuple[str, str], name: str) -> float | None:
        """Return the attribute of an edge, falling back to the default; None when neither exists."""

        if not self.has_edge(edge):
            return None

        stored = self.edges[edge[0]][edge[1]]

        if name in stored.attributes:
            return stored.attributes[name]

        return self.default_edge_attributes.get(name)

    def set_edge_attribute(
        self, edge: tuple[str, str], name: str, value: float
    ) -> None:
        """Store an attribute on an edge, in both stored directions."""

        if not self.has_edge(edge):
            return

        u = edge[0]
        v = edge[1]

        self.edges[u][v].attributes[name] = value
        self.edges[v][u].attributes[name] = value

    def vertices_where(self, conditions: dict[str, float]) -> list[str]:
        """Return the vertices whose attributes match every (name, value) condition."""

        result = []

        for vertex_name in sorted(self.vertices):
            matched = True

            for name, value in conditions.items():
                if self.vertex_attribute(vertex_name, name) != value:
                    matched = False

            if matched:
                result.append(vertex_name)

        return result

    def edges_where(self, conditions: dict[str, float]) -> list[tuple[str, str]]:
        """Return the edges whose attributes match every (name, value) condition."""

        result = []

        for edge in self.get_edges():
            matched = True

            for name, value in conditions.items():
                if self.edge_attribute(edge, name) != value:
                    matched = False

            if matched:
                result.append(edge)

        return result

    def vertices_where_predicate(
        self, pred: Callable[[str, dict[str, float]], bool]
    ) -> list[str]:
        """Return the vertices for which pred(key, attributes) is true."""

        result = []

        for vertex_name in sorted(self.vertices):
            attributes = dict(self.default_vertex_attributes)

            for name, value in self.vertices[vertex_name].attributes.items():
                attributes[name] = value

            if pred(vertex_name, attributes):
                result.append(vertex_name)

        return result

    def edges_where_predicate(
        self, pred: Callable[[tuple[str, str], dict[str, float]], bool]
    ) -> list[tuple[str, str]]:
        """Return the edges for which pred(edge, attributes) is true."""

        result = []

        for edge in self.get_edges():
            stored = self.edges[edge[0]][edge[1]]
            attributes = dict(self.default_edge_attributes)

            for name, value in stored.attributes.items():
                attributes[name] = value

            if pred(edge, attributes):
                result.append(edge)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Algorithms
    # ═══════════════════════════════════════════════════════════════════════════
    def bfs(self, start: str) -> list[str]:
        """Return the breadth-first order from start."""

        result = []

        if not self.has_node(start):
            return result

        visited = set()
        queue = [start]
        visited.add(start)

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in self.neighbors(node):
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                queue.append(neighbor)

        return result

    def dfs(self, start: str) -> list[str]:
        """Return the depth-first order from start."""

        result = []

        if not self.has_node(start):
            return result

        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()

            if node in visited:
                continue

            visited.add(node)
            result.append(node)

            nbrs = self.neighbors(node)

            for i in range(len(nbrs) - 1, -1, -1):
                if nbrs[i] not in visited:
                    stack.append(nbrs[i])

        return result

    def connected_components(self) -> list[list[str]]:
        """Return the connected components as sorted node name lists."""

        visited = set()
        components = []

        for vertex_name in sorted(self.vertices):
            if vertex_name in visited:
                continue

            component = self.bfs(vertex_name)

            for node in component:
                visited.add(node)

            component.sort()
            components.append(component)

        return components

    def is_connected(self) -> bool:
        """Return whether the graph has at most one connected component."""
        return len(self.connected_components()) <= 1

    def number_connected_components(self) -> int:
        """Return the number of connected components."""
        return len(self.connected_components())

    def shortest_path(self, u: str, v: str) -> list[str]:
        """Return the shortest path between u and v, empty if disconnected."""

        path = []

        if not self.has_node(u) or not self.has_node(v):
            return path

        if u == v:
            return [u]

        parent = {u: ""}

        queue = [u]

        while queue:
            node = queue.pop(0)

            for neighbor in self.neighbors(node):
                if neighbor in parent:
                    continue

                parent[neighbor] = node

                if neighbor == v:
                    current = v

                    while current != u:
                        path.append(current)
                        current = parent[current]

                    path.append(u)
                    path.reverse()

                    return path

                queue.append(neighbor)

        return path

    def shortest_path_length(self, u: str, v: str) -> int:
        """Return the length of the shortest path between u and v, -1 if disconnected."""

        path = self.shortest_path(u, v)

        if not path:
            return -1

        return len(path) - 1

    def has_cycle(self) -> bool:
        """Return whether the graph contains a cycle."""

        visited = set()

        for vertex_name in sorted(self.vertices):
            if vertex_name in visited:
                continue

            parent = {vertex_name: ""}

            queue = [vertex_name]
            visited.add(vertex_name)

            while queue:
                node = queue.pop(0)

                for neighbor in self.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif parent[node] != neighbor:
                        return True

        return False

    def cycle_basis(self) -> list[list[str]]:
        """Return a basis of fundamental cycles."""

        result = []
        order = {}
        parent = {}
        timer = 0

        for vertex_name in sorted(self.vertices):
            if vertex_name in order:
                continue

            parent[vertex_name] = ""
            order[vertex_name] = timer
            timer += 1

            stack = [[vertex_name, "", self.neighbors(vertex_name), 0]]

            while stack:
                frame = stack[-1]
                u = frame[0]
                p = frame[1]

                if frame[3] >= len(frame[2]):
                    stack.pop()
                    continue

                v = frame[2][frame[3]]
                frame[3] += 1

                if v not in order:
                    parent[v] = u
                    order[v] = timer
                    timer += 1
                    stack.append([v, u, self.neighbors(v), 0])
                elif v != p and order[v] < order[u]:
                    cycle = []
                    node = u

                    while node != v:
                        cycle.append(node)
                        node = parent[node]

                    cycle.append(v)
                    result.append(cycle)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON-ready dict."""

        vertices_json = []

        for vertex_name in sorted(self.vertices):
            vertices_json.append(self.vertices[vertex_name].__jsondump__())

        edges_json = []

        for u in sorted(self.edges):
            for v in sorted(self.edges[u]):
                if u < v:
                    edges_json.append(self.edges[u][v].__jsondump__())

        return {
            "default_edge_attributes": self.default_edge_attributes,
            "default_vertex_attributes": self.default_vertex_attributes,
            "edge_count": self.edge_count,
            "edges": edges_json,
            "guid": self.guid,
            "name": self.name,
            "type": "Graph",
            "vertex_count": self.vertex_count,
            "vertices": vertices_json,
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Graph:
        """Deserialize from a JSON dict."""

        graph = cls(name or data["name"])
        graph.guid = guid or data["guid"]
        graph.vertex_count = data["vertex_count"]
        graph.edge_count = data["edge_count"]

        if "default_edge_attributes" in data:
            graph.default_edge_attributes = dict(data["default_edge_attributes"])

        if "default_vertex_attributes" in data:
            graph.default_vertex_attributes = dict(data["default_vertex_attributes"])

        for vertex_data in data["vertices"]:
            vertex = vertex_data

            if isinstance(vertex_data, dict):
                vertex = Vertex.__jsonload__(vertex_data)

            graph.vertices[vertex.name] = vertex

        for edge_data in data["edges"]:
            edge = edge_data

            if isinstance(edge_data, dict):
                edge = Edge.__jsonload__(edge_data)

            graph.edges.setdefault(edge.v0, {})[edge.v1] = edge
            graph.edges.setdefault(edge.v1, {})[edge.v0] = edge

        return graph

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> Graph:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write JSON to a file."""

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=2)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Graph:
        """Read JSON from a file."""

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> graph_pb2.Graph:
        """Convert to the protobuf message, each edge once."""

        from .proto import graph_pb2

        proto = graph_pb2.Graph()
        proto.name = self.name

        if self.has_guid():
            proto.guid = self.guid

        proto.vertex_count = self.vertex_count
        proto.edge_count = self.edge_count

        for name, value in self.default_vertex_attributes.items():
            proto.default_vertex_attributes[name] = value

        for name, value in self.default_edge_attributes.items():
            proto.default_edge_attributes[name] = value

        for vertex_name in sorted(self.vertices):
            vertex = self.vertices[vertex_name]
            v = proto.vertices[vertex_name]
            v.name = vertex.name

            if vertex.has_guid():
                v.guid = vertex.guid

            v.attribute = vertex.attribute
            v.index = vertex.index

            for name, value in vertex.attributes.items():
                v.attributes[name] = value

        for u in sorted(self.edges):
            for v in sorted(self.edges[u]):
                if u > v:
                    continue

                edge = self.edges[u][v]
                e = proto.edges.add()

                if edge.has_guid():
                    e.guid = edge.guid

                e.name = edge.name
                e.v0 = edge.v0
                e.v1 = edge.v1
                e.attribute = edge.attribute
                e.index = edge.index

                for name, value in edge.attributes.items():
                    e.attributes[name] = value

        return proto

    @classmethod
    def from_proto(cls, proto: graph_pb2.Graph) -> Graph:
        """Construct from the protobuf message."""

        graph = cls(proto.name)

        if proto.guid:
            graph.guid = proto.guid

        graph.vertex_count = proto.vertex_count
        graph.edge_count = proto.edge_count

        for name, value in proto.default_vertex_attributes.items():
            graph.default_vertex_attributes[name] = value

        for name, value in proto.default_edge_attributes.items():
            graph.default_edge_attributes[name] = value

        for vertex_name in proto.vertices:
            v = proto.vertices[vertex_name]
            vertex = Vertex(v.name, v.attribute)
            vertex.guid = v.guid
            vertex.index = v.index

            for name, value in v.attributes.items():
                vertex.attributes[name] = value

            graph.vertices[vertex_name] = vertex

        for e in proto.edges:
            edge = Edge(e.v0, e.v1, e.attribute)
            edge.name = e.name
            edge.guid = e.guid
            edge.index = e.index

            for name, value in e.attributes.items():
                edge.attributes[name] = value

            graph.edges.setdefault(e.v0, {})[e.v1] = edge
            graph.edges.setdefault(e.v1, {})[e.v0] = edge

        return graph

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Graph:
        """Deserialize from protobuf bytes."""

        from .proto import graph_pb2

        proto = graph_pb2.Graph()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write protobuf bytes to a file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Graph:
        """Read protobuf bytes from a file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "<Graph with V vertices, E edges: name>"."""
        return f"<Graph with {self.vertex_count} vertices, {self.edge_count} edges: {self.name}>"

    def __repr__(self) -> str:
        """Return "Graph(guid, name, vertex_count, edge_count)"."""
        return (
            f"Graph({self.guid}, {self.name}, {self.vertex_count}, {self.edge_count})"
        )
