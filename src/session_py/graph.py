from __future__ import annotations
from typing import TYPE_CHECKING
import json
import uuid

if TYPE_CHECKING:
    from pathlib import Path

try:
    from .proto import graph_pb2

    _HAS_PROTOBUF = True
except ImportError:
    _HAS_PROTOBUF = False


# ═══════════════════════════════════════════════════════════════════════════
# Vertex
# ═══════════════════════════════════════════════════════════════════════════


class Vertex:
    """A graph vertex with a name, attribute string and integer index"""

    def __init__(self, name: str = "my_vertex", attribute: str = ""):
        """Construct from name and attribute"""
        self._guid = None
        self.name = name
        self.attribute = attribute
        self.index = -1

    def has_guid(self) -> bool:
        return self._guid is not None

    @property
    def guid(self) -> str:
        if self._guid is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def __jsondump__(self):
        return {
            "attribute": self.attribute,
            "guid": self.guid,
            "index": self.index,
            "name": self.name,
            "type": "Vertex",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        vertex = cls(name or data["name"], data["attribute"])
        vertex.guid = guid or data["guid"]
        vertex.index = data["index"]
        return vertex

    def __str__(self) -> str:
        return f"Vertex({self.guid}, {self.name}, {self.attribute}, {self.index})"

    def __repr__(self) -> str:
        return self.__str__()


# ═══════════════════════════════════════════════════════════════════════════
# Edge
# ═══════════════════════════════════════════════════════════════════════════


class Edge:
    """A graph edge connecting two vertices by name"""

    def __init__(self, v0: str = "", v1: str = "", attribute: str = ""):
        """Construct from endpoints and attribute"""
        self._guid = None
        self.name = "my_edge"
        self.v0 = v0
        self.v1 = v1
        self.attribute = attribute
        self.index = -1

    def has_guid(self) -> bool:
        return self._guid is not None

    @property
    def guid(self) -> str:
        if self._guid is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def vertices(self) -> tuple[str, str]:
        """The (v0, v1) tuple"""
        return (self.v0, self.v1)

    def connects(self, vertex_id: str) -> bool:
        """True if this edge touches the given vertex"""
        return self.v0 == vertex_id or self.v1 == vertex_id

    def other_vertex(self, vertex_id: str) -> str:
        """The other endpoint given one endpoint, empty if not connected"""
        if self.v0 == vertex_id:
            return self.v1
        if self.v1 == vertex_id:
            return self.v0
        return ""

    def __jsondump__(self):
        return {
            "attribute": self.attribute,
            "guid": self.guid,
            "index": self.index,
            "name": self.name,
            "type": "Edge",
            "v0": self.v0,
            "v1": self.v1,
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        edge = cls(data["v0"], data["v1"], data["attribute"])
        edge.name = name or data["name"]
        edge.guid = guid or data["guid"]
        edge.index = data["index"]
        return edge

    def __str__(self) -> str:
        return f"Edge({self.guid}, {self.name}, {self.v0}, {self.v1}, {self.attribute})"

    def __repr__(self) -> str:
        return self.__str__()


# ═══════════════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════════════


class Graph:
    """An undirected graph with string vertices and string attributes"""

    def __init__(self, name: str = "my_graph"):
        """Construct from name"""
        self._guid = None
        self.name = name
        self.vertices = {}
        self.edges = {}
        self.vertex_count = 0
        self.edge_count = 0

    def has_guid(self) -> bool:
        return self._guid is not None

    @property
    def guid(self) -> str:
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
        """True if a node with the given key exists"""
        return key in self.vertices

    def has_edge(self, key: tuple[str, str]) -> bool:
        """True if an edge between the given endpoints exists"""
        if key[0] not in self.edges:
            return False
        return key[1] in self.edges[key[0]]

    def add_node(self, key: str, attribute: str = "") -> str:
        """Add a node and return its key"""
        if self.has_node(key):
            return self.vertices[key].name
        vertex = Vertex(key, attribute)
        vertex.index = self.vertex_count
        self.vertices[key] = vertex
        self.vertex_count += 1
        return vertex.name

    def add_edge(self, u: str, v: str, attribute: str = "") -> tuple[str, str]:
        """Add an edge between u and v, creating missing nodes, and return (u, v)"""
        if not self.has_node(u):
            self.add_node(u)
        if not self.has_node(v):
            self.add_node(v)
        edge = Edge(u, v, attribute)
        edge.index = self.edge_count
        self.edges.setdefault(u, {})[v] = edge
        self.edges.setdefault(v, {})[u] = edge
        self.edge_count += 1
        return (u, v)

    def remove_node(self, key: str) -> None:
        """Remove a node and all its edges"""
        if not self.has_node(key):
            raise KeyError(f"Node {key} not in graph")
        if key in self.edges:
            for neighbor in self.edges[key]:
                self.edges[neighbor].pop(key, None)
            del self.edges[key]
        del self.vertices[key]
        self._reassign_indices()

    def remove_edge(self, edge: tuple[str, str]) -> None:
        """Remove an edge, keeping its nodes"""
        if not self.has_edge(edge):
            return
        u = edge[0]
        v = edge[1]
        del self.edges[u][v]
        del self.edges[v][u]
        self._reassign_edge_indices()

    def _reassign_indices(self) -> None:
        """Renumber vertex indices 0, 1, 2, ... keeping their relative order"""
        vertices = list(self.vertices.values())
        vertices.sort(key=lambda vertex: vertex.index)
        for i in range(len(vertices)):
            vertices[i].index = i
        self.vertex_count = len(vertices)

    def _reassign_edge_indices(self) -> None:
        """Renumber edge indices 0, 1, 2, ... keeping their relative order"""
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
        """All vertices in the graph"""
        result = []
        for vertex_name in sorted(self.vertices):
            result.append(self.vertices[vertex_name])
        return result

    def get_edges(self) -> list[tuple[str, str]]:
        """All edges in the graph as (u, v) tuples, each once"""
        result = []
        for u in sorted(self.edges):
            for v in sorted(self.edges[u]):
                if u < v:
                    result.append((u, v))
        return result

    def neighbors(self, node: str) -> list[str]:
        """All neighbors of a node"""
        if not self.has_node(node):
            raise KeyError(f"Node {node} not in graph")
        if node not in self.edges:
            return []
        return sorted(self.edges[node])

    def get_neighbors(self, node: str) -> list[str]:
        """Alias for neighbors()"""
        return self.neighbors(node)

    def edges_of(self, node: str) -> list[tuple[str, str, bool]]:
        """Incident edges as (other, attribute, forward); forward when node is the edge's v0"""
        result = []
        if node not in self.edges:
            return result
        for other in sorted(self.edges[node]):
            edge = self.edges[node][other]
            result.append((other, edge.attribute, edge.v0 == node))
        return result

    def number_of_vertices(self) -> int:
        """Number of vertices in the graph"""
        return len(self.vertices)

    def number_of_edges(self) -> int:
        """Number of edges in the graph"""
        count = 0
        for u in self.edges:
            for v in self.edges[u]:
                if u < v:
                    count += 1
        return count

    def clear(self) -> None:
        """Remove all vertices and edges"""
        self.vertices.clear()
        self.edges.clear()
        self.vertex_count = 0
        self.edge_count = 0

    def node_attribute(self, node: str, value: str = "") -> str:
        """Get or set node attribute (sets if value is non-empty)"""
        if not self.has_node(node):
            raise KeyError(f"Node {node} not in graph")
        if value == "":
            return self.vertices[node].attribute
        self.vertices[node].attribute = value
        return value

    def edge_attribute(self, u: str, v: str, value: str = "") -> str:
        """Get or set edge attribute (sets if value is non-empty)"""
        if not self.has_edge((u, v)):
            raise KeyError(f"Edge ({u}, {v}) not in graph")
        if value == "":
            return self.edges[u][v].attribute
        self.edges[u][v].attribute = value
        self.edges[v][u].attribute = value
        return value

    # ═══════════════════════════════════════════════════════════════════════════
    # Algorithms
    # ═══════════════════════════════════════════════════════════════════════════

    def bfs(self, start: str) -> list[str]:
        """Breadth-first order from start"""
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
        """Depth-first order from start"""
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
        """Connected components as sorted node name lists"""
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
        """True if the graph has at most one connected component"""
        return len(self.connected_components()) <= 1

    def number_connected_components(self) -> int:
        """Number of connected components"""
        return len(self.connected_components())

    def shortest_path(self, u: str, v: str) -> list[str]:
        """Shortest path between u and v, empty if disconnected"""
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
        """Length of the shortest path between u and v, -1 if disconnected"""
        path = self.shortest_path(u, v)
        if not path:
            return -1
        return len(path) - 1

    def has_cycle(self) -> bool:
        """True if the graph contains a cycle"""
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
        """A basis of fundamental cycles"""
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
                u = stack[-1][0]
                p = stack[-1][1]
                nbrs = stack[-1][2]
                if stack[-1][3] >= len(nbrs):
                    stack.pop()
                    continue
                v = nbrs[stack[-1][3]]
                stack[-1][3] += 1
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

    def __jsondump__(self):
        vertices_json = []
        for vertex_name in sorted(self.vertices):
            vertices_json.append(self.vertices[vertex_name].__jsondump__())
        edges_json = []
        for u in sorted(self.edges):
            for v in sorted(self.edges[u]):
                if u < v:
                    edges_json.append(self.edges[u][v].__jsondump__())
        return {
            "edge_count": self.edge_count,
            "edges": edges_json,
            "guid": self.guid,
            "name": self.name,
            "type": "Graph",
            "vertex_count": self.vertex_count,
            "vertices": vertices_json,
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        graph = cls(name or data["name"])
        graph.guid = guid or data["guid"]
        graph.vertex_count = data["vertex_count"]
        graph.edge_count = data["edge_count"]
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
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Graph":
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: str | Path) -> None:
        with open(filepath, "w") as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: str | Path) -> "Graph":
        with open(filepath) as f:
            return cls.__jsonload__(json.load(f))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def pb_dumps(self) -> bytes:
        if not _HAS_PROTOBUF:
            raise ImportError("protobuf not available")
        proto = graph_pb2.Graph()
        proto.name = self.name
        if self.has_guid():
            proto.guid = self.guid
        proto.vertex_count = self.vertex_count
        proto.edge_count = self.edge_count
        for vertex_name in sorted(self.vertices):
            vertex = self.vertices[vertex_name]
            v = proto.vertices[vertex_name]
            v.name = vertex.name
            if vertex.has_guid():
                v.guid = vertex.guid
            v.attribute = vertex.attribute
            v.index = vertex.index
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
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Graph":
        if not _HAS_PROTOBUF:
            raise ImportError("protobuf not available")
        proto = graph_pb2.Graph()
        proto.ParseFromString(data)
        graph = cls(proto.name)
        if proto.guid:
            graph.guid = proto.guid
        graph.vertex_count = proto.vertex_count
        graph.edge_count = proto.edge_count
        for vertex_name in proto.vertices:
            v = proto.vertices[vertex_name]
            vertex = Vertex(v.name, v.attribute)
            vertex.guid = v.guid
            vertex.index = v.index
            graph.vertices[vertex_name] = vertex
        for e in proto.edges:
            edge = Edge(e.v0, e.v1, e.attribute)
            edge.name = e.name
            edge.guid = e.guid
            edge.index = e.index
            graph.edges.setdefault(e.v0, {})[e.v1] = edge
            graph.edges.setdefault(e.v1, {})[e.v0] = edge
        return graph

    def pb_dump(self, filepath: str | Path) -> None:
        data = self.pb_dumps()
        with open(filepath, "wb") as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath: str | Path) -> "Graph":
        with open(filepath, "rb") as f:
            return cls.pb_loads(f.read())

    def __str__(self) -> str:
        return (
            f"Graph({self.guid}, {self.name}, {self.vertex_count}, {self.edge_count})"
        )

    def __repr__(self) -> str:
        return self.__str__()
