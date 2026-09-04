from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


# ═══════════════════════════════════════════════════════════════════════════
# Vertex
# ═══════════════════════════════════════════════════════════════════════════


@MINI_TEST("Vertex", "Constructor")
def test_vertex_constructor():
    from session_py import Vertex

    # Default constructor
    v0 = Vertex()

    # Constructor with name + attribute
    v = Vertex("v_named", "attr")

    MINI_CHECK(v0.name == "my_vertex")
    MINI_CHECK(v0.attribute == "")
    MINI_CHECK(v0.guid)
    MINI_CHECK(v.name == "v_named")
    MINI_CHECK(v.attribute == "attr")


@MINI_TEST("Vertex", "Json Roundtrip")
def test_vertex_json_roundtrip():
    from session_py import Vertex
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = Vertex("v0", "test_attribute")

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_vertex.json"
    file_json_dump(original, fname)
    loaded = file_json_load(fname)

    MINI_CHECK(loaded.name == original.name)
    MINI_CHECK(loaded.attribute == original.attribute)


# ═══════════════════════════════════════════════════════════════════════════
# Edge
# ═══════════════════════════════════════════════════════════════════════════


@MINI_TEST("Edge", "Constructor")
def test_edge_constructor():
    from session_py import Edge

    # Constructor with v0/v1/attribute
    e = Edge("a", "b", "attr")

    MINI_CHECK(e.v0 == "a")
    MINI_CHECK(e.v1 == "b")
    MINI_CHECK(e.attribute == "attr")
    MINI_CHECK(e.guid)


@MINI_TEST("Edge", "Json Roundtrip")
def test_edge_json_roundtrip():
    from session_py import Edge
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = Edge("v0", "v1", "test_edge_attr")

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_edge.json"
    file_json_dump(original, fname)
    loaded = file_json_load(fname)

    MINI_CHECK(loaded.name == original.name)
    MINI_CHECK(loaded.v0 == original.v0)
    MINI_CHECK(loaded.v1 == original.v1)


@MINI_TEST("Edge", "Vertices")
def test_edge_vertices():
    from session_py import Edge

    e = Edge("a", "b")
    u, v = e.vertices

    MINI_CHECK(u == "a" and v == "b")


@MINI_TEST("Edge", "Connects")
def test_edge_connects():
    from session_py import Edge

    e = Edge("a", "b")

    MINI_CHECK(e.connects("a"))
    MINI_CHECK(e.connects("b"))
    MINI_CHECK(not e.connects("c"))


@MINI_TEST("Edge", "Other Vertex")
def test_edge_other_vertex():
    from session_py import Edge

    e = Edge("a", "b")

    MINI_CHECK(e.other_vertex("a") == "b")
    MINI_CHECK(e.other_vertex("b") == "a")


# ═══════════════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════════════


@MINI_TEST("Graph", "Constructor")
def test_graph_constructor():
    from session_py import Graph

    # Default constructor
    g0 = Graph()

    # Constructor with name
    g = Graph("my_named_graph")

    MINI_CHECK(g0.name == "my_graph")
    MINI_CHECK(g0.guid)
    MINI_CHECK(g.name == "my_named_graph")


@MINI_TEST("Graph", "Json Roundtrip")
def test_graph_json_roundtrip():
    from session_py import Graph
    from pathlib import Path

    original = Graph("test_graph")
    original.add_node("node1", "Node 1")
    original.add_node("node2", "Node 2")
    original.add_edge("node1", "node2", "edge1")

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   file_json_dumps()    │ str          │ to JSON string
    #   file_json_loads(s)   │ str          │ from JSON string
    #   file_json_dump(path) │ file         │ write to file
    #   file_json_load(path) │ file         │ read from file

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_graph.json"
    original.file_json_dump(fname)
    loaded = Graph.file_json_load(fname)

    MINI_CHECK(loaded.number_of_vertices() == 2)
    MINI_CHECK(loaded.number_of_edges() == 1)
    MINI_CHECK(loaded.has_edge(("node1", "node2")))


@MINI_TEST("Graph", "Protobuf Roundtrip")
def test_graph_protobuf_roundtrip():
    from session_py import Graph
    from pathlib import Path

    original = Graph("test_graph")
    original.add_node("node1", "Node 1")
    original.add_node("node2", "Node 2")
    original.add_edge("node1", "node2", "edge1")

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_graph.bin"
    original.pb_dump(path)
    loaded = Graph.pb_load(path)

    MINI_CHECK(loaded.number_of_vertices() == 2)
    MINI_CHECK(loaded.number_of_edges() == 1)
    MINI_CHECK(loaded.has_edge(("node1", "node2")))


@MINI_TEST("Graph", "Has Node")
def test_graph_has_node():
    from session_py import Graph

    g = Graph("g")
    g.add_node("a")

    MINI_CHECK(g.has_node("a"))
    MINI_CHECK(not g.has_node("missing"))


@MINI_TEST("Graph", "Has Edge")
def test_graph_has_edge():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")

    MINI_CHECK(g.has_edge(("a", "b")))
    MINI_CHECK(not g.has_edge(("a", "c")))


@MINI_TEST("Graph", "Has Guid")
def test_graph_has_guid():
    from session_py.graph import Edge
    from session_py.graph import Vertex

    # A guid is lazily minted, so ASKING for one creates it. The writers used to ask for every
    # vertex and edge, which minted 34,592 UUIDs for one drawing sheet and wrote 1.3 MB of them
    # into a file whose reader discards them. has_guid() answers without minting, so a thing
    # nobody names never pays for one.
    v = Vertex("a")
    e = Edge("a", "b")

    MINI_CHECK(not v.has_guid())  # nobody has asked
    MINI_CHECK(not e.has_guid())

    minted = v.guid
    MINI_CHECK(minted != "")
    MINI_CHECK(v.has_guid())  # asking created it
    MINI_CHECK(v.guid == minted)  # and it is stable


@MINI_TEST("Graph", "Add Node")
def test_graph_add_node():
    from session_py import Graph

    g = Graph("g")
    key = g.add_node("a")

    MINI_CHECK(key == "a")
    MINI_CHECK(g.has_node("a"))
    MINI_CHECK(g.number_of_vertices() == 1)


@MINI_TEST("Graph", "Add Edge")
def test_graph_add_edge():
    from session_py import Graph

    g = Graph("g")
    edge = g.add_edge("a", "b")
    u, v = edge

    MINI_CHECK(u == "a" and v == "b")
    MINI_CHECK(g.number_of_edges() == 1)


@MINI_TEST("Graph", "Remove Node")
def test_graph_remove_node():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.remove_node("a")

    MINI_CHECK(not g.has_node("a"))
    MINI_CHECK(g.number_of_edges() == 0)


@MINI_TEST("Graph", "Remove Edge")
def test_graph_remove_edge():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.remove_edge(("a", "b"))

    MINI_CHECK(g.number_of_edges() == 0)
    MINI_CHECK(g.has_node("a"))
    MINI_CHECK(g.has_node("b"))


@MINI_TEST("Graph", "Get Vertices")
def test_graph_get_vertices():
    from session_py import Graph

    g = Graph("g")
    g.add_node("a")
    g.add_node("b")

    verts = g.get_vertices()

    MINI_CHECK(len(verts) == 2)


@MINI_TEST("Graph", "Get Edges")
def test_graph_get_edges():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")

    edges = list(g.get_edges())

    MINI_CHECK(len(edges) == 2)


@MINI_TEST("Graph", "Neighbors")
def test_graph_neighbors():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("a", "c")

    neigh = list(g.neighbors("a"))

    MINI_CHECK(len(neigh) == 2)


@MINI_TEST("Graph", "Get Neighbors")
def test_graph_get_neighbors():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("a", "c")

    neigh = g.get_neighbors("a")

    MINI_CHECK(len(neigh) == 2)


@MINI_TEST("Graph", "Number Of Vertices")
def test_graph_number_of_vertices():
    from session_py import Graph

    g = Graph("g")
    g.add_node("a")
    g.add_node("b")
    g.add_node("c")

    MINI_CHECK(g.number_of_vertices() == 3)


@MINI_TEST("Graph", "Number Of Edges")
def test_graph_number_of_edges():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")

    MINI_CHECK(g.number_of_edges() == 2)


@MINI_TEST("Graph", "Clear")
def test_graph_clear():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.clear()

    MINI_CHECK(g.number_of_vertices() == 0)
    MINI_CHECK(g.number_of_edges() == 0)


@MINI_TEST("Graph", "Node Attribute")
def test_graph_node_attribute():
    from session_py import Graph

    g = Graph("g")
    g.add_node("a", "initial")
    g.node_attribute("a", "updated")

    MINI_CHECK(g.node_attribute("a") == "updated")


@MINI_TEST("Graph", "Edge Attribute")
def test_graph_edge_attribute():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b", "initial")
    g.edge_attribute("a", "b", "updated")

    MINI_CHECK(g.edge_attribute("a", "b") == "updated")


@MINI_TEST("Graph", "Bfs")
def test_graph_bfs():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_edge("b", "d")
    g.add_edge("e", "f")
    result = g.bfs("a")

    MINI_CHECK(result == ["a", "b", "c", "d"])


@MINI_TEST("Graph", "Dfs")
def test_graph_dfs():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_edge("b", "d")
    g.add_edge("e", "f")
    result = g.dfs("a")

    MINI_CHECK(result == ["a", "b", "c", "d"])


@MINI_TEST("Graph", "Connected Components")
def test_graph_connected_components():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_edge("b", "d")
    g.add_edge("e", "f")
    comps = g.connected_components()

    MINI_CHECK(len(comps) == 2)
    MINI_CHECK(g.is_connected() == False)
    MINI_CHECK(g.number_connected_components() == 2)


@MINI_TEST("Graph", "Shortest Path")
def test_graph_shortest_path():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_edge("b", "d")
    g.add_edge("e", "f")

    MINI_CHECK(g.shortest_path("a", "d") == ["a", "b", "d"])
    MINI_CHECK(g.shortest_path_length("a", "d") == 2)
    MINI_CHECK(g.shortest_path("a", "e") == [])
    MINI_CHECK(g.shortest_path_length("a", "e") == -1)


@MINI_TEST("Graph", "Has Cycle")
def test_graph_has_cycle():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")

    MINI_CHECK(g.has_cycle() == True)
    g2 = Graph("g2")
    g2.add_edge("x", "y")
    g2.add_edge("y", "z")
    MINI_CHECK(g2.has_cycle() == False)


@MINI_TEST("Graph", "Cycle Basis")
def test_graph_cycle_basis():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    cycles = g.cycle_basis()

    MINI_CHECK(len(cycles) == 1)


if __name__ == "__main__":
    run_all(language="python")
