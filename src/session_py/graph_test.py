from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Vertex", "Json Roundtrip")
def test_vertex_json_roundtrip():
    from session_py import Vertex
    from session_py.encoders import json_dump
    from session_py.encoders import json_load
    from pathlib import Path

    original = Vertex("v0", "./serialization/test_attribute")

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_vertex.json"
    json_dump(original, fname)
    loaded = json_load(fname)

    MINI_CHECK(loaded.name == original.name)
    MINI_CHECK(loaded.attribute == original.attribute)


@MINI_TEST("Edge", "Json Roundtrip")
def test_edge_json_roundtrip():
    from session_py import Edge
    from session_py.encoders import json_dump
    from session_py.encoders import json_load
    from pathlib import Path

    original = Edge("v0", "v1", "./serialization/test_edge")

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_edge.json"
    json_dump(original, fname)
    loaded = json_load(fname)

    MINI_CHECK(loaded.name == original.name)
    MINI_CHECK(loaded.v0 == original.v0)
    MINI_CHECK(loaded.v1 == original.v1)


@MINI_TEST("Graph", "Json Roundtrip")
def test_graph_json_roundtrip():
    from session_py import Graph
    from session_py.encoders import json_dump
    from session_py.encoders import json_load
    from pathlib import Path

    original = Graph("./serialization/test_graph")
    original.add_node("node1", "Node 1")
    original.add_node("node2", "Node 2")
    original.add_edge("node1", "node2", "edge1")

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_graph.json"
    json_dump(original, fname)
    loaded = json_load(fname)

    MINI_CHECK(loaded.number_of_vertices() == 2)
    MINI_CHECK(loaded.number_of_edges() == 1)
    MINI_CHECK(loaded.has_edge(("node1", "node2")))


@MINI_TEST("Graph", "Bfs")
def test_graph_bfs():
    from session_py import Graph

    g = Graph("test")
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

    g = Graph("test")
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

    g = Graph("test")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_edge("b", "d")
    g.add_edge("e", "f")
    comps = g.connected_components()

    MINI_CHECK(len(comps) == 2)
    MINI_CHECK(comps[0] == ["a", "b", "c", "d"])
    MINI_CHECK(comps[1] == ["e", "f"])
    MINI_CHECK(g.is_connected() == False)
    MINI_CHECK(g.number_connected_components() == 2)


@MINI_TEST("Graph", "Shortest Path")
def test_graph_shortest_path():
    from session_py import Graph

    g = Graph("test")
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

    g = Graph("test")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_edge("b", "d")
    g.add_edge("e", "f")

    MINI_CHECK(g.has_cycle() == True)
    g2 = Graph("test2")
    g2.add_edge("x", "y")
    g2.add_edge("y", "z")
    MINI_CHECK(g2.has_cycle() == False)


@MINI_TEST("Graph", "Cycle Basis")
def test_graph_cycle_basis():
    from session_py import Graph

    g = Graph("test")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_edge("b", "d")
    g.add_edge("e", "f")
    cycles = g.cycle_basis()

    MINI_CHECK(len(cycles) == 1)
    cycle_set = set(cycles[0])
    MINI_CHECK(cycle_set == {"a", "b", "c"})


if __name__ == "__main__":
    run_all(language="python")
