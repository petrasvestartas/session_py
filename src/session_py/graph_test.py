from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


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


if __name__ == "__main__":
    run_all(language="python")
