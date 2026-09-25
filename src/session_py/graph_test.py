from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


# ═══════════════════════════════════════════════════════════════════════════
# Vertex
# ═══════════════════════════════════════════════════════════════════════════
@MINI_TEST("Vertex", "Constructor")
def test_vertex_constructor():
    from session_py import Vertex

    v0 = Vertex()
    v = Vertex("v_named", "attr")

    MINI_CHECK(v0.name == "my_vertex")
    MINI_CHECK(v0.attribute == "")
    MINI_CHECK(v0.guid != "")
    MINI_CHECK(v.name == "v_named")
    MINI_CHECK(v.attribute == "attr")


@MINI_TEST("Vertex", "Json Roundtrip")
def test_vertex_json_roundtrip():
    from session_py import Vertex
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = Vertex("v0", "test_attribute")
    original.attributes["load"] = 1.5

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_vertex.json"
    file_json_dump(original, fname)
    loaded = file_json_load(fname)

    MINI_CHECK(loaded.name == original.name)
    MINI_CHECK(loaded.attribute == original.attribute)
    MINI_CHECK(loaded.attributes == original.attributes)


# ═══════════════════════════════════════════════════════════════════════════
# Edge
# ═══════════════════════════════════════════════════════════════════════════
@MINI_TEST("Edge", "Constructor")
def test_edge_constructor():
    from session_py import Edge

    e = Edge("a", "b", "attr")

    MINI_CHECK(e.v0 == "a")
    MINI_CHECK(e.v1 == "b")
    MINI_CHECK(e.attribute == "attr")
    MINI_CHECK(e.guid != "")


@MINI_TEST("Edge", "Json Roundtrip")
def test_edge_json_roundtrip():
    from session_py import Edge
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = Edge("v0", "v1", "test_edge_attr")
    original.attributes["weight"] = 2.5

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_edge.json"
    file_json_dump(original, fname)
    loaded = file_json_load(fname)

    MINI_CHECK(loaded.name == original.name)
    MINI_CHECK(loaded.v0 == original.v0)
    MINI_CHECK(loaded.v1 == original.v1)
    MINI_CHECK(loaded.attributes == original.attributes)


@MINI_TEST("Edge", "Vertices")
def test_edge_vertices():
    from session_py import Edge

    e = Edge("a", "b")
    u, v = e.vertices()

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

    g0 = Graph()
    g = Graph("my_named_graph")
    gstr = str(g0)
    grepr = repr(g0)

    MINI_CHECK(g0.name == "my_graph")
    MINI_CHECK(g0.guid != "")
    MINI_CHECK(g0.vertex_count == 0)
    MINI_CHECK(g0.edge_count == 0)
    MINI_CHECK(g.name == "my_named_graph")
    MINI_CHECK(gstr == "<Graph with 0 vertices, 0 edges: my_graph>")
    MINI_CHECK(grepr == f"Graph({g0.guid}, my_graph, 0, 0)")


@MINI_TEST("Graph", "Json Roundtrip")
def test_graph_json_roundtrip():
    from session_py import Graph
    from pathlib import Path

    original = Graph("test_graph")
    original.add_node("node1", "Node 1")
    original.add_node("node2", "Node 2")
    original.add_edge("node1", "node2", "edge1")

    edge_key = ("node1", "node2")
    original.update_default_vertex_attributes({"load": 1.0})
    original.update_default_edge_attributes({"weight": 2.0})
    original.set_vertex_attribute("node1", "load", 3.0)
    original.set_edge_attribute(edge_key, "weight", 4.0)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_graph.json"
    original.file_json_dump(fname)
    loaded = Graph.file_json_load(fname)

    MINI_CHECK(loaded.number_of_vertices() == 2)
    MINI_CHECK(loaded.number_of_edges() == 1)
    MINI_CHECK(loaded.has_edge(edge_key))
    MINI_CHECK(loaded.default_vertex_attributes == original.default_vertex_attributes)
    MINI_CHECK(loaded.default_edge_attributes == original.default_edge_attributes)
    MINI_CHECK(loaded.vertex_attribute("node1", "load") == 3.0)
    MINI_CHECK(loaded.vertex_attribute("node2", "load") == 1.0)
    MINI_CHECK(loaded.edge_attribute(edge_key, "weight") == 4.0)


@MINI_TEST("Graph", "Protobuf Roundtrip")
def test_graph_protobuf_roundtrip():
    from session_py import Graph
    from pathlib import Path

    original = Graph("test_graph")
    original.add_node("node1", "Node 1")
    original.add_node("node2", "Node 2")
    original.add_edge("node1", "node2", "edge1")

    edge_key = ("node1", "node2")
    original.update_default_vertex_attributes({"load": 1.0})
    original.update_default_edge_attributes({"weight": 2.0})
    original.set_vertex_attribute("node1", "load", 3.0)
    original.set_edge_attribute(edge_key, "weight", 4.0)

    guid = original.guid
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_graph.bin"
    original.pb_dump(filename)

    loaded = Graph.pb_load(filename)
    converted = Graph.from_proto(original.to_proto())

    MINI_CHECK(loaded.number_of_vertices() == 2)
    MINI_CHECK(loaded.number_of_edges() == 1)
    MINI_CHECK(loaded.has_edge(edge_key))
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(loaded.default_vertex_attributes == original.default_vertex_attributes)
    MINI_CHECK(loaded.default_edge_attributes == original.default_edge_attributes)
    MINI_CHECK(loaded.vertex_attribute("node1", "load") == 3.0)
    MINI_CHECK(loaded.vertex_attribute("node2", "load") == 1.0)
    MINI_CHECK(loaded.edge_attribute(edge_key, "weight") == 4.0)
    MINI_CHECK(converted.number_of_edges() == 1)
    MINI_CHECK(converted.guid == guid)
    MINI_CHECK(converted.edge_attribute(edge_key, "weight") == 4.0)


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

    ab = ("a", "b")
    ac = ("a", "c")

    MINI_CHECK(g.has_edge(ab))
    MINI_CHECK(not g.has_edge(ac))


@MINI_TEST("Graph", "Has Guid")
def test_graph_has_guid():
    from session_py import Edge
    from session_py import Vertex

    v = Vertex("a")
    e = Edge("a", "b")

    MINI_CHECK(not v.has_guid())
    MINI_CHECK(not e.has_guid())

    minted = v.guid

    MINI_CHECK(minted != "")
    MINI_CHECK(v.has_guid())
    MINI_CHECK(v.guid == minted)


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
    g.add_edge("b", "a", "updated")

    MINI_CHECK(u == "a" and v == "b")
    MINI_CHECK(g.number_of_edges() == 1)
    MINI_CHECK(g.edge_count == 1)
    MINI_CHECK(g.edge_label("a", "b") == "updated")
    MINI_CHECK(g.edges["a"]["b"].guid == g.edges["b"]["a"].guid)


@MINI_TEST("Graph", "Remove Node")
def test_graph_remove_node():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.remove_node("a")

    MINI_CHECK(not g.has_node("a"))
    MINI_CHECK(g.number_of_edges() == 0)
    MINI_CHECK(g.edge_count == 0)


@MINI_TEST("Graph", "Remove Edge")
def test_graph_remove_edge():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    edge_key = ("a", "b")
    g.remove_edge(edge_key)

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

    edges = g.get_edges()

    MINI_CHECK(len(edges) == 2)


@MINI_TEST("Graph", "Neighbors")
def test_graph_neighbors():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("a", "c")

    neigh = g.neighbors("a")

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


@MINI_TEST("Graph", "Node Label")
def test_graph_node_label():
    from session_py import Graph

    g = Graph("g")
    g.add_node("a", "initial")
    g.node_label("a", "updated")

    MINI_CHECK(g.node_label("a") == "updated")


@MINI_TEST("Graph", "Edge Label")
def test_graph_edge_label():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b", "initial")
    g.edge_label("a", "b", "updated")

    MINI_CHECK(g.edge_label("a", "b") == "updated")


@MINI_TEST("Graph", "Update Default Vertex Attributes")
def test_graph_update_default_vertex_attributes():
    from session_py import Graph

    g = Graph("g")
    g.update_default_vertex_attributes({"is_support": 0.0, "load": 0.0})
    g.update_default_vertex_attributes({"load": -1.0})

    MINI_CHECK(len(g.default_vertex_attributes) == 2)
    MINI_CHECK(g.default_vertex_attributes["is_support"] == 0.0)
    MINI_CHECK(g.default_vertex_attributes["load"] == -1.0)


@MINI_TEST("Graph", "Update Default Edge Attributes")
def test_graph_update_default_edge_attributes():
    from session_py import Graph

    g = Graph("g")
    g.update_default_edge_attributes({"weight": 1.0, "stiffness": 0.0})
    g.update_default_edge_attributes({"weight": 2.0})

    MINI_CHECK(len(g.default_edge_attributes) == 2)
    MINI_CHECK(g.default_edge_attributes["weight"] == 2.0)
    MINI_CHECK(g.default_edge_attributes["stiffness"] == 0.0)


@MINI_TEST("Graph", "Vertex Attribute")
def test_graph_vertex_attribute():
    from session_py import Graph

    g = Graph("g")
    g.add_node("a")
    g.add_node("b")
    g.update_default_vertex_attributes({"is_support": 0.0})
    g.set_vertex_attribute("a", "is_support", 1.0)

    MINI_CHECK(g.vertex_attribute("a", "is_support") == 1.0)
    MINI_CHECK(g.vertex_attribute("b", "is_support") == 0.0)
    MINI_CHECK(g.vertex_attribute("a", "missing") is None)
    MINI_CHECK(g.vertex_attribute("missing", "is_support") is None)


@MINI_TEST("Graph", "Set Vertex Attribute")
def test_graph_set_vertex_attribute():
    from session_py import Graph

    g = Graph("g")
    g.add_node("a")
    g.set_vertex_attribute("a", "load", -2.5)
    g.set_vertex_attribute("missing", "load", 1.0)

    vertices = g.get_vertices()

    MINI_CHECK(vertices[0].attributes["load"] == -2.5)
    MINI_CHECK(not g.has_node("missing"))


@MINI_TEST("Graph", "Edge Attribute")
def test_graph_edge_attribute():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.update_default_edge_attributes({"weight": 1.0})

    ab = ("a", "b")
    ba = ("b", "a")
    bc = ("b", "c")
    ac = ("a", "c")
    g.set_edge_attribute(ab, "weight", 5.0)

    MINI_CHECK(g.edge_attribute(ab, "weight") == 5.0)
    MINI_CHECK(g.edge_attribute(ba, "weight") == 5.0)
    MINI_CHECK(g.edge_attribute(bc, "weight") == 1.0)
    MINI_CHECK(g.edge_attribute(ab, "missing") is None)
    MINI_CHECK(g.edge_attribute(ac, "weight") is None)


@MINI_TEST("Graph", "Set Edge Attribute")
def test_graph_set_edge_attribute():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")

    ba = ("b", "a")
    ac = ("a", "c")
    g.set_edge_attribute(ba, "weight", 3.0)
    g.set_edge_attribute(ac, "weight", 1.0)
    g.add_edge("a", "b")

    MINI_CHECK(g.edges["a"]["b"].attributes["weight"] == 3.0)
    MINI_CHECK(g.edges["b"]["a"].attributes["weight"] == 3.0)
    MINI_CHECK(not g.has_edge(ac))


@MINI_TEST("Graph", "Vertices Where")
def test_graph_vertices_where():
    from session_py import Graph

    g = Graph("g")
    g.add_node("a")
    g.add_node("b")
    g.add_node("c")
    g.update_default_vertex_attributes({"is_support": 0.0, "level": 1.0})
    g.set_vertex_attribute("a", "is_support", 1.0)
    g.set_vertex_attribute("c", "is_support", 1.0)
    g.set_vertex_attribute("c", "level", 2.0)

    MINI_CHECK(g.vertices_where({"is_support": 1.0}) == ["a", "c"])
    MINI_CHECK(g.vertices_where({"is_support": 1.0, "level": 1.0}) == ["a"])
    MINI_CHECK(g.vertices_where({"is_support": 0.0}) == ["b"])
    MINI_CHECK(g.vertices_where({"missing": 0.0}) == [])


@MINI_TEST("Graph", "Edges Where")
def test_graph_edges_where():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "d")
    g.update_default_edge_attributes({"weight": 0.0})

    bc = ("b", "c")
    cd = ("c", "d")
    dc = ("d", "c")
    g.set_edge_attribute(bc, "weight", 3.0)
    g.set_edge_attribute(dc, "weight", 3.0)

    heavy = g.edges_where({"weight": 3.0})
    light = g.edges_where({"weight": 0.0})

    MINI_CHECK(len(heavy) == 2)
    MINI_CHECK(heavy[0] == bc)
    MINI_CHECK(heavy[1] == cd)
    MINI_CHECK(len(light) == 1)


@MINI_TEST("Graph", "Vertices Where Predicate")
def test_graph_vertices_where_predicate():
    from session_py import Graph

    g = Graph("g")
    g.add_node("a")
    g.add_node("b")
    g.add_node("c")
    g.update_default_vertex_attributes({"load": 1.0})
    g.set_vertex_attribute("b", "load", 5.0)
    g.set_vertex_attribute("c", "load", 10.0)

    heavy = g.vertices_where_predicate(lambda key, attributes: attributes["load"] > 4.0)

    MINI_CHECK(heavy == ["b", "c"])


@MINI_TEST("Graph", "Edges Where Predicate")
def test_graph_edges_where_predicate():
    from session_py import Graph

    g = Graph("g")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.update_default_edge_attributes({"weight": 1.0})

    bc = ("b", "c")
    g.set_edge_attribute(bc, "weight", 5.0)

    heavy = g.edges_where_predicate(lambda edge, attributes: attributes["weight"] > 4.0)

    MINI_CHECK(len(heavy) == 1)
    MINI_CHECK(heavy[0] == bc)


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
    MINI_CHECK(not g.is_connected())
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

    g2 = Graph("g2")
    g2.add_edge("x", "y")
    g2.add_edge("y", "z")

    MINI_CHECK(g.has_cycle())
    MINI_CHECK(not g2.has_cycle())


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
