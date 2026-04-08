from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("Tree", "Constructor")
def test_tree_constructor():
    from session_py import Tree

    # Default constructor
    t0 = Tree()

    # Constructor with name
    t = Tree("my_named_tree")

    # Minimal string representation
    tstr = str(t)

    MINI_CHECK(t0.name == "my_tree")
    MINI_CHECK(t0.guid)
    MINI_CHECK(t.name == "my_named_tree")
    MINI_CHECK("Tree" in tstr)


@MINI_TEST("Tree", "Json Roundtrip")
def test_tree_json_roundtrip():
    from session_py import Tree
    from session_py import TreeNode
    from pathlib import Path

    original = Tree("test_tree")
    root_node = TreeNode("root_node")
    original.add(root_node)

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   json_dumps()    │ str          │ to JSON string
    #   json_loads(s)   │ str          │ from JSON string
    #   json_dump(path) │ file         │ write to file
    #   json_load(path) │ file         │ read from file

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_tree.json"
    original.json_dump(fname)
    loaded = Tree.json_load(fname)

    MINI_CHECK(loaded.name == original.name)
    MINI_CHECK(len(list(loaded.nodes)) == len(list(original.nodes)))


@MINI_TEST("Tree", "Protobuf Roundtrip")
def test_tree_protobuf_roundtrip():
    from session_py import Tree
    from session_py import TreeNode
    from pathlib import Path

    original = Tree("test_tree")
    root_node = TreeNode("root_node")
    original.add(root_node)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_tree.bin"
    original.pb_dump(fname)
    loaded = Tree.pb_load(fname)

    MINI_CHECK(loaded.name == original.name)
    MINI_CHECK(len(list(loaded.nodes)) == len(list(original.nodes)))


@MINI_TEST("Tree", "Root")
def test_tree_root():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    t.add(root)

    MINI_CHECK(t.root is root)


@MINI_TEST("Tree", "Add")
def test_tree_add():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    child = TreeNode("child")
    t.add(root)
    t.add(child, root)

    MINI_CHECK(len(list(t.nodes)) == 2)


@MINI_TEST("Tree", "Nodes")
def test_tree_nodes():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    child = TreeNode("child")
    t.add(root)
    t.add(child, root)

    all_nodes = list(t.nodes)

    MINI_CHECK(len(all_nodes) == 2)
    MINI_CHECK(all_nodes[0].name == "root")
    MINI_CHECK(all_nodes[1].name == "child")


@MINI_TEST("Tree", "Remove")
def test_tree_remove():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    child = TreeNode("child")
    t.add(root)
    t.add(child, root)
    t.remove(child)

    MINI_CHECK(len(list(t.nodes)) == 1)


@MINI_TEST("Tree", "Leaves")
def test_tree_leaves():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    a = TreeNode("a")
    b = TreeNode("b")
    t.add(root)
    t.add(a, root)
    t.add(b, root)

    lvs = list(t.leaves)

    MINI_CHECK(len(lvs) == 2)
    MINI_CHECK(lvs[0].name == "a")
    MINI_CHECK(lvs[1].name == "b")


@MINI_TEST("Tree", "Traverse")
def test_tree_traverse():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    a = TreeNode("a")
    b = TreeNode("b")
    t.add(root)
    t.add(a, root)
    t.add(b, root)

    preorder = list(t.traverse("depthfirst", "preorder"))
    bfs = list(t.traverse("breadthfirst", "preorder"))

    MINI_CHECK(len(preorder) == 3 and preorder[0].name == "root")
    MINI_CHECK(len(bfs) == 3 and bfs[0].name == "root")


@MINI_TEST("Tree", "Get Node By Name")
def test_tree_get_node_by_name():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    child = TreeNode("target")
    t.add(root)
    t.add(child, root)

    found = t.get_node_by_name("target")

    MINI_CHECK(found is not None and found.name == "target")
    MINI_CHECK(t.get_node_by_name("missing") is None)


@MINI_TEST("Tree", "Get Nodes By Name")
def test_tree_get_nodes_by_name():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    a = TreeNode("dup")
    b = TreeNode("dup")
    t.add(root)
    t.add(a, root)
    t.add(b, root)

    found = t.get_nodes_by_name("dup")

    MINI_CHECK(len(found) == 2)


@MINI_TEST("Tree", "Find Node By Guid")
def test_tree_find_node_by_guid():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    t.add(root)
    root_guid = root.guid

    found = t.find_node_by_guid(root_guid)

    MINI_CHECK(found is not None and found.guid == root_guid)
    MINI_CHECK(t.find_node_by_guid("missing-guid") is None)


@MINI_TEST("Tree", "Add Child By Guid")
def test_tree_add_child_by_guid():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    a = TreeNode("a")
    b = TreeNode("b")
    t.add(root)
    t.add(a, root)
    t.add(b, root)
    ok = t.add_child_by_guid(a.guid, b.guid)

    MINI_CHECK(ok)
    MINI_CHECK(len(a.children) == 1)


@MINI_TEST("Tree", "Get Children Guids")
def test_tree_get_children_guids():
    from session_py import Tree
    from session_py import TreeNode

    t = Tree("t")
    root = TreeNode("root")
    a = TreeNode("a")
    b = TreeNode("b")
    t.add(root)
    t.add(a, root)
    t.add(b, root)

    guids = t.get_children_guids(root.guid)

    MINI_CHECK(len(guids) == 2)
    MINI_CHECK(guids[0] == a.guid)
    MINI_CHECK(guids[1] == b.guid)


if __name__ == "__main__":
    run_all(language="python")
