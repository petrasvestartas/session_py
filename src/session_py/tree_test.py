from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("TreeNode", "Constructor")
def test_treenode_constructor():
    from session_py import TreeNode
    from session_py import Color

    # Default constructor
    n0 = TreeNode()

    # Constructor with name
    n = TreeNode("my_named_node")
    n.color = Color(1.0, 0.0, 0.0, 1.0)

    # Minimal string representation
    nstr = str(n)

    # Copies (compared by identity in Python)
    nother = TreeNode("my_named_node")

    MINI_CHECK(n0.name == "my_node")
    MINI_CHECK(n0.guid)
    MINI_CHECK(n.name == "my_named_node")
    MINI_CHECK(n.color is not None and n.color[0] == 1.0)
    MINI_CHECK("TreeNode(my_named_node" in nstr)
    MINI_CHECK(n == n)
    MINI_CHECK(n != nother)


@MINI_TEST("TreeNode", "Json Roundtrip")
def test_treenode_json_roundtrip():
    from session_py import TreeNode
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = TreeNode("test_node")
    child = TreeNode("child_node")
    original.add(child)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_treenode.json"
    file_json_dump(original, fname)
    loaded = file_json_load(fname)

    MINI_CHECK(loaded.name == original.name)
    MINI_CHECK(len(loaded.children) == 1)
    MINI_CHECK(loaded.children[0].name == "child_node")


@MINI_TEST("TreeNode", "Is Root")
def test_treenode_is_root():
    from session_py import TreeNode

    root = TreeNode("root")
    child = TreeNode("child")
    root.add(child)

    MINI_CHECK(root.is_root)
    MINI_CHECK(not child.is_root)


@MINI_TEST("TreeNode", "Is Leaf")
def test_treenode_is_leaf():
    from session_py import TreeNode

    parent = TreeNode("parent")
    child = TreeNode("child")
    parent.add(child)

    MINI_CHECK(child.is_leaf)
    MINI_CHECK(not parent.is_leaf)


@MINI_TEST("TreeNode", "Tree")
def test_treenode_tree():
    from session_py import TreeNode

    n = TreeNode("standalone")

    MINI_CHECK(n.tree is None)


@MINI_TEST("TreeNode", "Add")
def test_treenode_add():
    from session_py import TreeNode

    parent = TreeNode("parent")
    child = TreeNode("child")
    parent.add(child)

    MINI_CHECK(len(parent.children) == 1)
    MINI_CHECK(child.parent is parent)


@MINI_TEST("TreeNode", "Remove")
def test_treenode_remove():
    from session_py import TreeNode

    parent = TreeNode("parent")
    child = TreeNode("child")
    parent.add(child)
    parent.remove(child)

    MINI_CHECK(len(parent.children) == 0)
    MINI_CHECK(child.parent is None)


@MINI_TEST("TreeNode", "Parent")
def test_treenode_parent():
    from session_py import TreeNode

    root = TreeNode("root")
    child = TreeNode("child")
    root.add(child)

    MINI_CHECK(root.parent is None)
    MINI_CHECK(child.parent is root)


@MINI_TEST("TreeNode", "Ancestors")
def test_treenode_ancestors():
    from session_py import TreeNode

    root = TreeNode("root")
    mid = TreeNode("mid")
    leaf = TreeNode("leaf")
    root.add(mid)
    mid.add(leaf)

    anc = list(leaf.ancestors)

    MINI_CHECK(len(anc) == 2)
    MINI_CHECK(anc[0].name == "mid")
    MINI_CHECK(anc[1].name == "root")


@MINI_TEST("TreeNode", "Descendants")
def test_treenode_descendants():
    from session_py import TreeNode

    root = TreeNode("root")
    mid = TreeNode("mid")
    leaf = TreeNode("leaf")
    root.add(mid)
    mid.add(leaf)

    desc = list(root.descendants)

    MINI_CHECK(len(desc) == 2)
    MINI_CHECK(desc[0].name == "mid")
    MINI_CHECK(desc[1].name == "leaf")


@MINI_TEST("TreeNode", "Children")
def test_treenode_children():
    from session_py import TreeNode

    parent = TreeNode("parent")
    c1 = TreeNode("c1")
    c2 = TreeNode("c2")
    parent.add(c1)
    parent.add(c2)

    kids = parent.children

    MINI_CHECK(len(kids) == 2)
    MINI_CHECK(kids[0].name == "c1")
    MINI_CHECK(kids[1].name == "c2")


@MINI_TEST("TreeNode", "Traverse")
def test_treenode_traverse():
    from session_py import TreeNode

    root = TreeNode("root")
    a = TreeNode("a")
    b = TreeNode("b")
    root.add(a)
    root.add(b)

    preorder = list(root.traverse("depthfirst", "preorder"))
    postorder = list(root.traverse("depthfirst", "postorder"))
    bfs = list(root.traverse("breadthfirst", "preorder"))

    MINI_CHECK(len(preorder) == 3 and preorder[0].name == "root")
    MINI_CHECK(len(postorder) == 3 and postorder[2].name == "root")
    MINI_CHECK(len(bfs) == 3 and bfs[0].name == "root")


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
    #   file_json_dumps()    │ str          │ to JSON string
    #   file_json_loads(s)   │ str          │ from JSON string
    #   file_json_dump(path) │ file         │ write to file
    #   file_json_load(path) │ file         │ read from file

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_tree.json"
    original.file_json_dump(fname)
    loaded = Tree.file_json_load(fname)

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
