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
    n.color = Color(255, 0, 0, 255)

    # Minimal string representation
    nstr = str(n)

    # Copies (compared by identity in Python)
    nother = TreeNode("my_named_node")

    MINI_CHECK(n0.name == "my_node")
    MINI_CHECK(n0.guid)
    MINI_CHECK(n.name == "my_named_node")
    MINI_CHECK(n.color is not None and n.color[0] == 255)
    MINI_CHECK("TreeNode(my_named_node" in nstr)
    MINI_CHECK(n == n)
    MINI_CHECK(n != nother)


@MINI_TEST("TreeNode", "Json Roundtrip")
def test_treenode_json_roundtrip():
    from session_py import TreeNode
    from session_py.encoders import json_dump
    from session_py.encoders import json_load
    from pathlib import Path

    original = TreeNode("test_node")
    child = TreeNode("child_node")
    original.add(child)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_treenode.json"
    json_dump(original, fname)
    loaded = json_load(fname)

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


if __name__ == "__main__":
    run_all(language="python")
