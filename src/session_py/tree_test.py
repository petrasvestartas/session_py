from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Tree", "Json Roundtrip")
def test_tree_json_roundtrip():
    from session_py import Tree
    from session_py import TreeNode
    from session_py import Point
    from session_py.encoders import json_dump
    from session_py.encoders import json_load
    from pathlib import Path

    original = Tree("./serialization/test_tree")
    point1 = Point(1.0, 2.0, 3.0)
    node1 = TreeNode(point1.guid)
    original.add(node1)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_tree.json"
    json_dump(original, fname)
    loaded = json_load(fname)

    MINI_CHECK(loaded.name == original.name)
    MINI_CHECK(len(loaded.nodes()) == len(original.nodes()))


if __name__ == "__main__":
    run_all(language="python")
