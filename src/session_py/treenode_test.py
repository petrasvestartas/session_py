from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


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


if __name__ == "__main__":
    run_all(language="python")
