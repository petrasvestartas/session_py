from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


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


if __name__ == "__main__":
    run_all(language="python")
