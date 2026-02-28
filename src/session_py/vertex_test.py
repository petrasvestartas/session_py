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


if __name__ == "__main__":
    run_all(language="python")
