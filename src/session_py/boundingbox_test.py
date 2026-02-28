from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("BoundingBox", "Json Roundtrip")
def test_boundingbox_json_roundtrip():
    from session_py import BoundingBox
    from session_py import Point
    from session_py.encoders import json_dump
    from session_py.encoders import json_load
    from pathlib import Path

    original = BoundingBox.from_point(Point(1.0, 2.0, 3.0), 5.0)
    original.name = "test_bbox"

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_boundingbox.json"
    json_dump(original, fname)
    loaded = json_load(fname)

    MINI_CHECK(loaded.name == original.name)


if __name__ == "__main__":
    run_all(language="python")
