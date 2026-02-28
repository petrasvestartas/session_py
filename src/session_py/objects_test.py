from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Objects", "Json Roundtrip")
def test_objects_json_roundtrip():
    from session_py import Objects
    from session_py import Point
    from session_py.encoders import json_dump
    from session_py.encoders import json_load
    from pathlib import Path

    original = Objects()
    original.points.append(Point(1.0, 2.0, 3.0))
    original.points.append(Point(4.0, 5.0, 6.0))

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_objects.json"
    json_dump(original, fname)
    loaded = json_load(fname)

    MINI_CHECK(len(loaded.points) == len(original.points))


if __name__ == "__main__":
    run_all(language="python")
