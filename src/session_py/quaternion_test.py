from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Quaternion", "Json Roundtrip")
def test_quaternion_json_roundtrip():
    from session_py import Quaternion
    from session_py import Vector
    from session_py.encoders import json_dump
    from session_py.encoders import json_load
    from pathlib import Path

    axis = Vector(0.0, 0.0, 1.0)
    original = Quaternion.from_axis_angle(axis, 1.5708)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_quaternion.json"
    json_dump(original, fname)
    loaded = json_load(fname)

    MINI_CHECK(TOLERANCE.is_close(loaded.s, original.s))


if __name__ == "__main__":
    run_all(language="python")
