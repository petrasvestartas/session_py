from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Encoders", "Json Dump Load")
def test_json_dump_load():
    from session_py import Point
    from session_py.encoders import json_dump, json_load
    from pathlib import Path
    import json

    original = Point(1.5, 2.5, 3.5)
    original.name = "test_point"
    filepath = Path(__file__).resolve().parents[2] / "serialization" / "test_encoders_point.json"
    json_dump(original, filepath)
    loaded = json_load(filepath)

    MINI_CHECK(TOLERANCE.is_close(loaded[0], original[0]))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], original[1]))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], original[2]))
    MINI_CHECK(loaded.name == original.name)

    filepath.unlink()


@MINI_TEST("Encoders", "Json Dumps Loads")
def test_json_dumps_loads():
    from session_py import Vector
    from session_py.encoders import json_dumps, json_loads

    original = Vector(42.1, 84.2, 126.3)
    original.name = "test_vector"
    json_str = json_dumps(original)

    MINI_CHECK(isinstance(json_str, str))
    MINI_CHECK("Vector" in json_str)

    loaded = json_loads(json_str)

    MINI_CHECK(TOLERANCE.is_close(loaded[0], original[0]))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], original[1]))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], original[2]))
    MINI_CHECK(loaded.name == original.name)


@MINI_TEST("Encoders", "Encode Collection Values")
def test_encode_collection_values():
    from session_py import Point
    from session_py.encoders import json_dumps
    import json

    json_str = json_dumps([
        Point(1, 2, 3),
        Point(4, 5, 6),
        Point(7, 8, 9),
    ])
    d = json.loads(json_str)

    MINI_CHECK(isinstance(d, list))
    MINI_CHECK(len(d) == 3)
    MINI_CHECK(d[0]["type"] == "Point")
    MINI_CHECK(d[1]["x"] == 4.0)
    MINI_CHECK(d[2]["z"] == 9.0)


@MINI_TEST("Encoders", "Encode Collection Shared Ptr")
def test_encode_collection_shared_ptr():
    from session_py import Line
    from session_py.encoders import json_dumps
    import json

    d = json.loads(json_dumps([
        Line(0, 0, 0, 1, 0, 0),
        Line(0, 0, 0, 0, 1, 0),
    ]))

    MINI_CHECK(isinstance(d, list))
    MINI_CHECK(len(d) == 2)
    MINI_CHECK(d[0]["type"] == "Line")
    MINI_CHECK(d[1]["type"] == "Line")


@MINI_TEST("Encoders", "Decode Collection")
def test_decode_collection():
    from session_py import Point
    from session_py.encoders import json_dumps, json_loads

    decoded = json_loads(json_dumps([
        Point(1, 2, 3),
        Point(4, 5, 6),
    ]))

    MINI_CHECK(len(decoded) == 2)
    MINI_CHECK(TOLERANCE.is_close(decoded[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded[1][1], 5.0))


@MINI_TEST("Encoders", "Decode Collection Ptr")
def test_decode_collection_ptr():
    from session_py import Vector
    from session_py.encoders import json_dumps, json_loads

    decoded = json_loads(json_dumps([
        Vector(1, 0, 0),
        Vector(0, 1, 0),
    ]))

    MINI_CHECK(len(decoded) == 2)
    MINI_CHECK(TOLERANCE.is_close(decoded[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded[1][1], 1.0))


@MINI_TEST("Encoders", "Nested Collections")
def test_nested_collections():
    from session_py import Line
    from session_py.encoders import json_dumps, json_loads

    loaded = json_loads(json_dumps([
        Line(0, 0, 0, 1, 0, 0),
        Line(0, 0, 0, 0, 1, 0),
    ]))

    MINI_CHECK(len(loaded) == 2)
    MINI_CHECK(TOLERANCE.is_close(loaded[0].end()[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[1].end()[1], 1.0))


@MINI_TEST("Encoders", "Roundtrip File Io")
def test_roundtrip_file_io():
    from session_py import Vector
    from session_py.encoders import json_dump, json_load
    from pathlib import Path

    vectors = [
        Vector(1, 0, 0),
        Vector(0, 1, 0),
        Vector(0, 0, 1),
    ]
    filepath = Path(__file__).resolve().parents[2] / "serialization" / "test_encoders_collection.json"
    json_dump(vectors, filepath)
    loaded = json_load(filepath)

    MINI_CHECK(len(loaded) == 3)
    MINI_CHECK(TOLERANCE.is_close(loaded[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[1][1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[2][2], 1.0))

    filepath.unlink()


@MINI_TEST("Encoders", "Pretty Vs Compact")
def test_pretty_vs_compact():
    from session_py import Point
    from session_py.encoders import json_dumps, json_loads

    point = Point(1, 2, 3)
    pretty = json_dumps(point, pretty=True)
    compact = json_dumps(point, pretty=False)

    MINI_CHECK(len(pretty) > len(compact))
    MINI_CHECK("\n" in pretty)
    MINI_CHECK("\n" not in compact)

    loaded_pretty = json_loads(pretty)
    loaded_compact = json_loads(compact)

    MINI_CHECK(TOLERANCE.is_close(loaded_pretty[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded_compact[0], 1.0))


@MINI_TEST("Encoders", "Decode Primitives")
def test_decode_primitives():
    import json

    json_str = json.dumps(42)
    loaded = json.loads(json_str)
    MINI_CHECK(loaded == 42)

    json_str = json.dumps(3.14)
    loaded = json.loads(json_str)
    MINI_CHECK(TOLERANCE.is_close(loaded, 3.14))

    json_str = json.dumps("hello")
    loaded = json.loads(json_str)
    MINI_CHECK(loaded == "hello")

    json_str = json.dumps(True)
    loaded = json.loads(json_str)
    MINI_CHECK(loaded is True)


@MINI_TEST("Encoders", "Decode List")
def test_decode_list():
    import json
    from session_py import Point
    from session_py.encoders import json_dumps, json_loads

    loaded_vec = json.loads(json.dumps([1, 2, 3]))

    MINI_CHECK(len(loaded_vec) == 3)
    MINI_CHECK(loaded_vec[0] == 1)
    MINI_CHECK(loaded_vec[2] == 3)

    decoded = json_loads(json_dumps([
        Point(1, 2, 3),
        Point(4, 5, 6),
    ]))

    MINI_CHECK(len(decoded) == 2)
    MINI_CHECK(TOLERANCE.is_close(decoded[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded[1][0], 4.0))


@MINI_TEST("Encoders", "Decode Dict")
def test_decode_dict():
    import json
    from session_py import Vector
    from session_py.encoders import json_dumps, json_loads

    loaded = json.loads(json.dumps({"a": 1, "b": 2}))

    MINI_CHECK(loaded["a"] == 1)
    MINI_CHECK(loaded["b"] == 2)

    loaded_vec = json_loads(json_dumps(Vector(1, 2, 3)))

    MINI_CHECK(TOLERANCE.is_close(loaded_vec[0], 1.0))


@MINI_TEST("Encoders", "List In List In List")
def test_list_in_list_in_list():
    import json

    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    loaded = json.loads(json.dumps(data))

    MINI_CHECK(loaded[0][0][0] == 1)
    MINI_CHECK(loaded[1][1][1] == 8)
    MINI_CHECK(len(loaded) == 2)


@MINI_TEST("Encoders", "Dict Of Lists")
def test_dict_of_lists():
    from session_py import Point
    from session_py.encoders import json_dumps, json_loads

    data = {
        "numbers": [1, 2, 3],
        "letters": ["a", "b", "c"],
        "points": [
            Point(1, 0, 0),
            Point(0, 1, 0),
        ],
    }
    loaded = json_loads(json_dumps(data))

    MINI_CHECK(loaded["numbers"] == [1, 2, 3])
    MINI_CHECK(loaded["letters"][0] == "a")
    MINI_CHECK(len(loaded["points"]) == 2)
    MINI_CHECK(TOLERANCE.is_close(loaded["points"][0][0], 1.0))


@MINI_TEST("Encoders", "List Of Dict")
def test_list_of_dict():
    from session_py import Point
    from session_py.encoders import json_dumps, json_loads

    data = [
        {"name": "point1", "value": 10},
        {"name": "point2", "value": 20},
        {"geometry": Point(1, 2, 3)},
    ]
    loaded = json_loads(json_dumps(data))

    MINI_CHECK(loaded[0]["name"] == "point1")
    MINI_CHECK(loaded[1]["value"] == 20)
    MINI_CHECK(TOLERANCE.is_close(loaded[2]["geometry"][2], 3.0))


@MINI_TEST("Encoders", "Dict Of Dicts")
def test_dict_of_dicts():
    from session_py import Point
    from session_py import Vector
    from session_py.encoders import json_dumps, json_loads

    data = {
        "config": {"tolerance": 0.001, "scale": 1000},
        "geometry": {"point": Point(1, 2, 3), "vector": Vector(0, 0, 1)},
    }
    loaded = json_loads(json_dumps(data))

    MINI_CHECK(TOLERANCE.is_close(loaded["config"]["tolerance"], 0.001))
    MINI_CHECK(loaded["config"]["scale"] == 1000)
    MINI_CHECK(TOLERANCE.is_close(loaded["geometry"]["point"][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded["geometry"]["vector"][2], 1.0))


if __name__ == "__main__":
    run_all(language="python")
