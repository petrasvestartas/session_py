from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("FileEncoders", "Json Dump Load")
def test_file_json_dump_load():
    from session_py import Point
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    original = Point(1.5, 2.5, 3.5)
    original.name = "test_point"

    filepath = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_encoders_point.json"
    )
    file_json_dump(original, filepath)

    loaded = file_json_load(filepath)

    MINI_CHECK(TOLERANCE.is_close(loaded[0], original[0]))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], original[1]))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], original[2]))
    MINI_CHECK(loaded.name == original.name)

    filepath.unlink()


@MINI_TEST("FileEncoders", "Json Dumps Loads")
def test_file_json_dumps_loads():
    from session_py import Vector
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    original = Vector(42.1, 84.2, 126.3)
    original.name = "test_vector"

    json_str = file_json_dumps(original)

    MINI_CHECK(json_str != "")
    MINI_CHECK("Vector" in json_str)

    loaded = file_json_loads(json_str)

    MINI_CHECK(TOLERANCE.is_close(loaded[0], original[0]))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], original[1]))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], original[2]))
    MINI_CHECK(loaded.name == original.name)


@MINI_TEST("FileEncoders", "Encode Collection Values")
def test_file_encode_collection_values():
    from session_py import Point
    from session_py.file_encoders import file_json_dumps
    import json

    points = [
        Point(1.0, 2.0, 3.0),
        Point(4.0, 5.0, 6.0),
        Point(7.0, 8.0, 9.0),
    ]

    json_str = file_json_dumps(points)
    json_arr = json.loads(json_str)

    MINI_CHECK(isinstance(json_arr, list))
    MINI_CHECK(len(json_arr) == 3)
    MINI_CHECK(json_arr[0]["type"] == "Point")
    MINI_CHECK(json_arr[1]["x"] == 4.0)
    MINI_CHECK(json_arr[2]["z"] == 9.0)


@MINI_TEST("FileEncoders", "Encode Collection Shared Ptr")
def test_file_encode_collection_shared_ptr():
    from session_py import Line
    from session_py.file_encoders import file_json_dumps
    import json

    lines = [
        Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        Line(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ]

    json_str = file_json_dumps(lines)
    json_arr = json.loads(json_str)

    MINI_CHECK(isinstance(json_arr, list))
    MINI_CHECK(len(json_arr) == 2)
    MINI_CHECK(json_arr[0]["type"] == "Line")
    MINI_CHECK(json_arr[1]["type"] == "Line")


@MINI_TEST("FileEncoders", "Decode Collection")
def test_file_decode_collection():
    from session_py import Point
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    original_points = [
        Point(1.0, 2.0, 3.0),
        Point(4.0, 5.0, 6.0),
    ]

    json_str = file_json_dumps(original_points)
    decoded_points = file_json_loads(json_str)

    MINI_CHECK(len(decoded_points) == 2)
    MINI_CHECK(TOLERANCE.is_close(decoded_points[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded_points[1][1], 5.0))


@MINI_TEST("FileEncoders", "Decode Collection Ptr")
def test_file_decode_collection_ptr():
    from session_py import Vector
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    original_vectors = [
        Vector(1.0, 0.0, 0.0),
        Vector(0.0, 1.0, 0.0),
    ]

    json_str = file_json_dumps(original_vectors)
    decoded_vectors = file_json_loads(json_str)

    MINI_CHECK(len(decoded_vectors) == 2)
    MINI_CHECK(TOLERANCE.is_close(decoded_vectors[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded_vectors[1][1], 1.0))


@MINI_TEST("FileEncoders", "Nested Collections")
def test_nested_collections():
    from session_py import Line
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    lines = [
        Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        Line(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ]

    json_str = file_json_dumps(lines)

    MINI_CHECK(json_str != "")

    loaded = file_json_loads(json_str)

    MINI_CHECK(len(loaded) == 2)
    MINI_CHECK(TOLERANCE.is_close(loaded[0].end()[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[1].end()[1], 1.0))


@MINI_TEST("FileEncoders", "Roundtrip File Io")
def test_roundtrip_file_io():
    from session_py import Vector
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load
    from pathlib import Path

    vectors = [
        Vector(1.0, 0.0, 0.0),
        Vector(0.0, 1.0, 0.0),
        Vector(0.0, 0.0, 1.0),
    ]

    filepath = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_encoders_collection.json"
    )
    file_json_dump(vectors, filepath)

    decoded_vectors = file_json_load(filepath)

    MINI_CHECK(len(decoded_vectors) == 3)
    MINI_CHECK(TOLERANCE.is_close(decoded_vectors[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded_vectors[1][1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded_vectors[2][2], 1.0))

    filepath.unlink()


@MINI_TEST("FileEncoders", "Pretty Vs Compact")
def test_pretty_vs_compact():
    from session_py import Point
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    point = Point(1.0, 2.0, 3.0)

    pretty = file_json_dumps(point, True)
    compact = file_json_dumps(point, False)

    MINI_CHECK(len(pretty) > len(compact))
    MINI_CHECK("\n" in pretty)
    MINI_CHECK("\n" not in compact)

    loaded_pretty = file_json_loads(pretty)
    loaded_compact = file_json_loads(compact)

    MINI_CHECK(TOLERANCE.is_close(loaded_pretty[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded_compact[0], 1.0))


@MINI_TEST("FileEncoders", "Decode Primitives")
def test_decode_primitives():
    import json

    num = 42
    json_str = json.dumps(num)
    loaded = json.loads(json_str)

    MINI_CHECK(loaded == 42)

    float_val = 3.14
    json_str = json.dumps(float_val)
    loaded = json.loads(json_str)
    MINI_CHECK(TOLERANCE.is_close(loaded, 3.14))

    text = "hello"
    json_str = json.dumps(text)
    loaded = json.loads(json_str)
    MINI_CHECK(loaded == "hello")

    flag = True
    json_str = json.dumps(flag)
    loaded = json.loads(json_str)
    MINI_CHECK(loaded is True)


@MINI_TEST("FileEncoders", "Decode List")
def test_decode_list():
    from session_py import Point
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads
    import json

    data = [1, 2, 3]
    json_str = json.dumps(data)
    loaded_vec = json.loads(json_str)

    MINI_CHECK(len(loaded_vec) == 3)
    MINI_CHECK(loaded_vec[0] == 1)
    MINI_CHECK(loaded_vec[2] == 3)

    points = [
        Point(1.0, 2.0, 3.0),
        Point(4.0, 5.0, 6.0),
    ]

    json_str = file_json_dumps(points)
    decoded = file_json_loads(json_str)
    MINI_CHECK(len(decoded) == 2)
    MINI_CHECK(TOLERANCE.is_close(decoded[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded[1][0], 4.0))


@MINI_TEST("FileEncoders", "Decode Dict")
def test_decode_dict():
    from session_py import Vector
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads
    import json

    data = {"a": 1, "b": 2}
    json_str = json.dumps(data)
    loaded = json.loads(json_str)

    MINI_CHECK(loaded["a"] == 1)
    MINI_CHECK(loaded["b"] == 2)

    vec = Vector(1.0, 2.0, 3.0)
    vec_json = file_json_dumps(vec)
    loaded_vec = file_json_loads(vec_json)
    MINI_CHECK(TOLERANCE.is_close(loaded_vec[0], 1.0))


@MINI_TEST("FileEncoders", "List In List In List")
def test_list_in_list_in_list():
    import json

    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    json_str = json.dumps(data)
    loaded = json.loads(json_str)

    MINI_CHECK(loaded[0][0][0] == 1)
    MINI_CHECK(loaded[1][1][1] == 8)
    MINI_CHECK(len(loaded) == 2)


@MINI_TEST("FileEncoders", "Dict Of Lists")
def test_dict_of_lists():
    from session_py import Point
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    points = [
        Point(1.0, 0.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]

    data = {
        "numbers": [1, 2, 3],
        "letters": ["a", "b", "c"],
        "points": points,
    }

    json_str = file_json_dumps(data)
    loaded = file_json_loads(json_str)

    MINI_CHECK(len(loaded["numbers"]) == 3)
    MINI_CHECK(loaded["letters"][0] == "a")
    loaded_points = loaded["points"]
    MINI_CHECK(len(loaded_points) == 2)
    MINI_CHECK(TOLERANCE.is_close(loaded_points[0][0], 1.0))


@MINI_TEST("FileEncoders", "List Of Dict")
def test_list_of_dict():
    from session_py import Point
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    point = Point(1.0, 2.0, 3.0)

    data = [
        {"name": "point1", "value": 10},
        {"name": "point2", "value": 20},
        {"geometry": point},
    ]

    json_str = file_json_dumps(data)
    loaded = file_json_loads(json_str)

    MINI_CHECK(len(loaded) == 3)
    MINI_CHECK(loaded[0]["name"] == "point1")
    MINI_CHECK(loaded[1]["value"] == 20)
    loaded_point = loaded[2]["geometry"]
    MINI_CHECK(TOLERANCE.is_close(loaded_point[2], 3.0))


@MINI_TEST("FileEncoders", "Dict Of Dicts")
def test_dict_of_dicts():
    from session_py import Point
    from session_py import Vector
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    point = Point(1.0, 2.0, 3.0)
    vec = Vector(0.0, 0.0, 1.0)

    data = {
        "config": {"tolerance": 0.001, "scale": 1000},
        "geometry": {"point": point, "vector": vec},
    }

    json_str = file_json_dumps(data)
    loaded = file_json_loads(json_str)

    MINI_CHECK(TOLERANCE.is_close(loaded["config"]["tolerance"], 0.001))
    MINI_CHECK(loaded["config"]["scale"] == 1000)
    loaded_point = loaded["geometry"]["point"]
    loaded_vec = loaded["geometry"]["vector"]
    MINI_CHECK(TOLERANCE.is_close(loaded_point[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded_vec[2], 1.0))


@MINI_TEST("FileEncoders", "Write Error")
def test_write_error():
    from session_py import Point
    from session_py.file_encoders import file_json_dump

    threw = False
    try:
        file_json_dump(Point(1.0, 2.0, 3.0), "/definitely/missing-directory/test.json")
    except OSError:
        threw = True
    MINI_CHECK(threw)


if __name__ == "__main__":
    run_all(language="python")
