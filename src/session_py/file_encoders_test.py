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
    from session_py.file_encoders import file_encode_collection

    points = []
    points.append(Point(1.0, 2.0, 3.0))
    points.append(Point(4.0, 5.0, 6.0))
    points.append(Point(7.0, 8.0, 9.0))

    json_arr = file_encode_collection(points)

    MINI_CHECK(isinstance(json_arr, list))
    MINI_CHECK(len(json_arr) == 3)
    MINI_CHECK(json_arr[0]["type"] == "Point")
    MINI_CHECK(json_arr[1]["x"] == 4.0)
    MINI_CHECK(json_arr[2]["z"] == 9.0)


@MINI_TEST("FileEncoders", "Encode Collection Shared Ptr")
def test_file_encode_collection_shared_ptr():
    from session_py import Line
    from session_py.file_encoders import file_encode_collection

    lines = []
    lines.append(Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    lines.append(Line(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))

    json_arr = file_encode_collection(lines)

    MINI_CHECK(isinstance(json_arr, list))
    MINI_CHECK(len(json_arr) == 2)
    MINI_CHECK(json_arr[0]["type"] == "Line")
    MINI_CHECK(json_arr[1]["type"] == "Line")


@MINI_TEST("FileEncoders", "Decode Collection")
def test_file_decode_collection():
    from session_py import Point
    from session_py.file_encoders import file_encode_collection
    from session_py.file_encoders import file_decode_collection

    original_points = []
    original_points.append(Point(1.0, 2.0, 3.0))
    original_points.append(Point(4.0, 5.0, 6.0))

    json_arr = file_encode_collection(original_points)
    decoded_points = file_decode_collection(json_arr, Point)

    MINI_CHECK(len(decoded_points) == 2)
    MINI_CHECK(TOLERANCE.is_close(decoded_points[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded_points[1][1], 5.0))


@MINI_TEST("FileEncoders", "Decode Collection Ptr")
def test_file_decode_collection_ptr():
    from session_py import Vector
    from session_py.file_encoders import file_encode_collection
    from session_py.file_encoders import file_decode_collection

    original_vectors = []
    original_vectors.append(Vector(1.0, 0.0, 0.0))
    original_vectors.append(Vector(0.0, 1.0, 0.0))

    json_arr = file_encode_collection(original_vectors)
    decoded_vectors = file_decode_collection(json_arr, Vector)

    MINI_CHECK(len(decoded_vectors) == 2)
    MINI_CHECK(TOLERANCE.is_close(decoded_vectors[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded_vectors[1][1], 1.0))


@MINI_TEST("FileEncoders", "Nested Collections")
def test_nested_collections():
    from session_py import Line
    from session_py.file_encoders import file_encode_collection
    from session_py.file_encoders import file_decode_collection
    import json

    lines = []
    lines.append(Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    lines.append(Line(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))

    json_arr = file_encode_collection(lines)
    json_str = json.dumps(json_arr)

    MINI_CHECK(json_str != "")

    loaded_json = json.loads(json_str)
    loaded = file_decode_collection(loaded_json, Line)

    MINI_CHECK(len(loaded) == 2)
    MINI_CHECK(TOLERANCE.is_close(loaded[0].end()[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded[1].end()[1], 1.0))


@MINI_TEST("FileEncoders", "Roundtrip File Io")
def test_roundtrip_file_io():
    from session_py import Vector
    from session_py.file_encoders import file_encode_collection
    from session_py.file_encoders import file_decode_collection
    from session_py.file_encoders import file_json_dump
    from session_py.file_encoders import file_json_load_data
    from pathlib import Path

    vectors = []
    vectors.append(Vector(1.0, 0.0, 0.0))
    vectors.append(Vector(0.0, 1.0, 0.0))
    vectors.append(Vector(0.0, 0.0, 1.0))

    filepath = (
        Path(__file__).resolve().parents[2]
        / "serialization"
        / "test_encoders_collection.json"
    )
    json_arr = file_encode_collection(vectors)
    file_json_dump(json_arr, filepath)

    loaded_json = file_json_load_data(filepath)
    decoded_vectors = file_decode_collection(loaded_json, Vector)

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

    MINI_CHECK(loaded)


@MINI_TEST("FileEncoders", "Decode List")
def test_decode_list():
    from session_py import Point
    from session_py.file_encoders import file_encode_collection
    from session_py.file_encoders import file_decode_collection
    import json

    data = [1, 2, 3]
    json_str = json.dumps(data)
    loaded_vec = json.loads(json_str)

    MINI_CHECK(len(loaded_vec) == 3)
    MINI_CHECK(loaded_vec[0] == 1)
    MINI_CHECK(loaded_vec[2] == 3)

    points = []
    points.append(Point(1.0, 2.0, 3.0))
    points.append(Point(4.0, 5.0, 6.0))

    json_arr = file_encode_collection(points)
    decoded = file_decode_collection(json_arr, Point)

    MINI_CHECK(len(decoded) == 2)
    MINI_CHECK(TOLERANCE.is_close(decoded[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(decoded[1][0], 4.0))


@MINI_TEST("FileEncoders", "Decode Dict")
def test_decode_dict():
    from session_py import Vector
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads
    import json

    data = {}
    data["a"] = 1
    data["b"] = 2

    json_str = json.dumps(data)
    loaded = json.loads(json_str)

    MINI_CHECK(loaded["a"] == 1)
    MINI_CHECK(loaded["b"] == 2)

    vec = Vector(1.0, 2.0, 3.0)
    vec_json = file_json_dumps(vec)
    loaded_vec = file_json_loads(vec_json)

    MINI_CHECK(TOLERANCE.is_close(loaded_vec[0], 1.0))


@MINI_TEST("FileEncoders", "Decode Mesh")
def test_decode_mesh():
    from session_py import Mesh
    from session_py import Point
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    mesh = Mesh.from_vertices_and_faces(
        [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.0, 1.0, 0.0)], [[0, 1, 2]]
    )
    json_str = file_json_dumps(mesh)
    loaded = file_json_loads(json_str)

    MINI_CHECK(loaded.number_of_vertices() == 3)
    MINI_CHECK(loaded.number_of_faces() == 1)


@MINI_TEST("FileEncoders", "Decode Instance Ref")
def test_decode_instance_ref():
    from session_py import InstanceRef
    from session_py import Xform
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    instance = InstanceRef("def-abc", Xform.translation(1.0, 2.0, 3.0))
    json_str = file_json_dumps(instance)
    loaded = file_json_loads(json_str)

    MINI_CHECK(loaded.definition_guid == "def-abc")
    MINI_CHECK(TOLERANCE.is_close(loaded[12], 1.0))


@MINI_TEST("FileEncoders", "Decode Element Feature")
def test_decode_element_feature():
    from session_py import ElementFeature
    from session_py import Point
    from session_py import Polyline
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    outline = Polyline(
        [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0)]
    )
    feature = ElementFeature("cut", 2, [outline], "notch")
    json_str = file_json_dumps(feature)
    loaded = file_json_loads(json_str)

    MINI_CHECK(loaded.feature_type == "cut")
    MINI_CHECK(loaded.face_index == 2)
    MINI_CHECK(len(loaded.outlines) == 1)
    MINI_CHECK(loaded.outlines[0].point_count() == 3)


@MINI_TEST("FileEncoders", "Decode Component")
def test_decode_component():
    from session_py import Component
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    component = Component()
    component.type_name = "FloorBuilder"
    component.name = "floor"
    component.extra["height"] = 650
    json_str = file_json_dumps(component)
    loaded = file_json_loads(json_str)

    MINI_CHECK(loaded.type_name == "FloorBuilder")
    MINI_CHECK(loaded.name == "floor")
    MINI_CHECK(loaded.extra["height"] == 650)


@MINI_TEST("FileEncoders", "Decode Nurbs Surface Trimmed")
def test_decode_nurbs_surface_trimmed():
    from session_py import NurbsCurve
    from session_py import NurbsSurface
    from session_py import NurbsSurfaceTrimmed
    from session_py import Point
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    surface = NurbsSurface(3, False, 2, 2, 2, 2)
    surface.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surface.set_cv(1, 0, Point(5.0, 0.0, 0.0))
    surface.set_cv(0, 1, Point(0.0, 5.0, 0.0))
    surface.set_cv(1, 1, Point(5.0, 5.0, 0.0))

    outer = NurbsCurve.create(
        True,
        1,
        [
            Point(0.1, 0.1, 0.0),
            Point(0.9, 0.1, 0.0),
            Point(0.9, 0.9, 0.0),
            Point(0.1, 0.9, 0.0),
        ],
    )
    inner = NurbsCurve.create(
        True,
        1,
        [Point(0.4, 0.4, 0.0), Point(0.6, 0.4, 0.0), Point(0.6, 0.6, 0.0)],
    )

    trimmed = NurbsSurfaceTrimmed.create(surface, outer)
    trimmed.add_inner_loop(inner)
    trimmed.name = "trimmed"
    json_str = file_json_dumps(trimmed)
    loaded = file_json_loads(json_str)

    MINI_CHECK(loaded.name == "trimmed")
    MINI_CHECK(loaded.is_trimmed())
    MINI_CHECK(loaded.inner_loop_count() == 1)


@MINI_TEST("FileEncoders", "Decode Nurbs Surface")
def test_decode_nurbs_surface():
    from session_py import NurbsSurface
    from session_py import Point
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    surface = NurbsSurface(3, False, 2, 2, 2, 2)
    surface.set_cv(0, 0, Point(0.0, 0.0, 0.0))
    surface.set_cv(1, 0, Point(5.0, 0.0, 0.0))
    surface.set_cv(0, 1, Point(0.0, 5.0, 0.0))
    surface.set_cv(1, 1, Point(5.0, 5.0, 0.0))
    mesh = surface.mesh()
    json_str = file_json_dumps(surface)
    loaded = file_json_loads(json_str)

    MINI_CHECK(loaded.mesh().number_of_vertices() == mesh.number_of_vertices())
    MINI_CHECK(loaded.cv_count(0) == 2)
    MINI_CHECK(loaded.cv_count(1) == 2)


@MINI_TEST("FileEncoders", "Decode BRep")
def test_decode_brep():
    from session_py import BRep
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    brep = BRep.create_box(1.0, 2.0, 3.0)
    json_str = file_json_dumps(brep)
    loaded = file_json_loads(json_str)

    MINI_CHECK(loaded.face_count() == 6)


@MINI_TEST("FileEncoders", "Decode Element")
def test_decode_element():
    from session_py import Element
    from session_py import ElementFeature
    from session_py import Mesh
    from session_py import Point
    from session_py import Polyline
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    mesh = Mesh.from_vertices_and_faces(
        [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.0, 1.0, 0.0)], [[0, 1, 2]]
    )
    outline = Polyline(
        [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0)]
    )

    element = Element(mesh, "plate")
    element.add_feature(ElementFeature("cut", 0, [outline], "notch"))
    json_str = file_json_dumps(element)
    loaded = file_json_loads(json_str)

    MINI_CHECK(loaded.name == "plate")
    MINI_CHECK(loaded.features_count == 1)


@MINI_TEST("FileEncoders", "Decode Objects")
def test_decode_objects():
    from session_py import Component
    from session_py import ElementFeature
    from session_py import InstanceRef
    from session_py import Objects
    from session_py import Point
    from session_py import Polyline
    from session_py import Xform
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    component = Component()
    component.type_name = "FloorBuilder"
    outline = Polyline(
        [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0)]
    )
    instance = InstanceRef("def-abc", Xform.translation(1.0, 2.0, 3.0))
    instance.features.append(ElementFeature("drill", 0, [outline], "hole"))

    objects = Objects()
    objects.points.append(Point(1.0, 2.0, 3.0))
    objects.components.append(component)
    objects.instances.append(instance)
    json_str = file_json_dumps(objects)
    loaded = file_json_loads(json_str)

    MINI_CHECK(len(loaded.points) == 1)
    MINI_CHECK(len(loaded.components) == 1)
    MINI_CHECK(len(loaded.instances) == 1)
    MINI_CHECK(len(loaded.instances[0].features) == 1)


@MINI_TEST("FileEncoders", "Decode Tolerance")
def test_decode_tolerance():
    from session_py import Tolerance
    from session_py.file_encoders import file_json_dumps
    from session_py.file_encoders import file_json_loads

    tolerance = Tolerance("MM")
    tolerance.set_absolute(0.01)
    json_str = file_json_dumps(tolerance)
    loaded = file_json_loads(json_str)

    MINI_CHECK(loaded.unit() == "MM")
    MINI_CHECK(TOLERANCE.is_close(loaded.absolute(), 0.01))


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
    from session_py.file_encoders import file_encode_collection
    from session_py.file_encoders import file_decode_collection
    import json

    points = []
    points.append(Point(1.0, 0.0, 0.0))
    points.append(Point(0.0, 1.0, 0.0))

    data = {}
    data["numbers"] = [1, 2, 3]
    data["letters"] = ["a", "b", "c"]
    data["points"] = file_encode_collection(points)

    json_str = json.dumps(data)
    loaded = json.loads(json_str)

    MINI_CHECK(len(loaded["numbers"]) == 3)
    MINI_CHECK(loaded["letters"][0] == "a")

    loaded_points = file_decode_collection(loaded["points"], Point)

    MINI_CHECK(len(loaded_points) == 2)
    MINI_CHECK(TOLERANCE.is_close(loaded_points[0][0], 1.0))


@MINI_TEST("FileEncoders", "List Of Dict")
def test_list_of_dict():
    from session_py import Point
    import json

    point = Point(1.0, 2.0, 3.0)

    data = []
    data.append({"name": "point1", "value": 10})
    data.append({"name": "point2", "value": 20})
    data.append({"geometry": point.__jsondump__()})

    json_str = json.dumps(data)
    loaded = json.loads(json_str)

    MINI_CHECK(len(loaded) == 3)
    MINI_CHECK(loaded[0]["name"] == "point1")
    MINI_CHECK(loaded[1]["value"] == 20)

    loaded_point = Point.__jsonload__(loaded[2]["geometry"])

    MINI_CHECK(TOLERANCE.is_close(loaded_point[2], 3.0))


@MINI_TEST("FileEncoders", "Dict Of Dicts")
def test_dict_of_dicts():
    from session_py import Point
    from session_py import Vector
    import json

    point = Point(1.0, 2.0, 3.0)
    vec = Vector(0.0, 0.0, 1.0)

    data = {"config": {}, "geometry": {}}
    data["config"]["tolerance"] = 0.001
    data["config"]["scale"] = 1000
    data["geometry"]["point"] = point.__jsondump__()
    data["geometry"]["vector"] = vec.__jsondump__()

    json_str = json.dumps(data)
    loaded = json.loads(json_str)

    MINI_CHECK(TOLERANCE.is_close(loaded["config"]["tolerance"], 0.001))
    MINI_CHECK(loaded["config"]["scale"] == 1000)

    loaded_point = Point.__jsonload__(loaded["geometry"]["point"])
    loaded_vec = Vector.__jsonload__(loaded["geometry"]["vector"])

    MINI_CHECK(TOLERANCE.is_close(loaded_point[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded_vec[2], 1.0))


@MINI_TEST("FileEncoders", "Write Error")
def test_write_error():
    from pathlib import Path
    from session_py import Point
    from session_py.file_encoders import file_json_dump

    point = Point(1.0, 2.0, 3.0)
    threw = False

    try:
        file_json_dump(
            point,
            Path(__file__).resolve().parents[2]
            / "serialization"
            / "missing-directory"
            / "test.json",
        )
    except OSError:
        threw = True

    MINI_CHECK(threw)


if __name__ == "__main__":
    run_all(language="python")
