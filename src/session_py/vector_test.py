from .mini_test import MINI_TEST, MINI_CHECK, run_all


@MINI_TEST("Vector", "constructor")
def test_vector_constructor():
    from session_py import Vector

    # Constructor
    v = Vector(1.0, 2.0, 3.0)

    # Setters
    v[0] = 10.0
    v[1] = 20.0
    v[2] = 30.0

    # Getters
    x = v[0]
    y = v[1]
    z = v[2]

    # Copy
    vcopy = Vector(10.0, 20.0, 30.0)
    vcopy.guid = v.guid

    MINI_CHECK(
        v.name == "my_vector" and
        v[0] == 10.0 and
        v[1] == 20.0 and
        v[2] == 30.0 and
        v.guid
    )

    MINI_CHECK(x == 10.0 and y == 20.0 and z == 30.0)
    MINI_CHECK(vcopy == v)


@MINI_TEST("Vector", "arithmetic")
def test_vector_arithmetic():
    from session_py import Vector

    v1 = Vector(1.0, 2.0, 3.0)
    v2 = Vector(4.0, 5.0, 6.0)

    # Addition
    sum_vec = v1 + v2
    MINI_CHECK(sum_vec[0] == 5.0 and sum_vec[1] == 7.0 and sum_vec[2] == 9.0)

    # Subtraction
    diff = v2 - v1
    MINI_CHECK(diff[0] == 3.0 and diff[1] == 3.0 and diff[2] == 3.0)

    # Scalar multiplication
    scaled = v1 * 2.0
    MINI_CHECK(scaled[0] == 2.0 and scaled[1] == 4.0 and scaled[2] == 6.0)

    # Scalar division
    divided = v2 / 2.0
    MINI_CHECK(divided[0] == 2.0 and divided[1] == 2.5 and divided[2] == 3.0)


@MINI_TEST("Vector", "magnitude")
def test_vector_magnitude():
    from session_py import Vector

    v = Vector(3.0, 4.0, 0.0)
    length = v.magnitude()
    MINI_CHECK(abs(length - 5.0) < 1e-10)

    unit = Vector(1.0, 0.0, 0.0)
    MINI_CHECK(unit.magnitude() == 1.0)


@MINI_TEST("Vector", "normalize")
def test_vector_normalize():
    from session_py import Vector

    v = Vector(3.0, 4.0, 0.0)
    n = v.normalize()

    MINI_CHECK(abs(n.magnitude() - 1.0) < 1e-10)
    MINI_CHECK(abs(n[0] - 0.6) < 1e-10)
    MINI_CHECK(abs(n[1] - 0.8) < 1e-10)
    MINI_CHECK(n[2] == 0.0)


@MINI_TEST("Vector", "dot_product")
def test_vector_dot_product():
    from session_py import Vector

    v1 = Vector(1.0, 0.0, 0.0)
    v2 = Vector(0.0, 1.0, 0.0)
    v3 = Vector(1.0, 0.0, 0.0)

    # Perpendicular vectors
    MINI_CHECK(v1.dot(v2) == 0.0)

    # Parallel vectors
    MINI_CHECK(v1.dot(v3) == 1.0)


@MINI_TEST("Vector", "cross_product")
def test_vector_cross_product():
    from session_py import Vector

    v1 = Vector(1.0, 0.0, 0.0)
    v2 = Vector(0.0, 1.0, 0.0)

    cross = v1.cross(v2)
    MINI_CHECK(cross[0] == 0.0 and cross[1] == 0.0 and cross[2] == 1.0)


@MINI_TEST("Vector", "json_roundtrip")
def test_vector_json_roundtrip():
    from session_py import Vector
    from session_py.encoders import json_dump, json_load
    from pathlib import Path

    v = Vector(42.1, 84.2, 126.3)
    v.name = "test_vector"

    path = Path(__file__).resolve().parents[2] / "test_vector.json"
    json_dump(v, path)
    loaded = json_load(path)

    MINI_CHECK(loaded.name == "test_vector")
    MINI_CHECK(abs(loaded[0] - 42.1) < 1e-10)
    MINI_CHECK(abs(loaded[1] - 84.2) < 1e-10)
    MINI_CHECK(abs(loaded[2] - 126.3) < 1e-10)


@MINI_TEST("Vector", "static_axes")
def test_vector_static_axes():
    from session_py import Vector

    x = Vector.x_axis()
    y = Vector.y_axis()
    z = Vector.z_axis()

    MINI_CHECK(x[0] == 1.0 and x[1] == 0.0 and x[2] == 0.0)
    MINI_CHECK(y[0] == 0.0 and y[1] == 1.0 and y[2] == 0.0)
    MINI_CHECK(z[0] == 0.0 and z[1] == 0.0 and z[2] == 1.0)


if __name__ == "__main__":
    run_all("python")
