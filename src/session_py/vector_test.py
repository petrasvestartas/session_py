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

    # Minimal and full string representation
    vstr = v.str()
    vrepr = v.repr()

    # Copy (duplicate everything but guid)
    vcopy = v.duplicate()
    vother = Vector(1.0, 2.0, 3.0)

    # No-copy operators
    vmult = v.duplicate()
    vmult *= 2.0
    vdiv = v.duplicate()
    vdiv /= 2.0
    vadd = v.duplicate()
    vadd += Vector(1.0, 1.0, 1.0)
    vsub = v.duplicate()
    vsub -= Vector(1.0, 1.0, 1.0)

    # Copy operators
    result_mul = v * 2.0
    result_div = v / 2.0
    result_add = v + Vector(1.0, 1.0, 1.0)
    result_dif = v - Vector(1.0, 1.0, 1.0)

    # Static axis constructors
    vx = Vector.x_axis()
    vy = Vector.y_axis()
    vz = Vector.z_axis()
    vzero = Vector.zero()

    MINI_CHECK(
        v.name == "my_vector" and
        v[0] == 10.0 and
        v[1] == 20.0 and
        v[2] == 30.0 and
        v.guid
    )
    MINI_CHECK(x == 10.0 and y == 20.0 and z == 30.0)
    MINI_CHECK(vstr == "10.000000, 20.000000, 30.000000")
    MINI_CHECK(vrepr == "Vector(my_vector, 10.000000, 20.000000, 30.000000, 37.416574)")
    MINI_CHECK(vcopy == v and vcopy.guid != v.guid)
    MINI_CHECK(vother != v)
    MINI_CHECK(vmult[0] == 20.0 and vmult[1] == 40.0 and vmult[2] == 60.0)
    MINI_CHECK(vdiv[0] == 5.0 and vdiv[1] == 10.0 and vdiv[2] == 15.0)
    MINI_CHECK(vadd[0] == 11.0 and vadd[1] == 21.0 and vadd[2] == 31.0)
    MINI_CHECK(vsub[0] == 9.0 and vsub[1] == 19.0 and vsub[2] == 29.0)
    MINI_CHECK(result_mul[0] == 20.0 and result_mul[1] == 40.0 and result_mul[2] == 60.0)
    MINI_CHECK(result_div[0] == 5.0 and result_div[1] == 10.0 and result_div[2] == 15.0)
    MINI_CHECK(result_add[0] == 11.0 and result_add[1] == 21.0 and result_add[2] == 31.0)
    MINI_CHECK(result_dif[0] == 9.0 and result_dif[1] == 19.0 and result_dif[2] == 29.0)
    MINI_CHECK(vx[0] == 1.0 and vx[1] == 0.0 and vx[2] == 0.0)
    MINI_CHECK(vy[0] == 0.0 and vy[1] == 1.0 and vy[2] == 0.0)
    MINI_CHECK(vz[0] == 0.0 and vz[1] == 0.0 and vz[2] == 1.0)
    MINI_CHECK(vzero[0] == 0.0 and vzero[1] == 0.0 and vzero[2] == 0.0)


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


if __name__ == "__main__":
    run_all("python")
