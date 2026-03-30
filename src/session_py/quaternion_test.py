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
    import math

    axis = Vector(0.0, 0.0, 1.0)
    original = Quaternion.from_axis_angle(axis, 1.5708)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_quaternion.json"
    json_dump(original, fname)
    loaded = json_load(fname)

    MINI_CHECK(TOLERANCE.is_close(loaded.s, original.s))


@MINI_TEST("Quaternion", "From Arc")
def test_quaternion_from_arc():
    from session_py import Quaternion
    from session_py import Vector

    src = Vector(1.0, 0.0, 0.0)
    dst = Vector(0.0, 1.0, 0.0)
    q = Quaternion.from_arc(src, dst)
    rotated = q.rotate_vector(src)

    MINI_CHECK(TOLERANCE.is_close(rotated[0], dst[0]))
    MINI_CHECK(TOLERANCE.is_close(rotated[1], dst[1]))
    MINI_CHECK(TOLERANCE.is_close(rotated[2], dst[2]))
    src2 = Vector(1.0, 0.0, 0.0)
    dst2 = Vector(-1.0, 0.0, 0.0)
    q2 = Quaternion.from_arc(src2, dst2)
    rot = q2.rotate_vector(src2)
    MINI_CHECK(TOLERANCE.is_close(rot[0], -1.0))
    MINI_CHECK(TOLERANCE.is_close(rot[1], 0.0))


@MINI_TEST("Quaternion", "From Euler")
def test_quaternion_from_euler():
    from session_py import Quaternion
    from session_py import Vector
    import math

    q_euler = Quaternion.from_euler(0.0, 0.0, math.pi / 2.0)
    q_axis = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), math.pi / 2.0)

    MINI_CHECK(TOLERANCE.is_close(q_euler.s, q_axis.s))
    MINI_CHECK(TOLERANCE.is_close(q_euler.v[2], q_axis.v[2]))


@MINI_TEST("Quaternion", "slerp")
def test_quaternion_slerp():
    from session_py import Quaternion
    from session_py import Vector
    import math

    q1 = Quaternion.identity()
    q2 = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), math.pi / 2.0)
    mid = q1.slerp(q2, 0.5)
    expected = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), math.pi / 4.0)

    MINI_CHECK(TOLERANCE.is_close(mid.s, expected.s))
    MINI_CHECK(TOLERANCE.is_close(mid.v[2], expected.v[2]))
    q3 = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), 0.001)
    mid2 = q1.slerp(q3, 0.5)
    half = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), 0.0005)
    MINI_CHECK(TOLERANCE.is_close(mid2.s, half.s))


@MINI_TEST("Quaternion", "nlerp")
def test_quaternion_nlerp():
    from session_py import Quaternion
    from session_py import Vector
    import math

    q1 = Quaternion.identity()
    q2 = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), math.pi / 2.0)
    r0 = q1.nlerp(q2, 0.0)
    r1 = q1.nlerp(q2, 1.0)

    MINI_CHECK(TOLERANCE.is_close(r0.s, q1.s))
    MINI_CHECK(TOLERANCE.is_close(r1.s, q2.s))


@MINI_TEST("Quaternion", "invert")
def test_quaternion_invert():
    from session_py import Quaternion
    from session_py import Vector
    import math

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), math.pi / 3.0)
    result = q * q.invert()

    MINI_CHECK(TOLERANCE.is_close(result.s, 1.0))
    MINI_CHECK(TOLERANCE.is_close(result.v[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(result.v[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(result.v[2], 0.0))


@MINI_TEST("Quaternion", "dot")
def test_quaternion_dot():
    from session_py import Quaternion

    q = Quaternion.identity()

    MINI_CHECK(TOLERANCE.is_close(q.dot(q), 1.0))


@MINI_TEST("Quaternion", "add")
def test_quaternion_add():
    from session_py import Quaternion
    from session_py import Vector

    a = Quaternion.from_sv(1.0, Vector(0.0, 0.0, 0.0))
    b = Quaternion.from_sv(0.0, Vector(0.0, 0.0, 1.0))
    r = a + b

    MINI_CHECK(TOLERANCE.is_close(r.s, 1.0))
    MINI_CHECK(TOLERANCE.is_close(r.v[2], 1.0))


@MINI_TEST("Quaternion", "sub")
def test_quaternion_sub():
    from session_py import Quaternion
    from session_py import Vector
    import math

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), math.pi / 4.0)
    r = q - q

    MINI_CHECK(TOLERANCE.is_close(r.s, 0.0))
    MINI_CHECK(TOLERANCE.is_close(r.v[2], 0.0))


@MINI_TEST("Quaternion", "neg")
def test_quaternion_neg():
    from session_py import Quaternion

    q = Quaternion.identity()
    r = -q

    MINI_CHECK(TOLERANCE.is_close(r.s, -1.0))


@MINI_TEST("Quaternion", "Mul Scalar")
def test_quaternion_mul_scalar():
    from session_py import Quaternion

    q = Quaternion.identity()
    r = q * 2.0

    MINI_CHECK(TOLERANCE.is_close(r.s, 2.0))


@MINI_TEST("Quaternion", "magnitude2")
def test_quaternion_magnitude2():
    from session_py import Quaternion
    from session_py import Vector
    import math

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), math.pi / 4.0)

    MINI_CHECK(TOLERANCE.is_close(q.magnitude2(), q.magnitude() * q.magnitude()))


@MINI_TEST("Quaternion", "conjugate")
def test_quaternion_conjugate():
    from session_py import Quaternion
    from session_py import Vector
    import math

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), math.pi / 4.0)
    r = q.conjugate()

    MINI_CHECK(TOLERANCE.is_close(r.s, q.s))
    MINI_CHECK(TOLERANCE.is_close(r.v[0], -q.v[0]))
    MINI_CHECK(TOLERANCE.is_close(r.v[2], -q.v[2]))


@MINI_TEST("Quaternion", "mul")
def test_quaternion_mul():
    from session_py import Quaternion
    from session_py import Vector
    import math

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), math.pi / 2.0)
    r = q * q
    v = r.rotate_vector(Vector(1.0, 0.0, 0.0))

    MINI_CHECK(TOLERANCE.is_close(v[0], -1.0))
    MINI_CHECK(TOLERANCE.is_close(v[1], 0.0))


if __name__ == "__main__":
    run_all(language="python")
