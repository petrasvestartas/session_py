from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("Quaternion", "Constructor")
def test_quaternion_constructor():
    from session_py import Quaternion
    from session_py import Vector

    # Default constructor (identity)
    q0 = Quaternion()

    # Constructor with arguments
    q = Quaternion.from_scalar_and_vector(2.0, Vector(1.0, 0.0, 0.0))

    # Setters
    q[0] = 5.0
    q[1] = 0.0
    q[2] = 1.0
    q[3] = 0.0

    # Getters
    s_val = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    # Minimal and Full String Representation
    qstr = str(q)
    qrepr = repr(q)

    # Copy (duplicates everything except guid)
    qcopy = q.duplicate()
    qother = Quaternion.from_scalar_and_vector(2.0, Vector(1.0, 0.0, 0.0))

    # Copy operators
    qrot = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 2.0)
    qmul = qrot * qrot
    qscaled = Quaternion.identity() * 2.0
    a = Quaternion.from_scalar_and_vector(1.0, Vector(0.0, 0.0, 0.0))
    b = Quaternion.from_scalar_and_vector(0.0, Vector(0.0, 0.0, 1.0))
    qsum = a + b
    qdiff = qrot - qrot
    qneg = -Quaternion.identity()

    MINI_CHECK(q0.name == "my_quaternion")
    MINI_CHECK(q0.guid)
    MINI_CHECK(TOLERANCE.is_close(q0.scalar, 1.0))
    MINI_CHECK(TOLERANCE.is_close(q0.vector[0], 0.0) and TOLERANCE.is_close(q0.vector[1], 0.0) and TOLERANCE.is_close(q0.vector[2], 0.0))
    MINI_CHECK(q[0] == 5.0 and q[1] == 0.0 and q[2] == 1.0 and q[3] == 0.0)
    MINI_CHECK(s_val == 5.0 and x == 0.0 and y == 1.0 and z == 0.0)
    MINI_CHECK(qstr == "5.0, 0.0, 1.0, 0.0")
    MINI_CHECK(qrepr == "Quaternion(my_quaternion, 5.0, 0.0, 1.0, 0.0)")
    MINI_CHECK(qcopy == q and qcopy.guid != q.guid)
    MINI_CHECK(qother != q)
    MINI_CHECK(TOLERANCE.is_close(qmul.scalar, 0.0) and TOLERANCE.is_close(qmul.vector[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(qscaled.scalar, 2.0))
    MINI_CHECK(TOLERANCE.is_close(qsum.scalar, 1.0) and TOLERANCE.is_close(qsum.vector[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(qdiff.scalar, 0.0) and TOLERANCE.is_close(qdiff.vector[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(qneg.scalar, -1.0))


@MINI_TEST("Quaternion", "Identity")
def test_quaternion_identity():
    from session_py import Quaternion

    q = Quaternion.identity()

    MINI_CHECK(TOLERANCE.is_close(q.scalar, 1.0))
    MINI_CHECK(TOLERANCE.is_close(q.vector[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(q.vector[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(q.vector[2], 0.0))


@MINI_TEST("Quaternion", "From Scalar And Vector")
def test_quaternion_from_scalar_and_vector():
    from session_py import Quaternion
    from session_py import Vector

    q = Quaternion.from_scalar_and_vector(2.0, Vector(1.0, 2.0, 3.0))

    MINI_CHECK(TOLERANCE.is_close(q.scalar, 2.0))
    MINI_CHECK(TOLERANCE.is_close(q.vector[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(q.vector[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(q.vector[2], 3.0))


@MINI_TEST("Quaternion", "From Axis Angle")
def test_quaternion_from_axis_angle():
    from session_py import Quaternion
    from session_py import Vector
    import math

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 2.0)

    MINI_CHECK(TOLERANCE.is_close(q.scalar, math.cos(PI / 4.0)))
    MINI_CHECK(TOLERANCE.is_close(q.vector[2], math.sin(PI / 4.0)))


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

    q_euler = Quaternion.from_euler(0.0, 0.0, PI / 2.0)
    q_axis = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 2.0)

    MINI_CHECK(TOLERANCE.is_close(q_euler.scalar, q_axis.scalar))
    MINI_CHECK(TOLERANCE.is_close(q_euler.vector[2], q_axis.vector[2]))


@MINI_TEST("Quaternion", "From Rotation")
def test_quaternion_from_rotation():
    from session_py import Quaternion
    from session_py import Vector
    from session_py import Plane
    from session_py import Point

    plane_a = Plane(Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0))
    plane_b = Plane(Point(0.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(-1.0, 0.0, 0.0))
    q = Quaternion.from_rotation(plane_a, plane_b)
    rotated_x = q.rotate_vector(plane_a.x_axis)

    MINI_CHECK(TOLERANCE.is_close(rotated_x[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(rotated_x[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(rotated_x[2], 0.0))


@MINI_TEST("Quaternion", "Rotate Vector")
def test_quaternion_rotate_vector():
    from session_py import Quaternion
    from session_py import Vector

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 2.0)
    rotated = q.rotate_vector(Vector(1.0, 0.0, 0.0))

    MINI_CHECK(TOLERANCE.is_close(rotated[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(rotated[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(rotated[2], 0.0))


@MINI_TEST("Quaternion", "Get Rotation")
def test_quaternion_get_rotation():
    from session_py import Quaternion
    from session_py import Vector

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 2.0)
    p = q.get_rotation()

    MINI_CHECK(TOLERANCE.is_close(p.x_axis[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(p.x_axis[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(p.y_axis[0], -1.0))
    MINI_CHECK(TOLERANCE.is_close(p.y_axis[1], 0.0))


@MINI_TEST("Quaternion", "Magnitude")
def test_quaternion_magnitude():
    from session_py import Quaternion
    from session_py import Vector

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 4.0)

    MINI_CHECK(TOLERANCE.is_close(q.magnitude(), 1.0))


@MINI_TEST("Quaternion", "Magnitude Squared")
def test_quaternion_magnitude_squared():
    from session_py import Quaternion
    from session_py import Vector

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 4.0)

    MINI_CHECK(TOLERANCE.is_close(q.magnitude_squared(), q.magnitude() * q.magnitude()))


@MINI_TEST("Quaternion", "Normalized")
def test_quaternion_normalized():
    from session_py import Quaternion
    from session_py import Vector

    q = Quaternion.from_scalar_and_vector(2.0, Vector(0.0, 0.0, 2.0))
    n = q.normalized()

    MINI_CHECK(TOLERANCE.is_close(n.magnitude(), 1.0))


@MINI_TEST("Quaternion", "Conjugate")
def test_quaternion_conjugate():
    from session_py import Quaternion
    from session_py import Vector

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 4.0)
    r = q.conjugate()

    MINI_CHECK(TOLERANCE.is_close(r.scalar, q.scalar))
    MINI_CHECK(TOLERANCE.is_close(r.vector[0], -q.vector[0]))
    MINI_CHECK(TOLERANCE.is_close(r.vector[2], -q.vector[2]))


@MINI_TEST("Quaternion", "Invert")
def test_quaternion_invert():
    from session_py import Quaternion
    from session_py import Vector

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 3.0)
    result = q * q.invert()

    MINI_CHECK(TOLERANCE.is_close(result.scalar, 1.0))
    MINI_CHECK(TOLERANCE.is_close(result.vector[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(result.vector[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(result.vector[2], 0.0))


@MINI_TEST("Quaternion", "Dot")
def test_quaternion_dot():
    from session_py import Quaternion

    q = Quaternion.identity()

    MINI_CHECK(TOLERANCE.is_close(q.dot(q), 1.0))


@MINI_TEST("Quaternion", "Slerp")
def test_quaternion_slerp():
    from session_py import Quaternion
    from session_py import Vector

    q1 = Quaternion.identity()
    q2 = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 2.0)
    mid = q1.slerp(q2, 0.5)
    expected = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 4.0)

    MINI_CHECK(TOLERANCE.is_close(mid.scalar, expected.scalar))
    MINI_CHECK(TOLERANCE.is_close(mid.vector[2], expected.vector[2]))
    q3 = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), 0.001)
    mid2 = q1.slerp(q3, 0.5)
    half = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), 0.0005)
    MINI_CHECK(TOLERANCE.is_close(mid2.scalar, half.scalar))


@MINI_TEST("Quaternion", "Nlerp")
def test_quaternion_nlerp():
    from session_py import Quaternion
    from session_py import Vector

    q1 = Quaternion.identity()
    q2 = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 2.0)
    r0 = q1.nlerp(q2, 0.0)
    r1 = q1.nlerp(q2, 1.0)

    MINI_CHECK(TOLERANCE.is_close(r0.scalar, q1.scalar))
    MINI_CHECK(TOLERANCE.is_close(r1.scalar, q2.scalar))


@MINI_TEST("Quaternion", "Json Roundtrip")
def test_quaternion_json_roundtrip():
    from session_py import Quaternion
    from session_py import Vector
    from pathlib import Path

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 2.0)
    q.name = "test_quaternion"

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   json_dumps()    │ str          │ to JSON string
    #   json_loads(s)   │ str          │ from JSON string
    #   json_dump(path) │ file         │ write to file
    #   json_load(path) │ file         │ read from file

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_quaternion.json"
    q.json_dump(fname)
    loaded = Quaternion.json_load(fname)

    MINI_CHECK(loaded.name == "test_quaternion")
    MINI_CHECK(TOLERANCE.is_close(loaded.scalar, q.scalar))
    MINI_CHECK(TOLERANCE.is_close(loaded.vector[2], q.vector[2]))


@MINI_TEST("Quaternion", "Protobuf Roundtrip")
def test_quaternion_protobuf_roundtrip():
    from session_py import Quaternion
    from session_py import Vector
    from pathlib import Path

    q = Quaternion.from_axis_angle(Vector(0.0, 0.0, 1.0), PI / 2.0)
    q.name = "test_quaternion"

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_quaternion.bin"
    q.pb_dump(path)
    loaded = Quaternion.pb_load(path)

    MINI_CHECK(loaded.name == "test_quaternion")
    MINI_CHECK(TOLERANCE.is_close(loaded.scalar, q.scalar))
    MINI_CHECK(TOLERANCE.is_close(loaded.vector[2], q.vector[2]))


if __name__ == "__main__":
    run_all(language="python")
