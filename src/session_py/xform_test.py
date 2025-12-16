from .mini_test import MINI_TEST, MINI_CHECK, run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("Xform", "constructor")
def test_xform_constructor():
    from session_py import Xform

    # Constructor (identity by default)
    x = Xform()

    # Matrix access
    m00 = x.m[0]
    m11 = x.m[5]
    m22 = x.m[10]
    m33 = x.m[15]

    # Check identity
    is_id = x.is_identity()

    # From matrix constructor
    xfrom = Xform.from_matrix([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 10.0, 15.0, 1.0])

    MINI_CHECK(x.name == "my_xform" and x.guid != "")
    MINI_CHECK(m00 == 1.0 and m11 == 1.0 and m22 == 1.0 and m33 == 1.0)
    MINI_CHECK(is_id == True)
    MINI_CHECK(xfrom.m[12] == 5.0 and xfrom.m[13] == 10.0 and xfrom.m[14] == 15.0)


@MINI_TEST("Xform", "translation")
def test_xform_translation():
    from session_py import Xform
    from session_py import Point

    # Translation matrix
    t = Xform.translation(1.0, 2.0, 3.0)

    # Apply to point
    p = Point(4.0, 5.0, 6.0)
    tp = t.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(tp[0], 5.0))
    MINI_CHECK(TOLERANCE.is_close(tp[1], 7.0))
    MINI_CHECK(TOLERANCE.is_close(tp[2], 9.0))


@MINI_TEST("Xform", "scaling")
def test_xform_scaling():
    from session_py import Xform
    from session_py import Point

    # Scaling matrix
    s = Xform.scaling(2.0, 3.0, 4.0)

    # Apply to point
    p = Point(1.0, 1.0, 1.0)
    sp = s.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(sp[0], 2.0))
    MINI_CHECK(TOLERANCE.is_close(sp[1], 3.0))
    MINI_CHECK(TOLERANCE.is_close(sp[2], 4.0))


@MINI_TEST("Xform", "rotation_z")
def test_xform_rotation_z():
    from session_py import Xform
    from session_py import Point

    # Rotation around Z axis by 90 degrees
    r = Xform.rotation_z(PI / 2.0)

    # Apply to point (1,0,0) -> (0,1,0)
    p = Point(1.0, 0.0, 0.0)
    rp = r.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(rp[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(rp[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(rp[2], 0.0))


@MINI_TEST("Xform", "inverse")
def test_xform_inverse():
    from session_py import Xform

    # Create composite transformation
    t = Xform.translation(1.0, 2.0, 3.0)
    s = Xform.scaling(2.0, 2.0, 2.0)
    composite = t * s

    # Compute inverse
    inv = composite.inverse()

    # Multiply should give identity
    result = composite * inv

    MINI_CHECK(result.is_identity())


@MINI_TEST("Xform", "mul_operator")
def test_xform_mul_operator():
    from session_py import Xform
    from session_py import Point

    # Matrix multiplication
    t = Xform.translation(10.0, 0.0, 0.0)
    s = Xform.scaling(2.0, 1.0, 1.0)

    # Combined: first scale, then translate
    combined = t * s

    # Apply to point
    p = Point(1.0, 0.0, 0.0)
    result = combined.transformed_point(p)

    # (1,0,0) * scale(2,1,1) = (2,0,0), then translate(10,0,0) = (12,0,0)
    MINI_CHECK(TOLERANCE.is_close(result[0], 12.0))
    MINI_CHECK(TOLERANCE.is_close(result[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(result[2], 0.0))


@MINI_TEST("Xform", "transform_vector")
def test_xform_transform_vector():
    from session_py import Xform
    from session_py import Vector

    # Translation should not affect vectors (only direction)
    t = Xform.translation(100.0, 200.0, 300.0)
    v = Vector(1.0, 0.0, 0.0)
    tv = t.transformed_vector(v)

    # Scaling should affect vectors
    s = Xform.scaling(2.0, 3.0, 4.0)
    v2 = Vector(1.0, 1.0, 1.0)
    sv = s.transformed_vector(v2)

    MINI_CHECK(TOLERANCE.is_close(tv[0], 1.0) and TOLERANCE.is_close(tv[1], 0.0) and TOLERANCE.is_close(tv[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(sv[0], 2.0) and TOLERANCE.is_close(sv[1], 3.0) and TOLERANCE.is_close(sv[2], 4.0))


if __name__ == "__main__":
    run_all("python")
