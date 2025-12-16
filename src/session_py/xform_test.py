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


@MINI_TEST("Xform", "rotation_x")
def test_xform_rotation_x():
    from session_py import Xform
    from session_py import Point

    # Rotation around X axis by 90 degrees
    r = Xform.rotation_x(PI / 2.0)

    # Apply to point (0,1,0) -> (0,0,1)
    p = Point(0.0, 1.0, 0.0)
    rp = r.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(rp[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(rp[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(rp[2], 1.0))


@MINI_TEST("Xform", "rotation_y")
def test_xform_rotation_y():
    from session_py import Xform
    from session_py import Point

    # Rotation around Y axis by 90 degrees
    r = Xform.rotation_y(PI / 2.0)

    # Apply to point (0,0,1) -> (1,0,0)
    p = Point(0.0, 0.0, 1.0)
    rp = r.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(rp[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(rp[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(rp[2], 0.0))


@MINI_TEST("Xform", "rotation")
def test_xform_rotation():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    # Rotation around arbitrary axis (1,1,1) by 120 degrees
    # This cycles x->y->z->x
    axis = Vector(1.0, 1.0, 1.0)
    r = Xform.rotation(axis, 2.0 * PI / 3.0)

    # Apply to point (1,0,0) -> (0,1,0)
    p = Point(1.0, 0.0, 0.0)
    rp = r.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(rp[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(rp[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(rp[2], 0.0))


@MINI_TEST("Xform", "change_basis")
def test_xform_change_basis():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    # Create a coordinate system at origin with rotated axes
    origin = Point(10.0, 20.0, 30.0)
    x_axis = Vector(1.0, 0.0, 0.0)
    y_axis = Vector(0.0, 1.0, 0.0)
    z_axis = Vector(0.0, 0.0, 1.0)

    # Change basis transform
    xform = Xform.change_basis(origin, x_axis, y_axis, z_axis)

    # Point at local origin should map to world origin
    p = Point(10.0, 20.0, 30.0)
    tp = xform.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(tp[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(tp[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(tp[2], 0.0))


@MINI_TEST("Xform", "plane_to_plane")
def test_xform_plane_to_plane():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    # Source plane at origin, XY plane
    origin_0 = Point(0.0, 0.0, 0.0)
    x_axis_0 = Vector(1.0, 0.0, 0.0)
    y_axis_0 = Vector(0.0, 1.0, 0.0)
    z_axis_0 = Vector(0.0, 0.0, 1.0)

    # Target plane translated and rotated
    origin_1 = Point(10.0, 0.0, 0.0)
    x_axis_1 = Vector(0.0, 1.0, 0.0)
    y_axis_1 = Vector(-1.0, 0.0, 0.0)
    z_axis_1 = Vector(0.0, 0.0, 1.0)

    xform = Xform.plane_to_plane(origin_0, x_axis_0, y_axis_0, z_axis_0, origin_1, x_axis_1, y_axis_1, z_axis_1)

    # Origin of source should map to origin of target
    p = Point(0.0, 0.0, 0.0)
    tp = xform.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(tp[0], 10.0))
    MINI_CHECK(TOLERANCE.is_close(tp[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(tp[2], 0.0))


@MINI_TEST("Xform", "look_at_rh")
def test_xform_look_at_rh():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    # Camera at (0,0,10) looking at origin
    eye = Point(0.0, 0.0, 10.0)
    target = Point(0.0, 0.0, 0.0)
    up = Vector(0.0, 1.0, 0.0)

    xform = Xform.look_at_rh(eye, target, up)

    # The target point should be on the negative Z axis in view space
    tp = xform.transformed_point(target)

    MINI_CHECK(TOLERANCE.is_close(tp[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(tp[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(tp[2], -10.0))


@MINI_TEST("Xform", "json_roundtrip")
def test_xform_json_roundtrip():
    from session_py import Xform
    from pathlib import Path

    # Create a non-identity xform
    xform = Xform.translation(1.0, 2.0, 3.0)
    xform.name = "test_xform"

    # json_dump(fname) / json_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "test_xform.json"
    xform.json_dump(fname)
    loaded = Xform.json_load(fname)

    MINI_CHECK(loaded.name == "test_xform")
    MINI_CHECK(TOLERANCE.is_close(loaded.m[12], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.m[13], 2.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.m[14], 3.0))


if __name__ == "__main__":
    run_all("python")
