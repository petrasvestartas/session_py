from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("Xform", "Constructor")
def test_xform_constructor():
    from session_py import Xform
    from session_py import Point

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

    # Minimal and Full String Representation
    xstr = str(x)
    xrepr = repr(x)

    # Copy (duplicates everything except guid)
    xcopy = x.duplicate()
    xother = Xform()

    # Equality operators
    x_eq = (x == xother)
    x_ne = (x != xfrom)

    # Matrix multiplication (*)
    t = Xform.translation(10.0, 0.0, 0.0)
    s = Xform.scaling(2.0, 1.0, 1.0)
    combined = t * s
    p = Point(1.0, 0.0, 0.0)
    result = combined.transformed_point(p)

    # In-place multiplication (*=)
    t2 = Xform.translation(10.0, 0.0, 0.0)
    t2 *= s
    result2 = t2.transformed_point(p)

    MINI_CHECK(x.name == "my_xform" and x.guid != "")
    MINI_CHECK(m00 == 1.0 and m11 == 1.0 and m22 == 1.0 and m33 == 1.0)
    MINI_CHECK(is_id == True)
    MINI_CHECK(xfrom.m[12] == 5.0 and xfrom.m[13] == 10.0 and xfrom.m[14] == 15.0)
    MINI_CHECK("1.000000" in xstr)
    MINI_CHECK("Xform(" in xrepr and "my_xform" in xrepr)
    MINI_CHECK(xcopy == x and xcopy.guid != x.guid)
    MINI_CHECK(x_eq == True and x_ne == True)
    # (1,0,0) * scale(2,1,1) = (2,0,0), then translate(10,0,0) = (12,0,0)
    MINI_CHECK(TOLERANCE.is_close(result[0], 12.0) and TOLERANCE.is_close(result[1], 0.0) and TOLERANCE.is_close(result[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(result2[0], 12.0) and TOLERANCE.is_close(result2[1], 0.0) and TOLERANCE.is_close(result2[2], 0.0))


@MINI_TEST("Xform", "Translation")
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


@MINI_TEST("Xform", "Scaling")
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


@MINI_TEST("Xform", "Rotation")
def test_xform_rotation():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    # Rotation around X axis by 90 degrees
    rx = Xform.rotation_x(PI / 2.0)
    # Apply to point (0,1,0) -> (0,0,1)
    px = Point(0.0, 1.0, 0.0)
    rpx = rx.transformed_point(px)

    # Rotation around Y axis by 90 degrees
    ry = Xform.rotation_y(PI / 2.0)
    # Apply to point (0,0,1) -> (1,0,0)
    py = Point(0.0, 0.0, 1.0)
    rpy = ry.transformed_point(py)

    # Rotation around Z axis by 90 degrees
    rz = Xform.rotation_z(PI / 2.0)
    # Apply to point (1,0,0) -> (0,1,0)
    pz = Point(1.0, 0.0, 0.0)
    rpz = rz.transformed_point(pz)

    # Rotation around arbitrary axis (1,1,1) by 120 degrees
    # This cycles x->y->z->x
    axis = Vector(1.0, 1.0, 1.0)
    r = Xform.rotation(axis, 2.0 * PI / 3.0)
    # Apply to point (1,0,0) -> (0,1,0)
    p = Point(1.0, 0.0, 0.0)
    rp = r.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(rpx[0], 0.0) and TOLERANCE.is_close(rpx[1], 0.0) and TOLERANCE.is_close(rpx[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(rpy[0], 1.0) and TOLERANCE.is_close(rpy[1], 0.0) and TOLERANCE.is_close(rpy[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(rpz[0], 0.0) and TOLERANCE.is_close(rpz[1], 1.0) and TOLERANCE.is_close(rpz[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(rp[0], 0.0) and TOLERANCE.is_close(rp[1], 1.0) and TOLERANCE.is_close(rp[2], 0.0))


@MINI_TEST("Xform", "Inverse")
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


@MINI_TEST("Xform", "Transform_geometry")
def test_xform_transform_geometry():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector
    from session_py import Line
    from session_py import Plane
    from session_py import Polyline

    # Simple translation by (10, 20, 30)
    t = Xform.translation(10.0, 20.0, 30.0)

    # Transform Point: (1,2,3) -> (11,22,33)
    pt = Point(1.0, 2.0, 3.0)
    pt.xform = t.duplicate()
    pt_transformed = pt.transformed()

    # Transform Vector: translation should NOT affect vectors
    v = Vector(1.0, 0.0, 0.0)
    v_transformed = t.transformed_vector(v)

    # Transform Line: (0,0,0)-(1,0,0) -> (10,20,30)-(11,20,30)
    ln = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    ln.xform = t.duplicate()
    ln_transformed = ln.transformed()

    # Transform Plane: origin (0,0,0) -> (10,20,30)
    pl = Plane(Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0))
    pl.xform = t.duplicate()
    pl_transformed = pl.transformed()

    # Transform Polyline: 3 points translated
    poly = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0)])
    poly.xform = t.duplicate()
    poly_transformed = poly.transformed()
    pts = poly_transformed.get_points()

    MINI_CHECK(TOLERANCE.is_close(pt_transformed[0], 11.0) and TOLERANCE.is_close(pt_transformed[1], 22.0) and TOLERANCE.is_close(pt_transformed[2], 33.0))
    MINI_CHECK(TOLERANCE.is_close(v_transformed[0], 1.0) and TOLERANCE.is_close(v_transformed[1], 0.0) and TOLERANCE.is_close(v_transformed[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(ln_transformed[0], 10.0) and TOLERANCE.is_close(ln_transformed[1], 20.0) and TOLERANCE.is_close(ln_transformed[2], 30.0))
    MINI_CHECK(TOLERANCE.is_close(ln_transformed[3], 11.0) and TOLERANCE.is_close(ln_transformed[4], 20.0) and TOLERANCE.is_close(ln_transformed[5], 30.0))
    MINI_CHECK(TOLERANCE.is_close(pl_transformed.origin[0], 10.0) and TOLERANCE.is_close(pl_transformed.origin[1], 20.0) and TOLERANCE.is_close(pl_transformed.origin[2], 30.0))
    MINI_CHECK(TOLERANCE.is_close(pts[0][0], 10.0) and TOLERANCE.is_close(pts[0][1], 20.0) and TOLERANCE.is_close(pts[0][2], 30.0))
    MINI_CHECK(TOLERANCE.is_close(pts[1][0], 11.0) and TOLERANCE.is_close(pts[1][1], 20.0) and TOLERANCE.is_close(pts[1][2], 30.0))
    MINI_CHECK(TOLERANCE.is_close(pts[2][0], 11.0) and TOLERANCE.is_close(pts[2][1], 21.0) and TOLERANCE.is_close(pts[2][2], 30.0))


@MINI_TEST("Xform", "Change_basis")
def test_xform_change_basis():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    # System 0: standard XY plane at origin
    origin_0 = Point(0.0, 0.0, 0.0)
    x_axis_0 = Vector(1.0, 0.0, 0.0)
    y_axis_0 = Vector(0.0, 1.0, 0.0)
    z_axis_0 = Vector(0.0, 0.0, 1.0)

    # System 1: translated and rotated 90 degrees around Z
    origin_1 = Point(10.0, 20.0, 0.0)
    x_axis_1 = Vector(0.0, 1.0, 0.0)
    y_axis_1 = Vector(-1.0, 0.0, 0.0)
    z_axis_1 = Vector(0.0, 0.0, 1.0)

    # Transform maps points FROM system 1 TO system 0
    xform = Xform.change_basis(origin_1, x_axis_1, y_axis_1, z_axis_1, origin_0, x_axis_0, y_axis_0, z_axis_0)

    # Point at origin_1 should map to origin_0
    p = Point(10.0, 20.0, 0.0)
    tp = xform.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(tp[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(tp[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(tp[2], 0.0))


@MINI_TEST("Xform", "Plane_to_plane")
def test_xform_plane_to_plane():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector
    from session_py import Plane

    # Source plane at origin, XY plane
    plane_from = Plane(Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0))

    # Target plane translated and rotated
    plane_to = Plane(Point(10.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(-1.0, 0.0, 0.0))

    xform = Xform.plane_to_plane(plane_from, plane_to)

    # Origin of source should map to origin of target
    p = Point(0.0, 0.0, 0.0)
    tp = xform.transformed_point(p)

    MINI_CHECK(TOLERANCE.is_close(tp[0], 10.0))
    MINI_CHECK(TOLERANCE.is_close(tp[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(tp[2], 0.0))


@MINI_TEST("Xform", "Look_at_rh")
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


@MINI_TEST("Xform", "Json_roundtrip")
def test_xform_json_roundtrip():
    from session_py import Xform
    from pathlib import Path

    # Create a non-identity xform
    xform = Xform.translation(1.0, 2.0, 3.0)
    xform.name = "test_xform"

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   json_dumps()    │ str          │ to JSON string
    #   json_loads(s)   │ str          │ from JSON string
    #   json_dump(path) │ file         │ write to file
    #   json_load(path) │ file         │ read from file

    # json_dump(fname) / json_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_xform.json"
    xform.json_dump(fname)
    loaded = Xform.json_load(fname)

    MINI_CHECK(loaded.name == "test_xform")
    MINI_CHECK(TOLERANCE.is_close(loaded.m[12], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.m[13], 2.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.m[14], 3.0))


@MINI_TEST("Xform", "Protobuf_roundtrip")
def test_xform_protobuf_roundtrip():
    from session_py import Xform
    from pathlib import Path

    # Create a non-identity xform
    xform = Xform.translation(1.0, 2.0, 3.0)
    xform.name = "test_xform_proto"

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    # pb_dump(fname) / pb_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_xform.bin"
    xform.pb_dump(fname)
    loaded = Xform.pb_load(fname)

    MINI_CHECK(loaded.name == "test_xform_proto")
    MINI_CHECK(TOLERANCE.is_close(loaded.m[12], 1.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.m[13], 2.0))
    MINI_CHECK(TOLERANCE.is_close(loaded.m[14], 3.0))


if __name__ == "__main__":
    run_all("python")
