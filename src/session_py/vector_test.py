import math
from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI
from .tolerance import SCALE


@MINI_TEST("Vector", "Constructor")
def test_vector_constructor():
    from session_py import Vector
    from session_py import Point

    v = Vector(1.0, 2.0, 3.0)
    p0 = Point(1.0, 2.0, 3.0)
    p1 = Point(2.0, 4.0, 6.0)
    v_2p = Vector.from_points(p0, p1)

    v[0] = 10.0
    v[1] = 20.0
    v[2] = 30.0

    x = v[0]
    y = v[1]
    z = v[2]

    vstr = str(v)
    vrepr = repr(v)

    vcopy = v.duplicate()
    vother = Vector(1.0, 2.0, 3.0)

    vmult = v.duplicate()
    vmult *= 2.0

    vdiv = v.duplicate()
    vdiv /= 2.0

    vadd = v.duplicate()
    vadd += Vector(1.0, 1.0, 1.0)

    vsub = v.duplicate()
    vsub -= Vector(1.0, 1.0, 1.0)

    result_mul = v * 2.0
    result_div = v / 2.0
    result_add = v + Vector(1.0, 1.0, 1.0)
    result_sub = v - Vector(1.0, 1.0, 1.0)
    result_neg = -v

    vx = Vector.x_axis()
    vy = Vector.y_axis()
    vz = Vector.z_axis()
    vzero = Vector.zero()

    MINI_CHECK(v.name == "my_vector")
    MINI_CHECK(v[0] == 10.0 and v[1] == 20.0 and v[2] == 30.0)
    MINI_CHECK(v.guid != "")
    MINI_CHECK(x == 10.0 and y == 20.0 and z == 30.0)
    MINI_CHECK(v_2p[0] == 1.0 and v_2p[1] == 2.0 and v_2p[2] == 3.0)
    MINI_CHECK(vstr == "10.000000, 20.000000, 30.000000")
    MINI_CHECK(vrepr == "Vector(my_vector, 10.000000, 20.000000, 30.000000, 37.416574)")
    MINI_CHECK(vcopy == v and vcopy.guid != v.guid)
    MINI_CHECK(vother != v)
    MINI_CHECK(vmult[0] == 20.0 and vmult[1] == 40.0 and vmult[2] == 60.0)
    MINI_CHECK(vdiv[0] == 5.0 and vdiv[1] == 10.0 and vdiv[2] == 15.0)
    MINI_CHECK(vadd[0] == 11.0 and vadd[1] == 21.0 and vadd[2] == 31.0)
    MINI_CHECK(vsub[0] == 9.0 and vsub[1] == 19.0 and vsub[2] == 29.0)
    MINI_CHECK(
        result_mul[0] == 20.0 and result_mul[1] == 40.0 and result_mul[2] == 60.0
    )
    MINI_CHECK(result_div[0] == 5.0 and result_div[1] == 10.0 and result_div[2] == 15.0)
    MINI_CHECK(
        result_add[0] == 11.0 and result_add[1] == 21.0 and result_add[2] == 31.0
    )
    MINI_CHECK(result_sub[0] == 9.0 and result_sub[1] == 19.0 and result_sub[2] == 29.0)
    MINI_CHECK(
        result_neg[0] == -10.0 and result_neg[1] == -20.0 and result_neg[2] == -30.0
    )
    MINI_CHECK(vx[0] == 1.0 and vx[1] == 0.0 and vx[2] == 0.0)
    MINI_CHECK(vy[0] == 0.0 and vy[1] == 1.0 and vy[2] == 0.0)
    MINI_CHECK(vz[0] == 0.0 and vz[1] == 0.0 and vz[2] == 1.0)
    MINI_CHECK(vzero[0] == 0.0 and vzero[1] == 0.0 and vzero[2] == 0.0)


@MINI_TEST("Vector", "Transformation")
def test_vector_transformation():
    from session_py import Vector
    from session_py import Xform

    v = Vector(1.0, 2.0, 3.0)
    xform = Xform.translation(10.0, 20.0, 30.0)
    moved = v.transformed(xform)
    v.transform(xform)

    v2 = Vector(1.0, 0.0, 0.0)
    rotation = Xform.rotation_z(PI / 2.0)
    v2.transform(rotation)

    MINI_CHECK(moved[0] == 1.0 and moved[1] == 2.0 and moved[2] == 3.0)
    MINI_CHECK(v[0] == 1.0 and v[1] == 2.0 and v[2] == 3.0)
    MINI_CHECK(
        TOLERANCE.is_close(v2[0], 0.0)
        and TOLERANCE.is_close(v2[1], 1.0)
        and TOLERANCE.is_close(v2[2], 0.0)
    )


@MINI_TEST("Vector", "Magnitude")
def test_vector_magnitude():
    from session_py import Vector

    v = Vector(3.0, 4.0, 0.0)
    length = v.magnitude()
    length_squared = v.magnitude_squared()

    MINI_CHECK(length == 5.0)
    MINI_CHECK(length_squared == 25.0)


@MINI_TEST("Vector", "Normalize")
def test_vector_normalize():
    from session_py import Vector

    v0 = Vector(3.0, 4.0, 0.0)
    ok = v0.normalize_self()

    v1 = Vector(3.0, 4.0, 0.0)
    v2 = v1.normalized()

    zero = Vector(0.0, 0.0, 0.0)
    zero_ok = zero.normalize_self()

    MINI_CHECK(ok and TOLERANCE.is_close(v0.magnitude(), 1.0))
    MINI_CHECK(TOLERANCE.is_close(v2.magnitude(), 1.0))
    MINI_CHECK(not zero_ok)


@MINI_TEST("Vector", "Reverse")
def test_vector_reverse():
    from session_py import Vector

    v = Vector(1.0, -2.0, 3.0)
    v.reverse()

    MINI_CHECK(v[0] == -1.0 and v[1] == 2.0 and v[2] == -3.0)


@MINI_TEST("Vector", "Dot Product")
def test_vector_dot_product():
    from session_py import Vector

    v1 = Vector(1.0, 0.0, 0.0)
    v2 = Vector(0.0, 1.0, 0.0)
    v3 = Vector(1.0, 0.0, 0.0)
    dot_perp = v1.dot(v2)
    dot_paral = v1.dot(v3)

    a = Vector(3.0, 4.0, 0.0)
    b = Vector(1.0, 0.0, 0.0)
    b2 = Vector(2.0, 0.0, 0.0)
    proj_scalar = a.dot(b) / b.magnitude()
    proj_coeff = a.dot(b2) / b2.magnitude_squared()

    MINI_CHECK(TOLERANCE.is_close(dot_perp, 0.0))
    MINI_CHECK(TOLERANCE.is_close(dot_paral, 1.0))
    MINI_CHECK(TOLERANCE.is_close(proj_scalar, 3.0))
    MINI_CHECK(TOLERANCE.is_close(proj_coeff, 1.5))


@MINI_TEST("Vector", "Cross Product")
def test_vector_cross_product():
    from session_py import Vector

    v1 = Vector(1.0, 0.0, 0.0)
    v2 = Vector(0.0, 1.0, 0.0)
    vn = v1.cross(v2)

    a = Vector(3.0, 0.0, 0.0)
    b = Vector(0.0, 4.0, 0.0)
    area = a.cross(b).magnitude()

    MINI_CHECK(vn[0] == 0.0 and vn[1] == 0.0 and vn[2] == 1.0)
    MINI_CHECK(TOLERANCE.is_close(area, 12.0))


@MINI_TEST("Vector", "Angle")
def test_vector_angle():
    from session_py import Vector

    v1 = Vector(1.0, 0.0, 0.0)
    v2 = Vector(0.0, 1.0, 0.0)
    v3 = Vector(1.0, 1.0, 0.0)
    angle_90 = v1.angle(v2, False)
    angle_45 = v1.angle(v3, False)
    angle_signed = v2.angle(v1, True)
    angle_rad = v1.angle(v2, False, False)

    v_30 = Vector(math.sqrt(3.0), 1.0, 0.0)
    v_60 = Vector(1.0, math.sqrt(3.0), 0.0)
    v_135 = Vector(-1.0, 1.0, 0.0)
    xy_angle_30 = Vector.angle_between_vector_xy_components(v_30)
    xy_angle_60 = Vector.angle_between_vector_xy_components(v_60)
    xy_angle_135 = Vector.angle_between_vector_xy_components(v_135)

    v_dir = Vector(35.4, 35.4, 86.6)
    abg = v_dir.coordinate_direction_3angles(True)

    v_sph = Vector(1.0, 1.0, math.sqrt(2.0))
    pt = v_sph.coordinate_direction_2angles(True)

    MINI_CHECK(TOLERANCE.is_close(angle_90, 90.0))
    MINI_CHECK(TOLERANCE.is_close(angle_45, 45.0))
    MINI_CHECK(TOLERANCE.is_close(angle_signed, -90.0))
    MINI_CHECK(TOLERANCE.is_close(angle_rad, PI / 2.0))
    MINI_CHECK(TOLERANCE.is_close(xy_angle_30, 30.0))
    MINI_CHECK(TOLERANCE.is_close(xy_angle_60, 60.0))
    MINI_CHECK(TOLERANCE.is_close(xy_angle_135, 135.0))
    MINI_CHECK(TOLERANCE.is_close(abg[0], 69.2742))
    MINI_CHECK(TOLERANCE.is_close(abg[1], 69.2742))
    MINI_CHECK(TOLERANCE.is_close(abg[2], 30.032058))
    MINI_CHECK(TOLERANCE.is_close(pt[0], 45.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 45.0))


@MINI_TEST("Vector", "Projection")
def test_vector_projection():
    from session_py import Vector

    v = Vector(1.0, 1.0, 1.0)
    x = Vector.x_axis()
    y = Vector.y_axis()
    z = Vector.z_axis()

    proj_x, len_x, perp_x, perp_len_x = v.projection(x)
    proj_y, len_y, _, _ = v.projection(y)
    proj_z, len_z, _, _ = v.projection(z)

    MINI_CHECK(proj_x[0] == 1.0 and proj_x[1] == 0.0 and proj_x[2] == 0.0)
    MINI_CHECK(proj_y[0] == 0.0 and proj_y[1] == 1.0 and proj_y[2] == 0.0)
    MINI_CHECK(proj_z[0] == 0.0 and proj_z[1] == 0.0 and proj_z[2] == 1.0)
    MINI_CHECK(
        TOLERANCE.is_close(len_x, 1.0)
        and TOLERANCE.is_close(len_y, 1.0)
        and TOLERANCE.is_close(len_z, 1.0)
    )
    MINI_CHECK(perp_x[0] == 0.0 and perp_x[1] == 1.0 and perp_x[2] == 1.0)
    MINI_CHECK(TOLERANCE.is_close(perp_len_x, math.sqrt(2.0)))


@MINI_TEST("Vector", "Is Parallel To")
def test_vector_is_parallel_to():
    from session_py import Vector

    v1 = Vector(2.0, 2.0, 2.0)
    v2 = Vector(4.0, 4.0, 4.0)
    v3 = Vector(-1.0, -1.0, -1.0)
    v4 = Vector(1.0, 0.0, 0.0)

    MINI_CHECK(v1.is_parallel_to(v2) == 1)
    MINI_CHECK(v1.is_parallel_to(v3) == -1)
    MINI_CHECK(v1.is_parallel_to(v4) == 0)


@MINI_TEST("Vector", "Is Perpendicular To")
def test_vector_is_perpendicular_to():
    from session_py import Vector

    v1 = Vector(1.0, 0.0, 0.0)
    v2 = Vector(0.0, 1.0, 0.0)
    v3 = Vector(0.0, 0.0, 1.0)
    v4 = Vector(1.0, 1.0, 0.0)

    z_axis = Vector(0.0, 0.0, 1.0)
    x_axis = Vector.zero()
    x_ok = x_axis.perpendicular_to(z_axis)

    arbitrary = Vector(1.0, 2.0, 3.0)
    perp = Vector.zero()
    perp_ok = perp.perpendicular_to(arbitrary)

    MINI_CHECK(v1.is_perpendicular_to(v2))
    MINI_CHECK(v1.is_perpendicular_to(v3))
    MINI_CHECK(not v1.is_perpendicular_to(v4))
    MINI_CHECK(x_ok and x_axis.is_perpendicular_to(z_axis))
    MINI_CHECK(perp_ok and perp.is_perpendicular_to(arbitrary))


@MINI_TEST("Vector", "Get Leveled Vector")
def test_vector_get_leveled_vector():
    from session_py import Vector

    v = Vector(1.0, 1.0, 1.0)
    vertical_height = 1.0
    leveled = v.get_leveled_vector(vertical_height)

    MINI_CHECK(TOLERANCE.is_close(leveled.magnitude(), math.sqrt(3.0)))
    MINI_CHECK(TOLERANCE.is_close(leveled[2], vertical_height))


@MINI_TEST("Vector", "Cos Sin Laws")
def test_vector_cos_sin_laws():
    from session_py import Vector

    a = 3.0
    b = 4.0
    c = 5.0

    angle_a = Vector.angle_from_cosine_law(b, c, a, True)
    angle_b = Vector.angle_from_cosine_law(a, c, b, True)
    angle_c = Vector.angle_from_cosine_law(a, b, c, True)

    side_a = Vector.side_from_sine_law(angle_a, angle_b, b, True)
    side_b = Vector.side_from_sine_law(angle_b, angle_c, c, True)
    side_c = Vector.side_from_sine_law(angle_c, angle_a, a, True)

    computed_c = Vector.cosine_law(a, b, angle_c, True)
    computed_a = Vector.cosine_law(b, c, angle_a, True)
    computed_b = Vector.cosine_law(a, c, angle_b, True)

    computed_angle_b = Vector.sine_law_angle(a, angle_a, b, True)
    computed_angle_a = Vector.sine_law_angle(b, angle_b, a, True)

    computed_side_b = Vector.sine_law_length(a, angle_a, angle_b, True)
    computed_side_a = Vector.sine_law_length(b, angle_b, angle_a, True)

    MINI_CHECK(TOLERANCE.is_close(angle_a, 36.86989764584402))
    MINI_CHECK(TOLERANCE.is_close(angle_b, 53.13010235415599))
    MINI_CHECK(TOLERANCE.is_close(angle_c, 90.0))
    MINI_CHECK(TOLERANCE.is_close(angle_a + angle_b + angle_c, 180.0))
    MINI_CHECK(TOLERANCE.is_close(side_a, a))
    MINI_CHECK(TOLERANCE.is_close(side_b, b))
    MINI_CHECK(TOLERANCE.is_close(side_c, c))
    MINI_CHECK(TOLERANCE.is_close(computed_c, c))
    MINI_CHECK(TOLERANCE.is_close(computed_a, a))
    MINI_CHECK(TOLERANCE.is_close(computed_b, b))
    MINI_CHECK(TOLERANCE.is_close(computed_angle_b, angle_b))
    MINI_CHECK(TOLERANCE.is_close(computed_angle_a, angle_a))
    MINI_CHECK(TOLERANCE.is_close(computed_side_b, b))
    MINI_CHECK(TOLERANCE.is_close(computed_side_a, a))


@MINI_TEST("Vector", "Sum Of Vectors")
def test_vector_sum_of_vectors():
    from session_py import Vector

    vecs = [
        Vector(1.0, 1.0, 1.0),
        Vector(2.0, 2.0, 2.0),
        Vector(3.0, 3.0, 3.0),
    ]
    sum = Vector.sum_of_vectors(vecs)

    empty = []
    zero = Vector.sum_of_vectors(empty)

    MINI_CHECK(sum[0] == 6.0 and sum[1] == 6.0 and sum[2] == 6.0)
    MINI_CHECK(zero[0] == 0.0 and zero[1] == 0.0 and zero[2] == 0.0)


@MINI_TEST("Vector", "Average")
def test_vector_average():
    from session_py import Vector

    vecs = [
        Vector(1.0, 2.0, 3.0),
        Vector(3.0, 4.0, 5.0),
        Vector(5.0, 6.0, 7.0),
    ]
    avg = Vector.average(vecs)

    empty = []
    zero = Vector.average(empty)

    MINI_CHECK(avg[0] == 3.0 and avg[1] == 4.0 and avg[2] == 5.0)
    MINI_CHECK(zero[0] == 0.0 and zero[1] == 0.0 and zero[2] == 0.0)


@MINI_TEST("Vector", "Is Zero")
def test_vector_is_zero():
    from session_py import Vector

    zero = Vector(0.0, 0.0, 0.0)
    nonzero = Vector(1.0, 0.0, 0.0)
    tiny = Vector(1e-13, 1e-13, 1e-13)

    MINI_CHECK(zero.is_zero())
    MINI_CHECK(not nonzero.is_zero())
    MINI_CHECK(tiny.is_zero())


@MINI_TEST("Vector", "Scale")
def test_vector_scale():
    from session_py import Vector

    v_up = Vector(1.0, 2.0, 3.0)
    v_up.scale_up()

    v_rt = Vector(1.0, 2.0, 3.0)
    v_rt.scale_up()
    v_rt.scale_down()

    MINI_CHECK(v_up[0] == SCALE)
    MINI_CHECK(
        TOLERANCE.is_close(v_rt[0], 1.0)
        and TOLERANCE.is_close(v_rt[1], 2.0)
        and TOLERANCE.is_close(v_rt[2], 3.0)
    )


@MINI_TEST("Vector", "Reflect")
def test_vector_reflect():
    from session_py import Vector

    v = Vector(1.0, 2.0, 3.0)
    n = Vector.x_axis()
    r = v.reflect(n)

    MINI_CHECK(TOLERANCE.is_close(r[0], -1.0))
    MINI_CHECK(TOLERANCE.is_close(r[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(r[2], 3.0))


@MINI_TEST("Vector", "Average Normal")
def test_vector_average_normal():
    from session_py import Point
    from session_py import Vector

    square = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 0.0),
    ]
    n = Vector.average_normal(square)
    empty = Vector.average_normal([])

    MINI_CHECK(TOLERANCE.is_close(abs(n[2]), 1.0))
    MINI_CHECK(TOLERANCE.is_close(n[0], 0.0) and TOLERANCE.is_close(n[1], 0.0))
    MINI_CHECK(empty.is_zero())


@MINI_TEST("Vector", "Average Normal Polyline")
def test_vector_average_normal_polyline():
    from session_py import Point
    from session_py import Polyline
    from session_py import Vector

    square = Polyline([
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 0.0),
    ])
    n = Vector.average_normal_polyline(square)
    empty = Vector.average_normal_polyline(Polyline())

    MINI_CHECK(TOLERANCE.is_close(abs(n[2]), 1.0))
    MINI_CHECK(TOLERANCE.is_close(n[0], 0.0) and TOLERANCE.is_close(n[1], 0.0))
    MINI_CHECK(empty.is_zero())


@MINI_TEST("Vector", "Json Roundtrip")
def test_vector_json_roundtrip():
    from session_py import Vector
    from pathlib import Path

    v = Vector(42.1, 84.2, 126.3)
    v.name = "test_vector"

    filename = (
        Path(__file__).resolve().parents[2] / "serialization" / "test_vector.json"
    )
    v.file_json_dump(filename)
    loaded = Vector.file_json_load(filename)

    json_string = v.file_json_dumps()
    parsed = Vector.file_json_loads(json_string)

    MINI_CHECK(loaded.name == "test_vector")
    MINI_CHECK(loaded.guid == v.guid)
    MINI_CHECK(TOLERANCE.is_close(loaded[0], 42.1))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], 84.2))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], 126.3))
    MINI_CHECK(parsed == v)


@MINI_TEST("Vector", "Protobuf Roundtrip")
def test_vector_protobuf_roundtrip():
    from session_py import Vector
    from pathlib import Path

    v = Vector(42.1, 84.2, 126.3)
    v.name = "test_vector"

    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_vector.bin"
    v.pb_dump(filename)
    loaded = Vector.pb_load(filename)

    data = v.pb_dumps()
    parsed = Vector.pb_loads(data)
    converted = Vector.from_proto(v.to_proto())

    MINI_CHECK(loaded.name == "test_vector")
    MINI_CHECK(TOLERANCE.is_close(loaded[0], 42.1))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], 84.2))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], 126.3))
    MINI_CHECK(parsed == v)
    MINI_CHECK(converted == v)


if __name__ == "__main__":
    run_all("python")
