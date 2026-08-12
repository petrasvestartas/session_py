import math
from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Vector", "Constructor")
def test_vector_constructor():
    from session_py import Vector, Point

    # Constructor
    v = Vector(1.0, 2.0, 3.0)
    p0 = Point(1.0, 2.0, 3.0)
    p1 = Point(2.0, 4.0, 6.0)
    v_2p = Vector.from_points(p0, p1)

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

    MINI_CHECK(v.name == "my_vector" and v[0] == 10.0 and v[1] == 20.0 and v[2] == 30.0 and v.guid)
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
    MINI_CHECK(result_mul[0] == 20.0 and result_mul[1] == 40.0 and result_mul[2] == 60.0)
    MINI_CHECK(result_div[0] == 5.0 and result_div[1] == 10.0 and result_div[2] == 15.0)
    MINI_CHECK(result_add[0] == 11.0 and result_add[1] == 21.0 and result_add[2] == 31.0)
    MINI_CHECK(result_dif[0] == 9.0 and result_dif[1] == 19.0 and result_dif[2] == 29.0)
    MINI_CHECK(vx[0] == 1.0 and vx[1] == 0.0 and vx[2] == 0.0)
    MINI_CHECK(vy[0] == 0.0 and vy[1] == 1.0 and vy[2] == 0.0)
    MINI_CHECK(vz[0] == 0.0 and vz[1] == 0.0 and vz[2] == 1.0)
    MINI_CHECK(vzero[0] == 0.0 and vzero[1] == 0.0 and vzero[2] == 0.0)


@MINI_TEST("Vector", "Transformation")
def test_vector_transformation():
    from session_py import Vector, Xform
    from session_py.tolerance import PI

    v = Vector(1.0, 2.0, 3.0)
    v_xf = Xform.translation(10.0, 20.0, 30.0)
    v_transformed = v.transformed(v_xf)
    v.transform(v_xf)

    MINI_CHECK(v_transformed[0] == 1.0 and v_transformed[1] == 2.0 and v_transformed[2] == 3.0)
    MINI_CHECK(v[0] == 1.0 and v[1] == 2.0 and v[2] == 3.0)

    v2 = Vector(1.0, 0.0, 0.0)
    v2_xf = Xform.rotation_z(PI / 2.0)
    v2.transform(v2_xf)
    MINI_CHECK(TOLERANCE.is_close(v2[0], 0.0) and TOLERANCE.is_close(v2[1], 1.0) and TOLERANCE.is_close(v2[2], 0.0))


@MINI_TEST("Vector", "Magnitude")
def test_vector_magnitude():
    from session_py import Vector

    v = Vector(3.0, 4.0, 0.0)
    length = v.magnitude()
    len_squared = v.magnitude_squared()

    MINI_CHECK(length == 5.0)
    MINI_CHECK(len_squared == 25.0)


@MINI_TEST("Vector", "Normalize")
def test_vector_normalize():
    from session_py import Vector

    v0 = Vector(3.0, 4.0, 0.0)
    v0.normalize_self()

    v1 = Vector(3.0, 4.0, 0.0)
    v2 = v1.normalized()

    MINI_CHECK(TOLERANCE.is_close(v0.magnitude(), 1.0))
    MINI_CHECK(TOLERANCE.is_close(v2.magnitude(), 1.0))


@MINI_TEST("Vector", "Reverse")
def test_vector_reverse():
    from session_py import Vector

    v = Vector(1.0, -2.0, 3.0)
    v.reverse()

    MINI_CHECK(v[0] == -1.0 and v[1] == 2.0 and v[2] == -3.0)


@MINI_TEST("Vector", "Dot Product")
def test_vector_dot_product():
    from session_py import Vector

    # Orthogonality and parallelism via dot product
    # Perpendicular vectors are close to 0.0
    # Parallel vectors are close to 1.0
    v1 = Vector(1.0, 0.0, 0.0)
    v2 = Vector(0.0, 1.0, 0.0)
    v3 = Vector(1.0, 0.0, 0.0)
    dot_perp = v1.dot(v2)
    dot_paral = v1.dot(v3)

    # Projection of a onto b
    # Scalar projection:
    # (a . b) / ||b|| (here ||b||=1, so just a_x = 3.0)
    # Projection coefficient:
    # (a . b) / ||b||^2 = 6/4 = 1.5 (how many b2's fit in projection)
    a = Vector(3.0, 4.0, 0.0)
    b = Vector(1.0, 0.0, 0.0)
    b2 = Vector(2.0, 0.0, 0.0)
    proj_scalar = a.dot(b) / math.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    proj_coeff = a.dot(b2) / (b2[0]**2 + b2[1]**2 + b2[2]**2)

    MINI_CHECK(TOLERANCE.is_close(dot_perp, 0.0))
    MINI_CHECK(TOLERANCE.is_close(dot_paral, 1.0))
    MINI_CHECK(TOLERANCE.is_close(proj_scalar, 3.0))
    MINI_CHECK(TOLERANCE.is_close(proj_coeff, 1.5))


@MINI_TEST("Vector", "Cross Product")
def test_vector_cross_product():
    from session_py import Vector

    # Get normal
    v1 = Vector(1.0, 0.0, 0.0)
    v2 = Vector(0.0, 1.0, 0.0)
    vn = v1.cross(v2)

    # Perpendicular, area = 3*4=12
    a = Vector(3.0, 0.0, 0.0)
    b = Vector(0.0, 4.0, 0.0)
    cross = a.cross(b)
    area = math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)

    MINI_CHECK(vn[0] == 0.0 and vn[1] == 0.0 and vn[2] == 1.0)
    MINI_CHECK(TOLERANCE.is_close(area, 12.0))


@MINI_TEST("Vector", "Angle")
def test_vector_angle():
    from session_py import Vector

    # angle(): Angle between two vectors (degrees)
    v1 = Vector(1.0, 0.0, 0.0)  # x-axis
    v2 = Vector(0.0, 1.0, 0.0)  # y-axis
    v3 = Vector(1.0, 1.0, 0.0)  # 45 deg from x-axis

    angle_90 = v1.angle(v2, False)  # v1 to v2: 90 deg
    angle_45 = v1.angle(v3, False)  # v1 to v3: 45 deg

    # angle_between_vector_xy_components(): Angle of vector's XY projection from +X axis (atan2)
    v_30 = Vector(math.sqrt(3.0), 1.0, 0.0)  # 30 deg from x-axis
    v_60 = Vector(1.0, math.sqrt(3.0), 0.0)  # 60 deg from x-axis
    v_neg = Vector(-1.0, 1.0, 0.0)           # 135 deg from x-axis

    xy_angle_30 = Vector.angle_between_vector_xy_components(v_30)
    xy_angle_60 = Vector.angle_between_vector_xy_components(v_60)
    xy_angle_135 = Vector.angle_between_vector_xy_components(v_neg)

    # coordinate_direction_3angles(): Angles (alpha, beta, gamma) to x, y, z axes
    v_dir = Vector(35.4, 35.4, 86.6)
    abg = v_dir.coordinate_direction_3angles(True)

    # coordinate_direction_2angles(): Spherical angles (theta azimuth, phi elevation)
    v_sph = Vector(1.0, 1.0, math.sqrt(2.0))
    pt = v_sph.coordinate_direction_2angles(True)

    MINI_CHECK(TOLERANCE.is_close(angle_90, 90.0))
    MINI_CHECK(TOLERANCE.is_close(angle_45, 45.0))
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

    # Project vector v=(1,1,1) onto each axis
    # Returns: (projection_vector, scalar_length, perpendicular_vector, perp_length)
    v = Vector(1.0, 1.0, 1.0)
    x = Vector.x_axis()
    y = Vector.y_axis()
    z = Vector.z_axis()
    proj_x, lenx, perp_x, perp_lenx = v.projection(x)
    proj_y, leny, perp_y, perp_leny = v.projection(y)
    proj_z, lenz, perp_z, perp_lenz = v.projection(z)

    MINI_CHECK(proj_x[0] == 1.0 and proj_x[1] == 0.0 and proj_x[2] == 0.0)
    MINI_CHECK(proj_y[0] == 0.0 and proj_y[1] == 1.0 and proj_y[2] == 0.0)
    MINI_CHECK(proj_z[0] == 0.0 and proj_z[1] == 0.0 and proj_z[2] == 1.0)


@MINI_TEST("Vector", "Is Parallel To")
def test_vector_is_parallel_to():
    from session_py import Vector

    # is_parallel_to returns: 1 (parallel), -1 (anti-parallel), 0 (not parallel)
    v1 = Vector(2.0, 2.0, 2.0)
    v2 = Vector(4.0, 4.0, 4.0)      # parallel (same direction)
    v3 = Vector(-1.0, -1.0, -1.0)   # anti-parallel (opposite direction)
    v4 = Vector(1.0, 0.0, 0.0)      # not parallel

    MINI_CHECK(v1.is_parallel_to(v2) == 1)
    MINI_CHECK(v1.is_parallel_to(v3) == -1)
    MINI_CHECK(v1.is_parallel_to(v4) == 0)


@MINI_TEST("Vector", "Is Perpendicular To")
def test_vector_is_perpendicular_to():
    from session_py import Vector

    # is_perpendicular_to: checks if two vectors are perpendicular (dot product ~ 0)
    v1 = Vector(1.0, 0.0, 0.0)
    v2 = Vector(0.0, 1.0, 0.0)  # perpendicular
    v3 = Vector(0.0, 0.0, 1.0)  # perpendicular
    v4 = Vector(1.0, 1.0, 0.0)  # not perpendicular

    # perpendicular_to: sets a vector to be perpendicular to another
    z_axis = Vector(0.0, 0.0, 1.0)
    x_axis = Vector.zero()
    # x_axis is now perpendicular to z_axis
    x_axis.perpendicular_to(z_axis)

    arbitrary = Vector(1.0, 2.0, 3.0)
    perp = Vector.zero()
    # perp is now perpendicular to arbitrary
    perp.perpendicular_to(arbitrary)

    MINI_CHECK(v1.is_perpendicular_to(v2))
    MINI_CHECK(v1.is_perpendicular_to(v3))
    MINI_CHECK(not v1.is_perpendicular_to(v4))
    MINI_CHECK(x_axis.is_perpendicular_to(z_axis))
    MINI_CHECK(perp.is_perpendicular_to(arbitrary))


@MINI_TEST("Vector", "Get Leveled Vector")
def test_vector_get_leveled_vector():
    from session_py import Vector

    # Scale vector along its direction so its Z-component equals vertical_height.
    v = Vector(1.0, 1.0, 1.0)
    vertical_height = 1.0
    v_leveled = v.get_leveled_vector(vertical_height)

    MINI_CHECK(TOLERANCE.is_close(v_leveled.magnitude(), math.sqrt(3.0)))


@MINI_TEST("Vector", "Cos Sin Laws")
def test_vector_cos_sin_laws():
    from session_py import Vector

    # Given a 3-4-5 right triangle
    a = 3.0  # side opposite to angle A
    b = 4.0  # side opposite to angle B
    c = 5.0  # hypotenuse, opposite to angle C (90 deg)

    # angle_from_cosine_law(adj1, adj2, opposite) -> angle opposite to 'opposite' side
    angle_a = Vector.angle_from_cosine_law(b, c, a, True)
    angle_b = Vector.angle_from_cosine_law(a, c, b, True)
    angle_c = Vector.angle_from_cosine_law(a, b, c, True)

    # given 2 angles + 1 side, find other side
    side_a = Vector.side_from_sine_law(angle_a, angle_b, b, True)
    side_b = Vector.side_from_sine_law(angle_b, angle_c, c, True)
    side_c = Vector.side_from_sine_law(angle_c, angle_a, a, True)

    # given 2 sides + included angle, find 3rd side
    computed_c = Vector.cosine_law(a, b, angle_c, True)
    computed_a = Vector.cosine_law(b, c, angle_a, True)
    computed_b = Vector.cosine_law(a, c, angle_b, True)

    # given 2 sides + 1 angle, find other angle
    computed_angle_b = Vector.sine_law_angle(a, angle_a, b, True)
    computed_angle_a = Vector.sine_law_angle(b, angle_b, a, True)

    # given 1 side + 2 angles, find other side
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

    # Sum of multiple vectors
    vecs = [
        Vector(1.0, 1.0, 1.0),
        Vector(2.0, 2.0, 2.0),
        Vector(3.0, 3.0, 3.0),
    ]
    sum_v = Vector.sum_of_vectors(vecs)
    MINI_CHECK(sum_v[0] == 6.0)
    MINI_CHECK(sum_v[1] == 6.0)
    MINI_CHECK(sum_v[2] == 6.0)

    # Empty list returns zero vector
    empty = []
    zero = Vector.sum_of_vectors(empty)
    MINI_CHECK(zero[0] == 0.0)
    MINI_CHECK(zero[1] == 0.0)
    MINI_CHECK(zero[2] == 0.0)


@MINI_TEST("Vector", "Average")
def test_vector_average():
    from session_py import Vector

    # Average of multiple vectors
    vecs = [
        Vector(1.0, 2.0, 3.0),
        Vector(3.0, 4.0, 5.0),
        Vector(5.0, 6.0, 7.0),
    ]
    avg = Vector.average(vecs)
    MINI_CHECK(avg[0] == 3.0)
    MINI_CHECK(avg[1] == 4.0)
    MINI_CHECK(avg[2] == 5.0)


@MINI_TEST("Vector", "Is Zero")
def test_vector_is_zero():
    from session_py import Vector

    zero = Vector(0.0, 0.0, 0.0)
    nonzero = Vector(1.0, 0.0, 0.0)
    tiny = Vector(1e-13, 1e-13, 1e-13)  # Magnitude ~ 1.7e-13 < 1e-12

    MINI_CHECK(zero.is_zero())
    MINI_CHECK(not nonzero.is_zero())
    MINI_CHECK(tiny.is_zero())


@MINI_TEST("Vector", "Scale")
def test_vector_scale():
    from session_py import Vector

    v = Vector(2.0, 4.0, 6.0)
    v.scale(0.5)
    v_up = Vector(1.0, 2.0, 3.0)
    v_up.scale_up()
    v_rt = Vector(1.0, 2.0, 3.0)
    v_rt.scale_up()
    v_rt.scale_down()

    MINI_CHECK(v[0] == 1.0 and v[1] == 2.0 and v[2] == 3.0)
    MINI_CHECK(v_up[0] > 1.0)
    MINI_CHECK(TOLERANCE.is_close(v_rt[0], 1.0) and TOLERANCE.is_close(v_rt[1], 2.0) and TOLERANCE.is_close(v_rt[2], 3.0))


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

    sq = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(0.0, 0.0, 0.0),
    ]
    n = Vector.average_normal(sq)

    MINI_CHECK(TOLERANCE.is_close(abs(n[2]), 1.0))
    MINI_CHECK(TOLERANCE.is_close(n[0], 0.0) and TOLERANCE.is_close(n[1], 0.0))


@MINI_TEST("Vector", "Interpolate Points")
def test_vector_interpolate_points():
    from session_py import Point
    from session_py.vector import interpolate_points

    from_pt = Point(0.0, 0.0, 0.0)
    to_pt = Point(1.0, 0.0, 0.0)
    pts0 = interpolate_points(from_pt, to_pt, 2, 0)
    pts1 = interpolate_points(from_pt, to_pt, 1, 1)

    MINI_CHECK(len(pts0) == 2)
    MINI_CHECK(TOLERANCE.is_close(pts0[0][0], 1.0 / 3.0))
    MINI_CHECK(TOLERANCE.is_close(pts0[1][0], 2.0 / 3.0))
    MINI_CHECK(len(pts1) == 3)
    MINI_CHECK(TOLERANCE.is_close(pts1[0][0], 0.0) and TOLERANCE.is_close(pts1[2][0], 1.0))


@MINI_TEST("Vector", "Json Roundtrip")
def test_vector_json_roundtrip():
    from session_py import Vector
    from pathlib import Path

    v = Vector(42.1, 84.2, 126.3)
    v.name = "test_vector"

    #   __jsondump__()  │ dict         │ to JSON object (internal use)
    #   __jsonload__(d) │ dict         │ from JSON object (internal use)
    #   file_json_dumps()    │ str          │ to JSON string
    #   file_json_loads(s)   │ str          │ from JSON string
    #   file_json_dump(path) │ file         │ write to file
    #   file_json_load(path) │ file         │ read from file

    # file_json_dump(fname) / file_json_load(fname) - file-based serialization
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_vector.json"
    v.file_json_dump(fname)
    loaded = Vector.file_json_load(fname)

    MINI_CHECK(loaded.name == "test_vector")
    MINI_CHECK(TOLERANCE.is_close(loaded[0], 42.1))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], 84.2))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], 126.3))


@MINI_TEST("Vector", "Protobuf Roundtrip")
def test_vector_protobuf_roundtrip():
    from session_py import Vector
    from pathlib import Path

    v = Vector(42.1, 84.2, 126.3)
    v.name = "test_vector"

    #   pb_dumps()      │ bytes        │ to protobuf bytes
    #   pb_loads(b)     │ bytes        │ from protobuf bytes
    #   pb_dump(path)   │ file         │ write to file
    #   pb_load(path)   │ file         │ read from file

    # pb_dump(filename) / pb_load(filename) - file-based serialization
    path = Path(__file__).resolve().parents[2] / "serialization" / "test_vector.bin"
    v.pb_dump(path)
    loaded = Vector.pb_load(path)

    MINI_CHECK(loaded.name == "test_vector")
    MINI_CHECK(TOLERANCE.is_close(loaded[0], 42.1))
    MINI_CHECK(TOLERANCE.is_close(loaded[1], 84.2))
    MINI_CHECK(TOLERANCE.is_close(loaded[2], 126.3))


if __name__ == "__main__":
    run_all("python")
