from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("Tolerance", "Is Zero")
def test_tolerance_is_zero():
    from session_py.tolerance import TOLERANCE
    result = TOLERANCE.is_zero(1e-10)

    MINI_CHECK(result)


@MINI_TEST("Tolerance", "Is Close")
def test_tolerance_is_close():
    from session_py.tolerance import TOLERANCE
    result = TOLERANCE.is_close(1.0, 1.0 + 1e-7)

    MINI_CHECK(result)


@MINI_TEST("Tolerance", "Is Positive")
def test_tolerance_is_positive():
    from session_py.tolerance import TOLERANCE
    result = TOLERANCE.is_positive(1.0)

    MINI_CHECK(result)


@MINI_TEST("Tolerance", "Is Negative")
def test_tolerance_is_negative():
    from session_py.tolerance import TOLERANCE
    result = TOLERANCE.is_negative(-1.0)

    MINI_CHECK(result)


@MINI_TEST("Tolerance", "Is Between")
def test_tolerance_is_between():
    from session_py.tolerance import TOLERANCE
    result = TOLERANCE.is_between(0.5, 0.0, 1.0)

    MINI_CHECK(result)


@MINI_TEST("Tolerance", "Format Number")
def test_tolerance_format_number():
    from session_py.tolerance import TOLERANCE
    result = TOLERANCE.format_number(3.14159, precision=2)

    MINI_CHECK(result == "3.14")


@MINI_TEST("Tolerance", "Key")
def test_tolerance_key():
    from session_py.tolerance import TOLERANCE
    result = TOLERANCE.key([1.0, 2.0, 3.0])

    MINI_CHECK(result == "1.000,2.000,3.000")


@MINI_TEST("Tolerance", "To Radians")
def test_tolerance_to_radians():
    from session_py.tolerance import Tolerance
    from session_py.tolerance import PI
    r0 = Tolerance.to_radians(180.0)
    r1 = Tolerance.to_radians(90.0)
    r2 = Tolerance.to_radians(0.0)

    MINI_CHECK(abs(r0 - PI) < 1e-9)
    MINI_CHECK(abs(r1 - PI / 2.0) < 1e-9)
    MINI_CHECK(abs(r2) < 1e-9)


@MINI_TEST("Tolerance", "To Degrees")
def test_tolerance_to_degrees():
    from session_py.tolerance import Tolerance
    from session_py.tolerance import PI
    d0 = Tolerance.to_degrees(PI)
    d1 = Tolerance.to_degrees(PI / 2.0)
    d2 = Tolerance.to_degrees(0.0)

    MINI_CHECK(abs(d0 - 180.0) < 1e-9)
    MINI_CHECK(abs(d1 - 90.0) < 1e-9)
    MINI_CHECK(abs(d2) < 1e-9)


@MINI_TEST("Tolerance", "Runtime Modification")
def test_tolerance_runtime_modification():
    from session_py.tolerance import TOLERANCE
    # Get current default values
    original_absolute = TOLERANCE.absolute
    original_relative = TOLERANCE.relative

    MINI_CHECK(original_absolute == 1e-9)
    MINI_CHECK(original_relative == 1e-6)

    # Modify tolerance values at runtime
    TOLERANCE.absolute = 1e-12
    TOLERANCE.relative = 1e-12
    MINI_CHECK(TOLERANCE.absolute == 1e-12)
    MINI_CHECK(TOLERANCE.relative == 1e-12)

    # Test with new tolerance - 1e-11 difference now fails is_close
    close_with_tight = TOLERANCE.is_close(1.0, 1.0 + 1e-11)
    MINI_CHECK(not close_with_tight)

    # Reset to defaults
    TOLERANCE.reset()
    MINI_CHECK(TOLERANCE.absolute == 1e-9)
    MINI_CHECK(TOLERANCE.relative == 1e-6)

    # Same test now passes with default tolerance
    close_with_default = TOLERANCE.is_close(1.0, 1.0 + 1e-11)
    MINI_CHECK(close_with_default)


@MINI_TEST("Tolerance", "Unique From Two Int")
def test_tolerance_unique_from_two_int():
    from session_py.tolerance import unique_from_two_int
    r0 = unique_from_two_int(3, 7)
    r1 = unique_from_two_int(7, 3)

    MINI_CHECK(r0 == r1)
    MINI_CHECK(r0 == (7 << 32) | 3)


@MINI_TEST("Tolerance", "Wrap Index")
def test_tolerance_wrap_index():
    from session_py.tolerance import wrap_index
    r0 = wrap_index(0, 4)
    r1 = wrap_index(3, 4)
    r2 = wrap_index(4, 4)
    r3 = wrap_index(-1, 4)
    r4 = wrap_index(0, 0)

    MINI_CHECK(r0 == 0)
    MINI_CHECK(r1 == 3)
    MINI_CHECK(r2 == 0)
    MINI_CHECK(r3 == 3)
    MINI_CHECK(r4 == 0)


@MINI_TEST("Tolerance", "Triangle Edge By Angle")
def test_tolerance_triangle_edge_by_angle():
    from session_py.tolerance import triangle_edge_by_angle
    r = triangle_edge_by_angle(1.0, 45.0)

    MINI_CHECK(abs(r - 1.0) < 1e-9)
    r2 = triangle_edge_by_angle(5.0, 0.0)
    MINI_CHECK(abs(r2) < 1e-9)


@MINI_TEST("Tolerance", "Rad Deg Conversion")
def test_tolerance_rad_deg():
    from session_py.tolerance import rad_to_deg
    from session_py.tolerance import deg_to_rad
    from session_py.tolerance import PI
    r0 = rad_to_deg(PI)
    r1 = deg_to_rad(180.0)
    r2 = deg_to_rad(rad_to_deg(1.234))

    MINI_CHECK(abs(r0 - 180.0) < 1e-9)
    MINI_CHECK(abs(r1 - PI) < 1e-9)
    MINI_CHECK(abs(r2 - 1.234) < 1e-9)


@MINI_TEST("Tolerance", "Count Digits")
def test_tolerance_count_digits():
    from session_py.tolerance import count_digits
    r0 = count_digits(0.0)
    r1 = count_digits(1.0)
    r2 = count_digits(9.9)
    r3 = count_digits(10.0)
    r4 = count_digits(100.5)
    r5 = count_digits(-42.0)

    MINI_CHECK(r0 == 0)
    MINI_CHECK(r1 == 1)
    MINI_CHECK(r2 == 1)
    MINI_CHECK(r3 == 2)
    MINI_CHECK(r4 == 3)
    MINI_CHECK(r5 == 2)


@MINI_TEST("Tolerance", "Is Angle Zero")
def test_tolerance_is_angle_zero():
    from session_py.tolerance import TOLERANCE
    # Angular tolerance default is 1e-6
    r0 = TOLERANCE.is_angle_zero(1e-8)
    r1 = TOLERANCE.is_angle_zero(0.1)

    MINI_CHECK(r0)
    MINI_CHECK(not r1)


@MINI_TEST("Tolerance", "Is Angles Close")
def test_tolerance_is_angles_close():
    from session_py.tolerance import TOLERANCE
    r0 = TOLERANCE.is_angles_close(1.0, 1.0 + 1e-8)
    r1 = TOLERANCE.is_angles_close(1.0, 2.0)

    MINI_CHECK(r0)
    MINI_CHECK(not r1)


@MINI_TEST("Tolerance", "Is Point Close")
def test_tolerance_is_point_close():
    from session_py.tolerance import TOLERANCE
    from session_py import Point

    a = Point(1.0, 2.0, 3.0)
    b = Point(1.0, 2.0, 3.0 + 1e-12)
    c = Point(1.0, 2.0, 4.0)

    MINI_CHECK(TOLERANCE.is_point_close(a, b))
    MINI_CHECK(not TOLERANCE.is_point_close(a, c))


@MINI_TEST("Tolerance", "Is Allclose")
def test_tolerance_is_allclose():
    from session_py.tolerance import TOLERANCE
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0 + 1e-12]
    c = [1.0, 2.0, 4.0]

    MINI_CHECK(TOLERANCE.is_allclose(a, b))
    MINI_CHECK(not TOLERANCE.is_allclose(a, c))


@MINI_TEST("Tolerance", "Key Xy")
def test_tolerance_key_xy():
    from session_py.tolerance import TOLERANCE
    result = TOLERANCE.key_xy([1.0, 2.0])

    MINI_CHECK(result == "1.000,2.000")


@MINI_TEST("Tolerance", "Round To")
def test_tolerance_round_to():
    from session_py.tolerance import Tolerance
    r0 = Tolerance.round_to(3.14159, 2)
    r1 = Tolerance.round_to(2.5, 0)

    MINI_CHECK(abs(r0 - 3.14) < 1e-9)
    MINI_CHECK(abs(r1 - 2.0) < 1e-9)


@MINI_TEST("Tolerance", "Precision From Tolerance")
def test_tolerance_precision_from_tolerance():
    from session_py.tolerance import TOLERANCE
    # Default absolute tolerance is 1e-9 -> precision should be 9
    prec = TOLERANCE.precision_from_tolerance()

    MINI_CHECK(prec == 9)


@MINI_TEST("Tolerance", "Tolerance")
def test_tolerance_tolerance():
    from session_py.tolerance import TOLERANCE
    # rtol * abs(truevalue) + atol
    result = TOLERANCE.tolerance(1.0, 1e-6, 1e-9)

    MINI_CHECK(abs(result - (1e-6 + 1e-9)) < 1e-18)


@MINI_TEST("Tolerance", "Compare")
def test_tolerance_compare():
    from session_py.tolerance import TOLERANCE
    r0 = TOLERANCE.compare(1.0, 1.0 + 1e-7, 1e-6, 1e-9)
    r1 = TOLERANCE.compare(1.0, 2.0, 1e-6, 1e-9)

    MINI_CHECK(r0)
    MINI_CHECK(not r1)


@MINI_TEST("Tolerance", "Is Finite")
def test_tolerance_is_finite():
    from session_py.tolerance import is_finite
    r0 = is_finite(1.0)
    r1 = is_finite(float("inf"))

    MINI_CHECK(r0)
    MINI_CHECK(not r1)


@MINI_TEST("Tolerance", "Is Vector Close")
def test_tolerance_is_vector_close():
    from session_py.tolerance import TOLERANCE
    from session_py import Vector

    a = Vector(1.0, 2.0, 3.0)
    b = Vector(1.0, 2.0, 3.0 + 1e-12)
    c = Vector(1.0, 2.0, 4.0)

    MINI_CHECK(TOLERANCE.is_vector_close(a, b))
    MINI_CHECK(not TOLERANCE.is_vector_close(a, c))


@MINI_TEST("Tolerance", "Temporary")
def test_tolerance_temporary():
    from session_py.tolerance import TOLERANCE
    original = TOLERANCE.absolute
    with TOLERANCE.temporary(absolute=1e-12):
        MINI_CHECK(TOLERANCE.absolute == 1e-12)
    MINI_CHECK(TOLERANCE.absolute == original)


if __name__ == "__main__":
    run_all("python")
