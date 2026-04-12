from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from session_py.tolerance import (
    TOLERANCE, Tolerance,
    unique_from_two_int, wrap_index, triangle_edge_by_angle,
    rad_to_deg, deg_to_rad, count_digits, PI,
)


@MINI_TEST("Tolerance", "Is Zero")
def test_tolerance_is_zero():
    result = TOLERANCE.is_zero(1e-10)

    MINI_CHECK(result)


@MINI_TEST("Tolerance", "Is Close")
def test_tolerance_is_close():
    result = TOLERANCE.is_close(1.0, 1.0 + 1e-7)

    MINI_CHECK(result)


@MINI_TEST("Tolerance", "Is Positive")
def test_tolerance_is_positive():
    result = TOLERANCE.is_positive(1.0)

    MINI_CHECK(result)


@MINI_TEST("Tolerance", "Is Negative")
def test_tolerance_is_negative():
    result = TOLERANCE.is_negative(-1.0)

    MINI_CHECK(result)


@MINI_TEST("Tolerance", "Is Between")
def test_tolerance_is_between():
    result = TOLERANCE.is_between(0.5, 0.0, 1.0)

    MINI_CHECK(result)


@MINI_TEST("Tolerance", "Format Number")
def test_tolerance_format_number():
    result = TOLERANCE.format_number(3.14159, precision=2)

    MINI_CHECK(result == "3.14")


@MINI_TEST("Tolerance", "Key")
def test_tolerance_key():
    result = TOLERANCE.key([1.0, 2.0, 3.0])

    MINI_CHECK(result == "1.000,2.000,3.000")


@MINI_TEST("Tolerance", "Runtime Modification")
def test_tolerance_runtime_modification():
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
    r0 = unique_from_two_int(3, 7)
    r1 = unique_from_two_int(7, 3)

    MINI_CHECK(r0 == r1)
    MINI_CHECK(r0 == (7 << 32) | 3)


@MINI_TEST("Tolerance", "Wrap Index")
def test_tolerance_wrap_index():
    MINI_CHECK(wrap_index(0, 4)  == 0)
    MINI_CHECK(wrap_index(3, 4)  == 3)
    MINI_CHECK(wrap_index(4, 4)  == 0)
    MINI_CHECK(wrap_index(-1, 4) == 3)
    MINI_CHECK(wrap_index(0, 0)  == 0)


@MINI_TEST("Tolerance", "Triangle Edge By Angle")
def test_tolerance_triangle_edge_by_angle():
    MINI_CHECK(abs(triangle_edge_by_angle(1.0, 45.0) - 1.0) < 1e-9)
    MINI_CHECK(abs(triangle_edge_by_angle(5.0, 0.0)) < 1e-9)


@MINI_TEST("Tolerance", "Rad Deg Conversion")
def test_tolerance_rad_deg():
    MINI_CHECK(abs(rad_to_deg(PI) - 180.0) < 1e-9)
    MINI_CHECK(abs(deg_to_rad(180.0) - PI) < 1e-9)
    MINI_CHECK(abs(deg_to_rad(rad_to_deg(1.234)) - 1.234) < 1e-9)


@MINI_TEST("Tolerance", "To Radians")
def test_tolerance_to_radians():
    MINI_CHECK(abs(Tolerance.to_radians(180.0) - PI) < 1e-9)
    MINI_CHECK(abs(Tolerance.to_radians(90.0) - PI / 2.0) < 1e-9)
    MINI_CHECK(abs(Tolerance.to_radians(0.0)) < 1e-9)


@MINI_TEST("Tolerance", "To Degrees")
def test_tolerance_to_degrees():
    MINI_CHECK(abs(Tolerance.to_degrees(PI) - 180.0) < 1e-9)
    MINI_CHECK(abs(Tolerance.to_degrees(PI / 2.0) - 90.0) < 1e-9)
    MINI_CHECK(abs(Tolerance.to_degrees(0.0)) < 1e-9)


@MINI_TEST("Tolerance", "Count Digits")
def test_tolerance_count_digits():
    MINI_CHECK(count_digits(0.0)   == 0)
    MINI_CHECK(count_digits(1.0)   == 1)
    MINI_CHECK(count_digits(9.9)   == 1)
    MINI_CHECK(count_digits(10.0)  == 2)
    MINI_CHECK(count_digits(100.5) == 3)
    MINI_CHECK(count_digits(-42.0) == 2)


@MINI_TEST("Tolerance", "Is Angle Zero")
def test_tolerance_is_angle_zero():
    # Angular tolerance default is 1e-6
    MINI_CHECK(TOLERANCE.is_angle_zero(1e-8))
    MINI_CHECK(not TOLERANCE.is_angle_zero(0.1))


@MINI_TEST("Tolerance", "Is Angles Close")
def test_tolerance_is_angles_close():
    MINI_CHECK(TOLERANCE.is_angles_close(1.0, 1.0 + 1e-8))
    MINI_CHECK(not TOLERANCE.is_angles_close(1.0, 2.0))


@MINI_TEST("Tolerance", "Is Point Close")
def test_tolerance_is_point_close():
    from session_py import Point

    a = Point(1.0, 2.0, 3.0)
    b = Point(1.0, 2.0, 3.0 + 1e-12)
    c = Point(1.0, 2.0, 4.0)

    MINI_CHECK(TOLERANCE.is_point_close(a, b))
    MINI_CHECK(not TOLERANCE.is_point_close(a, c))


@MINI_TEST("Tolerance", "Is Allclose")
def test_tolerance_is_allclose():
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0 + 1e-12]
    c = [1.0, 2.0, 4.0]

    MINI_CHECK(TOLERANCE.is_allclose(a, b))
    MINI_CHECK(not TOLERANCE.is_allclose(a, c))


@MINI_TEST("Tolerance", "Key Xy")
def test_tolerance_key_xy():
    result = TOLERANCE.key_xy([1.0, 2.0])

    MINI_CHECK(result == "1.000,2.000")


@MINI_TEST("Tolerance", "Round To")
def test_tolerance_round_to():
    MINI_CHECK(abs(Tolerance.round_to(3.14159, 2) - 3.14) < 1e-9)
    MINI_CHECK(abs(Tolerance.round_to(2.5, 0) - 2.0) < 1e-9)


@MINI_TEST("Tolerance", "Precision From Tolerance")
def test_tolerance_precision_from_tolerance():
    # Default absolute tolerance is 1e-9 -> precision should be 9
    prec = TOLERANCE.precision_from_tolerance()

    MINI_CHECK(prec == 9)


if __name__ == "__main__":
    run_all("python")
