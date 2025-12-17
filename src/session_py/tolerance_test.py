from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from session_py.tolerance import TOLERANCE, Tolerance


@MINI_TEST("Tolerance", "is_zero")
def test_tolerance_is_zero():
    result = TOLERANCE.is_zero(1e-10)
    MINI_CHECK(result == True)


@MINI_TEST("Tolerance", "is_close")
def test_tolerance_is_close():
    result = TOLERANCE.is_close(1.0, 1.0 + 1e-7)
    MINI_CHECK(result == True)


@MINI_TEST("Tolerance", "is_positive")
def test_tolerance_is_positive():
    result = TOLERANCE.is_positive(1.0)
    MINI_CHECK(result == True)


@MINI_TEST("Tolerance", "is_negative")
def test_tolerance_is_negative():
    result = TOLERANCE.is_negative(-1.0)
    MINI_CHECK(result == True)


@MINI_TEST("Tolerance", "is_between")
def test_tolerance_is_between():
    result = TOLERANCE.is_between(0.5, 0.0, 1.0)
    MINI_CHECK(result == True)


@MINI_TEST("Tolerance", "format_number")
def test_tolerance_format_number():
    result = TOLERANCE.format_number(3.14159, precision=2)
    MINI_CHECK(result == "3.14")


@MINI_TEST("Tolerance", "key")
def test_tolerance_key():
    result = TOLERANCE.key([1.0, 2.0, 3.0])
    MINI_CHECK(result == "1.000,2.000,3.000")


if __name__ == "__main__":
    run_all("python")
