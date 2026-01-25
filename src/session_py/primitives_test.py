from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all


@MINI_TEST("Primitives", "circle")
def test_circle():
    from session_py import Primitives

    c = Primitives.circle(0.0, 0.0, 0.0, 1.0)

    MINI_CHECK(c.cv_count() == 9)
    MINI_CHECK(c.order() == 3)
    MINI_CHECK(c.is_rational() == True)


@MINI_TEST("Primitives", "ellipse")
def test_ellipse():
    from session_py import Primitives

    c = Primitives.ellipse(0.0, 0.0, 0.0, 2.0, 1.0)

    MINI_CHECK(c.cv_count() == 9)
    MINI_CHECK(c.order() == 3)
    MINI_CHECK(c.is_rational() == True)


@MINI_TEST("Primitives", "arc")
def test_arc():
    from session_py import Primitives
    from session_py import Point

    start = Point(0.0, 0.0, 0.0)
    mid = Point(1.0, 1.0, 0.0)
    end = Point(2.0, 0.0, 0.0)
    c = Primitives.arc(start, mid, end)

    MINI_CHECK(c.cv_count() == 3)
    MINI_CHECK(c.order() == 3)
    MINI_CHECK(c.is_rational() == True)


@MINI_TEST("Primitives", "parabola")
def test_parabola():
    from session_py import Primitives
    from session_py import Point

    p0 = Point(-1.0, 1.0, 0.0)
    p1 = Point(0.0, 0.0, 0.0)
    p2 = Point(1.0, 1.0, 0.0)
    c = Primitives.parabola(p0, p1, p2)

    MINI_CHECK(c.cv_count() == 3)
    MINI_CHECK(c.order() == 3)
    MINI_CHECK(c.is_rational() == False)


@MINI_TEST("Primitives", "hyperbola")
def test_hyperbola():
    from session_py import Primitives
    from session_py import Point

    center = Point(0.0, 0.0, 0.0)
    c = Primitives.hyperbola(center, 1.0, 1.0, 1.0)

    MINI_CHECK(c.cv_count() >= 4)
    MINI_CHECK(c.order() == 4)
    MINI_CHECK(c.is_rational() == False)


@MINI_TEST("Primitives", "spiral")
def test_spiral():
    from session_py import Primitives

    c = Primitives.spiral(1.0, 2.0, 1.0, 5.0)

    MINI_CHECK(c.cv_count() >= 4)
    MINI_CHECK(c.order() == 4)
    MINI_CHECK(c.is_rational() == False)


if __name__ == "__main__":
    run_all(language="python")
