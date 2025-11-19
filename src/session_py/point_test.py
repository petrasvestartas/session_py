from .point import Point
from .vector import Vector
from .color import Color
from .mini_test import MINI_TEST, MINI_CHECK, run_all

# /home/petras/code/code_session/uvsession/bin/python -m session_py.point_test


@MINI_TEST("Point", "constructor")
def test_point_constructor():
    from .point import Point
    point = Point(1.0, 2.0, 3.0)
    point[0] = 10.0
    MINI_CHECK(point.name == "my_point")
    MINI_CHECK(bool(point.guid))
    MINI_CHECK(point.x == 10.0)
    MINI_CHECK(point.y == 2.0)
    MINI_CHECK(point.z == 3.0)
    MINI_CHECK(point.width == 1.0)
    MINI_CHECK(point.pointcolor == Color.white())


@MINI_TEST("Point", "equality_equal")
def test_point_equality_equal():
    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(1.0, 2.0, 3.0)

    # Timed code: perform equality and inequality checks
    eq_result = (p1 == p2)
    neq_result = (p1 != p2)

    # Assertions at the end
    MINI_CHECK(eq_result == True)
    MINI_CHECK(neq_result == False)


@MINI_TEST("Point", "equality_not_equal")
def test_point_equality_not_equal():
    p3 = Point(1.0, 2.0, 3.0)
    p4 = Point(1.1, 2.0, 3.0)

    # Timed code: perform equality and inequality checks
    eq_result = (p3 == p4)
    neq_result = (p3 != p4)

    # Assertions at the end
    MINI_CHECK(eq_result == False)
    MINI_CHECK(neq_result == True)

if __name__ == "__main__":
    run_all(language="python")

###########################################################################################
# JSON
###########################################################################################


def test_point_json_roundtrip():
    from pathlib import Path
    from session_py.encoders import json_dump, json_load

    point = Point(1.5, 2.5, 3.5)
    point.name = "test_point"
    point.width = 2.0
    point.pointcolor = Color(255, 128, 64, 255)

    path = Path(__file__).resolve().parents[2] / "test_point.json"
    json_dump(point, path)
    loaded = json_load(path)

    assert isinstance(loaded, Point)
    assert loaded.x == point.x
    assert loaded.y == point.y
    assert loaded.z == point.z
    assert loaded.name == point.name
    assert loaded.width == point.width
    assert loaded.pointcolor.r == 255


###########################################################################################
# No-copy Operators
###########################################################################################


def test_point_getitem():
    import time
    
    point = Point(1.0, 2.0, 3.0)
    assert point[0] == 1.0
    assert point[1] == 2.0
    assert point[2] == 3.0
    
    # Benchmark
    iterations = 100_000
    start = time.perf_counter()
    for _ in range(iterations):
        _ = point[0] + point[1] + point[2]
    duration = time.perf_counter() - start
    print(f"  Point indexing: {duration/iterations*1e6:.2f}µs per op ({iterations} iterations)")


def test_point_setitem():
    point = Point(1.0, 2.0, 3.0)
    point[0] = 4.0
    point[1] = 5.0
    point[2] = 6.0
    assert point.x == 4.0
    assert point.y == 5.0
    assert point.z == 6.0


def test_point_imul():
    point = Point(1.0, 2.0, 3.0)
    point *= 2.0
    assert point.x == 2.0
    assert point.y == 4.0
    assert point.z == 6.0


def test_point_itruediv():
    point = Point(2.0, 4.0, 6.0)
    point /= 2.0
    assert point.x == 1.0
    assert point.y == 2.0
    assert point.z == 3.0


def test_point_iadd():
    point = Point(1.0, 2.0, 3.0)
    vec = Vector(4.0, 5.0, 6.0)
    point += vec
    assert point.x == 5.0
    assert point.y == 7.0
    assert point.z == 9.0


def test_point_isub():
    point = Point(5.0, 7.0, 9.0)
    vec = Vector(4.0, 5.0, 6.0)
    point -= vec
    assert point.x == 1.0
    assert point.y == 2.0
    assert point.z == 3.0


###########################################################################################
# Copy Operators
###########################################################################################


def test_point_mul():
    point = Point(1.0, 2.0, 3.0)
    result = point * 2.0
    assert result.x == 2.0
    assert result.y == 4.0
    assert result.z == 6.0


def test_point_truediv():
    point = Point(2.0, 4.0, 6.0)
    result = point / 2.0
    assert result.x == 1.0
    assert result.y == 2.0
    assert result.z == 3.0


def test_point_add():
    point = Point(1.0, 2.0, 3.0)
    vec = Vector(4.0, 5.0, 6.0)
    result = point + vec
    assert result.x == 5.0
    assert result.y == 7.0
    assert result.z == 9.0


def test_point_sub():
    p1 = Point(5.0, 7.0, 9.0)
    p2 = Point(4.0, 5.0, 6.0)
    result = p1 - p2
    assert isinstance(result, Vector)
    assert result.x == 1.0
    assert result.y == 2.0
    assert result.z == 3.0

    vec = Vector(1.0, 1.0, 1.0)
    result2 = p1 - vec
    assert isinstance(result2, Point)
    assert result2.x == 4.0
    assert result2.y == 6.0
    assert result2.z == 8.0


###########################################################################################
# Details
###########################################################################################


def test_point_ccw():
    a = Point(0.0, 0.0, 0.0)
    b = Point(1.0, 0.0, 0.0)
    c = Point(0.0, 1.0, 0.0)
    assert Point.ccw(a, b, c)
    assert not Point.ccw(b, a, c)


def test_point_mid_point():
    import time
    
    p1 = Point(0.0, 0.0, 0.0)
    p2 = Point(1.0, 0.0, 0.0)
    mid = p1.mid_point(p2)
    assert round(mid.x, 6) == 0.5
    assert round(mid.y, 6) == 0.0
    assert round(mid.z, 6) == 0.0
    
    # Benchmark
    iterations = 100_000
    start = time.perf_counter()
    for _ in range(iterations):
        _ = p1.mid_point(p2)
    duration = time.perf_counter() - start
    print(f"  Point midpoint: {duration/iterations*1e6:.2f}µs per op ({iterations} iterations)")


def test_point_distance():
    import time
    
    p1 = Point(0.0, 0.0, 0.0)
    p2 = Point(1.0, 0.0, 0.0)
    assert round(p1.distance(p2), 6) == 1.0
    
    # Benchmark
    iterations = 100_000
    start = time.perf_counter()
    for _ in range(iterations):
        _ = p1.distance(p2)
    duration = time.perf_counter() - start
    print(f"  Point distance: {duration/iterations*1e6:.2f}µs per op ({iterations} iterations)")


def test_point_area():
    points = [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.0, 1.0, 0.0)]
    assert Point.area(points) == 0.5


def test_point_centroid_quad():
    vertices = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
    ]
    centroid = Point.centroid_quad(vertices)
    assert round(centroid.x, 6) == 0.5
    assert round(centroid.y, 6) == 0.5
    assert round(centroid.z, 6) == 0.0


