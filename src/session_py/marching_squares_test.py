import math
from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("MarchingSquares", "Extract")
def test_marching_squares_extract():
    from session_py import MarchingSquares
    # 3x3 grid: center > 0, corners < 0 → circle-like
    grid = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    result = MarchingSquares.extract(grid, 0.5)

    MINI_CHECK(len(result) == 1)
    MINI_CHECK(result[0].point_count() == 5)
    MINI_CHECK(result[0].is_closed())


@MINI_TEST("MarchingSquares", "ExtractFromFunc")
def test_marching_squares_extract_from_func():
    from session_py import MarchingSquares
    # Circle: x^2 + y^2 = 1
    def circle(x, y):
        return 1.0 - (x * x + y * y)
    result = MarchingSquares.extract_from_func(circle, (-2.0, 2.0), (-2.0, 2.0), 20, 20, 0.0)

    MINI_CHECK(len(result) > 0)


@MINI_TEST("MarchingSquares", "EmptyGrid")
def test_marching_squares_empty_grid():
    from session_py import MarchingSquares
    result = MarchingSquares.extract([], 0.5)

    MINI_CHECK(len(result) == 0)


@MINI_TEST("MarchingSquares", "AllAbove")
def test_marching_squares_all_above():
    from session_py import MarchingSquares
    grid = [
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0],
    ]
    result = MarchingSquares.extract(grid, 1.0)

    MINI_CHECK(len(result) == 0)


@MINI_TEST("MarchingSquares", "AllBelow")
def test_marching_squares_all_below():
    from session_py import MarchingSquares
    grid = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    result = MarchingSquares.extract(grid, 1.0)

    MINI_CHECK(len(result) == 0)


@MINI_TEST("MarchingSquares", "Interpolation")
def test_marching_squares_interpolation():
    from session_py import MarchingSquares
    # Single cell: v0=0, v1=2, v2=2, v3=0 → iso=1 bisects left-right
    grid = [
        [0.0, 2.0],
        [0.0, 2.0],
    ]
    result = MarchingSquares.extract(grid, 1.0)

    MINI_CHECK(len(result) == 1)
    seg = result[0]
    MINI_CHECK(TOLERANCE.is_close(seg.get_point(0)[0], 0.5))
    MINI_CHECK(TOLERANCE.is_close(seg.get_point(1)[0], 0.5))


if __name__ == "__main__":
    run_all("python")
