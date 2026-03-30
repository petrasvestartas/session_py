import math
from typing import Callable, List, Tuple

from .point import Point
from .polyline import Polyline


class MarchingSquares:
    """Marching squares iso-contour extraction.

    Extracts polylines at a given iso-value from a 2D scalar field.
    Uses the standard 16-case lookup table with linear interpolation.
    """

    # Edge table: for each of 16 cases, list of edge pairs that form line segments
    # Edges: 0=bottom, 1=right, 2=top, 3=left  (CCW)
    # Each case is a list of (edge_a, edge_b) pairs
    _EDGE_TABLE = [
        [],           # 0000
        [(3, 0)],     # 0001 (only corner 0 inside)
        [(0, 1)],     # 0010
        [(3, 1)],     # 0011
        [(1, 2)],     # 0100
        [(3, 2), (0, 1)],  # 0101 ambiguous: two segments
        [(0, 2)],     # 0110
        [(3, 2)],     # 0111
        [(2, 3)],     # 1000
        [(2, 0)],     # 1001
        [(2, 1), (3, 0)],  # 1010 ambiguous
        [(2, 1)],     # 1011
        [(1, 3)],     # 1100
        [(1, 0)],     # 1101
        [(0, 3)],     # 1110
        [],           # 1111
    ]

    @staticmethod
    def _interp(a: float, b: float, va: float, vb: float, iso: float) -> float:
        if abs(vb - va) < 1e-12:
            return (a + b) * 0.5
        t = (iso - va) / (vb - va)
        return a + t * (b - a)

    @staticmethod
    def extract(grid: List[List[float]], iso_value: float, cell_size: float = 1.0) -> List[Polyline]:
        rows = len(grid)
        if rows == 0:
            return []
        cols = len(grid[0])
        if cols == 0:
            return []

        segments: List[Tuple[Point, Point]] = []

        for r in range(rows - 1):
            for c in range(cols - 1):
                v0 = grid[r][c]
                v1 = grid[r][c + 1]
                v2 = grid[r + 1][c + 1]
                v3 = grid[r + 1][c]

                x0 = c * cell_size
                y0 = r * cell_size
                x1 = (c + 1) * cell_size
                y1 = (r + 1) * cell_size

                case = (
                    (1 if v3 >= iso_value else 0) << 3 |
                    (1 if v2 >= iso_value else 0) << 2 |
                    (1 if v1 >= iso_value else 0) << 1 |
                    (1 if v0 >= iso_value else 0)
                )

                # Edge midpoints (with interpolation)
                def edge_pt(e: int) -> Point:
                    if e == 0:  # bottom: v0-v1
                        x = MarchingSquares._interp(x0, x1, v0, v1, iso_value)
                        return Point(x, y0, 0.0)
                    elif e == 1:  # right: v1-v2
                        y = MarchingSquares._interp(y0, y1, v1, v2, iso_value)
                        return Point(x1, y, 0.0)
                    elif e == 2:  # top: v3-v2
                        x = MarchingSquares._interp(x0, x1, v3, v2, iso_value)
                        return Point(x, y1, 0.0)
                    else:  # left: v0-v3
                        y = MarchingSquares._interp(y0, y1, v0, v3, iso_value)
                        return Point(x0, y, 0.0)

                for ea, eb in MarchingSquares._EDGE_TABLE[case]:
                    segments.append((edge_pt(ea), edge_pt(eb)))

        return MarchingSquares._connect_segments(segments)

    @staticmethod
    def extract_from_func(
        func: Callable[[float, float], float],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        nx: int,
        ny: int,
        iso_value: float,
    ) -> List[Polyline]:
        x0, x1 = x_range
        y0, y1 = y_range
        dx = (x1 - x0) / (nx - 1) if nx > 1 else 0.0
        dy = (y1 - y0) / (ny - 1) if ny > 1 else 0.0
        grid = []
        for r in range(ny):
            row = []
            for c in range(nx):
                x = x0 + c * dx
                y = y0 + r * dy
                row.append(func(x, y))
            grid.append(row)
        segments: List[Tuple[Point, Point]] = []
        for r in range(ny - 1):
            for c in range(nx - 1):
                v0 = grid[r][c]
                v1 = grid[r][c + 1]
                v2 = grid[r + 1][c + 1]
                v3 = grid[r + 1][c]
                px0 = x0 + c * dx
                py0 = y0 + r * dy
                px1 = x0 + (c + 1) * dx
                py1 = y0 + (r + 1) * dy

                case = (
                    (1 if v3 >= iso_value else 0) << 3 |
                    (1 if v2 >= iso_value else 0) << 2 |
                    (1 if v1 >= iso_value else 0) << 1 |
                    (1 if v0 >= iso_value else 0)
                )

                def edge_pt_f(e: int) -> Point:
                    if e == 0:
                        x = MarchingSquares._interp(px0, px1, v0, v1, iso_value)
                        return Point(x, py0, 0.0)
                    elif e == 1:
                        y = MarchingSquares._interp(py0, py1, v1, v2, iso_value)
                        return Point(px1, y, 0.0)
                    elif e == 2:
                        x = MarchingSquares._interp(px0, px1, v3, v2, iso_value)
                        return Point(x, py1, 0.0)
                    else:
                        y = MarchingSquares._interp(py0, py1, v0, v3, iso_value)
                        return Point(px0, y, 0.0)

                for ea, eb in MarchingSquares._EDGE_TABLE[case]:
                    segments.append((edge_pt_f(ea), edge_pt_f(eb)))

        return MarchingSquares._connect_segments(segments)

    @staticmethod
    def _connect_segments(segments: List[Tuple[Point, Point]]) -> List[Polyline]:
        if not segments:
            return []

        def key(p: Point):
            return (round(p[0], 8), round(p[1], 8))

        ep: dict = {}
        for i, (a, b) in enumerate(segments):
            ep.setdefault(key(a), []).append((i, 0))
            ep.setdefault(key(b), []).append((i, 1))

        used = [False] * len(segments)

        def pop_next(pt):
            for entry in ep.get(key(pt), []):
                i, end = entry
                if not used[i]:
                    used[i] = True
                    sa, sb = segments[i]
                    return sb if end == 0 else sa
            return None

        result = []
        for start in range(len(segments)):
            if used[start]:
                continue
            used[start] = True
            a, b = segments[start]
            forward: List[Point] = [b]
            while True:
                nxt = pop_next(forward[-1])
                if nxt is None:
                    break
                forward.append(nxt)
            backward: List[Point] = [a]
            while True:
                nxt = pop_next(backward[-1])
                if nxt is None:
                    break
                backward.append(nxt)
            pts = list(reversed(backward)) + forward
            result.append(Polyline(pts))

        return result
