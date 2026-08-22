from __future__ import annotations
# SpatialKDTree — alternating-axis median split over bare 3D points.
# Use for: k-nearest-neighbor queries on point clouds (fastest option).
#   Points only — no volumes, no boxes, no rotation.
# Prefer over SpatialAABBTree/SpatialBVH when data is a point cloud, not triangle faces.
# Prefer over SpatialRTree   when queries are k-NN, not region overlap.
# Note: static structure; rebuild required after point insertion.
from typing import List
from typing import Optional
from typing import Tuple
import math

from .point import Point


def _nth_element(a: list[int], lo: int, mid: int, hi: int, key) -> None:
    """Place the mid-th element in sorted position within a[lo:hi] (quickselect)."""
    while hi - lo > 1:
        pivot = key(a[(lo + hi) // 2])
        i = lo
        j = hi - 1
        while i <= j:
            while key(a[i]) < pivot:
                i += 1
            while key(a[j]) > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        if mid <= j:
            hi = j + 1
        elif mid >= i:
            lo = i
        else:
            return


def _heap_push(heap: list, item) -> None:
    heap.append(item)
    i = len(heap) - 1
    while i > 0 and heap[(i - 1) // 2][0] < heap[i][0]:
        heap[i], heap[(i - 1) // 2] = heap[(i - 1) // 2], heap[i]
        i = (i - 1) // 2


def _heap_replace(heap: list, item) -> None:
    heap[0] = item
    i = 0
    while True:
        l = 2 * i + 1
        r = 2 * i + 2
        m = i
        if l < len(heap) and heap[l][0] > heap[m][0]:
            m = l
        if r < len(heap) and heap[r][0] > heap[m][0]:
            m = r
        if m == i:
            break
        heap[i], heap[m] = heap[m], heap[i]
        i = m


class SpatialKDTree:
    """KD-tree for point-to-point nearest-neighbor queries.

    Build on construction using alternating-axis median split.
    Complements SpatialRTree (box queries) and SpatialBVH (collision/ray).
    """

    class _Node:
        __slots__ = ("idx", "axis", "left", "right")

        def __init__(self, idx: int, axis: int, left: Optional["SpatialKDTree._Node"], right: Optional["SpatialKDTree._Node"]):
            self.idx = idx
            self.axis = axis
            self.left = left
            self.right = right

    def __init__(self, points: list[Point]):
        self._points = list(points)
        self._root = self._build(list(range(len(points))), 0, len(points), 0) if points else None

    def _build(self, indices: list[int], lo: int, hi: int, depth: int) -> Optional["SpatialKDTree._Node"]:
        if lo >= hi:
            return None
        axis = depth % 3
        mid = lo + (hi - lo) // 2
        _nth_element(indices, lo, mid, hi, lambda i: self._points[i][axis])
        return SpatialKDTree._Node(
            idx=indices[mid],
            axis=axis,
            left=self._build(indices, lo, mid, depth + 1),
            right=self._build(indices, mid + 1, hi, depth + 1),
        )

    @staticmethod
    def _dist_sq(a: Point, b: Point) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return dx * dx + dy * dy + dz * dz

    def _nearest_1(self, node: Optional["SpatialKDTree._Node"], query: Point, best: list) -> None:
        if node is None:
            return
        d = self._dist_sq(query, self._points[node.idx])
        if d < best[1]:
            best[0] = node.idx
            best[1] = d
        diff = query[node.axis] - self._points[node.idx][node.axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._nearest_1(near, query, best)
        if diff * diff < best[1]:
            self._nearest_1(far, query, best)

    def nearest(self, query: Point) -> tuple[int, float]:
        best = [0, float("inf")]
        self._nearest_1(self._root, query, best)
        return best[0], math.sqrt(best[1])

    def _nearest_k(self, node: Optional["SpatialKDTree._Node"], query: Point, k: int, heap: list) -> None:
        if node is None:
            return
        d = self._dist_sq(query, self._points[node.idx])
        if len(heap) < k:
            _heap_push(heap, (d, node.idx))
        elif d < heap[0][0]:
            _heap_replace(heap, (d, node.idx))
        diff = query[node.axis] - self._points[node.idx][node.axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._nearest_k(near, query, k, heap)
        if len(heap) < k or diff * diff < heap[0][0]:
            self._nearest_k(far, query, k, heap)

    def nearest_k(self, query: Point, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        heap: list = []
        self._nearest_k(self._root, query, k, heap)
        return sorted([(idx, math.sqrt(d)) for d, idx in heap], key=lambda x: x[1])

    def _radius(self, node: Optional["SpatialKDTree._Node"], query: Point, radius_sq: float, result: list) -> None:
        if node is None:
            return
        d = self._dist_sq(query, self._points[node.idx])
        if d <= radius_sq:
            result.append((node.idx, math.sqrt(d)))
        diff = query[node.axis] - self._points[node.idx][node.axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._radius(near, query, radius_sq, result)
        if diff * diff <= radius_sq:
            self._radius(far, query, radius_sq, result)

    def radius_search(self, query: Point, radius: float) -> list[tuple[int, float]]:
        result: list = []
        self._radius(self._root, query, radius * radius, result)
        return sorted(result, key=lambda x: x[1])
