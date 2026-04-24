# SpatialKDTree — alternating-axis median split over bare 3D points.
# Use for: k-nearest-neighbor queries on point clouds (fastest option).
#   Points only — no volumes, no boxes, no rotation.
# Prefer over SpatialAABBTree/SpatialBVH when data is a point cloud, not triangle faces.
# Prefer over SpatialRTree   when queries are k-NN, not region overlap.
# Note: static structure; rebuild required after point insertion.
import math
from typing import List, Optional, Tuple

from .point import Point


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

    def __init__(self, points: List[Point]):
        self._points = list(points)
        self._root = self._build(list(range(len(points))), 0) if points else None

    def _build(self, indices: List[int], depth: int) -> Optional["SpatialKDTree._Node"]:
        if not indices:
            return None
        axis = depth % 3
        indices.sort(key=lambda i: self._points[i][axis])
        mid = len(indices) // 2
        return SpatialKDTree._Node(
            idx=indices[mid],
            axis=axis,
            left=self._build(indices[:mid], depth + 1),
            right=self._build(indices[mid + 1:], depth + 1),
        )

    @staticmethod
    def _dist_sq(a: Point, b: Point) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return dx * dx + dy * dy + dz * dz

    def _nearest_1(self, node: Optional["SpatialKDTree._Node"], query: Point, best: List) -> None:
        if node is None:
            return
        d = self._dist_sq(query, self._points[node.idx])
        if best[0] is None or d < best[1]:
            best[0] = node.idx
            best[1] = d
        diff = query[node.axis] - self._points[node.idx][node.axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._nearest_1(near, query, best)
        if diff * diff < best[1]:
            self._nearest_1(far, query, best)

    def nearest(self, query: Point) -> Tuple[int, float]:
        best = [None, float("inf")]
        self._nearest_1(self._root, query, best)
        return best[0], math.sqrt(best[1])

    def _nearest_k(self, node: Optional["SpatialKDTree._Node"], query: Point, k: int, heap: List) -> None:
        if node is None:
            return
        d = self._dist_sq(query, self._points[node.idx])
        if len(heap) < k:
            heap.append((-d, node.idx))
            heap.sort()
        elif d < -heap[0][0]:
            heap[0] = (-d, node.idx)
            heap.sort()
        diff = query[node.axis] - self._points[node.idx][node.axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._nearest_k(near, query, k, heap)
        if len(heap) < k or diff * diff < -heap[0][0]:
            self._nearest_k(far, query, k, heap)

    def nearest_k(self, query: Point, k: int) -> List[Tuple[int, float]]:
        heap: List = []
        self._nearest_k(self._root, query, k, heap)
        return sorted([(idx, math.sqrt(-neg_d)) for neg_d, idx in heap], key=lambda x: x[1])

    def _radius(self, node: Optional["SpatialKDTree._Node"], query: Point, radius_sq: float, result: List) -> None:
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

    def radius_search(self, query: Point, radius: float) -> List[Tuple[int, float]]:
        result: List = []
        self._radius(self._root, query, radius * radius, result)
        return sorted(result, key=lambda x: x[1])
