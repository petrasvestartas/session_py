from __future__ import annotations

import math
from operator import itemgetter

from .point import Point

STACK_SIZE = 64
NULL_IDX = -1


class _Node:
    """A split node: point index, axis and children."""

    def __init__(self, idx: int, axis: int, left: int, right: int):
        """Construct a node."""

        self.idx = idx
        self.axis = axis
        self.left = left
        self.right = right


class _Range:
    """A pending index range on the build stack."""

    def __init__(self, lo: int, hi: int, depth: int, parent: int, is_left: bool):
        """Construct a range."""

        self.lo = lo
        self.hi = hi
        self.depth = depth
        self.parent = parent
        self.is_left = is_left


class _Visit:
    """A pending node on the query stack with its distance bound."""

    def __init__(self, node: int, bound: float):
        """Construct a visit."""
        self.node = node
        self.bound = bound


class SpatialKDTree:
    """KD-tree with alternating-axis median split over points for nearest, k-nearest and radius queries."""

    def __init__(self, points: list[Point]):
        """Construct the tree over points."""
        self._points = list(points)
        self._nodes: list[_Node] = []
        self._build()

    def _build(self) -> None:
        """Build the nodes by iterative median splits over an explicit stack."""

        n = len(self._points)
        indices = list(range(n))
        stack: list[_Range] = []

        if n > 0:
            stack.append(_Range(0, n, 0, NULL_IDX, False))

        while len(stack) > 0:
            range_ = stack.pop()
            axis = range_.depth % 3
            mid = range_.lo + (range_.hi - range_.lo) // 2
            self._nth_element(indices, range_.lo, mid, range_.hi, axis)
            node = len(self._nodes)
            self._nodes.append(_Node(indices[mid], axis, NULL_IDX, NULL_IDX))

            if range_.parent != NULL_IDX and range_.is_left:
                self._nodes[range_.parent].left = node

            if range_.parent != NULL_IDX and not range_.is_left:
                self._nodes[range_.parent].right = node

            if range_.lo < mid:
                assert len(stack) < STACK_SIZE
                stack.append(_Range(range_.lo, mid, range_.depth + 1, node, True))

            if mid + 1 < range_.hi:
                assert len(stack) < STACK_SIZE
                stack.append(_Range(mid + 1, range_.hi, range_.depth + 1, node, False))

    def _nth_element(
        self, indices: list[int], lo: int, mid: int, hi: int, axis: int
    ) -> None:
        """Partition indices[lo:hi] so indices[mid] holds the median along axis."""

        while hi - lo > 1:
            pivot = self._points[indices[(lo + hi) // 2]][axis]
            i = lo
            j = hi - 1

            while i <= j:
                while self._points[indices[i]][axis] < pivot:
                    i += 1

                while self._points[indices[j]][axis] > pivot:
                    j -= 1

                if i <= j:
                    indices[i], indices[j] = indices[j], indices[i]
                    i += 1
                    j -= 1

            if mid <= j:
                hi = j + 1
            elif mid >= i:
                lo = i
            else:
                return

    def _push(self, stack: list[_Visit], node: int, bound: float) -> None:
        """Push a node with its bound onto the visit stack."""

        if node == NULL_IDX:
            return

        assert len(stack) < STACK_SIZE
        stack.append(_Visit(node, bound))

    def _dist_sq(self, a: Point, b: Point) -> float:
        """Return the squared distance between a and b."""

        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]

        return dx * dx + dy * dy + dz * dz

    def _insert_sorted(
        self, best: list[tuple[int, float]], idx: int, d2: float, k: int
    ) -> None:
        """Insert (idx, d2) into best keeping it sorted and at most k long."""

        pos = len(best)

        while pos > 0 and best[pos - 1][1] > d2:
            pos -= 1

        best.insert(pos, (idx, d2))

        if len(best) > k:
            best.pop()

    def nearest(self, query: Point) -> tuple[int, float]:
        """Return the index and distance of the nearest point."""

        best = 0
        best_d2 = math.inf
        stack: list[_Visit] = []

        if len(self._nodes) > 0:
            self._push(stack, 0, 0.0)

        while len(stack) > 0:
            visit = stack.pop()

            if visit.bound >= best_d2:
                continue

            node = self._nodes[visit.node]
            d2 = self._dist_sq(query, self._points[node.idx])

            if d2 < best_d2:
                best_d2 = d2
                best = node.idx

            diff = query[node.axis] - self._points[node.idx][node.axis]
            near = node.left if diff <= 0 else node.right
            far = node.right if diff <= 0 else node.left
            self._push(stack, far, diff * diff)
            self._push(stack, near, 0.0)

        return best, math.sqrt(best_d2)

    def nearest_k(self, query: Point, k: int) -> list[tuple[int, float]]:
        """Return the k nearest (index, distance) pairs sorted by distance."""

        best: list[tuple[int, float]] = []

        if k <= 0:
            return best

        stack: list[_Visit] = []

        if len(self._nodes) > 0:
            self._push(stack, 0, 0.0)

        while len(stack) > 0:
            visit = stack.pop()
            full = len(best) == k

            if full and visit.bound >= best[-1][1]:
                continue

            node = self._nodes[visit.node]
            d2 = self._dist_sq(query, self._points[node.idx])

            if not full or d2 < best[-1][1]:
                self._insert_sorted(best, node.idx, d2, k)

            diff = query[node.axis] - self._points[node.idx][node.axis]
            near = node.left if diff <= 0 else node.right
            far = node.right if diff <= 0 else node.left
            self._push(stack, far, diff * diff)
            self._push(stack, near, 0.0)

        for i in range(len(best)):
            best[i] = (best[i][0], math.sqrt(best[i][1]))

        return best

    def radius_search(self, query: Point, radius: float) -> list[tuple[int, float]]:
        """Return every (index, distance) pair within radius sorted by distance."""

        result: list[tuple[int, float]] = []
        r2 = radius * radius
        stack: list[_Visit] = []

        if len(self._nodes) > 0:
            self._push(stack, 0, 0.0)

        while len(stack) > 0:
            visit = stack.pop()

            if visit.bound > r2:
                continue

            node = self._nodes[visit.node]
            d2 = self._dist_sq(query, self._points[node.idx])

            if d2 <= r2:
                result.append((node.idx, math.sqrt(d2)))

            diff = query[node.axis] - self._points[node.idx][node.axis]
            near = node.left if diff <= 0 else node.right
            far = node.right if diff <= 0 else node.left
            self._push(stack, far, diff * diff)
            self._push(stack, near, 0.0)

        result.sort(key=itemgetter(1))

        return result
