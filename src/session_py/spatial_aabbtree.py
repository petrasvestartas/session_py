from __future__ import annotations

from .aabb import AABB

STACK_SIZE = 64
NULL_IDX = -1


class Node:
    """Tree node."""

    def __init__(self, aabb: AABB, right: int, object_id: int):
        """Construct from bounds, right child index and object id."""
        self.aabb = aabb
        self.right = right
        self.object_id = object_id


class _Range:
    """Pending id range of the build stack."""

    def __init__(self, lo: int, hi: int, parent: int, is_left: bool):
        """Construct from id range, parent index and side."""

        self.lo = lo
        self.hi = hi
        self.parent = parent
        self.is_left = is_left


class SpatialAABBTree:
    """Flat AABB tree with longest-axis median split; the left child of node i is i + 1, the right child is stored."""

    def __init__(self):
        """Construct an empty tree."""
        self.nodes: list[Node] = []

    def empty(self) -> bool:
        """Return whether the tree has no nodes."""
        return len(self.nodes) == 0

    def size(self) -> int:
        """Return the node count."""
        return len(self.nodes)

    def build(self, aabbs: list[AABB]) -> None:
        """Build the tree over the boxes, one leaf per box."""

        self.nodes = []
        n = len(aabbs)
        ids = list(range(n))
        stack: list[_Range] = []

        if n > 0:
            stack.append(_Range(0, n, NULL_IDX, False))

        while len(stack) > 0:
            range_ = stack.pop()
            node = len(self.nodes)
            aabb = self._bounds(ids, range_.lo, range_.hi, aabbs)
            self.nodes.append(Node(aabb, NULL_IDX, NULL_IDX))

            if range_.parent != NULL_IDX and not range_.is_left:
                self.nodes[range_.parent].right = node

            if range_.hi - range_.lo == 1:
                self.nodes[node].object_id = ids[range_.lo]
                continue

            axis = self._longest_axis(self.nodes[node].aabb)
            mid = range_.lo + (range_.hi - range_.lo) // 2
            self._nth_element(ids, range_.lo, mid, range_.hi, axis, aabbs)

            assert len(stack) + 2 <= STACK_SIZE
            stack.append(_Range(mid, range_.hi, node, False))
            stack.append(_Range(range_.lo, mid, node, True))

    def query_aabb(self, query: AABB) -> list[int]:
        """Return the ids of every leaf box that intersects query."""

        hits: list[int] = []
        stack: list[int] = []

        if len(self.nodes) > 0:
            stack.append(0)

        while len(stack) > 0:
            idx = stack.pop()
            node = self.nodes[idx]

            if not node.aabb.intersects(query):
                continue

            if node.object_id != NULL_IDX:
                hits.append(node.object_id)
                continue

            assert len(stack) + 2 <= STACK_SIZE
            stack.append(idx + 1)
            stack.append(node.right)

        return hits

    def _bounds(self, ids: list[int], lo: int, hi: int, aabbs: list[AABB]) -> AABB:
        """Return the box enclosing ids[lo, hi)."""

        aabb = aabbs[ids[lo]]

        for i in range(lo + 1, hi):
            aabb = AABB.merge(aabb, aabbs[ids[i]])

        return aabb

    def _longest_axis(self, aabb: AABB) -> int:
        """Return the axis of the largest half-size."""

        if aabb.hx >= aabb.hy and aabb.hx >= aabb.hz:
            return 0

        if aabb.hy >= aabb.hz:
            return 1

        return 2

    def _center(self, aabb: AABB, axis: int) -> float:
        """Return the center coordinate of aabb along axis."""

        if axis == 0:
            return aabb.cx

        if axis == 1:
            return aabb.cy

        return aabb.cz

    def _nth_element(
        self, ids: list[int], lo: int, mid: int, hi: int, axis: int, aabbs: list[AABB]
    ) -> None:
        """Partition ids[lo, hi) so that ids[mid] sits at its sorted rank by center along axis."""

        while hi - lo > 1:
            pivot = self._center(aabbs[ids[(lo + hi) // 2]], axis)
            i = lo
            j = hi - 1

            while i <= j:
                while self._center(aabbs[ids[i]], axis) < pivot:
                    i += 1

                while self._center(aabbs[ids[j]], axis) > pivot:
                    j -= 1

                if i <= j:
                    ids[i], ids[j] = ids[j], ids[i]
                    i += 1
                    j -= 1

            if mid <= j:
                hi = j + 1
            elif mid >= i:
                lo = i
            else:
                return
