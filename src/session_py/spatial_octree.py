from __future__ import annotations

import math

from .point import Point

MAX_LEVEL = 21
STACK_SIZE = 8 * MAX_LEVEL
NULL_IDX = -1


class _Node:
    def __init__(self, min_: list[float], size: float, level: int, spacing: float, first: int, count: int, children: list[int]):
        self.min = min_
        self.size = size
        self.level = level
        self.spacing = spacing
        self.first = first
        self.count = count
        self.children = children


class _Task:
    def __init__(self, min_: list[float], size: float, level: int, spacing: float, lo: int, hi: int, parent: int, octant: int):
        self.min = min_
        self.size = size
        self.level = level
        self.spacing = spacing
        self.lo = lo
        self.hi = hi
        self.parent = parent
        self.octant = octant


class SpatialOctree:
    """Potree-style LOD octree: every node keeps a spacing-limited subsample and order() makes each node's points contiguous."""

    def __init__(self, points: list[Point], root_spacing: float, leaf_capacity: int):
        coords: list[float] = []
        for p in points:
            coords.append(p[0])
            coords.append(p[1])
            coords.append(p[2])
        self._nodes: list[_Node] = []
        self._order: list[int] = []
        self._build(coords, root_spacing, leaf_capacity)

    @classmethod
    def from_coords(cls, coords: list[float], root_spacing: float, leaf_capacity: int) -> SpatialOctree:
        tree = cls.__new__(cls)
        tree._nodes = []
        tree._order = []
        tree._build(coords, root_spacing, leaf_capacity)
        return tree

    def _root_cube(self, coords: list[float]) -> tuple[list[float], float]:
        n = len(coords) // 3
        lo = [coords[0], coords[1], coords[2]]
        hi = [coords[0], coords[1], coords[2]]
        for i in range(1, n):
            for k in range(3):
                lo[k] = min(lo[k], coords[i * 3 + k])
                hi[k] = max(hi[k], coords[i * 3 + k])
        size = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        if size <= 0.0:
            size = 1.0
        min_ = [0.0, 0.0, 0.0]
        for k in range(3):
            min_[k] = (lo[k] + hi[k]) * 0.5 - size * 0.5
        return min_, size

    def _build(self, coords: list[float], root_spacing: float, leaf_capacity: int) -> None:
        n = len(coords) // 3
        if n == 0:
            return
        root_min, root_size = self._root_cube(coords)
        indices = list(range(n))
        stack: list[_Task] = []
        self._push(stack, _Task(root_min, root_size, 0, root_spacing, 0, n, NULL_IDX, 0))
        while len(stack) > 0:
            task = stack.pop()
            node = len(self._nodes)
            self._nodes.append(_Node(task.min, task.size, task.level, task.spacing, len(self._order), 0, [NULL_IDX] * 8))
            if task.parent != NULL_IDX:
                self._nodes[task.parent].children[task.octant] = node
            if task.hi - task.lo <= leaf_capacity or task.level >= MAX_LEVEL:
                self._order.extend(indices[task.lo:task.hi])
                self._nodes[node].count = task.hi - task.lo
                continue
            bounds = self._accept(coords, task, indices)
            self._nodes[node].count = len(self._order) - self._nodes[node].first
            half = task.size * 0.5
            for b in range(7, -1, -1):
                if bounds[b] == bounds[b + 1]:
                    continue
                min_ = [
                    task.min[0] + (b & 1) * half,
                    task.min[1] + ((b >> 1) & 1) * half,
                    task.min[2] + ((b >> 2) & 1) * half,
                ]
                self._push(stack, _Task(min_, half, task.level + 1, task.spacing * 0.5, bounds[b], bounds[b + 1], node, b))

    def _accept(self, coords: list[float], task: _Task, indices: list[int]) -> list[int]:
        cells = max(1, math.ceil(task.size / task.spacing))
        half = task.size * 0.5
        center = [task.min[0] + half, task.min[1] + half, task.min[2] + half]
        seen: set[tuple[int, int, int]] = set()
        buckets: list[list[int]] = [[], [], [], [], [], [], [], []]
        for i in range(task.lo, task.hi):
            idx = indices[i]
            key = [0, 0, 0]
            for k in range(3):
                key[k] = min(max(math.floor((coords[idx * 3 + k] - task.min[k]) / task.spacing), 0), cells - 1)
            if (key[0], key[1], key[2]) not in seen:
                seen.add((key[0], key[1], key[2]))
                self._order.append(idx)
                continue
            octant = 0
            for k in range(3):
                if coords[idx * 3 + k] >= center[k]:
                    octant |= 1 << k
            buckets[octant].append(idx)
        bounds = [0] * 9
        bounds[0] = task.lo
        for b in range(8):
            bounds[b + 1] = bounds[b] + len(buckets[b])
            indices[bounds[b]:bounds[b + 1]] = buckets[b]
        return bounds

    def _push(self, stack: list[_Task], task: _Task) -> None:
        assert len(stack) < STACK_SIZE
        stack.append(task)

    def node_count(self) -> int:
        return len(self._nodes)

    def node_cube(self, i: int) -> tuple[Point, float]:
        node = self._nodes[i]
        half = node.size * 0.5
        return Point(node.min[0] + half, node.min[1] + half, node.min[2] + half), node.size

    def node_level(self, i: int) -> int:
        return self._nodes[i].level

    def node_spacing(self, i: int) -> float:
        return self._nodes[i].spacing

    def node_range(self, i: int) -> tuple[int, int]:
        node = self._nodes[i]
        return node.first, node.count

    def children(self, i: int) -> list[int]:
        result: list[int] = []
        for c in self._nodes[i].children:
            if c != NULL_IDX:
                result.append(c)
        return result

    def order(self) -> list[int]:
        return self._order
