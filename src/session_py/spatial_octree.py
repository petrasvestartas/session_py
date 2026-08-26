from __future__ import annotations
# SpatialOctree — Potree-style multi-resolution LOD octree over bare 3D points.
# Use for: level-of-detail point-cloud rendering. Each node owns a spacing-limited
#   SUBSAMPLE of the points (grid accept, first point wins, leftovers descend into
#   octants at half the spacing), so drawing shallow nodes far away and deep nodes up
#   close gives Potree's uniform on-screen density. `order()` is the permutation that
#   makes every node's points CONTIGUOUS - upload points in that order and a node is
#   one (first, count) range.
# Prefer over SpatialKDTree when the question is "which points at what density",
#   not "which point is nearest".
# Note: static structure; rebuild required after point insertion.
import math

from .point import Point

# Duplicate points can never be separated by subdivision: below this level the node
# absorbs everything instead of recursing forever (spacing has shrunk by 2^21 anyway).
_MAX_LEVEL = 21


class SpatialOctree:
    """LOD octree: per-node spacing-limited subsamples over a reordered point set.

    Build on construction; the root cube is the bounding box grown to a cube.
    Complements SpatialKDTree (nearest queries) and SpatialRTree (box queries).
    """

    class _Node:
        __slots__ = ("min", "size", "level", "spacing", "first", "count", "children")

        def __init__(self, min_: list[float], size: float, level: int, spacing: float, first: int):
            self.min = min_
            self.size = size
            self.level = level
            self.spacing = spacing
            self.first = first
            self.count = 0
            self.children = [-1] * 8

    def __init__(self, points: list[Point], root_spacing: float, leaf_capacity: int):
        coords: list[float] = []
        for p in points:
            coords.append(p[0])
            coords.append(p[1])
            coords.append(p[2])
        self._init(coords, root_spacing, leaf_capacity)

    @classmethod
    def from_coords(cls, coords: list[float], root_spacing: float, leaf_capacity: int) -> "SpatialOctree":
        # Coords are only read during construction - nothing is stored, so a renderer
        # can hand its flat table over without a copy.
        tree = cls.__new__(cls)
        tree._init(coords, root_spacing, leaf_capacity)
        return tree

    def _init(self, coords: list[float], root_spacing: float, leaf_capacity: int) -> None:
        self._nodes: list[SpatialOctree._Node] = []
        self._order: list[int] = []
        n = len(coords) // 3
        if n == 0:
            return
        lo = [coords[0], coords[1], coords[2]]
        hi = [coords[0], coords[1], coords[2]]
        for i in range(1, n):
            for k in range(3):
                lo[k] = min(lo[k], coords[i * 3 + k])
                hi[k] = max(hi[k], coords[i * 3 + k])
        size = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        if size <= 0.0:
            size = 1.0
        root_min = [(lo[k] + hi[k]) * 0.5 - size * 0.5 for k in range(3)]
        self._build(coords, root_min, size, 0, root_spacing, list(range(n)), leaf_capacity)

    def _build(self, coords: list[float], min_: list[float], size: float, level: int, spacing: float, idxs: list[int], leaf_capacity: int) -> int:
        node = SpatialOctree._Node(min_, size, level, spacing, len(self._order))
        node_id = len(self._nodes)
        self._nodes.append(node)
        if len(idxs) <= leaf_capacity or level >= _MAX_LEVEL:
            self._order.extend(idxs)
            node.count = len(idxs)
            return node_id
        cells = max(1, int(math.ceil(size / spacing)))
        center = [min_[k] + size * 0.5 for k in range(3)]
        seen: set = set()
        accepted: list[int] = []
        buckets: list[list[int]] = [[] for _ in range(8)]
        for i in idxs:
            key = []
            for k in range(3):
                c = int(math.floor((coords[i * 3 + k] - min_[k]) / spacing))
                key.append(min(max(c, 0), cells - 1))
            key = tuple(key)
            if key not in seen:
                seen.add(key)
                accepted.append(i)
            else:
                b = 0
                if coords[i * 3] >= center[0]:
                    b |= 1
                if coords[i * 3 + 1] >= center[1]:
                    b |= 2
                if coords[i * 3 + 2] >= center[2]:
                    b |= 4
                buckets[b].append(i)
        self._order.extend(accepted)
        node.count = len(accepted)
        half = size * 0.5
        for b in range(8):
            if buckets[b]:
                child_min = [
                    min_[0] + (b & 1) * half,
                    min_[1] + ((b >> 1) & 1) * half,
                    min_[2] + ((b >> 2) & 1) * half,
                ]
                child_id = self._build(coords, child_min, half, level + 1, spacing * 0.5, buckets[b], leaf_capacity)
                self._nodes[node_id].children[b] = child_id
        return node_id

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
        return [c for c in self._nodes[i].children if c >= 0]

    def order(self) -> list[int]:
        return self._order
