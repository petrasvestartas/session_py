# SpatialAABBTree — flat contiguous BVH over axis-aligned boxes (SAH median split).
# Use for: closest-point on static mesh faces, ray-mesh intersection.
#   Build once, query many times. Cache-friendly nodes.
# Prefer over SpatialBVH  when geometry is static and all volumes are world-aligned.
# Prefer over SpatialRTree when no dynamic insert/delete is needed.
# Prefer over SpatialKDTree when querying faces/volumes, not bare point clouds.
from typing import List
from .aabb import AABB


class _Node:
    __slots__ = ("aabb", "right", "object_id")

    def __init__(self, aabb: AABB = None, right: int = -1, object_id: int = -1):
        self.aabb = aabb if aabb is not None else AABB(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.right = right
        self.object_id = object_id


class SpatialAABBTree:
    def __init__(self):
        self.nodes: List[_Node] = []

    def empty(self) -> bool:
        return len(self.nodes) == 0

    def size(self) -> int:
        return len(self.nodes)

    def build(self, aabbs: List[AABB]) -> None:
        self.nodes = []
        if not aabbs:
            return
        ids = list(range(len(aabbs)))
        self._build_node(ids, aabbs)

    def query_aabb(self, query: AABB) -> List[int]:
        hits: List[int] = []
        if self.empty():
            return hits
        stack = [0]
        while stack:
            idx = stack.pop()
            node = self.nodes[idx]
            if not node.aabb.intersects(query):
                continue
            if node.object_id >= 0:
                hits.append(node.object_id)
            else:
                stack.append(idx + 1)
                stack.append(node.right)
        return hits

    def _build_node(self, ids: List[int], aabbs: List[AABB]) -> None:
        idx = len(self.nodes)
        self.nodes.append(_Node())

        lo_x = lo_y = lo_z = 1e308
        hi_x = hi_y = hi_z = -1e308
        for i in ids:
            b = aabbs[i]
            lo_x = min(lo_x, b.cx - b.hx); hi_x = max(hi_x, b.cx + b.hx)
            lo_y = min(lo_y, b.cy - b.hy); hi_y = max(hi_y, b.cy + b.hy)
            lo_z = min(lo_z, b.cz - b.hz); hi_z = max(hi_z, b.cz + b.hz)
        self.nodes[idx].aabb = AABB(
            (lo_x + hi_x) * 0.5, (lo_y + hi_y) * 0.5, (lo_z + hi_z) * 0.5,
            (hi_x - lo_x) * 0.5, (hi_y - lo_y) * 0.5, (hi_z - lo_z) * 0.5,
        )

        if len(ids) == 1:
            self.nodes[idx].object_id = ids[0]
            return

        dx = hi_x - lo_x; dy = hi_y - lo_y; dz = hi_z - lo_z
        axis = 0 if (dx >= dy and dx >= dz) else (1 if dy >= dz else 2)
        mid = len(ids) // 2

        key = (lambda i: aabbs[i].cx) if axis == 0 else (lambda i: aabbs[i].cy) if axis == 1 else (lambda i: aabbs[i].cz)
        ids.sort(key=key)

        left_ids = ids[:mid]
        right_ids = ids[mid:]
        self._build_node(left_ids, aabbs)
        self.nodes[idx].right = len(self.nodes)
        self._build_node(right_ids, aabbs)
