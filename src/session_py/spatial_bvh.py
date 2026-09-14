from __future__ import annotations

import math
import uuid
from .aabb import AABB
from .obb import OBB
from .point import Point
from .vector import Vector

STACK_SIZE = 64
NULL_IDX = -1


class Node:
    def __init__(
        self,
        aabb: AABB | None = None,
        left: int = NULL_IDX,
        right: int = NULL_IDX,
        object_id: int = NULL_IDX,
    ):
        self.aabb = AABB() if aabb is None else aabb
        self.left = left
        self.right = right
        self.object_id = object_id

    def is_leaf(self) -> bool:
        return self.object_id != NULL_IDX


def _quantize(t: float) -> int:
    return int(min(max(t, 0.0), 1.0) * 1023.0)


class SpatialBVH:
    """Linear BVH (Karras 2012): leaves in Morton order, internal node i splits the sorted range it covers, node 0 is the root."""

    def __init__(self, world_size: float = 1000.0):
        self._guid = None
        self.name = "my_bvh"
        self.world_size = world_size
        self.object_guids: list[str] = []
        self.nodes: list[Node] = []

    def has_guid(self) -> bool:
        return getattr(self, "_guid", None) is not None

    @property
    def guid(self) -> str:
        if getattr(self, "_guid", None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    @staticmethod
    def from_boxes(bounding_boxes: list[OBB], world_size: float) -> SpatialBVH:
        bvh = SpatialBVH(world_size)
        bvh.build(bounding_boxes)
        return bvh

    def empty(self) -> bool:
        return len(self.nodes) == 0

    def size(self) -> int:
        return len(self.nodes)

    @staticmethod
    def compute_world_size(bounding_boxes: list[OBB]) -> float:
        """Largest absolute box coordinate times 2.2, at least 10"""
        if len(bounding_boxes) == 0:
            return 1000.0
        max_extent = 0.0
        for bbox in bounding_boxes:
            for k in range(3):
                max_extent = max(max_extent, abs(bbox.center[k]) + bbox.half_size[k])
        return max(max_extent * 2.2, 10.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # Build
    # ═══════════════════════════════════════════════════════════════════════════

    def build(self, bounding_boxes: list[OBB]) -> None:
        self.build_from_boxes(bounding_boxes, self.world_size)

    def build_from_boxes(self, boxes: list[OBB], ws: float) -> None:
        aabbs = []
        for bbox in boxes:
            aabbs.append(self._aabb_from_obb(bbox))
        self.build_from_aabbs(aabbs, ws)

    def build_from_aabbs(self, aabbs: list[AABB], ws: float) -> None:
        self.world_size = ws
        self.nodes = []
        n = len(aabbs)
        if n == 0:
            return
        codes = self._sorted_codes(aabbs)
        leaf = n - 1
        for i in range(n - 1):
            self.nodes.append(Node())
        for code in codes:
            self.nodes.append(Node(aabbs[code[1]], NULL_IDX, NULL_IDX, code[1]))
        order = []
        for i in range(n - 1):
            first, last = self._determine_range(codes, i)
            split = self._find_split(codes, first, last)
            self.nodes[i].left = leaf + split if split == first else split
            self.nodes[i].right = leaf + split + 1 if split + 1 == last else split + 1
            order.append((last - first, i))
        order.sort()
        for item in order:
            node = self.nodes[item[1]]
            node.aabb = AABB.merge(
                self.nodes[node.left].aabb, self.nodes[node.right].aabb
            )

    def build_with_guids(self, boxes_with_guids: list[tuple[OBB, str]]) -> None:
        """Boxes paired with their guids, world size computed from the boxes"""
        bounding_boxes = []
        self.object_guids = []
        for bbox, guid in boxes_with_guids:
            bounding_boxes.append(bbox)
            self.object_guids.append(guid)
        self.world_size = self.compute_world_size(bounding_boxes)
        self.build(bounding_boxes)

    def _sorted_codes(self, aabbs: list[AABB]) -> list[tuple[int, int]]:
        """(morton code, id) sorted by code, codes quantized over the bounding cube of the box centers"""
        lo = [0.0, 0.0, 0.0]
        hi = [0.0, 0.0, 0.0]
        for k in range(3):
            lo[k] = self._center(aabbs[0], k)
            hi[k] = lo[k]
        for aabb in aabbs[1:]:
            for k in range(3):
                lo[k] = min(lo[k], self._center(aabb, k))
                hi[k] = max(hi[k], self._center(aabb, k))
        ext = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        codes = []
        for i, aabb in enumerate(aabbs):
            code = 0
            for k in range(3):
                t = (self._center(aabb, k) - lo[k]) / ext if ext > 0.0 else 0.0
                code |= expand_bits(_quantize(t)) << k
            codes.append((code, i))
        codes.sort()
        return codes

    def _common_prefix(self, codes: list[tuple[int, int]], i: int, j: int) -> int:
        """Leading bits shared by codes i and j, ties broken by index; -1 when j is out of range"""
        if j < 0 or j >= len(codes):
            return -1
        if codes[i][0] != codes[j][0]:
            return 32 - (codes[i][0] ^ codes[j][0]).bit_length()
        return 32 + 32 - (i ^ j).bit_length()

    def _determine_range(self, codes: list[tuple[int, int]], i: int) -> tuple[int, int]:
        """Sorted range [first, last] covered by internal node i"""
        d = (
            1
            if self._common_prefix(codes, i, i + 1)
            > self._common_prefix(codes, i, i - 1)
            else -1
        )
        delta_min = self._common_prefix(codes, i, i - d)
        length = 1
        while self._common_prefix(codes, i, i + length * d) > delta_min:
            length *= 2
        bound = 0
        step = length // 2
        while step > 0:
            if self._common_prefix(codes, i, i + (bound + step) * d) > delta_min:
                bound += step
            step //= 2
        j = i + bound * d
        return (min(i, j), max(i, j))

    def _find_split(self, codes: list[tuple[int, int]], first: int, last: int) -> int:
        """Last index of the left half of [first, last]"""
        common = self._common_prefix(codes, first, last)
        split = first
        step = last - first
        while step > 1:
            step = (step + 1) // 2
            if (
                split + step < last
                and self._common_prefix(codes, first, split + step) > common
            ):
                split += step
        return split

    # ═══════════════════════════════════════════════════════════════════════════
    # Queries
    # ═══════════════════════════════════════════════════════════════════════════

    def check_all_collisions(
        self, bounding_boxes: list[OBB]
    ) -> tuple[list[tuple[int, int]], list[int], int]:
        """Overlapping (i, j) pairs with i < j, the ids in any pair, and the number of nodes tested"""
        pairs = []
        visited = [False] * len(bounding_boxes)
        total_checks = 0
        for i in range(len(bounding_boxes)):
            found = self.find_collisions(i, bounding_boxes[i], bounding_boxes)
            total_checks += found[1]
            for j in found[0]:
                if j < i:
                    continue
                pairs.append((i, j))
                visited[i] = True
                visited[j] = True
        colliding_indices = []
        for i in range(len(visited)):
            if visited[i]:
                colliding_indices.append(i)
        return (pairs, colliding_indices, total_checks)

    def check_all_collisions_guids(
        self, bounding_boxes: list[OBB]
    ) -> list[tuple[str, str]]:
        pairs, colliding_indices, total_checks = self.check_all_collisions(
            bounding_boxes
        )
        guid_pairs = []
        for i, j in pairs:
            if i < len(self.object_guids) and j < len(self.object_guids):
                guid_pairs.append((self.object_guids[i], self.object_guids[j]))
        return guid_pairs

    def find_collisions(
        self, object_id: int, query_bbox: OBB, bounding_boxes: list[OBB]
    ) -> tuple[list[int], int]:
        """Ids overlapping query_bbox other than object_id, and the number of nodes tested"""
        collisions = []
        check_count = 0
        query = self._aabb_from_obb(query_bbox)
        stack: list[int] = []
        if len(self.nodes) > 0:
            stack.append(0)
        while len(stack) > 0:
            node = self.nodes[stack.pop()]
            if not node.aabb.intersects(query):
                continue
            check_count += 1
            if node.is_leaf():
                id = node.object_id
                if (
                    id != object_id
                    and id < len(bounding_boxes)
                    and query.intersects(self._aabb_from_obb(bounding_boxes[id]))
                ):
                    collisions.append(id)
                continue
            assert len(stack) + 2 <= STACK_SIZE
            stack.append(node.left)
            stack.append(node.right)
        return (collisions, check_count)

    def query_aabb(self, query: AABB | OBB) -> list[int]:
        """Ids of every leaf box that intersects query"""
        if isinstance(query, OBB):
            query = self._aabb_from_obb(query)
        hits: list[int] = []
        stack: list[int] = []
        if len(self.nodes) > 0:
            stack.append(0)
        while len(stack) > 0:
            node = self.nodes[stack.pop()]
            if not node.aabb.intersects(query):
                continue
            if node.is_leaf():
                hits.append(node.object_id)
                continue
            assert len(stack) + 2 <= STACK_SIZE
            stack.append(node.left)
            stack.append(node.right)
        return hits

    def nearest_neighbors(
        self, object_id: int, bounding_boxes: list[OBB], inflate: float = 1.2
    ) -> list[int]:
        """Ids overlapping the box of object_id with its half-sizes scaled by inflate, object_id excluded"""
        result: list[int] = []
        if object_id < 0 or object_id >= len(bounding_boxes):
            return result
        query = self._aabb_from_obb(bounding_boxes[object_id])
        query.hx *= inflate
        query.hy *= inflate
        query.hz *= inflate
        for id in self.query_aabb(query):
            if id != object_id:
                result.append(id)
        return result

    def ray_cast(
        self,
        origin: Point,
        direction: Vector,
        candidate_leaf_ids: list[int],
        find_all: bool = False,
    ) -> bool:
        """Leaf ids whose box the ray enters, nearest entry first; True when any"""
        candidate_leaf_ids.clear()
        found = []
        stack: list[int] = []
        if len(self.nodes) > 0:
            stack.append(0)
        while len(stack) > 0:
            node = self.nodes[stack.pop()]
            span = self._ray_aabb(origin, direction, node.aabb)
            if span[1] < span[0] or span[1] < 0.0:
                continue
            if node.is_leaf():
                found.append((span[0], node.object_id))
                continue
            assert len(stack) + 2 <= STACK_SIZE
            stack.append(node.left)
            stack.append(node.right)
        found.sort()
        for hit in found:
            candidate_leaf_ids.append(hit[1])
        return len(candidate_leaf_ids) > 0

    def _ray_aabb(
        self, origin: Point, direction: Vector, aabb: AABB
    ) -> tuple[float, float]:
        """(entry, exit) ray parameters of the box slabs; a miss when exit < entry"""
        tmin = -math.inf
        tmax = math.inf
        for k in range(3):
            inv = 1.0 / direction[k] if direction[k] != 0.0 else math.inf
            t1 = (self._center(aabb, k) - self._half(aabb, k) - origin[k]) * inv
            t2 = (self._center(aabb, k) + self._half(aabb, k) - origin[k]) * inv
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
        return (tmin, tmax)

    # ═══════════════════════════════════════════════════════════════════════════
    # Boxes
    # ═══════════════════════════════════════════════════════════════════════════

    def merge_aabb(self, aabb1: OBB, aabb2: OBB) -> OBB:
        """Axis-aligned box enclosing both boxes"""
        merged = AABB.merge(self._aabb_from_obb(aabb1), self._aabb_from_obb(aabb2))
        return OBB(
            merged.center(),
            Vector(1, 0, 0),
            Vector(0, 1, 0),
            Vector(0, 0, 1),
            Vector(merged.hx, merged.hy, merged.hz),
        )

    def aabb_intersect(self, aabb1: AABB | OBB, aabb2: AABB | OBB) -> bool:
        if isinstance(aabb1, OBB):
            aabb1 = self._aabb_from_obb(aabb1)
        if isinstance(aabb2, OBB):
            aabb2 = self._aabb_from_obb(aabb2)
        return aabb1.intersects(aabb2)

    @staticmethod
    def _aabb_from_obb(obb: OBB) -> AABB:
        half = [0.0, 0.0, 0.0]
        for k in range(3):
            half[k] = (
                abs(obb.x_axis[k]) * obb.half_size[0]
                + abs(obb.y_axis[k]) * obb.half_size[1]
                + abs(obb.z_axis[k]) * obb.half_size[2]
            )
        return AABB(
            obb.center[0], obb.center[1], obb.center[2], half[0], half[1], half[2]
        )

    def _center(self, aabb: AABB, axis: int) -> float:
        if axis == 0:
            return aabb.cx
        if axis == 1:
            return aabb.cy
        return aabb.cz

    def _half(self, aabb: AABB, axis: int) -> float:
        if axis == 0:
            return aabb.hx
        if axis == 1:
            return aabb.hy
        return aabb.hz


# ═══════════════════════════════════════════════════════════════════════════
# Morton codes
# ═══════════════════════════════════════════════════════════════════════════


def expand_bits(v: int) -> int:
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v


def calculate_morton_code(
    x: float, y: float, z: float, world_size: float = 100.0
) -> int:
    half = world_size * 0.5
    ix = _quantize((x + half) / world_size)
    iy = _quantize((y + half) / world_size)
    iz = _quantize((z + half) / world_size)
    return expand_bits(ix) | (expand_bits(iy) << 1) | (expand_bits(iz) << 2)
