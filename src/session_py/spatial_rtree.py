from __future__ import annotations

from collections.abc import Callable

MAXNODES = 8  # Fan-out ceiling.
MINNODES = 4  # Fan-out floor.
NOT_TAKEN = -1  # Partition slot not yet assigned.
STACK_SIZE = 64  # Explicit traversal stack depth.


class _Rect:
    """Axis-aligned box."""

    def __init__(self):
        """Construct with zeroed fields."""

        self.m_min = [0.0, 0.0, 0.0]  # Minimum corner.
        self.m_max = [0.0, 0.0, 0.0]  # Maximum corner.

    def copy(self) -> _Rect:
        """Copy by value."""

        rect = _Rect()
        rect.m_min = list(self.m_min)
        rect.m_max = list(self.m_max)

        return rect


class _Branch:
    """Child pointer or leaf datum with its cover."""

    def __init__(self):
        """Construct with zeroed fields."""

        self.m_rect = _Rect()  # Cover of the child or datum.
        self.m_child = None  # Child node, none on leaves.
        self.m_data = 0  # Leaf datum.

    def copy(self) -> _Branch:
        """Copy by value."""

        branch = _Branch()
        branch.m_rect = self.m_rect.copy()
        branch.m_child = self.m_child
        branch.m_data = self.m_data

        return branch


class _Node:
    """Inner or leaf node with up to MAXNODES + 1 branches during a split."""

    def __init__(self):
        """Construct with zeroed fields."""

        self.m_count = 0  # Branches in use.
        self.m_level = 0  # 0 for leaves.
        self.m_branch = [None] * (MAXNODES + 1)  # Branch slots.

        for i in range(MAXNODES + 1):
            self.m_branch[i] = _Branch()

    def is_leaf(self) -> bool:
        """Whether the node is a leaf."""
        return self.m_level == 0


class _Visit:
    """Traversal stack entry."""

    def __init__(self, node: _Node, index: int):
        """Construct from a node and its next branch index."""

        self.node = node  # Node being walked.
        self.index = index  # Next branch to visit.


class _PartitionVars:
    """Scratch state for a quadratic split."""

    def __init__(self):
        """Construct with zeroed fields."""

        self.m_partition = [NOT_TAKEN] * (MAXNODES + 1)  # Group of each buffered branch.
        self.m_total = 0  # Buffered branch count.
        self.m_min_fill = 0  # Minimum branches per group.
        self.m_count = [0, 0]  # Branches per group.
        self.m_cover = [_Rect(), _Rect()]  # Cover per group.
        self.m_area = [0.0, 0.0]  # Cover volume per group.
        self.m_branch_buf = [None] * (MAXNODES + 1)  # Branches being split.
        self.m_branch_count = 0  # Branches in the buffer.
        self.m_cover_split = _Rect()  # Cover of the whole buffer.
        self.m_cover_split_area = 0.0  # Volume of the whole buffer cover.

        for i in range(MAXNODES + 1):
            self.m_branch_buf[i] = _Branch()


class SpatialRTree:
    """R-tree with dynamic insert and remove (Guttman quadratic split, fan-out 4 to 8) for box overlap queries."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self):
        """Construct an empty tree with a single leaf root."""

        self._m_root = self._alloc_node()  # Tree root.
        self._m_size = 0  # Stored item count.

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def count(self) -> int:
        """Number of stored items."""
        return self._m_size

    # ═══════════════════════════════════════════════════════════════════════════
    # Mutators
    # ═══════════════════════════════════════════════════════════════════════════
    def insert(self, a_min: list[float], a_max: list[float], a_data: int) -> None:
        """Insert an item with its bounding box."""

        branch = _Branch()
        branch.m_rect = self._make_rect(a_min, a_max)
        branch.m_child = None
        branch.m_data = a_data

        self._insert_branch_internal(branch, 0)
        self._m_size += 1

    def remove(self, a_min: list[float], a_max: list[float], a_data: int) -> bool:
        """Remove an item by its bounding box and data; false when not found."""

        rect = self._make_rect(a_min, a_max)
        reinsert_list = []

        if not self._remove_rect_internal(rect, a_data, reinsert_list):
            return False

        for node in reinsert_list:
            for i in range(node.m_count):
                self._insert_branch_internal(node.m_branch[i], node.m_level)

        while not self._m_root.is_leaf() and self._m_root.m_count == 1:
            self._m_root = self._m_root.m_branch[0].m_child

        self._m_size -= 1

        return True

    def remove_all(self) -> None:
        """Remove every item."""

        self._m_root = self._alloc_node()
        self._m_size = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # Queries
    # ═══════════════════════════════════════════════════════════════════════════
    def search(self, a_min: list[float], a_max: list[float], a_callback: Callable[[int], bool]) -> int:
        """Visit every item overlapping the box until the callback returns false; returns the visit count."""

        rect = self._make_rect(a_min, a_max)
        stack: list[_Visit] = [_Visit(self._m_root, 0)]
        count = 0

        while len(stack) > 0:
            visit = stack[-1]

            if visit.index == visit.node.m_count:
                stack.pop()
                continue

            branch = visit.node.m_branch[visit.index]
            visit.index += 1

            if not self._overlaps(rect, branch.m_rect):
                continue

            if not visit.node.is_leaf():
                assert len(stack) < STACK_SIZE
                stack.append(_Visit(branch.m_child, 0))
                continue

            count += 1

            if not a_callback(branch.m_data):
                return count

        return count

    # ═══════════════════════════════════════════════════════════════════════════
    # Node allocation
    # ═══════════════════════════════════════════════════════════════════════════
    def _alloc_node(self) -> _Node:
        """Allocate an empty leaf node."""

        node = _Node()
        node.m_count = 0
        node.m_level = 0

        return node

    # ═══════════════════════════════════════════════════════════════════════════
    # Rect math
    # ═══════════════════════════════════════════════════════════════════════════
    def _make_rect(self, a_min: list[float], a_max: list[float]) -> _Rect:
        """Build a rect from min and max corners."""

        rect = _Rect()

        for i in range(3):
            rect.m_min[i] = min(a_min[i], a_max[i])
            rect.m_max[i] = max(a_min[i], a_max[i])

        return rect

    def _calc_rect_volume(self, rect: _Rect) -> float:
        """Volume of a rect."""

        volume = 1.0

        for i in range(3):
            volume *= rect.m_max[i] - rect.m_min[i]

        return volume

    def _combine_rect(self, a: _Rect, b: _Rect) -> _Rect:
        """Smallest rect covering both."""

        rect = _Rect()

        for i in range(3):
            rect.m_min[i] = min(a.m_min[i], b.m_min[i])
            rect.m_max[i] = max(a.m_max[i], b.m_max[i])

        return rect

    def _overlaps(self, a: _Rect, b: _Rect) -> bool:
        """Whether two rects overlap."""

        for i in range(3):
            if a.m_max[i] < b.m_min[i] or b.m_max[i] < a.m_min[i]:
                return False

        return True

    def _node_cover(self, node: _Node) -> _Rect:
        """Rect covering every branch of a node."""

        rect = node.m_branch[0].m_rect.copy()

        for i in range(1, node.m_count):
            rect = self._combine_rect(rect, node.m_branch[i].m_rect)

        return rect

    # ═══════════════════════════════════════════════════════════════════════════
    # Branches
    # ═══════════════════════════════════════════════════════════════════════════
    def _add_branch(self, branch: _Branch, node: _Node) -> _Node | None:
        """Add a branch, splitting the node when full; returns the new sibling or none."""

        if node.m_count == MAXNODES:
            return self._split_node(node, branch)

        node.m_branch[node.m_count] = branch
        node.m_count += 1

        return None

    def _disconnect_branch(self, node: _Node, index: int) -> None:
        """Remove a branch by swapping in the last one."""

        assert 0 <= index < node.m_count

        node.m_branch[index] = node.m_branch[node.m_count - 1]
        node.m_count -= 1

    def _pick_branch(self, rect: _Rect, node: _Node) -> int:
        """Branch whose rect grows least when covering the rect."""

        best_incr = -1.0
        best_area = -1.0
        best = 0

        for i in range(node.m_count):
            cur = node.m_branch[i].m_rect
            area = self._calc_rect_volume(cur)
            combined = self._combine_rect(rect, cur)
            incr = self._calc_rect_volume(combined) - area

            if i == 0 or incr < best_incr or (incr == best_incr and area < best_area):
                best = i
                best_incr = incr
                best_area = area

        return best

    # ═══════════════════════════════════════════════════════════════════════════
    # Quadratic split
    # ═══════════════════════════════════════════════════════════════════════════
    def _get_branches(self, node: _Node, branch: _Branch, part_vars: _PartitionVars) -> None:
        """Collect the node's branches plus one extra into the partition buffer."""

        assert node.m_count == MAXNODES

        for i in range(MAXNODES):
            part_vars.m_branch_buf[i] = node.m_branch[i].copy()

        part_vars.m_branch_buf[MAXNODES] = branch.copy()
        part_vars.m_branch_count = MAXNODES + 1
        part_vars.m_cover_split = part_vars.m_branch_buf[0].m_rect.copy()

        for i in range(1, MAXNODES + 1):
            part_vars.m_cover_split = self._combine_rect(part_vars.m_cover_split, part_vars.m_branch_buf[i].m_rect)

        part_vars.m_cover_split_area = self._calc_rect_volume(part_vars.m_cover_split)
        node.m_count = 0

    def _init_part_vars(self, part_vars: _PartitionVars, max_rects: int, min_fill: int) -> None:
        """Reset the partition buffer."""

        part_vars.m_count[0] = 0
        part_vars.m_count[1] = 0
        part_vars.m_area[0] = 0.0
        part_vars.m_area[1] = 0.0
        part_vars.m_total = max_rects
        part_vars.m_min_fill = min_fill

        for i in range(max_rects):
            part_vars.m_partition[i] = NOT_TAKEN

    def _classify_branch(self, index: int, group: int, part_vars: _PartitionVars) -> None:
        """Assign a branch to a group and grow the group cover."""

        assert part_vars.m_partition[index] == NOT_TAKEN
        part_vars.m_partition[index] = group

        if part_vars.m_count[group] == 0:
            part_vars.m_cover[group] = part_vars.m_branch_buf[index].m_rect.copy()
        else:
            part_vars.m_cover[group] = self._combine_rect(part_vars.m_branch_buf[index].m_rect, part_vars.m_cover[group])

        part_vars.m_area[group] = self._calc_rect_volume(part_vars.m_cover[group])
        part_vars.m_count[group] += 1

    def _pick_seeds(self, part_vars: _PartitionVars) -> None:
        """Seed the two groups with the most wasteful pair."""

        seed0 = 0
        seed1 = 1
        worst = -part_vars.m_cover_split_area - 1.0
        area = [0.0] * (MAXNODES + 1)

        for i in range(part_vars.m_total):
            area[i] = self._calc_rect_volume(part_vars.m_branch_buf[i].m_rect)

        for i in range(part_vars.m_total - 1):
            for j in range(i + 1, part_vars.m_total):
                combined = self._combine_rect(part_vars.m_branch_buf[i].m_rect, part_vars.m_branch_buf[j].m_rect)
                waste = self._calc_rect_volume(combined) - area[i] - area[j]

                if waste > worst:
                    worst = waste
                    seed0 = i
                    seed1 = j

        self._classify_branch(seed0, 0, part_vars)
        self._classify_branch(seed1, 1, part_vars)

    def _choose_partition(self, part_vars: _PartitionVars, min_fill: int) -> None:
        """Quadratic split of the partition buffer into two groups."""

        self._init_part_vars(part_vars, part_vars.m_branch_count, min_fill)
        self._pick_seeds(part_vars)

        while (part_vars.m_count[0] + part_vars.m_count[1]) < part_vars.m_total and \
              part_vars.m_count[0] < (part_vars.m_total - part_vars.m_min_fill) and \
              part_vars.m_count[1] < (part_vars.m_total - part_vars.m_min_fill):

            biggest_diff = -1.0
            chosen = 0
            better_group = 0

            for i in range(part_vars.m_total):
                if part_vars.m_partition[i] != NOT_TAKEN:
                    continue

                r0 = self._combine_rect(part_vars.m_branch_buf[i].m_rect, part_vars.m_cover[0])
                r1 = self._combine_rect(part_vars.m_branch_buf[i].m_rect, part_vars.m_cover[1])
                growth0 = self._calc_rect_volume(r0) - part_vars.m_area[0]
                growth1 = self._calc_rect_volume(r1) - part_vars.m_area[1]
                diff = growth1 - growth0
                group = 0

                if diff < 0:
                    group = 1
                    diff = -diff

                if diff > biggest_diff:
                    biggest_diff = diff
                    chosen = i
                    better_group = group
                elif diff == biggest_diff and part_vars.m_count[group] < part_vars.m_count[better_group]:
                    chosen = i
                    better_group = group

            self._classify_branch(chosen, better_group, part_vars)

        if (part_vars.m_count[0] + part_vars.m_count[1]) < part_vars.m_total:
            group = 1 if part_vars.m_count[0] >= part_vars.m_total - part_vars.m_min_fill else 0

            for i in range(part_vars.m_total):
                if part_vars.m_partition[i] == NOT_TAKEN:
                    self._classify_branch(i, group, part_vars)

    def _load_nodes(self, node_a: _Node, node_b: _Node, part_vars: _PartitionVars) -> None:
        """Move partitioned branches into the two nodes."""

        for i in range(part_vars.m_total):
            target = node_a if part_vars.m_partition[i] == 0 else node_b

            self._add_branch(part_vars.m_branch_buf[i], target)

    def _split_node(self, node: _Node, branch: _Branch) -> _Node:
        """Split a full node with the extra branch; returns the new sibling."""

        part_vars = _PartitionVars()
        self._get_branches(node, branch, part_vars)
        self._choose_partition(part_vars, MINNODES)

        new_node = self._alloc_node()
        new_node.m_level = node.m_level
        self._load_nodes(node, new_node, part_vars)

        return new_node

    # ═══════════════════════════════════════════════════════════════════════════
    # Insertion
    # ═══════════════════════════════════════════════════════════════════════════
    def _insert_rect_internal(self, branch: _Branch, level: int) -> _Node | None:
        """Insert a branch at a level; returns the root's new sibling or none."""

        stack: list[_Visit] = []
        node = self._m_root

        while node.m_level != level:
            assert node.m_level > level
            assert len(stack) < STACK_SIZE

            idx = self._pick_branch(branch.m_rect, node)
            stack.append(_Visit(node, idx))
            node = node.m_branch[idx].m_child

        other = self._add_branch(branch, node)

        for d in range(len(stack) - 1, -1, -1):
            parent = stack[d].node
            idx = stack[d].index

            if other is None:
                parent.m_branch[idx].m_rect = self._combine_rect(parent.m_branch[idx].m_rect, branch.m_rect)
                continue

            parent.m_branch[idx].m_rect = self._node_cover(parent.m_branch[idx].m_child)

            new_b = _Branch()
            new_b.m_rect = self._node_cover(other)
            new_b.m_child = other

            other = self._add_branch(new_b, parent)

        return other

    def _insert_branch_internal(self, branch: _Branch, level: int) -> None:
        """Insert a branch and grow the root when it splits."""

        new_node = self._insert_rect_internal(branch, level)

        if new_node is None:
            return

        old_root = self._m_root
        self._m_root = self._alloc_node()
        self._m_root.m_level = old_root.m_level + 1

        b1 = _Branch()
        b1.m_rect = self._node_cover(old_root)
        b1.m_child = old_root

        b2 = _Branch()
        b2.m_rect = self._node_cover(new_node)
        b2.m_child = new_node

        self._add_branch(b1, self._m_root)
        self._add_branch(b2, self._m_root)

    # ═══════════════════════════════════════════════════════════════════════════
    # Removal
    # ═══════════════════════════════════════════════════════════════════════════
    def _remove_rect_internal(self, rect: _Rect, data: int, reinsert_list: list[_Node]) -> bool:
        """Remove the matching leaf branch; underfull nodes go to the reinsert list."""

        stack: list[_Visit] = [_Visit(self._m_root, 0)]

        while len(stack) > 0:
            visit = stack[-1]

            if visit.index == visit.node.m_count:
                stack.pop()
                continue

            branch = visit.node.m_branch[visit.index]
            visit.index += 1

            if not self._overlaps(rect, branch.m_rect):
                continue

            if not visit.node.is_leaf():
                assert len(stack) < STACK_SIZE
                stack.append(_Visit(branch.m_child, 0))
                continue

            if branch.m_data != data:
                continue

            self._disconnect_branch(visit.node, visit.index - 1)

            for d in range(len(stack) - 2, -1, -1):
                self._shrink_branch(stack[d].node, stack[d].index - 1, reinsert_list)

            return True

        return False

    def _shrink_branch(self, node: _Node, index: int, reinsert_list: list[_Node]) -> None:
        """Recompute a child's cover or queue it for reinsertion when underfull."""

        child = node.m_branch[index].m_child

        if child.m_count >= MINNODES:
            node.m_branch[index].m_rect = self._node_cover(child)

            return

        reinsert_list.append(child)
        self._disconnect_branch(node, index)
