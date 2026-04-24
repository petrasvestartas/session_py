# SpatialRTree — R-tree with dynamic insert/delete (Guttman split, fan-out 4–8).
# Use for: "find all objects overlapping this region" (spatial range queries).
#   Supports live insertion and deletion; good for mutable object sets.
# Prefer over SpatialAABBTree/SpatialBVH when data changes frequently.
# Prefer over SpatialKDTree  when querying volumes/boxes, not bare point clouds.
# Note: k-NN is possible but slower than SpatialKDTree for pure point queries.
MAXNODES = 8
MINNODES = 4
NOT_TAKEN = -1


class _Rect:
    def __init__(self):
        self.m_min = [0.0, 0.0, 0.0]
        self.m_max = [0.0, 0.0, 0.0]

    def copy(self):
        r = _Rect()
        r.m_min = list(self.m_min)
        r.m_max = list(self.m_max)
        return r


class _Branch:
    def __init__(self):
        self.m_rect = _Rect()
        self.m_child = None
        self.m_data = 0

    def copy(self):
        b = _Branch()
        b.m_rect = self.m_rect.copy()
        b.m_child = self.m_child
        b.m_data = self.m_data
        return b


class _Node:
    def __init__(self):
        self.m_count = 0
        self.m_level = 0
        self.m_branch = [None] * (MAXNODES + 1)
        for i in range(MAXNODES + 1):
            self.m_branch[i] = _Branch()

    def is_leaf(self):
        return self.m_level == 0


class _PartitionVars:
    def __init__(self):
        self.m_partition = [NOT_TAKEN] * (MAXNODES + 1)
        self.m_total = 0
        self.m_min_fill = 0
        self.m_count = [0, 0]
        self.m_cover = [_Rect(), _Rect()]
        self.m_area = [0.0, 0.0]
        self.m_branch_buf = [None] * (MAXNODES + 1)
        for i in range(MAXNODES + 1):
            self.m_branch_buf[i] = _Branch()
        self.m_branch_count = 0
        self.m_cover_split = _Rect()
        self.m_cover_split_area = 0.0


class SpatialRTree:
    def __init__(self):
        self._m_root = self._alloc_node()
        self._m_root.m_level = 0
        self._m_size = 0

    def _alloc_node(self):
        node = _Node()
        node.m_count = 0
        node.m_level = 0
        return node

    def _make_rect(self, a_min, a_max):
        rect = _Rect()
        for i in range(3):
            rect.m_min[i] = a_min[i]
            rect.m_max[i] = a_max[i]
        return rect

    def _calc_rect_volume(self, rect):
        volume = 1.0
        for i in range(3):
            volume *= rect.m_max[i] - rect.m_min[i]
        return volume

    def _combine_rect(self, a, b):
        rect = _Rect()
        for i in range(3):
            rect.m_min[i] = min(a.m_min[i], b.m_min[i])
            rect.m_max[i] = max(a.m_max[i], b.m_max[i])
        return rect

    def _overlaps(self, a, b):
        for i in range(3):
            if a.m_max[i] < b.m_min[i] or b.m_max[i] < a.m_min[i]:
                return False
        return True

    def _node_cover(self, node):
        rect = node.m_branch[0].m_rect.copy()
        for i in range(1, node.m_count):
            rect = self._combine_rect(rect, node.m_branch[i].m_rect)
        return rect

    def _add_branch(self, branch, node, new_node_ref):
        if node.m_count < MAXNODES:
            node.m_branch[node.m_count] = branch
            node.m_count += 1
            return False
        else:
            assert new_node_ref is not None
            new_node_ref[0] = self._split_node(node, branch)
            return True

    def _disconnect_branch(self, node, index):
        assert 0 <= index < node.m_count
        node.m_branch[index] = node.m_branch[node.m_count - 1]
        node.m_count -= 1

    def _pick_branch(self, rect, node):
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

    def _get_branches(self, node, branch, part_vars):
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

    def _init_part_vars(self, part_vars, max_rects, min_fill):
        part_vars.m_count[0] = part_vars.m_count[1] = 0
        part_vars.m_area[0] = part_vars.m_area[1] = 0.0
        part_vars.m_total = max_rects
        part_vars.m_min_fill = min_fill
        for i in range(max_rects):
            part_vars.m_partition[i] = NOT_TAKEN

    def _classify_branch(self, index, group, part_vars):
        assert part_vars.m_partition[index] == NOT_TAKEN
        part_vars.m_partition[index] = group
        if part_vars.m_count[group] == 0:
            part_vars.m_cover[group] = part_vars.m_branch_buf[index].m_rect.copy()
        else:
            part_vars.m_cover[group] = self._combine_rect(part_vars.m_branch_buf[index].m_rect, part_vars.m_cover[group])
        part_vars.m_area[group] = self._calc_rect_volume(part_vars.m_cover[group])
        part_vars.m_count[group] += 1

    def _pick_seeds(self, part_vars):
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

    def _choose_partition(self, part_vars, min_fill):
        self._init_part_vars(part_vars, part_vars.m_branch_count, min_fill)
        self._pick_seeds(part_vars)
        while (part_vars.m_count[0] + part_vars.m_count[1]) < part_vars.m_total and \
              part_vars.m_count[0] < (part_vars.m_total - part_vars.m_min_fill) and \
              part_vars.m_count[1] < (part_vars.m_total - part_vars.m_min_fill):
            biggest_diff = -1.0
            chosen = 0
            better_group = 0
            for i in range(part_vars.m_total):
                if part_vars.m_partition[i] == NOT_TAKEN:
                    r0 = self._combine_rect(part_vars.m_branch_buf[i].m_rect, part_vars.m_cover[0])
                    r1 = self._combine_rect(part_vars.m_branch_buf[i].m_rect, part_vars.m_cover[1])
                    growth0 = self._calc_rect_volume(r0) - part_vars.m_area[0]
                    growth1 = self._calc_rect_volume(r1) - part_vars.m_area[1]
                    diff = growth1 - growth0
                    if diff >= 0:
                        group = 0
                    else:
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

    def _load_nodes(self, node_a, node_b, part_vars):
        for i in range(part_vars.m_total):
            g = part_vars.m_partition[i]
            target = node_a if g == 0 else node_b
            self._add_branch(part_vars.m_branch_buf[i], target, None)

    def _split_node(self, node, branch):
        part_vars = _PartitionVars()
        self._get_branches(node, branch, part_vars)
        self._choose_partition(part_vars, MINNODES)
        new_node = self._alloc_node()
        new_node.m_level = node.m_level
        self._load_nodes(node, new_node, part_vars)
        return new_node

    def _insert_rect_rec(self, branch, node, level):
        if node.m_level > level:
            idx = self._pick_branch(branch.m_rect, node)
            other = self._insert_rect_rec(branch, node.m_branch[idx].m_child, level)
            if other is None:
                node.m_branch[idx].m_rect = self._combine_rect(node.m_branch[idx].m_rect, branch.m_rect)
                return None
            else:
                node.m_branch[idx].m_rect = self._node_cover(node.m_branch[idx].m_child)
                new_b = _Branch()
                new_b.m_rect = self._node_cover(other)
                new_b.m_child = other
                nn = [None]
                self._add_branch(new_b, node, nn)
                return nn[0]
        elif node.m_level == level:
            nn = [None]
            self._add_branch(branch, node, nn)
            return nn[0]
        return None

    def _insert_branch_internal(self, branch, level):
        new_node = self._insert_rect_rec(branch, self._m_root, level)
        if new_node is not None:
            old_root = self._m_root
            self._m_root = self._alloc_node()
            self._m_root.m_level = old_root.m_level + 1
            b1 = _Branch()
            b1.m_rect = self._node_cover(old_root)
            b1.m_child = old_root
            b2 = _Branch()
            b2.m_rect = self._node_cover(new_node)
            b2.m_child = new_node
            self._add_branch(b1, self._m_root, None)
            self._add_branch(b2, self._m_root, None)

    def _search_internal(self, rect, node, count_ref, callback):
        if node.is_leaf():
            for i in range(node.m_count):
                if self._overlaps(rect, node.m_branch[i].m_rect):
                    count_ref[0] += 1
                    if not callback(node.m_branch[i].m_data):
                        return False
        else:
            for i in range(node.m_count):
                if self._overlaps(rect, node.m_branch[i].m_rect):
                    if not self._search_internal(rect, node.m_branch[i].m_child, count_ref, callback):
                        return False
        return True

    def _remove_rect_internal(self, rect, data, node, reinsert_list):
        if node.is_leaf():
            for i in range(node.m_count):
                if node.m_branch[i].m_data == data and self._overlaps(rect, node.m_branch[i].m_rect):
                    self._disconnect_branch(node, i)
                    return True
            return False
        else:
            for i in range(node.m_count):
                if self._overlaps(rect, node.m_branch[i].m_rect):
                    if self._remove_rect_internal(rect, data, node.m_branch[i].m_child, reinsert_list):
                        if node.m_branch[i].m_child.m_count >= MINNODES:
                            node.m_branch[i].m_rect = self._node_cover(node.m_branch[i].m_child)
                        else:
                            reinsert_list.append(node.m_branch[i].m_child)
                            self._disconnect_branch(node, i)
                        return True
            return False

    def insert(self, a_min, a_max, a_data):
        branch = _Branch()
        branch.m_rect = self._make_rect(a_min, a_max)
        branch.m_child = None
        branch.m_data = a_data
        self._insert_branch_internal(branch, 0)
        self._m_size += 1

    def remove(self, a_min, a_max, a_data):
        rect = self._make_rect(a_min, a_max)
        reinsert_list = []
        if not self._remove_rect_internal(rect, a_data, self._m_root, reinsert_list):
            return False
        for node in reinsert_list:
            for i in range(node.m_count):
                self._insert_branch_internal(node.m_branch[i], node.m_level)
        while not self._m_root.is_leaf() and self._m_root.m_count == 1:
            self._m_root = self._m_root.m_branch[0].m_child
        self._m_size -= 1
        return True

    def search(self, a_min, a_max, a_callback):
        rect = self._make_rect(a_min, a_max)
        count_ref = [0]
        self._search_internal(rect, self._m_root, count_ref, a_callback)
        return count_ref[0]

    def remove_all(self):
        self._m_root = self._alloc_node()
        self._m_root.m_level = 0
        self._m_size = 0

    def count(self):
        return self._m_size
