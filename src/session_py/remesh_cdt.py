from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .point import Point
from .polyline import Polyline
from .session_config import SESSION_CONFIG
from .vector import Vector

if TYPE_CHECKING:
    from .mesh import Mesh

# ═══════════════════════════════════════════════════════════════════════════
# Integer geometry
# ═══════════════════════════════════════════════════════════════════════════

NULL_IDX = None
MAX_COORD64 = 9e17
MAX_PRECISION = 6


def _to_int64(x: float) -> int:
    """Round to the nearest int64."""

    magnitude = abs(x)
    whole = math.floor(magnitude)

    if magnitude - whole >= 0.5:
        whole += 1

    return whole if x >= 0.0 else -whole


def _to_point64(p, scale: float) -> tuple[int, int]:
    """Scale a 2D point to integer coordinates."""
    return (_to_int64(p[0] * scale), _to_int64(p[1] * scale))


def _div3(s: int) -> int:
    """Integer division truncating towards zero like C++."""

    if s >= 0:
        return s // 3

    return -((-s) // 3)


def _cross_sign(p1, p2, p3) -> int:
    """Sign of the turn p1 -> p2 -> p3."""

    cp = float(p2[0] - p1[0]) * float(p3[1] - p2[1]) - float(p2[1] - p1[1]) * float(
        p3[0] - p2[0]
    )

    if cp > 0:
        return 1

    if cp < 0:
        return -1

    return 0


def _left_turning(p1, p2, p3) -> bool:
    """True when p1 -> p2 -> p3 turns left."""
    return _cross_sign(p1, p2, p3) < 0


def _right_turning(p1, p2, p3) -> bool:
    """True when p1 -> p2 -> p3 turns right."""
    return _cross_sign(p1, p2, p3) > 0


def _sweep_before(a, b) -> bool:
    """True when a is swept before b: higher y first, then lower x."""

    if a[1] == b[1]:
        return a[0] < b[0]

    return a[1] > b[1]


def _dist_sqr(a, b) -> float:
    """Squared distance between two integer points."""

    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])

    return dx * dx + dy * dy


def _in_circle(a, b, c, d) -> float:
    """Positive when d lies inside the circumcircle of the counter-clockwise triangle a, b, c."""

    m00 = float(a[0] - d[0])
    m01 = float(a[1] - d[1])
    m02 = m00 * m00 + m01 * m01
    m10 = float(b[0] - d[0])
    m11 = float(b[1] - d[1])
    m12 = m10 * m10 + m11 * m11
    m20 = float(c[0] - d[0])
    m21 = float(c[1] - d[1])
    m22 = m20 * m20 + m21 * m21

    return (
        m00 * (m11 * m22 - m21 * m12)
        - m10 * (m01 * m22 - m21 * m02)
        + m20 * (m01 * m12 - m11 * m02)
    )


def _dist_sqr_segment(p, a, b) -> float:
    """Squared distance from p to the segment a-b."""

    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    ax = float(p[0] - a[0])
    ay = float(p[1] - a[1])
    q = ax * dx + ay * dy

    if q < 0:
        return _dist_sqr(p, a)

    if q > dx * dx + dy * dy:
        return _dist_sqr(p, b)

    return (ax * dy - dx * ay) * (ax * dy - dx * ay) / (dx * dx + dy * dy)


def _segments_intersect(a1, a2, b1, b2) -> bool:
    """True when a1-a2 and b1-b2 cross strictly inside both segments."""

    if a1 == b1 or a2 == b1 or a2 == b2 or a1 == b2:
        return False

    dy1 = float(a2[1] - a1[1])
    dx1 = float(a2[0] - a1[0])
    dy2 = float(b2[1] - b1[1])
    dx2 = float(b2[0] - b1[0])
    cp = dy1 * dx2 - dy2 * dx1

    if cp == 0:
        return False

    t = float(a1[0] - b1[0]) * dy2 - float(a1[1] - b1[1]) * dx2

    if t >= 0 and (cp < 0 or t >= cp):
        return False

    if t < 0 and (cp > 0 or t <= cp):
        return False

    u = float(a1[0] - b1[0]) * dy1 - float(a1[1] - b1[1]) * dx1

    if u >= 0:
        return cp > 0 and u < cp

    return cp < 0 and u > cp


def _inside_path64(p, poly) -> bool:
    """Even-odd test of an integer point against an integer ring."""

    inside = False
    n = len(poly)
    j = n - 1

    for i in range(n):
        if (poly[i][1] > p[1]) != (poly[j][1] > p[1]):
            x = float(poly[i][0]) + float(p[1] - poly[i][1]) * float(
                poly[j][0] - poly[i][0]
            ) / float(poly[j][1] - poly[i][1])

            if float(p[0]) < x:
                inside = not inside

        j = i

    return inside


def _prev_index(i: int, n: int) -> int:
    """Index before i on a ring of n."""
    return n - 1 if i == 0 else i - 1


def _next_index(i: int, n: int) -> int:
    """Index after i on a ring of n."""
    return (i + 1) % n


def _find_loc_min(path, i: int) -> tuple[bool, int]:
    """Advance i to the next vertex that ends a rising run and starts a falling one; false when the path is flat."""

    n = len(path)

    if n < 3:
        return False, i

    i0 = i
    k = _next_index(i, n)

    while path[k][1] <= path[i][1]:
        i = k
        k = _next_index(k, n)

        if i == i0:
            return False, i

    while path[k][1] >= path[i][1]:
        i = k
        k = _next_index(k, n)

    return True, i


# ═══════════════════════════════════════════════════════════════════════════
# Sweep graph
# ═══════════════════════════════════════════════════════════════════════════

LOOSE = 0  # Diagonal between two boundary edges.
ASCEND = 1  # Boundary edge on the left side.
DESCEND = 2  # Boundary edge on the right side.


class _Vertex:
    """Sweep vertex with its incident edges."""

    __slots__ = ("edges", "inner_lm", "pt")

    def __init__(self, pt: tuple[int, int]):
        """Construct at an integer point."""
        self.pt = pt  # Integer position.
        self.edges = []  # Edges touching the vertex.
        self.inner_lm = False  # True at a local minimum of a hole.


class _Edge:
    """Sweep edge with its endpoints, triangles and active-list links."""

    __slots__ = (
        "active",
        "kind",
        "next",
        "prev",
        "tri_a",
        "tri_b",
        "vb",
        "vl",
        "vr",
        "vt",
    )

    def __init__(self):
        """Construct an unlinked edge."""
        self.vl = NULL_IDX  # Left vertex.
        self.vr = NULL_IDX  # Right vertex.
        self.vb = NULL_IDX  # Bottom vertex.
        self.vt = NULL_IDX  # Top vertex.
        self.kind = LOOSE  # Boundary side or loose diagonal.
        self.tri_a = NULL_IDX  # First triangle.
        self.tri_b = NULL_IDX  # Second triangle.
        self.active = False  # True while on the active list.
        self.next = NULL_IDX  # Next active edge.
        self.prev = NULL_IDX  # Previous active edge.


class _Tri:
    """Triangle on three edges."""

    __slots__ = ("edges",)

    def __init__(self, e1: int, e2: int, e3: int):
        """Construct on three edge indices."""
        self.edges = [e1, e2, e3]  # Edge indices.


class _Delaunay:
    """Sweep-line constrained Delaunay: boundary edges ascend on the left and descend on the right, diagonals are loose."""

    def __init__(self):
        """Construct an empty sweep graph."""
        self.vs = []  # Vertices.
        self.es = []  # Edges.
        self.ts = []  # Triangles.
        self.pending = []  # Loose edges waiting to be legalized.
        self.horz = []  # Horizontal edges deferred from the current row.
        self.loc_mins = []  # Hole local minima on the current row.
        self.lowermost = NULL_IDX  # Lowest vertex of the outer path.
        self.first_active = NULL_IDX  # Head of the active edge list.

    def _is_horizontal(self, e: int) -> bool:
        """True when both ends of e share a row."""
        return self.vs[self.es[e].vb].pt[1] == self.vs[self.es[e].vt].pt[1]

    def _completed(self, e: int) -> bool:
        """An edge is done with two triangles, or with one when it is a boundary edge."""

        if self.es[e].tri_a is NULL_IDX:
            return False

        if self.es[e].tri_b is not NULL_IDX:
            return True

        return self.es[e].kind != LOOSE

    def _other(self, e: int, v: int) -> int:
        """The endpoint of e that is not v."""
        return self.es[e].vt if self.es[e].vb == v else self.es[e].vb

    def _add_vertex(self, p: tuple[int, int]) -> int:
        """Append a vertex and return its index."""

        self.vs.append(_Vertex(p))

        return len(self.vs) - 1

    def _add_active(self, e: int) -> None:
        """Prepend e to the doubly-linked active list."""

        if self.es[e].active:
            return

        self.es[e].prev = NULL_IDX
        self.es[e].next = self.first_active
        self.es[e].active = True

        if self.first_active is not NULL_IDX:
            self.es[self.first_active].prev = e

        self.first_active = e

    def _remove_active(self, e: int) -> None:
        """Unlink e from the active list and from both endpoint edge lists."""

        self._remove_from_vertex(self.es[e].vb, e)
        self._remove_from_vertex(self.es[e].vt, e)

        prev = self.es[e].prev
        next = self.es[e].next

        if next is not NULL_IDX:
            self.es[next].prev = prev

        if prev is not NULL_IDX:
            self.es[prev].next = next

        self.es[e].active = False

        if self.first_active == e:
            self.first_active = next

    def _remove_from_vertex(self, v: int, e: int) -> None:
        """Drop e from the edge list of v."""

        edges = self.vs[v].edges

        if e in edges:
            edges.remove(e)

    def _create_edge(self, v1: int, v2: int, kind: int) -> int:
        """New edge between v1 and v2; loose edges go straight to the active list and the legalize queue."""

        e = len(self.es)
        self.es.append(_Edge())

        p1 = self.vs[v1].pt
        p2 = self.vs[v2].pt

        self.es[e].vb = v2 if p1[1] < p2[1] else v1
        self.es[e].vt = v1 if p1[1] < p2[1] else v2
        self.es[e].vl = v1 if p1[0] <= p2[0] else v2
        self.es[e].vr = v2 if p1[0] <= p2[0] else v1
        self.es[e].kind = kind

        self.vs[v1].edges.append(e)
        self.vs[v2].edges.append(e)

        if kind == LOOSE:
            self.pending.append(e)
            self._add_active(e)

        return e

    def _create_tri(self, e1: int, e2: int, e3: int) -> int:
        """New triangle on three edges; an edge leaves the active list when it is completed."""

        t = len(self.ts)
        self.ts.append(_Tri(e1, e2, e3))

        for e in (e1, e2, e3):
            if self.es[e].tri_a is not NULL_IDX:
                self.es[e].tri_b = t
                self._remove_active(e)
            else:
                self.es[e].tri_a = t

                if self.es[e].kind != LOOSE:
                    self._remove_active(e)

        return t

    def _split_edge(self, long_e: int, short_e: int) -> None:
        """Shorten long_e to end at short_e's top and continue it with a new edge to the old top."""

        old_t = self.es[long_e].vt
        new_t = self.es[short_e].vt

        self._remove_from_vertex(old_t, long_e)
        self.es[long_e].vt = new_t

        if self.es[long_e].vl == old_t:
            self.es[long_e].vl = new_t
        else:
            self.es[long_e].vr = new_t

        self.vs[new_t].edges.append(long_e)
        self._create_edge(new_t, old_t, self.es[long_e].kind)

    def _split_collinear(self, v: int) -> None:
        """Split the longer of two collinear non-horizontal edges leaving v downwards."""

        snapshot = list(self.vs[v].edges)

        for e1 in snapshot:
            if self._is_horizontal(e1) or self.es[e1].vb != v:
                continue

            for e2 in snapshot:
                if e2 == e1 or self.es[e2].vb != v:
                    continue

                t1 = self.vs[self.es[e1].vt].pt
                t2 = self.vs[self.es[e2].vt].pt

                if t1[1] == t2[1] or _cross_sign(t1, self.vs[v].pt, t2) != 0:
                    continue

                if t1[1] < t2[1]:
                    self._split_edge(e1, e2)
                else:
                    self._split_edge(e2, e1)

                break

    def _merge_duplicates(self, order: list[int]) -> None:
        """Merge coincident vertices that are neighbours in sweep order into the first one."""

        v1 = order[0]

        for v2 in order[1:]:
            if self.vs[v1].pt != self.vs[v2].pt:
                v1 = v2
                continue

            if not self.vs[v1].inner_lm or not self.vs[v2].inner_lm:
                self.vs[v1].inner_lm = False

            for e in self.vs[v2].edges:
                if self.es[e].vb == v2:
                    self.es[e].vb = v1
                else:
                    self.es[e].vt = v1

                if self.es[e].vl == v2:
                    self.es[e].vl = v1
                else:
                    self.es[e].vr = v1

            self.vs[v1].edges.extend(self.vs[v2].edges)
            self.vs[v2].edges = []
            self._split_collinear(v1)

    def _find_linking_edge(self, v1: int, v2: int, prefer_ascend: bool) -> int:
        """Edge of v1 that reaches v2, a loose one or one of the preferred kind first."""

        res = NULL_IDX

        for e in self.vs[v1].edges:
            if self.es[e].vl != v2 and self.es[e].vr != v2:
                continue

            if self.es[e].kind == LOOSE or (self.es[e].kind == ASCEND) == prefer_ascend:
                return e

            res = e

        return res

    def _horizontal_between(self, v1: int, v2: int) -> bool:
        """True when an active horizontal edge lies on the row of v1 between v1 and v2."""

        y = self.vs[v1].pt[1]
        lo = min(self.vs[v1].pt[0], self.vs[v2].pt[0])
        hi = max(self.vs[v1].pt[0], self.vs[v2].pt[0])
        e = self.first_active

        while e is not NULL_IDX:
            pl = self.vs[self.es[e].vl].pt
            pr = self.vs[self.es[e].vr].pt

            if (
                pl[1] == y
                and pr[1] == y
                and pl[0] >= lo
                and pr[0] <= hi
                and (pl[0] != lo or pl[0] != hi)
            ):
                return True

            e = self.es[e].next

        return False

    def _edge_below(self, v_above: int) -> int:
        """Nearest active edge spanning the x of v_above below it, NULL_IDX when there is none."""

        pa = self.vs[v_above].pt
        best = NULL_IDX
        best_d = -1.0
        e = self.first_active

        while e is not NULL_IDX:
            pl = self.vs[self.es[e].vl].pt
            pr = self.vs[self.es[e].vr].pt
            spans = (
                pl[0] <= pa[0]
                and pr[0] >= pa[0]
                and self.vs[self.es[e].vb].pt[1] >= pa[1]
            )

            if (
                spans
                and self.es[e].vb != v_above
                and self.es[e].vt != v_above
                and not _left_turning(pl, pa, pr)
            ):
                d = _dist_sqr_segment(pa, pl, pr)

                if best is NULL_IDX or d < best_d:
                    best = e
                    best_d = d

            e = self.es[e].next

        return best

    def _visible_vertex(self, e_below: int, v_above: int) -> int:
        """Endpoint of e_below visible from v_above, moved past every active edge crossing the connection."""

        pa = self.vs[v_above].pt
        best = (
            self.es[e_below].vb
            if self.vs[self.es[e_below].vt].pt[1] <= pa[1]
            else self.es[e_below].vt
        )
        left = self.vs[best].pt[0] < pa[0]
        e = self.first_active

        while e is not NULL_IDX:
            pb = self.vs[best].pt
            pl = self.vs[self.es[e].vl].pt
            pr = self.vs[self.es[e].vr].pt
            eb = self.vs[self.es[e].vb].pt
            et = self.vs[self.es[e].vt].pt
            spans = (
                (pr[0] > pb[0] and pl[0] < pa[0])
                if left
                else (pr[0] < pb[0] and pl[0] > pa[0])
            )

            if (
                spans
                and eb[1] > pa[1]
                and et[1] < pb[1]
                and _segments_intersect(eb, et, pb, pa)
            ):
                best = self.es[e].vt if et[1] > pa[1] else self.es[e].vb

            e = self.es[e].next

        return best

    def _create_loc_min_edge(self, v_above: int) -> int:
        """Connect a hole local minimum to the visible vertex of the nearest active edge below it."""

        below = self._edge_below(v_above)

        if below is NULL_IDX:
            return NULL_IDX

        return self._create_edge(self._visible_vertex(below, v_above), v_above, LOOSE)

    def _fan_vertex(self, edge: int, pivot: int, left: bool) -> tuple[int, int]:
        """Tightest active fan candidate around pivot on the left (or right) side of edge, turns read with the side as sign; NULL_IDX when there is none."""

        v = self._other(edge, pivot)
        side = 1 if left else -1
        v_alt = NULL_IDX
        e_alt = NULL_IDX

        for e in self.vs[pivot].edges:
            if e == edge or not self.es[e].active:
                continue

            vx = self._other(e, pivot)

            if vx == v:
                continue

            sign = side * _cross_sign(self.vs[v].pt, self.vs[pivot].pt, self.vs[vx].pt)

            if sign == 0:
                if (self.vs[v].pt[0] > self.vs[pivot].pt[0]) == (
                    self.vs[pivot].pt[0] > self.vs[vx].pt[0]
                ):
                    continue
            elif sign > 0 or (
                v_alt is not NULL_IDX
                and side
                * _cross_sign(self.vs[vx].pt, self.vs[pivot].pt, self.vs[v_alt].pt)
                >= 0
            ):
                continue

            v_alt = vx
            e_alt = e

        return v_alt, e_alt

    def _triangulate_fan(self, edge: int, pivot: int, min_y: int, left: bool) -> None:
        """Fan triangles around pivot on one side of edge, walking onto each new diagonal, never below min_y."""

        max_fan = 2 * len(self.vs) + 2

        for _step in range(max_fan):
            v_alt, e_alt = self._fan_vertex(edge, pivot, left)

            if v_alt is NULL_IDX or self.vs[v_alt].pt[1] < min_y:
                return

            kind_below = ASCEND if left else DESCEND
            kind_above = DESCEND if left else ASCEND

            if (
                self.vs[v_alt].pt[1] < self.vs[pivot].pt[1]
                and self.es[e_alt].kind == kind_below
            ):
                return

            if (
                self.vs[v_alt].pt[1] > self.vs[pivot].pt[1]
                and self.es[e_alt].kind == kind_above
            ):
                return

            v = self._other(edge, pivot)
            prefer_ascend = (
                self.vs[v_alt].pt[1] < self.vs[v].pt[1]
                if left
                else self.vs[v_alt].pt[1] > self.vs[v].pt[1]
            )
            ex = self._find_linking_edge(v_alt, v, prefer_ascend)

            if ex is NULL_IDX:
                if (
                    self.vs[v_alt].pt[1] == self.vs[v].pt[1]
                    and self.vs[v].pt[1] == min_y
                    and self._horizontal_between(v_alt, v)
                ):
                    return

                ex = self._create_edge(v_alt, v, LOOSE)

            if left:
                self._create_tri(edge, e_alt, ex)
            else:
                self._create_tri(edge, ex, e_alt)

            if self._completed(ex):
                return

            edge = ex
            pivot = v_alt

    def _opposite(self, tri: int, edge: int, vl: int) -> tuple[int, int, int]:
        """Of the two edges of tri other than edge, a gets the one touching vl and b the other; returns the far vertex."""

        far = NULL_IDX
        a = NULL_IDX
        b = NULL_IDX

        for e in self.ts[tri].edges:
            if e == edge:
                continue

            if self.es[e].vl == vl:
                a = e
                far = self.es[e].vr
            elif self.es[e].vr == vl:
                a = e
                far = self.es[e].vl
            else:
                b = e

        return far, a, b

    def _rewire(self, tri: int, other: int, edge: int, e1: int, e2: int) -> None:
        """Give tri the edges (edge, e1, e2) and move e1/e2 from the other triangle onto it."""

        self.ts[tri].edges = [edge, e1, e2]

        for e in (e1, e2):
            if self.es[e].kind == LOOSE:
                self.pending.append(e)

            if self.es[e].tri_a == tri or self.es[e].tri_b == tri:
                continue

            if self.es[e].tri_a == other:
                self.es[e].tri_a = tri
            elif self.es[e].tri_b == other:
                self.es[e].tri_b = tri

    def _force_legal(self, edge: int) -> None:
        """Flip edge when the far vertex of one triangle lies inside the circumcircle of the other."""

        ta = self.es[edge].tri_a
        tb = self.es[edge].tri_b

        if ta is NULL_IDX or tb is NULL_IDX:
            return

        vl = self.es[edge].vl
        vr = self.es[edge].vr
        va, a1, b1 = self._opposite(ta, edge, vl)
        vb, a2, b2 = self._opposite(tb, edge, vl)

        if va is NULL_IDX or vb is NULL_IDX or b1 is NULL_IDX or b2 is NULL_IDX:
            return

        if _cross_sign(self.vs[va].pt, self.vs[vl].pt, self.vs[vr].pt) == 0:
            return

        ict = _in_circle(self.vs[va].pt, self.vs[vl].pt, self.vs[vr].pt, self.vs[vb].pt)

        if ict == 0 or _right_turning(
            self.vs[va].pt, self.vs[vl].pt, self.vs[vr].pt
        ) == (ict < 0):
            return

        self.es[edge].vl = va
        self.es[edge].vr = vb
        self._rewire(ta, tb, edge, a1, a2)
        self._rewire(tb, ta, edge, b1, b2)

    def _walk_path(self, path, i0: int, i: int, v0: int) -> bool:
        """Walk the path from i back round to i0 creating boundary edges; false when the step budget of a degenerate path is blown."""

        n = len(path)
        budget = 16 * n + 256
        steps = 0
        v_prev = v0

        while steps < budget:
            steps += 1

            self.loc_mins.append(v_prev)

            if self.lowermost is NULL_IDX or _sweep_before(
                self.vs[v_prev].pt, self.vs[self.lowermost].pt
            ):
                self.lowermost = v_prev

            i_next = _next_index(i, n)

            if _cross_sign(self.vs[v_prev].pt, path[i], path[i_next]) == 0:
                i = i_next
                continue

            while path[i][1] <= self.vs[v_prev].pt[1]:
                steps += 1

                if steps > budget:
                    return False

                v = self._add_vertex(path[i])

                self._create_edge(v_prev, v, ASCEND)
                v_prev = v
                i = i_next
                i_next = _next_index(i, n)

                while _cross_sign(self.vs[v_prev].pt, path[i], path[i_next]) == 0:
                    steps += 1

                    if steps > budget:
                        return False

                    i = i_next
                    i_next = _next_index(i, n)

            v_prev_prev = v_prev

            while i != i0 and path[i][1] >= self.vs[v_prev].pt[1]:
                steps += 1

                if steps > budget:
                    return False

                v = self._add_vertex(path[i])

                self._create_edge(v, v_prev, DESCEND)
                v_prev_prev = v_prev
                v_prev = v
                i = i_next
                i_next = _next_index(i, n)

                while _cross_sign(self.vs[v_prev].pt, path[i], path[i_next]) == 0:
                    steps += 1

                    if steps > budget:
                        return False

                    i = i_next
                    i_next = _next_index(i, n)

            if i == i0:
                self._create_edge(v0, v_prev, DESCEND)

                return True

            if _left_turning(self.vs[v_prev_prev].pt, self.vs[v_prev].pt, path[i]):
                self.vs[v_prev].inner_lm = True

        return False

    def _discard(self, start: int) -> None:
        """Detach the edges of every vertex added since start."""

        for v in range(start, len(self.vs)):
            self.vs[v].edges = []

    def _add_path(self, path) -> None:
        """Register one closed path; paths that are flat, degenerate or too tiny to hold a triangle are dropped."""

        n = len(path)
        ok, i = _find_loc_min(path, 0)

        if not ok:
            return

        i0 = i
        i_prev = _prev_index(i, n)

        while path[i_prev] == path[i]:
            i_prev = _prev_index(i_prev, n)

        i_next = _next_index(i, n)

        while _cross_sign(path[i_prev], path[i], path[i_next]) == 0:
            ok, i = _find_loc_min(path, i)

            if not ok or i == i0:
                return

            i_prev = _prev_index(i, n)

            while path[i_prev] == path[i]:
                i_prev = _prev_index(i_prev, n)

            i_next = _next_index(i, n)

        start = len(self.vs)
        v0 = self._add_vertex(path[i])

        if _left_turning(path[i_prev], path[i], path[i_next]):
            self.vs[v0].inner_lm = True

        if not self._walk_path(path, i0, i_next, v0):
            self._discard(start)

            return

        count = len(self.vs) - start
        tiny = count == 3 and (
            _dist_sqr(self.vs[start].pt, self.vs[start + 1].pt) <= 1
            or _dist_sqr(self.vs[start + 1].pt, self.vs[start + 2].pt) <= 1
            or _dist_sqr(self.vs[start + 2].pt, self.vs[start].pt) <= 1
        )

        if count < 3 or tiny:
            self._discard(start)

    def _add_paths(self, paths) -> bool:
        """Register every path; false when none survives."""

        total = 0

        for path in paths:
            total += len(path)

        if total == 0:
            return False

        for path in paths:
            self._add_path(path)

        return len(self.vs) > 2

    def _flip_winding(self) -> None:
        """The outer path was wound clockwise: swap the hole flags and the boundary sides."""

        for v in self.loc_mins:
            self.vs[v].inner_lm = not self.vs[v].inner_lm

        for e in self.es:
            if e.kind == ASCEND:
                e.kind = DESCEND
            elif e.kind == DESCEND:
                e.kind = ASCEND

    def _sweep_loc_mins(self, curr_y: int) -> bool:
        """Connect and fan the hole local minima collected on the finished row; false when one cannot be reached."""

        while self.loc_mins:
            lm = self.loc_mins.pop()

            e = self._create_loc_min_edge(lm)

            if e is NULL_IDX:
                return False

            vb = self.es[e].vb

            if self._is_horizontal(e):
                self._triangulate_fan(e, vb, curr_y, self.es[e].vl == vb)
            else:
                self._triangulate_fan(e, vb, curr_y, True)

                if not self._completed(e):
                    self._triangulate_fan(e, vb, curr_y, False)

            if len(self.vs[lm].edges) < 2:
                continue

            self._add_active(self.vs[lm].edges[0])
            self._add_active(self.vs[lm].edges[1])

        return True

    def _sweep_horizontals(self, curr_y: int) -> None:
        """Fan the horizontal edges deferred from the finished row."""

        while self.horz:
            e = self.horz.pop()

            if self._completed(e):
                continue

            if self.es[e].vb == self.es[e].vl:
                if self.es[e].kind == ASCEND:
                    self._triangulate_fan(e, self.es[e].vb, curr_y, True)
            elif self.es[e].kind == DESCEND:
                self._triangulate_fan(e, self.es[e].vb, curr_y, False)

    def _sweep_vertex(self, v: int) -> None:
        """Activate the boundary edges starting at v and fan the ones ending at it."""

        for i in range(len(self.vs[v].edges) - 1, -1, -1):
            if i >= len(self.vs[v].edges):
                continue

            e = self.vs[v].edges[i]

            if self._completed(e) or self.es[e].kind == LOOSE:
                continue

            if self._is_horizontal(e):
                self.horz.append(e)

            if v == self.es[e].vb:
                if not self.vs[v].inner_lm:
                    self._add_active(e)
            elif not self._is_horizontal(e):
                self._triangulate_fan(
                    e, self.es[e].vb, self.vs[v].pt[1], self.es[e].kind == ASCEND
                )

    def _sweep(self, order: list[int]) -> bool:
        """Sweep the vertices top to bottom filling triangles row by row; false when a hole cannot be connected."""

        curr_y = self.vs[order[0]].pt[1]

        for v in order:
            if not self.vs[v].edges:
                continue

            if self.vs[v].pt[1] != curr_y:
                if not self._sweep_loc_mins(curr_y):
                    return False

                self._sweep_horizontals(curr_y)
                curr_y = self.vs[v].pt[1]

            self._sweep_vertex(v)

            if self.vs[v].inner_lm:
                self.loc_mins.append(v)

        while self.horz:
            e = self.horz.pop()

            if not self._completed(e) and self.es[e].vb == self.es[e].vl:
                self._triangulate_fan(e, self.es[e].vb, curr_y, True)

        return True

    def _legalize(self) -> None:
        """Flip loose edges until Delaunay, capped so near-cocircular integer points cannot flip-flop forever."""

        max_flips = 64 * len(self.vs) + 4096

        for _flips in range(max_flips):
            if not self.pending:
                return

            e = self.pending.pop()

            self._force_legal(e)

    def _tri_points(self, t: _Tri) -> list[tuple[int, int]]:
        """Both ends of edge 0 and the far end of edge 1."""

        e0 = self.es[t.edges[0]]
        e1 = self.es[t.edges[1]]
        p0 = self.vs[e0.vl].pt
        p1 = self.vs[e0.vr].pt
        p2 = (
            self.vs[e1.vr].pt
            if self.vs[e1.vl].pt == p0 or self.vs[e1.vl].pt == p1
            else self.vs[e1.vl].pt
        )

        return [p0, p1, p2]

    def _triangles(self) -> list[list[tuple[int, int]]]:
        """Counter-clockwise triangles, flat ones dropped."""

        res = []

        for t in self.ts:
            p = self._tri_points(t)
            sign = _cross_sign(p[0], p[1], p[2])

            if sign == 0:
                continue

            if sign < 0:
                p[0], p[2] = p[2], p[0]

            res.append(p)

        return res

    def execute(self, paths) -> list[list[tuple[int, int]]]:
        """Triangles of the paths, empty when they hold no polygon or a hole cannot be connected."""

        if not self._add_paths(paths):
            return []

        if self.vs[self.lowermost].inner_lm:
            self._flip_winding()

        self.loc_mins = []

        order = list(range(len(self.vs)))
        order.sort(key=lambda v: (-self.vs[v].pt[1], self.vs[v].pt[0]))

        self._merge_duplicates(order)

        if not self._sweep(order):
            return []

        self._legalize()

        return self._triangles()


# ═══════════════════════════════════════════════════════════════════════════
# Triangulation
# ═══════════════════════════════════════════════════════════════════════════


def _cdt_scale(border_2d, holes_2d) -> float:
    """Power of ten keeping the largest coordinate inside int64 headroom."""

    max_coord = 1.0

    for p in border_2d:
        max_coord = max(max_coord, abs(p[0]), abs(p[1]))

    for hole in holes_2d:
        for p in hole:
            max_coord = max(max_coord, abs(p[0]), abs(p[1]))

    precision = MAX_PRECISION

    while precision > 0 and max_coord * 10.0**precision > MAX_COORD64:
        precision -= 1

    return 10.0**precision


def _shift_hole_rows(
    border_2d, holes_2d, scale: float
) -> list[list[tuple[float, float]]]:
    """Hole rows sharing an integer y with a border row move one unit down so the sweep never sees a collinear constraint."""

    border_ys = set()

    for p in border_2d:
        border_ys.add(_to_int64(p[1] * scale))

    holes = []

    for hole in holes_2d:
        shifted = []

        for p in hole:
            iy = _to_int64(p[1] * scale)

            if iy in border_ys:
                shifted.append((p[0], float(iy - 1) / scale))
            else:
                shifted.append((p[0], p[1]))

        holes.append(shifted)

    return holes


def _to_path64(pts, scale: float) -> list[tuple[int, int]]:
    """Integer ring, closing duplicate dropped."""

    path = []

    for p in pts:
        path.append(_to_point64(p, scale))

    if len(path) > 1 and path[0] == path[-1]:
        path.pop()

    return path


def _index_map(border_2d, holes_2d, scale: float) -> dict[tuple[int, int], int]:
    """Index of every integer point in the flat list [border..., hole0..., hole1...], first occurrence wins."""

    indices = {}
    index = 0

    for p in border_2d:
        indices.setdefault(_to_point64(p, scale), index)
        index += 1

    for hole in holes_2d:
        for p in hole:
            indices.setdefault(_to_point64(p, scale), index)
            index += 1

    return indices


def _inside_hole(tri, paths, hole_sets) -> bool:
    """A triangle lies in a hole when all its corners are on one hole ring or its centroid is outside the border or inside a hole."""

    for hole_set in hole_sets:
        if tri[0] in hole_set and tri[1] in hole_set and tri[2] in hole_set:
            return True

    c = (
        _div3(tri[0][0] + tri[1][0] + tri[2][0]),
        _div3(tri[0][1] + tri[1][1] + tri[2][1]),
    )

    if not _inside_path64(c, paths[0]):
        return True

    for path in paths[1:]:
        if _inside_path64(c, path):
            return True

    return False


def _remove_hole_triangles(tris, paths) -> None:
    """Drop the triangles the sweep filled inside the holes; edge midpoints are not tested because valid triangles touch the hole rings."""

    hole_sets = []

    for path in paths[1:]:
        hole_sets.append(set(path))

    kept = []

    for tri in tris:
        if not _inside_hole(tri, paths, hole_sets):
            kept.append(tri)

    tris[:] = kept


def _to_indices(tris, indices) -> list[tuple[int, int, int]]:
    """Corner indices into the flat list, triangles with an unknown corner dropped."""

    out = []

    for tri in tris:
        f = [0, 0, 0]
        known = True

        for k in range(3):
            if tri[k] not in indices:
                known = False
            else:
                f[k] = indices[tri[k]]

        if known:
            out.append((f[0], f[1], f[2]))

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Mesh assembly
# ═══════════════════════════════════════════════════════════════════════════


def _strip_close(polyline: Polyline) -> list[Point]:
    """Polyline points without the closing duplicate."""

    pts = polyline.get_points()

    if len(pts) > 1:
        f = pts[0]
        b = pts[-1]

        if (
            abs(f[0] - b[0]) < 1e-12
            and abs(f[1] - b[1]) < 1e-12
            and abs(f[2] - b[2]) < 1e-12
        ):
            pts.pop()

    return pts


def _signed_area(pts) -> float:
    """Signed area of a 2D ring, positive when counter-clockwise."""

    area = 0.0
    n = len(pts)

    for i in range(n):
        j = (i + 1) % n

        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]

    return area * 0.5


def _border_index(polylines: list[Polyline]) -> int:
    """Index of the polyline with the largest bounding-box diagonal."""

    border = 0
    max_diag = 0.0

    for i in range(len(polylines)):
        pts = polylines[i].get_points()

        if len(pts) < 3:
            continue

        lo = Point(pts[0][0], pts[0][1], pts[0][2])
        hi = Point(pts[0][0], pts[0][1], pts[0][2])

        for p in pts:
            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])

        diag = lo.distance(hi)

        if diag > max_diag:
            max_diag = diag
            border = i

    return border


def _project_2d(
    pts: list[Point], origin: Point, xaxis: Vector, yaxis: Vector
) -> list[tuple[float, float]]:
    """Plane coordinates of the points in the frame (origin, xaxis, yaxis)."""

    out = []

    for p in pts:
        dx = p[0] - origin[0]
        dy = p[1] - origin[1]
        dz = p[2] - origin[2]

        out.append(
            (
                dx * xaxis[0] + dy * xaxis[1] + dz * xaxis[2],
                dx * yaxis[0] + dy * yaxis[1] + dz * yaxis[2],
            )
        )

    return out


def _cover_missing(tri_list: list[list[int]], vkeys: list[int], n: int) -> None:
    """Ear triangles for border vertices no triangle touches, so every vertex is drawn."""

    covered = set()

    for t in tri_list:
        covered.update(t)

    for m in range(n):
        if vkeys[m] not in covered:
            tri_list.append([vkeys[(m + n - 1) % n], vkeys[m], vkeys[(m + 1) % n]])


def _build_mesh(
    border: list[Point], holes: list[list[Point]], tris: list[tuple[int, int, int]]
) -> Mesh:
    """One face over the border with the holes as face holes, or one face per triangle under SESSION_CONFIG.explode_mesh_faces."""

    from .mesh import Mesh

    mesh = Mesh()
    vkeys = []

    for p in border:
        vkeys.append(mesh.add_vertex(p))

    for hole in holes:
        for p in hole:
            vkeys.append(mesh.add_vertex(p))

    if SESSION_CONFIG.explode_mesh_faces:
        for t in tris:
            mesh.add_face([vkeys[t[0]], vkeys[t[1]], vkeys[t[2]]])

        return mesh

    ring = vkeys[: len(border)]
    fkey = mesh.add_face(ring)

    if fkey is None:
        return mesh

    tri_list = []

    for t in tris:
        f = [vkeys[t[0]], vkeys[t[1]], vkeys[t[2]]]

        if f[0] != f[1] and f[1] != f[2] and f[2] != f[0]:
            tri_list.append(f)

    if not holes:
        _cover_missing(tri_list, vkeys, len(border))
    else:
        hole_rings = []
        off = len(border)

        for hole in holes:
            hole_rings.append(vkeys[off : off + len(hole)])
            off += len(hole)

        mesh.set_face_holes(fkey, hole_rings)

    mesh.set_face_triangulation(fkey, tri_list)

    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# RemeshCDT
# ═══════════════════════════════════════════════════════════════════════════


def cdt_triangulate(border_2d, holes_2d) -> list[tuple[int, int, int]]:
    """Triangle index triples of a counter-clockwise 2D border with clockwise holes into the flat list [border..., hole0..., hole1...]."""

    scale = _cdt_scale(border_2d, holes_2d)
    holes = _shift_hole_rows(border_2d, holes_2d, scale)
    paths = [_to_path64(border_2d, scale)]

    for hole in holes:
        paths.append(_to_path64(hole, scale))

    delaunay = _Delaunay()
    tris = delaunay.execute(paths)

    if holes:
        _remove_hole_triangles(tris, paths)

    return _to_indices(tris, _index_map(border_2d, holes, scale))


class RemeshCDT:
    """Constrained Delaunay triangulation of a border polyline with hole polylines."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Triangulation
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def triangulate(polylines: list[Polyline]) -> list[tuple[int, int, int]]:
        """Triangle index triples into the flat list [border..., hole0..., hole1...], closing duplicates stripped."""

        if not polylines:
            return []

        border = _strip_close(polylines[0])

        if len(border) < 3:
            return []

        border_2d = []

        for p in border:
            border_2d.append((p[0], p[1]))

        holes_2d = []

        for i in range(1, len(polylines)):
            hole_2d = []

            for p in _strip_close(polylines[i]):
                hole_2d.append((p[0], p[1]))

            holes_2d.append(hole_2d)

        return cdt_triangulate(border_2d, holes_2d)

    @staticmethod
    def from_polylines(
        polylines: list[Polyline], is_2d: bool = False, is_first_boundary: bool = True
    ) -> Mesh:
        """Mesh of one face with holes, or one face per triangle under SESSION_CONFIG.explode_mesh_faces; is_2d skips the plane projection, is_first_boundary=false picks the border by largest bbox diagonal."""

        from .mesh import Mesh

        if not polylines:
            return Mesh()

        border_idx = (
            0 if is_first_boundary or len(polylines) == 1 else _border_index(polylines)
        )
        border = _strip_close(polylines[border_idx])

        if len(border) < 3:
            return Mesh()

        holes = []

        for i in range(len(polylines)):
            if i == border_idx:
                continue

            hole = _strip_close(polylines[i])

            if len(hole) >= 3:
                holes.append(hole)

        origin = Point(0.0, 0.0, 0.0)
        xaxis = Vector(1.0, 0.0, 0.0)
        yaxis = Vector(0.0, 1.0, 0.0)

        if not is_2d:
            all_pts = list(border)

            for hole in holes:
                all_pts.extend(hole)

            origin, xaxis, yaxis, _zaxis = Polyline(all_pts).get_average_plane()

        border_2d = _project_2d(border, origin, xaxis, yaxis)

        if _signed_area(border_2d) < 0.0:
            border.reverse()
            border_2d.reverse()

        holes_2d = []

        for hole in holes:
            hole_2d = _project_2d(hole, origin, xaxis, yaxis)

            if _signed_area(hole_2d) > 0.0:
                hole.reverse()
                hole_2d.reverse()

            holes_2d.append(hole_2d)

        return _build_mesh(border, holes, cdt_triangulate(border_2d, holes_2d))
