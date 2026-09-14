from __future__ import annotations
import heapq
import math
from .polyline import Polyline

_VF_NONE = 0
_VF_LOCAL_MAX = 4
_VF_LOCAL_MIN = 8
_JW_NONE = 0
_JW_LEFT = 1
_JW_RIGHT = 2
_INF = float("inf")


class _BIVec2:
    __slots__ = ("x", "y")
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y
    def __eq__(self, o):
        return self.x == o.x and self.y == o.y
    def __ne__(self, o):
        return self.x != o.x or self.y != o.y


class _VVertex:
    __slots__ = ("pt", "next", "prev", "flags")
    def __init__(self):
        self.pt = _BIVec2()
        self.next = None
        self.prev = None
        self.flags = _VF_NONE


class _VLocalMinima:
    __slots__ = ("vertex", "polytype")
    def __init__(self, vertex: _VVertex, polytype: int):
        self.vertex = vertex
        self.polytype = polytype


class _VOutPt:
    __slots__ = ("pt", "next", "prev", "outrec", "horz")
    def __init__(self, pt: _BIVec2, outrec: "_VOutRec"):
        self.pt = pt
        self.outrec = outrec
        self.next = self
        self.prev = self
        self.horz = None


class _VOutRec:
    __slots__ = ("idx", "front_edge", "back_edge", "pts", "owner")
    def __init__(self, idx: int = 0):
        self.idx = idx
        self.front_edge = None
        self.back_edge = None
        self.pts = None
        self.owner = None


class _VActive:
    __slots__ = ("bot", "top", "curr_x", "dx", "wind_dx", "wind_cnt", "wind_cnt2",
                 "outrec", "prev_in_ael", "next_in_ael", "prev_in_sel", "next_in_sel",
                 "jump", "vertex_top", "local_min", "is_left_bound", "join_with")
    def __init__(self):
        self.bot = _BIVec2()
        self.top = _BIVec2()
        self.curr_x = 0
        self.dx = 0.0
        self.wind_dx = 1
        self.wind_cnt = 0
        self.wind_cnt2 = 0
        self.outrec = None
        self.prev_in_ael = None
        self.next_in_ael = None
        self.prev_in_sel = None
        self.next_in_sel = None
        self.jump = None
        self.vertex_top = None
        self.local_min = None
        self.is_left_bound = False
        self.join_with = _JW_NONE


class _VIntersectNode:
    __slots__ = ("pt", "edge1", "edge2")
    def __init__(self, pt: _BIVec2, edge1: _VActive, edge2: _VActive):
        self.pt = pt
        self.edge1 = edge1
        self.edge2 = edge2


class _VHorzSeg:
    __slots__ = ("left_op", "right_op", "left_to_right")
    def __init__(self, left_op: _VOutPt):
        self.left_op = left_op
        self.right_op = None
        self.left_to_right = True


class _VHorzJoin:
    __slots__ = ("op1", "op2")
    def __init__(self, op1: _VOutPt, op2: _VOutPt):
        self.op1 = op1
        self.op2 = op2


class _ScanlineHeap:
    def __init__(self):
        self._heap = []
    def clear(self) -> None:
        self._heap.clear()
    def empty(self) -> bool:
        return len(self._heap) == 0
    def push(self, y: int) -> None:
        heapq.heappush(self._heap, -y)
    def top(self) -> int:
        return -self._heap[0]
    def pop(self) -> None:
        heapq.heappop(self._heap)


class _VattiScratch:
    def __init__(self):
        self.locmin_list = []
        self.intersect_nodes = []
        self.horz_seg_list = []
        self.horz_join_list = []
        self.outrec_list = []
        self.scanline_list = _ScanlineHeap()
        self.actives = None
        self.sel = None
        self.bot_y = 0
        self.locmin_idx = 0
        self.succeeded = True

    def new_outpt(self, pt: _BIVec2, rec: _VOutRec) -> _VOutPt:
        o = _VOutPt(_BIVec2(pt.x, pt.y), rec)
        return o

    def new_outrec(self) -> _VOutRec:
        r = _VOutRec(len(self.outrec_list))
        self.outrec_list.append(r)
        return r


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def _v_get_dx(p1: _BIVec2, p2: _BIVec2) -> float:
    dy = float(p2.y - p1.y)
    if dy != 0:
        return float(p2.x - p1.x) / dy
    return -_INF if p2.x > p1.x else _INF

def _v_top_x(ae: _VActive, y: int) -> int:
    if y == ae.top.y or ae.top.x == ae.bot.x:
        return ae.top.x
    if y == ae.bot.y:
        return ae.bot.x
    return ae.bot.x + round(ae.dx * float(y - ae.bot.y))

def _v_is_horizontal(e: _VActive) -> bool:
    return e.top.y == e.bot.y

def _v_is_hot(e: _VActive) -> bool:
    return e.outrec is not None

def _v_is_maxima_v(v: _VVertex) -> bool:
    return (v.flags & _VF_LOCAL_MAX) != 0

def _v_is_maxima_e(e: _VActive) -> bool:
    return _v_is_maxima_v(e.vertex_top)

def _v_is_front(e: _VActive) -> bool:
    return e is e.outrec.front_edge

def _v_is_joined(e: _VActive) -> bool:
    return e.join_with != _JW_NONE

def _v_same_polytype(a: _VActive, b: _VActive) -> bool:
    return a.local_min.polytype == b.local_min.polytype

def _v_polytype(e: _VActive) -> int:
    return e.local_min.polytype

def _v_set_dx(e: _VActive) -> None:
    e.dx = _v_get_dx(e.bot, e.top)

def _v_next_vertex(e: _VActive) -> _VVertex:
    return e.vertex_top.next if e.wind_dx > 0 else e.vertex_top.prev

def _v_prev_prev_vertex(ae: _VActive) -> _VVertex:
    return ae.vertex_top.prev.prev if ae.wind_dx > 0 else ae.vertex_top.next.next

def _v_cross_product(p1: _BIVec2, p2: _BIVec2, p3: _BIVec2) -> float:
    return float(p2.x - p1.x) * float(p3.y - p2.y) - float(p2.y - p1.y) * float(p3.x - p2.x)

def _v_dot_product(p1: _BIVec2, p2: _BIVec2, p3: _BIVec2) -> float:
    return float(p2.x - p1.x) * float(p3.x - p2.x) + float(p2.y - p1.y) * float(p3.y - p2.y)

def _v_products_equal(a: int, b: int, c: int, d: int) -> bool:
    return a * b == c * d

def _v_is_collinear(p1: _BIVec2, shared: _BIVec2, p2: _BIVec2) -> bool:
    return _v_products_equal(shared.x - p1.x, p2.y - shared.y, shared.y - p1.y, p2.x - shared.x)

def _v_perpendic_dist_sq(pt: _BIVec2, l1: _BIVec2, l2: _BIVec2) -> float:
    a = float(pt.x - l1.x)
    b = float(pt.y - l1.y)
    c = float(l2.x - l1.x)
    d = float(l2.y - l1.y)
    if c == 0 and d == 0:
        return 0.0
    e = a * d - c * b
    return (e * e) / (c * c + d * d)

def _v_get_seg_isect_pt(a: _BIVec2, b: _BIVec2, c: _BIVec2, d: _BIVec2) -> tuple[bool, _BIVec2]:
    dx1 = float(b.x - a.x)
    dy1 = float(b.y - a.y)
    dx2 = float(d.x - c.x)
    dy2 = float(d.y - c.y)
    det = dy1 * dx2 - dy2 * dx1
    if det == 0.0:
        return False, _BIVec2()
    t = (float(a.x - c.x) * dy2 - float(a.y - c.y) * dx2) / det
    if t <= 0.0:
        ip = _BIVec2(a.x, a.y)
    elif t >= 1.0:
        ip = _BIVec2(b.x, b.y)
    else:
        ip = _BIVec2(a.x + round(t * dx1), a.y + round(t * dy1))
    return True, ip

def _v_closest_pt_on_seg(pt: _BIVec2, s1: _BIVec2, s2: _BIVec2) -> _BIVec2:
    if s1 == s2:
        return _BIVec2(s1.x, s1.y)
    dx = float(s2.x - s1.x)
    dy = float(s2.y - s1.y)
    q = (float(pt.x - s1.x) * dx + float(pt.y - s1.y) * dy) / (dx * dx + dy * dy)
    if q < 0:
        q = 0.0
    elif q > 1:
        q = 1.0
    return _BIVec2(s1.x + round(q * dx), s1.y + round(q * dy))

def _v_sign_d(v: float) -> int:
    return (v > 0) - (v < 0)

def _v_segs_intersect(a: _BIVec2, b: _BIVec2, c: _BIVec2, d: _BIVec2) -> bool:
    return (_v_sign_d(_v_cross_product(a, c, d)) * _v_sign_d(_v_cross_product(b, c, d)) < 0) and \
           (_v_sign_d(_v_cross_product(c, a, b)) * _v_sign_d(_v_cross_product(d, a, b)) < 0)

def _v_area_outpt(op: _VOutPt) -> float:
    r = 0.0
    o = op
    while True:
        r += float(o.prev.pt.y + o.pt.y) * float(o.prev.pt.x - o.pt.x)
        o = o.next
        if o is op:
            break
    return r * 0.5

def _v_area_tri(p1: _BIVec2, p2: _BIVec2, p3: _BIVec2) -> float:
    return float(p3.y + p1.y) * float(p3.x - p1.x) + float(p1.y + p2.y) * float(p1.x - p2.x) + float(p2.y + p3.y) * float(p2.x - p3.x)

def _v_pts_close(a: _BIVec2, b: _BIVec2) -> bool:
    return abs(a.x - b.x) < 2 and abs(a.y - b.y) < 2

def _v_very_small_tri(op: _VOutPt) -> bool:
    return op.next.next is op.prev and (_v_pts_close(op.prev.pt, op.next.pt) or _v_pts_close(op.pt, op.next.pt) or _v_pts_close(op.pt, op.prev.pt))

def _v_valid_closed(op: _VOutPt) -> bool:
    return op is not None and op.next is not op and op.next is not op.prev and not _v_very_small_tri(op)

def _v_winding_step(pt: _BIVec2, a: _BIVec2, b: _BIVec2) -> int:
    cross = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x)
    if a.y <= pt.y:
        return 1 if (b.y > pt.y and cross > 0) else 0
    return -1 if (b.y <= pt.y and cross < 0) else 0

def _pip_i(pt: _BIVec2, poly: list[_BIVec2]) -> bool:
    winding = 0
    n = len(poly)
    for i in range(n):
        winding += _v_winding_step(pt, poly[i], poly[(i + 1) % n])
    return winding != 0

def _pip_vertex(pt: _BIVec2, head: _VVertex) -> bool:
    winding = 0
    v = head
    while True:
        winding += _v_winding_step(pt, v.pt, v.next.pt)
        v = v.next
        if v is head:
            break
    return winding != 0


# ═══════════════════════════════════════════════════════════════════════════
# Vertex building and local minima detection
# ═══════════════════════════════════════════════════════════════════════════

def _v_find_local_minima(head: _VVertex, polytype: int, sc: _VattiScratch) -> None:
    pv = head.prev
    while pv is not head and pv.pt.y == head.pt.y:
        pv = pv.prev
    if pv is head:
        return
    going_up = pv.pt.y > head.pt.y
    going_up0 = going_up
    pv = head
    cv = head.next
    while cv is not head:
        if cv.pt.y > pv.pt.y and going_up:
            pv.flags |= _VF_LOCAL_MAX
            going_up = False
        elif cv.pt.y < pv.pt.y and not going_up:
            going_up = True
            pv.flags |= _VF_LOCAL_MIN
            sc.locmin_list.append(_VLocalMinima(pv, polytype))
        pv = cv
        cv = cv.next
    if going_up != going_up0:
        if going_up0:
            pv.flags |= _VF_LOCAL_MIN
            sc.locmin_list.append(_VLocalMinima(pv, polytype))
        else:
            pv.flags |= _VF_LOCAL_MAX


def _v_link_path(pts: list[_BIVec2], n: int, polytype: int, sc: _VattiScratch) -> _VVertex | None:
    """Links n scaled points into a circular vertex list; returns its head or None if degenerate."""
    head = _VVertex()
    head.pt = pts[0]
    prev_v = head
    cnt = 1
    for i in range(1, n):
        if pts[i] == prev_v.pt:
            continue
        cv = _VVertex()
        cv.pt = pts[i]
        cv.prev = prev_v
        prev_v.next = cv
        prev_v = cv
        cnt += 1
    if cnt >= 3 and prev_v.pt == head.pt:
        prev_v = prev_v.prev
        cnt -= 1
    if cnt < 3:
        return None
    prev_v.next = head
    head.prev = prev_v
    _v_find_local_minima(head, polytype, sc)
    return head


def _v_add_path_from_doubles(coords: list[float], n: int, polytype: int, bool_scale: float, sc: _VattiScratch) -> tuple[_VVertex | None, int, int, int, int]:
    if n < 3:
        return None, 0, 0, 0, 0
    pts = []
    for i in range(n):
        pts.append(_BIVec2(round(coords[i * 3] * bool_scale), round(coords[i * 3 + 1] * bool_scale)))
    minX, maxX, minY, maxY = _v_bounds(pts)
    return _v_link_path(pts, n, polytype, sc), minX, maxX, minY, maxY


def _v_add_path(pts: list[_BIVec2], n: int, polytype: int, sc: _VattiScratch) -> None:
    if n < 3:
        return
    _v_link_path(pts, n, polytype, sc)


# ═══════════════════════════════════════════════════════════════════════════
# AEL operations
# ═══════════════════════════════════════════════════════════════════════════

def _v_get_maxima_pair(e: _VActive) -> _VActive | None:
    e2 = e.next_in_ael
    while e2 is not None:
        if e2.vertex_top is e.vertex_top:
            return e2
        e2 = e2.next_in_ael
    return None

def _v_get_curr_y_maxima(e: _VActive) -> _VVertex | None:
    r = e.vertex_top
    if e.wind_dx > 0:
        while r.next.pt.y == r.pt.y:
            r = r.next
    else:
        while r.prev.pt.y == r.pt.y:
            r = r.prev
    return r if _v_is_maxima_v(r) else None

def _v_get_prev_hot(e: _VActive) -> _VActive | None:
    p = e.prev_in_ael
    while p is not None and not _v_is_hot(p):
        p = p.prev_in_ael
    return p

def _v_is_valid_ael_order(resident: _VActive, newcomer: _VActive) -> bool:
    if newcomer.curr_x != resident.curr_x:
        return newcomer.curr_x > resident.curr_x
    d = _v_cross_product(resident.top, newcomer.bot, newcomer.top)
    if d != 0:
        return d < 0
    if not _v_is_maxima_e(resident) and resident.top.y > newcomer.top.y:
        return _v_cross_product(newcomer.bot, resident.top, _v_next_vertex(resident).pt) <= 0
    if not _v_is_maxima_e(newcomer) and newcomer.top.y > resident.top.y:
        return _v_cross_product(newcomer.bot, newcomer.top, _v_next_vertex(newcomer).pt) >= 0
    y = newcomer.bot.y
    if resident.bot.y != y or resident.local_min.vertex.pt.y != y:
        return newcomer.is_left_bound
    if resident.is_left_bound != newcomer.is_left_bound:
        return newcomer.is_left_bound
    if _v_is_collinear(_v_prev_prev_vertex(resident).pt, resident.bot, resident.top):
        return True
    return (_v_cross_product(_v_prev_prev_vertex(resident).pt, newcomer.bot, _v_prev_prev_vertex(newcomer).pt) > 0) == newcomer.is_left_bound

def _v_insert_left_edge(sc: _VattiScratch, e: _VActive) -> None:
    if sc.actives is None:
        e.prev_in_ael = None
        e.next_in_ael = None
        sc.actives = e
    elif not _v_is_valid_ael_order(sc.actives, e):
        e.prev_in_ael = None
        e.next_in_ael = sc.actives
        sc.actives.prev_in_ael = e
        sc.actives = e
    else:
        e2 = sc.actives
        while e2.next_in_ael is not None and _v_is_valid_ael_order(e2.next_in_ael, e):
            e2 = e2.next_in_ael
        if e2.join_with == _JW_RIGHT:
            e2 = e2.next_in_ael
        if e2 is None:
            return
        e.next_in_ael = e2.next_in_ael
        if e2.next_in_ael is not None:
            e2.next_in_ael.prev_in_ael = e
        e.prev_in_ael = e2
        e2.next_in_ael = e

def _v_insert_right_edge(e: _VActive, e2: _VActive) -> None:
    e2.next_in_ael = e.next_in_ael
    if e.next_in_ael is not None:
        e.next_in_ael.prev_in_ael = e2
    e2.prev_in_ael = e
    e.next_in_ael = e2

def _v_swap_positions_in_ael(sc: _VattiScratch, e1: _VActive, e2: _VActive) -> None:
    nxt = e2.next_in_ael
    if nxt is not None:
        nxt.prev_in_ael = e1
    prv = e1.prev_in_ael
    if prv is not None:
        prv.next_in_ael = e2
    e2.prev_in_ael = prv
    e2.next_in_ael = e1
    e1.prev_in_ael = e2
    e1.next_in_ael = nxt
    if e2.prev_in_ael is None:
        sc.actives = e2

def _v_delete_from_ael(sc: _VattiScratch, e: _VActive) -> None:
    prv = e.prev_in_ael
    nxt = e.next_in_ael
    if prv is None and nxt is None and e is not sc.actives:
        return
    if prv is not None:
        prv.next_in_ael = nxt
    else:
        sc.actives = nxt
    if nxt is not None:
        nxt.prev_in_ael = prv


# ═══════════════════════════════════════════════════════════════════════════
# Scanline
# ═══════════════════════════════════════════════════════════════════════════

def _v_insert_scanline(sc: _VattiScratch, y: int) -> None:
    sc.scanline_list.push(y)

def _v_pop_scanline(sc: _VattiScratch) -> tuple[bool, int]:
    sl = sc.scanline_list
    if sl.empty():
        return False, 0
    y = sl.top()
    sl.pop()
    while not sl.empty() and y == sl.top():
        sl.pop()
    return True, y

def _v_pop_locmin(sc: _VattiScratch, y: int) -> tuple[bool, _VLocalMinima | None]:
    if sc.locmin_idx >= len(sc.locmin_list) or sc.locmin_list[sc.locmin_idx].vertex.pt.y != y:
        return False, None
    lm = sc.locmin_list[sc.locmin_idx]
    sc.locmin_idx += 1
    return True, lm

def _v_push_horz(sc: _VattiScratch, e: _VActive) -> None:
    e.next_in_sel = sc.sel
    sc.sel = e

def _v_pop_horz(sc: _VattiScratch) -> tuple[bool, _VActive | None]:
    e = sc.sel
    if e is None:
        return False, None
    sc.sel = sc.sel.next_in_sel
    return True, e


# ═══════════════════════════════════════════════════════════════════════════
# Winding and contribution
# ═══════════════════════════════════════════════════════════════════════════

def _v_set_wind_count(sc: _VattiScratch, e: _VActive) -> None:
    e2 = e.prev_in_ael
    pt = _v_polytype(e)
    while e2 is not None and _v_polytype(e2) != pt:
        e2 = e2.prev_in_ael
    if e2 is None:
        e.wind_cnt = e.wind_dx
        e2 = sc.actives
    else:
        if e2.wind_cnt * e2.wind_dx < 0:
            if abs(e2.wind_cnt) > 1:
                e.wind_cnt = e2.wind_cnt if e2.wind_dx * e.wind_dx < 0 else e2.wind_cnt + e.wind_dx
            else:
                e.wind_cnt = e.wind_dx
        else:
            e.wind_cnt = e2.wind_cnt if e2.wind_dx * e.wind_dx < 0 else e2.wind_cnt + e.wind_dx
        e.wind_cnt2 = e2.wind_cnt2
        e2 = e2.next_in_ael
    while e2 is not e:
        if _v_polytype(e2) != pt:
            e.wind_cnt2 += e2.wind_dx
        e2 = e2.next_in_ael

def _v_is_contributing(e: _VActive, cliptype: int) -> bool:
    if abs(e.wind_cnt) != 1:
        return False
    wc2 = abs(e.wind_cnt2)
    if cliptype == 0:
        return wc2 != 0
    if cliptype == 1:
        return wc2 == 0
    r = (wc2 == 0)
    return r if _v_polytype(e) == 0 else not r


# ═══════════════════════════════════════════════════════════════════════════
# Output operations
# ═══════════════════════════════════════════════════════════════════════════

def _v_set_sides(or_: _VOutRec, f: _VActive, b: _VActive) -> None:
    or_.front_edge = f
    or_.back_edge = b

def _v_swap_outrecs(e1: _VActive, e2: _VActive) -> None:
    or1 = e1.outrec
    or2 = e2.outrec
    if or1 is or2:
        or1.front_edge, or1.back_edge = or1.back_edge, or1.front_edge
        return
    if or1 is not None:
        if e1 is or1.front_edge:
            or1.front_edge = e2
        else:
            or1.back_edge = e2
    if or2 is not None:
        if e2 is or2.front_edge:
            or2.front_edge = e1
        else:
            or2.back_edge = e1
    e1.outrec = or2
    e2.outrec = or1

def _v_add_outpt(e: _VActive, pt: _BIVec2, sc: _VattiScratch) -> _VOutPt:
    outrec = e.outrec
    to_front = _v_is_front(e)
    op_front = outrec.pts
    op_back = op_front.next
    if to_front and pt == op_front.pt:
        return op_front
    if not to_front and pt == op_back.pt:
        return op_back
    nop = sc.new_outpt(pt, outrec)
    op_back.prev = nop
    nop.prev = op_front
    nop.next = op_back
    op_front.next = nop
    if to_front:
        outrec.pts = nop
    return nop

def _v_add_local_min_poly(e1: _VActive, e2: _VActive, pt: _BIVec2, sc: _VattiScratch, is_new: bool) -> _VOutPt:
    outrec = sc.new_outrec()
    e1.outrec = outrec
    e2.outrec = outrec
    prev_hot = _v_get_prev_hot(e1)
    if prev_hot is not None:
        if (prev_hot is prev_hot.outrec.front_edge) == is_new:
            _v_set_sides(outrec, e2, e1)
        else:
            _v_set_sides(outrec, e1, e2)
    else:
        outrec.owner = None
        if is_new:
            _v_set_sides(outrec, e1, e2)
        else:
            _v_set_sides(outrec, e2, e1)
    op = sc.new_outpt(pt, outrec)
    outrec.pts = op
    return op

def _v_uncouple(ae: _VActive) -> None:
    or_ = ae.outrec
    if or_ is None:
        return
    or_.front_edge.outrec = None
    or_.back_edge.outrec = None
    or_.front_edge = None
    or_.back_edge = None

def _v_join_outrec_paths(e1: _VActive, e2: _VActive) -> None:
    p1_st = e1.outrec.pts
    p2_st = e2.outrec.pts
    p1_end = p1_st.next
    p2_end = p2_st.next
    if _v_is_front(e1):
        p2_end.prev = p1_st
        p1_st.next = p2_end
        p2_st.next = p1_end
        p1_end.prev = p2_st
        e1.outrec.pts = p2_st
        e1.outrec.front_edge = e2.outrec.front_edge
        if e1.outrec.front_edge is not None:
            e1.outrec.front_edge.outrec = e1.outrec
    else:
        p1_end.prev = p2_st
        p2_st.next = p1_end
        p1_st.next = p2_end
        p2_end.prev = p1_st
        e1.outrec.back_edge = e2.outrec.back_edge
        if e1.outrec.back_edge is not None:
            e1.outrec.back_edge.outrec = e1.outrec
    e2.outrec.front_edge = None
    e2.outrec.back_edge = None
    e2.outrec.pts = None
    e2.outrec.owner = e1.outrec
    e1.outrec = None
    e2.outrec = None


def _v_add_local_max_poly(e1: _VActive, e2: _VActive, pt: _BIVec2, sc: _VattiScratch) -> _VOutPt | None:
    if _v_is_joined(e1):
        _v_split(e1, pt, sc)
    if _v_is_joined(e2):
        _v_split(e2, pt, sc)
    if _v_is_front(e1) == _v_is_front(e2):
        sc.succeeded = False
        return None
    result = _v_add_outpt(e1, pt, sc)
    if e1.outrec is e2.outrec:
        outrec = e1.outrec
        outrec.pts = result
        _v_uncouple(e1)
        result = outrec.pts
    elif e1.outrec.idx < e2.outrec.idx:
        _v_join_outrec_paths(e1, e2)
    else:
        _v_join_outrec_paths(e2, e1)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Split and check join
# ═══════════════════════════════════════════════════════════════════════════

def _v_split(e: _VActive, pt: _BIVec2, sc: _VattiScratch) -> None:
    if e.join_with == _JW_RIGHT:
        e.join_with = _JW_NONE
        e.next_in_ael.join_with = _JW_NONE
        _v_add_local_min_poly(e, e.next_in_ael, pt, sc, True)
    else:
        e.join_with = _JW_NONE
        e.prev_in_ael.join_with = _JW_NONE
        _v_add_local_min_poly(e.prev_in_ael, e, pt, sc, True)

def _v_check_join_left(e: _VActive, pt: _BIVec2, sc: _VattiScratch, check_curr_x: bool = False) -> None:
    prev = e.prev_in_ael
    if prev is None or not _v_is_hot(e) or not _v_is_hot(prev) or _v_is_horizontal(e) or _v_is_horizontal(prev):
        return
    if (pt.y < e.top.y + 2 or pt.y < prev.top.y + 2) and (e.bot.y > pt.y or prev.bot.y > pt.y):
        return
    if check_curr_x:
        if _v_perpendic_dist_sq(pt, prev.bot, prev.top) > 0.25:
            return
    else:
        if e.curr_x != prev.curr_x:
            return
    if not _v_is_collinear(e.top, pt, prev.top):
        return
    if e.outrec.idx == prev.outrec.idx:
        _v_add_local_max_poly(prev, e, pt, sc)
    elif e.outrec.idx < prev.outrec.idx:
        _v_join_outrec_paths(e, prev)
    else:
        _v_join_outrec_paths(prev, e)
    prev.join_with = _JW_RIGHT
    e.join_with = _JW_LEFT

def _v_check_join_right(e: _VActive, pt: _BIVec2, sc: _VattiScratch, check_curr_x: bool = False) -> None:
    nxt = e.next_in_ael
    if nxt is None or not _v_is_hot(e) or not _v_is_hot(nxt) or _v_is_horizontal(e) or _v_is_horizontal(nxt):
        return
    if (pt.y < e.top.y + 2 or pt.y < nxt.top.y + 2) and (e.bot.y > pt.y or nxt.bot.y > pt.y):
        return
    if check_curr_x:
        if _v_perpendic_dist_sq(pt, nxt.bot, nxt.top) > 0.35:
            return
    else:
        if e.curr_x != nxt.curr_x:
            return
    if not _v_is_collinear(e.top, pt, nxt.top):
        return
    if e.outrec.idx == nxt.outrec.idx:
        _v_add_local_max_poly(e, nxt, pt, sc)
    elif e.outrec.idx < nxt.outrec.idx:
        _v_join_outrec_paths(e, nxt)
    else:
        _v_join_outrec_paths(nxt, e)
    e.join_with = _JW_RIGHT
    nxt.join_with = _JW_LEFT


# ═══════════════════════════════════════════════════════════════════════════
# Intersect edges
# ═══════════════════════════════════════════════════════════════════════════

def _v_intersect_edges(e1: _VActive, e2: _VActive, pt: _BIVec2, sc: _VattiScratch, cliptype: int) -> None:
    if _v_is_joined(e1):
        _v_split(e1, pt, sc)
    if _v_is_joined(e2):
        _v_split(e2, pt, sc)

    if _v_polytype(e1) == _v_polytype(e2):
        if e1.wind_cnt + e2.wind_dx == 0:
            e1.wind_cnt = -e1.wind_cnt
        else:
            e1.wind_cnt += e2.wind_dx
        if e2.wind_cnt - e1.wind_dx == 0:
            e2.wind_cnt = -e2.wind_cnt
        else:
            e2.wind_cnt -= e1.wind_dx
    else:
        e1.wind_cnt2 += e2.wind_dx
        e2.wind_cnt2 -= e1.wind_dx

    old_e1_wc = abs(e1.wind_cnt)
    old_e2_wc = abs(e2.wind_cnt)
    e1_in01 = old_e1_wc == 0 or old_e1_wc == 1
    e2_in01 = old_e2_wc == 0 or old_e2_wc == 1
    if (not _v_is_hot(e1) and not e1_in01) or (not _v_is_hot(e2) and not e2_in01):
        return

    if _v_is_hot(e1) and _v_is_hot(e2):
        if (old_e1_wc != 0 and old_e1_wc != 1) or (old_e2_wc != 0 and old_e2_wc != 1) or (_v_polytype(e1) != _v_polytype(e2)):
            _v_add_local_max_poly(e1, e2, pt, sc)
        elif _v_is_front(e1) or e1.outrec is e2.outrec:
            _v_add_local_max_poly(e1, e2, pt, sc)
            _v_add_local_min_poly(e1, e2, pt, sc, False)
        else:
            _v_add_outpt(e1, pt, sc)
            _v_add_outpt(e2, pt, sc)
            _v_swap_outrecs(e1, e2)
    elif _v_is_hot(e1):
        _v_add_outpt(e1, pt, sc)
        _v_swap_outrecs(e1, e2)
    elif _v_is_hot(e2):
        _v_add_outpt(e2, pt, sc)
        _v_swap_outrecs(e1, e2)
    else:
        e1Wc2 = abs(e1.wind_cnt2)
        e2Wc2 = abs(e2.wind_cnt2)
        if not _v_same_polytype(e1, e2):
            _v_add_local_min_poly(e1, e2, pt, sc, False)
        elif old_e1_wc == 1 and old_e2_wc == 1:
            if cliptype == 0:
                if e1Wc2 > 0 and e2Wc2 > 0:
                    _v_add_local_min_poly(e1, e2, pt, sc, False)
            elif cliptype == 1:
                if e1Wc2 <= 0 and e2Wc2 <= 0:
                    _v_add_local_min_poly(e1, e2, pt, sc, False)
            else:
                if (_v_polytype(e1) == 1 and e1Wc2 > 0 and e2Wc2 > 0) or \
                   (_v_polytype(e1) == 0 and e1Wc2 <= 0 and e2Wc2 <= 0):
                    _v_add_local_min_poly(e1, e2, pt, sc, False)


# ═══════════════════════════════════════════════════════════════════════════
# Horizontal edges
# ═══════════════════════════════════════════════════════════════════════════

def _v_add_trial_horz_join(sc: _VattiScratch, op: _VOutPt) -> None:
    sc.horz_seg_list.append(_VHorzSeg(op))

def _v_get_last_op(e: _VActive) -> _VOutPt:
    r = e.outrec.pts
    if e is not e.outrec.front_edge:
        r = r.next
    return r

def _v_update_edge_into_ael(sc: _VattiScratch, e: _VActive) -> None:
    e.bot = _BIVec2(e.top.x, e.top.y)
    e.vertex_top = _v_next_vertex(e)
    e.top = _BIVec2(e.vertex_top.pt.x, e.vertex_top.pt.y)
    e.curr_x = e.bot.x
    _v_set_dx(e)
    if _v_is_joined(e):
        _v_split(e, e.bot, sc)
    if _v_is_horizontal(e):
        pt = _v_next_vertex(e).pt
        while pt.y == e.top.y:
            if (pt.x < e.top.x) != (e.bot.x < e.top.x):
                break
            e.vertex_top = _v_next_vertex(e)
            e.top = _BIVec2(pt.x, pt.y)
            if _v_is_maxima_e(e):
                break
            pt = _v_next_vertex(e).pt
        _v_set_dx(e)
        return
    _v_insert_scanline(sc, e.top.y)
    _v_check_join_left(e, e.bot, sc)
    _v_check_join_right(e, e.bot, sc, True)

def _v_reset_horz_dir(horz: _VActive, max_v: _VVertex) -> tuple[bool, int, int]:
    if horz.bot.x == horz.top.x:
        left = horz.curr_x
        right = horz.curr_x
        e = horz.next_in_ael
        while e is not None and e.vertex_top is not max_v:
            e = e.next_in_ael
        return e is not None, left, right
    if horz.curr_x < horz.top.x:
        return True, horz.curr_x, horz.top.x
    return False, horz.top.x, horz.curr_x

def _v_do_horizontal(horz: _VActive, sc: _VattiScratch, cliptype: int) -> None:
    y = horz.bot.y
    vertex_max = _v_get_curr_y_maxima(horz)
    is_ltr, horz_left, horz_right = _v_reset_horz_dir(horz, vertex_max)
    if _v_is_hot(horz):
        op = _v_add_outpt(horz, _BIVec2(horz.curr_x, y), sc)
        _v_add_trial_horz_join(sc, op)

    while True:
        e = horz.next_in_ael if is_ltr else horz.prev_in_ael
        while e is not None:
            if e.vertex_top is vertex_max:
                if _v_is_hot(horz) and _v_is_joined(e):
                    _v_split(e, e.top, sc)
                if _v_is_hot(horz):
                    while horz.vertex_top is not vertex_max:
                        _v_add_outpt(horz, _BIVec2(horz.top.x, horz.top.y), sc)
                        _v_update_edge_into_ael(sc, horz)
                    if is_ltr:
                        _v_add_local_max_poly(horz, e, _BIVec2(horz.top.x, horz.top.y), sc)
                    else:
                        _v_add_local_max_poly(e, horz, _BIVec2(horz.top.x, horz.top.y), sc)
                _v_delete_from_ael(sc, e)
                _v_delete_from_ael(sc, horz)
                return
            if vertex_max is not horz.vertex_top:
                if (is_ltr and e.curr_x > horz_right) or (not is_ltr and e.curr_x < horz_left):
                    break
                if e.curr_x == horz.top.x and not _v_is_horizontal(e):
                    pt2 = _v_next_vertex(horz).pt
                    if is_ltr:
                        if _v_top_x(e, pt2.y) >= pt2.x:
                            break
                    else:
                        if _v_top_x(e, pt2.y) <= pt2.x:
                            break
            pt = _BIVec2(e.curr_x, horz.bot.y)
            if is_ltr:
                _v_intersect_edges(horz, e, pt, sc, cliptype)
                _v_swap_positions_in_ael(sc, horz, e)
                _v_check_join_left(e, pt, sc)
                horz.curr_x = e.curr_x
                e = horz.next_in_ael
            else:
                _v_intersect_edges(e, horz, pt, sc, cliptype)
                _v_swap_positions_in_ael(sc, e, horz)
                _v_check_join_right(e, pt, sc)
                horz.curr_x = e.curr_x
                e = horz.prev_in_ael
            if horz.outrec is not None:
                _v_add_trial_horz_join(sc, _v_get_last_op(horz))
        if _v_next_vertex(horz).pt.y != horz.top.y:
            break
        if _v_is_hot(horz):
            _v_add_outpt(horz, _BIVec2(horz.top.x, horz.top.y), sc)
        _v_update_edge_into_ael(sc, horz)
        is_ltr, horz_left, horz_right = _v_reset_horz_dir(horz, vertex_max)
    if _v_is_hot(horz):
        op = _v_add_outpt(horz, _BIVec2(horz.top.x, horz.top.y), sc)
        _v_add_trial_horz_join(sc, op)
    _v_update_edge_into_ael(sc, horz)


# ═══════════════════════════════════════════════════════════════════════════
# Horizontal joins
# ═══════════════════════════════════════════════════════════════════════════

def _v_dup_outpt(op: _VOutPt, after: _VOutPt, sc: _VattiScratch) -> _VOutPt:
    r = sc.new_outpt(op.pt, op.outrec)
    if after:
        r.next = op.next
        r.next.prev = r
        r.prev = op
        op.next = r
    else:
        r.prev = op.prev
        r.prev.next = r
        r.next = op
        op.prev = r
    return r

def _v_convert_horz_segs_to_joins(sc: _VattiScratch) -> None:
    valid = 0
    for hs in sc.horz_seg_list:
        op = hs.left_op
        outrec = op.outrec
        while outrec is not None and outrec.pts is None:
            outrec = outrec.owner
        if outrec is None:
            hs.right_op = None
            continue
        has_edges = outrec.front_edge is not None
        cy = op.pt.y
        opP = op
        opN = op
        if has_edges:
            opA = outrec.pts
            opZ = opA.next
            while opP is not opZ and opP.prev.pt.y == cy:
                opP = opP.prev
            while opN is not opA and opN.next.pt.y == cy:
                opN = opN.next
        else:
            while opP.prev is not opN and opP.prev.pt.y == cy:
                opP = opP.prev
            while opN.next is not opP and opN.next.pt.y == cy:
                opN = opN.next
        if opP.pt.x == opN.pt.x:
            hs.right_op = None
            continue
        if opP.pt.x < opN.pt.x:
            hs.left_op = opP
            hs.right_op = opN
            hs.left_to_right = True
        else:
            hs.left_op = opN
            hs.right_op = opP
            hs.left_to_right = False
        if hs.left_op.horz is not None:
            hs.right_op = None
            continue
        hs.left_op.horz = hs
        valid += 1
    if valid < 2:
        return
    sc.horz_seg_list.sort(key=lambda hs: (-1 if hs.right_op is None else 0, -hs.left_op.pt.x if hs.right_op is not None else 0))
    j = valid
    for i in range(j - 1):
        hs1 = sc.horz_seg_list[i]
        if hs1.right_op is None:
            continue
        for k in range(i + 1, j):
            hs2 = sc.horz_seg_list[k]
            if hs2.right_op is None:
                continue
            if hs2.left_op.pt.x >= hs1.right_op.pt.x or hs2.left_to_right == hs1.left_to_right or hs2.right_op.pt.x <= hs1.left_op.pt.x:
                continue
            cy = hs1.left_op.pt.y
            if hs1.left_to_right:
                while hs1.left_op.next.pt.y == cy and hs1.left_op.next.pt.x <= hs2.left_op.pt.x:
                    hs1.left_op = hs1.left_op.next
                while hs2.left_op.prev.pt.y == cy and hs2.left_op.prev.pt.x <= hs1.left_op.pt.x:
                    hs2.left_op = hs2.left_op.prev
                sc.horz_join_list.append(_VHorzJoin(_v_dup_outpt(hs1.left_op, True, sc), _v_dup_outpt(hs2.left_op, False, sc)))
            else:
                while hs1.left_op.prev.pt.y == cy and hs1.left_op.prev.pt.x <= hs2.left_op.pt.x:
                    hs1.left_op = hs1.left_op.prev
                while hs2.left_op.next.pt.y == cy and hs2.left_op.next.pt.x <= hs1.left_op.pt.x:
                    hs2.left_op = hs2.left_op.next
                sc.horz_join_list.append(_VHorzJoin(_v_dup_outpt(hs2.left_op, True, sc), _v_dup_outpt(hs1.left_op, False, sc)))

def _v_fix_outrec_pts(outrec: _VOutRec) -> None:
    op = outrec.pts
    while True:
        op.outrec = outrec
        op = op.next
        if op is outrec.pts:
            break

def _v_process_horz_joins(sc: _VattiScratch) -> None:
    for j in sc.horz_join_list:
        or1 = j.op1.outrec
        while or1 is not None and or1.pts is None:
            or1 = or1.owner
        or2 = j.op2.outrec
        while or2 is not None and or2.pts is None:
            or2 = or2.owner
        op1b = j.op1.next
        op2b = j.op2.prev
        j.op1.next = j.op2
        j.op2.prev = j.op1
        op1b.prev = op2b
        op2b.next = op1b
        if or1 is or2:
            or2 = sc.new_outrec()
            or2.pts = op1b
            _v_fix_outrec_pts(or2)
            if or1.pts.outrec is or2:
                or1.pts = j.op1
                or1.pts.outrec = or1
            or2.owner = or1
        else:
            or2.pts = None
            or2.owner = or1


# ═══════════════════════════════════════════════════════════════════════════
# Intersection detection
# ═══════════════════════════════════════════════════════════════════════════

def _v_adjust_curr_x_copy_to_sel(sc: _VattiScratch, top_y: int) -> None:
    e = sc.actives
    sc.sel = e
    while e is not None:
        e.prev_in_sel = e.prev_in_ael
        e.next_in_sel = e.next_in_ael
        e.jump = e.next_in_sel
        if e.join_with == _JW_LEFT:
            e.curr_x = e.prev_in_ael.curr_x
        else:
            e.curr_x = _v_top_x(e, top_y)
        e = e.next_in_ael

def _v_extract_from_sel(ae: _VActive) -> _VActive | None:
    res = ae.next_in_sel
    if res is not None:
        res.prev_in_sel = ae.prev_in_sel
    ae.prev_in_sel.next_in_sel = res
    return res

def _v_insert1_before2_in_sel(a1: _VActive, a2: _VActive) -> None:
    a1.prev_in_sel = a2.prev_in_sel
    if a1.prev_in_sel is not None:
        a1.prev_in_sel.next_in_sel = a1
    a1.next_in_sel = a2
    a2.prev_in_sel = a1

def _v_add_new_isect_node(sc: _VattiScratch, e1: _VActive, e2: _VActive, top_y: int) -> None:
    ok, ip = _v_get_seg_isect_pt(e1.bot, e1.top, e2.bot, e2.top)
    if not ok:
        ip = _BIVec2(e1.curr_x, top_y)
    if ip.y > sc.bot_y or ip.y < top_y:
        ad1 = abs(e1.dx)
        ad2 = abs(e2.dx)
        if ad1 > 100 and ad2 > 100:
            ip = _v_closest_pt_on_seg(ip, e1.bot, e1.top) if ad1 > ad2 else _v_closest_pt_on_seg(ip, e2.bot, e2.top)
        elif ad1 > 100:
            ip = _v_closest_pt_on_seg(ip, e1.bot, e1.top)
        elif ad2 > 100:
            ip = _v_closest_pt_on_seg(ip, e2.bot, e2.top)
        else:
            if ip.y < top_y:
                ip.y = top_y
            else:
                ip.y = sc.bot_y
            ip.x = _v_top_x(e1, ip.y) if ad1 < ad2 else _v_top_x(e2, ip.y)
    sc.intersect_nodes.append(_VIntersectNode(ip, e1, e2))

def _v_build_intersect_list(sc: _VattiScratch, top_y: int) -> bool:
    if sc.actives is None or sc.actives.next_in_ael is None:
        return False
    _v_adjust_curr_x_copy_to_sel(sc, top_y)
    left = sc.sel
    while left is not None and left.jump is not None:
        prev_base = None
        while left is not None and left.jump is not None:
            curr_base = left
            right = left.jump
            l_end = right
            r_end = right.jump
            left.jump = r_end
            while left is not l_end and right is not r_end:
                if right.curr_x < left.curr_x:
                    tmp = right.prev_in_sel
                    while True:
                        _v_add_new_isect_node(sc, tmp, right, top_y)
                        if tmp is left:
                            break
                        tmp = tmp.prev_in_sel
                    tmp = right
                    right = _v_extract_from_sel(tmp)
                    l_end = right
                    _v_insert1_before2_in_sel(tmp, left)
                    if left is curr_base:
                        curr_base = tmp
                        curr_base.jump = r_end
                        if prev_base is None:
                            sc.sel = curr_base
                        else:
                            prev_base.jump = curr_base
                else:
                    left = left.next_in_sel
            prev_base = curr_base
            left = r_end
        left = sc.sel
    return len(sc.intersect_nodes) > 0

def _v_process_intersect_list(sc: _VattiScratch, cliptype: int) -> None:
    sc.intersect_nodes.sort(key=lambda n: (-n.pt.y, n.pt.x))
    for i in range(len(sc.intersect_nodes)):
        node = sc.intersect_nodes[i]
        if not (node.edge1.next_in_ael is node.edge2 or node.edge1.prev_in_ael is node.edge2):
            for j_idx in range(i + 1, len(sc.intersect_nodes)):
                n2 = sc.intersect_nodes[j_idx]
                if n2.edge1.next_in_ael is n2.edge2 or n2.edge1.prev_in_ael is n2.edge2:
                    sc.intersect_nodes[i], sc.intersect_nodes[j_idx] = sc.intersect_nodes[j_idx], sc.intersect_nodes[i]
                    node = sc.intersect_nodes[i]
                    break
        _v_intersect_edges(node.edge1, node.edge2, node.pt, sc, cliptype)
        _v_swap_positions_in_ael(sc, node.edge1, node.edge2)
        node.edge1.curr_x = node.pt.x
        node.edge2.curr_x = node.pt.x
        _v_check_join_left(node.edge2, node.pt, sc, True)
        _v_check_join_right(node.edge1, node.pt, sc, True)


# ═══════════════════════════════════════════════════════════════════════════
# Local minima insertion
# ═══════════════════════════════════════════════════════════════════════════

def _v_insert_local_minima_into_ael(sc: _VattiScratch, bot_y: int, cliptype: int) -> None:
    while True:
        ok, lm = _v_pop_locmin(sc, bot_y)
        if not ok:
            break
        lb = _VActive()
        lb.bot = _BIVec2(lm.vertex.pt.x, lm.vertex.pt.y)
        lb.curr_x = lb.bot.x
        lb.wind_dx = -1
        lb.vertex_top = lm.vertex.prev
        lb.top = _BIVec2(lb.vertex_top.pt.x, lb.vertex_top.pt.y)
        lb.local_min = lm
        _v_set_dx(lb)

        rb = _VActive()
        rb.bot = _BIVec2(lm.vertex.pt.x, lm.vertex.pt.y)
        rb.curr_x = rb.bot.x
        rb.wind_dx = 1
        rb.vertex_top = lm.vertex.next
        rb.top = _BIVec2(rb.vertex_top.pt.x, rb.vertex_top.pt.y)
        rb.local_min = lm
        _v_set_dx(rb)

        if _v_is_horizontal(lb):
            if lb.dx == -_INF:
                lb, rb = rb, lb
        elif _v_is_horizontal(rb):
            if rb.dx == _INF:
                lb, rb = rb, lb
        elif lb.dx < rb.dx:
            lb, rb = rb, lb

        lb.is_left_bound = True
        _v_insert_left_edge(sc, lb)
        _v_set_wind_count(sc, lb)
        contributing = _v_is_contributing(lb, cliptype)

        rb.is_left_bound = False
        rb.wind_cnt = lb.wind_cnt
        rb.wind_cnt2 = lb.wind_cnt2
        _v_insert_right_edge(lb, rb)
        if contributing:
            _v_add_local_min_poly(lb, rb, _BIVec2(lb.bot.x, lb.bot.y), sc, True)
            if not _v_is_horizontal(lb):
                _v_check_join_left(lb, lb.bot, sc)
        while rb.next_in_ael is not None and _v_is_valid_ael_order(rb.next_in_ael, rb):
            _v_intersect_edges(rb, rb.next_in_ael, _BIVec2(rb.bot.x, rb.bot.y), sc, cliptype)
            _v_swap_positions_in_ael(sc, rb, rb.next_in_ael)
        if _v_is_horizontal(rb):
            _v_push_horz(sc, rb)
        else:
            _v_check_join_right(rb, rb.bot, sc)
            _v_insert_scanline(sc, rb.top.y)
        if _v_is_horizontal(lb):
            _v_push_horz(sc, lb)
        else:
            _v_insert_scanline(sc, lb.top.y)


# ═══════════════════════════════════════════════════════════════════════════
# Maxima
# ═══════════════════════════════════════════════════════════════════════════

def _v_do_maxima(e: _VActive, sc: _VattiScratch, cliptype: int) -> _VActive | None:
    prev_e = e.prev_in_ael
    next_e = e.next_in_ael
    max_pair = _v_get_maxima_pair(e)
    if max_pair is None:
        return next_e
    if _v_is_joined(e):
        _v_split(e, e.top, sc)
    if _v_is_joined(max_pair):
        _v_split(max_pair, max_pair.top, sc)
    while next_e is not max_pair:
        _v_intersect_edges(e, next_e, _BIVec2(e.top.x, e.top.y), sc, cliptype)
        _v_swap_positions_in_ael(sc, e, next_e)
        next_e = e.next_in_ael
    if _v_is_hot(e):
        _v_add_local_max_poly(e, max_pair, _BIVec2(e.top.x, e.top.y), sc)
    _v_delete_from_ael(sc, max_pair)
    _v_delete_from_ael(sc, e)
    return prev_e.next_in_ael if prev_e is not None else sc.actives


# ═══════════════════════════════════════════════════════════════════════════
# Top of scanbeam
# ═══════════════════════════════════════════════════════════════════════════

def _v_do_top_of_scanbeam(sc: _VattiScratch, y: int, cliptype: int) -> None:
    sc.sel = None
    e = sc.actives
    while e is not None:
        if e.top.y == y:
            e.curr_x = e.top.x
            if _v_is_maxima_e(e):
                e = _v_do_maxima(e, sc, cliptype)
                continue
            if _v_is_hot(e):
                _v_add_outpt(e, _BIVec2(e.top.x, e.top.y), sc)
            _v_update_edge_into_ael(sc, e)
            if _v_is_horizontal(e):
                _v_push_horz(sc, e)
        else:
            e.curr_x = _v_top_x(e, y)
        e = e.next_in_ael


# ═══════════════════════════════════════════════════════════════════════════
# Collinear cleanup and self intersections
# ═══════════════════════════════════════════════════════════════════════════

def _v_dispose_outpt(op: _VOutPt) -> _VOutPt | None:
    r = op.next
    op.prev.next = op.next
    op.next.prev = op.prev
    return r

def _v_do_split_op(sc: _VattiScratch, outrec: _VOutRec, splitOp: _VOutPt) -> None:
    prevOp = splitOp.prev
    nnOp = splitOp.next.next
    outrec.pts = prevOp
    ok, ip = _v_get_seg_isect_pt(prevOp.pt, splitOp.pt, splitOp.next.pt, nnOp.pt)
    area1 = _v_area_outpt(outrec.pts)
    if abs(area1) < 2:
        outrec.pts = None
        return
    area2 = _v_area_tri(ip, splitOp.pt, splitOp.next.pt)
    absA2 = abs(area2)
    if ip == prevOp.pt or ip == nnOp.pt:
        nnOp.prev = prevOp
        prevOp.next = nnOp
    else:
        nop = sc.new_outpt(ip, prevOp.outrec)
        nop.prev = prevOp
        nop.next = nnOp
        nnOp.prev = nop
        prevOp.next = nop
    if absA2 >= 1 and (absA2 > abs(area1) or (area2 > 0) == (area1 > 0)):
        nr = sc.new_outrec()
        nr.owner = outrec.owner
        splitOp.outrec = nr
        splitOp.next.outrec = nr
        nop2 = sc.new_outpt(ip, nr)
        nop2.prev = splitOp.next
        nop2.next = splitOp
        nr.pts = nop2
        splitOp.prev = nop2
        splitOp.next.next = nop2

def _v_fix_self_intersects(sc: _VattiScratch, outrec: _VOutRec) -> None:
    op2 = outrec.pts
    while True:
        if op2.prev is op2.next.next:
            break
        if _v_segs_intersect(op2.prev.pt, op2.pt, op2.next.pt, op2.next.next.pt):
            if op2 is outrec.pts or op2.next is outrec.pts:
                outrec.pts = outrec.pts.prev
            _v_do_split_op(sc, outrec, op2)
            if outrec.pts is None:
                break
            op2 = outrec.pts
            continue
        op2 = op2.next
        if op2 is outrec.pts:
            break

def _v_clean_collinear(sc: _VattiScratch, outrec: _VOutRec) -> None:
    while outrec is not None and outrec.pts is None:
        outrec = outrec.owner
    if outrec is None:
        return
    if not _v_valid_closed(outrec.pts):
        outrec.pts = None
        return
    startOp = outrec.pts
    op2 = startOp
    while True:
        if _v_is_collinear(op2.prev.pt, op2.pt, op2.next.pt) and \
           (op2.pt == op2.prev.pt or op2.pt == op2.next.pt or _v_dot_product(op2.prev.pt, op2.pt, op2.next.pt) < 0):
            if op2 is outrec.pts:
                outrec.pts = op2.prev
            op2 = _v_dispose_outpt(op2)
            if not _v_valid_closed(op2):
                outrec.pts = None
                return
            startOp = op2
            continue
        op2 = op2.next
        if op2 is startOp:
            break
    _v_fix_self_intersects(sc, outrec)


# ═══════════════════════════════════════════════════════════════════════════
# Sweep
# ═══════════════════════════════════════════════════════════════════════════

def _v_execute_internal(sc: _VattiScratch, cliptype: int) -> bool:
    sc.locmin_list.sort(key=lambda lm: (-lm.vertex.pt.y, lm.vertex.pt.x))
    for lm in sc.locmin_list:
        _v_insert_scanline(sc, lm.vertex.pt.y)
    sc.locmin_idx = 0

    ok, y = _v_pop_scanline(sc)
    if not ok:
        return True
    while sc.succeeded:
        _v_insert_local_minima_into_ael(sc, y, cliptype)
        while True:
            ok_h, e = _v_pop_horz(sc)
            if not ok_h:
                break
            _v_do_horizontal(e, sc, cliptype)
        if sc.horz_seg_list:
            _v_convert_horz_segs_to_joins(sc)
            sc.horz_seg_list.clear()
        sc.bot_y = y
        ok, y = _v_pop_scanline(sc)
        if not ok:
            break
        if sc.succeeded and _v_build_intersect_list(sc, y):
            _v_process_intersect_list(sc, cliptype)
            sc.intersect_nodes.clear()
        _v_do_top_of_scanbeam(sc, y, cliptype)
        while True:
            ok_h, e = _v_pop_horz(sc)
            if not ok_h:
                break
            _v_do_horizontal(e, sc, cliptype)
    if sc.succeeded:
        _v_process_horz_joins(sc)
    return sc.succeeded


# ═══════════════════════════════════════════════════════════════════════════
# Fast paths and extraction
# ═══════════════════════════════════════════════════════════════════════════

def _v_strip_closing(c: list[float], n: int) -> int:
    """Drops a closing point that repeats the first one."""
    if n < 2:
        return n
    dx = c[(n - 1) * 3] - c[0]
    dy = c[(n - 1) * 3 + 1] - c[1]
    if dx * dx + dy * dy < 1e-20:
        n -= 1
    return n


def _v_bool_scale(ca: list[float], na: int, cb: list[float], nb: int) -> float:
    """Integer scale so that (max_coord * scale)^2 fits in int64."""
    max_coord = 0.0
    for i in range(na):
        max_coord = max(max_coord, abs(ca[i * 3]), abs(ca[i * 3 + 1]))
    for i in range(nb):
        max_coord = max(max_coord, abs(cb[i * 3]), abs(cb[i * 3 + 1]))
    if max_coord < 1e-12:
        max_coord = 1.0
    return math.floor(math.sqrt(float(2**63 - 1)) / (2.0 * max_coord))


def _v_select(a: Polyline, b: Polyline, a_in_b: bool, b_in_a: bool, clip_type: int) -> list[Polyline]:
    """Result of a boolean when one polygon contains the other or they are disjoint."""
    if clip_type == 0:
        if a_in_b:
            return [a]
        if b_in_a:
            return [b]
        return []
    if clip_type == 1:
        if a_in_b:
            return [b]
        if b_in_a:
            return [a]
        return [a, b]
    if a_in_b:
        return []
    return [a]


def _v_bounds(v: list[_BIVec2]) -> tuple[int, int, int, int]:
    minX = maxX = v[0].x
    minY = maxY = v[0].y
    for i in range(1, len(v)):
        if v[i].x < minX:
            minX = v[i].x
        elif v[i].x > maxX:
            maxX = v[i].x
        if v[i].y < minY:
            minY = v[i].y
        elif v[i].y > maxY:
            maxY = v[i].y
    return minX, maxX, minY, maxY


def _v_any_cross(va: list[_BIVec2], vb: list[_BIVec2]) -> bool:
    na = len(va)
    nb = len(vb)
    for i in range(na):
        a1 = va[i]
        a2 = va[(i + 1) % na]
        axmin = min(a1.x, a2.x)
        axmax = max(a1.x, a2.x)
        aymin = min(a1.y, a2.y)
        aymax = max(a1.y, a2.y)
        for j in range(nb):
            b1 = vb[j]
            b2 = vb[(j + 1) % nb]
            if max(b1.x, b2.x) < axmin or min(b1.x, b2.x) > axmax or max(b1.y, b2.y) < aymin or min(b1.y, b2.y) > aymax:
                continue
            if _v_segs_intersect(a1, a2, b1, b2):
                return True
    return False


def _v_centroid(v: list[_BIVec2]) -> _BIVec2:
    c = _BIVec2(0, 0)
    for i in range(len(v)):
        c.x += v[i].x
        c.y += v[i].y
    c.x = int(c.x / len(v))
    c.y = int(c.y / len(v))
    return c


def _v_contains(va: list[_BIVec2], vb: list[_BIVec2]) -> tuple[bool, bool]:
    """Containment of non-crossing polygons: vertex test, validated by the centroid, then the centroid
    nudged by one unit when it sits on the boundary."""
    a_in_b = _pip_i(va[0], vb)
    b_in_a = _pip_i(vb[0], va)
    ca_cen = _v_centroid(va)
    cb_cen = _v_centroid(vb)
    if a_in_b and not _pip_i(ca_cen, vb):
        a_in_b = False
    if b_in_a and not _pip_i(cb_cen, va):
        b_in_a = False
    if a_in_b or b_in_a:
        return a_in_b, b_in_a
    a_in_b = _pip_i(ca_cen, vb)
    b_in_a = _pip_i(cb_cen, va)
    if a_in_b or b_in_a:
        return a_in_b, b_in_a
    a_in_b = _pip_i(_BIVec2(ca_cen.x + 1, ca_cen.y + 1), vb)
    b_in_a = _pip_i(_BIVec2(cb_cen.x + 1, cb_cen.y + 1), va)
    return a_in_b, b_in_a


def _v_ring_start(sc: _VattiScratch, outrec: _VOutRec) -> _VOutPt | None:
    """First output point of a finished ring, or None when the ring is degenerate."""
    if outrec.pts is None:
        return None
    _v_clean_collinear(sc, outrec)
    if outrec.pts is None:
        return None
    op = outrec.pts
    if op.next is op or op.next is op.prev or _v_very_small_tri(op):
        return None
    return op


def _v_extract(sc: _VattiScratch, inv_scale: float) -> list[Polyline]:
    out = []
    for outrec in sc.outrec_list:
        op = _v_ring_start(sc, outrec)
        if op is None:
            continue
        coords = []
        o = op.next
        last = o.pt
        coords.extend([last.x * inv_scale, last.y * inv_scale, 0.0])
        o = o.next
        while o is not op.next:
            if o.pt != last:
                last = o.pt
                coords.extend([last.x * inv_scale, last.y * inv_scale, 0.0])
            o = o.next
        if len(coords) < 9:
            continue
        out.append(Polyline.from_coords(coords))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Open subject against closed clip
# ═══════════════════════════════════════════════════════════════════════════

def _v_point_in_poly(cc: list[float], nc: int, px: float, py: float) -> bool:
    """Even-odd ray cast of (px, py) against the first nc points of cc."""
    inside = False
    j = nc - 1
    for i in range(nc):
        xi = cc[i * 3]
        yi = cc[i * 3 + 1]
        xj = cc[j * 3]
        yj = cc[j * 3 + 1]
        j = i
        if (yi > py) == (yj > py):
            continue
        xint = xj + (py - yj) * (xi - xj) / (yi - yj)
        if px < xint:
            inside = not inside
    return inside


def _v_crossings(cc: list[float], nc: int, ax: float, ay: float, dx: float, dy: float) -> list[float]:
    """Sorted parameters in (0, 1] where segment (a, b) crosses an edge of the clip."""
    ts = []
    j = nc - 1
    for i in range(nc):
        ex = cc[i * 3] - cc[j * 3]
        ey = cc[i * 3 + 1] - cc[j * 3 + 1]
        rx = cc[j * 3] - ax
        ry = cc[j * 3 + 1] - ay
        j = i
        denom = dy * ex - dx * ey
        if abs(denom) < 1e-18:
            continue
        t = (ry * ex - rx * ey) / denom
        u = (ry * dx - rx * dy) / denom
        if t > 1e-12 and t <= 1.0 + 1e-12 and u >= -1e-9 and u <= 1.0 + 1e-9:
            ts.append(min(max(t, 0.0), 1.0))
    ts.sort()
    return ts


def _v_push_xy(cur: list[float], x: float, y: float) -> None:
    n = len(cur)
    if n >= 3 and abs(cur[n - 3] - x) < 1e-9 and abs(cur[n - 2] - y) < 1e-9:
        return
    cur.append(x)
    cur.append(y)
    cur.append(0.0)


def _v_flush(cur: list[float], result: list[Polyline]) -> None:
    if len(cur) >= 6:
        result.append(Polyline.from_coords(list(cur)))
    cur.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Boolean operations
# ═══════════════════════════════════════════════════════════════════════════

class BooleanPolyline:
    @staticmethod
    def compute(a: Polyline, b: Polyline, clip_type: int) -> list[Polyline]:
        """Vatti boolean of two closed planar polylines; clip_type 0 intersection, 1 union, 2 a minus b."""
        ca = a.coords
        cb = b.coords
        na = _v_strip_closing(ca, len(ca) // 3)
        nb = _v_strip_closing(cb, len(cb) // 3)
        if na < 3 or nb < 3:
            return []
        bool_scale = _v_bool_scale(ca, na, cb, nb)
        sc = _VattiScratch()
        if na * nb <= 400:
            va = []
            vb = []
            for i in range(na):
                va.append(_BIVec2(round(ca[i * 3] * bool_scale), round(ca[i * 3 + 1] * bool_scale)))
            for i in range(nb):
                vb.append(_BIVec2(round(cb[i * 3] * bool_scale), round(cb[i * 3 + 1] * bool_scale)))
            aMinX, aMaxX, aMinY, aMaxY = _v_bounds(va)
            bMinX, bMaxX, bMinY, bMaxY = _v_bounds(vb)
            if aMaxX < bMinX or bMaxX < aMinX or aMaxY < bMinY or bMaxY < aMinY:
                return _v_select(a, b, _pip_i(va[0], vb), _pip_i(vb[0], va), clip_type)
            if not _v_any_cross(va, vb):
                a_in_b, b_in_a = _v_contains(va, vb)
                return _v_select(a, b, a_in_b, b_in_a, clip_type)
            _v_add_path(va, na, 0, sc)
            _v_add_path(vb, nb, 1, sc)
        else:
            va_head, aMinX, aMaxX, aMinY, aMaxY = _v_add_path_from_doubles(ca, na, 0, bool_scale, sc)
            vb_head, bMinX, bMaxX, bMinY, bMaxY = _v_add_path_from_doubles(cb, nb, 1, bool_scale, sc)
            if va_head is None or vb_head is None:
                return []
            if aMaxX < bMinX or bMaxX < aMinX or aMaxY < bMinY or bMaxY < aMinY:
                return _v_select(a, b, _pip_vertex(va_head.pt, vb_head), _pip_vertex(vb_head.pt, va_head), clip_type)
        if not _v_execute_internal(sc, clip_type):
            return []
        return _v_extract(sc, 1.0 / bool_scale)

    @staticmethod
    def compute_count(a: Polyline, b: Polyline, clip_type: int) -> int:
        """Number of output points of compute, without building polylines."""
        ca = a.coords
        cb = b.coords
        na = _v_strip_closing(ca, len(ca) // 3)
        nb = _v_strip_closing(cb, len(cb) // 3)
        if na < 3 or nb < 3:
            return 0
        bool_scale = _v_bool_scale(ca, na, cb, nb)
        sc = _VattiScratch()
        va_head, aMinX, aMaxX, aMinY, aMaxY = _v_add_path_from_doubles(ca, na, 0, bool_scale, sc)
        vb_head, bMinX, bMaxX, bMinY, bMaxY = _v_add_path_from_doubles(cb, nb, 1, bool_scale, sc)
        if va_head is None or vb_head is None:
            return 0
        if aMaxX < bMinX or bMaxX < aMinX or aMaxY < bMinY or bMaxY < aMinY:
            return 0
        if not _v_execute_internal(sc, clip_type):
            return 0
        total = 0
        for outrec in sc.outrec_list:
            op = _v_ring_start(sc, outrec)
            if op is None:
                continue
            o = op
            while True:
                total += 1
                o = o.next
                if o is op:
                    break
        return total

    @staticmethod
    def compute_raw(a_xy: list[float], na: int, b_xy: list[float], nb: int, clip_type: int, out_xy: list[float], max_out: int) -> int:
        """compute on flat xy arrays; writes up to max_out result points to out_xy and returns the total."""
        a = Polyline.from_coords([0.0] * (na * 3))
        b = Polyline.from_coords([0.0] * (nb * 3))
        for i in range(na):
            a.coords[i * 3] = a_xy[i * 2]
            a.coords[i * 3 + 1] = a_xy[i * 2 + 1]
        for i in range(nb):
            b.coords[i * 3] = b_xy[i * 2]
            b.coords[i * 3 + 1] = b_xy[i * 2 + 1]
        result = BooleanPolyline.compute(a, b, clip_type)
        total = 0
        for r in range(len(result)):
            c = result[r].coords
            for i in range(len(c) // 3):
                if total < max_out:
                    out_xy[total * 2] = c[i * 3]
                    out_xy[total * 2 + 1] = c[i * 3 + 1]
                total += 1
        return total

    @staticmethod
    def clip_open_against_closed(open_subject: Polyline, closed_clip: Polyline) -> list[Polyline]:
        """Pieces of an open polyline that lie inside a closed clip polygon, in the xy plane."""
        result = []
        cs = open_subject.coords
        cc = closed_clip.coords
        ns = len(cs) // 3
        nc = _v_strip_closing(cc, len(cc) // 3)
        if ns < 2 or nc < 3:
            return result
        cur = []
        if _v_point_in_poly(cc, nc, cs[0], cs[1]):
            _v_push_xy(cur, cs[0], cs[1])
        for si in range(ns - 1):
            ax = cs[si * 3]
            ay = cs[si * 3 + 1]
            bx = cs[(si + 1) * 3]
            by = cs[(si + 1) * 3 + 1]
            dx = bx - ax
            dy = by - ay
            ts = _v_crossings(cc, nc, ax, ay, dx, dy)
            prev_t = 0.0
            for k in range(len(ts)):
                t = ts[k]
                if t - prev_t < 1e-12:
                    prev_t = t
                    continue
                mid_t = 0.5 * (prev_t + t)
                _v_push_xy(cur, ax + dx * t, ay + dy * t)
                if _v_point_in_poly(cc, nc, ax + dx * mid_t, ay + dy * mid_t):
                    _v_flush(cur, result)
                prev_t = t
            if prev_t >= 1.0 - 1e-12:
                continue
            mid_t = 0.5 * (prev_t + 1.0)
            if _v_point_in_poly(cc, nc, ax + dx * mid_t, ay + dy * mid_t):
                _v_push_xy(cur, bx, by)
        _v_flush(cur, result)
        return result
