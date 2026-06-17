import heapq
import math
from .polyline import Polyline

VF_None = 0
VF_LocalMax = 4
VF_LocalMin = 8

JW_None = 0
JW_Left = 1
JW_Right = 2


class BIVec2:
    __slots__ = ("x", "y")
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __eq__(self, o):
        return self.x == o.x and self.y == o.y
    def __ne__(self, o):
        return self.x != o.x or self.y != o.y


class VVertex:
    __slots__ = ("pt", "next", "prev", "flags")
    def __init__(self):
        self.pt = BIVec2()
        self.next = None
        self.prev = None
        self.flags = VF_None


class VLocalMinima:
    __slots__ = ("vertex", "polytype")
    def __init__(self, vertex, polytype):
        self.vertex = vertex
        self.polytype = polytype


class VOutPt:
    __slots__ = ("pt", "next", "prev", "outrec", "horz")
    def __init__(self, pt, outrec):
        self.pt = pt
        self.outrec = outrec
        self.next = self
        self.prev = self
        self.horz = None


class VOutRec:
    __slots__ = ("idx", "front_edge", "back_edge", "pts", "owner")
    def __init__(self, idx=0):
        self.idx = idx
        self.front_edge = None
        self.back_edge = None
        self.pts = None
        self.owner = None


class VActive:
    __slots__ = ("bot", "top", "curr_x", "dx", "wind_dx", "wind_cnt", "wind_cnt2",
                 "outrec", "prev_in_ael", "next_in_ael", "prev_in_sel", "next_in_sel",
                 "jump", "vertex_top", "local_min", "is_left_bound", "join_with")
    def __init__(self):
        self.bot = BIVec2()
        self.top = BIVec2()
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
        self.join_with = JW_None


class VIntersectNode:
    __slots__ = ("pt", "edge1", "edge2")
    def __init__(self, pt, edge1, edge2):
        self.pt = pt
        self.edge1 = edge1
        self.edge2 = edge2


class VHorzSeg:
    __slots__ = ("left_op", "right_op", "left_to_right")
    def __init__(self, left_op):
        self.left_op = left_op
        self.right_op = None
        self.left_to_right = True


class VHorzJoin:
    __slots__ = ("op1", "op2")
    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2


class ScanlineHeap:
    def __init__(self):
        self._heap = []
    def clear(self):
        self._heap.clear()
    def empty(self):
        return len(self._heap) == 0
    def push(self, y):
        heapq.heappush(self._heap, -y)
    def top(self):
        return -self._heap[0]
    def pop(self):
        heapq.heappop(self._heap)


class VattiScratch:
    def __init__(self):
        self.locmin_list = []
        self.intersect_nodes = []
        self.horz_seg_list = []
        self.horz_join_list = []
        self.outrec_list = []
        self.scanline_list = ScanlineHeap()
        self.actives = None
        self.sel = None
        self.bot_y = 0
        self.locmin_idx = 0
        self.succeeded = True

    def reset(self):
        self.locmin_list.clear()
        self.intersect_nodes.clear()
        self.horz_seg_list.clear()
        self.horz_join_list.clear()
        self.outrec_list.clear()
        self.scanline_list.clear()
        self.actives = None
        self.sel = None
        self.bot_y = 0
        self.locmin_idx = 0
        self.succeeded = True

    def new_outpt(self, pt, rec):
        o = VOutPt(BIVec2(pt.x, pt.y), rec)
        return o

    def new_outrec(self):
        r = VOutRec(len(self.outrec_list))
        self.outrec_list.append(r)
        return r


# -- Geometry helpers --

_INF = float("inf")

def v_get_dx(p1, p2):
    dy = float(p2.y - p1.y)
    if dy != 0:
        return float(p2.x - p1.x) / dy
    return -_INF if p2.x > p1.x else _INF

def v_top_x(ae, y):
    if y == ae.top.y or ae.top.x == ae.bot.x:
        return ae.top.x
    if y == ae.bot.y:
        return ae.bot.x
    return ae.bot.x + round(ae.dx * float(y - ae.bot.y))

def v_is_horizontal(e):
    return e.top.y == e.bot.y

def v_is_hot(e):
    return e.outrec is not None

def v_is_maxima_v(v):
    return (v.flags & VF_LocalMax) != 0

def v_is_maxima_e(e):
    return v_is_maxima_v(e.vertex_top)

def v_is_front(e):
    return e is e.outrec.front_edge

def v_is_joined(e):
    return e.join_with != JW_None

def v_same_polytype(a, b):
    return a.local_min.polytype == b.local_min.polytype

def v_polytype(e):
    return e.local_min.polytype

def v_set_dx(e):
    e.dx = v_get_dx(e.bot, e.top)

def v_next_vertex(e):
    return e.vertex_top.next if e.wind_dx > 0 else e.vertex_top.prev

def v_prev_prev_vertex(ae):
    return ae.vertex_top.prev.prev if ae.wind_dx > 0 else ae.vertex_top.next.next

def v_cross_product(p1, p2, p3):
    return float(p2.x - p1.x) * float(p3.y - p2.y) - float(p2.y - p1.y) * float(p3.x - p2.x)

def v_dot_product(p1, p2, p3):
    return float(p2.x - p1.x) * float(p3.x - p2.x) + float(p2.y - p1.y) * float(p3.y - p2.y)

def v_products_equal(a, b, c, d):
    return a * b == c * d

def v_is_collinear(p1, shared, p2):
    return v_products_equal(shared.x - p1.x, p2.y - shared.y, shared.y - p1.y, p2.x - shared.x)

def v_perpendic_dist_sq(pt, l1, l2):
    a = float(pt.x - l1.x)
    b = float(pt.y - l1.y)
    c = float(l2.x - l1.x)
    d = float(l2.y - l1.y)
    if c == 0 and d == 0:
        return 0.0
    e = a * d - c * b
    return (e * e) / (c * c + d * d)

def v_get_seg_isect_pt(a, b, c, d):
    dx1 = float(b.x - a.x)
    dy1 = float(b.y - a.y)
    dx2 = float(d.x - c.x)
    dy2 = float(d.y - c.y)
    det = dy1 * dx2 - dy2 * dx1
    if det == 0.0:
        return False, BIVec2()
    t = (float(a.x - c.x) * dy2 - float(a.y - c.y) * dx2) / det
    if t <= 0.0:
        ip = BIVec2(a.x, a.y)
    elif t >= 1.0:
        ip = BIVec2(b.x, b.y)
    else:
        ip = BIVec2(a.x + round(t * dx1), a.y + round(t * dy1))
    return True, ip

def v_closest_pt_on_seg(pt, s1, s2):
    if s1 == s2:
        return BIVec2(s1.x, s1.y)
    dx = float(s2.x - s1.x)
    dy = float(s2.y - s1.y)
    q = (float(pt.x - s1.x) * dx + float(pt.y - s1.y) * dy) / (dx * dx + dy * dy)
    if q < 0:
        q = 0.0
    elif q > 1:
        q = 1.0
    return BIVec2(s1.x + round(q * dx), s1.y + round(q * dy))

def v_segs_intersect(a, b, c, d):
    def sign(v):
        if not v:
            return 0
        return 1 if v > 0 else -1
    return (sign(v_cross_product(a, c, d)) * sign(v_cross_product(b, c, d)) < 0) and \
           (sign(v_cross_product(c, a, b)) * sign(v_cross_product(d, a, b)) < 0)

def v_area_outpt(op):
    r = 0.0
    o = op
    while True:
        r += float(o.prev.pt.y + o.pt.y) * float(o.prev.pt.x - o.pt.x)
        o = o.next
        if o is op:
            break
    return r * 0.5

def v_area_tri(p1, p2, p3):
    return float(p3.y + p1.y) * float(p3.x - p1.x) + float(p1.y + p2.y) * float(p1.x - p2.x) + float(p2.y + p3.y) * float(p2.x - p3.x)

def v_pts_close(a, b):
    return abs(a.x - b.x) < 2 and abs(a.y - b.y) < 2

def v_very_small_tri(op):
    return op.next.next is op.prev and (v_pts_close(op.prev.pt, op.next.pt) or v_pts_close(op.pt, op.next.pt) or v_pts_close(op.pt, op.prev.pt))

def v_valid_closed(op):
    return op is not None and op.next is not op and op.next is not op.prev and not v_very_small_tri(op)

def pip_i(pt, poly):
    winding = 0
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if a.y <= pt.y:
            if b.y > pt.y and ((b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x)) > 0:
                winding += 1
        else:
            if b.y <= pt.y and ((b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x)) < 0:
                winding -= 1
    return winding != 0

def pip_vertex(pt, head):
    winding = 0
    v = head
    while True:
        a = v.pt
        b = v.next.pt
        if a.y <= pt.y:
            if b.y > pt.y and ((b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x)) > 0:
                winding += 1
        else:
            if b.y <= pt.y and ((b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x)) < 0:
                winding -= 1
        v = v.next
        if v is head:
            break
    return winding != 0


# -- Vertex building + local minima detection --

def v_add_path_from_doubles(coords, n, polytype, sc, bool_scale):
    if n < 3:
        return None, 0, 0, 0, 0
    vertices = []
    pt0 = BIVec2(round(coords[0] * bool_scale), round(coords[1] * bool_scale))
    v0 = VVertex()
    v0.pt = pt0
    vertices.append(v0)
    minX = maxX = pt0.x
    minY = maxY = pt0.y
    prev_v = v0
    for i in range(1, n):
        px = round(coords[i * 3] * bool_scale)
        py = round(coords[i * 3 + 1] * bool_scale)
        pt = BIVec2(px, py)
        if pt == prev_v.pt:
            continue
        cv = VVertex()
        cv.pt = pt
        cv.prev = prev_v
        prev_v.next = cv
        prev_v = cv
        vertices.append(cv)
        if pt.x < minX:
            minX = pt.x
        elif pt.x > maxX:
            maxX = pt.x
        if pt.y < minY:
            minY = pt.y
        elif pt.y > maxY:
            maxY = pt.y
    cnt = len(vertices)
    if cnt < 3:
        return None, 0, 0, 0, 0
    if prev_v.pt == vertices[0].pt:
        prev_v = prev_v.prev
        cnt -= 1
    if cnt < 3:
        return None, 0, 0, 0, 0
    prev_v.next = vertices[0]
    vertices[0].prev = prev_v

    # Local minima detection
    pv = vertices[0].prev
    while pv is not vertices[0] and pv.pt.y == vertices[0].pt.y:
        pv = pv.prev
    if pv is vertices[0]:
        return vertices[0], minX, maxX, minY, maxY
    going_up = pv.pt.y > vertices[0].pt.y
    going_up0 = going_up
    pv = vertices[0]
    cv = vertices[0].next
    while cv is not vertices[0]:
        if cv.pt.y > pv.pt.y and going_up:
            pv.flags |= VF_LocalMax
            going_up = False
        elif cv.pt.y < pv.pt.y and not going_up:
            going_up = True
            pv.flags |= VF_LocalMin
            sc.locmin_list.append(VLocalMinima(pv, polytype))
        pv = cv
        cv = cv.next
    if going_up != going_up0:
        if going_up0:
            pv.flags |= VF_LocalMin
            sc.locmin_list.append(VLocalMinima(pv, polytype))
        else:
            pv.flags |= VF_LocalMax
    return vertices[0], minX, maxX, minY, maxY

def v_add_path(pts, n, polytype, sc):
    if n < 3:
        return
    vertices = []
    v0 = VVertex()
    v0.pt = BIVec2(pts[0].x, pts[0].y)
    vertices.append(v0)
    prev_v = v0
    for i in range(1, n):
        if pts[i] == prev_v.pt:
            continue
        cv = VVertex()
        cv.pt = BIVec2(pts[i].x, pts[i].y)
        cv.prev = prev_v
        prev_v.next = cv
        prev_v = cv
        vertices.append(cv)
    cnt = len(vertices)
    if cnt < 3:
        return
    if prev_v.pt == vertices[0].pt:
        prev_v = prev_v.prev
        cnt -= 1
    if cnt < 3:
        return
    prev_v.next = vertices[0]
    vertices[0].prev = prev_v

    pv = vertices[0].prev
    while pv is not vertices[0] and pv.pt.y == vertices[0].pt.y:
        pv = pv.prev
    if pv is vertices[0]:
        return
    going_up = pv.pt.y > vertices[0].pt.y
    going_up0 = going_up
    pv = vertices[0]
    cv = vertices[0].next
    while cv is not vertices[0]:
        if cv.pt.y > pv.pt.y and going_up:
            pv.flags |= VF_LocalMax
            going_up = False
        elif cv.pt.y < pv.pt.y and not going_up:
            going_up = True
            pv.flags |= VF_LocalMin
            sc.locmin_list.append(VLocalMinima(pv, polytype))
        pv = cv
        cv = cv.next
    if going_up != going_up0:
        if going_up0:
            pv.flags |= VF_LocalMin
            sc.locmin_list.append(VLocalMinima(pv, polytype))
        else:
            pv.flags |= VF_LocalMax


# -- AEL operations --

def v_get_maxima_pair(e):
    e2 = e.next_in_ael
    while e2 is not None:
        if e2.vertex_top is e.vertex_top:
            return e2
        e2 = e2.next_in_ael
    return None

def v_get_curr_y_maxima(e):
    r = e.vertex_top
    if e.wind_dx > 0:
        while r.next.pt.y == r.pt.y:
            r = r.next
    else:
        while r.prev.pt.y == r.pt.y:
            r = r.prev
    return r if v_is_maxima_v(r) else None

def v_get_prev_hot(e):
    p = e.prev_in_ael
    while p is not None and not v_is_hot(p):
        p = p.prev_in_ael
    return p

def v_is_valid_ael_order(resident, newcomer):
    if newcomer.curr_x != resident.curr_x:
        return newcomer.curr_x > resident.curr_x
    d = v_cross_product(resident.top, newcomer.bot, newcomer.top)
    if d != 0:
        return d < 0
    if not v_is_maxima_e(resident) and resident.top.y > newcomer.top.y:
        return v_cross_product(newcomer.bot, resident.top, v_next_vertex(resident).pt) <= 0
    if not v_is_maxima_e(newcomer) and newcomer.top.y > resident.top.y:
        return v_cross_product(newcomer.bot, newcomer.top, v_next_vertex(newcomer).pt) >= 0
    y = newcomer.bot.y
    if resident.bot.y != y or resident.local_min.vertex.pt.y != y:
        return newcomer.is_left_bound
    if resident.is_left_bound != newcomer.is_left_bound:
        return newcomer.is_left_bound
    if v_is_collinear(v_prev_prev_vertex(resident).pt, resident.bot, resident.top):
        return True
    return (v_cross_product(v_prev_prev_vertex(resident).pt, newcomer.bot, v_prev_prev_vertex(newcomer).pt) > 0) == newcomer.is_left_bound

def v_insert_left_edge(sc, e):
    if sc.actives is None:
        e.prev_in_ael = None
        e.next_in_ael = None
        sc.actives = e
    elif not v_is_valid_ael_order(sc.actives, e):
        e.prev_in_ael = None
        e.next_in_ael = sc.actives
        sc.actives.prev_in_ael = e
        sc.actives = e
    else:
        e2 = sc.actives
        while e2.next_in_ael is not None and v_is_valid_ael_order(e2.next_in_ael, e):
            e2 = e2.next_in_ael
        if e2.join_with == JW_Right:
            e2 = e2.next_in_ael
        if e2 is None:
            return
        e.next_in_ael = e2.next_in_ael
        if e2.next_in_ael is not None:
            e2.next_in_ael.prev_in_ael = e
        e.prev_in_ael = e2
        e2.next_in_ael = e

def v_insert_right_edge(e, e2):
    e2.next_in_ael = e.next_in_ael
    if e.next_in_ael is not None:
        e.next_in_ael.prev_in_ael = e2
    e2.prev_in_ael = e
    e.next_in_ael = e2

def v_swap_positions_in_ael(sc, e1, e2):
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

def v_delete_from_ael(sc, e):
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


# -- Scanline --

def v_insert_scanline(sc, y):
    sc.scanline_list.push(y)

def v_pop_scanline(sc):
    sl = sc.scanline_list
    if sl.empty():
        return False, 0
    y = sl.top()
    sl.pop()
    while not sl.empty() and y == sl.top():
        sl.pop()
    return True, y

def v_pop_locmin(sc, y):
    if sc.locmin_idx >= len(sc.locmin_list) or sc.locmin_list[sc.locmin_idx].vertex.pt.y != y:
        return False, None
    lm = sc.locmin_list[sc.locmin_idx]
    sc.locmin_idx += 1
    return True, lm

def v_push_horz(sc, e):
    e.next_in_sel = sc.sel
    sc.sel = e

def v_pop_horz(sc):
    e = sc.sel
    if e is None:
        return False, None
    sc.sel = sc.sel.next_in_sel
    return True, e


# -- Winding + contribution --

def v_set_wind_count(sc, e):
    e2 = e.prev_in_ael
    pt = v_polytype(e)
    while e2 is not None and v_polytype(e2) != pt:
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
        if v_polytype(e2) != pt:
            e.wind_cnt2 += e2.wind_dx
        e2 = e2.next_in_ael

def v_is_contributing(e, cliptype):
    if abs(e.wind_cnt) != 1:
        return False
    wc2 = abs(e.wind_cnt2)
    if cliptype == 0:
        return wc2 != 0
    if cliptype == 1:
        return wc2 == 0
    r = (wc2 == 0)
    return r if v_polytype(e) == 0 else not r


# -- Output operations --

def v_set_sides(or_, f, b):
    or_.front_edge = f
    or_.back_edge = b

def v_swap_outrecs(e1, e2):
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

def v_add_outpt(e, pt, sc):
    outrec = e.outrec
    to_front = v_is_front(e)
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

def v_add_local_min_poly(e1, e2, pt, sc, is_new):
    outrec = sc.new_outrec()
    e1.outrec = outrec
    e2.outrec = outrec
    prev_hot = v_get_prev_hot(e1)
    if prev_hot is not None:
        if (prev_hot is prev_hot.outrec.front_edge) == is_new:
            v_set_sides(outrec, e2, e1)
        else:
            v_set_sides(outrec, e1, e2)
    else:
        outrec.owner = None
        if is_new:
            v_set_sides(outrec, e1, e2)
        else:
            v_set_sides(outrec, e2, e1)
    op = sc.new_outpt(pt, outrec)
    outrec.pts = op
    return op

def v_uncouple(ae):
    or_ = ae.outrec
    if or_ is None:
        return
    or_.front_edge.outrec = None
    or_.back_edge.outrec = None
    or_.front_edge = None
    or_.back_edge = None

def v_join_outrec_paths(e1, e2):
    p1_st = e1.outrec.pts
    p2_st = e2.outrec.pts
    p1_end = p1_st.next
    p2_end = p2_st.next
    if v_is_front(e1):
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


def v_add_local_max_poly(e1, e2, pt, sc):
    if v_is_joined(e1):
        v_split(e1, pt, sc)
    if v_is_joined(e2):
        v_split(e2, pt, sc)
    if v_is_front(e1) == v_is_front(e2):
        sc.succeeded = False
        return None
    result = v_add_outpt(e1, pt, sc)
    if e1.outrec is e2.outrec:
        outrec = e1.outrec
        outrec.pts = result
        v_uncouple(e1)
        result = outrec.pts
    elif e1.outrec.idx < e2.outrec.idx:
        v_join_outrec_paths(e1, e2)
    else:
        v_join_outrec_paths(e2, e1)
    return result


# -- Split + CheckJoin --

def v_split(e, pt, sc):
    if e.join_with == JW_Right:
        e.join_with = JW_None
        e.next_in_ael.join_with = JW_None
        v_add_local_min_poly(e, e.next_in_ael, pt, sc, True)
    else:
        e.join_with = JW_None
        e.prev_in_ael.join_with = JW_None
        v_add_local_min_poly(e.prev_in_ael, e, pt, sc, True)

def v_check_join_left(e, pt, sc, check_curr_x=False):
    prev = e.prev_in_ael
    if prev is None or not v_is_hot(e) or not v_is_hot(prev) or v_is_horizontal(e) or v_is_horizontal(prev):
        return
    if (pt.y < e.top.y + 2 or pt.y < prev.top.y + 2) and (e.bot.y > pt.y or prev.bot.y > pt.y):
        return
    if check_curr_x:
        if v_perpendic_dist_sq(pt, prev.bot, prev.top) > 0.25:
            return
    else:
        if e.curr_x != prev.curr_x:
            return
    if not v_is_collinear(e.top, pt, prev.top):
        return
    if e.outrec.idx == prev.outrec.idx:
        v_add_local_max_poly(prev, e, pt, sc)
    elif e.outrec.idx < prev.outrec.idx:
        v_join_outrec_paths(e, prev)
    else:
        v_join_outrec_paths(prev, e)
    prev.join_with = JW_Right
    e.join_with = JW_Left

def v_check_join_right(e, pt, sc, check_curr_x=False):
    nxt = e.next_in_ael
    if nxt is None or not v_is_hot(e) or not v_is_hot(nxt) or v_is_horizontal(e) or v_is_horizontal(nxt):
        return
    if (pt.y < e.top.y + 2 or pt.y < nxt.top.y + 2) and (e.bot.y > pt.y or nxt.bot.y > pt.y):
        return
    if check_curr_x:
        if v_perpendic_dist_sq(pt, nxt.bot, nxt.top) > 0.35:
            return
    else:
        if e.curr_x != nxt.curr_x:
            return
    if not v_is_collinear(e.top, pt, nxt.top):
        return
    if e.outrec.idx == nxt.outrec.idx:
        v_add_local_max_poly(e, nxt, pt, sc)
    elif e.outrec.idx < nxt.outrec.idx:
        v_join_outrec_paths(e, nxt)
    else:
        v_join_outrec_paths(nxt, e)
    e.join_with = JW_Right
    nxt.join_with = JW_Left


# -- IntersectEdges --

def v_intersect_edges(e1, e2, pt, sc, cliptype):
    if v_is_joined(e1):
        v_split(e1, pt, sc)
    if v_is_joined(e2):
        v_split(e2, pt, sc)

    if v_polytype(e1) == v_polytype(e2):
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
    if (not v_is_hot(e1) and not e1_in01) or (not v_is_hot(e2) and not e2_in01):
        return

    if v_is_hot(e1) and v_is_hot(e2):
        if (old_e1_wc != 0 and old_e1_wc != 1) or (old_e2_wc != 0 and old_e2_wc != 1) or (v_polytype(e1) != v_polytype(e2)):
            v_add_local_max_poly(e1, e2, pt, sc)
        elif v_is_front(e1) or e1.outrec is e2.outrec:
            v_add_local_max_poly(e1, e2, pt, sc)
            v_add_local_min_poly(e1, e2, pt, sc, False)
        else:
            v_add_outpt(e1, pt, sc)
            v_add_outpt(e2, pt, sc)
            v_swap_outrecs(e1, e2)
    elif v_is_hot(e1):
        v_add_outpt(e1, pt, sc)
        v_swap_outrecs(e1, e2)
    elif v_is_hot(e2):
        v_add_outpt(e2, pt, sc)
        v_swap_outrecs(e1, e2)
    else:
        e1Wc2 = abs(e1.wind_cnt2)
        e2Wc2 = abs(e2.wind_cnt2)
        if not v_same_polytype(e1, e2):
            v_add_local_min_poly(e1, e2, pt, sc, False)
        elif old_e1_wc == 1 and old_e2_wc == 1:
            if cliptype == 0:
                if e1Wc2 > 0 and e2Wc2 > 0:
                    v_add_local_min_poly(e1, e2, pt, sc, False)
            elif cliptype == 1:
                if e1Wc2 <= 0 and e2Wc2 <= 0:
                    v_add_local_min_poly(e1, e2, pt, sc, False)
            else:
                if (v_polytype(e1) == 1 and e1Wc2 > 0 and e2Wc2 > 0) or \
                   (v_polytype(e1) == 0 and e1Wc2 <= 0 and e2Wc2 <= 0):
                    v_add_local_min_poly(e1, e2, pt, sc, False)


# -- Horizontal edge processing --

def v_add_trial_horz_join(sc, op):
    sc.horz_seg_list.append(VHorzSeg(op))

def v_get_last_op(e):
    r = e.outrec.pts
    if e is not e.outrec.front_edge:
        r = r.next
    return r

def v_update_edge_into_ael(sc, e, cliptype):
    e.bot = BIVec2(e.top.x, e.top.y)
    e.vertex_top = v_next_vertex(e)
    e.top = BIVec2(e.vertex_top.pt.x, e.vertex_top.pt.y)
    e.curr_x = e.bot.x
    v_set_dx(e)
    if v_is_joined(e):
        v_split(e, e.bot, sc)
    if v_is_horizontal(e):
        pt = v_next_vertex(e).pt
        while pt.y == e.top.y:
            if (pt.x < e.top.x) != (e.bot.x < e.top.x):
                break
            e.vertex_top = v_next_vertex(e)
            e.top = BIVec2(pt.x, pt.y)
            if v_is_maxima_e(e):
                break
            pt = v_next_vertex(e).pt
        v_set_dx(e)
        return
    v_insert_scanline(sc, e.top.y)
    v_check_join_left(e, e.bot, sc)
    v_check_join_right(e, e.bot, sc, True)

def v_reset_horz_dir(horz, max_v):
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

def v_do_horizontal(horz, sc, cliptype):
    y = horz.bot.y
    vertex_max = v_get_curr_y_maxima(horz)
    is_ltr, horz_left, horz_right = v_reset_horz_dir(horz, vertex_max)
    if v_is_hot(horz):
        op = v_add_outpt(horz, BIVec2(horz.curr_x, y), sc)
        v_add_trial_horz_join(sc, op)

    while True:
        e = horz.next_in_ael if is_ltr else horz.prev_in_ael
        while e is not None:
            if e.vertex_top is vertex_max:
                if v_is_hot(horz) and v_is_joined(e):
                    v_split(e, e.top, sc)
                if v_is_hot(horz):
                    while horz.vertex_top is not vertex_max:
                        v_add_outpt(horz, BIVec2(horz.top.x, horz.top.y), sc)
                        v_update_edge_into_ael(sc, horz, cliptype)
                    if is_ltr:
                        v_add_local_max_poly(horz, e, BIVec2(horz.top.x, horz.top.y), sc)
                    else:
                        v_add_local_max_poly(e, horz, BIVec2(horz.top.x, horz.top.y), sc)
                v_delete_from_ael(sc, e)
                v_delete_from_ael(sc, horz)
                return
            if vertex_max is not horz.vertex_top:
                if (is_ltr and e.curr_x > horz_right) or (not is_ltr and e.curr_x < horz_left):
                    break
                if e.curr_x == horz.top.x and not v_is_horizontal(e):
                    pt2 = v_next_vertex(horz).pt
                    if is_ltr:
                        if v_top_x(e, pt2.y) >= pt2.x:
                            break
                    else:
                        if v_top_x(e, pt2.y) <= pt2.x:
                            break
            pt = BIVec2(e.curr_x, horz.bot.y)
            if is_ltr:
                v_intersect_edges(horz, e, pt, sc, cliptype)
                v_swap_positions_in_ael(sc, horz, e)
                v_check_join_left(e, pt, sc)
                horz.curr_x = e.curr_x
                e = horz.next_in_ael
            else:
                v_intersect_edges(e, horz, pt, sc, cliptype)
                v_swap_positions_in_ael(sc, e, horz)
                v_check_join_right(e, pt, sc)
                horz.curr_x = e.curr_x
                e = horz.prev_in_ael
            if horz.outrec is not None:
                v_add_trial_horz_join(sc, v_get_last_op(horz))
        if v_next_vertex(horz).pt.y != horz.top.y:
            break
        if v_is_hot(horz):
            v_add_outpt(horz, BIVec2(horz.top.x, horz.top.y), sc)
        v_update_edge_into_ael(sc, horz, cliptype)
        is_ltr, horz_left, horz_right = v_reset_horz_dir(horz, vertex_max)
    if v_is_hot(horz):
        op = v_add_outpt(horz, BIVec2(horz.top.x, horz.top.y), sc)
        v_add_trial_horz_join(sc, op)
    v_update_edge_into_ael(sc, horz, cliptype)


# -- HorzSeg/HorzJoin handling --

def v_dup_outpt(op, after, sc):
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

def v_convert_horz_segs_to_joins(sc):
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
        hs.left_op.horz = True  # mark used
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
                sc.horz_join_list.append(VHorzJoin(v_dup_outpt(hs1.left_op, True, sc), v_dup_outpt(hs2.left_op, False, sc)))
            else:
                while hs1.left_op.prev.pt.y == cy and hs1.left_op.prev.pt.x <= hs2.left_op.pt.x:
                    hs1.left_op = hs1.left_op.prev
                while hs2.left_op.next.pt.y == cy and hs2.left_op.next.pt.x <= hs1.left_op.pt.x:
                    hs2.left_op = hs2.left_op.next
                sc.horz_join_list.append(VHorzJoin(v_dup_outpt(hs2.left_op, True, sc), v_dup_outpt(hs1.left_op, False, sc)))

def v_fix_outrec_pts(outrec):
    op = outrec.pts
    while True:
        op.outrec = outrec
        op = op.next
        if op is outrec.pts:
            break

def v_process_horz_joins(sc):
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
            v_fix_outrec_pts(or2)
            if or1.pts.outrec is or2:
                or1.pts = j.op1
                or1.pts.outrec = or1
            or2.owner = or1
        else:
            or2.pts = None
            or2.owner = or1


# -- Intersection detection (merge sort) --

def v_adjust_curr_x_copy_to_sel(sc, top_y):
    e = sc.actives
    sc.sel = e
    while e is not None:
        e.prev_in_sel = e.prev_in_ael
        e.next_in_sel = e.next_in_ael
        e.jump = e.next_in_sel
        if e.join_with == JW_Left:
            e.curr_x = e.prev_in_ael.curr_x
        else:
            e.curr_x = v_top_x(e, top_y)
        e = e.next_in_ael

def v_extract_from_sel(ae):
    res = ae.next_in_sel
    if res is not None:
        res.prev_in_sel = ae.prev_in_sel
    ae.prev_in_sel.next_in_sel = res
    return res

def v_insert1_before2_in_sel(a1, a2):
    a1.prev_in_sel = a2.prev_in_sel
    if a1.prev_in_sel is not None:
        a1.prev_in_sel.next_in_sel = a1
    a1.next_in_sel = a2
    a2.prev_in_sel = a1

def v_add_new_isect_node(sc, e1, e2, top_y):
    ok, ip = v_get_seg_isect_pt(e1.bot, e1.top, e2.bot, e2.top)
    if not ok:
        ip = BIVec2(e1.curr_x, top_y)
    if ip.y > sc.bot_y or ip.y < top_y:
        ad1 = abs(e1.dx)
        ad2 = abs(e2.dx)
        if ad1 > 100 and ad2 > 100:
            ip = v_closest_pt_on_seg(ip, e1.bot, e1.top) if ad1 > ad2 else v_closest_pt_on_seg(ip, e2.bot, e2.top)
        elif ad1 > 100:
            ip = v_closest_pt_on_seg(ip, e1.bot, e1.top)
        elif ad2 > 100:
            ip = v_closest_pt_on_seg(ip, e2.bot, e2.top)
        else:
            if ip.y < top_y:
                ip.y = top_y
            else:
                ip.y = sc.bot_y
            ip.x = v_top_x(e1, ip.y) if ad1 < ad2 else v_top_x(e2, ip.y)
    sc.intersect_nodes.append(VIntersectNode(ip, e1, e2))

def v_build_intersect_list(sc, top_y):
    if sc.actives is None or sc.actives.next_in_ael is None:
        return False
    v_adjust_curr_x_copy_to_sel(sc, top_y)
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
                        v_add_new_isect_node(sc, tmp, right, top_y)
                        if tmp is left:
                            break
                        tmp = tmp.prev_in_sel
                    tmp = right
                    right = v_extract_from_sel(tmp)
                    l_end = right
                    v_insert1_before2_in_sel(tmp, left)
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

def v_process_intersect_list(sc, cliptype):
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
        v_intersect_edges(node.edge1, node.edge2, node.pt, sc, cliptype)
        v_swap_positions_in_ael(sc, node.edge1, node.edge2)
        node.edge1.curr_x = node.pt.x
        node.edge2.curr_x = node.pt.x
        v_check_join_left(node.edge2, node.pt, sc, True)
        v_check_join_right(node.edge1, node.pt, sc, True)


# -- InsertLocalMinimaIntoAEL --

def v_insert_local_minima_into_ael(sc, bot_y, cliptype):
    while True:
        ok, lm = v_pop_locmin(sc, bot_y)
        if not ok:
            break
        lb = VActive()
        lb.bot = BIVec2(lm.vertex.pt.x, lm.vertex.pt.y)
        lb.curr_x = lb.bot.x
        lb.wind_dx = -1
        lb.vertex_top = lm.vertex.prev
        lb.top = BIVec2(lb.vertex_top.pt.x, lb.vertex_top.pt.y)
        lb.local_min = lm
        v_set_dx(lb)

        rb = VActive()
        rb.bot = BIVec2(lm.vertex.pt.x, lm.vertex.pt.y)
        rb.curr_x = rb.bot.x
        rb.wind_dx = 1
        rb.vertex_top = lm.vertex.next
        rb.top = BIVec2(rb.vertex_top.pt.x, rb.vertex_top.pt.y)
        rb.local_min = lm
        v_set_dx(rb)

        if v_is_horizontal(lb):
            if lb.dx == -_INF:
                lb, rb = rb, lb
        elif v_is_horizontal(rb):
            if rb.dx == _INF:
                lb, rb = rb, lb
        elif lb.dx < rb.dx:
            lb, rb = rb, lb

        lb.is_left_bound = True
        v_insert_left_edge(sc, lb)
        v_set_wind_count(sc, lb)
        contributing = v_is_contributing(lb, cliptype)

        rb.is_left_bound = False
        rb.wind_cnt = lb.wind_cnt
        rb.wind_cnt2 = lb.wind_cnt2
        v_insert_right_edge(lb, rb)
        if contributing:
            v_add_local_min_poly(lb, rb, BIVec2(lb.bot.x, lb.bot.y), sc, True)
            if not v_is_horizontal(lb):
                v_check_join_left(lb, lb.bot, sc)
        while rb.next_in_ael is not None and v_is_valid_ael_order(rb.next_in_ael, rb):
            v_intersect_edges(rb, rb.next_in_ael, BIVec2(rb.bot.x, rb.bot.y), sc, cliptype)
            v_swap_positions_in_ael(sc, rb, rb.next_in_ael)
        if v_is_horizontal(rb):
            v_push_horz(sc, rb)
        else:
            v_check_join_right(rb, rb.bot, sc)
            v_insert_scanline(sc, rb.top.y)
        if v_is_horizontal(lb):
            v_push_horz(sc, lb)
        else:
            v_insert_scanline(sc, lb.top.y)


# -- DoMaxima --

def v_do_maxima(e, sc, cliptype):
    prev_e = e.prev_in_ael
    next_e = e.next_in_ael
    max_pair = v_get_maxima_pair(e)
    if max_pair is None:
        return next_e
    if v_is_joined(e):
        v_split(e, e.top, sc)
    if v_is_joined(max_pair):
        v_split(max_pair, max_pair.top, sc)
    while next_e is not max_pair:
        v_intersect_edges(e, next_e, BIVec2(e.top.x, e.top.y), sc, cliptype)
        v_swap_positions_in_ael(sc, e, next_e)
        next_e = e.next_in_ael
    if v_is_hot(e):
        v_add_local_max_poly(e, max_pair, BIVec2(e.top.x, e.top.y), sc)
    v_delete_from_ael(sc, max_pair)
    v_delete_from_ael(sc, e)
    return prev_e.next_in_ael if prev_e is not None else sc.actives


# -- DoTopOfScanbeam --

def v_do_top_of_scanbeam(sc, y, cliptype):
    sc.sel = None
    e = sc.actives
    while e is not None:
        if e.top.y == y:
            e.curr_x = e.top.x
            if v_is_maxima_e(e):
                e = v_do_maxima(e, sc, cliptype)
                continue
            if v_is_hot(e):
                v_add_outpt(e, BIVec2(e.top.x, e.top.y), sc)
            v_update_edge_into_ael(sc, e, cliptype)
            if v_is_horizontal(e):
                v_push_horz(sc, e)
        else:
            e.curr_x = v_top_x(e, y)
        e = e.next_in_ael


# -- CleanCollinear + FixSelfIntersects --

def v_dispose_outpt(op):
    r = op.next
    op.prev.next = op.next
    op.next.prev = op.prev
    return r

def v_do_split_op(sc, outrec, splitOp):
    prevOp = splitOp.prev
    nnOp = splitOp.next.next
    outrec.pts = prevOp
    ok, ip = v_get_seg_isect_pt(prevOp.pt, splitOp.pt, splitOp.next.pt, nnOp.pt)
    area1 = v_area_outpt(outrec.pts)
    if abs(area1) < 2:
        outrec.pts = None
        return
    area2 = v_area_tri(ip, splitOp.pt, splitOp.next.pt)
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

def v_fix_self_intersects(sc, outrec):
    op2 = outrec.pts
    while True:
        if op2.prev is op2.next.next:
            break
        if v_segs_intersect(op2.prev.pt, op2.pt, op2.next.pt, op2.next.next.pt):
            if op2 is outrec.pts or op2.next is outrec.pts:
                outrec.pts = outrec.pts.prev
            v_do_split_op(sc, outrec, op2)
            if outrec.pts is None:
                break
            op2 = outrec.pts
            continue
        op2 = op2.next
        if op2 is outrec.pts:
            break

def v_clean_collinear(sc, outrec):
    while outrec is not None and outrec.pts is None:
        outrec = outrec.owner
    if outrec is None:
        return
    if not v_valid_closed(outrec.pts):
        outrec.pts = None
        return
    startOp = outrec.pts
    op2 = startOp
    while True:
        if v_is_collinear(op2.prev.pt, op2.pt, op2.next.pt) and \
           (op2.pt == op2.prev.pt or op2.pt == op2.next.pt or v_dot_product(op2.prev.pt, op2.pt, op2.next.pt) < 0):
            if op2 is outrec.pts:
                outrec.pts = op2.prev
            op2 = v_dispose_outpt(op2)
            if not v_valid_closed(op2):
                outrec.pts = None
                return
            startOp = op2
            continue
        op2 = op2.next
        if op2 is startOp:
            break
    v_fix_self_intersects(sc, outrec)


# -- ExecuteInternal --

def v_execute_internal(sc, cliptype):
    sc.locmin_list.sort(key=lambda lm: (-lm.vertex.pt.y, lm.vertex.pt.x))
    for lm in sc.locmin_list:
        v_insert_scanline(sc, lm.vertex.pt.y)
    sc.locmin_idx = 0

    ok, y = v_pop_scanline(sc)
    if not ok:
        return True
    while sc.succeeded:
        v_insert_local_minima_into_ael(sc, y, cliptype)
        while True:
            ok_h, e = v_pop_horz(sc)
            if not ok_h:
                break
            v_do_horizontal(e, sc, cliptype)
        if sc.horz_seg_list:
            v_convert_horz_segs_to_joins(sc)
            sc.horz_seg_list.clear()
        sc.bot_y = y
        ok, y = v_pop_scanline(sc)
        if not ok:
            break
        if sc.succeeded and v_build_intersect_list(sc, y):
            v_process_intersect_list(sc, cliptype)
            sc.intersect_nodes.clear()
        v_do_top_of_scanbeam(sc, y, cliptype)
        while True:
            ok_h, e = v_pop_horz(sc)
            if not ok_h:
                break
            v_do_horizontal(e, sc, cliptype)
    if sc.succeeded:
        v_process_horz_joins(sc)
    return sc.succeeded


# -- Build output paths --

def v_build_path(op):
    if op is None or op.next is op or op.next is op.prev:
        return None
    path = []
    last = op.next.pt
    path.append(BIVec2(last.x, last.y))
    op2 = op.next.next
    while op2 is not op.next:
        if op2.pt != last:
            last = op2.pt
            path.append(BIVec2(last.x, last.y))
        op2 = op2.next
    if len(path) < 3 or v_very_small_tri(op):
        return None
    return path


# -- Public API --

class BooleanPolyline:
    @staticmethod
    def compute(a, b, clip_type):
        ca = a.coords
        cb = b.coords
        na = len(ca) // 3
        nb = len(cb) // 3

        # Strip closing duplicate
        if na >= 2:
            dx = ca[(na - 1) * 3] - ca[0]
            dy = ca[(na - 1) * 3 + 1] - ca[1]
            if dx * dx + dy * dy < 1e-20:
                na -= 1
        if nb >= 2:
            dx = cb[(nb - 1) * 3] - cb[0]
            dy = cb[(nb - 1) * 3 + 1] - cb[1]
            if dx * dx + dy * dy < 1e-20:
                nb -= 1
        if na < 3 or nb < 3:
            return []

        # Compute safe scale from actual coordinate range to prevent int64 overflow
        # in cross products: (max_coord * scale)^2 must fit in int64
        max_coord = 0.0
        for i in range(na):
            max_coord = max(max_coord, abs(ca[i * 3]), abs(ca[i * 3 + 1]))
        for i in range(nb):
            max_coord = max(max_coord, abs(cb[i * 3]), abs(cb[i * 3 + 1]))
        if max_coord < 1e-12:
            max_coord = 1.0
        bool_scale = math.floor(math.sqrt(float(2**63 - 1)) / (2.0 * max_coord))
        bool_inv_scale = 1.0 / bool_scale

        sc = VattiScratch()

        # Small polygons: use intermediate arrays for containment check
        if na * nb <= 400:
            va = []
            vb = []
            for i in range(na):
                va.append(BIVec2(round(ca[i * 3] * bool_scale), round(ca[i * 3 + 1] * bool_scale)))
            aMinX = aMaxX = va[0].x
            aMinY = aMaxY = va[0].y
            for i in range(1, na):
                x, y = va[i].x, va[i].y
                if x < aMinX:
                    aMinX = x
                elif x > aMaxX:
                    aMaxX = x
                if y < aMinY:
                    aMinY = y
                elif y > aMaxY:
                    aMaxY = y

            for i in range(nb):
                vb.append(BIVec2(round(cb[i * 3] * bool_scale), round(cb[i * 3 + 1] * bool_scale)))
            bMinX = bMaxX = vb[0].x
            bMinY = bMaxY = vb[0].y
            for i in range(1, nb):
                x, y = vb[i].x, vb[i].y
                if x < bMinX:
                    bMinX = x
                elif x > bMaxX:
                    bMaxX = x
                if y < bMinY:
                    bMinY = y
                elif y > bMaxY:
                    bMaxY = y

            # AABB disjoint
            if aMaxX < bMinX or bMaxX < aMinX or aMaxY < bMinY or bMaxY < aMinY:
                a_in_b = pip_i(va[0], vb)
                b_in_a = pip_i(vb[0], va)
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
                if b_in_a:
                    return [a]
                return [a]

            # Containment: no crossings -> pure containment
            any_cross = False
            for i in range(na):
                if any_cross:
                    break
                a1 = va[i]
                a2 = va[(i + 1) % na]
                axmin = min(a1.x, a2.x)
                axmax = max(a1.x, a2.x)
                aymin = min(a1.y, a2.y)
                aymax = max(a1.y, a2.y)
                for j_idx in range(nb):
                    if any_cross:
                        break
                    b1 = vb[j_idx]
                    b2 = vb[(j_idx + 1) % nb]
                    if max(b1.x, b2.x) < axmin or min(b1.x, b2.x) > axmax or max(b1.y, b2.y) < aymin or min(b1.y, b2.y) > aymax:
                        continue
                    any_cross = v_segs_intersect(a1, a2, b1, b2)
            if not any_cross:
                a_in_b = pip_i(va[0], vb)
                b_in_a = pip_i(vb[0], va)
                # Validate containment with centroid -- pip_i can return true for a
                # vertex of A that lies on B's boundary (shared vertex / shared
                # edge), giving a false-positive containment. A convex polygon's
                # centroid is strictly interior to itself, so if A is truly
                # contained in B, A's centroid must also test inside B. If it
                # doesn't, the vertex test was a boundary artefact.
                ca_cen = BIVec2(0, 0)
                cb_cen = BIVec2(0, 0)
                for i in range(na):
                    ca_cen.x += va[i].x
                    ca_cen.y += va[i].y
                ca_cen.x = int(ca_cen.x / na)
                ca_cen.y = int(ca_cen.y / na)
                for i in range(nb):
                    cb_cen.x += vb[i].x
                    cb_cen.y += vb[i].y
                cb_cen.x = int(cb_cen.x / nb)
                cb_cen.y = int(cb_cen.y / nb)
                if a_in_b and not pip_i(ca_cen, vb):
                    a_in_b = False
                if b_in_a and not pip_i(cb_cen, va):
                    b_in_a = False
                # If vertex tests fail, try centroids -- catches near-identical
                # polygons with shared vertices (boundary points return false).
                if not a_in_b and not b_in_a:
                    a_in_b = pip_i(ca_cen, vb)
                    b_in_a = pip_i(cb_cen, va)
                    # If centroid also lands on boundary (exact y-match),
                    # perturb by 1 int64 unit to escape the edge.
                    if not a_in_b and not b_in_a:
                        a_in_b = pip_i(BIVec2(ca_cen.x + 1, ca_cen.y + 1), vb)
                        b_in_a = pip_i(BIVec2(cb_cen.x + 1, cb_cen.y + 1), va)
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
                if b_in_a:
                    return [a]
                return [a]
            v_add_path(va, na, 0, sc)
            v_add_path(vb, nb, 1, sc)
        else:
            # Large polygons: single pass double->VVertex
            va_head, aMinX, aMaxX, aMinY, aMaxY = v_add_path_from_doubles(ca, na, 0, sc, bool_scale)
            vb_head, bMinX, bMaxX, bMinY, bMaxY = v_add_path_from_doubles(cb, nb, 1, sc, bool_scale)
            if va_head is None or vb_head is None:
                return []
            # AABB disjoint
            if aMaxX < bMinX or bMaxX < aMinX or aMaxY < bMinY or bMaxY < aMinY:
                a_in_b = pip_vertex(va_head.pt, vb_head)
                b_in_a = pip_vertex(vb_head.pt, va_head)
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
                if b_in_a:
                    return [a]
                return [a]

        if not v_execute_internal(sc, clip_type):
            return []

        # Extract: OutPt -> Polyline._coords
        out = []
        for outrec in sc.outrec_list:
            if outrec.pts is None:
                continue
            v_clean_collinear(sc, outrec)
            if outrec.pts is None:
                continue
            op = outrec.pts
            if op is None or op.next is op or op.next is op.prev or v_very_small_tri(op):
                continue
            coords = []
            o = op.next
            last = o.pt
            coords.extend([last.x * bool_inv_scale, last.y * bool_inv_scale, 0.0])
            o = o.next
            while o is not op.next:
                if o.pt != last:
                    last = o.pt
                    coords.extend([last.x * bool_inv_scale, last.y * bool_inv_scale, 0.0])
                o = o.next
            cnt = len(coords) // 3
            if cnt < 3:
                continue
            out.append(Polyline.from_coords(coords))
        return out

    @staticmethod
    def clip_open_against_closed(open_subject, closed_clip):
        result = []
        cs = open_subject.coords
        cc = closed_clip.coords
        ns = len(cs) // 3
        nc = len(cc) // 3

        if nc >= 2:
            dx = cc[(nc - 1) * 3] - cc[0]
            dy = cc[(nc - 1) * 3 + 1] - cc[1]
            if dx * dx + dy * dy < 1e-20:
                nc -= 1
        if ns < 2 or nc < 3:
            return result

        def point_in_poly(px, py):
            inside = False
            j = nc - 1
            for i in range(nc):
                xi = cc[i * 3]
                yi = cc[i * 3 + 1]
                xj = cc[j * 3]
                yj = cc[j * 3 + 1]
                crosses = (yi > py) != (yj > py)
                if crosses:
                    xint = xj + (py - yj) * (xi - xj) / (yi - yj)
                    if px < xint:
                        inside = not inside
                j = i
            return inside

        cur = []

        def flush():
            if len(cur) < 6:
                cur.clear()
                return
            result.append(Polyline.from_coords(list(cur)))
            cur.clear()

        def push_xy(x, y):
            n = len(cur)
            if n >= 3:
                lx = cur[n - 3]
                ly = cur[n - 2]
                if abs(lx - x) < 1e-9 and abs(ly - y) < 1e-9:
                    return
            cur.append(x)
            cur.append(y)
            cur.append(0.0)

        a_inside = point_in_poly(cs[0], cs[1])
        if a_inside:
            push_xy(cs[0], cs[1])

        for si in range(ns - 1):
            ax = cs[si * 3]
            ay = cs[si * 3 + 1]
            bx = cs[(si + 1) * 3]
            by = cs[(si + 1) * 3 + 1]
            dx = bx - ax
            dy = by - ay
            ts = []
            ej = nc - 1
            for ei in range(nc):
                ex = cc[ei * 3] - cc[ej * 3]
                ey = cc[ei * 3 + 1] - cc[ej * 3 + 1]
                denom = dx * (-ey) - dy * (-ex)
                if abs(denom) >= 1e-18:
                    rx = cc[ej * 3] - ax
                    ry = cc[ej * 3 + 1] - ay
                    t = (rx * (-ey) - ry * (-ex)) / denom
                    u = (rx * (-dy) - ry * (-dx)) / denom
                    if t > 1e-12 and t <= 1.0 + 1e-12 and u >= -1e-9 and u <= 1.0 + 1e-9:
                        ts.append(0.0 if t < 0.0 else (1.0 if t > 1.0 else t))
                ej = ei
            ts.sort()
            prev_t = 0.0
            for t in ts:
                if t - prev_t < 1e-12:
                    prev_t = t
                    continue
                mid_t = 0.5 * (prev_t + t)
                mx = ax + dx * mid_t
                my = ay + dy * mid_t
                mid_in = point_in_poly(mx, my)
                ix = ax + dx * t
                iy = ay + dy * t
                if mid_in:
                    push_xy(ix, iy)
                    flush()
                else:
                    push_xy(ix, iy)
                prev_t = t
            if prev_t < 1.0 - 1e-12:
                mid_t = 0.5 * (prev_t + 1.0)
                mx = ax + dx * mid_t
                my = ay + dy * mid_t
                if point_in_poly(mx, my):
                    push_xy(bx, by)
        flush()
        return result
    @staticmethod
    def compute_raw(a_xy, na, b_xy, nb, clip_type, out_xy, max_out):
        # Raw-array version: takes flat 2D coords (x,y pairs, stride=2),
        # writes flat 2D result coords into out_xy. No Polyline construction.
        # Returns number of result points (0 if no intersection).
        if na >= 2:
            dx = a_xy[(na - 1) * 2] - a_xy[0]
            dy = a_xy[(na - 1) * 2 + 1] - a_xy[1]
            if dx * dx + dy * dy < 1e-20:
                na -= 1
        if nb >= 2:
            dx = b_xy[(nb - 1) * 2] - b_xy[0]
            dy = b_xy[(nb - 1) * 2 + 1] - b_xy[1]
            if dx * dx + dy * dy < 1e-20:
                nb -= 1
        if na < 3 or nb < 3:
            return 0

        max_coord = 0.0
        for i in range(na):
            max_coord = max(max_coord, abs(a_xy[i * 2]), abs(a_xy[i * 2 + 1]))
        for i in range(nb):
            max_coord = max(max_coord, abs(b_xy[i * 2]), abs(b_xy[i * 2 + 1]))
        if max_coord < 1e-12:
            max_coord = 1.0
        bool_scale = math.floor(math.sqrt(float(2**63 - 1)) / (2.0 * max_coord))
        bool_inv_scale = 1.0 / bool_scale

        sc = VattiScratch()

        va = []
        vb = []
        for i in range(na):
            va.append(BIVec2(round(a_xy[i * 2] * bool_scale), round(a_xy[i * 2 + 1] * bool_scale)))
        aMinX = aMaxX = va[0].x
        aMinY = aMaxY = va[0].y
        for i in range(1, na):
            x, y = va[i].x, va[i].y
            if x < aMinX:
                aMinX = x
            elif x > aMaxX:
                aMaxX = x
            if y < aMinY:
                aMinY = y
            elif y > aMaxY:
                aMaxY = y

        for i in range(nb):
            vb.append(BIVec2(round(b_xy[i * 2] * bool_scale), round(b_xy[i * 2 + 1] * bool_scale)))
        bMinX = bMaxX = vb[0].x
        bMinY = bMaxY = vb[0].y
        for i in range(1, nb):
            x, y = vb[i].x, vb[i].y
            if x < bMinX:
                bMinX = x
            elif x > bMaxX:
                bMaxX = x
            if y < bMinY:
                bMinY = y
            elif y > bMaxY:
                bMaxY = y

        if aMaxX < bMinX or bMaxX < aMinX or aMaxY < bMinY or bMaxY < aMinY:
            return 0

        v_add_path(va, na, 0, sc)
        v_add_path(vb, nb, 1, sc)

        if not v_execute_internal(sc, clip_type):
            return 0

        total_pts = 0
        for outrec in sc.outrec_list:
            if outrec.pts is None:
                continue
            v_clean_collinear(sc, outrec)
            if outrec.pts is None:
                continue
            op = outrec.pts
            if op is None or op.next is op or op.next is op.prev or v_very_small_tri(op):
                continue
            o = op.next
            last = o.pt
            if total_pts < max_out:
                out_xy[total_pts * 2] = last.x * bool_inv_scale
                out_xy[total_pts * 2 + 1] = last.y * bool_inv_scale
            total_pts += 1
            o = o.next
            while o is not op.next:
                if o.pt != last:
                    last = o.pt
                    if total_pts < max_out:
                        out_xy[total_pts * 2] = last.x * bool_inv_scale
                        out_xy[total_pts * 2 + 1] = last.y * bool_inv_scale
                    total_pts += 1
                o = o.next
        return total_pts
    @staticmethod
    def compute_count(a, b, clip_type):
        ca = a.coords
        cb = b.coords
        na = len(ca) // 3
        nb = len(cb) // 3
        if na >= 2:
            dx = ca[(na - 1) * 3] - ca[0]
            dy = ca[(na - 1) * 3 + 1] - ca[1]
            if dx * dx + dy * dy < 1e-20:
                na -= 1
        if nb >= 2:
            dx = cb[(nb - 1) * 3] - cb[0]
            dy = cb[(nb - 1) * 3 + 1] - cb[1]
            if dx * dx + dy * dy < 1e-20:
                nb -= 1
        if na < 3 or nb < 3:
            return 0

        max_coord = 0.0
        for i in range(na):
            max_coord = max(max_coord, abs(ca[i * 3]), abs(ca[i * 3 + 1]))
        for i in range(nb):
            max_coord = max(max_coord, abs(cb[i * 3]), abs(cb[i * 3 + 1]))
        if max_coord < 1e-12:
            max_coord = 1.0
        bool_scale = math.floor(math.sqrt(float(2**63 - 1)) / (2.0 * max_coord))

        sc = VattiScratch()

        va_head, aMinX, aMaxX, aMinY, aMaxY = v_add_path_from_doubles(ca, na, 0, sc, bool_scale)
        vb_head, bMinX, bMaxX, bMinY, bMaxY = v_add_path_from_doubles(cb, nb, 1, sc, bool_scale)
        if va_head is None or vb_head is None:
            return 0
        if aMaxX < bMinX or bMaxX < aMinX or aMaxY < bMinY or bMaxY < aMinY:
            return 0

        if not v_execute_internal(sc, clip_type):
            return 0

        # Just count output points -- no conversion, no Polyline allocation
        total_pts = 0
        for outrec in sc.outrec_list:
            if outrec.pts is None:
                continue
            v_clean_collinear(sc, outrec)
            if outrec.pts is None:
                continue
            op = outrec.pts
            if op is None or op.next is op or op.next is op.prev or v_very_small_tri(op):
                continue
            o = op
            while True:
                total_pts += 1
                o = o.next
                if o is op:
                    break
        return total_pts
