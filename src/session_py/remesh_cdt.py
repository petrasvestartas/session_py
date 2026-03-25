"""Constrained Delaunay Triangulation via sweep-line."""
import math

_LOOSE = 0
_ASCEND = 1
_DESCEND = 2
_IX_NONE = 0
_IX_COLLINEAR = 1
_IX_INTERSECT = 2
_EC_NEITHER = 0
_EC_LEFT = 1
_EC_RIGHT = 2


class _P64:
    __slots__ = ('x', 'y')

    def __init__(self, x=0, y=0):
        self.x = int(x)
        self.y = int(y)

    def __eq__(self, o):
        return self.x == o.x and self.y == o.y

    def __ne__(self, o):
        return not self.__eq__(o)

    def __hash__(self):
        return hash((self.x, self.y))


class _V2:
    __slots__ = ('pt', 'edges', 'innerLM')

    def __init__(self, pt):
        self.pt = pt
        self.edges = []
        self.innerLM = False


class _Edge:
    __slots__ = ('vL', 'vR', 'vB', 'vT', 'kind', 'triA', 'triB', 'isActive', 'nextE', 'prevE')

    def __init__(self):
        self.vL = self.vR = self.vB = self.vT = None
        self.kind = _LOOSE
        self.triA = self.triB = None
        self.isActive = False
        self.nextE = self.prevE = None


class _Tri:
    __slots__ = ('edges',)

    def __init__(self, e1, e2, e3):
        self.edges = [e1, e2, e3]


def _cps(p1, p2, p3):
    cp = float(p2.x - p1.x) * float(p3.y - p2.y) - float(p2.y - p1.y) * float(p3.x - p2.x)
    return 1 if cp > 0 else (-1 if cp < 0 else 0)


def _sqr(x):
    return x * x


def _dist_sqr(a, b):
    return _sqr(float(a.x - b.x)) + _sqr(float(a.y - b.y))


def _left_turning(p1, p2, p3):
    return _cps(p1, p2, p3) < 0


def _right_turning(p1, p2, p3):
    return _cps(p1, p2, p3) > 0


def _is_horiz(e):
    return e.vB.pt.y == e.vT.pt.y


def _is_loose(e):
    return e.kind == _LOOSE


def _is_left_edge(e):
    return e.kind == _ASCEND


def _is_right_edge(e):
    return e.kind == _DESCEND


def _edge_completed(e):
    if not e.triA:
        return False
    if e.triB:
        return True
    return e.kind != _LOOSE


def _edge_contains(e, v):
    if e.vL is v:
        return _EC_LEFT
    if e.vR is v:
        return _EC_RIGHT
    return _EC_NEITHER


def _remove_from_vert(vert, edge):
    vert.edges.remove(edge)


def _find_loc_min_idx(path, length, idx):
    if length < 3:
        return False, idx
    i0 = idx
    n = (idx + 1) % length
    while path[n].y <= path[idx].y:
        idx = n
        n = (n + 1) % length
        if idx == i0:
            return False, idx
    while path[n].y >= path[idx].y:
        idx = n
        n = (n + 1) % length
    return True, idx


def _prev(idx, length):
    return length - 1 if idx == 0 else idx - 1


def _next_idx(idx, length):
    return (idx + 1) % length


def _find_linking_edge(vert1, vert2, prefer_ascending):
    res = None
    for e in vert1.edges:
        if e.vL is vert2 or e.vR is vert2:
            if e.kind == _LOOSE or ((e.kind == _ASCEND) == prefer_ascending):
                return e
            res = e
    return res


def _path_from_tri(tri):
    e0 = tri.edges[0]
    res = [e0.vL.pt, e0.vR.pt]
    e1 = tri.edges[1]
    if e1.vL.pt == res[0] or e1.vL.pt == res[1]:
        res.append(e1.vR.pt)
    else:
        res.append(e1.vL.pt)
    return res


def _in_circle(pA, pB, pC, pD):
    m00 = float(pA.x - pD.x)
    m01 = float(pA.y - pD.y)
    m02 = _sqr(m00) + _sqr(m01)
    m10 = float(pB.x - pD.x)
    m11 = float(pB.y - pD.y)
    m12 = _sqr(m10) + _sqr(m11)
    m20 = float(pC.x - pD.x)
    m21 = float(pC.y - pD.y)
    m22 = _sqr(m20) + _sqr(m21)
    return m00*(m11*m22-m21*m12) - m10*(m01*m22-m21*m02) + m20*(m01*m12-m11*m02)


def _shortest_dist_seg(pt, sp1, sp2):
    dx = float(sp2.x - sp1.x)
    dy = float(sp2.y - sp1.y)
    ax = float(pt.x - sp1.x)
    ay = float(pt.y - sp1.y)
    qnum = ax*dx + ay*dy
    if qnum < 0:
        return _dist_sqr(pt, sp1)
    dd = dx*dx + dy*dy
    if qnum > dd:
        return _dist_sqr(pt, sp2)
    return _sqr(ax*dy - dx*ay) / dd


def _segs_intersect(s1a, s1b, s2a, s2b):
    if s1a == s2a or s1b == s2a or s1b == s2b:
        return _IX_NONE
    dy1 = float(s1b.y - s1a.y)
    dx1 = float(s1b.x - s1a.x)
    dy2 = float(s2b.y - s2a.y)
    dx2 = float(s2b.x - s2a.x)
    cp = dy1*dx2 - dy2*dx1
    if cp == 0:
        return _IX_COLLINEAR
    t = float(s1a.x - s2a.x)*dy2 - float(s1a.y - s2a.y)*dx2
    if t >= 0:
        if cp < 0 or t >= cp:
            return _IX_NONE
    else:
        if cp > 0 or t <= cp:
            return _IX_NONE
    t = float(s1a.x - s2a.x)*dy1 - float(s1a.y - s2a.y)*dx1
    if t >= 0:
        if cp > 0 and t < cp:
            return _IX_INTERSECT
    else:
        if cp < 0 and t > cp:
            return _IX_INTERSECT
    return _IX_NONE


class _Delaunay:
    def __init__(self, use_delaunay=True):
        self._verts = []
        self._edges = []
        self._tris = []
        self._pending = []
        self._horz = []
        self._loc_mins = []
        self._use_del = use_delaunay
        self._lowermost = None
        self._first_active = None

    def _cleanup(self):
        self._verts = []
        self._edges = []
        self._tris = []
        self._first_active = None
        self._lowermost = None

    def _add_active(self, edge):
        if edge.isActive:
            return
        edge.prevE = None
        edge.nextE = self._first_active
        edge.isActive = True
        if self._first_active:
            self._first_active.prevE = edge
        self._first_active = edge

    def _remove_active(self, edge):
        _remove_from_vert(edge.vB, edge)
        _remove_from_vert(edge.vT, edge)
        prev = edge.prevE
        nxt = edge.nextE
        if nxt:
            nxt.prevE = prev
        if prev:
            prev.nextE = nxt
        edge.isActive = False
        if self._first_active is edge:
            self._first_active = nxt

    def _create_edge(self, v1, v2, kind):
        res = _Edge()
        self._edges.append(res)
        if v1.pt.y == v2.pt.y:
            res.vB = v1; res.vT = v2
        elif v1.pt.y < v2.pt.y:
            res.vB = v2; res.vT = v1
        else:
            res.vB = v1; res.vT = v2
        if v1.pt.x <= v2.pt.x:
            res.vL = v1; res.vR = v2
        else:
            res.vL = v2; res.vR = v1
        res.kind = kind
        v1.edges.append(res)
        v2.edges.append(res)
        if kind == _LOOSE:
            self._pending.append(res)
            self._add_active(res)
        return res

    def _create_tri(self, e1, e2, e3):
        res = _Tri(e1, e2, e3)
        self._tris.append(res)
        for i in range(3):
            if res.edges[i].triA:
                res.edges[i].triB = res
                self._remove_active(res.edges[i])
            else:
                res.edges[i].triA = res
                if not _is_loose(res.edges[i]):
                    self._remove_active(res.edges[i])
        return res

    def _split_edge(self, longE, shortE):
        oldT = longE.vT
        newT = shortE.vT
        _remove_from_vert(oldT, longE)
        longE.vT = newT
        if longE.vL is oldT:
            longE.vL = newT
        else:
            longE.vR = newT
        newT.edges.append(longE)
        self._create_edge(newT, oldT, longE.kind)

    def _merge_dup_collinear(self):
        vIter1 = 0
        for vIter2 in range(1, len(self._verts)):
            v1 = self._verts[vIter1]
            v2 = self._verts[vIter2]
            if v1.pt != v2.pt:
                vIter1 = vIter2
                continue
            if not v1.innerLM or not v2.innerLM:
                v1.innerLM = False
            for e in v2.edges:
                if e.vB is v2:
                    e.vB = v1
                else:
                    e.vT = v1
                if e.vL is v2:
                    e.vL = v1
                else:
                    e.vR = v1
            v1.edges.extend(v2.edges)
            v2.edges = []
            for e1 in list(v1.edges):
                if _is_horiz(e1) or e1.vB is not v1:
                    continue
                for e2 in list(v1.edges):
                    if e2 is e1 or e2.vB is not v1:
                        continue
                    if e1.vT.pt.y == e2.vT.pt.y:
                        continue
                    if _cps(e1.vT.pt, v1.pt, e2.vT.pt) != 0:
                        continue
                    if e1.vT.pt.y < e2.vT.pt.y:
                        self._split_edge(e1, e2)
                    else:
                        self._split_edge(e2, e1)
                    break

    def _inner_loc_min_edge(self, vAbove):
        if not self._first_active:
            return None
        xA = vAbove.pt.x
        yA = vAbove.pt.y
        e = self._first_active
        eBelow = None
        bestD = -1.0
        while e:
            if (e.vL.pt.x <= xA and e.vR.pt.x >= xA and
                    e.vB.pt.y >= yA and e.vB is not vAbove and e.vT is not vAbove and
                    not _left_turning(e.vL.pt, vAbove.pt, e.vR.pt)):
                d = _shortest_dist_seg(vAbove.pt, e.vL.pt, e.vR.pt)
                if eBelow is None or d < bestD:
                    eBelow = e
                    bestD = d
            e = e.nextE
        if not eBelow:
            return None
        vBest = eBelow.vB if eBelow.vT.pt.y <= yA else eBelow.vT
        xBest = vBest.pt.x
        yBest = vBest.pt.y
        e = self._first_active
        if xBest < xA:
            while e:
                if (e.vR.pt.x > xBest and e.vL.pt.x < xA and
                        e.vB.pt.y > yA and e.vT.pt.y < yBest and
                        _segs_intersect(e.vB.pt, e.vT.pt, vBest.pt, vAbove.pt) == _IX_INTERSECT):
                    vBest = e.vT if e.vT.pt.y > yA else e.vB
                    xBest = vBest.pt.x
                    yBest = vBest.pt.y
                e = e.nextE
        else:
            while e:
                if (e.vR.pt.x < xBest and e.vL.pt.x > xA and
                        e.vB.pt.y > yA and e.vT.pt.y < yBest and
                        _segs_intersect(e.vB.pt, e.vT.pt, vBest.pt, vAbove.pt) == _IX_INTERSECT):
                    vBest = e.vT if e.vT.pt.y > yA else e.vB
                    xBest = vBest.pt.x
                    yBest = vBest.pt.y
                e = e.nextE
        return self._create_edge(vBest, vAbove, _LOOSE)

    def _horiz_between(self, v1, v2):
        y = v1.pt.y
        if v1.pt.x > v2.pt.x:
            l, r = v2.pt.x, v1.pt.x
        else:
            l, r = v1.pt.x, v2.pt.x
        res = self._first_active
        while res:
            if (res.vL.pt.y == y and res.vR.pt.y == y and
                    res.vL.pt.x >= l and res.vR.pt.x <= r and
                    (res.vL.pt.x != l or res.vL.pt.x != r)):
                return res
            res = res.nextE
        return None

    def _tri_left(self, edge, pivot, minY):
        vAlt = None
        eAlt = None
        v = edge.vT if edge.vB is pivot else edge.vB
        for e in pivot.edges:
            if e is edge or not e.isActive:
                continue
            vX = e.vB if e.vT is pivot else e.vT
            if vX is v:
                continue
            cps = _cps(v.pt, pivot.pt, vX.pt)
            if cps == 0:
                if (v.pt.x > pivot.pt.x) == (pivot.pt.x > vX.pt.x):
                    continue
            elif cps > 0 or (vAlt and not _left_turning(vX.pt, pivot.pt, vAlt.pt)):
                continue
            vAlt = vX
            eAlt = e
        if not vAlt or vAlt.pt.y < minY:
            return
        if vAlt.pt.y < pivot.pt.y:
            if _is_left_edge(eAlt):
                return
        elif vAlt.pt.y > pivot.pt.y:
            if _is_right_edge(eAlt):
                return
        eX = _find_linking_edge(vAlt, v, vAlt.pt.y < v.pt.y)
        if not eX:
            if vAlt.pt.y == v.pt.y == minY and self._horiz_between(vAlt, v):
                return
            eX = self._create_edge(vAlt, v, _LOOSE)
        self._create_tri(edge, eAlt, eX)
        if not _edge_completed(eX):
            self._tri_left(eX, vAlt, minY)

    def _tri_right(self, edge, pivot, minY):
        vAlt = None
        eAlt = None
        v = edge.vT if edge.vB is pivot else edge.vB
        for e in pivot.edges:
            if e is edge or not e.isActive:
                continue
            vX = e.vB if e.vT is pivot else e.vT
            if vX is v:
                continue
            cps = _cps(v.pt, pivot.pt, vX.pt)
            if cps == 0:
                if (v.pt.x > pivot.pt.x) == (pivot.pt.x > vX.pt.x):
                    continue
            elif cps < 0 or (vAlt and not _right_turning(vX.pt, pivot.pt, vAlt.pt)):
                continue
            vAlt = vX
            eAlt = e
        if not vAlt or vAlt.pt.y < minY:
            return
        if vAlt.pt.y < pivot.pt.y:
            if _is_right_edge(eAlt):
                return
        elif vAlt.pt.y > pivot.pt.y:
            if _is_left_edge(eAlt):
                return
        eX = _find_linking_edge(vAlt, v, vAlt.pt.y > v.pt.y)
        if not eX:
            if vAlt.pt.y == v.pt.y == minY and self._horiz_between(vAlt, v):
                return
            eX = self._create_edge(vAlt, v, _LOOSE)
        self._create_tri(edge, eX, eAlt)
        if not _edge_completed(eX):
            self._tri_right(eX, vAlt, minY)

    def _force_legal(self, edge):
        if not edge.triA or not edge.triB:
            return
        vertA = None
        vertB = None
        edgesA = [None, None, None]
        edgesB = [None, None, None]
        for i in range(3):
            if edge.triA.edges[i] is edge:
                continue
            ec = _edge_contains(edge.triA.edges[i], edge.vL)
            if ec == _EC_LEFT:
                edgesA[1] = edge.triA.edges[i]
                vertA = edge.triA.edges[i].vR
            elif ec == _EC_RIGHT:
                edgesA[1] = edge.triA.edges[i]
                vertA = edge.triA.edges[i].vL
            else:
                edgesB[1] = edge.triA.edges[i]
        for i in range(3):
            if edge.triB.edges[i] is edge:
                continue
            ec = _edge_contains(edge.triB.edges[i], edge.vL)
            if ec == _EC_LEFT:
                edgesA[2] = edge.triB.edges[i]
                vertB = edge.triB.edges[i].vR
            elif ec == _EC_RIGHT:
                edgesA[2] = edge.triB.edges[i]
                vertB = edge.triB.edges[i].vL
            else:
                edgesB[2] = edge.triB.edges[i]
        if _cps(vertA.pt, edge.vL.pt, edge.vR.pt) == 0:
            return
        ict = _in_circle(vertA.pt, edge.vL.pt, edge.vR.pt, vertB.pt)
        if ict == 0 or (_right_turning(vertA.pt, edge.vL.pt, edge.vR.pt) == (ict < 0)):
            return
        edge.vL = vertA
        edge.vR = vertB
        edge.triA.edges[0] = edge
        for i in range(1, 3):
            edge.triA.edges[i] = edgesA[i]
            if _is_loose(edgesA[i]):
                self._pending.append(edgesA[i])
            if edgesA[i].triA is edge.triA or edgesA[i].triB is edge.triA:
                continue
            if edgesA[i].triA is edge.triB:
                edgesA[i].triA = edge.triA
            elif edgesA[i].triB is edge.triB:
                edgesA[i].triB = edge.triA
        edge.triB.edges[0] = edge
        for i in range(1, 3):
            edge.triB.edges[i] = edgesB[i]
            if _is_loose(edgesB[i]):
                self._pending.append(edgesB[i])
            if edgesB[i].triA is edge.triB or edgesB[i].triB is edge.triB:
                continue
            if edgesB[i].triA is edge.triA:
                edgesB[i].triA = edge.triB
            elif edgesB[i].triB is edge.triA:
                edgesB[i].triB = edge.triB

    def _add_path(self, path):
        length = len(path)
        ok, i0 = _find_loc_min_idx(path, length, 0)
        if not ok:
            return
        iPrev = _prev(i0, length)
        while path[iPrev] == path[i0]:
            iPrev = _prev(iPrev, length)
        iNext = _next_idx(i0, length)
        i = i0
        while _cps(path[iPrev], path[i], path[iNext]) == 0:
            ok, i = _find_loc_min_idx(path, length, i)
            if not ok or i == i0:
                return
            iPrev = _prev(i, length)
            while path[iPrev] == path[i]:
                iPrev = _prev(iPrev, length)
            iNext = _next_idx(i, length)
        vert_cnt = len(self._verts)
        v0 = _V2(path[i])
        self._verts.append(v0)
        if _left_turning(path[iPrev], path[i], path[iNext]):
            v0.innerLM = True
        vPrev = v0
        i = iNext
        while True:
            self._loc_mins.append(vPrev)
            if (not self._lowermost or
                    vPrev.pt.y > self._lowermost.pt.y or
                    (vPrev.pt.y == self._lowermost.pt.y and vPrev.pt.x < self._lowermost.pt.x)):
                self._lowermost = vPrev
            iNext = _next_idx(i, length)
            if _cps(vPrev.pt, path[i], path[iNext]) == 0:
                i = iNext
                continue
            while path[i].y <= vPrev.pt.y:
                v = _V2(path[i])
                self._verts.append(v)
                self._create_edge(vPrev, v, _ASCEND)
                vPrev = v
                i = iNext
                iNext = _next_idx(i, length)
                while _cps(vPrev.pt, path[i], path[iNext]) == 0:
                    i = iNext
                    iNext = _next_idx(i, length)
            vPrevPrev = vPrev
            while i != i0 and path[i].y >= vPrev.pt.y:
                v = _V2(path[i])
                self._verts.append(v)
                self._create_edge(v, vPrev, _DESCEND)
                vPrevPrev = vPrev
                vPrev = v
                i = iNext
                iNext = _next_idx(i, length)
                while _cps(vPrev.pt, path[i], path[iNext]) == 0:
                    i = iNext
                    iNext = _next_idx(i, length)
            if i == i0:
                break
            if _left_turning(vPrevPrev.pt, vPrev.pt, path[i]):
                vPrev.innerLM = True
        self._create_edge(v0, vPrev, _DESCEND)
        n_new = len(self._verts) - vert_cnt
        if n_new < 3 or (n_new == 3 and (
                _dist_sqr(self._verts[vert_cnt].pt, self._verts[vert_cnt+1].pt) <= 1 or
                _dist_sqr(self._verts[vert_cnt+1].pt, self._verts[vert_cnt+2].pt) <= 1 or
                _dist_sqr(self._verts[vert_cnt+2].pt, self._verts[vert_cnt].pt) <= 1)):
            for j in range(vert_cnt, len(self._verts)):
                self._verts[j].edges = []

    def _add_paths(self, paths):
        total = sum(len(p) for p in paths)
        if total == 0:
            return False
        for path in paths:
            self._add_path(path)
        return len(self._verts) > 2

    def execute(self, paths):
        if not self._add_paths(paths):
            return []
        if self._lowermost.innerLM:
            for lm in self._loc_mins:
                lm.innerLM = not lm.innerLM
            for e in self._edges:
                if e.kind == _ASCEND:
                    e.kind = _DESCEND
                elif e.kind == _DESCEND:
                    e.kind = _ASCEND
        self._loc_mins = []
        self._edges.sort(key=lambda e: e.vL.pt.x)
        self._verts.sort(key=lambda v: (-v.pt.y, v.pt.x))
        self._merge_dup_collinear()
        currY = self._verts[0].pt.y
        for v in self._verts:
            if not v.edges:
                continue
            if v.pt.y != currY:
                while self._loc_mins:
                    lm = self._loc_mins.pop()
                    e = self._inner_loc_min_edge(lm)
                    if not e:
                        self._cleanup()
                        return None
                    if _is_horiz(e):
                        if e.vL is e.vB:
                            self._tri_left(e, e.vB, currY)
                        else:
                            self._tri_right(e, e.vB, currY)
                    else:
                        self._tri_left(e, e.vB, currY)
                        if not _edge_completed(e):
                            self._tri_right(e, e.vB, currY)
                    self._add_active(lm.edges[0])
                    self._add_active(lm.edges[1])
                while self._horz:
                    e = self._horz.pop()
                    if _edge_completed(e):
                        continue
                    if e.vB is e.vL:
                        if _is_left_edge(e):
                            self._tri_left(e, e.vB, currY)
                    else:
                        if _is_right_edge(e):
                            self._tri_right(e, e.vB, currY)
                currY = v.pt.y
            for i in range(len(v.edges) - 1, -1, -1):
                if i >= len(v.edges):
                    continue
                e = v.edges[i]
                if _edge_completed(e) or _is_loose(e):
                    continue
                if v is e.vB:
                    if _is_horiz(e):
                        self._horz.append(e)
                    if not v.innerLM:
                        self._add_active(e)
                else:
                    if _is_horiz(e):
                        self._horz.append(e)
                    elif _is_left_edge(e):
                        self._tri_left(e, e.vB, v.pt.y)
                    else:
                        self._tri_right(e, e.vB, v.pt.y)
            if v.innerLM:
                self._loc_mins.append(v)
        while self._horz:
            e = self._horz.pop()
            if not _edge_completed(e) and e.vB is e.vL:
                self._tri_left(e, e.vB, currY)
        if self._use_del:
            while self._pending:
                e = self._pending.pop()
                self._force_legal(e)
        res = []
        for tri in self._tris:
            p = _path_from_tri(tri)
            cps = _cps(p[0], p[1], p[2])
            if cps == 0:
                continue
            if cps < 0:
                p = [p[2], p[1], p[0]]
            res.append(p)
        self._cleanup()
        return res


def cdt_triangulate(border_2d, holes_2d=None):
    """Constrained Delaunay triangulation of a polygon with optional holes.
    border_2d: list of Point (uses [0],[1] as x,y)
    holes_2d: optional list of lists of Point
    Returns: list of (i,j,k) index tuples into flat array [border..., hole0..., hole1...]"""
    scale = 1e6
    flat = list(border_2d)
    if holes_2d:
        for h in holes_2d:
            flat.extend(h)
    pt_map = {}
    for i, p in enumerate(flat):
        key = (round(p[0] * scale), round(p[1] * scale))
        if key not in pt_map:
            pt_map[key] = i

    def make_path(pts):
        path = [_P64(round(p[0]*scale), round(p[1]*scale)) for p in pts]
        if len(path) > 1 and path[0] == path[-1]:
            path.pop()
        return path

    paths = [make_path(border_2d)]
    if holes_2d:
        for h in holes_2d:
            paths.append(make_path(h))
    d = _Delaunay(True)
    tris = d.execute(paths)
    if not tris:
        return []
    out = []
    for tri in tris:
        f = []
        ok = True
        for pt in tri:
            key = (pt.x, pt.y)
            if key not in pt_map:
                ok = False
                break
            f.append(pt_map[key])
        if ok:
            out.append((f[0], f[1], f[2]))
    return out
