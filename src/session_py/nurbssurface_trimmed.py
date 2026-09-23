from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import json
import math
import uuid
from .closest import Closest
from .color import Color
from .mesh import Mesh
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .point import Point
from .primitives import Primitives
from .tolerance import PI
from .vector import Vector

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import nurbssurface_trimmed_pb2
    from .xform import Xform


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _crease_side_normal(
    surface: NurbsSurface,
    knots: list[list[float]],
    center: tuple[float, float],
    uv: list[float],
) -> Vector:
    """Normal on the side of a C0 knot line that belongs to the triangle around center."""

    for direction in range(2):
        if uv[direction] in knots[direction]:
            uv[direction] = math.nextafter(uv[direction], center[direction])

    return surface.normal_at(uv[0], uv[1])


def _point_in_polygon_2d(u: float, v: float, poly: list[Point]) -> bool:
    """Winding-number test of (u, v) against a closed UV polygon."""

    winding = 0
    n = len(poly)

    for i in range(n):
        j = (i + 1) % n
        x0 = poly[i][0]
        y0 = poly[i][1]
        x1 = poly[j][0]
        y1 = poly[j][1]
        cross = (x1 - x0) * (v - y0) - (y1 - y0) * (u - x0)

        if y0 <= v and y1 > v and cross > 0.0:
            winding += 1

        if y0 > v and y1 <= v and cross < 0.0:
            winding -= 1

    return winding != 0


def _inside_loops(u: float, v: float, loops_uv: list[list[Point]]) -> bool:
    """True when (u, v) lies inside the outer loop and outside every hole."""

    if not _point_in_polygon_2d(u, v, loops_uv[0]):
        return False

    for li in range(1, len(loops_uv)):
        if _point_in_polygon_2d(u, v, loops_uv[li]):
            return False

    return True


def _eval3(srf: NurbsSurface, u: float, v: float) -> list[float]:
    """Surface point at (u, v) as a plain array."""

    p = srf.point_at(u, v)

    return [p[0], p[1], p[2]]


def _plane_field(
    srf: NurbsSurface, q: list[float], n: list[float], u: float, v: float
) -> float:
    """Signed distance of the surface point at (u, v) to the plane (q, n)."""

    p = _eval3(srf, u, v)

    return (p[0] - q[0]) * n[0] + (p[1] - q[1]) * n[1] + (p[2] - q[2]) * n[2]


def _refine_crossing(
    srf: NurbsSurface, q: list[float], n: list[float], u: float, v: float
) -> tuple[float, float]:
    """Newton steps of (u, v) onto the plane (q, n) along the field gradient."""

    for it in range(12):
        fv = _plane_field(srf, q, n, u, v)

        if abs(fv) < 1e-9:
            break

        h = 1e-4
        a = _eval3(srf, u + h, v)
        b = _eval3(srf, u - h, v)
        c = _eval3(srf, u, v + h)
        d = _eval3(srf, u, v - h)
        gu = ((a[0] - b[0]) * n[0] + (a[1] - b[1]) * n[1] + (a[2] - b[2]) * n[2]) / (
            2 * h
        )
        gv = ((c[0] - d[0]) * n[0] + (c[1] - d[1]) * n[1] + (c[2] - d[2]) * n[2]) / (
            2 * h
        )
        g2 = gu * gu + gv * gv

        if g2 < 1e-20:
            break

        u -= fv * gu / g2
        v -= fv * gv / g2

    return u, v


def _span_subdivisions(
    srf: NurbsSurface,
    dir: int,
    sp: list[float],
    osp: list[float],
    deg: int,
    max_angle_deg: float,
    chord_tol: float,
) -> list[int]:
    """Subdivisions per span along dir from the normal turn (max_angle_deg) and the chord deviation (chord_tol) at the mid line of the other direction."""

    n = len(sp) - 1
    subs = [2 if deg > 1 else 1] * n
    smid = (osp[0] + osp[-1]) * 0.5

    for i in range(n):
        t0 = sp[i]
        t1 = sp[i + 1]

        if deg > 1:
            ma = 0.0
            pn = Vector(0.0, 0.0, 0.0)

            for k in range(5):
                t = t0 + k * (t1 - t0) / 4.0
                nm = srf.normal_at(t, smid) if dir == 0 else srf.normal_at(smid, t)

                if k > 0:
                    d = max(-1.0, min(1.0, pn.dot(nm)))
                    ma += math.acos(d) * 180.0 / PI

                pn = nm

            subs[i] = max(subs[i], max(1, min(math.ceil(ma / max_angle_deg), 64)))

        p0 = _eval3(srf, t0, smid) if dir == 0 else _eval3(srf, smid, t0)
        p1 = _eval3(srf, t1, smid) if dir == 0 else _eval3(srf, smid, t1)
        dev = 0.0

        for k in range(1, 4):
            fr = k / 4.0
            tm = t0 + fr * (t1 - t0)
            pm = _eval3(srf, tm, smid) if dir == 0 else _eval3(srf, smid, tm)
            lx = p0[0] + fr * (p1[0] - p0[0])
            ly = p0[1] + fr * (p1[1] - p0[1])
            lz = p0[2] + fr * (p1[2] - p0[2])
            dd = math.sqrt((pm[0] - lx) ** 2 + (pm[1] - ly) ** 2 + (pm[2] - lz) ** 2)
            dev = max(dev, dd)

        if dev > chord_tol:
            subs[i] = max(subs[i], min(math.ceil(math.sqrt(dev / chord_tol)), 64))

    return subs


def _span_parameters(sp: list[float], subs: list[int]) -> list[float]:
    """Grid parameters: each span of sp cut into subs[i] equal steps, ending on the last knot."""

    out = []

    for i in range(len(sp) - 1):
        for st in range(subs[i]):
            out.append(sp[i] + st * (sp[i + 1] - sp[i]) / subs[i])

    out.append(sp[-1])

    return out


def _unit3(n: Vector) -> list[float] | None:
    """Unit normal as a plain array, or none when degenerate."""

    nl = math.sqrt(n.magnitude_squared())

    if nl < 1e-12:
        return None

    return [n[0] / nl, n[1] / nl, n[2] / nl]


def _segment_intersection(
    p1: list[float], p2: list[float], p3: list[float], p4: list[float]
) -> tuple[float, float] | None:
    """Parameters of the 2D segment crossing p1p2 x p3p4, or false when parallel or outside."""

    d1u = p2[0] - p1[0]
    d1v = p2[1] - p1[1]
    d2u = p4[0] - p3[0]
    d2v = p4[1] - p3[1]
    den = d1u * d2v - d1v * d2u

    if abs(den) < 1e-20:
        return None

    s = ((p3[0] - p1[0]) * d2v - (p3[1] - p1[1]) * d2u) / den
    t = ((p3[0] - p1[0]) * d1v - (p3[1] - p1[1]) * d1u) / den

    if s < -1e-12 or s > 1.0 + 1e-12 or t < -1e-12 or t > 1.0 + 1e-12:
        return None

    return (s, t)


def _newton_curve_curve(
    ca: NurbsCurve, ta: float, cb: NurbsCurve, tb: float, tol: float
) -> tuple[float, float]:
    """Newton refinement of a UV curve-curve crossing (ta, tb), clamped to the domains."""

    for it in range(8):
        da = ca.evaluate(ta, 1)
        db = cb.evaluate(tb, 1)
        fu = da[0][0] - db[0][0]
        fv = da[0][1] - db[0][1]

        if math.hypot(fu, fv) < tol:
            break

        j00 = da[1][0]
        j01 = -db[1][0]
        j10 = da[1][1]
        j11 = -db[1][1]
        den = j00 * j11 - j01 * j10

        if abs(den) < 1e-20:
            break

        ta -= (fu * j11 - j01 * fv) / den
        tb -= (j00 * fv - fu * j10) / den
        a0, a1 = ca.domain()
        b0, b1 = cb.domain()
        ta = min(max(ta, a0), a1)
        tb = min(max(tb, b0), b1)

    return ta, tb


def _loop_signed_area(loop: NurbsCurve) -> float:
    """Signed area of a closed UV loop sampled at 64 parameters."""

    n = 64
    l0, l1 = loop.domain()
    s = 0.0
    prev = loop.point_at(l0)

    for i in range(1, n + 1):
        p = loop.point_at(l0 + (l1 - l0) * i / n)
        s += prev[0] * p[1] - p[0] * prev[1]
        prev = p

    return s * 0.5


def _project_to_uv(
    pt: Point, p00: Point, u_axis: Vector, v_axis: Vector, u_len2: float, v_len2: float
) -> Point:
    """Plane coordinates of pt in the affine frame (p00, u_axis, v_axis)."""

    d = pt - p00

    return Point(d.dot(u_axis) / u_len2, d.dot(v_axis) / v_len2, 0.0)


def _snap_to_border(
    p: list[float], u0: float, u1: float, v0: float, v1: float, snap_uv: float
) -> None:
    """Snap a UV point onto the domain border when within snap_uv of it."""

    if abs(p[0] - u0) < snap_uv:
        p[0] = u0

    if abs(p[0] - u1) < snap_uv:
        p[0] = u1

    if abs(p[1] - v0) < snap_uv:
        p[1] = v0

    if abs(p[1] - v1) < snap_uv:
        p[1] = v1


# ═══════════════════════════════════════════════════════════════════════════
# VertexWelder
# ═══════════════════════════════════════════════════════════════════════════
class _VertexWelder:
    """Adds 3D points to a mesh, returning the existing vertex when one lies within tol."""

    def __init__(self, mesh: Mesh, tol: float, cell: float):
        """Construct over a mesh with a weld tolerance and a hash cell size."""

        self._mesh = mesh  # Mesh receiving the vertices.
        self._tol = tol  # Weld tolerance.
        self._cell = cell  # Hash cell size.
        self._cells = {}  # Vertices per cell.

    def weld(self, p: Point) -> int:
        """Weld a 3D point, returning the existing vertex within tol or a new one."""

        ci = math.floor(p[0] / self._cell)
        cj = math.floor(p[1] / self._cell)
        ck = math.floor(p[2] / self._cell)

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    bucket = self._cells.get((ci + di, cj + dj, ck + dk))

                    if bucket is None:
                        continue

                    for q, vk in bucket:
                        if (q - p).magnitude_squared() <= self._tol * self._tol:
                            return vk

        vk = self._mesh.add_vertex(p)
        self._cells.setdefault((ci, cj, ck), []).append((p, vk))

        return vk

    def weld_surface(self, srf: NurbsSurface, u: float, v: float) -> int:
        """Weld the surface point at (u, v); a new vertex gets the surface normal."""

        before = self._mesh.number_of_vertices()
        vk = self.weld(srf.point_at(u, v))

        if self._mesh.number_of_vertices() > before:
            nm = srf.normal_at(u, v)
            self._mesh.vertex[vk].set_normal(nm[0], nm[1], nm[2])

        return vk


# ═══════════════════════════════════════════════════════════════════════════
# UVGraph
# ═══════════════════════════════════════════════════════════════════════════
class _UVVertexPool:
    """Snapped UV vertices of the split graph: points within snap of each other share one id."""

    def __init__(self, snap: float):
        """Construct with the snap distance."""

        self._snap = snap  # Snap distance.
        self._cells = {}  # Ids per cell.
        self.verts = []  # UV position per id.

    def id(self, p: list[float]) -> int:
        """Id of the vertex within snap of p, a new one when none."""

        ci = math.floor(p[0] / self._snap)
        cj = math.floor(p[1] / self._snap)

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                bucket = self._cells.get((ci + di, cj + dj))

                if bucket is None:
                    continue

                for vk in bucket:
                    q = self.verts[vk]

                    if math.hypot(q[0] - p[0], q[1] - p[1]) <= self._snap:
                        return vk

        vk = len(self.verts)
        self.verts.append([p[0], p[1]])
        self._cells.setdefault((ci, cj), []).append(vk)

        return vk


def _cycle_area(cycle: list[int], hes: list, verts: list[list[float]]) -> float:
    """Signed area of a half-edge cycle."""

    s = 0.0

    for hi in cycle:
        a = verts[hes[hi][0]]
        b = verts[hes[hi][1]]
        s += a[0] * b[1] - b[0] * a[1]

    return s * 0.5


def _point_in_cycle(
    p: list[float], cycle: list[int], hes: list, verts: list[list[float]]
) -> bool:
    """Even-odd test of p against a half-edge cycle."""

    inside = False

    for hi in cycle:
        a = verts[hes[hi][0]]
        b = verts[hes[hi][1]]

        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            inside = not inside

    return inside


def _cycle_to_segments(
    cycle: list[int], hes: list, verts: list[list[float]], pcurves: list[NurbsCurve]
) -> list[NurbsCurve]:
    """Pieces of a cycle: trimmed pcurve runs, straight UV segments where a run cannot be cut."""

    runs = []

    for hi in cycle:
        tail, head, e, fwd = hes[hi]
        ta = e["ta"] if fwd else e["tb"]
        tb = e["tb"] if fwd else e["ta"]

        if runs and runs[-1]["cidx"] == e["cidx"] and runs[-1]["vb"] == tail:
            runs[-1]["vb"] = head
            runs[-1]["tb"] = tb
        else:
            runs.append({"cidx": e["cidx"], "va": tail, "vb": head, "ta": ta, "tb": tb})

    pieces = []

    for run in runs:
        made = False

        if run["cidx"] >= 0:
            crv = pcurves[run["cidx"]]
            c0, c1 = crv.domain()
            lo = max(c0, min(run["ta"], run["tb"]))
            hi_ = min(c1, max(run["ta"], run["tb"]))
            piece = crv.duplicate()
            piece_ok = True

            if hi_ - lo < (c1 - c0) - 1e-12 and hi_ - lo > 1e-14:
                if not piece.trim(lo, hi_):
                    piece_ok = False
            elif hi_ - lo <= 1e-14 and not (
                run["va"] == run["vb"] and piece.is_closed()
            ):
                piece_ok = False

            if piece_ok and piece.is_valid():
                if run["ta"] > run["tb"]:
                    piece.reverse()

                pieces.append(piece)
                made = True

        if not made:
            pa = verts[run["va"]]
            pb = verts[run["vb"]]

            if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 1e-14:
                seg_pts = [
                    Point(pa[0], pa[1], 0.0),
                    Point(pb[0], pb[1], 0.0),
                ]
                pieces.append(NurbsCurve.create(False, 1, seg_pts))

    return pieces


def _cycle_to_loop(
    cycle: list[int],
    hes: list,
    verts: list[list[float]],
    pcurves: list[NurbsCurve],
    snap_uv: float,
) -> NurbsCurve:
    """Closed loop of a cycle: the joined pieces when they close, else the polygon through its vertices."""

    pieces = _cycle_to_segments(cycle, hes, verts, pcurves)

    if not pieces:
        return NurbsCurve()

    join_tol = snap_uv * 4.0
    joined = NurbsCurve.join(pieces, join_tol)

    if len(joined) == 1 and joined[0].is_valid():
        J = joined[0]

        if (
            not J.is_closed()
            and J.point_at_start().distance(J.point_at_end()) <= join_tol
        ):
            x, y, z, w = J.get_cv_4d(0)
            xe, ye, ze, we = J.get_cv_4d(J.cv_count() - 1)
            J.set_cv_4d(J.cv_count() - 1, x, y, z, we)

        if J.is_closed():
            return J

    loop_pts = []

    for hi in cycle:
        a = verts[hes[hi][0]]
        loop_pts.append(Point(a[0], a[1], 0.0))

    loop_pts.append(Point(loop_pts[0][0], loop_pts[0][1], 0.0))

    return NurbsCurve.create(False, 1, loop_pts)


# ═══════════════════════════════════════════════════════════════════════════
# Delaunay2D
# ═══════════════════════════════════════════════════════════════════════════
class _Vertex2D:
    """UV vertex of the triangulation."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        """Construct at (x, y)."""

        self.x = x  # U coordinate.
        self.y = y  # V coordinate.


class _Triangle:
    """Triangle with per-edge neighbours; edge k is opposite vertex k."""

    __slots__ = ("v", "adj", "constrained", "alive")

    def __init__(self, v0: int, v1: int, v2: int):
        """Construct over three vertices with no neighbours."""

        self.v = [v0, v1, v2]  # Vertex indices.
        self.adj = [-1, -1, -1]  # Neighbour across each edge, -1 on the hull.
        self.constrained = [False, False, False]  # True where an edge is a constraint.
        self.alive = True  # False once removed.


class _Delaunay2D:
    """Incremental constrained Delaunay triangulation in UV with Bowyer-Watson insertion."""

    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        """Construct with a super triangle around the box."""

        dx = xmax - xmin
        dy = ymax - ymin
        d = max(dx, dy)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        scale = 20.0
        self.vertices = []  # Vertices, the super triangle first.
        self.triangles = []  # Triangle pool, dead ones flagged.
        self.edge_map = {}  # Hull edge -> (triangle, edge index).
        self.last_found = 0  # Triangle the last locate ended in.
        self.vertices.append(_Vertex2D(cx - scale * d, cy - scale * d))
        self.vertices.append(_Vertex2D(cx + scale * d, cy - scale * d))
        self.vertices.append(_Vertex2D(cx, cy + scale * d))
        self.super_v = [0, 1, 2]  # Super triangle vertices.
        self.triangles.append(_Triangle(0, 1, 2))
        self._register_edges(0)

    @staticmethod
    def _edge_key(a, b):
        """Order-independent key of an edge."""
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _in_circumcircle(ax, ay, bx, by, cx, cy, dx, dy):
        """Positive when d lies inside the circumcircle of a, b, c."""

        adx = ax - dx
        ady = ay - dy
        bdx = bx - dx
        bdy = by - dy
        cdx = cx - dx
        cdy = cy - dy

        return (
            (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
            + (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
            + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
        )

    @staticmethod
    def _orient2d(ax, ay, bx, by, cx, cy):
        """Twice the signed area of a, b, c."""
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def _register_edges(self, ti):
        """Record the hull edges of triangle ti in the edge map."""

        for k in range(3):
            a = self.triangles[ti].v[(k + 1) % 3]
            b = self.triangles[ti].v[(k + 2) % 3]
            key = self._edge_key(a, b)
            other = self.edge_map.get(key)

            if other is not None:
                oti, ok = other
                self.triangles[ti].adj[k] = oti
                self.triangles[oti].adj[ok] = ti
                del self.edge_map[key]
            else:
                self.edge_map[key] = (ti, k)

    def _unregister_edges(self, ti):
        """Drop the hull edges of triangle ti from the edge map and its neighbours."""

        for k in range(3):
            a = self.triangles[ti].v[(k + 1) % 3]
            b = self.triangles[ti].v[(k + 2) % 3]
            key = self._edge_key(a, b)
            adj_ti = self.triangles[ti].adj[k]

            if adj_ti >= 0 and self.triangles[adj_ti].alive:
                adj_kk = -1

                for kk in range(3):
                    if self.triangles[adj_ti].adj[kk] == ti:
                        adj_kk = kk
                        break

                if adj_kk >= 0:
                    self.triangles[adj_ti].adj[adj_kk] = -1
                    adj_a = self.triangles[adj_ti].v[(adj_kk + 1) % 3]
                    adj_b = self.triangles[adj_ti].v[(adj_kk + 2) % 3]
                    self.edge_map[self._edge_key(adj_a, adj_b)] = (adj_ti, adj_kk)
            else:
                e = self.edge_map.get(key)

                if e is not None and e[0] == ti:
                    del self.edge_map[key]

    def _locate(self, x, y, start_tri):
        """Triangle containing (x, y) by walking from start_tri, -1 when none."""

        if start_tri < 0 or start_tri >= len(self.triangles) or not self.triangles[start_tri].alive:
            start_tri = len(self.triangles) - 1

            while start_tri >= 0 and not self.triangles[start_tri].alive:
                start_tri -= 1

            if start_tri < 0:
                return -1

        cur = start_tri

        for _ in range(len(self.triangles)):
            t = self.triangles[cur]
            moved = False

            for k in range(3):
                a = t.v[k]
                b = t.v[(k + 1) % 3]
                ax = self.vertices[a].x
                ay = self.vertices[a].y
                bx = self.vertices[b].x
                by = self.vertices[b].y

                if self._orient2d(ax, ay, bx, by, x, y) < 0:
                    opp = t.adj[(k + 2) % 3]

                    if opp >= 0 and self.triangles[opp].alive:
                        cur = opp
                        moved = True
                        break

            if not moved:
                return cur

        return cur

    def insert(self, x: float, y: float) -> int:
        """Insert a point and return its vertex index, the existing one when coincident."""

        start = self._locate(x, y, self.last_found)

        if start >= 0:
            t = self.triangles[start]

            for k in range(3):
                vi2 = t.v[k]
                ddx = self.vertices[vi2].x - x
                ddy = self.vertices[vi2].y - y

                if ddx * ddx + ddy * ddy < 1e-12:
                    return vi2

        vi = len(self.vertices)
        self.vertices.append(_Vertex2D(x, y))
        bad = []
        visited = set()

        if start >= 0:
            bad.append(start)
            visited.add(start)

        bfs_front = 0

        while bfs_front < len(bad):
            ti = bad[bfs_front]
            bfs_front += 1

            if not self.triangles[ti].alive:
                continue

            v0, v1, v2 = self.triangles[ti].v
            ax = self.vertices[v0].x
            ay = self.vertices[v0].y
            bx = self.vertices[v1].x
            by = self.vertices[v1].y
            cx = self.vertices[v2].x
            cy = self.vertices[v2].y
            o = self._orient2d(ax, ay, bx, by, cx, cy)

            if o > 0:
                ic = self._in_circumcircle(ax, ay, bx, by, cx, cy, x, y)
            else:
                ic = self._in_circumcircle(ax, ay, cx, cy, bx, by, x, y)

            if ic > 0:
                for k in range(3):
                    if self.triangles[ti].constrained[k]:
                        continue

                    nb = self.triangles[ti].adj[k]

                    if nb >= 0 and nb not in visited:
                        visited.add(nb)
                        bad.append(nb)

        kept = []

        for ti in bad:
            if not self.triangles[ti].alive:
                continue

            v0, v1, v2 = self.triangles[ti].v
            ax = self.vertices[v0].x
            ay = self.vertices[v0].y
            bx = self.vertices[v1].x
            by = self.vertices[v1].y
            cx = self.vertices[v2].x
            cy = self.vertices[v2].y
            o = self._orient2d(ax, ay, bx, by, cx, cy)

            if o > 0:
                ic = self._in_circumcircle(ax, ay, bx, by, cx, cy, x, y)
            else:
                ic = self._in_circumcircle(ax, ay, cx, cy, bx, by, x, y)

            if ic > 0:
                kept.append(ti)

        bad = kept

        if not bad:
            self.vertices.pop()

            return -1

        bad_set = set(bad)
        polygon = []

        for ti in bad:
            t = self.triangles[ti]

            for k in range(3):
                nb = t.adj[k]

                if nb < 0 or nb not in bad_set:
                    polygon.append(
                        (t.v[(k + 1) % 3], t.v[(k + 2) % 3], t.constrained[k])
                    )

        for ti in bad:
            self._unregister_edges(ti)
            self.triangles[ti].alive = False

        for e0, e1, constr in polygon:
            o = self._orient2d(
                self.vertices[vi].x,
                self.vertices[vi].y,
                self.vertices[e0].x,
                self.vertices[e0].y,
                self.vertices[e1].x,
                self.vertices[e1].y,
            )

            if abs(o) < 1e-20:
                continue

            new_ti = len(self.triangles)

            if o > 0:
                va, vb = e0, e1
            else:
                va, vb = e1, e0

            tri = _Triangle(vi, va, vb)
            tri.constrained[0] = constr
            self.triangles.append(tri)
            self._register_edges(new_ti)

        self.last_found = len(self.triangles) - 1

        return vi

    def insert_constraint(self, v0: int, v1: int) -> None:
        """Force the edge v0-v1 into the triangulation by flipping the edges it crosses."""

        if v0 == v1:
            return

        for ti in range(len(self.triangles)):
            if not self.triangles[ti].alive:
                continue

            for k in range(3):
                e0 = self.triangles[ti].v[(k + 1) % 3]
                e1 = self.triangles[ti].v[(k + 2) % 3]

                if (e0 == v0 and e1 == v1) or (e0 == v1 and e1 == v0):
                    self.triangles[ti].constrained[k] = True
                    nb = self.triangles[ti].adj[k]

                    if nb >= 0 and self.triangles[nb].alive:
                        for kk in range(3):
                            if self.triangles[nb].adj[kk] == ti:
                                self.triangles[nb].constrained[kk] = True
                                break

                    return

        start_ti = -1

        for i in range(len(self.triangles) - 1, -1, -1):
            if not self.triangles[i].alive:
                continue

            for k in range(3):
                if self.triangles[i].v[k] == v0:
                    start_ti = i
                    break

        if start_ti < 0:
            return

        ax = self.vertices[v0].x
        ay = self.vertices[v0].y
        bx = self.vertices[v1].x
        by = self.vertices[v1].y
        ivl = -1
        ivr = -1
        it = -1
        ti = start_ti
        walk_guard = len(self.triangles) + 4

        for _ in range(walk_guard):
            if not self.triangles[ti].alive:
                break

            k_v0 = -1

            for i in range(3):
                if self.triangles[ti].v[i] == v0:
                    k_v0 = i
                    break

            if k_v0 < 0:
                break

            k = k_v0
            ip2 = self.triangles[ti].v[(k + 1) % 3]
            ip1 = self.triangles[ti].v[(k + 2) % 3]
            op2 = self._orient2d(
                ax, ay, bx, by, self.vertices[ip2].x, self.vertices[ip2].y
            )
            op1 = self._orient2d(
                ax, ay, bx, by, self.vertices[ip1].x, self.vertices[ip1].y
            )

            if op2 < 0 and op1 >= 0:
                ivl = ip1
                ivr = ip2
                it = ti
                break

            nxt = self.triangles[ti].adj[(k + 1) % 3]

            if nxt < 0 or not self.triangles[nxt].alive or nxt == start_ti:
                break

            ti = nxt

        if it < 0:
            return

        poly_l = [v0, ivl]
        poly_r = [v0, ivr]
        intersected = [it]
        iv = v0
        cur_it = it
        cross_guard = len(self.triangles) * 2 + 8

        for _ in range(cross_guard):
            if self._has_vertex(cur_it, v1):
                break

            k_iv = -1

            for i in range(3):
                if self.triangles[cur_it].v[i] == iv:
                    k_iv = i
                    break

            if k_iv < 0:
                break

            i_topo = self.triangles[cur_it].adj[k_iv]

            if i_topo < 0 or not self.triangles[i_topo].alive:
                break

            i_vopo = -1

            for k in range(3):
                if self.triangles[i_topo].adj[k] == cur_it:
                    i_vopo = self.triangles[i_topo].v[k]
                    break

            if i_vopo < 0:
                break

            o = self._orient2d(
                ax, ay, bx, by, self.vertices[i_vopo].x, self.vertices[i_vopo].y
            )

            if o < 0:
                if i_vopo != v1:
                    poly_r.append(i_vopo)

                iv = ivr
                ivr = i_vopo
            else:
                if i_vopo != v1:
                    poly_l.append(i_vopo)

                iv = ivl
                ivl = i_vopo

            intersected.append(i_topo)
            cur_it = i_topo

        poly_l.append(v1)
        poly_r.append(v1)
        first_new = len(self.triangles)

        for ti in intersected:
            self._unregister_edges(ti)
            self.triangles[ti].alive = False

        for i in range(max(len(poly_l) - 2, 0)):
            self._add_triangle(v1, poly_l[i + 1], poly_l[i])

        for i in range(1, max(len(poly_r) - 1, 1)):
            self._add_triangle(v0, poly_r[i], poly_r[i + 1])

        for new_ti in range(first_new, len(self.triangles)):
            if not self.triangles[new_ti].alive:
                continue

            for k in range(3):
                nb = self.triangles[new_ti].adj[k]

                if nb < 0 or nb >= first_new or not self.triangles[nb].alive:
                    continue

                for kk in range(3):
                    if (
                        self.triangles[nb].adj[kk] == new_ti
                        and self.triangles[nb].constrained[kk]
                    ):
                        self.triangles[new_ti].constrained[k] = True
                        break

        for ti in range(len(self.triangles)):
            if not self.triangles[ti].alive:
                continue

            for k in range(3):
                e0 = self.triangles[ti].v[(k + 1) % 3]
                e1 = self.triangles[ti].v[(k + 2) % 3]

                if (e0 == v0 and e1 == v1) or (e0 == v1 and e1 == v0):
                    self.triangles[ti].constrained[k] = True

    def _has_vertex(self, ti, v):
        """True when triangle ti has vertex v."""

        return (
            self.triangles[ti].v[0] == v
            or self.triangles[ti].v[1] == v
            or self.triangles[ti].v[2] == v
        )

    def _add_triangle(self, pa, pb, pc):
        """New counter-clockwise triangle over three vertices; skipped when degenerate."""

        o = self._orient2d(
            self.vertices[pa].x,
            self.vertices[pa].y,
            self.vertices[pb].x,
            self.vertices[pb].y,
            self.vertices[pc].x,
            self.vertices[pc].y,
        )

        if abs(o) < 1e-20:
            return

        new_ti = len(self.triangles)

        if o > 0:
            self.triangles.append(_Triangle(pa, pb, pc))
        else:
            self.triangles.append(_Triangle(pa, pc, pb))

        self._register_edges(new_ti)

    def cleanup(self) -> None:
        """Drop the triangles touching the super triangle."""

        sv = self.super_v

        for ti in range(len(self.triangles)):
            if not self.triangles[ti].alive:
                continue

            for k in range(3):
                if self.triangles[ti].v[k] in (sv[0], sv[1], sv[2]):
                    self._unregister_edges(ti)
                    self.triangles[ti].alive = False
                    break

        self.last_found = 0

        for i in range(len(self.triangles)):
            if self.triangles[i].alive:
                self.last_found = i
                break

    def get_triangles(self) -> list[tuple[int, int, int]]:
        """Vertex index triples of the live triangles."""

        result = []

        for t in self.triangles:
            if not t.alive:
                continue

            a, b, c = t.v[0], t.v[1], t.v[2]
            o = self._orient2d(
                self.vertices[a].x,
                self.vertices[a].y,
                self.vertices[b].x,
                self.vertices[b].y,
                self.vertices[c].x,
                self.vertices[c].y,
            )

            if o > 0:
                result.append((a, b, c))
            else:
                result.append((a, c, b))

        return result


# ═══════════════════════════════════════════════════════════════════════════
# TrimLoops
# ═══════════════════════════════════════════════════════════════════════════
class TrimLoops:
    """Trim wires of one face as UV polygons, optional 3D points per loop vertex shared bit for bit with the neighbouring face, and interior UV seeds."""

    def __init__(self):
        """Construct empty trim loops."""

        self.uv: list[list[Point]] = []  # UV polygon per loop.
        self.xyz: list[list[Point]] = []  # 3D point per loop vertex, empty when not shared.
        self.interior_uv: list[Point] = []  # UV seeds inside the face.


# ═══════════════════════════════════════════════════════════════════════════
# NurbsSurfaceTrimmed
# ═══════════════════════════════════════════════════════════════════════════
class NurbsSurfaceTrimmed:
    """A NURBS surface bounded by a closed outer loop and optional inner loops in its UV space."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self):
        """Construct an empty untrimmed face."""

        self._guid = None  # Lazily minted GUID.
        self.name = "my_nurbssurface_trimmed"  # Face name.
        self.width = 1.0  # Display width.
        self.surfacecolor = Color.black()  # Display color of the surface.
        self.m_surface = NurbsSurface()  # Underlying surface.
        self.m_outer_loop = NurbsCurve()  # Closed outer loop in UV space.
        self.m_inner_loops = []  # Closed hole loops in UV space.

    def duplicate(self) -> NurbsSurfaceTrimmed:
        """Copy with a new guid and the same data."""

        result = copy.deepcopy(self)
        result._guid = None

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def create(
        surface: NurbsSurface, outer_loop: NurbsCurve
    ) -> NurbsSurfaceTrimmed:
        """Surface with a closed outer loop given in its UV parameter space."""

        ts = NurbsSurfaceTrimmed()
        ts.m_surface = surface.duplicate()
        ts.m_outer_loop = outer_loop.duplicate()

        return ts

    @staticmethod
    def create_planar(boundary: NurbsCurve) -> NurbsSurfaceTrimmed:
        """Planar surface fitted to a closed 3D boundary, the boundary projected as the outer loop."""

        srf = Primitives.create_planar(boundary)

        if not srf.is_valid():
            return NurbsSurfaceTrimmed()

        p00 = srf.get_cv(0, 0)
        u_axis = srf.get_cv(1, 0) - p00
        v_axis = srf.get_cv(0, 1) - p00
        u_len2 = u_axis.magnitude_squared()
        v_len2 = v_axis.magnitude_squared()

        if u_len2 < 1e-28 or v_len2 < 1e-28:
            return NurbsSurfaceTrimmed()

        uv_pts = []

        if boundary.degree() <= 1:
            for i in range(boundary.cv_count()):
                uv_pts.append(
                    _project_to_uv(
                        boundary.get_cv(i), p00, u_axis, v_axis, u_len2, v_len2
                    )
                )
        else:
            spans = boundary.get_span_vector()
            n_sub = 10

            for si in range(len(spans) - 1):
                for k in range(n_sub + 1):
                    t = spans[si] + (spans[si + 1] - spans[si]) * k / n_sub
                    uv = _project_to_uv(boundary.point_at(t), p00, u_axis, v_axis, u_len2, v_len2)

                    if not uv_pts or (uv - uv_pts[-1]).magnitude_squared() > 1e-24:
                        uv_pts.append(uv)

        ts = NurbsSurfaceTrimmed()
        ts.m_surface = srf

        if len(uv_pts) >= 3:
            ts.m_outer_loop = NurbsCurve.create(False, 1, uv_pts)

        return ts

    @staticmethod
    def split_by_uv_curves(
        srf: NurbsSurface, pcurves: list[NurbsCurve], tolerance: float = 0.0
    ) -> list[NurbsSurfaceTrimmed]:
        """One trimmed face per region of the UV domain carved by the pcurves (x=u, y=v, z=0); dangling cutters are discarded."""

        if not srf.is_valid():
            return []

        u0, u1 = srf.domain(0)
        v0, v1 = srf.domain(1)
        range_u = u1 - u0
        range_v = v1 - v0

        spans_u = srf.get_span_vector(0)
        spans_v = srf.get_span_vector(1)
        nu = max(len(spans_u) - 1, 1) * 4
        nv = max(len(spans_v) - 1, 1) * 4
        du = range_u / nu
        dv = range_v / nv
        mu = (u0 + u1) * 0.5
        mv = (v0 + v1) * 0.5
        pmid = srf.point_at(mu, mv)
        uv_to_3d_u = pmid.distance(srf.point_at(min(mu + du, u1), mv)) / du
        uv_to_3d_v = pmid.distance(srf.point_at(mu, min(mv + dv, v1))) / dv
        uv_to_3d = max(uv_to_3d_u, uv_to_3d_v)

        if uv_to_3d < 1e-10:
            uv_to_3d = 1.0

        if tolerance > 0.0:
            snap_uv = max(1e-9, tolerance / uv_to_3d)
        else:
            snap_uv = min(range_u, range_v) * 1e-7

        samp_tol = max(range_u, range_v) * 2e-5
        polylines = []

        for cidx, crv in enumerate(pcurves):
            if not crv.is_valid():
                continue

            ct0, ct1 = crv.domain()
            entries = []
            n = min(max(crv.cv_count() * 4, 16), 2048)

            for i in range(n + 1):
                t = ct0 + (ct1 - ct0) * i / n
                p = crv.point_at(t)
                entries.append([t, p[0], p[1]])

            depth = 0

            while depth < 6:
                inserted = 0
                i = 0

                while i < len(entries) - 1:
                    a = entries[i]
                    b = entries[i + 1]
                    tm = (a[0] + b[0]) * 0.5
                    pm = crv.point_at(tm)
                    exu = b[1] - a[1]
                    exv = b[2] - a[2]
                    l2 = exu * exu + exv * exv

                    if l2 > 1e-30:
                        s = ((pm[0] - a[1]) * exu + (pm[1] - a[2]) * exv) / l2
                        cx = a[1] + s * exu
                        cy = a[2] + s * exv
                        dev = math.hypot(pm[0] - cx, pm[1] - cy)
                    else:
                        dev = 0.0

                    if dev > samp_tol and len(entries) < 4096:
                        entries.insert(i + 1, [tm, pm[0], pm[1]])
                        inserted += 1
                        i += 2
                    else:
                        i += 1

                if inserted == 0:
                    break

                depth += 1

            pts = []
            ts = []

            for t, pu, pv in entries:
                p = [min(max(pu, u0), u1), min(max(pv, v0), v1)]
                _snap_to_border(p, u0, u1, v0, v1, snap_uv)

                if (
                    pts
                    and abs(p[0] - pts[-1][0]) < 1e-15
                    and abs(p[1] - pts[-1][1]) < 1e-15
                ):
                    continue

                pts.append(p)
                ts.append(t)

            if len(pts) < 2:
                continue

            on_u0 = True
            on_u1 = True
            on_v0 = True
            on_v1 = True

            for p in pts:
                if abs(p[0] - u0) >= snap_uv:
                    on_u0 = False

                if abs(p[0] - u1) >= snap_uv:
                    on_u1 = False

                if abs(p[1] - v0) >= snap_uv:
                    on_v0 = False

                if abs(p[1] - v1) >= snap_uv:
                    on_v1 = False

            if on_u0 or on_u1 or on_v0 or on_v1:
                continue

            polylines.append({"cidx": cidx, "pts": pts, "ts": ts})

        polylines.append({"cidx": -1, "pts": [[u0, v0], [u1, v0]], "ts": [u0, u1]})
        polylines.append({"cidx": -2, "pts": [[u1, v0], [u1, v1]], "ts": [v0, v1]})
        polylines.append({"cidx": -3, "pts": [[u1, v1], [u0, v1]], "ts": [u1, u0]})
        polylines.append({"cidx": -4, "pts": [[u0, v1], [u0, v0]], "ts": [v1, v0]})

        min_ext = max(snap_uv * 8.0, min(range_u, range_v) * 1e-5)
        kept = []

        for poly in polylines:
            ext = 0.0

            for k in range(1, len(poly["pts"])):
                ext += math.hypot(
                    poly["pts"][k][0] - poly["pts"][k - 1][0],
                    poly["pts"][k][1] - poly["pts"][k - 1][1],
                )

            if poly["cidx"] < 0 or ext >= min_ext:
                kept.append(poly)

        polylines = kept

        splits = {}

        for pi in range(len(polylines)):
            for pj in range(pi + 1, len(polylines)):
                A = polylines[pi]
                B = polylines[pj]

                if A["cidx"] < 0 and B["cidx"] < 0:
                    continue

                aminu = A["pts"][0][0]
                amaxu = A["pts"][0][0]
                aminv = A["pts"][0][1]
                amaxv = A["pts"][0][1]

                for p in A["pts"]:
                    aminu = min(aminu, p[0])
                    amaxu = max(amaxu, p[0])
                    aminv = min(aminv, p[1])
                    amaxv = max(amaxv, p[1])

                aminu -= snap_uv
                amaxu += snap_uv
                aminv -= snap_uv
                amaxv += snap_uv
                bminu = B["pts"][0][0]
                bmaxu = B["pts"][0][0]
                bminv = B["pts"][0][1]
                bmaxv = B["pts"][0][1]

                for p in B["pts"]:
                    bminu = min(bminu, p[0])
                    bmaxu = max(bmaxu, p[0])
                    bminv = min(bminv, p[1])
                    bmaxv = max(bmaxv, p[1])

                if bminu > amaxu or bmaxu < aminu or bminv > amaxv or bmaxv < aminv:
                    continue

                for ia in range(len(A["pts"]) - 1):
                    for ib in range(len(B["pts"]) - 1):
                        hit = _segment_intersection(
                            A["pts"][ia],
                            A["pts"][ia + 1],
                            B["pts"][ib],
                            B["pts"][ib + 1],
                        )

                        if hit is None:
                            continue

                        s, t = hit
                        ta = A["ts"][ia] + (A["ts"][ia + 1] - A["ts"][ia]) * s
                        tb = B["ts"][ib] + (B["ts"][ib + 1] - B["ts"][ib]) * t
                        hu = (
                            A["pts"][ia][0]
                            + (A["pts"][ia + 1][0] - A["pts"][ia][0]) * s
                        )
                        hv = (
                            A["pts"][ia][1]
                            + (A["pts"][ia + 1][1] - A["pts"][ia][1]) * s
                        )

                        if A["cidx"] >= 0 and B["cidx"] >= 0:
                            ta, tb = _newton_curve_curve(
                                pcurves[A["cidx"]],
                                ta,
                                pcurves[B["cidx"]],
                                tb,
                                snap_uv * 0.01,
                            )
                            pa = pcurves[A["cidx"]].point_at(ta)
                            hu, hv = pa[0], pa[1]
                        elif A["cidx"] >= 0:
                            pa = pcurves[A["cidx"]].point_at(ta)
                            hu, hv = pa[0], pa[1]
                        elif B["cidx"] >= 0:
                            pb = pcurves[B["cidx"]].point_at(tb)
                            hu, hv = pb[0], pb[1]

                        hp = [hu, hv]
                        _snap_to_border(hp, u0, u1, v0, v1, snap_uv)

                        if B["cidx"] < 0:
                            if B["cidx"] in (-1, -3):
                                tb = hp[0]
                            else:
                                tb = hp[1]

                        if A["cidx"] < 0:
                            if A["cidx"] in (-1, -3):
                                ta = hp[0]
                            else:
                                ta = hp[1]

                        splits.setdefault((pi, ia), []).append((s, hp[0], hp[1], ta))
                        splits.setdefault((pj, ib), []).append((t, hp[0], hp[1], tb))

        pool = _UVVertexPool(snap_uv)
        edges = []

        for pi, poly in enumerate(polylines):
            chain = []

            for i in range(len(poly["pts"])):
                chain.append((pool.id(poly["pts"][i]), poly["ts"][i]))

                if i < len(poly["pts"]) - 1 and (pi, i) in splits:
                    evs = sorted(splits[(pi, i)])

                    for frac, hu, hv, tc in evs:
                        chain.append((pool.id([hu, hv]), tc))

            for i in range(len(chain) - 1):
                a, ta = chain[i]
                b, tb = chain[i + 1]

                if a == b:
                    continue

                edges.append({"a": a, "b": b, "cidx": poly["cidx"], "ta": ta, "tb": tb})

        alive = [True] * len(edges)
        changed = True

        for _ in range(len(edges) + 1):
            if not changed:
                break

            changed = False
            degree = {}

            for ei, e in enumerate(edges):
                if not alive[ei]:
                    continue

                degree[e["a"]] = degree.get(e["a"], 0) + 1
                degree[e["b"]] = degree.get(e["b"], 0) + 1

            for ei, e in enumerate(edges):
                if not alive[ei]:
                    continue

                if degree.get(e["a"], 0) == 1 or degree.get(e["b"], 0) == 1:
                    alive[ei] = False
                    changed = True

        live_edges = []

        for ei in range(len(edges)):
            if alive[ei]:
                live_edges.append(edges[ei])

        if not live_edges:
            return []

        verts = pool.verts
        hes = []

        for e in live_edges:
            hes.append([e["a"], e["b"], e, 1])
            hes.append([e["b"], e["a"], e, 0])

        out_map = []

        for vid in range(len(verts)):
            out_map.append([])

        for hi in range(len(hes)):
            out_map[hes[hi][0]].append(hi)

        for vid in range(len(out_map)):
            fan = []

            for hi in out_map[vid]:
                fan.append((math.atan2(verts[hes[hi][1]][1] - verts[vid][1], verts[hes[hi][1]][0] - verts[vid][0]), hi))

            fan.sort()

            for k in range(len(fan)):
                out_map[vid][k] = fan[k][1]

        next_he = [-1] * len(hes)

        for vid in range(len(out_map)):
            outs = out_map[vid]

            for pos in range(len(outs)):
                hi = outs[pos]
                tw = hi ^ 1
                nxt = outs[(pos + len(outs) - 1) % len(outs)]
                next_he[tw] = nxt

        visited = [False] * len(hes)
        faces = []

        for hi in range(len(hes)):
            if visited[hi]:
                continue

            cycle = []
            cur = hi

            while cur >= 0 and not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = next_he[cur]

            if len(cycle) >= 2:
                faces.append(cycle)

        border_vids = set()

        for e in live_edges:
            if e["cidx"] < 0:
                border_vids.add(e["a"])
                border_vids.add(e["b"])

        pos_faces = []
        neg_faces = []

        for cycle in faces:
            area = _cycle_area(cycle, hes, verts)

            if area > snap_uv * snap_uv:
                pos_faces.append((cycle, area))
            elif area < -snap_uv * snap_uv:
                touches_border = False

                for hi in cycle:
                    if hes[hi][0] in border_vids:
                        touches_border = True
                        break

                if not touches_border:
                    neg_faces.append(cycle)

        holes_of = []

        for fi in range(len(pos_faces)):
            holes_of.append([])

        for cycle in neg_faces:
            sample = verts[hes[cycle[0]][0]]
            best = -1
            best_area = float("inf")

            for fi, (fc, area) in enumerate(pos_faces):
                if area < best_area and _point_in_cycle(sample, fc, hes, verts):
                    hole_vids = set()
                    face_vids = set()

                    for hi in cycle:
                        hole_vids.add(hes[hi][0])

                    for hi in fc:
                        face_vids.add(hes[hi][0])

                    if hole_vids == face_vids:
                        continue

                    best = fi
                    best_area = area

            if best >= 0:
                holes_of[best].append(cycle)

        result = []

        for fi, (cycle, area) in enumerate(pos_faces):
            outer = _cycle_to_loop(cycle, hes, verts, pcurves, snap_uv)

            if not outer.is_valid():
                continue

            if _loop_signed_area(outer) < 0.0:
                outer.reverse()

            ts = NurbsSurfaceTrimmed.create(srf, outer)

            for hole_cycle in holes_of[fi]:
                hole = _cycle_to_loop(hole_cycle, hes, verts, pcurves, snap_uv)

                if hole.is_valid():
                    if _loop_signed_area(hole) > 0.0:
                        hole.reverse()

                    ts.add_inner_loop(hole)

            result.append(ts)

        return result

    @staticmethod
    def split_by_planes(
        srf: NurbsSurface, planes: list[tuple[Point, Vector]]
    ) -> list[NurbsSurfaceTrimmed]:
        """One trimmed face per non-empty region carved by the planes (all 2^K sign combinations)."""

        k = len(planes)

        if k == 0 or k > 16:
            return []

        out = []

        for mask in range(1 << k):
            cp = []

            for i in range(k):
                q, n = planes[i]
                flip = ((mask >> i) & 1) == 1

                if flip:
                    nn = Vector(-n[0], -n[1], -n[2])
                else:
                    nn = Vector(n[0], n[1], n[2])

                cp.append((q, nn))

            ts = NurbsSurfaceTrimmed()
            ts.m_surface = srf.duplicate()
            m = ts.mesh_by_planes(cp, 20.0, 0.01)

            if m.number_of_faces() > 0:
                out.append(ts)

        return out

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __eq__(self, other) -> bool:
        """Compare name, width, color and surface; guid and loops ignored."""

        if not isinstance(other, NurbsSurfaceTrimmed):
            return False

        if self.name != other.name:
            return False

        if self.width != other.width:
            return False

        if self.surfacecolor != other.surfacecolor:
            return False

        if self.m_surface != other.m_surface:
            return False

        return True

    def __ne__(self, other) -> bool:
        """Compare name, width, color and surface; guid and loops ignored."""
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════
    def transform(self, xform: Xform) -> None:
        """Transform the surface in place; the loops live in UV and stay."""
        self.m_surface.transform(xform)

    def transformed(self, xform: Xform) -> NurbsSurfaceTrimmed:
        """Return a transformed copy."""

        ts = self.duplicate()
        ts.transform(xform)

        return ts

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def has_guid(self) -> bool:
        """Return whether the lazy guid has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the guid, creating it on first access."""

        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        """Set the guid."""
        self._guid = value

    def surface(self) -> NurbsSurface:
        """Return a copy of the underlying surface."""
        return self.m_surface.duplicate()

    def get_outer_loop(self) -> NurbsCurve:
        """Return a copy of the outer loop."""
        return self.m_outer_loop.duplicate()

    def set_outer_loop(self, loop: NurbsCurve) -> None:
        """Replace the outer loop."""
        self.m_outer_loop = loop

    def is_trimmed(self) -> bool:
        """Return whether the outer loop is a valid curve."""
        return self.m_outer_loop.is_valid()

    def is_valid(self) -> bool:
        """Return whether the underlying surface is valid."""
        return self.m_surface.is_valid()

    # ═══════════════════════════════════════════════════════════════════════════
    # Inner loops
    # ═══════════════════════════════════════════════════════════════════════════
    def add_inner_loop(self, loop_2d: NurbsCurve) -> None:
        """Hole given directly as a closed 2D curve in UV space."""
        self.m_inner_loops.append(loop_2d)

    def add_hole(self, curve_3d: NurbsCurve) -> None:
        """Hole from a 3D curve pulled onto the surface and normalized into [0,1]^2."""

        dom = curve_3d.domain()
        sdom_u = self.m_surface.domain(0)
        sdom_v = self.m_surface.domain(1)
        range_u = sdom_u[1] - sdom_u[0]
        range_v = sdom_v[1] - sdom_v[0]
        n_samples = min(max(curve_3d.cv_count() * 4, 32), 2048)
        uv_pts = []

        for i in range(n_samples):
            t = dom[0] + (dom[1] - dom[0]) * i / n_samples
            pt3d = curve_3d.point_at(t)
            u, v, _ = Closest.surface_point(self.m_surface, pt3d)
            nu = (u - sdom_u[0]) / range_u
            nv = (v - sdom_v[0]) / range_v
            uv_pts.append(Point(nu, nv, 0.0))

        if len(uv_pts) >= 3:
            self.m_inner_loops.append(NurbsCurve.create(True, 1, uv_pts))

    def add_holes(self, curves_3d: list[NurbsCurve]) -> None:
        """Add one hole per 3D curve pulled onto the surface."""

        for crv in curves_3d:
            self.add_hole(crv)

    def get_inner_loop(self, index: int) -> NurbsCurve:
        """Return a copy of the inner loop at index."""
        return self.m_inner_loops[index].duplicate()

    def inner_loop_count(self) -> int:
        """Return the number of inner loops."""
        return len(self.m_inner_loops)

    def clear_inner_loops(self) -> None:
        """Remove every inner loop."""
        self.m_inner_loops.clear()

    # ═══════════════════════════════════════════════════════════════════════════
    # Evaluation
    # ═══════════════════════════════════════════════════════════════════════════
    def point_at(self, u: float, v: float) -> Point:
        """Return the surface point at (u, v)."""
        return self.m_surface.point_at(u, v)

    def normal_at(self, u: float, v: float) -> Vector:
        """Return the unit surface normal at (u, v)."""
        return self.m_surface.normal_at(u, v)

    # ═══════════════════════════════════════════════════════════════════════════
    # Meshing
    # ═══════════════════════════════════════════════════════════════════════════
    def mesh(self) -> Mesh:
        """Return mesh_q at 20 degrees and a chord factor of 0.005."""
        return self.mesh_q(20.0, 0.005)

    def mesh_q(self, max_angle_deg: float, chord_factor: float) -> Mesh:
        """Deflection-refined constrained Delaunay of the trim loops: angular bound in degrees, chord factor as a fraction of the bbox diagonal."""

        if not self.is_trimmed():
            return self.m_surface.mesh()

        deflection = self._bbox_diagonal() * chord_factor
        loops = TrimLoops()
        loops.uv.append(self._discretize_loop(self.m_outer_loop, deflection))

        for inner in self.m_inner_loops:
            loops.uv.append(self._discretize_loop(inner, deflection))

        return self._triangulate(loops, max_angle_deg, chord_factor)

    def mesh_loops(
        self, loops: TrimLoops, max_angle_deg: float, chord_factor: float
    ) -> Mesh:
        """Mesh sampled loops (outer first, then holes) keeping every loop vertex, tagged boundary/{loop}/{sample}; knot crossings add boundary_interval/{loop}/{segment}; empty mesh on invalid input."""

        if (
            not loops.uv
            or not math.isfinite(max_angle_deg)
            or max_angle_deg <= 0.0
            or not math.isfinite(chord_factor)
            or chord_factor <= 0.0
            or (loops.xyz and len(loops.xyz) != len(loops.uv))
        ):
            return Mesh()

        expected = 0

        for li, points in enumerate(loops.uv):
            if len(points) < 3 or (loops.xyz and len(loops.xyz[li]) != len(points)):
                return Mesh()

            for point in points:
                if not math.isfinite(point[0]) or not math.isfinite(point[1]):
                    return Mesh()

            if loops.xyz:
                for point in loops.xyz[li]:
                    if (
                        not math.isfinite(point[0])
                        or not math.isfinite(point[1])
                        or not math.isfinite(point[2])
                    ):
                        return Mesh()

            expected += len(points)

        result = self._triangulate(loops, max_angle_deg, chord_factor)
        actual = set()

        for vd in result.vertex.values():
            for name in vd.attributes:
                if name.startswith("boundary/"):
                    actual.add(name)

        return result if len(actual) == expected else Mesh()

    def mesh_by_plane(
        self, q0: Point, normal: Vector, max_angle_deg: float, chord_factor: float
    ) -> Mesh:
        """Mesh of the half (S-q0).n <= 0: span-adaptive grid, marching-squares clip with Newton-refined crossings, seams welded."""

        srf = self.m_surface
        n = _unit3(normal)

        if n is None:
            return srf.mesh()

        q = [q0[0], q0[1], q0[2]]

        usp = srf.get_span_vector(0)
        vsp = srf.get_span_vector(1)

        if len(usp) < 2 or len(vsp) < 2:
            return srf.mesh()

        bbox_diag = self._bbox_diagonal()
        chord_tol = bbox_diag * chord_factor
        us = _span_parameters(
            usp,
            _span_subdivisions(
                srf, 0, usp, vsp, srf.degree(0), max_angle_deg, chord_tol
            ),
        )
        vs = _span_parameters(
            vsp,
            _span_subdivisions(
                srf, 1, vsp, usp, srf.degree(1), max_angle_deg, chord_tol
            ),
        )
        nu = len(us)
        nv = len(vs)

        if nu < 2 or nv < 2:
            return srf.mesh()

        field = []

        for i in range(nu):
            row = []

            for j in range(nv):
                row.append(_plane_field(srf, q, n, us[i], vs[j]))

            field.append(row)

        result = Mesh()
        weld_tol = bbox_diag * 1e-5
        welder = _VertexWelder(result, weld_tol, weld_tol)

        for i in range(nu - 1):
            for j in range(nv - 1):
                cu = [us[i], us[i + 1], us[i + 1], us[i]]
                cv = [vs[j], vs[j], vs[j + 1], vs[j + 1]]
                fc = [field[i][j], field[i + 1][j], field[i + 1][j + 1], field[i][j + 1]]
                inn = [fc[0] <= 0, fc[1] <= 0, fc[2] <= 0, fc[3] <= 0]
                cnt = (
                    (1 if inn[0] else 0)
                    + (1 if inn[1] else 0)
                    + (1 if inn[2] else 0)
                    + (1 if inn[3] else 0)
                )

                if cnt == 0:
                    continue

                poly = []

                for k in range(4):
                    kn = (k + 1) % 4

                    if inn[k]:
                        poly.append(welder.weld_surface(srf, cu[k], cv[k]))

                    if inn[k] != inn[kn]:
                        t = (
                            fc[k] / (fc[k] - fc[kn])
                            if abs(fc[k] - fc[kn]) > 1e-30
                            else 0.5
                        )
                        u = cu[k] + (cu[kn] - cu[k]) * t
                        v = cv[k] + (cv[kn] - cv[k]) * t
                        u, v = _refine_crossing(srf, q, n, u, v)
                        poly.append(welder.weld_surface(srf, u, v))

                for t in range(1, len(poly) - 1):
                    a = poly[0]
                    b = poly[t]
                    c = poly[t + 1]

                    if a == b or b == c or c == a:
                        continue

                    result.add_face([a, b, c])

        if not result.face:
            return srf.mesh()

        return result

    def mesh_by_planes(
        self,
        planes: list[tuple[Point, Vector]],
        max_angle_deg: float,
        chord_factor: float,
    ) -> Mesh:
        """Mesh of the region inside every half-space (S-q).n <= 0: triangle soup clipped plane by plane, seams welded."""

        srf = self.m_surface
        pl = []

        for qn_q, qn_n in planes:
            n = _unit3(qn_n)

            if n is None:
                continue

            pl.append(([qn_q[0], qn_q[1], qn_q[2]], n))

        if not pl:
            return srf.mesh()

        usp = srf.get_span_vector(0)
        vsp = srf.get_span_vector(1)

        if len(usp) < 2 or len(vsp) < 2:
            return srf.mesh()

        bbox_diag = self._bbox_diagonal()
        chord_tol = bbox_diag * chord_factor
        us = _span_parameters(
            usp,
            _span_subdivisions(
                srf, 0, usp, vsp, srf.degree(0), max_angle_deg, chord_tol
            ),
        )
        vs = _span_parameters(
            vsp,
            _span_subdivisions(
                srf, 1, vsp, usp, srf.degree(1), max_angle_deg, chord_tol
            ),
        )
        nu = len(us)
        nv = len(vs)

        if nu < 2 or nv < 2:
            return srf.mesh()

        tris = []

        for i in range(nu - 1):
            for j in range(nv - 1):
                a = (us[i], vs[j])
                b = (us[i + 1], vs[j])
                c = (us[i + 1], vs[j + 1])
                d = (us[i], vs[j + 1])
                tris.append([a, b, c])
                tris.append([a, c, d])

        eps = 1e-9

        for k in range(len(pl)):
            q, n = pl[k]
            nxt = []

            for t in tris:
                poly = []

                for e in range(3):
                    p = t[e]
                    r = t[(e + 1) % 3]
                    fp = _plane_field(srf, q, n, p[0], p[1])
                    fr = _plane_field(srf, q, n, r[0], r[1])
                    pin = fp <= eps
                    rin = fr <= eps

                    if pin:
                        poly.append(p)

                    if pin != rin:
                        tt = fp / (fp - fr) if abs(fp - fr) > 1e-30 else 0.5
                        cu = p[0] + (r[0] - p[0]) * tt
                        cv = p[1] + (r[1] - p[1]) * tt
                        poly.append(_refine_crossing(srf, q, n, cu, cv))

                for w in range(1, len(poly) - 1):
                    nxt.append([poly[0], poly[w], poly[w + 1]])

            tris = nxt

            if not tris:
                break

        if not tris:
            return Mesh()

        result = Mesh()
        weld_tol = bbox_diag * 1e-5
        welder = _VertexWelder(result, weld_tol, weld_tol)

        for t in tris:
            a = welder.weld_surface(srf, t[0][0], t[0][1])
            b = welder.weld_surface(srf, t[1][0], t[1][1])
            c = welder.weld_surface(srf, t[2][0], t[2][1])

            if a == b or b == c or c == a:
                continue

            result.add_face([a, b, c])

        if not result.face:
            return Mesh()

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        data = {
            "guid": self.guid,
            "inner_loops": [],
            "name": self.name,
        }

        for loop in self.m_inner_loops:
            data["inner_loops"].append(loop.__jsondump__())

        if self.m_outer_loop.is_valid():
            data["outer_loop"] = self.m_outer_loop.__jsondump__()

        data["surface"] = self.m_surface.__jsondump__()
        data["surfacecolor"] = self.surfacecolor.__jsondump__()
        data["type"] = "NurbsSurfaceTrimmed"
        data["width"] = self.width

        return data

    @classmethod
    def __jsonload__(cls, data: dict) -> NurbsSurfaceTrimmed:
        """Deserialize from a JSON object."""

        ts = cls()

        if "guid" in data:
            ts.guid = data["guid"]

        if "name" in data:
            ts.name = data["name"]

        if "width" in data:
            ts.width = data["width"]

        if "surfacecolor" in data:
            ts.surfacecolor = Color.__jsonload__(data["surfacecolor"])

        if "surface" in data:
            ts.m_surface = NurbsSurface.__jsonload__(data["surface"])

        if "outer_loop" in data:
            ts.m_outer_loop = NurbsCurve.__jsonload__(data["outer_loop"])

        if "inner_loops" in data:
            for loop_data in data["inner_loops"]:
                ts.m_inner_loops.append(NurbsCurve.__jsonload__(loop_data))

        return ts

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> NurbsSurfaceTrimmed:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: str | Path) -> None:
        """Write to a JSON file."""

        with open(filepath, "w") as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: str | Path) -> NurbsSurfaceTrimmed:
        """Read from a JSON file."""

        with open(filepath) as f:
            return cls.__jsonload__(json.load(f))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> nurbssurface_trimmed_pb2.NurbsSurfaceTrimmed:
        """Convert to the protobuf message."""

        from .proto import nurbssurface_trimmed_pb2

        proto = nurbssurface_trimmed_pb2.NurbsSurfaceTrimmed()

        if self.has_guid():
            proto.guid = self.guid

        proto.name = self.name
        proto.width = self.width
        proto.surface.CopyFrom(self.m_surface.to_proto())

        if self.is_trimmed():
            proto.outer_loop.CopyFrom(self.m_outer_loop.to_proto())

        for inner in self.m_inner_loops:
            proto.inner_loops.add().CopyFrom(inner.to_proto())

        proto.surfacecolor.CopyFrom(self.surfacecolor.to_proto())

        return proto

    @classmethod
    def from_proto(cls, proto: nurbssurface_trimmed_pb2.NurbsSurfaceTrimmed) -> NurbsSurfaceTrimmed:
        """Construct from the protobuf message."""

        ts = cls()

        if proto.guid:
            ts.guid = proto.guid

        ts.name = proto.name
        ts.width = proto.width

        if proto.HasField("surface"):
            ts.m_surface = NurbsSurface.from_proto(proto.surface)

        if proto.HasField("outer_loop"):
            ts.m_outer_loop = NurbsCurve.from_proto(proto.outer_loop)

        for loop in proto.inner_loops:
            ts.m_inner_loops.append(NurbsCurve.from_proto(loop))

        ts.surfacecolor = Color.from_proto(proto.surfacecolor)

        return ts

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> NurbsSurfaceTrimmed:
        """Deserialize from protobuf bytes."""

        from .proto import nurbssurface_trimmed_pb2

        proto = nurbssurface_trimmed_pb2.NurbsSurfaceTrimmed()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filepath: str | Path) -> None:
        """Write to a protobuf file."""

        with open(filepath, "wb") as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: str | Path) -> NurbsSurfaceTrimmed:
        """Read from a protobuf file."""

        with open(filepath, "rb") as f:
            return cls.pb_loads(f.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "NurbsSurfaceTrimmed(name=..., trimmed=..., holes=...)"."""

        trimmed = "true" if self.is_trimmed() else "false"

        return f"NurbsSurfaceTrimmed(name={self.name}, trimmed={trimmed}, holes={self.inner_loop_count()})"

    def __repr__(self) -> str:
        """Return the multi-line form with the surface."""

        trimmed = "true" if self.is_trimmed() else "false"

        return f"NurbsSurfaceTrimmed(\n  name={self.name},\n  trimmed={trimmed},\n  holes={self.inner_loop_count()},\n  surface={self.m_surface}\n)"

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════════════════════════
    def _bbox_diagonal(self) -> float:
        """Return the diagonal of the control-point box, the scale every deflection tolerance is a fraction of."""

        bmin = [1e30, 1e30, 1e30]
        bmax = [-1e30, -1e30, -1e30]

        for i in range(self.m_surface.cv_count(0)):
            for j in range(self.m_surface.cv_count(1)):
                p = self.m_surface.get_cv(i, j)

                for k in range(3):
                    c = p[k]

                    if c < bmin[k]:
                        bmin[k] = c

                    if c > bmax[k]:
                        bmax[k] = c

        bbox_diag = math.sqrt(
            (bmax[0] - bmin[0]) ** 2
            + (bmax[1] - bmin[1]) ** 2
            + (bmax[2] - bmin[2]) ** 2
        )

        return 1.0 if bbox_diag < 1e-12 else bbox_diag

    def _discretize_loop(self, crv: NurbsCurve, deflection: float) -> list[Point]:
        """UV polygon of a trim loop: control points or samples, each edge split until its 3D chord is within deflection."""

        raw = []

        if crv.degree() <= 1 and not crv.is_rational():
            for i in range(crv.cv_count()):
                raw.append(crv.get_cv(i))
        else:
            n = min(max(crv.cv_count() * 4, 16), 2048)
            raw = crv.divide_by_count(n)[0]

        while len(raw) > 1:
            dx = raw[0][0] - raw[-1][0]
            dy = raw[0][1] - raw[-1][1]

            if dx * dx + dy * dy < 1e-20:
                raw.pop()
            else:
                break

        if len(raw) < 2:
            return raw

        out = []

        for i in range(len(raw)):
            stack = [(raw[i], raw[(i + 1) % len(raw)], 0)]

            while stack:
                a, b, depth = stack.pop()
                mu = (a[0] + b[0]) * 0.5
                mv = (a[1] + b[1]) * 0.5
                pa = self.m_surface.point_at(a[0], a[1])
                pm = self.m_surface.point_at(mu, mv)
                edge = self.m_surface.point_at(b[0], b[1]) - pa
                l2 = edge.magnitude_squared()

                if l2 > 1e-30:
                    t = (pm - pa).dot(edge) / l2
                    dev = math.sqrt((pm - (pa + edge * t)).magnitude_squared())
                else:
                    dev = math.sqrt((pm - pa).magnitude_squared())

                if dev > deflection and depth < 6:
                    stack.append((Point(mu, mv, 0.0), b, depth + 1))
                    stack.append((a, Point(mu, mv, 0.0), depth + 1))
                else:
                    out.append(a)

        return out

    def _triangulate(
        self, loops: TrimLoops, max_angle_deg: float, chord_factor: float
    ) -> Mesh:
        """Constrained Delaunay of the loops in UV, refined, trimmed, lifted and welded: the one body mesh_q and mesh_loops share."""

        from .remesh_nurbssurface_grid import RemeshNurbsSurfaceGrid

        if not loops.uv or len(loops.uv[0]) < 3:
            return self.m_surface.mesh()

        outer_uv = loops.uv[0]
        bbox_diag = self._bbox_diagonal()
        deflection = bbox_diag * chord_factor
        cos_max_angle = math.cos(min(max(max_angle_deg, 0.1), 179.0) * PI / 180.0)

        bb_umin = 1e30
        bb_vmin = 1e30
        bb_umax = -1e30
        bb_vmax = -1e30

        for p in outer_uv:
            if p[0] < bb_umin:
                bb_umin = p[0]

            if p[1] < bb_vmin:
                bb_vmin = p[1]

            if p[0] > bb_umax:
                bb_umax = p[0]

            if p[1] > bb_vmax:
                bb_vmax = p[1]

        crease_knots = [[], []]

        for direction in range(2):
            domain = self.m_surface.domain(direction)
            knots = self.m_surface.m_nurbsknot[direction]

            for knot in knots:
                if knot <= domain[0] or knot >= domain[1] or knot in crease_knots[direction]:
                    continue

                multiplicity = 0

                for value in knots:
                    if value == knot:
                        multiplicity += 1

                if multiplicity >= self.m_surface.degree(direction):
                    crease_knots[direction].append(knot)

        dt = _Delaunay2D(bb_umin, bb_vmin, bb_umax, bb_vmax)
        loop_vids = []
        boundary_intervals = {}

        for li, pts in enumerate(loops.uv):
            vis = []

            for p in pts:
                vis.append(dt.insert(p[0], p[1]))

            for i in range(len(vis)):
                j = (i + 1) % len(vis)
                events = [(0.0, vis[i]), (1.0, vis[j])]

                for direction in range(2):
                    delta = pts[j][direction] - pts[i][direction]

                    if delta == 0.0:
                        continue

                    for knot in crease_knots[direction]:
                        t = (knot - pts[i][direction]) / delta

                        if t <= 0.0 or t >= 1.0:
                            continue

                        uv = [
                            pts[i][0] + t * (pts[j][0] - pts[i][0]),
                            pts[i][1] + t * (pts[j][1] - pts[i][1]),
                        ]
                        uv[direction] = knot
                        vi = dt.insert(uv[0], uv[1])

                        if vi >= 0:
                            boundary_intervals[vi] = (li, i, t)

                        events.append((t, vi))

                events.sort()

                for k in range(1, len(events)):
                    if (
                        events[k - 1][1] >= 0
                        and events[k][1] >= 0
                        and events[k - 1][1] != events[k][1]
                    ):
                        dt.insert_constraint(events[k - 1][1], events[k][1])

            loop_vids.append(vis)

        for u in crease_knots[0]:
            for v in crease_knots[1]:
                if _inside_loops(u, v, loops.uv):
                    dt.insert(u, v)

        for direction in range(2):
            for knot in crease_knots[direction]:
                nodes = []

                for vi, vertex in enumerate(dt.vertices):
                    uv = [vertex.x, vertex.y]

                    if uv[direction] == knot:
                        nodes.append((uv[1 - direction], vi))

                nodes.sort()

                for k in range(1, len(nodes)):
                    uv = [knot, knot]
                    uv[1 - direction] = (nodes[k - 1][0] + nodes[k][0]) * 0.5

                    if _inside_loops(uv[0], uv[1], loops.uv):
                        dt.insert_constraint(nodes[k - 1][1], nodes[k][1])

        for p in loops.interior_uv:
            if _inside_loops(p[0], p[1], loops.uv):
                dt.insert(p[0], p[1])

        MAX_ITERS = 8
        MAX_VERTS = 200000
        iters = MAX_ITERS

        for _iter in range(iters):
            to_insert = []

            for tri in dt.triangles:
                if not tri.alive:
                    continue

                a = dt.vertices[tri.v[0]]
                b = dt.vertices[tri.v[1]]
                c = dt.vertices[tri.v[2]]
                cu = (a.x + b.x + c.x) / 3.0
                cv = (a.y + b.y + c.y) / 3.0

                if not _inside_loops(cu, cv, loops.uv):
                    continue

                pa = self.m_surface.point_at(a.x, a.y)
                pb = self.m_surface.point_at(b.x, b.y)
                pc = self.m_surface.point_at(c.x, c.y)
                pm = self.m_surface.point_at(cu, cv)
                n = (pb - pa).cross(pc - pa)
                nl = math.sqrt(n.magnitude_squared())

                if nl < 1e-30:
                    continue

                dev = abs((pm - pa).dot(n) / nl)
                refine = dev > deflection

                if not refine:
                    na = _crease_side_normal(
                        self.m_surface, crease_knots, (cu, cv), [a.x, a.y]
                    )
                    nb = _crease_side_normal(
                        self.m_surface, crease_knots, (cu, cv), [b.x, b.y]
                    )
                    nc2 = _crease_side_normal(
                        self.m_surface, crease_knots, (cu, cv), [c.x, c.y]
                    )
                    d1 = na.dot(nb)
                    d2 = nb.dot(nc2)
                    d3 = na.dot(nc2)
                    mind = min(d1, min(d2, d3))

                    if mind < cos_max_angle:
                        refine = True

                if refine:
                    to_insert.append((cu, cv))

            if not to_insert:
                break

            for cu, cv in to_insert:
                if len(dt.vertices) >= MAX_VERTS:
                    break

                dt.insert(cu, cv)

            if len(dt.vertices) >= MAX_VERTS:
                break

        dt.cleanup()

        for ti in range(len(dt.triangles)):
            if not dt.triangles[ti].alive:
                continue

            v0, v1, v2 = dt.triangles[ti].v
            cu = (dt.vertices[v0].x + dt.vertices[v1].x + dt.vertices[v2].x) / 3.0
            cv = (dt.vertices[v0].y + dt.vertices[v1].y + dt.vertices[v2].y) / 3.0

            if not _inside_loops(cu, cv, loops.uv):
                dt.triangles[ti].alive = False

        tris = dt.get_triangles()

        if not tris:
            return Mesh()

        for tri in tris:
            for direction in range(2):
                low = math.inf
                high = -math.inf

                for vi in tri:
                    value = dt.vertices[vi].x if direction == 0 else dt.vertices[vi].y
                    low = min(low, value)
                    high = max(high, value)

                for knot in crease_knots[direction]:
                    if low < knot < high:
                        return Mesh()

        nverts = len(dt.vertices)
        given = [None] * nverts

        for li, vids in enumerate(loop_vids):
            if li >= len(loops.xyz):
                break

            for k, vi in enumerate(vids):
                if vi >= 0 and k < len(loops.xyz[li]):
                    given[vi] = (li, k)

        result = Mesh()
        vert_map = [None] * nverts
        weld_tol = bbox_diag * 1e-5 if not loops.xyz else 0.0
        welder = _VertexWelder(result, weld_tol, bbox_diag * 1e-5)

        for tri in tris:
            for vi in tri:
                if vert_map[vi] is not None:
                    continue

                if given[vi] is not None:
                    p = loops.xyz[given[vi][0]][given[vi][1]]
                elif vi in boundary_intervals and loops.xyz:
                    li, segment, t = boundary_intervals[vi]
                    a = loops.xyz[li][segment]
                    b = loops.xyz[li][(segment + 1) % len(loops.xyz[li])]
                    p = a + (b - a) * t
                else:
                    p = self.m_surface.point_at(dt.vertices[vi].x, dt.vertices[vi].y)

                vert_map[vi] = welder.weld(p)

        for a, b, c in tris:
            v0 = vert_map[a]
            v1 = vert_map[b]
            v2 = vert_map[c]

            if v0 == v1 or v1 == v2 or v2 == v0:
                continue

            result.add_face([v0, v1, v2])

        fan = {}

        for fk in sorted(result.face):
            verts = result.face[fk]
            a = result.vertex[verts[0]].position()
            b = result.vertex[verts[1]].position()
            c = result.vertex[verts[2]].position()
            n = (b - a).cross(c - a)

            for vk in verts:
                if vk not in fan:
                    fan[vk] = Vector(0.0, 0.0, 0.0)

                fan[vk] += n

        for vi in range(nverts):
            if vert_map[vi] is not None:
                u = dt.vertices[vi].x
                v = dt.vertices[vi].y
                derivatives = self.m_surface.evaluate(u, v, 1)
                nrm = Vector(0.0, 0.0, 0.0)

                if len(derivatives) >= 3:
                    nrm = derivatives[2].cross(derivatives[1])

                nl = math.sqrt(nrm.magnitude_squared())

                if math.isfinite(nl) and nl > 0.0:
                    nrm = nrm / nl
                else:
                    f = fan.get(vert_map[vi], Vector(0.0, 0.0, 1.0))
                    fl = math.sqrt(f.magnitude_squared())
                    nrm = f / fl if math.isfinite(fl) and fl > 0.0 else Vector(0.0, 0.0, 1.0)

                vd = result.vertex[vert_map[vi]]
                vd.set_normal(nrm[0], nrm[1], nrm[2])
                vd.attributes["u"] = u
                vd.attributes["v"] = v

        for li, vids in enumerate(loop_vids):
            for k, vi in enumerate(vids):
                key = f"boundary/{li}/{k}"

                if vi >= 0 and vert_map[vi] is not None:
                    result.vertex[vert_map[vi]].attributes[key] = 1.0

        for vi in sorted(boundary_intervals):
            li, segment, t = boundary_intervals[vi]
            key = f"boundary_interval/{li}/{segment}"

            if vert_map[vi] is not None:
                result.vertex[vert_map[vi]].attributes[key] = t

        RemeshNurbsSurfaceGrid._split_crease_normals(self.m_surface, result)

        return result
