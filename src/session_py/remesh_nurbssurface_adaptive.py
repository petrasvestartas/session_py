from __future__ import annotations
from typing import TYPE_CHECKING
import math
from bisect import bisect_left
from bisect import bisect_right

from .tolerance import PI
from .point import Point
from .vector import Vector

if TYPE_CHECKING:
    from .nurbssurface import NurbsSurface
    from .mesh import Mesh

MAX_DEPTH = 8
STACK_SIZE = 64
KEY_SCALE = 1e10


class _Corner:
    """Surface sample: position and unit normal, zero where the surface has none"""

    def __init__(self, p: Point, n: Vector):
        self.p = p
        self.n = n


class _Node:
    """Quadtree cell: UV bounds, corners SW SE NE NW and the centre"""

    def __init__(
        self, u0: float, v0: float, u1: float, v1: float, c: list[_Corner], depth: int
    ):
        self.u0 = u0
        self.v0 = v0
        self.u1 = u1
        self.v1 = v1
        self.c = c
        self.depth = depth
        self.leaf = True


class _Quadtree:
    """Quadtree over the UV domain: the surface it samples, its tolerances, the cells, the leaf corners by key and the mesh it emits"""

    def __init__(
        self,
        s: NurbsSurface,
        norm_tol: float,
        chord_tol: float,
        max_edge: float,
        min_edge: float,
    ):
        from .mesh import Mesh

        self.s = s
        self.usp = s.get_span_vector(0)
        self.vsp = s.get_span_vector(1)
        self.closed = [s.is_closed(0), s.is_closed(1)]
        self.norm_tol = norm_tol
        self.chord_tol = chord_tol
        self.max_edge = max_edge
        self.min_edge = min_edge
        self.nodes: list[_Node] = []
        self.corners: dict[tuple[int, int], _Corner] = {}
        self.rows: dict[int, list[int]] = {}
        self.cols: dict[int, list[int]] = {}
        self.mesh = Mesh()
        self.keys: dict[tuple[int, int], int] = {}
        self.south: int | None = None
        self.north: int | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════


def _norm(v: Vector) -> float:
    """Euclidean length without the zero gate of magnitude()"""
    return math.sqrt(v.magnitude_squared())


def _dist2(a: Point, b: Point) -> float:
    """Squared distance between two points"""
    return (a - b).magnitude_squared()


def _midpoint(a: Point, b: Point) -> Point:
    """Midpoint of two points"""
    return Point.sum(a, b) * 0.5


def _bbox_diagonal(s: NurbsSurface) -> float:
    """Diagonal of the control point bounding box"""
    lo = Point(1e30, 1e30, 1e30)
    hi = Point(-1e30, -1e30, -1e30)
    for i in range(s.cv_count(0)):
        for j in range(s.cv_count(1)):
            p = s.get_cv(i, j)
            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])
    return _norm(hi - lo)


def _sample(s: NurbsSurface, u: float, v: float) -> _Corner:
    """Point and unit normal Su x Sv at (u, v); zero normal at a pole, where normal_at would give +Z"""
    c = _Corner(s.point_at(u, v), Vector(0.0, 0.0, 0.0))
    derivatives = s.evaluate(u, v, 1)
    if len(derivatives) < 3:
        return c
    n = derivatives[2].cross(derivatives[1])
    length = _norm(n)
    if length > 1e-10:
        c.n = n / length
    return c


def _make_node(
    s: NurbsSurface,
    u0: float,
    v0: float,
    u1: float,
    v1: float,
    corners: list[_Corner],
    depth: int,
) -> _Node:
    """Leaf cell over [u0, u1] x [v0, v1] from its sampled corners SW SE NE NW, centre sampled here"""
    centre = _sample(s, (u0 + u1) * 0.5, (v0 + v1) * 0.5)
    return _Node(
        u0, v0, u1, v1, [corners[0], corners[1], corners[2], corners[3], centre], depth
    )


def _sample_edges(s: NurbsSurface, p: _Node) -> list[_Corner]:
    """Edge midpoints S, E, N, W of the cell"""
    um = (p.u0 + p.u1) * 0.5
    vm = (p.v0 + p.v1) * 0.5
    return [
        _sample(s, um, p.v0),
        _sample(s, p.u1, vm),
        _sample(s, um, p.v1),
        _sample(s, p.u0, vm),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Splitting
# ═══════════════════════════════════════════════════════════════════════════


def _normals_turn(a: _Corner, b: _Corner, norm_tol: float) -> bool:
    """True when both normals exist and differ by more than norm_tol in squared length"""
    if a.n.magnitude_squared() <= 1e-20 or b.n.magnitude_squared() <= 1e-20:
        return False
    return (a.n - b.n).magnitude_squared() > norm_tol


def _chord_off(mid: _Corner, a: _Corner, b: _Corner, chord_tol: float) -> bool:
    """True when the edge midpoint sits more than chord_tol off the chord from a to b"""
    return _dist2(mid.p, _midpoint(a.p, b.p)) > chord_tol * chord_tol


def _too_short(p: _Node, min_edge: float) -> bool:
    """True when every edge is shorter than min_edge, so the cell is not split further"""
    if min_edge <= 0.0:
        return False
    longest = 0.0
    for i in range(4):
        longest = max(longest, _dist2(p.c[i].p, p.c[(i + 1) % 4].p))
    return longest < min_edge * min_edge


def _twisted(p: _Node, chord_tol: float) -> bool:
    """True when the centre sits more than twice chord_tol off a diagonal midpoint; never on a cell with a collapsed edge"""
    for i in range(4):
        if _dist2(p.c[i].p, p.c[(i + 1) % 4].p) < chord_tol * chord_tol:
            return False
    twist_tol2 = 4.0 * chord_tol * chord_tol
    return (
        _dist2(p.c[4].p, _midpoint(p.c[0].p, p.c[2].p)) > twist_tol2
        or _dist2(p.c[4].p, _midpoint(p.c[1].p, p.c[3].p)) > twist_tol2
    )


def _curved(s: NurbsSurface, p: _Node, chord_tol: float) -> tuple[bool, bool]:
    """Per direction, true when the arc at the centre, taken as a circle of its normal curvature, rises more than chord_tol over its chord"""
    d = s.evaluate((p.u0 + p.u1) * 0.5, (p.v0 + p.v1) * 0.5, 2)
    if len(d) < 6:
        return False, False
    su = d[3]
    sv = d[1]
    suu = d[5]
    svv = d[2]
    n = su.cross(sv)
    length = _norm(n)
    su2 = su.magnitude_squared()
    sv2 = sv.magnitude_squared()
    if length <= 1e-10 or su2 <= 1e-20 or sv2 <= 1e-20:
        return False, False
    unit = n * (1.0 / length)
    kappa_u = abs(suu.dot(unit)) / su2
    kappa_v = abs(svv.dot(unit)) / sv2
    span_u = math.sqrt(su2) * (p.u1 - p.u0)
    span_v = math.sqrt(sv2) * (p.v1 - p.v0)
    curved_u = kappa_u > 1e-20 and span_u * math.sqrt(kappa_u / (8.0 * chord_tol)) > 1.0
    curved_v = kappa_v > 1e-20 and span_v * math.sqrt(kappa_v / (8.0 * chord_tol)) > 1.0
    return curved_u, curved_v


def _split_flags(q: _Quadtree, p: _Node, mids: list[_Corner]) -> tuple[bool, bool]:
    """Directions to split: normals turning along an edge or from a corner to its midpoint, a midpoint off its chord, a twisted centre, an edge past max_edge, or the curvature at the centre"""
    sw = p.c[0]
    se = p.c[1]
    ne = p.c[2]
    nw = p.c[3]
    split_u = _normals_turn(sw, se, q.norm_tol) or _normals_turn(ne, nw, q.norm_tol)
    split_v = _normals_turn(se, ne, q.norm_tol) or _normals_turn(nw, sw, q.norm_tol)
    split_u = (
        split_u
        or _chord_off(mids[0], sw, se, q.chord_tol)
        or _chord_off(mids[2], nw, ne, q.chord_tol)
    )
    split_u = (
        split_u
        or _normals_turn(mids[0], sw, q.norm_tol)
        or _normals_turn(mids[2], nw, q.norm_tol)
    )
    split_v = (
        split_v
        or _chord_off(mids[3], sw, nw, q.chord_tol)
        or _chord_off(mids[1], se, ne, q.chord_tol)
    )
    split_v = (
        split_v
        or _normals_turn(mids[3], sw, q.norm_tol)
        or _normals_turn(mids[1], se, q.norm_tol)
    )
    if not split_u and not split_v and _twisted(p, q.chord_tol):
        split_u = True
        split_v = True
    if q.max_edge > 0.0:
        limit = q.max_edge * q.max_edge
        split_u = split_u or _dist2(sw.p, se.p) > limit or _dist2(ne.p, nw.p) > limit
        split_v = split_v or _dist2(se.p, ne.p) > limit or _dist2(nw.p, sw.p) > limit
    if not split_u or not split_v:
        curved_u, curved_v = _curved(q.s, p, q.chord_tol)
        split_u = split_u or curved_u
        split_v = split_v or curved_v
    return split_u, split_v


def _split_node(
    q: _Quadtree, idx: int, mids: list[_Corner], split_u: bool, split_v: bool
) -> None:
    """Children of cell idx appended to the pool: four quadrants, or two halves along the split direction"""
    p = q.nodes[idx]
    um = (p.u0 + p.u1) * 0.5
    vm = (p.v0 + p.v1) * 0.5
    depth = p.depth + 1
    p.leaf = False
    if split_u and split_v:
        q.nodes.append(
            _make_node(
                q.s, p.u0, p.v0, um, vm, [p.c[0], mids[0], p.c[4], mids[3]], depth
            )
        )
        q.nodes.append(
            _make_node(
                q.s, um, p.v0, p.u1, vm, [mids[0], p.c[1], mids[1], p.c[4]], depth
            )
        )
        q.nodes.append(
            _make_node(
                q.s, um, vm, p.u1, p.v1, [p.c[4], mids[1], p.c[2], mids[2]], depth
            )
        )
        q.nodes.append(
            _make_node(
                q.s, p.u0, vm, um, p.v1, [mids[3], p.c[4], mids[2], p.c[3]], depth
            )
        )
    elif split_u:
        q.nodes.append(
            _make_node(
                q.s, p.u0, p.v0, um, p.v1, [p.c[0], mids[0], mids[2], p.c[3]], depth
            )
        )
        q.nodes.append(
            _make_node(
                q.s, um, p.v0, p.u1, p.v1, [mids[0], p.c[1], p.c[2], mids[2]], depth
            )
        )
    else:
        q.nodes.append(
            _make_node(
                q.s, p.u0, p.v0, p.u1, vm, [p.c[0], p.c[1], mids[1], mids[3]], depth
            )
        )
        q.nodes.append(
            _make_node(
                q.s, p.u0, vm, p.u1, p.v1, [mids[3], mids[1], p.c[2], p.c[3]], depth
            )
        )


def _subdivide(q: _Quadtree, root: int) -> None:
    """Cells split from root down to MAX_DEPTH over an explicit stack, first child popped first so the pool fills depth first"""
    stack: list[int] = []
    stack.append(root)
    while len(stack) > 0:
        idx = stack.pop()
        p = q.nodes[idx]
        if p.depth >= MAX_DEPTH or _too_short(p, q.min_edge):
            continue
        mids = _sample_edges(q.s, p)
        split_u, split_v = _split_flags(q, p, mids)
        if not split_u and not split_v:
            continue
        first = len(q.nodes)
        _split_node(q, idx, mids, split_u, split_v)
        count = len(q.nodes) - first
        assert len(stack) + count <= STACK_SIZE
        for i in range(count - 1, -1, -1):
            stack.append(first + i)


def _build(q: _Quadtree) -> None:
    """One root cell per span pair, corners from the grid of span intersections, each subdivided before the next"""
    nu = len(q.usp)
    nv = len(q.vsp)
    grid: list[_Corner] = []
    for i in range(nu):
        for j in range(nv):
            grid.append(_sample(q.s, q.usp[i], q.vsp[j]))
    for i in range(nu - 1):
        for j in range(nv - 1):
            root = len(q.nodes)
            q.nodes.append(
                _make_node(
                    q.s,
                    q.usp[i],
                    q.vsp[j],
                    q.usp[i + 1],
                    q.vsp[j + 1],
                    [
                        grid[i * nv + j],
                        grid[(i + 1) * nv + j],
                        grid[(i + 1) * nv + j + 1],
                        grid[i * nv + j + 1],
                    ],
                    0,
                )
            )
            _subdivide(q, root)


# ═══════════════════════════════════════════════════════════════════════════
# Vertices and faces
# ═══════════════════════════════════════════════════════════════════════════


def _quantize(t: float) -> int:
    """t rounded at KEY_SCALE"""
    return round(t * KEY_SCALE)


def _wrap(closed: bool, sp: list[float], t: float) -> float:
    """t at the seam of a closed direction maps to the start of sp"""
    return sp[0] if closed and abs(t - sp[-1]) < 1e-10 else t


def _sort_unique(line: list[int]) -> list[int]:
    """Sorted without repeats"""
    return sorted(set(line))


def _index_leaves(q: _Quadtree) -> None:
    """Every leaf corner by key, and the u keys on each row and v keys on each column for the T-junction search"""
    for nd in q.nodes:
        if not nd.leaf:
            continue
        us = [nd.u0, nd.u1, nd.u1, nd.u0]
        vs = [nd.v0, nd.v0, nd.v1, nd.v1]
        for ci in range(4):
            key = (
                _quantize(_wrap(q.closed[0], q.usp, us[ci])),
                _quantize(_wrap(q.closed[1], q.vsp, vs[ci])),
            )
            q.corners.setdefault(key, nd.c[ci])
            q.rows.setdefault(key[1], []).append(_quantize(us[ci]))
            q.cols.setdefault(key[0], []).append(_quantize(vs[ci]))
    for key in q.rows:
        q.rows[key] = _sort_unique(q.rows[key])
    for key in q.cols:
        q.cols[key] = _sort_unique(q.cols[key])


def _between(
    lines: dict[int, list[int]], line: int, t0: float, t1: float
) -> list[float]:
    """Parameters on one line strictly between t0 and t1, in walk order from t0 to t1"""
    result: list[float] = []
    if line not in lines:
        return result
    lo = min(_quantize(t0), _quantize(t1))
    hi = max(_quantize(t0), _quantize(t1))
    first = bisect_right(lines[line], lo)
    last = bisect_left(lines[line], hi)
    for k in range(first, last):
        result.append(lines[line][k] / KEY_SCALE)
    if t0 > t1:
        result.reverse()
    return result


def _row_mids(q: _Quadtree, u0: float, u1: float, v: float) -> list[float]:
    """T-junction parameters between u0 and u1 on the row at v"""
    return _between(q.rows, _quantize(_wrap(q.closed[1], q.vsp, v)), u0, u1)


def _col_mids(q: _Quadtree, u: float, v0: float, v1: float) -> list[float]:
    """T-junction parameters between v0 and v1 on the column at u"""
    return _between(q.cols, _quantize(_wrap(q.closed[0], q.usp, u)), v0, v1)


def _vertex_at(q: _Quadtree, u: float, v: float) -> int:
    """Mesh vertex at (u, v): the pole on a singular side, else one per key, sampled when no leaf corner holds it"""
    if q.south is not None and abs(v - q.vsp[0]) < 1e-10:
        return q.south
    if q.north is not None and abs(v - q.vsp[-1]) < 1e-10:
        return q.north
    uw = _wrap(q.closed[0], q.usp, u)
    vw = _wrap(q.closed[1], q.vsp, v)
    key = (_quantize(uw), _quantize(vw))
    if key in q.keys:
        return q.keys[key]
    if key not in q.corners:
        q.corners[key] = _sample(q.s, uw, vw)
    vertex = q.mesh.add_vertex(q.corners[key].p)
    q.mesh.vertex[vertex].attributes["u"] = uw
    q.mesh.vertex[vertex].attributes["v"] = vw
    q.keys[key] = vertex
    return vertex


def _leaf_polygon(q: _Quadtree, nd: _Node) -> list[int]:
    """Vertices counter-clockwise around the leaf with the T-junction vertices on each edge, repeats at poles and seams dropped"""
    poly: list[int] = []
    poly.append(_vertex_at(q, nd.u0, nd.v0))
    for u in _row_mids(q, nd.u0, nd.u1, nd.v0):
        poly.append(_vertex_at(q, u, nd.v0))
    poly.append(_vertex_at(q, nd.u1, nd.v0))
    for v in _col_mids(q, nd.u1, nd.v0, nd.v1):
        poly.append(_vertex_at(q, nd.u1, v))
    poly.append(_vertex_at(q, nd.u1, nd.v1))
    for u in _row_mids(q, nd.u1, nd.u0, nd.v1):
        poly.append(_vertex_at(q, u, nd.v1))
    poly.append(_vertex_at(q, nd.u0, nd.v1))
    for v in _col_mids(q, nd.u0, nd.v1, nd.v0):
        poly.append(_vertex_at(q, nd.u0, v))
    unique: list[int] = []
    for vertex in poly:
        if len(unique) == 0 or unique[-1] != vertex:
            unique.append(vertex)
    while len(unique) > 1 and unique[0] == unique[-1]:
        unique.pop()
    return unique


def _add_leaf_faces(q: _Quadtree, nd: _Node) -> None:
    """Faces of one leaf: a triangle, a quad cut along its shorter diagonal, or a fan around the centre once T-junctions add vertices"""
    poly = _leaf_polygon(q, nd)
    n = len(poly)
    if n < 3:
        return
    if n == 3:
        q.mesh.add_face([poly[0], poly[1], poly[2]])
        return
    if n == 4:
        p0 = q.mesh.vertex[poly[0]].position()
        p1 = q.mesh.vertex[poly[1]].position()
        p2 = q.mesh.vertex[poly[2]].position()
        p3 = q.mesh.vertex[poly[3]].position()
        if _dist2(p0, p2) <= _dist2(p1, p3):
            q.mesh.add_face([poly[0], poly[1], poly[2]])
            q.mesh.add_face([poly[0], poly[2], poly[3]])
        else:
            q.mesh.add_face([poly[0], poly[1], poly[3]])
            q.mesh.add_face([poly[1], poly[2], poly[3]])
        return
    cu = _wrap(q.closed[0], q.usp, (nd.u0 + nd.u1) * 0.5)
    cv = _wrap(q.closed[1], q.vsp, (nd.v0 + nd.v1) * 0.5)
    q.corners.setdefault((_quantize(cu), _quantize(cv)), nd.c[4])
    centre = _vertex_at(q, cu, cv)
    for i in range(n):
        j = (i + 1) % n
        if poly[i] != poly[j] and poly[i] != centre and poly[j] != centre:
            q.mesh.add_face([poly[i], poly[j], centre])


# ═══════════════════════════════════════════════════════════════════════════
# Normals
# ═══════════════════════════════════════════════════════════════════════════


def _fan_normals(mesh: Mesh) -> list[Vector]:
    """Sum of the unnormalized face normals around each vertex key, faces taken in key order"""
    sums = [Vector(0.0, 0.0, 0.0) for key in range(len(mesh.vertex))]
    for key in sorted(mesh.face):
        vertices = mesh.face[key]
        p0 = mesh.vertex[vertices[0]].position()
        p1 = mesh.vertex[vertices[1]].position()
        p2 = mesh.vertex[vertices[2]].position()
        n = (p1 - p0).cross(p2 - p0)
        for vertex in vertices:
            sums[vertex] += n
    return sums


def _set_normals(mesh: Mesh) -> None:
    """Unit fan normal on every vertex, zero where the fan cancels"""
    sums = _fan_normals(mesh)
    for key, vd in mesh.vertex.items():
        length = _norm(sums[key])
        n = sums[key] / length if length > 1e-15 else sums[key]
        vd.set_normal(n[0], n[1], n[2])


# ═══════════════════════════════════════════════════════════════════════════
# RemeshNurbsSurfaceAdaptive
# ═══════════════════════════════════════════════════════════════════════════


class RemeshNurbsSurfaceAdaptive:
    """Adaptive mesh of a NURBS surface: a quadtree in UV split where normals turn or chords deviate, T-junctions fanned, poles and seams shared"""

    def __init__(self, surface: NurbsSurface):
        self._surface = surface
        self._max_angle = 20.0
        self._max_edge_length = 0.0
        self._min_edge_length = 0.0
        self._max_chord_height = 0.0

    def set_max_angle(self, degrees: float) -> RemeshNurbsSurfaceAdaptive:
        """Largest normal turn across a cell in degrees, 20 by default"""
        self._max_angle = degrees
        return self

    def set_max_edge_length(self, length: float) -> RemeshNurbsSurfaceAdaptive:
        """Longest cell edge; 0 for no limit"""
        self._max_edge_length = length
        return self

    def set_min_edge_length(self, length: float) -> RemeshNurbsSurfaceAdaptive:
        """Shortest cell edge still split; 0 for no limit"""
        self._min_edge_length = length
        return self

    def set_max_chord_height(self, height: float) -> RemeshNurbsSurfaceAdaptive:
        """Largest chord height; 0 for 0.5 percent of the bbox diagonal"""
        self._max_chord_height = height
        return self

    def get_max_angle(self) -> float:
        return self._max_angle

    def get_max_edge_length(self) -> float:
        return self._max_edge_length

    def get_min_edge_length(self) -> float:
        return self._min_edge_length

    def get_max_chord_height(self) -> float:
        return self._max_chord_height

    def mesh(self) -> Mesh:
        """Triangle mesh with u, v vertex attributes and fan normals"""
        norm_tol = 2.0 - 2.0 * math.cos(self._max_angle * PI / 180.0)
        chord_tol = (
            self._max_chord_height
            if self._max_chord_height > 0.0
            else _bbox_diagonal(self._surface) * 0.005
        )
        q = _Quadtree(
            self._surface,
            norm_tol,
            chord_tol,
            self._max_edge_length,
            self._min_edge_length,
        )
        _build(q)
        _index_leaves(q)
        if self._surface.is_singular(0):
            q.south = q.mesh.add_vertex(self._surface.point_at(q.usp[0], q.vsp[0]))
        if self._surface.is_singular(2):
            q.north = q.mesh.add_vertex(self._surface.point_at(q.usp[0], q.vsp[-1]))
        for nd in q.nodes:
            if nd.leaf:
                _add_leaf_faces(q, nd)
        _set_normals(q.mesh)
        return q.mesh
