from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from .brep import BRep
from .brep import BRepOrientation
from .brep import BRepRef
from .closest import Closest
from .line import Line
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .point import Point
from .polyline import Polyline
from .tolerance import Tolerance
from .vector import Vector

_EPSILON = Tolerance.ZERO_TOLERANCE
_FORWARD = BRepOrientation.Forward
_REVERSED = BRepOrientation.Reversed
_WORK_LIMIT = 200000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_tolerance(tolerance: float) -> None:
    _require(
        math.isfinite(tolerance) and tolerance > 0.0,
        "Split tolerance must be finite and positive",
    )


def _check_curve(curve: NurbsCurve) -> None:
    _require(curve.is_valid(), "Split requires valid curves")
    for i in range(curve.cv_count()):
        p = curve.get_cv(i)
        _require(
            math.isfinite(p[0])
            and math.isfinite(p[1])
            and math.isfinite(p[2])
            and math.isfinite(curve.weight(i))
            and curve.weight(i) > 0.0,
            "Split requires finite controls and positive rational weights",
        )


def _check_surface(surface: NurbsSurface) -> None:
    _require(surface.is_valid(), "Split requires a valid NURBS surface")
    for i in range(surface.cv_count(0)):
        for j in range(surface.cv_count(1)):
            p = surface.get_cv(i, j)
            w = surface.weight(i, j)
            _require(
                math.isfinite(p[0])
                and math.isfinite(p[1])
                and math.isfinite(p[2])
                and math.isfinite(w)
                and w > 0.0,
                "Split requires finite surface controls and positive rational weights",
            )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _interval(curve: NurbsCurve, a: float, b: float) -> NurbsCurve:
    result = curve.duplicate()
    lo, hi = curve.domain()
    a = _clamp(a, lo, hi)
    b = _clamp(b, lo, hi)
    _require(b > a, "Split produced an empty curve interval")
    if a > lo or b < hi:
        _require(result.trim(a, b), "Kernel refused a split interval")
    return result


def _closest(curve: NurbsCurve, point: Point) -> tuple[float, float]:
    t, gap = Closest.curve_point(curve, point)
    lo, hi = curve.domain()
    if curve.degree() == 1:
        spans = curve.get_span_vector()
        best = math.inf
        for i in range(1, len(spans)):
            a = curve.point_at(spans[i - 1])
            b = curve.point_at(spans[i])
            v = b - a
            length2 = v.dot(v)
            if length2 <= _EPSILON * _EPSILON:
                continue
            fraction = _clamp((point - a).dot(v) / length2, 0.0, 1.0)
            segment = _interval(curve, spans[i - 1], spans[i])
            w0 = segment.weight(0)
            w1 = segment.weight(segment.cv_count() - 1)
            normalized = fraction * w0 / (w1 * (1.0 - fraction) + fraction * w0)
            candidate = spans[i - 1] + normalized * (spans[i] - spans[i - 1])
            gap = curve.point_at(candidate).distance(point)
            if gap < best:
                best = gap
                t = candidate
        return t, curve.point_at(t).distance(point)
    for i in range(24):
        eval = curve.evaluate(t, 1)
        d = eval[1]
        r = eval[0] - Vector(point[0], point[1], point[2])
        dd = d.dot(d)
        if dd <= _EPSILON * _EPSILON:
            break
        next = _clamp(t - d.dot(r) / dd, lo, hi)
        if abs(next - t) <= _EPSILON * (hi - lo):
            t = next
            break
        t = next
    return t, curve.point_at(t).distance(point)


def _unique_parameters(values: list[float], lo: float, hi: float) -> list[float]:
    values = sorted(values)
    result = []
    for value in values:
        value = _clamp(value, lo, hi)
        if not result or value - result[-1] > (hi - lo) * _EPSILON * 16.0:
            result.append(value)
    return result


class _Box:
    def __init__(self, curve: NurbsCurve):
        p = curve.get_cv(0)
        self.lo = [p[0], p[1], p[2]]
        self.hi = [p[0], p[1], p[2]]
        for i in range(1, curve.cv_count()):
            p = curve.get_cv(i)
            for d in range(3):
                self.lo[d] = min(self.lo[d], p[d])
                self.hi[d] = max(self.hi[d], p[d])

    def diagonal(self) -> float:
        return math.hypot(
            self.hi[0] - self.lo[0], self.hi[1] - self.lo[1], self.hi[2] - self.lo[2]
        )

    def overlaps(self, other: _Box, tolerance: float) -> bool:
        for d in range(3):
            if (
                self.hi[d] + tolerance < other.lo[d]
                or other.hi[d] + tolerance < self.lo[d]
            ):
                return False
        return True


def _flat(curve: NurbsCurve, tolerance: float) -> bool:
    a = curve.point_at_start()
    b = curve.point_at_end()
    v = b - a
    length2 = v.dot(v)
    if length2 <= tolerance * tolerance:
        return _Box(curve).diagonal() <= tolerance
    for i in range(curve.cv_count()):
        p = curve.get_cv(i)
        t = (p - a).dot(v) / length2
        if t < -_EPSILON or t > 1.0 + _EPSILON or p.distance(a + v * t) > tolerance:
            return False
    return True


def _refine(a: NurbsCurve, b: NurbsCurve, ta: float, tb: float) -> tuple[float, float]:
    a0, a1 = a.domain()
    b0, b1 = b.domain()
    for k in range(40):
        da = a.evaluate(ta, 1)
        db = b.evaluate(tb, 1)
        r = da[0] - db[0]
        u = da[1]
        v = db[1]
        aa = u.dot(u)
        ab = u.dot(v)
        bb = v.dot(v)
        det = aa * bb - ab * ab
        if det <= _EPSILON * _EPSILON * aa * bb:
            break
        ar = u.dot(r)
        br = v.dot(r)
        na = _clamp(ta + (-bb * ar + ab * br) / det, a0, a1)
        nb = _clamp(tb + (-ab * ar + aa * br) / det, b0, b1)
        if abs(na - ta) < _EPSILON * (a1 - a0) and abs(nb - tb) < _EPSILON * (b1 - b0):
            ta = na
            tb = nb
            break
        ta = na
        tb = nb
    return ta, tb


def _intersections(
    a: NurbsCurve, b: NurbsCurve, tolerance: float, budget: list[int]
) -> list[tuple[float, float]]:
    work = []
    av = a.get_span_vector()
    bv = b.get_span_vector()
    _require(len(av) > 1 and len(bv) > 1, "Split requires nonempty curve spans")
    _require(
        len(av) - 1 <= budget[0] // (len(bv) - 1),
        "Curve intersection exceeds the bounded split workload",
    )
    for i in range(1, len(av)):
        for j in range(1, len(bv)):
            work.append(
                (_interval(a, av[i - 1], av[i]), _interval(b, bv[j - 1], bv[j]), 0)
            )
    hits = []
    while work:
        _require(budget[0] > 0, "Curve intersection exceeds the bounded split workload")
        budget[0] -= 1
        ca, cb, depth = work.pop()
        ba = _Box(ca)
        bb = _Box(cb)
        if not ba.overlaps(bb, tolerance):
            continue
        if (_flat(ca, tolerance * 0.1) and _flat(cb, tolerance * 0.1)) or depth >= 48:
            ap = ca.point_at_start()
            aq = ca.point_at_end()
            bp = cb.point_at_start()
            bq = cb.point_at_end()
            u = aq - ap
            v = bq - bp
            aa = u.dot(u)
            ab = u.dot(v)
            vv = v.dot(v)
            if (
                aa > tolerance * tolerance
                and vv > tolerance * tolerance
                and aa * vv - ab * ab < _EPSILON * _EPSILON * aa * vv
            ):
                t0 = (bp - ap).dot(u) / aa
                t1 = (bq - ap).dot(u) / aa
                gap = bp.distance(ap + u * t0)
                if gap <= tolerance and min(1.0, max(t0, t1)) - max(
                    0.0, min(t0, t1)
                ) > tolerance / math.sqrt(aa):
                    raise ValueError(
                        "Overlapping curves do not define isolated split points"
                    )
            ta, tb, d = Closest.curve_curve(ca, cb)
            if d > tolerance * 2.0:
                continue
            ta, tb = _refine(ca, cb, ta, tb)
            if a.point_at(ta).distance(b.point_at(tb)) > tolerance:
                continue
            duplicate = False
            for hit in hits:
                if (
                    a.point_at(hit[0]).distance(a.point_at(ta)) <= tolerance * 2.0
                    and a.point_at((hit[0] + ta) * 0.5).distance(a.point_at(ta))
                    <= tolerance * 2.0
                    and b.point_at(hit[1]).distance(b.point_at(tb)) <= tolerance * 2.0
                    and b.point_at((hit[1] + tb) * 0.5).distance(b.point_at(tb))
                    <= tolerance * 2.0
                ):
                    duplicate = True
                    break
            if not duplicate:
                hits.append((ta, tb))
            continue
        if ba.diagonal() >= bb.diagonal():
            lo, hi = ca.domain()
            mid = (lo + hi) * 0.5
            work.append((_interval(ca, lo, mid), cb, depth + 1))
            work.append((_interval(ca, mid, hi), cb, depth + 1))
        else:
            lo, hi = cb.domain()
            mid = (lo + hi) * 0.5
            work.append((ca, _interval(cb, lo, mid), depth + 1))
            work.append((ca, _interval(cb, mid, hi), depth + 1))
    hits.sort()
    return hits


def _pullback(
    surface: NurbsSurface, curve: NurbsCurve, tolerance: float
) -> list[NurbsCurve]:
    if (
        surface.m_cv_count[0] == 2
        and surface.m_cv_count[1] == 2
        and surface.m_order[0] == 2
        and surface.m_order[1] == 2
        and not surface.m_is_rat
    ):
        p = surface.get_cv(0, 0)
        u = surface.get_cv(1, 0) - p
        v = surface.get_cv(0, 1) - p
        last = surface.get_cv(1, 1)
        uu = u.dot(u)
        uv = u.dot(v)
        vv = v.dot(v)
        det = uu * vv - uv * uv
        if (
            det > _EPSILON * _EPSILON * uu * vv
            and last.distance(p + u + v) <= tolerance
        ):
            result = curve.duplicate()
            u0, u1 = surface.domain(0)
            v0, v1 = surface.domain(1)
            for i in range(curve.cv_count()):
                q = curve.get_cv(i)
                d = q - p
                du = d.dot(u)
                dv = d.dot(v)
                a = (du * vv - dv * uv) / det
                b = (dv * uu - du * uv) / det
                if q.distance(p + u * a + v * b) > tolerance:
                    return []
                w = curve.weight(i)
                result.set_cv_4d(
                    i, (u0 + a * (u1 - u0)) * w, (v0 + b * (v1 - v0)) * w, 0.0, w
                )
            return [result]
    return Closest.surface_curve(surface, curve, 0.0, 0.0, tolerance)


def _polygon(curve: NurbsCurve, tolerance: float) -> list[Point]:
    work = []
    spans = curve.get_span_vector()
    for i in range(len(spans), 1, -1):
        work.append((_interval(curve, spans[i - 2], spans[i - 1]), 0))
    result = []
    visited = 0
    while work:
        visited += 1
        _require(visited <= _WORK_LIMIT, "Trim sampling exceeds the bounded workload")
        part, depth = work.pop()
        if _flat(part, tolerance * 0.25):
            result.append(part.point_at_start())
            continue
        _require(depth < 40, "Trim sampling exceeds parameter precision")
        lo, hi = part.domain()
        mid = (lo + hi) * 0.5
        work.append((_interval(part, mid, hi), depth + 1))
        work.append((_interval(part, lo, mid), depth + 1))
    return result


def _inside(p: Point, polygon: list[Point]) -> bool:
    result = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[j]
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            result = not result
        j = i
    return result


def _inside_loops(p: Point, loops: list[list[Point]]) -> bool:
    if not loops or not _inside(p, loops[0]):
        return False
    for i in range(1, len(loops)):
        if _inside(p, loops[i]):
            return False
    return True


@dataclass
class _Source:
    edge: int
    world: NurbsCurve
    uv: NurbsCurve


@dataclass
class _Run:
    source: int
    a: float
    b: float


@dataclass
class _Span:
    source: int
    a: float
    b: float
    cuts: list[float]
    curve: NurbsCurve


@dataclass
class _Directed:
    a: int
    b: int
    run: _Run


@dataclass
class _Cycle:
    area: float
    loop: list[_Run]
    points: list[Point]


def _arrange(
    sources: list[_Source], original_loops: list[list[Point]], tolerance: float
) -> list[list[list[_Run]]]:
    spans = []
    for si in range(len(sources)):
        knots = sources[si].uv.get_span_vector()
        if len(knots) == 2 and sources[si].uv.is_closed():
            lo = knots[0]
            hi = knots[-1]
            knots = [
                lo,
                lo + (hi - lo) * 0.25,
                (lo + hi) * 0.5,
                lo + (hi - lo) * 0.75,
                hi,
            ]
        for i in range(1, len(knots)):
            spans.append(
                _Span(
                    si,
                    knots[i - 1],
                    knots[i],
                    [knots[i - 1], knots[i]],
                    _interval(sources[si].uv, knots[i - 1], knots[i]),
                )
            )
    _require(
        len(spans) > 0 and len(spans) <= _WORK_LIMIT // len(spans),
        "Face split exceeds the bounded workload",
    )
    budget = [_WORK_LIMIT]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            for a, b in _intersections(
                spans[i].curve, spans[j].curve, tolerance, budget
            ):
                spans[i].cuts.append(a)
                spans[j].cuts.append(b)
    vertices = []
    edges = []
    outgoing = []

    def node(p: Point) -> int:
        for i in range(len(vertices)):
            if p.distance(vertices[i]) <= tolerance * 4.0:
                return i
        vertices.append(p)
        outgoing.append([])
        return len(vertices) - 1

    def angle(edge: int) -> float:
        run = edges[edge].run
        d = sources[run.source].uv.evaluate(run.a, 1)[1]
        sign = 1.0 if run.b > run.a else -1.0
        return math.atan2(sign * d[1], sign * d[0])

    for span in spans:
        for k in range(len(span.cuts)):
            p = span.curve.point_at(span.cuts[k])
            if p.distance(span.curve.point_at(span.a)) <= tolerance:
                span.cuts[k] = span.a
            elif p.distance(span.curve.point_at(span.b)) <= tolerance:
                span.cuts[k] = span.b
        cuts = _unique_parameters(span.cuts, span.a, span.b)
        for i in range(1, len(cuts)):
            lo = cuts[i - 1]
            hi = cuts[i]
            source = sources[span.source]
            if source.edge < 0 and not _inside_loops(
                source.uv.point_at((lo + hi) * 0.5), original_loops
            ):
                continue
            a = node(source.uv.point_at(lo))
            b = node(source.uv.point_at(hi))
            if a == b:
                continue
            index = len(edges)
            edges.append(_Directed(a, b, _Run(span.source, lo, hi)))
            edges.append(_Directed(b, a, _Run(span.source, hi, lo)))
            outgoing[a].append(index)
            outgoing[b].append(index + 1)
    for choices in outgoing:
        choices.sort(key=angle)
    cycles = []
    used = [False] * len(edges)
    for initial in range(len(edges)):
        if used[initial]:
            continue
        loop = []
        points = []
        edge = initial
        while not used[edge]:
            used[edge] = True
            item = edges[edge]
            run = item.run
            loop.append(run)
            part = _interval(
                sources[run.source].uv, min(run.a, run.b), max(run.a, run.b)
            )
            if run.b < run.a:
                part.reverse()
            points.extend(_polygon(part, tolerance))
            options = outgoing[item.b]
            _require((edge ^ 1) in options, "Invalid trim graph adjacency")
            slot = options.index(edge ^ 1)
            edge = options[(slot + len(options) - 1) % len(options)]
        _require(edge == initial, "Invalid trim graph cycle")
        area = 0.0
        for i in range(len(points)):
            a = points[i]
            b = points[(i + 1) % len(points)]
            area += (a[0] * b[1] - b[0] * a[1]) * 0.5
        if abs(area) <= tolerance * tolerance:
            continue
        run = loop[0]
        curve = sources[run.source].uv
        t = (run.a + run.b) * 0.5
        p = curve.point_at(t)
        d = curve.evaluate(t, 1)[1]
        sign = 1.0 if run.b > run.a else -1.0
        length = math.hypot(d[0], d[1])
        _require(length > _EPSILON, "Cannot orient a degenerate trim fragment")
        left = Point(
            p[0] - sign * d[1] / length * tolerance * 8.0,
            p[1] + sign * d[0] / length * tolerance * 8.0,
            0.0,
        )
        if not _inside_loops(left, original_loops):
            continue
        cycles.append(_Cycle(area, loop, points))
    result = []
    positive = []
    for i in range(len(cycles)):
        if cycles[i].area > 0.0:
            positive.append(i)
            result.append([cycles[i].loop])
    for cycle in cycles:
        if cycle.area >= 0.0:
            continue
        parent = len(result)
        smallest = math.inf
        for i in range(len(positive)):
            outer = cycles[positive[i]]
            if (
                outer.area > abs(cycle.area) + tolerance * tolerance
                and outer.area < smallest
                and _inside(cycle.points[0], outer.points)
            ):
                parent = i
                smallest = outer.area
        _require(parent < len(result), "Unowned interior trim loop")
        result[parent].append(cycle.loop)
    return result


def _vertex(result: BRep, p: Point, tolerance: float) -> int:
    for i in range(len(result.m_vertices)):
        if result.m_vertices[i].point.distance(p) <= tolerance:
            return i
    return result.add_vertex(p, tolerance)


def _lifted_parameter(
    surface: NurbsSurface, uv: NurbsCurve, p: Point, expected: float, tolerance: float
) -> float:
    lo, hi = uv.domain()

    def gap(t: float) -> float:
        q = uv.point_at(t)
        return surface.point_at(q[0], q[1]).distance(p)

    if gap(expected) <= tolerance:
        return expected
    best = expected
    d = gap(best)
    index = 0
    for i in range(129):
        t = lo + (hi - lo) * i / 128.0
        value = gap(t)
        if value < d:
            d = value
            best = t
            index = i
    a = lo + (hi - lo) * max(0, index - 1) / 128.0
    b = lo + (hi - lo) * min(128, index + 1) / 128.0
    for k in range(60):
        x = a + (b - a) / 3.0
        y = b - (b - a) / 3.0
        if gap(x) < gap(y):
            b = y
        else:
            a = x
    mid = (a + b) * 0.5
    if gap(mid) < d:
        best = mid
    _require(
        gap(best) <= tolerance * 4.0,
        "Cannot keep an adjacent trim on its original shared edge",
    )
    return best


def _validate(result: BRep, original: BRep, tolerance: float) -> None:
    _require(result.is_valid(), "Split produced invalid BRep references")
    for s in range(len(original.m_shells)):
        if original.is_closed(s):
            _require(result.is_closed(s), "Split would open a joined shell")
    for face in result.m_faces:
        for wr in face.wires:
            edges = result.wire_edges(wr)
            for i in range(len(edges)):
                a = result.m_edges[edges[i].index]
                next = edges[(i + 1) % len(edges)]
                b = result.m_edges[next.index]
                tail = (
                    a.start_vertex
                    if edges[i].orientation == _REVERSED
                    else a.end_vertex
                )
                head = b.end_vertex if next.orientation == _REVERSED else b.start_vertex
                _require(tail == head, "Split produced an open face boundary")
    for edge in result.m_edges:
        if edge.degenerated:
            continue
        world = result.m_curves_3d[edge.curve_3d_index]
        _require(
            world.point_at_start().distance(result.m_vertices[edge.start_vertex].point)
            <= tolerance * 4.0
            and world.point_at_end().distance(result.m_vertices[edge.end_vertex].point)
            <= tolerance * 4.0,
            "Split edge does not meet its vertices",
        )
        for pc in edge.pcurves:
            for ci in [pc.curve_2d_index, pc.curve_2d_index_2]:
                if ci < 0:
                    continue
                uv = result.m_curves_2d[ci]
                lo, hi = uv.domain()
                for k in range(33):
                    q = uv.point_at(lo + (hi - lo) * k / 32.0)
                    p = result.m_surfaces[pc.surface_index].point_at(q[0], q[1])
                    _require(
                        _closest(world, p)[1] <= max(tolerance, edge.tolerance) * 8.0,
                        "Split edge and surface trim do not coincide",
                    )


def split_curve_by_curves(
    curve: NurbsCurve, cutters: list[NurbsCurve], tolerance: float
) -> list[NurbsCurve]:
    """Split a curve at isolated 3D intersections, retaining every piece and rejecting overlapping cutters."""
    _check_tolerance(tolerance)
    _check_curve(curve)
    _require(len(cutters) > 0, "Select at least one cutter")
    lo, hi = curve.domain()
    cuts = [lo, hi]
    cut_at_seam = False
    budget = [_WORK_LIMIT]
    for cutter in cutters:
        _check_curve(cutter)
        for hit in _intersections(curve, cutter, tolerance, budget):
            a = hit[0]
            if (
                abs(a - lo) <= (hi - lo) * _EPSILON * 16.0
                or abs(a - hi) <= (hi - lo) * _EPSILON * 16.0
            ):
                cut_at_seam = True
            cuts.append(a)
    cuts = _unique_parameters(cuts, lo, hi)
    result = []
    if len(cuts) == 2:
        return [copy.deepcopy(curve)]
    for i in range(1, len(cuts)):
        result.append(_interval(curve, cuts[i - 1], cuts[i]))
    if curve.is_closed() and len(result) > 1 and not cut_at_seam:
        joined = NurbsCurve.join([result[-1], result[0]], tolerance)
        _require(len(joined) == 1, "Cannot join the uncut seam of a closed curve")
        result[0] = joined[0]
        result.pop()
    return result


def split_brep_face_by_curves(
    brep: BRep, face_index: int, cutters: list[NurbsCurve], tolerance: float
) -> BRep:
    """Partition one face inside its owning BRep, retaining all regions and shared shell topology."""
    _check_tolerance(tolerance)
    _require(brep.is_valid(), "Split requires a valid BRep")
    _require(
        face_index >= 0 and face_index < brep.face_count(),
        "Select one BRep face to split",
    )
    _require(len(cutters) > 0, "Select at least one cutter")
    face = brep.m_faces[face_index]
    surface = brep.m_surfaces[face.surface_index]
    _check_surface(surface)
    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    origin = surface.point_at(u0, v0)
    scale = max(
        origin.distance(surface.point_at(u1, v0)) / (u1 - u0),
        origin.distance(surface.point_at(u0, v1)) / (v1 - v0),
    )
    _require(scale > _EPSILON, "Cannot split a degenerate surface domain")
    uv_tolerance = tolerance / scale
    sources = []
    original_loops = []
    for wr in face.wires:
        points = []
        for er in brep.wire_edges(wr):
            edge = brep.m_edges[er.index]
            _require(not edge.degenerated, "Pole-edge splitting is not supported")
            ci = brep.pcurve_index(er.index, face_index, er.orientation)
            _require(ci >= 0, "Face has no source UV boundary")
            uv = copy.deepcopy(brep.m_curves_2d[ci])
            _check_curve(uv)
            _check_curve(brep.m_curves_3d[edge.curve_3d_index])
            sources.append(
                _Source(
                    er.index, brep.m_curves_3d[edge.curve_3d_index], copy.deepcopy(uv)
                )
            )
            if er.orientation == _REVERSED:
                uv.reverse()
            points.extend(_polygon(uv, uv_tolerance))
        _require(len(points) >= 3, "Face has an invalid boundary")
        original_loops.append(points)
    for cutter in cutters:
        _check_curve(cutter)
        for uv in _pullback(surface, cutter, tolerance):
            sources.append(_Source(-1, cutter, uv))
    regions = _arrange(sources, original_loops, uv_tolerance)
    if len(regions) < 2:
        return copy.deepcopy(brep)
    result = copy.deepcopy(brep)
    pieces = []
    replacements = {}

    def make_edge(run: _Run) -> BRepRef:
        source = sources[run.source]
        uv = source.uv
        qa = uv.point_at(run.a)
        qb = uv.point_at(run.b)
        pa = surface.point_at(qa[0], qa[1])
        pb = surface.point_at(qb[0], qb[1])

        def parameter(t: float, p: Point) -> tuple[float, float]:
            lo, hi = source.world.domain()
            a, b = uv.domain()
            expected = lo + (t - a) / (b - a) * (hi - lo)
            gap = source.world.point_at(expected).distance(p)
            return (expected, gap) if gap <= tolerance else _closest(source.world, p)

        wa, da = parameter(run.a, pa)
        wb, db = parameter(run.b, pb)
        _require(
            da <= tolerance * 4.0 and db <= tolerance * 4.0,
            "Cutter is not on the selected surface",
        )
        w0, w1 = source.world.domain()
        c0, c1 = uv.domain()
        if source.world.is_closed():
            if abs(wa - w0) < (w1 - w0) * _EPSILON and run.a > (c0 + c1) * 0.5:
                wa = w1
            if abs(wb - w0) < (w1 - w0) * _EPSILON and run.b > (c0 + c1) * 0.5:
                wb = w1
        lo = min(wa, wb)
        hi = max(wa, wb)
        _require(hi - lo > (w1 - w0) * _EPSILON, "Split would create a collapsed edge")
        orientation = _FORWARD if wa < wb else _REVERSED
        for piece in pieces:
            same = (
                sources[piece[0]].edge == source.edge
                if source.edge >= 0
                else piece[0] == run.source
            )
            if (
                same
                and source.world.point_at(lo).distance(source.world.point_at(piece[1]))
                <= tolerance * 4.0
                and source.world.point_at(hi).distance(source.world.point_at(piece[2]))
                <= tolerance * 4.0
            ):
                return BRepRef(piece[3], orientation)
        world = _interval(source.world, lo, hi)
        a = _vertex(result, world.point_at_start(), tolerance * 4.0)
        b = _vertex(result, world.point_at_end(), tolerance * 4.0)
        ei = result.add_edge(result.add_curve_3d(world), a, b, tolerance)
        if source.edge >= 0:
            old = brep.m_edges[source.edge]
            for pc in old.pcurves:
                ids = [-1, -1]
                for at, ci in enumerate([pc.curve_2d_index, pc.curve_2d_index_2]):
                    if ci >= 0:
                        c = brep.m_curves_2d[ci]
                        c0, c1 = c.domain()
                        ca = _lifted_parameter(
                            brep.m_surfaces[pc.surface_index],
                            c,
                            world.point_at_start(),
                            c0 + (lo - w0) / (w1 - w0) * (c1 - c0),
                            tolerance,
                        )
                        cb = _lifted_parameter(
                            brep.m_surfaces[pc.surface_index],
                            c,
                            world.point_at_end(),
                            c0 + (hi - w0) / (w1 - w0) * (c1 - c0),
                            tolerance,
                        )
                        _require(
                            cb > ca, "A split crosses an unsupported periodic trim seam"
                        )
                        ids[at] = result.add_curve_2d(_interval(c, ca, cb))
                result.add_pcurve(ei, pc.surface_index, ids[0], ids[1])
            replacements.setdefault(source.edge, []).append((lo, ei))
        else:
            pc = _interval(uv, min(run.a, run.b), max(run.a, run.b))
            if (wb - wa) * (run.b - run.a) < 0.0:
                pc.reverse()
            result.add_pcurve(ei, face.surface_index, result.add_curve_2d(pc))
        pieces.append((run.source, lo, hi, ei))
        return BRepRef(ei, orientation)

    new_wires = []
    for region in regions:
        wires = []
        for loop in region:
            refs = []
            for run in loop:
                refs.append(make_edge(run))
            wires.append(BRepRef(result.add_wire(refs), _FORWARD))
        new_wires.append(wires)
    for edge, items in replacements.items():
        replacements[edge] = sorted(set(items))
    for wi in range(len(brep.m_wires)):
        refs = []
        for er in brep.m_wires[wi].edges:
            if er.index not in replacements:
                refs.append(copy.deepcopy(er))
                continue
            items = list(replacements[er.index])
            if er.orientation == _REVERSED:
                items.reverse()
            for item in items:
                refs.append(BRepRef(item[1], er.orientation))
        result.m_wires[wi].edges = refs
    result.m_faces[face_index].wires = new_wires[0]
    added = []
    for i in range(1, len(new_wires)):
        next = copy.deepcopy(face)
        next.wires = new_wires[i]
        added.append(result.face_count())
        result.m_faces.append(next)
    for shell in result.m_shells:
        refs = []
        for fr in shell.faces:
            refs.append(fr)
            if fr.index == face_index:
                for index in added:
                    refs.append(BRepRef(index, fr.orientation))
        shell.faces = refs
    _validate(result, brep, tolerance)
    return result


def split_surface_by_curves(
    surface: NurbsSurface, cutters: list[NurbsCurve], tolerance: float
) -> BRep:
    """Wrap a surface's natural boundary in a BRep and partition it with on-surface curves."""
    _check_tolerance(tolerance)
    _check_surface(surface)
    result = BRep()
    si = result.add_surface(surface)
    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    uv = [Point(u0, v0, 0), Point(u1, v0, 0), Point(u1, v1, 0), Point(u0, v1, 0)]
    at = [v0, u1, v1, u0]
    edges = []
    for i in range(4):
        curve = surface.iso_curve(i % 2, at[i])
        if i >= 2:
            curve.reverse()
        a = _vertex(result, curve.point_at_start(), tolerance)
        b = _vertex(result, curve.point_at_end(), tolerance)
        _require(
            a != b, "Closed or pole boundaries need a BRep with explicit seam topology"
        )
        edge = result.add_edge(result.add_curve_3d(curve), a, b, tolerance)
        pc = NurbsCurve.create(False, 1, [uv[i], uv[(i + 1) % 4]])
        result.add_pcurve(edge, si, result.add_curve_2d(pc))
        edges.append(BRepRef(edge, _FORWARD))
    result.add_face(si, [BRepRef(result.add_wire(edges), _FORWARD)])
    return split_brep_face_by_curves(result, 0, cutters, tolerance)


def split_line_by_curves(
    line: Line, cutters: list[NurbsCurve], tolerance: float
) -> list[Line]:
    """Split a line at isolated 3D intersections, retaining line types and display attributes."""
    curve = NurbsCurve.create(False, 1, [line.point_at(0), line.point_at(1)])
    result = []
    for piece in split_curve_by_curves(curve, cutters, tolerance):
        next = Line.from_points(piece.point_at_start(), piece.point_at_end())
        next.name = line.name
        next.width = line.width
        next.dash = copy.deepcopy(line.dash)
        next.linecolor = copy.deepcopy(line.linecolor)
        result.append(next)
    return result


def split_polyline_by_curves(
    polyline: Polyline, cutters: list[NurbsCurve], tolerance: float
) -> list[Polyline]:
    """Split a polyline, retaining each original corner, piece order and display attributes."""
    curve = NurbsCurve.create(False, 1, polyline.get_points())
    result = []
    for piece in split_curve_by_curves(curve, cutters, tolerance):
        points = []
        for t in piece.get_span_vector():
            points.append(piece.point_at(t))
        next = Polyline(points)
        next.name = polyline.name
        next.width = polyline.width
        next.dash = copy.deepcopy(polyline.dash)
        next.linecolor = copy.deepcopy(polyline.linecolor)
        result.append(next)
    return result
