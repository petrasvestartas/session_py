"""Intersection splits retaining original curve geometry and shared BRep topology."""

import copy
import math
from dataclasses import dataclass

from .brep import BRep, BRepOrientation, BRepRef
from .closest import Closest
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .point import Point
from .tolerance import Tolerance

_EPS = Tolerance.ZERO_TOLERANCE
_WORK_LIMIT = 200000
_FORWARD = BRepOrientation.Forward
_REVERSED = BRepOrientation.Reversed


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
            all(math.isfinite(p[d]) for d in range(3))
            and math.isfinite(curve.weight(i))
            and curve.weight(i) > 0.0,
            "Split requires finite controls and positive rational weights",
        )


def _check_surface(surface: NurbsSurface) -> None:
    _require(surface.is_valid(), "Split requires a valid NURBS surface")
    for i in range(surface.cv_count(0)):
        for j in range(surface.cv_count(1)):
            p, w = surface.get_cv(i, j), surface.weight(i, j)
            _require(
                all(math.isfinite(p[d]) for d in range(3))
                and math.isfinite(w)
                and w > 0,
                "Split requires finite surface controls and positive rational weights",
            )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _interval(curve: NurbsCurve, a: float, b: float) -> NurbsCurve:
    result = curve.duplicate()
    lo, hi = curve.domain()
    a, b = _clamp(a, lo, hi), _clamp(b, lo, hi)
    _require(b > a, "Split produced an empty curve interval")
    if a > lo or b < hi:
        _require(result.trim(a, b), "Kernel refused a split interval")
    return result


def _dot(a: Point, b: Point) -> float:
    return sum(a[d] * b[d] for d in range(3))


def _subtract(a: Point, b: Point) -> Point:
    return Point(*(a[d] - b[d] for d in range(3)))


def _closest(curve: NurbsCurve, point: Point) -> tuple[float, float]:
    t, _ = Closest.curve_point(curve, point)
    lo, hi = curve.domain()
    if curve.degree() == 1:
        spans = curve.get_span_vector()
        best = math.inf
        for i in range(1, len(spans)):
            a, b = curve.point_at(spans[i - 1]), curve.point_at(spans[i])
            v = _subtract(b, a)
            length2 = _dot(v, v)
            if length2 <= _EPS * _EPS:
                continue
            fraction = _clamp(_dot(_subtract(point, a), v) / length2, 0.0, 1.0)
            segment = _interval(curve, spans[i - 1], spans[i])
            w0, w1 = segment.weight(0), segment.weight(segment.cv_count() - 1)
            normalized = fraction * w0 / (w1 * (1.0 - fraction) + fraction * w0)
            candidate = spans[i - 1] + normalized * (spans[i] - spans[i - 1])
            gap = curve.point_at(candidate).distance(point)
            if gap < best:
                best, t = gap, candidate
        return t, curve.point_at(t).distance(point)
    for _ in range(24):
        value = curve.evaluate(t, 1)
        derivative = Point(*(value[1][d] for d in range(3)))
        residual = Point(*(value[0][d] - point[d] for d in range(3)))
        dd = _dot(derivative, derivative)
        if dd <= _EPS * _EPS:
            break
        next_t = _clamp(t - _dot(derivative, residual) / dd, lo, hi)
        if abs(next_t - t) <= _EPS * (hi - lo):
            t = next_t
            break
        t = next_t
    return t, curve.point_at(t).distance(point)


def _unique_parameters(values: list[float], lo: float, hi: float) -> list[float]:
    result = []
    for value in sorted(values):
        value = _clamp(value, lo, hi)
        if not result or value - result[-1] > (hi - lo) * _EPS * 16.0:
            result.append(value)
    return result


class _Box:
    def __init__(self, curve: NurbsCurve):
        points = [curve.get_cv(i) for i in range(curve.cv_count())]
        self.lo = [min(p[d] for p in points) for d in range(3)]
        self.hi = [max(p[d] for p in points) for d in range(3)]

    def diagonal(self) -> float:
        return math.sqrt(sum((self.hi[d] - self.lo[d]) ** 2 for d in range(3)))

    def overlaps(self, other: "_Box", tolerance: float) -> bool:
        return all(
            self.hi[d] + tolerance >= other.lo[d]
            and other.hi[d] + tolerance >= self.lo[d]
            for d in range(3)
        )


def _flat(curve: NurbsCurve, tolerance: float) -> bool:
    a, b = curve.point_at_start(), curve.point_at_end()
    v = _subtract(b, a)
    length2 = _dot(v, v)
    if length2 <= tolerance * tolerance:
        return _Box(curve).diagonal() <= tolerance
    for i in range(curve.cv_count()):
        p = curve.get_cv(i)
        t = _dot(_subtract(p, a), v) / length2
        q = Point(*(a[d] + v[d] * t for d in range(3)))
        if t < -_EPS or t > 1.0 + _EPS or p.distance(q) > tolerance:
            return False
    return True


def _refine(a: NurbsCurve, b: NurbsCurve, ta: float, tb: float) -> tuple[float, float]:
    a0, a1 = a.domain()
    b0, b1 = b.domain()
    for _ in range(40):
        da, db = a.evaluate(ta, 1), b.evaluate(tb, 1)
        residual = Point(*(da[0][d] - db[0][d] for d in range(3)))
        u, v = (
            Point(*(da[1][d] for d in range(3))),
            Point(*(db[1][d] for d in range(3))),
        )
        aa, ab, bb = _dot(u, u), _dot(u, v), _dot(v, v)
        determinant = aa * bb - ab * ab
        if determinant <= _EPS * _EPS * aa * bb:
            break
        ar, br = _dot(u, residual), _dot(v, residual)
        na = _clamp(ta + (-bb * ar + ab * br) / determinant, a0, a1)
        nb = _clamp(tb + (-ab * ar + aa * br) / determinant, b0, b1)
        if abs(na - ta) < _EPS * (a1 - a0) and abs(nb - tb) < _EPS * (b1 - b0):
            ta, tb = na, nb
            break
        ta, tb = na, nb
    return ta, tb


def _intersections(
    a: NurbsCurve, b: NurbsCurve, tolerance: float, budget: list[int]
) -> list[tuple[float, float]]:
    av, bv = a.get_span_vector(), b.get_span_vector()
    _require(len(av) > 1 and len(bv) > 1, "Split requires nonempty curve spans")
    _require(
        len(av) - 1 <= budget[0] // (len(bv) - 1),
        "Curve intersection exceeds the bounded split workload",
    )
    work = [
        (_interval(a, av[i - 1], av[i]), _interval(b, bv[j - 1], bv[j]), 0)
        for i in range(1, len(av))
        for j in range(1, len(bv))
    ]
    hits = []
    while work:
        budget[0] -= 1
        _require(
            budget[0] >= 0,
            "Curve intersection exceeds the bounded split workload",
        )
        ca, cb, depth = work.pop()
        ba, bb = _Box(ca), _Box(cb)
        if not ba.overlaps(bb, tolerance):
            continue
        if (_flat(ca, tolerance * 0.1) and _flat(cb, tolerance * 0.1)) or depth >= 48:
            ap, aq, bp, bq = (
                ca.point_at_start(),
                ca.point_at_end(),
                cb.point_at_start(),
                cb.point_at_end(),
            )
            u, v = _subtract(aq, ap), _subtract(bq, bp)
            aa, ab, vv = _dot(u, u), _dot(u, v), _dot(v, v)
            if (
                aa > tolerance**2
                and vv > tolerance**2
                and aa * vv - ab * ab < _EPS**2 * aa * vv
            ):
                t0, t1 = (
                    _dot(_subtract(bp, ap), u) / aa,
                    _dot(_subtract(bq, ap), u) / aa,
                )
                q = Point(*(ap[d] + u[d] * t0 for d in range(3)))
                if bp.distance(q) <= tolerance and min(1.0, max(t0, t1)) - max(
                    0.0, min(t0, t1)
                ) > tolerance / math.sqrt(aa):
                    raise ValueError(
                        "Overlapping curves do not define isolated split points"
                    )
            ta, tb, distance = Closest.curve_curve(ca, cb)
            if distance > tolerance * 2.0:
                continue
            ta, tb = _refine(ca, cb, ta, tb)
            if a.point_at(ta).distance(b.point_at(tb)) > tolerance:
                continue
            if not any(
                a.point_at(hit[0]).distance(a.point_at(ta)) <= tolerance * 2.0
                and a.point_at((hit[0] + ta) * 0.5).distance(a.point_at(ta))
                <= tolerance * 2.0
                and b.point_at(hit[1]).distance(b.point_at(tb)) <= tolerance * 2.0
                and b.point_at((hit[1] + tb) * 0.5).distance(b.point_at(tb))
                <= tolerance * 2.0
                for hit in hits
            ):
                hits.append((ta, tb))
        elif ba.diagonal() >= bb.diagonal():
            lo, hi = ca.domain()
            mid = (lo + hi) * 0.5
            work.extend(
                [
                    (_interval(ca, lo, mid), cb, depth + 1),
                    (_interval(ca, mid, hi), cb, depth + 1),
                ]
            )
        else:
            lo, hi = cb.domain()
            mid = (lo + hi) * 0.5
            work.extend(
                [
                    (ca, _interval(cb, lo, mid), depth + 1),
                    (ca, _interval(cb, mid, hi), depth + 1),
                ]
            )
    return sorted(hits)


def split_curve_by_curves(
    curve: NurbsCurve, cutters: list[NurbsCurve], tolerance: float
) -> list[NurbsCurve]:
    """Split at isolated 3D intersections, retaining every original curve interval.

    Parameters
    ----------
    curve : NurbsCurve
        Individual source curve; never modified.
    cutters : list[NurbsCurve]
        Curves intersecting the source in 3D, without projection.
    tolerance : float
        Positive finite distance tolerance in model units.

    Returns
    -------
    list[NurbsCurve]
        All pieces; one unchanged copy for a valid no-op.

    Raises
    ------
    ValueError
        Invalid input, overlapping curves, or excessive intersection workload.
    """
    _check_tolerance(tolerance)
    _check_curve(curve)
    _require(bool(cutters), "Select at least one cutter")
    lo, hi = curve.domain()
    cuts = [lo, hi]
    cut_at_seam = False
    budget = [_WORK_LIMIT]
    for cutter in cutters:
        _check_curve(cutter)
        for a, _ in _intersections(curve, cutter, tolerance, budget):
            if (
                abs(a - lo) <= (hi - lo) * _EPS * 16.0
                or abs(a - hi) <= (hi - lo) * _EPS * 16.0
            ):
                cut_at_seam = True
            cuts.append(a)
    cuts = _unique_parameters(cuts, lo, hi)
    if len(cuts) == 2:
        return [copy.deepcopy(curve)]
    result = [_interval(curve, cuts[i - 1], cuts[i]) for i in range(1, len(cuts))]
    # A closed curve's arbitrary storage seam is not an additional cut.
    if curve.is_closed() and len(result) > 1 and not cut_at_seam:
        joined = NurbsCurve.join([result[-1], result[0]], tolerance)
        _require(len(joined) == 1, "Cannot join the uncut seam of a closed curve")
        result[0] = joined[0]
        result.pop()
    return result


def _pullback(
    surface: NurbsSurface, curve: NurbsCurve, tolerance: float
) -> list[NurbsCurve]:
    # Affine patches preserve the exact rational controls and parameterization.
    if (
        list(surface.m_cv_count) == [2, 2]
        and list(surface.m_order) == [2, 2]
        and not surface.m_is_rat
    ):
        p = surface.get_cv(0, 0)
        u, v = _subtract(surface.get_cv(1, 0), p), _subtract(surface.get_cv(0, 1), p)
        uu, uv, vv = _dot(u, u), _dot(u, v), _dot(v, v)
        determinant = uu * vv - uv * uv
        last = Point(*(p[d] + u[d] + v[d] for d in range(3)))
        if (
            determinant > _EPS**2 * uu * vv
            and surface.get_cv(1, 1).distance(last) <= tolerance
        ):
            result = curve.duplicate()
            u0, u1 = surface.domain(0)
            v0, v1 = surface.domain(1)
            for i in range(curve.cv_count()):
                q = curve.get_cv(i)
                delta = _subtract(q, p)
                du, dv = _dot(delta, u), _dot(delta, v)
                a, b = (
                    (du * vv - dv * uv) / determinant,
                    (dv * uu - du * uv) / determinant,
                )
                projected = Point(*(p[d] + a * u[d] + b * v[d] for d in range(3)))
                if q.distance(projected) > tolerance:
                    return []
                w = curve.weight(i)
                result.set_cv_4d(
                    i, (u0 + a * (u1 - u0)) * w, (v0 + b * (v1 - v0)) * w, 0.0, w
                )
            return [result]
    return Closest.surface_curve(surface, curve, 0.0, 0.0, tolerance)


def _polygon(curve: NurbsCurve, tolerance: float) -> list[Point]:
    spans = curve.get_span_vector()
    work = [
        (_interval(curve, spans[i - 1], spans[i]), 0)
        for i in range(len(spans) - 1, 0, -1)
    ]
    result, visited = [], 0
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
        work.extend(
            [
                (_interval(part, mid, hi), depth + 1),
                (_interval(part, lo, mid), depth + 1),
            ]
        )
    return result


def _inside(p: Point, polygon: list[Point]) -> bool:
    result = False
    for i, a in enumerate(polygon):
        b = polygon[i - 1]
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            result = not result
    return result


def _inside_loops(p: Point, loops: list[list[Point]]) -> bool:
    return (
        bool(loops)
        and _inside(p, loops[0])
        and not any(_inside(p, hole) for hole in loops[1:])
    )


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


def _arrange(
    sources: list[_Source], original_loops: list[list[Point]], tolerance: float
) -> list[list[list[_Run]]]:
    spans = []
    for si, source in enumerate(sources):
        knots = source.uv.get_span_vector()
        # A closed Bezier span needs distinct graph nodes on its interior.
        if len(knots) == 2 and source.uv.is_closed():
            lo, hi = knots
            knots = [lo + (hi - lo) * i / 4 for i in range(5)]
        for a, b in zip(knots, knots[1:]):
            spans.append([si, a, b, [a, b], _interval(source.uv, a, b)])
    _require(len(spans) ** 2 <= _WORK_LIMIT, "Face split exceeds bounded workload")
    budget = [_WORK_LIMIT]
    for i, left in enumerate(spans):
        for right in spans[i + 1 :]:
            for a, b in _intersections(left[4], right[4], tolerance, budget):
                left[3].append(a)
                right[3].append(b)
    vertices = []
    edges = []
    outgoing = []

    def vertex(p):
        for i, q in enumerate(vertices):
            if p.distance(q) <= tolerance * 4:
                return i
        vertices.append(p)
        outgoing.append([])
        return len(vertices) - 1

    def angle(run):
        curve = sources[run.source].uv
        derivative = curve.evaluate(run.a, 1)[1]
        sign = 1 if run.b > run.a else -1
        return math.atan2(sign * derivative[1], sign * derivative[0])

    for si, a, b, cuts, curve in spans:
        cuts = [
            a
            if curve.point_at(t).distance(curve.point_at(a)) <= tolerance
            else b
            if curve.point_at(t).distance(curve.point_at(b)) <= tolerance
            else t
            for t in cuts
        ]
        cuts = _unique_parameters(cuts, a, b)
        for lo, hi in zip(cuts, cuts[1:]):
            uv = sources[si].uv
            if sources[si].edge < 0 and not _inside_loops(
                uv.point_at((lo + hi) * 0.5), original_loops
            ):
                continue
            start, end = vertex(uv.point_at(lo)), vertex(uv.point_at(hi))
            if start == end:
                continue
            forward = _Run(si, lo, hi)
            back = _Run(si, hi, lo)
            index = len(edges)
            edges.extend([(start, end, forward), (end, start, back)])
            outgoing[start].append(index)
            outgoing[end].append(index + 1)
    for choices in outgoing:
        choices.sort(key=lambda i: angle(edges[i][2]))
    cycles = []
    used = set()
    for initial in range(len(edges)):
        if initial in used:
            continue
        loop = []
        points = []
        edge = initial
        while edge not in used:
            used.add(edge)
            a, b, run = edges[edge]
            loop.append(run)
            part = _interval(
                sources[run.source].uv, min(run.a, run.b), max(run.a, run.b)
            )
            if run.b < run.a:
                part.reverse()
            points.extend(_polygon(part, tolerance))
            options = outgoing[b]
            edge = options[(options.index(edge ^ 1) - 1) % len(options)]
        _require(edge == initial, "Invalid trim graph cycle")
        area = (
            sum(
                a[0] * b[1] - b[0] * a[1]
                for a, b in zip(points, points[1:] + points[:1])
            )
            * 0.5
        )
        if abs(area) <= tolerance * tolerance:
            continue
        run = loop[0]
        curve = sources[run.source].uv
        t = (run.a + run.b) * 0.5
        p = curve.point_at(t)
        d = curve.evaluate(t, 1)[1]
        sign = 1 if run.b > run.a else -1
        length = math.hypot(d[0], d[1])
        left = Point(
            p[0] - sign * d[1] / length * tolerance * 8,
            p[1] + sign * d[0] / length * tolerance * 8,
            0,
        )
        if not _inside_loops(left, original_loops):
            continue
        cycles.append((area, loop, points))
    result = [[loop] for area, loop, pts in cycles if area > 0]
    positives = [(area, pts) for area, loop, pts in cycles if area > 0]
    for area, loop, pts in cycles:
        if area >= 0:
            continue
        candidates = [
            i
            for i, (outer, poly) in enumerate(positives)
            if outer > abs(area) + tolerance * tolerance and _inside(pts[0], poly)
        ]
        _require(bool(candidates), "Unowned interior trim loop")
        parent = min(candidates, key=lambda i: positives[i][0])
        result[parent].append(loop)
    return result


def _vertex(result: BRep, point: Point, tolerance: float) -> int:
    for i, vertex in enumerate(result.m_vertices):
        if vertex.point.distance(point) <= tolerance:
            return i
    return result.add_vertex(point, tolerance)


def _lifted_parameter(
    surface: NurbsSurface,
    uv: NurbsCurve,
    point: Point,
    expected: float,
    tolerance: float,
) -> float:
    lo, hi = uv.domain()

    def gap(t: float) -> float:
        q = uv.point_at(t)
        return surface.point_at(q[0], q[1]).distance(point)

    if gap(expected) <= tolerance:
        return expected
    best, distance, index = expected, gap(expected), 0
    for i in range(129):
        t = lo + (hi - lo) * i / 128.0
        value = gap(t)
        if value < distance:
            distance, best, index = value, t, i
    a, b = (
        lo + (hi - lo) * max(0, index - 1) / 128.0,
        lo + (hi - lo) * min(128, index + 1) / 128.0,
    )
    for _ in range(60):
        x, y = a + (b - a) / 3.0, b - (b - a) / 3.0
        if gap(x) < gap(y):
            b = y
        else:
            a = x
    mid = (a + b) * 0.5
    if gap(mid) < distance:
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
        for wire in face.wires:
            edges = result.wire_edges(wire)
            for i, edge in enumerate(edges):
                next_edge = edges[(i + 1) % len(edges)]
                a, b = result.m_edges[edge.index], result.m_edges[next_edge.index]
                tail = a.start_vertex if edge.orientation == _REVERSED else a.end_vertex
                head = (
                    b.end_vertex
                    if next_edge.orientation == _REVERSED
                    else b.start_vertex
                )
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


def split_brep_face_by_curves(
    brep: BRep, face_index: int, cutters: list[NurbsCurve], tolerance: float
) -> BRep:
    """Partition a face, retaining every region and propagating shared edge splits.

    Parameters
    ----------
    brep : BRep
        Owning BRep, never modified.
    face_index : int
        Index of the selected face in the owning BRep.
    cutters : list[NurbsCurve]
        On-surface cutting curves; no implicit projection is performed.
    tolerance : float
        Positive finite distance tolerance in model units.

    Returns
    -------
    BRep
        Updated owning BRep, or an unchanged copy when no region is divided.

    Raises
    ------
    ValueError
        Invalid input or a split whose shared topology cannot be preserved.
    """
    _check_tolerance(tolerance)
    _require(brep.is_valid(), "Split requires a valid BRep")
    _require(0 <= face_index < brep.face_count(), "Select one BRep face to split")
    _require(bool(cutters), "Select at least one cutter")
    face = brep.m_faces[face_index]
    surface = brep.m_surfaces[face.surface_index]
    _check_surface(surface)
    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    scale = max(
        surface.point_at(u0, v0).distance(surface.point_at(u1, v0)) / (u1 - u0),
        surface.point_at(u0, v0).distance(surface.point_at(u0, v1)) / (v1 - v0),
    )
    _require(scale > _EPS, "Cannot split a degenerate surface domain")
    uv_tolerance = tolerance / scale
    sources, original_loops = [], []
    for wire in face.wires:
        points = []
        for ref in brep.wire_edges(wire):
            edge = brep.m_edges[ref.index]
            _require(not edge.degenerated, "Pole-edge splitting is not supported")
            ci = brep.pcurve_index(ref.index, face_index, ref.orientation)
            _require(ci >= 0, "Face has no source UV boundary")
            uv = copy.deepcopy(brep.m_curves_2d[ci])
            _check_curve(uv)
            _check_curve(brep.m_curves_3d[edge.curve_3d_index])
            sources.append(
                _Source(
                    ref.index, brep.m_curves_3d[edge.curve_3d_index], copy.deepcopy(uv)
                )
            )
            if ref.orientation == _REVERSED:
                uv.reverse()
            points.extend(_polygon(uv, uv_tolerance))
        _require(len(points) >= 3, "Face has an invalid boundary")
        original_loops.append(points)
    for cutter in cutters:
        _check_curve(cutter)
        sources.extend(
            _Source(-1, cutter, uv) for uv in _pullback(surface, cutter, tolerance)
        )
    regions = _arrange(sources, original_loops, uv_tolerance)
    if len(regions) < 2:
        return copy.deepcopy(brep)
    result = copy.deepcopy(brep)
    pieces, replacements = [], {}

    def make_edge(run: _Run) -> BRepRef:
        source = sources[run.source]
        uv = source.uv
        qa, qb = uv.point_at(run.a), uv.point_at(run.b)
        pa, pb = surface.point_at(qa[0], qa[1]), surface.point_at(qb[0], qb[1])

        def parameter(t, p):
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
            if abs(wa - w0) < (w1 - w0) * _EPS and run.a > (c0 + c1) * 0.5:
                wa = w1
            if abs(wb - w0) < (w1 - w0) * _EPS and run.b > (c0 + c1) * 0.5:
                wb = w1
        lo, hi = min(wa, wb), max(wa, wb)
        _require(hi - lo > (w1 - w0) * _EPS, "Split would create a collapsed edge")
        for source_index, piece_lo, piece_hi, edge_index in pieces:
            same = (
                sources[source_index].edge == source.edge
                if source.edge >= 0
                else source_index == run.source
            )
            if (
                same
                and source.world.point_at(lo).distance(source.world.point_at(piece_lo))
                <= tolerance * 4.0
                and source.world.point_at(hi).distance(source.world.point_at(piece_hi))
                <= tolerance * 4.0
            ):
                return BRepRef(edge_index, _FORWARD if wa < wb else _REVERSED)
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
        return BRepRef(ei, _FORWARD if wa < wb else _REVERSED)

    new_wires = [
        [
            BRepRef(result.add_wire([make_edge(run) for run in loop]), _FORWARD)
            for loop in region
        ]
        for region in regions
    ]
    replacements = {edge: sorted(set(items)) for edge, items in replacements.items()}
    # Every old wire receives the same ordered fragments of a shared edge.
    for wi, wire in enumerate(brep.m_wires):
        refs = []
        for ref in wire.edges:
            if ref.index not in replacements:
                refs.append(copy.deepcopy(ref))
                continue
            items = replacements[ref.index]
            if ref.orientation == _REVERSED:
                items = list(reversed(items))
            refs.extend(BRepRef(edge, ref.orientation) for _, edge in items)
        result.m_wires[wi].edges = refs
    result.m_faces[face_index].wires = new_wires[0]
    added = []
    for wires in new_wires[1:]:
        next_face = copy.deepcopy(face)
        next_face.wires = wires
        added.append(result.face_count())
        result.m_faces.append(next_face)
    for shell in result.m_shells:
        refs = []
        for ref in shell.faces:
            refs.append(ref)
            if ref.index == face_index:
                refs.extend(BRepRef(index, ref.orientation) for index in added)
        shell.faces = refs
    _validate(result, brep, tolerance)
    return result


def split_surface_by_curves(
    surface: NurbsSurface, cutters: list[NurbsCurve], tolerance: float
) -> BRep:
    """Wrap a surface's natural boundary in a BRep and retain every split region.

    Parameters
    ----------
    surface : NurbsSurface
        Individual source surface; never modified.
    cutters : list[NurbsCurve]
        Curves on the source surface, without projection.
    tolerance : float
        Positive finite distance tolerance in model units.

    Returns
    -------
    BRep
        Trimmed regions on the original surface.

    Raises
    ------
    ValueError
        Invalid input, unsupported natural seam/pole, or invalid split topology.
    """
    _check_tolerance(tolerance)
    _check_surface(surface)
    result = BRep()
    si = result.add_surface(surface)
    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    uv = [Point(u0, v0, 0), Point(u1, v0, 0), Point(u1, v1, 0), Point(u0, v1, 0)]
    edges = []
    for i in range(4):
        curve = surface.iso_curve(0 if i % 2 == 0 else 1, [v0, u1, v1, u0][i])
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
