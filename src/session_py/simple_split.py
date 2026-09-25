from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from dataclasses import field
from .brep import BRep
from .brep import BRepFace
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

# ═══════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════
_EPSILON = Tolerance.ZERO_TOLERANCE  # Relative parameter and determinant epsilon.
_FORWARD = BRepOrientation.Forward  # Use along the curve direction.
_REVERSED = BRepOrientation.Reversed  # Use against the curve direction.
_WORK_LIMIT = 200000  # Upper bound on subdivision steps.


class _Bounds:
    """Axis-aligned box around the control points of a curve."""

    def __init__(self, curve: NurbsCurve):
        """Box around the control points of a curve."""
        p = curve.get_cv(0)
        self.lo = [p[0], p[1], p[2]]  # Minimum corner.
        self.hi = [p[0], p[1], p[2]]  # Maximum corner.

        for i in range(1, curve.cv_count()):
            p = curve.get_cv(i)

            for d in range(3):
                self.lo[d] = min(self.lo[d], p[d])
                self.hi[d] = max(self.hi[d], p[d])

    def diagonal(self) -> float:
        """Length of the box diagonal."""
        return math.hypot(
            self.hi[0] - self.lo[0], self.hi[1] - self.lo[1], self.hi[2] - self.lo[2]
        )

    def overlaps(self, other: _Bounds, tolerance: float) -> bool:
        """True when the boxes overlap within tolerance."""
        for d in range(3):
            if (
                self.hi[d] + tolerance < other.lo[d]
                or other.hi[d] + tolerance < self.lo[d]
            ):
                return False

        return True


@dataclass
class _Pair:
    """Two curve pieces tested for intersection."""

    a: NurbsCurve  # Piece of the first curve.
    b: NurbsCurve  # Piece of the second curve.
    depth: int  # Subdivision depth.


@dataclass
class _Part:
    """Curve piece waiting to be flattened."""

    curve: NurbsCurve  # Piece of the curve.
    depth: int  # Subdivision depth.


@dataclass
class _Source:
    """Trim or cutter curve in world and surface parameter space."""

    edge: int  # BRep edge, -1 for a cutter.
    world: NurbsCurve  # 3D curve.
    uv: NurbsCurve  # Curve in surface parameter space.


@dataclass
class _Run:
    """Parameter run along one source curve."""

    source: int  # Index into the sources.
    a: float  # Start parameter.
    b: float  # End parameter.


@dataclass
class _Span:
    """Knot span of a source curve with its cut parameters."""

    source: int  # Index into the sources.
    a: float  # Start parameter.
    b: float  # End parameter.
    cuts: list[float]  # Cut parameters, span ends included.
    curve: NurbsCurve  # Span of the source uv curve.


@dataclass
class _Directed:
    """Directed half-edge of the trim graph."""

    b: int  # Head vertex.
    run: _Run  # Parameter run along the source.


@dataclass
class _Graph:
    """Planar trim graph, twin half-edges at index ^ 1."""

    edges: list[_Directed] = field(default_factory=list)  # Half-edges.
    outgoing: list[list[int]] = field(
        default_factory=list
    )  # Half-edges leaving each vertex, sorted by angle.


@dataclass
class _Cycle:
    """Closed loop of runs with its sampled polygon."""

    area: float  # Signed area in parameter space.
    loop: list[_Run]  # Runs around the loop.
    points: list[Point]  # Sampled polygon.


@dataclass
class _Piece:
    """New BRep edge cut from a source."""

    source: int  # Index into the sources.
    lo: float  # Start parameter on the world curve.
    hi: float  # End parameter on the world curve.
    edge: int  # BRep edge.


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════
def _require(condition: bool, message: str) -> None:
    """Raise ValueError when the condition fails."""
    if not condition:
        raise ValueError(message)


def _check_tolerance(tolerance: float) -> None:
    """Reject a tolerance that is not finite and positive."""
    _require(
        math.isfinite(tolerance) and tolerance > 0.0,
        "Split tolerance must be finite and positive",
    )


def _check_curve(curve: NurbsCurve) -> None:
    """Reject an invalid curve or one with non-finite controls or non-positive weights."""
    _require(curve.is_valid(), "Split requires valid curves")

    for i in range(curve.cv_count()):
        p = curve.get_cv(i)
        w = curve.weight(i)
        _require(
            math.isfinite(p[0])
            and math.isfinite(p[1])
            and math.isfinite(p[2])
            and math.isfinite(w)
            and w > 0.0,
            "Split requires finite controls and positive rational weights",
        )


def _check_surface(surface: NurbsSurface) -> None:
    """Reject an invalid surface or one with non-finite controls or non-positive weights."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Curve parameters
# ═══════════════════════════════════════════════════════════════════════════
def _clamp(value: float, lo: float, hi: float) -> float:
    """Value limited to [lo, hi]."""
    return max(lo, min(hi, value))


def _interval(curve: NurbsCurve, a: float, b: float) -> NurbsCurve:
    """Copy of a curve trimmed to [a, b], clamped to its domain."""
    result = curve.duplicate()
    lo = curve.domain_start()
    hi = curve.domain_end()
    a = _clamp(a, lo, hi)
    b = _clamp(b, lo, hi)
    _require(b > a, "Split produced an empty curve interval")

    if a > lo or b < hi:
        _require(result.trim(a, b), "Kernel refused a split interval")

    return result


def _closest_segments(curve: NurbsCurve, point: Point, t: float) -> tuple[float, float]:
    """Closest parameter and distance on a degree-1 curve, exact per segment."""
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


def _closest(curve: NurbsCurve, point: Point) -> tuple[float, float]:
    """Closest parameter and distance from a point to a curve, polished by Newton steps."""
    t = Closest.curve_point(curve, point)[0]

    if curve.degree() == 1:
        return _closest_segments(curve, point, t)

    lo = curve.domain_start()
    hi = curve.domain_end()

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
    """Sorted parameters clamped to [lo, hi], dropping near duplicates."""
    values = sorted(values)
    result = []

    for value in values:
        value = _clamp(value, lo, hi)

        if not result or value - result[-1] > (hi - lo) * _EPSILON * 16.0:
            result.append(value)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Curve intersection
# ═══════════════════════════════════════════════════════════════════════════
def _flat(curve: NurbsCurve, tolerance: float) -> bool:
    """True when every control point lies within tolerance of the chord."""
    a = curve.point_at_start()
    b = curve.point_at_end()
    v = b - a
    length2 = v.dot(v)

    if length2 <= tolerance * tolerance:
        return _Bounds(curve).diagonal() <= tolerance

    for i in range(curve.cv_count()):
        p = curve.get_cv(i)
        t = (p - a).dot(v) / length2

        if t < -_EPSILON or t > 1.0 + _EPSILON or p.distance(a + v * t) > tolerance:
            return False

    return True


def _refine(a: NurbsCurve, b: NurbsCurve, ta: float, tb: float) -> tuple[float, float]:
    """Newton refinement of a curve-curve intersection seed."""
    a0 = a.domain_start()
    a1 = a.domain_end()
    b0 = b.domain_start()
    b1 = b.domain_end()

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
        converged = abs(na - ta) < _EPSILON * (a1 - a0) and abs(nb - tb) < _EPSILON * (
            b1 - b0
        )
        ta = na
        tb = nb

        if converged:
            break

    return ta, tb


def _check_overlap(pair: _Pair, tolerance: float) -> None:
    """Raise when two flat pieces overlap along a shared line."""
    ap = pair.a.point_at_start()
    aq = pair.a.point_at_end()
    bp = pair.b.point_at_start()
    bq = pair.b.point_at_end()
    u = aq - ap
    v = bq - bp
    aa = u.dot(u)
    ab = u.dot(v)
    vv = v.dot(v)
    parallel = (
        aa > tolerance * tolerance
        and vv > tolerance * tolerance
        and aa * vv - ab * ab < _EPSILON * _EPSILON * aa * vv
    )

    if not parallel:
        return

    t0 = (bp - ap).dot(u) / aa
    t1 = (bq - ap).dot(u) / aa
    gap = bp.distance(ap + u * t0)
    shared = min(1.0, max(t0, t1)) - max(0.0, min(t0, t1))

    if gap <= tolerance and shared > tolerance / math.sqrt(aa):
        raise ValueError("Overlapping curves do not define isolated split points")


def _duplicate(
    a: NurbsCurve,
    b: NurbsCurve,
    hits: list[tuple[float, float]],
    ta: float,
    tb: float,
    tolerance: float,
) -> bool:
    """True when a hit is already recorded within tolerance on both curves."""
    for hit in hits:
        near_a = (
            a.point_at(hit[0]).distance(a.point_at(ta)) <= tolerance * 2.0
            and a.point_at((hit[0] + ta) * 0.5).distance(a.point_at(ta))
            <= tolerance * 2.0
        )
        near_b = (
            b.point_at(hit[1]).distance(b.point_at(tb)) <= tolerance * 2.0
            and b.point_at((hit[1] + tb) * 0.5).distance(b.point_at(tb))
            <= tolerance * 2.0
        )

        if near_a and near_b:
            return True

    return False


def _add_hit(
    a: NurbsCurve,
    b: NurbsCurve,
    pair: _Pair,
    tolerance: float,
    hits: list[tuple[float, float]],
) -> None:
    """Record the refined crossing of two flat pieces unless it is a duplicate."""
    _check_overlap(pair, tolerance)
    ta, tb, d = Closest.curve_curve(pair.a, pair.b)

    if d > tolerance * 2.0:
        return

    ta, tb = _refine(pair.a, pair.b, ta, tb)

    if a.point_at(ta).distance(b.point_at(tb)) > tolerance:
        return

    if not _duplicate(a, b, hits, ta, tb, tolerance):
        hits.append((ta, tb))


def _subdivide(pair: _Pair, ba: _Bounds, bb: _Bounds, work: list[_Pair]) -> None:
    """Halve the piece with the larger box at its parameter midpoint."""
    if ba.diagonal() >= bb.diagonal():
        lo = pair.a.domain_start()
        hi = pair.a.domain_end()
        mid = (lo + hi) * 0.5
        work.append(_Pair(_interval(pair.a, lo, mid), pair.b, pair.depth + 1))
        work.append(_Pair(_interval(pair.a, mid, hi), pair.b, pair.depth + 1))
        return

    lo = pair.b.domain_start()
    hi = pair.b.domain_end()
    mid = (lo + hi) * 0.5
    work.append(_Pair(pair.a, _interval(pair.b, lo, mid), pair.depth + 1))
    work.append(_Pair(pair.a, _interval(pair.b, mid, hi), pair.depth + 1))


def _intersections(
    a: NurbsCurve, b: NurbsCurve, tolerance: float, budget: list[int]
) -> list[tuple[float, float]]:
    """Sorted parameter pairs where two curves cross, drawing on a shared work budget."""
    av = a.get_span_vector()
    bv = b.get_span_vector()
    _require(len(av) > 1 and len(bv) > 1, "Split requires nonempty curve spans")
    _require(
        len(av) - 1 <= budget[0] // (len(bv) - 1),
        "Curve intersection exceeds the bounded split workload",
    )
    work = []

    for i in range(1, len(av)):
        for j in range(1, len(bv)):
            work.append(
                _Pair(_interval(a, av[i - 1], av[i]), _interval(b, bv[j - 1], bv[j]), 0)
            )

    hits = []

    while work:
        _require(budget[0] > 0, "Curve intersection exceeds the bounded split workload")
        budget[0] -= 1
        pair = work.pop()
        ba = _Bounds(pair.a)
        bb = _Bounds(pair.b)

        if not ba.overlaps(bb, tolerance):
            continue

        if (
            _flat(pair.a, tolerance * 0.1) and _flat(pair.b, tolerance * 0.1)
        ) or pair.depth >= 48:
            _add_hit(a, b, pair, tolerance, hits)
        else:
            _subdivide(pair, ba, bb, work)

    hits.sort()
    return hits


# ═══════════════════════════════════════════════════════════════════════════
# Trim polygons
# ═══════════════════════════════════════════════════════════════════════════
def _pullback(
    surface: NurbsSurface, curve: NurbsCurve, tolerance: float
) -> list[NurbsCurve]:
    """Curves in surface parameter space, exact on a bilinear parallelogram patch."""
    bilinear = (
        surface.m_cv_count[0] == 2
        and surface.m_cv_count[1] == 2
        and surface.m_order[0] == 2
        and surface.m_order[1] == 2
        and not surface.m_is_rat
    )

    if not bilinear:
        return Closest.surface_curve(surface, curve, 0.0, 0.0, tolerance)

    p = surface.get_cv(0, 0)
    u = surface.get_cv(1, 0) - p
    v = surface.get_cv(0, 1) - p
    last = surface.get_cv(1, 1)
    uu = u.dot(u)
    uv = u.dot(v)
    vv = v.dot(v)
    det = uu * vv - uv * uv
    parallelogram = (
        det > _EPSILON * _EPSILON * uu * vv and last.distance(p + u + v) <= tolerance
    )

    if not parallelogram:
        return Closest.surface_curve(surface, curve, 0.0, 0.0, tolerance)

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
        _require(
            result.set_cv_4d(
                i, (u0 + a * (u1 - u0)) * w, (v0 + b * (v1 - v0)) * w, 0.0, w
            ),
            "Kernel refused a pullback control",
        )

    return [result]


def _polygon(curve: NurbsCurve, tolerance: float) -> list[Point]:
    """Points sampling a curve until each piece is flat within tolerance."""
    spans = curve.get_span_vector()
    work = []

    for i in range(len(spans), 1, -1):
        work.append(_Part(_interval(curve, spans[i - 2], spans[i - 1]), 0))

    result = []
    visited = 0

    while work:
        visited += 1
        _require(visited <= _WORK_LIMIT, "Trim sampling exceeds the bounded workload")
        part = work.pop()

        if _flat(part.curve, tolerance * 0.25):
            result.append(part.curve.point_at_start())
            continue

        _require(part.depth < 40, "Trim sampling exceeds parameter precision")
        lo = part.curve.domain_start()
        hi = part.curve.domain_end()
        mid = (lo + hi) * 0.5
        work.append(_Part(_interval(part.curve, mid, hi), part.depth + 1))
        work.append(_Part(_interval(part.curve, lo, mid), part.depth + 1))

    return result


def _inside(p: Point, polygon: list[Point]) -> bool:
    """Even-odd point in polygon test in the xy plane."""
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
    """True inside the first loop and outside every hole loop."""
    if not loops or not _inside(p, loops[0]):
        return False

    for i in range(1, len(loops)):
        if _inside(p, loops[i]):
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Trim arrangement
# ═══════════════════════════════════════════════════════════════════════════
def _compute_spans(sources: list[_Source], tolerance: float) -> list[_Span]:
    """Knot spans of every source, cut at their mutual intersections."""
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
            hits = _intersections(spans[i].curve, spans[j].curve, tolerance, budget)

            for hit in hits:
                spans[i].cuts.append(hit[0])
                spans[j].cuts.append(hit[1])

    return spans


def _span_cuts(span: _Span, tolerance: float) -> list[float]:
    """Cut parameters of a span, snapped to its ends and deduplicated."""
    cuts = list(span.cuts)

    for k in range(len(cuts)):
        p = span.curve.point_at(cuts[k])

        if p.distance(span.curve.point_at(span.a)) <= tolerance:
            cuts[k] = span.a
        elif p.distance(span.curve.point_at(span.b)) <= tolerance:
            cuts[k] = span.b

    return _unique_parameters(cuts, span.a, span.b)


def _node(graph: _Graph, vertices: list[Point], p: Point, tolerance: float) -> int:
    """Index of the graph vertex at a point, added when none lies within tolerance."""
    for i in range(len(vertices)):
        if p.distance(vertices[i]) <= tolerance * 4.0:
            return i

    vertices.append(p)
    graph.outgoing.append([])
    return len(vertices) - 1


def _angle(sources: list[_Source], run: _Run) -> float:
    """Direction angle of a run leaving its start vertex."""
    d = sources[run.source].uv.evaluate(run.a, 1)[1]
    sign = 1.0 if run.b > run.a else -1.0
    return math.atan2(sign * d[1], sign * d[0])


def _compute_graph(
    spans: list[_Span],
    sources: list[_Source],
    original_loops: list[list[Point]],
    tolerance: float,
) -> _Graph:
    """Half-edge graph of the cut spans inside the original loops."""
    graph = _Graph()
    vertices = []

    for span in spans:
        cuts = _span_cuts(span, tolerance)
        source = sources[span.source]

        for i in range(1, len(cuts)):
            lo = cuts[i - 1]
            hi = cuts[i]

            if source.edge < 0 and not _inside_loops(
                source.uv.point_at((lo + hi) * 0.5), original_loops
            ):
                continue

            a = _node(graph, vertices, source.uv.point_at(lo), tolerance)
            b = _node(graph, vertices, source.uv.point_at(hi), tolerance)

            if a == b:
                continue

            index = len(graph.edges)
            graph.edges.append(_Directed(b, _Run(span.source, lo, hi)))
            graph.edges.append(_Directed(a, _Run(span.source, hi, lo)))
            graph.outgoing[a].append(index)
            graph.outgoing[b].append(index + 1)

    angles = []

    for edge in graph.edges:
        angles.append(_angle(sources, edge.run))

    for choices in graph.outgoing:
        choices.sort(key=angles.__getitem__)

    return graph


def _signed_area(points: list[Point]) -> float:
    """Signed shoelace area of a closed polygon in the xy plane."""
    area = 0.0

    for i in range(len(points)):
        a = points[i]
        b = points[(i + 1) % len(points)]
        area += (a[0] * b[1] - b[0] * a[1]) * 0.5

    return area


def _trace_cycle(
    graph: _Graph,
    sources: list[_Source],
    initial: int,
    used: list[bool],
    tolerance: float,
) -> _Cycle:
    """Loop traced from one half-edge by turning to the previous outgoing half-edge at each vertex."""
    cycle = _Cycle(0.0, [], [])
    edge = initial

    while not used[edge]:
        used[edge] = True
        item = graph.edges[edge]
        run = item.run
        cycle.loop.append(run)
        part = _interval(sources[run.source].uv, min(run.a, run.b), max(run.a, run.b))

        if run.b < run.a:
            _require(part.reverse(), "Kernel refused to reverse a trim fragment")

        cycle.points.extend(_polygon(part, tolerance))
        options = graph.outgoing[item.b]
        _require((edge ^ 1) in options, "Invalid trim graph adjacency")
        slot = options.index(edge ^ 1)
        edge = options[(slot + len(options) - 1) % len(options)]

    _require(edge == initial, "Invalid trim graph cycle")
    cycle.area = _signed_area(cycle.points)
    return cycle


def _left_of(sources: list[_Source], run: _Run, tolerance: float) -> Point:
    """Point eight tolerances left of the middle of a run."""
    curve = sources[run.source].uv
    t = (run.a + run.b) * 0.5
    p = curve.point_at(t)
    d = curve.evaluate(t, 1)[1]
    sign = 1.0 if run.b > run.a else -1.0
    length = math.hypot(d[0], d[1])
    _require(length > _EPSILON, "Cannot orient a degenerate trim fragment")
    return Point(
        p[0] - sign * d[1] / length * tolerance * 8.0,
        p[1] + sign * d[0] / length * tolerance * 8.0,
        0.0,
    )


def _compute_cycles(
    graph: _Graph,
    sources: list[_Source],
    original_loops: list[list[Point]],
    tolerance: float,
) -> list[_Cycle]:
    """Non-degenerate graph cycles whose interior lies inside the original loops."""
    cycles = []
    used = [False] * len(graph.edges)

    for initial in range(len(graph.edges)):
        if used[initial]:
            continue

        cycle = _trace_cycle(graph, sources, initial, used, tolerance)

        if abs(cycle.area) <= tolerance * tolerance:
            continue

        if _inside_loops(_left_of(sources, cycle.loop[0], tolerance), original_loops):
            cycles.append(cycle)

    return cycles


def _nest_cycles(cycles: list[_Cycle], tolerance: float) -> list[list[list[_Run]]]:
    """Regions as outer loops, each hole nested in the smallest outer loop around it."""
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


def _arrange(
    sources: list[_Source], original_loops: list[list[Point]], tolerance: float
) -> list[list[list[_Run]]]:
    """Regions of the planar arrangement of the sources inside the original loops."""
    spans = _compute_spans(sources, tolerance)
    graph = _compute_graph(spans, sources, original_loops, tolerance)
    cycles = _compute_cycles(graph, sources, original_loops, tolerance)
    return _nest_cycles(cycles, tolerance)


# ═══════════════════════════════════════════════════════════════════════════
# BRep assembly
# ═══════════════════════════════════════════════════════════════════════════
def _vertex(result: BRep, p: Point, tolerance: float) -> int:
    """Index of the BRep vertex at a point, added when none lies within tolerance."""
    for i in range(len(result.m_vertices)):
        if result.m_vertices[i].point.distance(p) <= tolerance:
            return i

    return result.add_vertex(p, tolerance)


def _lifted_gap(surface: NurbsSurface, uv: NurbsCurve, p: Point, t: float) -> float:
    """Distance from a point to the surface under a pcurve parameter."""
    q = uv.point_at(t)
    return surface.point_at(q[0], q[1]).distance(p)


def _lifted_parameter(
    surface: NurbsSurface, uv: NurbsCurve, p: Point, expected: float, tolerance: float
) -> float:
    """Pcurve parameter whose surface point meets a world point, sampled then refined by ternary search."""
    lo = uv.domain_start()
    hi = uv.domain_end()

    if _lifted_gap(surface, uv, p, expected) <= tolerance:
        return expected

    best = expected
    d = _lifted_gap(surface, uv, p, best)
    index = 0

    for i in range(129):
        t = lo + (hi - lo) * i / 128.0
        value = _lifted_gap(surface, uv, p, t)

        if value < d:
            d = value
            best = t
            index = i

    a = lo + (hi - lo) * max(0, index - 1) / 128.0
    b = lo + (hi - lo) * min(128, index + 1) / 128.0

    for k in range(60):
        x = a + (b - a) / 3.0
        y = b - (b - a) / 3.0

        if _lifted_gap(surface, uv, p, x) < _lifted_gap(surface, uv, p, y):
            b = y
        else:
            a = x

    mid = (a + b) * 0.5

    if _lifted_gap(surface, uv, p, mid) < d:
        best = mid

    _require(
        _lifted_gap(surface, uv, p, best) <= tolerance * 4.0,
        "Cannot keep an adjacent trim on its original shared edge",
    )
    return best


def _world_parameter(
    source: _Source, t: float, p: Point, tolerance: float
) -> tuple[float, float]:
    """World curve parameter and distance for a surface point, proportional guess first."""
    lo = source.world.domain_start()
    hi = source.world.domain_end()
    a = source.uv.domain_start()
    b = source.uv.domain_end()
    expected = lo + (t - a) / (b - a) * (hi - lo)
    gap = source.world.point_at(expected).distance(p)

    if gap <= tolerance:
        return expected, gap

    return _closest(source.world, p)


def _world_run(
    surface: NurbsSurface, source: _Source, run: _Run, tolerance: float
) -> tuple[float, float]:
    """World curve parameters of both run ends, a closed curve's seam end moved to the domain end."""
    qa = source.uv.point_at(run.a)
    qb = source.uv.point_at(run.b)
    wa, da = _world_parameter(source, run.a, surface.point_at(qa[0], qa[1]), tolerance)
    wb, db = _world_parameter(source, run.b, surface.point_at(qb[0], qb[1]), tolerance)
    _require(
        da <= tolerance * 4.0 and db <= tolerance * 4.0,
        "Cutter is not on the selected surface",
    )

    w0 = source.world.domain_start()
    w1 = source.world.domain_end()
    c0 = source.uv.domain_start()
    c1 = source.uv.domain_end()

    if source.world.is_closed():
        if abs(wa - w0) < (w1 - w0) * _EPSILON and run.a > (c0 + c1) * 0.5:
            wa = w1

        if abs(wb - w0) < (w1 - w0) * _EPSILON and run.b > (c0 + c1) * 0.5:
            wb = w1

    return wa, wb


def _add_shared_pcurves(
    result: BRep,
    brep: BRep,
    source: _Source,
    piece: _Piece,
    world: NurbsCurve,
    tolerance: float,
) -> None:
    """Pcurves of a piece of a shared BRep edge, cut from every adjacent face trim."""
    w0 = source.world.domain_start()
    w1 = source.world.domain_end()

    for pc in brep.m_edges[source.edge].pcurves:
        surface = brep.m_surfaces[pc.surface_index]
        sides = [pc.curve_2d_index, pc.curve_2d_index_2]
        ids = [-1, -1]

        for at in range(2):
            if sides[at] < 0:
                continue

            c = brep.m_curves_2d[sides[at]]
            c0 = c.domain_start()
            c1 = c.domain_end()
            ca = _lifted_parameter(
                surface,
                c,
                world.point_at_start(),
                c0 + (piece.lo - w0) / (w1 - w0) * (c1 - c0),
                tolerance,
            )
            cb = _lifted_parameter(
                surface,
                c,
                world.point_at_end(),
                c0 + (piece.hi - w0) / (w1 - w0) * (c1 - c0),
                tolerance,
            )
            _require(cb > ca, "A split crosses an unsupported periodic trim seam")
            ids[at] = result.add_curve_2d(_interval(c, ca, cb))

        result.add_pcurve(piece.edge, pc.surface_index, ids[0], ids[1])


def _add_run_edge(
    result: BRep,
    pieces: list[_Piece],
    brep: BRep,
    surface_index: int,
    sources: list[_Source],
    run: _Run,
    tolerance: float,
) -> BRepRef:
    """Oriented BRep edge for a run, reusing the edge already cut for the same stretch of its source."""
    source = sources[run.source]
    wa, wb = _world_run(brep.m_surfaces[surface_index], source, run, tolerance)
    lo = min(wa, wb)
    hi = max(wa, wb)
    w0 = source.world.domain_start()
    w1 = source.world.domain_end()
    _require(hi - lo > (w1 - w0) * _EPSILON, "Split would create a collapsed edge")
    orientation = _FORWARD if wa < wb else _REVERSED

    for piece in pieces:
        same = (
            sources[piece.source].edge == source.edge
            if source.edge >= 0
            else piece.source == run.source
        )

        if (
            same
            and source.world.point_at(lo).distance(source.world.point_at(piece.lo))
            <= tolerance * 4.0
            and source.world.point_at(hi).distance(source.world.point_at(piece.hi))
            <= tolerance * 4.0
        ):
            return BRepRef(piece.edge, orientation)

    world = _interval(source.world, lo, hi)
    a = _vertex(result, world.point_at_start(), tolerance * 4.0)
    b = _vertex(result, world.point_at_end(), tolerance * 4.0)
    edge = result.add_edge(result.add_curve_3d(world), a, b, tolerance)
    piece = _Piece(run.source, lo, hi, edge)

    if source.edge >= 0:
        _add_shared_pcurves(result, brep, source, piece, world, tolerance)
    else:
        pc = _interval(source.uv, min(run.a, run.b), max(run.a, run.b))

        if (wb - wa) * (run.b - run.a) < 0.0:
            _require(pc.reverse(), "Kernel refused to reverse a cutter trim")

        result.add_pcurve(edge, surface_index, result.add_curve_2d(pc))

    pieces.append(piece)
    return BRepRef(edge, orientation)


def _boundary_loops(
    brep: BRep,
    face_index: int,
    uv_tolerance: float,
    sources: list[_Source],
) -> list[list[Point]]:
    """Sampled trim loops of a face, each boundary edge added to the sources."""
    loops = []

    for wr in brep.m_faces[face_index].wires:
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
                _require(uv.reverse(), "Kernel refused to reverse a face trim")

            points.extend(_polygon(uv, uv_tolerance))

        _require(len(points) >= 3, "Face has an invalid boundary")
        loops.append(points)

    return loops


def _replace_wires(
    result: BRep, brep: BRep, sources: list[_Source], pieces: list[_Piece]
) -> None:
    """Replace every use of a cut boundary edge in the wires by its pieces in order."""
    replacements = {}

    for piece in pieces:
        if sources[piece.source].edge >= 0:
            replacements.setdefault(sources[piece.source].edge, []).append(
                (piece.lo, piece.edge)
            )

    for wi in range(len(brep.m_wires)):
        refs = []

        for er in brep.m_wires[wi].edges:
            if er.index not in replacements:
                refs.append(copy.deepcopy(er))
                continue

            items = sorted(set(replacements[er.index]))

            if er.orientation == _REVERSED:
                items.reverse()

            for item in items:
                refs.append(BRepRef(item[1], er.orientation))

        result.m_wires[wi].edges = refs


def _add_faces(
    result: BRep, face: BRepFace, face_index: int, new_wires: list[list[BRepRef]]
) -> None:
    """Put the first region on the split face and the others on new faces beside it in every shell."""
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


def _check_wires(result: BRep) -> None:
    """Reject a face boundary whose consecutive edges do not share a vertex."""
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


def _validate(result: BRep, original: BRep, tolerance: float) -> None:
    """Reject a split that opens a shell, leaves a wire open or moves an edge off its vertices or trims."""
    _require(result.is_valid(), "Split produced invalid BRep references")

    for s in range(len(original.m_shells)):
        if original.is_closed(s):
            _require(result.is_closed(s), "Split would open a joined shell")

    _check_wires(result)

    for edge in result.m_edges:
        if edge.degenerated:
            continue

        world = result.m_curves_3d[edge.curve_3d_index]
        start = world.point_at_start().distance(
            result.m_vertices[edge.start_vertex].point
        )
        end = world.point_at_end().distance(result.m_vertices[edge.end_vertex].point)
        _require(
            start <= tolerance * 4.0 and end <= tolerance * 4.0,
            "Split edge does not meet its vertices",
        )

        for pc in edge.pcurves:
            for ci in [pc.curve_2d_index, pc.curve_2d_index_2]:
                if ci < 0:
                    continue

                uv = result.m_curves_2d[ci]
                lo = uv.domain_start()
                hi = uv.domain_end()

                for k in range(33):
                    q = uv.point_at(lo + (hi - lo) * k / 32.0)
                    p = result.m_surfaces[pc.surface_index].point_at(q[0], q[1])
                    _require(
                        _closest(world, p)[1] <= max(tolerance, edge.tolerance) * 8.0,
                        "Split edge and surface trim do not coincide",
                    )


# ═══════════════════════════════════════════════════════════════════════════
# Split
# ═══════════════════════════════════════════════════════════════════════════
def split_curve_by_curves(
    curve: NurbsCurve, cutters: list[NurbsCurve], tolerance: float
) -> list[NurbsCurve]:
    """Split a curve at isolated 3D intersections, retaining every piece and rejecting overlapping cutters."""
    _check_tolerance(tolerance)
    _check_curve(curve)
    _require(len(cutters) > 0, "Select at least one cutter")
    lo = curve.domain_start()
    hi = curve.domain_end()
    cuts = [lo, hi]
    cut_at_seam = False
    budget = [_WORK_LIMIT]

    for cutter in cutters:
        _check_curve(cutter)
        hits = _intersections(curve, cutter, tolerance, budget)

        for hit in hits:
            a = hit[0]

            if (
                abs(a - lo) <= (hi - lo) * _EPSILON * 16.0
                or abs(a - hi) <= (hi - lo) * _EPSILON * 16.0
            ):
                cut_at_seam = True

            cuts.append(a)

    cuts = _unique_parameters(cuts, lo, hi)

    if len(cuts) == 2:
        return [copy.deepcopy(curve)]

    result = []

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
    original_loops = _boundary_loops(brep, face_index, uv_tolerance, sources)

    for cutter in cutters:
        _check_curve(cutter)

        for uv in _pullback(surface, cutter, tolerance):
            sources.append(_Source(-1, cutter, uv))

    regions = _arrange(sources, original_loops, uv_tolerance)

    if len(regions) < 2:
        return copy.deepcopy(brep)

    result = copy.deepcopy(brep)
    pieces = []
    new_wires = []

    for region in regions:
        wires = []

        for loop in region:
            refs = []

            for run in loop:
                refs.append(
                    _add_run_edge(
                        result,
                        pieces,
                        brep,
                        face.surface_index,
                        sources,
                        run,
                        tolerance,
                    )
                )

            wires.append(BRepRef(result.add_wire(refs), _FORWARD))

        new_wires.append(wires)

    _replace_wires(result, brep, sources, pieces)
    _add_faces(result, face, face_index, new_wires)
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
    uv = [
        Point(u0, v0, 0.0),
        Point(u1, v0, 0.0),
        Point(u1, v1, 0.0),
        Point(u0, v1, 0.0),
    ]
    at = [v0, u1, v1, u0]
    edges = []

    for i in range(4):
        curve = surface.iso_curve(i % 2, at[i])

        if i >= 2:
            _require(curve.reverse(), "Kernel refused to reverse a natural boundary")

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
    curve = NurbsCurve.create(False, 1, [line.point_at(0.0), line.point_at(1.0)])
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
