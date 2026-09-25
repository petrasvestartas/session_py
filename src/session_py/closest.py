from __future__ import annotations
from typing import TYPE_CHECKING
import math
from .aabb import AABB
from .line import Line
from .nurbscurve import NurbsCurve
from .nurbsknot import CurveNurbsKnotStyle
from .point import Point
from .polyline import Polyline
from .spatial_aabbtree import SpatialAABBTree
from .spatial_kdtree import SpatialKDTree

if TYPE_CHECKING:
    from .mesh import Mesh
    from .nurbssurface import NurbsSurface
    from .pointcloud import PointCloud

STACK_SIZE = 64


# ═══════════════════════════════════════════════════════════════════════════
# Curve helpers
# ═══════════════════════════════════════════════════════════════════════════
def _curve_seed(curve: NurbsCurve, test_point: Point, t0: float, t1: float) -> float:
    """Parameter of the closest sample on a dense grid over [t0, t1]."""

    num_samples = max(50, curve.cv_count() * 10)
    dt = (t1 - t0) / num_samples
    best_t = t0
    best_dist = curve.point_at(t0).distance(test_point)

    for i in range(num_samples + 1):
        t = t0 + i * dt
        dist = curve.point_at(t).distance(test_point)

        if dist < best_dist:
            best_dist = dist
            best_t = t

    return best_t


def _curve_newton(
    curve: NurbsCurve, test_point: Point, t0: float, t1: float, t: float
) -> float:
    """Newton on (C(t) - P) . C'(t) = 0 from t, clamped to [t0, t1]."""

    max_iterations = 32
    step_tolerance = (t1 - t0) * 1e-12

    for _ in range(max_iterations):
        derivs = curve.evaluate(t, 2)

        if len(derivs) < 3:
            break

        pt = derivs[0]
        d1 = derivs[1]
        d2 = derivs[2]
        rx = pt[0] - test_point[0]
        ry = pt[1] - test_point[1]
        rz = pt[2] - test_point[2]
        f = rx * d1[0] + ry * d1[1] + rz * d1[2]

        if abs(f) < step_tolerance:
            break

        df = (
            d1[0] * d1[0]
            + d1[1] * d1[1]
            + d1[2] * d1[2]
            + rx * d2[0]
            + ry * d2[1]
            + rz * d2[2]
        )

        if abs(df) < 1e-14:
            break

        dt_step = -f / df

        if abs(dt_step) > (t1 - t0) * 0.5:
            dt_step = math.copysign((t1 - t0) * 0.5, dt_step)

        t += dt_step

        if t < t0:
            t = t0

        if t > t1:
            t = t1

        if abs(dt_step) < step_tolerance:
            break

    return t


def _curve_curve_seed(curve0: NurbsCurve, curve1: NurbsCurve) -> tuple[float, float]:
    """Parameters of the closest pair on dense grids over both domains."""

    u0 = curve0.domain_start()
    u1 = curve0.domain_end()
    v0 = curve1.domain_start()
    v1 = curve1.domain_end()
    n0 = max(40, curve0.cv_count() * 8)
    n1 = max(40, curve1.cv_count() * 8)
    p0 = []
    p1 = []

    for i in range(n0 + 1):
        p0.append(curve0.point_at(u0 + (u1 - u0) * i / n0))

    for j in range(n1 + 1):
        p1.append(curve1.point_at(v0 + (v1 - v0) * j / n1))

    best = math.inf
    u = u0
    v = v0

    for i in range(n0 + 1):
        for j in range(n1 + 1):
            d2 = (p0[i] - p1[j]).magnitude_squared()

            if d2 < best:
                best = d2
                u = u0 + (u1 - u0) * i / n0
                v = v0 + (v1 - v0) * j / n1

    return u, v


# ═══════════════════════════════════════════════════════════════════════════
# Surface helpers
# ═══════════════════════════════════════════════════════════════════════════
def _surface_seed(
    surface: NurbsSurface,
    test_point: Point,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
) -> tuple[float, float]:
    """Parameters of the closest sample on a grid whose resolution follows the window size."""

    domain_u0, domain_u1 = surface.domain(0)
    domain_v0, domain_v1 = surface.domain(1)

    full_u = max(10, surface.order(0))
    full_v = max(10, surface.order(1))
    u_frac = (u1 - u0) / max(domain_u1 - domain_u0, 1e-12)
    v_frac = (v1 - v0) / max(domain_v1 - domain_v0, 1e-12)
    u_samples = max(3, math.ceil(full_u * min(1.0, u_frac)))
    v_samples = max(3, math.ceil(full_v * min(1.0, v_frac)))
    du_param = (u1 - u0) / u_samples
    dv_param = (v1 - v0) / v_samples
    best_u = u0
    best_v = v0
    best_dist = math.inf

    for i in range(u_samples + 1):
        for j in range(v_samples + 1):
            uu = u0 + i * du_param
            vv = v0 + j * dv_param
            dist = surface.point_at(uu, vv).distance(test_point)

            if dist < best_dist:
                best_dist = dist
                best_u = uu
                best_v = vv

    return best_u, best_v


def _surface_newton(
    surface: NurbsSurface,
    test_point: Point,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    seed: tuple[float, float],
) -> tuple[float, float]:
    """Newton on the perpendicular-foot conditions from the seed, clamped to the window."""

    u, v = seed
    max_iterations = 20
    step_tolerance = min(u1 - u0, v1 - v0) * 1e-10
    max_step = min(u1 - u0, v1 - v0) * 0.5

    for _ in range(max_iterations):
        derivs = surface.evaluate(u, v, 1)

        if len(derivs) < 3:
            break

        pt = surface.point_at(u, v)
        du_vec = derivs[2]
        dv_vec = derivs[1]
        delta = test_point - pt
        fu = -delta.dot(du_vec)
        fv = -delta.dot(dv_vec)

        if abs(fu) < step_tolerance and abs(fv) < step_tolerance:
            break

        duu = du_vec.dot(du_vec)
        dvv = dv_vec.dot(dv_vec)
        duv = du_vec.dot(dv_vec)
        det = duu * dvv - duv * duv

        if abs(det) < 1e-12:
            break

        du_step = (dvv * fu - duv * fv) / det
        dv_step = (duu * fv - duv * fu) / det

        if abs(du_step) > max_step:
            du_step = math.copysign(max_step, du_step)

        if abs(dv_step) > max_step:
            dv_step = math.copysign(max_step, dv_step)

        u = max(u0, min(u1, u - du_step))
        v = max(v0, min(v1, v - dv_step))

        if abs(du_step) < step_tolerance and abs(dv_step) < step_tolerance:
            break

    return u, v


# ═══════════════════════════════════════════════════════════════════════════
# Pullback helpers
# ═══════════════════════════════════════════════════════════════════════════
class _Pullback:
    """Surface domain, trace step and tolerances shared by the surface_curve steps."""

    def __init__(self):
        self.u0 = 0.0  # Surface domain start in u.
        self.u1 = 0.0  # Surface domain end in u.
        self.v0 = 0.0  # Surface domain start in v.
        self.v1 = 0.0  # Surface domain end in v.
        self.range_u = 0.0  # Domain length in u.
        self.range_v = 0.0  # Domain length in v.
        self.closed_u = False  # Surface closed in u.
        self.closed_v = False  # Surface closed in v.
        self.du = 0.0  # Quarter-span step in u.
        self.dv = 0.0  # Quarter-span step in v.
        self.step = 0.0  # Uv deviation bound of a fitted pcurve.
        self.fit_tol = 0.0  # 3d deviation bound of a lifted uv midpoint.
        self.reject_tol = 0.0  # Residual above which a sample is re-inverted globally.
        self.on_surf_tol = 0.0  # Residual above which the curve is off the surface.


def _pullback_setup(surface: NurbsSurface, tolerance: float) -> _Pullback:
    """Domain, steps and tolerances of the surface for one pullback."""

    pb = _Pullback()
    pb.u0, pb.u1 = surface.domain(0)
    pb.v0, pb.v1 = surface.domain(1)
    pb.range_u = pb.u1 - pb.u0
    pb.range_v = pb.v1 - pb.v0
    pb.closed_u = surface.is_closed(0)
    pb.closed_v = surface.is_closed(1)

    nu = max(len(surface.get_span_vector(0)) - 1, 1) * 4
    nv = max(len(surface.get_span_vector(1)) - 1, 1) * 4
    pb.du = pb.range_u / nu
    pb.dv = pb.range_v / nv

    mu = (pb.u0 + pb.u1) * 0.5
    mv = (pb.v0 + pb.v1) * 0.5
    pmid = surface.point_at(mu, mv)
    wu_probe = min(mu + pb.du, pb.u1)
    wv_probe = min(mv + pb.dv, pb.v1)
    uv_to_3d_u = pmid.distance(surface.point_at(wu_probe, mv)) / pb.du
    uv_to_3d_v = pmid.distance(surface.point_at(mu, wv_probe)) / pb.dv
    uv_to_3d = max(uv_to_3d_u, uv_to_3d_v)
    uv_to_3d_min = min(uv_to_3d_u, uv_to_3d_v)

    if uv_to_3d < 1e-10:
        uv_to_3d = 1.0

    if uv_to_3d_min < 1e-10:
        uv_to_3d_min = 1.0

    pb.step = min(pb.du, pb.dv) * 0.25
    pb.fit_tol = (
        tolerance if tolerance > 0.0 else pb.step * (uv_to_3d + uv_to_3d_min) * 0.5
    )
    pb.reject_tol = pb.fit_tol * 100.0

    corner_diag = surface.point_at(pb.u0, pb.v0).distance(
        surface.point_at(pb.u1, pb.v1)
    )

    if corner_diag < 1e-12:
        corner_diag = max(pb.range_u, pb.range_v)

    pb.on_surf_tol = corner_diag * 0.05

    return pb


def _pullback_wrap(x: float, x0: float, x1: float, closed: bool) -> float:
    """x folded into [x0, x1] by period when closed, clamped otherwise."""

    if not closed:
        return max(x0, min(x, x1))

    t = math.fmod(x - x0, x1 - x0)

    if t < 0:
        t += x1 - x0

    return x0 + t


def _pullback_unwrap(prev: float, x: float, range_: float, closed: bool) -> float:
    """x shifted by whole periods to within half a period of prev."""

    if not closed:
        return x

    while x - prev > range_ * 0.5:
        x -= range_

    while x - prev < -range_ * 0.5:
        x += range_

    return x


def _pullback_invert(
    surface: NurbsSurface,
    pb: _Pullback,
    pt: Point,
    up: float,
    vp: float,
    wu: float,
    wv: float,
) -> tuple[float, float, float]:
    """Windowed inversion of pt around (up, vp), trying the seam-mirrored windows when closed."""

    u_centers = [up]

    if pb.closed_u and up - wu < pb.u0:
        u_centers.append(up + pb.range_u)

    if pb.closed_u and up + wu > pb.u1:
        u_centers.append(up - pb.range_u)

    v_centers = [vp]

    if pb.closed_v and vp - wv < pb.v0:
        v_centers.append(vp + pb.range_v)

    if pb.closed_v and vp + wv > pb.v1:
        v_centers.append(vp - pb.range_v)

    best = (up, vp, math.inf)

    for uc in u_centers:
        for vc in v_centers:
            wu0 = max(uc - wu, pb.u0)
            wu1 = min(uc + wu, pb.u1)
            wv0 = max(vc - wv, pb.v0)
            wv1 = min(vc + wv, pb.v1)

            if wu1 - wu0 < 1e-14 or wv1 - wv0 < 1e-14:
                continue

            res = Closest.surface_point(surface, pt, wu0, wu1, wv0, wv1)

            if res[2] < best[2]:
                best = res

            if best[2] < pb.fit_tol * 0.01:
                break

    return best


def _pullback_samples(
    surface: NurbsSurface,
    curve: NurbsCurve,
    pb: _Pullback,
    t0: float,
    t1: float,
) -> list[list[float]]:
    """Warm-started samples [t, u, v, residual] along [t0, t1], empty when the curve is off the surface."""

    n0 = max(16, 4 * curve.span_count())
    samples = []
    max_residual = 0.0
    min_residual = math.inf

    for i in range(n0 + 1):
        t = t0 + (t1 - t0) * i / n0
        pt = curve.point_at(t)

        if i == 0:
            uu, vv, rd = Closest.surface_point(surface, pt, 0.0, 0.0, 0.0, 0.0)
        else:
            prev = samples[-1]
            prev2 = samples[max(0, len(samples) - 2)]
            wu = max(pb.du, pb.dv) * 2.0 + abs(prev[1] - prev2[1])
            wv = max(pb.du, pb.dv) * 2.0 + abs(prev[2] - prev2[2])
            up = _pullback_wrap(prev[1], pb.u0, pb.u1, pb.closed_u)
            vp = _pullback_wrap(prev[2], pb.v0, pb.v1, pb.closed_v)
            ru, rv, rd = _pullback_invert(surface, pb, pt, up, vp, wu, wv)

            if rd > pb.reject_tol:
                ru, rv, rd = Closest.surface_point(surface, pt, 0.0, 0.0, 0.0, 0.0)

            uu = _pullback_unwrap(prev[1], ru, pb.range_u, pb.closed_u)
            vv = _pullback_unwrap(prev[2], rv, pb.range_v, pb.closed_v)

        samples.append([t, uu, vv, rd])
        max_residual = max(max_residual, rd)
        min_residual = min(min_residual, rd)

    if max_residual > pb.reject_tol or min_residual > pb.on_surf_tol:
        samples.clear()

    return samples


def _pullback_refine(
    surface: NurbsSurface,
    curve: NurbsCurve,
    pb: _Pullback,
    samples: list[list[float]],
) -> None:
    """Bisect every span whose lifted uv midpoint strays from the curve, up to 8 rounds or 4096 samples."""

    for _ in range(8):
        inserted = 0
        i = 0

        while i + 1 < len(samples):
            a = samples[i]
            b = samples[i + 1]
            tm = (a[0] + b[0]) * 0.5
            um = _pullback_wrap((a[1] + b[1]) * 0.5, pb.u0, pb.u1, pb.closed_u)
            vm = _pullback_wrap((a[2] + b[2]) * 0.5, pb.v0, pb.v1, pb.closed_v)
            pm = curve.point_at(tm)

            if (
                surface.point_at(um, vm).distance(pm) <= pb.fit_tol
                or len(samples) >= 4096
            ):
                i += 1
                continue

            wu = max(abs(b[1] - a[1]), pb.du)
            wv = max(abs(b[2] - a[2]), pb.dv)
            ru, rv, rd = _pullback_invert(surface, pb, pm, um, vm, wu, wv)

            if rd > pb.on_surf_tol:
                i += 1
                continue

            uu = _pullback_unwrap(a[1], ru, pb.range_u, pb.closed_u)
            vv = _pullback_unwrap(a[2], rv, pb.range_v, pb.closed_v)

            samples.insert(i + 1, [tm, uu, vv, rd])
            inserted += 1
            i += 2

        if inserted == 0:
            break


def _pullback_seam_axis(
    a: float,
    b: float,
    x0: float,
    range_: float,
    closed: bool,
    bestt: float,
) -> tuple[bool, float, float]:
    """Smallest seam crossing of one axis between a and b that beats bestt, as (found, bestt, level)."""

    if not closed or abs(b - a) <= 1e-15:
        return False, bestt, 0.0

    k0 = math.floor((a - x0) / range_)
    k1 = math.floor((b - x0) / range_)
    found = False
    level = 0.0

    for k in range(min(k0, k1) + 1, max(k0, k1) + 1):
        seam = x0 + k * range_
        t = (seam - a) / (b - a)

        if t > 1e-9 and t < 1.0 - 1e-9 and t < bestt:
            bestt = t
            level = seam
            found = True

    return found, bestt, level


def _pullback_first_seam(
    pb: _Pullback, a: list[float], b: list[float]
) -> tuple[bool, float, float]:
    """First seam crossing on segment a -> b, as (found, cu, cv)."""

    bestt = 2.0
    found = False
    cu = 0.0
    cv = 0.0
    found_u, bestt, level = _pullback_seam_axis(
        a[0], b[0], pb.u0, pb.range_u, pb.closed_u, bestt
    )

    if found_u:
        found = True
        cu = level
        cv = a[1] + (b[1] - a[1]) * bestt

    found_v, bestt, level = _pullback_seam_axis(
        a[1], b[1], pb.v0, pb.range_v, pb.closed_v, bestt
    )

    if found_v:
        found = True
        cv = level
        cu = a[0] + (b[0] - a[0]) * bestt

    return found, cu, cv


def _pullback_at_seam(x: float, x0: float, range_: float, closed: bool) -> bool:
    """True when x sits on a seam level of a closed axis."""

    if not closed:
        return False

    seam = x0 + round((x - x0) / range_) * range_

    return abs(x - seam) < range_ * 1e-6


def _pullback_on_seam(pb: _Pullback, p: list[float]) -> bool:
    """True when p sits on a seam of either closed axis."""

    return _pullback_at_seam(p[0], pb.u0, pb.range_u, pb.closed_u) or _pullback_at_seam(
        p[1], pb.v0, pb.range_v, pb.closed_v
    )


def _pullback_shift(pb: _Pullback, seg: list[list[float]]) -> None:
    """Shift a segment by whole periods so its middle point lies inside the domain."""

    mid = seg[len(seg) // 2]
    k_u = math.floor((mid[0] - pb.u0) / pb.range_u) if pb.closed_u else 0
    k_v = math.floor((mid[1] - pb.v0) / pb.range_v) if pb.closed_v else 0

    for p in seg:
        p[0] -= k_u * pb.range_u
        p[1] -= k_v * pb.range_v


def _pullback_split(
    pb: _Pullback, pts: list[list[float]], rejoin: bool
) -> tuple[list[list[list[float]]], bool]:
    """Split the unwrapped uv polyline at every seam crossing; rejoin the two arcs of a mid-arc loop start."""

    raw = []
    cur = [pts[0]]
    any_cross = False

    for i in range(1, len(pts)):
        a = pts[i - 1]
        b = pts[i]
        found, cu, cv = _pullback_first_seam(pb, a, b)

        while found:
            cur.append([cu, cv])
            raw.append(cur)
            cur = [[cu, cv]]
            any_cross = True
            a = [cu, cv]
            found, cu, cv = _pullback_first_seam(pb, a, b)

        cur.append(b)

        if i + 1 < len(pts) and _pullback_on_seam(pb, b):
            raw.append(cur)
            cur = [[b[0], b[1]]]
            any_cross = True

    raw.append(cur)

    if rejoin and len(raw) > 1:
        merged = raw[-1]

        for k in range(1, len(raw[0])):
            merged.append(raw[0][k])

        raw.pop(0)
        raw[-1] = merged

    return raw, any_cross


def _pullback_pieces(
    pb: _Pullback, curve: NurbsCurve, samples: list[list[float]]
) -> list[tuple[list[list[float]], bool]]:
    """In-domain uv pieces with a closed flag, slivers dropped."""

    pts = []

    for s in samples:
        pts.append([s[1], s[2]])

    p_first = curve.point_at(samples[0][0])
    p_last = curve.point_at(samples[-1][0])
    is_loop = p_first.distance(p_last) < pb.fit_tol * 4.0 and len(pts) >= 6

    if is_loop:
        pts.pop()

    wind_u = samples[-1][1] - samples[0][1] if pb.closed_u else 0.0
    wind_v = samples[-1][2] - samples[0][2] if pb.closed_v else 0.0
    crosses = abs(wind_u) > pb.range_u * 0.5 or abs(wind_v) > pb.range_v * 0.5
    rejoin = is_loop and not crosses and not _pullback_on_seam(pb, pts[0])
    raw, any_cross = _pullback_split(pb, pts, rejoin)
    pieces = []

    for seg in raw:
        if len(seg) < 2:
            continue

        _pullback_shift(pb, seg)

        umin = 1e300
        umax = -1e300
        vmin = 1e300
        vmax = -1e300
        length = 0.0

        for i in range(len(seg)):
            umin = min(umin, seg[i][0])
            umax = max(umax, seg[i][0])
            vmin = min(vmin, seg[i][1])
            vmax = max(vmax, seg[i][1])

            if i > 0:
                length += math.hypot(
                    seg[i][0] - seg[i - 1][0], seg[i][1] - seg[i - 1][1]
                )

        if length < min(pb.range_u, pb.range_v) * 1e-4:
            continue

        seg_loop = (
            is_loop
            and not any_cross
            and umax - umin < pb.range_u * 0.9
            and vmax - vmin < pb.range_v * 0.9
        )

        pieces.append((seg, seg_loop))

    return pieces


def _pullback_turning(pts_uv: list[Point]) -> float:
    """Total turning angle of a uv polyline."""

    mp = len(pts_uv)
    total_turning = 0.0

    for i in range(1, mp - 1):
        dx1 = pts_uv[i][0] - pts_uv[i - 1][0]
        dy1 = pts_uv[i][1] - pts_uv[i - 1][1]
        dx2 = pts_uv[i + 1][0] - pts_uv[i][0]
        dy2 = pts_uv[i + 1][1] - pts_uv[i][1]
        l1 = math.hypot(dx1, dy1)
        l2 = math.hypot(dx2, dy2)

        if l1 <= 1e-14 or l2 <= 1e-14:
            continue

        c = max(-1.0, min(1.0, (dx1 * dx2 + dy1 * dy2) / (l1 * l2)))
        total_turning += math.acos(c)

    return total_turning


def _pullback_chords(pts_uv: list[Point], piece_loop: bool) -> list[float]:
    """Normalized chord-length parameters of a uv polyline."""

    mp = len(pts_uv)
    chords = [0.0] * mp
    total_len = 0.0

    for i in range(1, mp):
        total_len += pts_uv[i].distance(pts_uv[i - 1])
        chords[i] = total_len

    if piece_loop:
        total_len += pts_uv[0].distance(pts_uv[mp - 1])

    if total_len > 1e-14:
        for i in range(1, mp):
            chords[i] /= total_len

    return chords


def _pullback_fit(
    pb: _Pullback, piece_pts: list[list[float]], piece_loop: bool
) -> NurbsCurve:
    """Fit one piece as a uv pcurve on [0, 1]; interpolation and a degree-1 polyline are the fallbacks."""

    _pullback_shift(pb, piece_pts)

    pts_uv = []

    for p in piece_pts:
        pts_uv.append(Point(p[0], p[1], 0.0))

    mp = len(pts_uv)
    chords = _pullback_chords(pts_uv, piece_loop)
    target_cvs = max(8, int(_pullback_turning(pts_uv) / 0.5) + 6)
    max_cvs = mp - 1
    pcurve = NurbsCurve()

    for _ in range(5):
        if target_cvs > max_cvs:
            break

        pcurve = NurbsCurve.create_fitted(pts_uv, target_cvs, 3, piece_loop)

        if not pcurve.is_valid():
            break

        ft0 = pcurve.domain_start()
        ft1 = pcurve.domain_end()
        max_dev = 0.0

        for i in range(mp):
            max_dev = max(
                max_dev,
                pcurve.point_at(ft0 + (ft1 - ft0) * chords[i]).distance(pts_uv[i]),
            )

        if max_dev < pb.step:
            break

        target_cvs = min(target_cvs * 2, max_cvs)

    if not pcurve.is_valid():
        if piece_loop:
            pcurve = NurbsCurve.create_interpolated(
                pts_uv, CurveNurbsKnotStyle.ChordPeriodic
            )
        else:
            pcurve = NurbsCurve.create_interpolated(pts_uv)

    if not pcurve.is_valid():
        pcurve = NurbsCurve.create(False, 1, pts_uv)

    if pcurve.is_valid():
        pcurve.set_domain(0.0, 1.0)

    return pcurve


# ═══════════════════════════════════════════════════════════════════════════
# Mesh helpers
# ═══════════════════════════════════════════════════════════════════════════
def _closest_point_on_triangle(p: Point, a: Point, b: Point, c: Point) -> Point:
    """Closest point on triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5)."""

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)

    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2

    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)

        return a + ab * v

    cp = p - c
    d5 = ab.dot(cp)
    d6 = ac.dot(cp)

    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6

    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)

        return a + ac * w

    va = d3 * d6 - d5 * d4

    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))

        return b + (c - b) * w

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return a + ab * v + ac * w


def _aabb_min_distance(aabb: AABB, p: Point) -> float:
    """Distance from p to the box, zero inside."""

    dx = max(0.0, abs(p[0] - aabb.cx) - aabb.hx)
    dy = max(0.0, abs(p[1] - aabb.cy) - aabb.hy)
    dz = max(0.0, abs(p[2] - aabb.cz) - aabb.hz)

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _aabb_to_aabb_min_dist(a: AABB, b: AABB) -> float:
    """Distance between two boxes, zero when they overlap."""

    dx = max(0.0, abs(a.cx - b.cx) - a.hx - b.hx)
    dy = max(0.0, abs(a.cy - b.cy) - a.hy - b.hy)
    dz = max(0.0, abs(a.cz - b.cz) - a.hz - b.hz)

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _mesh_face_keys(mesh: Mesh) -> list[int]:
    """Face keys in the face-index order used by the triangle caches."""
    return sorted(mesh.face.keys())


class Closest:
    """Closest-point queries between points, curves, surfaces, meshes and clouds."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Curves
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def curve_point(
        curve: NurbsCurve, test_point: Point, t0: float = 0.0, t1: float = 0.0
    ) -> tuple[float, float]:
        """Parameter and distance of the closest curve point within [t0, t1] (0 means the domain end)."""

        if not curve.is_valid():
            return (0.0, math.inf)

        domain_start = curve.domain_start()
        domain_end = curve.domain_end()

        if t0 <= 0.0:
            t0 = domain_start

        if t1 <= 0.0:
            t1 = domain_end

        t0 = max(t0, domain_start)
        t1 = min(t1, domain_end)

        t = _curve_newton(
            curve, test_point, t0, t1, _curve_seed(curve, test_point, t0, t1)
        )
        final_dist = curve.point_at(t).distance(test_point)
        dist_start = curve.point_at(t0).distance(test_point)
        dist_end = curve.point_at(t1).distance(test_point)

        if dist_start < final_dist:
            t = t0
            final_dist = dist_start

        if dist_end < final_dist:
            t = t1
            final_dist = dist_end

        return (t, final_dist)

    @staticmethod
    def curve_curve(
        curve0: NurbsCurve, curve1: NurbsCurve
    ) -> tuple[float, float, float]:
        """Return the parameters and distance of the closest approach between two curves."""

        if not curve0.is_valid() or not curve1.is_valid():
            return (0.0, 0.0, math.inf)

        u0 = curve0.domain_start()
        u1 = curve0.domain_end()
        v0 = curve1.domain_start()
        v1 = curve1.domain_end()
        u, v = _curve_curve_seed(curve0, curve1)

        for _ in range(64):
            e0 = curve0.evaluate(u, 2)
            e1 = curve1.evaluate(v, 2)

            if len(e0) < 3 or len(e1) < 3:
                break

            c0 = e0[0]
            c0p = e0[1]
            c0pp = e0[2]
            c1 = e1[0]
            c1p = e1[1]
            c1pp = e1[2]
            rx = c0[0] - c1[0]
            ry = c0[1] - c1[1]
            rz = c0[2] - c1[2]
            gu = rx * c0p[0] + ry * c0p[1] + rz * c0p[2]
            gv = -(rx * c1p[0] + ry * c1p[1] + rz * c1p[2])
            huu = (
                c0p[0] * c0p[0]
                + c0p[1] * c0p[1]
                + c0p[2] * c0p[2]
                + rx * c0pp[0]
                + ry * c0pp[1]
                + rz * c0pp[2]
            )
            huv = -(c0p[0] * c1p[0] + c0p[1] * c1p[1] + c0p[2] * c1p[2])
            hvv = (
                c1p[0] * c1p[0]
                + c1p[1] * c1p[1]
                + c1p[2] * c1p[2]
                - (rx * c1pp[0] + ry * c1pp[1] + rz * c1pp[2])
            )
            det = huu * hvv - huv * huv

            if abs(det) < 1e-14:
                break

            du = -(hvv * gu - huv * gv) / det
            dv = -(-huv * gu + huu * gv) / det

            if abs(du) > (u1 - u0) * 0.5:
                du = math.copysign((u1 - u0) * 0.5, du)

            if abs(dv) > (v1 - v0) * 0.5:
                dv = math.copysign((v1 - v0) * 0.5, dv)

            u = min(max(u + du, u0), u1)
            v = min(max(v + dv, v0), v1)

            if max(abs(du), abs(dv)) < 1e-13:
                break

        dist = curve0.point_at(u).distance(curve1.point_at(v))

        return (u, v, dist)

    @staticmethod
    def line_point(line: Line, test_point: Point) -> tuple[Point, float, float]:
        """Return the closest point, parameter in [0, 1] and distance on a segment."""

        start = line.start()
        end = line.end()
        direction = end - start
        len_sq = direction.magnitude_squared()

        if len_sq < 1e-20:
            return (start, 0.0, start.distance(test_point))

        t = max(0.0, min(1.0, (test_point - start).dot(direction) / len_sq))
        closest = start + direction * t

        return (closest, t, closest.distance(test_point))

    @staticmethod
    def polyline_point(
        polyline: Polyline, test_point: Point
    ) -> tuple[Point, float, float]:
        """Return the closest point, length parameter in [0, 1] and distance on a polyline."""

        points = polyline.get_points()

        if not points:
            return (Point(0, 0, 0), 0.0, math.inf)

        if len(points) == 1:
            return (points[0], 0.0, points[0].distance(test_point))

        best_point = points[0]
        best_param = 0.0
        best_dist = math.inf
        cumulative_length = 0.0
        total_length = polyline.length()

        for i in range(len(points) - 1):
            segment = Line.from_points(points[i], points[i + 1])
            segment_length = segment.length()
            closest, t, dist = Closest.line_point(segment, test_point)

            if dist < best_dist:
                best_dist = dist
                best_point = closest

                if total_length > 1e-20:
                    best_param = (cumulative_length + t * segment_length) / total_length
                else:
                    best_param = float(i) / (len(points) - 1)

            cumulative_length += segment_length

        return (best_point, best_param, best_dist)

    # ═══════════════════════════════════════════════════════════════════════════
    # Surfaces
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def surface_point(
        surface: NurbsSurface,
        test_point: Point,
        u0: float = 0.0,
        u1: float = 0.0,
        v0: float = 0.0,
        v1: float = 0.0,
    ) -> tuple[float, float, float]:
        """Parameters and distance of the closest surface point within a uv window (0 means the domain end)."""

        if not surface.is_valid():
            return (0.0, 0.0, math.inf)

        domain_u0, domain_u1 = surface.domain(0)
        domain_v0, domain_v1 = surface.domain(1)

        if u0 <= 0.0:
            u0 = domain_u0

        if u1 <= 0.0:
            u1 = domain_u1

        if v0 <= 0.0:
            v0 = domain_v0

        if v1 <= 0.0:
            v1 = domain_v1

        u0 = max(u0, domain_u0)
        u1 = min(u1, domain_u1)
        v0 = max(v0, domain_v0)
        v1 = min(v1, domain_v1)

        seed = _surface_seed(surface, test_point, u0, u1, v0, v1)
        u, v = _surface_newton(surface, test_point, u0, u1, v0, v1, seed)

        return (u, v, surface.point_at(u, v).distance(test_point))

    @staticmethod
    def surface_curve(
        surface: NurbsSurface,
        curve: NurbsCurve,
        t0: float = 0.0,
        t1: float = 0.0,
        tolerance: float = 0.0,
    ) -> list[NurbsCurve]:
        """Seam-split uv pcurves of a curve lying on the surface, empty when it does not."""

        if not surface.is_valid() or not curve.is_valid():
            return []

        ct0 = curve.domain_start()
        ct1 = curve.domain_end()

        if t0 <= 0.0:
            t0 = ct0

        if t1 <= 0.0:
            t1 = ct1

        t0 = max(t0, ct0)
        t1 = min(t1, ct1)

        if t1 - t0 < 1e-14:
            return []

        pb = _pullback_setup(surface, tolerance)
        samples = _pullback_samples(surface, curve, pb, t0, t1)

        if not samples:
            return []

        _pullback_refine(surface, curve, pb, samples)

        result = []

        for piece_pts, piece_loop in _pullback_pieces(pb, curve, samples):
            pcurve = _pullback_fit(pb, piece_pts, piece_loop)

            if pcurve.is_valid():
                result.append(pcurve)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Meshes and clouds
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def mesh_point(mesh: Mesh, test_point: Point) -> tuple[Point, int, float]:
        """Return the closest point, face key and distance on a mesh via its triangle BVH."""

        best_point = Point(0, 0, 0)
        best_face_key = 0
        best_dist = math.inf

        if mesh.number_of_faces() == 0:
            return (best_point, best_face_key, best_dist)

        mesh.build_triangle_bvh()
        bvh = mesh.get_cached_bvh()

        if bvh is None or bvh.empty():
            return (best_point, best_face_key, best_dist)

        face_keys = _mesh_face_keys(mesh)
        stack = [0]

        while len(stack) > 0:
            node = bvh.nodes[stack.pop()]

            if _aabb_min_distance(node.aabb, test_point) >= best_dist:
                continue

            if node.is_leaf():
                found, face_idx, sub_idx, v0, v1, v2 = mesh.get_triangle_by_id(
                    node.object_id
                )

                if not found:
                    continue

                cp = _closest_point_on_triangle(test_point, v0, v1, v2)
                dist = cp.distance(test_point)

                if dist < best_dist:
                    best_dist = dist
                    best_point = cp
                    best_face_key = face_keys[face_idx]

                continue

            ld = _aabb_min_distance(bvh.nodes[node.left].aabb, test_point)
            rd = _aabb_min_distance(bvh.nodes[node.right].aabb, test_point)

            assert len(stack) + 2 <= STACK_SIZE

            if ld <= rd:
                if rd < best_dist:
                    stack.append(node.right)

                if ld < best_dist:
                    stack.append(node.left)
            else:
                if ld < best_dist:
                    stack.append(node.left)

                if rd < best_dist:
                    stack.append(node.right)

        return (best_point, best_face_key, best_dist)

    @staticmethod
    def mesh_point_aabb(mesh: Mesh, test_point: Point) -> tuple[Point, int, float]:
        """Return the closest point, face key and distance on a mesh via its triangle AABB tree."""

        best_point = Point(0, 0, 0)
        best_face_key = 0
        best_dist = math.inf

        if mesh.number_of_faces() == 0:
            return (best_point, best_face_key, best_dist)

        mesh.build_triangle_aabb_tree()
        tree = mesh.get_cached_aabb_tree()

        if tree is None or tree.empty():
            return (best_point, best_face_key, best_dist)

        face_keys = _mesh_face_keys(mesh)
        stack = [0]

        while len(stack) > 0:
            ni = stack.pop()
            node = tree.nodes[ni]

            if _aabb_min_distance(node.aabb, test_point) >= best_dist:
                continue

            if node.object_id >= 0:
                found, face_idx, sub_idx, v0, v1, v2 = mesh.get_triangle_by_id(
                    node.object_id
                )

                if not found:
                    continue

                cp = _closest_point_on_triangle(test_point, v0, v1, v2)
                dist = cp.distance(test_point)

                if dist < best_dist:
                    best_dist = dist
                    best_point = cp
                    best_face_key = face_keys[face_idx]

                continue

            left = ni + 1
            right = node.right
            ld = _aabb_min_distance(tree.nodes[left].aabb, test_point)
            rd = _aabb_min_distance(tree.nodes[right].aabb, test_point)

            assert len(stack) + 2 <= STACK_SIZE

            if ld <= rd:
                if rd < best_dist:
                    stack.append(right)

                if ld < best_dist:
                    stack.append(left)
            else:
                if ld < best_dist:
                    stack.append(left)

                if rd < best_dist:
                    stack.append(right)

        return (best_point, best_face_key, best_dist)

    @staticmethod
    def pointcloud_point(
        cloud: PointCloud, test_point: Point
    ) -> tuple[Point, int, float]:
        """Return the closest point, index and distance in a cloud by linear scan."""

        if cloud.point_count() == 0:
            return (Point(0, 0, 0), 0, math.inf)

        best_point = cloud.get_point(0)
        best_index = 0
        best_dist = best_point.distance(test_point)

        for i in range(1, cloud.point_count()):
            p = cloud.get_point(i)
            dist = p.distance(test_point)

            if dist < best_dist:
                best_dist = dist
                best_point = p
                best_index = i

        return (best_point, best_index, best_dist)

    @staticmethod
    def pointcloud_point_kdtree(
        cloud: PointCloud, test_point: Point
    ) -> tuple[Point, int, float]:
        """Return the closest point, index and distance in a cloud via a kd-tree."""

        if cloud.point_count() == 0:
            return (Point(0, 0, 0), 0, math.inf)

        pts = []

        for i in range(cloud.point_count()):
            pts.append(cloud.get_point(i))

        kd = SpatialKDTree(pts)
        idx, dist = kd.nearest(test_point)

        return (cloud.get_point(idx), idx, dist)

    # ═══════════════════════════════════════════════════════════════════════════
    # Collections
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def lines_closest(
        lines: list[Line], threshold: float = 0.0
    ) -> list[tuple[int, int]]:
        """Return the index pairs of lines whose endpoints come within threshold of each other."""

        pairs = []

        if threshold < 0.0 or len(lines) < 2:
            return pairs

        aabbs = []

        for ln in lines:
            aabbs.append(AABB.from_line(ln, threshold))

        tree = SpatialAABBTree()
        tree.build(aabbs)

        for i in range(len(lines)):
            for j in tree.query_aabb(aabbs[i]):
                if j <= i:
                    continue

                d_a = Closest.line_point(lines[j], lines[i].start())[2]
                d_b = Closest.line_point(lines[j], lines[i].end())[2]
                d_c = Closest.line_point(lines[i], lines[j].start())[2]
                d_d = Closest.line_point(lines[i], lines[j].end())[2]

                if min(d_a, d_b, d_c, d_d) <= threshold:
                    pairs.append((i, j))

        return pairs

    @staticmethod
    def polylines_closest(
        polylines: list[Polyline], threshold: float = 0.0
    ) -> list[tuple[int, int]]:
        """Index pairs of polylines whose vertices come within threshold of each other."""

        pairs = []

        if threshold < 0.0 or len(polylines) < 2:
            return pairs

        aabbs = []

        for pl in polylines:
            aabbs.append(AABB.from_polyline(pl, threshold))

        tree = SpatialAABBTree()
        tree.build(aabbs)

        for i in range(len(polylines)):
            for j in tree.query_aabb(aabbs[i]):
                if j <= i:
                    continue

                dist = math.inf

                for pt in polylines[i].get_points():
                    d = Closest.polyline_point(polylines[j], pt)[2]

                    if d < dist:
                        dist = d

                if dist <= threshold:
                    pairs.append((i, j))

        return pairs

    @staticmethod
    def nurbscurves_closest(
        curves: list[NurbsCurve], threshold: float = 0.0
    ) -> list[tuple[int, int]]:
        """Index pairs of curves whose endpoints come within threshold of each other."""

        pairs = []

        if threshold < 0.0 or len(curves) < 2:
            return pairs

        aabbs = []

        for crv in curves:
            aabbs.append(AABB.from_nurbscurve(crv, threshold, False))

        tree = SpatialAABBTree()
        tree.build(aabbs)

        for i in range(len(curves)):
            for j in tree.query_aabb(aabbs[i]):
                if j <= i:
                    continue

                p_start = curves[i].point_at(curves[i].domain_start())
                p_end = curves[i].point_at(curves[i].domain_end())
                d_a = Closest.curve_point(curves[j], p_start)[1]
                d_b = Closest.curve_point(curves[j], p_end)[1]

                if min(d_a, d_b) <= threshold:
                    pairs.append((i, j))

        return pairs

    @staticmethod
    def boxes_closest(
        boxes: list[AABB], threshold: float = 0.0
    ) -> list[tuple[int, int]]:
        """Return the index pairs of boxes within threshold of each other."""

        pairs = []

        if threshold < 0.0 or len(boxes) < 2:
            return pairs

        inflated = []

        for b in boxes:
            inf = AABB(b.cx, b.cy, b.cz, b.hx, b.hy, b.hz)
            inf.inflate(threshold)
            inflated.append(inf)

        tree = SpatialAABBTree()
        tree.build(inflated)

        for i in range(len(boxes)):
            for j in tree.query_aabb(inflated[i]):
                if j <= i:
                    continue

                if _aabb_to_aabb_min_dist(boxes[i], boxes[j]) <= threshold:
                    pairs.append((i, j))

        return pairs
