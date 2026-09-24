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


def _span_turn(srf: NurbsSurface, dir: int, t0: float, t1: float, smid: float) -> float:
    """Normal turn in degrees along dir over [t0, t1] on the line smid of the other direction, summed over four steps."""

    ma = 0.0
    pn = Vector(0.0, 0.0, 0.0)

    for k in range(5):
        t = t0 + k * (t1 - t0) / 4.0
        nm = srf.normal_at(t, smid) if dir == 0 else srf.normal_at(smid, t)

        if k > 0:
            d = max(-1.0, min(1.0, pn.dot(nm)))
            ma += math.acos(d) * 180.0 / PI

        pn = nm

    return ma


def _span_deviation(srf: NurbsSurface, dir: int, t0: float, t1: float, smid: float) -> float:
    """Largest distance of the quarter points along dir over [t0, t1] on the line smid from their chord."""

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

    return dev


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
            ma = _span_turn(srf, dir, t0, t1, smid)
            subs[i] = max(subs[i], max(1, min(math.ceil(ma / max_angle_deg), 64)))

        dev = _span_deviation(srf, dir, t0, t1, smid)

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


def _span_grid(
    srf: NurbsSurface, max_angle_deg: float, chord_tol: float
) -> tuple[list[float], list[float]] | None:
    """Span-adaptive grid parameters in u and v; none when the surface has no span in a direction."""

    usp = srf.get_span_vector(0)
    vsp = srf.get_span_vector(1)

    if len(usp) < 2 or len(vsp) < 2:
        return None

    us = _span_parameters(usp, _span_subdivisions(srf, 0, usp, vsp, srf.degree(0), max_angle_deg, chord_tol))
    vs = _span_parameters(vsp, _span_subdivisions(srf, 1, vsp, usp, srf.degree(1), max_angle_deg, chord_tol))

    if len(us) < 2 or len(vs) < 2:
        return None

    return us, vs


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
# Plane clipping
# ═══════════════════════════════════════════════════════════════════════════
def _clip_cell(
    welder: _VertexWelder,
    srf: NurbsSurface,
    q: list[float],
    n: list[float],
    cu: list[float],
    cv: list[float],
    fc: list[float],
) -> list[int]:
    """Welded polygon of the part of a grid cell where the field is <= 0: kept corners and Newton-refined edge crossings in order."""

    inn = [fc[0] <= 0, fc[1] <= 0, fc[2] <= 0, fc[3] <= 0]
    poly = []

    for k in range(4):
        kn = (k + 1) % 4

        if inn[k]:
            poly.append(welder.weld_surface(srf, cu[k], cv[k]))

        if inn[k] != inn[kn]:
            t = fc[k] / (fc[k] - fc[kn]) if abs(fc[k] - fc[kn]) > 1e-30 else 0.5
            u = cu[k] + (cu[kn] - cu[k]) * t
            v = cv[k] + (cv[kn] - cv[k]) * t
            u, v = _refine_crossing(srf, q, n, u, v)
            poly.append(welder.weld_surface(srf, u, v))

    return poly


def _add_fan(mesh: Mesh, poly: list[int]) -> None:
    """Fan a welded polygon into the mesh from its first vertex, skipping triangles with a repeated vertex."""

    for t in range(1, len(poly) - 1):
        a = poly[0]
        b = poly[t]
        c = poly[t + 1]

        if a == b or b == c or c == a:
            continue

        mesh.add_face([a, b, c])


def _grid_triangles(us: list[float], vs: list[float]) -> list[list[tuple[float, float]]]:
    """Two UV triangles per cell of the grid us x vs."""

    tris = []

    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            a = (us[i], vs[j])
            b = (us[i + 1], vs[j])
            c = (us[i + 1], vs[j + 1])
            d = (us[i], vs[j + 1])
            tris.append([a, b, c])
            tris.append([a, c, d])

    return tris


def _clip_triangles(
    srf: NurbsSurface, q: list[float], n: list[float], tris: list[list[tuple[float, float]]]
) -> list[list[tuple[float, float]]]:
    """UV triangles clipped to the half (S-q).n <= 1e-9, each kept part fanned from its first corner."""

    eps = 1e-9
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

    return nxt


def _weld_triangles(srf: NurbsSurface, tris: list[list[tuple[float, float]]], weld_tol: float) -> Mesh:
    """Mesh of UV triangles lifted onto the surface, seams welded within weld_tol, degenerate faces skipped."""

    result = Mesh()
    welder = _VertexWelder(result, weld_tol, weld_tol)

    for t in tris:
        a = welder.weld_surface(srf, t[0][0], t[0][1])
        b = welder.weld_surface(srf, t[1][0], t[1][1])
        c = welder.weld_surface(srf, t[2][0], t[2][1])

        if a == b or b == c or c == a:
            continue

        result.add_face([a, b, c])

    return result


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


class _SplitDomain:
    """UV domain of the split surface and the distance under which UV points snap together."""

    __slots__ = ("u0", "u1", "v0", "v1", "snap")

    def __init__(self, u0: float, u1: float, v0: float, v1: float, snap: float):
        """Construct from the domain bounds and the snap distance."""

        self.u0 = u0  # Start of the u domain.
        self.u1 = u1  # End of the u domain.
        self.v0 = v0  # Start of the v domain.
        self.v1 = v1  # End of the v domain.
        self.snap = snap  # Snap distance in UV.


def _split_domain(srf: NurbsSurface, tolerance: float) -> _SplitDomain:
    """Domain of the surface with the snap distance: tolerance carried from 3D into UV, else 1e-7 of the shorter side."""

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
        return _SplitDomain(u0, u1, v0, v1, max(1e-9, tolerance / uv_to_3d))

    return _SplitDomain(u0, u1, v0, v1, min(range_u, range_v) * 1e-7)


def _snap_to_border(p: list[float], dom: _SplitDomain) -> None:
    """Snap a UV point onto the domain border when within the snap distance of it."""

    if abs(p[0] - dom.u0) < dom.snap:
        p[0] = dom.u0

    if abs(p[0] - dom.u1) < dom.snap:
        p[0] = dom.u1

    if abs(p[1] - dom.v0) < dom.snap:
        p[1] = dom.v0

    if abs(p[1] - dom.v1) < dom.snap:
        p[1] = dom.v1


def _refine_samples(crv: NurbsCurve, entries: list[list[float]], samp_tol: float) -> int:
    """One pass inserting the parameter midpoint of every chord farther than samp_tol from the curve; the count inserted."""

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
        dev = 0.0

        if l2 > 1e-30:
            s = ((pm[0] - a[1]) * exu + (pm[1] - a[2]) * exv) / l2
            cx = a[1] + s * exu
            cy = a[2] + s * exv
            dev = math.hypot(pm[0] - cx, pm[1] - cy)

        if dev > samp_tol and len(entries) < 4096:
            entries.insert(i + 1, [tm, pm[0], pm[1]])
            inserted += 1
            i += 2
        else:
            i += 1

    return inserted


def _sample_pcurve(crv: NurbsCurve, samp_tol: float) -> list[list[float]]:
    """Samples (t, u, v) of a pcurve: uniform in t, then up to six passes of chord refinement."""

    ct0, ct1 = crv.domain()
    n = min(max(crv.cv_count() * 4, 16), 2048)
    entries = []

    for i in range(n + 1):
        t = ct0 + (ct1 - ct0) * i / n
        p = crv.point_at(t)
        entries.append([t, p[0], p[1]])

    for depth in range(6):
        if _refine_samples(crv, entries, samp_tol) == 0:
            break

    return entries


def _clamp_samples(entries: list[list[float]], cidx: int, dom: _SplitDomain) -> dict:
    """Polyline of pcurve cidx from its samples: clamped into the domain, snapped to the border, repeats dropped."""

    poly = {"cidx": cidx, "pts": [], "ts": []}

    for t, pu, pv in entries:
        p = [min(max(pu, dom.u0), dom.u1), min(max(pv, dom.v0), dom.v1)]
        _snap_to_border(p, dom)

        if poly["pts"] and abs(p[0] - poly["pts"][-1][0]) < 1e-15 and abs(p[1] - poly["pts"][-1][1]) < 1e-15:
            continue

        poly["pts"].append(p)
        poly["ts"].append(t)

    return poly


def _on_border(pts: list[list[float]], dom: _SplitDomain) -> bool:
    """True when every point lies within the snap distance of one domain side."""

    on_u0 = True
    on_u1 = True
    on_v0 = True
    on_v1 = True

    for p in pts:
        if abs(p[0] - dom.u0) >= dom.snap:
            on_u0 = False

        if abs(p[0] - dom.u1) >= dom.snap:
            on_u1 = False

        if abs(p[1] - dom.v0) >= dom.snap:
            on_v0 = False

        if abs(p[1] - dom.v1) >= dom.snap:
            on_v1 = False

    return on_u0 or on_u1 or on_v0 or on_v1


def _polyline_length(pts: list[list[float]]) -> float:
    """Length of a UV polyline."""

    ext = 0.0

    for k in range(1, len(pts)):
        ext += math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1])

    return ext


def _uv_polylines(pcurves: list[NurbsCurve], dom: _SplitDomain) -> list[dict]:
    """Polylines of the valid pcurves that neither hug the border nor fall short of min_ext, then the four domain sides."""

    range_u = dom.u1 - dom.u0
    range_v = dom.v1 - dom.v0
    samp_tol = max(range_u, range_v) * 2e-5
    min_ext = max(dom.snap * 8.0, min(range_u, range_v) * 1e-5)
    polylines = []

    for cidx, crv in enumerate(pcurves):
        if not crv.is_valid():
            continue

        poly = _clamp_samples(_sample_pcurve(crv, samp_tol), cidx, dom)

        if len(poly["pts"]) >= 2 and not _on_border(poly["pts"], dom) and _polyline_length(poly["pts"]) >= min_ext:
            polylines.append(poly)

    polylines.append({"cidx": -1, "pts": [[dom.u0, dom.v0], [dom.u1, dom.v0]], "ts": [dom.u0, dom.u1]})
    polylines.append({"cidx": -2, "pts": [[dom.u1, dom.v0], [dom.u1, dom.v1]], "ts": [dom.v0, dom.v1]})
    polylines.append({"cidx": -3, "pts": [[dom.u1, dom.v1], [dom.u0, dom.v1]], "ts": [dom.u1, dom.u0]})
    polylines.append({"cidx": -4, "pts": [[dom.u0, dom.v1], [dom.u0, dom.v0]], "ts": [dom.v1, dom.v0]})

    return polylines


def _uv_bounds(pts: list[list[float]]) -> list[float]:
    """UV bounds (umin, umax, vmin, vmax) of a polyline."""

    bounds = [pts[0][0], pts[0][0], pts[0][1], pts[0][1]]

    for p in pts:
        bounds[0] = min(bounds[0], p[0])
        bounds[1] = max(bounds[1], p[0])
        bounds[2] = min(bounds[2], p[1])
        bounds[3] = max(bounds[3], p[1])

    return bounds


def _boxes_overlap(A: dict, B: dict, snap: float) -> bool:
    """True when the bounds of B meet the bounds of A grown by snap."""

    a = _uv_bounds(A["pts"])
    b = _uv_bounds(B["pts"])

    return not (b[0] > a[1] + snap or b[1] < a[0] - snap or b[2] > a[3] + snap or b[3] < a[2] - snap)


def _border_parameter(cidx: int, hp: list[float]) -> float:
    """Parameter of a point along domain side cidx: u on the bottom and top sides, v on the left and right."""
    return hp[0] if cidx in (-1, -3) else hp[1]


def _crossing_point(
    acidx: int,
    ta: float,
    bcidx: int,
    tb: float,
    hit: list[float],
    pcurves: list[NurbsCurve],
    dom: _SplitDomain,
) -> tuple[list[float], float, float]:
    """UV point of a crossing moved onto its pcurves, Newton-refined when both are pcurves, snapped to the border; ta and tb follow it."""

    hp = [hit[0], hit[1]]

    if acidx >= 0 and bcidx >= 0:
        ta, tb = _newton_curve_curve(pcurves[acidx], ta, pcurves[bcidx], tb, dom.snap * 0.01)

    if acidx >= 0:
        pa = pcurves[acidx].point_at(ta)
        hp = [pa[0], pa[1]]
    elif bcidx >= 0:
        pb = pcurves[bcidx].point_at(tb)
        hp = [pb[0], pb[1]]

    _snap_to_border(hp, dom)

    if bcidx < 0:
        tb = _border_parameter(bcidx, hp)

    if acidx < 0:
        ta = _border_parameter(acidx, hp)

    return hp, ta, tb


def _add_crossings(
    polylines: list[dict],
    pi: int,
    pj: int,
    pcurves: list[NurbsCurve],
    dom: _SplitDomain,
    splits: dict,
) -> None:
    """Crossings of polylines pi and pj as events (fraction, u, v, parameter) on each crossed segment."""

    A = polylines[pi]
    B = polylines[pj]

    for ia in range(len(A["pts"]) - 1):
        for ib in range(len(B["pts"]) - 1):
            hit = _segment_intersection(A["pts"][ia], A["pts"][ia + 1], B["pts"][ib], B["pts"][ib + 1])

            if hit is None:
                continue

            s, t = hit
            ta = A["ts"][ia] + (A["ts"][ia + 1] - A["ts"][ia]) * s
            tb = B["ts"][ib] + (B["ts"][ib + 1] - B["ts"][ib]) * t
            hu = A["pts"][ia][0] + (A["pts"][ia + 1][0] - A["pts"][ia][0]) * s
            hv = A["pts"][ia][1] + (A["pts"][ia + 1][1] - A["pts"][ia][1]) * s
            hp, ta, tb = _crossing_point(A["cidx"], ta, B["cidx"], tb, [hu, hv], pcurves, dom)
            splits.setdefault((pi, ia), []).append((s, hp[0], hp[1], ta))
            splits.setdefault((pj, ib), []).append((t, hp[0], hp[1], tb))


def _polyline_crossings(polylines: list[dict], pcurves: list[NurbsCurve], dom: _SplitDomain) -> dict:
    """Crossing events of every pair of overlapping polylines with at least one pcurve, keyed by (polyline, segment)."""

    splits = {}

    for pi in range(len(polylines)):
        for pj in range(pi + 1, len(polylines)):
            if (polylines[pi]["cidx"] >= 0 or polylines[pj]["cidx"] >= 0) and _boxes_overlap(
                polylines[pi], polylines[pj], dom.snap
            ):
                _add_crossings(polylines, pi, pj, pcurves, dom, splits)

    return splits


def _split_edges(polylines: list[dict], splits: dict, pool: _UVVertexPool) -> list[dict]:
    """Graph edges along every polyline between consecutive pool vertices, its crossings inserted in order."""

    edges = []

    for pi, poly in enumerate(polylines):
        chain = []

        for i in range(len(poly["pts"])):
            chain.append((pool.id(poly["pts"][i]), poly["ts"][i]))

            if i < len(poly["pts"]) - 1 and (pi, i) in splits:
                for frac, hu, hv, tc in sorted(splits[(pi, i)]):
                    chain.append((pool.id([hu, hv]), tc))

        for i in range(len(chain) - 1):
            a, ta = chain[i]
            b, tb = chain[i + 1]

            if a == b:
                continue

            edges.append({"a": a, "b": b, "cidx": poly["cidx"], "ta": ta, "tb": tb})

    return edges


def _prune_dangling(edges: list[dict]) -> list[dict]:
    """Edges left after repeatedly dropping every edge with an end of degree one."""

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

    return live_edges


def _half_edges(edges: list[dict]) -> list[list[int]]:
    """Two opposite half-edges [tail, head, edge, forward] per edge, the forward one at the even index."""

    hes = []

    for ei, e in enumerate(edges):
        hes.append([e["a"], e["b"], ei, 1])
        hes.append([e["b"], e["a"], ei, 0])

    return hes


def _next_half_edges(hes: list[list[int]], verts: list[list[float]]) -> list[int]:
    """Successor of every half-edge around its face: the twin of an outgoing half-edge continues with its predecessor in the angle-sorted fan."""

    out_map = []

    for vid in range(len(verts)):
        out_map.append([])

    for hi in range(len(hes)):
        out_map[hes[hi][0]].append(hi)

    for vid in range(len(out_map)):
        fan = []

        for hi in out_map[vid]:
            angle = math.atan2(verts[hes[hi][1]][1] - verts[vid][1], verts[hes[hi][1]][0] - verts[vid][0])
            fan.append((angle, hi))

        fan.sort()

        for k in range(len(fan)):
            out_map[vid][k] = fan[k][1]

    next_he = [-1] * len(hes)

    for outs in out_map:
        for pos in range(len(outs)):
            next_he[outs[pos] ^ 1] = outs[(pos + len(outs) - 1) % len(outs)]

    return next_he


def _face_cycles(next_he: list[int]) -> list[list[int]]:
    """Cycles of at least two half-edges traced through next_he, each half-edge in one cycle."""

    visited = [False] * len(next_he)
    faces = []

    for hi in range(len(next_he)):
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

    return faces


def _cycle_area(cycle: list[int], hes: list[list[int]], verts: list[list[float]]) -> float:
    """Signed area of a half-edge cycle."""

    s = 0.0

    for hi in cycle:
        a = verts[hes[hi][0]]
        b = verts[hes[hi][1]]
        s += a[0] * b[1] - b[0] * a[1]

    return s * 0.5


def _touches_border(cycle: list[int], hes: list[list[int]], border_vids: set[int]) -> bool:
    """True when a cycle passes through a vertex of the domain border."""

    for hi in cycle:
        if hes[hi][0] in border_vids:
            return True

    return False


def _classify_faces(
    faces: list[list[int]],
    hes: list[list[int]],
    verts: list[list[float]],
    edges: list[dict],
    snap: float,
) -> tuple[list[tuple[list[int], float]], list[list[int]]]:
    """Face cycles by orientation: counter-clockwise faces with their area, clockwise holes clear of the domain border."""

    border_vids = set()

    for e in edges:
        if e["cidx"] < 0:
            border_vids.add(e["a"])
            border_vids.add(e["b"])

    pos_faces = []
    neg_faces = []

    for cycle in faces:
        area = _cycle_area(cycle, hes, verts)

        if area > snap * snap:
            pos_faces.append((cycle, area))
        elif area < -snap * snap and not _touches_border(cycle, hes, border_vids):
            neg_faces.append(cycle)

    return pos_faces, neg_faces


def _point_in_cycle(p: list[float], cycle: list[int], hes: list[list[int]], verts: list[list[float]]) -> bool:
    """Even-odd test of p against a half-edge cycle."""

    inside = False

    for hi in cycle:
        a = verts[hes[hi][0]]
        b = verts[hes[hi][1]]

        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside

    return inside


def _same_vertices(a: list[int], b: list[int], hes: list[list[int]]) -> bool:
    """True when two cycles pass through the same set of vertices."""

    a_vids = set()
    b_vids = set()

    for hi in a:
        a_vids.add(hes[hi][0])

    for hi in b:
        b_vids.add(hes[hi][0])

    return a_vids == b_vids


def _assign_holes(
    neg_faces: list[list[int]],
    pos_faces: list[tuple[list[int], float]],
    hes: list[list[int]],
    verts: list[list[float]],
) -> list[list[list[int]]]:
    """Holes per positive face: each hole goes to the smallest face that contains it and is not its own vertex ring."""

    holes_of = []

    for fi in range(len(pos_faces)):
        holes_of.append([])

    for cycle in neg_faces:
        sample = verts[hes[cycle[0]][0]]
        best = -1
        best_area = float("inf")

        for fi, (fc, area) in enumerate(pos_faces):
            if area < best_area and _point_in_cycle(sample, fc, hes, verts) and not _same_vertices(cycle, fc, hes):
                best = fi
                best_area = area

        if best >= 0:
            holes_of[best].append(cycle)

    return holes_of


def _cycle_runs(cycle: list[int], hes: list[list[int]], edges: list[dict]) -> list[dict]:
    """Runs of a cycle: consecutive half-edges on one pcurve merged."""

    runs = []

    for hi in cycle:
        tail, head, eidx, fwd = hes[hi]
        e = edges[eidx]
        ta = e["ta"] if fwd else e["tb"]
        tb = e["tb"] if fwd else e["ta"]

        if runs and runs[-1]["cidx"] == e["cidx"] and runs[-1]["vb"] == tail:
            runs[-1]["vb"] = head
            runs[-1]["tb"] = tb
        else:
            runs.append({"cidx": e["cidx"], "va": tail, "vb": head, "ta": ta, "tb": tb})

    return runs


def _run_piece(run: dict, pcurves: list[NurbsCurve]) -> NurbsCurve | None:
    """Pcurve piece of a run, trimmed to its parameters and oriented along it; none when the run cannot be cut."""

    if run["cidx"] < 0:
        return None

    crv = pcurves[run["cidx"]]
    c0, c1 = crv.domain()
    lo = max(c0, min(run["ta"], run["tb"]))
    hi_ = min(c1, max(run["ta"], run["tb"]))
    piece = crv.duplicate()

    if hi_ - lo < (c1 - c0) - 1e-12 and hi_ - lo > 1e-14:
        if not piece.trim(lo, hi_):
            return None
    elif hi_ - lo <= 1e-14 and not (run["va"] == run["vb"] and piece.is_closed()):
        return None

    if not piece.is_valid():
        return None

    if run["ta"] > run["tb"] and not piece.reverse():
        return None

    return piece


def _cycle_to_segments(
    cycle: list[int],
    hes: list[list[int]],
    edges: list[dict],
    verts: list[list[float]],
    pcurves: list[NurbsCurve],
) -> list[NurbsCurve]:
    """Pieces of a cycle: trimmed pcurve runs, straight UV segments where a run cannot be cut."""

    pieces = []

    for run in _cycle_runs(cycle, hes, edges):
        piece = _run_piece(run, pcurves)

        if piece is not None:
            pieces.append(piece)
            continue

        pa = verts[run["va"]]
        pb = verts[run["vb"]]

        if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 1e-14:
            pieces.append(NurbsCurve.create(False, 1, [Point(pa[0], pa[1], 0.0), Point(pb[0], pb[1], 0.0)]))

    return pieces


def _close_curve(curve: NurbsCurve, tol: float) -> bool:
    """Close a curve whose ends lie within tol by moving its last control point onto the first; true when it ends closed."""

    if curve.is_closed():
        return True

    if curve.point_at_start().distance(curve.point_at_end()) > tol:
        return False

    last = curve.cv_count() - 1
    x, y, z, w = curve.get_cv_4d(0)
    xe, ye, ze, we = curve.get_cv_4d(last)

    return curve.set_cv_4d(last, x, y, z, we) and curve.is_closed()


def _cycle_to_loop(
    cycle: list[int],
    hes: list[list[int]],
    edges: list[dict],
    verts: list[list[float]],
    pcurves: list[NurbsCurve],
    snap_uv: float,
) -> NurbsCurve:
    """Closed loop of a cycle: the joined pieces when they close, else the polygon through its vertices."""

    pieces = _cycle_to_segments(cycle, hes, edges, verts, pcurves)

    if not pieces:
        return NurbsCurve()

    join_tol = snap_uv * 4.0
    joined = NurbsCurve.join(pieces, join_tol)

    if len(joined) == 1 and joined[0].is_valid() and _close_curve(joined[0], join_tol):
        return joined[0]

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
        self._visit_epoch = 0  # Stamp of the current search.
        self._visit_stamp = []  # Last search stamp per triangle.
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
        existing = self._find_coincident(start, x, y)

        if existing >= 0:
            return existing

        vi = len(self.vertices)
        self.vertices.append(_Vertex2D(x, y))
        bad = self._collect_cavity(start, x, y)

        if not bad:
            self.vertices.pop()

            return -1

        self._fill_cavity(vi, bad, self._cavity_polygon(bad))
        self.last_found = len(self.triangles) - 1

        return vi

    def _vertex_index(self, ti, v):
        """Corner of triangle ti holding vertex v, -1 when none does."""

        for k in range(3):
            if self.triangles[ti].v[k] == v:
                return k

        return -1

    def _opposite_vertex(self, ti, nb):
        """Vertex of triangle ti across its edge shared with triangle nb, -1 when they are not neighbours."""

        for k in range(3):
            if self.triangles[ti].adj[k] == nb:
                return self.triangles[ti].v[k]

        return -1

    def _find_coincident(self, start, x, y):
        """Vertex of triangle start within 1e-6 of (x, y), -1 when none."""

        if start < 0 or not self.triangles[start].alive:
            return -1

        for vi in self.triangles[start].v:
            ddx = self.vertices[vi].x - x
            ddy = self.vertices[vi].y - y

            if ddx * ddx + ddy * ddy < 1e-12:
                return vi

        return -1

    def _circumcircle_contains(self, ti, x, y):
        """True when (x, y) lies inside the circumcircle of triangle ti."""

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

        return ic > 0

    def _collect_cavity(self, start, x, y):
        """Triangles whose circumcircle holds (x, y), grown from start across unconstrained edges."""

        self._visit_epoch += 1

        if len(self._visit_stamp) < len(self.triangles) + 64:
            self._visit_stamp.extend([0] * (len(self.triangles) + 64 - len(self._visit_stamp)))

        bad = []

        if start >= 0:
            bad.append(start)
            self._visit_stamp[start] = self._visit_epoch

        front = 0

        while front < len(bad):
            ti = bad[front]
            front += 1

            if not self.triangles[ti].alive or not self._circumcircle_contains(ti, x, y):
                bad[front - 1] = -1
                continue

            for k in range(3):
                nb = self.triangles[ti].adj[k]

                if self.triangles[ti].constrained[k] or nb < 0 or self._visit_stamp[nb] == self._visit_epoch:
                    continue

                self._visit_stamp[nb] = self._visit_epoch
                bad.append(nb)

        return [ti for ti in bad if ti >= 0]

    def _cavity_polygon(self, bad):
        """Edges of the bad triangles that face a good neighbour or the hull."""

        bad_set = set(bad)
        polygon = []

        for ti in bad:
            t = self.triangles[ti]

            for k in range(3):
                nb = t.adj[k]

                if nb < 0 or nb not in bad_set:
                    polygon.append((t.v[(k + 1) % 3], t.v[(k + 2) % 3], t.constrained[k]))

        return polygon

    def _fill_cavity(self, vi, bad, polygon):
        """Replace the bad triangles by a fan from vertex vi to the polygon edges."""

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

    def insert_constraint(self, v0: int, v1: int) -> None:
        """Force the edge v0-v1 into the triangulation by flipping the edges it crosses."""

        if v0 == v1 or self._constrain_existing(v0, v1):
            return

        start_ti = self._first_triangle_at(v0)

        if start_ti < 0:
            return

        it, ivl, ivr = self._first_crossed(start_ti, v0, v1)

        if it < 0:
            return

        poly_l = [v0, ivl]
        poly_r = [v0, ivr]
        intersected = [it]
        self._walk_crossed(v0, v1, ivl, ivr, poly_l, poly_r, intersected)
        poly_l.append(v1)
        poly_r.append(v1)
        self._retriangulate(v0, v1, poly_l, poly_r, intersected)

    def _constrain_existing(self, v0, v1):
        """Mark v0-v1 constrained when it already is a triangle edge; false when it is not."""

        for ti in range(len(self.triangles)):
            if not self.triangles[ti].alive:
                continue

            for k in range(3):
                e0 = self.triangles[ti].v[(k + 1) % 3]
                e1 = self.triangles[ti].v[(k + 2) % 3]

                if not ((e0 == v0 and e1 == v1) or (e0 == v1 and e1 == v0)):
                    continue

                self.triangles[ti].constrained[k] = True
                nb = self.triangles[ti].adj[k]

                if nb >= 0 and self.triangles[nb].alive:
                    for kk in range(3):
                        if self.triangles[nb].adj[kk] == ti:
                            self.triangles[nb].constrained[kk] = True
                            break

                return True

        return False

    def _first_triangle_at(self, v):
        """Lowest live triangle with vertex v, -1 when none."""

        for ti in range(len(self.triangles)):
            if self.triangles[ti].alive and self._has_vertex(ti, v):
                return ti

        return -1

    def _first_crossed(self, start_ti, v0, v1):
        """Triangle around v0 whose opposite edge the segment v0-v1 crosses, with that edge's left and right ends; -1 when none."""

        ax = self.vertices[v0].x
        ay = self.vertices[v0].y
        bx = self.vertices[v1].x
        by = self.vertices[v1].y
        ti = start_ti

        for _ in range(len(self.triangles) + 4):
            if not self.triangles[ti].alive:
                break

            k = self._vertex_index(ti, v0)

            if k < 0:
                break

            ip2 = self.triangles[ti].v[(k + 1) % 3]
            ip1 = self.triangles[ti].v[(k + 2) % 3]
            op2 = self._orient2d(ax, ay, bx, by, self.vertices[ip2].x, self.vertices[ip2].y)
            op1 = self._orient2d(ax, ay, bx, by, self.vertices[ip1].x, self.vertices[ip1].y)

            if op2 < 0 and op1 >= 0:
                return ti, ip1, ip2

            nxt = self.triangles[ti].adj[(k + 1) % 3]

            if nxt < 0 or not self.triangles[nxt].alive or nxt == start_ti:
                break

            ti = nxt

        return -1, -1, -1

    def _walk_crossed(self, v0, v1, ivl, ivr, poly_l, poly_r, intersected):
        """Walk the triangles crossed by v0-v1 from intersected[0], collecting the vertices left and right of it."""

        ax = self.vertices[v0].x
        ay = self.vertices[v0].y
        bx = self.vertices[v1].x
        by = self.vertices[v1].y
        iv = v0
        cur_it = intersected[0]

        for _ in range(len(self.triangles) * 2 + 8):
            if self._has_vertex(cur_it, v1):
                break

            k_iv = self._vertex_index(cur_it, iv)

            if k_iv < 0:
                break

            i_topo = self.triangles[cur_it].adj[k_iv]

            if i_topo < 0 or not self.triangles[i_topo].alive:
                break

            i_vopo = self._opposite_vertex(i_topo, cur_it)

            if i_vopo < 0:
                break

            o = self._orient2d(ax, ay, bx, by, self.vertices[i_vopo].x, self.vertices[i_vopo].y)

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

    def _retriangulate(self, v0, v1, poly_l, poly_r, intersected):
        """Replace the crossed triangles by the two fans on either side of v0-v1 and constrain it."""

        for ti in intersected:
            self._unregister_edges(ti)
            self.triangles[ti].alive = False

        first_new = len(self.triangles)

        for i in range(len(poly_l) - 2):
            self._add_triangle(v1, poly_l[i + 1], poly_l[i])

        for i in range(1, len(poly_r) - 1):
            self._add_triangle(v0, poly_r[i], poly_r[i + 1])

        self._inherit_constraints(first_new)
        self._mark_edge(v0, v1)

    def _inherit_constraints(self, first_new):
        """Constrain every edge of a triangle from first_new on that its older neighbour holds constrained."""

        for new_ti in range(first_new, len(self.triangles)):
            if not self.triangles[new_ti].alive:
                continue

            for k in range(3):
                nb = self.triangles[new_ti].adj[k]

                if nb < 0 or nb >= first_new or not self.triangles[nb].alive:
                    continue

                for kk in range(3):
                    if self.triangles[nb].adj[kk] == new_ti and self.triangles[nb].constrained[kk]:
                        self.triangles[new_ti].constrained[k] = True
                        break

    def _mark_edge(self, v0, v1):
        """Mark the edge v0-v1 constrained in every live triangle that has it."""

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
# Triangulation
# ═══════════════════════════════════════════════════════════════════════════
def _loop_points(crv: NurbsCurve) -> list[Point]:
    """Loop polygon in UV before refinement: the control points of a polyline, else samples, the closing repeat dropped."""

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

    return raw


def _subdivide_edge(srf: NurbsSurface, start: Point, end: Point, deflection: float, out: list[Point]) -> None:
    """Append the UV points of the edge start-end without end, halved up to six times until each 3D chord is within deflection."""

    stack = [(start, end, 0)]

    while stack:
        a, b, depth = stack.pop()
        mu = (a[0] + b[0]) * 0.5
        mv = (a[1] + b[1]) * 0.5
        pa = srf.point_at(a[0], a[1])
        pm = srf.point_at(mu, mv)
        edge = srf.point_at(b[0], b[1]) - pa
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


def _find_crease_knots(surface: NurbsSurface) -> list[list[float]]:
    """Interior knots per direction whose multiplicity reaches the degree: the C0 lines of the surface."""

    crease_knots = [[], []]

    for direction in range(2):
        domain = surface.domain(direction)
        knots = surface.m_nurbsknot[direction]

        for knot in knots:
            if knot <= domain[0] or knot >= domain[1] or knot in crease_knots[direction]:
                continue

            multiplicity = 0

            for value in knots:
                if value == knot:
                    multiplicity += 1

            if multiplicity >= surface.degree(direction):
                crease_knots[direction].append(knot)

    return crease_knots


def _loop_bounds(pts: list[Point]) -> list[float]:
    """UV bounds (umin, vmin, umax, vmax) of a loop polygon."""

    bounds = [1e30, 1e30, -1e30, -1e30]

    for p in pts:
        if p[0] < bounds[0]:
            bounds[0] = p[0]

        if p[1] < bounds[1]:
            bounds[1] = p[1]

        if p[0] > bounds[2]:
            bounds[2] = p[0]

        if p[1] > bounds[3]:
            bounds[3] = p[1]

    return bounds


def _insert_loop_edge(
    dt: _Delaunay2D,
    pts: list[Point],
    vis: list[int],
    li: int,
    i: int,
    crease_knots: list[list[float]],
    boundary_intervals: dict,
) -> None:
    """Constrain loop edge i in pieces cut where it crosses a crease knot line, each crossing inserted and recorded."""

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

            uv = [pts[i][0] + t * (pts[j][0] - pts[i][0]), pts[i][1] + t * (pts[j][1] - pts[i][1])]
            uv[direction] = knot
            vi = dt.insert(uv[0], uv[1])

            if vi >= 0:
                boundary_intervals[vi] = (li, i, t)

            events.append((t, vi))

    events.sort()

    for k in range(1, len(events)):
        if events[k - 1][1] >= 0 and events[k][1] >= 0 and events[k - 1][1] != events[k][1]:
            dt.insert_constraint(events[k - 1][1], events[k][1])


def _insert_loops(
    dt: _Delaunay2D, loops_uv: list[list[Point]], crease_knots: list[list[float]], boundary_intervals: dict
) -> list[list[int]]:
    """Insert each loop's vertices, then constrain its edges; the vertex index of every loop sample."""

    loop_vids = []

    for li, pts in enumerate(loops_uv):
        vis = []

        for p in pts:
            vis.append(dt.insert(p[0], p[1]))

        for i in range(len(vis)):
            _insert_loop_edge(dt, pts, vis, li, i, crease_knots, boundary_intervals)

        loop_vids.append(vis)

    return loop_vids


def _insert_crease_lines(dt: _Delaunay2D, loops_uv: list[list[Point]], crease_knots: list[list[float]]) -> None:
    """Insert the crease knot crossings inside the loops and constrain each knot line between consecutive vertices on it."""

    for u in crease_knots[0]:
        for v in crease_knots[1]:
            if _inside_loops(u, v, loops_uv):
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

                if _inside_loops(uv[0], uv[1], loops_uv):
                    dt.insert_constraint(nodes[k - 1][1], nodes[k][1])


def _min_normal_dot(
    surface: NurbsSurface,
    crease_knots: list[list[float]],
    center: tuple[float, float],
    a: _Vertex2D,
    b: _Vertex2D,
    c: _Vertex2D,
) -> float:
    """Smallest dot product between the crease-side normals at the corners of triangle abc around its centroid."""

    na = _crease_side_normal(surface, crease_knots, center, [a.x, a.y])
    nb = _crease_side_normal(surface, crease_knots, center, [b.x, b.y])
    nc2 = _crease_side_normal(surface, crease_knots, center, [c.x, c.y])
    d1 = na.dot(nb)
    d2 = nb.dot(nc2)
    d3 = na.dot(nc2)

    return min(d1, min(d2, d3))


def _refinement_points(
    dt: _Delaunay2D,
    surface: NurbsSurface,
    loops_uv: list[list[Point]],
    crease_knots: list[list[float]],
    deflection: float,
    cos_max_angle: float,
) -> list[tuple[float, float]]:
    """Centroids of the live triangles inside the loops whose chord leaves deflection or whose corner normals turn past the angle bound."""

    to_insert = []

    for tri in dt.triangles:
        if not tri.alive:
            continue

        a = dt.vertices[tri.v[0]]
        b = dt.vertices[tri.v[1]]
        c = dt.vertices[tri.v[2]]
        cu = (a.x + b.x + c.x) / 3.0
        cv = (a.y + b.y + c.y) / 3.0

        if not _inside_loops(cu, cv, loops_uv):
            continue

        pa = surface.point_at(a.x, a.y)
        pb = surface.point_at(b.x, b.y)
        pc = surface.point_at(c.x, c.y)
        pm = surface.point_at(cu, cv)
        n = (pb - pa).cross(pc - pa)
        nl = math.sqrt(n.magnitude_squared())

        if nl < 1e-30:
            continue

        dev = abs((pm - pa).dot(n) / nl)

        if dev > deflection or _min_normal_dot(surface, crease_knots, (cu, cv), a, b, c) < cos_max_angle:
            to_insert.append((cu, cv))

    return to_insert


def _refine(
    dt: _Delaunay2D,
    surface: NurbsSurface,
    loops_uv: list[list[Point]],
    crease_knots: list[list[float]],
    deflection: float,
    cos_max_angle: float,
) -> None:
    """Insert refinement centroids for up to eight rounds, until none is needed or the vertex cap is hit."""

    MAX_ITERS = 8
    MAX_VERTS = 200000

    for _iter in range(MAX_ITERS):
        to_insert = _refinement_points(dt, surface, loops_uv, crease_knots, deflection, cos_max_angle)

        if not to_insert:
            break

        for cu, cv in to_insert:
            if len(dt.vertices) >= MAX_VERTS:
                break

            dt.insert(cu, cv)

        if len(dt.vertices) >= MAX_VERTS:
            break


def _trim_outside(dt: _Delaunay2D, loops_uv: list[list[Point]]) -> None:
    """Drop the super triangle and every triangle whose centroid lies outside the loops."""

    dt.cleanup()

    for tri in dt.triangles:
        if not tri.alive:
            continue

        v0, v1, v2 = tri.v
        cu = (dt.vertices[v0].x + dt.vertices[v1].x + dt.vertices[v2].x) / 3.0
        cv = (dt.vertices[v0].y + dt.vertices[v1].y + dt.vertices[v2].y) / 3.0

        if not _inside_loops(cu, cv, loops_uv):
            tri.alive = False


def _crosses_crease(tris: list[tuple[int, int, int]], dt: _Delaunay2D, crease_knots: list[list[float]]) -> bool:
    """True when a triangle spans a crease knot line in either direction."""

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
                    return True

    return False


def _given_points(count: int, loops: TrimLoops, loop_vids: list[list[int]]) -> list[tuple[int, int] | None]:
    """Loop and sample of the 3D point given for each triangulation vertex, none where none is."""

    given = [None] * count

    for li, vids in enumerate(loop_vids):
        if li >= len(loops.xyz):
            break

        for k, vi in enumerate(vids):
            if vi >= 0 and k < len(loops.xyz[li]):
                given[vi] = (li, k)

    return given


def _vertex_point(
    surface: NurbsSurface,
    dt: _Delaunay2D,
    vi: int,
    loops: TrimLoops,
    given: list[tuple[int, int] | None],
    boundary_intervals: dict,
) -> Point:
    """3D point of triangulation vertex vi: its given loop point, the loop chord at a knot crossing, else the surface point."""

    if given[vi] is not None:
        return loops.xyz[given[vi][0]][given[vi][1]]

    if vi in boundary_intervals and loops.xyz:
        li, segment, t = boundary_intervals[vi]
        a = loops.xyz[li][segment]
        b = loops.xyz[li][(segment + 1) % len(loops.xyz[li])]

        return a + (b - a) * t

    return surface.point_at(dt.vertices[vi].x, dt.vertices[vi].y)


def _weld_vertices(
    welder: _VertexWelder,
    surface: NurbsSurface,
    dt: _Delaunay2D,
    tris: list[tuple[int, int, int]],
    loops: TrimLoops,
    loop_vids: list[list[int]],
    boundary_intervals: dict,
) -> list[int | None]:
    """Welded mesh vertex of every triangulation vertex a triangle uses, none for the others."""

    given = _given_points(len(dt.vertices), loops, loop_vids)
    vert_map = [None] * len(dt.vertices)

    for tri in tris:
        for vi in tri:
            if vert_map[vi] is None:
                vert_map[vi] = welder.weld(_vertex_point(surface, dt, vi, loops, given, boundary_intervals))

    return vert_map


def _add_faces(mesh: Mesh, tris: list[tuple[int, int, int]], vert_map: list[int | None]) -> None:
    """One face per triangle over its welded vertices, collapsed ones skipped."""

    for a, b, c in tris:
        v0 = vert_map[a]
        v1 = vert_map[b]
        v2 = vert_map[c]

        if v0 == v1 or v1 == v2 or v2 == v0:
            continue

        mesh.add_face([v0, v1, v2])


def _fan_normals(mesh: Mesh) -> dict:
    """Area-weighted sum of the face normals around each mesh vertex."""

    fan = {}

    for fk in sorted(mesh.face):
        verts = mesh.face[fk]
        a = mesh.vertex[verts[0]].position()
        b = mesh.vertex[verts[1]].position()
        c = mesh.vertex[verts[2]].position()
        n = (b - a).cross(c - a)

        for vk in verts:
            if vk not in fan:
                fan[vk] = Vector(0.0, 0.0, 0.0)

            fan[vk] += n

    return fan


def _set_vertex_normals(mesh: Mesh, surface: NurbsSurface, dt: _Delaunay2D, vert_map: list[int | None]) -> None:
    """Normal of every used vertex from the surface derivatives, the fan normal where they degenerate, and its u and v."""

    fan = _fan_normals(mesh)

    for vi in range(len(vert_map)):
        if vert_map[vi] is None:
            continue

        u = dt.vertices[vi].x
        v = dt.vertices[vi].y
        derivatives = surface.evaluate(u, v, 1)
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

        vd = mesh.vertex[vert_map[vi]]
        vd.set_normal(nrm[0], nrm[1], nrm[2])
        vd.attributes["u"] = u
        vd.attributes["v"] = v


def _tag_boundary(mesh: Mesh, loop_vids: list[list[int]], boundary_intervals: dict, vert_map: list[int | None]) -> None:
    """Tag loop vertices boundary/{loop}/{sample} and knot crossings boundary_interval/{loop}/{segment} with their chord parameter."""

    for li, vids in enumerate(loop_vids):
        for k, vi in enumerate(vids):
            key = f"boundary/{li}/{k}"

            if vi >= 0 and vert_map[vi] is not None:
                mesh.vertex[vert_map[vi]].attributes[key] = 1.0

    for vi in sorted(boundary_intervals):
        li, segment, t = boundary_intervals[vi]
        key = f"boundary_interval/{li}/{segment}"

        if vert_map[vi] is not None:
            mesh.vertex[vert_map[vi]].attributes[key] = t


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

        dom = _split_domain(srf, tolerance)
        polylines = _uv_polylines(pcurves, dom)
        pool = _UVVertexPool(dom.snap)
        splits = _polyline_crossings(polylines, pcurves, dom)
        live_edges = _prune_dangling(_split_edges(polylines, splits, pool))

        if not live_edges:
            return []

        verts = pool.verts
        hes = _half_edges(live_edges)
        faces = _face_cycles(_next_half_edges(hes, verts))
        pos_faces, neg_faces = _classify_faces(faces, hes, verts, live_edges, dom.snap)
        holes_of = _assign_holes(neg_faces, pos_faces, hes, verts)
        result = []

        for fi, (cycle, area) in enumerate(pos_faces):
            outer = _cycle_to_loop(cycle, hes, live_edges, verts, pcurves, dom.snap)

            if not outer.is_valid() or (_loop_signed_area(outer) < 0.0 and not outer.reverse()):
                continue

            ts = NurbsSurfaceTrimmed.create(srf, outer)

            for hole_cycle in holes_of[fi]:
                hole = _cycle_to_loop(hole_cycle, hes, live_edges, verts, pcurves, dom.snap)

                if not hole.is_valid() or (_loop_signed_area(hole) > 0.0 and not hole.reverse()):
                    continue

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
        """Compare name, width, color, surface and trim loops; guid ignored."""

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

        if self.m_outer_loop != other.m_outer_loop:
            return False

        return self.m_inner_loops == other.m_inner_loops

    def __ne__(self, other) -> bool:
        """Compare name, width, color, surface and trim loops; guid ignored."""
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
        bbox_diag = self._bbox_diagonal()
        grid = _span_grid(srf, max_angle_deg, bbox_diag * chord_factor)

        if grid is None:
            return srf.mesh()

        us, vs = grid
        field = []

        for i in range(len(us)):
            row = []

            for j in range(len(vs)):
                row.append(_plane_field(srf, q, n, us[i], vs[j]))

            field.append(row)

        result = Mesh()
        weld_tol = bbox_diag * 1e-5
        welder = _VertexWelder(result, weld_tol, weld_tol)

        for i in range(len(us) - 1):
            for j in range(len(vs) - 1):
                cu = [us[i], us[i + 1], us[i + 1], us[i]]
                cv = [vs[j], vs[j], vs[j + 1], vs[j + 1]]
                fc = [field[i][j], field[i + 1][j], field[i + 1][j + 1], field[i][j + 1]]
                _add_fan(result, _clip_cell(welder, srf, q, n, cu, cv, fc))

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

        bbox_diag = self._bbox_diagonal()
        grid = _span_grid(srf, max_angle_deg, bbox_diag * chord_factor)

        if grid is None:
            return srf.mesh()

        tris = _grid_triangles(grid[0], grid[1])

        for q, n in pl:
            tris = _clip_triangles(srf, q, n, tris)

            if not tris:
                break

        if not tris:
            return Mesh()

        result = _weld_triangles(srf, tris, bbox_diag * 1e-5)

        return result if result.face else Mesh()

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

        raw = _loop_points(crv)

        if len(raw) < 2:
            return raw

        out = []

        for i in range(len(raw)):
            _subdivide_edge(self.m_surface, raw[i], raw[(i + 1) % len(raw)], deflection, out)

        return out

    def _triangulate(
        self, loops: TrimLoops, max_angle_deg: float, chord_factor: float
    ) -> Mesh:
        """Constrained Delaunay of the loops in UV, refined, trimmed, lifted and welded: the one body mesh_q and mesh_loops share."""

        from .remesh_nurbssurface_grid import RemeshNurbsSurfaceGrid

        if not loops.uv or len(loops.uv[0]) < 3:
            return self.m_surface.mesh()

        bbox_diag = self._bbox_diagonal()
        deflection = bbox_diag * chord_factor
        cos_max_angle = math.cos(min(max(max_angle_deg, 0.1), 179.0) * PI / 180.0)
        crease_knots = _find_crease_knots(self.m_surface)
        bounds = _loop_bounds(loops.uv[0])

        dt = _Delaunay2D(bounds[0], bounds[1], bounds[2], bounds[3])
        boundary_intervals = {}
        loop_vids = _insert_loops(dt, loops.uv, crease_knots, boundary_intervals)
        _insert_crease_lines(dt, loops.uv, crease_knots)

        for p in loops.interior_uv:
            if _inside_loops(p[0], p[1], loops.uv):
                dt.insert(p[0], p[1])

        _refine(dt, self.m_surface, loops.uv, crease_knots, deflection, cos_max_angle)
        _trim_outside(dt, loops.uv)
        tris = dt.get_triangles()

        if not tris or _crosses_crease(tris, dt, crease_knots):
            return Mesh()

        result = Mesh()
        welder = _VertexWelder(result, bbox_diag * 1e-5 if not loops.xyz else 0.0, bbox_diag * 1e-5)
        vert_map = _weld_vertices(welder, self.m_surface, dt, tris, loops, loop_vids, boundary_intervals)
        _add_faces(result, tris, vert_map)
        _set_vertex_normals(result, self.m_surface, dt, vert_map)
        _tag_boundary(result, loop_vids, boundary_intervals, vert_map)
        RemeshNurbsSurfaceGrid._split_crease_normals(self.m_surface, result)

        return result
