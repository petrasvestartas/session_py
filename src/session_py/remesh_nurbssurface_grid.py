from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import math
import sys

from .tolerance import PI
from .point import Point
from .vector import Vector

if TYPE_CHECKING:
    from .nurbssurface import NurbsSurface
    from .mesh import Mesh


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
MAX_SUBS = 24


# ═══════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════
def _norm(v: Vector) -> float:
    """Euclidean length without the zero gate of magnitude()."""
    return math.sqrt(v.magnitude_squared())


def _point_along(s: NurbsSurface, dir: int, t: float, fixed: float) -> Point:
    """Surface point at t along dir with the other parameter fixed."""
    return s.point_at(t, fixed) if dir == 0 else s.point_at(fixed, t)


def _normal_along(s: NurbsSurface, dir: int, t: float, fixed: float) -> Vector:
    """Surface normal at t along dir with the other parameter fixed."""
    return s.normal_at(t, fixed) if dir == 0 else s.normal_at(fixed, t)


def _raw_normal(s: NurbsSurface, u: float, v: float) -> Vector:
    """Sv x Su unnormalized, zero when the surface cannot be evaluated; normal_at would give a +Z sentinel at a pole."""

    derivatives = s.evaluate(u, v, 1)

    if len(derivatives) < 3:
        return Vector(0.0, 0.0, 0.0)

    return derivatives[2].cross(derivatives[1])


def _bbox_diagonal(s: NurbsSurface) -> float:
    """Diagonal of the control point bounding box."""

    lo = Point(1e30, 1e30, 1e30)
    hi = Point(-1e30, -1e30, -1e30)

    for i in range(s.cv_count(0)):
        for j in range(s.cv_count(1)):
            p = s.get_cv(i, j)

            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])

    return _norm(hi - lo)


# ═══════════════════════════════════════════════════════════════════════════
# Subdivisions
# ═══════════════════════════════════════════════════════════════════════════
def _span_angle(
    s: NurbsSurface, dir: int, t0: float, t1: float, osp: list[float]
) -> float:
    """Largest turn of the unit normal in degrees over [t0, t1], sampled at the span midpoints of the other direction."""

    max_angle = 0.0

    for si in range(len(osp) - 1):
        fixed = (osp[si] + osp[si + 1]) * 0.5

        first = Vector(0.0, 0.0, 0.0)
        last = Vector(0.0, 0.0, 0.0)
        has_first = False

        for k in range(5):
            n = _normal_along(s, dir, t0 + k * (t1 - t0) / 4.0, fixed)
            length = _norm(n)

            if length < 1e-10:
                continue

            unit = n / length

            if not has_first:
                first = unit

            has_first = True
            last = unit

        if not has_first:
            continue

        dot = max(-1.0, min(1.0, first.dot(last)))

        max_angle = max(max_angle, math.acos(dot) * 180.0 / PI)

    return max_angle


def _span_deviation(
    s: NurbsSurface, dir: int, t0: float, t1: float, osp: list[float]
) -> float:
    """Largest height of [t0, t1] over its chord, at up to four positions across the other direction."""

    max_dev = 0.0
    nc = min(len(osp) - 1, 3)

    for ci in range(nc + 1):
        fixed = osp[0] + ci * (osp[-1] - osp[0]) / max(nc, 1)
        p0 = _point_along(s, dir, t0, fixed)
        p1 = _point_along(s, dir, t1, fixed)

        for k in range(1, 4):
            frac = k / 4.0
            pm = _point_along(s, dir, t0 + frac * (t1 - t0), fixed)

            max_dev = max(max_dev, _norm(pm - (p0 + (p1 - p0) * frac)))

    return max_dev


def _span_subs(
    s: NurbsSurface,
    dir: int,
    sp: list[float],
    osp: list[float],
    max_angle_deg: float,
    chord_tol: float,
) -> list[int]:
    """Subdivisions per span along dir: the normal turn against max_angle_deg, the chord height against chord_tol, at least two on a curved span."""

    degree = s.degree(dir)
    subs = [1] * (len(sp) - 1)

    for i in range(len(sp) - 1):
        if degree > 1:
            angle = _span_angle(s, dir, sp[i], sp[i + 1], osp)

            subs[i] = min(max(math.ceil(angle / max_angle_deg), 1), MAX_SUBS)

        dev = _span_deviation(s, dir, sp[i], sp[i + 1], osp)

        if dev > chord_tol:
            subs[i] = max(
                subs[i],
                min(max(math.ceil(math.sqrt(dev / chord_tol)), 2), MAX_SUBS),
            )

        if degree > 1:
            subs[i] = max(subs[i], 2)

    return subs


def _isocurve_length(
    s: NurbsSurface, dir: int, sp: list[float], fixed: float, n: int
) -> float:
    """Length of the iso-curve at fixed along dir as a polyline of n steps."""

    length = 0.0
    prev = _point_along(s, dir, sp[0], fixed)

    for i in range(1, n + 1):
        next = _point_along(s, dir, sp[0] + i * (sp[-1] - sp[0]) / n, fixed)

        length += _norm(next - prev)
        prev = next

    return length


def _balance_subs(
    s: NurbsSurface,
    usp: list[float],
    vsp: list[float],
    u_subs: list[int],
    v_subs: list[int],
) -> None:
    """Scale up the curved direction whose spacing is more than twice the other's."""

    total_u = 1
    total_v = 1

    for sub in u_subs:
        total_u += sub

    for sub in v_subs:
        total_v += sub

    u_len = _isocurve_length(s, 0, usp, (vsp[0] + vsp[-1]) * 0.5, max(total_u, 10))
    v_len = _isocurve_length(s, 1, vsp, (usp[0] + usp[-1]) * 0.5, max(total_v, 10))

    if u_len <= 1e-14 or v_len <= 1e-14:
        return

    ratio = (u_len / total_u) / (v_len / total_v)

    if ratio > 2.0 and s.degree(0) > 1:
        scale = math.sqrt(ratio)

        for i in range(len(u_subs)):
            u_subs[i] = min(MAX_SUBS, math.ceil(u_subs[i] * scale))
    elif ratio < 0.5 and s.degree(1) > 1:
        scale = math.sqrt(1.0 / ratio)

        for i in range(len(v_subs)):
            v_subs[i] = min(MAX_SUBS, math.ceil(v_subs[i] * scale))


def _twist_subs(
    s: NurbsSurface, usp: list[float], vsp: list[float], twist_tol: float
) -> int:
    """Subdivisions both directions of a bilinear surface need for its twist, 1 when every span centre lies within twist_tol of its diagonal midpoint."""

    max_twist = 0.0

    for i in range(len(usp) - 1):
        for j in range(len(vsp) - 1):
            pm = s.point_at((usp[i] + usp[i + 1]) * 0.5, (vsp[j] + vsp[j + 1]) * 0.5)
            p00 = s.point_at(usp[i], vsp[j])
            p11 = s.point_at(usp[i + 1], vsp[j + 1])

            max_twist = max(max_twist, _norm(pm - Point.sum(p00, p11) * 0.5))

    if max_twist <= twist_tol:
        return 1

    return min(max(math.ceil(2.0 * math.sqrt(max_twist / twist_tol)), 4), MAX_SUBS)


def _make_odd(subs: list[int]) -> None:
    """One more subdivision on the largest span when the total is even, so a closed direction triangulates seamlessly."""

    total = 0

    for sub in subs:
        total += sub

    if total % 2 == 0:
        subs[subs.index(max(subs))] += 1


# ═══════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════
def _arclen_params(
    s: NurbsSurface, dir: int, n: int, sp: list[float], fixed: float
) -> list[float]:
    """n parameters spaced evenly by arc length along the iso-curve at fixed."""

    nsample = max(n * 20, 200)

    st = [0.0] * (nsample + 1)
    sl = [0.0] * (nsample + 1)
    prev = _point_along(s, dir, sp[0], fixed)

    for k in range(nsample + 1):
        st[k] = sp[0] + k * (sp[-1] - sp[0]) / nsample

        if k == 0:
            continue

        next = _point_along(s, dir, st[k], fixed)

        sl[k] = sl[k - 1] + _norm(next - prev)
        prev = next

    params = [sp[0]]
    j = 0

    for i in range(1, n - 1):
        target = sl[nsample] * i / (n - 1)

        while j < nsample and sl[j] < target:
            j += 1

        a = j - 1 if j > 0 else 0
        frac = (target - sl[a]) / (sl[j] - sl[a]) if sl[j] > sl[a] else 0.0

        params.append(st[a] + frac * (st[j] - st[a]))

    params.append(sp[-1])

    return params


def _span_params(sp: list[float], subs: list[int]) -> list[float]:
    """Every span split into its subdivisions, ending on the last span boundary."""

    params = []

    for i in range(len(sp) - 1):
        for sub in range(subs[i]):
            params.append(sp[i] + sub * (sp[i + 1] - sp[i]) / subs[i])

    params.append(sp[-1])

    return params


def _fix_closed_gap(params: list[float], domain_end: float) -> None:
    """Closed direction: drop the duplicate end and fill a wrap gap wider than 1.5 times the largest step."""

    if len(params) < 3:
        return

    params.pop()

    wrap_gap = domain_end - params[-1]
    max_gap = 0.0

    for i in range(1, len(params)):
        max_gap = max(max_gap, params[i] - params[i - 1])

    if max_gap <= 0.0 or wrap_gap <= max_gap * 1.5:
        return

    extra = math.ceil(wrap_gap / max_gap) - 1
    step = wrap_gap / (extra + 1)

    for e in range(1, extra + 1):
        params.append(params[-1] + step)


def _grid_params(
    s: NurbsSurface,
    dir: int,
    count: int,
    sp: list[float],
    fixed: float,
    subs: list[int],
) -> list[float]:
    """Parameters along dir: arc-length spaced when count is positive, else the span subdivisions; a closed direction made odd and its wrap gap filled."""

    closed = s.is_closed(dir)

    if closed and count == 0:
        _make_odd(subs)

    params = (
        _arclen_params(s, dir, max(count, 2), sp, fixed)
        if count > 0
        else _span_params(sp, subs)
    )

    if closed:
        _fix_closed_gap(params, sp[-1])

    return params


# ═══════════════════════════════════════════════════════════════════════════
# Vertices and faces
# ═══════════════════════════════════════════════════════════════════════════
def _add_vertex_uv(s: NurbsSurface, mesh: Mesh, u: float, v: float) -> int:
    """Vertex at S(u, v) tagged with its parameters."""

    key = mesh.add_vertex(s.point_at(u, v))

    mesh.vertex[key].attributes["u"] = u
    mesh.vertex[key].attributes["v"] = v

    return key


def _add_grid(
    s: NurbsSurface,
    mesh: Mesh,
    us: list[float],
    vs: list[float],
    j_start: int,
    j_end: int,
) -> list[int]:
    """Grid vertices row by row over us and the rows j_start..j_end of vs."""

    grid = []

    for u in us:
        for j in range(j_start, j_end):
            grid.append(_add_vertex_uv(s, mesh, u, vs[j]))

    return grid


def _add_faces(
    mesh: Mesh,
    grid: list[int],
    nu: int,
    closed_u: bool,
    wrap_v: bool,
    south: int | None,
    north: int | None,
) -> None:
    """Fans from the south pole, checkerboard-split quads, fans to the north pole."""

    nv = len(grid) // nu
    nu_faces = nu if closed_u else nu - 1
    nv_faces = nv if wrap_v else nv - 1

    if south is not None:
        for i in range(nu_faces):
            mesh.add_face([south, grid[((i + 1) % nu) * nv], grid[i * nv]])

    for i in range(nu_faces):
        for j in range(nv_faces):
            i1 = (i + 1) % nu
            j1 = (j + 1) % nv
            v00 = grid[i * nv + j]
            v10 = grid[i1 * nv + j]
            v01 = grid[i * nv + j1]
            v11 = grid[i1 * nv + j1]

            if (i + j) % 2 == 0:
                mesh.add_face([v00, v10, v11])
                mesh.add_face([v00, v11, v01])
            else:
                mesh.add_face([v00, v10, v01])
                mesh.add_face([v10, v11, v01])

    if north is not None:
        for i in range(nu_faces):
            mesh.add_face(
                [grid[i * nv + nv - 1], grid[((i + 1) % nu) * nv + nv - 1], north]
            )


# ═══════════════════════════════════════════════════════════════════════════
# Normals
# ═══════════════════════════════════════════════════════════════════════════
def _fan_normals(mesh: Mesh) -> list[Vector]:
    """Sum of the unnormalized face normals around each vertex key, faces taken in key order."""

    sums = []

    for key in range(len(mesh.vertex)):
        sums.append(Vector(0.0, 0.0, 0.0))

    for key in sorted(mesh.face):
        vertices = mesh.face[key]

        if len(vertices) < 3:
            continue

        p0 = mesh.vertex[vertices[0]].position()
        p1 = mesh.vertex[vertices[1]].position()
        p2 = mesh.vertex[vertices[2]].position()
        n = (p1 - p0).cross(p2 - p0)

        for vertex in vertices:
            sums[vertex] += n

    return sums


def _set_normals(
    s: NurbsSurface, mesh: Mesh, south: int | None, north: int | None
) -> None:
    """Unit surface normal on the side of the fan normal; the fan normal at the poles and where the surface normal vanishes, +Z when the fan vanishes too."""

    sums = _fan_normals(mesh)

    for key, vd in mesh.vertex.items():
        fan_length = _norm(sums[key])

        n = Vector(0.0, 0.0, 1.0)

        if math.isfinite(fan_length) and fan_length > 0.0:
            n = sums[key] / fan_length

        if key != south and key != north:
            raw = _raw_normal(s, vd.attributes["u"], vd.attributes["v"])
            length = _norm(raw)

            if math.isfinite(length) and length > 0.0:
                n = -raw / length if raw.dot(n) < 0.0 else raw / length

        vd.set_normal(n[0], n[1], n[2])


def _crease_flags(s: NurbsSurface, u: float, v: float) -> int:
    """Bit per direction where (u, v) sits on an internal knot of full multiplicity whose one-sided normals disagree."""

    uv = [u, v]
    flags = 0

    for dir in range(2):
        domain = s.domain(dir)
        value = uv[dir]

        if value <= domain[0] or value >= domain[1]:
            continue

        if list(s.m_nurbsknot[dir]).count(value) < s.degree(dir):
            continue

        lo = [u, v]
        hi = [u, v]

        lo[dir] = math.nextafter(value, -math.inf)
        hi[dir] = math.nextafter(value, math.inf)

        a = s.normal_at(lo[0], lo[1])
        b = s.normal_at(hi[0], hi[1])
        length = math.sqrt(a.magnitude_squared() * b.magnitude_squared())

        if length == 0.0:
            continue

        dot = a.dot(b) / length

        if math.isfinite(dot) and dot < 1.0 - 64.0 * sys.float_info.epsilon:
            flags |= 1 << dir

    return flags


def _crease_side(center: list[float], uv: list[float], flags: int) -> int:
    """Nudge uv one ulp toward center in each flagged direction; bit per direction nudged upward."""

    side = 0

    for dir in range(2):
        if not flags & (1 << dir):
            continue

        high = center[dir] > uv[dir]

        if high:
            side |= 1 << dir

        uv[dir] = math.nextafter(uv[dir], math.inf if high else -math.inf)

    return side


def _crease_target(mesh: Mesh, copies: dict, used: set, key: int, side: int) -> int:
    """Vertex carrying a corner: the original the first time its key is met, then one copy per (key, side)."""

    identity = (key, side)

    if identity in copies:
        return copies[identity]

    if key not in used:
        used.add(key)
        copies[identity] = key

        return key

    target = mesh.add_vertex(mesh.vertex[key].position())

    mesh.vertex[target] = copy.deepcopy(mesh.vertex[key])
    copies[identity] = target

    return target


class RemeshNurbsSurfaceGrid:
    """Grid mesh of a NURBS surface: spans split by normal turn and chord height, poles fanned, seams closed."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def from_u_v(s: NurbsSurface, max_u: int, max_v: int) -> Mesh:
        """Grid at 20 degrees and 0.5 percent of the bbox diagonal; max_u and max_v fix the parameter counts when positive."""
        return RemeshNurbsSurfaceGrid.from_u_v_q(s, max_u, max_v, 20.0, 0.005)

    @staticmethod
    def from_u_v_q(
        s: NurbsSurface,
        max_u: int,
        max_v: int,
        max_angle_deg: float,
        chord_factor: float,
    ) -> Mesh:
        """Grid with the normal turn per subdivision capped at max_angle_deg and the chord height at chord_factor of the bbox diagonal; vertex normals are unit surface normals on the fan side, fan normals at poles."""

        from .mesh import Mesh

        usp = s.get_span_vector(0)
        vsp = s.get_span_vector(1)
        bbox_diag = _bbox_diagonal(s)
        chord_tol = bbox_diag * chord_factor

        u_subs = _span_subs(s, 0, usp, vsp, max_angle_deg, chord_tol)
        v_subs = _span_subs(s, 1, vsp, usp, max_angle_deg, chord_tol)

        _balance_subs(s, usp, vsp, u_subs, v_subs)

        sing_v0 = s.is_singular(0)
        sing_v1 = s.is_singular(2)

        if s.degree(0) == 1 and s.degree(1) == 1 and not sing_v0 and not sing_v1:
            twist = _twist_subs(s, usp, vsp, chord_tol if bbox_diag > 0.0 else 1e-6)

            for i in range(len(u_subs)):
                u_subs[i] = max(u_subs[i], twist)

            for i in range(len(v_subs)):
                v_subs[i] = max(v_subs[i], twist)

        u_mid = (usp[0] + usp[-1]) * 0.5
        v_mid = (vsp[0] + vsp[-1]) * 0.5

        us = _grid_params(s, 0, max_u, usp, v_mid, u_subs)
        vs = _grid_params(s, 1, max_v, vsp, u_mid, v_subs)
        nv = len(vs)

        if sing_v0 and sing_v1 and nv < 3:
            return Mesh()

        mesh = Mesh()
        south = None
        north = None

        if sing_v0:
            south = _add_vertex_uv(s, mesh, us[0], vs[0])

        if sing_v1:
            north = _add_vertex_uv(s, mesh, us[0], vs[nv - 1])

        grid = _add_grid(
            s, mesh, us, vs, 1 if sing_v0 else 0, nv - 1 if sing_v1 else nv
        )

        _add_faces(
            mesh,
            grid,
            len(us),
            s.is_closed(0),
            s.is_closed(1) and not sing_v0 and not sing_v1,
            south,
            north,
        )
        _set_normals(s, mesh, south, north)
        RemeshNurbsSurfaceGrid._split_crease_normals(s, mesh)

        return mesh

    # ═══════════════════════════════════════════════════════════════════════════
    # Normals
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _split_crease_normals(s: NurbsSurface, mesh: Mesh) -> None:
        """Split shading vertices at internal C0 knots whose one-sided normals disagree."""

        candidates = {}

        for key, vd in mesh.vertex.items():
            if "u" not in vd.attributes or "v" not in vd.attributes:
                continue

            flags = _crease_flags(s, vd.attributes["u"], vd.attributes["v"])

            if flags:
                candidates[key] = flags

        if not candidates:
            return

        copies = {}
        used = set()

        for face_key, vertices in mesh.face.items():
            center = [0.0, 0.0]

            for key in vertices:
                center[0] += mesh.vertex[key].attributes["u"]
                center[1] += mesh.vertex[key].attributes["v"]

            center[0] /= len(vertices)
            center[1] /= len(vertices)

            face_normal = mesh.face_normal(face_key)

            for corner in range(len(vertices)):
                key = vertices[corner]

                if key not in candidates:
                    continue

                uv = [
                    mesh.vertex[key].attributes["u"],
                    mesh.vertex[key].attributes["v"],
                ]

                side = _crease_side(center, uv, candidates[key])
                target = _crease_target(mesh, copies, used, key, side)
                n = s.normal_at(uv[0], uv[1])
                length = _norm(n)

                if math.isfinite(length) and length > 0.0:
                    sign = (
                        -1.0
                        if face_normal is not None and n.dot(face_normal) < 0.0
                        else 1.0
                    )

                    mesh.vertex[target].set_normal(
                        sign * n[0] / length, sign * n[1] / length, sign * n[2] / length
                    )

                vertices[corner] = target

        mesh.rebuild_halfedges()
