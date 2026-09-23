from __future__ import annotations
import math
import numpy as np
from .point import Point
from .vector import Vector
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .nurbssurface_trimmed import NurbsSurfaceTrimmed
from .brep import BRep
from .brep import BRepRef
from .brep import BRepOrientation
from .closest import Closest
from .tolerance import PI

# ═══════════════════════════════════════════════════════════════════════════
# ISO 10303-21 parser
# ═══════════════════════════════════════════════════════════════════════════

REF = 0  # Entity reference.
NUM = 1  # Real or integer.
STR = 2  # Quoted string.
ENUM = 3  # Enum literal or bare identifier.
LIST = 4  # Parenthesised list.
NULL = 5  # Unset or derived value.


class StepParam:
    """One parameter of an entity instance."""

    def __init__(self):
        """Construct a null parameter."""

        self.tag = NULL  # Which member is set.
        self.ref_id = 0  # Entity id for Ref.
        self.num = 0.0  # Value for Num.
        self.str = ""  # Text for Str and Enum.
        self.list = []  # Items for List.


class StepSubEntity:
    """One TYPE(params) part of an entity instance."""

    def __init__(self):
        """Construct an empty part."""

        self.type = ""  # Entity type name.
        self.params = []  # Parameters in file order.


class StepEntity:
    """One entity: a single part for a simple instance, several for a complex one."""

    def __init__(self):
        """Construct an empty entity."""
        self.parts = []  # Sub-entities of the instance.

    def has(self, t: str) -> bool:
        """Return whether any sub-entity carries type t."""

        for p in self.parts:
            if p.type == t:
                return True

        return False

    def find(self, t: str) -> StepSubEntity | None:
        """Return the first sub-entity of type t, or null."""

        for p in self.parts:
            if p.type == t:
                return p

        return None


class StepFile:
    """Entities of a parsed file by id."""

    def __init__(self):
        """Construct an empty file."""
        self.entities = {}  # Entities by id.

    def ids_of_type(self, t: str) -> list[int]:
        """Return the sorted ids of every entity carrying type t."""

        out = []

        for k, e in self.entities.items():
            if e.has(t):
                out.append(k)

        out.sort()

        return out


PI_2 = 1.5707963267948966  # Quarter turn.
MAX_DEPTH = 8  # Deepest list nesting parsed recursively.
NS = 17  # Samples per side of a surface grid.


class Cursor:
    """Read position in a STEP text."""

    def __init__(self, text: str):
        """Construct at the start of text."""

        self.s = text  # Text being parsed.
        self.p = 0  # Current position.
        self.end = len(text)  # End of text.


def _skip_ws(c: Cursor) -> None:
    """Advance the cursor past whitespace."""
    while c.p < c.end and c.s[c.p].isspace():
        c.p += 1


def _consume(c: Cursor, ch: str) -> bool:
    """Advance past ch when it is next, skipping whitespace first."""

    _skip_ws(c)

    if c.p < c.end and c.s[c.p] == ch:
        c.p += 1

        return True

    return False


def _isident(ch: str) -> bool:
    """Return whether ch can start or continue an identifier."""
    return ch.isupper() or ch.isdigit() or ch == "_"


def _parse_int(c: Cursor) -> int:
    """Read an optionally signed integer."""

    id_ = 0

    while c.p < c.end and c.s[c.p].isdigit():
        id_ = id_ * 10 + (ord(c.s[c.p]) - 48)
        c.p += 1

    return id_


def _parse_number(c: Cursor) -> float:
    """Read a real, an integer or an enum literal as a double."""

    start = c.p

    if c.p < c.end and c.s[c.p] in "+-":
        c.p += 1

    while c.p < c.end and (c.s[c.p].isdigit() or c.s[c.p] == "."):
        c.p += 1

    if c.p < c.end and c.s[c.p] in "eE":
        q = c.p + 1

        if q < c.end and c.s[q] in "+-":
            q += 1

        if q < c.end and c.s[q].isdigit():
            c.p = q

            while c.p < c.end and c.s[c.p].isdigit():
                c.p += 1

    try:
        return float(c.s[start : c.p])
    except ValueError:
        return 0.0


def _parse_ident(c: Cursor) -> str:
    """Read an identifier."""

    start = c.p

    while c.p < c.end and _isident(c.s[c.p]):
        c.p += 1

    return c.s[start : c.p]


def _parse_string(c: Cursor) -> str:
    """Read a quoted string, unescaping doubled quotes."""

    out = []
    c.p += 1

    while c.p < c.end:
        if c.s[c.p] != "'":
            out.append(c.s[c.p])
            c.p += 1
            continue

        c.p += 1

        if c.p >= c.end or c.s[c.p] != "'":
            break

        out.append("'")
        c.p += 1

    return "".join(out)


def _parse_params(c: Cursor, depth: int) -> list[StepParam]:
    """Read a parenthesised parameter list, recursing one level deeper."""

    out = []
    _consume(c, "(")

    while c.p < c.end:
        _skip_ws(c)

        if c.p >= c.end or c.s[c.p] == ")":
            break

        out.append(_parse_param(c, depth))
        _skip_ws(c)

        if c.p < c.end and c.s[c.p] == ",":
            c.p += 1

    _consume(c, ")")

    return out


def _skip_list(c: Cursor) -> None:
    """Skip a parenthesised group without recursing, for lists nested deeper than MAX_DEPTH."""

    open_ = 0

    while c.p < c.end:
        if c.s[c.p] == "(":
            open_ += 1

        if c.s[c.p] == ")":
            open_ -= 1

        c.p += 1

        if open_ == 0:
            return


def _parse_param(c: Cursor, depth: int) -> StepParam:
    """Read one parameter: reference, number, string, enum, list or sub-entity."""

    _skip_ws(c)

    r = StepParam()

    if c.p >= c.end:
        return r

    ch = c.s[c.p]

    if ch == "#":
        c.p += 1
        r.tag = REF
        r.ref_id = _parse_int(c)
    elif ch == "$" or ch == "*":
        c.p += 1
    elif ch == "(":
        r.tag = LIST

        if depth < MAX_DEPTH:
            r.list = _parse_params(c, depth + 1)
        else:
            _skip_list(c)
    elif ch == "'":
        r.tag = STR
        r.str = _parse_string(c)
    elif ch == ".":
        c.p += 1

        start = c.p

        while c.p < c.end and c.s[c.p] != ".":
            c.p += 1

        r.tag = ENUM
        r.str = c.s[start : c.p]

        if c.p < c.end:
            c.p += 1
    elif ch.isdigit() or ch == "-" or ch == "+":
        r.tag = NUM
        r.num = _parse_number(c)
    elif ch.isupper():
        r.tag = ENUM
        r.str = _parse_ident(c)
        _skip_ws(c)

        if c.p < c.end and c.s[c.p] == "(":
            _parse_params(c, depth + 1)
    else:
        c.p += 1

    return r


def _parse_sub_entity(c: Cursor) -> StepSubEntity:
    """Read one TYPE(params) instance."""

    sub = StepSubEntity()
    sub.type = _parse_ident(c)
    _skip_ws(c)

    if c.p < c.end and c.s[c.p] == "(":
        sub.params = _parse_params(c, 0)

    return sub


def _skip_statement(c: Cursor) -> None:
    """Advance past the next semicolon."""

    in_str = False

    while c.p < c.end:
        ch = c.s[c.p]
        c.p += 1

        if ch == "'":
            in_str = not in_str

        if ch == ";" and not in_str:
            return


def _parse_step_string(content: str, sf: StepFile) -> None:
    """Fill sf from the DATA section of a STEP text."""

    c = Cursor(content)

    while c.p < c.end:
        _skip_ws(c)

        if c.p >= c.end:
            break

        if c.s[c.p] != "#":
            while c.p < c.end and c.s[c.p] != "\n":
                c.p += 1

            continue

        c.p += 1

        id_ = _parse_int(c)

        if not _consume(c, "="):
            continue

        _skip_ws(c)

        if c.p >= c.end:
            break

        ent = StepEntity()

        if c.s[c.p] == "(":
            c.p += 1

            while c.p < c.end:
                _skip_ws(c)

                if c.p >= c.end or not c.s[c.p].isupper():
                    break

                ent.parts.append(_parse_sub_entity(c))

            _consume(c, ")")
        else:
            ent.parts.append(_parse_sub_entity(c))

        if id_ not in sf.entities:
            sf.entities[id_] = ent

        _skip_statement(c)


def _strip_comments(raw: str) -> str:
    """Remove /* */ comments from the raw text."""

    text = []
    i = 0
    n = len(raw)

    while i < n:
        if i + 1 < n and raw[i] == "/" and raw[i + 1] == "*":
            i += 2

            while i + 1 < n and not (raw[i] == "*" and raw[i + 1] == "/"):
                i += 1

            i += 2
        else:
            text.append(raw[i])
            i += 1

    return "".join(text)


def _parse_step_file(filepath: str) -> StepFile:
    """Read and parse a STEP file."""

    sf = StepFile()

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        return sf

    text = _strip_comments(raw)
    lo = text.find("DATA")

    if lo < 0:
        return sf

    semi = text.find(";", lo)

    if semi < 0:
        return sf

    endsec = text.find("ENDSEC", semi)

    if endsec < 0:
        return sf

    _parse_step_string(text[semi + 1 : endsec], sf)

    return sf


# ═══════════════════════════════════════════════════════════════════════════
# Parameter access
# ═══════════════════════════════════════════════════════════════════════════


def _first_ref(params: list[StepParam]) -> int:
    """Return the first reference parameter, or -1."""

    for p in params:
        if p.tag == REF:
            return p.ref_id

    return -1


def _all_refs(params: list[StepParam]) -> list[int]:
    """Return every reference parameter in order."""

    out = []

    for p in params:
        if p.tag == REF:
            out.append(p.ref_id)

    return out


def _list_refs(params: list[StepParam]) -> list[int]:
    """Return every reference inside the list parameters."""

    out = []

    for p in params:
        out.extend(_all_refs(p.list))

    return out


def _nums(params: list[StepParam]) -> list[float]:
    """Return every numeric parameter in order."""

    out = []

    for p in params:
        if p.tag == NUM:
            out.append(p.num)

    return out


def _int_list(p: StepParam) -> list[int]:
    """Return the numbers of a list parameter as integers."""

    out = []

    for v in _nums(p.list):
        out.append(int(v))

    return out


def _dbl_list(p: StepParam) -> list[float]:
    """Return the numbers of a list parameter."""
    return _nums(p.list)


def _dbl_list_list(p: StepParam) -> list[list[float]]:
    """Return the numbers of a list-of-lists parameter."""

    out = []

    for row in p.list:
        out.append(_dbl_list(row))

    return out


def _ref_list_list(p: StepParam) -> list[list[int]]:
    """Return the references of a list-of-lists parameter."""

    out = []

    for row in p.list:
        out.append(_all_refs(row.list))

    return out


def _coords(params: list[StepParam]) -> list[float]:
    """Numbers of the first list parameter that holds any."""

    for p in params:
        out = _dbl_list(p)

        if out:
            return out

    return []


def _last_flag(params: list[StepParam], fallback: bool) -> bool:
    """Last enum parameter as a flag (.T. is true), fallback when there is none."""

    out = fallback

    for p in params:
        if p.tag == ENUM:
            out = p.str == "T"

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Knot utilities
# ═══════════════════════════════════════════════════════════════════════════


def _expand_knots(vals: list[float], mults: list[int]) -> list[float]:
    """Repeat each knot value by its multiplicity."""

    flat = []

    for i in range(min(len(vals), len(mults))):
        for j in range(mults[i]):
            flat.append(vals[i])

    return flat


def _compress_knots(flat) -> tuple[list[float], list[int]]:
    """Collapse a flat knot vector into values and multiplicities."""

    vals = []
    mults = []

    for v in flat:
        if not vals or abs(v - vals[-1]) > 1e-12:
            vals.append(float(v))
            mults.append(1)
        else:
            mults[-1] += 1

    return vals, mults


def _full_from_internal(internal) -> list[float]:
    """Add the two clamped end knots to an internal knot vector."""

    if len(internal) == 0:
        return []

    full = [float(internal[0])]

    for v in internal:
        full.append(float(v))

    full.append(float(internal[-1]))

    return full


def _internal_from_full(full: list[float]) -> list[float]:
    """Drop the two clamped end knots of a full knot vector."""

    if len(full) < 2:
        return list(full)

    return list(full[1:-1])


# ═══════════════════════════════════════════════════════════════════════════
# Analytic geometry
# ═══════════════════════════════════════════════════════════════════════════


class Axis2:
    """Orthonormal frame of an AXIS2_PLACEMENT_3D."""

    def __init__(self):
        """Construct the world frame."""

        self.origin = Point(0, 0, 0)  # Frame origin.
        self.ax = Vector(1, 0, 0)  # Frame x axis.
        self.ay = Vector(0, 1, 0)  # Frame y axis.
        self.az = Vector(0, 0, 1)  # Frame z axis.
        self.ok = False  # Whether the frame was read.


class Proj:
    """Parameter projector of a surface: a plane or a cylinder on the quarter-arc chart."""

    def __init__(self):
        """Construct a projector of no kind."""

        self.kind = 0  # 0 none, 1 plane, 2 cylinder.
        self.a = Axis2()  # Surface frame.


class AnFace:
    """Analytic surface of a face."""

    def __init__(self):
        """Construct a face of no kind."""

        self.kind = 0  # 2 cylinder, 3 cone, 4 sphere, 5 torus.
        self.a = Axis2()  # Surface frame.
        self.radius = 0.0  # Main radius.
        self.r2 = 0.0  # Cone semi-angle or torus minor radius.


def _axis_point(a: Axis2, lx: float, ly: float, lz: float) -> Point:
    """Return the point at local coordinates in the axis frame."""
    return a.origin + a.ax * lx + a.ay * ly + a.az * lz


def _angle_of(a: Axis2, pt: Point) -> float:
    """Return the angle of pt around the axis in radians."""

    d = pt - a.origin

    return math.atan2(d.dot(a.ay), d.dot(a.ax))


def _round_half_away(x: float) -> int:
    """Round to the nearest integer, halves away from zero like std::round."""

    r = math.floor(abs(x))

    if abs(x) - r >= 0.5:
        r += 1

    return int(math.copysign(r, x))


def _arc_param_of_angle(theta: float) -> float:
    """Parameter within one quarter-arc rational span (w = sqrt(2)/2) whose angle is theta."""

    if theta <= 0:
        return 0.0

    if theta >= PI_2:
        return 1.0

    w = math.sqrt(2.0) / 2.0
    tau = theta / PI_2

    for it in range(8):
        o = 1.0 - tau
        x = o * o + 2 * w * tau * o
        y = 2 * w * tau * o + tau * tau
        dx = -2 * o + 2 * w * (1 - 2 * tau)
        dy = 2 * w * (1 - 2 * tau) + 2 * tau
        f = math.atan2(y, x) - theta
        df = (x * dy - y * dx) / max(x * x + y * y, 1e-30)

        if abs(df) < 1e-30:
            break

        step = f / df
        tau = min(max(tau - step, 0.0), 1.0)

        if abs(step) < 1e-15:
            break

    return tau


def _chart_u_of_angle(ang: float, q0: int) -> float:
    """Chart coordinate of an angle: quarter-arc spans counted from quarter q0, corrected for the projective span parameterization."""

    q = ang / PI_2 - q0
    spanf = math.floor(q + 1e-12)

    return spanf + _arc_param_of_angle((q - spanf) * PI_2)


def _arc_nodes(q0: float, nspans: int) -> tuple[list[float], list[float], list[float]]:
    """Cos, sin and weight of the quarter-arc chart nodes from quarter q0: even nodes on the arc, odd nodes at the tangent corners."""

    n = 2 * nspans + 1
    ca = [0.0] * n
    sa = [0.0] * n
    cw = [0.0] * n

    for i in range(n):
        mid = i % 2 == 1
        a0 = (q0 + i // 2) * PI_2
        a1 = (q0 + i // 2 + 1) * PI_2
        ca[i] = math.cos(a0) + math.cos(a1) if mid else math.cos(a0)
        sa[i] = math.sin(a0) + math.sin(a1) if mid else math.sin(a0)
        cw[i] = math.sqrt(2.0) / 2.0 if mid else 1.0

    return ca, sa, cw


def _quarter_knots(nspans: int) -> list[float]:
    """Integer knots of nspans quarter arcs: 0, 0, 1, 1, ..., nspans, nspans."""

    knots = [0.0, 0.0]

    for s in range(1, nspans):
        knots.append(float(s))
        knots.append(float(s))

    knots.append(float(nspans))
    knots.append(float(nspans))

    return knots


def _an_st_of(an: AnFace, p: Point) -> tuple[float, float, bool]:
    """Canonical (s, t) of a 3D point; radial_ok is false at a pole or apex where the angle is undefined."""

    d = p - an.a.origin
    x = d.dot(an.a.ax)
    y = d.dot(an.a.ay)
    z = d.dot(an.a.az)
    rho = math.sqrt(x * x + y * y)
    s = math.atan2(y, x)
    radial_ok = rho > 1e-9

    if an.kind == 2:
        return s, z, radial_ok

    if an.kind == 3:
        ca = math.cos(an.r2)

        return s, z / ca if abs(ca) > 1e-12 else z, radial_ok

    if an.kind == 4:
        return s, math.atan2(z, rho), radial_ok

    return s, math.atan2(z, rho - an.radius), radial_ok


def _an_eval(an: AnFace, s: float, t: float) -> Point:
    """Evaluate the analytic surface at chart parameters s, t."""

    cs = math.cos(s)
    sn = math.sin(s)

    if an.kind == 2:
        return _axis_point(an.a, an.radius * cs, an.radius * sn, t)

    if an.kind == 3:
        r = an.radius + t * math.sin(an.r2)

        return _axis_point(an.a, r * cs, r * sn, t * math.cos(an.r2))

    ct = math.cos(t)
    st = math.sin(t)

    if an.kind == 4:
        return _axis_point(
            an.a, an.radius * ct * cs, an.radius * ct * sn, an.radius * st
        )

    r = an.radius + an.r2 * ct

    return _axis_point(an.a, r * cs, r * sn, an.r2 * st)


def _build_analytic_nurbs(
    an: AnFace, su0: int, nsu: int, t0: float, t1: float, sv0: int, nsv: int
) -> NurbsSurface:
    """Kernel NURBS window of an analytic surface: nsu quarter arcs from quarter su0 in u; v is linear on [t0, t1] for cylinder and cone, nsv quarter arcs from sv0 for sphere and torus."""

    ca, sa, cw = _arc_nodes(su0, nsu)
    nu = 2 * nsu + 1

    if an.kind == 2 or an.kind == 3:
        srf = NurbsSurface(3, True, 3, 2, nu, 2)
        srf.m_nurbsknot[0] = np.array(_quarter_knots(nsu), dtype=np.float64)
        srf.m_nurbsknot[1] = np.array([t0, t1], dtype=np.float64)

        for j in range(2):
            t = t0 if j == 0 else t1
            r = an.radius if an.kind == 2 else an.radius + t * math.sin(an.r2)
            z = t if an.kind == 2 else t * math.cos(an.r2)

            for i in range(nu):
                p = _axis_point(an.a, r * ca[i], r * sa[i], z)

                if not srf.set_cv_4d(
                    i, j, cw[i] * p[0], cw[i] * p[1], cw[i] * p[2], cw[i]
                ):
                    return NurbsSurface()

        return srf

    cb, sb, vw = _arc_nodes(sv0, nsv)
    nv = 2 * nsv + 1
    srf = NurbsSurface(3, True, 3, 3, nu, nv)
    srf.m_nurbsknot[0] = np.array(_quarter_knots(nsu), dtype=np.float64)
    srf.m_nurbsknot[1] = np.array(_quarter_knots(nsv), dtype=np.float64)

    for j in range(nv):
        r = an.radius * cb[j] if an.kind == 4 else an.radius + an.r2 * cb[j]
        z = an.radius * sb[j] if an.kind == 4 else an.r2 * sb[j]

        for i in range(nu):
            p = _axis_point(an.a, r * ca[i], r * sa[i], z)
            wij = cw[i] * vw[j]

            if not srf.set_cv_4d(i, j, wij * p[0], wij * p[1], wij * p[2], wij):
                return NurbsSurface()

    return srf


def _plane_surface(
    a: Axis2, u0: float, u1: float, v0: float, v1: float
) -> NurbsSurface:
    """Bilinear patch of the plane with axis a over [u0, u1] x [v0, v1]."""

    out = NurbsSurface(3, False, 2, 2, 2, 2)
    out.m_nurbsknot[0] = np.array([u0, u1], dtype=np.float64)
    out.m_nurbsknot[1] = np.array([v0, v1], dtype=np.float64)

    ok = (
        out.set_cv(0, 0, _axis_point(a, u0, v0, 0.0))
        and out.set_cv(0, 1, _axis_point(a, u0, v1, 0.0))
        and out.set_cv(1, 0, _axis_point(a, u1, v0, 0.0))
        and out.set_cv(1, 1, _axis_point(a, u1, v1, 0.0))
    )

    return out if ok else NurbsSurface()


def _cylinder_surface(
    a: Axis2, radius: float, u0: float, u1: float, v0: float, v1: float
) -> NurbsSurface:
    """Rational cylinder patch on the quarter-arc chart (1 unit = 90 degrees) over [u0, u1] x [v0, v1]; a span of 4 closes it."""

    closed = abs((u1 - u0) - 4.0) < 0.2
    n_spans = 4 if closed else max(1, math.ceil(abs(u1 - u0) - 1e-9))

    if closed:
        u1 = u0 + 4.0

    n_u = 2 * n_spans + 1
    out = NurbsSurface(3, True, 3, 2, n_u, 2)
    knots = [u0, u0]

    for s in range(1, n_spans):
        knots.append(u0 + s)
        knots.append(u0 + s)

    knots.append(u1)
    knots.append(u1)
    out.m_nurbsknot[0] = np.array(knots, dtype=np.float64)
    out.m_nurbsknot[1] = np.array([v0, v1], dtype=np.float64)

    ca, sa, cw = _arc_nodes(u0, n_spans)

    for i in range(n_u):
        for j in range(2):
            p = _axis_point(a, radius * ca[i], radius * sa[i], v0 if j == 0 else v1)

            if not out.set_cv_4d(i, j, cw[i] * p[0], cw[i] * p[1], cw[i] * p[2], cw[i]):
                return NurbsSurface()

    return out


def _project(pr: Proj, pt: Point) -> tuple[float, float]:
    """Parameter-space image of a 3D point: plane coordinates, or cylinder (angle in quarter turns, height)."""

    d = pt - pr.a.origin

    if pr.kind == 1:
        return d.dot(pr.a.ax), d.dot(pr.a.ay)

    return math.atan2(d.dot(pr.a.ay), d.dot(pr.a.ax)) * 2.0 / PI, d.dot(pr.a.az)


def _bilinear_projector(srf: NurbsSurface) -> Proj:
    """Affine projector of a bilinear patch from its corner p00; kind 0 when the patch is not bilinear or degenerate."""

    pr = Proj()

    if not srf.is_valid() or srf.degree(0) != 1 or srf.degree(1) != 1:
        return pr

    p00 = srf.get_cv(0, 0)
    eu = srf.get_cv(1, 0) - p00
    ev = srf.get_cv(0, 1) - p00
    eu2 = eu.dot(eu)
    ev2 = ev.dot(ev)

    if eu2 <= 1e-28 or ev2 <= 1e-28:
        return pr

    pr.kind = 1
    pr.a.origin = p00
    pr.a.ax = eu * (1.0 / eu2)
    pr.a.ay = ev * (1.0 / ev2)
    pr.a.ok = True

    return pr


# ═══════════════════════════════════════════════════════════════════════════
# Curve helpers
# ═══════════════════════════════════════════════════════════════════════════


def _sample_nurbs(nc: NurbsCurve, n: int) -> list[Point]:
    """n points evenly spaced in parameter over the curve domain."""

    tmin, tmax = nc.domain()
    pts = []

    for i in range(n):
        pts.append(nc.point_at(tmin + (tmax - tmin) * i / (n - 1) if n > 1 else tmin))

    return pts


def _polyline_nurbs(pts: list[Point], dim: int) -> NurbsCurve:
    """Degree-1 curve through the points with integer knots, dim 2 or 3; invalid for fewer than two points."""

    n = len(pts)

    if n < 2:
        return NurbsCurve()

    nc = NurbsCurve(dim, False, 2, n)

    for i in range(n):
        nc.m_nurbsknot[i] = float(i)

        if not nc.set_cv(i, pts[i]):
            return NurbsCurve()

    return nc


def _circle_nurbs(a: Axis2, rad: float, vs: Point, ve: Point) -> NurbsCurve:
    """Exact rational arc on the circle (axis a, radius rad) from vs to ve, the full circle when they coincide."""

    sa = _angle_of(a, vs)
    ea = _angle_of(a, ve)

    if vs.distance(ve) < 1e-10:
        ea = sa + 2.0 * PI
    elif ea <= sa:
        ea += 2.0 * PI

    span = ea - sa
    ns = max(1, math.ceil(abs(span) / PI_2))
    n_cp = 2 * ns + 1
    wm = math.cos(span / (2.0 * ns))
    crv = NurbsCurve(3, True, 3, n_cp)
    crv.m_nurbsknot[0] = sa
    crv.m_nurbsknot[1] = sa

    for s in range(1, ns):
        crv.m_nurbsknot[2 * s] = sa + s * span / ns
        crv.m_nurbsknot[2 * s + 1] = sa + s * span / ns

    crv.m_nurbsknot[2 * ns] = ea
    crv.m_nurbsknot[2 * ns + 1] = ea

    for i in range(n_cp):
        mid = i % 2 == 1
        ang = sa + (i // 2 + (0.5 if mid else 0.0)) * span / ns
        w = wm if mid else 1.0
        r2 = rad / wm if mid else rad
        p = a.origin + (a.ax * math.cos(ang) + a.ay * math.sin(ang)) * r2

        if not crv.set_cv_4d(i, w * p[0], w * p[1], w * p[2], w):
            return NurbsCurve()

    return crv


def _uv_line(u0: float, v0: float, u1: float, v1: float) -> NurbsCurve:
    """Degree-1 pcurve from (u0, v0) to (u1, v1)."""

    return NurbsCurve.create(
        False,
        1,
        [
            Point(u0, v0, 0),
            Point(u1, v1, 0),
        ],
    )


def _exact_pcurve(proj: Proj, c3: NurbsCurve) -> NurbsCurve:
    """Exact pcurve of a 3D curve under an affine projector: control points map one to one, weights unchanged."""

    if proj.kind != 1 or not c3.is_valid() or c3.cv_count() < 2:
        return NurbsCurve()

    p2 = NurbsCurve(3, c3.is_rational(), c3.order(), c3.cv_count())
    p2.m_nurbsknot = np.array(c3.m_nurbsknot, dtype=np.float64)

    for ci in range(c3.cv_count()):
        wx, wy, wz, w = c3.get_cv_4d(ci)

        if abs(w) < 1e-300:
            return NurbsCurve()

        u, v = _project(proj, Point(wx / w, wy / w, wz / w))

        if not p2.set_cv_4d(ci, u * w, v * w, 0.0, w):
            return NurbsCurve()

    return p2 if p2.is_valid() else NurbsCurve()


def _unwrap_seam(uv: list[Point]) -> None:
    """Keep consecutive cylinder samples on one branch of the quarter-arc chart (period 4)."""

    for k in range(1, len(uv)):
        du = uv[k][0] - uv[k - 1][0]

        if du > 2.0:
            uv[k][0] -= 4.0
        elif du < -2.0:
            uv[k][0] += 4.0


# ═══════════════════════════════════════════════════════════════════════════
# Surface grid
# ═══════════════════════════════════════════════════════════════════════════


def _surface_grid(srf: NurbsSurface, ns: int) -> list[Point]:
    """ns x ns surface points over the domain, row-major with u slowest."""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    grid = []

    for i in range(ns):
        for j in range(ns):
            grid.append(
                srf.point_at(
                    u0 + (u1 - u0) * i / (ns - 1), v0 + (v1 - v0) * j / (ns - 1)
                )
            )

    return grid


def _grid_scale(grid: list[Point]) -> float:
    """Return the largest distance from the first grid point."""

    scale = 0.0

    for p in grid:
        scale = max(scale, p.distance(grid[0]))

    return scale


def _grid_closed(grid: list[Point], ns: int, tol: float, along_u: bool) -> bool:
    """True when the first and last row (along_u) or column coincide within tol."""

    for k in range(ns):
        a = grid[k] if along_u else grid[k * ns]
        b = grid[(ns - 1) * ns + k] if along_u else grid[k * ns + ns - 1]

        if a.distance(b) > tol:
            return False

    return True


def _grid_degenerate(grid: list[Point], ns: int, tol: float, j: int) -> bool:
    """True when column j collapses to one point (a pole or apex)."""

    for k in range(1, ns):
        if grid[k * ns + j].distance(grid[j]) > tol:
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# StepReader
# ═══════════════════════════════════════════════════════════════════════════


class StepReader:
    """Entity access over a parsed file with points, directions and frames cached by id."""

    def __init__(self, sf: StepFile):
        """Construct over a parsed file."""

        self.sf = sf  # Parsed file.
        self.pt_cache = {}  # Points by id.
        self.dir_cache = {}  # Directions by id.
        self.ax_cache = {}  # Frames by id.

    def get(self, id_: int) -> StepEntity | None:
        """Return the entity with this id, or null."""
        return self.sf.entities.get(id_)

    def get_point(self, id_: int) -> Point:
        """Read a CARTESIAN_POINT, caching by id."""

        if id_ in self.pt_cache:
            return self.pt_cache[id_]

        pt = Point(0, 0, 0)
        e = self.get(id_)
        sub = e.find("CARTESIAN_POINT") if e is not None else None
        c = _coords(sub.params) if sub is not None else []

        if len(c) >= 3:
            pt = Point(c[0], c[1], c[2])
        elif len(c) == 2:
            pt = Point(c[0], c[1], 0.0)

        self.pt_cache[id_] = pt

        return pt

    def get_direction(self, id_: int) -> Vector:
        """Read a DIRECTION as a unit vector, caching by id."""

        if id_ in self.dir_cache:
            return self.dir_cache[id_]

        v = Vector(0, 0, 1)
        e = self.get(id_)
        sub = e.find("DIRECTION") if e is not None else None
        c = _coords(sub.params) if sub is not None else []

        if len(c) >= 3:
            v = Vector(c[0], c[1], c[2])

        self.dir_cache[id_] = v

        return v

    def get_axis2(self, id_: int) -> Axis2:
        """AXIS2_PLACEMENT_3D as an orthonormal frame: az normalized, ax made orthogonal to it, ay = az x ax."""

        if id_ in self.ax_cache:
            return self.ax_cache[id_]

        a = Axis2()
        e = self.get(id_)
        sub = e.find("AXIS2_PLACEMENT_3D") if e is not None else None
        refs = _all_refs(sub.params) if sub is not None else []

        if not refs:
            self.ax_cache[id_] = a

            return a

        a.origin = self.get_point(refs[0])
        a.az = self.get_direction(refs[1]) if len(refs) > 1 else Vector(0, 0, 1)

        ln = a.az.magnitude()

        if ln > 1e-12:
            a.az = a.az * (1.0 / ln)

        if len(refs) > 2:
            a.ax = self.get_direction(refs[2])
        else:
            a.ax = Vector(1, 0, 0) if abs(a.az[0]) < 0.9 else Vector(0, 1, 0)

        a.ax = a.ax - a.az * a.ax.dot(a.az)

        xn = a.ax.magnitude()

        if xn > 1e-12:
            a.ax = a.ax * (1.0 / xn)

        a.ay = a.az.cross(a.ax)
        a.ok = True
        self.ax_cache[id_] = a

        return a

    def get_nurbs_curve(self, id_: int) -> NurbsCurve:
        """B_SPLINE_CURVE_WITH_KNOTS, simple or complex with RATIONAL_B_SPLINE_CURVE; invalid when malformed."""

        e = self.get(id_)
        bsc = e.find("B_SPLINE_CURVE_WITH_KNOTS") if e is not None else None

        if bsc is None:
            return NurbsCurve()

        base = e.find("B_SPLINE_CURVE")
        rat = e.find("RATIONAL_B_SPLINE_CURVE")

        if base is None:
            pp = bsc.params

            if len(pp) < 8:
                return NurbsCurve()

            degree = int(pp[1].num)
            pt_refs = _all_refs(pp[2].list)
            mults = _int_list(pp[6])
            knots = _dbl_list(pp[7])
        else:
            bp = base.params
            kp = bsc.params

            if len(bp) < 2 or len(kp) < 2:
                return NurbsCurve()

            degree = int(bp[0].num)
            pt_refs = _all_refs(bp[1].list)
            mults = _int_list(kp[0])
            knots = _dbl_list(kp[1])

        if not pt_refs or not mults or not knots:
            return NurbsCurve()

        order = degree + 1
        cv_count = len(pt_refs)
        full = _expand_knots(knots, mults)

        if len(full) != cv_count + order:
            return NurbsCurve()

        internal = _internal_from_full(full)
        is_rat = rat is not None
        weights = _dbl_list(rat.params[0]) if is_rat and rat.params else []
        nc = NurbsCurve(3, is_rat, order, cv_count)

        if len(nc.m_nurbsknot) != len(internal):
            return NurbsCurve()

        nc.m_nurbsknot = np.array(internal, dtype=np.float64)

        for i in range(cv_count):
            pt = self.get_point(pt_refs[i])
            w = weights[i] if is_rat and i < len(weights) else 1.0

            if not nc.set_cv_4d(i, w * pt[0], w * pt[1], w * pt[2], w):
                return NurbsCurve()

        return nc

    def get_nurbs_surface(self, id_: int) -> NurbsSurface:
        """B_SPLINE_SURFACE_WITH_KNOTS, simple or complex with RATIONAL_B_SPLINE_SURFACE; invalid when malformed."""

        e = self.get(id_)
        bss = e.find("B_SPLINE_SURFACE_WITH_KNOTS") if e is not None else None

        if bss is None:
            return NurbsSurface()

        base = e.find("B_SPLINE_SURFACE")
        rat = e.find("RATIONAL_B_SPLINE_SURFACE")

        if base is None:
            pp = bss.params

            if len(pp) < 12:
                return NurbsSurface()

            u_deg = int(pp[1].num)
            v_deg = int(pp[2].num)
            ctrl_pts = _ref_list_list(pp[3])
            u_mults = _int_list(pp[8])
            v_mults = _int_list(pp[9])
            u_knots = _dbl_list(pp[10])
            v_knots = _dbl_list(pp[11])
        else:
            bp = base.params
            kp = bss.params

            if len(bp) < 3 or len(kp) < 4:
                return NurbsSurface()

            u_deg = int(bp[0].num)
            v_deg = int(bp[1].num)
            ctrl_pts = _ref_list_list(bp[2])
            u_mults = _int_list(kp[0])
            v_mults = _int_list(kp[1])
            u_knots = _dbl_list(kp[2])
            v_knots = _dbl_list(kp[3])

        if not ctrl_pts or not ctrl_pts[0] or not u_mults or not v_mults:
            return NurbsSurface()

        cv_u = len(ctrl_pts)
        cv_v = len(ctrl_pts[0])
        full_u = _expand_knots(u_knots, u_mults)
        full_v = _expand_knots(v_knots, v_mults)

        if len(full_u) != cv_u + u_deg + 1 or len(full_v) != cv_v + v_deg + 1:
            return NurbsSurface()

        is_rat = rat is not None
        weights = _dbl_list_list(rat.params[0]) if is_rat and rat.params else []
        srf = NurbsSurface(3, is_rat, u_deg + 1, v_deg + 1, cv_u, cv_v)
        srf.m_nurbsknot[0] = np.array(_internal_from_full(full_u), dtype=np.float64)
        srf.m_nurbsknot[1] = np.array(_internal_from_full(full_v), dtype=np.float64)

        for u in range(cv_u):
            for v in range(min(cv_v, len(ctrl_pts[u]))):
                pt = self.get_point(ctrl_pts[u][v])
                w = (
                    weights[u][v]
                    if is_rat and u < len(weights) and v < len(weights[u])
                    else 1.0
                )

                if not srf.set_cv_4d(u, v, w * pt[0], w * pt[1], w * pt[2], w):
                    return NurbsSurface()

        return srf if srf.is_valid() else NurbsSurface()

    def basis_curve_of(self, curve_id: int) -> int:
        """The 3D basis curve behind a SURFACE_CURVE or SEAM_CURVE, the id itself otherwise."""

        e = self.get(curve_id)
        sc = e.find("SURFACE_CURVE") if e is not None else None

        if e is not None and sc is None:
            sc = e.find("SEAM_CURVE")

        ref = _first_ref(sc.params) if sc is not None else -1

        return ref if ref >= 0 else curve_id

    def sample_curve(
        self, curve_id: int, v_start: Point, v_end: Point, n: int
    ) -> list[Point]:
        """n points along a curve entity: a B-spline or a circle arc between the vertices, else the two vertices."""

        ends = [v_start, v_end]
        id_ = self.basis_curve_of(curve_id)

        for depth in range(MAX_DEPTH):
            e = self.get(id_)
            tc = e.find("TRIMMED_CURVE") if e is not None else None

            if tc is None:
                break

            id_ = self.basis_curve_of(_first_ref(tc.params))

        e = self.get(id_)

        if e is None:
            return ends

        if e.has("B_SPLINE_CURVE_WITH_KNOTS"):
            nc = self.get_nurbs_curve(id_)

            return _sample_nurbs(nc, n) if nc.is_valid() else ends

        circle = e.find("CIRCLE")

        if circle is None:
            return ends

        ax_ref = _first_ref(circle.params)
        rr = _nums(circle.params)
        rad = rr[0] if rr else 0.0
        a = self.get_axis2(ax_ref)

        if ax_ref < 0 or rad == 0 or not a.ok:
            return ends

        sa = _angle_of(a, v_start)
        ea = _angle_of(a, v_end)

        if ea <= sa:
            ea += 2.0 * PI

        pts = []

        for i in range(n):
            ang = sa + (ea - sa) * i / (n - 1) if n > 1 else sa
            pts.append(a.origin + (a.ax * math.cos(ang) + a.ay * math.sin(ang)) * rad)

        return pts

    def get_projector(self, surface_id: int) -> Proj:
        """Return the parameter projector of a surface, caching by id."""

        pr = Proj()
        e = self.get(surface_id)
        plane = e.find("PLANE") if e is not None else None
        cyl = e.find("CYLINDRICAL_SURFACE") if e is not None else None
        sub = plane if plane is not None else cyl
        ref = _first_ref(sub.params) if sub is not None else -1

        if ref < 0:
            return pr

        pr.a = self.get_axis2(ref)
        pr.kind = 1 if plane is not None else 2

        return pr

    def fill_surface(
        self, id_: int, u0: float, u1: float, v0: float, v1: float
    ) -> NurbsSurface:
        """Kernel surface of a surface entity: the B-spline itself, or a plane or cylinder patch over the padded uv window."""

        e = self.get(id_)

        if e is None:
            return NurbsSurface()

        if e.has("B_SPLINE_SURFACE_WITH_KNOTS"):
            return self.get_nurbs_surface(id_)

        plane = e.find("PLANE")
        cyl = e.find("CYLINDRICAL_SURFACE")
        sub = plane if plane is not None else cyl
        ax_ref = _first_ref(sub.params) if sub is not None else -1
        a = self.get_axis2(ax_ref)

        if ax_ref < 0 or not a.ok:
            return NurbsSurface()

        pad_v = max(1e-6, 0.01 * (v1 - v0))

        if plane is not None:
            pad_u = max(1e-6, 0.01 * (u1 - u0))

            return _plane_surface(a, u0 - pad_u, u1 + pad_u, v0 - pad_v, v1 + pad_v)

        rr = _nums(sub.params)

        return _cylinder_surface(
            a, rr[0] if rr else 1.0, u0, u1, v0 - pad_v, v1 + pad_v
        )

    def get_analytic_srf(self, id_: int) -> AnFace:
        """CYLINDRICAL, CONICAL, SPHERICAL or TOROIDAL_SURFACE as an analytic face; kind 0 otherwise."""

        an = AnFace()
        e = self.get(id_)

        if e is None:
            return an

        kinds = [
            "CYLINDRICAL_SURFACE",
            "CONICAL_SURFACE",
            "SPHERICAL_SURFACE",
            "TOROIDAL_SURFACE",
        ]

        for i in range(len(kinds)):
            sub = e.find(kinds[i])

            if sub is None:
                continue

            ax_ref = _first_ref(sub.params)

            if ax_ref < 0:
                return an

            an.a = self.get_axis2(ax_ref)

            if not an.a.ok:
                return an

            rr = _nums(sub.params)
            an.kind = i + 2
            an.radius = rr[0] if rr else 0.0
            an.r2 = rr[1] if len(rr) > 1 else 0.0

            return an

        return an

    def pcurve_st_samples(
        self, ec_geom_id: int, surface_ref: int, forward_use: bool, n: int
    ) -> list[Point]:
        """Canonical (s, t) samples of the pcurve an edge carries on a surface; a SEAM_CURVE holds two, the second for the reversed use."""

        e = self.get(ec_geom_id)
        sc = e.find("SURFACE_CURVE") if e is not None else None
        is_seam = e is not None and sc is None and e.has("SEAM_CURVE")

        if is_seam:
            sc = e.find("SEAM_CURVE")

        if sc is None:
            return []

        mine = []

        for pid in _list_refs(sc.params):
            pe = self.get(pid)
            pc = pe.find("PCURVE") if pe is not None else None
            refs = _all_refs(pc.params) if pc is not None else []

            if len(refs) >= 2 and refs[0] == surface_ref:
                mine.append(refs[1])

        if not mine:
            return []

        pick = 1 if is_seam and len(mine) > 1 and not forward_use else 0
        dr = self.get(mine[pick])
        drs = dr.find("DEFINITIONAL_REPRESENTATION") if dr is not None else None
        c2_refs = _list_refs(drs.params) if drs is not None else []

        if not c2_refs:
            return []

        c2 = self.get_nurbs_curve(c2_refs[0])

        return _sample_nurbs(c2, n) if c2.is_valid() else []


# ═══════════════════════════════════════════════════════════════════════════
# Topology access
# ═══════════════════════════════════════════════════════════════════════════


class Bound:
    """One face bound: outer flag, orientation and the ORIENTED_EDGE ids of its EDGE_LOOP."""

    def __init__(self):
        """Construct an empty bound."""

        self.is_outer = False  # Whether the bound is FACE_OUTER_BOUND.
        self.orient = True  # Bound orientation flag.
        self.oe_refs = []  # ORIENTED_EDGE ids.


def _bound_loop(r: StepReader, bid: int) -> Bound | None:
    """Read a FACE_BOUND or FACE_OUTER_BOUND with its EDGE_LOOP."""

    bent = r.get(bid)
    bsub = bent.find("FACE_OUTER_BOUND") if bent is not None else None

    if bent is not None and bsub is None:
        bsub = bent.find("FACE_BOUND")

    if bsub is None:
        return None

    lent = r.get(_first_ref(bsub.params))
    loop = lent.find("EDGE_LOOP") if lent is not None else None

    if loop is None:
        return None

    b = Bound()
    b.is_outer = bent.has("FACE_OUTER_BOUND")
    b.orient = _last_flag(bsub.params, True)
    b.oe_refs = _list_refs(loop.params)

    return b


def _oriented_edge(r: StepReader, oe_id: int) -> tuple[int, bool]:
    """EDGE_CURVE id (-1 when missing) and orientation of an ORIENTED_EDGE."""

    oent = r.get(oe_id)
    oe = oent.find("ORIENTED_EDGE") if oent is not None else None

    if oe is None:
        return -1, True

    refs = _all_refs(oe.params)

    return refs[-1] if refs else -1, _last_flag(oe.params, True)


def _edge_refs(r: StepReader, ec_ref: int) -> list[int]:
    """Start vertex, end vertex and geometry ids of an EDGE_CURVE; empty when missing."""

    ecent = r.get(ec_ref)
    ec = ecent.find("EDGE_CURVE") if ecent is not None else None

    return _all_refs(ec.params) if ec is not None else []


def _edge_geom_id(r: StepReader, ec_ref: int) -> int:
    """Return the geometry id of an EDGE_CURVE, or -1."""

    refs = _edge_refs(r, ec_ref)

    return refs[2] if len(refs) >= 3 else -1


# ═══════════════════════════════════════════════════════════════════════════
# BRep assembly from STEP
# ═══════════════════════════════════════════════════════════════════════════


class PendingEdge:
    """One edge use in loop-traversal order; c2d is flipped into the edge direction when stored."""

    def __init__(self, edge: int, reversed_: bool, c2d: NurbsCurve):
        """Construct from edge index, direction and pcurve."""

        self.edge = edge  # Brep edge index.
        self.reversed = reversed_  # Whether the edge runs against the loop.
        self.c2d = c2d  # Parameter-space curve.


class LoopEdge:
    """One edge use with its parameter-space samples in curve order; pc2d is an exact pcurve when exact."""

    def __init__(self):
        """Construct an empty loop edge."""

        self.edge_idx = -1  # Brep edge index.
        self.reversed = False  # Whether the edge runs against the loop.
        self.uv = []  # Sampled uv points.
        self.pc2d = NurbsCurve()  # Parameter-space curve.
        self.exact = False  # Whether pc2d is exact rather than sampled.


class Loop:
    """One face loop with its edge uses in traversal order."""

    def __init__(self):
        """Construct an empty loop."""

        self.is_outer = False  # Whether the loop is the outer boundary.
        self.projected = False  # Whether the uv came from projection.
        self.edges = []  # Edges in traversal order.


class Window:
    """Chart window of an analytic face: quarter arcs from su0 in u, from sv0 in v for sphere and torus, [t0, t1] otherwise."""

    def __init__(self):
        """Construct an empty window."""

        self.su0 = 0  # First quarter arc in u.
        self.nsu = 0  # Quarter arcs in u.
        self.sv0 = 0  # First quarter arc in v.
        self.nsv = 0  # Quarter arcs in v.
        self.t0 = 0.0  # Start of the linear domain.
        self.t1 = 0.0  # End of the linear domain.


def _loop_bounds(lp: Loop) -> tuple[float, float, float, float]:
    """(umin, umax, vmin, vmax) over the samples of one loop; umin > umax when there are none."""

    umin = 1e300
    umax = -1e300
    vmin = 1e300
    vmax = -1e300

    for le in lp.edges:
        for p in le.uv:
            umin = min(umin, p[0])
            umax = max(umax, p[0])
            vmin = min(vmin, p[1])
            vmax = max(vmax, p[1])

    return umin, umax, vmin, vmax


def _loops_bounds(loops: list[Loop]) -> tuple[float, float, float, float]:
    """Return the uv bounds over every loop."""

    umin = 1e300
    umax = -1e300
    vmin = 1e300
    vmax = -1e300

    for lp in loops:
        u0, u1, v0, v1 = _loop_bounds(lp)
        umin = min(umin, u0)
        umax = max(umax, u1)
        vmin = min(vmin, v0)
        vmax = max(vmax, v1)

    return umin, umax, vmin, vmax


def _pick_outer_loop(loops: list[Loop]) -> None:
    """Mark the loop with the largest uv extent as outer when none is marked (OCCT and FreeCAD write FACE_BOUND for the outer boundary)."""

    for lp in loops:
        if lp.is_outer:
            return

    if not loops:
        return

    best = 0
    best_a = -1.0

    for i in range(len(loops)):
        u0, u1, v0, v1 = _loop_bounds(loops[i])
        a = (u1 - u0) * (v1 - v0) if u1 > u0 and v1 > v0 else 0.0

        if a > best_a:
            best_a = a
            best = i

    loops[best].is_outer = True


def _outer_first(loops: list[Loop]) -> list[Loop]:
    """Reorder so outer loops come before inner ones."""

    ordered = []

    for lp in loops:
        if lp.is_outer:
            ordered.append(lp)

    for lp in loops:
        if not lp.is_outer:
            ordered.append(lp)

    return ordered


def _loop_ucenter(lp: Loop) -> float | None:
    """Mean u of the samples of a loop, none when it has no samples."""

    total = 0.0
    cnt = 0

    for le in lp.edges:
        for p in le.uv:
            total += p[0]
            cnt += 1

    if cnt == 0:
        return None

    return total / cnt


def _surface_periods(proj: Proj, proj_srf: NurbsSurface) -> tuple[float, float]:
    """Periods of the parameter chart: 4 in u for the analytic cylinder, the domain span of each closed direction of a B-spline surface, 0 when open."""

    if proj.kind == 2:
        return 4.0, 0.0

    if not proj_srf.is_valid():
        return 0.0, 0.0

    du0, du1 = proj_srf.domain(0)
    dv0, dv1 = proj_srf.domain(1)
    scale = proj_srf.point_at(du0, dv0).distance(proj_srf.point_at(du1, dv1)) + 1e-9
    closed_u = True
    closed_v = True

    for k in range(5):
        fu = du0 + (du1 - du0) * k / 4.0
        fv = dv0 + (dv1 - dv0) * k / 4.0

        if (
            proj_srf.point_at(du0, fv).distance(proj_srf.point_at(du1, fv))
            > scale * 1e-6
        ):
            closed_u = False

        if (
            proj_srf.point_at(fu, dv0).distance(proj_srf.point_at(fu, dv1))
            > scale * 1e-6
        ):
            closed_v = False

    return du1 - du0 if closed_u else 0.0, dv1 - dv0 if closed_v else 0.0


def _chain_loops(loops: list[Loop], tau_u: float, tau_v: float) -> None:
    """Shift each edge by whole periods so its traversal start meets the previous edge's end."""

    if tau_u <= 0.0 and tau_v <= 0.0:
        return

    for lp in loops:
        prev_end = Point(0, 0, 0)
        have_prev = False

        for le in lp.edges:
            if not le.uv:
                continue

            st = le.uv[-1] if le.reversed else le.uv[0]
            n = (
                _round_half_away((prev_end[0] - st[0]) / tau_u)
                if have_prev and tau_u > 0.0
                else 0
            )
            m = (
                _round_half_away((prev_end[1] - st[1]) / tau_v)
                if have_prev and tau_v > 0.0
                else 0
            )

            for p in le.uv:
                p[0] += n * tau_u
                p[1] += m * tau_v

            prev_end = le.uv[0] if le.reversed else le.uv[-1]
            have_prev = True


def _center_inner_loops(loops: list[Loop], tau_u: float) -> None:
    """Shift each inner loop by whole u periods onto the outer loop's u window."""

    if tau_u <= 0.0:
        return

    outer = None

    for lp in loops:
        if lp.is_outer:
            outer = _loop_ucenter(lp)
            break

    if outer is None:
        return

    for lp in loops:
        center = None if lp.is_outer else _loop_ucenter(lp)

        if center is None:
            continue

        n = _round_half_away((outer - center) / tau_u)

        for le in lp.edges:
            for p in le.uv:
                p[0] += n * tau_u


def _pending_of(lp: Loop) -> list[PendingEdge]:
    """Pending edges of a loop in traversal order: the exact pcurve when there is one, else the sampled polyline."""

    pl = []

    for le in lp.edges:
        crv2d = le.pc2d.duplicate()

        if not le.exact or (le.reversed and not crv2d.reverse()):
            uv = list(le.uv)

            if le.reversed:
                uv.reverse()

            crv2d = _polyline_nurbs(uv, 2)

        pl.append(PendingEdge(le.edge_idx, le.reversed, crv2d))

    return pl


def _uv_of_samples(
    proj: Proj, proj_srf: NurbsSurface, samples: list[Point]
) -> list[Point]:
    """Parameter-space images of 3D samples: the analytic projection, or a warm-started closest-point search on proj_srf."""

    uv = []

    if proj.kind != 0:
        for s in samples:
            u, v = _project(proj, s)
            uv.append(Point(u, v, 0.0))

        return uv

    if not proj_srf.is_valid() or not samples:
        return [
            Point(0, 0, 0),
            Point(1, 0, 0),
        ]

    du0, du1 = proj_srf.domain(0)
    dv0, dv1 = proj_srf.domain(1)
    wu = (du1 - du0) * 0.1
    wv = (dv1 - dv0) * 0.1
    d_ref = 0.0
    pu = 0.0
    pv = 0.0

    for k in range(len(samples)):
        if k == 0:
            u, v, d = Closest.surface_point(proj_srf, samples[k])
            d_ref = d
        else:
            u, v, d = Closest.surface_point(
                proj_srf, samples[k], pu - wu, pu + wu, pv - wv, pv + wv
            )

            if d > 10 * d_ref + 1e-9:
                u, v, d = Closest.surface_point(proj_srf, samples[k])

        uv.append(Point(u, v, 0.0))
        pu = u
        pv = v

    return uv


def _chart_point(an: AnFace, w: Window, p: Point) -> Point:
    """Map a surface parameter point into the window chart."""

    angular = an.kind == 4 or an.kind == 5

    return Point(
        _chart_u_of_angle(p[0], w.su0),
        _chart_u_of_angle(p[1], w.sv0) if angular else p[1],
        0.0,
    )


def _chart_eval(an: AnFace, w: Window, q: Point) -> Point:
    """Evaluate the analytic surface at a window chart point."""

    angular = an.kind == 4 or an.kind == 5

    return _an_eval(
        an, (w.su0 + q[0]) * PI_2, (w.sv0 + q[1]) * PI_2 if angular else q[1]
    )


def _analytic_window(loops: list[Loop], an: AnFace) -> Window | None:
    """Chart window of the loops; none when they are empty or wider than 16 quarter arcs."""

    smin, smax, tmin, tmax = _loops_bounds(loops)

    if smin > smax:
        return None

    w = Window()
    w.su0 = math.floor(smin / PI_2 + 1e-9)
    w.nsu = max(1, math.ceil(smax / PI_2 - 1e-9) - w.su0)
    w.t0 = tmin
    w.t1 = tmax

    if w.nsu > 16:
        return None

    if an.kind == 4 or an.kind == 5:
        w.sv0 = math.floor(tmin / PI_2 + 1e-9)

        sv1 = math.ceil(tmax / PI_2 - 1e-9)

        if an.kind == 4:
            w.sv0 = max(w.sv0, -1)
            sv1 = min(sv1, 1)

        w.nsv = max(1, sv1 - w.sv0)

        if w.nsv > 16:
            return None
    elif tmax - tmin < 1e-12:
        return None

    return w


class BRepBuilder:
    """BRep of one STEP shell, built face by face."""

    def __init__(self, reader: StepReader):
        """Construct over a reader."""

        self.r = reader  # Entity reader.
        self.brep = BRep()  # Brep under construction.
        self.vmap = {}  # Brep vertex by VERTEX_POINT id.
        self.emap = {}  # Brep edge by EDGE_CURVE id.
        self.face_refs = []  # Face references in file order.

    def vertex_at(self, q: Point, tol: float) -> int:
        """Existing vertex within tol of q, else a new one."""

        for i in range(len(self.brep.m_vertices)):
            if self.brep.m_vertices[i].point.distance(q) <= tol:
                return i

        return self.brep.add_vertex(q)

    def attach_pcurve(self, edge: int, si: int, c2: int, reversed_use: bool) -> None:
        """The second use of an edge on the same surface is a seam: the forward use keeps curve_2d_index, the reversed one curve_2d_index_2."""

        for pc in self.brep.m_edges[edge].pcurves:
            if pc.surface_index != si:
                continue

            if reversed_use:
                pc.curve_2d_index_2 = c2
            else:
                pc.curve_2d_index_2 = pc.curve_2d_index
                pc.curve_2d_index = c2

            return

        self.brep.add_pcurve(edge, si, c2)

    def finish_face(
        self, si: int, reversed_face: bool, loops: list[list[PendingEdge]]
    ) -> None:
        """Face from its surface and loops (outer first), oriented in the shell by reversed_face."""

        wires = []

        for lp in loops:
            refs = []

            for pe in lp:
                c = pe.c2d.duplicate()

                if not pe.reversed or c.reverse():
                    self.attach_pcurve(
                        pe.edge, si, self.brep.add_curve_2d(c), pe.reversed
                    )

                refs.append(
                    BRepRef(
                        pe.edge,
                        BRepOrientation.Reversed
                        if pe.reversed
                        else BRepOrientation.Forward,
                    )
                )

            if refs:
                wires.append(BRepRef(self.brep.add_wire(refs), BRepOrientation.Forward))

        fi = self.brep.add_face(si, wires)
        self.face_refs.append(
            BRepRef(
                fi,
                BRepOrientation.Reversed if reversed_face else BRepOrientation.Forward,
            )
        )

    def get_vertex(self, vp_id: int) -> int:
        """Return the brep vertex of a VERTEX_POINT, creating it once."""

        if vp_id in self.vmap:
            return self.vmap[vp_id]

        e = self.r.get(vp_id)
        sub = e.find("VERTEX_POINT") if e is not None else None
        ref = _first_ref(sub.params) if sub is not None else -1
        pt = self.r.get_point(ref) if ref >= 0 else Point(0, 0, 0)
        idx = self.brep.add_vertex(pt)
        self.vmap[vp_id] = idx

        return idx

    def edge_curve(self, curve_id: int, vs: Point, ve: Point) -> NurbsCurve:
        """Exact 3D curve of an edge basis: the B-spline itself or a rational arc of a CIRCLE, invalid otherwise."""

        e = self.r.get(curve_id)

        if e is None:
            return NurbsCurve()

        if e.has("B_SPLINE_CURVE_WITH_KNOTS"):
            return self.r.get_nurbs_curve(curve_id)

        circle = e.find("CIRCLE")

        if circle is None:
            return NurbsCurve()

        ax_ref = _first_ref(circle.params)
        rr = _nums(circle.params)
        rad = rr[0] if rr else 0.0
        a = self.r.get_axis2(ax_ref)

        if ax_ref < 0 or rad <= 0 or not a.ok:
            return NurbsCurve()

        return _circle_nurbs(a, rad, vs, ve)

    def get_edge(self, ec_id: int) -> int:
        """BRep edge of an EDGE_CURVE, made once: exact curve when possible, else a sampled polyline."""

        if ec_id in self.emap:
            return self.emap[ec_id]

        refs = _edge_refs(self.r, ec_id)

        if len(refs) < 3:
            return -1

        sv = self.get_vertex(refs[0])
        ev = self.get_vertex(refs[1])
        curve_id = self.r.basis_curve_of(refs[2])
        vs = self.brep.m_vertices[sv].point
        ve = self.brep.m_vertices[ev].point
        crv3d = self.edge_curve(curve_id, vs, ve)

        if not crv3d.is_valid():
            crv3d = _polyline_nurbs(self.r.sample_curve(curve_id, vs, ve, 16), 3)

        idx = self.brep.add_edge(self.brep.add_curve_3d(crv3d), sv, ev)
        self.emap[ec_id] = idx

        return idx

    def st_projected(
        self, an: AnFace, geom_id: int, edge_idx: int, rev: bool, lp: Loop
    ) -> list[Point]:
        """Projection fallback: 3D samples of an edge mapped to canonical (s, t), branch-unwrapped along the loop traversal."""

        be = self.brep.m_edges[edge_idx]
        ordered = self.r.sample_curve(
            geom_id,
            self.brep.m_vertices[be.start_vertex].point,
            self.brep.m_vertices[be.end_vertex].point,
            48,
        )

        if len(ordered) < 2:
            return []

        if rev:
            ordered.reverse()

        ps = 0.0
        pt = 0.0
        have_prev = False

        if lp.edges and lp.edges[-1].uv:
            pe = lp.edges[-1]
            q = pe.uv[0] if pe.reversed else pe.uv[-1]
            ps = q[0]
            pt = q[1]
            have_prev = True

        for k in range(len(ordered)):
            if have_prev:
                break

            s2, t2, ok2 = _an_st_of(an, ordered[k])

            if not ok2:
                continue

            ps = s2
            pt = t2
            have_prev = True

        st = []

        for k in range(len(ordered)):
            s, t, ok = _an_st_of(an, ordered[k])

            if not ok and (k > 0 or have_prev):
                s = st[-1][0] if k > 0 else ps

            rs = st[-1][0] if k > 0 else (ps if have_prev else s)
            s -= 2 * PI * _round_half_away((s - rs) / (2 * PI))

            if an.kind == 5:
                rt = st[-1][1] if k > 0 else (pt if have_prev else t)
                t -= 2 * PI * _round_half_away((t - rt) / (2 * PI))

            st.append(Point(s, t, 0.0))

        if rev:
            st.reverse()

        return st

    def analytic_loops(
        self, bound_refs: list[int], surface_ref: int, an: AnFace, loops: list[Loop]
    ) -> bool:
        """Loops of an analytic face with canonical (s, t) samples from the file pcurves or from projection."""

        for bid in bound_refs:
            b = _bound_loop(self.r, bid)

            if b is None:
                continue

            lp = Loop()
            lp.is_outer = b.is_outer

            for oe_id in b.oe_refs:
                ec_ref, oe_orient = _oriented_edge(self.r, oe_id)
                edge_idx = -1 if ec_ref < 0 else self.get_edge(ec_ref)

                if edge_idx < 0:
                    continue

                geom_id = _edge_geom_id(self.r, ec_ref)
                le = LoopEdge()
                le.edge_idx = edge_idx
                le.reversed = oe_orient != b.orient

                if geom_id >= 0:
                    le.uv = self.r.pcurve_st_samples(
                        geom_id, surface_ref, oe_orient, 48
                    )

                if len(le.uv) < 2:
                    lp.projected = True
                    le.uv = self.st_projected(an, geom_id, edge_idx, le.reversed, lp)

                    if not le.uv:
                        return False

                lp.edges.append(le)

            if lp.edges:
                loops.append(lp)

            _pick_outer_loop(loops)

        return len(loops) > 0

    def analytic_pending(
        self, lp: Loop, an: AnFace, w: Window, scale3: float
    ) -> list[PendingEdge]:
        """Pending edges of one loop in the chart, plus a degenerated edge across each pole or apex gap between consecutive edges."""

        period = 4.0
        chains = []

        for le in lp.edges:
            uv = []

            for p in le.uv:
                uv.append(_chart_point(an, w, p))

            if le.reversed:
                uv.reverse()

            chains.append(uv)

        for k in range(1, len(chains)):
            if not lp.projected:
                break

            if not chains[k] or not chains[k - 1]:
                continue

            n = _round_half_away((chains[k - 1][-1][0] - chains[k][0][0]) / period)
            m = (
                _round_half_away((chains[k - 1][-1][1] - chains[k][0][1]) / period)
                if an.kind == 5
                else 0
            )

            for p in chains[k]:
                p[0] += n * period
                p[1] += m * period

        pl = []

        for k in range(len(lp.edges)):
            if len(chains[k]) < 2:
                continue

            pl.append(
                PendingEdge(
                    lp.edges[k].edge_idx,
                    lp.edges[k].reversed,
                    _polyline_nurbs(chains[k], 2),
                )
            )

            nxt = chains[(k + 1) % len(chains)]

            if not nxt:
                continue

            a2 = chains[k][-1]
            b2 = nxt[0]

            if abs(a2[0] - b2[0]) + abs(a2[1] - b2[1]) <= 1e-7:
                continue

            p3a = _chart_eval(an, w, a2)
            p3b = _chart_eval(an, w, b2)

            if p3a.distance(p3b) >= scale3 * 1e-6:
                continue

            vd = self.vertex_at(p3a, scale3 * 1e-6)
            pl.append(
                PendingEdge(
                    self.brep.add_edge(-1, vd, vd), False, _polyline_nurbs([a2, b2], 2)
                )
            )

        return pl

    def add_face_analytic(
        self, bound_refs: list[int], surface_ref: int, same_sense: bool, an: AnFace
    ) -> bool:
        """Face on a cylinder, cone, sphere or torus: the exact kernel window with the file pcurves bound in it; false falls back to projection."""

        loops = []

        if not self.analytic_loops(bound_refs, surface_ref, an, loops):
            return False

        w = _analytic_window(loops, an)

        if w is None:
            return False

        srf = _build_analytic_nurbs(an, w.su0, w.nsu, w.t0, w.t1, w.sv0, w.nsv)

        if not srf.is_valid():
            return False

        scale3 = an.radius + abs(an.r2) + 1.0
        srf_idx = self.brep.add_surface(srf)
        loops = _outer_first(loops)
        pending = []

        for lp in loops:
            pending.append(self.analytic_pending(lp, an, w, scale3))

        self.finish_face(srf_idx, not same_sense, pending)

        return True

    def step_point_of(self, vp_id: int) -> Point:
        """Point of a VERTEX_POINT, far away when missing."""

        e = self.r.get(vp_id)
        sub = e.find("VERTEX_POINT") if e is not None else None
        ref = _first_ref(sub.params) if sub is not None else -1

        return self.r.get_point(ref) if ref >= 0 else Point(1e300, 1e300, 1e300)

    def topo_vertex_at(self, vl_vertex_ids: list[int], q: Point, tol: float) -> int:
        """The file vertex at q when one of the given VERTEX_POINTs sits there, else a new vertex."""

        for vid in vl_vertex_ids:
            if self.step_point_of(vid).distance(q) <= tol:
                return self.get_vertex(vid)

        return self.brep.add_vertex(q)

    def add_face_vertex_loop(
        self, vl_vertex_ids: list[int], surface_ref: int, same_sense: bool
    ) -> bool:
        """Face bounded only by VERTEX_LOOPs: the whole surface, with seam and pole edges read off the surface (sphere-like or torus-like)."""

        an = self.r.get_analytic_srf(surface_ref)
        srf = NurbsSurface()

        if an.kind == 4:
            srf = _build_analytic_nurbs(an, 0, 4, 0, 0, -1, 2)
        elif an.kind == 5:
            srf = _build_analytic_nurbs(an, 0, 4, 0, 0, 0, 4)
        elif an.kind == 0:
            srf = self.r.get_nurbs_surface(surface_ref)

        if not srf.is_valid():
            return False

        u0, u1 = srf.domain(0)
        v0, v1 = srf.domain(1)
        grid = _surface_grid(srf, NS)
        tol = _grid_scale(grid) * 1e-7

        if not tol > 0:
            return False

        closed_u = _grid_closed(grid, NS, tol, True)
        closed_v = _grid_closed(grid, NS, tol, False)
        degen_v0 = _grid_degenerate(grid, NS, tol, 0)
        degen_v1 = _grid_degenerate(grid, NS, tol, NS - 1)

        if not closed_u or not ((degen_v0 and degen_v1) or closed_v):
            return False

        si = self.brep.add_surface(srf)
        wire = []

        if degen_v0 and degen_v1:
            v_lo = self.topo_vertex_at(vl_vertex_ids, grid[0], tol)
            v_hi = self.topo_vertex_at(vl_vertex_ids, grid[NS - 1], tol)
            seam = srf.iso_curve(1, u0)

            if not seam.is_valid():
                return False

            ei_seam = self.brep.add_edge(self.brep.add_curve_3d(seam), v_lo, v_hi)
            ei_lo = self.brep.add_edge(-1, v_lo, v_lo)
            ei_hi = self.brep.add_edge(-1, v_hi, v_hi)
            wire.append(PendingEdge(ei_lo, False, _uv_line(u0, v0, u1, v0)))
            wire.append(PendingEdge(ei_seam, False, _uv_line(u1, v0, u1, v1)))
            wire.append(PendingEdge(ei_hi, False, _uv_line(u1, v1, u0, v1)))
            wire.append(PendingEdge(ei_seam, True, _uv_line(u0, v1, u0, v0)))
        else:
            vtx = self.topo_vertex_at(vl_vertex_ids, grid[0], tol)
            c_u = srf.iso_curve(1, u0)
            c_v = srf.iso_curve(0, v0)

            if not c_u.is_valid() or not c_v.is_valid():
                return False

            ei_u = self.brep.add_edge(self.brep.add_curve_3d(c_u), vtx, vtx)
            ei_v = self.brep.add_edge(self.brep.add_curve_3d(c_v), vtx, vtx)
            wire.append(PendingEdge(ei_v, False, _uv_line(u0, v0, u1, v0)))
            wire.append(PendingEdge(ei_u, False, _uv_line(u1, v0, u1, v1)))
            wire.append(PendingEdge(ei_v, True, _uv_line(u1, v1, u0, v1)))
            wire.append(PendingEdge(ei_u, True, _uv_line(u0, v1, u0, v0)))

        self.finish_face(si, not same_sense, [wire])

        return True

    def vertex_loop_ids(self, bound_refs: list[int]) -> list[int]:
        """VERTEX_POINT ids of the VERTEX_LOOP bounds; empty when any bound is an EDGE_LOOP."""

        ids = []

        for bid in bound_refs:
            bent = self.r.get(bid)
            bsub = bent.find("FACE_OUTER_BOUND") if bent is not None else None

            if bent is not None and bsub is None:
                bsub = bent.find("FACE_BOUND")

            lent = self.r.get(_first_ref(bsub.params)) if bsub is not None else None

            if lent is None:
                continue

            if lent.has("EDGE_LOOP"):
                return []

            vl = lent.find("VERTEX_LOOP")
            ref = _first_ref(vl.params) if vl is not None else -1

            if ref >= 0:
                ids.append(ref)

        return ids

    def projected_loops(
        self,
        bound_refs: list[int],
        proj: Proj,
        proj_srf: NurbsSurface,
        loops: list[Loop],
    ) -> None:
        """Loops of a face on a projected surface: uv samples in curve order, exact pcurves under an affine projector."""

        n = 48 if proj_srf.is_valid() else 16
        exact = Proj()

        if proj.kind == 1:
            exact = proj
        elif proj.kind == 0:
            exact = _bilinear_projector(proj_srf)

        for bid in bound_refs:
            b = _bound_loop(self.r, bid)

            if b is None:
                continue

            lp = Loop()
            lp.is_outer = b.is_outer

            for oe_id in b.oe_refs:
                ec_ref, oe_orient = _oriented_edge(self.r, oe_id)
                edge_idx = -1 if ec_ref < 0 else self.get_edge(ec_ref)

                if edge_idx < 0:
                    continue

                be = self.brep.m_edges[edge_idx]
                vs = self.brep.m_vertices[be.start_vertex].point
                ve = self.brep.m_vertices[be.end_vertex].point
                le = LoopEdge()
                le.edge_idx = edge_idx
                le.reversed = oe_orient != b.orient
                le.uv = _uv_of_samples(
                    proj,
                    proj_srf,
                    self.r.sample_curve(_edge_geom_id(self.r, ec_ref), vs, ve, n),
                )

                if proj.kind == 2:
                    _unwrap_seam(le.uv)

                if be.curve_3d_index >= 0:
                    le.pc2d = _exact_pcurve(
                        exact, self.brep.m_curves_3d[be.curve_3d_index]
                    )

                le.exact = le.pc2d.is_valid()
                lp.edges.append(le)

            if lp.edges:
                loops.append(lp)

            _pick_outer_loop(loops)

    def add_face(self, face_id: int) -> None:
        """ADVANCED_FACE: vertex-loop face, analytic face, or projection onto the plane, cylinder chart or B-spline surface."""

        fent = self.r.get(face_id)
        face = fent.find("ADVANCED_FACE") if fent is not None else None

        if face is None:
            return

        bound_refs = _list_refs(face.params)
        surface_ref = _first_ref(face.params)
        same_sense = _last_flag(face.params, True)
        vl_ids = self.vertex_loop_ids(bound_refs)

        if vl_ids and self.add_face_vertex_loop(vl_ids, surface_ref, same_sense):
            return

        an = self.r.get_analytic_srf(surface_ref)

        if an.kind >= 2 and self.add_face_analytic(
            bound_refs, surface_ref, same_sense, an
        ):
            return

        proj = self.r.get_projector(surface_ref)
        proj_srf = (
            self.r.fill_surface(surface_ref, 0, 1, 0, 1)
            if proj.kind == 0
            else NurbsSurface()
        )
        loops = []
        self.projected_loops(bound_refs, proj, proj_srf, loops)

        tau_u, tau_v = _surface_periods(proj, proj_srf)
        _chain_loops(loops, tau_u, tau_v)
        _center_inner_loops(loops, tau_u)

        srf = proj_srf

        if not srf.is_valid():
            umin, umax, vmin, vmax = _loops_bounds(loops)

            if umin > umax:
                umin = -1.0
                umax = 1.0
                vmin = -1.0
                vmax = 1.0

            srf = self.r.fill_surface(surface_ref, umin, umax, vmin, vmax)

        srf_idx = self.brep.add_surface(srf)
        loops = _outer_first(loops)
        pending = []

        for lp in loops:
            pending.append(_pending_of(lp))

        self.finish_face(srf_idx, not same_sense, pending)

    def build_from_shell(self, shell_id: int) -> BRep:
        """BRep of a CLOSED_SHELL (one solid) or OPEN_SHELL (one shell), empty for anything else."""

        sent = self.r.get(shell_id)
        shell = sent.find("CLOSED_SHELL") if sent is not None else None

        if sent is not None and shell is None:
            shell = sent.find("OPEN_SHELL")

        if shell is None:
            return BRep()

        self.brep.name = "step_brep"

        for f in _list_refs(shell.params):
            self.add_face(f)

        if self.face_refs:
            sh = self.brep.add_shell(self.face_refs)

            if sent.has("CLOSED_SHELL"):
                self.brep.add_solid([BRepRef(sh, BRepOrientation.Forward)])

        return self.brep


# ═══════════════════════════════════════════════════════════════════════════
# StepWriter
# ═══════════════════════════════════════════════════════════════════════════


def _fmt(v: float) -> str:
    """ISO 10303-21 REAL: a decimal point in the mantissa and an uppercase E."""

    if abs(v) < 1e15 and v == int(v):
        s = f"{int(v)}."
    else:
        s = f"{v:.15g}"

    e = s.find("e")

    if e >= 0:
        s = s[:e] + "E" + s[e + 1 :]

        if "." not in s[:e]:
            s = s[:e] + "." + s[e:]
    elif "." not in s:
        s += "."

    return s


def _fmt_int_list(items: list[int]) -> str:
    """Format integers as a STEP list."""

    s = "("

    for i in range(len(items)):
        s += ("," if i else "") + str(items[i])

    return s + ")"


def _fmt_dbl_list(items: list[float]) -> str:
    """Format doubles as a STEP list."""

    s = "("

    for i in range(len(items)):
        s += ("," if i else "") + _fmt(items[i])

    return s + ")"


def _fmt_ref_list(ids: list[int]) -> str:
    """Format ids as a STEP reference list."""

    s = "("

    for i in range(len(ids)):
        s += (",#" if i else "#") + str(ids[i])

    return s + ")"


def _fmt_ref_grid(rows: list[list[int]]) -> str:
    """Format id rows as a STEP list of reference lists."""

    s = "("

    for i in range(len(rows)):
        s += ("," if i else "") + _fmt_ref_list(rows[i])

    return s + ")"


def _fmt_dbl_grid(rows: list[list[float]]) -> str:
    """Format double rows as a STEP list of lists."""

    s = "("

    for i in range(len(rows)):
        s += ("," if i else "") + _fmt_dbl_list(rows[i])

    return s + ")"


class StepWriter:
    """Entity lines of a STEP file under construction."""

    def __init__(self):
        """Construct with no entities."""

        self.next_id = 1  # Next free entity id.
        self.lines = []  # Emitted entity lines.

    def new_id(self) -> int:
        """Return the next free entity id."""

        id_ = self.next_id
        self.next_id += 1

        return id_

    def write_raw(self, body: str) -> int:
        """Emit "#id=body;" with a fresh id and return the id."""

        id_ = self.new_id()
        self.lines.append(f"#{id_}={body};")

        return id_

    def write_point(self, x: float, y: float, z: float) -> int:
        """Emit a CARTESIAN_POINT and return its id."""
        return self.write_raw(f"CARTESIAN_POINT('',({_fmt(x)},{_fmt(y)},{_fmt(z)}))")

    def write_nurbs_curve(self, nc: NurbsCurve) -> int:
        """B_SPLINE_CURVE_WITH_KNOTS, as a complete complex instance when rational; -1 for an invalid curve."""

        if not nc.is_valid():
            return -1

        pt_ids = []
        weights = []

        for i in range(nc.cv_count()):
            x, y, z, w = nc.get_cv_4d(i)

            if abs(w) < 1e-14:
                w = 1.0

            pt_ids.append(self.write_point(x / w, y / w, z / w))
            weights.append(w)

        kvals, kmults = _compress_knots(_full_from_internal(nc.m_nurbsknot))
        degree = str(nc.m_order - 1)

        if not nc.is_rational():
            return self.write_raw(
                "B_SPLINE_CURVE_WITH_KNOTS('',"
                + degree
                + ","
                + _fmt_ref_list(pt_ids)
                + ",.UNSPECIFIED.,.F.,.U.,"
                + _fmt_int_list(kmults)
                + ","
                + _fmt_dbl_list(kvals)
                + ",.UNSPECIFIED.)"
            )

        return self.write_raw(
            "(BOUNDED_CURVE()B_SPLINE_CURVE("
            + degree
            + ","
            + _fmt_ref_list(pt_ids)
            + ",.UNSPECIFIED.,.F.,.U.)"
            + "B_SPLINE_CURVE_WITH_KNOTS("
            + _fmt_int_list(kmults)
            + ","
            + _fmt_dbl_list(kvals)
            + ",.UNSPECIFIED.)"
            + "CURVE()GEOMETRIC_REPRESENTATION_ITEM()RATIONAL_B_SPLINE_CURVE("
            + _fmt_dbl_list(weights)
            + ")REPRESENTATION_ITEM(''))"
        )

    def write_nurbs_surface(self, srf: NurbsSurface) -> int:
        """B_SPLINE_SURFACE_WITH_KNOTS, as a complete complex instance when rational; -1 for an invalid surface."""

        if not srf.is_valid():
            return -1

        cv_u = srf.m_cv_count[0]
        cv_v = srf.m_cv_count[1]
        pt_ids = []
        weight_grid = []

        for u in range(cv_u):
            pt_ids.append([0] * cv_v)
            weight_grid.append([1.0] * cv_v)

            for v in range(cv_v):
                ok, x, y, z, w = srf.get_cv_4d(u, v)

                if not ok:
                    return -1

                if abs(w) < 1e-14:
                    w = 1.0

                pt_ids[u][v] = self.write_point(x / w, y / w, z / w)
                weight_grid[u][v] = w

        ku_vals, ku_mults = _compress_knots(_full_from_internal(srf.m_nurbsknot[0]))
        kv_vals, kv_mults = _compress_knots(_full_from_internal(srf.m_nurbsknot[1]))
        degrees = str(srf.m_order[0] - 1) + "," + str(srf.m_order[1] - 1)
        knots = (
            _fmt_int_list(ku_mults)
            + ","
            + _fmt_int_list(kv_mults)
            + ","
            + _fmt_dbl_list(ku_vals)
            + ","
            + _fmt_dbl_list(kv_vals)
        )

        if not srf.is_rational():
            return self.write_raw(
                "B_SPLINE_SURFACE_WITH_KNOTS('',"
                + degrees
                + ","
                + _fmt_ref_grid(pt_ids)
                + ",.UNSPECIFIED.,.F.,.F.,.U.,"
                + knots
                + ",.UNSPECIFIED.)"
            )

        return self.write_raw(
            "(BOUNDED_SURFACE()B_SPLINE_SURFACE("
            + degrees
            + ","
            + _fmt_ref_grid(pt_ids)
            + ",.UNSPECIFIED.,.F.,.F.,.U.)"
            + "B_SPLINE_SURFACE_WITH_KNOTS("
            + knots
            + ",.UNSPECIFIED.)GEOMETRIC_REPRESENTATION_ITEM()"
            + "RATIONAL_B_SPLINE_SURFACE("
            + _fmt_dbl_grid(weight_grid)
            + ")REPRESENTATION_ITEM('')SURFACE())"
        )

    def write_loop_as_face_bound(
        self, trimmed: NurbsSurfaceTrimmed, loop_2d: NurbsCurve, is_outer: bool
    ) -> int:
        """FACE_OUTER_BOUND or FACE_BOUND of one closed trim loop: its 3D image sampled as a polyline edge on one vertex."""

        if not loop_2d.is_valid():
            return -1

        pts3d = []

        for uv in _sample_nurbs(loop_2d, max(2, loop_2d.cv_count() * 2)):
            pts3d.append(trimmed.m_surface.point_at(uv[0], uv[1]))

        v0 = self.write_raw(
            f"VERTEX_POINT('',#{self.write_point(pts3d[0][0], pts3d[0][1], pts3d[0][2])})"
        )
        crv3d = self.write_nurbs_curve(_polyline_nurbs(pts3d, 3))

        if crv3d < 0:
            return -1

        self.write_nurbs_curve(loop_2d)

        ec = self.write_raw(f"EDGE_CURVE('',#{v0},#{v0},#{crv3d},.T.)")
        oe = self.write_raw(f"ORIENTED_EDGE('',*,*,#{ec},.T.)")
        el = self.write_raw(f"EDGE_LOOP('',(#{oe}))")
        fb_type = "FACE_OUTER_BOUND" if is_outer else "FACE_BOUND"

        return self.write_raw(f"{fb_type}('',#{el},.T.)")

    def write_trimmed_face(self, trimmed: NurbsSurfaceTrimmed) -> int:
        """ADVANCED_FACE of a trimmed surface; -1 when the surface or the outer loop cannot be written."""

        srf_id = self.write_nurbs_surface(trimmed.m_surface)

        if srf_id < 0:
            return -1

        bounds = [self.write_loop_as_face_bound(trimmed, trimmed.m_outer_loop, True)]

        if bounds[0] < 0:
            return -1

        for inner in trimmed.m_inner_loops:
            ib = self.write_loop_as_face_bound(trimmed, inner, False)

            if ib >= 0:
                bounds.append(ib)

        return self.write_raw(
            f"ADVANCED_FACE('',{_fmt_ref_list(bounds)},#{srf_id},.T.)"
        )

    def color_style(self, r: float, g: float, b: float) -> int:
        """AP214 surface-color chain; returns the PRESENTATION_STYLE_ASSIGNMENT for STYLED_ITEMs."""

        c = self.write_raw(f"COLOUR_RGB('',{_fmt(r)},{_fmt(g)},{_fmt(b)})")
        fc = self.write_raw(f"FILL_AREA_STYLE_COLOUR('',#{c})")
        fa = self.write_raw(f"FILL_AREA_STYLE('',(#{fc}))")
        sf = self.write_raw(f"SURFACE_STYLE_FILL_AREA(#{fa})")
        ss = self.write_raw(f"SURFACE_SIDE_STYLE('',(#{sf}))")
        su = self.write_raw(f"SURFACE_STYLE_USAGE(.BOTH.,#{ss})")

        return self.write_raw(f"PRESENTATION_STYLE_ASSIGNMENT((#{su}))")

    def write_body(self, faces: list[int], closed: bool) -> int:
        """CLOSED_SHELL + MANIFOLD_SOLID_BREP or OPEN_SHELL + SHELL_BASED_SURFACE_MODEL over the faces; -1 when there are none."""

        if not faces:
            return -1

        shell_type = "CLOSED_SHELL" if closed else "OPEN_SHELL"
        shell = self.write_raw(f"{shell_type}('',{_fmt_ref_list(faces)})")

        if closed:
            return self.write_raw(f"MANIFOLD_SOLID_BREP('',#{shell})")

        return self.write_raw(f"SHELL_BASED_SURFACE_MODEL('',(#{shell}))")

    def finish_product(
        self,
        bodies: list[int],
        closed: bool,
        name: str,
        uncertainty: float = 1e-6,
        styled_items: list[int] | None = None,
    ) -> None:
        """AP214 PRODUCT and SHAPE_DEFINITION_REPRESENTATION skeleton importers need to find the bodies; uncertainty is the sewing tolerance."""

        o = self.write_point(0, 0, 0)
        dz = self.write_raw("DIRECTION('',(0.,0.,1.))")
        dx = self.write_raw("DIRECTION('',(1.,0.,0.))")
        ax = self.write_raw(f"AXIS2_PLACEMENT_3D('',#{o},#{dz},#{dx})")
        lu = self.write_raw("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
        au = self.write_raw("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
        su = self.write_raw("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")

        if not math.isfinite(uncertainty) or uncertainty <= 0.0:
            uncertainty = 1e-6

        un = self.write_raw(
            f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({_fmt(uncertainty)}),#{lu},'distance_accuracy_value','')"
        )
        gc = self.write_raw(
            f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{un}))GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{au},#{su}))REPRESENTATION_CONTEXT('',''))"
        )
        ac = self.write_raw(
            "APPLICATION_CONTEXT('core data for automotive mechanical design processes')"
        )
        self.write_raw(
            f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{ac})"
        )

        pc = self.write_raw(f"PRODUCT_CONTEXT('',#{ac},'mechanical')")
        pr = self.write_raw(f"PRODUCT('{name}','{name}','',(#{pc}))")
        pf = self.write_raw(f"PRODUCT_DEFINITION_FORMATION('','',#{pr})")
        dc = self.write_raw(
            f"PRODUCT_DEFINITION_CONTEXT('part definition',#{ac},'design')"
        )
        pd = self.write_raw(f"PRODUCT_DEFINITION('design','',#{pf},#{dc})")
        ps = self.write_raw(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
        rep_type = "MANIFOLD_SURFACE_SHAPE_REPRESENTATION"

        if not bodies:
            rep_type = "SHAPE_REPRESENTATION"
        elif closed:
            rep_type = "ADVANCED_BREP_SHAPE_REPRESENTATION"

        items = [ax] + bodies
        rp = self.write_raw(f"{rep_type}('{name}',{_fmt_ref_list(items)},#{gc})")
        self.write_raw(f"SHAPE_DEFINITION_REPRESENTATION(#{ps},#{rp})")

        if styled_items:
            self.write_raw(
                f"MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',{_fmt_ref_list(styled_items)},#{gc})"
            )

    def emit(self) -> str:
        """Return the complete STEP text with header and data sections."""

        out = "ISO-10303-21;\nHEADER;\n"
        out += "FILE_DESCRIPTION((''),'2;1');\n"
        out += "FILE_NAME('','',(''),(''),'','','');\n"
        out += "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n"
        out += "ENDSEC;\nDATA;\n"

        for line in self.lines:
            out += line + "\n"

        out += "ENDSEC;\nEND-ISO-10303-21;\n"

        return out


class BRepEmitter:
    """Write one BRep into a StepWriter: vertices, edges and surfaces once each, degenerated edges omitted, a wire of only degenerated edges as a VERTEX_LOOP."""

    def __init__(self, writer: StepWriter, b: BRep):
        """Construct over a writer and the brep to emit."""

        self.w = writer  # Entity writer.
        self.brep = b  # Brep being emitted.
        self.vid = {}  # VERTEX_POINT id by vertex.
        self.eid = {}  # EDGE_CURVE id by edge.
        self.sid = {}  # Surface id by surface.

    def vertex_id(self, vi: int) -> int:
        """Return the VERTEX_POINT id of a brep vertex, emitting it once."""

        if vi in self.vid:
            return self.vid[vi]

        p = self.brep.m_vertices[vi].point
        self.vid[vi] = self.w.write_raw(
            f"VERTEX_POINT('',#{self.w.write_point(p[0], p[1], p[2])})"
        )

        return self.vid[vi]

    def edge_id(self, ei: int) -> int:
        """Return the EDGE_CURVE id of a brep edge, emitting it once."""

        if ei in self.eid:
            return self.eid[ei]

        e = self.brep.m_edges[ei]
        c = self.w.write_nurbs_curve(self.brep.m_curves_3d[e.curve_3d_index])

        if c < 0:
            self.eid[ei] = -1

            return -1

        sv = self.vertex_id(e.start_vertex)
        ev = self.vertex_id(e.end_vertex)
        self.eid[ei] = self.w.write_raw(f"EDGE_CURVE('',#{sv},#{ev},#{c},.T.)")

        return self.eid[ei]

    def surface_id(self, si: int) -> int:
        """Return the surface id of a brep surface, emitting it once."""

        if si in self.sid:
            return self.sid[si]

        self.sid[si] = self.w.write_nurbs_surface(self.brep.m_surfaces[si])

        return self.sid[si]

    def wire_id(self, wire: BRepRef) -> int:
        """EDGE_LOOP of the non-degenerated edges, a VERTEX_LOOP when there are none, -1 for an empty wire."""

        oes = []
        any_vertex = -1

        for er in self.brep.wire_edges(wire):
            e = self.brep.m_edges[er.index]

            if any_vertex < 0:
                any_vertex = e.start_vertex

            if e.degenerated:
                continue

            ec = self.edge_id(er.index)

            if ec < 0:
                continue

            sense = "T" if er.orientation == BRepOrientation.Forward else "F"
            oes.append(self.w.write_raw(f"ORIENTED_EDGE('',*,*,#{ec},.{sense}.)"))

        if oes:
            return self.w.write_raw(f"EDGE_LOOP('',{_fmt_ref_list(oes)})")

        if any_vertex >= 0:
            return self.w.write_raw(f"VERTEX_LOOP('',#{self.vertex_id(any_vertex)})")

        return -1

    def face_id(self, fi: int, fo: int) -> int:
        """Return the ADVANCED_FACE id of a brep face, emitting it once."""

        f = self.brep.m_faces[fi]
        srf = self.surface_id(f.surface_index)

        if srf < 0:
            return -1

        bounds = []

        for wi in range(len(f.wires)):
            loop = self.wire_id(f.wires[wi])

            if loop < 0:
                continue

            fb_type = "FACE_OUTER_BOUND" if wi == 0 else "FACE_BOUND"
            bounds.append(self.w.write_raw(f"{fb_type}('',#{loop},.T.)"))

        if not bounds:
            return -1

        sense = "T" if fo == BRepOrientation.Forward else "F"

        return self.w.write_raw(
            f"ADVANCED_FACE('',{_fmt_ref_list(bounds)},#{srf},.{sense}.)"
        )


def _emit_brep_shells(w: StepWriter, brep: BRep) -> list[tuple[list[int], bool]]:
    """Face-id groups of a brep written into w: one per shell with its closed flag, then the free faces as an open group."""

    em = BRepEmitter(w, brep)
    groups = []
    in_shell = [False] * len(brep.m_faces)

    for si in range(brep.shell_count()):
        ids = []

        for fr in brep.m_shells[si].faces:
            in_shell[fr.index] = True
            id_ = em.face_id(fr.index, fr.orientation)

            if id_ >= 0:
                ids.append(id_)

        if ids:
            groups.append((ids, brep.is_closed(si)))

    free_ids = []

    for fi in range(brep.face_count()):
        id_ = -1 if in_shell[fi] else em.face_id(fi, BRepOrientation.Forward)

        if id_ >= 0:
            free_ids.append(id_)

    if free_ids:
        groups.append((free_ids, False))

    return groups


def _vertex_diagonal(brep: BRep) -> float:
    """Bounding-box diagonal of the vertices, 1 when there are none."""

    if not brep.m_vertices:
        return 1.0

    lo = Point(1e300, 1e300, 1e300)
    hi = Point(-1e300, -1e300, -1e300)

    for v in brep.m_vertices:
        for k in range(3):
            lo[k] = min(lo[k], v.point[k])
            hi[k] = max(hi[k], v.point[k])

    return lo.distance(hi)


def _write_step_string(content: str, filepath: str) -> None:
    """Write the STEP text to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def read_file_step_points(filepath: str) -> list[Point]:
    """Every CARTESIAN_POINT of the file in entity-id order."""

    sf = _parse_step_file(filepath)
    r = StepReader(sf)
    out = []

    for id_ in sf.ids_of_type("CARTESIAN_POINT"):
        out.append(r.get_point(id_))

    return out


def read_file_step_nurbscurves(filepath: str) -> list[NurbsCurve]:
    """Every B_SPLINE_CURVE_WITH_KNOTS of the file that reads as a valid curve."""

    sf = _parse_step_file(filepath)
    r = StepReader(sf)
    out = []

    for id_ in sf.ids_of_type("B_SPLINE_CURVE_WITH_KNOTS"):
        nc = r.get_nurbs_curve(id_)

        if nc.is_valid():
            out.append(nc)

    return out


def read_file_step_nurbssurfaces(filepath: str) -> list[NurbsSurface]:
    """Every B_SPLINE_SURFACE_WITH_KNOTS of the file that reads as a valid surface."""

    sf = _parse_step_file(filepath)
    r = StepReader(sf)
    out = []

    for id_ in sf.ids_of_type("B_SPLINE_SURFACE_WITH_KNOTS"):
        srf = r.get_nurbs_surface(id_)

        if srf.is_valid():
            out.append(srf)

    return out


def _trimmed_outer_loop(r: StepReader, bound_refs: list[int]) -> NurbsCurve:
    """Outer trim of a face as a dim-2 polyline of sampled 3D edge points (x, y), from the first bound with an EDGE_LOOP."""

    for bid in bound_refs:
        b = _bound_loop(r, bid)

        if b is None:
            continue

        uv_pts = []

        for oe_id in b.oe_refs:
            ecr = _edge_refs(r, _oriented_edge(r, oe_id)[0])

            if len(ecr) < 3:
                continue

            vs = r.get_point(ecr[0])
            ve = r.get_point(ecr[1])

            for s in r.sample_curve(ecr[2], vs, ve, 8):
                uv_pts.append(Point(s[0], s[1], 0.0))

        return _polyline_nurbs(uv_pts, 2)

    return NurbsCurve()


def read_file_step_nurbssurfaces_trimmed(filepath: str) -> list[NurbsSurfaceTrimmed]:
    """Every ADVANCED_FACE on a B-spline surface with its first edge loop sampled as the outer trim."""

    sf = _parse_step_file(filepath)
    r = StepReader(sf)
    out = []

    for face_id in sf.ids_of_type("ADVANCED_FACE"):
        fent = r.get(face_id)
        face = fent.find("ADVANCED_FACE") if fent is not None else None

        if face is None:
            continue

        surface_ref = _first_ref(face.params)
        surf = r.get(surface_ref)

        if surf is None or not surf.has("B_SPLINE_SURFACE_WITH_KNOTS"):
            continue

        nst = NurbsSurfaceTrimmed()
        nst.m_surface = r.get_nurbs_surface(surface_ref)
        nst.m_outer_loop = _trimmed_outer_loop(r, _list_refs(face.params))

        if nst.m_surface.is_valid() and nst.m_outer_loop.is_valid():
            out.append(nst)

    return out


def read_file_step_breps(filepath: str) -> list[BRep]:
    """One BRep per shell of every MANIFOLD_SOLID_BREP, BREP_WITH_VOIDS and SHELL_BASED_SURFACE_MODEL, in file order."""

    sf = _parse_step_file(filepath)
    r = StepReader(sf)
    ids = sorted(sf.entities.keys())
    shell_refs = []

    for id_ in ids:
        e = r.get(id_)
        root = e.find("MANIFOLD_SOLID_BREP")

        if root is None:
            root = e.find("BREP_WITH_VOIDS")

        if root is None:
            root = e.find("SHELL_BASED_SURFACE_MODEL")

        if root is None:
            continue

        shell_refs.extend(_all_refs(root.params))
        shell_refs.extend(_list_refs(root.params))

    out = []

    for shell_ref in shell_refs:
        sh = r.get(shell_ref)
        os_ = sh.find("ORIENTED_CLOSED_SHELL") if sh is not None else None
        inner = _first_ref(os_.params) if os_ is not None else -1
        builder = BRepBuilder(r)
        b = builder.build_from_shell(inner if inner >= 0 else shell_ref)

        if b.m_faces:
            out.append(b)

    return out


def write_file_step_nurbscurves(curves: list[NurbsCurve], filepath: str) -> None:
    """One file holding the curves as bare B_SPLINE_CURVE_WITH_KNOTS entities."""

    w = StepWriter()

    for nc in curves:
        w.write_nurbs_curve(nc)

    _write_step_string(w.emit(), filepath)


def write_file_step_nurbssurfaces(surfaces: list[NurbsSurface], filepath: str) -> None:
    """One file holding the surfaces as bare B_SPLINE_SURFACE_WITH_KNOTS entities."""

    w = StepWriter()

    for srf in surfaces:
        w.write_nurbs_surface(srf)

    _write_step_string(w.emit(), filepath)


def write_file_step_nurbssurfaces_trimmed(
    trimmed: list[NurbsSurfaceTrimmed], filepath: str
) -> None:
    """One file holding the trimmed surfaces as ADVANCED_FACEs of an open shell."""

    w = StepWriter()
    face_ids = []

    for t in trimmed:
        fid = w.write_trimmed_face(t)

        if fid >= 0:
            face_ids.append(fid)

    bodies = []
    body = w.write_body(face_ids, False)

    if body >= 0:
        bodies.append(body)

    w.finish_product(bodies, False, "trimmed")
    _write_step_string(w.emit(), filepath)


def write_file_step_brep(brep: BRep, filepath: str) -> None:
    """One AP214 file holding the brep, one body per shell."""

    w = StepWriter()
    bodies = []
    any_closed = False

    for ids, closed in _emit_brep_shells(w, brep):
        body = w.write_body(ids, closed)

        if body >= 0:
            bodies.append(body)

        any_closed = any_closed or closed

    w.finish_product(
        bodies,
        any_closed,
        brep.name if brep.name else "brep",
        _vertex_diagonal(brep) * 1e-4,
    )
    _write_step_string(w.emit(), filepath)


def write_file_step_breps(breps: list[BRep], name: str, filepath: str) -> None:
    """One AP214 file holding several breps side by side, each face colored from its brep's surfacecolor."""

    w = StepWriter()
    bodies = []
    styled = []
    any_closed = False
    diag = 1.0

    for b in breps:
        if b is None:
            continue

        groups = _emit_brep_shells(w, b)
        diag = max(diag, _vertex_diagonal(b))
        psa = w.color_style(b.surfacecolor.r, b.surfacecolor.g, b.surfacecolor.b)

        for ids, closed in groups:
            body = w.write_body(ids, closed)

            if body >= 0:
                bodies.append(body)

            any_closed = any_closed or closed

            for fid in ids:
                styled.append(w.write_raw(f"STYLED_ITEM('',(#{psa}),#{fid})"))

    w.finish_product(bodies, any_closed, name, diag * 1e-4, styled)
    _write_step_string(w.emit(), filepath)
