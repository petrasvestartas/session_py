import uuid
import copy
import math

from .point import Point
from .vector import Vector
from .xform import Xform
from .color import Color
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface

import numpy as _np
_IDENTITY_XFORM = Xform.identity()
_ZERO_GUID = "00000000-0000-0000-0000-000000000000"
_BLACK_COLOR = Color.black()
_NURBSKNOT_01 = _np.array([0., 1.], dtype=_np.float64)


def _aabb_from_surface(srf, n=6):
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    lo = [1e30, 1e30, 1e30]
    hi = [-1e30, -1e30, -1e30]
    for i in range(n + 1):
        for j in range(n + 1):
            p = srf.point_at(u0 + (u1 - u0) * i / n, v0 + (v1 - v0) * j / n)
            for k in range(3):
                if p[k] < lo[k]: lo[k] = p[k]
                if p[k] > hi[k]: hi[k] = p[k]
    return lo, hi


def _aabb_from_curve(crv, n=16):
    c0, c1 = crv.domain()
    lo = [1e30, 1e30, 1e30]
    hi = [-1e30, -1e30, -1e30]
    for i in range(n + 1):
        p = crv.point_at(c0 + (c1 - c0) * i / n)
        for k in range(3):
            if p[k] < lo[k]: lo[k] = p[k]
            if p[k] > hi[k]: hi[k] = p[k]
    return lo, hi


def _srf_is_planar(s):
    du = s.domain(0); dv = s.domain(1)
    n0 = s.normal_at(0.5 * (du[0] + du[1]), 0.5 * (dv[0] + dv[1]))
    for uu, vv in ((0.25, 0.3), (0.75, 0.8)):
        n = s.normal_at(du[0] + (du[1] - du[0]) * uu, dv[0] + (dv[1] - dv[0]) * vv)
        cx = n0[1] * n[2] - n0[2] * n[1]
        cy = n0[2] * n[0] - n0[0] * n[2]
        cz = n0[0] * n[1] - n0[1] * n[0]
        if (cx * cx + cy * cy + cz * cz) ** 0.5 > 1e-7:
            return False
    return True


def _recognize_solid(X):
    """Recognize a solid BRep as an analytic primitive for O(1) point classification (skips
    the tessellate + ray-cast). Returns a descriptor dict or None when unrecognized -- the
    caller then falls back to the mesh ray-cast, so GENERAL / freeform solids stay correct;
    this is purely a fast path for the box/cylinder/sphere the kernel builds most. Recognition
    SELF-VERIFIES, so a wrong guess yields None (mesh fallback), never a wrong classification.
    Geometry is derived from SURFACE-sampled points, NOT m_vertices (which can be stale vs the
    surfaces after transformed())."""
    from .brep import BRepLoopType  # local to avoid forward-ref at import time
    if not X.m_faces or not X.m_surfaces:
        return None
    spts = []
    for f in X.m_faces:
        if f.surface_index < 0 or f.surface_index >= len(X.m_surfaces):
            return None
        s = X.m_surfaces[f.surface_index]
        du = s.domain(0); dv = s.domain(1)
        for a in (0.0, 0.5, 1.0):
            for b in (0.0, 0.5, 1.0):
                spts.append(s.point_at(du[0] + (du[1] - du[0]) * a, dv[0] + (dv[1] - dv[0]) * b))
    if not spts:
        return None
    xmn = ymn = zmn = 1e300; xmx = ymx = zmx = -1e300
    cx = cy = cz = 0.0
    for p in spts:
        xmn = min(xmn, p[0]); ymn = min(ymn, p[1]); zmn = min(zmn, p[2])
        xmx = max(xmx, p[0]); ymx = max(ymx, p[1]); zmx = max(zmx, p[2])
        cx += p[0]; cy += p[1]; cz += p[2]
    nv = len(spts)
    C = (cx / nv, cy / nv, cz / nv)
    diag = ((xmx - xmn) ** 2 + (ymx - ymn) ** 2 + (zmx - zmn) ** 2) ** 0.5
    if diag < 1e-12:
        return None
    tol = diag * 1e-6

    all_planar = True
    for f in X.m_faces:
        if not _srf_is_planar(X.m_surfaces[f.surface_index]):
            all_planar = False; break
    if all_planar:
        hs = []
        for f in X.m_faces:
            s = X.m_surfaces[f.surface_index]
            du = s.domain(0); dv = s.domain(1)
            q = s.point_at(0.5 * (du[0] + du[1]), 0.5 * (dv[0] + dv[1]))
            n = s.normal_at(0.5 * (du[0] + du[1]), 0.5 * (dv[0] + dv[1]))
            if n[0] * (C[0] - q[0]) + n[1] * (C[1] - q[1]) + n[2] * (C[2] - q[2]) > 0:
                n = (-n[0], -n[1], -n[2])
            hs.append((n[0], n[1], n[2], n[0] * q[0] + n[1] * q[1] + n[2] * q[2]))
        for vtx in spts:
            for h in hs:
                if h[0] * vtx[0] + h[1] * vtx[1] + h[2] * vtx[2] > h[3] + tol * 50:
                    return None  # not convex -> fall back to mesh
        return {'kind': 1, 'tol': tol, 'hs': hs}

    # Sphere: a single (non-planar) face. Least-squares centre fit, then verify equidistant.
    if len(X.m_faces) == 1:
        s = X.m_surfaces[X.m_faces[0].surface_index]
        du = s.domain(0); dv = s.domain(1)
        sp = []
        for i in range(1, 6):
            for j in range(1, 6):
                sp.append(s.point_at(du[0] + (du[1] - du[0]) * i / 6.0, dv[0] + (dv[1] - dv[0]) * j / 6.0))
        A = [[0.0] * 3 for _ in range(3)]; bb = [0.0, 0.0, 0.0]
        p0 = sp[0]; p0d = p0[0] * p0[0] + p0[1] * p0[1] + p0[2] * p0[2]
        for i in range(1, len(sp)):
            row = (2 * (sp[i][0] - p0[0]), 2 * (sp[i][1] - p0[1]), 2 * (sp[i][2] - p0[2]))
            rhs = sp[i][0] ** 2 + sp[i][1] ** 2 + sp[i][2] ** 2 - p0d
            for r in range(3):
                for c in range(3):
                    A[r][c] += row[r] * row[c]
                bb[r] += row[r] * rhs

        def det3(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        D = det3(A)
        if abs(D) > 1e-12:
            cc = []
            for k in range(3):
                M = [[bb[r] if c == k else A[r][c] for c in range(3)] for r in range(3)]
                cc.append(det3(M) / D)
            center = (cc[0], cc[1], cc[2])
            rs = sum(((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2 + (p[2] - center[2]) ** 2) ** 0.5 for p in sp)
            radius = rs / len(sp)
            ok = radius > tol
            for p in sp:
                d = ((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2 + (p[2] - center[2]) ** 2) ** 0.5
                if abs(d - radius) > diag * 1e-3:
                    ok = False; break
            if ok:
                return {'kind': 3, 'tol': tol, 'sc': center, 'sr': radius}

    # Cylinder: exactly 2 planar caps + 1 curved lateral.
    planar, curved = [], []
    for fi in range(len(X.m_faces)):
        (planar if _srf_is_planar(X.m_surfaces[X.m_faces[fi].surface_index]) else curved).append(fi)
    if len(planar) == 2 and len(curved) == 1:
        def cap_circle(fi):
            face = X.m_faces[fi]; s = X.m_surfaces[face.surface_index]
            bpts = []
            for li in face.loop_indices:
                if li < 0 or li >= len(X.m_loops) or X.m_loops[li].type != BRepLoopType.Outer:
                    continue
                for ti in X.m_loops[li].trim_indices:
                    if ti < 0 or ti >= len(X.m_trims):
                        continue
                    c2 = X.m_trims[ti].curve_2d_index
                    if c2 < 0 or c2 >= len(X.m_curves_2d):
                        continue
                    pc = X.m_curves_2d[c2]; dc = pc.domain()
                    for k in range(16):
                        uv = pc.point_at(dc[0] + (dc[1] - dc[0]) * k / 16.0)
                        bpts.append(s.point_at(uv[0], uv[1]))
            if len(bpts) < 6:
                return None
            mx = sum(p[0] for p in bpts) / len(bpts)
            my = sum(p[1] for p in bpts) / len(bpts)
            mz = sum(p[2] for p in bpts) / len(bpts)
            rs = sum(((p[0] - mx) ** 2 + (p[1] - my) ** 2 + (p[2] - mz) ** 2) ** 0.5 for p in bpts)
            radius = rs / len(bpts)
            for p in bpts:
                if abs(((p[0] - mx) ** 2 + (p[1] - my) ** 2 + (p[2] - mz) ** 2) ** 0.5 - radius) > diag * 1e-3:
                    return None
            return ((mx, my, mz), radius)
        c0 = cap_circle(planar[0]); c1 = cap_circle(planar[1])
        if c0 and c1 and abs(c0[1] - c1[1]) < diag * 1e-3:
            (a0, r0), (a1, _) = c0, c1
            h = ((a1[0] - a0[0]) ** 2 + (a1[1] - a0[1]) ** 2 + (a1[2] - a0[2]) ** 2) ** 0.5
            if h > tol:
                d = ((a1[0] - a0[0]) / h, (a1[1] - a0[1]) / h, (a1[2] - a0[2]) / h)
                L = X.m_surfaces[X.m_faces[curved[0]].surface_index]
                du = L.domain(0); dv = L.domain(1); ok = True
                for i in range(4):
                    for j in range(4):
                        p = L.point_at(du[0] + (du[1] - du[0]) * i / 3.0, dv[0] + (dv[1] - dv[0]) * j / 3.0)
                        w = (p[0] - a0[0], p[1] - a0[1], p[2] - a0[2])
                        t = w[0] * d[0] + w[1] * d[1] + w[2] * d[2]
                        rad = max(0.0, (w[0] ** 2 + w[1] ** 2 + w[2] ** 2) - t * t) ** 0.5
                        if abs(rad - r0) > diag * 1e-3 or t < -diag * 1e-3 or t > h + diag * 1e-3:
                            ok = False
                    if not ok:
                        break
                if ok:
                    return {'kind': 2, 'tol': tol, 'ca': a0, 'cd': d, 'ch': h, 'cr': r0}
    return None


def _inside_prim(prim, p):
    tol = prim['tol']
    k = prim['kind']
    if k == 1:
        for h in prim['hs']:
            if h[0] * p[0] + h[1] * p[1] + h[2] * p[2] > h[3] + tol:
                return False
        return True
    if k == 2:
        ca = prim['ca']; cd = prim['cd']
        w = (p[0] - ca[0], p[1] - ca[1], p[2] - ca[2])
        t = w[0] * cd[0] + w[1] * cd[1] + w[2] * cd[2]
        if t < -tol or t > prim['ch'] + tol:
            return False
        rad = max(0.0, (w[0] ** 2 + w[1] ** 2 + w[2] ** 2) - t * t) ** 0.5
        return rad <= prim['cr'] + tol
    if k == 3:
        sc = prim['sc']
        d = ((p[0] - sc[0]) ** 2 + (p[1] - sc[1]) ** 2 + (p[2] - sc[2]) ** 2) ** 0.5
        return d <= prim['sr'] + tol
    return False


def _aabb_overlap(a, b, m):
    for k in range(3):
        if a[0][k] - m > b[1][k] or b[0][k] - m > a[1][k]:
            return False
    return True


class BRepTrimType:
    Boundary = 0
    Mated = 1
    Seam = 2
    Singular = 3

    _TO_STR = {0: "boundary", 1: "mated", 2: "seam", 3: "singular"}
    _FROM_STR = {"boundary": 0, "mated": 1, "seam": 2, "singular": 3}

    @staticmethod
    def to_str(v):
        return BRepTrimType._TO_STR.get(v, "boundary")

    @staticmethod
    def from_str(s):
        return BRepTrimType._FROM_STR.get(s, 0)


class BRepLoopType:
    Outer = 0
    Inner = 1

    @staticmethod
    def to_str(v):
        return "inner" if v == 1 else "outer"

    @staticmethod
    def from_str(s):
        return 1 if s == "inner" else 0


class BRepVertex:
    def __init__(self):
        self.point_index = -1
        self.edge_indices = []


class BRepEdge:
    def __init__(self):
        self.curve_3d_index = -1
        self.start_vertex = -1
        self.end_vertex = -1
        self.trim_indices = []


class BRepTrim:
    def __init__(self):
        self.curve_2d_index = -1
        self.edge_index = -1
        self.loop_index = -1
        self.reversed = False
        self.type = BRepTrimType.Boundary


class BRepLoop:
    def __init__(self):
        self.trim_indices = []
        self.face_index = -1
        self.type = BRepLoopType.Outer


class BRepFace:
    def __init__(self):
        self.surface_index = -1
        self.loop_indices = []
        self.reversed = False
        self.facecolor = None


class BRep:
    def __init__(self):
        self._guid = None
        self.name = "my_brep"
        self.width = 1.0
        self._surfacecolor = None
        self._xform = None
        self.m_surfaces = []
        self.m_curves_3d = []
        self.m_curves_2d = []
        self.m_vertices = []
        self.m_topology_vertices = []
        self.m_topology_edges = []
        self.m_trims = []
        self.m_loops = []
        self.m_faces = []

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

    def refresh_guid(self):
        """Clear the guid so a FRESH one mints lazily on next read — the duplicate/copy enabler."""
        self._guid = None

    @property
    def surfacecolor(self):
        if self._surfacecolor is None:
            self._surfacecolor = Color.black()
        return self._surfacecolor

    @surfacecolor.setter
    def surfacecolor(self, value):
        self._surfacecolor = value

    @property
    def xform(self):
        if getattr(self, '_xform', None) is None:
            self._xform = Xform.identity()
        return self._xform

    @xform.setter
    def xform(self, value):
        self._xform = value

    def __str__(self):
        return f"BRep(name={self.name}, faces={self.face_count()}, edges={self.edge_count()}, vertices={self.vertex_count()})"

    def __repr__(self):
        return f"BRep(\n  name={self.name},\n  faces={self.face_count()},\n  edges={self.edge_count()},\n  vertices={self.vertex_count()},\n  solid={'true' if self.is_solid() else 'false'}\n)"

    def __eq__(self, other):
        if not isinstance(other, BRep):
            return False
        if self.name != other.name:
            return False
        if self.width != other.width:
            return False
        if self.surfacecolor != other.surfacecolor:
            return False
        if self.xform != other.xform:
            return False
        if len(self.m_faces) != len(other.m_faces):
            return False
        if len(self.m_surfaces) != len(other.m_surfaces):
            return False
        if len(self.m_topology_edges) != len(other.m_topology_edges):
            return False
        if len(self.m_vertices) != len(other.m_vertices):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def duplicate(self):
        b = BRep()
        b.guid = str(uuid.uuid4())
        b.name = self.name
        b.width = self.width
        b.surfacecolor = copy.deepcopy(self.surfacecolor)
        b.xform = copy.deepcopy(self.xform)
        b.m_surfaces = copy.deepcopy(self.m_surfaces)
        b.m_curves_3d = copy.deepcopy(self.m_curves_3d)
        b.m_curves_2d = copy.deepcopy(self.m_curves_2d)
        b.m_vertices = copy.deepcopy(self.m_vertices)
        b.m_topology_vertices = copy.deepcopy(self.m_topology_vertices)
        b.m_topology_edges = copy.deepcopy(self.m_topology_edges)
        b.m_trims = copy.deepcopy(self.m_trims)
        b.m_loops = copy.deepcopy(self.m_loops)
        b.m_faces = copy.deepcopy(self.m_faces)
        return b

    ###########################################################################
    # Accessors
    ###########################################################################

    def face_count(self):
        return len(self.m_faces)

    def edge_count(self):
        return len(self.m_topology_edges)

    def vertex_count(self):
        return len(self.m_vertices)

    def is_valid(self):
        if not self.m_faces or not self.m_surfaces or not self.m_vertices:
            return False
        for f in self.m_faces:
            if f.surface_index < 0 or f.surface_index >= len(self.m_surfaces):
                return False
        for l in self.m_loops:
            if l.face_index < 0 or l.face_index >= len(self.m_faces):
                return False
        for t in self.m_trims:
            if t.curve_2d_index < 0 or t.curve_2d_index >= len(self.m_curves_2d):
                return False
            if t.loop_index < 0 or t.loop_index >= len(self.m_loops):
                return False
        for e in self.m_topology_edges:
            if e.start_vertex < 0 or e.start_vertex >= len(self.m_topology_vertices):
                return False
            if e.end_vertex < 0 or e.end_vertex >= len(self.m_topology_vertices):
                return False
        return True

    def is_solid(self):
        if not self.m_topology_edges:
            return False
        diag = self._brep_bbox_diag()
        deg_tol = max(diag * 1e-7, 1e-12)
        for e in self.m_topology_edges:
            if len(e.trim_indices) == 2:
                continue
            # A DEGENERATE edge (3D curve collapsed to a point, e.g. a sphere/cone pole)
            # is watertight by construction — the pole is a single point fully enclosed by
            # its face — and OCCT excludes such degenerate edges from manifold checks. Skip
            # them; only genuine (non-zero-length) edges must be 2-trim.
            ci = e.curve_3d_index
            if 0 <= ci < len(self.m_curves_3d):
                c = self.m_curves_3d[ci]
                d0, d1 = c.domain()
                p0 = c.point_at(d0)
                ext = 0.0
                for k in range(1, 5):
                    pk = c.point_at(d0 + (d1 - d0) * k / 4.0)
                    ext = max(ext, ((pk[0]-p0[0])**2 + (pk[1]-p0[1])**2 + (pk[2]-p0[2])**2) ** 0.5)
                if ext < deg_tol:
                    continue
            return False
        return True

    def volume(self):
        """Volume via the divergence theorem: V = (1/3) sum_faces flux_outward. The per-face
        OUTWARD sign is determined GEOMETRICALLY (step off the face along its natural normal and
        test inside/outside), so it is independent of stored orientation flags (which differ
        between primitive constructors and the boolean rebuilder). Planar faces use the exact
        chained boundary integral; curved faces use trim-masked composite Gauss of S.(Su x Sv).
        Matches OCCT BRepGProp to machine precision for planar + rectangular-trim curved solids;
        accurate (mesh/grid resolution) for arbitrary curved trims (e.g. sphere caps)."""
        import math
        import numpy as np
        from .point import Point
        from .vector import Vector
        from .polyline import Polyline
        from .remesh_cdt import RemeshCDT
        GN = (-0.9061798459386640, -0.5384693101056831, 0.0,
              0.5384693101056831, 0.9061798459386640)
        GW = (0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
              0.4786286704993665, 0.2369268850561891)

        def cross(a, b):
            return Vector(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

        def is_planar(s):
            u0, u1 = s.domain(0); v0, v1 = s.domain(1)
            n0 = s.normal_at(u0+(u1-u0)*0.5, v0+(v1-v0)*0.5)
            for uu, vv in ((0.25, 0.3), (0.5, 0.6), (0.75, 0.8)):
                if cross(n0, s.normal_at(u0+(u1-u0)*uu, v0+(v1-v0)*vv)).magnitude() > 1e-7:
                    return False
            return True

        def loop_vector_area(srf, loop):
            # 1/2 * closed integral C x C', chaining the loop's pcurves head-to-tail by matching
            # UV endpoints (orientation-independent of the stored reversed flags).
            segs = []
            for ti in loop.trim_indices:
                if ti < 0 or ti >= len(self.m_trims):
                    continue
                c2 = self.m_trims[ti].curve_2d_index
                if c2 < 0 or c2 >= len(self.m_curves_2d):
                    continue
                pc = self.m_curves_2d[c2]
                d0, d1 = pc.domain()
                segs.append((pc, d0, d1, pc.point_at(d0), pc.point_at(d1)))
            if not segs:
                return Vector(0, 0, 0)

            def d2(a, b):
                return (a[0]-b[0])**2 + (a[1]-b[1])**2
            order = [(0, True)]
            used = [False]*len(segs); used[0] = True
            tail = segs[0][4]
            for _ in range(1, len(segs)):
                best, fwd, bd = -1, True, 1e300
                for j in range(len(segs)):
                    if used[j]:
                        continue
                    ds, de = d2(segs[j][3], tail), d2(segs[j][4], tail)
                    if ds < bd: bd, best, fwd = ds, j, True
                    if de < bd: bd, best, fwd = de, j, False
                if best < 0:
                    break
                used[best] = True; order.append((best, fwd))
                tail = segs[best][4] if fwd else segs[best][3]
            ax = ay = az = 0.0
            NS = 24
            for (idx, fwd) in order:
                pc, t0, t1, _ps, _pe = segs[idx]
                for s in range(NS):
                    a = t0 + (t1-t0)*s/NS; b = t0 + (t1-t0)*(s+1)/NS
                    mid = 0.5*(a+b); half = 0.5*(b-a)
                    for g in range(5):
                        t = mid + half*GN[g]
                        pe = pc.evaluate(t, 1)
                        if len(pe) < 2:
                            continue
                        uv = pe[0]; duv = pe[1]
                        if not fwd:
                            duv = Vector(-duv[0], -duv[1], -duv[2])
                        se = srf.evaluate(uv[0], uv[1], 1)
                        if len(se) < 3:
                            continue
                        S, Sv, Su = se[0], se[1], se[2]
                        Cp = Vector(Su[0]*duv[0]+Sv[0]*duv[1], Su[1]*duv[0]+Sv[1]*duv[1],
                                    Su[2]*duv[0]+Sv[2]*duv[1])
                        cr = cross(S, Cp)
                        w = GW[g]*half
                        ax += w*cr[0]; ay += w*cr[1]; az += w*cr[2]
            return Vector(0.5*ax, 0.5*ay, 0.5*az)

        def face_interior(face, srf):
            outers, inners = [], []
            for li in face.loop_indices:
                if li < 0 or li >= len(self.m_loops):
                    continue
                pts = []
                for ti in self.m_loops[li].trim_indices:
                    if ti < 0 or ti >= len(self.m_trims):
                        continue
                    c2 = self.m_trims[ti].curve_2d_index
                    if c2 < 0 or c2 >= len(self.m_curves_2d):
                        continue
                    pc = self.m_curves_2d[c2]
                    d0, d1 = pc.domain()
                    n = max(pc.cv_count()*3, 6)
                    for i in range(n):
                        uv = pc.point_at(d0 + (d1-d0)*i/n)
                        pts.append(Point(uv[0], uv[1], 0))
                if len(pts) < 3:
                    continue
                (outers if self.m_loops[li].type == BRepLoopType.Outer else inners).append(Polyline(pts))
            u0, u1 = srf.domain(0); v0, v1 = srf.domain(1)

            def pip(uu, vv, poly):  # point-in-polygon (ray cast) over a Polyline's points
                pp = poly.get_points()
                inside = False; j = len(pp) - 1
                for i in range(len(pp)):
                    if (pp[i][1] > vv) != (pp[j][1] > vv) and \
                            uu < (pp[j][0]-pp[i][0])*(vv-pp[i][1])/(pp[j][1]-pp[i][1] or 1e-300) + pp[i][0]:
                        inside = not inside
                    j = i
                return inside

            def in_material(uu, vv):
                if outers and not any(pip(uu, vv, op) for op in outers):
                    return False
                return not any(pip(uu, vv, ip) for ip in inners)

            cu, cv = 0.5*(u0+u1), 0.5*(v0+v1)
            if outers:
                allp = [outers[0]] + inners
                try:
                    tris = RemeshCDT.triangulate(allp)
                except Exception:
                    tris = []
                flat = []
                for pl in allp:
                    flat.extend(pl.get_points())
                # Largest-area triangle whose centroid is on the face MATERIAL (inside the
                # outer loop, outside every hole). The first triangle / domain centre can land
                # in a hole (e.g. an annulus' centre) -> a wrong outward-sign probe -> wrong
                # flux. The largest valid triangle's centroid is robustly interior.
                best_a = -1.0
                for t in tris:
                    if t[0] >= len(flat) or t[1] >= len(flat) or t[2] >= len(flat):
                        continue
                    a, b, c = flat[t[0]], flat[t[1]], flat[t[2]]
                    tcu = (a[0]+b[0]+c[0])/3.0; tcv = (a[1]+b[1]+c[1])/3.0
                    ar = abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                    if ar > best_a and in_material(tcu, tcv):
                        best_a = ar; cu, cv = tcu, tcv
                if best_a < 0.0:
                    # CDT gave nothing usable -> grid-sample for any in-material point.
                    for iu in range(1, 12):
                        for iv in range(1, 12):
                            su = u0 + (u1-u0)*iu/12.0; sv = v0 + (v1-v0)*iv/12.0
                            if in_material(su, sv):
                                cu, cv = su, sv; best_a = 0.0; break
                        if best_a >= 0.0:
                            break
            return srf.point_at(cu, cv), srf.normal_at(cu, cv)

        def sphere_flux(face, srf):
            # Analytic sphere boundary-integral flux (exact, ~300x fewer surface evals than the
            # masked Gauss; used for sphere cap-cut faces -- a sphere minus circular caps).
            # flux_nat = integral P.(Su x Sv) du dv with P = C + R*er (er radial), Su x Sv || er:
            #   = C.(vector area) - R^2 * closed_integral(h dtheta), where h = (P-C).Zs and
            #   theta = atan2((P-C).Ys, (P-C).Xs) (GEOMETRIC longitude, seam-independent). So
            #   flux_nat = sum_loops [ C.A_loop - R^2 * H_loop ] -- depends ONLY on the boundary
            #   curve's 3D geometry, sidestepping the masked-Gauss region/staircase error.
            # Returns (have_analytic, flux_analytic). Recognizes the sphere from the surface itself
            # (poles -> center/axis/radius + grid verify); have_analytic stays False otherwise.
            su0, su1 = srf.domain(0); sv0, sv1 = srf.domain(1)
            umid = 0.5*(su0+su1)
            Ps = srf.point_at(umid, sv0)
            Pn = srf.point_at(umid, sv1)
            Pm = srf.point_at(umid, 0.5*(sv0+sv1))
            axis = Vector(Pn[0]-Ps[0], Pn[1]-Ps[1], Pn[2]-Ps[2])
            R = 0.5*axis.magnitude()
            if R <= 1e-9:
                return False, 0.0, False
            Cc = Point(0.5*(Ps[0]+Pn[0]), 0.5*(Ps[1]+Pn[1]), 0.5*(Ps[2]+Pn[2]))
            Zs = Vector(axis[0]/(2*R), axis[1]/(2*R), axis[2]/(2*R))
            # verify sphere: sample grid, |P-C| ~ R
            for i in range(4):
                for j in range(4):
                    p = srf.point_at(su0+(su1-su0)*i/3.0, sv0+(sv1-sv0)*j/3.0)
                    rr = math.sqrt((p[0]-Cc[0])**2 + (p[1]-Cc[1])**2 + (p[2]-Cc[2])**2)
                    if abs(rr - R) > R*1e-4 + 1e-6:
                        return False, 0.0, False
            dm = Vector(Pm[0]-Cc[0], Pm[1]-Cc[1], Pm[2]-Cc[2])
            dz = dm[0]*Zs[0]+dm[1]*Zs[1]+dm[2]*Zs[2]
            Xs = Vector(dm[0]-dz*Zs[0], dm[1]-dz*Zs[1], dm[2]-dz*Zs[2])
            xn = Xs.magnitude()
            if xn < 1e-12:
                return False, 0.0, False
            Xs = Vector(Xs[0]/xn, Xs[1]/xn, Xs[2]/xn)
            Ys = cross(Zs, Xs)
            Cv = np.array([Cc[0], Cc[1], Cc[2]])
            Xv = np.array([Xs[0], Xs[1], Xs[2]])
            Yv = np.array([Ys[0], Ys[1], Ys[2]])
            Zv = np.array([Zs[0], Zs[1], Zs[2]])
            flux_analytic = 0.0
            pole_face = False
            for li in face.loop_indices:
                if li < 0 or li >= len(self.m_loops):
                    continue
                # chained, ordered 3D polyline of the loop (greedy walk, like loop_vector_area)
                segs = []
                for ti in self.m_loops[li].trim_indices:
                    if ti < 0 or ti >= len(self.m_trims):
                        continue
                    c2 = self.m_trims[ti].curve_2d_index
                    if c2 < 0 or c2 >= len(self.m_curves_2d):
                        continue
                    pc = self.m_curves_2d[c2]
                    d0, d1 = pc.domain()
                    segs.append((pc, d0, d1, pc.point_at(d0), pc.point_at(d1)))
                if not segs:
                    continue

                def d2(a, b):
                    return (a[0]-b[0])**2 + (a[1]-b[1])**2
                order = [(0, True)]
                used = [False]*len(segs); used[0] = True
                tail = segs[0][4]
                for _ in range(1, len(segs)):
                    best, fwd, bd = -1, True, 1e300
                    for j in range(len(segs)):
                        if used[j]:
                            continue
                        ds, de = d2(segs[j][3], tail), d2(segs[j][4], tail)
                        if ds < bd: bd, best, fwd = ds, j, True
                        if de < bd: bd, best, fwd = de, j, False
                    if best < 0:
                        break
                    used[best] = True; order.append((best, fwd))
                    tail = segs[best][4] if fwd else segs[best][3]
                # sample ordered UV polyline densely (boundary integral converges to the pcurve's
                # enclosed flux; residual is the pullback pcurve's CV density). Batch the curve +
                # surface evals -- per-point NURBS evals dominate the cost.
                us_list = []
                vs_list = []
                for (idx, fwd) in order:
                    pc, ct0, ct1, _ps, _pe = segs[idx]
                    ns = 200
                    ss = np.arange(ns, dtype=np.float64)
                    ts = (ct0 + (ct1-ct0)*ss/ns) if fwd else (ct1 - (ct1-ct0)*ss/ns)
                    cpts = pc._batch_point_at(ts)
                    us_list.append(cpts[:, 0]); vs_list.append(cpts[:, 1])
                if not us_list:
                    continue
                uu = np.concatenate(us_list); vv = np.concatenate(vs_list)
                if uu.shape[0] < 3:
                    continue
                uu = np.append(uu, uu[0]); vv = np.append(vv, vv[0])  # close it
                uvArea = 0.5*float(np.sum(uu[:-1]*vv[1:] - uu[1:]*vv[:-1]))
                P = srf.batch_point_at(uu, vv)  # (m, 3)
                dd = P - Cv
                xv = dd @ Xv; yv = dd @ Yv; hv = dd @ Zv
                theta = np.arctan2(yv, xv)
                # vector area A = 0.5 * sum P_i x P_{i+1}
                Pa, Pb = P[:-1], P[1:]
                Ax = 0.5*float(np.sum(Pa[:, 1]*Pb[:, 2] - Pa[:, 2]*Pb[:, 1]))
                Ay = 0.5*float(np.sum(Pa[:, 2]*Pb[:, 0] - Pa[:, 0]*Pb[:, 2]))
                Az = 0.5*float(np.sum(Pa[:, 0]*Pb[:, 1] - Pa[:, 1]*Pb[:, 0]))
                # H = sum 0.5*(h_i + h_{i-1}) * dtheta, dtheta wrapped to (-pi, pi]
                dth = np.diff(theta)
                dth = (dth + math.pi) % (2*math.pi) - math.pi
                H = float(np.sum(0.5*(hv[1:] + hv[:-1])*dth))
                # Net longitude winding of this loop (sum of the SAME wrapped per-segment dtheta
                # used for H). A pole-encircling loop winds ~+-2pi, injecting a spurious +-2pi*R^2
                # pole residue into the Green's reduction -- detect and skip the analytic override.
                wind = float(np.sum(dth))
                loopflux = (Cc[0]*Ax + Cc[1]*Ay + Cc[2]*Az) - R*R*H
                osign = (1.0 if self.m_loops[li].type == BRepLoopType.Outer else -1.0) \
                    * (1.0 if uvArea >= 0 else -1.0)
                flux_analytic += osign*loopflux
                if abs(wind) > math.pi:
                    pole_face = True
            return True, flux_analytic, pole_face

        bmesh = self.mesh()
        diag = self._brep_bbox_diag()
        eps = diag * 1e-3

        total = 0.0
        for face in self.m_faces:
            si = face.surface_index
            if si < 0 or si >= len(self.m_surfaces):
                continue
            srf = self.m_surfaces[si]
            P3, Nnat = face_interior(face, srf)
            if Nnat.magnitude() < 1e-12:
                continue
            probe = Point(P3[0]+eps*Nnat[0], P3[1]+eps*Nnat[1], P3[2]+eps*Nnat[2])
            sign = -1.0 if self.contains_point(probe, bmesh) else 1.0

            curved_rect = False
            have_analytic = False
            pole_face = False
            if is_planar(srf):
                area = 0.0
                for li in face.loop_indices:
                    if li < 0 or li >= len(self.m_loops):
                        continue
                    la = loop_vector_area(srf, self.m_loops[li]).magnitude()
                    area += la if self.m_loops[li].type == BRepLoopType.Outer else -la
                qn = P3[0]*Nnat[0] + P3[1]*Nnat[1] + P3[2]*Nnat[2]
                flux_nat = qn * area
            else:
                umin = vmin = 1e300; umax = vmax = -1e300
                outer_polys, inner_polys = [], []
                for li in face.loop_indices:
                    if li < 0 or li >= len(self.m_loops):
                        continue
                    poly = []
                    for ti in self.m_loops[li].trim_indices:
                        if ti < 0 or ti >= len(self.m_trims):
                            continue
                        c2 = self.m_trims[ti].curve_2d_index
                        if c2 < 0 or c2 >= len(self.m_curves_2d):
                            continue
                        pc = self.m_curves_2d[c2]
                        d0, d1 = pc.domain()
                        n = max(pc.cv_count()*4, 12)
                        for i in range(n):
                            uv = pc.point_at(d0 + (d1-d0)*i/n)
                            umin = min(umin, uv[0]); umax = max(umax, uv[0])
                            vmin = min(vmin, uv[1]); vmax = max(vmax, uv[1])
                            poly.append((uv[0], uv[1]))
                    if len(poly) >= 3:
                        (outer_polys if self.m_loops[li].type == BRepLoopType.Outer else inner_polys).append(poly)
                u0, u1 = srf.domain(0); v0, v1 = srf.domain(1)
                if umax <= umin or vmax <= vmin:
                    umin, umax, vmin, vmax = u0, u1, v0, v1
                # A RECTANGULAR trim (cylinder band, full sphere) needs only NU=24 -- every Gauss
                # point is inside, so the quadrature is exact. A NON-rectangular trim (sphere caps,
                # a band with circular holes) has a curved mask boundary whose staircase error scales
                # ~1/NU. Sphere cap-cut faces are handled exactly below by the analytic boundary
                # integral; this Gauss is the fallback for other curved faces.
                rect_trim = (not inner_polys
                             and abs(umin-u0) < (u1-u0)*1e-3 and abs(umax-u1) < (u1-u0)*1e-3
                             and abs(vmin-v0) < (v1-v0)*1e-3 and abs(vmax-v1) < (v1-v0)*1e-3)
                curved_rect = rect_trim

                # A sphere cap-cut face (non-rectangular trim) is integrated EXACTLY by the analytic
                # boundary integral below; compute it up-front so the (discarded) masked Gauss is
                # skipped (NU=0). Rectangular trims (full sphere / equatorial band) and non-sphere
                # curved faces keep the masked Gauss.
                have_analytic, flux_analytic, pole_face = (False, 0.0, False)
                if not curved_rect:
                    have_analytic, flux_analytic, pole_face = sphere_flux(face, srf)

                def in_poly(u, v, p):
                    inside = False
                    j = len(p) - 1
                    for i in range(len(p)):
                        if (p[i][1] > v) != (p[j][1] > v) and \
                                u < (p[j][0]-p[i][0])*(v-p[i][1])/(p[j][1]-p[i][1]) + p[i][0]:
                            inside = not inside
                        j = i
                    return inside

                def in_trim(u, v):
                    ok = not outer_polys
                    for op in outer_polys:
                        if in_poly(u, v, op):
                            ok = True; break
                    if not ok:
                        return False
                    for ip in inner_polys:
                        if in_poly(u, v, ip):
                            return False
                    return True

                # Skip the (discarded) masked Gauss only when the analytic override will be used.
                # A pole-encircling sphere face falls BACK to the masked Gauss, so it must run here.
                NU = NV = (0 if (have_analytic and not pole_face) else 24)
                flux_nat = 0.0
                for iu in range(NU):
                    ua = umin+(umax-umin)*iu/NU; ub = umin+(umax-umin)*(iu+1)/NU
                    um = 0.5*(ua+ub); uh = 0.5*(ub-ua)
                    for iv in range(NV):
                        va_ = vmin+(vmax-vmin)*iv/NV; vb = vmin+(vmax-vmin)*(iv+1)/NV
                        vm = 0.5*(va_+vb); vh = 0.5*(vb-va_)
                        for gu in range(5):
                            u = um + uh*GN[gu]
                            for gv in range(5):
                                v = vm + vh*GN[gv]
                                if not in_trim(u, v):
                                    continue
                                d = srf.evaluate(u, v, 1)
                                if len(d) < 3:
                                    continue
                                nrm = cross(d[2], d[1])
                                integ = d[0][0]*nrm[0] + d[0][1]*nrm[1] + d[0][2]*nrm[2]
                                flux_nat += GW[gu]*GW[gv]*uh*vh*integ

            # The analytic boundary integral (computed above for non-rect sphere caps) is exact;
            # use it in place of the masked Gauss there. Gauss stands for rectangular-trim and
            # non-sphere curved faces (where it is already exact / the only available method).
            if have_analytic and not curved_rect and not pole_face:
                flux_nat = flux_analytic
            total += sign * flux_nat
        return abs(total) / 3.0

    def contains_point(self, p, boundary=None):
        """True if `p` is strictly inside the (closed) solid, by ray-cast parity against
        the tessellated boundary. Robust for interior points; matches OCCT
        BRepClass3d_SolidClassifier for IN/OUT. Pass a precomputed `boundary` mesh to
        avoid re-tessellation when classifying many points.
        """
        from .line import Line
        from . import intersection
        if boundary is None:
            boundary = self.mesh()
        dx, dy, dz = 0.5773502691, 0.6539124, 0.5023147  # irregular, ~unit
        big = 1e6
        ray = Line(p[0], p[1], p[2], p[0] + dx*big, p[1] + dy*big, p[2] + dz*big)
        hits = intersection.ray_mesh(ray, boundary, 1e-9, True)
        if not hits:
            return False
        return (len(hits) % 2) == 1

    ###########################################################################
    # Building
    ###########################################################################

    def add_surface(self, srf):
        self.m_surfaces.append(srf)
        return len(self.m_surfaces) - 1

    def add_curve_3d(self, crv):
        self.m_curves_3d.append(crv)
        return len(self.m_curves_3d) - 1

    def add_curve_2d(self, crv):
        self.m_curves_2d.append(crv)
        return len(self.m_curves_2d) - 1

    def add_vertex(self, pt):
        self.m_vertices.append(pt)
        return len(self.m_vertices) - 1

    def add_edge(self, curve_3d_idx, start_vertex, end_vertex):
        e = BRepEdge()
        e.curve_3d_index = curve_3d_idx
        e.start_vertex = start_vertex
        e.end_vertex = end_vertex
        self.m_topology_edges.append(e)
        return len(self.m_topology_edges) - 1

    def add_trim(self, curve_2d_idx, edge_idx, loop_idx, reversed, trim_type):
        t = BRepTrim()
        t.curve_2d_index = curve_2d_idx
        t.edge_index = edge_idx
        t.loop_index = loop_idx
        t.reversed = reversed
        t.type = trim_type
        idx = len(self.m_trims)
        self.m_trims.append(t)
        if 0 <= loop_idx < len(self.m_loops):
            self.m_loops[loop_idx].trim_indices.append(idx)
        if 0 <= edge_idx < len(self.m_topology_edges):
            self.m_topology_edges[edge_idx].trim_indices.append(idx)
        return idx

    def add_loop(self, face_idx, loop_type):
        l = BRepLoop()
        l.face_index = face_idx
        l.type = loop_type
        idx = len(self.m_loops)
        self.m_loops.append(l)
        if 0 <= face_idx < len(self.m_faces):
            self.m_faces[face_idx].loop_indices.append(idx)
        return idx

    def add_face(self, surface_idx, reversed):
        f = BRepFace()
        f.surface_index = surface_idx
        f.reversed = reversed
        self.m_faces.append(f)
        return len(self.m_faces) - 1

    ###########################################################################
    # Factory
    ###########################################################################

    @staticmethod
    def create_box(sx, sy, sz):
        brep = BRep()
        brep.name = "box"
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        corners = [
            Point(-hx, -hy, -hz), Point(hx, -hy, -hz),
            Point(hx, hy, -hz), Point(-hx, hy, -hz),
            Point(-hx, -hy, hz), Point(hx, -hy, hz),
            Point(hx, hy, hz), Point(-hx, hy, hz),
        ]
        for c in corners:
            brep.add_vertex(c)
        face_verts = [
            [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
            [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
        ]
        edge_verts = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7],
        ]
        for ev in edge_verts:
            crv = NurbsCurve.create(False, 1, [corners[ev[0]], corners[ev[1]]])
            brep.add_curve_3d(crv)
        for i in range(8):
            tv = BRepVertex()
            tv.point_index = i
            brep.m_topology_vertices.append(tv)
        for i, ev in enumerate(edge_verts):
            brep.add_edge(i, ev[0], ev[1])
        uv_corners = [
            Point(0, 0, 0),
            Point(1, 0, 0),
            Point(1, 1, 0),
            Point(0, 1, 0),
        ]
        for fi in range(6):
            fv = face_verts[fi]
            p00, p10 = corners[fv[0]], corners[fv[1]]
            p01, p11 = corners[fv[3]], corners[fv[2]]
            srf = NurbsSurface(3, False, 2, 2, 2, 2)
            srf.set_cv(0, 0, p00); srf.set_cv(1, 0, p10)
            srf.set_cv(0, 1, p01); srf.set_cv(1, 1, p11)
            si = brep.add_surface(srf)
            face_idx = brep.add_face(si, False)
            loop_idx = brep.add_loop(face_idx, BRepLoopType.Outer)
            for ei in range(4):
                nxt = (ei + 1) % 4
                trim_crv = NurbsCurve.create(False, 1, [uv_corners[ei], uv_corners[nxt]])
                c2d_idx = brep.add_curve_2d(trim_crv)
                edge_idx = -1
                for e_i, ev in enumerate(edge_verts):
                    if (ev[0] == fv[ei] and ev[1] == fv[nxt]) or (ev[0] == fv[nxt] and ev[1] == fv[ei]):
                        edge_idx = e_i
                        break
                rev = edge_verts[edge_idx][0] != fv[ei] if edge_idx >= 0 else False
                brep.add_trim(c2d_idx, edge_idx, loop_idx, rev, BRepTrimType.Mated)
        for ei, e in enumerate(brep.m_topology_edges):
            brep.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            brep.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return brep

    @staticmethod
    def create_cylinder(radius, height):
        from .primitives import Primitives
        brep = BRep()
        brep.name = "cylinder"
        body = Primitives.cylinder_surface(0, 0, 0, radius, height)
        dom_u = body.domain(0)
        dom_v = body.domain(1)
        p_bot = body.point_at(dom_u[0], dom_v[0])
        p_top = body.point_at(dom_u[0], dom_v[1])
        vi_bot = brep.add_vertex(p_bot)
        vi_top = brep.add_vertex(p_top)
        tv0 = BRepVertex(); tv0.point_index = vi_bot
        tv1 = BRepVertex(); tv1.point_index = vi_top
        brep.m_topology_vertices.extend([tv0, tv1])
        circle_bot = Primitives.circle(0, 0, 0, radius)
        circle_top = Primitives.circle(0, 0, height, radius)
        seam_line = NurbsCurve.create(False, 1, [p_bot, p_top])
        ci_bot = brep.add_curve_3d(circle_bot)
        ci_top = brep.add_curve_3d(circle_top)
        ci_seam = brep.add_curve_3d(seam_line)
        ei_bot = brep.add_edge(ci_bot, 0, 0)
        ei_top = brep.add_edge(ci_top, 1, 1)
        ei_seam = brep.add_edge(ci_seam, 0, 1)
        si_body = brep.add_surface(body)
        cap_bot = NurbsSurface(3, False, 2, 2, 2, 2)
        cap_bot.set_cv(0, 0, Point(-radius, -radius, 0))
        cap_bot.set_cv(1, 0, Point(radius, -radius, 0))
        cap_bot.set_cv(0, 1, Point(-radius, radius, 0))
        cap_bot.set_cv(1, 1, Point(radius, radius, 0))
        si_bot = brep.add_surface(cap_bot)
        cap_top = NurbsSurface(3, False, 2, 2, 2, 2)
        cap_top.set_cv(0, 0, Point(-radius, -radius, height))
        cap_top.set_cv(1, 0, Point(radius, -radius, height))
        cap_top.set_cv(0, 1, Point(-radius, radius, height))
        cap_top.set_cv(1, 1, Point(radius, radius, height))
        si_top = brep.add_surface(cap_top)
        fi_body = brep.add_face(si_body, False)
        li_body = brep.add_loop(fi_body, BRepLoopType.Outer)
        c2d_bot = NurbsCurve.create(False, 1, [Point(dom_u[0], dom_v[0], 0), Point(dom_u[1], dom_v[0], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_bot), ei_bot, li_body, False, BRepTrimType.Mated)
        c2d_sr = NurbsCurve.create(False, 1, [Point(dom_u[1], dom_v[0], 0), Point(dom_u[1], dom_v[1], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_sr), ei_seam, li_body, False, BRepTrimType.Seam)
        c2d_top = NurbsCurve.create(False, 1, [Point(dom_u[1], dom_v[1], 0), Point(dom_u[0], dom_v[1], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_top), ei_top, li_body, True, BRepTrimType.Mated)
        c2d_sl = NurbsCurve.create(False, 1, [Point(dom_u[0], dom_v[1], 0), Point(dom_u[0], dom_v[0], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_sl), ei_seam, li_body, True, BRepTrimType.Seam)
        # Circular 2D trim in UV space: circle at (0.5,0.5) radius 0.5
        import math as _m
        _w = _m.sqrt(2.0) / 2.0
        _cx = [1, 1, 0, -1, -1, -1, 0, 1, 1]; _cy = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        _cw = [1, _w, 1, _w, 1, _w, 1, _w, 1]; _kn = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        def _cap_circle():
            import numpy as np
            c = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=9)
            c.m_nurbsknot = np.array(_kn, dtype=np.float64)
            c.m_cv = np.zeros(9 * 4, dtype=np.float64)
            for i in range(9):
                c.set_cv_4d(i, (0.5+0.5*_cx[i])*_cw[i], (0.5+0.5*_cy[i])*_cw[i], 0.0, _cw[i])
            return c
        fi_bot = brep.add_face(si_bot, True)
        li_bot = brep.add_loop(fi_bot, BRepLoopType.Outer)
        brep.add_trim(brep.add_curve_2d(_cap_circle()), ei_bot, li_bot, True, BRepTrimType.Mated)
        fi_top = brep.add_face(si_top, False)
        li_top = brep.add_loop(fi_top, BRepLoopType.Outer)
        brep.add_trim(brep.add_curve_2d(_cap_circle()), ei_top, li_top, False, BRepTrimType.Mated)
        for ei, e in enumerate(brep.m_topology_edges):
            brep.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            brep.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return brep

    @staticmethod
    def create_sphere(radius):
        from .primitives import Primitives
        brep = BRep()
        brep.name = "sphere"
        srf = Primitives.sphere_surface(0, 0, 0, radius)
        dom_u = srf.domain(0)
        dom_v = srf.domain(1)
        p_south = Point(0, 0, -radius)
        p_north = Point(0, 0, radius)
        vi_south = brep.add_vertex(p_south)
        vi_north = brep.add_vertex(p_north)
        tv0 = BRepVertex(); tv0.point_index = vi_south
        tv1 = BRepVertex(); tv1.point_index = vi_north
        brep.m_topology_vertices.extend([tv0, tv1])
        seam_crv = NurbsCurve.create(False, 1, [p_south, p_north])
        ci_seam = brep.add_curve_3d(seam_crv)
        ei_seam = brep.add_edge(ci_seam, 0, 1)
        si = brep.add_surface(srf)
        fi = brep.add_face(si, False)
        li = brep.add_loop(fi, BRepLoopType.Outer)
        c2d_south = NurbsCurve.create(False, 1, [Point(dom_u[0], dom_v[0], 0), Point(dom_u[1], dom_v[0], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_south), -1, li, False, BRepTrimType.Singular)
        c2d_sr = NurbsCurve.create(False, 1, [Point(dom_u[1], dom_v[0], 0), Point(dom_u[1], dom_v[1], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_sr), ei_seam, li, False, BRepTrimType.Seam)
        c2d_north = NurbsCurve.create(False, 1, [Point(dom_u[1], dom_v[1], 0), Point(dom_u[0], dom_v[1], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_north), -1, li, False, BRepTrimType.Singular)
        c2d_sl = NurbsCurve.create(False, 1, [Point(dom_u[0], dom_v[1], 0), Point(dom_u[0], dom_v[0], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_sl), ei_seam, li, True, BRepTrimType.Seam)
        for ei, e in enumerate(brep.m_topology_edges):
            brep.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            brep.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return brep

    @staticmethod
    def create_cone(radius, height):
        # Side face = cone_surface (u in [0,4] = circle, v in [0,1] = base->apex; v=1 is a SINGULAR
        # apex pole, like a sphere pole) + one planar base cap. Mirrors create_cylinder's base+seam
        # but with a singular apex instead of a top cap.
        from .primitives import Primitives
        import math as _m
        import numpy as np
        brep = BRep()
        brep.name = "cone"
        body = Primitives.cone_surface(0, 0, 0, radius, height)
        dom_u = body.domain(0)
        dom_v = body.domain(1)
        p_base = body.point_at(dom_u[0], dom_v[0])   # on base circle at u=0
        p_apex = Point(0, 0, height)
        vi_base = brep.add_vertex(p_base)
        vi_apex = brep.add_vertex(p_apex)
        tv0 = BRepVertex(); tv0.point_index = vi_base
        tv1 = BRepVertex(); tv1.point_index = vi_apex
        brep.m_topology_vertices.extend([tv0, tv1])
        circle_base = Primitives.circle(0, 0, 0, radius)
        seam_line = NurbsCurve.create(False, 1, [p_base, p_apex])
        ci_base = brep.add_curve_3d(circle_base)
        ci_seam = brep.add_curve_3d(seam_line)
        ei_base = brep.add_edge(ci_base, vi_base, vi_base)
        ei_seam = brep.add_edge(ci_seam, vi_base, vi_apex)
        si_body = brep.add_surface(body)
        cap_base = NurbsSurface(3, False, 2, 2, 2, 2)
        cap_base.set_cv(0, 0, Point(-radius, -radius, 0))
        cap_base.set_cv(1, 0, Point(radius, -radius, 0))
        cap_base.set_cv(0, 1, Point(-radius, radius, 0))
        cap_base.set_cv(1, 1, Point(radius, radius, 0))
        si_base = brep.add_surface(cap_base)
        fi_body = brep.add_face(si_body, False)
        li_body = brep.add_loop(fi_body, BRepLoopType.Outer)
        c2d_base = NurbsCurve.create(False, 1, [Point(dom_u[0], dom_v[0], 0), Point(dom_u[1], dom_v[0], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_base), ei_base, li_body, False, BRepTrimType.Mated)
        c2d_sr = NurbsCurve.create(False, 1, [Point(dom_u[1], dom_v[0], 0), Point(dom_u[1], dom_v[1], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_sr), ei_seam, li_body, False, BRepTrimType.Seam)
        c2d_apex = NurbsCurve.create(False, 1, [Point(dom_u[1], dom_v[1], 0), Point(dom_u[0], dom_v[1], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_apex), -1, li_body, False, BRepTrimType.Singular)
        c2d_sl = NurbsCurve.create(False, 1, [Point(dom_u[0], dom_v[1], 0), Point(dom_u[0], dom_v[0], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_sl), ei_seam, li_body, True, BRepTrimType.Seam)
        _w = _m.sqrt(2.0) / 2.0
        _cx = [1, 1, 0, -1, -1, -1, 0, 1, 1]; _cy = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        _cw = [1, _w, 1, _w, 1, _w, 1, _w, 1]; _kn = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        def _cap_circle():
            c = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=9)
            c.m_nurbsknot = np.array(_kn, dtype=np.float64)
            c.m_cv = np.zeros(9 * 4, dtype=np.float64)
            for i in range(9):
                c.set_cv_4d(i, (0.5+0.5*_cx[i])*_cw[i], (0.5+0.5*_cy[i])*_cw[i], 0.0, _cw[i])
            return c
        fi_base = brep.add_face(si_base, True)
        li_base = brep.add_loop(fi_base, BRepLoopType.Outer)
        brep.add_trim(brep.add_curve_2d(_cap_circle()), ei_base, li_base, True, BRepTrimType.Mated)
        for ei, e in enumerate(brep.m_topology_edges):
            brep.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            brep.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return brep

    @staticmethod
    def create_torus(major_radius, minor_radius):
        # Torus: a single closed face, periodic in BOTH u (major circle) and v (minor circle). No
        # caps, no poles -- two seams: the u-seam (minor circle at u=0) and the v-seam (outer major
        # circle at v=0), meeting at one corner vertex. The loop is the UV rectangle [0,4]x[0,4] = 4
        # Seam trims.
        from .primitives import Primitives
        brep = BRep()
        brep.name = "torus"
        body = Primitives.torus_surface(0, 0, 0, major_radius, minor_radius)
        dom_u = body.domain(0)
        dom_v = body.domain(1)
        p_corner = body.point_at(dom_u[0], dom_v[0])   # (major+minor, 0, 0)
        vi = brep.add_vertex(p_corner)
        tv = BRepVertex(); tv.point_index = vi
        brep.m_topology_vertices.append(tv)

        # u-seam: v -> point_at(u0, v), the minor circle at u=0. v-seam: u -> point_at(u, v0), outer
        # major circle at v=0. Sample each as a closed polyline (the exact rational pcurve is a
        # circle).
        def iso_curve(along_v):
            pts = []
            n = 64
            for i in range(n + 1):
                t = i / n
                if along_v:
                    p = body.point_at(dom_u[0], dom_v[0] + (dom_v[1] - dom_v[0]) * t)
                else:
                    p = body.point_at(dom_u[0] + (dom_u[1] - dom_u[0]) * t, dom_v[0])
                pts.append(p)
            return NurbsCurve.create(False, 3, pts)
        ci_useam = brep.add_curve_3d(iso_curve(True))    # minor circle (varies v)
        ci_vseam = brep.add_curve_3d(iso_curve(False))   # major circle (varies u)
        ei_useam = brep.add_edge(ci_useam, vi, vi)
        ei_vseam = brep.add_edge(ci_vseam, vi, vi)
        si_body = brep.add_surface(body)
        fi_body = brep.add_face(si_body, False)
        li_body = brep.add_loop(fi_body, BRepLoopType.Outer)
        # Loop around the UV rectangle: bottom(v-seam), right(u-seam), top(v-seam), left(u-seam).
        c2d_bottom = NurbsCurve.create(False, 1, [Point(dom_u[0], dom_v[0], 0), Point(dom_u[1], dom_v[0], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_bottom), ei_vseam, li_body, False, BRepTrimType.Seam)
        c2d_right = NurbsCurve.create(False, 1, [Point(dom_u[1], dom_v[0], 0), Point(dom_u[1], dom_v[1], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_right), ei_useam, li_body, False, BRepTrimType.Seam)
        c2d_top = NurbsCurve.create(False, 1, [Point(dom_u[1], dom_v[1], 0), Point(dom_u[0], dom_v[1], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_top), ei_vseam, li_body, True, BRepTrimType.Seam)
        c2d_left = NurbsCurve.create(False, 1, [Point(dom_u[0], dom_v[1], 0), Point(dom_u[0], dom_v[0], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_left), ei_useam, li_body, True, BRepTrimType.Seam)
        for ei, e in enumerate(brep.m_topology_edges):
            brep.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            brep.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return brep

    @staticmethod
    def create_block_with_hole(sx, sy, sz, hole_radius):
        from .primitives import Primitives
        import math as _m
        import numpy as np
        brep = BRep()
        brep.name = "block_with_hole"
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        corners = [
            Point(-hx, -hy, -hz), Point(hx, -hy, -hz),
            Point(hx, hy, -hz), Point(-hx, hy, -hz),
            Point(-hx, -hy, hz), Point(hx, -hy, hz),
            Point(hx, hy, hz), Point(-hx, hy, hz),
        ]
        for c in corners:
            brep.add_vertex(c)
        for i in range(8):
            tv = BRepVertex(); tv.point_index = i
            brep.m_topology_vertices.append(tv)
        edge_verts = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7],
        ]
        for ev in edge_verts:
            brep.add_curve_3d(NurbsCurve.create(False, 1, [corners[ev[0]], corners[ev[1]]]))
        for i, ev in enumerate(edge_verts):
            brep.add_edge(i, ev[0], ev[1])
        side_faces = [[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
        def find_edge(v0, v1):
            for e in range(12):
                if (edge_verts[e][0]==v0 and edge_verts[e][1]==v1) or (edge_verts[e][0]==v1 and edge_verts[e][1]==v0):
                    return e
            return -1
        uv = [
            Point(0, 0, 0),
            Point(1, 0, 0),
            Point(1, 1, 0),
            Point(0, 1, 0),
        ]
        for fv in side_faces:
            p00, p10, p01, p11 = corners[fv[0]], corners[fv[1]], corners[fv[3]], corners[fv[2]]
            srf = NurbsSurface(3, False, 2, 2, 2, 2)
            srf.set_cv(0, 0, p00); srf.set_cv(1, 0, p10)
            srf.set_cv(0, 1, p01); srf.set_cv(1, 1, p11)
            si = brep.add_surface(srf)
            face_idx = brep.add_face(si, False)
            loop_idx = brep.add_loop(face_idx, BRepLoopType.Outer)
            for ei in range(4):
                nxt = (ei + 1) % 4
                tc = NurbsCurve.create(False, 1, [uv[ei], uv[nxt]])
                c2d = brep.add_curve_2d(tc)
                eidx = find_edge(fv[ei], fv[nxt])
                rev = edge_verts[eidx][0] != fv[ei]
                brep.add_trim(c2d, eidx, loop_idx, rev, BRepTrimType.Mated)
        cyl_srf = Primitives.cylinder_surface(0, 0, -hz, hole_radius, sz)
        dom_u = cyl_srf.domain(0)
        dom_v = cyl_srf.domain(1)
        si_cyl = brep.add_surface(cyl_srf)
        fi_cyl = brep.add_face(si_cyl, True)
        li_cyl = brep.add_loop(fi_cyl, BRepLoopType.Outer)
        circle_bot = Primitives.circle(0, 0, -hz, hole_radius)
        circle_top = Primitives.circle(0, 0, hz, hole_radius)
        seam_line = NurbsCurve.create(False, 1, [Point(hole_radius, 0, -hz), Point(hole_radius, 0, hz)])
        ci_bot = brep.add_curve_3d(circle_bot)
        ci_top = brep.add_curve_3d(circle_top)
        ci_seam = brep.add_curve_3d(seam_line)
        vi_seam_bot = brep.add_vertex(Point(hole_radius, 0, -hz))
        vi_seam_top = brep.add_vertex(Point(hole_radius, 0, hz))
        tv_b = BRepVertex(); tv_b.point_index = vi_seam_bot
        tv_t = BRepVertex(); tv_t.point_index = vi_seam_top
        brep.m_topology_vertices.extend([tv_b, tv_t])
        ei_bot = brep.add_edge(ci_bot, 8, 8)
        ei_top = brep.add_edge(ci_top, 9, 9)
        ei_seam = brep.add_edge(ci_seam, 8, 9)
        c2d_bot = NurbsCurve.create(False, 1, [Point(dom_u[0], dom_v[0], 0), Point(dom_u[1], dom_v[0], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_bot), ei_bot, li_cyl, False, BRepTrimType.Mated)
        c2d_sr = NurbsCurve.create(False, 1, [Point(dom_u[1], dom_v[0], 0), Point(dom_u[1], dom_v[1], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_sr), ei_seam, li_cyl, False, BRepTrimType.Seam)
        c2d_top = NurbsCurve.create(False, 1, [Point(dom_u[1], dom_v[1], 0), Point(dom_u[0], dom_v[1], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_top), ei_top, li_cyl, True, BRepTrimType.Mated)
        c2d_sl = NurbsCurve.create(False, 1, [Point(dom_u[0], dom_v[1], 0), Point(dom_u[0], dom_v[0], 0)])
        brep.add_trim(brep.add_curve_2d(c2d_sl), ei_seam, li_cyl, True, BRepTrimType.Seam)
        _w = _m.sqrt(2.0) / 2.0
        _cx = [1, 1, 0, -1, -1, -1, 0, 1, 1]; _cy = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        _cw = [1, _w, 1, _w, 1, _w, 1, _w, 1]; _kn = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        def make_cap(z, reversed, circle_edge_idx):
            r = max(hx, hy)
            cap = NurbsSurface(3, False, 2, 2, 2, 2)
            cap.set_cv(0, 0, Point(-r, -r, z)); cap.set_cv(1, 0, Point(r, -r, z))
            cap.set_cv(0, 1, Point(-r, r, z)); cap.set_cv(1, 1, Point(r, r, z))
            si = brep.add_surface(cap)
            fi = brep.add_face(si, reversed)
            outer_li = brep.add_loop(fi, BRepLoopType.Outer)
            fv = [0,3,2,1] if z < 0 else [4,5,6,7]
            for ei in range(4):
                nxt = (ei + 1) % 4
                u0 = (corners[fv[ei]].x + r) / (2.0 * r)
                v0 = (corners[fv[ei]].y + r) / (2.0 * r)
                u1 = (corners[fv[nxt]].x + r) / (2.0 * r)
                v1 = (corners[fv[nxt]].y + r) / (2.0 * r)
                tc = NurbsCurve.create(False, 1, [Point(u0, v0, 0), Point(u1, v1, 0)])
                c2d = brep.add_curve_2d(tc)
                eidx = find_edge(fv[ei], fv[nxt])
                rev = edge_verts[eidx][0] != fv[ei]
                brep.add_trim(c2d, eidx, outer_li, rev, BRepTrimType.Mated)
            inner_li = brep.add_loop(fi, BRepLoopType.Inner)
            hole_crv = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=9)
            hole_crv.m_nurbsknot = np.array(_kn, dtype=np.float64)
            hole_crv.m_cv = np.zeros(9 * 4, dtype=np.float64)
            cr = hole_radius / (2.0 * r)
            cx_uv, cy_uv = 0.5, 0.5
            for i in range(9):
                hole_crv.set_cv_4d(i, (cx_uv+cr*_cx[i])*_cw[i], (cy_uv+cr*_cy[i])*_cw[i], 0.0, _cw[i])
            brep.add_trim(brep.add_curve_2d(hole_crv), circle_edge_idx, inner_li, reversed, BRepTrimType.Mated)
        make_cap(-hz, True, ei_bot)
        make_cap(hz, False, ei_top)
        for ei, e in enumerate(brep.m_topology_edges):
            brep.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            brep.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return brep

    @staticmethod
    def from_polylines(polylines):
        from .plane import Plane
        brep = BRep()
        brep.name = "polysurface"
        tol = 1e-6

        _vertex_map = {}
        def find_or_add(p):
            key = (int(p[0]*1000000+0.5), int(p[1]*1000000+0.5), int(p[2]*1000000+0.5))
            existing = _vertex_map.get(key)
            if existing is not None:
                return existing
            idx = brep.add_vertex(p)
            tv = BRepVertex(); tv.point_index = idx
            brep.m_topology_vertices.append(tv)
            _vertex_map[key] = idx
            return idx

        poly_vi = []
        for pl in polylines:
            pts = pl.get_points()
            n = len(pts) - 1 if pl.is_closed() else len(pts)
            poly_vi.append([find_or_add(pts[i]) for i in range(n)])

        edge_map = {}
        def get_edge(v0, v1):
            lo, hi = min(v0, v1), max(v0, v1)
            if (lo, hi) in edge_map:
                return edge_map[(lo, hi)], v0 != lo
            line = NurbsCurve.create(False, 1, [brep.m_vertices[v0], brep.m_vertices[v1]])
            ci = brep.add_curve_3d(line)
            ei = brep.add_edge(ci, lo, hi)
            edge_map[(lo, hi)] = ei
            return ei, v0 != lo

        for pi, pl in enumerate(polylines):
            vi = poly_vi[pi]
            n = len(vi)
            if n < 3:
                continue
            org, plane = pl.get_fast_plane()
            if not plane.is_valid():
                continue
            xa, ya = plane.x_axis, plane.y_axis
            us, vs_list = [], []
            umin = vmin = 1e30
            umax = vmax = -1e30
            for i in range(n):
                dx = brep.m_vertices[vi[i]][0] - org[0]
                dy = brep.m_vertices[vi[i]][1] - org[1]
                dz = brep.m_vertices[vi[i]][2] - org[2]
                u = dx*xa[0] + dy*xa[1] + dz*xa[2]
                v = dx*ya[0] + dy*ya[1] + dz*ya[2]
                us.append(u); vs_list.append(v)
                umin = min(umin, u); umax = max(umax, u)
                vmin = min(vmin, v); vmax = max(vmax, v)
            pad = max(umax - umin, vmax - vmin) * 0.01
            umin -= pad; umax += pad; vmin -= pad; vmax += pad
            du, dv = umax - umin, vmax - vmin

            def pt3d(u, v):
                return Point(org[0]+u*xa[0]+v*ya[0], org[1]+u*xa[1]+v*ya[1], org[2]+u*xa[2]+v*ya[2])
            srf = NurbsSurface(3, False, 2, 2, 2, 2)
            srf.set_cv(0, 0, pt3d(umin, vmin)); srf.set_cv(1, 0, pt3d(umax, vmin))
            srf.set_cv(0, 1, pt3d(umin, vmax)); srf.set_cv(1, 1, pt3d(umax, vmax))
            si = brep.add_surface(srf)
            f_idx = brep.add_face(si, False)
            l_idx = brep.add_loop(f_idx, BRepLoopType.Outer)
            for i in range(n):
                j = (i + 1) % n
                u0 = (us[i] - umin) / du; v0 = (vs_list[i] - vmin) / dv
                u1 = (us[j] - umin) / du; v1 = (vs_list[j] - vmin) / dv
                tc = NurbsCurve.create(False, 1, [Point(u0, v0, 0), Point(u1, v1, 0)])
                c2d = brep.add_curve_2d(tc)
                ei, rev = get_edge(vi[i], vi[j])
                tt = BRepTrimType.Boundary if not brep.m_topology_edges[ei].trim_indices else BRepTrimType.Mated
                for ti in brep.m_topology_edges[ei].trim_indices:
                    brep.m_trims[ti].type = BRepTrimType.Mated
                brep.add_trim(c2d, ei, l_idx, rev, tt)

        for ei, e in enumerate(brep.m_topology_edges):
            brep.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            brep.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return brep

    @staticmethod
    def from_nurbscurves(curves, holes=None):
        brep = BRep.__new__(BRep)
        brep.guid = _ZERO_GUID; brep.name = "polysurface"; brep.width = 1.0
        brep.surfacecolor = _BLACK_COLOR; brep.xform = _IDENTITY_XFORM
        brep.m_surfaces = []; brep.m_curves_3d = []; brep.m_curves_2d = []
        brep.m_vertices = []; brep.m_topology_vertices = []; brep.m_topology_edges = []
        brep.m_trims = []; brep.m_loops = []; brep.m_faces = []
        tol = 1e-6
        if holes is None:
            holes = []

        _vertex_map = {}
        def find_or_add(p):
            key = (int(p[0]*1000000+0.5), int(p[1]*1000000+0.5), int(p[2]*1000000+0.5))
            existing = _vertex_map.get(key)
            if existing is not None:
                return existing
            idx = brep.add_vertex(p)
            tv = BRepVertex(); tv.point_index = idx
            brep.m_topology_vertices.append(tv)
            _vertex_map[key] = idx
            return idx

        def project_curve_to_uv(crv, org, xa, ya, umin, vmin, du, dv):
            nc = crv.m_cv_count
            c2d = NurbsCurve.__new__(NurbsCurve)
            c2d.guid = _ZERO_GUID; c2d.name = ""; c2d.width = 1.0
            c2d.pointcolors = []; c2d.linecolors = []; c2d.xform = _IDENTITY_XFORM; c2d._rmf_cache = None
            c2d.m_dim = 3; c2d.m_is_rat = crv.m_is_rat; c2d.m_order = crv.m_order
            c2d.m_cv_count = nc; c2d.m_cv_stride = (4 if crv.m_is_rat else 3)
            c2d.m_nurbsknot = crv.m_nurbsknot
            ox=float(org[0]); oy=float(org[1]); oz=float(org[2])
            xa0=float(xa[0]); xa1=float(xa[1]); xa2=float(xa[2])
            ya0=float(ya[0]); ya1=float(ya[1]); ya2=float(ya[2])
            if crv.m_is_rat:
                cl = crv.m_cv.tolist()
                out = _np.empty(nc * 4, dtype=_np.float64)
                for ki in range(nc):
                    k4 = ki * 4
                    w = cl[k4+3]
                    _x = cl[k4]/w - ox; _y = cl[k4+1]/w - oy; _z = cl[k4+2]/w - oz
                    out[k4] = (_x*xa0+_y*xa1+_z*xa2 - umin)/du * w
                    out[k4+1] = (_x*ya0+_y*ya1+_z*ya2 - vmin)/dv * w
                    out[k4+2] = 0.0; out[k4+3] = w
            else:
                cl = crv.m_cv.tolist()
                out = _np.empty(nc * 3, dtype=_np.float64)
                for ki in range(nc):
                    k3 = ki * 3
                    _x = cl[k3] - ox; _y = cl[k3+1] - oy; _z = cl[k3+2] - oz
                    out[k3] = (_x*xa0+_y*xa1+_z*xa2 - umin)/du
                    out[k3+1] = (_x*ya0+_y*ya1+_z*ya2 - vmin)/dv
                    out[k3+2] = 0.0
            c2d.m_cv = out
            return c2d

        def add_curve_loop(crv, face_idx, loop_type, org, xa, ya, umin, vmin, du, dv):
            li = brep.add_loop(face_idx, loop_type)
            ci3d = brep.add_curve_3d(crv)
            crv2d = project_curve_to_uv(crv, org, xa, ya, umin, vmin, du, dv)
            c2d = brep.add_curve_2d(crv2d)
            c = crv.m_cv
            if crv.m_is_rat:
                w0 = float(c[3]); sp = (float(c[0])/w0, float(c[1])/w0, float(c[2])/w0)
                n4 = crv.m_cv_count * 4
                wn = float(c[n4-1]); ep = (float(c[n4-4])/wn, float(c[n4-3])/wn, float(c[n4-2])/wn)
            else:
                sp = (float(c[0]), float(c[1]), float(c[2]))
                ep = (float(c[-3]), float(c[-2]), float(c[-1]))
            vi_s = find_or_add(sp)
            sdx = sp[0]-ep[0]; sdy = sp[1]-ep[1]; sdz = sp[2]-ep[2]
            vi_e = vi_s if sdx*sdx+sdy*sdy+sdz*sdz < tol*tol else find_or_add(ep)
            lo, hi = min(vi_s, vi_e), max(vi_s, vi_e)
            ei = brep.add_edge(ci3d, lo, hi)
            brep.add_trim(c2d, ei, li, False, BRepTrimType.Boundary)

        for ci_idx, crv in enumerate(curves):
            if crv.order() == 2:
                pts = crv.m_cv.reshape(-1, 3).tolist()
            else:
                pts, _ = crv.divide_by_count(max(crv.cv_count() * 2, 4))
            dx0 = float(pts[0][0]-pts[-1][0]); dy0 = float(pts[0][1]-pts[-1][1]); dz0 = float(pts[0][2]-pts[-1][2])
            n = len(pts) - 1 if dx0*dx0+dy0*dy0+dz0*dz0 < tol*tol else len(pts)
            if n < 3:
                continue
            _nx = _ny = _nz = 0.0
            for _i in range(n):
                _a = pts[_i]; _b = pts[(_i+1) % n]
                _nx += (_a[1]-_b[1]) * (_a[2]+_b[2])
                _ny += (_a[2]-_b[2]) * (_a[0]+_b[0])
                _nz += (_a[0]-_b[0]) * (_a[1]+_b[1])
            _nlen = math.sqrt(_nx*_nx + _ny*_ny + _nz*_nz)
            if _nlen < 1e-12:
                continue
            _nx /= _nlen; _ny /= _nlen; _nz /= _nlen
            _ax, _ay, _az = abs(_nx), abs(_ny), abs(_nz)
            if _ay <= _ax and _ay <= _az:
                _px, _py, _pz = -_nz, 0.0, _nx
            elif _ax <= _ay and _ax <= _az:
                _px, _py, _pz = 0.0, _nz, -_ny
            else:
                _px, _py, _pz = _ny, -_nx, 0.0
            _pm = math.sqrt(_px*_px + _py*_py + _pz*_pz)
            _px /= _pm; _py /= _pm; _pz /= _pm
            org = pts[0]
            xa = (_px, _py, _pz)
            ya = (_ny*_pz - _nz*_py, _nz*_px - _nx*_pz, _nx*_py - _ny*_px)
            ox = float(org[0]); oy = float(org[1]); oz = float(org[2])
            xa0=xa[0]; xa1=xa[1]; xa2=xa[2]; ya0=ya[0]; ya1=ya[1]; ya2=ya[2]
            umin=vmin=1e30; umax=vmax=-1e30
            for _i in range(n):
                _p = pts[_i]
                _dx=float(_p[0])-ox; _dy=float(_p[1])-oy; _dz=float(_p[2])-oz
                _u=_dx*xa0+_dy*xa1+_dz*xa2; _v=_dx*ya0+_dy*ya1+_dz*ya2
                if _u<umin: umin=_u
                if _u>umax: umax=_u
                if _v<vmin: vmin=_v
                if _v>vmax: vmax=_v
            if ci_idx < len(holes):
                for hcrv in holes[ci_idx]:
                    _c = hcrv.m_cv.tolist(); _nc = hcrv.m_cv_count
                    if hcrv.m_is_rat:
                        for _ki in range(_nc):
                            _w = _c[_ki*4+3]
                            _dx = _c[_ki*4+0]/_w - ox; _dy = _c[_ki*4+1]/_w - oy; _dz = _c[_ki*4+2]/_w - oz
                            _u = _dx*xa0+_dy*xa1+_dz*xa2; _v = _dx*ya0+_dy*ya1+_dz*ya2
                            if _u<umin: umin=_u
                            if _u>umax: umax=_u
                            if _v<vmin: vmin=_v
                            if _v>vmax: vmax=_v
                    else:
                        _st = hcrv.m_cv_stride
                        for _ki in range(_nc):
                            _dx = _c[_ki*_st+0] - ox; _dy = _c[_ki*_st+1] - oy; _dz = _c[_ki*_st+2] - oz
                            _u = _dx*xa0+_dy*xa1+_dz*xa2; _v = _dx*ya0+_dy*ya1+_dz*ya2
                            if _u<umin: umin=_u
                            if _u>umax: umax=_u
                            if _v<vmin: vmin=_v
                            if _v>vmax: vmax=_v
            pad = max(umax - umin, vmax - vmin) * 0.01
            umin -= pad; umax += pad; vmin -= pad; vmax += pad
            du, dv = umax - umin, vmax - vmin

            srf = NurbsSurface.__new__(NurbsSurface)
            srf.guid = _ZERO_GUID; srf.name = ""; srf.surfacecolor = _BLACK_COLOR
            srf.width = 1.0; srf.pointcolors = []; srf.facecolors = []; srf.linecolors = []
            srf.xform = _IDENTITY_XFORM; srf.m_mesh = None
            srf.m_dim = 3; srf.m_is_rat = 0; srf.m_order = [2, 2]; srf.m_cv_count = [2, 2]
            srf.m_cv_stride = [6, 3]; srf.m_nurbsknot = [_NURBSKNOT_01, _NURBSKNOT_01]
            cv = _np.zeros(12, dtype=_np.float64); srf.m_cv = cv
            cv[0]=ox+umin*xa[0]+vmin*ya[0]; cv[1]=oy+umin*xa[1]+vmin*ya[1]; cv[2]=oz+umin*xa[2]+vmin*ya[2]
            cv[3]=ox+umin*xa[0]+vmax*ya[0]; cv[4]=oy+umin*xa[1]+vmax*ya[1]; cv[5]=oz+umin*xa[2]+vmax*ya[2]
            cv[6]=ox+umax*xa[0]+vmin*ya[0]; cv[7]=oy+umax*xa[1]+vmin*ya[1]; cv[8]=oz+umax*xa[2]+vmin*ya[2]
            cv[9]=ox+umax*xa[0]+vmax*ya[0]; cv[10]=oy+umax*xa[1]+vmax*ya[1]; cv[11]=oz+umax*xa[2]+vmax*ya[2]
            si = brep.add_surface(srf)
            fi = brep.add_face(si, False)
            add_curve_loop(crv, fi, BRepLoopType.Outer, org, xa, ya, umin, vmin, du, dv)
            if ci_idx < len(holes):
                for hcrv in holes[ci_idx]:
                    add_curve_loop(hcrv, fi, BRepLoopType.Inner, org, xa, ya, umin, vmin, du, dv)

        for ei, e in enumerate(brep.m_topology_edges):
            brep.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            brep.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return brep

    ###########################################################################
    # Splitting
    ###########################################################################

    def _split(self, cut_pcurves_for, tolerance=None):
        """Split every face by per-face cut pcurves; rebuild a new BRep.

        cut_pcurves_for(surface) returns the cutter's UV pcurves on that
        surface (empty if the cutter misses it). Faces the cutter crosses are
        subdivided via the loop-aware arrangement; uncut faces and faces with
        inner loops are copied unchanged. Vertices and edges are deduplicated
        in 3D so shared boundaries become mated.
        """
        from .nurbssurface_trimmed import NurbsSurfaceTrimmed

        result = BRep()
        result.name = self.name
        _vmap = {}
        _emap = {}

        def find_or_add_vertex(p):
            key = (int(round(p[0]*1e6)), int(round(p[1]*1e6)), int(round(p[2]*1e6)))
            existing = _vmap.get(key)
            if existing is not None:
                return existing
            idx = result.add_vertex(p)
            tv = BRepVertex(); tv.point_index = idx
            result.m_topology_vertices.append(tv)
            _vmap[key] = idx
            return idx

        def lift(srf, devtol, pc):
            # Adaptive lift: a straight lift stays at ~2 points; a 2-CV UV line that wraps a
            # cylinder/sphere into a full circle is subdivided to the chord tolerance, so the same
            # intersection lifted from two surfaces agrees within the sew tolerance.
            c0, c1 = pc.domain()

            def ev(t):
                uv = pc.point_at(t)
                return srf.point_at(uv[0], uv[1])

            def rec(ta, pa, tb, pb, depth, acc):
                tm = 0.5 * (ta + tb)
                pmid = ev(tm)
                ex, ey, ez = pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]
                l2 = ex*ex + ey*ey + ez*ez
                if l2 > 1e-30:
                    tt = ((pmid[0]-pa[0])*ex + (pmid[1]-pa[1])*ey + (pmid[2]-pa[2])*ez) / l2
                    cx, cy, cz = pa[0]+tt*ex, pa[1]+tt*ey, pa[2]+tt*ez
                    dev = math.sqrt((pmid[0]-cx)**2 + (pmid[1]-cy)**2 + (pmid[2]-cz)**2)
                else:
                    dev = math.sqrt((pmid[0]-pa[0])**2 + (pmid[1]-pa[1])**2 + (pmid[2]-pa[2])**2)
                if dev > devtol and depth < 9:
                    rec(ta, pa, tm, pmid, depth+1, acc)
                    acc.append(pmid)
                    rec(tm, pmid, tb, pb, depth+1, acc)

            n0 = max(pc.cv_count() - 1, 1)
            pts3 = []
            prev = ev(c0)
            pts3.append(prev)
            for i in range(n0):
                ta = c0 + (c1 - c0) * i / n0
                tb = c0 + (c1 - c0) * (i + 1) / n0
                pa = prev if i == 0 else ev(ta)
                pb = ev(tb)
                rec(ta, pa, tb, pb, 0, pts3)
                pts3.append(pb)
            c3d = NurbsCurve.create(False, 1, pts3)
            # Edge key midpoint: the lifted parameter-midpoint is orientation/sampling stable, so
            # two adjacent faces' opposite-oriented copies of a shared segment key the same edge.
            pm = ev(0.5 * (c0 + c1))
            return c3d, pts3[0], pts3[-1], pm

        def append_face(srf, loops):
            si = result.add_surface(srf)
            fi = result.add_face(si, False)
            # One deviation tolerance per face (chord error target = 2e-3 of surface size):
            # straight lifts stay 2-pt, wrapped circles subdivide to within the sew tolerance.
            bmn = [1e30, 1e30, 1e30]
            bmx = [-1e30, -1e30, -1e30]
            for ii in range(srf.cv_count(0)):
                for jj in range(srf.cv_count(1)):
                    p = srf.get_cv(ii, jj)
                    for k in range(3):
                        if p[k] < bmn[k]: bmn[k] = p[k]
                        if p[k] > bmx[k]: bmx[k] = p[k]
            sd = math.sqrt(sum((bmx[k]-bmn[k])**2 for k in range(3)))
            if sd < 1e-12:
                sd = 1.0
            devtol = sd * 2e-3
            for ltype, pcs in loops:
                li = result.add_loop(fi, ltype)
                for pc in pcs:
                    if pc is None or not pc.is_valid():
                        continue
                    c3d, p0, p1, pm = lift(srf, devtol, pc)
                    ci3d = result.add_curve_3d(c3d)
                    va = find_or_add_vertex(p0)
                    vb = find_or_add_vertex(p1)
                    lo, hi = (va, vb) if va <= vb else (vb, va)
                    ekey = (lo, hi,
                            int(round(pm[0]*1e6)), int(round(pm[1]*1e6)), int(round(pm[2]*1e6)))
                    prior = _emap.get(ekey)
                    if prior is not None:
                        ei = prior
                        ttype = BRepTrimType.Mated
                    else:
                        ei = result.add_edge(ci3d, lo, hi)
                        _emap[ekey] = ei
                        ttype = BRepTrimType.Boundary
                    ci2d = result.add_curve_2d(pc)
                    result.add_trim(ci2d, ei, li, False, ttype)

        for face in self.m_faces:
            if face.surface_index < 0 or face.surface_index >= len(self.m_surfaces):
                continue
            srf = self.m_surfaces[face.surface_index]
            outer_pcs = []
            inner_loops = []
            has_inner = False
            for li in face.loop_indices:
                if li < 0 or li >= len(self.m_loops):
                    continue
                loop = self.m_loops[li]
                pcs = []
                for ti in loop.trim_indices:
                    if ti < 0 or ti >= len(self.m_trims):
                        continue
                    trim = self.m_trims[ti]
                    if 0 <= trim.curve_2d_index < len(self.m_curves_2d):
                        pcs.append(self.m_curves_2d[trim.curve_2d_index])
                if loop.type == BRepLoopType.Inner:
                    has_inner = True
                    inner_loops.append(pcs)
                else:
                    outer_pcs = pcs

            cut_pcs = cut_pcurves_for(srf)
            if not cut_pcs or has_inner:
                loops = [(BRepLoopType.Outer, outer_pcs)]
                for il in inner_loops:
                    loops.append((BRepLoopType.Inner, il))
                append_face(srf, loops)
                continue

            parts = NurbsSurfaceTrimmed.split_by_uv_curves(
                srf, outer_pcs + cut_pcs, tolerance,
                use_domain_border=False, n_boundary=len(outer_pcs))
            if len(parts) <= 1:
                loops = [(BRepLoopType.Outer, outer_pcs)]
                append_face(srf, loops)
                continue
            for part in parts:
                # Prefer per-run segmentation (each boundary run a separate pcurve edge) so runs
                # mate with the matching segment edge of an adjacent face -> watertight imprint.
                osegs = getattr(part, 'm_outer_segments', None)
                loops = [(BRepLoopType.Outer, osegs if osegs else [part.m_outer_loop])]
                isegs = getattr(part, 'm_inner_segments', None) or []
                for k, il in enumerate(part.m_inner_loops):
                    seg = isegs[k] if k < len(isegs) else []
                    loops.append((BRepLoopType.Inner, seg if seg else [il]))
                append_face(part.m_surface, loops)

        for ei, e in enumerate(result.m_topology_edges):
            if 0 <= e.start_vertex < len(result.m_topology_vertices):
                result.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            if e.end_vertex != e.start_vertex and 0 <= e.end_vertex < len(result.m_topology_vertices):
                result.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return result

    def split_by_plane(self, plane, tolerance=None):
        """Split this BRep by a plane. Returns a new subdivided BRep."""
        from .intersection import surface_plane_uv

        def cut_for(srf):
            return [pair[1] for pair in surface_plane_uv(srf, plane, tolerance)]
        return self._split(cut_for, tolerance)

    def split_by_surface(self, cutter, tolerance=None):
        """Split this BRep by another surface. Returns a new subdivided BRep."""
        from .intersection import surface_surface
        cutter_bb = _aabb_from_surface(cutter)

        def cut_for(srf):
            srf_bb = _aabb_from_surface(srf)
            margin = max(srf_bb[1][0] - srf_bb[0][0], srf_bb[1][1] - srf_bb[0][1],
                         srf_bb[1][2] - srf_bb[0][2]) * 1e-3
            if not _aabb_overlap(srf_bb, cutter_bb, margin):
                return []
            return [triple[1] for triple in surface_surface(srf, cutter, tolerance)]
        return self._split(cut_for, tolerance)

    def split_by_curves(self, curves, tolerance=None):
        """Split this BRep by 3D curves pulled onto each face. New BRep."""
        from .closest import Closest
        curve_bbs = [_aabb_from_curve(c) for c in curves]

        def cut_for(srf):
            srf_bb = _aabb_from_surface(srf)
            margin = max(srf_bb[1][0] - srf_bb[0][0], srf_bb[1][1] - srf_bb[0][1],
                         srf_bb[1][2] - srf_bb[0][2]) * 1e-3
            out = []
            for crv, cbb in zip(curves, curve_bbs):
                if not _aabb_overlap(srf_bb, cbb, margin):
                    continue
                for pc in Closest.surface_curve(srf, crv, 0.0, 0.0, tolerance):
                    out.append(pc)
            return out
        return self._split(cut_for, tolerance)

    def split_by_line(self, line, tolerance=None):
        """Split this BRep by a line pulled onto each face. New BRep."""
        pts = [line.start(), line.end()]
        crv = NurbsCurve.create(False, 1, pts)
        return self.split_by_curves([crv], tolerance)

    def _subset(self, face_indices):
        """Build a standalone BRep from a subset of this BRep's faces.

        Copies only the referenced geometry and topology, remapping indices.
        """
        sub = BRep()
        sub.name = self.name
        s_map = {}
        c3_map = {}
        c2_map = {}
        v_map = {}
        e_map = {}
        l_map = {}

        def map_surface(i):
            if i not in s_map:
                s_map[i] = sub.add_surface(self.m_surfaces[i])
            return s_map[i]

        def map_vertex(i):
            if i not in v_map:
                pt = self.m_vertices[self.m_topology_vertices[i].point_index]
                idx = sub.add_vertex(pt)
                tv = BRepVertex(); tv.point_index = idx
                sub.m_topology_vertices.append(tv)
                v_map[i] = len(sub.m_topology_vertices) - 1
            return v_map[i]

        def map_edge(i):
            if i < 0:
                return -1
            if i not in e_map:
                e = self.m_topology_edges[i]
                ci3 = c3_map.get(e.curve_3d_index)
                if ci3 is None and 0 <= e.curve_3d_index < len(self.m_curves_3d):
                    ci3 = sub.add_curve_3d(self.m_curves_3d[e.curve_3d_index])
                    c3_map[e.curve_3d_index] = ci3
                sv = map_vertex(e.start_vertex) if 0 <= e.start_vertex < len(self.m_topology_vertices) else -1
                ev = map_vertex(e.end_vertex) if 0 <= e.end_vertex < len(self.m_topology_vertices) else -1
                e_map[i] = sub.add_edge(ci3 if ci3 is not None else -1, sv, ev)
            return e_map[i]

        for fi in face_indices:
            face = self.m_faces[fi]
            si = map_surface(face.surface_index)
            new_fi = sub.add_face(si, face.reversed)
            for li in face.loop_indices:
                loop = self.m_loops[li]
                new_li = sub.add_loop(new_fi, loop.type)
                for ti in loop.trim_indices:
                    trim = self.m_trims[ti]
                    ci2 = c2_map.get(trim.curve_2d_index)
                    if ci2 is None and 0 <= trim.curve_2d_index < len(self.m_curves_2d):
                        ci2 = sub.add_curve_2d(self.m_curves_2d[trim.curve_2d_index])
                        c2_map[trim.curve_2d_index] = ci2
                    sub.add_trim(ci2 if ci2 is not None else -1, map_edge(trim.edge_index),
                                 new_li, trim.reversed, trim.type)
        for ei, e in enumerate(sub.m_topology_edges):
            if 0 <= e.start_vertex < len(sub.m_topology_vertices):
                sub.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            if e.end_vertex != e.start_vertex and 0 <= e.end_vertex < len(sub.m_topology_vertices):
                sub.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return sub

    def split_by_plane_pieces(self, plane, tolerance=None):
        """Split this BRep by a plane and separate the result into the pieces
        on each side of the plane. Returns a list of BReps (one per side)."""
        whole = self.split_by_plane(plane, tolerance)
        o = plane.origin
        n = plane.z_axis
        pos = []
        neg = []
        for fi, face in enumerate(whole.m_faces):
            srf = whole.m_surfaces[face.surface_index]
            # Classify by the centroid of the face's outer loop lifted to 3D
            # (the underlying surface center is shared by both cut halves).
            sx = sy = sz = 0.0
            cnt = 0
            for li in face.loop_indices:
                loop = whole.m_loops[li]
                if loop.type != BRepLoopType.Outer:
                    continue
                for ti in loop.trim_indices:
                    pc = whole.m_curves_2d[whole.m_trims[ti].curve_2d_index]
                    d0, d1 = pc.domain()
                    for k in range(8):
                        uv = pc.point_at(d0 + (d1 - d0) * k / 8.0)
                        p = srf.point_at(uv[0], uv[1])
                        sx += p[0]; sy += p[1]; sz += p[2]; cnt += 1
            if cnt == 0:
                continue
            cx, cy, cz = sx / cnt, sy / cnt, sz / cnt
            d = (cx - o[0]) * n[0] + (cy - o[1]) * n[1] + (cz - o[2]) * n[2]
            (pos if d >= 0.0 else neg).append(fi)
        pieces = []
        for idxs in (pos, neg):
            if idxs:
                pieces.append(whole._subset(idxs))
        return pieces

    def split_by_brep(self, cutter, tolerance=None):
        """Split this BRep by every face of another BRep. New BRep.

        Each target face is cut by every overlapping cutter face (planar faces
        via the fast plane path, others via surface/surface).
        """
        from .intersection import cut_curves_on_surface
        cutter_surfaces = cutter.m_surfaces
        cutter_bbs = [_aabb_from_surface(cs) for cs in cutter_surfaces]

        def cut_for(srf):
            srf_bb = _aabb_from_surface(srf)
            margin = max(srf_bb[1][0] - srf_bb[0][0], srf_bb[1][1] - srf_bb[0][1],
                         srf_bb[1][2] - srf_bb[0][2]) * 1e-3
            out = []
            for cs, cbb in zip(cutter_surfaces, cutter_bbs):
                if not _aabb_overlap(srf_bb, cbb, margin):
                    continue
                for pc in cut_curves_on_surface(srf, cs, tolerance):
                    out.append(pc)
            return out
        return self._split(cut_for, tolerance)

    ###########################################################################
    # Booleans
    ###########################################################################

    def subset(self, face_indices):
        """Build a standalone BRep from a subset of this BRep's faces, preserving edge topology."""
        sub = BRep()
        sub.name = self.name
        s_map, c2_map, c3_map, v_map, e_map = {}, {}, {}, {}, {}

        def map_surface(i):
            if i in s_map:
                return s_map[i]
            x = sub.add_surface(self.m_surfaces[i]); s_map[i] = x; return x

        def map_c2(i):
            if i < 0 or i >= len(self.m_curves_2d):
                return -1
            if i in c2_map:
                return c2_map[i]
            x = sub.add_curve_2d(self.m_curves_2d[i]); c2_map[i] = x; return x

        def map_vertex(i):
            if i < 0 or i >= len(self.m_topology_vertices):
                return -1
            if i in v_map:
                return v_map[i]
            idx = sub.add_vertex(self.m_vertices[self.m_topology_vertices[i].point_index])
            tv = BRepVertex(); tv.point_index = idx
            sub.m_topology_vertices.append(tv)
            nv = len(sub.m_topology_vertices) - 1
            v_map[i] = nv
            return nv

        def map_edge(i):
            if i < 0 or i >= len(self.m_topology_edges):
                return -1
            if i in e_map:
                return e_map[i]
            e = self.m_topology_edges[i]
            ci3 = -1
            if 0 <= e.curve_3d_index < len(self.m_curves_3d):
                if e.curve_3d_index in c3_map:
                    ci3 = c3_map[e.curve_3d_index]
                else:
                    ci3 = sub.add_curve_3d(self.m_curves_3d[e.curve_3d_index]); c3_map[e.curve_3d_index] = ci3
            sv = map_vertex(e.start_vertex); ev = map_vertex(e.end_vertex)
            ne = sub.add_edge(ci3, sv, ev); e_map[i] = ne; return ne

        for fi in face_indices:
            face = self.m_faces[fi]
            si = map_surface(face.surface_index)
            new_fi = sub.add_face(si, face.reversed)
            for li in face.loop_indices:
                lp = self.m_loops[li]
                new_li = sub.add_loop(new_fi, lp.type)
                for ti in lp.trim_indices:
                    trim = self.m_trims[ti]
                    ci2 = map_c2(trim.curve_2d_index)
                    ei = map_edge(trim.edge_index)
                    sub.add_trim(ci2, ei, new_li, trim.reversed, trim.type)
        for ei, e in enumerate(sub.m_topology_edges):
            if 0 <= e.start_vertex < len(sub.m_topology_vertices):
                sub.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            if e.end_vertex != e.start_vertex and 0 <= e.end_vertex < len(sub.m_topology_vertices):
                sub.m_topology_vertices[e.end_vertex].edge_indices.append(ei)
        return sub

    def _brep_bbox_diag(self):
        xmn = ymn = zmn = 1e300
        xmx = ymx = zmx = -1e300
        for p in self.m_vertices:
            xmn = min(xmn, p[0]); ymn = min(ymn, p[1]); zmn = min(zmn, p[2])
            xmx = max(xmx, p[0]); ymx = max(ymx, p[1]); zmx = max(zmx, p[2])
        d = ((xmx-xmn)**2 + (ymx-ymn)**2 + (zmx-zmn)**2) ** 0.5
        return d if d > 0 else 1.0

    def imprint_edges(self, tol=0.0):
        """Split an under-mated edge at interior points coinciding with another edge's endpoint,
        so a long edge spanning shorter coincident edges is broken to match them (T-junctions)."""
        if tol <= 0.0:
            tol = self._brep_bbox_diag() * 1e-6
        vpos = []
        for tv in self.m_topology_vertices:
            if 0 <= tv.point_index < len(self.m_vertices):
                vpos.append(self.m_vertices[tv.point_index])
            else:
                vpos.append(None)

        def split_multi(c, params):
            params = sorted(params)
            out = []
            rem = c
            for t in params:
                l, r = rem.split(t)
                if l is None or r is None or not l.is_valid() or not r.is_valid():
                    return None
                out.append(l); rem = r
            out.append(rem)
            return out

        ne0 = len(self.m_topology_edges)
        for ei in range(ne0):
            if len(self.m_topology_edges[ei].trim_indices) >= 2:
                continue
            ci = self.m_topology_edges[ei].curve_3d_index
            if ci < 0 or ci >= len(self.m_curves_3d):
                continue
            C = self.m_curves_3d[ci]
            if not C.is_valid():
                continue
            pA = C.point_at_start(); pB = C.point_at_end()
            clen = pA.distance(pB)
            if clen < tol:
                continue
            cd0, cd1 = C.domain()
            ebb = [1e300, 1e300, 1e300, -1e300, -1e300, -1e300]
            for k in range(7):
                p = C.point_at(cd0 + (cd1 - cd0) * k / 6)
                for d in range(3):
                    if p[d] < ebb[d]: ebb[d] = p[d]
                    if p[d] > ebb[d+3]: ebb[d+3] = p[d]
            splits = []
            for V in vpos:
                if V is None:
                    continue
                if (V[0] < ebb[0]-tol or V[0] > ebb[3]+tol or V[1] < ebb[1]-tol or
                        V[1] > ebb[4]+tol or V[2] < ebb[2]-tol or V[2] > ebb[5]+tol):
                    continue
                if V.distance(pA) < tol or V.distance(pB) < tol:
                    continue
                tc = C.closest_parameter(V)
                if C.point_at(tc).distance(V) > tol:
                    continue
                frac = (tc - cd0) / (cd1 - cd0)
                if frac <= 1e-6 or frac >= 1.0 - 1e-6:
                    continue
                if any(s[1].distance(V) < tol for s in splits):
                    continue
                splits.append((tc, V))
            if not splits:
                continue
            splits.sort(key=lambda s: s[0])
            c3pieces = split_multi(C, [s[0] for s in splits])
            if c3pieces is None or len(c3pieces) != len(splits) + 1:
                continue
            orig_trims = list(self.m_topology_edges[ei].trim_indices)
            vids = [self.m_topology_edges[ei].start_vertex]
            for (_, V) in splits:
                pidx = self.add_vertex(V)
                tv = BRepVertex(); tv.point_index = pidx
                self.m_topology_vertices.append(tv)
                vids.append(len(self.m_topology_vertices) - 1)
            vids.append(self.m_topology_edges[ei].end_vertex)
            edge_ids = []
            for j in range(len(c3pieces)):
                if j == 0:
                    c3idx = ci; eidx = ei; self.m_curves_3d[ci] = c3pieces[0]
                else:
                    c3idx = self.add_curve_3d(c3pieces[j]); eidx = len(self.m_topology_edges)
                    self.m_topology_edges.append(BRepEdge())
                e = self.m_topology_edges[eidx]
                e.curve_3d_index = c3idx; e.start_vertex = vids[j]; e.end_vertex = vids[j+1]
                e.trim_indices = []
                edge_ids.append(eidx)
            for ti in orig_trims:
                if ti < 0 or ti >= len(self.m_trims):
                    continue
                T = self.m_trims[ti]
                li = T.loop_index
                fi = self.m_loops[li].face_index if 0 <= li < len(self.m_loops) else -1
                si = self.m_faces[fi].surface_index if 0 <= fi < len(self.m_faces) else -1
                c2 = T.curve_2d_index
                p2pieces = None
                if 0 <= si < len(self.m_surfaces) and 0 <= c2 < len(self.m_curves_2d):
                    from .closest import Closest
                    S = self.m_surfaces[si]; P = self.m_curves_2d[c2]
                    tps = []
                    for (_, V) in splits:
                        uu, vv, _dd = Closest.surface_point(S, V)
                        tps.append(P.closest_parameter(Point(uu, vv, 0.0)))
                    p2pieces = split_multi(P, tps)
                ok = p2pieces is not None and len(p2pieces) == len(edge_ids)
                newtrims = []
                for j in range(len(edge_ids)):
                    if ok:
                        pc = p2pieces[j]
                    else:
                        pc = self.m_curves_2d[max(c2, 0)] if j == 0 else NurbsCurve()
                    if j == 0:
                        c2idx = c2 if c2 >= 0 else self.add_curve_2d(pc)
                        if c2 >= 0:
                            self.m_curves_2d[c2] = pc
                        tidx = ti
                    else:
                        c2idx = self.add_curve_2d(pc); tidx = len(self.m_trims)
                        self.m_trims.append(BRepTrim())
                    tr = self.m_trims[tidx]
                    tr.curve_2d_index = c2idx; tr.edge_index = edge_ids[j]
                    tr.loop_index = li; tr.reversed = T.reversed; tr.type = T.type
                    newtrims.append(tidx)
                    self.m_topology_edges[edge_ids[j]].trim_indices.append(tidx)
                if T.reversed:
                    newtrims.reverse()
                if 0 <= li < len(self.m_loops):
                    tl = self.m_loops[li].trim_indices
                    if ti in tl:
                        pos = tl.index(ti)
                        tl[pos:pos+1] = newtrims
        for v in self.m_topology_vertices:
            v.edge_indices = []
        for ei, e in enumerate(self.m_topology_edges):
            if 0 <= e.start_vertex < len(self.m_topology_vertices):
                self.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            if e.end_vertex != e.start_vertex and 0 <= e.end_vertex < len(self.m_topology_vertices):
                self.m_topology_vertices[e.end_vertex].edge_indices.append(ei)

    def co_refine_coincident_edges(self, tol=0.0):
        """Co-refine the A<->B section: where one operand imprinted the shared curve as a single
        closed circle and the other as 2+ open arcs (periodic-seam straddle) -- or as partially
        overlapping arcs -- split the longer at the shorter's endpoints so they mate 1:1. Strictly
        coincidence-gated, so each solid's own edges are untouched (the OCCT "shared section edge"
        guarantee done as a co-refinement). After it, sew merges arc-for-arc identical segments."""
        from .closest import Closest
        diag = self._brep_bbox_diag()
        if tol <= 0.0:
            tol = diag * 5e-3   # sew-level coincidence tolerance

        def split_multi(c, params):
            params = sorted(params)
            out = []
            rem = c
            for t in params:
                l, r = rem.split(t)
                if l is None or r is None or not l.is_valid() or not r.is_valid():
                    return None
                out.append(l); rem = r
            out.append(rem)
            return out

        def p2pl(p, pts):
            best = 1e300
            for j in range(len(pts) - 1):
                a, b = pts[j], pts[j+1]
                ex, ey, ez = b[0]-a[0], b[1]-a[1], b[2]-a[2]
                l2 = ex*ex + ey*ey + ez*ez
                t = ((p[0]-a[0])*ex + (p[1]-a[1])*ey + (p[2]-a[2])*ez) / l2 if l2 > 1e-30 else 0.0
                t = min(max(t, 0.0), 1.0)
                cx, cy, cz = a[0]+t*ex, a[1]+t*ey, a[2]+t*ez
                d = ((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2) ** 0.5
                if d < best: best = d
            return best

        ne0 = len(self.m_topology_edges)

        def cand(e):
            return (0 <= e < ne0 and len(self.m_topology_edges[e].trim_indices) < 2
                    and 0 <= self.m_topology_edges[e].curve_3d_index < len(self.m_curves_3d))

        NS = 24
        samp = [[] for _ in range(ne0)]
        bbox = [None] * ne0
        for e in range(ne0):
            if not cand(e):
                continue
            C = self.m_curves_3d[self.m_topology_edges[e].curve_3d_index]
            t0, t1 = C.domain()
            bb = [1e300, 1e300, 1e300, -1e300, -1e300, -1e300]
            for k in range(NS + 1):
                p = C.point_at(t0 + (t1 - t0) * k / NS)
                samp[e].append(p)
                for d in range(3):
                    if p[d] < bb[d]: bb[d] = p[d]
                    if p[d] > bb[d+3]: bb[d+3] = p[d]
            bbox[e] = bb

        def bbox_far(i, j):
            a, b = bbox[i], bbox[j]
            return (a[0] > b[3]+tol or b[0] > a[3]+tol or a[1] > b[4]+tol or
                    b[1] > a[4]+tol or a[2] > b[5]+tol or b[2] > a[5]+tol)

        # ej is an arc-SUBSET of ei: every ej sample lies within tol of ei's polyline. (One-directional;
        # a full circle is NOT a subset of one of its arcs, but each arc IS a subset of the circle.)
        def subset_of(ej, ei):
            if len(samp[ej]) < 2 or len(samp[ei]) < 2:
                return False
            for p in samp[ej]:
                if p2pl(p, samp[ei]) > tol:
                    return False
            return True

        for ei in range(ne0):
            if not cand(ei):
                continue
            C = self.m_curves_3d[self.m_topology_edges[ei].curve_3d_index]
            if not C.is_valid():
                continue
            dom0, dom1 = C.domain()
            pA = C.point_at_start(); pB = C.point_at_end()
            closed = pA.distance(pB) < tol

            # Split points on C = endpoints of DISTINCT under-mated edges that are arc-subsets of C
            # and land strictly interior on C (a circle split at its coincident arcs' shared
            # endpoints).
            sp = []
            seam_has_split = False
            for ej in range(ne0):
                if ej == ei or not cand(ej) or bbox_far(ei, ej) or not subset_of(ej, ei):
                    continue
                Cj = self.m_curves_3d[self.m_topology_edges[ej].curve_3d_index]
                # Only an OPEN arc subdivides ei at its endpoints. A closed circle coincident with ei
                # (e.g. the same cut circle imprinted whole on both operands) has only its arbitrary
                # param-seam as an "endpoint" -- splitting there is spurious; sew merges them whole.
                if Cj.point_at_start().distance(Cj.point_at_end()) < tol:
                    continue
                ends = [Cj.point_at_start(), Cj.point_at_end()]
                for V in ends:
                    tc = C.closest_parameter(V)
                    if C.point_at(tc).distance(V) > tol:
                        continue
                    frac = (tc - dom0) / (dom1 - dom0)
                    if frac <= 1e-6 or frac >= 1.0 - 1e-6:
                        seam_has_split = True
                        continue
                    if any(s[1].distance(V) < tol for s in sp):
                        continue
                    sp.append((tc, V))
            if not sp:
                continue
            sp.sort(key=lambda s: s[0])
            iparams = [s[0] for s in sp]
            ipts = [s[1] for s in sp]

            c3pieces = split_multi(C, iparams)
            if c3pieces is None or len(c3pieces) != len(iparams) + 1:
                continue   # split failed -> keep edge

            # wrap-join the first+last 3D piece across the param seam ONLY for a closed edge whose
            # seam is interior to an arc (no split point at the seam). Else (open, or seam coincides
            # with a split point) the pieces are already the arcs.
            do_wrap = closed and not seam_has_split

            orig_trims = list(self.m_topology_edges[ei].trim_indices)
            # vertices at the interior split points
            svids = []
            for V in ipts:
                pidx = self.add_vertex(V)
                tv = BRepVertex(); tv.point_index = pidx
                svids.append(len(self.m_topology_vertices))
                self.m_topology_vertices.append(tv)

            # edge-piece curves + their (startV,endV)
            arcs = []; arc_v = []
            if do_wrap:
                wrap = NurbsCurve.join([c3pieces[-1], c3pieces[0]], tol)
                if len(wrap) != 1 or not wrap[0].is_valid():
                    continue
                for k in range(1, len(c3pieces) - 1):
                    arcs.append(c3pieces[k]); arc_v.append((svids[k-1], svids[k]))
                arcs.append(wrap[0]); arc_v.append((svids[-1], svids[0]))
            else:
                vids = [self.m_topology_edges[ei].start_vertex]
                vids.extend(svids)
                vids.append(self.m_topology_edges[ei].end_vertex)
                for k in range(len(c3pieces)):
                    arcs.append(c3pieces[k]); arc_v.append((vids[k], vids[k+1]))

            # create the piece edges (piece 0 reuses ei + its 3D-curve slot)
            edge_ids = []
            for k in range(len(arcs)):
                if k == 0:
                    c3idx = self.m_topology_edges[ei].curve_3d_index
                    eidx = ei
                    self.m_curves_3d[c3idx] = arcs[0]
                else:
                    c3idx = self.add_curve_3d(arcs[k])
                    eidx = len(self.m_topology_edges)
                    self.m_topology_edges.append(BRepEdge())
                e = self.m_topology_edges[eidx]
                e.curve_3d_index = c3idx
                e.start_vertex = arc_v[k][0]; e.end_vertex = arc_v[k][1]
                e.trim_indices = []
                edge_ids.append(eidx)

            # split each trim's 2D pcurve at the same params (project the split points onto the
            # surface)
            for ti in orig_trims:
                if ti < 0 or ti >= len(self.m_trims):
                    continue
                T = self.m_trims[ti]
                li = T.loop_index
                fi = self.m_loops[li].face_index if 0 <= li < len(self.m_loops) else -1
                si = self.m_faces[fi].surface_index if 0 <= fi < len(self.m_faces) else -1
                c2 = T.curve_2d_index
                # Build each arc's 2D pcurve. PLANAR face: project the (already-split) 3D arc onto
                # the plane (exact, cheap). CURVED face: split the original open pcurve at the
                # projected params (cheap, exact) -- projecting onto a curved surface per-sample is
                # slow/unstable near singularities (e.g. a cone apex). (NurbsCurve.split silently
                # fails on a CLOSED rational circle, so closed circles must be on planar faces here.)
                p2 = []
                if 0 <= si < len(self.m_surfaces):
                    S = self.m_surfaces[si]
                    if S.is_planar(None, 1e-6):
                        for arc3 in arcs:
                            ad0, ad1 = arc3.domain()
                            n = max(arc3.cv_count() * 2, 12)
                            uvs = []
                            for s in range(n + 1):
                                P3 = arc3.point_at(ad0 + (ad1 - ad0) * s / n)
                                uu, vv, _dd = Closest.surface_point(S, P3)
                                uvs.append(Point(uu, vv, 0.0))
                            p2.append(NurbsCurve.create(False, 1, uvs))
                    elif 0 <= c2 < len(self.m_curves_2d):
                        P = self.m_curves_2d[c2]
                        tps = []
                        for V in ipts:
                            uu, vv, _dd = Closest.surface_point(S, V)
                            tps.append(P.closest_parameter(Point(uu, vv, 0.0)))
                        sm = split_multi(P, tps)
                        p2 = sm if sm is not None else []
                        if do_wrap and len(p2) >= 2:
                            w2 = NurbsCurve.join([p2[-1], p2[0]], tol)
                            if len(w2) == 1 and w2[0].is_valid():
                                a2 = list(p2[1:-1]); a2.append(w2[0]); p2 = a2
                            else:
                                p2 = []
                ok = len(p2) == len(edge_ids)
                newtrims = []
                for k in range(len(edge_ids)):
                    if ok:
                        pc = p2[k]
                    else:
                        pc = self.m_curves_2d[max(c2, 0)] if k == 0 else NurbsCurve()
                    if k == 0:
                        c2idx = c2 if c2 >= 0 else self.add_curve_2d(pc)
                        if c2 >= 0:
                            self.m_curves_2d[c2] = pc
                        tidx = ti
                    else:
                        c2idx = self.add_curve_2d(pc)
                        tidx = len(self.m_trims)
                        self.m_trims.append(BRepTrim())
                    tr = self.m_trims[tidx]
                    tr.curve_2d_index = c2idx; tr.edge_index = edge_ids[k]
                    tr.loop_index = li; tr.reversed = T.reversed; tr.type = T.type
                    newtrims.append(tidx)
                    self.m_topology_edges[edge_ids[k]].trim_indices.append(tidx)
                if T.reversed:
                    newtrims.reverse()
                if 0 <= li < len(self.m_loops):
                    tl = self.m_loops[li].trim_indices
                    if ti in tl:
                        pos = tl.index(ti)
                        tl[pos:pos+1] = newtrims

        # rebuild vertex->edge adjacency
        for v in self.m_topology_vertices:
            v.edge_indices = []
        for ei, e in enumerate(self.m_topology_edges):
            if 0 <= e.start_vertex < len(self.m_topology_vertices):
                self.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            if e.end_vertex != e.start_vertex and 0 <= e.end_vertex < len(self.m_topology_vertices):
                self.m_topology_vertices[e.end_vertex].edge_indices.append(ei)

    def sew_coincident_edges(self, tol=0.0):
        """Merge edges whose 3D curves coincide (point-to-segment Hausdorff < tol) into single
        mated edges, so independently-imprinted intersection curves on A and B share one edge."""
        diag = self._brep_bbox_diag()
        if tol <= 0.0:
            tol = diag * 5e-3
        ne = len(self.m_topology_edges)
        NS = 16
        samp = [[] for _ in range(ne)]
        bbox = [None] * ne
        # Only UNDER-mated edges (fewer than 2 trims) are sewing candidates: a shared
        # intersection curve is imprinted as a 1-trim edge on each side and the two halves must
        # merge into one 2-trim edge. Already-2-trim edges are watertight and cannot legitimately
        # coincide with another edge, so skip sampling AND comparing them (O(ne^2) -> O(k^2)).
        for ei in range(ne):
            if len(self.m_topology_edges[ei].trim_indices) >= 2:
                continue
            ci = self.m_topology_edges[ei].curve_3d_index
            if ci < 0 or ci >= len(self.m_curves_3d):
                continue
            t0, t1 = self.m_curves_3d[ci].domain()
            bb = [1e300, 1e300, 1e300, -1e300, -1e300, -1e300]
            for k in range(NS + 1):
                p = self.m_curves_3d[ci].point_at(t0 + (t1 - t0) * k / NS)
                samp[ei].append(p)
                for d in range(3):
                    if p[d] < bb[d]: bb[d] = p[d]
                    if p[d] > bb[d+3]: bb[d+3] = p[d]
            bbox[ei] = bb

        def bbox_far(i, j):
            a, b = bbox[i], bbox[j]
            return (a[0] > b[3]+tol or b[0] > a[3]+tol or a[1] > b[4]+tol or
                    b[1] > a[4]+tol or a[2] > b[5]+tol or b[2] > a[5]+tol)

        def pt_to_polyline(p, pts):
            best = 1e300
            for j in range(len(pts) - 1):
                a, b = pts[j], pts[j+1]
                ex, ey, ez = b[0]-a[0], b[1]-a[1], b[2]-a[2]
                l2 = ex*ex + ey*ey + ez*ez
                t = ((p[0]-a[0])*ex + (p[1]-a[1])*ey + (p[2]-a[2])*ez) / l2 if l2 > 1e-30 else 0.0
                t = min(max(t, 0.0), 1.0)
                cx, cy, cz = a[0]+t*ex, a[1]+t*ey, a[2]+t*ez
                d = ((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2) ** 0.5
                if d < best: best = d
            return best

        def coincident_within(a, b):
            # Directed point-to-polyline Hausdorff both ways, EARLY-EXIT as soon as any sample
            # exceeds tol (then provably not coincident). Same accept/reject as full Hausdorff
            # < tol, but rejection is O(1) instead of O(NS^2). Midpoint first: non-coincident
            # edges sharing an endpoint diverge most in the middle, so it rejects fastest.
            if len(a) < 2 or len(b) < 2:
                return False
            if pt_to_polyline(a[len(a)//2], b) > tol: return False
            if pt_to_polyline(b[len(b)//2], a) > tol: return False
            for p in a:
                if pt_to_polyline(p, b) > tol: return False
            for p in b:
                if pt_to_polyline(p, a) > tol: return False
            return True

        rep = [-1] * ne
        reps = []
        for ei in range(ne):
            if not samp[ei]:
                rep[ei] = ei; reps.append(ei); continue
            for r in reps:
                if samp[r] and bbox[ei] is not None and bbox[r] is not None and \
                        not bbox_far(ei, r) and coincident_within(samp[ei], samp[r]):
                    rep[ei] = r; break
            if rep[ei] < 0:
                rep[ei] = ei; reps.append(ei)
        old2new = {}
        newedges = []
        for r in reps:
            old2new[r] = len(newedges)
            e = self.m_topology_edges[r]
            ne_ = BRepEdge()
            ne_.curve_3d_index = e.curve_3d_index; ne_.start_vertex = e.start_vertex
            ne_.end_vertex = e.end_vertex; ne_.trim_indices = []
            newedges.append(ne_)
        for ti in range(len(self.m_trims)):
            oe = self.m_trims[ti].edge_index
            if oe < 0 or oe >= ne:
                continue
            ni = old2new[rep[oe]]
            self.m_trims[ti].edge_index = ni
            newedges[ni].trim_indices.append(ti)
        self.m_topology_edges = newedges
        for v in self.m_topology_vertices:
            v.edge_indices = []
        for ei, e in enumerate(self.m_topology_edges):
            if 0 <= e.start_vertex < len(self.m_topology_vertices):
                self.m_topology_vertices[e.start_vertex].edge_indices.append(ei)
            if e.end_vertex != e.start_vertex and 0 <= e.end_vertex < len(self.m_topology_vertices):
                self.m_topology_vertices[e.end_vertex].edge_indices.append(ei)

    def boolean(self, other, op, tolerance=None):
        """Boolean of two solids via imprint -> classify -> select -> sew. op in {'union','difference','intersection'}."""
        from .remesh_cdt import RemeshCDT
        from .polyline import Polyline
        A2 = self.split_by_brep(other, tolerance)
        B2 = other.split_by_brep(self, tolerance)
        # Classify fragments against the OTHER solid. When an operand is a recognized primitive
        # (box/cylinder/sphere) the point-in-solid test is analytic (O(1)) so we skip building
        # its mesh AND the ray-cast; unrecognized (general/freeform) operands fall back to the
        # mesh ray-cast unchanged.
        primA = _recognize_solid(self)
        primB = _recognize_solid(other)
        meshA = None if primA else self.mesh()
        meshB = None if primB else other.mesh()

        def face_sample(X, fi):
            face = X.m_faces[fi]
            srf = X.m_surfaces[face.surface_index]
            outers, inners = [], []
            for li in face.loop_indices:
                if li < 0 or li >= len(X.m_loops):
                    continue
                pts = []
                for ti in X.m_loops[li].trim_indices:
                    if ti < 0 or ti >= len(X.m_trims):
                        continue
                    c2 = X.m_trims[ti].curve_2d_index
                    if c2 < 0 or c2 >= len(X.m_curves_2d):
                        continue
                    pc = X.m_curves_2d[c2]
                    d0, d1 = pc.domain()
                    n = max(pc.cv_count() * 3, 6)
                    for i in range(n):
                        uv = pc.point_at(d0 + (d1 - d0) * i / n)
                        pts.append(Point(uv[0], uv[1], 0))
                if len(pts) < 3:
                    continue
                (outers if X.m_loops[li].type == BRepLoopType.Outer else inners).append(Polyline(pts))

            def fallback():
                u0, u1 = srf.domain(0); v0, v1 = srf.domain(1)
                return srf.point_at(0.5*(u0+u1), 0.5*(v0+v1))

            if not outers:
                return fallback()
            allp = [outers[0]] + inners
            try:
                tris = RemeshCDT.triangulate(allp)
            except Exception:
                return fallback()
            flat = []
            for pl in allp:
                flat.extend(pl.get_points())
            if not tris or not flat:
                return fallback()
            best, best_area = -1, -1.0
            for ti, t in enumerate(tris):
                if t[0] < 0 or t[2] >= len(flat):
                    continue
                ar = abs((flat[t[1]][0]-flat[t[0]][0])*(flat[t[2]][1]-flat[t[0]][1])
                         - (flat[t[2]][0]-flat[t[0]][0])*(flat[t[1]][1]-flat[t[0]][1])) * 0.5
                if ar > best_area:
                    best_area = ar; best = ti
            if best < 0:
                return fallback()
            t = tris[best]
            cu = (flat[t[0]][0] + flat[t[1]][0] + flat[t[2]][0]) / 3.0
            cv = (flat[t[0]][1] + flat[t[1]][1] + flat[t[2]][1]) / 3.0
            return srf.point_at(cu, cv)

        def classify(X2, solid, solid_mesh, prim, is_first):
            kept, rev = [], []
            for fi in range(len(X2.m_faces)):
                if X2.m_faces[fi].surface_index < 0:
                    continue
                sp = face_sample(X2, fi)
                inside = _inside_prim(prim, sp) if prim else solid.contains_point(sp, solid_mesh)
                if op == 'union':
                    keep, r = (not inside), False
                elif op == 'intersection':
                    keep, r = inside, False
                else:  # difference
                    if is_first:
                        keep, r = (not inside), False
                    else:
                        keep, r = inside, True
                if keep:
                    kept.append(fi); rev.append(r)
            return kept, rev

        keptA, _ = classify(A2, other, meshB, primB, True)
        keptB, revB = classify(B2, self, meshA, primA, False)

        subA = A2.subset(keptA)
        subB = B2.subset(keptB)
        for k in range(min(len(revB), len(subB.m_faces))):
            if revB[k]:
                subB.m_faces[k].reversed = not subB.m_faces[k].reversed

        result = subA
        result.name = "boolean"
        voff = len(result.m_vertices); tvoff = len(result.m_topology_vertices)
        soff = len(result.m_surfaces); c2off = len(result.m_curves_2d)
        c3off = len(result.m_curves_3d); eoff = len(result.m_topology_edges)
        loff = len(result.m_loops); foff = len(result.m_faces); toff = len(result.m_trims)
        result.m_vertices.extend(subB.m_vertices)
        result.m_surfaces.extend(subB.m_surfaces)
        result.m_curves_2d.extend(subB.m_curves_2d)
        result.m_curves_3d.extend(subB.m_curves_3d)
        for tv in subB.m_topology_vertices:
            nv = BRepVertex(); nv.point_index = tv.point_index + voff; nv.edge_indices = []
            result.m_topology_vertices.append(nv)
        for e in subB.m_topology_edges:
            ne_ = BRepEdge()
            ne_.curve_3d_index = e.curve_3d_index + c3off if e.curve_3d_index >= 0 else e.curve_3d_index
            ne_.start_vertex = e.start_vertex + tvoff if e.start_vertex >= 0 else e.start_vertex
            ne_.end_vertex = e.end_vertex + tvoff if e.end_vertex >= 0 else e.end_vertex
            ne_.trim_indices = []
            result.m_topology_edges.append(ne_)
        for t in subB.m_trims:
            nt = BRepTrim()
            nt.curve_2d_index = t.curve_2d_index + c2off if t.curve_2d_index >= 0 else t.curve_2d_index
            nt.edge_index = t.edge_index + eoff if t.edge_index >= 0 else t.edge_index
            nt.loop_index = t.loop_index + loff if t.loop_index >= 0 else t.loop_index
            nt.reversed = t.reversed; nt.type = t.type
            result.m_trims.append(nt)
        for lp in subB.m_loops:
            nl = BRepLoop()
            nl.trim_indices = [ti + toff for ti in lp.trim_indices]
            nl.face_index = lp.face_index + foff if lp.face_index >= 0 else lp.face_index
            nl.type = lp.type
            result.m_loops.append(nl)
        for f in subB.m_faces:
            nf = BRepFace()
            nf.surface_index = f.surface_index + soff if f.surface_index >= 0 else f.surface_index
            nf.loop_indices = [li + loff for li in f.loop_indices]
            nf.reversed = f.reversed
            result.m_faces.append(nf)
        for e in result.m_topology_edges:
            e.trim_indices = []
        for ti in range(len(result.m_trims)):
            ei = result.m_trims[ti].edge_index
            if 0 <= ei < len(result.m_topology_edges):
                result.m_topology_edges[ei].trim_indices.append(ti)

        # Resolve T-junctions first (imprint), then co-refine the A<->B section so an under-mated
        # closed cut-circle is split at the endpoints of coincident open arcs (mate 1:1), then sew
        # the now arc-for-arc-identical segments. co_refine is the OCCT "shared section edge"
        # guarantee; gate it with SESSION_NO_COREFINE for A/B comparison.
        import os as _os
        result.imprint_edges()
        if not _os.environ.get("SESSION_NO_COREFINE"):
            result.co_refine_coincident_edges()
        result.sew_coincident_edges()
        return result

    def boolean_union(self, other, tolerance=None):
        return self.boolean(other, 'union', tolerance)

    def boolean_difference(self, other, tolerance=None):
        return self.boolean(other, 'difference', tolerance)

    def boolean_intersection(self, other, tolerance=None):
        return self.boolean(other, 'intersection', tolerance)

    def __add__(self, other):
        return self.boolean(other, 'union')

    def __sub__(self, other):
        return self.boolean(other, 'difference')

    def __and__(self, other):
        return self.boolean(other, 'intersection')

    ###########################################################################
    # Meshing
    ###########################################################################

    def mesh(self):
        from .mesh import Mesh
        from .nurbssurface_trimmed import NurbsSurfaceTrimmed
        nf = len(self.m_faces)

        # Phase 1: Classify faces as direct (RemeshNurbsSurfaceGrid) or CDT
        face_direct = [False] * nf
        for fi in range(nf):
            face = self.m_faces[fi]
            if face.surface_index < 0 or face.surface_index >= len(self.m_surfaces):
                continue
            srf = self.m_surfaces[face.surface_index]
            has_inner = False
            all_linear = True
            outer_pts = []
            for li in face.loop_indices:
                if li < 0 or li >= len(self.m_loops):
                    continue
                loop = self.m_loops[li]
                if loop.type == BRepLoopType.Inner:
                    has_inner = True
                for ti in loop.trim_indices:
                    if ti < 0 or ti >= len(self.m_trims):
                        continue
                    trim = self.m_trims[ti]
                    if trim.curve_2d_index < 0 or trim.curve_2d_index >= len(self.m_curves_2d):
                        continue
                    crv = self.m_curves_2d[trim.curve_2d_index]
                    if crv.degree() > 1 or crv.is_rational():
                        all_linear = False
                    if loop.type == BRepLoopType.Outer and crv.degree() <= 1 and not crv.is_rational():
                        for k in range(max(crv.cv_count() - 1, 0)):
                            p = crv.get_cv(k)
                            if p is not None:
                                outer_pts.append(p)
            direct = (not has_inner) and all_linear
            if direct and outer_pts:
                u0, u1 = srf.domain(0)
                v0, v1 = srf.domain(1)
                tol = max(u1 - u0, v1 - v0) * 0.01
                bb_umin = bb_vmin = 1e30
                bb_umax = bb_vmax = -1e30
                for p in outer_pts:
                    if p[0] < bb_umin: bb_umin = p[0]
                    if p[0] > bb_umax: bb_umax = p[0]
                    if p[1] < bb_vmin: bb_vmin = p[1]
                    if p[1] > bb_vmax: bb_vmax = p[1]
                if (abs(bb_umin - u0) > tol or abs(bb_umax - u1) > tol or
                    abs(bb_vmin - v0) > tol or abs(bb_vmax - v1) > tol):
                    direct = False
            face_direct[fi] = direct

        # Phase 2: Mesh direct faces, extract boundary 3D points for shared edges
        fmesh = [Mesh() for _ in range(nf)]
        edge_bnd = {}
        for fi in range(nf):
            if not face_direct[fi]:
                continue
            face = self.m_faces[fi]
            srf = self.m_surfaces[face.surface_index]
            fmesh[fi] = srf.mesh()

            u0, u1 = srf.domain(0)
            v0, v1 = srf.domain(1)
            utol = (u1 - u0) * 0.001
            vtol = (v1 - v0) * 0.001

            for li in face.loop_indices:
                if li < 0 or li >= len(self.m_loops):
                    continue
                for ti in self.m_loops[li].trim_indices:
                    if ti < 0 or ti >= len(self.m_trims):
                        continue
                    eidx = self.m_trims[ti].edge_index
                    if eidx < 0 or eidx >= len(self.m_topology_edges):
                        continue
                    if eidx in edge_bnd:
                        continue

                    shared = False
                    for oti in self.m_topology_edges[eidx].trim_indices:
                        if oti == ti:
                            continue
                        oli = self.m_trims[oti].loop_index
                        if oli < 0 or oli >= len(self.m_loops):
                            continue
                        ofi = self.m_loops[oli].face_index
                        if 0 <= ofi < nf and not face_direct[ofi]:
                            shared = True
                            break
                    if not shared:
                        continue

                    c2di = self.m_trims[ti].curve_2d_index
                    if c2di < 0 or c2di >= len(self.m_curves_2d):
                        continue
                    c2d = self.m_curves_2d[c2di]
                    sp = c2d.get_cv(0)
                    ep = c2d.get_cv(c2d.cv_count() - 1)

                    at_v0 = abs(sp[1] - v0) < vtol and abs(ep[1] - v0) < vtol
                    at_v1 = abs(sp[1] - v1) < vtol and abs(ep[1] - v1) < vtol
                    at_u0 = abs(sp[0] - u0) < utol and abs(ep[0] - u0) < utol
                    at_u1 = abs(sp[0] - u1) < utol and abs(ep[0] - u1) < utol
                    if not at_v0 and not at_v1 and not at_u0 and not at_u1:
                        continue

                    pts = []
                    for vk, vd in fmesh[fi].vertex.items():
                        if "u" not in vd.attributes or "v" not in vd.attributes:
                            continue
                        iu = vd.attributes["u"]
                        iv = vd.attributes["v"]
                        if at_v0 and abs(iv - v0) < vtol * 0.1:
                            pts.append((iu, vd.position()))
                        elif at_v1 and abs(iv - v1) < vtol * 0.1:
                            pts.append((iu, vd.position()))
                        elif at_u0 and abs(iu - u0) < utol * 0.1:
                            pts.append((iv, vd.position()))
                        elif at_u1 and abs(iu - u1) < utol * 0.1:
                            pts.append((iv, vd.position()))
                    pts.sort(key=lambda a: a[0])
                    if len(pts) >= 2:
                        edge_bnd[eidx] = [pt for _, pt in pts]

        # Phase 3: Mesh CDT faces, using matched boundary points for shared edges
        for fi in range(nf):
            if face_direct[fi]:
                continue
            face = self.m_faces[fi]
            if face.surface_index < 0 or face.surface_index >= len(self.m_surfaces):
                continue
            srf = self.m_surfaces[face.surface_index]

            p00 = srf.get_cv(0, 0)
            p10 = srf.get_cv(1, 0)
            p01 = srf.get_cv(0, 1)
            eux = p10[0] - p00[0]; euy = p10[1] - p00[1]; euz = p10[2] - p00[2]
            evx = p01[0] - p00[0]; evy = p01[1] - p00[1]; evz = p01[2] - p00[2]
            eu2 = eux * eux + euy * euy + euz * euz
            ev2 = evx * evx + evy * evy + evz * evz
            can_project = (srf.degree(0) == 1 and srf.degree(1) == 1 and eu2 > 1e-28 and ev2 > 1e-28)
            u0c, u1c = srf.domain(0); v0c, v1c = srf.domain(1)

            ts = NurbsSurfaceTrimmed()
            ts.m_surface = srf
            for li in face.loop_indices:
                if li < 0 or li >= len(self.m_loops):
                    continue
                loop = self.m_loops[li]
                loop_pts = []
                for ti in loop.trim_indices:
                    if ti < 0 or ti >= len(self.m_trims):
                        continue
                    trim = self.m_trims[ti]
                    if trim.type == BRepTrimType.Singular:
                        continue
                    eidx = trim.edge_index

                    if can_project and eidx >= 0 and eidx in edge_bnd:
                        for pt in edge_bnd[eidx]:
                            dx = pt[0] - p00[0]; dy = pt[1] - p00[1]; dz = pt[2] - p00[2]
                            u = (dx * eux + dy * euy + dz * euz) / eu2
                            v = (dx * evx + dy * evy + dz * evz) / ev2
                            loop_pts.append(Point(u, v, 0))
                    else:
                        if trim.curve_2d_index < 0 or trim.curve_2d_index >= len(self.m_curves_2d):
                            continue
                        crv = self.m_curves_2d[trim.curve_2d_index]
                        if crv.degree() <= 1 and not crv.is_rational():
                            # A UV-straight edge can still map to a CURVED 3D path when it
                            # spans the surface's curved parametric direction (e.g. a cylinder
                            # rim along the angular u). Sampling it by its 2 CVs lets CDT
                            # chord-cut straight across the interior (a spurious membrane);
                            # densify along the curved span so the band follows the surface.
                            sp = crv.get_cv(0); ep = crv.get_cv(crv.cv_count() - 1)
                            span = 0.0
                            if srf.degree(0) > 1 and (u1c - u0c) > 0:
                                span = max(span, abs(ep[0] - sp[0]) / (u1c - u0c))
                            if srf.degree(1) > 1 and (v1c - v0c) > 0:
                                span = max(span, abs(ep[1] - sp[1]) / (v1c - v0c))
                            if span > 1e-9:
                                n = max(int(round(span * 48)), 8)
                                pts, _ = crv.divide_by_count(n)
                                for k in range(len(pts) - 1):
                                    loop_pts.append(pts[k])
                            else:
                                for k in range(max(crv.cv_count() - 1, 0)):
                                    p = crv.get_cv(k)
                                    if p is not None:
                                        loop_pts.append(p)
                        else:
                            n = max(crv.cv_count() * 4, 16)
                            pts, _ = crv.divide_by_count(n)
                            for k in range(len(pts) - 1):
                                loop_pts.append(pts[k])
                if len(loop_pts) >= 3:
                    loop_crv = NurbsCurve.create(True, 1, loop_pts)
                    if loop.type == BRepLoopType.Outer:
                        ts.m_outer_loop = loop_crv
                    else:
                        ts.m_inner_loops.append(loop_crv)
            fmesh[fi] = ts.mesh()

        # Phase 4: Combine
        all_polygons = []
        for fi in range(nf):
            face = self.m_faces[fi]
            fm = fmesh[fi]
            if not fm.face:
                continue
            # Reversed faces must have their triangle winding flipped so the facet
            # orientation matches the face's outward normal (from_polylines rebuilds
            # vertices from positions, so flipping per-vertex normals here has no effect).
            for fk, fverts in fm.face.items():
                poly = [fm.vertex[vi].position() for vi in fverts]
                if face.reversed:
                    poly.reverse()
                all_polygons.append(poly)
        return Mesh.from_polylines(all_polygons, 1e-6)
    def face_meshes(self):
        return self.face_meshes_q(None)

    def face_meshes_q(self, quality=None):
        from .mesh import Mesh
        from .nurbssurface_trimmed import NurbsSurfaceTrimmed
        from .remesh_nurbssurface_grid import RemeshNurbsSurfaceGrid
        nf = len(self.m_faces)

        # Phase 1: classify
        face_direct = [False] * nf
        for fi in range(nf):
            face = self.m_faces[fi]
            if face.surface_index < 0 or face.surface_index >= len(self.m_surfaces):
                continue
            srf = self.m_surfaces[face.surface_index]
            has_inner = False
            all_linear = True
            outer_pts = []
            for li in face.loop_indices:
                if li < 0 or li >= len(self.m_loops):
                    continue
                loop = self.m_loops[li]
                if loop.type == BRepLoopType.Inner:
                    has_inner = True
                for ti in loop.trim_indices:
                    if ti < 0 or ti >= len(self.m_trims):
                        continue
                    trim = self.m_trims[ti]
                    if trim.curve_2d_index < 0 or trim.curve_2d_index >= len(self.m_curves_2d):
                        continue
                    crv = self.m_curves_2d[trim.curve_2d_index]
                    if crv.degree() > 1 or crv.is_rational():
                        all_linear = False
                    if loop.type == BRepLoopType.Outer and crv.degree() <= 1 and not crv.is_rational():
                        for k in range(max(crv.cv_count() - 1, 0)):
                            p = crv.get_cv(k)
                            if p is not None:
                                outer_pts.append(p)
            direct = (not has_inner) and all_linear
            if direct and outer_pts:
                u0, u1 = srf.domain(0)
                v0, v1 = srf.domain(1)
                tol = max(u1 - u0, v1 - v0) * 0.01
                bb_umin = bb_vmin = 1e30
                bb_umax = bb_vmax = -1e30
                for p in outer_pts:
                    if p[0] < bb_umin: bb_umin = p[0]
                    if p[0] > bb_umax: bb_umax = p[0]
                    if p[1] < bb_vmin: bb_vmin = p[1]
                    if p[1] > bb_vmax: bb_vmax = p[1]
                if (abs(bb_umin - u0) > tol or abs(bb_umax - u1) > tol or
                    abs(bb_vmin - v0) > tol or abs(bb_vmax - v1) > tol):
                    direct = False
            face_direct[fi] = direct

        # Phase 2: direct faces. Mesh each via the grid mesher, then record the 3D
        # boundary discretisation along every edge shared with a CDT face.
        fmesh = [Mesh() for _ in range(nf)]
        edge_bnd = {}
        for fi in range(nf):
            if not face_direct[fi]:
                continue
            face = self.m_faces[fi]
            srf = self.m_surfaces[face.surface_index]
            if quality is not None:
                a, c = quality
                fmesh[fi] = RemeshNurbsSurfaceGrid.from_u_v_q(srf, 0, 0, a, c)
            else:
                fmesh[fi] = srf.mesh()
            u0, u1 = srf.domain(0)
            v0, v1 = srf.domain(1)
            utol = (u1 - u0) * 0.001
            vtol = (v1 - v0) * 0.001
            for li in face.loop_indices:
                if li < 0 or li >= len(self.m_loops):
                    continue
                for ti in self.m_loops[li].trim_indices:
                    if ti < 0 or ti >= len(self.m_trims):
                        continue
                    eidx = self.m_trims[ti].edge_index
                    if eidx < 0 or eidx >= len(self.m_topology_edges):
                        continue
                    if eidx in edge_bnd:
                        continue
                    shared = False
                    for oti in self.m_topology_edges[eidx].trim_indices:
                        if oti == ti or oti < 0 or oti >= len(self.m_trims):
                            continue
                        oli = self.m_trims[oti].loop_index
                        if oli < 0 or oli >= len(self.m_loops):
                            continue
                        ofi = self.m_loops[oli].face_index
                        if ofi >= 0 and ofi < nf and not face_direct[ofi]:
                            shared = True
                            break
                    if not shared:
                        continue
                    c2di = self.m_trims[ti].curve_2d_index
                    if c2di < 0 or c2di >= len(self.m_curves_2d):
                        continue
                    c2d = self.m_curves_2d[c2di]
                    sp = c2d.get_cv(0)
                    ep = c2d.get_cv(max(c2d.cv_count() - 1, 0))
                    if sp is None or ep is None:
                        continue
                    at_v0 = abs(sp[1] - v0) < vtol and abs(ep[1] - v0) < vtol
                    at_v1 = abs(sp[1] - v1) < vtol and abs(ep[1] - v1) < vtol
                    at_u0 = abs(sp[0] - u0) < utol and abs(ep[0] - u0) < utol
                    at_u1 = abs(sp[0] - u1) < utol and abs(ep[0] - u1) < utol
                    if not at_v0 and not at_v1 and not at_u0 and not at_u1:
                        continue
                    pts = []
                    for vd in fmesh[fi].vertex.values():
                        if "u" not in vd.attributes or "v" not in vd.attributes:
                            continue
                        iu = vd.attributes["u"]
                        iv = vd.attributes["v"]
                        if at_v0 and abs(iv - v0) < vtol * 0.1:
                            pts.append((iu, vd.position()))
                        elif at_v1 and abs(iv - v1) < vtol * 0.1:
                            pts.append((iu, vd.position()))
                        elif at_u0 and abs(iu - u0) < utol * 0.1:
                            pts.append((iv, vd.position()))
                        elif at_u1 and abs(iu - u1) < utol * 0.1:
                            pts.append((iv, vd.position()))
                    pts.sort(key=lambda a: a[0])
                    if len(pts) >= 2:
                        edge_bnd[eidx] = [p for _, p in pts]

        # Phase 3: Mesh CDT faces via NurbsSurfaceTrimmed. For edges shared with a
        # direct face, reuse that face's boundary points projected into this bilinear UV.
        for fi in range(nf):
            if face_direct[fi]:
                continue
            face = self.m_faces[fi]
            if face.surface_index < 0 or face.surface_index >= len(self.m_surfaces):
                continue
            srf = self.m_surfaces[face.surface_index]
            proj = None
            a_cv = srf.get_cv(0, 0)
            b_cv = srf.get_cv(1, 0)
            c_cv = srf.get_cv(0, 1)
            if a_cv is not None and b_cv is not None and c_cv is not None and srf.degree(0) == 1 and srf.degree(1) == 1:
                eu = [b_cv[0] - a_cv[0], b_cv[1] - a_cv[1], b_cv[2] - a_cv[2]]
                ev = [c_cv[0] - a_cv[0], c_cv[1] - a_cv[1], c_cv[2] - a_cv[2]]
                eu2 = eu[0] * eu[0] + eu[1] * eu[1] + eu[2] * eu[2]
                ev2 = ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2]
                if eu2 > 1e-28 and ev2 > 1e-28:
                    proj = (a_cv, eu, ev, eu2, ev2)
            u0c, u1c = srf.domain(0); v0c, v1c = srf.domain(1)
            ts = NurbsSurfaceTrimmed()
            ts.m_surface = srf
            for li in face.loop_indices:
                if li < 0 or li >= len(self.m_loops):
                    continue
                loop = self.m_loops[li]
                loop_pts = []
                for ti in loop.trim_indices:
                    if ti < 0 or ti >= len(self.m_trims):
                        continue
                    trim = self.m_trims[ti]
                    if trim.type == BRepTrimType.Singular:
                        continue
                    eidx = trim.edge_index
                    if proj is not None and eidx >= 0 and eidx in edge_bnd:
                        a, eu, ev, eu2, ev2 = proj
                        for pt in edge_bnd[eidx]:
                            d = [pt[0] - a[0], pt[1] - a[1], pt[2] - a[2]]
                            u = (d[0] * eu[0] + d[1] * eu[1] + d[2] * eu[2]) / eu2
                            v = (d[0] * ev[0] + d[1] * ev[1] + d[2] * ev[2]) / ev2
                            loop_pts.append(Point(u, v, 0.0))
                    else:
                        if trim.curve_2d_index < 0 or trim.curve_2d_index >= len(self.m_curves_2d):
                            continue
                        crv = self.m_curves_2d[trim.curve_2d_index]
                        if crv.degree() <= 1 and not crv.is_rational():
                            # UV-straight edges spanning a curved parametric direction (e.g. a
                            # cylinder rim along the angular u) must be densified so CDT follows
                            # the surface instead of chord-cutting across the interior.
                            sp = crv.get_cv(0); ep = crv.get_cv(crv.cv_count() - 1)
                            span = 0.0
                            if srf.degree(0) > 1 and (u1c - u0c) > 0:
                                span = max(span, abs(ep[0] - sp[0]) / (u1c - u0c))
                            if srf.degree(1) > 1 and (v1c - v0c) > 0:
                                span = max(span, abs(ep[1] - sp[1]) / (v1c - v0c))
                            if span > 1e-9:
                                n = max(int(round(span * 48)), 8)
                                pts, _ = crv.divide_by_count(n)
                                for k in range(len(pts) - 1):
                                    loop_pts.append(pts[k])
                            else:
                                for k in range(max(crv.cv_count() - 1, 0)):
                                    p = crv.get_cv(k)
                                    if p is not None:
                                        loop_pts.append(p)
                        else:
                            n = max(crv.cv_count() * 4, 16)
                            pts, _ = crv.divide_by_count(n)
                            for k in range(len(pts) - 1):
                                loop_pts.append(pts[k])
                if len(loop_pts) >= 3:
                    loop_crv = NurbsCurve.create(True, 1, loop_pts)
                    if loop.type == BRepLoopType.Outer:
                        ts.m_outer_loop = loop_crv
                    else:
                        ts.m_inner_loops.append(loop_crv)
            fmesh[fi] = ts.mesh()

        # Apply reversed flag: flip BOTH winding and normals.
        for fi in range(nf):
            if self.m_faces[fi].reversed:
                fmesh[fi].flip()
                for vd in fmesh[fi].vertex.values():
                    n = vd.normal()
                    if n is not None:
                        vd.set_normal(-n[0], -n[1], -n[2])

        return fmesh


    ###########################################################################
    # Evaluation
    ###########################################################################

    def point_at(self, face_idx, u, v):
        if face_idx < 0 or face_idx >= len(self.m_faces):
            return Point()
        si = self.m_faces[face_idx].surface_index
        if si < 0 or si >= len(self.m_surfaces):
            return Point()
        return self.m_surfaces[si].point_at(u, v)

    def normal_at(self, face_idx, u, v):
        if face_idx < 0 or face_idx >= len(self.m_faces):
            return Vector()
        si = self.m_faces[face_idx].surface_index
        if si < 0 or si >= len(self.m_surfaces):
            return Vector()
        n = self.m_surfaces[si].normal_at(u, v)
        if self.m_faces[face_idx].reversed:
            return -n
        return n

    ###########################################################################
    # Transformation
    ###########################################################################

    def transform(self, xform=None):
        if xform is None:
            xform = self.xform
        for srf in self.m_surfaces:
            srf.xform = xform
            srf.transform()
        for crv in self.m_curves_3d:
            crv.xform = xform
            crv.transform()
        for i, pt in enumerate(self.m_vertices):
            m = xform.m
            x = m[0]*pt[0] + m[1]*pt[1] + m[2]*pt[2] + m[3]
            y = m[4]*pt[0] + m[5]*pt[1] + m[6]*pt[2] + m[7]
            z = m[8]*pt[0] + m[9]*pt[1] + m[10]*pt[2] + m[11]
            self.m_vertices[i] = Point(x, y, z)
        self.xform = Xform.identity()

    def transformed(self, xform=None):
        b = self.duplicate()
        b.transform(xform)
        return b

    ###########################################################################
    # JSON Serialization
    ###########################################################################

    def __jsondump__(self):
        j = {}
        j["curves_2d"] = [c.__jsondump__() for c in self.m_curves_2d]
        j["curves_3d"] = [c.__jsondump__() for c in self.m_curves_3d]
        faces = []
        for f in self.m_faces:
            d = {"loop_indices": f.loop_indices, "reversed": f.reversed, "surface_index": f.surface_index}
            if f.facecolor is not None:
                d = {"facecolor": f.facecolor.__jsondump__(), **d}
            faces.append(d)
        j["faces"] = faces
        j["guid"] = self.guid
        loops = []
        for l in self.m_loops:
            loops.append({"face_index": l.face_index, "trim_indices": l.trim_indices, "type": BRepLoopType.to_str(l.type)})
        j["loops"] = loops
        j["name"] = self.name
        j["surfaces"] = [s.__jsondump__() for s in self.m_surfaces]
        j["surfacecolor"] = self.surfacecolor.__jsondump__()
        edges = []
        for e in self.m_topology_edges:
            edges.append({"curve_3d_index": e.curve_3d_index, "end_vertex": e.end_vertex, "start_vertex": e.start_vertex, "trim_indices": e.trim_indices})
        j["topology_edges"] = edges
        verts = []
        for v in self.m_topology_vertices:
            verts.append({"edge_indices": v.edge_indices, "point_index": v.point_index})
        j["topology_vertices"] = verts
        trims = []
        for t in self.m_trims:
            trims.append({"curve_2d_index": t.curve_2d_index, "edge_index": t.edge_index, "loop_index": t.loop_index, "reversed": t.reversed, "type": BRepTrimType.to_str(t.type)})
        j["trims"] = trims
        j["type"] = "BRep"
        j["vertices"] = [[v[0], v[1], v[2]] for v in self.m_vertices]
        j["width"] = self.width
        j["xform"] = self.xform.__jsondump__()
        return j

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        b = cls()
        b.guid = guid if guid is not None else data.get("guid", b.guid)
        b.name = name if name is not None else data.get("name", "my_brep")
        b.width = data.get("width", 1.0)
        if "surfacecolor" in data:
            b.surfacecolor = Color.__jsonload__(data["surfacecolor"])
        if "xform" in data:
            b.xform = Xform.__jsonload__(data["xform"])
        if "curves_2d" in data:
            b.m_curves_2d = [NurbsCurve.__jsonload__(c) for c in data["curves_2d"]]
        if "curves_3d" in data:
            b.m_curves_3d = [NurbsCurve.__jsonload__(c) for c in data["curves_3d"]]
        if "surfaces" in data:
            b.m_surfaces = [NurbsSurface.__jsonload__(s) for s in data["surfaces"]]
        if "vertices" in data:
            b.m_vertices = [Point(v[0], v[1], v[2]) for v in data["vertices"]]
        if "topology_vertices" in data:
            for v in data["topology_vertices"]:
                tv = BRepVertex()
                tv.point_index = v["point_index"]
                tv.edge_indices = list(v["edge_indices"])
                b.m_topology_vertices.append(tv)
        if "topology_edges" in data:
            for e in data["topology_edges"]:
                te = BRepEdge()
                te.curve_3d_index = e["curve_3d_index"]
                te.start_vertex = e["start_vertex"]
                te.end_vertex = e["end_vertex"]
                te.trim_indices = list(e["trim_indices"])
                b.m_topology_edges.append(te)
        if "trims" in data:
            for t in data["trims"]:
                bt = BRepTrim()
                bt.curve_2d_index = t["curve_2d_index"]
                bt.edge_index = t["edge_index"]
                bt.loop_index = t["loop_index"]
                bt.reversed = t["reversed"]
                bt.type = BRepTrimType.from_str(t["type"])
                b.m_trims.append(bt)
        if "loops" in data:
            for l in data["loops"]:
                bl = BRepLoop()
                bl.face_index = l["face_index"]
                bl.trim_indices = list(l["trim_indices"])
                bl.type = BRepLoopType.from_str(l["type"])
                b.m_loops.append(bl)
        if "faces" in data:
            for f in data["faces"]:
                bf = BRepFace()
                bf.surface_index = f["surface_index"]
                bf.loop_indices = list(f["loop_indices"])
                bf.reversed = f["reversed"]
                if "facecolor" in f:
                    bf.facecolor = Color.__jsonload__(f["facecolor"])
                b.m_faces.append(bf)
        return b

    def file_json_dump(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=4)

    @classmethod
    def file_json_load(cls, filepath):
        import json
        with open(filepath, 'r') as f:
            return cls.__jsonload__(json.load(f))

    def file_json_dumps(self):
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, s):
        import json
        return cls.__jsonload__(json.loads(s))

    ###########################################################################
    # Protobuf Serialization
    ###########################################################################

    def pb_dumps(self):
        from .proto import brep_pb2
        proto = brep_pb2.BRep()
        proto.guid = self.guid
        proto.name = self.name
        proto.width = self.width
        for c in self.m_curves_2d:
            c.pb_fill(proto.curves_2d.add())
        for c in self.m_curves_3d:
            c.pb_fill(proto.curves_3d.add())
        for s in self.m_surfaces:
            s.pb_fill(proto.surfaces.add())
        for v in self.m_vertices:
            p = proto.vertices.add()
            p.x = v[0]; p.y = v[1]; p.z = v[2]
        for tv in self.m_topology_vertices:
            p = proto.topology_vertices.add()
            p.point_index = tv.point_index
            p.edge_indices.extend(tv.edge_indices)
        for te in self.m_topology_edges:
            p = proto.topology_edges.add()
            p.curve_3d_index = te.curve_3d_index
            p.start_vertex = te.start_vertex
            p.end_vertex = te.end_vertex
            p.trim_indices.extend(te.trim_indices)
        for t in self.m_trims:
            p = proto.trims.add()
            p.curve_2d_index = t.curve_2d_index
            p.edge_index = t.edge_index
            p.loop_index = t.loop_index
            p.reversed = t.reversed
            p.type = t.type
        for l in self.m_loops:
            p = proto.loops.add()
            p.trim_indices.extend(l.trim_indices)
            p.face_index = l.face_index
            p.type = l.type
        for f in self.m_faces:
            p = proto.faces.add()
            p.surface_index = f.surface_index
            p.loop_indices.extend(f.loop_indices)
            p.reversed = f.reversed
            if f.facecolor is not None:
                p.facecolor.r = f.facecolor.r
                p.facecolor.g = f.facecolor.g
                p.facecolor.b = f.facecolor.b
                p.facecolor.a = f.facecolor.a
        proto.surfacecolor.name = self.surfacecolor.name
        proto.surfacecolor.r = self.surfacecolor.r
        proto.surfacecolor.g = self.surfacecolor.g
        proto.surfacecolor.b = self.surfacecolor.b
        proto.surfacecolor.a = self.surfacecolor.a
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        from .proto import brep_pb2
        proto = brep_pb2.BRep()
        proto.ParseFromString(data)
        b = cls()
        b.guid = proto.guid
        b.name = proto.name
        b.width = proto.width
        for c in proto.curves_2d:
            b.m_curves_2d.append(NurbsCurve.pb_loads(c.SerializeToString()))
        for c in proto.curves_3d:
            b.m_curves_3d.append(NurbsCurve.pb_loads(c.SerializeToString()))
        for s in proto.surfaces:
            b.m_surfaces.append(NurbsSurface.pb_loads(s.SerializeToString()))
        for v in proto.vertices:
            b.m_vertices.append(Point(v.x, v.y, v.z))
        for tv in proto.topology_vertices:
            bv = BRepVertex()
            bv.point_index = tv.point_index
            bv.edge_indices = list(tv.edge_indices)
            b.m_topology_vertices.append(bv)
        for te in proto.topology_edges:
            be = BRepEdge()
            be.curve_3d_index = te.curve_3d_index
            be.start_vertex = te.start_vertex
            be.end_vertex = te.end_vertex
            be.trim_indices = list(te.trim_indices)
            b.m_topology_edges.append(be)
        for t in proto.trims:
            bt = BRepTrim()
            bt.curve_2d_index = t.curve_2d_index
            bt.edge_index = t.edge_index
            bt.loop_index = t.loop_index
            bt.reversed = t.reversed
            bt.type = t.type
            b.m_trims.append(bt)
        for l in proto.loops:
            bl = BRepLoop()
            bl.face_index = l.face_index
            bl.trim_indices = list(l.trim_indices)
            bl.type = l.type
            b.m_loops.append(bl)
        for f in proto.faces:
            bf = BRepFace()
            bf.surface_index = f.surface_index
            bf.loop_indices = list(f.loop_indices)
            bf.reversed = f.reversed
            if f.HasField('facecolor'):
                bf.facecolor = Color(f.facecolor.r, f.facecolor.g, f.facecolor.b, f.facecolor.a)
            b.m_faces.append(bf)
        cp = proto.surfacecolor
        b.surfacecolor = Color(cp.r, cp.g, cp.b, cp.a)
        b.surfacecolor.name = cp.name
        xp = proto.xform
        b.xform = Xform()
        b.xform.name = xp.name
        b.xform.m = list(xp.matrix)
        return b

    def pb_dump(self, filepath):
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath):
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())
