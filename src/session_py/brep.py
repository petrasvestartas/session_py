from __future__ import annotations
from typing import TYPE_CHECKING
import uuid
import copy
import math

from .point import Point
from .vector import Vector
from .color import Color
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface

if TYPE_CHECKING:
    from pathlib import Path
    from .xform import Xform
    from .mesh import Mesh
    from .polyline import Polyline


class BRepOrientation:
    """TopAbs_Orientation: carried by the parent -> child reference, never by the shape."""
    Forward = 0
    Reversed = 1
    Internal = 2
    External = 3

    _TO_STR = {0: "forward", 1: "reversed", 2: "internal", 3: "external"}
    _FROM_STR = {"forward": 0, "reversed": 1, "internal": 2, "external": 3}

    @staticmethod
    def to_str(v: int) -> str:
        return BRepOrientation._TO_STR.get(v, "forward")

    @staticmethod
    def from_str(s: str) -> int:
        return BRepOrientation._FROM_STR.get(s, 0)


def brep_reverse(o: int) -> int:
    """TopAbs::Reverse."""
    if o == BRepOrientation.Forward:
        return BRepOrientation.Reversed
    if o == BRepOrientation.Reversed:
        return BRepOrientation.Forward
    return o


def brep_compose(a: int, b: int) -> int:
    """TopAbs::Compose: the orientation of a sub-shape reached through a parent with orientation `a`."""
    if a == BRepOrientation.Internal or a == BRepOrientation.External:
        return a
    if a == BRepOrientation.Forward:
        return b
    return brep_reverse(b)


class BRepRef:
    """TopoDS_Shape: an oriented reference to a sub-shape (index into the owning table)."""
    def __init__(self, index: int = -1, orientation: int = BRepOrientation.Forward):
        self.index = index
        self.orientation = orientation

    def __eq__(self, other):
        return isinstance(other, BRepRef) and self.index == other.index and self.orientation == other.orientation


class BRepVertex:
    """BRep_TVertex."""
    def __init__(self, point: Point | None = None, tolerance: float = 0.0):
        self.point = point if point is not None else Point(0.0, 0.0, 0.0)
        self.tolerance = tolerance


class BRepCurveOnSurface:
    """BRep_CurveOnSurface / BRep_CurveOnClosedSurface. curve_2d_index_2 is the pcurve of the
    REVERSED use of the edge on a closed surface (seam); -1 otherwise. Pcurves run in the
    edge's own direction (OCCT SameParameter convention)."""
    def __init__(self, surface_index: int = -1, curve_2d_index: int = -1, curve_2d_index_2: int = -1):
        self.surface_index = surface_index
        self.curve_2d_index = curve_2d_index
        self.curve_2d_index_2 = curve_2d_index_2


class BRepEdge:
    """BRep_TEdge. curve_3d_index is -1 for a degenerated edge (sphere pole, cone apex)."""
    def __init__(self):
        self.curve_3d_index = -1
        self.start_vertex = -1
        self.end_vertex = -1
        self.tolerance = 0.0
        self.degenerated = False
        self.pcurves: list[BRepCurveOnSurface] = []


class BRepWire:
    """TopoDS_TWire."""
    def __init__(self, edges: list[BRepRef] | None = None):
        self.edges: list[BRepRef] = list(edges) if edges else []


class BRepFace:
    """BRep_TFace. The first wire is the outer boundary."""
    def __init__(self):
        self.surface_index = -1
        self.wires: list[BRepRef] = []
        self.tolerance = 0.0
        self.facecolor = None


class BRepShell:
    """TopoDS_TShell."""
    def __init__(self, faces: list[BRepRef] | None = None):
        self.faces: list[BRepRef] = list(faces) if faces else []


class BRepSolid:
    """TopoDS_TSolid."""
    def __init__(self, shells: list[BRepRef] | None = None):
        self.shells: list[BRepRef] = list(shells) if shells else []


F = BRepOrientation.Forward
R = BRepOrientation.Reversed


def _bilinear_patch(p00: Point, p10: Point, p01: Point, p11: Point) -> NurbsSurface:
    """Bilinear planar patch: u runs p00 -> p10, v runs p00 -> p01, natural normal = u x v."""
    srf = NurbsSurface(3, False, 2, 2, 2, 2)
    srf.set_cv(0, 0, p00)
    srf.set_cv(1, 0, p10)
    srf.set_cv(0, 1, p01)
    srf.set_cv(1, 1, p11)
    return srf


def _uv_line(u0: float, v0: float, u1: float, v1: float) -> NurbsCurve:
    """Straight pcurve from (u0, v0) to (u1, v1)."""
    return NurbsCurve.create(False, 1, [Point(u0, v0, 0.0), Point(u1, v1, 0.0)])


def _project_to_patch(crv: NurbsCurve, srf: NurbsSurface) -> NurbsCurve:
    """Exact pcurve of a 3D curve lying on a bilinear planar patch: the affine image of its CVs."""
    p00 = srf.get_cv(0, 0)
    p10 = srf.get_cv(1, 0)
    p01 = srf.get_cv(0, 1)
    eu = [p10[0] - p00[0], p10[1] - p00[1], p10[2] - p00[2]]
    ev = [p01[0] - p00[0], p01[1] - p00[1], p01[2] - p00[2]]
    eu2 = eu[0] * eu[0] + eu[1] * eu[1] + eu[2] * eu[2]
    ev2 = ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2]
    c2 = NurbsCurve(3, crv.is_rational(), crv.order(), crv.cv_count())
    for i in range(crv.nurbsknot_count()):
        c2.set_nurbsknot(i, crv.nurbsknot(i))
    for i in range(crv.cv_count()):
        wx, wy, wz, w = crv.get_cv_4d(i)
        dx = wx / w - p00[0]
        dy = wy / w - p00[1]
        dz = wz / w - p00[2]
        u = (dx * eu[0] + dy * eu[1] + dz * eu[2]) / eu2
        v = (dx * ev[0] + dy * ev[1] + dz * ev[2]) / ev2
        if crv.is_rational():
            c2.set_cv_4d(i, u * w, v * w, 0.0, w)
        else:
            c2.set_cv(i, Point(u, v, 0.0))
    return c2


def _uv_signed_area(c2d: NurbsCurve) -> float:
    """Signed area of a closed pcurve's sampled polygon (positive = counter-clockwise)."""
    pts, _ = c2d.divide_by_count(max(c2d.cv_count() * 4, 16))
    a = 0.0
    for i in range(len(pts) - 1):
        a += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]
    return 0.5 * a


def _polygon_signed_area(pts: list[Point]) -> float:
    a = 0.0
    n = len(pts)
    for i in range(n):
        p = pts[i]
        q = pts[(i + 1) % n]
        a += p[0] * q[1] - q[0] * p[1]
    return 0.5 * a


class _PolyFaceBuilder:
    """Shared builder for planar polygon faces whose vertices, edges (lo -> hi vertex) and
    surfaces are made from a point table. The face order lists vertices counter-clockwise
    seen from the outside, so the natural normal of the patch points outward (Forward face)."""
    def __init__(self, b: "BRep"):
        self.b = b
        self.edge_map: dict[tuple[int, int], int] = {}

    def edge(self, v0: int, v1: int) -> int:
        lo, hi = min(v0, v1), max(v0, v1)
        if (lo, hi) in self.edge_map:
            return self.edge_map[(lo, hi)]
        line = NurbsCurve.create(False, 1, [self.b.m_vertices[lo].point, self.b.m_vertices[hi].point])
        ei = self.b.add_edge(self.b.add_curve_3d(line), lo, hi)
        self.edge_map[(lo, hi)] = ei
        return ei

    def wire_refs(self, srf_index: int, vi: list[int]) -> list[BRepRef]:
        b = self.b
        srf = b.m_surfaces[srf_index]
        refs = []
        n = len(vi)
        for i in range(n):
            va, vb = vi[i], vi[(i + 1) % n]
            ei = self.edge(va, vb)
            b.add_pcurve(ei, srf_index, b.add_curve_2d(_project_to_patch(b.m_curves_3d[b.m_edges[ei].curve_3d_index], srf)))
            refs.append(BRepRef(ei, F if b.m_edges[ei].start_vertex == va else R))
        return refs

    def face(self, srf: NurbsSurface, vi: list[int]) -> int:
        si = self.b.add_surface(srf)
        return self.b.add_face(si, [BRepRef(self.b.add_wire(self.wire_refs(si, vi)), F)])


_BOX_FACES = [
    [0, 3, 2, 1],  # bottom (z=-hz), normal -Z
    [4, 5, 6, 7],  # top (z=+hz), normal +Z
    [0, 1, 5, 4],  # front (y=-hy), normal -Y
    [1, 2, 6, 5],  # right (x=+hx), normal +X
    [2, 3, 7, 6],  # back (y=+hy), normal +Y
    [3, 0, 4, 7],  # left (x=-hx), normal -X
]


def _quad_patch(b: "BRep", fv: list[int]) -> NurbsSurface:
    """Bilinear patch spanned by four vertex indices in face order (p00, p10, p11, p01)."""
    return _bilinear_patch(b.m_vertices[fv[0]].point, b.m_vertices[fv[1]].point,
                           b.m_vertices[fv[3]].point, b.m_vertices[fv[2]].point)


def _box_corners(b: "BRep", sx: float, sy: float, sz: float) -> None:
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    b.add_vertex(Point(-hx, -hy, -hz))
    b.add_vertex(Point(hx, -hy, -hz))
    b.add_vertex(Point(hx, hy, -hz))
    b.add_vertex(Point(-hx, hy, -hz))
    b.add_vertex(Point(-hx, -hy, hz))
    b.add_vertex(Point(hx, -hy, hz))
    b.add_vertex(Point(hx, hy, hz))
    b.add_vertex(Point(-hx, hy, hz))


def _cap_patch(r: float, z: float, up: bool) -> NurbsSurface:
    """Planar cap at height z with natural normal +Z (up) or -Z (down), spanning [-r, r]^2."""
    if up:
        return _bilinear_patch(Point(-r, -r, z), Point(r, -r, z), Point(-r, r, z), Point(r, r, z))
    return _bilinear_patch(Point(-r, -r, z), Point(-r, r, z), Point(r, -r, z), Point(r, r, z))


def _cap_face(b: "BRep", cap: NurbsSurface, edge: int) -> int:
    """Cap face bounded by one closed edge: outer wire counter-clockwise in the patch's UV."""
    si = b.add_surface(cap)
    c2d = _project_to_patch(b.m_curves_3d[b.m_edges[edge].curve_3d_index], cap)
    o = F if _uv_signed_area(c2d) > 0.0 else R
    b.add_pcurve(edge, si, b.add_curve_2d(c2d))
    return b.add_face(si, [BRepRef(b.add_wire([BRepRef(edge, o)]), F)])


def _body_face(b: "BRep", si: int, e_bot: int, e_seam: int, e_top: int) -> int:
    """Periodic body face (cylinder / cone / bore): seam from v0 to v1 at u0 == u1, bottom ring
    forward at v0, top ring (or degenerated apex) reversed at v1."""
    u0, u1 = b.m_surfaces[si].domain(0)
    v0, v1 = b.m_surfaces[si].domain(1)
    b.add_pcurve(e_bot, si, b.add_curve_2d(_uv_line(u0, v0, u1, v0)))
    b.add_pcurve(e_top, si, b.add_curve_2d(_uv_line(u0, v1, u1, v1)))
    b.add_pcurve(e_seam, si, b.add_curve_2d(_uv_line(u1, v0, u1, v1)), b.add_curve_2d(_uv_line(u0, v0, u0, v1)))
    wire = b.add_wire([BRepRef(e_bot, F), BRepRef(e_seam, F), BRepRef(e_top, R), BRepRef(e_seam, R)])
    return b.add_face(si, [BRepRef(wire, F)])


def _planar_patch_through(pts: list[Point], org: Point, xa: Vector, ya: Vector) -> NurbsSurface:
    """Padded bilinear patch through `pts` in the plane (org, xa, ya)."""
    umin = vmin = 1e30
    umax = vmax = -1e30
    for p in pts:
        dx, dy, dz = p[0] - org[0], p[1] - org[1], p[2] - org[2]
        u = dx * xa[0] + dy * xa[1] + dz * xa[2]
        v = dx * ya[0] + dy * ya[1] + dz * ya[2]
        umin, umax = min(umin, u), max(umax, u)
        vmin, vmax = min(vmin, v), max(vmax, v)
    pad = max(umax - umin, vmax - vmin) * 0.01
    umin -= pad
    umax += pad
    vmin -= pad
    vmax += pad

    def pt3d(u, v):
        return Point(org[0] + u * xa[0] + v * ya[0], org[1] + u * xa[1] + v * ya[1], org[2] + u * xa[2] + v * ya[2])

    return _bilinear_patch(pt3d(umin, vmin), pt3d(umax, vmin), pt3d(umin, vmax), pt3d(umax, vmax))


def _find_or_add_vertex(b: "BRep", p: Point, tol: float) -> int:
    for i, v in enumerate(b.m_vertices):
        if v.point.distance(p) < tol:
            return i
    return b.add_vertex(p)


def _cv_points(c: NurbsCurve) -> list[Point]:
    pts = []
    for k in range(c.cv_count()):
        wx, wy, wz, w = c.get_cv_4d(k)
        if w != 0.0:
            pts.append(Point(wx / w, wy / w, wz / w))
    return pts


def _signed_volume(meshes: list["Mesh"]) -> float:
    """Signed volume of face meshes (positive when the windings point outward)."""
    total = 0.0
    for fm in meshes:
        for fverts in fm.face.values():
            a = fm.vertex[fverts[0]].position()
            for k in range(1, len(fverts) - 1):
                b = fm.vertex[fverts[k]].position()
                c = fm.vertex[fverts[k + 1]].position()
                total += (a[0] * (b[1] * c[2] - b[2] * c[1])
                          - a[1] * (b[0] * c[2] - b[2] * c[0])
                          + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return total / 6.0


def _close_free_faces(b: "BRep") -> None:
    """BRepBuilderAPI_Sewing + MakeSolid for free faces: when every edge is shared by exactly two
    face uses, orient the faces consistently across shared edges, one shell per connected
    component wound outward, and one solid per shell."""
    nf = b.face_count()
    if nf == 0:
        return
    uses: list[list[tuple[int, int]]] = [[] for _ in b.m_edges]
    for fi in range(nf):
        for wr in b.m_faces[fi].wires:
            for er in b.wire_edges(wr):
                uses[er.index].append((fi, er.orientation))
    for u in uses:
        if len(u) != 2:
            return
    fo = [F] * nf
    seen = [False] * nf
    components = []
    for seed in range(nf):
        if seen[seed]:
            continue
        comp = []
        stack = [seed]
        seen[seed] = True
        while stack:
            fi = stack.pop()
            comp.append(fi)
            for wr in b.m_faces[fi].wires:
                for er in b.wire_edges(wr):
                    for g, og in uses[er.index]:
                        if g == fi or seen[g]:
                            continue
                        fo[g] = brep_reverse(fo[fi]) if og == er.orientation else fo[fi]
                        seen[g] = True
                        stack.append(g)
        components.append(comp)
    shells = []
    for comp in components:
        shells.append(BRepRef(b.add_shell([BRepRef(fi, fo[fi]) for fi in comp]), F))
    fm = b.face_meshes()
    for sr in shells:
        part = [fm[fr.index] for fr in b.m_shells[sr.index].faces]
        if _signed_volume(part) < 0.0:
            for fr in b.m_shells[sr.index].faces:
                fr.orientation = brep_reverse(fr.orientation)
        b.add_solid([sr])


class BRep:
    """Boundary representation after OCCT's TopoDS/BRep model, with indexed tables.

    Geometry pools (surfaces, 3D curves, 2D pcurves) and shape tables (vertices, edges, wires,
    faces, shells, solids). Every parent -> child link is a BRepRef carrying the orientation.
    The BRep itself is the compound: its free shapes are those no parent references."""

    def __init__(self):
        self._guid = None
        self.name = "my_brep"
        self.width = 1.0
        self._surfacecolor = None
        self.m_surfaces: list[NurbsSurface] = []
        self.m_curves_3d: list[NurbsCurve] = []
        self.m_curves_2d: list[NurbsCurve] = []
        self.m_vertices: list[BRepVertex] = []
        self.m_edges: list[BRepEdge] = []
        self.m_wires: list[BRepWire] = []
        self.m_faces: list[BRepFace] = []
        self.m_shells: list[BRepShell] = []
        self.m_solids: list[BRepSolid] = []

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def refresh_guid(self) -> None:
        """Clear the guid so a FRESH one mints lazily on next read."""
        self._guid = None

    @property
    def surfacecolor(self) -> "Color":
        if self._surfacecolor is None:
            self._surfacecolor = Color.black()
        return self._surfacecolor

    @surfacecolor.setter
    def surfacecolor(self, value: "Color") -> None:
        self._surfacecolor = value

    def __str__(self):
        return f"BRep(name={self.name}, faces={self.face_count()}, edges={self.edge_count()}, vertices={self.vertex_count()})"

    def __repr__(self):
        return f"BRep(\n  name={self.name},\n  faces={self.face_count()},\n  edges={self.edge_count()},\n  vertices={self.vertex_count()},\n  solid={'true' if self.is_solid() else 'false'}\n)"

    def __eq__(self, other):
        if not isinstance(other, BRep):
            return False
        return (self.name == other.name and self.width == other.width
                and self.surfacecolor == other.surfacecolor
                and len(self.m_surfaces) == len(other.m_surfaces)
                and len(self.m_vertices) == len(other.m_vertices)
                and len(self.m_edges) == len(other.m_edges)
                and len(self.m_wires) == len(other.m_wires)
                and len(self.m_faces) == len(other.m_faces)
                and len(self.m_shells) == len(other.m_shells)
                and len(self.m_solids) == len(other.m_solids))

    def __ne__(self, other):
        return not self.__eq__(other)

    def duplicate(self) -> "BRep":
        b = copy.deepcopy(self)
        b._guid = str(uuid.uuid4())
        return b

    ###########################################################################
    # Static Factory Methods
    ###########################################################################

    @staticmethod
    def create_box(sx: float, sy: float, sz: float) -> "BRep":
        """Axis-aligned box centered at the origin: 6 faces, 12 edges, 8 vertices, one solid."""
        b = BRep()
        b.name = "box"
        _box_corners(b, sx, sy, sz)
        pb = _PolyFaceBuilder(b)
        faces = [BRepRef(pb.face(_quad_patch(b, fv), fv), F) for fv in _BOX_FACES]
        b.add_solid([BRepRef(b.add_shell(faces), F)])
        return b

    @staticmethod
    def create_cylinder(radius: float, height: float) -> "BRep":
        """Cylinder along +Z: one periodic body face (seam edge) and two planar caps."""
        from .primitives import Primitives
        b = BRep()
        b.name = "cylinder"
        body = Primitives.cylinder_surface(0, 0, 0, radius, height)
        p_bot = body.point_at_corner(0, 0)
        p_top = body.point_at_corner(0, 1)
        v_bot = b.add_vertex(p_bot)
        v_top = b.add_vertex(p_top)
        e_bot = b.add_edge(b.add_curve_3d(Primitives.circle(0, 0, 0, radius)), v_bot, v_bot)
        e_top = b.add_edge(b.add_curve_3d(Primitives.circle(0, 0, height, radius)), v_top, v_top)
        e_seam = b.add_edge(b.add_curve_3d(NurbsCurve.create(False, 1, [p_bot, p_top])), v_bot, v_top)
        f_body = _body_face(b, b.add_surface(body), e_bot, e_seam, e_top)
        f_bot = _cap_face(b, _cap_patch(radius, 0.0, False), e_bot)
        f_top = _cap_face(b, _cap_patch(radius, height, True), e_top)
        b.add_solid([BRepRef(b.add_shell([BRepRef(f_body, F), BRepRef(f_bot, F), BRepRef(f_top, F)]), F)])
        return b

    @staticmethod
    def create_sphere(radius: float) -> "BRep":
        """Sphere centered at the origin: one face, a seam meridian and two degenerated pole edges."""
        from .primitives import Primitives
        b = BRep()
        b.name = "sphere"
        srf = Primitives.sphere_surface(0, 0, 0, radius)
        u0, u1 = srf.domain(0)
        v0, v1 = srf.domain(1)
        v_s = b.add_vertex(Point(0.0, 0.0, -radius))
        v_n = b.add_vertex(Point(0.0, 0.0, radius))
        e_seam = b.add_edge(b.add_curve_3d(srf.iso_curve(1, u0)), v_s, v_n)
        e_south = b.add_edge(-1, v_s, v_s)
        e_north = b.add_edge(-1, v_n, v_n)
        si = b.add_surface(srf)
        b.add_pcurve(e_south, si, b.add_curve_2d(_uv_line(u0, v0, u1, v0)))
        b.add_pcurve(e_north, si, b.add_curve_2d(_uv_line(u0, v1, u1, v1)))
        b.add_pcurve(e_seam, si, b.add_curve_2d(_uv_line(u1, v0, u1, v1)), b.add_curve_2d(_uv_line(u0, v0, u0, v1)))
        wire = b.add_wire([BRepRef(e_south, F), BRepRef(e_seam, F), BRepRef(e_north, R), BRepRef(e_seam, R)])
        fi = b.add_face(si, [BRepRef(wire, F)])
        b.add_solid([BRepRef(b.add_shell([BRepRef(fi, F)]), F)])
        return b

    @staticmethod
    def create_cone(radius: float, height: float) -> "BRep":
        """Cone along +Z: base circle at z=0, apex at z=height (degenerated apex edge), planar base."""
        from .primitives import Primitives
        b = BRep()
        b.name = "cone"
        body = Primitives.cone_surface(0, 0, 0, radius, height)
        p_base = body.point_at_corner(0, 0)
        p_apex = Point(0.0, 0.0, height)
        v_base = b.add_vertex(p_base)
        v_apex = b.add_vertex(p_apex)
        e_base = b.add_edge(b.add_curve_3d(Primitives.circle(0, 0, 0, radius)), v_base, v_base)
        e_seam = b.add_edge(b.add_curve_3d(NurbsCurve.create(False, 1, [p_base, p_apex])), v_base, v_apex)
        e_apex = b.add_edge(-1, v_apex, v_apex)
        f_body = _body_face(b, b.add_surface(body), e_base, e_seam, e_apex)
        f_base = _cap_face(b, _cap_patch(radius, 0.0, False), e_base)
        b.add_solid([BRepRef(b.add_shell([BRepRef(f_body, F), BRepRef(f_base, F)]), F)])
        return b

    @staticmethod
    def create_pyramid(base: float, height: float) -> "BRep":
        """Square pyramid: base edge `base` centered at the origin in z=0, apex at (0,0,height)."""
        b = BRep()
        b.name = "pyramid"
        h = base * 0.5
        b.add_vertex(Point(-h, -h, 0.0))
        b.add_vertex(Point(h, -h, 0.0))
        b.add_vertex(Point(h, h, 0.0))
        b.add_vertex(Point(-h, h, 0.0))
        v_apex = b.add_vertex(Point(0.0, 0.0, height))
        pb = _PolyFaceBuilder(b)
        fv = [0, 3, 2, 1]
        faces = [BRepRef(pb.face(_quad_patch(b, fv), fv), F)]
        for i in range(4):
            a, c = i, (i + 1) % 4
            srf = _bilinear_patch(b.m_vertices[a].point, b.m_vertices[c].point, b.m_vertices[v_apex].point, b.m_vertices[v_apex].point)
            si = b.add_surface(srf)
            e_ac, e_c, e_a = pb.edge(a, c), pb.edge(c, v_apex), pb.edge(a, v_apex)
            e_deg = b.add_edge(-1, v_apex, v_apex)
            ac_fwd = b.m_edges[e_ac].start_vertex == a
            b.add_pcurve(e_ac, si, b.add_curve_2d(_uv_line(0, 0, 1, 0) if ac_fwd else _uv_line(1, 0, 0, 0)))
            b.add_pcurve(e_c, si, b.add_curve_2d(_uv_line(1, 0, 1, 1)))
            b.add_pcurve(e_deg, si, b.add_curve_2d(_uv_line(1, 1, 0, 1)))
            b.add_pcurve(e_a, si, b.add_curve_2d(_uv_line(0, 0, 0, 1)))
            wire = b.add_wire([BRepRef(e_ac, F if ac_fwd else R), BRepRef(e_c, F), BRepRef(e_deg, F), BRepRef(e_a, R)])
            faces.append(BRepRef(b.add_face(si, [BRepRef(wire, F)]), F))
        b.add_solid([BRepRef(b.add_shell(faces), F)])
        return b

    @staticmethod
    def create_torus(major_radius: float, minor_radius: float) -> "BRep":
        """Torus in the XY plane: one face closed in both directions, two seam edges, one vertex."""
        from .primitives import Primitives
        b = BRep()
        b.name = "torus"
        srf = Primitives.torus_surface(0, 0, 0, major_radius, minor_radius)
        u0, u1 = srf.domain(0)
        v0, v1 = srf.domain(1)
        v = b.add_vertex(srf.point_at_corner(0, 0))
        e_u = b.add_edge(b.add_curve_3d(srf.iso_curve(1, u0)), v, v)   # minor circle at u0
        e_v = b.add_edge(b.add_curve_3d(srf.iso_curve(0, v0)), v, v)   # major circle at v0
        si = b.add_surface(srf)
        b.add_pcurve(e_v, si, b.add_curve_2d(_uv_line(u0, v0, u1, v0)), b.add_curve_2d(_uv_line(u0, v1, u1, v1)))
        b.add_pcurve(e_u, si, b.add_curve_2d(_uv_line(u1, v0, u1, v1)), b.add_curve_2d(_uv_line(u0, v0, u0, v1)))
        wire = b.add_wire([BRepRef(e_v, F), BRepRef(e_u, F), BRepRef(e_v, R), BRepRef(e_u, R)])
        fi = b.add_face(si, [BRepRef(wire, F)])
        b.add_solid([BRepRef(b.add_shell([BRepRef(fi, F)]), F)])
        return b

    @staticmethod
    def create_block_with_hole(sx: float, sy: float, sz: float, hole_radius: float) -> "BRep":
        """Axis-aligned box with a cylindrical through-hole along Z."""
        from .primitives import Primitives
        b = BRep()
        b.name = "block_with_hole"
        hz = sz * 0.5
        _box_corners(b, sx, sy, sz)
        pb = _PolyFaceBuilder(b)
        faces = [BRepRef(pb.face(_quad_patch(b, fv), fv), F) for fv in _BOX_FACES[2:]]
        p_bot = Point(hole_radius, 0.0, -hz)
        p_top = Point(hole_radius, 0.0, hz)
        v_bot = b.add_vertex(p_bot)
        v_top = b.add_vertex(p_top)
        e_bot = b.add_edge(b.add_curve_3d(Primitives.circle(0, 0, -hz, hole_radius)), v_bot, v_bot)
        e_top = b.add_edge(b.add_curve_3d(Primitives.circle(0, 0, hz, hole_radius)), v_top, v_top)
        e_seam = b.add_edge(b.add_curve_3d(NurbsCurve.create(False, 1, [p_bot, p_top])), v_bot, v_top)
        bore = Primitives.cylinder_surface(0, 0, -hz, hole_radius, sz)
        faces.append(BRepRef(_body_face(b, b.add_surface(bore), e_bot, e_seam, e_top), R))
        for fi in range(2):
            fv = _BOX_FACES[fi]
            cap = _quad_patch(b, fv)
            si = b.add_surface(cap)
            outer = pb.wire_refs(si, fv)
            e_hole = e_bot if fi == 0 else e_top
            c2d = _project_to_patch(b.m_curves_3d[b.m_edges[e_hole].curve_3d_index], cap)
            o = F if _uv_signed_area(c2d) < 0.0 else R
            b.add_pcurve(e_hole, si, b.add_curve_2d(c2d))
            faces.append(BRepRef(b.add_face(si, [BRepRef(b.add_wire(outer), F), BRepRef(b.add_wire([BRepRef(e_hole, o)]), F)]), F))
        b.add_solid([BRepRef(b.add_shell(faces), F)])
        return b

    @staticmethod
    def from_polylines(polylines: list["Polyline"]) -> "BRep":
        """One planar face per closed polyline; coincident vertices and edges are shared."""
        b = BRep()
        b.name = "polysurface"
        tol = 1e-6
        pb = _PolyFaceBuilder(b)
        for pl in polylines:
            pts = pl.get_points()
            n = len(pts) - 1 if pl.is_closed() else len(pts)
            if n < 3:
                continue
            org, plane = pl.get_fast_plane()
            if not plane.is_valid():
                continue
            vi = [_find_or_add_vertex(b, pts[i], tol) for i in range(n)]
            pb.face(_planar_patch_through(pts[:n], org, plane.x_axis, plane.y_axis), vi)
        _close_free_faces(b)
        return b

    @staticmethod
    def from_nurbscurves(curves: list["NurbsCurve"], holes: list[list["NurbsCurve"]] | None = None) -> "BRep":
        """One planar face per closed curve with optional hole curves (inner wires)."""
        from .polyline import Polyline
        b = BRep()
        b.name = "polysurface"
        tol = 1e-6
        holes = holes or []

        def curve_wire(crv: NurbsCurve, si: int) -> int:
            sp = crv.point_at(crv.domain()[0])
            ep = crv.point_at(crv.domain()[1])
            vs = _find_or_add_vertex(b, sp, tol)
            ve = vs if crv.is_closed() else _find_or_add_vertex(b, ep, tol)
            ei = b.add_edge(b.add_curve_3d(crv), vs, ve)
            b.add_pcurve(ei, si, b.add_curve_2d(_project_to_patch(crv, b.m_surfaces[si])))
            return b.add_wire([BRepRef(ei, F)])

        for ci, crv in enumerate(curves):
            pts = _cv_points(crv)
            if len(pts) >= 2 and pts[0].distance(pts[-1]) < tol:
                pts.pop()
            if len(pts) < 3:
                continue
            org, plane = Polyline(pts).get_fast_plane()
            if not plane.is_valid():
                continue
            if ci < len(holes):
                for h in holes[ci]:
                    pts.extend(_cv_points(h))
            si = b.add_surface(_planar_patch_through(pts, org, plane.x_axis, plane.y_axis))
            wires = [BRepRef(curve_wire(crv, si), F)]
            if ci < len(holes):
                for h in holes[ci]:
                    wires.append(BRepRef(curve_wire(h, si), F))
            b.add_face(si, wires)
        _close_free_faces(b)
        return b

    ###########################################################################
    # Accessors
    ###########################################################################

    def vertex_count(self) -> int:
        return len(self.m_vertices)

    def edge_count(self) -> int:
        return len(self.m_edges)

    def wire_count(self) -> int:
        return len(self.m_wires)

    def face_count(self) -> int:
        return len(self.m_faces)

    def shell_count(self) -> int:
        return len(self.m_shells)

    def solid_count(self) -> int:
        return len(self.m_solids)

    def is_valid(self) -> bool:
        """Every reference resolves into its table, every face has a surface and an outer wire,
        every edge has two vertices and (unless degenerated) a 3D curve."""
        if not self.m_faces:
            return False

        def ok(i, n):
            return 0 <= i < n

        for e in self.m_edges:
            if not ok(e.start_vertex, len(self.m_vertices)) or not ok(e.end_vertex, len(self.m_vertices)):
                return False
            if not e.degenerated and not ok(e.curve_3d_index, len(self.m_curves_3d)):
                return False
            for pc in e.pcurves:
                if not ok(pc.surface_index, len(self.m_surfaces)) or not ok(pc.curve_2d_index, len(self.m_curves_2d)):
                    return False
                if pc.curve_2d_index_2 >= 0 and not ok(pc.curve_2d_index_2, len(self.m_curves_2d)):
                    return False
        for w in self.m_wires:
            if not w.edges:
                return False
            for r in w.edges:
                if not ok(r.index, len(self.m_edges)):
                    return False
        for f in self.m_faces:
            if not ok(f.surface_index, len(self.m_surfaces)) or not f.wires:
                return False
            for r in f.wires:
                if not ok(r.index, len(self.m_wires)):
                    return False
        for s in self.m_shells:
            for r in s.faces:
                if not ok(r.index, len(self.m_faces)):
                    return False
        for s in self.m_solids:
            for r in s.shells:
                if not ok(r.index, len(self.m_shells)):
                    return False
        return True

    def is_closed(self, shell_index: int) -> bool:
        """BRep_Tool::IsClosed(shell): every non-degenerated edge is used exactly twice by the
        shell's faces (a seam counts twice through its two pcurves)."""
        if shell_index < 0 or shell_index >= len(self.m_shells):
            return False
        uses = [0] * len(self.m_edges)
        for fr in self.m_shells[shell_index].faces:
            for wr in self.m_faces[fr.index].wires:
                for er in self.wire_edges(wr):
                    uses[er.index] += 1
        for i, e in enumerate(self.m_edges):
            if not e.degenerated and uses[i] != 0 and uses[i] != 2:
                return False
        return len(self.m_shells[shell_index].faces) > 0

    def is_solid(self) -> bool:
        """At least one solid, and every shell of every solid is closed."""
        if not self.m_solids:
            return False
        for s in self.m_solids:
            for r in s.shells:
                if not self.is_closed(r.index):
                    return False
        return True

    def face_orientation(self, face_index: int) -> int:
        """Orientation of a face inside its first parent shell; Forward for a free face."""
        for s in self.m_shells:
            for r in s.faces:
                if r.index == face_index:
                    return r.orientation
        return BRepOrientation.Forward

    def pcurve_index(self, edge_index: int, face_index: int, orientation: int) -> int:
        """BRep_Tool::CurveOnSurface(E, F): the pcurve index of an edge on a face's surface for the
        given use orientation (the REVERSED pcurve on a seam); -1 if none."""
        if edge_index < 0 or edge_index >= len(self.m_edges):
            return -1
        if face_index < 0 or face_index >= len(self.m_faces):
            return -1
        si = self.m_faces[face_index].surface_index
        for pc in self.m_edges[edge_index].pcurves:
            if pc.surface_index == si:
                if orientation == BRepOrientation.Reversed and pc.curve_2d_index_2 >= 0:
                    return pc.curve_2d_index_2
                return pc.curve_2d_index
        return -1

    def wire_edges(self, wire: BRepRef) -> list[BRepRef]:
        """The edges of a wire composed with the wire's own orientation (a Reversed wire is
        traversed backwards with every edge reversed)."""
        if wire.index < 0 or wire.index >= len(self.m_wires):
            return []
        out = [BRepRef(r.index, brep_compose(wire.orientation, r.orientation)) for r in self.m_wires[wire.index].edges]
        if wire.orientation == BRepOrientation.Reversed:
            out.reverse()
        return out

    def edge_faces(self, edge_index: int) -> list[BRepRef]:
        """Faces sharing an edge, each with the orientation of that edge use."""
        out = []
        for fi, f in enumerate(self.m_faces):
            fo = self.face_orientation(fi)
            for wr in f.wires:
                for er in self.wire_edges(wr):
                    if er.index == edge_index:
                        out.append(BRepRef(fi, brep_compose(fo, er.orientation)))
        return out

    def vertex_points(self) -> list[Point]:
        """Vertex positions, in vertex order."""
        return [v.point for v in self.m_vertices]

    def update_tolerances(self) -> float:
        """BRepLib::UpdateTolerances: raise every edge tolerance to the worst distance between its
        curve ends (3D curve and each pcurve lifted through its surface) and its vertices, and
        every vertex tolerance to the worst incident edge end. Returns the largest tolerance."""
        worst = 0.0
        for e in self.m_edges:
            vs = self.m_vertices[e.start_vertex]
            ve = self.m_vertices[e.end_vertex]
            tol = e.tolerance
            if e.curve_3d_index >= 0:
                c = self.m_curves_3d[e.curve_3d_index]
                d0, d1 = c.domain()
                tol = max(tol, c.point_at(d0).distance(vs.point))
                tol = max(tol, c.point_at(d1).distance(ve.point))
            for pc in e.pcurves:
                srf = self.m_surfaces[pc.surface_index]
                for ci in (pc.curve_2d_index, pc.curve_2d_index_2):
                    if ci < 0:
                        continue
                    c2 = self.m_curves_2d[ci]
                    d0, d1 = c2.domain()
                    a = c2.point_at(d0)
                    z = c2.point_at(d1)
                    tol = max(tol, srf.point_at(a[0], a[1]).distance(vs.point))
                    tol = max(tol, srf.point_at(z[0], z[1]).distance(ve.point))
            e.tolerance = tol
            vs.tolerance = max(vs.tolerance, tol)
            ve.tolerance = max(ve.tolerance, tol)
            worst = max(worst, tol)
        return worst

    def volume(self) -> float:
        """Volume of the tessellated boundary (divergence theorem); meaningful for solids only."""
        return self.mesh().volume()

    ###########################################################################
    # Building (BRep_Builder)
    ###########################################################################

    def add_surface(self, srf: "NurbsSurface") -> int:
        self.m_surfaces.append(srf)
        return len(self.m_surfaces) - 1

    def add_curve_3d(self, crv: "NurbsCurve") -> int:
        self.m_curves_3d.append(crv)
        return len(self.m_curves_3d) - 1

    def add_curve_2d(self, crv: "NurbsCurve") -> int:
        self.m_curves_2d.append(crv)
        return len(self.m_curves_2d) - 1

    def add_vertex(self, pt: "Point", tolerance: float = 0.0) -> int:
        """MakeVertex."""
        self.m_vertices.append(BRepVertex(pt, tolerance))
        return len(self.m_vertices) - 1

    def add_edge(self, curve_3d_index: int, start_vertex: int, end_vertex: int, tolerance: float = 0.0) -> int:
        """MakeEdge: curve_3d_index -1 makes a degenerated edge (start == end vertex)."""
        e = BRepEdge()
        e.curve_3d_index = curve_3d_index
        e.start_vertex = start_vertex
        e.end_vertex = end_vertex
        e.tolerance = tolerance
        e.degenerated = curve_3d_index < 0
        self.m_edges.append(e)
        return len(self.m_edges) - 1

    def add_pcurve(self, edge_index: int, surface_index: int, curve_2d_index: int, curve_2d_index_2: int = -1) -> None:
        """UpdateEdge(E, pcurve, S): attach a pcurve on a surface; curve_2d_index_2 for the
        reversed use on a closed surface. Replaces an existing record for the same surface."""
        for pc in self.m_edges[edge_index].pcurves:
            if pc.surface_index == surface_index:
                pc.curve_2d_index = curve_2d_index
                pc.curve_2d_index_2 = curve_2d_index_2
                return
        self.m_edges[edge_index].pcurves.append(BRepCurveOnSurface(surface_index, curve_2d_index, curve_2d_index_2))

    def add_wire(self, edges: list[BRepRef]) -> int:
        """MakeWire + Add(edges)."""
        self.m_wires.append(BRepWire(edges))
        return len(self.m_wires) - 1

    def add_face(self, surface_index: int, wires: list[BRepRef], tolerance: float = 0.0) -> int:
        """MakeFace(S) + Add(wires); the first wire is the outer boundary."""
        f = BRepFace()
        f.surface_index = surface_index
        f.wires = list(wires)
        f.tolerance = tolerance
        self.m_faces.append(f)
        return len(self.m_faces) - 1

    def add_shell(self, faces: list[BRepRef]) -> int:
        """MakeShell + Add(faces)."""
        self.m_shells.append(BRepShell(faces))
        return len(self.m_shells) - 1

    def add_solid(self, shells: list[BRepRef]) -> int:
        """MakeSolid + Add(shells)."""
        self.m_solids.append(BRepSolid(shells))
        return len(self.m_solids) - 1

    ###########################################################################
    # Meshing
    ###########################################################################

    def _wire_uv_points(self, face_index: int, wire: BRepRef) -> list[Point]:
        """UV polygon of one wire of a face (pcurves sampled in traversal order)."""
        pts = []
        for er in self.wire_edges(wire):
            ci = self.pcurve_index(er.index, face_index, er.orientation)
            if ci < 0:
                continue
            crv = self.m_curves_2d[ci]
            if crv.degree() <= 1 and not crv.is_rational():
                seg = [crv.get_cv(k) for k in range(crv.cv_count())]
            else:
                seg, _ = crv.divide_by_count(max(crv.cv_count() * 4, 16))
            if er.orientation == BRepOrientation.Reversed:
                seg.reverse()
            pts.extend(seg[:-1])
        return pts

    def mesh(self) -> "Mesh":
        """One welded triangle mesh of every face, wound to the face's outward orientation."""
        from .mesh import Mesh
        polygons = []
        for fm in self.face_meshes():
            for fverts in fm.face.values():
                polygons.append([fm.vertex[vi].position() for vi in fverts])
        return Mesh.from_polylines(polygons, 1e-6)

    def face_meshes(self) -> list["Mesh"]:
        """One mesh per face, in face order (vertices not shared across faces)."""
        return self.face_meshes_q(None)

    def face_meshes_q(self, quality: tuple[float, float] | None = None) -> list["Mesh"]:
        """As face_meshes with a tessellation-quality (max_angle_deg, chord_factor) override for the
        grid-meshed faces."""
        from .mesh import Mesh
        from .nurbssurface_trimmed import NurbsSurfaceTrimmed
        from .remesh_nurbssurface_grid import RemeshNurbsSurfaceGrid
        nf = len(self.m_faces)

        # Phase 1: a face whose outer wire is the full UV rectangle (straight pcurves enclosing the
        # whole domain area, no holes) is meshed directly on the surface grid; everything else goes
        # through the trimmed CDT.
        face_direct = [False] * nf
        for fi in range(nf):
            face = self.m_faces[fi]
            srf = self.m_surfaces[face.surface_index]
            if len(face.wires) != 1:
                continue
            all_linear = True
            for er in self.wire_edges(face.wires[0]):
                ci = self.pcurve_index(er.index, fi, er.orientation)
                if ci < 0:
                    continue
                if self.m_curves_2d[ci].degree() > 1 or self.m_curves_2d[ci].is_rational():
                    all_linear = False
            if not all_linear:
                continue
            outer = self._wire_uv_points(fi, face.wires[0])
            if len(outer) < 3:
                continue
            u0, u1 = srf.domain(0)
            v0, v1 = srf.domain(1)
            domain_area = (u1 - u0) * (v1 - v0)
            face_direct[fi] = abs(abs(_polygon_signed_area(outer)) - domain_area) < 1e-3 * domain_area

        # Phase 2: direct faces. Record the 3D boundary discretisation along every edge shared
        # with a CDT face so both sides tessellate the seam with the same points.
        fmesh = [Mesh() for _ in range(nf)]
        edge_bnd: dict[int, list[Point]] = {}
        for fi in range(nf):
            if not face_direct[fi]:
                continue
            face = self.m_faces[fi]
            srf = self.m_surfaces[face.surface_index]
            if quality is not None:
                fmesh[fi] = RemeshNurbsSurfaceGrid.from_u_v_q(srf, 0, 0, quality[0], quality[1])
            else:
                fmesh[fi] = srf.mesh()
            u0, u1 = srf.domain(0)
            v0, v1 = srf.domain(1)
            utol = (u1 - u0) * 0.001
            vtol = (v1 - v0) * 0.001
            for er in self.wire_edges(face.wires[0]):
                eidx = er.index
                if eidx in edge_bnd:
                    continue
                shared = any(fr.index != fi and not face_direct[fr.index] for fr in self.edge_faces(eidx))
                if not shared:
                    continue
                ci = self.pcurve_index(eidx, fi, er.orientation)
                if ci < 0:
                    continue
                c2d = self.m_curves_2d[ci]
                sp = c2d.get_cv(0)
                ep = c2d.get_cv(c2d.cv_count() - 1)
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

        # Phase 3: CDT faces. Shared edges reuse the direct face's boundary points projected into
        # this face's planar patch; every other edge samples its own pcurve.
        for fi in range(nf):
            if face_direct[fi]:
                continue
            face = self.m_faces[fi]
            srf = self.m_surfaces[face.surface_index]
            p00 = srf.get_cv(0, 0)
            p10 = srf.get_cv(1, 0)
            p01 = srf.get_cv(0, 1)
            eu = [p10[0] - p00[0], p10[1] - p00[1], p10[2] - p00[2]]
            ev = [p01[0] - p00[0], p01[1] - p00[1], p01[2] - p00[2]]
            eu2 = eu[0] * eu[0] + eu[1] * eu[1] + eu[2] * eu[2]
            ev2 = ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2]
            can_project = srf.degree(0) == 1 and srf.degree(1) == 1 and eu2 > 1e-28 and ev2 > 1e-28

            ts = NurbsSurfaceTrimmed()
            ts.m_surface = srf
            for wi, wr in enumerate(face.wires):
                loop_pts = []
                for er in self.wire_edges(wr):
                    ci = self.pcurve_index(er.index, fi, er.orientation)
                    if ci < 0:
                        continue
                    crv = self.m_curves_2d[ci]
                    if can_project and er.index in edge_bnd:
                        seg = []
                        for pt in edge_bnd[er.index]:
                            dx, dy, dz = pt[0] - p00[0], pt[1] - p00[1], pt[2] - p00[2]
                            seg.append(Point((dx * eu[0] + dy * eu[1] + dz * eu[2]) / eu2,
                                             (dx * ev[0] + dy * ev[1] + dz * ev[2]) / ev2, 0.0))
                        d0, d1 = crv.domain()
                        start = crv.point_at(d1 if er.orientation == BRepOrientation.Reversed else d0)
                        if seg[0].distance(start) > seg[-1].distance(start):
                            seg.reverse()
                    else:
                        if crv.degree() <= 1 and not crv.is_rational():
                            seg = [crv.get_cv(k) for k in range(crv.cv_count())]
                        else:
                            seg, _ = crv.divide_by_count(max(crv.cv_count() * 4, 16))
                        if er.orientation == BRepOrientation.Reversed:
                            seg.reverse()
                    loop_pts.extend(seg[:-1])
                if len(loop_pts) < 3:
                    continue
                loop_crv = NurbsCurve.create(True, 1, loop_pts)
                if wi == 0:
                    ts.m_outer_loop = loop_crv
                else:
                    ts.m_inner_loops.append(loop_crv)
            fmesh[fi] = ts.mesh()

        # A Reversed face has its outward normal opposite to the surface normal: flip winding
        # and stored normals together so shading agrees with the geometry.
        for fi in range(nf):
            if self.face_orientation(fi) != BRepOrientation.Reversed:
                continue
            fmesh[fi].flip()
            for vd in fmesh[fi].vertex.values():
                n = vd.normal()
                if n is not None:
                    vd.set_normal(-n[0], -n[1], -n[2])
        return fmesh

    ###########################################################################
    # Evaluation
    ###########################################################################

    def point_at(self, face_index: int, u: float, v: float) -> "Point":
        """Surface point of a face at (u, v)."""
        if face_index < 0 or face_index >= len(self.m_faces):
            return Point(0.0, 0.0, 0.0)
        return self.m_surfaces[self.m_faces[face_index].surface_index].point_at(u, v)

    def normal_at(self, face_index: int, u: float, v: float) -> "Vector":
        """Surface normal of a face at (u, v), flipped when the face is Reversed in its shell."""
        if face_index < 0 or face_index >= len(self.m_faces):
            return Vector(0.0, 0.0, 0.0)
        n = self.m_surfaces[self.m_faces[face_index].surface_index].normal_at(u, v)
        if self.face_orientation(face_index) == BRepOrientation.Reversed:
            return Vector(-n[0], -n[1], -n[2])
        return n

    ###########################################################################
    # Transformation
    ###########################################################################

    def transform(self, xform: "Xform") -> None:
        """Transform surfaces, 3D curves and vertices in place (pcurves are parametric, untouched)."""
        for srf in self.m_surfaces:
            srf.transform(xform)
        for crv in self.m_curves_3d:
            crv.transform(xform)
        for v in self.m_vertices:
            v.point = xform.transform_point(v.point)

    def transformed(self, xform: "Xform") -> "BRep":
        """Return a transformed copy."""
        b = self.duplicate()
        b.transform(xform)
        return b

    ###########################################################################
    # JSON Serialization
    ###########################################################################

    @staticmethod
    def _refs_dump(refs: list[BRepRef]) -> list:
        return [{"index": r.index, "orientation": BRepOrientation.to_str(r.orientation)} for r in refs]

    @staticmethod
    def _refs_load(arr) -> list[BRepRef]:
        return [BRepRef(r["index"], BRepOrientation.from_str(r["orientation"])) for r in arr]

    def __jsondump__(self):
        j = {}
        j["curves_2d"] = [c.__jsondump__() for c in self.m_curves_2d]
        j["curves_3d"] = [c.__jsondump__() for c in self.m_curves_3d]
        edges = []
        for e in self.m_edges:
            edges.append({
                "curve_3d_index": e.curve_3d_index,
                "degenerated": e.degenerated,
                "end_vertex": e.end_vertex,
                "pcurves": [{"curve_2d_index": pc.curve_2d_index, "curve_2d_index_2": pc.curve_2d_index_2, "surface_index": pc.surface_index} for pc in e.pcurves],
                "start_vertex": e.start_vertex,
                "tolerance": e.tolerance,
            })
        j["edges"] = edges
        faces = []
        for f in self.m_faces:
            d = {"surface_index": f.surface_index, "tolerance": f.tolerance, "wires": BRep._refs_dump(f.wires)}
            if f.facecolor is not None:
                d = {"facecolor": f.facecolor.__jsondump__(), **d}
            faces.append(d)
        j["faces"] = faces
        j["guid"] = self.guid
        j["name"] = self.name
        j["shells"] = [{"faces": BRep._refs_dump(s.faces)} for s in self.m_shells]
        j["solids"] = [{"shells": BRep._refs_dump(s.shells)} for s in self.m_solids]
        j["surfacecolor"] = self.surfacecolor.__jsondump__()
        j["surfaces"] = [s.__jsondump__() for s in self.m_surfaces]
        j["type"] = "BRep"
        j["vertices"] = [{"point": [v.point[0], v.point[1], v.point[2]], "tolerance": v.tolerance} for v in self.m_vertices]
        j["width"] = self.width
        j["wires"] = [{"edges": BRep._refs_dump(w.edges)} for w in self.m_wires]
        return j

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        b = cls()
        b.guid = guid if guid is not None else data.get("guid", b.guid)
        b.name = name if name is not None else data.get("name", "my_brep")
        b.width = data.get("width", 1.0)
        if "surfacecolor" in data:
            b.surfacecolor = Color.__jsonload__(data["surfacecolor"])
        b.m_curves_2d = [NurbsCurve.__jsonload__(c) for c in data.get("curves_2d", [])]
        b.m_curves_3d = [NurbsCurve.__jsonload__(c) for c in data.get("curves_3d", [])]
        b.m_surfaces = [NurbsSurface.__jsonload__(s) for s in data.get("surfaces", [])]
        for v in data.get("vertices", []):
            b.m_vertices.append(BRepVertex(Point(v["point"][0], v["point"][1], v["point"][2]), v["tolerance"]))
        for e in data.get("edges", []):
            be = BRepEdge()
            be.curve_3d_index = e["curve_3d_index"]
            be.degenerated = e["degenerated"]
            be.end_vertex = e["end_vertex"]
            be.pcurves = [BRepCurveOnSurface(pc["surface_index"], pc["curve_2d_index"], pc["curve_2d_index_2"]) for pc in e["pcurves"]]
            be.start_vertex = e["start_vertex"]
            be.tolerance = e["tolerance"]
            b.m_edges.append(be)
        b.m_wires = [BRepWire(BRep._refs_load(w["edges"])) for w in data.get("wires", [])]
        for f in data.get("faces", []):
            bf = BRepFace()
            if "facecolor" in f:
                bf.facecolor = Color.__jsonload__(f["facecolor"])
            bf.surface_index = f["surface_index"]
            bf.tolerance = f["tolerance"]
            bf.wires = BRep._refs_load(f["wires"])
            b.m_faces.append(bf)
        b.m_shells = [BRepShell(BRep._refs_load(s["faces"])) for s in data.get("shells", [])]
        b.m_solids = [BRepSolid(BRep._refs_load(s["shells"])) for s in data.get("solids", [])]
        return b

    def file_json_dump(self, filepath: str | Path) -> None:
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=4)

    @classmethod
    def file_json_load(cls, filepath: str | Path) -> "BRep":
        import json
        with open(filepath) as f:
            return cls.__jsonload__(json.load(f))

    def file_json_dumps(self) -> str:
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, s: str) -> "BRep":
        import json
        return cls.__jsonload__(json.loads(s))

    ###########################################################################
    # Protobuf Serialization
    ###########################################################################

    @staticmethod
    def _refs_fill(refs: list[BRepRef], out) -> None:
        for r in refs:
            p = out.add()
            p.index = r.index
            p.orientation = r.orientation

    @staticmethod
    def _refs_from_proto(arr) -> list[BRepRef]:
        return [BRepRef(r.index, int(r.orientation)) for r in arr]

    def pb_dumps(self) -> bytes:
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
            p.point.x = v.point[0]
            p.point.y = v.point[1]
            p.point.z = v.point[2]
            p.tolerance = v.tolerance
        for e in self.m_edges:
            p = proto.edges.add()
            p.curve_3d_index = e.curve_3d_index
            p.start_vertex = e.start_vertex
            p.end_vertex = e.end_vertex
            p.tolerance = e.tolerance
            p.degenerated = e.degenerated
            for pc in e.pcurves:
                q = p.pcurves.add()
                q.surface_index = pc.surface_index
                q.curve_2d_index = pc.curve_2d_index
                q.curve_2d_index_2 = pc.curve_2d_index_2
        for w in self.m_wires:
            BRep._refs_fill(w.edges, proto.wires.add().edges)
        for f in self.m_faces:
            p = proto.faces.add()
            p.surface_index = f.surface_index
            BRep._refs_fill(f.wires, p.wires)
            p.tolerance = f.tolerance
            if f.facecolor is not None:
                p.facecolor.r = f.facecolor.r
                p.facecolor.g = f.facecolor.g
                p.facecolor.b = f.facecolor.b
                p.facecolor.a = f.facecolor.a
        for s in self.m_shells:
            BRep._refs_fill(s.faces, proto.shells.add().faces)
        for s in self.m_solids:
            BRep._refs_fill(s.shells, proto.solids.add().shells)
        proto.surfacecolor.name = self.surfacecolor.name
        proto.surfacecolor.r = self.surfacecolor.r
        proto.surfacecolor.g = self.surfacecolor.g
        proto.surfacecolor.b = self.surfacecolor.b
        proto.surfacecolor.a = self.surfacecolor.a
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "BRep":
        from .proto import brep_pb2
        proto = brep_pb2.BRep()
        proto.ParseFromString(data)
        b = cls()
        b.guid = proto.guid
        b.name = proto.name
        b.width = proto.width
        b.m_curves_2d = [NurbsCurve.pb_loads(c.SerializeToString()) for c in proto.curves_2d]
        b.m_curves_3d = [NurbsCurve.pb_loads(c.SerializeToString()) for c in proto.curves_3d]
        b.m_surfaces = [NurbsSurface.pb_loads(s.SerializeToString()) for s in proto.surfaces]
        for v in proto.vertices:
            b.m_vertices.append(BRepVertex(Point(v.point.x, v.point.y, v.point.z), v.tolerance))
        for e in proto.edges:
            be = BRepEdge()
            be.curve_3d_index = e.curve_3d_index
            be.start_vertex = e.start_vertex
            be.end_vertex = e.end_vertex
            be.tolerance = e.tolerance
            be.degenerated = e.degenerated
            be.pcurves = [BRepCurveOnSurface(pc.surface_index, pc.curve_2d_index, pc.curve_2d_index_2) for pc in e.pcurves]
            b.m_edges.append(be)
        b.m_wires = [BRepWire(BRep._refs_from_proto(w.edges)) for w in proto.wires]
        for f in proto.faces:
            bf = BRepFace()
            bf.surface_index = f.surface_index
            bf.wires = BRep._refs_from_proto(f.wires)
            bf.tolerance = f.tolerance
            if f.HasField('facecolor'):
                bf.facecolor = Color(f.facecolor.r, f.facecolor.g, f.facecolor.b, f.facecolor.a)
            b.m_faces.append(bf)
        b.m_shells = [BRepShell(BRep._refs_from_proto(s.faces)) for s in proto.shells]
        b.m_solids = [BRepSolid(BRep._refs_from_proto(s.shells)) for s in proto.solids]
        cp = proto.surfacecolor
        b.surfacecolor = Color(cp.r, cp.g, cp.b, cp.a)
        b.surfacecolor.name = cp.name
        return b

    def pb_dump(self, filepath: str | Path) -> None:
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: str | Path) -> "BRep":
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())
