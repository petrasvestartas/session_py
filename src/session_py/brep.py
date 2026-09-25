from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import json
import math
import sys
import uuid

from .color import Color
from .mesh import Mesh
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .nurbssurface_trimmed import NurbsSurfaceTrimmed
from .nurbssurface_trimmed import TrimLoops
from .plane import Plane
from .point import Point
from .polyline import Polyline
from .primitives import Primitives
from .remesh_nurbssurface_grid import RemeshNurbsSurfaceGrid
from .tolerance import PI
from .tolerance import Tolerance
from .vector import Vector

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import brep_pb2
    from .xform import Xform


# ═══════════════════════════════════════════════════════════════════════════
# Orientation
# ═══════════════════════════════════════════════════════════════════════════
class BRepOrientation:
    """TopAbs_Orientation: carried by the parent -> child reference, never by the shape"""

    Forward = 0  # Same direction as the shape.
    Reversed = 1  # Opposite direction.
    Internal = 2  # Inside the parent, both sides.
    External = 3  # Outside the parent, no side.


def brep_reverse(o: int) -> int:
    """TopAbs::Reverse"""

    if o == BRepOrientation.Forward:
        return BRepOrientation.Reversed

    if o == BRepOrientation.Reversed:
        return BRepOrientation.Forward

    return o


def brep_compose(a: int, b: int) -> int:
    """TopAbs::Compose: the orientation of a sub-shape reached through a parent with orientation `a`"""

    if a == BRepOrientation.Internal or a == BRepOrientation.External:
        return a

    if a == BRepOrientation.Forward:
        return b

    return brep_reverse(b)


F = BRepOrientation.Forward
R = BRepOrientation.Reversed


def _orientation_to_str(o: int) -> str:
    """JSON name of an orientation"""

    if o == BRepOrientation.Reversed:
        return "reversed"

    if o == BRepOrientation.Internal:
        return "internal"

    if o == BRepOrientation.External:
        return "external"

    return "forward"


def _orientation_from_str(s: str) -> int:
    """Orientation of a JSON name, Forward when unknown"""

    if s == "reversed":
        return BRepOrientation.Reversed

    if s == "internal":
        return BRepOrientation.Internal

    if s == "external":
        return BRepOrientation.External

    return BRepOrientation.Forward


def _in_range(index: int, count: int) -> bool:
    """True when `index` addresses one of `count` table entries"""

    return 0 <= index < count


# ═══════════════════════════════════════════════════════════════════════════
# Shapes
# ═══════════════════════════════════════════════════════════════════════════
class BRepRef:
    """TopoDS_Shape: an oriented reference to a sub-shape (index into the owning table)"""

    def __init__(self, index: int = -1, orientation: int = BRepOrientation.Forward):
        self.index = index  # Index into the owning table.
        self.orientation = orientation  # Orientation of this use.

    def __eq__(self, other):
        """Compare index and orientation."""

        return (
            isinstance(other, BRepRef)
            and self.index == other.index
            and self.orientation == other.orientation
        )


class BRepVertex:
    """BRep_TVertex"""

    def __init__(self, point: Point | None = None, tolerance: float = 0.0):
        self.point = point if point is not None else Point(0.0, 0.0, 0.0)  # Position.
        self.tolerance = tolerance  # Vertex tolerance.


class BRepCurveOnSurface:
    """BRep_CurveOnSurface: curve_2d_index_2 is the pcurve of the REVERSED use on a closed surface (seam), -1 otherwise; pcurves run in the edge's own direction"""

    def __init__(
        self,
        surface_index: int = -1,
        curve_2d_index: int = -1,
        curve_2d_index_2: int = -1,
    ):
        self.surface_index = surface_index  # Surface the pcurve lies on.
        self.curve_2d_index = curve_2d_index  # Pcurve of the forward use.
        self.curve_2d_index_2 = curve_2d_index_2  # Reversed use on a seam, else -1.


class BRepEdge:
    """BRep_TEdge: curve_3d_index is -1 for a degenerated edge (sphere pole, cone apex)"""

    def __init__(self):
        self.curve_3d_index = -1  # 3D curve, -1 when degenerated.
        self.start_vertex = -1  # Start vertex.
        self.end_vertex = -1  # End vertex.
        self.tolerance = 0.0  # Edge tolerance.
        self.degenerated = False  # True for a pole or apex edge.
        self.pcurves: list[BRepCurveOnSurface] = []  # One per surface.


class BRepWire:
    """TopoDS_TWire"""

    def __init__(self, edges: list[BRepRef] | None = None):
        self.edges: list[BRepRef] = list(edges) if edges else []  # Traversal order.


class BRepFace:
    """BRep_TFace: the first wire is the outer boundary; facecolor None means unset"""

    def __init__(self):
        self.surface_index = -1  # Underlying surface.
        self.wires: list[BRepRef] = []  # Outer wire first, then holes.
        self.tolerance = 0.0  # Face tolerance.
        self.facecolor: Color | None = None  # Display color, None when unset.


class BRepShell:
    """TopoDS_TShell"""

    def __init__(self, faces: list[BRepRef] | None = None):
        self.faces: list[BRepRef] = list(faces) if faces else []  # Oriented faces.


class BRepSolid:
    """TopoDS_TSolid"""

    def __init__(self, shells: list[BRepRef] | None = None):
        self.shells: list[BRepRef] = list(shells) if shells else []  # Outer first.


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════
def _bilinear_patch(p00: Point, p10: Point, p01: Point, p11: Point) -> NurbsSurface:
    """Bilinear planar patch: u runs p00 -> p10, v runs p00 -> p01, natural normal = u x v"""

    srf = NurbsSurface(3, False, 2, 2, 2, 2)

    if (
        not srf.set_cv(0, 0, p00)
        or not srf.set_cv(1, 0, p10)
        or not srf.set_cv(0, 1, p01)
        or not srf.set_cv(1, 1, p11)
    ):
        return NurbsSurface()

    return srf


def _uv_line(u0: float, v0: float, u1: float, v1: float) -> NurbsCurve:
    """Straight pcurve from (u0, v0) to (u1, v1)"""

    return NurbsCurve.create(
        False,
        1,
        [
            Point(u0, v0, 0.0),
            Point(u1, v1, 0.0),
        ],
    )


def _project_to_patch(crv: NurbsCurve, srf: NurbsSurface) -> NurbsCurve:
    """Exact pcurve of a 3D curve lying on a bilinear planar patch: the affine image of its CVs"""

    p00 = srf.get_cv(0, 0)
    eu = srf.get_cv(1, 0) - p00
    ev = srf.get_cv(0, 1) - p00
    eu2 = eu.dot(eu)
    ev2 = ev.dot(ev)
    c2 = NurbsCurve(3, crv.is_rational(), crv.order(), crv.cv_count())

    for i in range(crv.nurbsknot_count()):
        if not c2.set_nurbsknot(i, crv.nurbsknot(i)):
            return NurbsCurve()

    for i in range(crv.cv_count()):
        wx, wy, wz, w = crv.get_cv_4d(i)
        d = Point(wx / w, wy / w, wz / w) - p00
        u = d.dot(eu) / eu2
        v = d.dot(ev) / ev2

        written = (
            c2.set_cv_4d(i, u * w, v * w, 0.0, w)
            if crv.is_rational()
            else c2.set_cv(i, Point(u, v, 0.0))
        )

        if not written:
            return NurbsCurve()

    return c2


def _uv_signed_area(c2d: NurbsCurve) -> float:
    """Signed area of a closed pcurve's sampled polygon (positive = counter-clockwise)"""

    pts = c2d.divide_by_count(max(c2d.cv_count() * 4, 16), True)[0]
    area = 0.0

    for i in range(len(pts) - 1):
        area += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]

    return 0.5 * area


def _polygon_signed_area(pts: list[Point]) -> float:
    """Signed area of a closed UV polygon (positive = counter-clockwise)"""

    n = len(pts)
    area = 0.0

    for i in range(n):
        p = pts[i]
        q = pts[(i + 1) % n]
        area += p[0] * q[1] - q[0] * p[1]

    return 0.5 * area


def _bbox_diagonal(srf: NurbsSurface) -> float:
    """Diagonal of the control point bounding box"""

    lo = Point(1e30, 1e30, 1e30)
    hi = Point(-1e30, -1e30, -1e30)

    for i in range(srf.cv_count(0)):
        for j in range(srf.cv_count(1)):
            p = srf.get_cv(i, j)

            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])

    return hi.distance(lo)


# ═══════════════════════════════════════════════════════════════════════════
# Factory helpers
# ═══════════════════════════════════════════════════════════════════════════
class _PolyFaceBuilder:
    """Planar polygon faces from a vertex table: edges run lo -> hi vertex and are shared, a face lists its vertices counter-clockwise seen from outside so the patch normal points outward"""

    def __init__(self, b: BRep):
        self.b = b  # BRep being built.
        self.edge_map: dict[tuple[int, int], int] = {}  # Edge per (lo, hi) pair.

    def edge(self, v0: int, v1: int) -> int:
        """Straight edge between two vertices, shared by every face that uses it"""

        lo = min(v0, v1)
        hi = max(v0, v1)

        if (lo, hi) in self.edge_map:
            return self.edge_map[(lo, hi)]

        line = NurbsCurve.create(
            False, 1, [self.b.m_vertices[lo].point, self.b.m_vertices[hi].point]
        )
        ei = self.b.add_edge(self.b.add_curve_3d(line), lo, hi)
        self.edge_map[(lo, hi)] = ei

        return ei

    def wire_refs(self, si: int, vi: list[int]) -> list[BRepRef]:
        """Oriented edge references of the vertex cycle `vi`, each with its pcurve on surface `si`"""

        b = self.b
        srf = b.m_surfaces[si]
        refs = []
        n = len(vi)

        for i in range(n):
            va = vi[i]
            vb = vi[(i + 1) % n]
            ei = self.edge(va, vb)
            b.add_pcurve(
                ei,
                si,
                b.add_curve_2d(
                    _project_to_patch(b.m_curves_3d[b.m_edges[ei].curve_3d_index], srf)
                ),
            )
            refs.append(BRepRef(ei, F if b.m_edges[ei].start_vertex == va else R))

        return refs

    def face(
        self, srf: NurbsSurface, vi: list[int], holes: list[list[int]] | None = None
    ) -> int:
        """Face on `srf` bounded by the vertex cycle `vi`, with one inner wire per hole cycle; returns the face index"""

        si = self.b.add_surface(srf)
        wires = [BRepRef(self.b.add_wire(self.wire_refs(si, vi)), F)]

        for hole in holes or []:
            wires.append(BRepRef(self.b.add_wire(self.wire_refs(si, hole)), F))

        return self.b.add_face(si, wires)


_BOX_FACES = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
]


def _quad_patch(b: BRep, fv: list[int]) -> NurbsSurface:
    """Bilinear patch spanned by four vertex indices in face order (p00, p10, p11, p01)"""

    return _bilinear_patch(
        b.m_vertices[fv[0]].point,
        b.m_vertices[fv[1]].point,
        b.m_vertices[fv[3]].point,
        b.m_vertices[fv[2]].point,
    )


def _box_corners(b: BRep, sx: float, sy: float, sz: float) -> None:
    """The eight corners of an origin-centered box, bottom ring then top ring"""

    hx = sx * 0.5
    hy = sy * 0.5
    hz = sz * 0.5

    b.add_vertex(Point(-hx, -hy, -hz))
    b.add_vertex(Point(hx, -hy, -hz))
    b.add_vertex(Point(hx, hy, -hz))
    b.add_vertex(Point(-hx, hy, -hz))
    b.add_vertex(Point(-hx, -hy, hz))
    b.add_vertex(Point(hx, -hy, hz))
    b.add_vertex(Point(hx, hy, hz))
    b.add_vertex(Point(-hx, hy, hz))


def _cap_patch(r: float, z: float, up: bool) -> NurbsSurface:
    """Planar cap at height z with natural normal +Z (up) or -Z (down), spanning [-r, r]^2"""

    if up:
        return _bilinear_patch(
            Point(-r, -r, z), Point(r, -r, z), Point(-r, r, z), Point(r, r, z)
        )

    return _bilinear_patch(
        Point(-r, -r, z), Point(-r, r, z), Point(r, -r, z), Point(r, r, z)
    )


def _cap_face(b: BRep, cap: NurbsSurface, edge: int) -> int:
    """Cap face bounded by one closed edge: outer wire counter-clockwise in the patch's UV"""

    si = b.add_surface(cap)
    c2d = _project_to_patch(b.m_curves_3d[b.m_edges[edge].curve_3d_index], cap)
    o = F if _uv_signed_area(c2d) > 0.0 else R
    b.add_pcurve(edge, si, b.add_curve_2d(c2d))

    return b.add_face(si, [BRepRef(b.add_wire([BRepRef(edge, o)]), F)])


def _body_face(b: BRep, si: int, e_bot: int, e_seam: int, e_top: int) -> int:
    """Periodic body face (cylinder / cone / bore): seam from v0 to v1 at u0 == u1, bottom ring forward at v0, top ring (or degenerated apex) reversed at v1"""

    u0, u1 = b.m_surfaces[si].domain(0)
    v0, v1 = b.m_surfaces[si].domain(1)

    b.add_pcurve(e_bot, si, b.add_curve_2d(_uv_line(u0, v0, u1, v0)))
    b.add_pcurve(e_top, si, b.add_curve_2d(_uv_line(u0, v1, u1, v1)))
    b.add_pcurve(
        e_seam,
        si,
        b.add_curve_2d(_uv_line(u1, v0, u1, v1)),
        b.add_curve_2d(_uv_line(u0, v0, u0, v1)),
    )
    wire = b.add_wire(
        [
            BRepRef(e_bot, F),
            BRepRef(e_seam, F),
            BRepRef(e_top, R),
            BRepRef(e_seam, R),
        ]
    )

    return b.add_face(si, [BRepRef(wire, F)])


def _plane_point(org: Point, xa: Vector, ya: Vector, u: float, v: float) -> Point:
    """Point of the plane (org, xa, ya) at (u, v)"""

    return org + xa * u + ya * v


def _planar_patch_through(
    pts: list[Point], org: Point, xa: Vector, ya: Vector
) -> NurbsSurface:
    """Padded bilinear patch through `pts` in the plane (org, xa, ya)"""

    umin = 1e30
    umax = -1e30
    vmin = 1e30
    vmax = -1e30

    for p in pts:
        d = p - org
        u = d.dot(xa)
        v = d.dot(ya)
        umin = min(umin, u)
        umax = max(umax, u)
        vmin = min(vmin, v)
        vmax = max(vmax, v)

    pad = max(umax - umin, vmax - vmin) * 0.01
    umin -= pad
    umax += pad
    vmin -= pad
    vmax += pad

    return _bilinear_patch(
        _plane_point(org, xa, ya, umin, vmin),
        _plane_point(org, xa, ya, umax, vmin),
        _plane_point(org, xa, ya, umin, vmax),
        _plane_point(org, xa, ya, umax, vmax),
    )


def _signed_area_in_plane(
    pts: list[Point], org: Point, xa: Vector, ya: Vector
) -> float:
    """Signed area of a closed cycle of points seen in the plane (org, xa, ya): positive when it runs counter-clockwise"""

    area = 0.0
    n = len(pts)

    for i in range(n):
        a = pts[i] - org
        b = pts[(i + 1) % n] - org
        area += a.dot(xa) * b.dot(ya) - b.dot(xa) * a.dot(ya)

    return area * 0.5


def _open_points(pl: Polyline) -> list[Point]:
    """The vertices of a polyline without the closing duplicate"""

    pts = pl.get_points()
    n = len(pts) - 1 if pl.is_closed() else len(pts)

    return pts[:n]


def _find_or_add_vertex(b: BRep, p: Point, tol: float) -> int:
    """Index of the first vertex within `tol` of `p`, a new vertex when none is"""

    for i in range(len(b.m_vertices)):
        if b.m_vertices[i].point.distance(p) < tol:
            return i

    return b.add_vertex(p)


def _cv_points(c: NurbsCurve) -> list[Point]:
    """Euclidean control points of a curve, zero weights skipped"""

    pts = []

    for k in range(c.cv_count()):
        wx, wy, wz, w = c.get_cv_4d(k)

        if w != 0.0:
            pts.append(Point(wx / w, wy / w, wz / w))

    return pts


def _curve_wire(b: BRep, crv: NurbsCurve, si: int, tol: float) -> int:
    """One-edge wire of a closed or open curve on planar surface `si`, sharing vertices within `tol`"""

    sp = crv.point_at(crv.domain()[0])
    ep = crv.point_at(crv.domain()[1])
    vs = _find_or_add_vertex(b, sp, tol)
    ve = vs if crv.is_closed() else _find_or_add_vertex(b, ep, tol)
    ei = b.add_edge(b.add_curve_3d(crv), vs, ve)
    b.add_pcurve(ei, si, b.add_curve_2d(_project_to_patch(crv, b.m_surfaces[si])))

    return b.add_wire([BRepRef(ei, F)])


# ═══════════════════════════════════════════════════════════════════════════
# Sewing helpers
# ═══════════════════════════════════════════════════════════════════════════
def _signed_volume(meshes: list[Mesh]) -> float:
    """Signed volume of face meshes (positive when the windings point outward)"""

    total = 0.0

    for fm in meshes:
        for fverts in fm.face.values():
            for k in range(1, len(fverts) - 1):
                a = fm.vertex[fverts[0]].position()
                b = fm.vertex[fverts[k]].position()
                c = fm.vertex[fverts[k + 1]].position()
                total += (
                    a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])
                )

    return total / 6.0


def _edge_uses(b: BRep) -> list[list[tuple[int, int]]]:
    """Face uses of every edge as (face, composed orientation); empty when some edge is not used exactly twice"""

    uses: list[list[tuple[int, int]]] = []

    for _ in b.m_edges:
        uses.append([])

    for fi in range(b.face_count()):
        for wr in b.m_faces[fi].wires:
            for er in b.wire_edges(wr):
                uses[er.index].append((fi, er.orientation))

    for use in uses:
        if len(use) != 2:
            return []

    return uses


def _face_components(
    b: BRep, uses: list[list[tuple[int, int]]], fo: list[int]
) -> list[list[int]]:
    """Connected components of faces, each face oriented consistently with the neighbour it was reached from"""

    nf = b.face_count()
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

    return components


def _close_free_faces(b: BRep) -> None:
    """BRepBuilderAPI_Sewing + MakeSolid for free faces: when every edge is shared by exactly two face uses, one shell per connected component wound outward and one solid per shell"""

    nf = b.face_count()

    if nf == 0:
        return

    uses = _edge_uses(b)

    if not uses:
        return

    fo = [F] * nf
    shells = []

    for comp in _face_components(b, uses, fo):
        refs = []

        for fi in comp:
            refs.append(BRepRef(fi, fo[fi]))

        shells.append(BRepRef(b.add_shell(refs), F))

    fm = b.face_meshes()

    for sr in shells:
        part = []

        for fr in b.m_shells[sr.index].faces:
            part.append(fm[fr.index])

        if _signed_volume(part) < 0.0:
            for fr in b.m_shells[sr.index].faces:
                fr.orientation = brep_reverse(fr.orientation)

        b.add_solid([sr])


# ═══════════════════════════════════════════════════════════════════════════
# Planar face helpers
# ═══════════════════════════════════════════════════════════════════════════
CURVED_EDGE_SAMPLES = 16  # Samples per curved edge of a planar face.


def _face_outline(b: BRep, fi: int) -> list[Point]:
    """Open outline of a face's outer wire in wire order: vertices of straight edges, samples of curved ones"""

    points = []

    for er in b.wire_edges(b.m_faces[fi].wires[0]):
        if er.index < 0 or er.index >= len(b.m_edges):
            continue

        edge = b.m_edges[er.index]

        if edge.degenerated:
            continue

        reversed_ = er.orientation == BRepOrientation.Reversed
        curved = (
            edge.curve_3d_index >= 0 and b.m_curves_3d[edge.curve_3d_index].degree() > 1
        )

        if curved:
            c = b.m_curves_3d[edge.curve_3d_index]
            d0, d1 = c.domain()

            for s in range(CURVED_EDGE_SAMPLES):
                u = s / CURVED_EDGE_SAMPLES
                t = d1 + (d0 - d1) * u if reversed_ else d0 + (d1 - d0) * u
                points.append(c.point_at(t))
        else:
            start = edge.end_vertex if reversed_ else edge.start_vertex

            if 0 <= start < len(b.m_vertices):
                points.append(b.m_vertices[start].point)

    return points


def _outline_volume(polylines: list[Polyline]) -> float:
    """Signed volume enclosed by closed outlines (tetrahedra fans from the origin), positive when wound outward"""

    total = 0.0

    for pl in polylines:
        pts = pl.get_points()

        if len(pts) < 3:
            continue

        p0 = pts[0]

        for k in range(1, len(pts) - 1):
            p1 = pts[k]
            p2 = pts[k + 1]
            total += (
                p0[0] * (p1[1] * p2[2] - p1[2] * p2[1])
                + p0[1] * (p1[2] * p2[0] - p1[0] * p2[2])
                + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0])
            )

    return total / 6.0


def _planar_faces(b: BRep) -> tuple[list[Polyline], list[Plane]]:
    """Outer polyline and outward plane of every planar face in one walk, so the two stay index-aligned"""

    polylines = []
    planes = []

    for fi in range(b.face_count()):
        face = b.m_faces[fi]

        if face.surface_index < 0 or not face.wires:
            continue

        if not b.m_surfaces[face.surface_index].is_planar():
            continue

        points = _face_outline(b, fi)

        if len(points) < 3:
            continue

        origin = Point.centroid(points)
        normal = Vector.average_normal(points)

        if b.face_orientation(fi) == BRepOrientation.Reversed:
            normal.reverse()

        points.append(points[0])
        polylines.append(Polyline(points))
        planes.append(Plane.from_point_normal(origin, normal))

    if b.is_solid() and _outline_volume(polylines) < 0.0:
        for i in range(len(planes)):
            n = planes[i].z_axis
            n.reverse()
            planes[i] = Plane.from_point_normal(planes[i].origin, n)

    return polylines, planes


# ═══════════════════════════════════════════════════════════════════════════
# Meshing helpers
# ═══════════════════════════════════════════════════════════════════════════
class _EdgeBoundary:
    """Canonical boundary of every shared edge: model points, the (face, pcurve, parameters) that produced them, and refined (t, uv) samples"""

    def __init__(self):
        self.points: dict[int, list[Point]] = {}  # Canonical model points per edge.
        self.basis = {}  # (face, pcurve, parameters) per edge.
        self.samples = {}  # Refined (t, uv) samples per edge.


def _parameter(item: tuple) -> float:
    """Sort key of a (parameter, ...) tuple: its parameter"""

    return item[0]


def _uv_key(p: Point) -> tuple[float, float]:
    """Sort key of a UV point: u, then v"""

    return (p[0], p[1])


def _wire_uv_points(b: BRep, face_index: int, wire: BRepRef) -> list[Point]:
    """UV polygon of one wire of a face (pcurves sampled in traversal order)"""

    pts = []

    for er in b.wire_edges(wire):
        ci = b.pcurve_index(er.index, face_index, er.orientation)

        if ci < 0:
            continue

        crv = b.m_curves_2d[ci]
        seg = []

        if crv.degree() <= 1 and not crv.is_rational():
            for k in range(crv.cv_count()):
                seg.append(crv.get_cv(k))
        else:
            seg = crv.divide_by_count(max(crv.cv_count() * 4, 16), True)[0]

        if er.orientation == BRepOrientation.Reversed:
            seg.reverse()

        for k in range(len(seg) - 1):
            pts.append(seg[k])

    return pts


def _lifted_distance(
    surface: NurbsSurface, curve: NurbsCurve, point: Point, t: float
) -> float:
    """Distance from `point` to the surface point the pcurve reaches at t"""

    uv = curve.point_at(t)

    return surface.point_at(uv[0], uv[1]).distance(point)


def _boundary_parameter(
    surface: NurbsSurface, curve: NurbsCurve, point: Point
) -> float:
    """Parameter of the lifted pcurve closest to `point`: a coarse scan then 64 golden-section steps in the best cell"""

    start, end = curve.domain()
    count = min(max(curve.cv_count() * 4, 32), 4096)
    step = (end - start) / count
    best = start
    error = _lifted_distance(surface, curve, point, start)

    for index in range(1, count + 1):
        t = end if index == count else start + index * step
        candidate = _lifted_distance(surface, curve, point, t)

        if candidate < error:
            best = t
            error = candidate

    left = max(best - step, start)
    right = min(best + step, end)
    ratio = (math.sqrt(5.0) - 1.0) * 0.5
    a = right - ratio * (right - left)
    b = left + ratio * (right - left)
    da = _lifted_distance(surface, curve, point, a)
    db = _lifted_distance(surface, curve, point, b)

    for _ in range(64):
        if da < db:
            right = b
            b = a
            db = da
            a = right - ratio * (right - left)
            da = _lifted_distance(surface, curve, point, a)
        else:
            left = a
            a = b
            da = db
            b = left + ratio * (right - left)
            db = _lifted_distance(surface, curve, point, b)

    if da < error:
        best = a
        error = da

    if db < error:
        best = b

    return best


def _boundary_normal(
    surface: NurbsSurface, curve: NurbsCurve, t: float, toward: float
) -> Vector | None:
    """Unit normal on a boundary, taking the one-sided limit toward `toward` at a singular endpoint"""

    for at in (t, t + (toward - t) * 1e-6):
        uv = curve.point_at(at)
        derivatives = surface.evaluate(uv[0], uv[1], 1)

        if len(derivatives) < 3:
            continue

        n = derivatives[1].cross(derivatives[2])
        scale = max(abs(n[0]), abs(n[1]), abs(n[2]))

        if not math.isfinite(scale) or scale == 0.0:
            continue

        n = n / scale
        length = n.magnitude()

        if math.isfinite(length) and length > 0.0:
            return n / length

    return None


def _boundary_turns(
    surface: NurbsSurface,
    curve: NurbsCurve,
    ta: float,
    t: float,
    tb: float,
    cosine: float,
) -> bool:
    """True when the normals at ta, t and tb turn more than the angle whose cosine is given"""

    normals = [
        _boundary_normal(surface, curve, ta, tb),
        _boundary_normal(surface, curve, t, ta),
        _boundary_normal(surface, curve, tb, ta),
    ]

    for j in range(3):
        for k in range(j + 1, 3):
            if normals[j] is None or normals[k] is None:
                continue

            if normals[j].dot(normals[k]) < cosine:
                return True

    return False


def _refine_surface_boundary(
    surface: NurbsSurface,
    curve: NurbsCurve,
    samples: list[tuple[float, Point, Point]],
    angle: float,
    chord: float,
) -> list[tuple[float, Point, Point]]:
    """Refine samples of a lifted pcurve until chord and angle hold; existing samples stay exact, eight split levels and 4096 added points per edge bound the work"""

    if len(samples) < 2:
        return samples

    tolerance = _bbox_diagonal(surface) * chord
    cosine = math.cos(min(max(angle, 0.1), 179.0) * PI / 180.0)
    result = []
    added = 0

    for i in range(1, len(samples)):
        stack = [(samples[i - 1], samples[i], 0)]

        while stack:
            a, b, depth = stack.pop()
            t = (a[0] + b[0]) * 0.5
            uv = curve.point_at(t)
            point = surface.point_at(uv[0], uv[1])
            pa = a[2]
            pb = b[2]
            center = Point(
                (pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5
            )
            split = (
                (
                    point.distance(center) > tolerance
                    or _boundary_turns(surface, curve, a[0], t, b[0], cosine)
                )
                and depth < 8
                and added < 4096
            )

            if not split:
                result.append(a)
                continue

            added += 1
            middle = (t, uv, point)
            stack.append((middle, b, depth + 1))
            stack.append((a, middle, depth + 1))

    result.append(samples[-1])

    return result


def _same_boundary_point(a: Point, b: Point) -> bool:
    """Compare canonical boundary positions exactly, without tolerance"""

    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def _direct_face(b: BRep, fi: int) -> bool:
    """Phase 1: the outer wire is the full UV rectangle (straight pcurves enclosing the whole domain, no holes), so the face meshes directly on the surface grid"""

    face = b.m_faces[fi]

    if len(face.wires) != 1:
        return False

    for er in b.wire_edges(face.wires[0]):
        ci = b.pcurve_index(er.index, fi, er.orientation)

        if ci < 0:
            continue

        if b.m_curves_2d[ci].degree() > 1 or b.m_curves_2d[ci].is_rational():
            return False

    outer = _wire_uv_points(b, fi, face.wires[0])

    if len(outer) < 3:
        return False

    srf = b.m_surfaces[face.surface_index]
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)

    for er in b.wire_edges(face.wires[0]):
        ci = b.pcurve_index(er.index, fi, er.orientation)

        if ci < 0:
            continue

        curve = b.m_curves_2d[ci]

        for k in (0, max(0, curve.cv_count() - 1)):
            p = curve.get_cv(k)
            corner_u = min(abs(p[0] - u0), abs(p[0] - u1)) <= (u1 - u0) * 1e-9
            corner_v = min(abs(p[1] - v0), abs(p[1] - v1)) <= (v1 - v0) * 1e-9

            if not corner_u or not corner_v:
                return False

    domain_area = (u1 - u0) * (v1 - v0)

    return abs(abs(_polygon_signed_area(outer)) - domain_area) < 1e-3 * domain_area


def _grid_side_points(
    grid: Mesh,
    srf: NurbsSurface,
    at_v0: bool,
    at_v1: bool,
    at_u0: bool,
    at_u1: bool,
) -> list[tuple[float, Point]]:
    """Grid vertices on the domain sides flagged by (at_v0, at_v1, at_u0, at_u1), as (parameter along the side, model point) sorted and unique"""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    utol = (u1 - u0) * 0.001
    vtol = (v1 - v0) * 0.001
    pts: list[tuple[float, Point]] = []

    for vd in grid.vertex.values():
        if "u" not in vd.attributes or "v" not in vd.attributes:
            continue

        iu = vd.attributes["u"]
        iv = vd.attributes["v"]

        if (at_v0 and abs(iv - v0) < vtol * 0.1) or (
            at_v1 and abs(iv - v1) < vtol * 0.1
        ):
            pts.append((iu, vd.position()))
        elif (at_u0 and abs(iu - u0) < utol * 0.1) or (
            at_u1 and abs(iu - u1) < utol * 0.1
        ):
            pts.append((iv, vd.position()))

    pts.sort(key=_parameter)
    unique: list[tuple[float, Point]] = []

    for p in pts:
        if not unique or unique[-1][0] != p[0]:
            unique.append(p)

    return unique


def _grid_edge_samples(
    b: BRep, fi: int, grid: Mesh, er: BRepRef
) -> list[tuple[float, Point]]:
    """Phase 2: grid vertices of a direct face along a shared edge that runs on a domain side, as (pcurve parameter, model point) sorted along the edge; empty elsewhere"""

    samples: list[tuple[float, Point]] = []
    shared = False

    for fr in b.edge_faces(er.index):
        if fr.index != fi:
            shared = True

    if not shared:
        return samples

    ci = b.pcurve_index(er.index, fi, er.orientation)

    if ci < 0:
        return samples

    srf = b.m_surfaces[b.m_faces[fi].surface_index]
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    utol = (u1 - u0) * 0.001
    vtol = (v1 - v0) * 0.001
    c2d = b.m_curves_2d[ci]
    sp = c2d.get_cv(0)
    ep = c2d.get_cv(c2d.cv_count() - 1)
    at_v0 = abs(sp[1] - v0) < vtol and abs(ep[1] - v0) < vtol
    at_v1 = abs(sp[1] - v1) < vtol and abs(ep[1] - v1) < vtol
    at_u0 = abs(sp[0] - u0) < utol and abs(ep[0] - u0) < utol
    at_u1 = abs(sp[0] - u1) < utol and abs(ep[0] - u1) < utol

    if not at_v0 and not at_v1 and not at_u0 and not at_u1:
        return samples

    pts = _grid_side_points(grid, srf, at_v0, at_v1, at_u0, at_u1)

    if len(pts) < 2:
        return samples

    varying = 0 if at_v0 or at_v1 else 1
    t0, t1 = c2d.domain()

    for p, pt in pts:
        samples.append(
            (t0 + (p - sp[varying]) / (ep[varying] - sp[varying]) * (t1 - t0), pt)
        )

    return samples


def _grid_boundaries(b: BRep, fi: int, grid: Mesh, boundary: _EdgeBoundary) -> bool:
    """Phase 2: the first incident grid supplies the canonical polygon of every shared edge; True when this face's grid disagrees with an earlier one and must be rebuilt"""

    rebuild = False

    for er in b.wire_edges(b.m_faces[fi].wires[0]):
        eidx = er.index
        samples = _grid_edge_samples(b, fi, grid, er)

        if not samples:
            continue

        parameters = []
        bnd = []

        for t, pt in samples:
            parameters.append(t)
            bnd.append(pt)

        if eidx not in boundary.points:
            boundary.points[eidx] = bnd
            boundary.basis[eidx] = (
                fi,
                b.pcurve_index(eidx, fi, er.orientation),
                parameters,
            )
            continue

        canonical = boundary.points[eidx]
        forward = True
        backward = True

        for k in range(min(len(canonical), len(bnd))):
            forward = forward and _same_boundary_point(canonical[k], bnd[k])
            backward = backward and _same_boundary_point(
                canonical[k], bnd[len(bnd) - 1 - k]
            )

        matches = len(canonical) == len(bnd) and (forward or backward)
        rebuild = rebuild or not matches

    return rebuild


def _refine_shared_boundaries(
    b: BRep,
    face_direct: list[bool],
    rebuild_grid: list[bool],
    boundary: _EdgeBoundary,
    angle: float,
    chord: float,
) -> None:
    """Refine the canonical polygon of every edge shared with a curved CDT face, then mark every incident face for rebuild with the same refined polygon"""

    for edge in sorted(boundary.basis):
        face, pcurve, parameters = boundary.basis[edge]
        curved_cdt = False

        for incident in b.edge_faces(edge):
            fi = incident.index
            cdt = not face_direct[fi] or rebuild_grid[fi]
            curved_cdt = curved_cdt or (
                cdt
                and not b.m_surfaces[b.m_faces[fi].surface_index].is_planar(None, 0.0)
            )

        if not curved_cdt:
            continue

        surface = b.m_surfaces[b.m_faces[face].surface_index]
        curve = b.m_curves_2d[pcurve]
        points = boundary.points[edge]
        samples = []

        for i in range(len(parameters)):
            samples.append((parameters[i], curve.point_at(parameters[i]), points[i]))

        samples.sort(key=_parameter)
        end = curve.domain()[1]

        if (
            b.m_edges[edge].start_vertex == b.m_edges[edge].end_vertex
            and samples
            and samples[-1][0] < end
        ):
            samples.append((end, curve.point_at(end), samples[0][2]))

        refined = _refine_surface_boundary(surface, curve, samples, angle, chord)

        if len(refined) <= len(samples):
            continue

        refined_points = []
        refined_samples = []

        for t, uv, p in refined:
            refined_points.append(p)
            refined_samples.append((t, uv))

        boundary.points[edge] = refined_points
        boundary.samples[edge] = refined_samples

        for incident in b.edge_faces(edge):
            rebuild_grid[incident.index] = True


def _grid_interior_uv(srf: NurbsSurface, grid: Mesh) -> list[Point]:
    """Interior UV seeds of a rebuilt face: its grid vertices strictly inside the domain"""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    seeds = []

    for vertex in grid.vertex.values():
        if "u" not in vertex.attributes or "v" not in vertex.attributes:
            continue

        u = vertex.attributes["u"]
        v = vertex.attributes["v"]

        if u0 < u < u1 and v0 < v < v1:
            seeds.append(Point(u, v, 0.0))

    seeds.sort(key=_uv_key)

    return seeds


def _planar_patch_tolerance(srf: NurbsSurface) -> float:
    """Planarity tolerance for a surface of any size: 1e-9 of its control-point bounding box diagonal, never below the zero tolerance"""

    lo = [1e300, 1e300, 1e300]
    hi = [-1e300, -1e300, -1e300]

    for i in range(srf.cv_count(0)):
        for j in range(srf.cv_count(1)):
            p = srf.get_cv(i, j)

            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])

    diagonal = math.sqrt(
        (hi[0] - lo[0]) * (hi[0] - lo[0])
        + (hi[1] - lo[1]) * (hi[1] - lo[1])
        + (hi[2] - lo[2]) * (hi[2] - lo[2])
    )

    return max(1e-9 * diagonal, Tolerance.ZERO_TOLERANCE)


def _is_planar_patch(srf: NurbsSurface) -> bool:
    """True for a surface flat within _planar_patch_tolerance, whatever its coordinates"""

    return srf.is_planar(None, _planar_patch_tolerance(srf))


def _planar_patch_uv(srf: NurbsSurface, p: Point) -> tuple[float, float] | None:
    """Surface parameters of a point on a degree-1 parallelogram patch by two dot products; None when the patch is not that shape"""

    if (
        srf.degree(0) != 1
        or srf.degree(1) != 1
        or srf.cv_count(0) != 2
        or srf.cv_count(1) != 2
    ):
        return None

    p00 = srf.get_cv(0, 0)
    p10 = srf.get_cv(1, 0)
    p01 = srf.get_cv(0, 1)
    p11 = srf.get_cv(1, 1)
    eu = p10 - p00
    ev = p01 - p00
    skew = (p11 - p10) - ev

    if skew.magnitude() > _planar_patch_tolerance(srf):
        return None

    eu2 = eu.dot(eu)
    ev2 = ev.dot(ev)

    if eu2 <= 0.0 or ev2 <= 0.0:
        return None

    d = p - p00
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)

    return (u0 + d.dot(eu) / eu2 * (u1 - u0), v0 + d.dot(ev) / ev2 * (v1 - v0))


def _linear_pcurve_parameter(crv: NurbsCurve, uv: tuple[float, float]) -> float | None:
    """Parameter of the closest point on a two-point degree-1 pcurve by one projection; None for any other curve"""

    if crv.degree() != 1 or crv.is_rational() or crv.cv_count() != 2:
        return None

    c0 = crv.get_cv(0)
    c1 = crv.get_cv(1)
    dx = c1[0] - c0[0]
    dy = c1[1] - c0[1]
    length_squared = dx * dx + dy * dy

    if length_squared <= 0.0:
        return None

    fraction = ((uv[0] - c0[0]) * dx + (uv[1] - c0[1]) * dy) / length_squared
    fraction = min(max(fraction, 0.0), 1.0)
    t0, t1 = crv.domain()

    return t0 + fraction * (t1 - t0)


def _lift_canonical(
    b: BRep,
    fi: int,
    ei: int,
    ci: int,
    boundary: _EdgeBoundary,
    samples: list[tuple[float, Point, Point]],
) -> bool:
    """Phase 3: map the canonical points of edge `ei` onto pcurve `ci` of face `fi`, checked in model space; False when a point cannot be lifted"""

    face = b.m_faces[fi]
    edge = b.m_edges[ei]
    srf = b.m_surfaces[face.surface_index]
    crv = b.m_curves_2d[ci]
    cached = (
        ei in boundary.basis
        and boundary.basis[ei][0] == fi
        and boundary.basis[ei][1] == ci
        and ei in boundary.samples
    )
    points = boundary.points[ei]
    planar = _is_planar_patch(srf)

    for index in range(len(points)):
        p = points[index]

        if cached:
            t, q = boundary.samples[ei][index]
        else:
            uv = _planar_patch_uv(srf, p) if planar else None
            u, v = uv if uv is not None else srf.closest_parameters(p)
            t = _linear_pcurve_parameter(crv, (u, v)) if uv is not None else None

            if t is None:
                t = crv.closest_parameter(Point(u, v, 0.0))

            q = crv.point_at(t)

        scale = max(abs(p[0]), abs(p[1]), abs(p[2]), 1.0)
        tolerance = max(
            edge.tolerance, face.tolerance, math.sqrt(sys.float_info.epsilon) * scale
        )

        if srf.point_at(q[0], q[1]).distance(p) > tolerance:
            t = _boundary_parameter(srf, crv, p)
            q = crv.point_at(t)

            if srf.point_at(q[0], q[1]).distance(p) > tolerance:
                return False

        samples.append((t, q, p))

    samples.sort(key=_parameter)
    unique = []

    for sample in samples:
        if not unique or unique[-1][0] != sample[0]:
            unique.append(sample)

    samples[:] = unique

    return True


def _fresh_samples(
    srf: NurbsSurface, crv: NurbsCurve, angle: float, chord: float
) -> list[tuple[float, Point, Point]]:
    """Phase 3: fresh samples of a pcurve nobody has sampled yet, refined to the face's angle and chord"""

    points = []
    parameters = []

    if crv.degree() <= 1 and not crv.is_rational() and _is_planar_patch(srf):
        for k in range(crv.cv_count()):
            points.append(crv.get_cv(k))
            parameters.append(crv.greville_abcissa(k))
    else:
        count = min(max(crv.cv_count() * 4, math.ceil(360.0 / max(angle, 0.1))), 4096)
        points, parameters = crv.divide_by_count(count, True)

    samples = []

    for k in range(len(points)):
        q = points[k]
        samples.append((parameters[k], q, srf.point_at(q[0], q[1])))

    return _refine_surface_boundary(srf, crv, samples, angle, chord)


def _edge_use_samples(
    b: BRep,
    fi: int,
    er: BRepRef,
    boundary: _EdgeBoundary,
    angle: float,
    chord: float,
    samples: list[tuple[float, Point, Point]],
) -> bool:
    """Phase 3: samples of one edge use of a CDT face in traversal direction, a closed edge repeating its first point at the end; False when the edge has no pcurve or cannot be lifted"""

    ei = er.index
    edge = b.m_edges[ei]
    ci = b.pcurve_index(ei, fi, er.orientation)

    if ci < 0:
        return False

    crv = b.m_curves_2d[ci]

    if ei in boundary.points:
        if not _lift_canonical(b, fi, ei, ci, boundary, samples):
            return False
    else:
        samples[:] = _fresh_samples(
            b.m_surfaces[b.m_faces[fi].surface_index], crv, angle, chord
        )
        positions = []

        for sample in samples:
            positions.append(sample[2])

        boundary.points[ei] = positions

    if edge.start_vertex == edge.end_vertex and len(samples) > 1:
        first = samples[0]

        if not _same_boundary_point(first[2], samples[-1][2]):
            samples.append((crv.domain()[1], first[1], first[2]))

    if er.orientation == BRepOrientation.Reversed:
        samples.reverse()

    return len(samples) >= 2


def _trim_loops(
    b: BRep,
    fi: int,
    boundary: _EdgeBoundary,
    angle: float,
    chord: float,
    loops: TrimLoops,
    uses: list[tuple[int, int, int, int]],
) -> bool:
    """Phase 3: trim loops of a CDT face, every edge use keeping its boundary-node identities; False when some edge cannot be sampled"""

    face = b.m_faces[fi]

    for wi in range(len(face.wires)):
        uv = []
        xyz = []

        for er in b.wire_edges(face.wires[wi]):
            samples: list[tuple[float, Point, Point]] = []

            if not _edge_use_samples(b, fi, er, boundary, angle, chord, samples):
                return False

            uses.append((er.index, wi, len(uv), len(samples)))

            for k in range(len(samples) - 1):
                uv.append(samples[k][1])
                xyz.append(samples[k][2])

        loops.uv.append(uv)
        loops.xyz.append(xyz)

    return True


def _tag_loop_vertices(mesh: Mesh, loops: TrimLoops, normal: Vector) -> None:
    """Set every vertex normal and tag every loop vertex boundary/{loop}/{sample} as mesh_loops does"""

    lookup = {}

    for li in range(len(loops.xyz)):
        for k in range(len(loops.xyz[li])):
            p = loops.xyz[li][k]
            lookup.setdefault((p[0], p[1], p[2]), (li, k))

    for vd in mesh.vertex.values():
        vd.set_normal(normal[0], normal[1], normal[2])
        position = vd.position()
        hit = lookup.get((position[0], position[1], position[2]))

        if hit is not None:
            vd.attributes[f"boundary/{hit[0]}/{hit[1]}"] = 1.0


def _add_ring_faces(
    mesh: Mesh,
    border: list[Point],
    holes: list[list[Point]],
    border_2d: list[Point],
    holes_2d: list[list[Point]],
) -> None:
    """Add the border and hole vertices and the CDT triangles of their 2D rings, degenerate triangles skipped"""

    from .remesh_cdt import cdt_triangulate

    vkeys = []

    for p in border:
        vkeys.append(mesh.add_vertex(p))

    for hole in holes:
        for p in hole:
            vkeys.append(mesh.add_vertex(p))

    for t in cdt_triangulate(border_2d, holes_2d):
        if t[0] != t[1] and t[1] != t[2] and t[2] != t[0]:
            mesh.add_face([vkeys[t[0]], vkeys[t[1]], vkeys[t[2]]])


def _wind_to_normal(mesh: Mesh, normal: Vector) -> None:
    """Flip the mesh when its first face winds against the normal"""

    if not mesh.face:
        return

    fverts = next(iter(mesh.face.values()))
    a = mesh.vertex[fverts[0]].position()
    b = mesh.vertex[fverts[1]].position()
    c = mesh.vertex[fverts[2]].position()

    if (b - a).cross(c - a).dot(normal) < 0.0:
        mesh.flip()


def _planar_loops_mesh(srf: NurbsSurface, loops: TrimLoops) -> Mesh:
    """Phase 3 for a planar face: the sampled loops triangulated as one polygon with holes, wound to the surface normal, every loop vertex tagged boundary/{loop}/{sample} as mesh_loops does; no grid, no surface evaluation"""

    from .remesh_cdt import _project_2d
    from .remesh_cdt import _signed_area

    mesh = Mesh()

    if not loops.xyz or len(loops.xyz[0]) < 3:
        return mesh

    all_pts = []

    for loop in loops.xyz:
        all_pts.extend(loop)

    origin, xaxis, yaxis, _zaxis = Polyline(all_pts).get_average_plane()
    border = list(loops.xyz[0])
    border_2d = _project_2d(border, origin, xaxis, yaxis)

    if _signed_area(border_2d) < 0.0:
        border.reverse()
        border_2d.reverse()

    holes = []
    holes_2d = []

    for li in range(1, len(loops.xyz)):
        if len(loops.xyz[li]) < 3:
            continue

        hole = list(loops.xyz[li])
        hole_2d = _project_2d(hole, origin, xaxis, yaxis)

        if _signed_area(hole_2d) > 0.0:
            hole.reverse()
            hole_2d.reverse()

        holes.append(hole)
        holes_2d.append(hole_2d)

    _add_ring_faces(mesh, border, holes, border_2d, holes_2d)

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    normal = srf.normal_at(0.5 * (u0 + u1), 0.5 * (v0 + v1))
    _wind_to_normal(mesh, normal)
    _tag_loop_vertices(mesh, loops, normal)

    return mesh


def _tag_edge_uses(
    mesh: Mesh, loops: TrimLoops, uses: list[tuple[int, int, int, int]]
) -> None:
    """Tag every boundary vertex of a CDT mesh with the edge use it samples; each use keeps both ends, including the next edge's start"""

    for use_id in range(len(uses)):
        edge, li, start, count = uses[use_id]
        length = len(loops.uv[li])

        if length == 0:
            continue

        for sample in range(count):
            key = f"boundary/{li}/{(start + sample) % length}"
            tag = f"brep_edge/{edge}/{use_id}/{sample}"

            for vd in mesh.vertex.values():
                if key in vd.attributes:
                    vd.attributes[tag] = 1.0

            if sample + 1 >= count:
                continue

            interval = f"boundary_interval/{li}/{(start + sample) % length}"
            interval_tag = f"brep_edge_interval/{edge}/{use_id}/{sample}"

            for vd in mesh.vertex.values():
                if interval in vd.attributes:
                    vd.attributes[interval_tag] = vd.attributes[interval]


def _flip_reversed_faces(b: BRep, fmesh: list[Mesh]) -> None:
    """Flip every face mesh of a face Reversed in its shell, vertex normals included"""

    for fi in range(len(fmesh)):
        if b.face_orientation(fi) != BRepOrientation.Reversed:
            continue

        fmesh[fi].flip()

        for vd in fmesh[fi].vertex.values():
            n = vd.normal()

            if n is not None:
                vd.set_normal(-n[0], -n[1], -n[2])


# ═══════════════════════════════════════════════════════════════════════════
# Cutting helpers
# ═══════════════════════════════════════════════════════════════════════════
def _face_rings(b: BRep, rings: Mesh) -> bool:
    """Vertex rings of every face in one mesh keyed by BRep vertex index, outer rings wound to the face normal, holes as face holes; False when some face or edge is curved"""

    planar = True

    for vi in range(b.vertex_count()):
        rings.add_vertex(b.m_vertices[vi].point, vi)

    for fi in range(b.face_count()):
        surface = b.m_surfaces[b.m_faces[fi].surface_index]
        u = surface.domain(0)
        v = surface.domain(1)
        frame = Plane.from_point_normal(
            Point(0.0, 0.0, 0.0),
            b.normal_at(fi, (u[0] + u[1]) * 0.5, (v[0] + v[1]) * 0.5),
        )
        loops = []
        outer = []
        planar = planar and _is_planar_patch(surface)

        for wire in b.m_faces[fi].wires:
            loops.append([])

            for er in b.wire_edges(wire):
                edge = b.m_edges[er.index]

                if edge.degenerated:
                    continue

                planar = planar and b.m_curves_3d[edge.curve_3d_index].degree() == 1
                loops[-1].append(
                    edge.end_vertex
                    if er.orientation == BRepOrientation.Reversed
                    else edge.start_vertex
                )

        for vi in loops[0]:
            outer.append(b.m_vertices[vi].point)

        if _signed_area_in_plane(outer, frame.origin, frame.x_axis, frame.y_axis) < 0.0:
            loops[0].reverse()

        fk = rings.add_face(loops[0])

        if fk is not None and len(loops) > 1:
            rings.set_face_holes(fk, loops[1:])

    return planar


def _ring_polyline(mesh: Mesh, ring: list[int]) -> Polyline:
    """Closed polyline through the positions of a vertex ring of `mesh`"""

    points = []

    for vk in ring:
        points.append(mesh.vertex[vk].position())

    points.append(points[0])

    return Polyline(points)


# ═══════════════════════════════════════════════════════════════════════════
# Serialization helpers
# ═══════════════════════════════════════════════════════════════════════════
def _refs_to_json(refs: list[BRepRef]) -> list:
    """JSON array of oriented references"""

    arr = []

    for r in refs:
        arr.append(
            {"index": r.index, "orientation": _orientation_to_str(r.orientation)}
        )

    return arr


def _refs_from_json(arr) -> list[BRepRef]:
    """Oriented references of a JSON array"""

    refs = []

    for r in arr:
        refs.append(BRepRef(r["index"], _orientation_from_str(r["orientation"])))

    return refs


def _edge_to_json(e: BRepEdge) -> dict:
    """JSON object of an edge, fields in alphabetical order"""

    ej = {}
    ej["curve_3d_index"] = e.curve_3d_index
    ej["degenerated"] = e.degenerated
    ej["end_vertex"] = e.end_vertex
    ej["pcurves"] = []

    for pc in e.pcurves:
        ej["pcurves"].append(
            {
                "curve_2d_index": pc.curve_2d_index,
                "curve_2d_index_2": pc.curve_2d_index_2,
                "surface_index": pc.surface_index,
            }
        )

    ej["start_vertex"] = e.start_vertex
    ej["tolerance"] = e.tolerance

    return ej


def _edge_from_json(e: dict) -> BRepEdge:
    """Edge of a JSON object"""

    be = BRepEdge()
    be.curve_3d_index = e["curve_3d_index"]
    be.degenerated = e["degenerated"]
    be.end_vertex = e["end_vertex"]

    for pc in e["pcurves"]:
        be.pcurves.append(
            BRepCurveOnSurface(
                pc["surface_index"], pc["curve_2d_index"], pc["curve_2d_index_2"]
            )
        )

    be.start_vertex = e["start_vertex"]
    be.tolerance = e["tolerance"]

    return be


def _face_to_json(f: BRepFace) -> dict:
    """JSON object of a face, facecolor only when set"""

    fj = {}

    if f.facecolor is not None:
        fj["facecolor"] = f.facecolor.__jsondump__()

    fj["surface_index"] = f.surface_index
    fj["tolerance"] = f.tolerance
    fj["wires"] = _refs_to_json(f.wires)

    return fj


def _face_from_json(f: dict) -> BRepFace:
    """Face of a JSON object"""

    from .file_encoders import file_decode_node

    bf = BRepFace()

    if "facecolor" in f:
        bf.facecolor = file_decode_node(f["facecolor"])

    bf.surface_index = f["surface_index"]
    bf.tolerance = f["tolerance"]
    bf.wires = _refs_from_json(f["wires"])

    return bf


def _refs_to_proto(refs: list[BRepRef], out) -> None:
    """Append oriented references to a repeated proto field"""

    for r in refs:
        p = out.add()
        p.index = r.index
        p.orientation = r.orientation


def _refs_from_proto(arr) -> list[BRepRef]:
    """Oriented references of a repeated proto field"""

    refs = []

    for r in arr:
        refs.append(BRepRef(r.index, int(r.orientation)))

    return refs


def _edge_to_proto(e: BRepEdge, p) -> None:
    """Fill a proto edge"""

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


def _edge_from_proto(e) -> BRepEdge:
    """Edge of a proto edge"""

    be = BRepEdge()
    be.curve_3d_index = e.curve_3d_index
    be.start_vertex = e.start_vertex
    be.end_vertex = e.end_vertex
    be.tolerance = e.tolerance
    be.degenerated = e.degenerated

    for pc in e.pcurves:
        be.pcurves.append(
            BRepCurveOnSurface(pc.surface_index, pc.curve_2d_index, pc.curve_2d_index_2)
        )

    return be


def _face_to_proto(f: BRepFace, p) -> None:
    """Fill a proto face, facecolor only when set"""

    p.surface_index = f.surface_index
    _refs_to_proto(f.wires, p.wires)
    p.tolerance = f.tolerance

    if f.facecolor is not None:
        p.facecolor.CopyFrom(f.facecolor.to_proto())


def _face_from_proto(f) -> BRepFace:
    """Face of a proto face"""

    bf = BRepFace()
    bf.surface_index = f.surface_index
    bf.wires = _refs_from_proto(f.wires)
    bf.tolerance = f.tolerance

    if f.HasField("facecolor"):
        bf.facecolor = Color.from_proto(f.facecolor)

    return bf


# ═══════════════════════════════════════════════════════════════════════════
# BRep
# ═══════════════════════════════════════════════════════════════════════════
class BRep:
    """Boundary representation after OCCT's TopoDS/BRep model: geometry pools, indexed shape tables, every parent -> child link a BRepRef carrying the orientation"""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self):
        """Construct an empty BRep."""

        self._guid = None  # Lazily minted GUID.
        self.name = "my_brep"  # BRep name.
        self.width = 1.0  # Display width.
        self.surfacecolor = Color.lightgrey()  # Display color of the faces.
        self.m_surfaces: list[NurbsSurface] = []  # Surface pool.
        self.m_curves_3d: list[NurbsCurve] = []  # 3D edge curve pool.
        self.m_curves_2d: list[NurbsCurve] = []  # Pcurve pool.
        self.m_vertices: list[BRepVertex] = []  # Vertex table.
        self.m_edges: list[BRepEdge] = []  # Edge table.
        self.m_wires: list[BRepWire] = []  # Wire table.
        self.m_faces: list[BRepFace] = []  # Face table.
        self.m_shells: list[BRepShell] = []  # Shell table.
        self.m_solids: list[BRepSolid] = []  # Solid table.

    def duplicate(self) -> BRep:
        """Copy with a new guid and the same data."""

        b = copy.deepcopy(self)
        b._guid = None

        return b

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def create_box(sx: float, sy: float, sz: float) -> BRep:
        """Axis-aligned box centered at the origin: 6 faces, 12 edges, 8 vertices, one solid"""

        b = BRep()
        b.name = "box"
        _box_corners(b, sx, sy, sz)
        pb = _PolyFaceBuilder(b)
        faces = []

        for fv in _BOX_FACES:
            faces.append(BRepRef(pb.face(_quad_patch(b, fv), fv), F))

        b.add_solid([BRepRef(b.add_shell(faces), F)])

        return b

    @staticmethod
    def create_cylinder(radius: float, height: float) -> BRep:
        """Cylinder along +Z: one periodic body face (seam edge) and two planar caps"""

        b = BRep()
        b.name = "cylinder"
        body = Primitives.cylinder_surface(0, 0, 0, radius, height)
        p_bot = body.point_at_corner(0, 0)
        p_top = body.point_at_corner(0, 1)
        v_bot = b.add_vertex(p_bot)
        v_top = b.add_vertex(p_top)
        e_bot = b.add_edge(
            b.add_curve_3d(Primitives.circle(0, 0, 0, radius)), v_bot, v_bot
        )
        e_top = b.add_edge(
            b.add_curve_3d(Primitives.circle(0, 0, height, radius)), v_top, v_top
        )
        e_seam = b.add_edge(
            b.add_curve_3d(NurbsCurve.create(False, 1, [p_bot, p_top])), v_bot, v_top
        )
        f_body = _body_face(b, b.add_surface(body), e_bot, e_seam, e_top)
        f_bot = _cap_face(b, _cap_patch(radius, 0.0, False), e_bot)
        f_top = _cap_face(b, _cap_patch(radius, height, True), e_top)
        shell = b.add_shell([BRepRef(f_body, F), BRepRef(f_bot, F), BRepRef(f_top, F)])
        b.add_solid([BRepRef(shell, F)])

        return b

    @staticmethod
    def create_sphere(radius: float) -> BRep:
        """Sphere centered at the origin: one face, a seam meridian and two degenerated pole edges"""

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
        b.add_pcurve(
            e_seam,
            si,
            b.add_curve_2d(_uv_line(u1, v0, u1, v1)),
            b.add_curve_2d(_uv_line(u0, v0, u0, v1)),
        )
        wire = b.add_wire(
            [
                BRepRef(e_south, F),
                BRepRef(e_seam, F),
                BRepRef(e_north, R),
                BRepRef(e_seam, R),
            ]
        )
        fi = b.add_face(si, [BRepRef(wire, F)])
        b.add_solid([BRepRef(b.add_shell([BRepRef(fi, F)]), F)])

        return b

    @staticmethod
    def create_cone(radius: float, height: float) -> BRep:
        """Cone along +Z: base circle at z=0, apex at z=height (degenerated apex edge), planar base"""

        b = BRep()
        b.name = "cone"
        body = Primitives.cone_surface(0, 0, 0, radius, height)
        p_base = body.point_at_corner(0, 0)
        p_apex = Point(0.0, 0.0, height)
        v_base = b.add_vertex(p_base)
        v_apex = b.add_vertex(p_apex)
        e_base = b.add_edge(
            b.add_curve_3d(Primitives.circle(0, 0, 0, radius)), v_base, v_base
        )
        e_seam = b.add_edge(
            b.add_curve_3d(NurbsCurve.create(False, 1, [p_base, p_apex])),
            v_base,
            v_apex,
        )
        e_apex = b.add_edge(-1, v_apex, v_apex)
        f_body = _body_face(b, b.add_surface(body), e_base, e_seam, e_apex)
        f_base = _cap_face(b, _cap_patch(radius, 0.0, False), e_base)
        b.add_solid([BRepRef(b.add_shell([BRepRef(f_body, F), BRepRef(f_base, F)]), F)])

        return b

    @staticmethod
    def create_pyramid(base: float, height: float) -> BRep:
        """Square pyramid: base edge `base` centered at the origin in z=0, apex at (0,0,height)"""

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
            a = i
            c = (i + 1) % 4
            srf = _bilinear_patch(
                b.m_vertices[a].point,
                b.m_vertices[c].point,
                b.m_vertices[v_apex].point,
                b.m_vertices[v_apex].point,
            )
            si = b.add_surface(srf)
            e_ac = pb.edge(a, c)
            e_c = pb.edge(c, v_apex)
            e_a = pb.edge(a, v_apex)
            e_deg = b.add_edge(-1, v_apex, v_apex)
            ac_fwd = b.m_edges[e_ac].start_vertex == a
            b.add_pcurve(
                e_ac,
                si,
                b.add_curve_2d(
                    _uv_line(0, 0, 1, 0) if ac_fwd else _uv_line(1, 0, 0, 0)
                ),
            )
            b.add_pcurve(e_c, si, b.add_curve_2d(_uv_line(1, 0, 1, 1)))
            b.add_pcurve(e_deg, si, b.add_curve_2d(_uv_line(1, 1, 0, 1)))
            b.add_pcurve(e_a, si, b.add_curve_2d(_uv_line(0, 0, 0, 1)))
            wire = b.add_wire(
                [
                    BRepRef(e_ac, F if ac_fwd else R),
                    BRepRef(e_c, F),
                    BRepRef(e_deg, F),
                    BRepRef(e_a, R),
                ]
            )
            faces.append(BRepRef(b.add_face(si, [BRepRef(wire, F)]), F))

        b.add_solid([BRepRef(b.add_shell(faces), F)])

        return b

    @staticmethod
    def create_torus(major_radius: float, minor_radius: float) -> BRep:
        """Torus in the XY plane: one face closed in both directions, two seam edges, one vertex"""

        b = BRep()
        b.name = "torus"
        srf = Primitives.torus_surface(0, 0, 0, major_radius, minor_radius)
        u0, u1 = srf.domain(0)
        v0, v1 = srf.domain(1)
        v = b.add_vertex(srf.point_at_corner(0, 0))
        e_u = b.add_edge(b.add_curve_3d(srf.iso_curve(1, u0)), v, v)
        e_v = b.add_edge(b.add_curve_3d(srf.iso_curve(0, v0)), v, v)
        si = b.add_surface(srf)
        b.add_pcurve(
            e_v,
            si,
            b.add_curve_2d(_uv_line(u0, v0, u1, v0)),
            b.add_curve_2d(_uv_line(u0, v1, u1, v1)),
        )
        b.add_pcurve(
            e_u,
            si,
            b.add_curve_2d(_uv_line(u1, v0, u1, v1)),
            b.add_curve_2d(_uv_line(u0, v0, u0, v1)),
        )
        wire = b.add_wire(
            [
                BRepRef(e_v, F),
                BRepRef(e_u, F),
                BRepRef(e_v, R),
                BRepRef(e_u, R),
            ]
        )
        fi = b.add_face(si, [BRepRef(wire, F)])
        b.add_solid([BRepRef(b.add_shell([BRepRef(fi, F)]), F)])

        return b

    @staticmethod
    def create_block_with_hole(
        sx: float, sy: float, sz: float, hole_radius: float
    ) -> BRep:
        """Axis-aligned box with a cylindrical through-hole along Z"""

        b = BRep()
        b.name = "block_with_hole"
        hz = sz * 0.5
        _box_corners(b, sx, sy, sz)
        pb = _PolyFaceBuilder(b)
        faces = []

        for fi in range(2, 6):
            faces.append(
                BRepRef(pb.face(_quad_patch(b, _BOX_FACES[fi]), _BOX_FACES[fi]), F)
            )

        p_bot = Point(hole_radius, 0.0, -hz)
        p_top = Point(hole_radius, 0.0, hz)
        v_bot = b.add_vertex(p_bot)
        v_top = b.add_vertex(p_top)
        e_bot = b.add_edge(
            b.add_curve_3d(Primitives.circle(0, 0, -hz, hole_radius)), v_bot, v_bot
        )
        e_top = b.add_edge(
            b.add_curve_3d(Primitives.circle(0, 0, hz, hole_radius)), v_top, v_top
        )
        e_seam = b.add_edge(
            b.add_curve_3d(NurbsCurve.create(False, 1, [p_bot, p_top])), v_bot, v_top
        )
        bore = Primitives.cylinder_surface(0, 0, -hz, hole_radius, sz)
        faces.append(
            BRepRef(_body_face(b, b.add_surface(bore), e_bot, e_seam, e_top), R)
        )

        for fi in range(2):
            fv = _BOX_FACES[fi]
            cap = _quad_patch(b, fv)
            si = b.add_surface(cap)
            outer = pb.wire_refs(si, fv)
            e_hole = e_bot if fi == 0 else e_top
            c2d = _project_to_patch(
                b.m_curves_3d[b.m_edges[e_hole].curve_3d_index], cap
            )
            o = F if _uv_signed_area(c2d) < 0.0 else R
            b.add_pcurve(e_hole, si, b.add_curve_2d(c2d))
            wires = [
                BRepRef(b.add_wire(outer), F),
                BRepRef(b.add_wire([BRepRef(e_hole, o)]), F),
            ]
            faces.append(BRepRef(b.add_face(si, wires), F))

        b.add_solid([BRepRef(b.add_shell(faces), F)])

        return b

    @staticmethod
    def from_polylines(
        polylines: list[Polyline], holes: list[list[Polyline]] | None = None
    ) -> BRep:
        """One planar face per closed polyline, holes[i] the closed polylines bounding the holes of face i; coincident vertices and edges are shared, closed sheets become solids"""

        b = BRep()
        b.name = "polysurface"
        tol = 1e-6
        holes = holes if holes is not None else []
        pb = _PolyFaceBuilder(b)

        for pi in range(len(polylines)):
            pts = _open_points(polylines[pi])

            if len(pts) < 3:
                continue

            org, plane = polylines[pi].get_fast_plane()

            if not plane.is_valid():
                continue

            xa = plane.x_axis
            ya = plane.y_axis
            outer_area = _signed_area_in_plane(pts, org, xa, ya)
            vi = []

            for pt in pts:
                vi.append(_find_or_add_vertex(b, pt, tol))

            all_pts = list(pts)
            hole_cycles = []

            if pi < len(holes):
                for h in holes[pi]:
                    hp = _open_points(h)

                    if len(hp) < 3:
                        continue

                    if _signed_area_in_plane(hp, org, xa, ya) * outer_area > 0.0:
                        hp.reverse()

                    cycle = []

                    for pt in hp:
                        cycle.append(_find_or_add_vertex(b, pt, tol))

                    hole_cycles.append(cycle)
                    all_pts.extend(hp)

            pb.face(_planar_patch_through(all_pts, org, xa, ya), vi, hole_cycles)

        _close_free_faces(b)

        return b

    @staticmethod
    def from_nurbscurves(
        curves: list[NurbsCurve], holes: list[list[NurbsCurve]] | None = None
    ) -> BRep:
        """One planar face per closed curve with optional hole curves (inner wires); closed sheets become solids"""

        b = BRep()
        b.name = "polysurface"
        tol = 1e-6
        holes = holes if holes is not None else []

        for ci in range(len(curves)):
            crv = curves[ci]
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

            si = b.add_surface(
                _planar_patch_through(pts, org, plane.x_axis, plane.y_axis)
            )
            wires = [BRepRef(_curve_wire(b, crv, si, tol), F)]

            if ci < len(holes):
                for h in holes[ci]:
                    wires.append(BRepRef(_curve_wire(b, h, si, tol), F))

            b.add_face(si, wires)

        _close_free_faces(b)

        return b

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __eq__(self, other):
        """Compare name, width, color and table sizes; guid ignored."""

        if not isinstance(other, BRep):
            return False

        return (
            self.name == other.name
            and self.width == other.width
            and self.surfacecolor == other.surfacecolor
            and len(self.m_surfaces) == len(other.m_surfaces)
            and len(self.m_vertices) == len(other.m_vertices)
            and len(self.m_edges) == len(other.m_edges)
            and len(self.m_wires) == len(other.m_wires)
            and len(self.m_faces) == len(other.m_faces)
            and len(self.m_shells) == len(other.m_shells)
            and len(self.m_solids) == len(other.m_solids)
        )

    def __ne__(self, other):
        """Compare name, width, color and table sizes; guid ignored."""

        return not self.__eq__(other)

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

    def refresh_guid(self) -> None:
        """Clear the guid so a fresh one mints lazily on next read"""

        self._guid = None

    def vertex_count(self) -> int:
        """Return the number of vertices."""

        return len(self.m_vertices)

    def edge_count(self) -> int:
        """Return the number of edges."""

        return len(self.m_edges)

    def wire_count(self) -> int:
        """Return the number of wires."""

        return len(self.m_wires)

    def face_count(self) -> int:
        """Return the number of faces."""

        return len(self.m_faces)

    def shell_count(self) -> int:
        """Return the number of shells."""

        return len(self.m_shells)

    def solid_count(self) -> int:
        """Return the number of solids."""

        return len(self.m_solids)

    def is_valid(self) -> bool:
        """Every reference resolves into its table, every face has a surface and an outer wire, every edge two vertices and (unless degenerated) a 3D curve"""

        if not self.m_faces:
            return False

        for e in self.m_edges:
            if not _in_range(e.start_vertex, len(self.m_vertices)) or not _in_range(
                e.end_vertex, len(self.m_vertices)
            ):
                return False

            if not e.degenerated and not _in_range(
                e.curve_3d_index, len(self.m_curves_3d)
            ):
                return False

            for pc in e.pcurves:
                if not _in_range(
                    pc.surface_index, len(self.m_surfaces)
                ) or not _in_range(pc.curve_2d_index, len(self.m_curves_2d)):
                    return False

                if pc.curve_2d_index_2 >= 0 and not _in_range(
                    pc.curve_2d_index_2, len(self.m_curves_2d)
                ):
                    return False

        for w in self.m_wires:
            if not w.edges:
                return False

            for r in w.edges:
                if not _in_range(r.index, len(self.m_edges)):
                    return False

        for f in self.m_faces:
            if not _in_range(f.surface_index, len(self.m_surfaces)) or not f.wires:
                return False

            for r in f.wires:
                if not _in_range(r.index, len(self.m_wires)):
                    return False

        for s in self.m_shells:
            for r in s.faces:
                if not _in_range(r.index, len(self.m_faces)):
                    return False

        for s in self.m_solids:
            for r in s.shells:
                if not _in_range(r.index, len(self.m_shells)):
                    return False

        return True

    def is_closed(self, shell_index: int) -> bool:
        """BRep_Tool::IsClosed(shell): every non-degenerated edge is used exactly twice by the shell's faces (a seam counts twice through its two pcurves)"""

        if shell_index < 0 or shell_index >= len(self.m_shells):
            return False

        uses = [0] * len(self.m_edges)

        for fr in self.m_shells[shell_index].faces:
            for wr in self.m_faces[fr.index].wires:
                for er in self.wire_edges(wr):
                    uses[er.index] += 1

        for i in range(len(self.m_edges)):
            if not self.m_edges[i].degenerated and uses[i] != 0 and uses[i] != 2:
                return False

        return len(self.m_shells[shell_index].faces) > 0

    def is_solid(self) -> bool:
        """At least one solid, and every shell of every solid is closed"""

        if not self.m_solids:
            return False

        for s in self.m_solids:
            for r in s.shells:
                if not self.is_closed(r.index):
                    return False

        return True

    def face_orientation(self, face_index: int) -> int:
        """Orientation of a face inside its first parent shell; Forward for a free face"""

        for s in self.m_shells:
            for r in s.faces:
                if r.index == face_index:
                    return r.orientation

        return BRepOrientation.Forward

    def pcurve_index(self, edge_index: int, face_index: int, orientation: int) -> int:
        """BRep_Tool::CurveOnSurface(E, F): the pcurve index of an edge on a face's surface for the given use orientation (the REVERSED pcurve on a seam); -1 if none"""

        if edge_index < 0 or edge_index >= len(self.m_edges):
            return -1

        if face_index < 0 or face_index >= len(self.m_faces):
            return -1

        si = self.m_faces[face_index].surface_index

        for pc in self.m_edges[edge_index].pcurves:
            if pc.surface_index == si:
                return (
                    pc.curve_2d_index_2
                    if orientation == BRepOrientation.Reversed
                    and pc.curve_2d_index_2 >= 0
                    else pc.curve_2d_index
                )

        return -1

    def wire_edges(self, wire: BRepRef) -> list[BRepRef]:
        """The edges of a wire composed with the wire's own orientation (a Reversed wire is traversed backwards with every edge reversed)"""

        out = []

        if wire.index < 0 or wire.index >= len(self.m_wires):
            return out

        for r in self.m_wires[wire.index].edges:
            out.append(BRepRef(r.index, brep_compose(wire.orientation, r.orientation)))

        if wire.orientation == BRepOrientation.Reversed:
            out.reverse()

        return out

    def edge_faces(self, edge_index: int) -> list[BRepRef]:
        """Faces sharing an edge, each with the orientation of that edge use"""

        out = []

        for fi in range(len(self.m_faces)):
            fo = self.face_orientation(fi)

            for wr in self.m_faces[fi].wires:
                for er in self.wire_edges(wr):
                    if er.index == edge_index:
                        out.append(BRepRef(fi, brep_compose(fo, er.orientation)))

        return out

    def vertex_points(self) -> list[Point]:
        """Vertex positions, in vertex order"""

        pts = []

        for v in self.m_vertices:
            pts.append(v.point)

        return pts

    def face_polylines(self) -> list[Polyline]:
        """One closed polyline per PLANAR face: the outer wire walked in wire order with its winding untouched (lofts pair loops by it), inner wires ignored; index-aligned with face_planes"""

        return _planar_faces(self)[0]

    def face_planes(self) -> list[Plane]:
        """The plane of every face face_polylines emits: centroid origin, Newell normal flipped for a Reversed face, and the whole set flipped when a closed solid encloses negative volume so normals point outward; free faces keep the wire's sign"""

        return _planar_faces(self)[1]

    def update_tolerances(self) -> float:
        """BRepLib::UpdateTolerances: raise every edge tolerance to the worst gap between its curve ends (3D and lifted pcurves) and its vertices, every vertex to its worst edge; returns the largest"""

        worst = 0.0

        for e in self.m_edges:
            vs = self.m_vertices[e.start_vertex]
            ve = self.m_vertices[e.end_vertex]
            tol = e.tolerance

            if e.curve_3d_index >= 0:
                c = self.m_curves_3d[e.curve_3d_index]
                tol = max(tol, c.point_at(c.domain()[0]).distance(vs.point))
                tol = max(tol, c.point_at(c.domain()[1]).distance(ve.point))

            for pc in e.pcurves:
                srf = self.m_surfaces[pc.surface_index]

                for ci in (pc.curve_2d_index, pc.curve_2d_index_2):
                    if ci < 0:
                        continue

                    c2 = self.m_curves_2d[ci]
                    a = c2.point_at(c2.domain()[0])
                    z = c2.point_at(c2.domain()[1])
                    tol = max(tol, srf.point_at(a[0], a[1]).distance(vs.point))
                    tol = max(tol, srf.point_at(z[0], z[1]).distance(ve.point))

            e.tolerance = tol
            vs.tolerance = max(vs.tolerance, tol)
            ve.tolerance = max(ve.tolerance, tol)
            worst = max(worst, tol)

        return worst

    def volume(self) -> float:
        """Volume of the tessellated boundary (divergence theorem); meaningful for solids only"""

        return self.mesh().volume()

    # ═══════════════════════════════════════════════════════════════════════════
    # Building
    # ═══════════════════════════════════════════════════════════════════════════
    def add_surface(self, srf: NurbsSurface) -> int:
        """Append a surface to the pool; returns its index."""

        self.m_surfaces.append(srf)

        return len(self.m_surfaces) - 1

    def add_curve_3d(self, crv: NurbsCurve) -> int:
        """Append a 3D curve to the pool; returns its index."""

        self.m_curves_3d.append(crv)

        return len(self.m_curves_3d) - 1

    def add_curve_2d(self, crv: NurbsCurve) -> int:
        """Append a pcurve to the pool; returns its index."""

        self.m_curves_2d.append(crv)

        return len(self.m_curves_2d) - 1

    def add_vertex(self, pt: Point, tolerance: float = 0.0) -> int:
        """MakeVertex"""

        self.m_vertices.append(BRepVertex(pt, tolerance))

        return len(self.m_vertices) - 1

    def add_edge(
        self,
        curve_3d_index: int,
        start_vertex: int,
        end_vertex: int,
        tolerance: float = 0.0,
    ) -> int:
        """MakeEdge: curve_3d_index -1 makes a degenerated edge (start == end vertex)"""

        e = BRepEdge()
        e.curve_3d_index = curve_3d_index
        e.start_vertex = start_vertex
        e.end_vertex = end_vertex
        e.tolerance = tolerance
        e.degenerated = curve_3d_index < 0
        self.m_edges.append(e)

        return len(self.m_edges) - 1

    def add_pcurve(
        self,
        edge_index: int,
        surface_index: int,
        curve_2d_index: int,
        curve_2d_index_2: int = -1,
    ) -> None:
        """UpdateEdge(E, pcurve, S): attach a pcurve on a surface, curve_2d_index_2 for the reversed use on a closed surface; replaces an existing record for the same surface"""

        for pc in self.m_edges[edge_index].pcurves:
            if pc.surface_index == surface_index:
                pc.curve_2d_index = curve_2d_index
                pc.curve_2d_index_2 = curve_2d_index_2
                return

        self.m_edges[edge_index].pcurves.append(
            BRepCurveOnSurface(surface_index, curve_2d_index, curve_2d_index_2)
        )

    def add_wire(self, edges: list[BRepRef]) -> int:
        """MakeWire + Add(edges)"""

        self.m_wires.append(BRepWire(edges))

        return len(self.m_wires) - 1

    def add_face(
        self, surface_index: int, wires: list[BRepRef], tolerance: float = 0.0
    ) -> int:
        """MakeFace(S) + Add(wires); the first wire is the outer boundary"""

        f = BRepFace()
        f.surface_index = surface_index
        f.wires = list(wires)
        f.tolerance = tolerance
        self.m_faces.append(f)

        return len(self.m_faces) - 1

    def add_shell(self, faces: list[BRepRef]) -> int:
        """MakeShell + Add(faces)"""

        self.m_shells.append(BRepShell(faces))

        return len(self.m_shells) - 1

    def add_solid(self, shells: list[BRepRef]) -> int:
        """MakeSolid + Add(shells)"""

        self.m_solids.append(BRepSolid(shells))

        return len(self.m_solids) - 1

    # ═══════════════════════════════════════════════════════════════════════════
    # Meshing
    # ═══════════════════════════════════════════════════════════════════════════
    def mesh(self) -> Mesh:
        """One welded triangle mesh of every face, wound to the face's outward orientation"""

        polygons = []

        for fm in self.face_meshes():
            for fverts in fm.face.values():
                poly = []

                for vi in fverts:
                    poly.append(fm.vertex[vi].position())

                polygons.append(poly)

        return Mesh.from_polylines(polygons, 1e-6)

    def face_meshes(self) -> list[Mesh]:
        """One mesh per face, in face order (vertices not shared across faces)"""

        return self.face_meshes_q(False, 0.0, 0.0)

    def face_meshes_q(
        self, has_quality: bool, max_angle_deg: float, chord_factor: float
    ) -> list[Mesh]:
        """As face_meshes with a tessellation-quality override (max_angle_deg, chord_factor) when has_quality"""

        nf = len(self.m_faces)
        angle = max_angle_deg if has_quality else 20.0
        chord = chord_factor if has_quality else 0.005
        face_direct = [False] * nf
        rebuild_grid = [False] * nf
        fmesh = []
        boundary = _EdgeBoundary()

        for fi in range(nf):
            face_direct[fi] = _direct_face(self, fi)
            fmesh.append(Mesh())

        for fi in range(nf):
            if not face_direct[fi]:
                continue

            srf = self.m_surfaces[self.m_faces[fi].surface_index]
            fmesh[fi] = (
                RemeshNurbsSurfaceGrid.from_u_v_q(
                    srf, 0, 0, max_angle_deg, chord_factor
                )
                if has_quality
                else srf.mesh()
            )
            rebuild_grid[fi] = _grid_boundaries(self, fi, fmesh[fi], boundary)

        _refine_shared_boundaries(
            self, face_direct, rebuild_grid, boundary, angle, chord
        )

        for fi in range(nf):
            if rebuild_grid[fi]:
                face_direct[fi] = False

        for fi in range(nf):
            if face_direct[fi]:
                continue

            srf = self.m_surfaces[self.m_faces[fi].surface_index]
            loops = TrimLoops()

            if rebuild_grid[fi]:
                loops.interior_uv = _grid_interior_uv(srf, fmesh[fi])

            uses: list[tuple[int, int, int, int]] = []

            if not _trim_loops(self, fi, boundary, angle, chord, loops, uses):
                continue

            if not loops.interior_uv and _is_planar_patch(srf):
                fmesh[fi] = _planar_loops_mesh(srf, loops)
            else:
                ts = NurbsSurfaceTrimmed()
                ts.m_surface = srf
                fmesh[fi] = ts.mesh_loops(loops, angle, chord)

            _tag_edge_uses(fmesh[fi], loops, uses)

        _flip_reversed_faces(self, fmesh)

        return fmesh

    # ═══════════════════════════════════════════════════════════════════════════
    # Evaluation
    # ═══════════════════════════════════════════════════════════════════════════
    def point_at(self, face_index: int, u: float, v: float) -> Point:
        """Surface point of a face at (u, v)"""

        if face_index < 0 or face_index >= len(self.m_faces):
            return Point(0.0, 0.0, 0.0)

        return self.m_surfaces[self.m_faces[face_index].surface_index].point_at(u, v)

    def normal_at(self, face_index: int, u: float, v: float) -> Vector:
        """Surface normal of a face at (u, v), flipped when the face is Reversed in its shell"""

        if face_index < 0 or face_index >= len(self.m_faces):
            return Vector(0.0, 0.0, 0.0)

        n = self.m_surfaces[self.m_faces[face_index].surface_index].normal_at(u, v)

        if self.face_orientation(face_index) == BRepOrientation.Reversed:
            return -n

        return n

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════
    def transform(self, xform: Xform) -> None:
        """Transform surfaces, 3D curves and vertices in place (pcurves are parametric, untouched)"""

        for srf in self.m_surfaces:
            srf.transform(xform)

        for crv in self.m_curves_3d:
            crv.transform(xform)

        for v in self.m_vertices:
            v.point = xform.transform_point(v.point)

    def transformed(self, xform: Xform) -> BRep:
        """Return a transformed copy"""

        b = self.duplicate()
        b.transform(xform)

        return b

    # ═══════════════════════════════════════════════════════════════════════════
    # Cutting
    # ═══════════════════════════════════════════════════════════════════════════
    def cut_by_plane(self, plane: Plane) -> BRep:
        """Return the part on the side the plane normal points to, every section loop capped by one planar face; a copy when everything lies on that side, empty when the plane cuts a BRep with a curved face or edge"""

        rings = Mesh()

        if not _face_rings(self, rings):
            tessellation = self.mesh()

            return (
                self.duplicate()
                if tessellation.cut_by_plane(plane) == tessellation
                else BRep()
            )

        cut = rings.cut_by_plane(plane)

        if cut == rings:
            return self.duplicate()

        if cut.is_empty():
            return BRep()

        polylines = []
        holes = []

        for fk, ring in cut.face.items():
            polylines.append(_ring_polyline(cut, ring))
            holes.append([])

            for hole in cut.face_holes.get(fk, []):
                holes[-1].append(_ring_polyline(cut, hole))

        result = BRep.from_polylines(polylines, holes)
        result.name = self.name
        result.width = self.width
        result.surfacecolor = self.surfacecolor

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        j = {}
        j["curves_2d"] = []

        for c in self.m_curves_2d:
            j["curves_2d"].append(c.__jsondump__())

        j["curves_3d"] = []

        for c in self.m_curves_3d:
            j["curves_3d"].append(c.__jsondump__())

        j["edges"] = []

        for e in self.m_edges:
            j["edges"].append(_edge_to_json(e))

        j["faces"] = []

        for f in self.m_faces:
            j["faces"].append(_face_to_json(f))

        j["guid"] = self.guid
        j["name"] = self.name
        j["shells"] = []

        for s in self.m_shells:
            j["shells"].append({"faces": _refs_to_json(s.faces)})

        j["solids"] = []

        for s in self.m_solids:
            j["solids"].append({"shells": _refs_to_json(s.shells)})

        j["surfacecolor"] = self.surfacecolor.__jsondump__()
        j["surfaces"] = []

        for s in self.m_surfaces:
            j["surfaces"].append(s.__jsondump__())

        j["type"] = "BRep"
        j["vertices"] = []

        for v in self.m_vertices:
            j["vertices"].append(
                {
                    "point": [v.point[0], v.point[1], v.point[2]],
                    "tolerance": v.tolerance,
                }
            )

        j["width"] = self.width
        j["wires"] = []

        for w in self.m_wires:
            j["wires"].append({"edges": _refs_to_json(w.edges)})

        return j

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None) -> BRep:
        """Deserialize from a JSON object."""

        from .file_encoders import file_decode_node

        b = cls()
        b.guid = guid if guid is not None else data.get("guid", b.guid)
        b.name = name if name is not None else data.get("name", "my_brep")
        b.width = data.get("width", 1.0)

        if "surfacecolor" in data:
            b.surfacecolor = file_decode_node(data["surfacecolor"])

        for c in data.get("curves_2d", []):
            b.m_curves_2d.append(file_decode_node(c))

        for c in data.get("curves_3d", []):
            b.m_curves_3d.append(file_decode_node(c))

        for s in data.get("surfaces", []):
            b.m_surfaces.append(file_decode_node(s))

        for v in data.get("vertices", []):
            b.m_vertices.append(
                BRepVertex(
                    Point(v["point"][0], v["point"][1], v["point"][2]), v["tolerance"]
                )
            )

        for e in data.get("edges", []):
            b.m_edges.append(_edge_from_json(e))

        for w in data.get("wires", []):
            b.m_wires.append(BRepWire(_refs_from_json(w["edges"])))

        for f in data.get("faces", []):
            b.m_faces.append(_face_from_json(f))

        for s in data.get("shells", []):
            b.m_shells.append(BRepShell(_refs_from_json(s["faces"])))

        for s in data.get("solids", []):
            b.m_solids.append(BRepSolid(_refs_from_json(s["shells"])))

        return b

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""

        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> BRep:
        """Deserialize from a JSON string."""

        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file."""

        with open(filename, "w") as f:
            json.dump(self.__jsondump__(), f, indent=4)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> BRep:
        """Read from a JSON file."""

        with open(filename) as f:
            return cls.__jsonload__(json.load(f))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> brep_pb2.BRep:
        """Convert to the protobuf message."""

        from .proto import brep_pb2

        proto = brep_pb2.BRep()

        if self.has_guid():
            proto.guid = self.guid

        proto.name = self.name
        proto.width = self.width

        for c in self.m_curves_2d:
            proto.curves_2d.add().CopyFrom(c.to_proto())

        for c in self.m_curves_3d:
            proto.curves_3d.add().CopyFrom(c.to_proto())

        for s in self.m_surfaces:
            proto.surfaces.add().CopyFrom(s.to_proto())

        for v in self.m_vertices:
            p = proto.vertices.add()
            p.point.x = v.point[0]
            p.point.y = v.point[1]
            p.point.z = v.point[2]
            p.tolerance = v.tolerance

        for e in self.m_edges:
            _edge_to_proto(e, proto.edges.add())

        for w in self.m_wires:
            _refs_to_proto(w.edges, proto.wires.add().edges)

        for f in self.m_faces:
            _face_to_proto(f, proto.faces.add())

        for s in self.m_shells:
            _refs_to_proto(s.faces, proto.shells.add().faces)

        for s in self.m_solids:
            _refs_to_proto(s.shells, proto.solids.add().shells)

        proto.surfacecolor.CopyFrom(self.surfacecolor.to_proto())

        return proto

    @classmethod
    def from_proto(cls, proto: brep_pb2.BRep) -> BRep:
        """Construct from the protobuf message."""

        b = cls()

        if proto.guid:
            b.guid = proto.guid

        b.name = proto.name
        b.width = proto.width

        for c in proto.curves_2d:
            b.m_curves_2d.append(NurbsCurve.from_proto(c))

        for c in proto.curves_3d:
            b.m_curves_3d.append(NurbsCurve.from_proto(c))

        for s in proto.surfaces:
            b.m_surfaces.append(NurbsSurface.from_proto(s))

        for v in proto.vertices:
            b.m_vertices.append(
                BRepVertex(Point(v.point.x, v.point.y, v.point.z), v.tolerance)
            )

        for e in proto.edges:
            b.m_edges.append(_edge_from_proto(e))

        for w in proto.wires:
            b.m_wires.append(BRepWire(_refs_from_proto(w.edges)))

        for f in proto.faces:
            b.m_faces.append(_face_from_proto(f))

        for s in proto.shells:
            b.m_shells.append(BRepShell(_refs_from_proto(s.faces)))

        for s in proto.solids:
            b.m_solids.append(BRepSolid(_refs_from_proto(s.shells)))

        b.surfacecolor = Color.from_proto(proto.surfacecolor)

        return b

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""

        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> BRep:
        """Deserialize from protobuf bytes."""

        from .proto import brep_pb2

        proto = brep_pb2.BRep()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write to a protobuf file."""

        with open(filename, "wb") as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> BRep:
        """Read from a protobuf file."""

        with open(filename, "rb") as f:
            return cls.pb_loads(f.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self):
        """Return "BRep(name=..., faces=..., edges=..., vertices=...)"."""

        return f"BRep(name={self.name}, faces={self.face_count()}, edges={self.edge_count()}, vertices={self.vertex_count()})"

    def __repr__(self):
        """Return the multi-line form with the solid flag."""

        return f"BRep(\n  name={self.name},\n  faces={self.face_count()},\n  edges={self.edge_count()},\n  vertices={self.vertex_count()},\n  solid={'true' if self.is_solid() else 'false'}\n)"
