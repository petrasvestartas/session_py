from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import overload
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Callable
import uuid
import math
from enum import Enum

if TYPE_CHECKING:
    from .proto import mesh_pb2
    from pathlib import Path
    from .xform import Xform
    from .line import Line
    from .polyline import Polyline

from .point import Point
from .vector import Vector
from .tolerance import Tolerance
from .tolerance import PI
from .color import Color
from .obb import OBB
from .spatial_bvh import SpatialBVH
from .remesh_cdt import _cdt_triangulate as _cdt_triangulate


class ColorMode(Enum):
    OBJECTCOLOR = "objectcolor"
    POINTCOLORS = "pointcolors"
    FACECOLORS = "facecolors"
    NONE = "none"


class NormalWeighting(Enum):
    AREA = "area"
    ANGLE = "angle"
    UNIFORM = "uniform"


class VertexData:
    """Vertex data containing position and attributes.

    Parameters
    ----------
    point : Point, optional
        Initial position. Defaults to origin.

    Attributes
    ----------
    x : float
        X coordinate.
    y : float
        Y coordinate.
    z : float
        Z coordinate.
    attributes : dict
        Custom vertex attributes.
    """

    def __init__(self, point: Optional[Point] = None):
        if point is None:
            point = Point(0.0, 0.0, 0.0)
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]
        self.attributes = {}

    def __getitem__(self, index):
        """Access coordinate by index (0=x, 1=y, 2=z)."""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        """Set coordinate by index (0=x, 1=y, 2=z)."""
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError("Index out of range")

    def position(self) -> Point:
        """Get the vertex position as a Point."""
        return Point(self.x, self.y, self.z)

    def set_position(self, point: Point) -> None:
        """Set the vertex position from a Point."""
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]

    def color(self) -> List[float]:
        """Get the vertex color as [r, g, b]."""
        return [
            self.attributes.get("r", 0.5),
            self.attributes.get("g", 0.5),
            self.attributes.get("b", 0.5),
        ]

    def set_color(self, r: float, g: float, b: float) -> None:
        """Set the vertex color."""
        self.attributes["r"] = r
        self.attributes["g"] = g
        self.attributes["b"] = b

    def normal(self) -> Optional[List[float]]:
        """Get the vertex normal as [nx, ny, nz]."""
        if (
            "nx" in self.attributes
            and "ny" in self.attributes
            and "nz" in self.attributes
        ):
            return [self.attributes["nx"], self.attributes["ny"], self.attributes["nz"]]
        return None

    def set_normal(self, nx: float, ny: float, nz: float) -> None:
        """Set the vertex normal."""
        self.attributes["nx"] = nx
        self.attributes["ny"] = ny
        self.attributes["nz"] = nz

    def __eq__(self, other):
        if not isinstance(other, VertexData):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z and self.attributes == other.attributes

    def __ne__(self, other):
        return not self.__eq__(other)


def _lp_newell_normal(pts):
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        a = pts[i]; b = pts[(i+1)%n]
        nx += (a[1]-b[1]) * (a[2]+b[2])
        ny += (a[2]-b[2]) * (a[0]+b[0])
        nz += (a[0]-b[0]) * (a[1]+b[1])
    return nx, ny, nz


def _lp_merge_collinear(pts, vkeys):
    tol = Tolerance.APPROXIMATION
    zt2 = Tolerance.ZERO_TOLERANCE * Tolerance.ZERO_TOLERANCE
    changed = True
    while changed:
        changed = False
        m = len(pts)
        if m < 3: break
        np_, nk = [], []
        for i in range(m):
            p = (i-1+m)%m; nxt = (i+1)%m
            ax = pts[i][0]-pts[p][0]; ay = pts[i][1]-pts[p][1]; az = pts[i][2]-pts[p][2]
            bx = pts[nxt][0]-pts[i][0]; by = pts[nxt][1]-pts[i][1]; bz = pts[nxt][2]-pts[i][2]
            cx = ay*bz-az*by; cy = az*bx-ax*bz; cz = ax*by-ay*bx
            a2 = ax*ax+ay*ay+az*az; b2 = bx*bx+by*by+bz*bz
            if a2 < zt2 or b2 < zt2 or cx*cx+cy*cy+cz*cz < tol*tol*a2*b2:
                changed = True
            else:
                np_.append(pts[i]); nk.append(vkeys[i])
        pts, vkeys = np_, nk
    return pts, vkeys


def _lp_offset_toward(p, cx, cy, cz, gap):
    dx = cx-p[0]; dy = cy-p[1]; dz = cz-p[2]
    length = math.sqrt(dx*dx+dy*dy+dz*dz)
    if length > 1e-10: dx *= gap/length; dy *= gap/length; dz *= gap/length
    return Point(p[0]+dx, p[1]+dy, p[2]+dz)


def _lp_face_centroid(m, fk):
    vkeys = m.face_vertices(fk)
    cx = cy = cz = 0.0
    for vk in vkeys: p = m.vertex_point(vk); cx += p[0]; cy += p[1]; cz += p[2]
    n = len(vkeys)
    return Point(cx/n, cy/n, cz/n)


class LoftWallFace:
    def __init__(self):
        self.face_key = 0
        self.face_index = 0
        self.is_quad = False
        self.top_v0 = 0
        self.top_v1 = 0
        self.bot_v0 = 0
        self.bot_v1 = 0


class LoftPanel:
    def __init__(self):
        self.mesh = Mesh()
        self.top_face_key = 0
        self.bot_face_key = 0
        self.wall_faces = []
        self.orig_top_to_local = {}
        self.orig_bot_to_local = {}
        self.top_vertices = []
        self.bot_vertices = []
        self.face_roles = {}


class LoftAdjPair:
    def __init__(self, pi: int, wi: "LoftWallFace", pj: int, wj: "LoftWallFace"):
        self.pi = pi
        self.wi = wi
        self.pj = pj
        self.wj = wj


class Mesh:
    """A halfedge mesh data structure for representing polygonal surfaces.

    Attributes
    ----------
    halfedge : dict
        Halfedge connectivity structure mapping vertex pairs to faces.
    vertex : dict
        Vertex data dictionary mapping vertex keys to VertexData.
    face : dict
        Face vertex lists mapping face keys to vertex key lists.
    facedata : dict
        Face attributes dictionary.
    edgedata : dict
        Edge attributes dictionary.
    default_vertex_attributes : dict
        Default attributes for new vertices.
    default_face_attributes : dict
        Default attributes for new faces.
    default_edge_attributes : dict
        Default attributes for new edges.
    guid : str
        Unique identifier for the mesh.
    name : str
        Name of the mesh.
    pointcolors : list
        Vertex colors (one per vertex).
    facecolors : list
        Face colors (one per face).
    linecolors : list
        Edge colors (one per edge).
    widths : list
        Edge widths (one per edge).
    """

    def __init__(self):
        self.halfedge = {}
        self.vertex = {}
        self.face = {}
        self.facedata = {}
        self.edgedata = {}
        self.default_vertex_attributes = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.default_face_attributes = {}
        self.default_edge_attributes = {}
        self.triangulation = {}
        self.face_holes = {}
        self._max_vertex = 0
        self._max_face = 0
        self._guid = None
        self.name = "my_mesh"
        self._pointcolors = []
        self._facecolors = []
        self._linecolors = []
        self._widths = []
        self._objectcolor = None
        self.color_mode = ColorMode.OBJECTCOLOR
        self._triangle_bvh_built = False
        self._triangle_bvh = None
        self._triangle_aabbs_cache = []
        self._triangle_indices_cache = []
        self._triangle_face_subidx_cache = []
        self._vertices_cache = []

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def refresh_guid(self) -> None:
        """Clear the guid so a FRESH one mints lazily on next read — the duplicate/copy enabler."""
        self._guid = None

    def duplicate(self) -> "Mesh":
        import copy
        result = copy.copy(self)
        result.guid = str(uuid.uuid4())
        return result

    def __copy__(self):
        m = Mesh()
        m.name = self.name
        m.halfedge = {u: dict(v) for u, v in self.halfedge.items()}
        m.vertex = {k: VertexData(v.position()) for k, v in self.vertex.items()}
        for k, v in self.vertex.items():
            m.vertex[k].attributes = dict(v.attributes)
        m.face = {k: list(v) for k, v in self.face.items()}
        m.facedata = {k: dict(v) for k, v in self.facedata.items()}
        m.edgedata = {k: dict(v) for k, v in self.edgedata.items()}
        m.default_vertex_attributes = dict(self.default_vertex_attributes)
        m.default_face_attributes = dict(self.default_face_attributes)
        m.default_edge_attributes = dict(self.default_edge_attributes)
        m.triangulation = {k: list(v) for k, v in self.triangulation.items()}
        m.face_holes = {k: [list(r) for r in v] for k, v in self.face_holes.items()}
        m._max_vertex = self._max_vertex
        m._max_face = self._max_face
        m._pointcolors = list(self._pointcolors)
        m._facecolors = list(self._facecolors)
        m._linecolors = list(self._linecolors)
        m._widths = list(self._widths)
        m._objectcolor = self._objectcolor
        m.color_mode = self.color_mode
        return m

    def __eq__(self, other):
        if not isinstance(other, Mesh):
            return NotImplemented
        if self.name != other.name:
            return False
        if self.vertex != other.vertex:
            return False
        if self.face != other.face:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"Mesh(name={self.name}, vertices={self.number_of_vertices()}, faces={self.number_of_faces()})"

    def __repr__(self):
        return f"Mesh(\n  name={self.name},\n  vertices={self.number_of_vertices()},\n  faces={self.number_of_faces()},\n  edges={self.number_of_edges()}\n)"

    ###########################################################################################
    # Construction
    ###########################################################################################

    @staticmethod
    def from_polylines(
        polygons: List[List[Point]], precision: Optional[float] = None
    ) -> "Mesh":
        """Create a mesh from a list of polygons.

        Parameters
        ----------
        polygons : list[list[Point]]
            List of polygons, each polygon is a list of points.
        precision : float, optional
            Precision for vertex merging. If None, exact matching is used.

        Returns
        -------
        Mesh
            The constructed mesh with merged vertices.
        """
        mesh = Mesh()
        map_eps = {}
        map_exact = {}

        def get_vkey(p: Point) -> int:
            if precision is not None:
                kx = round(p[0] / precision)
                ky = round(p[1] / precision)
                kz = round(p[2] / precision)
                key = (kx, ky, kz)
                if key in map_eps:
                    return map_eps[key]
                vk = mesh.add_vertex(p)
                map_eps[key] = vk
                return vk
            else:
                key = (p[0], p[1], p[2])
                if key in map_exact:
                    return map_exact[key]
                vk = mesh.add_vertex(p)
                map_exact[key] = vk
                return vk

        for poly in polygons:
            if len(poly) < 3:
                continue
            vkeys = [get_vkey(p) for p in poly]
            if len(vkeys) > 1 and vkeys[-1] == vkeys[0]:
                vkeys = vkeys[:-1]
            if len(vkeys) < 3:
                continue
            fkey = mesh.add_face(vkeys)
            if len(vkeys) >= 4:
                np_ = len(vkeys)
                nx, ny, nz = 0.0, 0.0, 0.0
                for i in range(np_):
                    a, b = poly[i], poly[(i + 1) % np_]
                    nx += (a[1] - b[1]) * (a[2] + b[2])
                    ny += (a[2] - b[2]) * (a[0] + b[0])
                    nz += (a[0] - b[0]) * (a[1] + b[1])
                nlen = (nx*nx + ny*ny + nz*nz) ** 0.5
                if nlen > 1e-12:
                    nx /= nlen; ny /= nlen; nz /= nlen
                    ux, uy, uz = 1.0, 0.0, 0.0
                    if abs(nx) > 0.9:
                        ux, uy, uz = 0.0, 1.0, 0.0
                    dot = ux*nx + uy*ny + uz*nz
                    ux -= dot*nx; uy -= dot*ny; uz -= dot*nz
                    um = (ux*ux + uy*uy + uz*uz) ** 0.5
                    ux /= um; uy /= um; uz /= um
                    vx = ny*uz - nz*uy; vy = nz*ux - nx*uz; vz = nx*uy - ny*ux
                    pts2d = [Point(poly[i][0]*ux + poly[i][1]*uy + poly[i][2]*uz,
                                   poly[i][0]*vx + poly[i][1]*vy + poly[i][2]*vz, 0.0) for i in range(np_)]
                    tris = _cdt_triangulate(pts2d, [])
                    mesh.triangulation[fkey] = [[vkeys[t[0]], vkeys[t[1]], vkeys[t[2]]] for t in tris]

        return mesh

    @staticmethod
    def from_vertices_and_faces(
        vertices: List[Point], faces: List[List[int]]
    ) -> "Mesh":
        mesh = Mesh()
        vkeys = []
        for pt in vertices:
            vkeys.append(mesh.add_vertex(pt))
        for f in faces:
            mesh.add_face([vkeys[i] for i in f])
        return mesh

    @staticmethod
    def create_box(x: float, y: float, z: float) -> "Mesh":
        hx, hy, hz = x * 0.5, y * 0.5, z * 0.5
        vertices = [
            Point(-hx, -hy, -hz),
            Point( hx, -hy, -hz),
            Point( hx,  hy, -hz),
            Point(-hx,  hy, -hz),
            Point(-hx, -hy,  hz),
            Point( hx, -hy,  hz),
            Point( hx,  hy,  hz),
            Point(-hx,  hy,  hz),
        ]
        faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ]
        return Mesh.from_vertices_and_faces(vertices, faces)

    @staticmethod
    def create_dodecahedron(edge: float = 2.0) -> "Mesh":
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        ip = 1.0 / phi
        s = edge / (2.0 * ip)
        verts = [
            Point( s,  s,  s), Point( s,  s, -s), Point( s, -s,  s), Point( s, -s, -s),
            Point(-s,  s,  s), Point(-s,  s, -s), Point(-s, -s,  s), Point(-s, -s, -s),
            Point(0,  s*ip,  s*phi), Point(0,  s*ip, -s*phi),
            Point(0, -s*ip,  s*phi), Point(0, -s*ip, -s*phi),
            Point( s*ip,  s*phi, 0), Point( s*ip, -s*phi, 0),
            Point(-s*ip,  s*phi, 0), Point(-s*ip, -s*phi, 0),
            Point( s*phi, 0,  s*ip), Point( s*phi, 0, -s*ip),
            Point(-s*phi, 0,  s*ip), Point(-s*phi, 0, -s*ip),
        ]
        idx = [
            [0, 8,10, 2,16], [0,16,17, 1,12], [0,12,14, 4, 8],
            [1,17, 3,11, 9], [1, 9, 5,14,12], [2,10, 6,15,13],
            [2,13, 3,17,16], [3,13,15, 7,11], [4,14, 5,19,18],
            [4,18, 6,10, 8], [5, 9,11, 7,19], [6,18,19, 7,15],
        ]
        faces = [[verts[f[0]], verts[f[1]], verts[f[2]], verts[f[3]], verts[f[4]]] for f in idx]
        return Mesh.from_polylines(faces)

    @staticmethod
    def from_lines(
        lines: List, delete_boundary_face: bool = False, precision: Optional[float] = None
    ) -> "Mesh":
        if not lines:
            return Mesh()

        all_pts = []
        for ln in lines:
            all_pts.append(ln.start())
            all_pts.append(ln.end())

        eps = precision if precision is not None else 0.0
        if eps <= 0.0:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            zs = [p[2] for p in all_pts]
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            dz = max(zs) - min(zs)
            diag = math.sqrt(dx*dx + dy*dy + dz*dz)
            eps = diag * 1e-6
            if eps < 1e-12:
                eps = 1e-12

        vmap = {}
        verts = []
        def get_vid(p):
            kx = round(p[0] / eps)
            ky = round(p[1] / eps)
            kz = round(p[2] / eps)
            key = (kx, ky, kz)
            if key in vmap:
                return vmap[key]
            vid = len(verts)
            verts.append(p)
            vmap[key] = vid
            return vid

        adj = {}
        for ln in lines:
            a = get_vid(ln.start())
            b = get_vid(ln.end())
            if a == b:
                continue
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        nv = len(verts)

        for v in adj:
            nbrs = sorted(set(adj[v]))
            vx, vy = verts[v][0], verts[v][1]
            nbrs.sort(key=lambda n: math.atan2(verts[n][1] - vy, verts[n][0] - vx))
            adj[v] = nbrs

        visited = set()
        face_cycles = []

        for u in adj:
            for v in adj[u]:
                if (u, v) in visited:
                    continue
                cycle = []
                cu, cv = u, v
                valid = True
                while True:
                    if (cu, cv) in visited:
                        break
                    visited.add((cu, cv))
                    cycle.append(cu)
                    cv_nbrs = adj.get(cv, [])
                    if cu not in cv_nbrs:
                        valid = False
                        break
                    idx = cv_nbrs.index(cu)
                    prev_idx = (idx - 1) % len(cv_nbrs)
                    nxt = cv_nbrs[prev_idx]
                    cu, cv = cv, nxt
                    if len(cycle) > nv * 2:
                        valid = False
                        break
                if valid and len(cycle) >= 3:
                    face_cycles.append(cycle)

        if delete_boundary_face and face_cycles:
            min_idx = 0
            min_area = float('inf')
            for i, cyc in enumerate(face_cycles):
                cn = len(cyc)
                area = 0.0
                for j in range(cn):
                    a, b = cyc[j], cyc[(j+1)%cn]
                    area += verts[a][0]*verts[b][1] - verts[b][0]*verts[a][1]
                area *= 0.5
                if area < min_area:
                    min_area = area
                    min_idx = i
            face_cycles.pop(min_idx)

        mesh = Mesh()
        vkeys = []
        for pt in verts:
            vkeys.append(mesh.add_vertex(pt))
        for cycle in face_cycles:
            fvkeys = [vkeys[i] for i in cycle]
            fkey = mesh.add_face(fvkeys)
            bpts = [(verts[i][0], verts[i][1]) for i in cycle]
            area = sum(bpts[j][0]*bpts[(j+1)%len(bpts)][1] - bpts[(j+1)%len(bpts)][0]*bpts[j][1]
                       for j in range(len(bpts))) * 0.5
            ordered = list(cycle)
            if area < 0:
                bpts.reverse(); ordered.reverse()
            tris = _cdt_triangulate(bpts, [])
            mesh.triangulation[fkey] = [[vkeys[ordered[t[0]]], vkeys[ordered[t[1]]], vkeys[ordered[t[2]]]] for t in tris]
        return mesh

    @staticmethod
    def from_polygon_with_holes(
        polylines: List[List[Point]], sort_by_bbox: bool = False
    ) -> "Mesh":
        from .remesh_cdt import RemeshCDT
        from .polyline import Polyline
        if not polylines:
            return Mesh()
        pls = [Polyline(v) for v in polylines]
        return RemeshCDT.from_polylines(pls, False, not sort_by_bbox)

    @staticmethod
    def loft(polylines0: List, polylines1: List, cap: bool = True, fix_collinear: bool = True) -> "Mesh":
        if not polylines0 or not polylines1 or len(polylines0) != len(polylines1):
            return Mesh()
        border_idx = 0
        max_diag = 0.0
        for i, pl in enumerate(polylines0):
            pts = pl.get_points()
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
            dx = max(xs) - min(xs); dy = max(ys) - min(ys); dz = max(zs) - min(zs)
            diag = math.sqrt(dx*dx + dy*dy + dz*dz)
            if diag > max_diag:
                max_diag = diag; border_idx = i
        def get_open(pl):
            pts = pl.get_points()
            if len(pts) > 1:
                f, b = pts[0], pts[-1]
                if abs(f[0]-b[0]) < 1e-12 and abs(f[1]-b[1]) < 1e-12 and abs(f[2]-b[2]) < 1e-12:
                    return pts[:-1]
            return pts
        origin, xaxis, yaxis, zaxis = polylines0[border_idx].get_average_plane()
        c0 = polylines0[border_idx].center(); c1 = polylines1[border_idx].center()
        btt = Vector(c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2])
        if zaxis.dot(btt) < 0:
            yaxis = Vector(-yaxis[0], -yaxis[1], -yaxis[2])
        def proj(p):
            dx = p[0] - origin[0]; dy = p[1] - origin[1]; dz = p[2] - origin[2]
            return (dx*xaxis[0] + dy*xaxis[1] + dz*xaxis[2], dx*yaxis[0] + dy*yaxis[1] + dz*yaxis[2])
        def sarea(pts):
            a = 0.0; n = len(pts)
            for i in range(n):
                j = (i + 1) % n
                xi, yi = proj(pts[i]); xj, yj = proj(pts[j])
                a += xi*yj - xj*yi
            return a * 0.5
        order = [border_idx] + [i for i in range(len(polylines0)) if i != border_idx]
        poly_infos = []  # (bot_off, bot_n, top_off, top_n)
        all_bot = []; all_top = []
        def strip_shared_collinear(bot, top):
            cdt_scale = 1e6
            def cross_q(a, b, c):
                pa = proj(a); pb = proj(b); pc = proj(c)
                iax = round(pa[0] * cdt_scale); iay = round(pa[1] * cdt_scale)
                ibx = round(pb[0] * cdt_scale); iby = round(pb[1] * cdt_scale)
                icx = round(pc[0] * cdt_scale); icy = round(pc[1] * cdt_scale)
                return (ibx - iax) * (icy - iay) - (iby - iay) * (icx - iax)
            changed = True
            while changed and len(bot) > 3:
                changed = False
                n = len(bot)
                for i in range(n):
                    prev = (i + n - 1) % n
                    nxt = (i + 1) % n
                    if (cross_q(bot[prev], bot[i], bot[nxt]) == 0 and
                            cross_q(top[prev], top[i], top[nxt]) == 0):
                        bot.pop(i); top.pop(i)
                        changed = True
                        break
        for oi, idx in enumerate(order):
            bot = get_open(polylines0[idx]); top = get_open(polylines1[idx])
            if (oi == 0 and sarea(bot) < 0) or (oi != 0 and sarea(bot) > 0):
                bot.reverse(); top.reverse()
            if len(bot) == len(top): strip_shared_collinear(bot, top)
            poly_infos.append((len(all_bot), len(bot), len(all_top), len(top)))
            all_bot.extend(bot); all_top.extend(top)
        mesh = Mesh()
        bvk = [mesh.add_vertex(p) for p in all_bot]
        tvk = [mesh.add_vertex(p) for p in all_top]
        if cap:
            _, bot_n0, _, top_n0 = poly_infos[0]
            bpts = [Point(*proj(all_bot[i]), 0.0) for i in range(bot_n0)]
            b_hpts = [[Point(*proj(all_bot[i]), 0.0) for i in range(off, off+cnt)] for off, cnt, _, _ in poly_infos[1:]]
            bot_tris = _cdt_triangulate(bpts, b_hpts if b_hpts else [])
            fk_bot = mesh.add_face([bvk[bot_n0 - 1 - i] for i in range(bot_n0)])
            if fk_bot is not None:
                if b_hpts:
                    mesh.face_holes[fk_bot] = [[bvk[off + j] for j in range(cnt)] for off, cnt, _, _ in poly_infos[1:]]
                mesh.triangulation[fk_bot] = [[bvk[t[0]], bvk[t[2]], bvk[t[1]]] for t in bot_tris]
                if fix_collinear:
                    sc = 1e6
                    vk2d = {bvk[i]: proj(all_bot[i]) for i in range(bot_n0)}
                    fv = [bvk[bot_n0 - 1 - i] for i in range(bot_n0)]
                    tl = mesh.triangulation[fk_bot]
                    chg = True
                    while chg:
                        chg = False
                        tv = set(v for t in tl for v in t)
                        n = len(fv)
                        for k in range(n):
                            B = fv[k]
                            if B in tv:
                                continue
                            A = fv[(k + n - 1) % n]; C = fv[(k + 1) % n]
                            for j, t in enumerate(tl):
                                if A in t and C in t:
                                    if (t[0]==A or t[0]==C) and (t[1]==A or t[1]==C):
                                        tl[j] = [t[0], B, t[2]]; tl.append([B, t[1], t[2]])
                                    elif (t[1]==A or t[1]==C) and (t[2]==A or t[2]==C):
                                        tl[j] = [t[0], t[1], B]; tl.append([t[0], B, t[2]])
                                    else:
                                        tl[j] = [t[0], t[1], B]; tl.append([B, t[1], t[2]])
                                    chg = True; break
                            if chg:
                                break
                    def _zero(t):
                        u0, v0 = vk2d.get(t[0], (0.0, 0.0)); u1, v1 = vk2d.get(t[1], (0.0, 0.0)); u2, v2 = vk2d.get(t[2], (0.0, 0.0))
                        return (round(u1*sc)-round(u0*sc))*(round(v2*sc)-round(v0*sc)) - (round(v1*sc)-round(v0*sc))*(round(u2*sc)-round(u0*sc)) == 0
                    mesh.triangulation[fk_bot] = [t for t in tl if not _zero(t)]
            tpts = [Point(*proj(all_top[i]), 0.0) for i in range(top_n0)]
            t_hpts = [[Point(*proj(all_top[i]), 0.0) for i in range(off, off+cnt)] for _, _, off, cnt in poly_infos[1:]]
            top_tris = _cdt_triangulate(tpts, t_hpts if t_hpts else [])
            fk_top = mesh.add_face([tvk[i] for i in range(top_n0)])
            if fk_top is not None:
                if t_hpts:
                    mesh.face_holes[fk_top] = [[tvk[off + j] for j in range(cnt)] for _, _, off, cnt in poly_infos[1:]]
                mesh.triangulation[fk_top] = [[tvk[t[0]], tvk[t[1]], tvk[t[2]]] for t in top_tris]
                if fix_collinear:
                    sc = 1e6
                    vk2d_t = {tvk[i]: proj(all_top[i]) for i in range(top_n0)}
                    fv = [tvk[i] for i in range(top_n0)]
                    tl = mesh.triangulation[fk_top]
                    chg = True
                    while chg:
                        chg = False
                        tv = set(v for t in tl for v in t)
                        n = len(fv)
                        for k in range(n):
                            B = fv[k]
                            if B in tv:
                                continue
                            A = fv[(k + n - 1) % n]; C = fv[(k + 1) % n]
                            for j, t in enumerate(tl):
                                if A in t and C in t:
                                    if (t[0]==A or t[0]==C) and (t[1]==A or t[1]==C):
                                        tl[j] = [t[0], B, t[2]]; tl.append([B, t[1], t[2]])
                                    elif (t[1]==A or t[1]==C) and (t[2]==A or t[2]==C):
                                        tl[j] = [t[0], t[1], B]; tl.append([t[0], B, t[2]])
                                    else:
                                        tl[j] = [t[0], t[1], B]; tl.append([B, t[1], t[2]])
                                    chg = True; break
                            if chg:
                                break
                    def _zero_t(t):
                        u0, v0 = vk2d_t.get(t[0], (0.0, 0.0)); u1, v1 = vk2d_t.get(t[1], (0.0, 0.0)); u2, v2 = vk2d_t.get(t[2], (0.0, 0.0))
                        return (round(u1*sc)-round(u0*sc))*(round(v2*sc)-round(v0*sc)) - (round(v1*sc)-round(v0*sc))*(round(u2*sc)-round(u0*sc)) == 0
                    mesh.triangulation[fk_top] = [t for t in tl if not _zero_t(t)]
        def side_faces(bot_off, bot_n, top_off, top_n, bpts, tpts):
            def edsq(pts, i):
                j = (i + 1) % len(pts)
                dx = pts[j][0] - pts[i][0]; dy = pts[j][1] - pts[i][1]; dz = pts[j][2] - pts[i][2]
                return dx*dx + dy*dy + dz*dz
            ia = max(range(bot_n), key=lambda i: edsq(bpts, i))
            ib = 0
            if bot_n == top_n:
                def align_cost(cand):
                    total = 0.0
                    for k in range(bot_n):
                        xb, yb = proj(bpts[(ia+k)%bot_n])
                        xt, yt = proj(tpts[(cand+k)%top_n])
                        total += (xt-xb)**2 + (yt-yb)**2
                    return total
                ib = min(range(top_n), key=align_cost)
            if bot_n == top_n:
                for k in range(bot_n):
                    cb = bot_off + (ia + k) % bot_n; ct = top_off + (ib + k) % top_n
                    nb = bot_off + (ia + k + 1) % bot_n; nt = top_off + (ib + k + 1) % top_n
                    mesh.add_face([bvk[cb], bvk[nb], tvk[nt], tvk[ct]])
                return
            b_arcs = [0.0] * (bot_n + 1)
            for k in range(bot_n):
                i = (ia + k) % bot_n; j = (ia + k + 1) % bot_n
                dx = bpts[j][0] - bpts[i][0]; dy = bpts[j][1] - bpts[i][1]; dz = bpts[j][2] - bpts[i][2]
                b_arcs[k + 1] = b_arcs[k] + math.sqrt(dx*dx + dy*dy + dz*dz)
            t_arcs = [0.0] * (top_n + 1)
            for k in range(top_n):
                i = (ib + k) % top_n; j = (ib + k + 1) % top_n
                dx = tpts[j][0] - tpts[i][0]; dy = tpts[j][1] - tpts[i][1]; dz = tpts[j][2] - tpts[i][2]
                t_arcs[k + 1] = t_arcs[k] + math.sqrt(dx*dx + dy*dy + dz*dz)
            inv_b = 1.0 / b_arcs[bot_n] if b_arcs[bot_n] > 0 else 1.0
            inv_t = 1.0 / t_arcs[top_n] if t_arcs[top_n] > 0 else 1.0
            bi = ti = 0
            while bi < bot_n or ti < top_n:
                cb = bot_off + (ia + bi) % bot_n; ct = top_off + (ib + ti) % top_n
                nb = bot_off + (ia + bi + 1) % bot_n; nt = top_off + (ib + ti + 1) % top_n
                if bi >= bot_n:
                    mesh.add_face([bvk[cb], tvk[ct], tvk[nt]]); ti += 1
                elif ti >= top_n:
                    mesh.add_face([bvk[cb], bvk[nb], tvk[ct]]); bi += 1
                else:
                    bp = b_arcs[bi + 1] * inv_b; tp = t_arcs[ti + 1] * inv_t
                    if abs(bp - tp) < 1e-9:
                        mesh.add_face([bvk[cb], bvk[nb], tvk[nt], tvk[ct]]); bi += 1; ti += 1
                    elif bp < tp:
                        mesh.add_face([bvk[cb], bvk[nb], tvk[ct]]); bi += 1
                    else:
                        mesh.add_face([bvk[cb], tvk[ct], tvk[nt]]); ti += 1
        for bot_off, bot_n, top_off, top_n in poly_infos:
            side_faces(bot_off, bot_n, top_off, top_n, all_bot[bot_off:bot_off+bot_n], all_top[top_off:top_off+top_n])
        return mesh

    @staticmethod
    def loft_panels(
        top_polygons: List[List[Point]],
        bot_polygons: List[List[Point]],
        merge_precision: float,
        edge_gap: float = 0.0,
        edge_match_threshold: float = 2.0,
        add_caps: bool = True,
        skip_triangles: bool = False) -> List["LoftPanel"]:
        top_mesh = Mesh.from_polylines(top_polygons, merge_precision)
        bot_mesh = Mesh.from_polylines(bot_polygons, merge_precision)
        tfks = list(top_mesh.face.keys())
        bfks = list(bot_mesh.face.keys())
        import numpy as np
        top_cents = np.zeros((len(tfks), 3))
        for i, fk in enumerate(tfks):
            vkeys = top_mesh.face_vertices(fk)
            for vk in vkeys:
                p = top_mesh.vertex_point(vk)
                top_cents[i, 0] += p[0]; top_cents[i, 1] += p[1]; top_cents[i, 2] += p[2]
            top_cents[i] /= len(vkeys)
        bot_cents = np.zeros((len(bfks), 3))
        for i, fk in enumerate(bfks):
            vkeys = bot_mesh.face_vertices(fk)
            for vk in vkeys:
                p = bot_mesh.vertex_point(vk)
                bot_cents[i, 0] += p[0]; bot_cents[i, 1] += p[1]; bot_cents[i, 2] += p[2]
            bot_cents[i] /= len(vkeys)
        diff = top_cents[:, np.newaxis, :] - bot_cents[np.newaxis, :, :]
        dist_mat = np.sqrt((diff * diff).sum(axis=2))
        flat_order = np.argsort(dist_mat, axis=None)
        top_used = [False] * len(tfks)
        bot_used = [False] * len(bfks)
        face_match = []
        for flat_idx in flat_order:
            ti, bi = divmod(int(flat_idx), len(bfks))
            if top_used[ti] or bot_used[bi]:
                continue
            face_match.append((tfks[ti], bfks[bi]))
            top_used[ti] = True
            bot_used[bi] = True
        face_match.sort()
        panels = []
        for tfk, bfk in face_match:
            panel = LoftPanel()
            top_vkeys = list(top_mesh.face_vertices(tfk))
            bot_vkeys = list(bot_mesh.face_vertices(bfk))
            top_pts = [top_mesh.vertex_point(vk) for vk in top_vkeys]
            bot_pts = [bot_mesh.vertex_point(vk) for vk in bot_vkeys]
            top_pts, top_vkeys = _lp_merge_collinear(top_pts, top_vkeys)
            bot_pts, bot_vkeys = _lp_merge_collinear(bot_pts, bot_vkeys)
            max_te = 0.0
            sz = len(top_pts)
            for i in range(sz): max_te = max(max_te, top_pts[i].distance(top_pts[(i+1)%sz]))
            stol = max_te * 0.001
            tp, tk = [], []
            for i in range(len(top_pts)):
                if not tp or tp[-1].distance(top_pts[i]) > stol:
                    tp.append(top_pts[i]); tk.append(top_vkeys[i])
            while len(tp) >= 3 and tp[-1].distance(tp[0]) <= stol: tp.pop(); tk.pop()
            if len(tp) >= 3: top_pts, top_vkeys = tp, tk
            n = len(top_pts); m = len(bot_pts)
            tcx = tcy = tcz = bcx = bcy = bcz = 0.0
            for p in top_pts: tcx += p[0]; tcy += p[1]; tcz += p[2]
            for p in bot_pts: bcx += p[0]; bcy += p[1]; bcz += p[2]
            tcx /= n; tcy /= n; tcz /= n
            bcx /= m; bcy /= m; bcz /= m
            ax = tcx-bcx; ay = tcy-bcy; az = tcz-bcz
            alen = math.sqrt(ax*ax+ay*ay+az*az)
            if alen > 1e-12: ax /= alen; ay /= alen; az /= alen
            tnx, tny, tnz = _lp_newell_normal(top_pts)
            if tnx*ax+tny*ay+tnz*az < 0: top_pts.reverse(); top_vkeys.reverse()
            bnx, bny, bnz = _lp_newell_normal(bot_pts)
            if bnx*ax+bny*ay+bnz*az > 0: bot_pts.reverse(); bot_vkeys.reverse()
            for i in range(n):
                lk = panel.mesh.add_vertex(top_pts[i]); panel.orig_top_to_local[top_vkeys[i]] = lk; panel.top_vertices.append(lk)
            for j in range(m):
                lk = panel.mesh.add_vertex(bot_pts[j]); panel.orig_bot_to_local[bot_vkeys[j]] = lk; panel.bot_vertices.append(lk)
            if add_caps:
                top_cap = [panel.orig_top_to_local[vk] for vk in top_vkeys]
                fk = panel.mesh.add_face(top_cap)
                if fk is not None: panel.top_face_key = fk
                if fk is not None and len(top_cap) >= 3:
                    nx, ny, nz = _lp_newell_normal(top_pts)
                    mag = math.sqrt(nx*nx+ny*ny+nz*nz)
                    if mag > 1e-12:
                        nx /= mag; ny /= mag; nz /= mag
                        ux, uy, uz = (0.0, 1.0, 0.0) if abs(nx) > 0.9 else (1.0, 0.0, 0.0)
                        dot = ux*nx+uy*ny+uz*nz
                        ux -= dot*nx; uy -= dot*ny; uz -= dot*nz
                        um = math.sqrt(ux*ux+uy*uy+uz*uz); ux /= um; uy /= um; uz /= um
                        vx = ny*uz-nz*uy; vy = nz*ux-nx*uz; vz = nx*uy-ny*ux
                        bpts = [Point(p[0]*ux+p[1]*uy+p[2]*uz, p[0]*vx+p[1]*vy+p[2]*vz, 0.0) for p in top_pts]
                        tris = _cdt_triangulate(bpts, [])
                        if tris:
                            panel.mesh.triangulation[fk] = [[top_cap[t[0]], top_cap[t[1]], top_cap[t[2]]] for t in tris]
            top_arr = np.array([[p[0], p[1], p[2]] for p in top_pts])
            bot_arr = np.array([[p[0], p[1], p[2]] for p in bot_pts])
            top_mids_arr = (top_arr + np.roll(top_arr, -1, axis=0)) * 0.5
            bot_mids_arr = (bot_arr + np.roll(bot_arr, -1, axis=0)) * 0.5
            diff = bot_mids_arr[:, np.newaxis, :] - top_mids_arr[np.newaxis, :, :]
            mid_dist = np.sqrt((diff * diff).sum(axis=2))
            bot_to_top = mid_dist.argmin(axis=1).tolist()
            bot_dist = mid_dist[np.arange(m), bot_to_top].tolist()
            top_to_bot = mid_dist.argmin(axis=0).tolist()
            avg = sum(bot_dist) / m
            threshold = avg * edge_match_threshold
            top_used_edge = [False] * n
            for j in range(m):
                b0 = panel.orig_bot_to_local[bot_vkeys[j]]
                b1 = panel.orig_bot_to_local[bot_vkeys[(j+1)%m]]
                ti = bot_to_top[j]
                if ti >= 0 and bot_dist[j] <= threshold and top_to_bot[ti] == j:
                    t0 = panel.orig_top_to_local[top_vkeys[ti]]
                    t1 = panel.orig_top_to_local[top_vkeys[(ti+1)%n]]
                    if edge_gap > 0.0:
                        pb0 = panel.mesh.vertex_point(b0); pb1 = panel.mesh.vertex_point(b1)
                        pt0 = panel.mesh.vertex_point(t0); pt1 = panel.mesh.vertex_point(t1)
                        cx = (pb0[0]+pb1[0]+pt0[0]+pt1[0])*0.25
                        cy = (pb0[1]+pb1[1]+pt0[1]+pt1[1])*0.25
                        cz = (pb0[2]+pb1[2]+pt0[2]+pt1[2])*0.25
                        nb0 = panel.mesh.add_vertex(_lp_offset_toward(pb0, cx, cy, cz, edge_gap))
                        nb1 = panel.mesh.add_vertex(_lp_offset_toward(pb1, cx, cy, cz, edge_gap))
                        fk = panel.mesh.add_face([nb0, t1, t0, nb1])
                    else:
                        fk = panel.mesh.add_face([b0, t1, t0, b1])
                    if fk is not None:
                        w = LoftWallFace(); w.face_key = fk; w.is_quad = True
                        w.top_v0 = top_vkeys[ti]; w.top_v1 = top_vkeys[(ti+1)%n]
                        w.bot_v0 = bot_vkeys[(j+1)%m]; w.bot_v1 = bot_vkeys[j]
                        panel.wall_faces.append(w)
                    top_used_edge[ti] = True
                elif not skip_triangles:
                    diff_j = top_arr - bot_mids_arr[j]
                    best_tv = int((diff_j * diff_j).sum(axis=1).argmin())
                    tv = panel.orig_top_to_local[top_vkeys[best_tv]]
                    fk = panel.mesh.add_face([b0, tv, b1])
                    if fk is not None:
                        w = LoftWallFace(); w.face_key = fk; w.is_quad = False; panel.wall_faces.append(w)
            if not skip_triangles:
                for i in range(n):
                    if top_used_edge[i]: continue
                    t0 = panel.orig_top_to_local[top_vkeys[i]]
                    t1 = panel.orig_top_to_local[top_vkeys[(i+1)%n]]
                    diff_i = bot_arr - top_mids_arr[i]
                    best_bv = int((diff_i * diff_i).sum(axis=1).argmin())
                    bv = panel.orig_bot_to_local[bot_vkeys[best_bv]]
                    fk = panel.mesh.add_face([t1, t0, bv])
                    if fk is not None:
                        w = LoftWallFace(); w.face_key = fk; w.is_quad = False; panel.wall_faces.append(w)
            if add_caps:
                bot_cap = [panel.orig_bot_to_local[bot_vkeys[j]] for j in range(m)]
                bot_cap_fk = panel.mesh.add_face(bot_cap)
                if bot_cap_fk is not None: panel.bot_face_key = bot_cap_fk
                if bot_cap_fk is not None and len(bot_cap) >= 3:
                    bcnx, bcny, bcnz = _lp_newell_normal(bot_pts)
                    bcmag = math.sqrt(bcnx*bcnx+bcny*bcny+bcnz*bcnz)
                    if bcmag > 1e-12:
                        bcnx /= bcmag; bcny /= bcmag; bcnz /= bcmag
                        bcux, bcuy, bcuz = (0.0, 1.0, 0.0) if abs(bcnx) > 0.9 else (1.0, 0.0, 0.0)
                        bcdot = bcux*bcnx+bcuy*bcny+bcuz*bcnz
                        bcux -= bcdot*bcnx; bcuy -= bcdot*bcny; bcuz -= bcdot*bcnz
                        bcum = math.sqrt(bcux*bcux+bcuy*bcuy+bcuz*bcuz)
                        bcux /= bcum; bcuy /= bcum; bcuz /= bcum
                        bcvx = bcny*bcuz-bcnz*bcuy; bcvy = bcnz*bcux-bcnx*bcuz; bcvz = bcnx*bcuy-bcny*bcux
                        bpts2 = [Point(p[0]*bcux+p[1]*bcuy+p[2]*bcuz, p[0]*bcvx+p[1]*bcvy+p[2]*bcvz, 0.0) for p in bot_pts]
                        btris = _cdt_triangulate(bpts2, [])
                        if btris:
                            panel.mesh.triangulation[bot_cap_fk] = [[bot_cap[t[0]], bot_cap[t[1]], bot_cap[t[2]]] for t in btris]
            panels.append(panel)
        for panel in panels:
            fkey_to_idx = {}
            for fi, fk in enumerate(panel.mesh.face):
                fkey_to_idx[fk] = fi
            for w in panel.wall_faces:
                w.face_index = fkey_to_idx[w.face_key]
                panel.face_roles[w.face_key] = "QuadWall" if w.is_quad else "TriWall"
            if panel.top_face_key:
                panel.face_roles[panel.top_face_key] = "TopCap"
            if panel.bot_face_key:
                panel.face_roles[panel.bot_face_key] = "BotCap"
        edge_to_wall = {}
        for pi, panel in enumerate(panels):
            for wi, w in enumerate(panel.wall_faces):
                if not w.is_quad:
                    continue
                edge_to_wall[(w.top_v0, w.top_v1)] = (pi, wi)
        adjacency = []
        for pi, panel in enumerate(panels):
            for wi, w in enumerate(panel.wall_faces):
                if not w.is_quad:
                    continue
                key = (w.top_v1, w.top_v0)
                if key in edge_to_wall and edge_to_wall[key][0] > pi:
                    pj, wj = edge_to_wall[key]
                    adjacency.append(LoftAdjPair(pi, wi, pj, wj))
        top_ordered = Mesh()
        bot_ordered = Mesh()
        for i, panel in enumerate(panels):
            tvks = [top_ordered.add_vertex(panel.mesh.vertex_point(lk)) for lk in panel.top_vertices]
            bvks = [bot_ordered.add_vertex(panel.mesh.vertex_point(lk)) for lk in panel.bot_vertices]
            top_ordered.add_face(tvks, i)
            bot_ordered.add_face(bvks, i)
        return panels, adjacency, top_ordered, bot_ordered

    @staticmethod
    def from_polygon_with_holes_many(inputs: List, sort_by_bbox: bool = False, parallel: bool = False) -> List["Mesh"]:
        if parallel and len(inputs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as ex:
                return list(ex.map(lambda x: Mesh.from_polygon_with_holes(x, sort_by_bbox), inputs))
        return [Mesh.from_polygon_with_holes(x, sort_by_bbox) for x in inputs]

    @staticmethod
    def loft_many(pairs: List, cap: bool = True, parallel: bool = False, fix_collinear: bool = True) -> List["Mesh"]:
        if parallel and len(pairs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as ex:
                return list(ex.map(lambda p: Mesh.loft(p[0], p[1], cap, fix_collinear), pairs))
        return [Mesh.loft(p[0], p[1], cap, fix_collinear) for p in pairs]

    ###########################################################################################
    # Boolean Queries
    ###########################################################################################

    def is_empty(self) -> bool:
        """Check if the mesh is empty."""
        return len(self.vertex) == 0

    def is_valid(self) -> bool:
        if not self.vertex or not self.face:
            return False
        for fkey, vkeys in self.face.items():
            if len(vkeys) < 3:
                return False
            for vk in vkeys:
                if vk not in self.vertex:
                    return False
        return True

    def is_closed(self) -> bool:
        hole_edges = set()
        for fk, rings in self.face_holes.items():
            for ring in rings:
                n = len(ring)
                for i in range(n):
                    a = ring[i]; b = ring[(i + 1) % n]
                    hole_edges.add((a, b))
                    hole_edges.add((b, a))
        for u, nbrs in self.halfedge.items():
            for v, fkey in nbrs.items():
                if fkey is None and (u, v) not in hole_edges:
                    return False
        return bool(self.halfedge)

    def is_vertex_on_boundary(self, vertex_key: int) -> bool:
        """Check if a vertex is on the boundary."""
        if vertex_key not in self.halfedge:
            return False

        for v, face_opt in self.halfedge[vertex_key].items():
            if face_opt is None:
                return True

        for u, neighbors in self.halfedge.items():
            if vertex_key in neighbors and neighbors[vertex_key] is None:
                return True

        return False

    def is_edge_on_boundary(self, u: int, v: int) -> bool:
        """Check if an edge is on the boundary."""
        return self.halfedge.get(u, {}).get(v) is None or self.halfedge.get(v, {}).get(u) is None

    def is_face_on_boundary(self, face_key: int) -> bool:
        """Check if a face is on the boundary."""
        fe = self.face_edges(face_key)
        if fe is None:
            return False
        return any(self.is_edge_on_boundary(u, v) for u, v in fe)

    ###########################################################################################
    # Basic Queries
    ###########################################################################################

    def number_of_vertices(self) -> int:
        """Get the number of vertices."""
        return len(self.vertex)

    def number_of_faces(self) -> int:
        """Get the number of faces."""
        return len(self.face)

    def vertices(self) -> List[int]:
        return sorted(self.vertex.keys())

    def faces(self) -> List[int]:
        return sorted(self.face.keys())

    def number_of_edges(self) -> int:
        """Get the number of edges."""
        seen = set()
        for u, neighbors in self.halfedge.items():
            for v in neighbors:
                seen.add((min(u, v), max(u, v)))
        return len(seen)

    def edges(self) -> List[Tuple[int, int]]:
        seen = set()
        for u, neighbors in self.halfedge.items():
            for v in neighbors:
                seen.add((min(u, v), max(u, v)))
        return sorted(seen)

    def naked_edges(self, boundary: bool = True) -> List[Tuple[int, int]]:
        seen = set()
        for u, neighbors in self.halfedge.items():
            for v in neighbors:
                seen.add((min(u, v), max(u, v)))
        return [e for e in sorted(seen) if self.is_edge_on_boundary(e[0], e[1]) == boundary]

    def naked_vertices(self, boundary: bool = True) -> List[int]:
        result = []
        for vk in sorted(self.vertex.keys()):
            if self.is_vertex_on_boundary(vk) == boundary:
                result.append(vk)
        return result

    def naked_faces(self, boundary: bool = True) -> List[int]:
        result = []
        for fk in sorted(self.face.keys()):
            if self.is_face_on_boundary(fk) == boundary:
                result.append(fk)
        return result

    def euler(self) -> int:
        """Calculate Euler characteristic (V - E + F)."""
        return (
            self.number_of_vertices() - self.number_of_edges() + self.number_of_faces()
        )

    def clear(self) -> None:
        """Clear all mesh data."""
        self.halfedge.clear()
        self.vertex.clear()
        self.face.clear()
        self.facedata.clear()
        self.edgedata.clear()
        self.triangulation.clear()
        self.face_holes.clear()
        self._max_vertex = 0
        self._max_face = 0
        self._pointcolors.clear()
        self._facecolors.clear()
        self._linecolors.clear()
        self._widths.clear()
        self._objectcolor = None
        self.color_mode = ColorMode.OBJECTCOLOR
        self._triangle_bvh_built = False

    def set_pointcolors(self, colors: List[Color]) -> None:
        self._pointcolors = list(colors)
        self.color_mode = ColorMode.POINTCOLORS

    def set_facecolors(self, colors: List[Color]) -> None:
        self._facecolors = list(colors)
        self.color_mode = ColorMode.FACECOLORS

    def set_linecolors(self, colors: List[Color], widths: Optional[List[float]] = None) -> None:
        self._linecolors = list(colors)
        if widths is not None:
            self._widths = list(widths)

    def set_objectcolor(self, color: Color) -> None:
        self._objectcolor = color

    @property
    def pointcolors(self) -> list: return self._pointcolors
    @property
    def facecolors(self) -> list: return self._facecolors
    @property
    def linecolors(self) -> list: return self._linecolors
    def get_pointcolors(self) -> List[Color]: return self._pointcolors
    def get_facecolors(self) -> List[Color]: return self._facecolors
    def get_linecolors(self) -> List[Color]: return self._linecolors
    @property
    def widths(self) -> list: return self._widths
    @property
    def objectcolor(self) -> Optional[Color]:
        if getattr(self, '_objectcolor', None) is None:
            self._objectcolor = Color.white()
        return self._objectcolor

    @objectcolor.setter
    def objectcolor(self, value: Optional[Color]) -> None:
        self._objectcolor = value

    def clear_pointcolors(self) -> None:
        self._pointcolors.clear()
        if self.color_mode == ColorMode.POINTCOLORS:
            self.color_mode = ColorMode.OBJECTCOLOR

    def clear_facecolors(self) -> None:
        self._facecolors.clear()
        if self.color_mode == ColorMode.FACECOLORS:
            self.color_mode = ColorMode.OBJECTCOLOR

    def clear_linecolors(self) -> None:
        self._linecolors.clear()
        self._widths.clear()

    def unify_winding(self) -> bool:
        """Unify face winding by BFS over face adjacency; returns True if any face was flipped."""
        if len(self.face) < 2:
            return False

        edge_faces = {}
        for fkey, verts in self.face.items():
            n = len(verts)
            for i in range(n):
                u = verts[i]
                v = verts[(i + 1) % n]
                edge = (min(u, v), max(u, v))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append((fkey, u, v))

        visited = set()
        flipped = set()
        for seed in self.face:
            if seed in visited:
                continue
            visited.add(seed)
            queue = [seed]
            while queue:
                f = queue.pop()
                is_flipped = f in flipped
                verts = self.face[f]
                n = len(verts)
                for i in range(n):
                    u_orig = verts[i]
                    v_orig = verts[(i + 1) % n]
                    eff_u = v_orig if is_flipped else u_orig
                    eff_v = u_orig if is_flipped else v_orig
                    edge = (min(u_orig, v_orig), max(u_orig, v_orig))
                    for adj_key, adj_u, adj_v in edge_faces.get(edge, []):
                        if adj_key == f or adj_key in visited:
                            continue
                        if not (adj_u == eff_v and adj_v == eff_u):
                            flipped.add(adj_key)
                        visited.add(adj_key)
                        queue.append(adj_key)

        if not flipped:
            return False

        for fkey in flipped:
            self.face[fkey].reverse()

        for u in self.halfedge:
            self.halfedge[u].clear()
        for fkey, verts in self.face.items():
            n = len(verts)
            for i in range(n):
                u = verts[i]
                v = verts[(i + 1) % n]
                self.halfedge[u][v] = fkey
                if u not in self.halfedge[v]:
                    self.halfedge[v][u] = None

        self.orient_outward()
        return True

    def orient_outward(self) -> bool:
        if not self.face or self.naked_edges(True):
            return False
        vol = 0.0
        for fk, verts in self.face.items():
            n = len(verts)
            p0 = self.vertex_point(verts[0])
            for i in range(1, n - 1):
                p1 = self.vertex_point(verts[i])
                p2 = self.vertex_point(verts[i + 1])
                vol += (p0[0] * (p1[1] * p2[2] - p1[2] * p2[1])
                      + p0[1] * (p1[2] * p2[0] - p1[0] * p2[2])
                      + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0]))
        if vol >= 0.0:
            return False
        for fk in self.face:
            self.face[fk] = self.face[fk][::-1]
        for u in self.halfedge:
            self.halfedge[u].clear()
        for fk, verts in self.face.items():
            n = len(verts)
            for i in range(n):
                u = verts[i]
                v = verts[(i + 1) % n]
                self.halfedge[u][v] = fk
                if u not in self.halfedge[v]:
                    self.halfedge[v][u] = None
        return True

    def unweld(self) -> "Mesh":
        m = Mesh()
        for fkey in sorted(self.face):
            new_vkeys = []
            for vk in self.face[fkey]:
                pt = self.vertex[vk]
                new_vkeys.append(m.add_vertex(Point(pt[0], pt[1], pt[2])))
            m.add_face(new_vkeys)
        return m

    def weld(self, tolerance: float = 0.001) -> "Mesh":
        if not self.vertex:
            return Mesh()

        vkeys = sorted(self.vertex.keys())
        positions = [Point(self.vertex[k][0], self.vertex[k][1], self.vertex[k][2]) for k in vkeys]
        n = len(vkeys)

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        if tolerance > 0.0:
            boxes = [OBB.from_point(p, tolerance) for p in positions]
            ws = SpatialBVH.compute_world_size(boxes)
            bvh = SpatialBVH.from_boxes(boxes, ws)
            pairs, _, _ = bvh.check_all_collisions(boxes)
            for i, j in pairs:
                if positions[i].distance(positions[j]) <= tolerance:
                    ri = find(i)
                    rj = find(j)
                    if ri != rj:
                        parent[ri] = rj

        root_to_rep = {}
        for i in range(n):
            root = find(i)
            if root not in root_to_rep or vkeys[i] < root_to_rep[root]:
                root_to_rep[root] = vkeys[i]
        vkey_to_rep = {vkeys[i]: root_to_rep[find(i)] for i in range(n)}

        m = Mesh()
        added = set()
        for i in range(n):
            rep = vkey_to_rep[vkeys[i]]
            if rep not in added:
                added.add(rep)
                pt = self.vertex[rep]
                m.add_vertex(Point(pt[0], pt[1], pt[2]), rep)
        for fk in sorted(self.face):
            new_vkeys = [vkey_to_rep[vk] for vk in self.face[fk]]
            m.add_face(new_vkeys, fk)
        return m

    ###########################################################################################
    # Vertex and Face Operations
    ###########################################################################################

    def add_vertex(self, position: Point, vkey: Optional[int] = None) -> int:
        """Add a vertex to the mesh.

        Parameters
        ----------
        position : Point
            The position of the vertex.
        vkey : int, optional
            Optional vertex key. If None, auto-generated.

        Returns
        -------
        int
            The vertex key.
        """
        if vkey is None:
            vertex_key = self._max_vertex
            self._max_vertex += 1
        else:
            vertex_key = vkey
            if vertex_key >= self._max_vertex:
                self._max_vertex = vertex_key + 1

        self.vertex[vertex_key] = VertexData(position)
        self.halfedge[vertex_key] = {}
        self._pointcolors.append(Color.white())
        self._triangle_bvh_built = False

        return vertex_key

    def add_face(
        self, vertices: List[int], fkey: Optional[int] = None
    ) -> Optional[int]:
        """Add a face to the mesh.

        Parameters
        ----------
        vertices : list[int]
            The vertex keys forming the face.
        fkey : int, optional
            Optional face key. If None, auto-generated.

        Returns
        -------
        int or None
            The face key, or None if the face is invalid.
        """
        if len(vertices) < 3:
            return None

        if not all(v in self.vertex for v in vertices):
            return None

        if len(set(vertices)) != len(vertices):
            return None

        if fkey is None:
            face_key = self._max_face
            self._max_face += 1
        else:
            face_key = fkey
            if face_key >= self._max_face:
                self._max_face = face_key + 1

        self.face[face_key] = vertices.copy()
        self.triangulation.pop(face_key, None)
        self._facecolors.append(Color.white())
        self._triangle_bvh_built = False

        for i in range(len(vertices)):
            u = vertices[i]
            v = vertices[(i + 1) % len(vertices)]

            if u not in self.halfedge:
                self.halfedge[u] = {}
            if v not in self.halfedge:
                self.halfedge[v] = {}

            is_new_edge = u not in self.halfedge[v]

            self.halfedge[u][v] = face_key

            if is_new_edge:
                self.halfedge[v][u] = None
                self._linecolors.append(Color.black())
                self._widths.append(1.0)

        return face_key

    def remove_face(self, fkey: int) -> None:
        if fkey not in self.face:
            return
        vertices = self.face[fkey]
        n = len(vertices)
        for i in range(n):
            u = vertices[i]
            v = vertices[(i + 1) % n]
            if v in self.halfedge.get(u, {}):
                self.halfedge[u][v] = None
                if self.halfedge.get(v, {}).get(u) is None:
                    del self.halfedge[u][v]
                    del self.halfedge[v][u]
        del self.face[fkey]
        self.triangulation.pop(fkey, None)
        self.facedata.pop(fkey, None)
        self.face_holes.pop(fkey, None)
        n_edges = self.number_of_edges()
        if len(self._linecolors) > n_edges:
            self._linecolors = self._linecolors[:n_edges]
            self._widths = self._widths[:n_edges]
        n_faces = len(self.face)
        if len(self._facecolors) > n_faces:
            self._facecolors = self._facecolors[:n_faces]
        self._triangle_bvh_built = False

    def remove_vertex(self, vkey: int) -> None:
        if vkey not in self.vertex:
            return
        faces_to_remove = [fk for fk, verts in self.face.items() if vkey in verts]
        for fk in faces_to_remove:
            self.remove_face(fk)
        if vkey in self.halfedge:
            for v in list(self.halfedge[vkey].keys()):
                if vkey in self.halfedge.get(v, {}):
                    del self.halfedge[v][vkey]
            del self.halfedge[vkey]
        self.edgedata = {k: w for k, w in self.edgedata.items() if vkey not in k}
        del self.vertex[vkey]
        n_vertices = len(self.vertex)
        if len(self._pointcolors) > n_vertices:
            self._pointcolors = self._pointcolors[:n_vertices]
        self._triangle_bvh_built = False

    def remove_edge(self, u: int, v: int) -> None:
        faces_to_remove = set()
        f0 = self.halfedge.get(u, {}).get(v)
        if f0 is not None:
            faces_to_remove.add(f0)
        f1 = self.halfedge.get(v, {}).get(u)
        if f1 is not None:
            faces_to_remove.add(f1)
        for fk in faces_to_remove:
            self.remove_face(fk)
        if v in self.halfedge.get(u, {}):
            del self.halfedge[u][v]
        if u in self.halfedge.get(v, {}):
            del self.halfedge[v][u]
        self.edgedata.pop((u, v), None)
        self.edgedata.pop((v, u), None)
        n_edges = self.number_of_edges()
        if len(self._linecolors) > n_edges:
            self._linecolors = self._linecolors[:n_edges]
            self._widths = self._widths[:n_edges]
        self._triangle_bvh_built = False

    def flip_face(self, fkey: int) -> None:
        if fkey not in self.face:
            return
        fv = self.face[fkey][:]
        self.remove_face(fkey)
        self.add_face(fv[::-1], fkey)

    def flip(self) -> None:
        for fkey in self.face:
            self.face[fkey].reverse()
        for u in self.halfedge:
            self.halfedge[u].clear()
        for fkey, verts in self.face.items():
            n = len(verts)
            for i in range(n):
                u = verts[i]
                v = verts[(i + 1) % n]
                self.halfedge[u][v] = fkey
                if u not in self.halfedge[v]:
                    self.halfedge[v][u] = None

    ###########################################################################################
    # Connectivity Queries
    ###########################################################################################

    def edge_edges(self, u: int, v: int) -> Optional[List[Tuple[int, int]]]:
        """Get all edges sharing a vertex with (u,v), excluding (u,v) and (v,u)."""
        uv = v in self.halfedge.get(u, {})
        vu = u in self.halfedge.get(v, {})
        if not uv and not vu:
            return None
        edges = []
        for w in self.halfedge.get(u, {}):
            if w != v:
                edges.append((u, w))
        for w in self.halfedge.get(v, {}):
            if w != u:
                edges.append((v, w))
        return edges

    def edge_faces(self, u: int, v: int) -> Optional[List[int]]:
        """Get the faces adjacent to an edge."""
        f0 = self.halfedge.get(u, {}).get(v)
        f1 = self.halfedge.get(v, {}).get(u)
        if f0 is None and f1 is None:
            return None
        return [f for f in (f0, f1) if f is not None]

    def edge_line(self, u: int, v: int) -> Optional["Line"]:
        """Get the edge as a Line."""
        uv = v in self.halfedge.get(u, {})
        vu = u in self.halfedge.get(v, {})
        if not uv and not vu:
            return None
        from .line import Line
        return Line.from_points(self.vertex_point(u), self.vertex_point(v))

    def face_edges(self, face_key: int) -> Optional[List[Tuple[int, int]]]:
        """Get edges of a face as (vi, vi+1) pairs."""
        verts = self.face.get(face_key)
        if verts is None:
            return None
        n = len(verts)
        return [(verts[i], verts[(i + 1) % n]) for i in range(n)]

    def face_faces(self, face_key: int) -> Optional[List[int]]:
        """Get faces adjacent to a face (sharing an edge)."""
        fe = self.face_edges(face_key)
        if fe is None:
            return None
        neighbors = []
        for u, v in fe:
            f = self.halfedge.get(v, {}).get(u)
            if f is not None:
                neighbors.append(f)
        return neighbors

    def face_points(self, face_key: int) -> Optional[List[Point]]:
        """Get the point positions of a face's vertices."""
        fv = self.face_vertices(face_key)
        if fv is None:
            return None
        return [self.vertex_point(vk) for vk in fv]

    def face_polyline(self, face_key: int) -> Optional["Polyline"]:
        """Get the face as a Polyline."""
        pts = self.face_points(face_key)
        if pts is None:
            return None
        from .polyline import Polyline
        return Polyline(pts)

    def face_vertices(self, face_key: int) -> Optional[List[int]]:
        """Get the vertices of a face."""
        return self.face.get(face_key)

    def vertex_edges(self, vertex_key: int) -> Optional[List[Tuple[int, int]]]:
        """Get edges incident to a vertex as (vertex_key, neighbor) pairs."""
        if vertex_key not in self.halfedge:
            return None
        return [(vertex_key, u) for u in self.halfedge[vertex_key]]

    def vertex_faces(self, vertex_key: int) -> Optional[List[int]]:
        """Get the faces incident to a vertex."""
        if vertex_key not in self.halfedge:
            return None
        return [f for f in self.halfedge[vertex_key].values() if f is not None]

    def vertex_point(self, vertex_key: int) -> Optional[Point]:
        """Get the position of a vertex."""
        if vertex_key not in self.vertex:
            return None
        return self.vertex[vertex_key].position()

    def vertex_vertices(self, vertex_key: int) -> Optional[List[int]]:
        """Get the neighboring vertices of a vertex."""
        if vertex_key not in self.halfedge:
            return None
        return list(self.halfedge[vertex_key].keys())

    def vertex_neighbors(self, vertex_key: int, ordered: bool = False) -> Optional[List[int]]:
        """Alias of vertex_vertices. With ordered=True returns neighbors in face-cycle order
        around the vertex (boundary vertex starts/ends at boundary halfedges)."""
        if vertex_key not in self.halfedge:
            return None
        nbrs = list(self.halfedge[vertex_key].keys())
        if not ordered or len(nbrs) <= 1:
            return nbrs
        start = nbrs[0]
        for n in nbrs:
            if self.halfedge[vertex_key].get(n) is None:
                start = n
                break
        fkey = self.halfedge.get(start, {}).get(vertex_key)
        out = [start]
        guard = 0
        while fkey is not None and guard < 10000:
            guard += 1
            verts = self.face.get(fkey)
            if verts is None:
                break
            i = verts.index(vertex_key)
            nbr = verts[(i + 1) % len(verts)]
            if nbr == start:
                break
            out.append(nbr)
            fkey = self.halfedge.get(nbr, {}).get(vertex_key)
        return out

    ###########################################################################################
    # Boundary
    ###########################################################################################

    def vertices_on_boundary(self) -> List[int]:
        """Vertices touching at least one boundary halfedge."""
        out = []
        for v in self.vertex:
            if self.is_vertex_on_boundary(v):
                out.append(v)
        return out

    def edges_on_boundary(self) -> List[Tuple[int, int]]:
        """Edges with no face on one side, oriented as boundary halfedges (face is None on (u,v))."""
        out = []
        for u, nbrs in self.halfedge.items():
            for v, f in nbrs.items():
                if f is None:
                    out.append((u, v))
        return out

    def faces_on_boundary(self) -> List[int]:
        """Faces with at least one edge on the boundary."""
        out = []
        for fkey in self.face:
            if self.is_face_on_boundary(fkey):
                out.append(fkey)
        return out

    ###########################################################################################
    # Halfedge Navigation
    ###########################################################################################

    def halfedge_face(self, edge: Tuple[int, int]) -> Optional[int]:
        """Face on halfedge u->v, or None for boundary halfedges."""
        u, v = edge
        return self.halfedge.get(u, {}).get(v)

    def halfedge_after(self, edge: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Next halfedge in the same face cycle. Walks the boundary if face is None."""
        u, v = edge
        f = self.halfedge.get(u, {}).get(v)
        if f is not None:
            verts = self.face.get(f)
            if verts is None:
                return None
            n = len(verts)
            try:
                i = verts.index(v)
            except ValueError:
                return None
            return (v, verts[(i + 1) % n])
        if v not in self.halfedge:
            return None
        for w, fw in self.halfedge[v].items():
            if w != u and fw is None:
                return (v, w)
        return None

    def halfedge_before(self, edge: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Previous halfedge in the same face cycle. Walks the boundary if face is None."""
        u, v = edge
        f = self.halfedge.get(u, {}).get(v)
        if f is not None:
            verts = self.face.get(f)
            if verts is None:
                return None
            n = len(verts)
            try:
                i = verts.index(u)
            except ValueError:
                return None
            return (verts[(i - 1) % n], u)
        if u not in self.halfedge:
            return None
        for w, fw in self.halfedge[u].items():
            if w != v and self.halfedge.get(w, {}).get(u) is None:
                return (w, u)
        return None

    def halfedge_loop(self, edge: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compas-style: walk along the loop of halfedges in the direction of ``edge``,
        stepping to the opposite ordered neighbor (requires valence 4 interior)."""
        if self.is_edge_on_boundary(*edge):
            return self._halfedge_loop_on_boundary(edge)
        edges = [edge]
        u, v = edge
        guard = 0
        while guard < 10000:
            guard += 1
            nbrs = self.vertex_neighbors(v, ordered=True)
            if nbrs is None or len(nbrs) != 4:
                break
            try:
                i = nbrs.index(u)
            except ValueError:
                break
            u = v
            v = nbrs[i - 2]
            edges.append((u, v))
            if v == edges[0][0]:
                break
        return edges

    def _halfedge_loop_on_boundary(self, edge: Tuple[int, int]) -> List[Tuple[int, int]]:
        edges = [edge]
        u, v = edge
        guard = 0
        while guard < 10000:
            guard += 1
            nbrs = self.vertex_neighbors(v)
            if nbrs is None or len(nbrs) == 2:
                break
            nbr = None
            for temp in nbrs:
                if temp == u:
                    continue
                if self.is_edge_on_boundary(v, temp):
                    nbr = temp
                    break
            if nbr is None:
                break
            u, v = v, nbr
            edges.append((u, v))
            if v == edges[0][0]:
                break
        return edges

    def halfedge_strip(self, edge: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compas-style: walk across quads via the opposite halfedge. Closes by appending start."""
        u, v = edge
        edges = [edge]
        guard = 0
        while guard < 10000:
            guard += 1
            f = self.halfedge.get(u, {}).get(v)
            if f is None:
                break
            verts = self.face.get(f)
            if verts is None or len(verts) != 4:
                break
            i = verts.index(u)
            u = verts[i - 1]
            v = verts[i - 2]
            edges.append((u, v))
            if (u, v) == edge:
                break
        return edges

    ###########################################################################################
    # Sampling (deterministic LCG when seed given, for cross-language parity)
    ###########################################################################################

    @staticmethod
    def _lcg_sample(keys: List, size: int, seed: Optional[int]) -> List:
        if not keys or size <= 0:
            return []
        if seed is None:
            import random as _r
            return _r.sample(keys, min(size, len(keys)))
        s = (seed & 0x7FFFFFFF) or 1
        out = []
        used = set()
        n = len(keys)
        size = min(size, n)
        while len(out) < size:
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            i = s % n
            if i not in used:
                used.add(i)
                out.append(keys[i])
        return out

    def vertex_sample(self, size: int = 1, seed: Optional[int] = None) -> List[int]:
        return Mesh._lcg_sample(list(self.vertex.keys()), size, seed)

    def edge_sample(self, size: int = 1, seed: Optional[int] = None) -> List[Tuple[int, int]]:
        return Mesh._lcg_sample(self.edges(), size, seed)

    def face_sample(self, size: int = 1, seed: Optional[int] = None) -> List[int]:
        return Mesh._lcg_sample(list(self.face.keys()), size, seed)

    ###########################################################################################
    # Compas-style Aliases
    ###########################################################################################

    def face_center(self, face_key: int) -> Optional[Point]:
        """Alias of face_centroid."""
        return self.face_centroid(face_key)

    def face_polygon(self, face_key: int) -> Optional["Polyline"]:
        """Closed Polyline of the face boundary (Polyline acting as a polygon)."""
        pts = self.face_points(face_key)
        if pts is None:
            return None
        from .polyline import Polyline
        closed = list(pts)
        if len(closed) >= 1 and closed[0] != closed[-1]:
            closed.append(closed[0])
        return Polyline(closed)

    def flip_cycles(self) -> None:
        """Alias of flip()."""
        self.flip()

    ###########################################################################################
    # Attribute API
    ###########################################################################################

    def update_default_vertex_attributes(self, **kwargs: object) -> None:
        """Merge defaults; existing per-vertex attributes are unchanged."""
        for k, v in kwargs.items():
            self.default_vertex_attributes[k] = v

    def update_default_face_attributes(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            self.default_face_attributes[k] = v

    def update_default_edge_attributes(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            self.default_edge_attributes[k] = v

    @overload
    def vertex_attribute(self, key: int, name: str) -> Any: ...

    @overload
    def vertex_attribute(self, key: int, name: str, value: object) -> None: ...

    def vertex_attribute(self, key: int, name: str, value: object = None) -> Any:
        """Get when value is None; set otherwise. Returns default if name unset."""
        if key not in self.vertex:
            return None
        if value is None:
            attrs = self.vertex[key].attributes
            if name in attrs:
                return attrs[name]
            return self.default_vertex_attributes.get(name)
        self.vertex[key].attributes[name] = value
        return None

    @overload
    def face_attribute(self, fkey: int, name: str) -> Any: ...

    @overload
    def face_attribute(self, fkey: int, name: str, value: object) -> None: ...

    def face_attribute(self, fkey: int, name: str, value: object = None) -> Any:
        if fkey not in self.face:
            return None
        if value is None:
            attrs = self.facedata.get(fkey, {})
            if name in attrs:
                return attrs[name]
            return self.default_face_attributes.get(name)
        self.facedata.setdefault(fkey, {})[name] = value
        return None

    @overload
    def edge_attribute(self, edge: Tuple[int, int], name: str) -> Any: ...

    @overload
    def edge_attribute(self, edge: Tuple[int, int], name: str, value: object) -> None: ...

    def edge_attribute(self, edge: Tuple[int, int], name: str, value: object = None) -> Any:
        u, v = edge
        if v not in self.halfedge.get(u, {}) and u not in self.halfedge.get(v, {}):
            return None
        key = (u, v) if (u, v) in self.edgedata else ((v, u) if (v, u) in self.edgedata else (u, v))
        if value is None:
            attrs = self.edgedata.get(key, {})
            if name in attrs:
                return attrs[name]
            return self.default_edge_attributes.get(name)
        self.edgedata.setdefault(key, {})[name] = value
        return None

    @overload
    def vertices_attribute(self, name: str, value: None = None, keys: Optional[List[int]] = None) -> List[Any]: ...

    @overload
    def vertices_attribute(self, name: str, value: object = None, keys: Optional[List[int]] = None) -> None: ...

    def vertices_attribute(self, name: str, value: object = None, keys: Optional[List[int]] = None) -> Optional[List[Any]]:
        """Bulk get/set. With value=None and keys=None, returns list over all vertices."""
        if keys is None:
            keys = list(self.vertex.keys())
        if value is None:
            return [self.vertex_attribute(k, name) for k in keys]
        for k in keys:
            self.vertex_attribute(k, name, value)
        return None

    @overload
    def faces_attribute(self, name: str, value: None = None, keys: Optional[List[int]] = None) -> List[Any]: ...

    @overload
    def faces_attribute(self, name: str, value: object = None, keys: Optional[List[int]] = None) -> None: ...

    def faces_attribute(self, name: str, value: object = None, keys: Optional[List[int]] = None) -> Optional[List[Any]]:
        if keys is None:
            keys = list(self.face.keys())
        if value is None:
            return [self.face_attribute(k, name) for k in keys]
        for k in keys:
            self.face_attribute(k, name, value)
        return None

    @overload
    def edges_attribute(self, name: str, value: None = None, keys: Optional[List[Tuple[int, int]]] = None) -> List[Any]: ...

    @overload
    def edges_attribute(self, name: str, value: object = None, keys: Optional[List[Tuple[int, int]]] = None) -> None: ...

    def edges_attribute(self, name: str, value: object = None, keys: Optional[List[Tuple[int, int]]] = None) -> Optional[List[Any]]:
        if keys is None:
            keys = self.edges()
        if value is None:
            return [self.edge_attribute(e, name) for e in keys]
        for e in keys:
            self.edge_attribute(e, name, value)
        return None

    def vertices_where(self, conditions: Dict[str, object]) -> List[int]:
        """Vertices whose attributes match all (name, value) pairs."""
        out = []
        for k in self.vertex:
            if all(self.vertex_attribute(k, n) == v for n, v in conditions.items()):
                out.append(k)
        return out

    def faces_where(self, conditions: Dict[str, object]) -> List[int]:
        out = []
        for k in self.face:
            if all(self.face_attribute(k, n) == v for n, v in conditions.items()):
                out.append(k)
        return out

    def edges_where(self, conditions: Dict[str, object]) -> List[Tuple[int, int]]:
        out = []
        for e in self.edges():
            if all(self.edge_attribute(e, n) == v for n, v in conditions.items()):
                out.append(e)
        return out

    def vertices_where_predicate(self, predicate: Callable) -> List[int]:
        """predicate(key, attrs_dict) -> bool. attrs_dict merges defaults with overrides."""
        out = []
        for k in self.vertex:
            attrs = dict(self.default_vertex_attributes)
            attrs.update(self.vertex[k].attributes)
            if predicate(k, attrs):
                out.append(k)
        return out

    def faces_where_predicate(self, predicate: Callable) -> List[int]:
        out = []
        for k in self.face:
            attrs = dict(self.default_face_attributes)
            attrs.update(self.facedata.get(k, {}))
            if predicate(k, attrs):
                out.append(k)
        return out

    def edges_where_predicate(self, predicate: Callable) -> List[Tuple[int, int]]:
        out = []
        for e in self.edges():
            attrs = dict(self.default_edge_attributes)
            ed = self.edgedata.get(e) or self.edgedata.get((e[1], e[0])) or {}
            attrs.update(ed)
            if predicate(e, attrs):
                out.append(e)
        return out

    ###########################################################################################
    # Geometric Properties
    ###########################################################################################

    def area(self) -> float:
        total = 0.0
        for vkeys in self.face.values():
            if len(vkeys) < 3:
                continue
            vd0 = self.vertex.get(vkeys[0])
            if vd0 is None:
                continue
            x0, y0, z0 = vd0.x, vd0.y, vd0.z
            for i in range(1, len(vkeys) - 1):
                vd1 = self.vertex.get(vkeys[i])
                vd2 = self.vertex.get(vkeys[i + 1])
                if vd1 is None or vd2 is None:
                    continue
                ux = vd1.x - x0; uy = vd1.y - y0; uz = vd1.z - z0
                vx = vd2.x - x0; vy = vd2.y - y0; vz = vd2.z - z0
                cx = uy * vz - uz * vy; cy = uz * vx - ux * vz; cz = ux * vy - uy * vx
                total += math.sqrt(cx*cx + cy*cy + cz*cz) * 0.5
        return total

    def centroid(self) -> Point:
        """Get the centroid of all vertices."""
        x, y, z = 0.0, 0.0, 0.0
        for vk in self.vertex:
            p = self.vertex_point(vk)
            x += p[0]; y += p[1]; z += p[2]
        n = max(len(self.vertex), 1)
        return Point(x / n, y / n, z / n)

    def dihedral_angle(self, u: int, v: int) -> Optional[float]:
        """Calculate the dihedral angle between two faces sharing edge (u,v)."""
        ef = self.edge_faces(u, v)
        if ef is None or len(ef) < 2:
            return None
        n0 = self.face_normal(ef[0])
        n1 = self.face_normal(ef[1])
        if n0 is None or n1 is None:
            return None
        dot = max(-1.0, min(1.0, n0[0]*n1[0] + n0[1]*n1[1] + n0[2]*n1[2]))
        return (PI - math.acos(dot)) * 180.0 / PI

    def dihedral_angles(self, scale: float = 0.3, with_arcs: bool = True, with_points: bool = True) -> Tuple[Dict[Tuple[int, int], float], List["Polyline"], List[Point]]:
        """Calculate dihedral angles for all interior edges.
        Returns (angles, arcs, points): angles dict (u,v)->radians; arcs slerp polylines if scale>0;
        points at arc midpoint (scale>0) or edge midpoint (scale==0). arcs/points empty if flags false."""
        from .polyline import Polyline
        angles = {}
        arcs = []
        points = []
        arc_n = 12
        for u, v in self.edges():
            da = self.dihedral_angle(u, v)
            if da is None:
                continue
            angles[(u, v)] = da
            deg = da
            ep0 = self.vertex_point(u)
            ep1 = self.vertex_point(v)
            if ep0 is None or ep1 is None:
                continue
            mx = (ep0[0]+ep1[0])*0.5
            my = (ep0[1]+ep1[1])*0.5
            mz = (ep0[2]+ep1[2])*0.5
            if scale == 0.0:
                if with_points:
                    pt = Point(mx, my, mz, str(deg))
                    pt.pointcolor = Color(240, 220, 0, 255)
                    points.append(pt)
                continue
            ef = self.edge_faces(u, v)
            if ef is None or len(ef) < 2:
                continue
            ex = ep1[0]-ep0[0]; ey = ep1[1]-ep0[1]; ez = ep1[2]-ep0[2]
            elen = math.sqrt(ex*ex+ey*ey+ez*ez)
            if elen < 1e-10:
                continue
            ex /= elen; ey /= elen; ez /= elen
            fc0 = self.face_centroid(ef[0])
            fc1 = self.face_centroid(ef[1])
            if fc0 is None or fc1 is None:
                continue
            d0x = fc0[0]-mx; d0y = fc0[1]-my; d0z = fc0[2]-mz
            dot0 = d0x*ex+d0y*ey+d0z*ez
            d0x -= dot0*ex; d0y -= dot0*ey; d0z -= dot0*ez
            d0len = math.sqrt(d0x*d0x+d0y*d0y+d0z*d0z)
            if d0len < 1e-10:
                continue
            d0x /= d0len; d0y /= d0len; d0z /= d0len
            d1x = fc1[0]-mx; d1y = fc1[1]-my; d1z = fc1[2]-mz
            dot1 = d1x*ex+d1y*ey+d1z*ez
            d1x -= dot1*ex; d1y -= dot1*ey; d1z -= dot1*ez
            d1len = math.sqrt(d1x*d1x+d1y*d1y+d1z*d1z)
            if d1len < 1e-10:
                continue
            d1x /= d1len; d1y /= d1len; d1z /= d1len
            theta = math.acos(max(-1.0, min(1.0, d0x*d1x+d0y*d1y+d0z*d1z)))
            if abs(math.sin(theta)) < 1e-10:
                continue
            arc_pts = []
            for j in range(arc_n+1):
                t = j / arc_n
                w1 = math.sin((1.0-t)*theta) / math.sin(theta)
                w2 = math.sin(t*theta) / math.sin(theta)
                arc_pts.append(Point(
                    mx+(w1*d0x+w2*d1x)*scale,
                    my+(w1*d0y+w2*d1y)*scale,
                    mz+(w1*d0z+w2*d1z)*scale))
            if with_arcs:
                arc = Polyline(arc_pts)
                arc.name = "dihedral_e"+str(u)+"_"+str(v)+"="+str(deg)
                arc.linecolor = Color(240, 220, 0, 255)
                arcs.append(arc)
            if with_points:
                mid = arc_pts[arc_n//2]
                pt = Point(mid[0], mid[1], mid[2], str(deg))
                pt.pointcolor = Color(240, 220, 0, 255)
                points.append(pt)
        return angles, arcs, points

    def face_area(self, face_key: int) -> Optional[float]:
        """Calculate the area of a face."""
        vkeys = self.face.get(face_key)
        if vkeys is None or len(vkeys) < 3:
            return 0.0
        vd0 = self.vertex.get(vkeys[0])
        if vd0 is None:
            return None
        x0, y0, z0 = vd0.x, vd0.y, vd0.z
        area = 0.0
        for i in range(1, len(vkeys) - 1):
            vd1 = self.vertex.get(vkeys[i])
            vd2 = self.vertex.get(vkeys[i + 1])
            if vd1 is None or vd2 is None:
                return None
            ux = vd1.x - x0; uy = vd1.y - y0; uz = vd1.z - z0
            vx = vd2.x - x0; vy = vd2.y - y0; vz = vd2.z - z0
            cx = uy * vz - uz * vy; cy = uz * vx - ux * vz; cz = ux * vy - uy * vx
            area += math.sqrt(cx*cx + cy*cy + cz*cz) * 0.5
        return area

    def face_centroid(self, face_key: int) -> Optional[Point]:
        """Get the centroid of a face."""
        verts = self.face.get(face_key)
        if not verts:
            return None
        x, y, z = 0.0, 0.0, 0.0
        for vk in verts:
            p = self.vertex_point(vk)
            if p is None:
                return None
            x += p[0]; y += p[1]; z += p[2]
        n = len(verts)
        return Point(x / n, y / n, z / n)

    def face_normal(self, face_key: int, unitized: bool = True) -> Optional[Vector]:
        """Calculate the normal of a face. When unitized=False, length encodes 2x face area."""
        vertices = self.face_vertices(face_key)
        if vertices is None or len(vertices) < 3:
            return None

        p0 = self.vertex_point(vertices[0])
        p1 = self.vertex_point(vertices[1])
        p2 = self.vertex_point(vertices[2])

        if p0 is None or p1 is None or p2 is None:
            return None

        u = Vector(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        v = Vector(p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])

        normal = u.cross(v)
        if not unitized:
            return Vector(normal[0], normal[1], normal[2])
        length = normal.magnitude()

        if length > Tolerance.ZERO_TOLERANCE:
            return Vector(normal[0] / length, normal[1] / length, normal[2] / length)

        return None

    def face_normals(self) -> Dict[int, Vector]:
        """Calculate normals for all faces."""
        normals = {}
        for face_key in self.face:
            normal = self.face_normal(face_key)
            if normal is not None:
                normals[face_key] = normal
        return normals

    def vertex_angle_in_face(self, vertex_key: int, face_key: int) -> Optional[float]:
        """Calculate the angle at a vertex in a face."""
        vertices = self.face_vertices(face_key)
        if vertices is None or vertex_key not in vertices:
            return None

        vertex_index = vertices.index(vertex_key)
        n = len(vertices)
        prev_vertex = vertices[(vertex_index - 1) % n]
        next_vertex = vertices[(vertex_index + 1) % n]

        center = self.vertex_point(vertex_key)
        prev_pos = self.vertex_point(prev_vertex)
        next_pos = self.vertex_point(next_vertex)

        if center is None or prev_pos is None or next_pos is None:
            return None

        u = Vector(prev_pos[0] - center[0], prev_pos[1] - center[1], prev_pos[2] - center[2])
        v = Vector(next_pos[0] - center[0], next_pos[1] - center[1], next_pos[2] - center[2])

        u_len = u.magnitude()
        v_len = v.magnitude()

        if u_len < Tolerance.ZERO_TOLERANCE or v_len < Tolerance.ZERO_TOLERANCE:
            return 0.0

        cos_angle = u.dot(v) / (u_len * v_len)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.acos(cos_angle)

    def vertex_normal(self, vertex_key: int) -> Optional[Vector]:
        """Calculate the normal of a vertex (area-weighted)."""
        return self.vertex_normal_weighted(vertex_key, NormalWeighting.AREA)

    def vertex_normal_weighted(
        self, vertex_key: int, weighting: NormalWeighting
    ) -> Optional[Vector]:
        """Calculate the normal of a vertex with specified weighting."""
        faces = self.vertex_faces(vertex_key)
        if not faces:
            return None

        normal_acc = Vector(0.0, 0.0, 0.0)

        for face_key in faces:
            face_normal = self.face_normal(face_key)
            if face_normal is None:
                continue

            if weighting == NormalWeighting.AREA:
                weight = self.face_area(face_key) or 1.0
            elif weighting == NormalWeighting.ANGLE:
                weight = self.vertex_angle_in_face(vertex_key, face_key) or 1.0
            else:  # UNIFORM
                weight = 1.0

            normal_acc[0] += face_normal[0] * weight
            normal_acc[1] += face_normal[1] * weight
            normal_acc[2] += face_normal[2] * weight

        length = normal_acc.magnitude()
        if length > Tolerance.ZERO_TOLERANCE:
            return Vector(
                normal_acc[0] / length, normal_acc[1] / length, normal_acc[2] / length
            )

        return None

    def vertex_normals(self) -> Dict[int, Vector]:
        """Calculate normals for all vertices (area-weighted)."""
        return self.vertex_normals_weighted(NormalWeighting.AREA)

    def vertex_normals_weighted(self, weighting: NormalWeighting) -> Dict[int, Vector]:
        """Calculate normals for all vertices with specified weighting."""
        acc = {}
        for fk, vkeys in self.face.items():
            n = len(vkeys)
            if n < 3:
                continue
            pts = []
            ok = True
            for vk in vkeys:
                vd = self.vertex.get(vk)
                if vd is None:
                    ok = False
                    break
                pts.append((vd.x, vd.y, vd.z))
            if not ok:
                continue
            ex = pts[1][0]-pts[0][0]; ey = pts[1][1]-pts[0][1]; ez = pts[1][2]-pts[0][2]
            fx = pts[2][0]-pts[0][0]; fy = pts[2][1]-pts[0][1]; fz = pts[2][2]-pts[0][2]
            cnx = ey*fz-ez*fy; cny = ez*fx-ex*fz; cnz = ex*fy-ey*fx
            length = math.sqrt(cnx*cnx + cny*cny + cnz*cnz)
            if length < Tolerance.ZERO_TOLERANCE:
                continue
            ux = cnx/length; uy = cny/length; uz = cnz/length
            area = 0.0
            if weighting == NormalWeighting.AREA:
                for i in range(1, n-1):
                    ax = pts[i][0]-pts[0][0]; ay = pts[i][1]-pts[0][1]; az = pts[i][2]-pts[0][2]
                    bx = pts[i+1][0]-pts[0][0]; by = pts[i+1][1]-pts[0][1]; bz = pts[i+1][2]-pts[0][2]
                    cx = ay*bz-az*by; cy = az*bx-ax*bz; cz = ax*by-ay*bx
                    area += math.sqrt(cx*cx + cy*cy + cz*cz) * 0.5
            for i in range(n):
                if weighting == NormalWeighting.UNIFORM:
                    weight = 1.0
                elif weighting == NormalWeighting.AREA:
                    weight = area
                else:
                    prev = (i + n - 1) % n; nxt = (i + 1) % n
                    ax = pts[prev][0]-pts[i][0]; ay = pts[prev][1]-pts[i][1]; az = pts[prev][2]-pts[i][2]
                    bx = pts[nxt][0]-pts[i][0]; by = pts[nxt][1]-pts[i][1]; bz = pts[nxt][2]-pts[i][2]
                    a_len = math.sqrt(ax*ax + ay*ay + az*az)
                    b_len = math.sqrt(bx*bx + by*by + bz*bz)
                    if a_len < Tolerance.ZERO_TOLERANCE or b_len < Tolerance.ZERO_TOLERANCE:
                        continue
                    cos_a = max(-1.0, min(1.0, (ax*bx + ay*by + az*bz) / (a_len * b_len)))
                    weight = math.acos(cos_a)
                vk = vkeys[i]
                if vk not in acc:
                    acc[vk] = [0.0, 0.0, 0.0]
                acc[vk][0] += ux * weight
                acc[vk][1] += uy * weight
                acc[vk][2] += uz * weight
        normals = {}
        for vk, v in acc.items():
            length = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            if length > Tolerance.ZERO_TOLERANCE:
                normals[vk] = Vector(v[0]/length, v[1]/length, v[2]/length)
        return normals

    def compute_vertex_normals(self) -> None:
        """Compute area-weighted vertex normals and store them on each vertex."""
        normals = self.vertex_normals()
        for key, n in normals.items():
            self.vertex[key].set_normal(n[0], n[1], n[2])

    def volume(self) -> float:
        total = 0.0
        for vkeys in self.face.values():
            if len(vkeys) < 3:
                continue
            vd0 = self.vertex.get(vkeys[0])
            if vd0 is None:
                continue
            x0, y0, z0 = vd0.x, vd0.y, vd0.z
            for i in range(1, len(vkeys) - 1):
                vd1 = self.vertex.get(vkeys[i])
                vd2 = self.vertex.get(vkeys[i + 1])
                if vd1 is None or vd2 is None:
                    continue
                total += (x0 * (vd1.y * vd2.z - vd1.z * vd2.y)
                        + y0 * (vd1.z * vd2.x - vd1.x * vd2.z)
                        + z0 * (vd1.x * vd2.y - vd1.y * vd2.x))
        return abs(total) / 6.0

    ###########################################################################################
    # Export
    ###########################################################################################

    def vertex_index(self) -> Dict[int, int]:
        """Create a mapping from sparse vertex keys to sequential indices.

        Returns
        -------
        dict[int, int]
            A dictionary mapping vertex_key -> sequential_index (0, 1, 2, ...).
        """
        # Sort keys to ensure consistent ordering
        sorted_keys = sorted(self.vertex.keys())
        return {key: index for index, key in enumerate(sorted_keys)}

    def to_vertices_and_faces(self) -> Tuple[List[Point], List[List[int]]]:
        """Export vertices and faces with sequential 0-based indices.

        Returns
        -------
        tuple
            A tuple of (vertices, faces) where:
            - vertices: List of Point objects in sequential order
            - faces: List of face vertex lists using sequential indices
        """
        vertex_idx = self.vertex_index()
        vertices = [None] * len(self.vertex)

        for key, vdata in self.vertex.items():
            idx = vertex_idx[key]
            vertices[idx] = vdata.position()

        # Sort face keys to ensure consistent ordering
        sorted_face_keys = sorted(self.face.keys())
        faces = []
        for face_key in sorted_face_keys:
            face_vertices = self.face[face_key]
            remapped = [vertex_idx[v] for v in face_vertices]
            faces.append(remapped)

        return vertices, faces

    ###########################################################################################
    # Transformation
    ###########################################################################################

    def transform(self, xform: "Xform") -> None:
        for vdata in self.vertex.values():
            pos = vdata.position()
            pos.transform(xform)
            vdata[0] = pos[0]
            vdata[1] = pos[1]
            vdata[2] = pos[2]
        self._triangle_bvh_built = False

    def transformed(self, xform: "Xform") -> "Mesh":
        import copy
        result = copy.deepcopy(self)
        result.transform(xform)
        return result

    ###########################################################################################
    # Triangle BVH (closest point / ray queries)
    ###########################################################################################

    def build_triangle_bvh(self, force: bool = False) -> None:
        """Build (and cache) a BVH over the mesh's triangulated faces."""
        if self._triangle_bvh_built and not force:
            return

        from session_py.spatial_bvh import SpatialBVH
        from session_py.aabb import AABB

        self._triangle_aabbs_cache = []
        self._triangle_indices_cache = []
        self._triangle_face_subidx_cache = []
        self._vertices_cache = []

        vertices, faces_vec = self.to_vertices_and_faces()
        self._vertices_cache = vertices

        vertex_keys = sorted(self.vertex.keys())
        vkey_to_idx = {}
        for i in range(len(vertex_keys)):
            vkey_to_idx[vertex_keys[i]] = i

        face_keys = sorted(self.face.keys())

        tasks = []
        for fi in range(len(faces_vec)):
            fv = faces_vec[fi]
            if len(fv) < 3:
                continue
            if len(fv) >= 5 and fi < len(face_keys):
                tri = self.triangulation.get(face_keys[fi])
                if tri is not None:
                    for j in range(len(tri)):
                        t = tri[j]
                        tasks.append((vkey_to_idx[t[0]], vkey_to_idx[t[1]], vkey_to_idx[t[2]], fi, j))
                    continue
            for j in range(1, len(fv) - 1):
                tasks.append((fv[0], fv[j], fv[j + 1], fi, j))

        for i0, i1, i2, face_idx, sub_idx in tasks:
            p0 = self._vertices_cache[i0]
            p1 = self._vertices_cache[i1]
            p2 = self._vertices_cache[i2]

            min_x = min(p0[0], p1[0], p2[0]) - 0.001
            min_y = min(p0[1], p1[1], p2[1]) - 0.001
            min_z = min(p0[2], p1[2], p2[2]) - 0.001
            max_x = max(p0[0], p1[0], p2[0]) + 0.001
            max_y = max(p0[1], p1[1], p2[1]) + 0.001
            max_z = max(p0[2], p1[2], p2[2]) + 0.001

            cx = (min_x + max_x) * 0.5
            cy = (min_y + max_y) * 0.5
            cz = (min_z + max_z) * 0.5
            hx = (max_x - min_x) * 0.5
            hy = (max_y - min_y) * 0.5
            hz = (max_z - min_z) * 0.5

            self._triangle_aabbs_cache.append(AABB(cx, cy, cz, hx, hy, hz))
            self._triangle_indices_cache.append((i0, i1, i2))
            self._triangle_face_subidx_cache.append((face_idx, sub_idx))

        # Compute world size from object bounds (triangle AABBs)
        min_x = float('inf')
        min_y = float('inf')
        min_z = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        max_z = float('-inf')
        for bb in self._triangle_aabbs_cache:
            bx0 = bb.cx - bb.hx
            bx1 = bb.cx + bb.hx
            by0 = bb.cy - bb.hy
            by1 = bb.cy + bb.hy
            bz0 = bb.cz - bb.hz
            bz1 = bb.cz + bb.hz
            if bx0 < min_x:
                min_x = bx0
            if bx1 > max_x:
                max_x = bx1
            if by0 < min_y:
                min_y = by0
            if by1 > max_y:
                max_y = by1
            if bz0 < min_z:
                min_z = bz0
            if bz1 > max_z:
                max_z = bz1
        extent_x = max(abs(min_x), abs(max_x))
        extent_y = max(abs(min_y), abs(max_y))
        extent_z = max(abs(min_z), abs(max_z))
        max_extent = max(extent_x, extent_y, extent_z)
        world_size = max(2.2 * max_extent, 10.0)

        self._triangle_bvh = SpatialBVH()
        self._triangle_bvh.build_from_aabbs(self._triangle_aabbs_cache, world_size)
        self._triangle_bvh_built = True

    def get_cached_bvh(self) -> Optional["SpatialBVH"]:
        """Return the cached triangle BVH (or None if not built)."""
        return self._triangle_bvh

    def get_triangle_by_id(self, tri_id: int) -> Tuple[bool, int, int, Optional[Point], Optional[Point], Optional[Point]]:
        """Return (found, face_idx, sub_idx, v0, v1, v2) for a cached triangle id."""
        if tri_id < 0:
            return (False, 0, 0, None, None, None)
        if tri_id >= len(self._triangle_indices_cache) or tri_id >= len(self._triangle_face_subidx_cache):
            return (False, 0, 0, None, None, None)
        tri = self._triangle_indices_cache[tri_id]
        fs = self._triangle_face_subidx_cache[tri_id]
        face_idx = fs[0]
        sub_idx = fs[1]
        if tri[0] >= len(self._vertices_cache) or tri[1] >= len(self._vertices_cache) or tri[2] >= len(self._vertices_cache):
            return (False, 0, 0, None, None, None)
        v0 = self._vertices_cache[tri[0]]
        v1 = self._vertices_cache[tri[1]]
        v2 = self._vertices_cache[tri[2]]
        return (True, face_idx, sub_idx, v0, v1, v2)

    def clear_triangle_bvh(self) -> None:
        """Drop the cached triangle BVH."""
        self._triangle_bvh_built = False
        self._triangle_bvh = None
        self._triangle_aabbs_cache = []
        self._triangle_indices_cache = []
        self._triangle_face_subidx_cache = []
        self._vertices_cache = []

    ###########################################################################################
    # JSON
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field.

        Returns
        -------
        dict
            Dictionary with fields in alphabetical order (matching Rust).

        """
        # Halfedge connectivity
        halfedge_data = {}
        for u, neighbors in self.halfedge.items():
            halfedge_data[str(u)] = {
                str(v): face_key for v, face_key in neighbors.items()
            }

        # Vertex data (alphabetical: attributes, x, y, z)
        vertex_data = {}
        for key, vdata in self.vertex.items():
            vertex_data[str(key)] = {
                "attributes": vdata.attributes,
                "x": vdata[0],
                "y": vdata[1],
                "z": vdata[2],
            }

        # Face data
        face_data = {}
        for key, vertices in self.face.items():
            face_data[str(key)] = vertices

        # Face attributes
        facedata_json = {}
        for key, attrs in self.facedata.items():
            facedata_json[str(key)] = attrs

        # Edge attributes
        edgedata_json = {}
        for (u, v), attrs in self.edgedata.items():
            edgedata_json[f"{u},{v}"] = attrs

        # Colors as flat RGBA arrays
        pointcolors_flat = []
        for c in self._pointcolors:
            pointcolors_flat.extend([c[0], c[1], c[2], c[3]])

        facecolors_flat = []
        for c in self._facecolors:
            facecolors_flat.extend([c[0], c[1], c[2], c[3]])

        linecolors_flat = []
        for c in self._linecolors:
            linecolors_flat.extend([c[0], c[1], c[2], c[3]])

        # Return fields in alphabetical order to match Rust's serde_json
        return {
            "color_mode": self.color_mode.value,
            "default_edge_attributes": self.default_edge_attributes,
            "default_face_attributes": self.default_face_attributes,
            "default_vertex_attributes": self.default_vertex_attributes,
            "edgedata": edgedata_json,
            "face": face_data,
            "face_holes": {str(fk): [list(r) for r in rings] for fk, rings in self.face_holes.items()},
            "facecolors": facecolors_flat,
            "facedata": facedata_json,
            "guid": self.guid,
            "halfedge": halfedge_data,
            "linecolors": linecolors_flat,
            "max_face": self._max_face,
            "max_vertex": self._max_vertex,
            "name": self.name,
            "objectcolor": self.objectcolor.__jsondump__(),
            "pointcolors": pointcolors_flat,
            "triangulation": {
                str(fk): [[t[0], t[1], t[2]] for t in tris]
                for fk, tris in self.triangulation.items()
            },
            "type": f"{self.__class__.__name__}",
            "vertex": vertex_data,
            "widths": self._widths,
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format.

        Parameters
        ----------
        data : dict
            Dictionary containing mesh data.
        guid : str, optional
            GUID for the mesh.
        name : str, optional
            Name for the mesh.

        Returns
        -------
        :class:`Mesh`
            Reconstructed mesh instance.

        """
        mesh = cls()
        mesh.guid = guid if guid is not None else data.get("guid", mesh.guid)
        mesh.name = name if name is not None else data.get("name", mesh.name)

        # Load halfedge connectivity
        if "halfedge" in data:
            for u_str, neighbors in data["halfedge"].items():
                u = int(u_str)
                mesh.halfedge[u] = {}
                for v_str, face_key in neighbors.items():
                    v = int(v_str)
                    mesh.halfedge[u][v] = face_key

        # Load vertex data
        if "vertex" in data:
            for key_str, vdata in data["vertex"].items():
                key = int(key_str)
                vertex_data = VertexData()
                vertex_data.x = vdata["x"]
                vertex_data.y = vdata["y"]
                vertex_data.z = vdata["z"]
                if "attributes" in vdata:
                    vertex_data.attributes = vdata["attributes"]
                mesh.vertex[key] = vertex_data
                if "halfedge" not in data:
                    mesh.halfedge[key] = {}
                if key >= mesh._max_vertex:
                    mesh._max_vertex = key + 1

        # Load face data
        if "face" in data:
            for key_str, vertices in data["face"].items():
                key = int(key_str)
                mesh.face[key] = vertices
                if key >= mesh._max_face:
                    mesh._max_face = key + 1

        # Load face attributes
        if data.get("facedata"):
            for key_str, attrs in data["facedata"].items():
                key = int(key_str)
                mesh.facedata[key] = attrs

        # Load edge attributes
        if data.get("edgedata"):
            for edge_str, attrs in data["edgedata"].items():
                u, v = map(int, edge_str.split(","))
                mesh.edgedata[(u, v)] = attrs

        if data.get("face_holes"):
            for fk_str, rings in data["face_holes"].items():
                mesh.face_holes[int(fk_str)] = [list(r) for r in rings]

        if data.get("triangulation"):
            for fk_str, tris_json in data["triangulation"].items():
                mesh.triangulation[int(fk_str)] = [[t[0], t[1], t[2]] for t in tris_json]

        if "default_vertex_attributes" in data:
            mesh.default_vertex_attributes = data["default_vertex_attributes"]
        if "default_face_attributes" in data:
            mesh.default_face_attributes = data["default_face_attributes"]
        if "default_edge_attributes" in data:
            mesh.default_edge_attributes = data["default_edge_attributes"]

        if "max_vertex" in data:
            mesh._max_vertex = data["max_vertex"]
        if "max_face" in data:
            mesh._max_face = data["max_face"]

        # Load colors from flat RGBA arrays
        if "pointcolors" in data:
            arr = data["pointcolors"]
            mesh._pointcolors = [Color(arr[i], arr[i+1], arr[i+2], arr[i+3]) for i in range(0, len(arr) - 3, 4)]

        if "facecolors" in data:
            arr = data["facecolors"]
            mesh._facecolors = [Color(arr[i], arr[i+1], arr[i+2], arr[i+3]) for i in range(0, len(arr) - 3, 4)]

        if "linecolors" in data:
            arr = data["linecolors"]
            mesh._linecolors = [Color(arr[i], arr[i+1], arr[i+2], arr[i+3]) for i in range(0, len(arr) - 3, 4)]

        if "widths" in data:
            mesh._widths = data["widths"]

        if "objectcolor" in data:
            mesh._objectcolor = Color.__jsonload__(data["objectcolor"])
        if "color_mode" in data:
            mesh.color_mode = ColorMode(data["color_mode"]) if data["color_mode"] in {m.value for m in ColorMode} else ColorMode.OBJECTCOLOR

        return mesh

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write JSON to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "Mesh":
        """Read JSON from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def file_json_dumps(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Mesh":
        """Load from JSON string."""
        import json
        return cls.__jsonload__(json.loads(json_string))

    ###########################################################################################
    # Protobuf
    ###########################################################################################

    def pb_dumps(self) -> bytes:
        """Convert to protobuf binary format."""
        from .proto import mesh_pb2

        proto = mesh_pb2.Mesh()
        proto.guid = self.guid
        proto.name = self.name

        # Vertices
        for vkey, vdata in self.vertex.items():
            vertex_proto = proto.vertices[vkey]
            vertex_proto.x = vdata.x
            vertex_proto.y = vdata.y
            vertex_proto.z = vdata.z
            for k, v in vdata.attributes.items():
                vertex_proto.attributes[k] = v

        # Faces
        for fkey, fverts in self.face.items():
            face_proto = proto.faces[fkey]
            face_proto.vertices.extend(fverts)
            if fkey in self.facedata:
                for k, v in self.facedata[fkey].items():
                    face_proto.attributes[k] = v
            if fkey in self.face_holes:
                for ring in self.face_holes[fkey]:
                    hole_proto = mesh_pb2.HoleRing()
                    hole_proto.vertices.extend(ring)
                    face_proto.holes.append(hole_proto)

        # Triangulation
        for fkey, tris in self.triangulation.items():
            tri_list = proto.triangulation[fkey]
            for t in tris:
                tri_list.vertices.append(t[0])
                tri_list.vertices.append(t[1])
                tri_list.vertices.append(t[2])

        # Halfedges
        for u, neighbors in self.halfedge.items():
            hmap = proto.halfedges[u]
            for v, fkey_opt in neighbors.items():
                hmap.neighbors[v] = fkey_opt if fkey_opt is not None else 0xFFFFFFFFFFFFFFFF

        # Edge data
        for (v1, v2), attrs in self.edgedata.items():
            edge_proto = mesh_pb2.EdgeData()
            edge_proto.vertex1 = v1
            edge_proto.vertex2 = v2
            for k, v in attrs.items():
                edge_proto.attributes[k] = v
            proto.edge_data.append(edge_proto)

        # Default attributes
        for k, v in self.default_vertex_attributes.items():
            proto.default_vertex_attributes[k] = v
        for k, v in self.default_face_attributes.items():
            proto.default_face_attributes[k] = v
        for k, v in self.default_edge_attributes.items():
            proto.default_edge_attributes[k] = v

        # Colors
        from .proto import color_pb2
        for c in self._pointcolors:
            color_proto = color_pb2.Color()
            color_proto.guid = c.guid
            color_proto.name = c.name
            color_proto.r = c[0]
            color_proto.g = c[1]
            color_proto.b = c[2]
            color_proto.a = c[3]
            proto.pointcolors.append(color_proto)

        for c in self._facecolors:
            color_proto = color_pb2.Color()
            color_proto.guid = c.guid
            color_proto.name = c.name
            color_proto.r = c[0]
            color_proto.g = c[1]
            color_proto.b = c[2]
            color_proto.a = c[3]
            proto.facecolors.append(color_proto)

        for c in self._linecolors:
            color_proto = color_pb2.Color()
            color_proto.guid = c.guid
            color_proto.name = c.name
            color_proto.r = c[0]
            color_proto.g = c[1]
            color_proto.b = c[2]
            color_proto.a = c[3]
            proto.linecolors.append(color_proto)

        # Widths
        proto.widths.extend(self._widths)

        # Object color
        proto.objectcolor.guid = self.objectcolor.guid
        proto.objectcolor.name = self.objectcolor.name
        proto.objectcolor.r = self.objectcolor[0]
        proto.objectcolor.g = self.objectcolor[1]
        proto.objectcolor.b = self.objectcolor[2]
        proto.objectcolor.a = self.objectcolor[3]
        _cm_map = {"objectcolor": 0, "pointcolors": 1, "facecolors": 2, "none": 3}
        proto.color_mode = _cm_map.get(self.color_mode.value, 0)

        return proto.SerializeToString()

    def pb_fill(self, proto: "mesh_pb2.Mesh") -> None:
        """Fill an existing Mesh proto message directly (avoids serialize/deserialize cycle)."""
        from .proto import mesh_pb2
        from .proto import color_pb2
        proto.guid = self.guid
        proto.name = self.name
        for vkey, vdata in self.vertex.items():
            vp = proto.vertices[vkey]
            vp.x = vdata.x; vp.y = vdata.y; vp.z = vdata.z
            for k, v in vdata.attributes.items():
                vp.attributes[k] = v
        for fkey, fverts in self.face.items():
            fp = proto.faces[fkey]
            fp.vertices.extend(fverts)
            if fkey in self.facedata:
                for k, v in self.facedata[fkey].items():
                    fp.attributes[k] = v
            if fkey in self.face_holes:
                for ring in self.face_holes[fkey]:
                    hp = mesh_pb2.HoleRing()
                    hp.vertices.extend(ring)
                    fp.holes.append(hp)
        for fkey, tris in self.triangulation.items():
            tl = proto.triangulation[fkey]
            for t in tris:
                tl.vertices.append(t[0]); tl.vertices.append(t[1]); tl.vertices.append(t[2])
        for u, neighbors in self.halfedge.items():
            hmap = proto.halfedges[u]
            for v, fkey_opt in neighbors.items():
                hmap.neighbors[v] = fkey_opt if fkey_opt is not None else 0xFFFFFFFFFFFFFFFF
        for (v1, v2), attrs in self.edgedata.items():
            ep = mesh_pb2.EdgeData()
            ep.vertex1 = v1; ep.vertex2 = v2
            for k, v in attrs.items():
                ep.attributes[k] = v
            proto.edge_data.append(ep)
        for k, v in self.default_vertex_attributes.items():
            proto.default_vertex_attributes[k] = v
        for k, v in self.default_face_attributes.items():
            proto.default_face_attributes[k] = v
        for k, v in self.default_edge_attributes.items():
            proto.default_edge_attributes[k] = v
        for c in self._pointcolors:
            cp = color_pb2.Color()
            cp.guid = c.guid; cp.name = c.name
            cp.r = c[0]; cp.g = c[1]; cp.b = c[2]; cp.a = c[3]
            proto.pointcolors.append(cp)
        for c in self._facecolors:
            cp = color_pb2.Color()
            cp.guid = c.guid; cp.name = c.name
            cp.r = c[0]; cp.g = c[1]; cp.b = c[2]; cp.a = c[3]
            proto.facecolors.append(cp)
        for c in self._linecolors:
            cp = color_pb2.Color()
            cp.guid = c.guid; cp.name = c.name
            cp.r = c[0]; cp.g = c[1]; cp.b = c[2]; cp.a = c[3]
            proto.linecolors.append(cp)
        proto.widths.extend(self._widths)
        proto.objectcolor.guid = self.objectcolor.guid
        proto.objectcolor.name = self.objectcolor.name
        proto.objectcolor.r = self.objectcolor[0]
        proto.objectcolor.g = self.objectcolor[1]
        proto.objectcolor.b = self.objectcolor[2]
        proto.objectcolor.a = self.objectcolor[3]
        _cm_map = {"objectcolor": 0, "pointcolors": 1, "facecolors": 2, "none": 3}
        proto.color_mode = _cm_map.get(self.color_mode.value, 0)

    @classmethod
    def pb_loads(cls, data: bytes) -> "Mesh":
        """Create Mesh from protobuf binary data."""
        from .proto import mesh_pb2
        from .color import Color

        proto = mesh_pb2.Mesh()
        proto.ParseFromString(data)

        mesh = cls()
        mesh.guid = proto.guid
        mesh.name = proto.name

        # Vertices
        for vkey, vdata in proto.vertices.items():
            attrs = dict(vdata.attributes)
            mesh.vertex[vkey] = VertexData(Point(vdata.x, vdata.y, vdata.z))
            mesh.vertex[vkey].attributes = attrs
            if vkey not in mesh.halfedge:
                mesh.halfedge[vkey] = {}

        # Faces
        for fkey, fdata in proto.faces.items():
            mesh.face[fkey] = list(fdata.vertices)
            if fdata.attributes:
                mesh.facedata[fkey] = dict(fdata.attributes)
            if fdata.holes:
                mesh.face_holes[fkey] = [list(h.vertices) for h in fdata.holes]

        # Triangulation
        if hasattr(proto, 'triangulation'):
            for fkey, tri_list in proto.triangulation.items():
                vlist = list(tri_list.vertices)
                tris = [[vlist[i], vlist[i+1], vlist[i+2]] for i in range(0, len(vlist) - 2, 3)]
                mesh.triangulation[fkey] = tris

        # Halfedges
        for u, hmap in proto.halfedges.items():
            neighbors = {}
            for v, fkey in hmap.neighbors.items():
                neighbors[v] = None if fkey == 0xFFFFFFFFFFFFFFFF else fkey
            mesh.halfedge[u] = neighbors

        # Edge data
        for edata in proto.edge_data:
            key = (edata.vertex1, edata.vertex2)
            mesh.edgedata[key] = dict(edata.attributes)

        # Default attributes
        mesh.default_vertex_attributes = dict(proto.default_vertex_attributes)
        mesh.default_face_attributes = dict(proto.default_face_attributes)
        mesh.default_edge_attributes = dict(proto.default_edge_attributes)

        # Colors
        mesh._pointcolors = []
        for c in proto.pointcolors:
            color = Color(c.r, c.g, c.b, c.a)
            color.guid = c.guid
            color.name = c.name
            mesh._pointcolors.append(color)

        mesh._facecolors = []
        for c in proto.facecolors:
            color = Color(c.r, c.g, c.b, c.a)
            color.guid = c.guid
            color.name = c.name
            mesh._facecolors.append(color)

        mesh._linecolors = []
        for c in proto.linecolors:
            color = Color(c.r, c.g, c.b, c.a)
            color.guid = c.guid
            color.name = c.name
            mesh._linecolors.append(color)

        # Widths
        mesh._widths = list(proto.widths)

        # Object color
        oc = proto.objectcolor
        mesh._objectcolor = Color(oc.r, oc.g, oc.b, oc.a)
        mesh._objectcolor.guid = oc.guid
        mesh._objectcolor.name = oc.name
        _cm_map = {0: "objectcolor", 1: "pointcolors", 2: "facecolors", 3: "none"}
        mesh.color_mode = ColorMode(_cm_map.get(getattr(proto, 'color_mode', 0), "objectcolor"))

        # Update max counters
        if mesh.vertex:
            mesh._max_vertex = max(mesh.vertex.keys()) + 1
        if mesh.face:
            mesh._max_face = max(mesh.face.keys()) + 1

        return mesh

    @classmethod
    def from_proto(cls, proto: "mesh_pb2.Mesh") -> "Mesh":
        """Create Mesh from proto message directly (no SerializeToString)."""
        from .color import Color

        mesh = cls()
        mesh.guid = proto.guid
        mesh.name = proto.name

        for vkey, vdata in proto.vertices.items():
            attrs = dict(vdata.attributes)
            mesh.vertex[vkey] = VertexData(Point(vdata.x, vdata.y, vdata.z))
            mesh.vertex[vkey].attributes = attrs
            if vkey not in mesh.halfedge:
                mesh.halfedge[vkey] = {}

        for fkey, fdata in proto.faces.items():
            mesh.face[fkey] = list(fdata.vertices)
            if fdata.attributes:
                mesh.facedata[fkey] = dict(fdata.attributes)
            if fdata.holes:
                mesh.face_holes[fkey] = [list(h.vertices) for h in fdata.holes]

        if hasattr(proto, 'triangulation'):
            for fkey, tri_list in proto.triangulation.items():
                vlist = list(tri_list.vertices)
                tris = [[vlist[i], vlist[i+1], vlist[i+2]] for i in range(0, len(vlist) - 2, 3)]
                mesh.triangulation[fkey] = tris

        for u, hmap in proto.halfedges.items():
            neighbors = {}
            for v, fkey in hmap.neighbors.items():
                neighbors[v] = None if fkey == 0xFFFFFFFFFFFFFFFF else fkey
            mesh.halfedge[u] = neighbors

        for edata in proto.edge_data:
            key = (edata.vertex1, edata.vertex2)
            mesh.edgedata[key] = dict(edata.attributes)

        mesh.default_vertex_attributes = dict(proto.default_vertex_attributes)
        mesh.default_face_attributes = dict(proto.default_face_attributes)
        mesh.default_edge_attributes = dict(proto.default_edge_attributes)

        mesh._pointcolors = []
        for c in proto.pointcolors:
            color = Color(c.r, c.g, c.b, c.a)
            color.guid = c.guid
            color.name = c.name
            mesh._pointcolors.append(color)

        mesh._facecolors = []
        for c in proto.facecolors:
            color = Color(c.r, c.g, c.b, c.a)
            color.guid = c.guid
            color.name = c.name
            mesh._facecolors.append(color)

        mesh._linecolors = []
        for c in proto.linecolors:
            color = Color(c.r, c.g, c.b, c.a)
            color.guid = c.guid
            color.name = c.name
            mesh._linecolors.append(color)

        mesh._widths = list(proto.widths)

        oc = proto.objectcolor
        mesh._objectcolor = Color(oc.r, oc.g, oc.b, oc.a)
        mesh._objectcolor.guid = oc.guid
        mesh._objectcolor.name = oc.name
        _cm_map = {0: "objectcolor", 1: "pointcolors", 2: "facecolors", 3: "none"}
        mesh.color_mode = ColorMode(_cm_map.get(getattr(proto, 'color_mode', 0), "objectcolor"))

        if mesh.vertex:
            mesh._max_vertex = max(mesh.vertex.keys()) + 1
        if mesh.face:
            mesh._max_face = max(mesh.face.keys()) + 1

        return mesh

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write protobuf to file."""
        data = self.pb_dumps()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "Mesh":
        """Read protobuf from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)

    ###########################################################################################
    # Color and Width Management
    ###########################################################################################

    def set_vertex_color(self, index: int, color: Color) -> None:
        """Set color for a specific vertex."""
        if 0 <= index < len(self._pointcolors):
            self._pointcolors[index] = color

    def set_face_color(self, index: int, color: Color) -> None:
        """Set color for a specific face."""
        if 0 <= index < len(self._facecolors):
            self._facecolors[index] = color

    def set_edge_color(self, index: int, color: Color) -> None:
        """Set color for a specific edge."""
        if 0 <= index < len(self._linecolors):
            self._linecolors[index] = color

    def set_edge_width(self, index: int, width: float) -> None:
        """Set width for a specific edge."""
        if 0 <= index < len(self._widths):
            self._widths[index] = width