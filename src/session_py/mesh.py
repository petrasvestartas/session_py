import uuid
import math
from enum import Enum
from typing import Optional, List, Dict, Tuple
from .point import Point
from .vector import Vector
from .tolerance import Tolerance
from .color import Color
from .xform import Xform
from .triangulation_2d import triangulate as _tri2d_triangulate
from .trimesh_cdt import cdt_triangulate as _cdt_triangulate


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

    def __init__(self, point: Point = None):
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

    def set_position(self, point: Point):
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

    def set_color(self, r: float, g: float, b: float):
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

    def set_normal(self, nx: float, ny: float, nz: float):
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
        self._max_vertex = 0
        self._max_face = 0
        self.guid = str(uuid.uuid4())
        self.name = "my_mesh"
        self._pointcolors = []
        self._facecolors = []
        self._linecolors = []
        self._widths = []
        self._objectcolor = Color.white()
        self.color_mode = ColorMode.OBJECTCOLOR
        self.xform = Xform.identity()

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
        m._max_vertex = self._max_vertex
        m._max_face = self._max_face
        m._pointcolors = list(self._pointcolors)
        m._facecolors = list(self._facecolors)
        m._linecolors = list(self._linecolors)
        m._widths = list(self._widths)
        m._objectcolor = self._objectcolor
        m.color_mode = self.color_mode
        m.xform = self.xform
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
        if self.xform != other.xform:
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
        polygons : list of list of Point
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
                kx = round(p.x / precision)
                ky = round(p.y / precision)
                kz = round(p.z / precision)
                key = (kx, ky, kz)
                if key in map_eps:
                    return map_eps[key]
                vk = mesh.add_vertex(p)
                map_eps[key] = vk
                return vk
            else:
                key = (p.x, p.y, p.z)
                if key in map_exact:
                    return map_exact[key]
                vk = mesh.add_vertex(p)
                map_exact[key] = vk
                return vk

        for poly in polygons:
            if len(poly) < 3:
                continue
            vkeys = [get_vkey(p) for p in poly]
            mesh.add_face(vkeys)

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
            xs = [p.x for p in all_pts]
            ys = [p.y for p in all_pts]
            zs = [p.z for p in all_pts]
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
            kx = round(p.x / eps)
            ky = round(p.y / eps)
            kz = round(p.z / eps)
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
            vx, vy = verts[v].x, verts[v].y
            nbrs.sort(key=lambda n: math.atan2(verts[n].y - vy, verts[n].x - vx))
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
            max_idx = max(range(len(face_cycles)), key=lambda i: len(face_cycles[i]))
            face_cycles.pop(max_idx)

        mesh = Mesh()
        vkeys = []
        for pt in verts:
            vkeys.append(mesh.add_vertex(pt))
        for cycle in face_cycles:
            pts = [verts[i] for i in cycle]
            tris = _tri2d_triangulate(pts)
            for t in tris:
                mesh.add_face([vkeys[cycle[t[0]]], vkeys[cycle[t[1]]], vkeys[cycle[t[2]]]])
        return mesh

    @staticmethod
    def from_polygon_with_holes(
        polylines: List[List[Point]], sort_by_bbox: bool = False
    ) -> "Mesh":
        if not polylines:
            return Mesh()
        border_idx = 0
        if sort_by_bbox and len(polylines) > 1:
            max_diag = 0.0
            for i, poly in enumerate(polylines):
                if len(poly) < 3:
                    continue
                xs = [p.x for p in poly]
                ys = [p.y for p in poly]
                zs = [p.z for p in poly]
                dx = max(xs) - min(xs)
                dy = max(ys) - min(ys)
                dz = max(zs) - min(zs)
                diag = math.sqrt(dx*dx + dy*dy + dz*dz)
                if diag > max_diag:
                    max_diag = diag
                    border_idx = i
        def strip_close(pts):
            if len(pts) > 1:
                f, b = pts[0], pts[-1]
                if abs(f.x-b.x) < 1e-12 and abs(f.y-b.y) < 1e-12 and abs(f.z-b.z) < 1e-12:
                    return pts[:-1]
            return pts
        border = strip_close(polylines[border_idx])
        if len(border) < 3:
            return Mesh()
        from .polyline import Polyline as _Polyline
        origin, xaxis, yaxis, zaxis = _Polyline(border).get_average_plane()
        def project_2d(p):
            dx = p.x - origin.x
            dy = p.y - origin.y
            dz = p.z - origin.z
            u = dx * xaxis.x + dy * xaxis.y + dz * xaxis.z
            v = dx * yaxis.x + dy * yaxis.y + dz * yaxis.z
            return Point(u, v, 0.0)
        boundary_2d = [project_2d(p) for p in border]
        def signed_area(pts):
            a = 0.0
            n = len(pts)
            for i in range(n):
                j = (i + 1) % n
                a += pts[i].x * pts[j].y - pts[j].x * pts[i].y
            return a * 0.5
        if signed_area(boundary_2d) < 0.0:
            border.reverse()
            boundary_2d.reverse()
        holes_2d = []
        hole_pts_3d = []
        for i, poly in enumerate(polylines):
            if i == border_idx:
                continue
            hole = strip_close(poly)
            if len(hole) < 3:
                continue
            hole_2d = [project_2d(p) for p in hole]
            if signed_area(hole_2d) > 0.0:
                hole.reverse()
                hole_2d.reverse()
            holes_2d.append(hole_2d)
            hole_pts_3d.append(hole)
        tris = _cdt_triangulate(boundary_2d, holes_2d if holes_2d else None)
        all_pts = list(border)
        for h in hole_pts_3d:
            all_pts.extend(h)
        mesh = Mesh()
        vkeys = []
        for p in all_pts:
            vkeys.append(mesh.add_vertex(p))
        for t in tris:
            mesh.add_face([vkeys[t[0]], vkeys[t[1]], vkeys[t[2]]])
        return mesh

    @staticmethod
    def loft(polylines0: List, polylines1: List, cap: bool = True) -> "Mesh":
        if not polylines0 or not polylines1 or len(polylines0) != len(polylines1):
            return Mesh()
        border_idx = 0
        max_diag = 0.0
        for i, pl in enumerate(polylines0):
            pts = pl.get_points()
            if not pts:
                continue
            xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
            dx = max(xs) - min(xs); dy = max(ys) - min(ys); dz = max(zs) - min(zs)
            diag = math.sqrt(dx*dx + dy*dy + dz*dz)
            if diag > max_diag:
                max_diag = diag; border_idx = i
        def get_open(pl):
            pts = pl.get_points()
            if len(pts) > 1:
                f, b = pts[0], pts[-1]
                if abs(f.x-b.x) < 1e-12 and abs(f.y-b.y) < 1e-12 and abs(f.z-b.z) < 1e-12:
                    return pts[:-1]
            return pts
        origin, xaxis, yaxis, _ = polylines0[border_idx].get_average_plane()
        def proj(p):
            dx = p.x - origin.x; dy = p.y - origin.y; dz = p.z - origin.z
            return (dx*xaxis.x + dy*xaxis.y + dz*xaxis.z, dx*yaxis.x + dy*yaxis.y + dz*yaxis.z)
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
        for oi, idx in enumerate(order):
            bot = get_open(polylines0[idx]); top = get_open(polylines1[idx])
            if (oi == 0 and sarea(bot) < 0) or (oi != 0 and sarea(bot) > 0):
                bot.reverse(); top.reverse()
            poly_infos.append((len(all_bot), len(bot), len(all_top), len(top)))
            all_bot.extend(bot); all_top.extend(top)
        mesh = Mesh()
        bvk = [mesh.add_vertex(p) for p in all_bot]
        tvk = [mesh.add_vertex(p) for p in all_top]
        if cap:
            _, bot_n0, _, top_n0 = poly_infos[0]
            bpts = [Point(*proj(all_bot[i]), 0.0) for i in range(bot_n0)]
            b_hpts = [[Point(*proj(all_bot[i]), 0.0) for i in range(off, off+cnt)] for off, cnt, _, _ in poly_infos[1:]]
            for t in _cdt_triangulate(bpts, b_hpts if b_hpts else None):
                mesh.add_face([bvk[t[0]], bvk[t[2]], bvk[t[1]]])
            tpts = [Point(*proj(all_top[i]), 0.0) for i in range(top_n0)]
            t_hpts = [[Point(*proj(all_top[i]), 0.0) for i in range(off, off+cnt)] for _, _, off, cnt in poly_infos[1:]]
            for t in _cdt_triangulate(tpts, t_hpts if t_hpts else None):
                mesh.add_face([tvk[t[0]], tvk[t[1]], tvk[t[2]]])
        def side_faces(bot_off, bot_n, top_off, top_n, bpts, tpts):
            def edsq(pts, i):
                j = (i + 1) % len(pts)
                dx = pts[j].x - pts[i].x; dy = pts[j].y - pts[i].y; dz = pts[j].z - pts[i].z
                return dx*dx + dy*dy + dz*dz
            ia = max(range(bot_n), key=lambda i: edsq(bpts, i))
            ib = max(range(top_n), key=lambda i: edsq(tpts, i))
            if bot_n == top_n:
                for k in range(bot_n):
                    cb = bot_off + (ia + k) % bot_n; ct = top_off + (ib + k) % top_n
                    nb = bot_off + (ia + k + 1) % bot_n; nt = top_off + (ib + k + 1) % top_n
                    mesh.add_face([bvk[cb], bvk[nb], tvk[nt], tvk[ct]])
                return
            b_arcs = [0.0] * (bot_n + 1)
            for k in range(bot_n):
                i = (ia + k) % bot_n; j = (ia + k + 1) % bot_n
                dx = bpts[j].x - bpts[i].x; dy = bpts[j].y - bpts[i].y; dz = bpts[j].z - bpts[i].z
                b_arcs[k + 1] = b_arcs[k] + math.sqrt(dx*dx + dy*dy + dz*dz)
            t_arcs = [0.0] * (top_n + 1)
            for k in range(top_n):
                i = (ib + k) % top_n; j = (ib + k + 1) % top_n
                dx = tpts[j].x - tpts[i].x; dy = tpts[j].y - tpts[i].y; dz = tpts[j].z - tpts[i].z
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
    def from_polygon_with_holes_many(inputs: List, sort_by_bbox: bool = False, parallel: bool = True) -> List["Mesh"]:
        if parallel and len(inputs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as ex:
                return list(ex.map(lambda x: Mesh.from_polygon_with_holes(x, sort_by_bbox), inputs))
        return [Mesh.from_polygon_with_holes(x, sort_by_bbox) for x in inputs]

    @staticmethod
    def loft_many(pairs: List, cap: bool = True, parallel: bool = True) -> List["Mesh"]:
        if parallel and len(pairs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as ex:
                return list(ex.map(lambda p: Mesh.loft(p[0], p[1], cap), pairs))
        return [Mesh.loft(p[0], p[1], cap) for p in pairs]

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
        for u, nbrs in self.halfedge.items():
            for v, fkey in nbrs.items():
                if fkey is None:
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
        return any(self.is_edge_on_boundary(u, v) for u, v in self.face_edges(face_key))

    ###########################################################################################
    # Basic Queries
    ###########################################################################################

    def number_of_vertices(self) -> int:
        """Get the number of vertices."""
        return len(self.vertex)

    def number_of_faces(self) -> int:
        """Get the number of faces."""
        return len(self.face)

    def number_of_edges(self) -> int:
        """Get the number of edges."""
        seen = set()
        count = 0
        for u in self.halfedge:
            for v in self.halfedge[u]:
                edge = tuple(sorted([u, v]))
                if edge not in seen:
                    seen.add(edge)
                    count += 1
        return count

    def edges(self):
        result = []
        for u in sorted(self.halfedge.keys()):
            for v in sorted(self.halfedge[u].keys()):
                if self.halfedge[u][v] is None:
                    result.append((u, v))
        return result

    def euler(self) -> int:
        """Calculate Euler characteristic (V - E + F)."""
        return (
            self.number_of_vertices() - self.number_of_edges() + self.number_of_faces()
        )

    def clear(self):
        """Clear all mesh data."""
        self.halfedge.clear()
        self.vertex.clear()
        self.face.clear()
        self.facedata.clear()
        self.edgedata.clear()
        self.triangulation.clear()
        self._max_vertex = 0
        self._max_face = 0
        self._pointcolors.clear()
        self._facecolors.clear()
        self._linecolors.clear()
        self._widths.clear()
        self._objectcolor = Color.white()
        self.color_mode = ColorMode.OBJECTCOLOR

    def set_pointcolors(self, colors):
        self._pointcolors = list(colors)
        self.color_mode = ColorMode.POINTCOLORS

    def set_facecolors(self, colors):
        self._facecolors = list(colors)
        self.color_mode = ColorMode.FACECOLORS

    def set_linecolors(self, colors, widths=None):
        self._linecolors = list(colors)
        if widths is not None:
            self._widths = list(widths)

    def set_objectcolor(self, color):
        self._objectcolor = color

    @property
    def pointcolors(self): return self._pointcolors
    @property
    def facecolors(self): return self._facecolors
    @property
    def linecolors(self): return self._linecolors
    def get_pointcolors(self): return self._pointcolors
    def get_facecolors(self): return self._facecolors
    def get_linecolors(self): return self._linecolors
    @property
    def widths(self): return self._widths
    @property
    def objectcolor(self): return self._objectcolor

    def clear_pointcolors(self):
        self._pointcolors.clear()
        if self.color_mode == ColorMode.POINTCOLORS:
            self.color_mode = ColorMode.OBJECTCOLOR

    def clear_facecolors(self):
        self._facecolors.clear()
        if self.color_mode == ColorMode.FACECOLORS:
            self.color_mode = ColorMode.OBJECTCOLOR

    def clear_linecolors(self):
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

        return vertex_key

    def add_face(
        self, vertices: List[int], fkey: Optional[int] = None
    ) -> Optional[int]:
        """Add a face to the mesh.

        Parameters
        ----------
        vertices : list of int
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
            self._max_face += 1
            face_key = self._max_face
        else:
            face_key = fkey
            if face_key >= self._max_face:
                self._max_face = face_key + 1

        self.face[face_key] = vertices.copy()
        self.triangulation.pop(face_key, None)
        self._facecolors.append(Color.white())

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
                self._linecolors.append(Color.white())
                self._widths.append(1.0)

        return face_key

    ###########################################################################################
    # Connectivity Queries
    ###########################################################################################

    def vertex_position(self, vertex_key: int) -> Optional[Point]:
        """Get the position of a vertex."""
        if vertex_key not in self.vertex:
            return None
        return self.vertex[vertex_key].position()

    def face_vertices(self, face_key: int) -> Optional[List[int]]:
        """Get the vertices of a face."""
        return self.face.get(face_key)

    def vertex_neighbors(self, vertex_key: int) -> List[int]:
        """Get the neighboring vertices of a vertex."""
        if vertex_key not in self.halfedge:
            return []
        return list(self.halfedge[vertex_key].keys())

    def vertex_faces(self, vertex_key: int) -> List[int]:
        """Get the faces incident to a vertex."""
        if vertex_key not in self.halfedge:
            return []
        return [f for f in self.halfedge[vertex_key].values() if f is not None]

    def vertex_edges(self, vertex_key: int) -> List[Tuple[int, int]]:
        """Get edges incident to a vertex as (vertex_key, neighbor) pairs."""
        if vertex_key not in self.halfedge:
            return []
        return [(vertex_key, u) for u in self.halfedge[vertex_key]]

    def face_edges(self, face_key: int) -> List[Tuple[int, int]]:
        """Get edges of a face as (vi, vi+1) pairs."""
        verts = self.face.get(face_key)
        if verts is None:
            return []
        n = len(verts)
        return [(verts[i], verts[(i + 1) % n]) for i in range(n)]

    def face_neighbors(self, face_key: int) -> List[int]:
        """Get faces adjacent to a face (sharing an edge)."""
        neighbors = []
        for u, v in self.face_edges(face_key):
            f = self.halfedge.get(v, {}).get(u)
            if f is not None:
                neighbors.append(f)
        return neighbors

    def edge_vertices(self, u: int, v: int) -> List[int]:
        """Get the two vertices of an edge."""
        return [u, v]

    def edge_faces(self, u: int, v: int) -> Tuple[Optional[int], Optional[int]]:
        """Get the faces on each side of an edge."""
        f0 = self.halfedge.get(u, {}).get(v)
        f1 = self.halfedge.get(v, {}).get(u)
        return (f0, f1)

    def edge_edges(self, u: int, v: int) -> List[Tuple[int, int]]:
        """Get all edges sharing a vertex with (u,v), excluding (u,v) and (v,u)."""
        edges = []
        for w in self.halfedge.get(u, {}):
            if w != v:
                edges.append((u, w))
        for w in self.halfedge.get(v, {}):
            if w != u:
                edges.append((v, w))
        return edges

    ###########################################################################################
    # Geometric Properties
    ###########################################################################################

    def face_normal(self, face_key: int) -> Optional[Vector]:
        """Calculate the normal of a face."""
        vertices = self.face_vertices(face_key)
        if vertices is None or len(vertices) < 3:
            return None

        p0 = self.vertex_position(vertices[0])
        p1 = self.vertex_position(vertices[1])
        p2 = self.vertex_position(vertices[2])

        if p0 is None or p1 is None or p2 is None:
            return None

        u = Vector(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
        v = Vector(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z)

        normal = u.cross(v)
        length = normal.magnitude()

        if length > Tolerance.ZERO_TOLERANCE:
            return Vector(normal[0] / length, normal[1] / length, normal[2] / length)

        return None

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

    def face_area(self, face_key: int) -> Optional[float]:
        """Calculate the area of a face."""
        vertices = self.face_vertices(face_key)
        if vertices is None or len(vertices) < 3:
            return 0.0

        area = 0.0
        p0 = self.vertex_position(vertices[0])
        if p0 is None:
            return None

        for i in range(1, len(vertices) - 1):
            p1 = self.vertex_position(vertices[i])
            p2 = self.vertex_position(vertices[i + 1])
            if p1 is None or p2 is None:
                return None

            u = Vector(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
            v = Vector(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z)

            area += u.cross(v).magnitude() * 0.5

        return area

    def vertex_angle_in_face(self, vertex_key: int, face_key: int) -> Optional[float]:
        """Calculate the angle at a vertex in a face."""
        vertices = self.face_vertices(face_key)
        if vertices is None or vertex_key not in vertices:
            return None

        vertex_index = vertices.index(vertex_key)
        n = len(vertices)
        prev_vertex = vertices[(vertex_index - 1) % n]
        next_vertex = vertices[(vertex_index + 1) % n]

        center = self.vertex_position(vertex_key)
        prev_pos = self.vertex_position(prev_vertex)
        next_pos = self.vertex_position(next_vertex)

        if center is None or prev_pos is None or next_pos is None:
            return None

        u = Vector(prev_pos.x - center.x, prev_pos.y - center.y, prev_pos.z - center.z)
        v = Vector(next_pos.x - center.x, next_pos.y - center.y, next_pos.z - center.z)

        u_len = u.magnitude()
        v_len = v.magnitude()

        if u_len < Tolerance.ZERO_TOLERANCE or v_len < Tolerance.ZERO_TOLERANCE:
            return 0.0

        cos_angle = u.dot(v) / (u_len * v_len)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.acos(cos_angle)

    def dihedral_angle(self, u: int, v: int) -> Optional[float]:
        """Calculate the dihedral angle between two faces sharing edge (u,v)."""
        f0_opt, f1_opt = self.edge_faces(u, v)
        if f0_opt is None or f1_opt is None:
            return None
        n0 = self.face_normal(f0_opt)
        n1 = self.face_normal(f1_opt)
        if n0 is None or n1 is None:
            return None
        dot = max(-1.0, min(1.0, n0[0]*n1[0] + n0[1]*n1[1] + n0[2]*n1[2]))
        return math.pi - math.acos(dot)

    def face_normals(self) -> Dict[int, Vector]:
        """Calculate normals for all faces."""
        normals = {}
        for face_key in self.face:
            normal = self.face_normal(face_key)
            if normal is not None:
                normals[face_key] = normal
        return normals

    def vertex_normals(self) -> Dict[int, Vector]:
        """Calculate normals for all vertices (area-weighted)."""
        return self.vertex_normals_weighted(NormalWeighting.AREA)

    def vertex_normals_weighted(self, weighting: NormalWeighting) -> Dict[int, Vector]:
        """Calculate normals for all vertices with specified weighting."""
        normals = {}
        for vertex_key in self.vertex:
            normal = self.vertex_normal_weighted(vertex_key, weighting)
            if normal is not None:
                normals[vertex_key] = normal
        return normals

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

    def transform(self, xf=None):
        xform = xf if xf is not None else self.xform
        for vdata in self.vertex.values():
            pos = vdata.position()
            xform.transform_point(pos)
            vdata[0] = pos.x
            vdata[1] = pos.y
            vdata[2] = pos.z

    def transformed(self, xf=None):
        import copy
        result = copy.deepcopy(self)
        result.transform(xf)
        return result

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
            "default_edge_attributes": self.default_edge_attributes,
            "default_face_attributes": self.default_face_attributes,
            "default_vertex_attributes": self.default_vertex_attributes,
            "edgedata": edgedata_json,
            "face": face_data,
            "facecolors": facecolors_flat,
            "facedata": facedata_json,
            "guid": self.guid,
            "halfedge": halfedge_data,
            "linecolors": linecolors_flat,
            "max_face": self._max_face,
            "max_vertex": self._max_vertex,
            "name": self.name,
            "objectcolor": self._objectcolor.__jsondump__(),
            "color_mode": self.color_mode.value,
            "pointcolors": pointcolors_flat,
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

        if "xform" in data:
            mesh.xform = decode_node(data["xform"])

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

    def json_dump(self, filepath):
        """Write JSON to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        """Read JSON from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def json_dumps(self):
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, json_string):
        """Load from JSON string."""
        import json
        return cls.__jsonload__(json.loads(json_string))

    ###########################################################################################
    # Protobuf
    ###########################################################################################

    def pb_dumps(self):
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
        proto.objectcolor.guid = self._objectcolor.guid
        proto.objectcolor.name = self._objectcolor.name
        proto.objectcolor.r = self._objectcolor[0]
        proto.objectcolor.g = self._objectcolor[1]
        proto.objectcolor.b = self._objectcolor[2]
        proto.objectcolor.a = self._objectcolor[3]
        _cm_map = {"objectcolor": 0, "pointcolors": 1, "facecolors": 2, "none": 3}
        proto.color_mode = _cm_map.get(self.color_mode.value, 0)

        # Xform
        from .proto import xform_pb2
        proto.xform.guid = self.xform.guid
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        """Create Mesh from protobuf binary data."""
        from .proto import mesh_pb2
        from .color import Color
        from .xform import Xform

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

        # Xform
        mesh.xform = Xform()
        mesh.xform.guid = proto.xform.guid
        mesh.xform.name = proto.xform.name
        mesh.xform.m = list(proto.xform.matrix)

        # Update max counters
        if mesh.vertex:
            mesh._max_vertex = max(mesh.vertex.keys()) + 1
        if mesh.face:
            mesh._max_face = max(mesh.face.keys()) + 1

        return mesh

    def pb_dump(self, filepath):
        """Write protobuf to file."""
        data = self.pb_dumps()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath):
        """Read protobuf from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)

    ###########################################################################################
    # Color and Width Management
    ###########################################################################################

    def set_vertex_color(self, index: int, color: Color):
        """Set color for a specific vertex."""
        if 0 <= index < len(self._pointcolors):
            self._pointcolors[index] = color

    def set_face_color(self, index: int, color: Color):
        """Set color for a specific face."""
        if 0 <= index < len(self._facecolors):
            self._facecolors[index] = color

    def set_edge_color(self, index: int, color: Color):
        """Set color for a specific edge."""
        if 0 <= index < len(self._linecolors):
            self._linecolors[index] = color

    def set_edge_width(self, index: int, width: float):
        """Set width for a specific edge."""
        if 0 <= index < len(self._widths):
            self._widths[index] = width
