import uuid
import copy
import json

from .closest import Closest
from .nurbssurface import NurbsSurface
from .nurbscurve import NurbsCurve
from .primitives import Primitives
from .xform import Xform
from .color import Color


# ---- Bowyer-Watson Constrained Delaunay Triangulation in 2D UV space ----

class _Vertex2D:
    __slots__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y


class _Triangle:
    __slots__ = ('v', 'adj', 'constrained', 'alive')

    def __init__(self, v0, v1, v2):
        self.v = [v0, v1, v2]
        self.adj = [-1, -1, -1]
        self.constrained = [False, False, False]
        self.alive = True


class _Delaunay2D:
    def __init__(self, xmin, ymin, xmax, ymax):
        dx = xmax - xmin
        dy = ymax - ymin
        d = dx if dx > dy else dy
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        scale = 20.0
        self.vertices = []
        self.triangles = []
        self.edge_map = {}
        self.last_found = 0
        self.vertices.append(_Vertex2D(cx - scale*d, cy - scale*d))
        self.vertices.append(_Vertex2D(cx + scale*d, cy - scale*d))
        self.vertices.append(_Vertex2D(cx, cy + scale*d))
        self.super_v = [0, 1, 2]
        self.triangles.append(_Triangle(0, 1, 2))
        self._register_edges(0)

    @staticmethod
    def _edge_key(a, b):
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _in_circumcircle(ax, ay, bx, by, cx, cy, dx, dy):
        adx = ax - dx; ady = ay - dy
        bdx = bx - dx; bdy = by - dy
        cdx = cx - dx; cdy = cy - dy
        return ((adx*adx + ady*ady) * (bdx*cdy - cdx*bdy)
              + (bdx*bdx + bdy*bdy) * (cdx*ady - adx*cdy)
              + (cdx*cdx + cdy*cdy) * (adx*bdy - bdx*ady))

    @staticmethod
    def _orient2d(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def _register_edges(self, ti):
        for k in range(3):
            a = self.triangles[ti].v[(k+1) % 3]
            b = self.triangles[ti].v[(k+2) % 3]
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
        for k in range(3):
            a = self.triangles[ti].v[(k+1) % 3]
            b = self.triangles[ti].v[(k+2) % 3]
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
                    adj_a = self.triangles[adj_ti].v[(adj_kk+1) % 3]
                    adj_b = self.triangles[adj_ti].v[(adj_kk+2) % 3]
                    self.edge_map[self._edge_key(adj_a, adj_b)] = (adj_ti, adj_kk)
            else:
                e = self.edge_map.get(key)
                if e is not None and e[0] == ti:
                    del self.edge_map[key]

    def _locate(self, x, y, start):
        if start < 0 or start >= len(self.triangles) or not self.triangles[start].alive:
            start = len(self.triangles) - 1
            while start >= 0 and not self.triangles[start].alive:
                start -= 1
            if start < 0:
                return -1
        cur = start
        for _ in range(len(self.triangles)):
            t = self.triangles[cur]
            moved = False
            for k in range(3):
                a = t.v[k]; b = t.v[(k+1) % 3]
                ax = self.vertices[a].x; ay = self.vertices[a].y
                bx = self.vertices[b].x; by = self.vertices[b].y
                if self._orient2d(ax, ay, bx, by, x, y) < 0:
                    opp = t.adj[(k+2) % 3]
                    if opp >= 0 and self.triangles[opp].alive:
                        cur = opp; moved = True; break
            if not moved:
                return cur
        return cur

    def insert(self, x, y):
        start = self._locate(x, y, self.last_found)
        if start >= 0:
            t = self.triangles[start]
            for k in range(3):
                vi2 = t.v[k]
                ddx = self.vertices[vi2].x - x
                ddy = self.vertices[vi2].y - y
                if ddx*ddx + ddy*ddy < 1e-12:
                    return vi2
        vi = len(self.vertices)
        self.vertices.append(_Vertex2D(x, y))
        bad = []
        visited = set()
        if start >= 0:
            bad.append(start); visited.add(start)
        bfs_front = 0
        while bfs_front < len(bad):
            ti = bad[bfs_front]; bfs_front += 1
            if not self.triangles[ti].alive:
                continue
            v0, v1, v2 = self.triangles[ti].v
            ax = self.vertices[v0].x; ay = self.vertices[v0].y
            bx = self.vertices[v1].x; by = self.vertices[v1].y
            cx = self.vertices[v2].x; cy = self.vertices[v2].y
            o = self._orient2d(ax, ay, bx, by, cx, cy)
            if o > 0:
                ic = self._in_circumcircle(ax, ay, bx, by, cx, cy, x, y)
            else:
                ic = self._in_circumcircle(ax, ay, cx, cy, bx, by, x, y)
            if ic > 0:
                for k in range(3):
                    if self.triangles[ti].constrained[k]:
                        continue
                    nb = self.triangles[ti].adj[k]
                    if nb >= 0 and nb not in visited:
                        visited.add(nb); bad.append(nb)
        kept = []
        for ti in bad:
            if not self.triangles[ti].alive:
                continue
            v0, v1, v2 = self.triangles[ti].v
            ax = self.vertices[v0].x; ay = self.vertices[v0].y
            bx = self.vertices[v1].x; by = self.vertices[v1].y
            cx = self.vertices[v2].x; cy = self.vertices[v2].y
            o = self._orient2d(ax, ay, bx, by, cx, cy)
            if o > 0:
                ic = self._in_circumcircle(ax, ay, bx, by, cx, cy, x, y)
            else:
                ic = self._in_circumcircle(ax, ay, cx, cy, bx, by, x, y)
            if ic > 0:
                kept.append(ti)
        bad = kept
        if not bad:
            self.vertices.pop()
            return -1
        bad_set = set(bad)
        polygon = []
        for ti in bad:
            t = self.triangles[ti]
            for k in range(3):
                nb = t.adj[k]
                if nb < 0 or nb not in bad_set:
                    polygon.append((t.v[(k+1) % 3], t.v[(k+2) % 3], t.constrained[k]))
        for ti in bad:
            self._unregister_edges(ti)
            self.triangles[ti].alive = False
        for (e0, e1, constr) in polygon:
            o = self._orient2d(self.vertices[vi].x, self.vertices[vi].y,
                               self.vertices[e0].x, self.vertices[e0].y,
                               self.vertices[e1].x, self.vertices[e1].y)
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
        self.last_found = len(self.triangles) - 1
        return vi

    def insert_constraint(self, v0, v1):
        if v0 == v1:
            return
        for ti in range(len(self.triangles)):
            if not self.triangles[ti].alive:
                continue
            for k in range(3):
                e0 = self.triangles[ti].v[(k+1) % 3]
                e1 = self.triangles[ti].v[(k+2) % 3]
                if (e0 == v0 and e1 == v1) or (e0 == v1 and e1 == v0):
                    self.triangles[ti].constrained[k] = True
                    nb = self.triangles[ti].adj[k]
                    if nb >= 0 and self.triangles[nb].alive:
                        for kk in range(3):
                            if self.triangles[nb].adj[kk] == ti:
                                self.triangles[nb].constrained[kk] = True
                                break
                    return
        start_ti = -1
        for i in range(len(self.triangles) - 1, -1, -1):
            if not self.triangles[i].alive:
                continue
            for k in range(3):
                if self.triangles[i].v[k] == v0:
                    start_ti = i
                    break
        if start_ti < 0:
            return
        ax = self.vertices[v0].x; ay = self.vertices[v0].y
        bx = self.vertices[v1].x; by = self.vertices[v1].y
        ivl = -1; ivr = -1; it = -1
        ti = start_ti
        guard = len(self.triangles) + 4
        g = 0
        while True:
            if g > guard or not self.triangles[ti].alive:
                break
            g += 1
            k_v0 = -1
            for i in range(3):
                if self.triangles[ti].v[i] == v0:
                    k_v0 = i
                    break
            if k_v0 < 0:
                break
            k = k_v0
            ip2 = self.triangles[ti].v[(k+1) % 3]
            ip1 = self.triangles[ti].v[(k+2) % 3]
            op2 = self._orient2d(ax, ay, bx, by, self.vertices[ip2].x, self.vertices[ip2].y)
            op1 = self._orient2d(ax, ay, bx, by, self.vertices[ip1].x, self.vertices[ip1].y)
            if op2 < 0 and op1 >= 0:
                ivl = ip1; ivr = ip2; it = ti
                break
            nxt = self.triangles[ti].adj[(k+1) % 3]
            if nxt < 0 or not self.triangles[nxt].alive:
                break
            if nxt == start_ti:
                break
            ti = nxt
        if it < 0:
            return
        poly_l = [v0, ivl]
        poly_r = [v0, ivr]
        intersected = [it]
        iv = v0
        cur_it = it
        guard = len(self.triangles) * 2 + 8
        g = 0

        def tri_has(ti, v):
            return self.triangles[ti].v[0] == v or self.triangles[ti].v[1] == v or self.triangles[ti].v[2] == v

        while not tri_has(cur_it, v1) and g < guard:
            g += 1
            k_iv = -1
            for i in range(3):
                if self.triangles[cur_it].v[i] == iv:
                    k_iv = i
                    break
            if k_iv < 0:
                break
            i_topo = self.triangles[cur_it].adj[k_iv]
            if i_topo < 0 or not self.triangles[i_topo].alive:
                break
            i_vopo = -1
            for k in range(3):
                if self.triangles[i_topo].adj[k] == cur_it:
                    i_vopo = self.triangles[i_topo].v[k]
                    break
            if i_vopo < 0:
                break
            o = self._orient2d(ax, ay, bx, by, self.vertices[i_vopo].x, self.vertices[i_vopo].y)
            if o < 0:
                if i_vopo != v1:
                    poly_r.append(i_vopo)
                iv = ivr; ivr = i_vopo
            else:
                if i_vopo != v1:
                    poly_l.append(i_vopo)
                iv = ivl; ivl = i_vopo
            intersected.append(i_topo)
            cur_it = i_topo
        poly_l.append(v1)
        poly_r.append(v1)
        first_new = len(self.triangles)
        for ti in intersected:
            self._unregister_edges(ti)
            self.triangles[ti].alive = False
        new_tris = []
        apex = v1
        for i in range(max(len(poly_l) - 2, 0)):
            new_tris.append((apex, poly_l[i+1], poly_l[i]))
        apex = v0
        for i in range(1, max(len(poly_r) - 1, 1)):
            new_tris.append((apex, poly_r[i], poly_r[i+1]))
        for (pa, pb, pc) in new_tris:
            o = self._orient2d(self.vertices[pa].x, self.vertices[pa].y,
                               self.vertices[pb].x, self.vertices[pb].y,
                               self.vertices[pc].x, self.vertices[pc].y)
            if abs(o) < 1e-20:
                continue
            new_ti = len(self.triangles)
            if o > 0:
                vb, vc = pb, pc
            else:
                vb, vc = pc, pb
            self.triangles.append(_Triangle(pa, vb, vc))
            self._register_edges(new_ti)
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
        for ti in range(len(self.triangles)):
            if not self.triangles[ti].alive:
                continue
            for k in range(3):
                e0 = self.triangles[ti].v[(k+1) % 3]
                e1 = self.triangles[ti].v[(k+2) % 3]
                if (e0 == v0 and e1 == v1) or (e0 == v1 and e1 == v0):
                    self.triangles[ti].constrained[k] = True

    def cleanup(self):
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

    def get_triangles(self):
        result = []
        for t in self.triangles:
            if not t.alive:
                continue
            a, b, c = t.v[0], t.v[1], t.v[2]
            o = self._orient2d(self.vertices[a].x, self.vertices[a].y,
                               self.vertices[b].x, self.vertices[b].y,
                               self.vertices[c].x, self.vertices[c].y)
            if o > 0:
                result.append((a, b, c))
            else:
                result.append((a, c, b))
        return result


class NurbsSurfaceTrimmed:

    def __init__(self):
        self._guid = None
        self.name = "my_nurbssurface_trimmed"
        self.width = 1.0
        self._surfacecolor = None
        self._xform = None
        self.m_surface = NurbsSurface()
        self.m_outer_loop = NurbsCurve()
        self.m_inner_loops = []

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

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

    @staticmethod
    def create(surface, outer_loop):
        ts = NurbsSurfaceTrimmed()
        ts.m_surface = surface.duplicate()
        ts.m_outer_loop = outer_loop.duplicate()
        return ts

    @staticmethod
    def create_planar(boundary):
        from .point import Point
        from .vector import Vector
        srf = Primitives.create_planar(boundary)
        if not srf.is_valid():
            return NurbsSurfaceTrimmed()
        p00 = srf.get_cv(0, 0)
        p10 = srf.get_cv(1, 0)
        p01 = srf.get_cv(0, 1)
        u_axis = Vector(p10[0]-p00[0], p10[1]-p00[1], p10[2]-p00[2])
        v_axis = Vector(p01[0]-p00[0], p01[1]-p00[1], p01[2]-p00[2])
        u_len2 = u_axis[0]**2 + u_axis[1]**2 + u_axis[2]**2
        v_len2 = v_axis[0]**2 + v_axis[1]**2 + v_axis[2]**2
        if u_len2 < 1e-28 or v_len2 < 1e-28:
            return NurbsSurfaceTrimmed()
        def project_to_uv(pt):
            dx, dy, dz = pt[0]-p00[0], pt[1]-p00[1], pt[2]-p00[2]
            nu = (dx*u_axis[0] + dy*u_axis[1] + dz*u_axis[2]) / u_len2
            nv = (dx*v_axis[0] + dy*v_axis[1] + dz*v_axis[2]) / v_len2
            return Point(nu, nv, 0.0)
        uv_pts = []
        if boundary.degree() <= 1:
            for i in range(boundary.cv_count()):
                uv_pts.append(project_to_uv(boundary.get_cv(i)))
        else:
            spans = boundary.get_span_vector()
            for si in range(len(spans) - 1):
                n_sub = 10
                for k in range(n_sub + 1):
                    t = spans[si] + (spans[si+1] - spans[si]) * k / n_sub
                    uv = project_to_uv(boundary.point_at(t))
                    if not uv_pts or (uv[0]-uv_pts[-1][0])**2 + (uv[1]-uv_pts[-1][1])**2 > 1e-24:
                        uv_pts.append(uv)
        ts = NurbsSurfaceTrimmed()
        ts.m_surface = srf
        if len(uv_pts) >= 3:
            ts.m_outer_loop = NurbsCurve.create(False, 1, uv_pts)
        return ts

    def surface(self):
        return self.m_surface

    def get_outer_loop(self):
        return self.m_outer_loop

    def set_outer_loop(self, loop):
        self.m_outer_loop = loop

    def is_trimmed(self):
        return self.m_outer_loop.is_valid()

    def is_valid(self):
        return self.m_surface.is_valid()

    def add_inner_loop(self, loop_2d):
        self.m_inner_loops.append(loop_2d)

    def add_hole(self, curve_3d):
        from .point import Point
        from .nurbscurve import NurbsCurve
        dom = curve_3d.domain()
        sdom_u = self.m_surface.domain(0)
        sdom_v = self.m_surface.domain(1)
        range_u = sdom_u[1] - sdom_u[0]
        range_v = sdom_v[1] - sdom_v[0]
        n_samples = max(curve_3d.cv_count() * 4, 32)
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

    def add_holes(self, curves_3d):
        for crv in curves_3d:
            self.add_hole(crv)

    def get_inner_loop(self, index):
        if 0 <= index < len(self.m_inner_loops):
            return self.m_inner_loops[index]
        return None

    def inner_loop_count(self):
        return len(self.m_inner_loops)

    def clear_inner_loops(self):
        self.m_inner_loops.clear()

    def point_at(self, u, v):
        return self.m_surface.point_at(u, v)

    def normal_at(self, u, v):
        return self.m_surface.normal_at(u, v)

    def mesh(self):
        return self.mesh_q(20.0, 0.005)

    # Unified trimmed-surface tessellation (BRepMesh / OpenNURBS model):
    #   1. Discretize each UV trim loop into a polygon, adaptively refining segments whose
    #      lifted 3D midpoint deviates from the chord — so the boundary follows the surface.
    #   2. Constrained Delaunay of the UV domain with the trim loops as boundary constraints.
    #   3. Refine the interior by surface DEFLECTION: repeatedly split any triangle whose surface
    #      point at its UV centroid lies farther than the deflection tolerance from the triangle's
    #      3D plane (or whose corner normals turn more than max_angle_deg). New points go in at the
    #      centroid only when it is inside the trim region, keeping the CDT boundary-conforming and
    #      producing graded, well-shaped triangles — dense where curved, coarse where flat.
    #   4. Delete exterior triangles, lift to 3D, set per-vertex analytic normals.
    # Planar surfaces need no interior refinement (deflection is ~0), so step 3 exits immediately
    # and the same code path yields the minimal boundary triangulation.
    def mesh_q(self, max_angle_deg, chord_factor):
        import math
        from .mesh import Mesh
        from .point import Point
        if not self.is_trimmed():
            return self.m_surface.mesh()

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
        bbox_diag = math.sqrt(sum((bmax[k]-bmin[k])**2 for k in range(3)))
        if bbox_diag < 1e-12:
            bbox_diag = 1.0
        deflection = bbox_diag * chord_factor
        cos_max_angle = math.cos(min(max(max_angle_deg, 0.1), 179.0) * math.pi / 180.0)

        def eval3(u, v):
            p = self.m_surface.point_at(u, v)
            return (p[0], p[1], p[2])

        # ---- 1. Adaptive trim-wire discretization in UV ----
        def disc_loop(crv):
            if crv.degree() <= 1 and not crv.is_rational():
                raw = [(crv.get_cv(i)[0], crv.get_cv(i)[1]) for i in range(crv.cv_count())]
            else:
                n = max(crv.cv_count() * 4, 16)
                sampled, _ = crv.divide_by_count(n)
                raw = [(p[0], p[1]) for p in sampled]
            while len(raw) > 1:
                dx = raw[0][0] - raw[-1][0]
                dy = raw[0][1] - raw[-1][1]
                if dx*dx + dy*dy < 1e-20:
                    raw.pop()
                else:
                    break
            if len(raw) < 2:
                return raw
            out = []
            m = len(raw)
            for i in range(m):
                a = raw[i]
                b = raw[(i+1) % m]
                stack = [(a, b, 0)]
                while stack:
                    sa, sb, depth = stack.pop()
                    mu = (sa[0]+sb[0]) * 0.5
                    mv = (sa[1]+sb[1]) * 0.5
                    pa = eval3(sa[0], sa[1])
                    pb = eval3(sb[0], sb[1])
                    pm = eval3(mu, mv)
                    ex = pb[0]-pa[0]; ey = pb[1]-pa[1]; ez = pb[2]-pa[2]
                    l2 = ex*ex + ey*ey + ez*ez
                    if l2 > 1e-30:
                        t = ((pm[0]-pa[0])*ex + (pm[1]-pa[1])*ey + (pm[2]-pa[2])*ez) / l2
                        cx = pa[0]+t*ex; cy = pa[1]+t*ey; cz = pa[2]+t*ez
                        dev = math.sqrt((pm[0]-cx)**2 + (pm[1]-cy)**2 + (pm[2]-cz)**2)
                    else:
                        dev = 0.0
                    if dev > deflection and depth < 6:
                        stack.append(((mu, mv), sb, depth+1))
                        stack.append((sa, (mu, mv), depth+1))
                    else:
                        out.append(sa)
            return out

        outer_uv = disc_loop(self.m_outer_loop)
        if len(outer_uv) < 3:
            return self.m_surface.mesh()
        hole_uvs = [disc_loop(inner) for inner in self.m_inner_loops]

        bb_umin = min(p[0] for p in outer_uv)
        bb_umax = max(p[0] for p in outer_uv)
        bb_vmin = min(p[1] for p in outer_uv)
        bb_vmax = max(p[1] for p in outer_uv)

        def point_in_polygon(u, v, poly):
            n = len(poly)
            if n < 3:
                return False
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > v) != (yj > v)) and (u < (xj-xi)*(v-yi)/(yj-yi)+xi):
                    inside = not inside
                j = i
            return inside

        def inside_trim(u, v):
            if not point_in_polygon(u, v, outer_uv):
                return False
            for h in hole_uvs:
                if point_in_polygon(u, v, h):
                    return False
            return True

        # ---- 2. Constrained Delaunay of the trim wire ----
        dt = _Delaunay2D(bb_umin, bb_vmin, bb_umax, bb_vmax)
        vis = [dt.insert(p[0], p[1]) for p in outer_uv]
        for i in range(len(vis)):
            j = (i + 1) % len(vis)
            if vis[i] >= 0 and vis[j] >= 0 and vis[i] != vis[j]:
                dt.insert_constraint(vis[i], vis[j])
        for h in hole_uvs:
            vis = [dt.insert(p[0], p[1]) for p in h]
            for i in range(len(vis)):
                j = (i + 1) % len(vis)
                if vis[i] >= 0 and vis[j] >= 0 and vis[i] != vis[j]:
                    dt.insert_constraint(vis[i], vis[j])

        # ---- 3. Interior refinement by surface deflection ----
        MAX_ITERS = 8
        MAX_VERTS = 200000
        for _iter in range(MAX_ITERS):
            to_insert = []
            for tri in dt.triangles:
                if not tri.alive:
                    continue
                a = dt.vertices[tri.v[0]]
                b = dt.vertices[tri.v[1]]
                c = dt.vertices[tri.v[2]]
                cu = (a.x + b.x + c.x) / 3.0
                cv = (a.y + b.y + c.y) / 3.0
                if not inside_trim(cu, cv):
                    continue
                pa = eval3(a.x, a.y)
                pb = eval3(b.x, b.y)
                pc = eval3(c.x, c.y)
                pm = eval3(cu, cv)
                ux = pb[0]-pa[0]; uy = pb[1]-pa[1]; uz = pb[2]-pa[2]
                vx = pc[0]-pa[0]; vy = pc[1]-pa[1]; vz = pc[2]-pa[2]
                nx = uy*vz - uz*vy
                ny = uz*vx - ux*vz
                nz = ux*vy - uy*vx
                nl = math.sqrt(nx*nx + ny*ny + nz*nz)
                if nl < 1e-30:
                    continue
                dev = abs(((pm[0]-pa[0])*nx + (pm[1]-pa[1])*ny + (pm[2]-pa[2])*nz) / nl)
                refine = dev > deflection
                if not refine:
                    na = self.m_surface.normal_at(a.x, a.y)
                    nb = self.m_surface.normal_at(b.x, b.y)
                    nc2 = self.m_surface.normal_at(c.x, c.y)
                    d1 = na[0]*nb[0] + na[1]*nb[1] + na[2]*nb[2]
                    d2 = nb[0]*nc2[0] + nb[1]*nc2[1] + nb[2]*nc2[2]
                    d3 = na[0]*nc2[0] + na[1]*nc2[1] + na[2]*nc2[2]
                    mind = min(d1, d2, d3)
                    if mind < cos_max_angle:
                        refine = True
                if refine:
                    to_insert.append((cu, cv))
            if not to_insert:
                break
            for (cu, cv) in to_insert:
                if len(dt.vertices) >= MAX_VERTS:
                    break
                dt.insert(cu, cv)
            if len(dt.vertices) >= MAX_VERTS:
                break

        # ---- 4. Trim, lift, normals ----
        dt.cleanup()
        for ti in range(len(dt.triangles)):
            if not dt.triangles[ti].alive:
                continue
            v0, v1, v2 = dt.triangles[ti].v
            cu = (dt.vertices[v0].x + dt.vertices[v1].x + dt.vertices[v2].x) / 3.0
            cv = (dt.vertices[v0].y + dt.vertices[v1].y + dt.vertices[v2].y) / 3.0
            if not inside_trim(cu, cv):
                dt.triangles[ti].alive = False

        tris = dt.get_triangles()
        if not tris:
            return self.m_surface.mesh()
        result = Mesh()
        nverts = len(dt.vertices)
        vert_map = [None] * nverts
        # Lift to 3D, welding coincident vertices so a closed/periodic surface (cylinder,
        # cone, torus, sphere) stitches at its seam: distinct UV columns u0 and u1 (or rows
        # v0/v1) evaluate to the SAME 3D point, so they must share one mesh vertex. Spatial
        # hash on a weld-tolerance grid; new points scan the 3x3x3 neighbour cells.
        weld_tol = bbox_diag * 1e-5
        cell = weld_tol if weld_tol > 0.0 else 1.0
        cell_map = {}

        def weld_vertex(x, y, z):
            ci = int(math.floor(x / cell)); cj = int(math.floor(y / cell)); ck = int(math.floor(z / cell))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        bucket = cell_map.get((ci+di, cj+dj, ck+dk))
                        if bucket is None:
                            continue
                        for (wx, wy, wz, wvk) in bucket:
                            if (wx-x)**2 + (wy-y)**2 + (wz-z)**2 <= weld_tol*weld_tol:
                                return wvk
            vk = result.add_vertex(Point(x, y, z))
            cell_map.setdefault((ci, cj, ck), []).append((x, y, z, vk))
            return vk

        for (a, b, c) in tris:
            for vi in (a, b, c):
                if vert_map[vi] is None:
                    p3d = self.m_surface.point_at(dt.vertices[vi].x, dt.vertices[vi].y)
                    vert_map[vi] = weld_vertex(p3d[0], p3d[1], p3d[2])
        for (a, b, c) in tris:
            v0 = vert_map[a]
            v1 = vert_map[b]
            v2 = vert_map[c]
            if v0 == v1 or v1 == v2 or v2 == v0:
                continue
            result.add_face([v0, v1, v2])
        for vi in range(nverts):
            if vert_map[vi] is not None:
                nrm = self.m_surface.normal_at(dt.vertices[vi].x, dt.vertices[vi].y)
                result.vertex[vert_map[vi]].set_normal(nrm[0], nrm[1], nrm[2])
        return result

    def mesh_by_plane(self, q0, normal, max_angle_deg, chord_factor):
        # Mesh the surface trimmed by a plane (q0, normal), keeping (S-q0).n <= 0. OCCT path-A:
        # span-adaptive UV grid + marching-squares clip at f(u,v)=(S-q0).n, every boundary
        # crossing Newton-refined onto the plane, coincident-3D vertices welded so periodic
        # seams (cylinder/torus/sphere) close watertight.
        import math
        from collections import defaultdict
        from .mesh import Mesh
        srf = self.m_surface
        nx, ny, nz = normal[0], normal[1], normal[2]
        nl = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nl < 1e-12:
            return srf.mesh()
        nx, ny, nz = nx/nl, ny/nl, nz/nl
        qx, qy, qz = q0[0], q0[1], q0[2]

        def e3(u, v):
            p = srf.point_at(u, v)
            return (p[0], p[1], p[2])

        def field(u, v):
            p = e3(u, v)
            return (p[0]-qx)*nx + (p[1]-qy)*ny + (p[2]-qz)*nz

        def refine(u, v):
            for _ in range(12):
                fv = field(u, v)
                if abs(fv) < 1e-9:
                    break
                h = 1e-4
                a = e3(u+h, v); b = e3(u-h, v); c = e3(u, v+h); d = e3(u, v-h)
                gu = ((a[0]-b[0])*nx + (a[1]-b[1])*ny + (a[2]-b[2])*nz) / (2*h)
                gv = ((c[0]-d[0])*nx + (c[1]-d[1])*ny + (c[2]-d[2])*nz) / (2*h)
                g2 = gu*gu + gv*gv
                if g2 < 1e-20:
                    break
                u -= fv*gu/g2; v -= fv*gv/g2
            return u, v

        usp = srf.get_span_vector(0)
        vsp = srf.get_span_vector(1)
        if len(usp) < 2 or len(vsp) < 2:
            return srf.mesh()
        deg_u = srf.degree(0); deg_v = srf.degree(1)
        bmin = [1e30]*3; bmax = [-1e30]*3
        for i in range(srf.cv_count(0)):
            for j in range(srf.cv_count(1)):
                p = srf.get_cv(i, j)
                for k in range(3):
                    bmin[k] = min(bmin[k], p[k]); bmax[k] = max(bmax[k], p[k])
        diag = math.sqrt(sum((bmax[k]-bmin[k])**2 for k in range(3))) or 1.0
        ctol = diag * chord_factor

        def span_subs(dr, sp, osp, deg):
            n = len(sp) - 1
            out = [2 if deg > 1 else 1] * n
            smid = (osp[0] + osp[-1]) * 0.5
            for i in range(n):
                t0, t1 = sp[i], sp[i+1]
                if deg > 1:
                    ma = 0.0; pn = None
                    for k in range(5):
                        t = t0 + k*(t1-t0)/4
                        nm = srf.normal_at(t, smid) if dr == 0 else srf.normal_at(smid, t)
                        if pn is not None:
                            dpd = max(-1.0, min(1.0, pn[0]*nm[0]+pn[1]*nm[1]+pn[2]*nm[2]))
                            ma += math.acos(dpd) * 180.0 / math.pi
                        pn = (nm[0], nm[1], nm[2])
                    out[i] = max(out[i], max(1, min(int(math.ceil(ma/max_angle_deg)), 64)))
                p0 = e3(t0, smid) if dr == 0 else e3(smid, t0)
                p1 = e3(t1, smid) if dr == 0 else e3(smid, t1)
                dev = 0.0
                for k in range(1, 4):
                    fr = k/4; tm = t0 + fr*(t1-t0)
                    pm = e3(tm, smid) if dr == 0 else e3(smid, tm)
                    lx, ly, lz = (p0[0]+fr*(p1[0]-p0[0]), p0[1]+fr*(p1[1]-p0[1]), p0[2]+fr*(p1[2]-p0[2]))
                    dev = max(dev, math.sqrt((pm[0]-lx)**2 + (pm[1]-ly)**2 + (pm[2]-lz)**2))
                if dev > ctol:
                    out[i] = max(out[i], min(int(math.ceil(math.sqrt(dev/ctol))), 64))
            return out

        us_subs = span_subs(0, usp, vsp, deg_u)
        vs_subs = span_subs(1, vsp, usp, deg_v)
        us = []
        for i in range(len(usp)-1):
            for s in range(us_subs[i]):
                us.append(usp[i] + s*(usp[i+1]-usp[i])/us_subs[i])
        us.append(usp[-1])
        vs = []
        for i in range(len(vsp)-1):
            for s in range(vs_subs[i]):
                vs.append(vsp[i] + s*(vsp[i+1]-vsp[i])/vs_subs[i])
        vs.append(vsp[-1])
        nu, nv = len(us), len(vs)
        if nu < 2 or nv < 2:
            return srf.mesh()
        F = [[field(us[i], vs[j]) for j in range(nv)] for i in range(nu)]
        result = Mesh()
        wt = diag * 1e-5; cell = wt or 1.0
        cmap = defaultdict(list)

        def weld(u, v):
            P = srf.point_at(u, v)
            x, y, z = P[0], P[1], P[2]
            ci, cj, ck = math.floor(x/cell), math.floor(y/cell), math.floor(z/cell)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        for (px, py, pz, vk) in cmap[(ci+di, cj+dj, ck+dk)]:
                            if (px-x)**2 + (py-y)**2 + (pz-z)**2 <= wt*wt:
                                return vk
            vk = result.add_vertex(P)
            nm = srf.normal_at(u, v)
            result.vertex[vk].set_normal(nm[0], nm[1], nm[2])
            cmap[(ci, cj, ck)].append((x, y, z, vk))
            return vk

        def cross(ua, va, ub, vb):
            fa = field(ua, va); fb = field(ub, vb)
            t = fa/(fa-fb) if abs(fa-fb) > 1e-30 else 0.5
            u, v = refine(ua + (ub-ua)*t, va + (vb-va)*t)
            return weld(u, v)

        for i in range(nu-1):
            for j in range(nv-1):
                cu = [us[i], us[i+1], us[i+1], us[i]]
                cv = [vs[j], vs[j], vs[j+1], vs[j+1]]
                inn = [F[i][j] <= 0, F[i+1][j] <= 0, F[i+1][j+1] <= 0, F[i][j+1] <= 0]
                if not any(inn):
                    continue
                poly = []
                for k in range(4):
                    kn = (k+1) % 4
                    if inn[k]:
                        poly.append(weld(cu[k], cv[k]))
                    if inn[k] != inn[kn]:
                        poly.append(cross(cu[k], cv[k], cu[kn], cv[kn]))
                for t in range(1, len(poly)-1):
                    a, b, c = poly[0], poly[t], poly[t+1]
                    if a == b or b == c or c == a:
                        continue
                    result.add_face([a, b, c])
        if not result.face:
            return srf.mesh()
        return result

    def transform(self, xf=None):
        if xf is None:
            self.m_surface.transform(self.xform)
            self.xform = Xform.identity()
        else:
            self.m_surface.transform(xf)

    def transformed(self):
        ts = self.duplicate()
        ts.transform()
        return ts

    def duplicate(self):
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def __eq__(self, other):
        if not isinstance(other, NurbsSurfaceTrimmed):
            return False
        if self.name != other.name:
            return False
        if self.width != other.width:
            return False
        if self.surfacecolor != other.surfacecolor:
            return False
        if self.xform != other.xform:
            return False
        if self.m_surface != other.m_surface:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_string(self):
        return f"NurbsSurfaceTrimmed(name={self.name}, trimmed={'true' if self.is_trimmed() else 'false'}, holes={self.inner_loop_count()})"

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return (f"NurbsSurfaceTrimmed(\n  name={self.name},\n"
                f"  trimmed={'true' if self.is_trimmed() else 'false'},\n"
                f"  holes={self.inner_loop_count()},\n"
                f"  surface={str(self.m_surface)}\n)")

    def __jsondump__(self):
        d = {
            'guid': self.guid,
            'inner_loops': [l.__jsondump__() for l in self.m_inner_loops],
            'name': self.name,
        }
        if self.m_outer_loop.is_valid():
            d['outer_loop'] = self.m_outer_loop.__jsondump__()
        d['surface'] = self.m_surface.__jsondump__()
        d['surfacecolor'] = self.surfacecolor.__jsondump__()
        d['type'] = 'NurbsSurfaceTrimmed'
        d['width'] = self.width
        d['xform'] = self.xform.__jsondump__()
        return d

    @classmethod
    def __jsonload__(cls, data):
        ts = cls()
        ts.guid = data.get('guid', ts.guid)
        ts.name = data.get('name', 'my_nurbssurface_trimmed')
        ts.width = data.get('width', 1.0)
        if 'surfacecolor' in data:
            ts.surfacecolor = Color.__jsonload__(data['surfacecolor'])
        if 'xform' in data:
            ts.xform = Xform.__jsonload__(data['xform'])
        if 'surface' in data:
            ts.m_surface = NurbsSurface.__jsonload__(data['surface'])
        if 'outer_loop' in data:
            ts.m_outer_loop = NurbsCurve.__jsonload__(data['outer_loop'])
        if 'inner_loops' in data:
            ts.m_inner_loops = [NurbsCurve.__jsonload__(l) for l in data['inner_loops']]
        return ts

    def file_json_dump(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def file_json_dumps(self):
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string):
        return cls.__jsonload__(json.loads(json_string))

    def pb_dumps(self):
        from .proto import nurbssurface_trimmed_pb2
        proto = nurbssurface_trimmed_pb2.NurbsSurfaceTrimmed()
        proto.guid = self.guid
        proto.name = self.name
        proto.width = self.width

        # Surface
        srf_data = self.m_surface.pb_dumps()
        proto.surface.ParseFromString(srf_data)

        # Outer loop
        if self.is_trimmed():
            loop_data = self.m_outer_loop.pb_dumps()
            proto.outer_loop.ParseFromString(loop_data)

        # Inner loops
        for inner in self.m_inner_loops:
            loop_data = inner.pb_dumps()
            il = proto.inner_loops.add()
            il.ParseFromString(loop_data)

        # Color
        proto.surfacecolor.name = self.surfacecolor.name
        proto.surfacecolor.r = self.surfacecolor[0]
        proto.surfacecolor.g = self.surfacecolor[1]
        proto.surfacecolor.b = self.surfacecolor[2]
        proto.surfacecolor.a = self.surfacecolor[3]

        # Transform
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        from .proto import nurbssurface_trimmed_pb2
        proto = nurbssurface_trimmed_pb2.NurbsSurfaceTrimmed()
        proto.ParseFromString(data)

        ts = cls()
        ts.guid = proto.guid
        ts.name = proto.name
        ts.width = proto.width

        # Surface
        if proto.HasField('surface'):
            srf_data = proto.surface.SerializeToString()
            ts.m_surface = NurbsSurface.pb_loads(srf_data)

        # Outer loop
        if proto.HasField('outer_loop') and proto.outer_loop.cv_count > 0:
            loop_data = proto.outer_loop.SerializeToString()
            ts.m_outer_loop = NurbsCurve.pb_loads(loop_data)

        # Inner loops
        for il in proto.inner_loops:
            loop_data = il.SerializeToString()
            ts.m_inner_loops.append(NurbsCurve.pb_loads(loop_data))

        # Color
        ts.surfacecolor = Color(
            proto.surfacecolor.r,
            proto.surfacecolor.g,
            proto.surfacecolor.b,
            proto.surfacecolor.a
        )
        ts.surfacecolor.name = proto.surfacecolor.name

        # Transform
        ts.xform = Xform()
        ts.xform.name = proto.xform.name
        ts.xform.m = list(proto.xform.matrix)

        return ts

    def pb_dump(self, filepath):
        data = self.pb_dumps()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)
