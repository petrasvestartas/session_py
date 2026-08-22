from typing import TYPE_CHECKING
import math
from bisect import bisect_left
from bisect import bisect_right

from .tolerance import PI

if TYPE_CHECKING:
    from .nurbssurface import NurbsSurface
    from .mesh import Mesh


class RemeshNurbsSurfaceAdaptive:
    def __init__(self, surface: "NurbsSurface"):
        self._surface = surface
        self._max_angle = 20.0
        self._max_edge_length = 0.0
        self._min_edge_length = 0.0
        self._max_chord_height = 0.0

    def get_max_angle(self) -> float:
        return self._max_angle

    def get_max_edge_length(self) -> float:
        return self._max_edge_length

    def get_min_edge_length(self) -> float:
        return self._min_edge_length

    def get_max_chord_height(self) -> float:
        return self._max_chord_height

    def set_max_angle(self, degrees: float) -> "RemeshNurbsSurfaceAdaptive":
        self._max_angle = degrees
        return self

    def set_max_edge_length(self, length: float) -> "RemeshNurbsSurfaceAdaptive":
        self._max_edge_length = length
        return self

    def set_min_edge_length(self, length: float) -> "RemeshNurbsSurfaceAdaptive":
        self._min_edge_length = length
        return self

    def set_max_chord_height(self, height: float) -> "RemeshNurbsSurfaceAdaptive":
        self._max_chord_height = height
        return self

    def _bbox_diagonal(self) -> float:
        s = self._surface
        minx = miny = minz = 1e30
        maxx = maxy = maxz = -1e30
        for i in range(s.cv_count(0)):
            for j in range(s.cv_count(1)):
                p = s.get_cv(i, j)
                if p[0] < minx: minx = p[0]
                if p[1] < miny: miny = p[1]
                if p[2] < minz: minz = p[2]
                if p[0] > maxx: maxx = p[0]
                if p[1] > maxy: maxy = p[1]
                if p[2] > maxz: maxz = p[2]
        dx, dy, dz = maxx - minx, maxy - miny, maxz - minz
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def mesh(self) -> "Mesh":
        from .mesh import Mesh
        from .point import Point
        import math as _math

        s = self._surface
        usp = s.get_span_vector(0)
        vsp = s.get_span_vector(1)
        ns_u = len(usp) - 1
        ns_v = len(vsp) - 1
        bbox_diag = self._bbox_diagonal()

        norm_tol = 2.0 - 2.0 * _math.cos(self._max_angle * PI / 180.0)
        chord_tol = self._max_chord_height if self._max_chord_height > 0 else bbox_diag * 0.005
        closed_u = s.is_closed(0)
        closed_v = s.is_closed(1)
        sing_v0 = s.is_singular(0)
        sing_v1 = s.is_singular(2)
        max_depth = 8

        def eval_corner(u, v):
            nx, ny, nz = 0.0, 0.0, 0.0
            pt = s.point_at(u, v)
            px, py, pz = pt[0], pt[1], pt[2]
            derivs = s.evaluate(u, v, 1)
            if len(derivs) >= 3:
                dux, duy, duz = derivs[1][0], derivs[1][1], derivs[1][2]
                dvx, dvy, dvz = derivs[2][0], derivs[2][1], derivs[2][2]
                cx = dvy*duz - dvz*duy
                cy = dvz*dux - dvx*duz
                cz = dvx*duy - dvy*dux
                ln = _math.sqrt(cx*cx + cy*cy + cz*cz)
                if ln > 1e-10:
                    nx, ny, nz = cx/ln, cy/ln, cz/ln
            return (px, py, pz, nx, ny, nz)

        def ndiff(a, b):
            dx, dy, dz = a[3]-b[3], a[4]-b[4], a[5]-b[5]
            return dx*dx + dy*dy + dz*dz

        def nvalid(c):
            return c[3]*c[3] + c[4]*c[4] + c[5]*c[5] > 1e-20

        def pdist2(a, b):
            dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
            return dx*dx + dy*dy + dz*dz

        def pmid(a, b):
            return ((a[0]+b[0])*0.5, (a[1]+b[1])*0.5, (a[2]+b[2])*0.5, 0.0, 0.0, 0.0)

        # Node: (u0, v0, u1, v1, c[0..4], ch[0..3], depth)
        # c[0]=SW, c[1]=SE, c[2]=NE, c[3]=NW, c[4]=center
        nodes = []

        def fill_child(u0, v0, u1, v1, sw, se, ne, nw, depth):
            ct = eval_corner((u0+u1)*0.5, (v0+v1)*0.5)
            nodes.append([u0, v0, u1, v1, [sw, se, ne, nw, ct], [-1,-1,-1,-1], depth])

        def subdivide(idx):
            stack = [idx]
            while stack:
                idx = stack.pop()
                nd = nodes[idx]
                if nd[6] >= max_depth:
                    continue
                u0, v0, u1, v1 = nd[0], nd[1], nd[2], nd[3]
                c = nd[4]
                um = (u0+u1)*0.5
                vm = (v0+v1)*0.5

                if self._min_edge_length > 0:
                    me = max(pdist2(c[i], c[(i+1)%4]) for i in range(4))
                    if me < self._min_edge_length**2:
                        continue

                split_u = False
                split_v = False
                if nvalid(c[0]) and nvalid(c[1]) and ndiff(c[0], c[1]) > norm_tol: split_u = True
                if nvalid(c[2]) and nvalid(c[3]) and ndiff(c[2], c[3]) > norm_tol: split_u = True
                if nvalid(c[1]) and nvalid(c[2]) and ndiff(c[1], c[2]) > norm_tol: split_v = True
                if nvalid(c[3]) and nvalid(c[0]) and ndiff(c[3], c[0]) > norm_tol: split_v = True

                s_mid = n_mid = w_mid = e_mid = None
                have_sn = have_we = False

                if not split_u:
                    s_mid = eval_corner(um, v0)
                    n_mid = eval_corner(um, v1)
                    have_sn = True
                    s_lin = pmid(c[0], c[1])
                    n_lin = pmid(c[3], c[2])
                    if pdist2(s_mid, s_lin) > chord_tol*chord_tol: split_u = True
                    if pdist2(n_mid, n_lin) > chord_tol*chord_tol: split_u = True
                    if nvalid(s_mid) and nvalid(c[0]) and ndiff(s_mid, c[0]) > norm_tol: split_u = True
                    if nvalid(n_mid) and nvalid(c[3]) and ndiff(n_mid, c[3]) > norm_tol: split_u = True

                if not split_v:
                    w_mid = eval_corner(u0, vm)
                    e_mid = eval_corner(u1, vm)
                    have_we = True
                    w_lin = pmid(c[0], c[3])
                    e_lin = pmid(c[1], c[2])
                    if pdist2(w_mid, w_lin) > chord_tol*chord_tol: split_v = True
                    if pdist2(e_mid, e_lin) > chord_tol*chord_tol: split_v = True
                    if nvalid(w_mid) and nvalid(c[0]) and ndiff(w_mid, c[0]) > norm_tol: split_v = True
                    if nvalid(e_mid) and nvalid(c[1]) and ndiff(e_mid, c[1]) > norm_tol: split_v = True

                if not split_u and not split_v:
                    degenerate = any(
                        pdist2(c[ci], c[(ci + 1) % 4]) < chord_tol * chord_tol
                        for ci in range(4)
                    )
                    if not degenerate:
                        twist_tol2 = 4.0 * chord_tol * chord_tol
                        for d in range(2):
                            mid = pmid(c[d], c[d+2])
                            if pdist2(c[4], mid) > twist_tol2:
                                split_u = True
                                split_v = True

                if self._max_edge_length > 0:
                    me2 = self._max_edge_length**2
                    if pdist2(c[0], c[1]) > me2 or pdist2(c[2], c[3]) > me2: split_u = True
                    if pdist2(c[1], c[2]) > me2 or pdist2(c[3], c[0]) > me2: split_v = True

                if not split_u or not split_v:
                    derivs2 = s.evaluate(um, vm, 2)
                    if len(derivs2) >= 6:
                        dux, duy, duz = derivs2[2][0], derivs2[2][1], derivs2[2][2]
                        dvx, dvy, dvz = derivs2[1][0], derivs2[1][1], derivs2[1][2]
                        d2ux, d2uy, d2uz = derivs2[5][0], derivs2[5][1], derivs2[5][2]
                        d2vx, d2vy, d2vz = derivs2[3][0], derivs2[3][1], derivs2[3][2]
                        nx = duy*dvz - duz*dvy
                        ny = duz*dvx - dux*dvz
                        nz = dux*dvy - duy*dvx
                        nlen = _math.sqrt(nx*nx + ny*ny + nz*nz)
                        E = dux*dux + duy*duy + duz*duz
                        G = dvx*dvx + dvy*dvy + dvz*dvz
                        if nlen > 1e-10 and E > 1e-20 and G > 1e-20:
                            nnx, nny, nnz = nx/nlen, ny/nlen, nz/nlen
                            e_coeff = d2ux*nnx + d2uy*nny + d2uz*nnz
                            g_coeff = d2vx*nnx + d2vy*nny + d2vz*nnz
                            kappa_u = abs(e_coeff) / E
                            kappa_v = abs(g_coeff) / G
                            span_u = _math.sqrt(E) * (u1 - u0)
                            span_v = _math.sqrt(G) * (v1 - v0)
                            if not split_u and kappa_u > 1e-20:
                                needed_u = span_u * _math.sqrt(kappa_u / (8.0 * chord_tol))
                                if needed_u > 1.0: split_u = True
                            if not split_v and kappa_v > 1e-20:
                                needed_v = span_v * _math.sqrt(kappa_v / (8.0 * chord_tol))
                                if needed_v > 1.0: split_v = True

                if not split_u and not split_v:
                    continue

                d = nd[6] + 1
                b = len(nodes)

                def make_node(nu0, nv0, nu1, nv1, sw, se, ne, nw):
                    ct = eval_corner((nu0+nu1)*0.5, (nv0+nv1)*0.5)
                    return [nu0, nv0, nu1, nv1, [sw, se, ne, nw, ct], [-1,-1,-1,-1], d]

                if split_u and split_v:
                    if not have_sn:
                        s_mid = eval_corner(um, v0)
                        n_mid = eval_corner(um, v1)
                    if not have_we:
                        w_mid = eval_corner(u0, vm)
                        e_mid = eval_corner(u1, vm)
                    nodes.extend([None]*4)
                    nd[5] = [b, b+1, b+2, b+3]
                    nodes[b] = make_node(u0, v0, um, vm, c[0], s_mid, c[4], w_mid)
                    nodes[b+1] = make_node(um, v0, u1, vm, s_mid, c[1], e_mid, c[4])
                    nodes[b+2] = make_node(um, vm, u1, v1, c[4], e_mid, c[2], n_mid)
                    nodes[b+3] = make_node(u0, vm, um, v1, w_mid, c[4], n_mid, c[3])
                    stack.extend([b, b+1, b+2, b+3])
                elif split_u:
                    if not have_sn:
                        s_mid = eval_corner(um, v0)
                        n_mid = eval_corner(um, v1)
                    nodes.extend([None]*2)
                    nd[5] = [b, b+1, -1, -1]
                    nodes[b] = make_node(u0, v0, um, v1, c[0], s_mid, n_mid, c[3])
                    nodes[b+1] = make_node(um, v0, u1, v1, s_mid, c[1], c[2], n_mid)
                    stack.extend([b, b+1])
                else:
                    if not have_we:
                        w_mid = eval_corner(u0, vm)
                        e_mid = eval_corner(u1, vm)
                    nodes.extend([None]*2)
                    nd[5] = [b, -1, b+1, -1]
                    nodes[b] = make_node(u0, v0, u1, vm, c[0], c[1], e_mid, w_mid)
                    nodes[b+1] = make_node(u0, vm, u1, v1, w_mid, e_mid, c[2], c[3])
                    stack.extend([b, b+1])

        # Pre-evaluate grid intersections
        grid = {}
        for i in range(ns_u+1):
            for j in range(ns_v+1):
                grid[(i, j)] = eval_corner(usp[i], vsp[j])

        for i in range(ns_u):
            for j in range(ns_v):
                sw = grid[(i, j)]
                se = grid[(i+1, j)]
                ne = grid[(i+1, j+1)]
                nw = grid[(i, j+1)]
                ct = eval_corner((usp[i]+usp[i+1])*0.5, (vsp[j]+vsp[j+1])*0.5)
                idx = len(nodes)
                nodes.append([usp[i], vsp[j], usp[i+1], vsp[j+1], [sw, se, ne, nw, ct], [-1,-1,-1,-1], 0])
                subdivide(idx)

        # Collect leaf corner UVs for T-junction detection
        SCALE = 1e10

        def qk(x):
            return int(round(x * SCALE))

        def norm_u(u):
            if closed_u and abs(u - usp[-1]) < 1e-10:
                return usp[0]
            return u

        def norm_v(v):
            if closed_v and abs(v - vsp[-1]) < 1e-10:
                return vsp[0]
            return v

        key_corner = {}
        by_v = {}
        by_u = {}

        def is_leaf(nd):
            ch = nd[5]
            return ch[0] < 0 and ch[2] < 0

        for nd in nodes:
            if not is_leaf(nd):
                continue
            us4 = [nd[0], nd[2], nd[2], nd[0]]
            vs4 = [nd[1], nd[1], nd[3], nd[3]]
            for ci in range(4):
                un = norm_u(us4[ci])
                vn = norm_v(vs4[ci])
                key = (qk(un), qk(vn))
                if key not in key_corner:
                    key_corner[key] = nd[4][ci]
                qv = qk(vn)
                if qv not in by_v:
                    by_v[qv] = []
                by_v[qv].append(qk(us4[ci]))
                qu = qk(un)
                if qu not in by_u:
                    by_u[qu] = []
                by_u[qu].append(qk(vs4[ci]))

        for k in by_v:
            by_v[k] = sorted(set(by_v[k]))
        for k in by_u:
            by_u[k] = sorted(set(by_u[k]))

        result = Mesh()
        south_pole = None
        north_pole = None

        if sing_v0:
            c0 = eval_corner(usp[0], vsp[0])
            south_pole = result.add_vertex(Point(c0[0], c0[1], c0[2]))
        if sing_v1:
            c1 = eval_corner(usp[0], vsp[-1])
            north_pole = result.add_vertex(Point(c1[0], c1[1], c1[2]))

        vtx_map = {}

        def get_vtx(u, v):
            if sing_v0 and abs(v - vsp[0]) < 1e-10:
                return south_pole
            if sing_v1 and abs(v - vsp[-1]) < 1e-10:
                return north_pole
            un = norm_u(u)
            vn = norm_v(v)
            key = (qk(un), qk(vn))
            if key in vtx_map:
                return vtx_map[key]
            if key in key_corner:
                c = key_corner[key]
            else:
                c = eval_corner(un, vn)
                key_corner[key] = c
            vi = result.add_vertex(Point(c[0], c[1], c[2]))
            result.vertex[vi].attributes["u"] = un
            result.vertex[vi].attributes["v"] = vn
            vtx_map[key] = vi
            return vi

        def find_h_mids(u_start, u_end, v_const):
            qv = qk(norm_v(v_const))
            if qv not in by_v:
                return []
            vec = by_v[qv]
            qa, qb = qk(u_start), qk(u_end)
            lo, hi = min(qa, qb), max(qa, qb)
            li = bisect_right(vec, lo)
            ri = bisect_left(vec, hi)
            res = [vec[k] / SCALE for k in range(li, ri)]
            if qa > qb:
                res.reverse()
            return res

        def find_v_mids(u_const, v_start, v_end):
            qu = qk(norm_u(u_const))
            if qu not in by_u:
                return []
            vec = by_u[qu]
            qa, qb = qk(v_start), qk(v_end)
            lo, hi = min(qa, qb), max(qa, qb)
            li = bisect_right(vec, lo)
            ri = bisect_left(vec, hi)
            res = [vec[k] / SCALE for k in range(li, ri)]
            if qa > qb:
                res.reverse()
            return res

        for nd in nodes:
            if not is_leaf(nd):
                continue
            u0, v0, u1, v1 = nd[0], nd[1], nd[2], nd[3]
            poly = []

            poly.append(get_vtx(u0, v0))
            for u in find_h_mids(u0, u1, v0):
                poly.append(get_vtx(u, v0))
            poly.append(get_vtx(u1, v0))
            for v in find_v_mids(u1, v0, v1):
                poly.append(get_vtx(u1, v))
            poly.append(get_vtx(u1, v1))
            for u in find_h_mids(u1, u0, v1):
                poly.append(get_vtx(u, v1))
            poly.append(get_vtx(u0, v1))
            for v in find_v_mids(u0, v1, v0):
                poly.append(get_vtx(u0, v))

            # Remove consecutive duplicates and wrap-around duplicate
            deduped = []
            for vi in poly:
                if not deduped or deduped[-1] != vi:
                    deduped.append(vi)
            while len(deduped) > 1 and deduped[0] == deduped[-1]:
                deduped.pop()
            poly = deduped

            n = len(poly)
            if n < 3:
                continue

            if n == 3:
                result.add_face([poly[0], poly[1], poly[2]])
            elif n == 4:
                v0v = result.vertex[poly[0]].position()
                v1v = result.vertex[poly[1]].position()
                v2v = result.vertex[poly[2]].position()
                v3v = result.vertex[poly[3]].position()
                d02 = (v0v[0]-v2v[0])**2 + (v0v[1]-v2v[1])**2 + (v0v[2]-v2v[2])**2
                d13 = (v1v[0]-v3v[0])**2 + (v1v[1]-v3v[1])**2 + (v1v[2]-v3v[2])**2
                if d02 <= d13:
                    result.add_face([poly[0], poly[1], poly[2]])
                    result.add_face([poly[0], poly[2], poly[3]])
                else:
                    result.add_face([poly[0], poly[1], poly[3]])
                    result.add_face([poly[1], poly[2], poly[3]])
            else:
                cu = norm_u((u0+u1)*0.5)
                cv = norm_v((v0+v1)*0.5)
                ck = (qk(cu), qk(cv))
                if ck not in key_corner:
                    key_corner[ck] = nd[4][4]
                cvi = get_vtx(cu, cv)
                for i in range(n):
                    j = (i+1) % n
                    if poly[i] != poly[j] and poly[i] != cvi and poly[j] != cvi:
                        result.add_face([poly[i], poly[j], cvi])

        # Compute vertex normals from face normals
        if result.vertex:
            max_vkey = max(result.vertex.keys())
            vnx = [0.0] * (max_vkey + 1)
            vny = [0.0] * (max_vkey + 1)
            vnz = [0.0] * (max_vkey + 1)
            for fi, vids in result.face.items():
                if len(vids) < 3:
                    continue
                p0 = result.vertex[vids[0]].position()
                p1 = result.vertex[vids[1]].position()
                p2 = result.vertex[vids[2]].position()
                e1x, e1y, e1z = p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]
                e2x, e2y, e2z = p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]
                fnx = e1y*e2z - e1z*e2y
                fny = e1z*e2x - e1x*e2z
                fnz = e1x*e2y - e1y*e2x
                for vi in vids:
                    vnx[vi] += fnx
                    vny[vi] += fny
                    vnz[vi] += fnz
            for vk in result.vertex:
                ln = math.sqrt(vnx[vk]**2 + vny[vk]**2 + vnz[vk]**2)
                if ln > 1e-15:
                    vnx[vk] /= ln
                    vny[vk] /= ln
                    vnz[vk] /= ln
                result.vertex[vk].set_normal(vnx[vk], vny[vk], vnz[vk])

        return result
