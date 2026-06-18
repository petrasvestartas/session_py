import math
from typing import TYPE_CHECKING

from .tolerance import PI

if TYPE_CHECKING:
    from .nurbssurface import NurbsSurface
    from .mesh import Mesh

def remesh_nurbssurface_grid(surface, max_u: int, max_v: int):
    return RemeshNurbsSurfaceGrid.from_u_v(surface, max_u, max_v)


class RemeshNurbsSurfaceGrid:
    @staticmethod
    def from_u_v(s, max_u: int, max_v: int):
        return RemeshNurbsSurfaceGrid.from_u_v_q(s, max_u, max_v, 20.0, 0.005)

    @staticmethod
    def from_u_v_q(s, max_u: int, max_v: int, max_angle_deg: float, chord_factor: float):
        from .mesh import Mesh
        MAX_ANGLE = max_angle_deg
        usp = list(s.get_span_vector(0))
        vsp = list(s.get_span_vector(1))
        ns_u = len(usp) - 1
        ns_v = len(vsp) - 1
        deg_u = s.degree(0)
        deg_v = s.degree(1)

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
        bbox_diag = math.sqrt(dx*dx + dy*dy + dz*dz)

        def span_subs(dir, sp, osp):
            n = len(sp) - 1
            subs = [1] * n
            n_other = len(osp) - 1
            s_positions = [(osp[k] + osp[k + 1]) * 0.5 for k in range(n_other)]
            degree_dir = deg_u if dir == 0 else deg_v
            for i in range(n):
                t0, t1 = sp[i], sp[i + 1]
                if degree_dir > 1:
                    max_angle = 0.0
                    for si in range(n_other):
                        sval = s_positions[si]
                        fn3 = [0.0, 0.0, 0.0]
                        ln3 = [0.0, 0.0, 0.0]
                        has_first = False
                        for k in range(5):
                            t = t0 + k * (t1 - t0) / 4.0
                            nrm = s.normal_at(t, sval) if dir == 0 else s.normal_at(sval, t)
                            nx, ny, nz = nrm[0], nrm[1], nrm[2]
                            length = math.sqrt(nx*nx + ny*ny + nz*nz)
                            if length < 1e-10:
                                continue
                            nx /= length
                            ny /= length
                            nz /= length
                            if not has_first:
                                fn3[0] = nx
                                fn3[1] = ny
                                fn3[2] = nz
                                has_first = True
                            ln3[0] = nx
                            ln3[1] = ny
                            ln3[2] = nz
                        total_angle = 0.0
                        if has_first:
                            dot = fn3[0]*ln3[0] + fn3[1]*ln3[1] + fn3[2]*ln3[2]
                            total_angle = math.acos(max(-1.0, min(1.0, dot))) * 180.0 / PI
                        if total_angle > max_angle:
                            max_angle = total_angle
                    subs[i] = max(1, min(int(math.ceil(max_angle / MAX_ANGLE)), 24))
                chord_tol = bbox_diag * chord_factor
                max_dev = 0.0
                nc = min(n_other, 3)
                for ci in range(nc + 1):
                    sv = osp[0] + ci * (osp[-1] - osp[0]) / max(nc, 1)
                    if dir == 0:
                        p0 = s.point_at(t0, sv)
                        p1 = s.point_at(t1, sv)
                    else:
                        p0 = s.point_at(sv, t0)
                        p1 = s.point_at(sv, t1)
                    px0, py0, pz0 = p0[0], p0[1], p0[2]
                    px1, py1, pz1 = p1[0], p1[1], p1[2]
                    for k in range(1, 4):
                        frac = k / 4.0
                        tm = t0 + frac * (t1 - t0)
                        if dir == 0:
                            pm = s.point_at(tm, sv)
                        else:
                            pm = s.point_at(sv, tm)
                        pmx, pmy, pmz = pm[0], pm[1], pm[2]
                        lx = px0 + frac * (px1 - px0)
                        ly = py0 + frac * (py1 - py0)
                        lz = pz0 + frac * (pz1 - pz0)
                        ddx = pmx - lx
                        ddy = pmy - ly
                        ddz = pmz - lz
                        dev = math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
                        if dev > max_dev:
                            max_dev = dev
                if max_dev > chord_tol:
                    chord_subs = max(2, int(math.ceil(math.sqrt(max_dev / chord_tol))))
                    subs[i] = max(subs[i], min(chord_subs, 24))
                if degree_dir > 1:
                    subs[i] = max(subs[i], 2)
            return subs

        u_subs = span_subs(0, usp, vsp)
        v_subs = span_subs(1, vsp, usp)

        # Arc-length aspect ratio balancing
        total_u = sum(u_subs) + 1
        total_v = sum(v_subs) + 1
        v_mid = (vsp[0] + vsp[-1]) * 0.5
        u_mid = (usp[0] + usp[-1]) * 0.5
        u_len = 0.0
        p0 = s.point_at(usp[0], v_mid)
        px0, py0, pz0 = p0[0], p0[1], p0[2]
        n_sample = max(total_u, 10)
        for i in range(1, n_sample + 1):
            u = usp[0] + i * (usp[-1] - usp[0]) / n_sample
            p1 = s.point_at(u, v_mid)
            px1, py1, pz1 = p1[0], p1[1], p1[2]
            u_len += math.sqrt((px1-px0)**2 + (py1-py0)**2 + (pz1-pz0)**2)
            px0, py0, pz0 = px1, py1, pz1
        v_len = 0.0
        p0 = s.point_at(u_mid, vsp[0])
        px0, py0, pz0 = p0[0], p0[1], p0[2]
        n_sample = max(total_v, 10)
        for i in range(1, n_sample + 1):
            v = vsp[0] + i * (vsp[-1] - vsp[0]) / n_sample
            p1 = s.point_at(u_mid, v)
            px1, py1, pz1 = p1[0], p1[1], p1[2]
            v_len += math.sqrt((px1-px0)**2 + (py1-py0)**2 + (pz1-pz0)**2)
            px0, py0, pz0 = px1, py1, pz1
        if u_len > 1e-14 and v_len > 1e-14 and total_u > 0 and total_v > 0:
            spacing_u = u_len / total_u
            spacing_v = v_len / total_v
            ratio = spacing_u / spacing_v
            if ratio > 2.0 and deg_u > 1:
                scale = math.sqrt(ratio)
                u_subs = [min(int(math.ceil(sub * scale)), 24) for sub in u_subs]
            elif ratio < 0.5 and deg_v > 1:
                scale = math.sqrt(1.0 / ratio)
                v_subs = [min(int(math.ceil(sub * scale)), 24) for sub in v_subs]

        # Bilinear twist check (skip for singular surfaces — fan triangulation handles those)
        if deg_u == 1 and deg_v == 1 and not s.is_singular(0) and not s.is_singular(2):
            import numpy as _np
            chord_tol = bbox_diag * chord_factor if bbox_diag > 0 else 1e-6
            u0_a = _np.array(usp[:-1], dtype=_np.float64)
            u1_a = _np.array(usp[1:], dtype=_np.float64)
            v0_a = _np.array(vsp[:-1], dtype=_np.float64)
            v1_a = _np.array(vsp[1:], dtype=_np.float64)
            um = ((u0_a + u1_a) * 0.5)
            vm = ((v0_a + v1_a) * 0.5)
            um_g = _np.repeat(um, ns_v)
            vm_g = _np.tile(vm, ns_u)
            u0_g = _np.repeat(u0_a, ns_v)
            v0_g = _np.tile(v0_a, ns_u)
            u1_g = _np.repeat(u1_a, ns_v)
            v1_g = _np.tile(v1_a, ns_u)
            all_u = _np.concatenate([um_g, u0_g, u1_g])
            all_v = _np.concatenate([vm_g, v0_g, v1_g])
            xyz_twist = s.batch_point_at(all_u, all_v)
            k_tw = ns_u * ns_v
            pm_arr = xyz_twist[:k_tw]
            p00_arr = xyz_twist[k_tw:2 * k_tw]
            p11_arr = xyz_twist[2 * k_tw:]
            mid = (p00_arr + p11_arr) * 0.5
            twist = _np.linalg.norm(pm_arr - mid, axis=1)
            max_twist = float(twist.max()) if twist.size else 0.0
            if max_twist > chord_tol:
                twist_subs = max(4, min(int(math.ceil(2.0 * math.sqrt(max_twist / chord_tol))), 24))
                u_subs = [max(sub, twist_subs) for sub in u_subs]
                v_subs = [max(sub, twist_subs) for sub in v_subs]

        closed_u = s.is_closed(0)
        closed_v = s.is_closed(1)

        # Ensure odd total subdivisions for closed directions (seamless checkerboard triangulation)
        if closed_u and max_u == 0:
            if sum(u_subs) % 2 == 0:
                u_subs[u_subs.index(max(u_subs))] += 1
        if closed_v and max_v == 0:
            if sum(v_subs) % 2 == 0:
                v_subs[v_subs.index(max(v_subs))] += 1

        def arclen_params(n, sp, fixed, is_u):
            nsample = max(n * 20, 200)
            st = [sp[0] + k * (sp[-1] - sp[0]) / nsample for k in range(nsample + 1)]
            sl = [0.0] * (nsample + 1)
            if is_u:
                p0 = s.point_at(sp[0], fixed)
            else:
                p0 = s.point_at(fixed, sp[0])
            px0, py0, pz0 = p0[0], p0[1], p0[2]
            for k in range(1, nsample + 1):
                if is_u:
                    p1 = s.point_at(st[k], fixed)
                else:
                    p1 = s.point_at(fixed, st[k])
                px1, py1, pz1 = p1[0], p1[1], p1[2]
                d = math.sqrt((px1-px0)**2 + (py1-py0)**2 + (pz1-pz0)**2)
                sl[k] = sl[k-1] + d
                px0, py0, pz0 = px1, py1, pz1
            total_len = sl[nsample]
            params = [sp[0]]
            j = 0
            for i in range(1, n - 1):
                target = total_len * i / (n - 1)
                while j < nsample and sl[j] < target:
                    j += 1
                ta = st[j-1] if j > 0 else st[0]
                tb = st[j]
                la = sl[j-1] if j > 0 else sl[0]
                lb = sl[j]
                frac = (target - la) / (lb - la) if lb > la else 0.0
                params.append(ta + frac * (tb - ta))
            params.append(sp[-1])
            return params

        # Build parameter arrays
        if max_u > 0:
            us = arclen_params(max(max_u, 2), usp, v_mid, True)
        else:
            us = []
            for i in range(ns_u):
                for sv in range(u_subs[i]):
                    us.append(usp[i] + sv * (usp[i + 1] - usp[i]) / u_subs[i])
            us.append(usp[-1])
        if max_v > 0:
            vs = arclen_params(max(max_v, 2), vsp, u_mid, False)
        else:
            vs = []
            for i in range(ns_v):
                for sv in range(v_subs[i]):
                    vs.append(vsp[i] + sv * (vsp[i + 1] - vsp[i]) / v_subs[i])
            vs.append(vsp[-1])

        def fix_closed_gap(params, spans, closed):
            if not closed or len(params) < 3:
                return params
            params = params[:-1]
            domain_end = spans[-1]
            wrap_gap = domain_end - params[-1]
            max_gap = 0.0
            for i in range(1, len(params)):
                g = params[i] - params[i - 1]
                if g > max_gap:
                    max_gap = g
            if max_gap > 0 and wrap_gap > max_gap * 1.5:
                extra = int(math.ceil(wrap_gap / max_gap)) - 1
                step = wrap_gap / (extra + 1)
                for e in range(1, extra + 1):
                    params.append(params[-1] + step)
            return params

        us = fix_closed_gap(us, usp, closed_u)
        vs = fix_closed_gap(vs, vsp, closed_v)
        nu = len(us)
        nv_count = len(vs)

        sing_v0 = s.is_singular(0)
        sing_v1 = s.is_singular(2)
        j_start = 1 if sing_v0 else 0
        j_end = nv_count - 1 if sing_v1 else nv_count
        nv_grid = j_end - j_start

        result = Mesh()
        south_pole = 0
        north_pole = 0
        if sing_v0:
            p = s.point_at(us[0], vs[0])
            south_pole = result.add_vertex(p)
            result.vertex[south_pole].attributes["u"] = us[0]
            result.vertex[south_pole].attributes["v"] = vs[0]
        if sing_v1:
            p = s.point_at(us[0], vs[nv_count - 1])
            north_pole = result.add_vertex(p)
            result.vertex[north_pole].attributes["u"] = us[0]
            result.vertex[north_pole].attributes["v"] = vs[nv_count - 1]
        grid_base = len(result.vertex)
        import numpy as _np
        _us_a = _np.asarray(us, dtype=_np.float64)
        _vs_a = _np.asarray(vs[j_start:j_end], dtype=_np.float64)
        _grid_us = _np.repeat(_us_a, len(_vs_a))
        _grid_vs = _np.tile(_vs_a, len(_us_a))
        _pts_arr = s.batch_point_at(_grid_us, _grid_vs)
        from .point import Point as _Pt
        for idx in range(len(_grid_us)):
            vk = result.add_vertex(_Pt(_pts_arr[idx, 0], _pts_arr[idx, 1], _pts_arr[idx, 2]))
            result.vertex[vk].attributes["u"] = float(_grid_us[idx])
            result.vertex[vk].attributes["v"] = float(_grid_vs[idx])

        def grid_idx(i, j):
            return grid_base + i * nv_grid + (j - j_start)

        nu_faces = nu if closed_u else nu - 1

        # South pole fan
        if sing_v0:
            for i in range(nu_faces):
                i1 = (i + 1) % nu
                result.add_face([south_pole, grid_idx(i1, j_start), grid_idx(i, j_start)])

        # Interior grid faces
        nv_interior = nv_grid - 1
        if closed_v and not sing_v0 and not sing_v1:
            nv_interior = nv_grid
        for i in range(nu_faces):
            for jj in range(nv_interior):
                j = jj + j_start
                i1 = (i + 1) % nu
                if closed_v and not sing_v0 and not sing_v1:
                    j1 = (jj + 1) % nv_grid + j_start
                else:
                    j1 = j + 1
                v00 = grid_idx(i, j)
                v10 = grid_idx(i1, j)
                v01 = grid_idx(i, j1)
                v11 = grid_idx(i1, j1)
                if (i + jj) % 2 == 0:
                    result.add_face([v00, v10, v11])
                    result.add_face([v00, v11, v01])
                else:
                    result.add_face([v00, v10, v01])
                    result.add_face([v10, v11, v01])

        # North pole fan
        if sing_v1:
            j_last = j_end - 1
            for i in range(nu_faces):
                i1 = (i + 1) % nu
                result.add_face([grid_idx(i, j_last), grid_idx(i1, j_last), north_pole])

        # Compute vertex normals from face normals
        if result.vertex:
            max_vkey = max(result.vertex.keys())
            vnx = [0.0] * (max_vkey + 1)
            vny = [0.0] * (max_vkey + 1)
            vnz = [0.0] * (max_vkey + 1)
            for fi, vids in result.face.items():
                if len(vids) < 3:
                    continue
                pos0 = result.vertex[vids[0]].position()
                pos1 = result.vertex[vids[1]].position()
                pos2 = result.vertex[vids[2]].position()
                e1x, e1y, e1z = pos1[0]-pos0[0], pos1[1]-pos0[1], pos1[2]-pos0[2]
                e2x, e2y, e2z = pos2[0]-pos0[0], pos2[1]-pos0[1], pos2[2]-pos0[2]
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
