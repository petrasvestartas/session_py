import uuid
import copy
import json

from .closest import Closest
from .nurbssurface import NurbsSurface
from .nurbscurve import NurbsCurve
from .primitives import Primitives
from .xform import Xform
from .color import Color




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
        import math
        from .mesh import Mesh
        if not self.is_trimmed():
            return self.m_surface.mesh()

        # Planar: boundary-conforming ear-clip triangulation
        if self.m_surface.is_planar():
            from .remesh_cdt import _cdt_triangulate as _RemeshCDT_cdt
            def disc(crv):
                if crv.degree() <= 1:
                    return [crv.get_cv(i) for i in range(crv.cv_count() - 1)]
                n = max(crv.cv_count() * 4, 16)
                pts, _ = crv.divide_by_count(n + 1)
                return pts
            outer_pts = disc(self.m_outer_loop)
            hole_pts = [disc(inner) for inner in self.m_inner_loops]
            import numpy as _np
            from .point import Point as _Pt
            def _trim_closed(lst):
                n = len(lst)
                if n > 1 and abs(lst[0][0]-lst[n-1][0]) < 1e-12 and abs(lst[0][1]-lst[n-1][1]) < 1e-12:
                    return lst[:-1]
                return lst
            all_uvs = _trim_closed(outer_pts)
            for hp in hole_pts:
                all_uvs = all_uvs + _trim_closed(hp)
            if all_uvs:
                u_arr = _np.array([p[0] for p in all_uvs], dtype=_np.float64)
                v_arr = _np.array([p[1] for p in all_uvs], dtype=_np.float64)
                xyz = self.m_surface.batch_point_at(u_arr, v_arr)
                pts3d = [_Pt(xyz[i, 0], xyz[i, 1], xyz[i, 2]) for i in range(len(u_arr))]
            else:
                pts3d = []
            def to_pairs(uvs):
                n = len(uvs)
                if n > 1 and abs(uvs[0][0]-uvs[n-1][0]) < 1e-12 and abs(uvs[0][1]-uvs[n-1][1]) < 1e-12:
                    uvs = uvs[:-1]
                return [(float(p[0]), float(p[1])) for p in uvs]
            border = to_pairs(outer_pts)
            holes = [to_pairs(h) for h in hole_pts] if hole_pts else []
            area = sum(border[j][0]*border[(j+1)%len(border)][1] - border[(j+1)%len(border)][0]*border[j][1]
                       for j in range(len(border))) * 0.5
            if area < 0: border.reverse()
            for h in holes:
                ha = sum(h[j][0]*h[(j+1)%len(h)][1] - h[(j+1)%len(h)][0]*h[j][1] for j in range(len(h))) * 0.5
                if ha > 0: h.reverse()
            tris = _RemeshCDT_cdt(border, holes if holes else None)
            np_ = len(pts3d)
            polygons = []
            for v0, v1, v2 in tris:
                if 0 <= v0 < np_ and 0 <= v1 < np_ and 0 <= v2 < np_:
                    polygons.append([pts3d[v0], pts3d[v1], pts3d[v2]])
            result = Mesh.from_polylines(polygons)
            dom_u = self.m_surface.domain(0)
            dom_v = self.m_surface.domain(1)
            nrm = self.m_surface.normal_at((dom_u[0]+dom_u[1])/2, (dom_v[0]+dom_v[1])/2)
            for vk in result.vertex:
                result.vertex[vk].set_normal(nrm[0], nrm[1], nrm[2])
            return result

        # Non-planar: UV grid triangulation with per-vertex parametric normals
        import numpy as _np
        from .point import Point as _Pt
        def disc_np(crv):
            if crv.degree() <= 1:
                return [(float(crv.get_cv(i)[0]), float(crv.get_cv(i)[1])) for i in range(crv.cv_count() - 1)]
            n = max(crv.cv_count() * 4, 16)
            pts, _ = crv.divide_by_count(n + 1)
            return [(float(p[0]), float(p[1])) for p in pts[:-1]]
        border_uv = disc_np(self.m_outer_loop)
        holes_uv = [disc_np(inner) for inner in self.m_inner_loops]
        def pip(px, py, poly):
            inside = False
            j = len(poly) - 1
            for i in range(len(poly)):
                xi, yi = poly[i]; xj, yj = poly[j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi):
                    inside = not inside
                j = i
            return inside
        def inside_trim(pu, pv):
            if not pip(pu, pv, border_uv): return False
            for h in holes_uv:
                if pip(pu, pv, h): return False
            return True
        u_min = min(p[0] for p in border_uv); u_max = max(p[0] for p in border_uv)
        v_min = min(p[1] for p in border_uv); v_max = max(p[1] for p in border_uv)
        n_grid = 10
        us = _np.linspace(u_min, u_max, n_grid + 1)
        vs = _np.linspace(v_min, v_max, n_grid + 1)
        uv_grid = [[(float(us[i]), float(vs[j])) for j in range(n_grid + 1)] for i in range(n_grid + 1)]
        u_flat = _np.array([uv_grid[i][j][0] for i in range(n_grid+1) for j in range(n_grid+1)], dtype=_np.float64)
        v_flat = _np.array([uv_grid[i][j][1] for i in range(n_grid+1) for j in range(n_grid+1)], dtype=_np.float64)
        xyz = self.m_surface.batch_point_at(u_flat, v_flat)
        pts3d_grid = [_Pt(float(xyz[k, 0]), float(xyz[k, 1]), float(xyz[k, 2])) for k in range(len(u_flat))]
        idx = lambda i, j: i * (n_grid + 1) + j
        polygons = []
        uv_per_poly = []
        for i in range(n_grid):
            for j in range(n_grid):
                cx0 = (uv_grid[i][j][0] + uv_grid[i+1][j][0] + uv_grid[i][j+1][0]) / 3
                cy0 = (uv_grid[i][j][1] + uv_grid[i+1][j][1] + uv_grid[i][j+1][1]) / 3
                if inside_trim(cx0, cy0):
                    a, b, c = idx(i, j), idx(i+1, j), idx(i, j+1)
                    polygons.append([pts3d_grid[a], pts3d_grid[b], pts3d_grid[c]])
                    uv_per_poly.append([(u_flat[a], v_flat[a]), (u_flat[b], v_flat[b]), (u_flat[c], v_flat[c])])
                cx1 = (uv_grid[i+1][j][0] + uv_grid[i+1][j+1][0] + uv_grid[i][j+1][0]) / 3
                cy1 = (uv_grid[i+1][j][1] + uv_grid[i+1][j+1][1] + uv_grid[i][j+1][1]) / 3
                if inside_trim(cx1, cy1):
                    a, b, c = idx(i+1, j), idx(i+1, j+1), idx(i, j+1)
                    polygons.append([pts3d_grid[a], pts3d_grid[b], pts3d_grid[c]])
                    uv_per_poly.append([(u_flat[a], v_flat[a]), (u_flat[b], v_flat[b]), (u_flat[c], v_flat[c])])
        if not polygons:
            return Mesh()
        result = Mesh.from_polylines(polygons)
        pts3d_arr = _np.array([[p[0], p[1], p[2]] for tri in polygons for p in tri], dtype=_np.float64)
        uv_flat2 = [(u, v) for tri_uv in uv_per_poly for (u, v) in tri_uv]
        for vk, vd in result.vertex.items():
            p = vd.position()
            diffs = pts3d_arr - _np.array([p.x, p.y, p.z])
            best_i = int(_np.argmin((diffs**2).sum(axis=1)))
            u_val, v_val = uv_flat2[best_i]
            nrm = self.m_surface.normal_at(u_val, v_val)
            vd.set_normal(nrm[0], nrm[1], nrm[2])
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
