import uuid
import copy
import json

from .closest import Closest
from .nurbssurface import NurbsSurface
from .nurbscurve import NurbsCurve
from .primitives import Primitives
from .xform import Xform
from .color import Color


def _point_in_polygon_2d(px, py, coords):
    winding = 0
    n = len(coords) // 2
    for i in range(n):
        j = (i + 1) % n
        y0, y1 = coords[i*2+1], coords[j*2+1]
        if y0 <= py:
            if y1 > py:
                x0, x1 = coords[i*2], coords[j*2]
                if (x1-x0)*(py-y0) - (px-x0)*(y1-y0) > 0:
                    winding += 1
        else:
            if y1 <= py:
                x0, x1 = coords[i*2], coords[j*2]
                if (x1-x0)*(py-y0) - (px-x0)*(y1-y0) < 0:
                    winding -= 1
    return winding != 0


class TrimmedSurface:

    def __init__(self):
        self.guid = str(uuid.uuid4())
        self.name = "my_trimmedsurface"
        self.width = 1.0
        self.surfacecolor = Color.black()
        self.xform = Xform.identity()
        self.m_surface = NurbsSurface()
        self.m_outer_loop = NurbsCurve()
        self.m_inner_loops = []

    @staticmethod
    def create(surface, outer_loop):
        ts = TrimmedSurface()
        ts.m_surface = surface.duplicate()
        ts.m_outer_loop = outer_loop.duplicate()
        return ts

    @staticmethod
    def create_planar(boundary):
        from .point import Point
        from .vector import Vector
        srf = Primitives.create_planar(boundary)
        if not srf.is_valid():
            return TrimmedSurface()
        p00 = srf.get_cv(0, 0)
        p10 = srf.get_cv(1, 0)
        p01 = srf.get_cv(0, 1)
        u_axis = Vector(p10[0]-p00[0], p10[1]-p00[1], p10[2]-p00[2])
        v_axis = Vector(p01[0]-p00[0], p01[1]-p00[1], p01[2]-p00[2])
        u_len2 = u_axis[0]**2 + u_axis[1]**2 + u_axis[2]**2
        v_len2 = v_axis[0]**2 + v_axis[1]**2 + v_axis[2]**2
        if u_len2 < 1e-28 or v_len2 < 1e-28:
            return TrimmedSurface()
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
        ts = TrimmedSurface()
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
            from . import triangulation_2d
            def disc(crv):
                n = max(crv.cv_count() * 4, 16) if crv.degree() > 1 else max(crv.cv_count() - 1, 4)
                pts, _ = crv.divide_by_count(n)
                return pts
            outer_pts = disc(self.m_outer_loop)
            hole_pts = [disc(inner) for inner in self.m_inner_loops]
            pts3d = []
            def add_pts(uv_list):
                n = len(uv_list)
                if n > 1 and abs(uv_list[0][0]-uv_list[n-1][0]) < 1e-12 and \
                   abs(uv_list[0][1]-uv_list[n-1][1]) < 1e-12:
                    n -= 1
                for i in range(n):
                    pts3d.append(self.m_surface.point_at(uv_list[i][0], uv_list[i][1]))
            add_pts(outer_pts)
            for hp in hole_pts:
                add_pts(hp)
            tris = triangulation_2d.triangulate(outer_pts, hole_pts if hole_pts else None)
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

        # Non-planar: grid + point-in-polygon discard
        dom_u = self.m_surface.domain(0)
        dom_v = self.m_surface.domain(1)
        range_u = dom_u[1] - dom_u[0]
        range_v = dom_v[1] - dom_v[0]
        if range_u < 1e-15 or range_v < 1e-15:
            return self.m_surface.mesh()
        p00 = self.m_surface.point_at(dom_u[0], dom_v[0])
        p10 = self.m_surface.point_at(dom_u[1], dom_v[0])
        p01 = self.m_surface.point_at(dom_u[0], dom_v[1])
        u_len = math.sqrt((p10[0]-p00[0])**2+(p10[1]-p00[1])**2+(p10[2]-p00[2])**2)
        v_len = math.sqrt((p01[0]-p00[0])**2+(p01[1]-p00[1])**2+(p01[2]-p00[2])**2)
        max_dim = max(u_len, v_len)
        max_edge = max_dim / 10.0 if max_dim > 1e-10 else 0.1
        nu = max(12, int(math.ceil(u_len / max_edge)) + 1) if u_len > 1e-10 else 12
        nv = max(12, int(math.ceil(v_len / max_edge)) + 1) if v_len > 1e-10 else 12
        us = [dom_u[0] + i * range_u / (nu - 1) for i in range(nu)]
        vs = [dom_v[0] + i * range_v / (nv - 1) for i in range(nv)]
        full = Mesh()
        for i in range(nu):
            for j in range(nv):
                vk = full.add_vertex(self.m_surface.point_at(us[i], vs[j]))
                full.vertex[vk].attributes["u"] = us[i]
                full.vertex[vk].attributes["v"] = vs[j]
        for i in range(nu - 1):
            for j in range(nv - 1):
                v00 = i * nv + j
                v10 = (i + 1) * nv + j
                v01 = i * nv + (j + 1)
                v11 = (i + 1) * nv + (j + 1)
                if (i + j) % 2 == 0:
                    full.add_face([v00, v10, v11])
                    full.add_face([v00, v11, v01])
                else:
                    full.add_face([v00, v10, v01])
                    full.add_face([v10, v11, v01])
        def discretize(crv):
            n = max(crv.cv_count() * 4, 16)
            pts, _ = crv.divide_by_count(n)
            coords = []
            for p in pts:
                coords.append(p[0])
                coords.append(p[1])
            return coords
        outer_coords = discretize(self.m_outer_loop)
        inner_coords = [discretize(inner) for inner in self.m_inner_loops]
        keep_verts = set()
        for vk, vd in full.vertex.items():
            u_raw = vd.attributes.get("u", 0.0)
            v_raw = vd.attributes.get("v", 0.0)
            if not _point_in_polygon_2d(u_raw, v_raw, outer_coords):
                continue
            in_hole = False
            for ic in inner_coords:
                if _point_in_polygon_2d(u_raw, v_raw, ic):
                    in_hole = True
                    break
            if not in_hole:
                keep_verts.add(vk)
        polygons = []
        for fk, fverts in full.face.items():
            if all(vi in keep_verts for vi in fverts):
                polygons.append([full.vertex[vi].position() for vi in fverts])
        result = Mesh.from_polylines(polygons)
        max_vkey = max(result.vertex.keys()) if result.vertex else 0
        vnx = [0.0] * (max_vkey + 1)
        vny = [0.0] * (max_vkey + 1)
        vnz = [0.0] * (max_vkey + 1)
        for fi, vids in result.face.items():
            if len(vids) < 3:
                continue
            p0 = result.vertex[vids[0]]
            p1 = result.vertex[vids[1]]
            p2 = result.vertex[vids[2]]
            e1x, e1y, e1z = p1.x-p0.x, p1.y-p0.y, p1.z-p0.z
            e2x, e2y, e2z = p2.x-p0.x, p2.y-p0.y, p2.z-p0.z
            fnx = e1y*e2z - e1z*e2y
            fny = e1z*e2x - e1x*e2z
            fnz = e1x*e2y - e1y*e2x
            for vi in vids:
                vnx[vi] += fnx; vny[vi] += fny; vnz[vi] += fnz
        for vk in result.vertex:
            ln = math.sqrt(vnx[vk]**2 + vny[vk]**2 + vnz[vk]**2)
            if ln > 1e-15:
                vnx[vk] /= ln; vny[vk] /= ln; vnz[vk] /= ln
            result.vertex[vk].set_normal(vnx[vk], vny[vk], vnz[vk])
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
        if not isinstance(other, TrimmedSurface):
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
        return f"TrimmedSurface(name={self.name}, trimmed={'true' if self.is_trimmed() else 'false'}, holes={self.inner_loop_count()})"

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return (f"TrimmedSurface(\n  name={self.name},\n"
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
        d['type'] = 'TrimmedSurface'
        d['width'] = self.width
        d['xform'] = self.xform.__jsondump__()
        return d

    @classmethod
    def __jsonload__(cls, data):
        ts = cls()
        ts.guid = data.get('guid', ts.guid)
        ts.name = data.get('name', 'my_trimmedsurface')
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

    def json_dump(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def json_dumps(self):
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, json_string):
        return cls.__jsonload__(json.loads(json_string))

    def pb_dumps(self):
        from .proto import trimmedsurface_pb2
        proto = trimmedsurface_pb2.TrimmedSurface()
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
        from .proto import trimmedsurface_pb2
        proto = trimmedsurface_pb2.TrimmedSurface()
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
