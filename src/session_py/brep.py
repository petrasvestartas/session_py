import uuid
import copy

from .point import Point
from .vector import Vector
from .xform import Xform
from .color import Color
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface


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


class BRep:
    def __init__(self):
        self.guid = str(uuid.uuid4())
        self.name = "my_brep"
        self.width = 1.0
        self.surfacecolor = Color.black()
        self.xform = Xform.identity()
        self.m_surfaces = []
        self.m_curves_3d = []
        self.m_curves_2d = []
        self.m_vertices = []
        self.m_topology_vertices = []
        self.m_topology_edges = []
        self.m_trims = []
        self.m_loops = []
        self.m_faces = []

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
        for e in self.m_topology_edges:
            if len(e.trim_indices) != 2:
                return False
        return True

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
        uv_corners = [Point(0,0,0), Point(1,0,0), Point(1,1,0), Point(0,1,0)]
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
        c2d_bot = NurbsCurve.create(False, 1, [Point(dom_u[0],dom_v[0],0), Point(dom_u[1],dom_v[0],0)])
        brep.add_trim(brep.add_curve_2d(c2d_bot), ei_bot, li_body, False, BRepTrimType.Mated)
        c2d_sr = NurbsCurve.create(False, 1, [Point(dom_u[1],dom_v[0],0), Point(dom_u[1],dom_v[1],0)])
        brep.add_trim(brep.add_curve_2d(c2d_sr), ei_seam, li_body, False, BRepTrimType.Seam)
        c2d_top = NurbsCurve.create(False, 1, [Point(dom_u[1],dom_v[1],0), Point(dom_u[0],dom_v[1],0)])
        brep.add_trim(brep.add_curve_2d(c2d_top), ei_top, li_body, True, BRepTrimType.Mated)
        c2d_sl = NurbsCurve.create(False, 1, [Point(dom_u[0],dom_v[1],0), Point(dom_u[0],dom_v[0],0)])
        brep.add_trim(brep.add_curve_2d(c2d_sl), ei_seam, li_body, True, BRepTrimType.Seam)
        # Circular 2D trim in UV space: circle at (0.5,0.5) radius 0.5
        import math as _m
        _w = _m.sqrt(2.0) / 2.0
        _cx = [1,1,0,-1,-1,-1,0,1,1]; _cy = [0,1,1,1,0,-1,-1,-1,0]
        _cw = [1,_w,1,_w,1,_w,1,_w,1]; _kn = [0,0,1,1,2,2,3,3,4,4]
        def _cap_circle():
            import numpy as np
            c = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=9)
            c.m_knot = np.array(_kn, dtype=np.float64)
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
        c2d_south = NurbsCurve.create(False, 1, [Point(dom_u[0],dom_v[0],0), Point(dom_u[1],dom_v[0],0)])
        brep.add_trim(brep.add_curve_2d(c2d_south), -1, li, False, BRepTrimType.Singular)
        c2d_sr = NurbsCurve.create(False, 1, [Point(dom_u[1],dom_v[0],0), Point(dom_u[1],dom_v[1],0)])
        brep.add_trim(brep.add_curve_2d(c2d_sr), ei_seam, li, False, BRepTrimType.Seam)
        c2d_north = NurbsCurve.create(False, 1, [Point(dom_u[1],dom_v[1],0), Point(dom_u[0],dom_v[1],0)])
        brep.add_trim(brep.add_curve_2d(c2d_north), -1, li, False, BRepTrimType.Singular)
        c2d_sl = NurbsCurve.create(False, 1, [Point(dom_u[0],dom_v[1],0), Point(dom_u[0],dom_v[0],0)])
        brep.add_trim(brep.add_curve_2d(c2d_sl), ei_seam, li, True, BRepTrimType.Seam)
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
        uv = [Point(0,0,0), Point(1,0,0), Point(1,1,0), Point(0,1,0)]
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
        c2d_bot = NurbsCurve.create(False, 1, [Point(dom_u[0],dom_v[0],0), Point(dom_u[1],dom_v[0],0)])
        brep.add_trim(brep.add_curve_2d(c2d_bot), ei_bot, li_cyl, False, BRepTrimType.Mated)
        c2d_sr = NurbsCurve.create(False, 1, [Point(dom_u[1],dom_v[0],0), Point(dom_u[1],dom_v[1],0)])
        brep.add_trim(brep.add_curve_2d(c2d_sr), ei_seam, li_cyl, False, BRepTrimType.Seam)
        c2d_top = NurbsCurve.create(False, 1, [Point(dom_u[1],dom_v[1],0), Point(dom_u[0],dom_v[1],0)])
        brep.add_trim(brep.add_curve_2d(c2d_top), ei_top, li_cyl, True, BRepTrimType.Mated)
        c2d_sl = NurbsCurve.create(False, 1, [Point(dom_u[0],dom_v[1],0), Point(dom_u[0],dom_v[0],0)])
        brep.add_trim(brep.add_curve_2d(c2d_sl), ei_seam, li_cyl, True, BRepTrimType.Seam)
        _w = _m.sqrt(2.0) / 2.0
        _cx = [1,1,0,-1,-1,-1,0,1,1]; _cy = [0,1,1,1,0,-1,-1,-1,0]
        _cw = [1,_w,1,_w,1,_w,1,_w,1]; _kn = [0,0,1,1,2,2,3,3,4,4]
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
            hole_crv.m_knot = np.array(_kn, dtype=np.float64)
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

        def find_or_add(p):
            for i, v in enumerate(brep.m_vertices):
                dx, dy, dz = p[0] - v[0], p[1] - v[1], p[2] - v[2]
                if dx*dx + dy*dy + dz*dz < tol*tol:
                    return i
            idx = brep.add_vertex(p)
            tv = BRepVertex(); tv.point_index = idx
            brep.m_topology_vertices.append(tv)
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
                tc = NurbsCurve.create(False, 1, [Point(u0,v0,0), Point(u1,v1,0)])
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
        from .plane import Plane
        from .polyline import Polyline
        brep = BRep()
        brep.name = "polysurface"
        tol = 1e-6
        if holes is None:
            holes = []

        def find_or_add(p):
            for i, v in enumerate(brep.m_vertices):
                dx, dy, dz = p[0] - v[0], p[1] - v[1], p[2] - v[2]
                if dx*dx + dy*dy + dz*dz < tol*tol:
                    return i
            idx = brep.add_vertex(p)
            tv = BRepVertex(); tv.point_index = idx
            brep.m_topology_vertices.append(tv)
            return idx

        def project_curve_to_uv(crv, org, xa, ya, umin, vmin, du, dv):
            import numpy as np
            c2d = NurbsCurve(dimension=3, is_rational=crv.is_rational(), order=crv.order(), cv_count=crv.cv_count())
            c2d.m_knot = np.array([crv.knot(i) for i in range(crv.knot_count())], dtype=np.float64)
            c2d.m_cv = np.zeros(crv.cv_count() * c2d.m_cv_stride, dtype=np.float64)
            for i in range(crv.cv_count()):
                if crv.is_rational():
                    wx, wy, wz, w = crv.get_cv_4d(i)
                    x, y, z = wx/w, wy/w, wz/w
                    dx, dy, dz = x-org[0], y-org[1], z-org[2]
                    u = (dx*xa[0]+dy*xa[1]+dz*xa[2] - umin) / du
                    v = (dx*ya[0]+dy*ya[1]+dz*ya[2] - vmin) / dv
                    c2d.set_cv_4d(i, u*w, v*w, 0.0, w)
                else:
                    cv = crv.get_cv(i)
                    dx, dy, dz = cv[0]-org[0], cv[1]-org[1], cv[2]-org[2]
                    u = (dx*xa[0]+dy*xa[1]+dz*xa[2] - umin) / du
                    v = (dx*ya[0]+dy*ya[1]+dz*ya[2] - vmin) / dv
                    c2d.set_cv(i, Point(u, v, 0))
            return c2d

        def add_curve_loop(crv, face_idx, loop_type, org, xa, ya, umin, vmin, du, dv):
            li = brep.add_loop(face_idx, loop_type)
            ci3d = brep.add_curve_3d(crv)
            crv2d = project_curve_to_uv(crv, org, xa, ya, umin, vmin, du, dv)
            c2d = brep.add_curve_2d(crv2d)
            dom = crv.domain()
            sp = crv.point_at(dom[0])
            ep = crv.point_at(dom[1])
            vi_s = find_or_add(sp)
            vi_e = vi_s if crv.is_closed() else find_or_add(ep)
            lo, hi = min(vi_s, vi_e), max(vi_s, vi_e)
            ei = brep.add_edge(ci3d, lo, hi)
            brep.add_trim(c2d, ei, li, False, BRepTrimType.Boundary)

        for ci_idx, crv in enumerate(curves):
            pts, _ = crv.divide_by_count(max(crv.cv_count() * 2, 4))
            n = len(pts) - 1 if crv.is_closed() else len(pts)
            if n < 3:
                continue
            pl = Polyline(pts)
            org, plane = pl.get_fast_plane()
            if not plane.is_valid():
                continue
            xa, ya = plane.x_axis, plane.y_axis
            us_list, vs_list = [], []
            umin = vmin = 1e30
            umax = vmax = -1e30
            for i in range(n):
                dx = pts[i][0]-org[0]; dy = pts[i][1]-org[1]; dz = pts[i][2]-org[2]
                u = dx*xa[0]+dy*xa[1]+dz*xa[2]; v = dx*ya[0]+dy*ya[1]+dz*ya[2]
                us_list.append(u); vs_list.append(v)
                umin = min(umin, u); umax = max(umax, u)
                vmin = min(vmin, v); vmax = max(vmax, v)
            if ci_idx < len(holes):
                for hcrv in holes[ci_idx]:
                    hpts, _ = hcrv.divide_by_count(max(hcrv.cv_count() * 2, 4))
                    for hp in hpts:
                        dx = hp[0]-org[0]; dy = hp[1]-org[1]; dz = hp[2]-org[2]
                        hu = dx*xa[0]+dy*xa[1]+dz*xa[2]; hv = dx*ya[0]+dy*ya[1]+dz*ya[2]
                        umin = min(umin, hu); umax = max(umax, hu)
                        vmin = min(vmin, hv); vmax = max(vmax, hv)
            pad = max(umax - umin, vmax - vmin) * 0.01
            umin -= pad; umax += pad; vmin -= pad; vmax += pad
            du, dv = umax - umin, vmax - vmin

            def pt3d(u, v):
                return Point(org[0]+u*xa[0]+v*ya[0], org[1]+u*xa[1]+v*ya[1], org[2]+u*xa[2]+v*ya[2])
            srf = NurbsSurface(3, False, 2, 2, 2, 2)
            srf.set_cv(0, 0, pt3d(umin, vmin)); srf.set_cv(1, 0, pt3d(umax, vmin))
            srf.set_cv(0, 1, pt3d(umin, vmax)); srf.set_cv(1, 1, pt3d(umax, vmax))
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
    # Meshing
    ###########################################################################

    def mesh(self):
        from .mesh import Mesh
        from .trimmedsurface import TrimmedSurface
        all_polygons = []
        for fi, face in enumerate(self.m_faces):
            if face.surface_index < 0 or face.surface_index >= len(self.m_surfaces):
                continue
            srf = self.m_surfaces[face.surface_index]
            ts = TrimmedSurface()
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
                    if trim.curve_2d_index < 0 or trim.curve_2d_index >= len(self.m_curves_2d):
                        continue
                    crv = self.m_curves_2d[trim.curve_2d_index]
                    ndiv = max(crv.cv_count() * 4, 16) if crv.degree() > 1 else max(crv.cv_count() - 1, 4)
                    pts, _ = crv.divide_by_count(ndiv)
                    for k in range(len(pts) - 1):
                        loop_pts.append(pts[k])
                if len(loop_pts) >= 3:
                    loop_crv = NurbsCurve.create(True, 1, loop_pts)
                    if loop.type == BRepLoopType.Outer:
                        ts.m_outer_loop = loop_crv
                    else:
                        ts.m_inner_loops.append(loop_crv)
            face_mesh = ts.mesh()
            if face.reversed:
                for vk, vd in face_mesh.vertex.items():
                    n = vd.normal()
                    if n is not None:
                        vd.set_normal(-n[0], -n[1], -n[2])
            for fk, fverts in face_mesh.face.items():
                poly = [face_mesh.vertex[vi].position() for vi in fverts]
                all_polygons.append(poly)
        return Mesh.from_polylines(all_polygons)

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
        return self.m_surfaces[si].normal_at(u, v)

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
            faces.append({"loop_indices": f.loop_indices, "reversed": f.reversed, "surface_index": f.surface_index})
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
                b.m_faces.append(bf)
        return b

    def json_dump(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=4)

    @classmethod
    def json_load(cls, filepath):
        import json
        with open(filepath, 'r') as f:
            return cls.__jsonload__(json.load(f))

    def json_dumps(self):
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, s):
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
            proto.curves_2d.add().ParseFromString(c.pb_dumps())
        for c in self.m_curves_3d:
            proto.curves_3d.add().ParseFromString(c.pb_dumps())
        for s in self.m_surfaces:
            proto.surfaces.add().ParseFromString(s.pb_dumps())
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
