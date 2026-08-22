from __future__ import annotations
from typing import List
from typing import Optional
from .mesh import Mesh
from .point import Point


def _offset_planes(mesh: Mesh, distance: float):
    planes = {}
    for fk in mesh.faces():
        c = mesh.face_centroid(fk)
        n = mesh.face_normal(fk)
        if c is None or n is None:
            continue
        # plane as (a, b, c, d) where ax+by+cz+d=0, normal=(a,b,c), d=-(n·origin)
        a, b, c_ = n[0], n[1], n[2]
        ox = c[0] + distance * a
        oy = c[1] + distance * b
        oz = c[2] + distance * c_
        d = -(a * ox + b * oy + c_ * oz)
        planes[fk] = (a, b, c_, d)
    return planes


def _intersect_planes(planes, fallback: Point) -> Point:
    n = len(planes)
    if n == 0:
        return fallback
    if n == 1:
        a, b, c, d = planes[0]
        d_rhs = -d
        t = d_rhs - (a * fallback[0] + b * fallback[1] + c * fallback[2])
        return Point(fallback[0] + t * a, fallback[1] + t * b, fallback[2] + t * c)
    eps = 1e-8
    A = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    rhs = [0.0, 0.0, 0.0]
    for (a, b, c, d) in planes:
        d_rhs = -d
        A[0][0] += a * a
        A[0][1] += a * b
        A[0][2] += a * c
        A[1][0] += b * a
        A[1][1] += b * b
        A[1][2] += b * c
        A[2][0] += c * a
        A[2][1] += c * b
        A[2][2] += c * c
        rhs[0] += a * d_rhs
        rhs[1] += b * d_rhs
        rhs[2] += c * d_rhs
    A[0][0] += eps
    A[1][1] += eps
    A[2][2] += eps
    rhs[0] += eps * fallback[0]
    rhs[1] += eps * fallback[1]
    rhs[2] += eps * fallback[2]
    det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
           - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
           + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    if abs(det) < 1e-20:
        return fallback
    inv_det = 1.0 / det
    x = inv_det * (rhs[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                   - A[0][1] * (rhs[1] * A[2][2] - A[1][2] * rhs[2])
                   + A[0][2] * (rhs[1] * A[2][1] - A[1][1] * rhs[2]))
    y = inv_det * (A[0][0] * (rhs[1] * A[2][2] - A[1][2] * rhs[2])
                   - rhs[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                   + A[0][2] * (A[1][0] * rhs[2] - rhs[1] * A[2][0]))
    z = inv_det * (A[0][0] * (A[1][1] * rhs[2] - rhs[1] * A[2][1])
                   - A[0][1] * (A[1][0] * rhs[2] - rhs[1] * A[2][0])
                   + rhs[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    return Point(x, y, z)


def _offset_vertices(mesh: Mesh, planes: dict) -> dict:
    result = {}
    for vk in mesh.vertices():
        vp = mesh.vertex_point(vk)
        if vp is None:
            continue
        fkeys = mesh.vertex_faces(vk)
        if fkeys is None or len(fkeys) == 0:
            result[vk] = vp
            continue
        adj = [planes[fk] for fk in fkeys if fk in planes]
        result[vk] = _intersect_planes(adj, vp)
    return result


class MeshOffset:
    class Layers:
        def __init__(self, top: Mesh, bottom: Mesh, sides: Mesh):
            self.top = top
            self.bottom = bottom
            self.sides = sides

    @staticmethod
    def from_mesh(mesh: Mesh, distance: float) -> Mesh:
        planes = _offset_planes(mesh, distance)
        off_verts = _offset_vertices(mesh, planes)

        result = Mesh()
        bot_vmap = {}
        top_vmap = {}
        for vk in mesh.vertices():
            bot_vmap[vk] = result.add_vertex(mesh.vertex_point(vk))
            top_vmap[vk] = result.add_vertex(off_verts[vk])

        for fk in mesh.faces():
            fv = mesh.face_vertices(fk)
            bkeys = [bot_vmap[v] for v in fv]
            tkeys = [top_vmap[v] for v in fv]
            result.add_face(list(reversed(bkeys)))
            result.add_face(tkeys)

        for (u, v) in mesh.naked_edges(True):
            result.add_face([bot_vmap[u], bot_vmap[v], top_vmap[v], top_vmap[u]])

        return result

    @staticmethod
    def from_mesh_layers(mesh: Mesh, distance: float) -> "MeshOffset.Layers":
        planes = _offset_planes(mesh, distance)
        off_verts = _offset_vertices(mesh, planes)

        bot = Mesh()
        top = Mesh()
        sides = Mesh()
        bot_vmap = {}
        top_vmap = {}
        for vk in mesh.vertices():
            bot_vmap[vk] = bot.add_vertex(mesh.vertex_point(vk))
            top_vmap[vk] = top.add_vertex(off_verts[vk])

        for fk in mesh.faces():
            fv = mesh.face_vertices(fk)
            bkeys = [bot_vmap[v] for v in fv]
            tkeys = [top_vmap[v] for v in fv]
            bot.add_face(list(reversed(bkeys)))
            top.add_face(tkeys)

        s_bot = {}
        s_top = {}
        for (u, v) in mesh.naked_edges(True):
            if u not in s_bot:
                s_bot[u] = sides.add_vertex(mesh.vertex_point(u))
            if v not in s_bot:
                s_bot[v] = sides.add_vertex(mesh.vertex_point(v))
            if u not in s_top:
                s_top[u] = sides.add_vertex(off_verts[u])
            if v not in s_top:
                s_top[v] = sides.add_vertex(off_verts[v])
            sides.add_face([s_bot[u], s_bot[v], s_top[v], s_top[u]])

        return MeshOffset.Layers(top, bot, sides)
