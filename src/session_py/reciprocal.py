from typing import List, Optional, Tuple
import copy

from .line import Line
from .plane import Plane
from .point import Point
from .vector import Vector
from .mesh import Mesh
from .xform import Xform
from .intersection import line_plane


def _get_lines(
    lines: List[Line],
    lp: List[Plane],
    fe: List[List[int]],
    end_planes: List,
    move: float = 0.0,
) -> List[Line]:
    ne = len(lines)
    nf = len(fe)

    moved = [copy.deepcopy(l) for l in lines]
    for i in range(ne):
        moved[i] += lp[i].y_axis * move

    pts = [[] for _ in range(ne)]
    pid = [[] for _ in range(ne)]

    for fi in range(nf):
        n = len(fe[fi])
        for j in range(n):
            cur  = fe[fi][j]
            prev = fe[fi][(j - 1) % n]
            nxt  = fe[fi][(j + 1) % n]

            p0 = line_plane(moved[cur], lp[prev], is_finite=False)
            if p0 is not None:
                pts[cur].append(p0)
                pid[cur].append(prev)

            p1 = line_plane(moved[cur], lp[nxt], is_finite=False)
            if p1 is not None:
                pts[cur].append(p1)
                pid[cur].append(nxt)

    out = [None] * ne
    end_planes.clear()
    for _ in range(ne):
        end_planes.append([None, None])

    for ei in range(ne):
        if len(pts[ei]) < 2:
            out[ei] = copy.deepcopy(moved[ei])
            end_planes[ei] = [copy.deepcopy(lp[ei]), copy.deepcopy(lp[ei])]
            continue

        np_ = len(pts[ei])
        ids = list(range(np_))
        ids.sort(key=lambda a: moved[ei].closest_point(pts[ei][a], limited=False)[0])

        s = ids[0]
        e = ids[-1]
        out[ei] = Line.from_points(pts[ei][s], pts[ei][e])

        ps = Plane(pts[ei][s], lp[pid[ei][s]].x_axis, lp[pid[ei][s]].y_axis)
        pe = Plane(pts[ei][e], lp[pid[ei][e]].x_axis, lp[pid[ei][e]].y_axis)
        end_planes[ei] = [ps, pe]

    return out


class ReciprocalResult:
    def __init__(self):
        self.center: List[Line] = []
        self.top: List[Line] = []
        self.bottom: List[Line] = []
        self.lineplanes: List[Plane] = []
        self.endplanes: List[List[Plane]] = []


class Reciprocal:
    @staticmethod
    def from_mesh(
        mesh: Mesh,
        angle: float,
        scale: float,
        use_ngon_normals: bool,
        height: float,
    ) -> ReciprocalResult:
        fkeys = mesh.faces()
        ekeys = mesh.edges()
        ne = len(ekeys)
        nf = len(fkeys)

        edge_idx = {}
        for i, (u, v) in enumerate(ekeys):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i

        fplane = {}
        for fk in fkeys:
            n = mesh.face_normal(fk)
            c = mesh.face_centroid(fk)
            if n is not None and c is not None:
                fplane[fk] = Plane.from_point_normal(c, n)

        fe = []
        for fi in range(nf):
            fk = fkeys[fi]
            edges_of_face = mesh.face_edges(fk) or []
            row = []
            for (u, v) in edges_of_face:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    row.append(edge_idx[key])
            fe.append(row)

        vecs = [Vector(0, 0, 0)] * ne
        for ei, (u, v) in enumerate(ekeys):
            adj = mesh.edge_faces(u, v)
            if not adj:
                continue
            sx, sy, sz = 0.0, 0.0, 0.0
            for fk in adj:
                if fk in fplane:
                    z = fplane[fk].z_axis
                    sx += z[0]; sy += z[1]; sz += z[2]
            count = len(adj)
            if count > 0:
                ax, ay, az = sx / count, sy / count, sz / count
                length = (ax*ax + ay*ay + az*az) ** 0.5
                if length > 1e-12:
                    vecs[ei] = Vector(ax / length, ay / length, az / length)

        lines = []
        for u, v in ekeys:
            el = mesh.edge_line(u, v)
            if el is not None:
                lines.append(copy.deepcopy(el))
            else:
                lines.append(Line())

        for ei in range(ne):
            v = vecs[ei]
            if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0:
                continue
            mid = lines[ei].center()
            lines[ei].xform = Xform.scale_uniform(mid, scale)
            lines[ei].transform()
            mid2 = lines[ei].center()
            axis_end = Point(mid2[0] + v[0], mid2[1] + v[1], mid2[2] + v[2])
            rot_axis = Line.from_points(mid2, axis_end)
            lines[ei].xform = Xform.rotation_around_line(rot_axis, angle)
            lines[ei].transform()

        lp = []
        for ei in range(ne):
            mid = lines[ei].center()
            d = lines[ei].to_direction()
            v = vecs[ei]
            if not (v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0):
                lp.append(Plane(mid, d, v))
            else:
                lp.append(Plane.from_point_normal(mid, d))

        result = ReciprocalResult()
        result.lineplanes = lp

        ep = []
        result.center = _get_lines(lines, lp, fe, ep, 0.0)
        result.endplanes = ep

        dummy = []
        result.top    = _get_lines(lines, lp, fe, dummy,  height)
        dummy2 = []
        result.bottom = _get_lines(lines, lp, fe, dummy2, -height)

        return result
