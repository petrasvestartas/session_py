from typing import List, Tuple
from .mesh import Mesh
from .point import Point
from .polyline import Polyline


def write_file_obj(mesh: Mesh, filepath: str):
    vertices, faces = mesh.to_vertices_and_faces()
    with open(filepath, "w") as f:
        for p in vertices:
            f.write(f"v {p.x} {p.y} {p.z}\n")
        for face in faces:
            if len(face) >= 3:
                idx = " ".join(str(i + 1) for i in face)
                f.write(f"f {idx}\n")


def read_file_obj(filepath: str) -> Mesh:
    verts: List[Point] = []
    faces: List[List[int]] = []

    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    verts.append(Point(x, y, z))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face: List[int] = []
                for tok in parts:
                    first = tok.split("/")[0]
                    if not first:
                        continue
                    idx = int(first)
                    if idx > 0:
                        vidx = idx - 1
                    else:
                        vidx = len(verts) + idx
                    face.append(vidx)
                if len(face) >= 3:
                    faces.append(face)

    mesh = Mesh()
    vkeys: List[int] = []
    for p in verts:
        vkeys.append(mesh.add_vertex(p))
    for f in faces:
        vlist = [vkeys[i] for i in f]
        mesh.add_face(vlist)
    return mesh


def read_file_obj_polylines(filepath: str) -> List[Polyline]:
    verts: List[Point] = []
    polylines: List[Polyline] = []
    curv_indices: List[int] = []
    in_curv = False
    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                verts.append(Point(float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("curv "):
                parts = line.split()[3:]
                curv_indices = [int(p) for p in parts]
                in_curv = True
            elif line.startswith("end") and in_curv:
                if curv_indices:
                    pts = [verts[i - 1] for i in curv_indices if 0 < i <= len(verts)]
                    if len(pts) >= 3:
                        polylines.append(Polyline(pts))
                in_curv = False
    return polylines


def pair_polylines(polylines: List[Polyline], search_radius: float = 500.0) -> List[Tuple[int, int]]:
    from .aabb import AABB
    from .spatial_rtree import SpatialRTree
    from .element_plate import ElementPlate
    NP = len(polylines)
    centroids: List[Point] = [Point(0, 0, 0)] * NP
    normals: List[List[float]] = [[0, 0, 0]] * NP
    aabbs = [None] * NP

    def open_pts(pl: Polyline) -> List[Point]:
        pts = pl.get_points()
        if len(pts) > 3:
            f, l = pts[0], pts[-1]
            if abs(f[0] - l[0]) < 1e-6 and abs(f[1] - l[1]) < 1e-6 and abs(f[2] - l[2]) < 1e-6:
                pts.pop()
        return pts

    tree = SpatialRTree()
    for i in range(NP):
        pts = open_pts(polylines[i])
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        centroids[i] = Point(cx, cy, cz)
        normals[i] = ElementPlate._polygon_normal(pts)
        aabbs[i] = AABB.from_polyline(polylines[i], search_radius)
        a = aabbs[i]
        tree.insert([a.cx - a.hx, a.cy - a.hy, a.cz - a.hz],
                    [a.cx + a.hx, a.cy + a.hy, a.cz + a.hz], i)

    paired = [False] * NP
    pairs: List[Tuple[int, int]] = []
    for i in range(NP):
        if paired[i]:
            continue
        best = [-1]
        best_d = [1e30]
        a = aabbs[i]
        ni = normals[i]
        ci = centroids[i]

        def cb(j, _i=i, _ni=ni, _ci=ci, _best=best, _best_d=best_d):
            if j <= _i or paired[j]:
                return True
            nj = normals[j]
            dot = _ni[0] * nj[0] + _ni[1] * nj[1] + _ni[2] * nj[2]
            if abs(dot) < 0.7:
                return True
            cj = centroids[j]
            dx = _ci[0] - cj[0]
            dy = _ci[1] - cj[1]
            dz = _ci[2] - cj[2]
            d = dx * dx + dy * dy + dz * dz
            if d < _best_d[0]:
                _best_d[0] = d
                _best[0] = j
            return True

        tree.search([a.cx - a.hx, a.cy - a.hy, a.cz - a.hz],
                    [a.cx + a.hx, a.cy + a.hy, a.cz + a.hz], cb)
        if best[0] >= 0:
            pairs.append((i, best[0]))
            paired[i] = True
            paired[best[0]] = True
    return pairs


save_file_obj = write_file_obj
load_file_obj = read_file_obj
