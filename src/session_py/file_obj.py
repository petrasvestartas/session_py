from typing import List
from .mesh import Mesh
from .point import Point
from .polyline import Polyline


def write_file_obj_to_string(mesh: Mesh) -> str:
    vertices, faces = mesh.to_vertices_and_faces()
    s = ""
    for p in vertices:
        s += f"v {p.x} {p.y} {p.z}\n"
    for face in faces:
        if len(face) >= 3:
            idx = " ".join(str(i + 1) for i in face)
            s += f"f {idx}\n"
    return s


def write_file_obj(mesh: Mesh, filepath: str):
    with open(filepath, "w") as f:
        f.write(write_file_obj_to_string(mesh))


def read_file_obj(filepath: str) -> Mesh:
    with open(filepath, "r") as f:
        content = f.read()
    return read_file_obj_from_str(content)


def read_file_obj_from_str(content: str) -> Mesh:
    verts: List[Point] = []
    faces: List[List[int]] = []

    for raw in content.splitlines():
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


save_file_obj = write_file_obj
load_file_obj = read_file_obj
