from __future__ import annotations
from .mesh import Mesh
from .point import Point
from .polyline import Polyline


def write_file_obj_to_string(mesh: Mesh) -> str:
    """OBJ text of the mesh: one v line per vertex, one f line per face with 1-based indices."""

    vertices, faces = mesh.to_vertices_and_faces()
    s = ""

    for p in vertices:
        s += f"v {p[0]} {p[1]} {p[2]}\n"

    for face in faces:
        if len(face) < 3:
            continue

        s += "f"

        for i in face:
            s += f" {i + 1}"

        s += "\n"

    return s


def write_file_obj(mesh: Mesh, filepath: str) -> None:
    """Write the mesh as an OBJ file."""
    with open(filepath, "w") as f:
        f.write(write_file_obj_to_string(mesh))


def read_file_obj_from_str(content: str) -> Mesh:
    """Mesh from OBJ text; v and f lines only, negative indices count from the end."""

    verts: list[Point] = []
    faces: list[list[int]] = []

    for line in content.splitlines():
        if not line or line[0] == "#":
            continue

        if line.startswith("v "):
            parts = line.split()

            if len(parts) < 4:
                continue

            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue

            verts.append(Point(x, y, z))
        elif line.startswith("f "):
            face: list[int] = []

            for tok in line.split()[1:]:
                try:
                    idx = int(tok.split("/")[0])
                except ValueError:
                    continue

                if idx == 0:
                    continue

                vidx = idx - 1 if idx > 0 else len(verts) + idx
                face.append(vidx)

            if len(face) >= 3:
                faces.append(face)

    return Mesh.from_vertices_and_faces(verts, faces)


def read_file_obj(filepath: str) -> Mesh:
    """Mesh from an OBJ file."""
    with open(filepath) as f:
        content = f.read()

    return read_file_obj_from_str(content)


def read_file_obj_polylines(filepath: str) -> list[Polyline]:
    """Polylines from the curv blocks of an OBJ file."""

    with open(filepath) as f:
        content = f.read()

    verts: list[Point] = []
    polylines: list[Polyline] = []
    curv: list[int] = []
    in_curv = False

    for line in content.splitlines():
        if not line or line[0] == "#":
            continue

        if line.startswith("v "):
            parts = line.split()

            if len(parts) < 4:
                continue

            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue

            verts.append(Point(x, y, z))
        elif line.startswith("curv "):
            curv = []

            for tok in line.split()[3:]:
                try:
                    curv.append(int(tok))
                except ValueError:
                    break

            in_curv = True
        elif line.startswith("end") and in_curv:
            pts: list[Point] = []

            for idx in curv:
                if 0 < idx <= len(verts):
                    pts.append(verts[idx - 1])

            if len(pts) >= 2:
                polylines.append(Polyline(pts))

            in_curv = False

    return polylines
