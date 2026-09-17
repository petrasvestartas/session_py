from __future__ import annotations
from .matrix import Matrix
from .mesh import Mesh
from .plane import Plane
from .point import Point


def _intersect_planes(planes: list[Plane], fallback: Point) -> Point:
    """Least-squares point on the planes, fallback fills any free direction."""

    if len(planes) == 0:
        return fallback

    if len(planes) == 1:
        plane = planes[0]
        t = -plane.d - (
            plane.a * fallback[0] + plane.b * fallback[1] + plane.c * fallback[2]
        )

        return Point(
            fallback[0] + t * plane.a,
            fallback[1] + t * plane.b,
            fallback[2] + t * plane.c,
        )

    eps = 1e-8
    lhs = Matrix(3, 3)
    rhs = Matrix(3, 1)

    for plane in planes:
        row = [plane.a, plane.b, plane.c]

        for i in range(3):
            for j in range(3):
                lhs[i, j] += row[i] * row[j]

            rhs[i, 0] -= row[i] * plane.d

    for i in range(3):
        lhs[i, i] += eps
        rhs[i, 0] += eps * fallback[i]

    solution = lhs.solve(rhs)

    if solution is None:
        return fallback

    return Point(solution[0, 0], solution[1, 0], solution[2, 0])


def _boundary_edges(mesh: Mesh) -> list[tuple[int, int]]:
    """Naked edges wound the way their face walks them."""

    directed = set()

    for fkey in mesh.faces():
        vertices = mesh.face[fkey]

        for i in range(len(vertices)):
            directed.add((vertices[i], vertices[(i + 1) % len(vertices)]))

    edges = []

    for u, v in mesh.naked_edges(True):
        if (u, v) in directed:
            edges.append((u, v))
        else:
            edges.append((v, u))

    return edges


class MeshOffset:
    """Thick shell of a mesh: original faces, offset faces, quads on naked edges."""

    class Layers:
        """Top, bottom and side meshes of a shell."""

        def __init__(self, top: Mesh, bottom: Mesh, sides: Mesh):
            """Construct from the three meshes."""
            self.top = top  # Offset faces.
            self.bottom = bottom  # Reversed original faces.
            self.sides = sides  # One quad per naked edge.

    @staticmethod
    def from_mesh(mesh: Mesh, distance: float) -> Mesh:
        """One closed mesh: reversed bottom, offset top, one quad per naked edge."""

        planes = MeshOffset.offset_planes(mesh, distance)
        offsets = MeshOffset.offset_vertices(mesh, planes)
        result = Mesh()
        bottom = {}
        top = {}

        for vkey in mesh.vertices():
            bottom[vkey] = result.add_vertex(mesh.vertex_point(vkey))
            top[vkey] = result.add_vertex(offsets[vkey])

        for fkey in mesh.faces():
            vertices = mesh.face_vertices(fkey)
            bottom_face = []
            top_face = []

            for vkey in vertices:
                bottom_face.append(bottom[vkey])
                top_face.append(top[vkey])

            bottom_face.reverse()
            result.add_face(bottom_face)
            result.add_face(top_face)

        for u, v in _boundary_edges(mesh):
            result.add_face([bottom[u], bottom[v], top[v], top[u]])

        return result

    @staticmethod
    def from_mesh_layers(mesh: Mesh, distance: float) -> MeshOffset.Layers:
        """The same shell as three meshes: top, bottom and sides."""

        planes = MeshOffset.offset_planes(mesh, distance)
        offsets = MeshOffset.offset_vertices(mesh, planes)
        layers = MeshOffset.Layers(Mesh(), Mesh(), Mesh())
        bottom = {}
        top = {}

        for vkey in mesh.vertices():
            bottom[vkey] = layers.bottom.add_vertex(mesh.vertex_point(vkey))
            top[vkey] = layers.top.add_vertex(offsets[vkey])

        for fkey in mesh.faces():
            vertices = mesh.face_vertices(fkey)
            bottom_face = []
            top_face = []

            for vkey in vertices:
                bottom_face.append(bottom[vkey])
                top_face.append(top[vkey])

            bottom_face.reverse()
            layers.bottom.add_face(bottom_face)
            layers.top.add_face(top_face)

        side_bottom = {}
        side_top = {}

        for u, v in _boundary_edges(mesh):
            for vkey in (u, v):
                if vkey not in side_bottom:
                    side_bottom[vkey] = layers.sides.add_vertex(mesh.vertex_point(vkey))

                if vkey not in side_top:
                    side_top[vkey] = layers.sides.add_vertex(offsets[vkey])

            layers.sides.add_face(
                [side_bottom[u], side_bottom[v], side_top[v], side_top[u]]
            )

        return layers

    @staticmethod
    def offset_planes(mesh: Mesh, distance: float) -> dict[int, Plane]:
        """Plane of each face translated by distance along its normal, by face key."""

        planes = {}

        for fkey in mesh.faces():
            centroid = mesh.face_centroid(fkey)
            normal = mesh.face_normal(fkey)

            if centroid is None or normal is None:
                continue

            planes[fkey] = Plane.from_point_normal(centroid + normal * distance, normal)

        return planes

    @staticmethod
    def offset_vertices(mesh: Mesh, planes: dict[int, Plane]) -> dict[int, Point]:
        """Offset position of each vertex: least-squares meet of its face planes, by vertex key."""

        vertex_faces = {}

        for fkey in mesh.faces():
            for vkey in mesh.face[fkey]:
                vertex_faces.setdefault(vkey, []).append(fkey)

        result = {}

        for vkey in mesh.vertices():
            point = mesh.vertex_point(vkey)

            if point is None:
                continue

            adjacent = []

            for fkey in vertex_faces.get(vkey, []):
                if fkey in planes:
                    adjacent.append(planes[fkey])

            result[vkey] = _intersect_planes(adjacent, point)

        return result
