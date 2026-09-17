from __future__ import annotations
from .mesh import Mesh
from .point import Point


def _cross_2d(o: Point, a: Point, b: Point) -> float:
    """Twice the signed area of o-a-b in XY, positive for a left turn."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _extend_chain(points: list[Point], chain: list[int], i: int) -> None:
    """Appends point i to the chain after popping every tail that no longer turns left towards it."""

    while (
        len(chain) >= 2
        and _cross_2d(points[chain[-2]], points[chain[-1]], points[i]) <= 0.0
    ):
        chain.pop()

    chain.append(i)


def _signed_volume(a: Point, b: Point, c: Point, d: Point) -> float:
    """Six times the signed volume of a-b-c-d, positive when d is on the normal side of a-b-c."""
    return (b - a).cross(c - a).dot(d - a)


def _visible_from(
    indices: list[int], points: list[Point], a: Point, b: Point, c: Point
) -> list[int]:
    """Indices of the points above the face a-b-c."""

    result = []

    for i in indices:
        if _signed_volume(a, b, c, points[i]) > 1e-10:
            result.append(i)

    return result


def _farthest_point(
    indices: list[int], points: list[Point], a: Point, b: Point, c: Point
) -> int:
    """Index of the point highest above the face a-b-c, -1 when none is above."""

    best = -1
    best_volume = 0.0

    for i in indices:
        volume = _signed_volume(a, b, c, points[i])

        if volume > best_volume:
            best_volume = volume
            best = i

    return best


def _quickhull_faces(
    points: list[Point],
    indices: list[int],
    a: int,
    b: int,
    c: int,
    faces: list[list[int]],
) -> None:
    """Hull faces over a-b-c: the face itself when no candidate is above it, else the three faces to the farthest candidate, recursively."""

    visible = _visible_from(indices, points, points[a], points[b], points[c])
    apex = _farthest_point(visible, points, points[a], points[b], points[c])

    if apex == -1:
        faces.append([a, b, c])

        return

    _quickhull_faces(
        points,
        _visible_from(visible, points, points[a], points[b], points[apex]),
        a,
        b,
        apex,
        faces,
    )
    _quickhull_faces(
        points,
        _visible_from(visible, points, points[b], points[c], points[apex]),
        b,
        c,
        apex,
        faces,
    )
    _quickhull_faces(
        points,
        _visible_from(visible, points, points[c], points[a], points[apex]),
        c,
        a,
        apex,
        faces,
    )


class ConvexHull:
    """Convex hull: monotone chain in XY for 2D, quickhull for 3D."""

    @staticmethod
    def hull_2d(points: list[Point]) -> list[Point]:
        """Counter-clockwise hull of the points projected to XY, collinear points dropped; fewer than three points come back as given."""

        n = len(points)

        if n < 3:
            return list(points)

        order = sorted(range(n), key=lambda i: (points[i][0], points[i][1]))
        lower = []

        for i in order:
            _extend_chain(points, lower, i)

        upper = []

        for i in reversed(order):
            _extend_chain(points, upper, i)

        lower.pop()
        upper.pop()
        hull = []

        for i in lower:
            hull.append(points[i])

        for i in upper:
            hull.append(points[i])

        return hull

    @staticmethod
    def hull_3d(points: list[Point]) -> Mesh:
        """Triangle mesh of the hull with outward faces; fewer than four points give the points and, for three, one face."""

        n = len(points)
        mesh = Mesh()

        if n < 4:
            vkeys = []

            for point in points:
                vkeys.append(mesh.add_vertex(point))

            if n == 3:
                mesh.add_face(vkeys)

            return mesh

        p0 = 0

        for i in range(1, n):
            if points[i][0] < points[p0][0]:
                p0 = i

        p1 = 0

        for i in range(1, n):
            if (points[i] - points[p0]).magnitude_squared() > (
                points[p1] - points[p0]
            ).magnitude_squared():
                p1 = i

        axis = points[p1] - points[p0]
        p2 = -1
        best_distance = -1.0

        for i in range(n):
            if i == p0 or i == p1:
                continue

            distance = axis.cross(points[i] - points[p0]).magnitude_squared()

            if distance > best_distance:
                best_distance = distance
                p2 = i

        p3 = -1
        best_volume = -1.0

        for i in range(n):
            if i == p0 or i == p1 or i == p2:
                continue

            volume = abs(_signed_volume(points[p0], points[p1], points[p2], points[i]))

            if volume > best_volume:
                best_volume = volume
                p3 = i

        if p2 < 0 or p3 < 0 or best_distance <= 1e-20 or best_volume <= 1e-20:
            for point in points:
                mesh.add_vertex(point)

            return mesh

        if _signed_volume(points[p0], points[p1], points[p2], points[p3]) > 0.0:
            p1, p2 = p2, p1

        rest = []

        for i in range(n):
            if i != p0 and i != p1 and i != p2 and i != p3:
                rest.append(i)

        faces = []
        _quickhull_faces(points, rest, p0, p1, p2, faces)
        _quickhull_faces(points, rest, p0, p3, p1, faces)
        _quickhull_faces(points, rest, p1, p3, p2, faces)
        _quickhull_faces(points, rest, p2, p3, p0, faces)
        used = set()

        for face in faces:
            used.update(face)

        vkeys = [0] * n

        for i in sorted(used):
            vkeys[i] = mesh.add_vertex(points[i])

        for face in faces:
            mesh.add_face([vkeys[face[0]], vkeys[face[1]], vkeys[face[2]]])

        return mesh
