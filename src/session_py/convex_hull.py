from typing import List
from typing import Tuple
import math

from .mesh import Mesh
from .point import Point


class ConvexHull:
    """Convex hull computation: Graham scan (2D) and Quickhull (3D)."""

    # -------------------------------------------------------------------------
    # 2D Graham scan (projects to XY plane)
    # -------------------------------------------------------------------------

    @staticmethod
    def _cross_2d(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    @staticmethod
    def hull_2d(points: List[Point]) -> List[Point]:
        n = len(points)
        if n < 3:
            return list(points)
        pts = sorted(points, key=lambda p: (p[0], p[1]))
        lower = []
        for p in pts:
            while len(lower) >= 2 and ConvexHull._cross_2d(lower[-2], lower[-1], p) <= 0.0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and ConvexHull._cross_2d(upper[-2], upper[-1], p) <= 0.0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    # -------------------------------------------------------------------------
    # 3D Quickhull
    # -------------------------------------------------------------------------

    @staticmethod
    def _normal(a: Point, b: Point, c: Point) -> Tuple[float, float, float]:
        ax = b[0] - a[0]
        ay = b[1] - a[1]
        az = b[2] - a[2]
        bx = c[0] - a[0]
        by = c[1] - a[1]
        bz = c[2] - a[2]
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx
        return nx, ny, nz

    @staticmethod
    def _signed_volume(a: Point, b: Point, c: Point, d: Point) -> float:
        nx, ny, nz = ConvexHull._normal(a, b, c)
        return nx * (d[0] - a[0]) + ny * (d[1] - a[1]) + nz * (d[2] - a[2])

    @staticmethod
    def _farthest_point(pts_idx: List[int], points: List[Point], a: Point, b: Point, c: Point) -> int:
        best_idx = -1
        best_vol = 0.0
        for i in pts_idx:
            v = ConvexHull._signed_volume(a, b, c, points[i])
            if v > best_vol:
                best_vol = v
                best_idx = i
        return best_idx

    @staticmethod
    def _visible_from(pts_idx: List[int], points: List[Point], a: Point, b: Point, c: Point) -> List[int]:
        result = []
        for i in pts_idx:
            if ConvexHull._signed_volume(a, b, c, points[i]) > 1e-10:
                result.append(i)
        return result

    @staticmethod
    def _quickhull_3d_faces(points: List[Point], pts_idx: List[int], a: int, b: int, c: int, faces: List[Tuple[int, int, int]]) -> None:
        pa = points[a]
        pb = points[b]
        pc = points[c]
        visible = ConvexHull._visible_from(pts_idx, points, pa, pb, pc)
        if not visible:
            faces.append((a, b, c))
            return
        apex = ConvexHull._farthest_point(visible, points, pa, pb, pc)
        if apex == -1:
            faces.append((a, b, c))
            return
        v_ab = ConvexHull._visible_from(visible, points, pa, pb, points[apex])
        v_bc = ConvexHull._visible_from(visible, points, pb, pc, points[apex])
        v_ca = ConvexHull._visible_from(visible, points, pc, pa, points[apex])
        ConvexHull._quickhull_3d_faces(points, v_ab, a, b, apex, faces)
        ConvexHull._quickhull_3d_faces(points, v_bc, b, c, apex, faces)
        ConvexHull._quickhull_3d_faces(points, v_ca, c, a, apex, faces)

    @staticmethod
    def hull_3d(points: List[Point]) -> Mesh:
        n = len(points)
        if n < 4:
            mesh = Mesh()
            vkeys = [mesh.add_vertex(p) for p in points]
            if n == 3:
                mesh.add_face(vkeys)
            return mesh
        # Find initial tetrahedron
        # Pick point with min x
        p0 = min(range(n), key=lambda i: points[i][0])
        # Pick farthest from p0
        p1 = max(range(n), key=lambda i: (points[i][0] - points[p0][0]) ** 2 + (points[i][1] - points[p0][1]) ** 2 + (points[i][2] - points[p0][2]) ** 2)
        # Pick farthest from line p0-p1
        ax = points[p1][0] - points[p0][0]
        ay = points[p1][1] - points[p0][1]
        az = points[p1][2] - points[p0][2]

        def dist_to_line(i: int) -> float:
            bx = points[i][0] - points[p0][0]
            by = points[i][1] - points[p0][1]
            bz = points[i][2] - points[p0][2]
            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx
            return cx * cx + cy * cy + cz * cz

        p2 = max((i for i in range(n) if i != p0 and i != p1), key=dist_to_line)
        # Pick farthest from triangle p0-p1-p2
        p3 = max((i for i in range(n) if i not in (p0, p1, p2)), key=lambda i: abs(ConvexHull._signed_volume(points[p0], points[p1], points[p2], points[i])))

        # Ensure consistent winding: p3 should be on negative side of p0-p1-p2
        if ConvexHull._signed_volume(points[p0], points[p1], points[p2], points[p3]) > 0:
            p1, p2 = p2, p1

        rest = [i for i in range(n) if i not in (p0, p1, p2, p3)]

        faces: List[Tuple[int, int, int]] = []
        # 4 faces of tetrahedron, outward-facing
        ConvexHull._quickhull_3d_faces(points, rest, p0, p1, p2, faces)
        ConvexHull._quickhull_3d_faces(points, rest, p0, p3, p1, faces)
        ConvexHull._quickhull_3d_faces(points, rest, p1, p3, p2, faces)
        ConvexHull._quickhull_3d_faces(points, rest, p2, p3, p0, faces)

        # Build mesh from unique vertices
        mesh = Mesh()
        used = set()
        for f in faces:
            for idx in f:
                used.add(idx)
        idx_to_vkey = {}
        for idx in sorted(used):
            vk = mesh.add_vertex(points[idx])
            idx_to_vkey[idx] = vk
        for f in faces:
            mesh.add_face([idx_to_vkey[f[0]], idx_to_vkey[f[1]], idx_to_vkey[f[2]]])
        return mesh
