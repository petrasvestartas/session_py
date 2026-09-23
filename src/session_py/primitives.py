from __future__ import annotations
import math
import numpy as np

from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .nurbsknot import CurveInterpStyle
from .nurbsknot import CurveNurbsKnotStyle
from .plane import Plane
from .point import Point
from .vector import Vector
from .line import Line
from .xform import Xform
from .mesh import Mesh
from .tolerance import Tolerance
from .tolerance import PI
from . import nurbsknot


# ═══════════════════════════════════════════════════════════════════════════
# Rational quadratic circle pattern
# ═══════════════════════════════════════════════════════════════════════════

_CIRCLE_W = 0.7071067811865476
_CIRCLE_X = [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0]
_CIRCLE_Y = [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0]
_CIRCLE_WEIGHTS = [1.0, _CIRCLE_W, 1.0, _CIRCLE_W, 1.0, _CIRCLE_W, 1.0, _CIRCLE_W, 1.0]
_CIRCLE_NURBSKNOTS = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]


def _set_circle_row(
    srf: NurbsSurface,
    j: int,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    weight: float,
) -> None:
    """Row j of a surface set to a circle in the plane z = cz, weights scaled by weight."""

    for i in range(9):
        w = _CIRCLE_WEIGHTS[i] * weight
        px = cx + radius * _CIRCLE_X[i]
        py = cy + radius * _CIRCLE_Y[i]
        srf.set_cv_4d(i, j, px * w, py * w, cz * w, w)


# ═══════════════════════════════════════════════════════════════════════════
# Mesh helpers
# ═══════════════════════════════════════════════════════════════════════════


def _add_ring(vertices: list[Point], n: int, radius: float, z: float) -> None:
    """Appends n points of a circle of the given radius in the plane z."""

    for i in range(n):
        angle = 2.0 * PI * i / n
        vertices.append(Point(radius * math.cos(angle), radius * math.sin(angle), z))


def _dedup_face(face: list[int]) -> list[int]:
    """Face without consecutive duplicate vertices."""

    unique = []

    for k in range(len(face)):
        if face[k] != face[(k + 1) % len(face)]:
            unique.append(face[k])

    return unique


def _surface_grid(
    surface: NurbsSurface, u_count: int, v_count: int, mesh: Mesh
) -> list[list[int]]:
    """Vertex keys of the surface sampled on a (u_count + 1) x (v_count + 1) grid; seam and poles share keys."""

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    closed_u = surface.is_closed(0)
    singular_south = surface.is_singular(0)
    singular_north = surface.is_singular(2)
    grid = [[0] * (v_count + 1) for _ in range(u_count + 1)]

    for i in range(u_count + 1):
        u = u0 + (u1 - u0) * i / u_count

        for j in range(v_count + 1):
            v = v0 + (v1 - v0) * j / v_count

            if closed_u and i == u_count:
                grid[i][j] = grid[0][j]
            elif singular_south and j == 0 and i > 0:
                grid[i][j] = grid[0][0]
            elif singular_north and j == v_count and i > 0:
                grid[i][j] = grid[0][v_count]
            else:
                grid[i][j] = mesh.add_vertex(surface.point_at(u, v))

    return grid


def _surface_mid_grid(
    surface: NurbsSurface, u_count: int, v_count: int, t: float, mesh: Mesh
) -> list[list[int]]:
    """Vertex keys of the surface sampled at v offset by t cells on a (u_count + 1) x v_count grid; seam shares keys."""

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    closed_u = surface.is_closed(0)
    grid = [[0] * v_count for _ in range(u_count + 1)]

    for i in range(u_count + 1):
        u = u0 + (u1 - u0) * i / u_count

        for j in range(v_count):
            v = v0 + (v1 - v0) * (j + t) / v_count

            if closed_u and i == u_count:
                grid[i][j] = grid[0][j]
            else:
                grid[i][j] = mesh.add_vertex(surface.point_at(u, v))

    return grid


# ═══════════════════════════════════════════════════════════════════════════
# Curve compatibility
# ═══════════════════════════════════════════════════════════════════════════


def _merge_nurbsknot_vectors(a: list[float], b: list[float]) -> list[float]:
    """Sorted union of two nurbsknot vectors, equal values kept once."""

    tol = 1e-10
    merged = []
    i = 0
    j = 0

    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) < tol:
            merged.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1

    while i < len(a):
        merged.append(a[i])
        i += 1

    while j < len(b):
        merged.append(b[j])
        j += 1

    return merged


def _nurbsknot_vectors_equal(a: list[float], b: list[float]) -> bool:
    """True when both nurbsknot vectors match within 1e-10."""

    tol = 1e-10

    if len(a) != len(b):
        return False

    for i in range(len(a)):
        if abs(a[i] - b[i]) > tol:
            return False

    return True


def _make_curves_compatible(curves: list[NurbsCurve]) -> None:
    """Same degree, rationality, domain [0, 1] and nurbsknot vector for every curve."""

    if len(curves) < 2:
        return

    max_degree = 0
    any_rational = False

    for c in curves:
        max_degree = max(max_degree, c.degree())
        any_rational = any_rational or c.is_rational()

    for c in curves:
        if c.degree() < max_degree:
            c.increase_degree(max_degree)

        if any_rational:
            c.make_rational()

    compatible = True

    for i in range(1, len(curves)):
        if curves[i].cv_count() != curves[0].cv_count() or not _nurbsknot_vectors_equal(
            curves[i].get_nurbsknots(), curves[0].get_nurbsknots()
        ):
            compatible = False

    if compatible:
        return

    for c in curves:
        c.set_domain(0.0, 1.0)

    unified = curves[0].get_nurbsknots()

    for i in range(1, len(curves)):
        unified = _merge_nurbsknot_vectors(unified, curves[i].get_nurbsknots())

    tol = 1e-10

    for c in curves:
        nurbsknots = c.get_nurbsknots()
        ci = 0

        for ui in range(len(unified)):
            if ci < len(nurbsknots) and abs(nurbsknots[ci] - unified[ui]) < tol:
                ci += 1
            else:
                c.insert_nurbsknot(unified[ui], 1)


# ═══════════════════════════════════════════════════════════════════════════
# Planar helpers
# ═══════════════════════════════════════════════════════════════════════════


def _bilinear_patch(p00: Point, p10: Point, p01: Point, p11: Point) -> NurbsSurface:
    """Bilinear patch: u runs p00 to p10, v runs p00 to p01."""

    srf = NurbsSurface(3, False, 2, 2, 2, 2)
    srf.set_cv(0, 0, p00)
    srf.set_cv(1, 0, p10)
    srf.set_cv(0, 1, p01)
    srf.set_cv(1, 1, p11)

    return srf


def _longest_edge_dir(pts: list[Point]) -> Vector:
    """Unit direction of the longest edge of a closed polygon."""

    best = Vector(0.0, 0.0, 0.0)

    for i in range(len(pts)):
        edge = pts[(i + 1) % len(pts)] - pts[i]

        if edge.magnitude() > best.magnitude():
            best = edge

    return best.normalized()


def _bounded_patch(
    pts: list[Point], origin: Point, x_axis: Vector, y_axis: Vector
) -> NurbsSurface:
    """Bilinear patch in the frame covering the points with a 5% margin."""

    min_u = 1e30
    max_u = -1e30
    min_v = 1e30
    max_v = -1e30

    for pt in pts:
        d = pt - origin
        min_u = min(min_u, d.dot(x_axis))
        max_u = max(max_u, d.dot(x_axis))
        min_v = min(min_v, d.dot(y_axis))
        max_v = max(max_v, d.dot(y_axis))

    pad = max(max_u - min_u, max_v - min_v) * 0.05

    if pad < 1e-6:
        pad = 1.0

    min_u -= pad
    max_u += pad
    min_v -= pad
    max_v += pad

    return _bilinear_patch(
        origin + x_axis * min_u + y_axis * min_v,
        origin + x_axis * max_u + y_axis * min_v,
        origin + x_axis * min_u + y_axis * max_v,
        origin + x_axis * max_u + y_axis * max_v,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Loft helpers
# ═══════════════════════════════════════════════════════════════════════════


def _loft_section_params(curves: list[NurbsCurve]) -> list[float]:
    """Section parameters in [0, 1] from the mean CV distance between consecutive sections."""

    n = len(curves)
    cv_count = curves[0].cv_count()
    v_params = [0.0] * n

    for k in range(1, n):
        total = 0.0

        for i in range(cv_count):
            total += curves[k - 1].get_cv(i).distance(curves[k].get_cv(i))

        v_params[k] = v_params[k - 1] + total / cv_count

    total = v_params[n - 1]

    for k in range(n):
        v_params[k] = v_params[k] / total if total > 1e-14 else float(k) / (n - 1)

    return v_params


def _loft_nurbsknots(v_params: list[float], order_v: int) -> list[float]:
    """Clamped nurbsknot vector averaging the section parameters."""

    n = len(v_params)
    degree_v = order_v - 1
    nurbsknots = [v_params[0]] * (order_v + n - 2)

    for j in range(1, n - order_v + 1):
        total = 0.0

        for i in range(j, j + degree_v):
            total += v_params[i]

        nurbsknots[degree_v - 1 + j] = total / degree_v

    for i in range(n - 1, order_v + n - 2):
        nurbsknots[i] = v_params[n - 1]

    return nurbsknots


def _loft_basis_row(
    nurbsknots: list[float], order: int, cv_count: int, t: float
) -> list[float]:
    """Row of the collocation matrix: the cv_count basis values at t."""

    row = [0.0] * cv_count
    span = nurbsknot.find_span(order, cv_count, nurbsknots, t)
    base = span + order - 1

    if nurbsknots[base - 1] == nurbsknots[base]:
        row[span if t <= nurbsknots[base] else span + order - 1] = 1.0

        return row

    basis = nurbsknot.eval_basis(order, nurbsknots, span, t)

    for j in range(order):
        if span + j < cv_count:
            row[span + j] = basis[j]

    return row


def _solve_linear(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Solves a x = b by Gaussian elimination with partial pivoting, one right-hand side per column of b."""

    a = [list(row) for row in a]
    b = [list(row) for row in b]
    n = len(a)
    dim = len(b[0])

    for col in range(n):
        max_row = col

        for row in range(col + 1, n):
            if abs(a[row][col]) > abs(a[max_row][col]):
                max_row = row

        if abs(a[max_row][col]) < 1e-14:
            continue

        a[col], a[max_row] = a[max_row], a[col]
        b[col], b[max_row] = b[max_row], b[col]

        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]

            for c in range(col, n):
                a[row][c] -= factor * a[col][c]

            for d in range(dim):
                b[row][d] -= factor * b[col][d]

    x = [[0.0] * dim for _ in range(n)]

    for row in range(n - 1, -1, -1):
        for d in range(dim):
            x[row][d] = b[row][d]

            for c in range(row + 1, n):
                x[row][d] -= a[row][c] * x[c][d]

            if abs(a[row][row]) > 1e-14:
                x[row][d] /= a[row][row]

    return x


# ═══════════════════════════════════════════════════════════════════════════
# Sweep helpers
# ═══════════════════════════════════════════════════════════════════════════


def _lerp_point(a: Point, b: Point, s: float) -> Point:
    """Point at fraction s from a to b."""
    return a + (b - a) * s


def _lerp_vector(a: Vector, b: Vector, s: float) -> Vector:
    """Vector at fraction s from a to b."""
    return a + (b - a) * s


def _profile_to_xy(profile: NurbsCurve) -> Xform:
    """World to the profile frame: centroid origin, x toward the start point, z the profile normal."""

    centroid = Vector(0.0, 0.0, 0.0)

    for i in range(profile.cv_count()):
        centroid += profile.get_cv(i) - Point(0.0, 0.0, 0.0)

    origin = Point(0.0, 0.0, 0.0) + centroid / profile.cv_count()
    t0, t1 = profile.domain()
    pa = profile.point_at(t0)
    pb = profile.point_at(t0 + (t1 - t0) / 3.0)
    pc = profile.point_at(t0 + 2.0 * (t1 - t0) / 3.0)
    normal = (pb - pa).cross(pc - pa)

    if not normal.normalize_self():
        normal = Vector(1.0, 0.0, 0.0)

    x_axis = pa - origin

    if not x_axis.normalize_self():
        x_axis = Vector(0.0, 1.0, 0.0)

    x_axis -= normal * x_axis.dot(normal)

    if not x_axis.normalize_self():
        x_axis = Vector(0.0, 1.0, 0.0)

    return Xform.world_to_frame(origin, x_axis, normal.cross(x_axis), normal)


def _shape_plane(shape: NurbsCurve) -> Plane:
    """Frame of a sweep shape: start point origin, x along the chord, z across it."""

    start = shape.point_at_start()
    direction = shape.point_at_end() - start

    if not direction.normalize_self():
        direction = Vector(1.0, 0.0, 0.0)

    side = direction.cross(Vector(0.0, 0.0, 1.0))

    if side.magnitude() < 1e-10:
        side = direction.cross(Vector(0.0, 1.0, 0.0))

    return Plane(start, direction, side.cross(direction))


def _shape_width(shape: NurbsCurve) -> float:
    """Chord length of a shape, 1 when degenerate."""

    width = shape.point_at_start().distance(shape.point_at_end())

    return 1.0 if width < 1e-14 else width


# ═══════════════════════════════════════════════════════════════════════════
# Edge helpers
# ═══════════════════════════════════════════════════════════════════════════


def _chain_curves(input_curves: list[NurbsCurve]) -> list[NurbsCurve]:
    """Curves ordered head to tail, reversed where needed; empty when they do not close a loop."""

    tol = 1e-6
    loop = [input_curves[0]]
    used = [False] * len(input_curves)
    used[0] = True

    for _step in range(1, len(input_curves)):
        tail = loop[-1].point_at_end()
        found = False

        for i in range(len(input_curves)):
            if found or used[i]:
                continue

            next_curve = input_curves[i].duplicate()

            if (
                next_curve.point_at_start().distance(tail) >= tol
                and next_curve.point_at_end().distance(tail) < tol
            ):
                next_curve.reverse()

            if next_curve.point_at_start().distance(tail) >= tol:
                continue

            loop.append(next_curve)
            used[i] = True
            found = True

        if not found:
            return []

    if loop[-1].point_at_end().distance(loop[0].point_at_start()) > tol:
        return []

    return loop


def _normalized_greville(curve: NurbsCurve) -> list[float]:
    """Greville abcissae mapped to [0, 1]."""

    grev = curve.get_greville_abcissae()
    t0, t1 = curve.domain()

    for i in range(len(grev)):
        grev[i] = (grev[i] - t0) / (t1 - t0) if t1 > t0 else 0.0

    return grev


class Primitives:
    """Factory for primitive meshes, NURBS curves and NURBS surfaces."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Mesh primitives
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def arrow_mesh(line: Line, radius: float) -> Mesh:
        """Arrow mesh along a line: cylinder body over 80% of the length, cone head of 1.5x radius over 20%."""

        start = line.start()
        axis = line.to_vector()
        length = line.length()
        body = Primitives._line_frame(line, start + axis * 0.4) * Xform.scale_xyz(
            radius * 2.0, radius * 2.0, length * 0.8
        )
        head = Primitives._line_frame(line, start + axis * 0.9) * Xform.scale_xyz(
            radius * 3.0, radius * 3.0, length * 0.2
        )
        mesh = Mesh()
        Primitives._add_geometry(mesh, Primitives._unit_cylinder_geometry(), body)
        Primitives._add_geometry(mesh, Primitives._unit_cone_geometry(), head)

        return mesh

    @staticmethod
    def cylinder_mesh(line: Line, radius: float) -> Mesh:
        """Ten-sided cylinder mesh along a line."""

        start = line.start()
        axis = line.to_vector()
        xform = Primitives._line_frame(line, start + axis * 0.5) * Xform.scale_xyz(
            radius * 2.0, radius * 2.0, line.length()
        )
        mesh = Mesh()
        Primitives._add_geometry(mesh, Primitives._unit_cylinder_geometry(), xform)

        return mesh

    @staticmethod
    def capsule_mesh(line: Line, radius: float) -> Mesh:
        """Ten-sided cylinder mesh with hemispherical caps along a line."""

        mesh = Mesh()
        Primitives._add_geometry(
            mesh,
            Primitives._capsule_geometry(line.length(), radius),
            Primitives._line_frame(line, line.start()),
        )

        return mesh

    @staticmethod
    def edge_pipes(mesh: Mesh, radius: float) -> list[Mesh]:
        """One capsule mesh per edge, colored by mesh.linecolors[i]."""

        edges = mesh.edges()
        colors = mesh.get_linecolors()
        count = min(len(edges), len(colors))
        pipes = []

        for i in range(count):
            u, v = edges[i]
            pipe = Primitives.capsule_mesh(
                Line.from_points(mesh.vertex[u].position(), mesh.vertex[v].position()),
                radius,
            )

            pipe.set_facecolors([colors[i]] * pipe.number_of_faces())
            pipes.append(pipe)

        return pipes

    @staticmethod
    def tetrahedron(edge: float = 2.0) -> Mesh:
        """Tetrahedron mesh (4 triangles) with the given edge length."""

        a = edge / 2.0
        h = edge * math.sqrt(2.0 / 3.0)
        r = edge / math.sqrt(3.0)
        z0 = -h / 4.0
        z1 = 3.0 * h / 4.0
        faces = [
            [
                Point(a, -r / 2.0, z0),
                Point(-a, -r / 2.0, z0),
                Point(0.0, r, z0),
            ],
            [
                Point(0.0, 0.0, z1),
                Point(-a, -r / 2.0, z0),
                Point(a, -r / 2.0, z0),
            ],
            [
                Point(0.0, 0.0, z1),
                Point(0.0, r, z0),
                Point(-a, -r / 2.0, z0),
            ],
            [
                Point(0.0, 0.0, z1),
                Point(a, -r / 2.0, z0),
                Point(0.0, r, z0),
            ],
        ]

        return Mesh.from_polylines(faces, 1e-10)

    @staticmethod
    def cube(edge: float = 2.0) -> Mesh:
        """Cube mesh (6 quads) with the given edge length."""

        a = edge / 2.0
        v0 = Point(-a, -a, -a)
        v1 = Point(a, -a, -a)
        v2 = Point(a, a, -a)
        v3 = Point(-a, a, -a)
        v4 = Point(-a, -a, a)
        v5 = Point(a, -a, a)
        v6 = Point(a, a, a)
        v7 = Point(-a, a, a)
        faces = [
            [v3, v2, v1, v0],
            [v4, v5, v6, v7],
            [v0, v1, v5, v4],
            [v2, v3, v7, v6],
            [v0, v4, v7, v3],
            [v1, v2, v6, v5],
        ]

        return Mesh.from_polylines(faces, 1e-10)

    @staticmethod
    def octahedron(edge: float = 2.0) -> Mesh:
        """Octahedron mesh (8 triangles) with the given edge length."""

        a = edge / math.sqrt(2.0)
        px = Point(a, 0.0, 0.0)
        nx = Point(-a, 0.0, 0.0)
        py = Point(0.0, a, 0.0)
        ny = Point(0.0, -a, 0.0)
        pz = Point(0.0, 0.0, a)
        nz = Point(0.0, 0.0, -a)
        faces = [
            [pz, px, py],
            [pz, py, nx],
            [pz, nx, ny],
            [pz, ny, px],
            [nz, py, px],
            [nz, nx, py],
            [nz, ny, nx],
            [nz, px, ny],
        ]

        return Mesh.from_polylines(faces, 1e-10)

    @staticmethod
    def icosahedron(edge: float = 2.0) -> Mesh:
        """Icosahedron mesh (20 triangles) with the given edge length."""

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        s = edge / 2.0
        sp = s * phi
        verts = [
            Point(-s, sp, 0.0),
            Point(s, sp, 0.0),
            Point(-s, -sp, 0.0),
            Point(s, -sp, 0.0),
            Point(0.0, -s, sp),
            Point(0.0, s, sp),
            Point(0.0, -s, -sp),
            Point(0.0, s, -sp),
            Point(sp, 0.0, -s),
            Point(sp, 0.0, s),
            Point(-sp, 0.0, -s),
            Point(-sp, 0.0, s),
        ]
        idx = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
        faces = []

        for f in idx:
            faces.append([verts[f[0]], verts[f[1]], verts[f[2]]])

        return Mesh.from_polylines(faces, 1e-10)

    # ═══════════════════════════════════════════════════════════════════════════
    # Curve primitives
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def circle(cx: float, cy: float, cz: float, radius: float) -> NurbsCurve:
        """Full circle as a rational quadratic NURBS (9 CVs)."""
        return Primitives.ellipse(cx, cy, cz, radius, radius)

    @staticmethod
    def ellipse(
        cx: float, cy: float, cz: float, major_radius: float, minor_radius: float
    ) -> NurbsCurve:
        """Full ellipse as a rational quadratic NURBS (9 CVs)."""

        curve = NurbsCurve(3, True, 3, 9)

        for i in range(10):
            curve.set_nurbsknot(i, _CIRCLE_NURBSKNOTS[i])

        for i in range(9):
            w = _CIRCLE_WEIGHTS[i]
            px = cx + major_radius * _CIRCLE_X[i]
            py = cy + minor_radius * _CIRCLE_Y[i]
            curve.set_cv_4d(i, px * w, py * w, cz * w, w)

        return curve

    @staticmethod
    def arc(start: Point, mid: Point, end: Point) -> NurbsCurve:
        """Circular arc from start through the arc midpoint to end as a rational quadratic NURBS; a line when collinear."""

        chord = end - start
        chord_mid = start + chord * 0.5
        sagitta = mid - chord_mid

        if chord.cross(sagitta).magnitude() < Tolerance.ZERO_TOLERANCE:
            return NurbsCurve.create(False, 1, [start, end])

        h = chord.magnitude() * 0.5
        s = sagitta.magnitude()
        radius = (h * h + s * s) / (2.0 * s)
        w = (radius - s) / radius

        if abs(w) < Tolerance.ZERO_TOLERANCE:
            w = Tolerance.ZERO_TOLERANCE

        curve = NurbsCurve(3, True, 3, 3)
        curve.m_nurbsknot = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        curve.set_cv_4d(0, start[0], start[1], start[2], 1.0)
        weighted = chord_mid * w + sagitta
        curve.set_cv_4d(1, weighted[0], weighted[1], weighted[2], w)
        curve.set_cv_4d(2, end[0], end[1], end[2], 1.0)

        return curve

    @staticmethod
    def parabola(p0: Point, p1: Point, p2: Point) -> NurbsCurve:
        """Parabola through three points with p1 as the apex, as a quadratic NURBS."""

        curve = NurbsCurve(3, False, 3, 3)
        curve.m_nurbsknot = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        curve.set_cv(0, p0)
        curve.set_cv(
            1,
            Point(
                2.0 * p1[0] - (p0[0] + p2[0]) / 2.0,
                2.0 * p1[1] - (p0[1] + p2[1]) / 2.0,
                2.0 * p1[2] - (p0[2] + p2[2]) / 2.0,
            ),
        )
        curve.set_cv(2, p2)

        return curve

    @staticmethod
    def hyperbola(center: Point, a: float, b: float, extent: float) -> NurbsCurve:
        """Hyperbola x = a cosh(t), y = b sinh(t) for t in [-extent, extent] as a cubic NURBS through 9 points."""

        segments = 8
        points = []

        for i in range(segments + 1):
            t = -extent + 2.0 * extent * i / segments
            points.append(
                Point(
                    center[0] + a * math.cosh(t),
                    center[1] + b * math.sinh(t),
                    center[2],
                )
            )

        curve = NurbsCurve()

        if not curve.create_clamped_uniform(3, 4, points, 1.0):
            return NurbsCurve()

        return curve

    @staticmethod
    def spiral(
        start_radius: float, end_radius: float, pitch: float, turns: float
    ) -> NurbsCurve:
        """Helix with linearly varying radius as a cubic NURBS, 8 points per turn."""

        segments = max(4, int(turns * 8))
        points = []

        for i in range(segments + 1):
            t = float(i) / segments
            angle = t * turns * 2.0 * PI
            r = start_radius + t * (end_radius - start_radius)
            points.append(
                Point(r * math.cos(angle), r * math.sin(angle), t * turns * pitch)
            )

        curve = NurbsCurve()

        if not curve.create_clamped_uniform(3, 4, points, 1.0):
            return NurbsCurve()

        return curve

    @staticmethod
    def create_interpolated(
        points: list[Point],
        parameterization: CurveNurbsKnotStyle = CurveNurbsKnotStyle.Chord,
        end_condition: CurveInterpStyle = CurveInterpStyle.Rhino,
    ) -> NurbsCurve:
        """Interpolated cubic NURBS through points."""
        return NurbsCurve.create_interpolated(points, parameterization, end_condition)

    # ═══════════════════════════════════════════════════════════════════════════
    # Surface primitives
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def cylinder_surface(
        cx: float, cy: float, cz: float, radius: float, height: float
    ) -> NurbsSurface:
        """Rational cylinder surface of degree 2x1 around the z axis through (cx, cy, cz)."""

        srf = NurbsSurface(3, True, 3, 2, 9, 2)

        for i in range(10):
            srf.set_nurbsknot(0, i, _CIRCLE_NURBSKNOTS[i])

        _set_circle_row(srf, 0, cx, cy, cz, radius, 1.0)
        _set_circle_row(srf, 1, cx, cy, cz + height, radius, 1.0)

        return srf

    @staticmethod
    def cone_surface(
        cx: float, cy: float, cz: float, radius: float, height: float
    ) -> NurbsSurface:
        """Rational cone surface of degree 2x1 with the apex at cz + height."""

        srf = NurbsSurface(3, True, 3, 2, 9, 2)

        for i in range(10):
            srf.set_nurbsknot(0, i, _CIRCLE_NURBSKNOTS[i])

        _set_circle_row(srf, 0, cx, cy, cz, radius, 1.0)
        _set_circle_row(srf, 1, cx, cy, cz + height, 0.0, 1.0)

        return srf

    @staticmethod
    def torus_surface(
        cx: float, cy: float, cz: float, major_radius: float, minor_radius: float
    ) -> NurbsSurface:
        """Rational torus surface of degree 2x2."""

        srf = NurbsSurface(3, True, 3, 3, 9, 9)

        for i in range(10):
            srf.set_nurbsknot(0, i, _CIRCLE_NURBSKNOTS[i])
            srf.set_nurbsknot(1, i, _CIRCLE_NURBSKNOTS[i])

        for j in range(9):
            _set_circle_row(
                srf,
                j,
                cx,
                cy,
                cz + minor_radius * _CIRCLE_Y[j],
                major_radius + minor_radius * _CIRCLE_X[j],
                _CIRCLE_WEIGHTS[j],
            )

        return srf

    @staticmethod
    def sphere_surface(cx: float, cy: float, cz: float, radius: float) -> NurbsSurface:
        """Rational sphere surface of degree 2x2 with poles on the z axis."""

        lat_r = [0.0, 1.0, 1.0, 1.0, 0.0]
        lat_z = [-1.0, -1.0, 0.0, 1.0, 1.0]
        lat_w = [1.0, _CIRCLE_W, 1.0, _CIRCLE_W, 1.0]
        v_nurbsknots = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        srf = NurbsSurface(3, True, 3, 3, 9, 5)

        for i in range(10):
            srf.set_nurbsknot(0, i, _CIRCLE_NURBSKNOTS[i])

        for i in range(6):
            srf.set_nurbsknot(1, i, v_nurbsknots[i])

        for j in range(5):
            _set_circle_row(
                srf, j, cx, cy, cz + radius * lat_z[j], radius * lat_r[j], lat_w[j]
            )

        return srf

    @staticmethod
    def quad_sphere(
        cx: float, cy: float, cz: float, radius: float
    ) -> list[NurbsSurface]:
        """Sphere as 6 rational biquadratic patches projected from the cube faces."""

        a = radius / math.sqrt(3.0)
        e = radius * math.sqrt(3.0) / 2.0
        wk = math.sqrt(2.0 / 3.0)
        wc = (
            -72.0
            - 32.0 * math.sqrt(6.0)
            + 48.0 * math.sqrt(3.0)
            + 56.0 * math.sqrt(2.0)
        ) / (
            48.0
            * (1.0 + math.sqrt(2.0 / 3.0) - 1.0 / math.sqrt(3.0) - 1.0 / math.sqrt(2.0))
        )
        k = radius * (
            1.0 - 1.0 / math.sqrt(3.0) + 2.0 * math.sqrt(2.0 / 3.0) - math.sqrt(2.0)
        )
        h = radius + k / wc
        zf = [
            [[-a, -a, a, 1.0], [-e, 0.0, e, wk], [-a, a, a, 1.0]],
            [[0.0, -e, e, wk], [0.0, 0.0, h, wc], [0.0, e, e, wk]],
            [[a, -a, a, 1.0], [e, 0.0, e, wk], [a, a, a, 1.0]],
        ]
        rot = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        ]
        faces = []

        for f in range(6):
            srf = NurbsSurface(3, True, 3, 3, 3, 3)

            for i in range(3):
                for j in range(3):
                    p = zf[i][j]
                    rx = (
                        rot[f][0][0] * p[0]
                        + rot[f][0][1] * p[1]
                        + rot[f][0][2] * p[2]
                        + cx
                    )
                    ry = (
                        rot[f][1][0] * p[0]
                        + rot[f][1][1] * p[1]
                        + rot[f][1][2] * p[2]
                        + cy
                    )
                    rz = (
                        rot[f][2][0] * p[0]
                        + rot[f][2][1] * p[1]
                        + rot[f][2][2] * p[2]
                        + cz
                    )
                    srf.set_cv_4d(i, j, rx * p[3], ry * p[3], rz * p[3], p[3])

            faces.append(srf)

        return faces

    @staticmethod
    def wave_surface(size: float, amplitude: float) -> NurbsSurface:
        """Tileable egg-crate surface z = amplitude sin(2 pi x / size) sin(2 pi y / size) as a 13x13 cubic NURBS."""

        n = 13
        pts = []

        for i in range(n):
            u = float(i) / (n - 1)

            for j in range(n):
                v = float(j) / (n - 1)
                pts.append(
                    Point(
                        size * u,
                        size * v,
                        amplitude * math.sin(2.0 * PI * u) * math.sin(2.0 * PI * v),
                    )
                )

        return NurbsSurface.create(False, False, 3, 3, n, n, pts)

    # ═══════════════════════════════════════════════════════════════════════════
    # Surface factories
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def create_ruled(curve_a: NurbsCurve, curve_b: NurbsCurve) -> NurbsSurface:
        """Ruled surface between two curves."""

        if not curve_a.is_valid() or not curve_b.is_valid():
            return NurbsSurface()

        curves = [curve_a.duplicate(), curve_b.duplicate()]
        curves[0].set_domain(0.0, 1.0)
        curves[1].set_domain(0.0, 1.0)
        _make_curves_compatible(curves)

        cv_count_u = curves[0].cv_count()
        is_rat = curves[0].is_rational()
        surface = NurbsSurface(3, is_rat, curves[0].order(), 2, cv_count_u, 2)

        if not surface.is_valid():
            return NurbsSurface()

        for i in range(surface.nurbsknot_count(0)):
            surface.set_nurbsknot(0, i, curves[0].nurbsknot(i))

        for i in range(cv_count_u):
            for j in range(2):
                if is_rat:
                    x, y, z, w = curves[j].get_cv_4d(i)
                    surface.set_cv_4d(i, j, x, y, z, w)
                else:
                    surface.set_cv(i, j, curves[j].get_cv(i))

        return surface

    @staticmethod
    def create_extrusion(curve: NurbsCurve, direction: Vector) -> NurbsSurface:
        """Extrusion of a curve along a direction."""

        if not curve.is_valid():
            return NurbsSurface()

        translated = curve.duplicate()
        translated.transform(
            Xform.translation(direction[0], direction[1], direction[2])
        )

        return Primitives.create_ruled(curve, translated)

    @staticmethod
    def create_planar(boundary: NurbsCurve) -> NurbsSurface:
        """Bilinear planar patch containing a closed boundary curve."""

        if not boundary.is_valid():
            return NurbsSurface()

        pts = []

        for i in range(boundary.cv_count()):
            pts.append(boundary.get_cv(i))

        if len(pts) >= 2 and pts[0].distance(pts[-1]) < 1e-10:
            pts.pop()

        if len(pts) < 3:
            return NurbsSurface()

        if boundary.degree() <= 1 and len(pts) == 3:
            return _bilinear_patch(pts[0], pts[1], pts[0], pts[2])

        if boundary.degree() <= 1 and len(pts) == 4:
            return _bilinear_patch(pts[0], pts[1], pts[3], pts[2])

        if boundary.degree() <= 1:
            normal = (pts[1] - pts[0]).cross(pts[2] - pts[0])

            if not normal.normalize_self():
                return NurbsSurface()

            x_axis = _longest_edge_dir(pts)
            y_axis = normal.cross(x_axis)

            if not y_axis.normalize_self():
                return NurbsSurface()

            return _bounded_patch(pts, pts[0], x_axis, y_axis)

        samples, _params = boundary.divide_by_count(max(20, boundary.cv_count() * 4))
        plane = Plane.from_points_pca(samples)

        if plane.z_axis.magnitude() < 1e-10:
            return NurbsSurface()

        return _bounded_patch(samples, plane.origin, plane.x_axis, plane.y_axis)

    @staticmethod
    def create_loft(input_curves: list[NurbsCurve], degree_v: int = 3) -> NurbsSurface:
        """Loft through section curves, interpolating them in v."""

        if len(input_curves) < 2:
            return NurbsSurface()

        for c in input_curves:
            if not c.is_valid():
                return NurbsSurface()

        curves = []

        for c in input_curves:
            curves.append(c.duplicate())

        _make_curves_compatible(curves)

        n = len(curves)
        cv_count_u = curves[0].cv_count()
        is_rat = curves[0].is_rational()
        order_v = max(1, min(degree_v, n - 1)) + 1
        v_params = _loft_section_params(curves)
        nurbsknots_v = _loft_nurbsknots(v_params, order_v)
        surface = NurbsSurface(3, is_rat, curves[0].order(), order_v, cv_count_u, n)

        if not surface.is_valid():
            return NurbsSurface()

        for i in range(surface.nurbsknot_count(0)):
            surface.set_nurbsknot(0, i, curves[0].nurbsknot(i))

        for i in range(surface.nurbsknot_count(1)):
            surface.set_nurbsknot(1, i, nurbsknots_v[i])

        basis = []

        for k in range(n):
            basis.append(_loft_basis_row(nurbsknots_v, order_v, n, v_params[k]))

        dim = 4 if is_rat else 3

        for i in range(cv_count_u):
            rhs = [[0.0] * dim for _ in range(n)]

            for k in range(n):
                if is_rat:
                    x, y, z, w = curves[k].get_cv_4d(i)
                    rhs[k] = [x, y, z, w]
                else:
                    p = curves[k].get_cv(i)
                    rhs[k] = [p[0], p[1], p[2]]

            q = _solve_linear(basis, rhs)

            for j in range(n):
                if is_rat:
                    surface.set_cv_4d(i, j, q[j][0], q[j][1], q[j][2], q[j][3])
                else:
                    surface.set_cv(i, j, Point(q[j][0], q[j][1], q[j][2]))

        return surface

    @staticmethod
    def create_revolve(
        profile: NurbsCurve,
        axis_origin: Point,
        axis_direction: Vector,
        angle: float = 2.0 * PI,
    ) -> NurbsSurface:
        """Surface of revolution of a profile around an axis."""

        if not profile.is_valid():
            return NurbsSurface()

        axis = Vector(axis_direction[0], axis_direction[1], axis_direction[2])

        if not axis.normalize_self():
            return NurbsSurface()

        angle = min(abs(angle), 2.0 * PI)

        if angle < 1e-14:
            return NurbsSurface()

        n_arcs = 4

        if angle <= PI / 2.0 + 1e-10:
            n_arcs = 1
        elif angle <= PI + 1e-10:
            n_arcs = 2
        elif angle <= 3.0 * PI / 2.0 + 1e-10:
            n_arcs = 3

        d_theta = angle / n_arcs
        w_mid = math.cos(d_theta / 2.0)
        n_u = 2 * n_arcs + 1
        cv_count_v = profile.cv_count()
        surface = NurbsSurface(3, True, 3, profile.order(), n_u, cv_count_v)

        if not surface.is_valid():
            return NurbsSurface()

        for i in range(surface.nurbsknot_count(0)):
            surface.set_nurbsknot(
                0, i, angle if i // 2 == n_arcs else (i // 2) * d_theta
            )

        for i in range(surface.nurbsknot_count(1)):
            surface.set_nurbsknot(1, i, profile.nurbsknot(i))

        for j in range(cv_count_v):
            p = profile.get_cv(j)
            profile_w = profile.weight(j) if profile.is_rational() else 1.0
            center = axis_origin + axis * (p - axis_origin).dot(axis)
            x_local = p - center
            r = x_local.magnitude()

            if r > 1e-14:
                x_local /= r

            y_local = axis.cross(x_local)

            for i in range(n_u):
                shoulder = i % 2 == 1
                theta = (i // 2) * d_theta + (d_theta / 2.0 if shoulder else 0.0)
                w = (w_mid if shoulder else 1.0) * profile_w
                q = center + (x_local * math.cos(theta) + y_local * math.sin(theta)) * (
                    r / w_mid if shoulder else r
                )
                surface.set_cv_4d(i, j, q[0] * w, q[1] * w, q[2] * w, w)

        return surface

    @staticmethod
    def create_sweep1(rail: NurbsCurve, profile: NurbsCurve) -> NurbsSurface:
        """Sweep of a closed profile along one rail."""

        if not rail.is_valid() or not profile.is_valid():
            return NurbsSurface()

        count = max(5, min(rail.span_count() * 2 + 1, 200))
        frames = rail.get_perpendicular_planes(count)

        if len(frames) == 0:
            return NurbsSurface()

        to_xy = _profile_to_xy(profile)
        sections = []

        for frame in frames:
            section = profile.duplicate()
            section.transform(Xform.to_frame(frame) * to_xy)
            sections.append(section)

        return Primitives.create_loft(sections, min(3, len(sections) - 1))

    @staticmethod
    def create_sweep2(
        rail1: NurbsCurve, rail2: NurbsCurve, shapes: list[NurbsCurve]
    ) -> NurbsSurface:
        """Sweep of shape curves between two rails."""

        if not rail1.is_valid() or not rail2.is_valid() or len(shapes) == 0:
            return NurbsSurface()

        for shape in shapes:
            if not shape.is_valid():
                return NurbsSurface()

        compat = []

        for shape in shapes:
            compat.append(shape.duplicate())

        _make_curves_compatible(compat)

        n_shapes = len(compat)
        planes = []
        widths = []

        for shape in compat:
            planes.append(_shape_plane(shape))
            widths.append(_shape_width(shape))

        count = max(5, min(max(rail1.span_count(), rail2.span_count()) * 2 + 1, 200))
        pts1, _params1 = rail1.divide_by_count(count + 1)
        pts2, _params2 = rail2.divide_by_count(count + 1)
        frames = rail1.get_perpendicular_planes(count)

        if len(frames) == 0:
            return NurbsSurface()

        sections = []

        for i in range(min(len(frames), len(pts1), len(pts2))):
            t = 0.0 if len(frames) <= 1 else float(i) / (len(frames) - 1)
            j = 0 if n_shapes == 1 else min(int(t * (n_shapes - 1)), n_shapes - 2)
            j1 = 0 if n_shapes == 1 else j + 1
            s = 0.0 if n_shapes == 1 else max(0.0, min(t * (n_shapes - 1) - j, 1.0))
            section = compat[j].duplicate()

            for c in range(section.cv_count()):
                section.set_cv(
                    c, _lerp_point(compat[j].get_cv(c), compat[j1].get_cv(c), s)
                )

            source = Plane(
                _lerp_point(planes[j].origin, planes[j1].origin, s),
                _lerp_vector(planes[j].x_axis, planes[j1].x_axis, s),
                _lerp_vector(planes[j].y_axis, planes[j1].y_axis, s),
            )
            width = widths[j] * (1.0 - s) + widths[j1] * s
            p1 = pts1[i]
            x_dir = pts2[i] - p1
            rail_dist = x_dir.magnitude()

            if not x_dir.normalize_self():
                x_dir = frames[i].x_axis

            y_dir = frames[i].z_axis.cross(x_dir)

            if not y_dir.normalize_self():
                y_dir = frames[i].y_axis

            if y_dir.dot(source.y_axis) < 0.0:
                y_dir = -y_dir

            scale = rail_dist / width if rail_dist > 1e-14 and width > 1e-14 else 1.0
            target = Plane(p1, x_dir, y_dir)
            to_source = Xform.world_to_frame(
                source.origin, source.x_axis, source.y_axis, source.z_axis
            )
            section.transform(
                Xform.to_frame(target)
                * Xform.scale_xyz(scale, scale, scale)
                * to_source
            )
            sections.append(section)

        return Primitives.create_loft(sections, min(3, len(sections) - 1))

    @staticmethod
    def create_edge(
        c0: NurbsCurve, c1: NurbsCurve, c2: NurbsCurve, c3: NurbsCurve
    ) -> NurbsSurface:
        """Coons patch from four boundary curves in any order and direction."""

        if (
            not c0.is_valid()
            or not c1.is_valid()
            or not c2.is_valid()
            or not c3.is_valid()
        ):
            return NurbsSurface()

        loop = _chain_curves(
            [c0.duplicate(), c1.duplicate(), c2.duplicate(), c3.duplicate()]
        )

        if len(loop) == 0:
            return NurbsSurface()

        v_pair = [loop[0].duplicate(), loop[2].duplicate()]
        v_pair[1].reverse()
        _make_curves_compatible(v_pair)

        u_pair = [loop[3].duplicate(), loop[1].duplicate()]
        u_pair[0].reverse()
        _make_curves_compatible(u_pair)

        south = v_pair[0]
        north = v_pair[1]
        west = u_pair[0]
        east = u_pair[1]
        cv_count_u = west.cv_count()
        cv_count_v = south.cv_count()
        surface = NurbsSurface(
            3,
            south.is_rational() or west.is_rational(),
            west.order(),
            south.order(),
            cv_count_u,
            cv_count_v,
        )

        if not surface.is_valid():
            return NurbsSurface()

        for i in range(surface.nurbsknot_count(0)):
            surface.set_nurbsknot(0, i, west.nurbsknot(i))

        for i in range(surface.nurbsknot_count(1)):
            surface.set_nurbsknot(1, i, south.nurbsknot(i))

        u_grev = _normalized_greville(west)
        v_grev = _normalized_greville(south)
        c00 = south.get_cv(0)
        c01 = south.get_cv(cv_count_v - 1)
        c10 = north.get_cv(0)
        c11 = north.get_cv(cv_count_v - 1)

        for i in range(cv_count_u):
            ui = u_grev[i]
            wi = west.get_cv(i)
            ei = east.get_cv(i)

            for j in range(cv_count_v):
                vj = v_grev[j]
                sj = south.get_cv(j)
                nj = north.get_cv(j)
                q = [0.0, 0.0, 0.0]

                for axis in range(3):
                    q[axis] = (
                        (1.0 - ui) * sj[axis]
                        + ui * nj[axis]
                        + (1.0 - vj) * wi[axis]
                        + vj * ei[axis]
                        - (1.0 - ui) * (1.0 - vj) * c00[axis]
                        - (1.0 - ui) * vj * c01[axis]
                        - ui * (1.0 - vj) * c10[axis]
                        - ui * vj * c11[axis]
                    )

                surface.set_cv(i, j, Point(q[0], q[1], q[2]))

        return surface

    # ═══════════════════════════════════════════════════════════════════════════
    # Surface to mesh
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def quad_mesh(surface: NurbsSurface, u_count: int, v_count: int) -> Mesh:
        """Quad mesh sampled on a u_count x v_count grid."""

        mesh = Mesh()
        grid = _surface_grid(surface, u_count, v_count, mesh)
        singular_south = surface.is_singular(0)
        singular_north = surface.is_singular(2)

        if singular_south:
            for i in range(u_count):
                mesh.add_face([grid[0][0], grid[i + 1][1], grid[i][1]])

        if singular_north:
            for i in range(u_count):
                mesh.add_face(
                    [grid[0][v_count], grid[i][v_count - 1], grid[i + 1][v_count - 1]]
                )

        j0 = 1 if singular_south else 0
        j1 = v_count - 1 if singular_north else v_count

        for i in range(u_count):
            for j in range(j0, j1):
                mesh.add_face(
                    [grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]]
                )

        return mesh

    @staticmethod
    def diamond_mesh(surface: NurbsSurface, u_count: int, v_count: int) -> Mesh:
        """Diamond mesh sampled on a u_count x v_count grid."""

        mesh = Mesh()
        grid = _surface_grid(surface, u_count, v_count, mesh)
        closed_u = surface.is_closed(0)
        u_end = u_count - 1 if closed_u else u_count

        for i in range(u_end + 1):
            for j in range(v_count + 1):
                if (i + j) % 2 != 0:
                    continue

                center = grid[i][j]
                il = i - 1 if i > 0 else (u_count - 1 if closed_u else -1)
                left = grid[il][j] if il >= 0 else center
                bottom = grid[i][j - 1] if j > 0 else center
                right = grid[i + 1][j] if i < u_count else center
                top = grid[i][j + 1] if j < v_count else center
                face = _dedup_face([left, bottom, right, top])

                if len(face) >= 3:
                    mesh.add_face(face)

        return mesh

    @staticmethod
    def hex_mesh(
        surface: NurbsSurface, u_count: int, v_count: int, t: float = 1.0 / 3.0
    ) -> Mesh:
        """Hexagonal mesh sampled on a u_count x v_count grid, t the split of each v cell."""

        mesh = Mesh()
        grid = _surface_grid(surface, u_count, v_count, mesh)
        mid_a = _surface_mid_grid(surface, u_count, v_count, t, mesh)
        mid_b = _surface_mid_grid(surface, u_count, v_count, 1.0 - t, mesh)
        closed_u = surface.is_closed(0)
        u_end = u_count - 1 if closed_u else u_count

        for i in range(u_end + 1):
            for j in range(v_count + 1):
                if (i + j) % 2 != 0:
                    continue

                center = grid[i][j]
                il = i - 1 if i > 0 else (u_count - 1 if closed_u else -1)
                ul = (
                    mid_a[il][j]
                    if il >= 0 and j < v_count
                    else (grid[il][j] if il >= 0 else center)
                )
                ll = (
                    mid_b[il][j - 1]
                    if il >= 0 and j > 0
                    else (grid[il][j] if il >= 0 else center)
                )
                bt = mid_a[i][j - 1] if j > 0 else center
                lr = (
                    mid_b[i + 1][j - 1]
                    if i < u_count and j > 0
                    else (grid[i + 1][j] if i < u_count else center)
                )
                ur = (
                    mid_a[i + 1][j]
                    if i < u_count and j < v_count
                    else (grid[i + 1][j] if i < u_count else center)
                )
                tp = mid_b[i][j] if j < v_count else center
                face = _dedup_face([ul, ll, bt, lr, ur, tp])

                if len(face) >= 3:
                    mesh.add_face(face)

        return mesh

    # ═══════════════════════════════════════════════════════════════════════════
    # Mesh geometry
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _unit_cylinder_geometry() -> tuple[list[Point], list[list[int]]]:
        """Ten-sided unit cylinder: radius 0.5, z from -0.5 to 0.5."""

        n = 10
        vertices = []
        _add_ring(vertices, n, 0.5, -0.5)
        _add_ring(vertices, n, 0.5, 0.5)

        triangles = []

        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([i, next_i, n + next_i])
            triangles.append([i, n + next_i, n + i])

        return vertices, triangles

    @staticmethod
    def _unit_cone_geometry() -> tuple[list[Point], list[list[int]]]:
        """Eight-sided unit cone: base radius 0.5 at z = -0.5, apex at z = 0.5."""

        n = 8
        vertices = [Point(0.0, 0.0, 0.5)]
        _add_ring(vertices, n, 0.5, -0.5)

        triangles = []

        for i in range(n):
            triangles.append([0, 1 + i, 1 + (i + 1) % n])

        return vertices, triangles

    @staticmethod
    def _capsule_geometry(
        length: float, radius: float
    ) -> tuple[list[Point], list[list[int]]]:
        """Ten-sided capsule along z from 0 to length with hemispherical caps."""

        n = 10
        r_hemi = radius * math.sin(PI / 4.0)
        off = radius * math.cos(PI / 4.0)
        top = n
        hemi_a = 2 * n
        pole_a = 3 * n
        hemi_b = 3 * n + 1
        pole_b = 4 * n + 1
        vertices = []
        _add_ring(vertices, n, radius, 0.0)
        _add_ring(vertices, n, radius, length)
        _add_ring(vertices, n, r_hemi, -off)
        vertices.append(Point(0.0, 0.0, -radius))
        _add_ring(vertices, n, r_hemi, length + off)
        vertices.append(Point(0.0, 0.0, length + radius))

        triangles = []

        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([i, next_i, top + next_i])
            triangles.append([i, top + next_i, top + i])
            triangles.append([hemi_a + i, next_i, i])
            triangles.append([hemi_a + i, hemi_a + next_i, next_i])
            triangles.append([top + i, top + next_i, hemi_b + next_i])
            triangles.append([top + i, hemi_b + next_i, hemi_b + i])
            triangles.append([pole_a, hemi_a + next_i, hemi_a + i])
            triangles.append([pole_b, hemi_b + i, hemi_b + next_i])

        return vertices, triangles

    @staticmethod
    def _line_frame(line: Line, origin: Point) -> Xform:
        """Frame at origin with z along the line."""

        z_axis = line.to_vector()

        if not z_axis.normalize_self():
            z_axis = Vector(0.0, 0.0, 1.0)

        pole = Vector(0.0, 0.0, 1.0) if abs(z_axis[2]) < 0.9 else Vector(1.0, 0.0, 0.0)
        x_axis = pole.cross(z_axis)

        return Xform.frame_to_world(origin, x_axis, z_axis.cross(x_axis), z_axis)

    @staticmethod
    def _add_geometry(
        mesh: Mesh, geometry: tuple[list[Point], list[list[int]]], xform: Xform
    ) -> None:
        """Appends transformed geometry to a mesh."""

        vertices, triangles = geometry
        keys = []

        for v in vertices:
            keys.append(mesh.add_vertex(v.transformed(xform)))

        for tri in triangles:
            mesh.add_face([keys[tri[0]], keys[tri[1]], keys[tri[2]]])
