from __future__ import annotations
from typing import TYPE_CHECKING
import functools
import math
import sys
from .boolean_polyline import BooleanPolyline
from .closest import Closest
from .line import Line
from .mesh import Mesh
from .nurbscurve import NurbsCurve
from .nurbsknot import CurveNurbsKnotStyle
from .obb import OBB
from .plane import Plane
from .point import Point
from .polyline import Polyline
from .spatial_bvh import SpatialBVH
from .tolerance import PI
from .tolerance import TWO_PI
from .tolerance import Tolerance
from .vector import Vector

if TYPE_CHECKING:
    from .element import Element
    from .nurbssurface import NurbsSurface


# ═══════════════════════════════════════════════════════════════════════════
# Lines and planes
# ═══════════════════════════════════════════════════════════════════════════
def _max_pivot_3x3(rows):
    """Largest absolute coefficient of a 3x3 system with its row and column, the first one on ties."""

    temp = abs(rows[0][0])
    i = 0
    j = 0

    for r in range(3):
        for c in range(3):
            val = abs(rows[r][c])

            if val > temp:
                temp = val
                i = r
                j = c

    return temp, i, j


def _load_rows_3x3(rows, ds, i):
    """Rows of a 3x3 system in a 3x4 work array, row i swapped to the top."""

    w = [0.0] * 12
    src = [0, 1, 2]
    src[0], src[i] = src[i], src[0]

    for r in range(3):
        for c in range(3):
            w[4 * r + c] = rows[src[r]][c]

        w[4 * r + 3] = ds[src[r]]

    return w


def _swap_columns(w, slot, c0, c1):
    """Swap two coefficient columns in all rows of the 3x4 work array, and the unknowns they solve for."""

    for r in range(3):
        w[4 * r + c0], w[4 * r + c1] = w[4 * r + c1], w[4 * r + c0]

    slot[c0], slot[c1] = slot[c1], slot[c0]


def _eliminate_first_column(w):
    """Scale the top row to a unit pivot and clear the first column of the rows below."""

    temp = 1.0 / w[0]
    w[1] *= temp
    w[2] *= temp
    w[3] *= temp

    for r in (4, 8):
        temp = -w[r]

        if temp != 0.0:
            for c in range(1, 4):
                w[r + c] += temp * w[c]


def _max_pivot_2x2(w):
    """Largest absolute coefficient of the lower-right 2x2 block with its row and column, the first one on ties."""

    temp = abs(w[5])
    i = 0
    j = 0

    for r in range(2):
        for c in range(2):
            val = abs(w[5 + 4 * r + c])

            if val > temp:
                temp = val
                i = r
                j = c

    return temp, i, j


def _update_pivot_range(val, maxpiv, minpiv):
    """Widen the [minpiv, maxpiv] pivot range by val."""

    if val > maxpiv:
        maxpiv = val
    elif val < minpiv:
        minpiv = val

    return maxpiv, minpiv


def _eliminate_last_columns(w, pivot, other, maxpiv, minpiv):
    """Eliminate the second and third columns using the rows at offsets pivot and other: (ok, maxpiv, minpiv)."""

    temp = 1.0 / w[pivot + 1]
    w[pivot + 2] *= temp
    w[pivot + 3] *= temp
    temp = -w[1]

    if temp != 0.0:
        w[2] += temp * w[pivot + 2]
        w[3] += temp * w[pivot + 3]

    temp = -w[other + 1]

    if temp != 0.0:
        w[other + 2] += temp * w[pivot + 2]
        w[other + 3] += temp * w[pivot + 3]

    temp = w[other + 2]

    if temp == 0.0:
        return False, maxpiv, minpiv

    maxpiv, minpiv = _update_pivot_range(abs(temp), maxpiv, minpiv)
    w[other + 3] /= temp
    temp = -w[pivot + 2]

    if temp != 0.0:
        w[pivot + 3] += temp * w[other + 3]

    temp = -w[2]

    if temp != 0.0:
        w[3] += temp * w[other + 3]

    return True, maxpiv, minpiv


def _solve_3x3(row0, row1, row2, d0, d1, d2):
    """Gaussian elimination of a 3x3 system with full pivoting: (rank, x, y, z, pivot_ratio)."""

    rows = [row0, row1, row2]
    temp, i, j = _max_pivot_3x3(rows)

    if temp == 0.0:
        return 0, 0.0, 0.0, 0.0, 0.0

    maxpiv = abs(temp)
    minpiv = maxpiv
    slot = [0, 1, 2]
    w = _load_rows_3x3(rows, [d0, d1, d2], i)

    if j != 0:
        _swap_columns(w, slot, 0, j)

    _eliminate_first_column(w)
    temp, i, j = _max_pivot_2x2(w)

    if temp == 0.0:
        return 1, 0.0, 0.0, 0.0, 0.0

    maxpiv, minpiv = _update_pivot_range(abs(temp), maxpiv, minpiv)

    if j != 0:
        _swap_columns(w, slot, 1, 2)

    pivot = 8 if i else 4
    other = 4 if i else 8
    ok, maxpiv, minpiv = _eliminate_last_columns(w, pivot, other, maxpiv, minpiv)

    if not ok:
        return 2, 0.0, 0.0, 0.0, 0.0

    sol = [0.0, 0.0, 0.0]
    sol[slot[0]] = w[3]
    sol[slot[1]] = w[pivot + 3]
    sol[slot[2]] = w[other + 3]

    return 3, sol[0], sol[1], sol[2], minpiv / maxpiv


def _plane_value_at(plane: Plane, point: Point) -> float:
    """Signed plane equation value at a point."""
    return plane.a * point[0] + plane.b * point[1] + plane.c * point[2] + plane.d


def line_line(line0: Line, line1: Line, tolerance: float) -> Point | None:
    """Intersection point of two segments, the midpoint of closest approach within tolerance."""

    result = line_line_parameters(line0, line1, tolerance, True, False)

    if result is None:
        return None

    t0, t1 = result
    p0 = line0.point_at(t0)
    p1 = line1.point_at(t1)

    return Point((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, (p0[2] + p1[2]) * 0.5)


def _shared_endpoint_parameters(line0: Line, line1: Line) -> tuple[float, float] | None:
    """Parameters (0 or 1) of an exactly shared endpoint of two segments, or None."""

    ends0 = [line0.start(), line0.end()]
    ends1 = [line1.start(), line1.end()]

    for i in range(2):
        for j in range(2):
            if (
                ends0[i][0] == ends1[j][0]
                and ends0[i][1] == ends1[j][1]
                and ends0[i][2] == ends1[j][2]
            ):
                return (float(i), float(j))

    return None


def _clamp_unit(t: float) -> float:
    """Clamp a parameter to [0, 1]."""

    if t < 0.0:
        return 0.0

    if t > 1.0:
        return 1.0

    return t


def line_line_parameters(
    line0: Line,
    line1: Line,
    tolerance: float,
    intersect_segments: bool = True,
    near_parallel_as_closest: bool = False,
) -> tuple[float, float] | None:
    """Parameters of closest approach of two lines, clamped to the segments when requested."""

    shared = _shared_endpoint_parameters(line0, line1)

    if shared is not None:
        return shared

    A = line0.to_vector()
    B = line1.to_vector()
    C = line1.start() - line0.start()

    AA = A.dot(A)
    BB = B.dot(B)
    AB = A.dot(B)
    AC = A.dot(C)
    BC = B.dot(C)

    det = AA * BB - AB * AB
    zero_tol = max(AA, BB) * sys.float_info.epsilon
    parallel = abs(det) < zero_tol

    if parallel and not near_parallel_as_closest:
        return None

    if parallel:
        t0 = (AC / AA) if AA > 0.0 else 0.0
        t1 = ((BC + t0 * AB) / BB) if BB > 0.0 else 0.0
    else:
        inv_det = 1.0 / det
        t0 = (BB * AC - AB * BC) * inv_det
        t1 = (AB * AC - AA * BC) * inv_det

    if intersect_segments:
        t0 = _clamp_unit(t0)
        t1 = _clamp_unit(t1)

    if tolerance > 0.0 and line0.point_at(t0).distance(line1.point_at(t1)) > tolerance:
        return None

    return (t0, t1)


def plane_plane(plane0: Plane, plane1: Plane) -> Line | None:
    """Intersection line of two planes, anchored on plane0's origin."""

    d = plane1.z_axis.cross(plane0.z_axis)

    p = Point(
        (plane0.origin[0] + plane1.origin[0]) * 0.5,
        (plane0.origin[1] + plane1.origin[1]) * 0.5,
        (plane0.origin[2] + plane1.origin[2]) * 0.5,
    )

    plane2 = Plane.from_point_normal(p, d)

    output_p = plane_plane_plane(plane0, plane1, plane2)

    if output_p is None:
        return None

    return Line(
        output_p[0],
        output_p[1],
        output_p[2],
        output_p[0] + d[0],
        output_p[1] + d[1],
        output_p[2] + d[2],
    )


def plane_plane_to_line_canonical(plane0: Plane, plane1: Plane) -> Line | None:
    """Intersection line of two planes, anchored at the foot of the world origin."""

    n0 = plane0.z_axis
    n1 = plane1.z_axis
    dx = n1[1] * n0[2] - n1[2] * n0[1]
    dy = n1[2] * n0[0] - n1[0] * n0[2]
    dz = n1[0] * n0[1] - n1[1] * n0[0]
    d_sq = dx * dx + dy * dy + dz * dz

    if d_sq < 1e-20:
        return None

    o0 = plane0.origin
    o1 = plane1.origin
    k0 = n0[0] * o0[0] + n0[1] * o0[1] + n0[2] * o0[2]
    k1 = n1[0] * o1[0] + n1[1] * o1[1] + n1[2] * o1[2]
    n0n0 = n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2]
    n1n1 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]
    n0n1 = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]
    det = n0n0 * n1n1 - n0n1 * n0n1

    if abs(det) < 1e-20:
        return None

    c0 = (k0 * n1n1 - k1 * n0n1) / det
    c1 = (k1 * n0n0 - k0 * n0n1) / det
    ax = c0 * n0[0] + c1 * n1[0]
    ay = c0 * n0[1] + c1 * n1[1]
    az = c0 * n0[2] + c1 * n1[2]

    return Line(ax, ay, az, ax + dx, ay + dy, az + dz)


def line_plane(line: Line, plane: Plane, is_finite: bool = True) -> Point | None:
    """Intersection point of a line and a plane."""

    pt0 = line.start()
    pt1 = line.end()

    a = _plane_value_at(plane, pt0)
    b = _plane_value_at(plane, pt1)
    d = a - b

    if d == 0.0:
        if abs(a) < abs(b):
            t = 0.0
        elif abs(b) < abs(a):
            t = 1.0
        else:
            t = 0.5

        rc = False
    else:
        d_inv = 1.0 / d
        fd = abs(d_inv)

        if fd > 1.0 and (abs(a) >= 1e38 / fd or abs(b) >= 1e38 / fd):
            t = 0.5
            rc = False
        else:
            t = a / (a - b)
            rc = True

    s = 1.0 - t

    output = Point(
        pt0[0] if line[0] == line[3] else s * line[0] + t * line[3],
        pt0[1] if line[1] == line[4] else s * line[1] + t * line[4],
        pt0[2] if line[2] == line[5] else s * line[2] + t * line[5],
    )

    if is_finite and (t < 0.0 or t > 1.0):
        return None

    return output if rc else None


def plane_plane_plane(plane0: Plane, plane1: Plane, plane2: Plane) -> Point | None:
    """Intersection point of three planes."""

    rank, x, y, z, pr = _solve_3x3(
        [plane0.a, plane0.b, plane0.c],
        [plane1.a, plane1.b, plane1.c],
        [plane2.a, plane2.b, plane2.c],
        -plane0.d,
        -plane1.d,
        -plane2.d,
    )

    if rank == 3 and pr > 1e-12:
        return Point(x, y, z)

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Rays
# ═══════════════════════════════════════════════════════════════════════════
class RayHit:
    """Ray-mesh hit."""

    def __init__(
        self,
        t: float = 0.0,
        point: Point | None = None,
        u: float = 0.0,
        v: float = 0.0,
        face_index: int = -1,
    ):
        self.t = t  # Parameter along the ray.
        self.point = point if point is not None else Point()  # Hit point.
        self.u = u  # Barycentric u.
        self.v = v  # Barycentric v.
        self.face_index = face_index  # Hit face.


def ray_box_parameters(
    origin: Point, direction: Vector, box: OBB, t0: float, t1: float
) -> tuple[bool, float, float]:
    """Ray-box slab test returning the entry and exit parameters."""

    box_min = box.min_point()
    box_max = box.max_point()

    inv_dir = Vector(
        1.0 / direction[0] if direction[0] != 0.0 else sys.float_info.max,
        1.0 / direction[1] if direction[1] != 0.0 else sys.float_info.max,
        1.0 / direction[2] if direction[2] != 0.0 else sys.float_info.max,
    )

    tx1 = (box_min[0] - origin[0]) * inv_dir[0]
    tx2 = (box_max[0] - origin[0]) * inv_dir[0]

    tmin = min(tx1, tx2)
    tmax = max(tx1, tx2)

    ty1 = (box_min[1] - origin[1]) * inv_dir[1]
    ty2 = (box_max[1] - origin[1]) * inv_dir[1]

    tmin = max(tmin, min(ty1, ty2))
    tmax = min(tmax, max(ty1, ty2))

    tz1 = (box_min[2] - origin[2]) * inv_dir[2]
    tz2 = (box_max[2] - origin[2]) * inv_dir[2]

    tmin = max(tmin, min(tz1, tz2))
    tmax = min(tmax, max(tz1, tz2))

    tmin = max(tmin, t0)
    tmax = min(tmax, t1)

    return (tmax >= tmin, tmin, tmax)


def ray_box(line: Line, box: OBB, t0: float, t1: float) -> list[Point] | None:
    """Line-box entry and exit points."""

    origin = line.start()
    direction = line.to_vector()

    hit, tmin, tmax = ray_box_parameters(origin, direction, box, t0, t1)

    if not hit:
        return None

    entry = origin + direction * tmin
    exit_point = origin + direction * tmax

    return [entry, exit_point]


def ray_sphere_parameters(
    origin: Point, direction: Vector, center: Point, radius: float
) -> tuple[int, float, float]:
    """Ray-sphere parameters, returning the hit count."""

    offset = origin - center
    a = direction.dot(direction)
    b = 2.0 * direction.dot(offset)
    c = offset.dot(offset) - (radius * radius)
    disc = b * b - 4.0 * a * c

    if disc < 0.0:
        return (0, 0.0, 0.0)

    root = math.sqrt(disc)
    q = (-b - root) / 2.0 if b < 0.0 else (-b + root) / 2.0

    t0 = q / a
    t1 = c / q

    if t1 == t0:
        return (1, t0, t1)

    if t0 > t1:
        t0, t1 = t1, t0

    return (2, t0, t1)


def ray_sphere(line: Line, center: Point, radius: float) -> list[Point] | None:
    """Line-sphere hit points."""

    origin = line.start()
    direction = line.to_vector()

    hits, t0, t1 = ray_sphere_parameters(origin, direction, center, radius)

    if hits == 0:
        return None

    points = [origin + direction * t0]

    if hits == 2:
        points.append(origin + direction * t1)

    return points


def ray_triangle_parameters(
    origin: Point,
    direction: Vector,
    v0: Point,
    v1: Point,
    v2: Point,
    epsilon: float,
) -> tuple[bool, float, float, float, bool]:
    """Moller-Trumbore ray-triangle test returning (hit, t, u, v, parallel)."""

    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = direction.cross(edge2)

    det = edge1.dot(pvec)

    if det > -epsilon and det < epsilon:
        return (False, 0.0, 0.0, 0.0, True)

    inv_det = 1.0 / det

    tvec = origin - v0
    u = tvec.dot(pvec) * inv_det

    if u < 0.0 - epsilon or u > 1.0 + epsilon:
        return (False, 0.0, u, 0.0, False)

    qvec = tvec.cross(edge1)
    v = direction.dot(qvec) * inv_det

    if v < 0.0 - epsilon or u + v > 1.0 + epsilon:
        return (False, 0.0, u, v, False)

    t = edge2.dot(qvec) * inv_det

    return (True, t, u, v, False)


def ray_triangle(
    line: Line, v0: Point, v1: Point, v2: Point, epsilon: float
) -> Point | None:
    """Line-triangle hit point."""

    origin = line.start()
    direction = line.to_vector()

    hit, t, _, _, _ = ray_triangle_parameters(origin, direction, v0, v1, v2, epsilon)

    if not hit:
        return None

    return origin + direction * t


def _ray_hit_before(a: RayHit, b: RayHit) -> bool:
    """Whether hit a sorts before hit b: smaller t, ties within 1e-6 broken by the lower face index."""

    eps = 1e-6
    dt = a.t - b.t

    if abs(dt) <= eps:
        return a.face_index < b.face_index

    return a.t < b.t


def _ray_hit_order(a: RayHit, b: RayHit) -> int:
    """Three-way comparison built on _ray_hit_before."""

    if _ray_hit_before(a, b):
        return -1

    if _ray_hit_before(b, a):
        return 1

    return 0


def _sort_ray_hits(hits: list[RayHit], find_all: bool) -> bool:
    """Sorts hits by t and keeps only the nearest unless find_all; false when there is none."""

    if not hits:
        return False

    hits.sort(key=functools.cmp_to_key(_ray_hit_order))

    if not find_all:
        del hits[1:]

    return True


def ray_mesh_hits(
    origin: Point,
    direction: Vector,
    mesh: Mesh,
    find_all: bool = False,
    epsilon: float = Tolerance.ZERO_TOLERANCE,
) -> tuple[bool, list[RayHit]]:
    """Ray-mesh hits by brute force sorted by t, only the nearest unless find_all."""

    hits: list[RayHit] = []

    vertices, faces = mesh.to_vertices_and_faces()

    for i in range(len(faces)):
        face = faces[i]

        if len(face) < 3:
            continue

        for j in range(1, len(face) - 1):
            v0 = vertices[face[0]]
            v1 = vertices[face[j]]
            v2 = vertices[face[j + 1]]

            hit, t, u, v, _ = ray_triangle_parameters(
                origin, direction, v0, v1, v2, epsilon
            )

            if not hit or t < 0.0:
                continue

            hits.append(RayHit(t, origin + direction * t, u, v, i))

    return (_sort_ray_hits(hits, find_all), hits)


def ray_mesh_bvh_hits(
    origin: Point,
    direction: Vector,
    mesh: Mesh,
    find_all: bool = False,
    epsilon: float = Tolerance.ZERO_TOLERANCE,
) -> tuple[bool, list[RayHit]]:
    """Ray-mesh hits through the mesh's triangle BVH sorted by t, only the nearest unless find_all."""

    hits: list[RayHit] = []

    candidates: list[int] = []

    if not mesh.triangle_bvh_ray_cast(origin, direction, candidates, find_all):
        return (False, hits)

    for tri_id in candidates:
        found, face_idx, _, v0, v1, v2 = mesh.get_triangle_by_id(tri_id)

        if not found:
            continue

        hit, t, u, v, _ = ray_triangle_parameters(
            origin, direction, v0, v1, v2, epsilon
        )

        if not hit or t < 0.0:
            continue

        hits.append(RayHit(t, origin + direction * t, u, v, face_idx))

    return (_sort_ray_hits(hits, find_all), hits)


def ray_mesh(
    line: Line, mesh: Mesh, epsilon: float, find_all: bool = False
) -> list[Point]:
    """Line-mesh hit points by brute force sorted by t, only the nearest unless find_all."""

    result: list[Point] = []

    found, hits = ray_mesh_hits(line.start(), line.to_vector(), mesh, find_all, epsilon)

    if not found:
        return result

    for hit in hits:
        result.append(hit.point)

    return result


def ray_mesh_bvh(
    line: Line, mesh: Mesh, epsilon: float, find_all: bool = False
) -> list[Point]:
    """Line-mesh hit points through the mesh's triangle BVH sorted by t, only the nearest unless find_all."""

    result: list[Point] = []

    found, hits = ray_mesh_bvh_hits(
        line.start(), line.to_vector(), mesh, find_all, epsilon
    )

    if not found:
        return result

    for hit in hits:
        result.append(hit.point)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# NURBS curve helpers
# ═══════════════════════════════════════════════════════════════════════════
def _unique_sorted(values: list[float], tolerance: float) -> list[float]:
    """Sorted values without neighbours closer than tolerance to the last kept one."""

    unique = []

    for value in values:
        if not unique or abs(unique[-1] - value) >= tolerance:
            unique.append(value)

    return unique


def _curve_signed_distance_to_plane(pt, plane):
    """Signed distance of a point to the plane."""

    v = pt - plane.origin

    return v.dot(plane.z_axis)


def _curve_plane_slope(curve, plane, t):
    """Rate of change of the signed plane distance with the curve parameter."""

    derivs = curve.evaluate(t, 1)

    return derivs[1].dot(plane.z_axis)


def _append_unique(values, t, tolerance):
    """Appends t unless a value within tolerance is already present."""

    for existing in values:
        if abs(existing - t) < tolerance:
            return

    values.append(t)


def _curve_plane_newton_bracket(curve, plane, tolerance, a, b):
    """Newton for the plane crossing in [a, b] from the midpoint as (converged, t), bisecting whenever a step is flat or leaves the bracket."""

    f_a = _curve_signed_distance_to_plane(curve.point_at(a), plane)
    t = (a + b) * 0.5

    for _ in range(10):
        f = _curve_signed_distance_to_plane(curve.point_at(t), plane)

        if abs(f) < tolerance:
            return True, t

        df = _curve_plane_slope(curve, plane, t)
        flat = abs(df) < 1e-14
        t_new = t if flat else t - f / df

        if flat or t_new < a or t_new > b:
            if f * f_a < 0:
                b = t
            else:
                a = t
                f_a = f

            t = (a + b) * 0.5
            continue

        if abs(t_new - t) < tolerance:
            return True, t_new

        t = t_new

    return False, t


def _curve_find_root_bisection(curve, plane, t0, t1, tolerance):
    """Bisect the plane crossing between t0 and t1 down to tolerance."""

    max_iterations = 50
    d0 = _curve_signed_distance_to_plane(curve.point_at(t0), plane)
    d1 = _curve_signed_distance_to_plane(curve.point_at(t1), plane)

    if d0 * d1 > 0:
        return False, 0.0

    for _ in range(max_iterations):
        t_mid = (t0 + t1) * 0.5
        d_mid = _curve_signed_distance_to_plane(curve.point_at(t_mid), plane)

        if abs(d_mid) < tolerance or (t1 - t0) < tolerance:
            return True, t_mid

        if d0 * d_mid < 0:
            t1 = t_mid
            d1 = d_mid
        else:
            t0 = t_mid
            d0 = d_mid

    t_result = (t0 + t1) * 0.5

    return abs(
        _curve_signed_distance_to_plane(curve.point_at(t_result), plane)
    ) < tolerance * 10.0, t_result


def _curve_refine_intersection_newton(curve, plane, t, tolerance):
    """Polish a plane crossing parameter with Newton steps."""

    max_iterations = 10
    step_tolerance = tolerance * 0.01

    for _ in range(max_iterations):
        pt = curve.point_at(t)

        f = _curve_signed_distance_to_plane(pt, plane)
        df = _curve_plane_slope(curve, plane, t)

        if abs(f) < tolerance:
            return t

        if abs(df) < 1e-12:
            return t

        dt = -f / df

        if abs(dt) < step_tolerance:
            return t

        t += dt

        t0, t1 = curve.domain()

        if t < t0:
            t = t0

        if t > t1:
            t = t1

    return t


def _curve_plane_refine(curve, plane, tolerance, ta, tb, results):
    """Newton refinement of the plane crossing in a tiny interval [ta, tb], kept when it stays inside and on the plane."""

    tm = (ta + tb) * 0.5
    pm = curve.point_at(tm)
    dist = _curve_signed_distance_to_plane(pm, plane)

    if abs(dist) >= tolerance:
        return

    t = tm

    for _ in range(10):
        pt = curve.point_at(t)
        f = _curve_signed_distance_to_plane(pt, plane)
        df = _curve_plane_slope(curve, plane, t)

        if abs(df) < 1e-12:
            break

        dt = -f / df
        t += dt

        if abs(dt) < tolerance * 0.01:
            break

        if t < ta or t > tb:
            t = tm
            break

    pt_final = curve.point_at(t)

    if (
        abs(_curve_signed_distance_to_plane(pt_final, plane)) < tolerance
        and t >= ta
        and t <= tb
    ):
        results.append(t)


def _curve_plane_clip_range(curve, plane, tolerance, ta, tb):
    """Part of [ta, tb] where the sampled distance crosses the plane, the whole interval when unclear; None when it misses."""

    num_samples = min(curve.order() + 1, 10)
    distances = []
    params = []
    dt = (tb - ta) / (num_samples - 1)

    for i in range(num_samples):
        t = ta + i * dt
        p = curve.point_at(t)
        distances.append(_curve_signed_distance_to_plane(p, plane))
        params.append(t)

    d_min = min(distances)
    d_max = max(distances)

    if d_min > tolerance or d_max < -tolerance:
        return None

    t_min = ta
    t_max = tb

    for i in range(len(distances) - 1):
        if distances[i] * distances[i + 1] < 0:
            d0 = distances[i]
            d1 = distances[i + 1]
            t_clip = params[i] - d0 * (params[i + 1] - params[i]) / (d1 - d0)

            if d0 > 0:
                t_max = min(t_max, t_clip + (tb - ta) * 0.1)
            else:
                t_min = max(t_min, t_clip - (tb - ta) * 0.1)

    if t_min >= t_max:
        t_min = ta
        t_max = tb

    t_min = max(ta, t_min)
    t_max = min(tb, t_max)

    return t_min, t_max


def _curve_plane_clip(curve, plane, tolerance, ta, tb, depth, results):
    """Bezier-clipping recursion of the curve-plane distance on [ta, tb]."""

    if depth > 50:
        tm = (ta + tb) * 0.5
        pm = curve.point_at(tm)
        dist = _curve_signed_distance_to_plane(pm, plane)

        if abs(dist) < tolerance:
            results.append(tm)

        return

    if abs(tb - ta) < tolerance * 0.01:
        _curve_plane_refine(curve, plane, tolerance, ta, tb, results)

        return

    clip_range = _curve_plane_clip_range(curve, plane, tolerance, ta, tb)

    if clip_range is None:
        return

    t_min, t_max = clip_range
    reduction = (t_max - t_min) / (tb - ta)

    if reduction > 0.8 or (t_max - t_min) < tolerance * 0.1:
        tm = (ta + tb) * 0.5
        _curve_plane_clip(curve, plane, tolerance, ta, tm, depth + 1, results)
        _curve_plane_clip(curve, plane, tolerance, tm, tb, depth + 1, results)
    else:
        _curve_plane_clip(curve, plane, tolerance, t_min, t_max, depth + 1, results)


def _curve_plane_subdivide_algebraic(curve, plane, tolerance, a, b, depth, results):
    """Hodograph subdivision of one span with Newton polishing of the crossings."""

    if depth > 30:
        return

    p_a = curve.point_at(a)
    p_b = curve.point_at(b)
    normal = plane.z_axis
    f_a = normal.dot(p_a - plane.origin)
    f_b = normal.dot(p_b - plane.origin)

    if f_a * f_b > 0:
        return

    mid_t = (a + b) * 0.5
    p_mid = curve.point_at(mid_t)
    line_dir = p_b - p_a
    line_len = line_dir.magnitude()

    if line_len > 1e-14:
        line_dir = line_dir / line_len

    deviation = abs((p_mid - p_a).cross(line_dir).magnitude())

    if deviation < tolerance * 10.0 or (b - a) < tolerance * 10.0:
        converged, t = _curve_plane_newton_bracket(curve, plane, tolerance, a, b)

        if converged and a <= t <= b:
            _append_unique(results, t, tolerance * 10.0)
    else:
        _curve_plane_subdivide_algebraic(
            curve, plane, tolerance, a, mid_t, depth + 1, results
        )
        _curve_plane_subdivide_algebraic(
            curve, plane, tolerance, mid_t, b, depth + 1, results
        )


def _curve_nearly_linear(curve, tolerance, a, b):
    """True when the chord of [a, b] deviates less than ten tolerances from the curve."""

    p_a = curve.point_at(a)
    p_b = curve.point_at(b)
    p_mid = curve.point_at((a + b) * 0.5)
    ab = p_b - p_a
    line_length = ab.magnitude()

    if line_length < 1e-14:
        return True

    am = p_mid - p_a
    cross_mag = ab.cross(am).magnitude()
    deviation = cross_mag / line_length

    return deviation < tolerance * 10.0


def _curve_plane_subdivide_production(curve, plane, tolerance, a, b, depth, results):
    """Span subdivision to nearly linear pieces with Newton polishing of the crossings."""

    if depth > 30:
        return

    p_a = curve.point_at(a)
    p_b = curve.point_at(b)
    normal = plane.z_axis
    f_a = normal.dot(p_a - plane.origin)
    f_b = normal.dot(p_b - plane.origin)

    if f_a * f_b > 0:
        return

    if _curve_nearly_linear(curve, tolerance, a, b) or (b - a) < tolerance * 10.0:
        converged, t = _curve_plane_newton_bracket(curve, plane, tolerance, a, b)

        if converged and a <= t <= b:
            _append_unique(results, t, tolerance * 10.0)
    else:
        mid = (a + b) * 0.5
        _curve_plane_subdivide_production(
            curve, plane, tolerance, a, mid, depth + 1, results
        )
        _curve_plane_subdivide_production(
            curve, plane, tolerance, mid, b, depth + 1, results
        )


# ═══════════════════════════════════════════════════════════════════════════
# NURBS curves
# ═══════════════════════════════════════════════════════════════════════════
def _append_parameter(params: list[float], t: float, tolerance: float) -> None:
    """Appends t unless it lies within tolerance of the last parameter."""

    if not params or abs(params[-1] - t) >= tolerance:
        params.append(t)


def _curve_plane_hidden_pairs(curve, plane, tolerance, t0, t1, intersections):
    """Crossing pairs hidden inside a span whose ends lie on one side, found on degree * 2 sub-intervals."""

    count = curve.degree() * 2
    dt = (t1 - t0) / count

    for i in range(count):
        s0 = t0 + i * dt
        s1 = t0 + (i + 1) * dt
        d0 = _curve_signed_distance_to_plane(curve.point_at(s0), plane)
        d1 = _curve_signed_distance_to_plane(curve.point_at(s1), plane)

        if d0 * d1 < 0:
            found, t_intersection = _curve_find_root_bisection(
                curve, plane, s0, s1, tolerance
            )

            if found:
                t_intersection = _curve_refine_intersection_newton(
                    curve, plane, t_intersection, tolerance
                )
                intersections.append(t_intersection)


def _curve_plane_spans(curve, plane, tolerance, intersections):
    """Crossings inside each knot span, plus span starts and the curve end lying on the plane."""

    span_params = curve.get_span_vector()

    for i in range(len(span_params) - 1):
        t0 = span_params[i]
        t1 = span_params[i + 1]

        if abs(t1 - t0) < tolerance:
            continue

        d0 = _curve_signed_distance_to_plane(curve.point_at(t0), plane)
        d1 = _curve_signed_distance_to_plane(curve.point_at(t1), plane)

        if d0 * d1 < 0:
            found, t_intersection = _curve_find_root_bisection(
                curve, plane, t0, t1, tolerance
            )

            if found:
                t_intersection = _curve_refine_intersection_newton(
                    curve, plane, t_intersection, tolerance
                )
                intersections.append(t_intersection)
        elif abs(d0) < tolerance:
            _append_parameter(intersections, t0, tolerance)
        elif curve.degree() > 1:
            _curve_plane_hidden_pairs(curve, plane, tolerance, t0, t1, intersections)

    t_end = curve.domain()[1]

    if abs(_curve_signed_distance_to_plane(curve.point_at(t_end), plane)) < tolerance:
        _append_parameter(intersections, t_end, tolerance)


def _curve_plane_samples(curve, plane, tolerance, intersections):
    """Extra crossings of a high-degree curve found on degree * 4 uniform samples."""

    t_start, t_end = curve.domain()
    num_samples = curve.degree() * 4
    dt = (t_end - t_start) / num_samples

    for i in range(num_samples):
        t0 = t_start + i * dt
        t1 = t_start + (i + 1) * dt
        d0 = _curve_signed_distance_to_plane(curve.point_at(t0), plane)
        d1 = _curve_signed_distance_to_plane(curve.point_at(t1), plane)
        crossing = False

        if d0 * d1 < 0:
            crossing, t_intersection = _curve_find_root_bisection(
                curve, plane, t0, t1, tolerance
            )

        if not crossing:
            continue

        is_new = True

        for existing in intersections:
            if abs(existing - t_intersection) < tolerance * 2.0:
                is_new = False
                break

        if is_new:
            t_intersection = _curve_refine_intersection_newton(
                curve, plane, t_intersection, tolerance
            )
            intersections.append(t_intersection)


def curve_plane(
    curve: NurbsCurve, plane: Plane, tolerance: float | None = None
) -> list[float]:
    """Curve-plane intersection parameters by sampling, bisection and Newton refinement."""

    intersections = []

    if not curve.is_valid():
        return intersections

    if tolerance is None or tolerance <= 0.0:
        tolerance = Tolerance.ZERO_TOLERANCE

    _curve_plane_spans(curve, plane, tolerance, intersections)

    if curve.degree() > 3 and len(intersections) < curve.degree():
        _curve_plane_samples(curve, plane, tolerance, intersections)

    intersections.sort()

    return _unique_sorted(intersections, tolerance * 2.0)


def curve_plane_points(
    curve: NurbsCurve, plane: Plane, tolerance: float | None = None
) -> list[Point]:
    """Curve-plane intersection points."""
    params = curve_plane(curve, plane, tolerance)

    return [curve.point_at(t) for t in params]


def curve_plane_bezier_clipping(
    curve: NurbsCurve, plane: Plane, tolerance: float | None = None
) -> list[float]:
    """Curve-plane intersection parameters by Bezier clipping."""

    results = []

    if not curve.is_valid():
        return results

    if tolerance is None:
        tolerance = Tolerance.ZERO_TOLERANCE

    t0, t1 = curve.domain()
    _curve_plane_clip(curve, plane, tolerance, t0, t1, 0, results)
    results.sort()

    return _unique_sorted(results, tolerance * 10.0)


def curve_plane_algebraic(
    curve: NurbsCurve, plane: Plane, tolerance: float | None = None
) -> list[float]:
    """Curve-plane intersection parameters by hodograph subdivision."""

    if not curve.is_valid():
        return []

    if tolerance is None:
        tolerance = Tolerance.ZERO_TOLERANCE

    results = []
    spans = curve.get_span_vector()

    if len(spans) < 2:
        return []

    for i in range(len(spans) - 1):
        span_t0 = spans[i]
        span_t1 = spans[i + 1]

        if abs(span_t1 - span_t0) < tolerance:
            continue

        _curve_plane_subdivide_algebraic(
            curve, plane, tolerance, span_t0, span_t1, 0, results
        )

    results.sort()

    return _unique_sorted(results, tolerance * 10.0)


def curve_plane_production(
    curve: NurbsCurve, plane: Plane, tolerance: float | None = None
) -> list[float]:
    """Curve-plane intersection parameters by span subdivision and Newton polishing."""

    if not curve.is_valid():
        return []

    if tolerance is None:
        tolerance = Tolerance.ZERO_TOLERANCE

    results = []
    spans = curve.get_span_vector()

    if len(spans) < 2:
        return []

    for i in range(len(spans) - 1):
        span_t0 = spans[i]
        span_t1 = spans[i + 1]

        if abs(span_t1 - span_t0) < tolerance:
            continue

        _curve_plane_subdivide_production(
            curve, plane, tolerance, span_t0, span_t1, 0, results
        )

    results.sort()

    return _unique_sorted(results, tolerance * 10.0)


def curve_closest_point(
    curve: NurbsCurve, test_point: Point, t0: float = 0.0, t1: float = 0.0
) -> tuple[float, float]:
    """Closest curve parameter and distance to a point, optionally within [t0, t1]."""
    return Closest.curve_point(curve, test_point, t0, t1)


# ═══════════════════════════════════════════════════════════════════════════
# NURBS surface helpers
# ═══════════════════════════════════════════════════════════════════════════
class _SurfacePlaneTrace:
    """One traced surface-plane curve in parameter space."""

    def __init__(self, uv_trace, uv_unwrapped, is_loop):
        self.uv_trace = uv_trace  # Traced (u, v) samples.
        self.uv_unwrapped = uv_unwrapped  # Samples with seam wraps undone.
        self.is_loop = is_loop  # Whether the trace closes on itself.


class _SurfacePlaneTraceResult:
    """All traces of one surface-plane section with the scales used."""

    def __init__(self, traces, step, uv_to_3d, uv_to_3d_min):
        self.traces = traces  # Traced curves.
        self.step = step  # uv step used.
        self.uv_to_3d = uv_to_3d  # Largest uv-to-3D scale seen.
        self.uv_to_3d_min = uv_to_3d_min  # Smallest uv-to-3D scale seen.


class _SurfacePlaneSeed:
    """Grid crossing of the surface-plane distance, the start of one trace."""

    def __init__(self, u, v, used):
        self.u = u  # Seed u.
        self.v = v  # Seed v.
        self.used = used  # Whether a trace already passed the seed.


class _SurfacePlaneField:
    """Signed surface-plane distance over the surface's UV domain with the tracing scales."""

    def __init__(self, surface, plane, tolerance):
        """Sample the domain and derive the tracing scales."""

        self.surface = surface  # Traced surface.
        self.pn = plane.z_axis  # Plane normal.
        self.p0 = plane.origin  # Plane origin.
        self.tolerance = tolerance  # Newton tolerance.
        self.u0, self.u1 = surface.domain(0)  # Domain in u.
        self.v0, self.v1 = surface.domain(1)  # Domain in v.
        self.range_u = self.u1 - self.u0  # Domain length in u.
        self.range_v = self.v1 - self.v0  # Domain length in v.
        self.closed_u = surface.is_closed(0)  # Whether u wraps around a seam.
        self.closed_v = surface.is_closed(1)  # Whether v wraps around a seam.

        spans_u = surface.get_span_vector(0)
        spans_v = surface.get_span_vector(1)
        self.nu = max(len(spans_u) - 1, 1) * 4  # Grid cells in u.
        self.nv = max(len(spans_v) - 1, 1) * 4  # Grid cells in v.
        self.du = self.range_u / self.nu  # Grid cell size in u.
        self.dv = self.range_v / self.nv  # Grid cell size in v.

        du = self.du
        dv = self.dv
        mu = (self.u0 + self.u1) * 0.5
        mv = (self.v0 + self.v1) * 0.5
        pmid = self.point((mu, mv))
        uv_to_3d_u = pmid.distance(self.point((self.wrap_u(mu + du), mv))) / du
        uv_to_3d_v = pmid.distance(self.point((mu, self.wrap_v(mv + dv)))) / dv
        self.uv_to_3d = max(uv_to_3d_u, uv_to_3d_v)  # Largest uv-to-3D scale.
        self.uv_to_3d_min = min(uv_to_3d_u, uv_to_3d_v)  # Smallest uv-to-3D scale.

        if self.uv_to_3d < 1e-10:
            self.uv_to_3d = 1.0

        if self.uv_to_3d_min < 1e-10:
            self.uv_to_3d_min = 1.0

        self.step = min(du, dv) * 0.25  # Marching step in uv.
        self.max_steps = self.nu * self.nv * 32  # Marching step cap per direction.
        self.close_tol_3d = (
            self.step * 4.0 * self.uv_to_3d_min
        )  # 3D distance that closes a loop.
        self.consume_tol_3d = (
            self.step * self.uv_to_3d * 2.0
        )  # 3D distance that consumes a seed.
        self.join_tol = (
            max(du, dv) * self.uv_to_3d * 1.5
        )  # 3D distance that joins two traces.

    def wrap_u(self, u):
        """Wrap u across a closed seam or clamp it to the domain."""

        if self.closed_u:
            t = math.fmod(u - self.u0, self.range_u)

            if t < 0:
                t += self.range_u

            return self.u0 + t

        return max(self.u0, min(u, self.u1))

    def wrap_v(self, v):
        """Wrap v across a closed seam or clamp it to the domain."""

        if self.closed_v:
            t = math.fmod(v - self.v0, self.range_v)

            if t < 0:
                t += self.range_v

            return self.v0 + t

        return max(self.v0, min(v, self.v1))

    def value(self, u, v):
        """Signed plane distance at (u, v)."""

        p = self.point((self.wrap_u(u), self.wrap_v(v)))
        p0 = self.p0
        pn = self.pn

        return (p[0] - p0[0]) * pn[0] + (p[1] - p0[1]) * pn[1] + (p[2] - p0[2]) * pn[2]

    def value_and_gradient(self, u, v):
        """Signed plane distance and its uv gradient at (u, v)."""

        derivs = self.surface.evaluate(self.wrap_u(u), self.wrap_v(v), 1)
        S = derivs[0]
        Su = derivs[2]
        Sv = derivs[1]
        p0 = self.p0
        pn = self.pn
        val = (S[0] - p0[0]) * pn[0] + (S[1] - p0[1]) * pn[1] + (S[2] - p0[2]) * pn[2]
        gu = Su[0] * pn[0] + Su[1] * pn[1] + Su[2] * pn[2]
        gv = Sv[0] * pn[0] + Sv[1] * pn[1] + Sv[2] * pn[2]

        return val, gu, gv

    def newton_correct(self, u, v):
        """Newton-project (u, v) onto the zero set as (converged, u, v)."""

        for _ in range(10):
            val, gu, gv = self.value_and_gradient(u, v)

            if abs(val) < self.tolerance:
                return True, u, v

            mag2 = gu * gu + gv * gv

            if mag2 < 1e-28:
                return False, u, v

            u -= val * gu / mag2
            v -= val * gv / mag2
            u = self.wrap_u(u)
            v = self.wrap_v(v)

        return abs(self.value(u, v)) < self.tolerance * 10.0, u, v

    def tangent(self, u, v, direction):
        """Unit uv tangent of the zero set at (u, v) in direction, or None."""

        _, gu, gv = self.value_and_gradient(u, v)
        mag = math.hypot(gu, gv)

        if mag < 1e-14:
            return None

        return (-gv / mag * direction, gu / mag * direction)

    def point(self, q):
        """Surface point at a uv sample."""

        return self.surface.point_at(q[0], q[1])

    def seam_newton(self, cu, cv, axis):
        """Newton-slide (cu, cv) along one seam line, axis 0 moving v and axis 1 moving u."""

        for _ in range(10):
            val, gu, gv = self.value_and_gradient(cu, cv)

            if abs(val) < self.tolerance:
                break

            if axis == 0:
                if abs(gv) < 1e-14:
                    break

                cv = cv - val / gv
            else:
                if abs(gu) < 1e-14:
                    break

                cu = cu - val / gu

        return cu, cv

    def polish(self, u, v):
        """Newton-project (u, v) onto the zero set to 1e-12 as (converged, u, v)."""

        for _ in range(8):
            val, gu, gv = self.value_and_gradient(u, v)

            if abs(val) < 1e-12:
                return True, u, v

            mag2 = gu * gu + gv * gv

            if mag2 < 1e-28:
                return False, u, v

            u -= val * gu / mag2
            v -= val * gv / mag2

        return True, u, v


def _surface_plane_grid(field):
    """Signed plane distance on the (nu + 1) x (nv + 1) grid, exact zeros nudged negative."""

    cols = field.nv + 1
    dist = [0.0] * ((field.nu + 1) * cols)

    for i in range(field.nu + 1):
        u = field.u0 + field.du * i

        for j in range(field.nv + 1):
            v = field.v0 + field.dv * j
            d = field.value(u, v)

            if d == 0.0:
                d = -1e-14

            dist[i * cols + j] = d

    return dist


def _surface_plane_seeds(field, dist):
    """Newton-corrected sign changes along the grid edges, near duplicates marked used."""

    seeds = []
    cols = field.nv + 1
    h_jmax = field.nv - 1 if field.closed_v else field.nv

    for i in range(field.nu):
        for j in range(h_jmax + 1):
            d0 = dist[i * cols + j]
            d1 = dist[(i + 1) * cols + j]

            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                ok, su, sv = field.newton_correct(
                    field.u0 + field.du * (i + t), field.v0 + field.dv * j
                )

                if ok:
                    seeds.append(_SurfacePlaneSeed(su, sv, False))

    v_imax = field.nu - 1 if field.closed_u else field.nu

    for i in range(v_imax + 1):
        for j in range(field.nv):
            d0 = dist[i * cols + j]
            d1 = dist[i * cols + j + 1]

            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                ok, su, sv = field.newton_correct(
                    field.u0 + field.du * i, field.v0 + field.dv * (j + t)
                )

                if ok:
                    seeds.append(_SurfacePlaneSeed(su, sv, False))

    seed_tol_3d = max(field.du, field.dv) * field.uv_to_3d

    for i in range(len(seeds)):
        if seeds[i].used:
            continue

        pi = field.point((seeds[i].u, seeds[i].v))

        for j in range(i + 1, len(seeds)):
            if seeds[j].used:
                continue

            if pi.distance(field.point((seeds[j].u, seeds[j].v))) < seed_tol_3d:
                seeds[j].used = True

    return seeds


def _domain_step(field, u, v, local_step, tu, tv):
    """Step (u, v) by local_step along (tu, tv), pulled back onto an open domain boundary, as (un, vn, clamped)."""

    un = u + local_step * tu
    vn = v + local_step * tv

    out_u = not field.closed_u and (un < field.u0 or un > field.u1)
    out_v = not field.closed_v and (vn < field.v0 or vn > field.v1)

    if not out_u and not out_v:
        return un, vn, False

    tc = 1.0

    if not field.closed_u and tu > 0 and un > field.u1:
        tc = min(tc, (field.u1 - u) / (local_step * tu))

    if not field.closed_u and tu < 0 and un < field.u0:
        tc = min(tc, (field.u0 - u) / (local_step * tu))

    if not field.closed_v and tv > 0 and vn > field.v1:
        tc = min(tc, (field.v1 - v) / (local_step * tv))

    if not field.closed_v and tv < 0 and vn < field.v0:
        tc = min(tc, (field.v0 - v) / (local_step * tv))

    return u + tc * local_step * tu, v + tc * local_step * tv, True


def _newton_retry(field, u, v, local_step, tu, tv):
    """Retry a failed Newton projection with the step halved up to four times, or None."""

    ls = local_step

    for _ in range(4):
        ls *= 0.5
        ok, un, vn = field.newton_correct(
            field.wrap_u(u + ls * tu), field.wrap_v(v + ls * tv)
        )

        if ok:
            return un, vn

    return None


def _consume_seeds(field, p, seeds):
    """Mark every unused seed within the consume distance of p as used."""

    for other in seeds:
        if (
            not other.used
            and p.distance(field.point((other.u, other.v))) < field.consume_tol_3d
        ):
            other.used = True


def _turn_step(field, tu, tv, prev_tu, prev_tv):
    """Step length for the turn between two unit tangents: a quarter or half step on sharp turns."""

    if math.hypot(prev_tu, prev_tv) <= 1e-14:
        return field.step

    dot = max(-1.0, min(1.0, tu * prev_tu + tv * prev_tv))

    if dot < 0.95:
        return field.step * 0.25

    if dot < 0.985:
        return field.step * 0.5

    return field.step


def _march_tangent(field, u, v, direction, prev_tu, prev_tv):
    """Midpoint tangent and step at (u, v) along direction, the previous tangent reused where the field has none; None when neither exists."""

    tangent = field.tangent(u, v, direction)

    if tangent is None:
        if math.hypot(prev_tu, prev_tv) < 1e-14:
            return None

        tangent = (prev_tu, prev_tv)

    tu, tv = tangent
    local_step = _turn_step(field, tu, tv, prev_tu, prev_tv)
    mid = field.tangent(u + local_step * 0.5 * tu, v + local_step * 0.5 * tv, direction)

    if mid is not None:
        tu, tv = mid

    return tu, tv, local_step


def _surface_plane_march(field, su, sv, direction, seeds, out):
    """March the zero set from (su, sv) in direction; true when it closes on its start."""

    u = su
    v = sv
    prev_tu = 0.0
    prev_tv = 0.0
    p_start = field.point((su, sv))
    p_prev = p_start
    dist_traveled = 0.0

    for _ in range(field.max_steps):
        tangent = _march_tangent(field, u, v, direction, prev_tu, prev_tv)

        if tangent is None:
            break

        tu, tv, local_step = tangent
        prev_tu = tu
        prev_tv = tv

        un_raw, vn_raw, hit_boundary = _domain_step(field, u, v, local_step, tu, tv)
        ok, un, vn = field.newton_correct(field.wrap_u(un_raw), field.wrap_v(vn_raw))

        if not ok:
            retry = _newton_retry(field, u, v, local_step, tu, tv)

            if retry is None:
                break

            un, vn = retry

        p_cur = field.point((un, vn))
        dist_traveled += p_prev.distance(p_cur)
        out.append((un, vn))

        if (
            dist_traveled > field.close_tol_3d * 3.0
            and p_start.distance(p_cur) < field.close_tol_3d
        ):
            return True

        u = un
        v = vn
        p_prev = p_cur

        if hit_boundary:
            break

        _consume_seeds(field, p_cur, seeds)

    return False


def _unwrap_trace(field, uv):
    """Undo the seam jumps of a closed domain in the unwrapped copy of a trace."""

    for i in range(1, len(uv)):
        u, v = uv[i]
        du_jump = u - uv[i - 1][0]
        dv_jump = v - uv[i - 1][1]

        if field.closed_u:
            if du_jump > field.range_u * 0.5:
                u -= field.range_u
            elif du_jump < -field.range_u * 0.5:
                u += field.range_u

        if field.closed_v:
            if dv_jump > field.range_v * 0.5:
                v -= field.range_v
            elif dv_jump < -field.range_v * 0.5:
                v += field.range_v

        uv[i] = (u, v)


def _surface_plane_trace_seed(field, seeds, index):
    """Trace one seed both ways into a trace, or None when it is too short to keep."""

    seed_u = seeds[index].u
    seed_v = seeds[index].v
    fwd = []
    bwd = []
    fwd_closed = _surface_plane_march(field, seed_u, seed_v, 1, seeds, fwd)

    if not fwd_closed:
        _surface_plane_march(field, seed_u, seed_v, -1, seeds, bwd)

    uv_trace = []

    for i in range(len(bwd) - 1, -1, -1):
        uv_trace.append(bwd[i])

    uv_trace.append((seed_u, seed_v))

    uv_trace.extend(fwd)

    if len(uv_trace) < 4:
        return None

    p_first = field.point(uv_trace[0])
    p_last = field.point(uv_trace[-1])
    is_loop = fwd_closed or (
        len(uv_trace) >= 6 and p_first.distance(p_last) < field.close_tol_3d
    )

    if is_loop:
        uv_trace.pop()

    if len(uv_trace) < 4:
        return None

    uv_unwrapped = list(uv_trace)
    _unwrap_trace(field, uv_unwrapped)

    return _SurfacePlaneTrace(uv_trace, uv_unwrapped, is_loop)


def _trace_covered_by(field, a, b):
    """Whether every eighth sample of trace a lies within the join distance of trace b."""

    stride = max(1, len(a.uv_trace) // 8)

    for k in range(0, len(a.uv_trace), stride):
        q = field.point(a.uv_trace[k])
        best = 1e300

        for r in b.uv_trace:
            best = min(best, q.distance(field.point(r)))

        if best > field.join_tol:
            return False

    return True


def _drop_covered_traces(field, traces):
    """Empty every open trace that a trace at least as long already covers."""

    for i in range(len(traces)):
        if not traces[i].uv_trace or traces[i].is_loop:
            continue

        for j in range(len(traces)):
            if i == j or not traces[j].uv_trace:
                continue

            if len(traces[j].uv_trace) < len(traces[i].uv_trace):
                continue

            if _trace_covered_by(field, traces[i], traces[j]):
                traces[i].uv_trace.clear()
                break


def _append_trace(field, a, b, reversed_):
    """Append trace b to the end of trace a, reversed when requested, and close a when it meets itself."""

    add = list(b.uv_trace)

    if reversed_:
        add.reverse()

    a.uv_trace.extend(add)
    b.uv_trace.clear()

    if (
        field.point(a.uv_trace[0]).distance(field.point(a.uv_trace[-1]))
        < field.join_tol
    ):
        a.is_loop = True
        a.uv_trace.pop()

    a.uv_unwrapped = list(a.uv_trace)
    _unwrap_trace(field, a.uv_unwrapped)


def _join_one_trace_pair(field, traces):
    """Join the first open trace pair whose end meets a start or end; false when none does."""

    for i in range(len(traces)):
        if len(traces[i].uv_trace) < 2 or traces[i].is_loop:
            continue

        ie = field.point(traces[i].uv_trace[-1])

        for j in range(len(traces)):
            if i == j or len(traces[j].uv_trace) < 2 or traces[j].is_loop:
                continue

            ja = field.point(traces[j].uv_trace[0])
            jb = field.point(traces[j].uv_trace[-1])
            fwd2 = ie.distance(ja) < field.join_tol
            rev2 = ie.distance(jb) < field.join_tol

            if not fwd2 and not rev2:
                continue

            _append_trace(field, traces[i], traces[j], rev2)

            return True

    return False


def _close_traces(field, traces):
    """Drop short traces and close the open ones whose ends meet."""

    kept = []

    for t in traces:
        if len(t.uv_trace) >= 4:
            kept.append(t)

    traces[:] = kept

    for t in traces:
        if t.is_loop or len(t.uv_trace) < 6:
            continue

        if (
            field.point(t.uv_trace[0]).distance(field.point(t.uv_trace[-1]))
            < field.join_tol
        ):
            t.is_loop = True
            t.uv_trace.pop()
            t.uv_unwrapped.pop()


def _snap_trace_end(field, q, qu):
    """Snap one open trace end within a grid cell of the domain boundary onto it, as (q, qu)."""

    qx, qy = q
    qux, quy = qu

    if not field.closed_u:
        if abs(qx - field.u0) < field.du:
            qx = field.u0
            qux = field.u0

        if abs(qx - field.u1) < field.du:
            qx = field.u1
            qux = field.u1
    elif qx - field.u0 < field.du:
        qx = field.u0
    elif field.u1 - qx < field.du:
        qx = field.u1

    if not field.closed_v:
        if abs(qy - field.v0) < field.dv:
            qy = field.v0
            quy = field.v0

        if abs(qy - field.v1) < field.dv:
            qy = field.v1
            quy = field.v1
    elif qy - field.v0 < field.dv:
        qy = field.v0
    elif field.v1 - qy < field.dv:
        qy = field.v1

    return (qx, qy), (qux, quy)


def _surface_plane_traces(surface, plane, tolerance):
    """Seed and trace surface/plane intersection curves in UV space."""

    field = _SurfacePlaneField(surface, plane, tolerance)
    dist = _surface_plane_grid(field)

    gmax = 0.0

    for d in dist:
        gmax = max(gmax, abs(d))

    if gmax < max(tolerance, 1e-9) * 10.0:
        return _SurfacePlaneTraceResult(
            [], field.step, field.uv_to_3d, field.uv_to_3d_min
        )

    seeds = _surface_plane_seeds(field, dist)
    traces = []

    for i in range(len(seeds)):
        if seeds[i].used:
            continue

        seeds[i].used = True
        trace = _surface_plane_trace_seed(field, seeds, i)

        if trace is not None:
            traces.append(trace)

    _drop_covered_traces(field, traces)

    for _ in range(len(traces)):
        if not _join_one_trace_pair(field, traces):
            break

    _close_traces(field, traces)

    for t in traces:
        if t.is_loop or not t.uv_trace:
            continue

        t.uv_trace[0], t.uv_unwrapped[0] = _snap_trace_end(
            field, t.uv_trace[0], t.uv_unwrapped[0]
        )
        t.uv_trace[-1], t.uv_unwrapped[-1] = _snap_trace_end(
            field, t.uv_trace[-1], t.uv_unwrapped[-1]
        )

    return _SurfacePlaneTraceResult(
        traces, field.step, field.uv_to_3d, field.uv_to_3d_min
    )


def _plane_points_2d(pts, plane):
    """Points projected into the plane's 2D frame, z = 0."""

    ax = plane.x_axis
    ay = plane.y_axis
    po = plane.origin
    pts_2d = []

    for p in pts:
        dx = p[0] - po[0]
        dy = p[1] - po[1]
        dz = p[2] - po[2]
        px = dx * ax[0] + dy * ax[1] + dz * ax[2]
        py = dx * ay[0] + dy * ay[1] + dz * ay[2]
        pts_2d.append(Point(px, py, 0))

    return pts_2d


def _chord_parameters(pts, is_loop):
    """Normalized cumulative chord length of each point, the closing chord included for loops."""

    m = len(pts)
    chords = [0.0] * m
    total_len = 0.0

    for i in range(1, m):
        total_len += pts[i].distance(pts[i - 1])
        chords[i] = total_len

    if is_loop and m > 1:
        total_len += pts[0].distance(pts[m - 1])

    if total_len > 1e-14:
        for i in range(1, m):
            chords[i] /= total_len

    return chords


def _total_turning(pts):
    """Sum of the turning angles along a planar polyline."""

    turning = 0.0

    for i in range(1, len(pts) - 1):
        dx1 = pts[i][0] - pts[i - 1][0]
        dy1 = pts[i][1] - pts[i - 1][1]
        dx2 = pts[i + 1][0] - pts[i][0]
        dy2 = pts[i + 1][1] - pts[i][1]
        l1 = math.hypot(dx1, dy1)
        l2 = math.hypot(dx2, dy2)

        if l1 > 1e-14 and l2 > 1e-14:
            c = (dx1 * dx2 + dy1 * dy2) / (l1 * l2)
            c = max(-1.0, min(1.0, c))
            turning += math.acos(c)

    return turning


def _fitted_max_deviation(cand, pts, chords, iterations):
    """Largest distance from each point to the curve, found by ternary search around its chord parameter."""

    m = len(pts)
    ft0, ft1 = cand.domain()
    max_dev = 0.0

    for i in range(m):
        t = ft0 + (ft1 - ft0) * chords[i]
        w2 = (ft1 - ft0) * 2.0 / max(m - 1, 1)
        lo = max(ft0, t - w2)
        hi = min(ft1, t + w2)

        for _ in range(iterations):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3

            if cand.point_at(m1).distance(pts[i]) < cand.point_at(m2).distance(pts[i]):
                hi = m2
            else:
                lo = m1

        max_dev = max(max_dev, cand.point_at(0.5 * (lo + hi)).distance(pts[i]))

    return max_dev


def _fit_freeform_2d(pts_2d, chords, is_loop, fit_tol):
    """Best cubic fitted to 2D points, CVs doubled until within fit_tol; invalid when no fit succeeds."""

    m = len(pts_2d)
    target_cvs = max(8, int(_total_turning(pts_2d) / 0.5) + 6)
    max_cvs = min(m - 1, 128)
    crv_2d = NurbsCurve()
    best_dev = 1e300

    for _ in range(6):
        if target_cvs > max_cvs:
            break

        cand = NurbsCurve.create_fitted(pts_2d, target_cvs, 3, is_loop)

        if not cand.is_valid():
            break

        max_dev = _fitted_max_deviation(cand, pts_2d, chords, 20)

        if max_dev < best_dev:
            best_dev = max_dev
            crv_2d = cand

        if max_dev < fit_tol:
            break

        target_cvs = min(target_cvs * 2, max_cvs + 1)

    return crv_2d


def _lift_to_plane(crv_2d, plane):
    """Move the CVs of a curve drawn in the plane's 2D frame to 3D, in place."""

    ax = plane.x_axis
    ay = plane.y_axis
    po = plane.origin

    for i in range(crv_2d.cv_count()):
        cv2 = crv_2d.get_cv(i)
        cx = cv2[0]
        cy = cv2[1]
        crv_2d.set_cv(
            i,
            Point(
                po[0] + cx * ax[0] + cy * ay[0],
                po[1] + cx * ax[1] + cy * ay[1],
                po[2] + cx * ax[2] + cy * ay[2],
            ),
        )


def _fit_planar_freeform(all_pts, is_loop, plane, fit_tol):
    """Cubic fitted to the points in the plane's frame, CVs doubled until within fit_tol, lifted back to 3D."""

    m = len(all_pts)

    if m < 4:
        return NurbsCurve()

    pts_2d = _plane_points_2d(all_pts, plane)
    chords = _chord_parameters(pts_2d, is_loop)
    crv_2d = _fit_freeform_2d(pts_2d, chords, is_loop, fit_tol)

    if not crv_2d.is_valid():
        if is_loop:
            crv_2d = NurbsCurve.create_interpolated(
                pts_2d, CurveNurbsKnotStyle.ChordPeriodic
            )
        else:
            crv_2d = NurbsCurve.create_interpolated(pts_2d)

    if not crv_2d.is_valid():
        return NurbsCurve()

    _lift_to_plane(crv_2d, plane)

    return crv_2d


def _circle_nurbs(cx, cy, cz, xa, ya, radius):
    """Rational 9-CV circle on knots 0..4 around (cx, cy, cz) in the plane of the unit axes xa, ya."""

    w = math.sqrt(2.0) / 2.0
    px = [1, 1, 0, -1, -1, -1, 0, 1, 1]
    py = [0, 1, 1, 1, 0, -1, -1, -1, 0]
    wts = [1, w, 1, w, 1, w, 1, w, 1]
    knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    crv = NurbsCurve(3, True, 3, 9)

    for i in range(10):
        crv.set_nurbsknot(i, knots[i])

    for i in range(9):
        x = cx + radius * (px[i] * xa[0] + py[i] * ya[0])
        y = cy + radius * (px[i] * xa[1] + py[i] * ya[1])
        z = cz + radius * (px[i] * xa[2] + py[i] * ya[2])
        crv.set_cv_4d(i, x * wts[i], y * wts[i], z * wts[i], wts[i])

    return crv


def _ellipse_nurbs(cx, cy, cz, ea, eb, semi_a, semi_b):
    """Rational 9-CV ellipse on knots 0..4 around (cx, cy, cz) with semi-axes along the unit axes ea, eb."""

    w = math.sqrt(2.0) / 2.0
    px = [1, 1, 0, -1, -1, -1, 0, 1, 1]
    py = [0, 1, 1, 1, 0, -1, -1, -1, 0]
    wts = [1, w, 1, w, 1, w, 1, w, 1]
    knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    crv = NurbsCurve(3, True, 3, 9)

    for i in range(10):
        crv.set_nurbsknot(i, knots[i])

    for i in range(9):
        x = cx + semi_a * px[i] * ea[0] + semi_b * py[i] * eb[0]
        y = cy + semi_a * px[i] * ea[1] + semi_b * py[i] * eb[1]
        z = cz + semi_a * px[i] * ea[2] + semi_b * py[i] * eb[2]
        crv.set_cv_4d(i, x * wts[i], y * wts[i], z * wts[i], wts[i])

    return crv


def _plane_coords_2d(p, po, ax, ay):
    """Coordinates of p in the 2D frame (po, ax, ay)."""

    dx = p[0] - po[0]
    dy = p[1] - po[1]
    dz = p[2] - po[2]

    return dx * ax[0] + dy * ax[1] + dz * ax[2], dx * ay[0] + dy * ay[1] + dz * ay[2]


def _fit_plane_circle(all_pts, plane):
    """Exact circle of a closed planar trace when every point lies on the circle through three of them."""

    ax = plane.x_axis
    ay = plane.y_axis
    po = plane.origin
    n = len(all_pts)
    x1, y1 = _plane_coords_2d(all_pts[0], po, ax, ay)
    x2, y2 = _plane_coords_2d(all_pts[n // 3], po, ax, ay)
    x3, y3 = _plane_coords_2d(all_pts[2 * n // 3], po, ax, ay)
    ax_ = x2 - x1
    ay_ = y2 - y1
    bx_ = x3 - x1
    by_ = y3 - y1
    dd = 2.0 * (ax_ * by_ - ay_ * bx_)

    if abs(dd) <= 1e-10:
        return NurbsCurve()

    a2 = ax_ * ax_ + ay_ * ay_
    b2 = bx_ * bx_ + by_ * by_
    ccx = x1 + (by_ * a2 - ay_ * b2) / dd
    ccy = y1 + (ax_ * b2 - bx_ * a2) / dd
    radius = math.hypot(x1 - ccx, y1 - ccy)
    max_dev = 0.0

    for p in all_pts:
        px, py = _plane_coords_2d(p, po, ax, ay)
        max_dev = max(max_dev, abs(math.hypot(px - ccx, py - ccy) - radius))

    if radius <= 1e-10 or max_dev >= max(radius * 1e-5, 1e-6):
        return NurbsCurve()

    cx3d = po[0] + ccx * ax[0] + ccy * ay[0]
    cy3d = po[1] + ccx * ax[1] + ccy * ay[1]
    cz3d = po[2] + ccx * ax[2] + ccy * ay[2]

    return _circle_nurbs(
        cx3d, cy3d, cz3d, [ax[0], ax[1], ax[2]], [ay[0], ay[1], ay[2]], radius
    )


def _conic_normal_equations(all_pts, po, ax, ay):
    """Augmented normal equations [AtA | Atb] of the conic fit through the points in the frame (po, ax, ay)."""

    ata = [[0.0] * 5 for _ in range(5)]
    atb = [0.0] * 5

    for p in all_pts:
        x, y = _plane_coords_2d(p, po, ax, ay)
        row = [x * x, x * y, y * y, x, y]

        for r in range(5):
            atb[r] += row[r]

            for c in range(5):
                ata[r][c] += row[r] * row[c]

    m = [[0.0] * 6 for _ in range(5)]

    for r in range(5):
        for c in range(5):
            m[r][c] = ata[r][c]

        m[r][5] = atb[r]

    return m


def _solve_augmented_5x5(m):
    """Solve an augmented 5x6 system by Gaussian elimination with partial pivoting; None when singular."""

    for col in range(5):
        pivot = col

        for r in range(col + 1, 5):
            if abs(m[r][col]) > abs(m[pivot][col]):
                pivot = r

        if abs(m[pivot][col]) < 1e-20:
            return None

        if pivot != col:
            for j in range(col, 6):
                m[col][j], m[pivot][j] = m[pivot][j], m[col][j]

        for r in range(col + 1, 5):
            f = m[r][col] / m[col][col]

            for j in range(col, 6):
                m[r][j] -= f * m[col][j]

    coef = [0.0] * 5

    for i in range(4, -1, -1):
        s = m[i][5]

        for j in range(i + 1, 5):
            s -= m[i][j] * coef[j]

        coef[i] = s / m[i][i]

    return coef


def _fit_plane_conic(all_pts, po, ax, ay):
    """Least-squares conic A x^2 + B xy + C y^2 + D x + E y = 1 through the points in the plane's frame, None when singular."""

    return _solve_augmented_5x5(_conic_normal_equations(all_pts, po, ax, ay))


def _conic_max_deviation(all_pts, po, ax, ay, coef):
    """Largest residual of the conic coef over the points in the frame (po, ax, ay)."""

    ca = coef[0]
    cb = coef[1]
    cc = coef[2]
    cd = coef[3]
    ce = coef[4]
    max_conic_dev = 0.0

    for p in all_pts:
        x, y = _plane_coords_2d(p, po, ax, ay)
        max_conic_dev = max(
            max_conic_dev,
            abs(ca * x * x + cb * x * y + cc * y * y + cd * x + ce * y - 1.0),
        )

    return max_conic_dev


def _plane_ellipse_deviation(all_pts, po, ax, ay, cx, cy, semi_a, semi_b, cos_t, sin_t):
    """Largest distance from the points to the ellipse (cx, cy, semi_a, semi_b, theta) in the plane's frame."""

    max_ell_dev = 0.0

    for p in all_pts:
        px2, py2 = _plane_coords_2d(p, po, ax, ay)
        lx = cos_t * (px2 - cx) + sin_t * (py2 - cy)
        ly = -sin_t * (px2 - cx) + cos_t * (py2 - cy)
        ang = math.atan2(ly / semi_b, lx / semi_a)
        ex = cx + semi_a * math.cos(ang) * cos_t - semi_b * math.sin(ang) * sin_t
        ey = cy + semi_a * math.cos(ang) * sin_t + semi_b * math.sin(ang) * cos_t
        max_ell_dev = max(max_ell_dev, math.hypot(px2 - ex, py2 - ey))

    return max_ell_dev


def _fit_plane_ellipse(all_pts, plane):
    """Exact ellipse of a closed planar trace from a least-squares conic, invalid when it deviates."""

    ax = plane.x_axis
    ay = plane.y_axis
    po = plane.origin
    coef = _fit_plane_conic(all_pts, po, ax, ay)

    if coef is None:
        return NurbsCurve()

    ca = coef[0]
    cb = coef[1]
    cc = coef[2]
    cd = coef[3]
    ce = coef[4]
    disc = cb * cb - 4 * ca * cc

    if disc >= -1e-10 or abs(ca) <= 1e-14:
        return NurbsCurve()

    max_conic_dev = _conic_max_deviation(all_pts, po, ax, ay, coef)

    if max_conic_dev / max(max(abs(ca), abs(cc)), 1e-10) >= 0.01:
        return NurbsCurve()

    det = 4 * ca * cc - cb * cb
    cx = (cb * ce - 2 * cc * cd) / det
    cy = (cb * cd - 2 * ca * ce) / det
    theta = 0.5 * math.atan2(cb, ca - cc)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    a2 = ca * cos_t * cos_t + cb * cos_t * sin_t + cc * sin_t * sin_t
    c2 = ca * sin_t * sin_t - cb * cos_t * sin_t + cc * cos_t * cos_t
    rhs = -(ca * cx * cx + cb * cx * cy + cc * cy * cy + cd * cx + ce * cy - 1.0)

    if rhs <= 1e-14 or a2 <= 1e-14 or c2 <= 1e-14:
        return NurbsCurve()

    semi_a = math.sqrt(rhs / a2)
    semi_b = math.sqrt(rhs / c2)
    cx3d = po[0] + cx * ax[0] + cy * ay[0]
    cy3d = po[1] + cx * ax[1] + cy * ay[1]
    cz3d = po[2] + cx * ax[2] + cy * ay[2]
    ea = [0.0, 0.0, 0.0]
    eb = [0.0, 0.0, 0.0]

    for d in range(3):
        ea[d] = cos_t * ax[d] + sin_t * ay[d]
        eb[d] = -sin_t * ax[d] + cos_t * ay[d]

    crv = _ellipse_nurbs(cx3d, cy3d, cz3d, ea, eb, semi_a, semi_b)

    ell_tol = max(max(semi_a, semi_b) * 1e-5, 2e-6)

    if (
        _plane_ellipse_deviation(
            all_pts, po, ax, ay, cx, cy, semi_a, semi_b, cos_t, sin_t
        )
        > ell_tol
    ):
        return NurbsCurve()

    return crv


def _surface_plane_fit_3d(
    all_pts, is_loop, plane, step, uv_to_3d, uv_to_3d_min, allow_conics=True
):
    """Fit a 3D plane-constrained NurbsCurve to traced intersection points."""

    crv = NurbsCurve()

    if allow_conics and is_loop and len(all_pts) >= 6:
        crv = _fit_plane_circle(all_pts, plane)

    if not crv.is_valid() and allow_conics and is_loop and len(all_pts) >= 8:
        crv = _fit_plane_ellipse(all_pts, plane)

    if not crv.is_valid():
        crv = _fit_planar_freeform(
            all_pts, is_loop, plane, step * (uv_to_3d + uv_to_3d_min) * 0.5 * 5e-4
        )

    return crv


class _SurfacePlanePiece:
    """Seam-free run of uv samples cut from one trace."""

    def __init__(self, uv, is_loop):
        self.uv = uv  # Samples in parameter space.
        self.is_loop = is_loop  # Whether the piece still closes on itself.


def _is_duplicate_trace(trace_pts3, kept_pts3, dup_tol):
    """Whether the quarter, half and three-quarter samples of a trace all lie within dup_tol of one kept trace."""

    m = len(trace_pts3)

    for other in kept_pts3:
        all_close = True

        for f in (0.25, 0.5, 0.75):
            cp = trace_pts3[int((m - 1) * f)]
            dmin = dup_tol + 1.0

            for k in range(0, len(other), 5):
                dmin = min(dmin, cp.distance(other[k]))

            if dmin > dup_tol:
                all_close = False
                break

        if all_close:
            return True

    return False


def _close_unwrapped_loop(field, pts):
    """Append the loop start shifted by whole periods after the end; returns the shift (closure_du, closure_dv)."""

    du_j = pts[0][0] - pts[-1][0]
    dv_j = pts[0][1] - pts[-1][1]

    if field.closed_u:
        while du_j > field.range_u * 0.5:
            du_j -= field.range_u

        while du_j < -field.range_u * 0.5:
            du_j += field.range_u

    if field.closed_v:
        while dv_j > field.range_v * 0.5:
            dv_j -= field.range_v

        while dv_j < -field.range_v * 0.5:
            dv_j += field.range_v

    closure_du = (pts[-1][0] + du_j) - pts[0][0]
    closure_dv = (pts[-1][1] + dv_j) - pts[0][1]
    pts.append((pts[0][0] + closure_du, pts[0][1] + closure_dv))

    return closure_du, closure_dv


def _seam_crossings(field, pa, pb):
    """Seam crossings (t, axis, seam value) of the segment pa-pb, sorted by t."""

    crossings = []

    if field.closed_u and abs(pb[0] - pa[0]) > 1e-15:
        k0 = math.floor((pa[0] - field.u0) / field.range_u)
        k1 = math.floor((pb[0] - field.u0) / field.range_u)

        for k in range(min(k0, k1) + 1, max(k0, k1) + 1):
            L = field.u0 + k * field.range_u
            t = (L - pa[0]) / (pb[0] - pa[0])

            if 0.0 < t < 1.0:
                crossings.append((t, 0, L))

    if field.closed_v and abs(pb[1] - pa[1]) > 1e-15:
        k0 = math.floor((pa[1] - field.v0) / field.range_v)
        k1 = math.floor((pb[1] - field.v0) / field.range_v)

        for k in range(min(k0, k1) + 1, max(k0, k1) + 1):
            L = field.v0 + k * field.range_v
            t = (L - pa[1]) / (pb[1] - pa[1])

            if 0.0 < t < 1.0:
                crossings.append((t, 1, L))

    crossings.sort()

    return crossings


def _snap_to_seam(field, pa, q):
    """Snap q onto a seam it lies on within 1e-9 of the period after a real move from pa, as (snapped, q)."""

    qu, qv = q
    on_seam = False

    if field.closed_u:
        k = _round_half_away((qu - field.u0) / field.range_u)
        L = field.u0 + k * field.range_u

        if (
            abs(qu - L) < field.range_u * 1e-9
            and abs(qu - pa[0]) > field.range_u * 1e-9
        ):
            qu = L
            on_seam = True

    if field.closed_v:
        k = _round_half_away((qv - field.v0) / field.range_v)
        L = field.v0 + k * field.range_v

        if (
            abs(qv - L) < field.range_v * 1e-9
            and abs(qv - pa[1]) > field.range_v * 1e-9
        ):
            qv = L
            on_seam = True

    return on_seam, (qu, qv)


def _round_half_away(x):
    """Round half away from zero like std::round."""

    r = math.floor(abs(x))

    if abs(x) - r >= 0.5:
        r += 1

    return r if x >= 0.0 else -r


def _split_at_seams(field, pts):
    """Samples with the seam crossings inserted, and the indices of the samples on a seam."""

    cross_idx = []
    out_pts = [pts[0]]

    for i in range(1, len(pts)):
        pa = pts[i - 1]
        pb = pts[i]

        for t, axis, L in _seam_crossings(field, pa, pb):
            cu = pa[0] + (pb[0] - pa[0]) * t
            cv_ = pa[1] + (pb[1] - pa[1]) * t

            if axis == 0:
                cv_ = field.seam_newton(L, cv_, 0)[1]
                cu = L
            else:
                cu = field.seam_newton(cu, L, 1)[0]
                cv_ = L

            out_pts.append((cu, cv_))
            cross_idx.append(len(out_pts) - 1)

        q = (pb[0], pb[1])
        on_seam = False

        if i < len(pts) - 1:
            on_seam, q = _snap_to_seam(field, pa, q)

        out_pts.append(q)

        if on_seam:
            cross_idx.append(len(out_pts) - 1)

    return out_pts, cross_idx


def _seam_pieces(out_pts, cross_idx, is_loop, wrap_drift, closure_du, closure_dv):
    """Cut the samples at the seam indices; a loop's last piece wraps around to its first seam."""

    pieces = []

    if not cross_idx:
        pieces.append(_SurfacePlanePiece(list(out_pts), is_loop and not wrap_drift))

        return pieces

    if is_loop:
        for ci in range(len(cross_idx) - 1):
            pieces.append(
                _SurfacePlanePiece(
                    out_pts[cross_idx[ci] : cross_idx[ci + 1] + 1], False
                )
            )

        wrap_piece = out_pts[cross_idx[-1] :]

        for pi in range(1, cross_idx[0] + 1):
            wrap_piece.append(
                (out_pts[pi][0] + closure_du, out_pts[pi][1] + closure_dv)
            )

        pieces.append(_SurfacePlanePiece(wrap_piece, False))

        return pieces

    bounds = [0]

    bounds.extend(cross_idx)

    bounds.append(len(out_pts) - 1)

    for bi in range(len(bounds) - 1):
        if bounds[bi + 1] > bounds[bi]:
            pieces.append(
                _SurfacePlanePiece(out_pts[bounds[bi] : bounds[bi + 1] + 1], False)
            )

    return pieces


def _trace_pieces(field, trace):
    """Seam-free uv pieces of one trace."""

    pts = list(trace.uv_unwrapped)
    closure_du = 0.0
    closure_dv = 0.0

    if trace.is_loop and len(pts) >= 2:
        closure_du, closure_dv = _close_unwrapped_loop(field, pts)

    out_pts, cross_idx = _split_at_seams(field, pts)
    wrap_drift = (
        abs(closure_du) > field.range_u * 0.5 or abs(closure_dv) > field.range_v * 0.5
    )

    return _seam_pieces(
        out_pts, cross_idx, trace.is_loop, wrap_drift, closure_du, closure_dv
    )


def _shift_piece_to_domain(field, piece_pts):
    """Shift a piece by whole periods so its middle sample lies in the base domain."""

    mid = piece_pts[len(piece_pts) // 2]

    if field.closed_u:
        k_u = math.floor((mid[0] - field.u0) / field.range_u)

        if k_u != 0:
            for i in range(len(piece_pts)):
                piece_pts[i] = (piece_pts[i][0] - k_u * field.range_u, piece_pts[i][1])

    if field.closed_v:
        k_v = math.floor((mid[1] - field.v0) / field.range_v)

        if k_v != 0:
            for i in range(len(piece_pts)):
                piece_pts[i] = (piece_pts[i][0], piece_pts[i][1] - k_v * field.range_v)


def _densify_segment(field, au, av, bu, bv, depth, pts_uv):
    """Insert zero-set samples between a and b while the chord midpoint sags more than step * 1e-4, four levels deep."""

    mu = 0.5 * (au + bu)
    mv = 0.5 * (av + bv)
    ok, cu, cv2 = field.polish(mu, mv)

    if not ok:
        return

    sag = math.hypot(cu - mu, cv2 - mv)

    if sag > field.step * 1e-4 and depth < 4:
        _densify_segment(field, au, av, cu, cv2, depth + 1, pts_uv)
        pts_uv.append(Point(cu, cv2, 0.0))
        _densify_segment(field, cu, cv2, bu, bv, depth + 1, pts_uv)
    else:
        pts_uv.append(Point(cu, cv2, 0.0))


def _densify_piece(field, piece_pts):
    """Piece samples with zero-set samples inserted where a segment sags."""

    pts_uv = []

    for i in range(1, len(piece_pts)):
        a = piece_pts[i - 1]
        b = piece_pts[i]
        pts_uv.append(Point(a[0], a[1], 0.0))
        _densify_segment(field, a[0], a[1], b[0], b[1], 0, pts_uv)

    pts_uv.append(Point(piece_pts[-1][0], piece_pts[-1][1], 0.0))

    return pts_uv


def _chord_max_deviation(cand, pts, chords):
    """Largest distance from each point to the curve at its chord parameter."""

    ft0, ft1 = cand.domain()
    max_dev = 0.0

    for i in range(len(pts)):
        t = ft0 + (ft1 - ft0) * chords[i]
        max_dev = max(max_dev, cand.point_at(t).distance(pts[i]))

    return max_dev


def _fit_pcurve(pts_uv, piece_loop, step):
    """Cubic pcurve through the uv samples, CVs doubled until within step * 2e-3, with the last CV count tried."""

    mp = len(pts_uv)
    chords = _chord_parameters(pts_uv, piece_loop)
    max_cvs = min(mp - 1, 96)
    pcurve = NurbsCurve()
    pcurve_dev = 1e300
    target_cvs = max(8, int(_total_turning(pts_uv) / 0.5) + 6)

    for _ in range(6):
        if target_cvs > max_cvs:
            break

        cand = NurbsCurve.create_fitted(pts_uv, target_cvs, 3, piece_loop)

        if not cand.is_valid():
            break

        max_dev = _chord_max_deviation(cand, pts_uv, chords)

        if max_dev < pcurve_dev:
            pcurve_dev = max_dev
            pcurve = cand

        if max_dev < step * 2e-3:
            break

        target_cvs = min(target_cvs * 2, max_cvs + 1)

    if not pcurve.is_valid():
        if piece_loop:
            pcurve = NurbsCurve.create_interpolated(
                pts_uv, CurveNurbsKnotStyle.ChordPeriodic
            )
        else:
            pcurve = NurbsCurve.create_interpolated(pts_uv)

    return pcurve, target_cvs


def _refit_pcurve(field, pts_uv, piece_loop, target_cvs, vali_tol, pcurve):
    """The pcurve refit with twice the CVs when it strays from the zero set by more than vali_tol."""

    max_cvs = min(len(pts_uv) - 1, 96)
    max_off = 0.0

    for i in range(17):
        pc = pcurve.point_at(i / 16.0)
        val, _, _ = field.value_and_gradient(pc[0], pc[1])
        max_off = max(max_off, abs(val))

    if max_off > vali_tol and target_cvs * 2 <= max_cvs:
        refit = NurbsCurve.create_fitted(pts_uv, target_cvs * 2, 3, piece_loop)

        if refit.is_valid():
            refit.set_domain(0.0, 1.0)

            return refit

    return pcurve


def _piece_curves(field, plane, piece):
    """3D section curve and uv pcurve of one seam-free piece, or None when a fit fails."""

    _shift_piece_to_domain(field, piece.uv)

    pts_uv = _densify_piece(field, piece.uv)
    pts3 = []

    for p in pts_uv:
        pts3.append(field.point((field.wrap_u(p[0]), field.wrap_v(p[1]))))

    crv3 = _surface_plane_fit_3d(
        pts3,
        piece.is_loop,
        plane,
        field.step,
        field.uv_to_3d,
        field.uv_to_3d_min,
        False,
    )

    if not crv3.is_valid():
        if piece.is_loop:
            crv3 = NurbsCurve.create_interpolated(
                pts3, CurveNurbsKnotStyle.ChordPeriodic
            )
        else:
            crv3 = NurbsCurve.create_interpolated(pts3)

    if not crv3.is_valid():
        return None

    pcurve, target_cvs = _fit_pcurve(pts_uv, piece.is_loop, field.step)

    if not pcurve.is_valid():
        return None

    crv3.set_domain(0.0, 1.0)
    pcurve.set_domain(0.0, 1.0)

    fit_tol = field.step * (field.uv_to_3d + field.uv_to_3d_min) * 0.5
    vali_tol = max(10.0 * field.tolerance, fit_tol * 2.0)
    pcurve = _refit_pcurve(field, pts_uv, piece.is_loop, target_cvs, vali_tol, pcurve)

    return crv3, pcurve


def _solve_gauss(m, rhs, n):
    """Solve an n x n linear system by Gaussian elimination with partial pivoting."""

    a = [list(m[r]) + [rhs[r]] for r in range(n)]

    for col in range(n):
        pivot = col

        for r in range(col + 1, n):
            if abs(a[r][col]) > abs(a[pivot][col]):
                pivot = r

        if abs(a[pivot][col]) < 1e-20:
            return None

        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]

        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]

            for j in range(col, n + 1):
                a[r][j] -= f * a[col][j]

    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        s = a[i][n]

        for j in range(i + 1, n):
            s -= a[i][j] * x[j]

        x[i] = s / a[i][i]

    return x


# ═══════════════════════════════════════════════════════════════════════════
# Analytic quadric surface intersection
# ═══════════════════════════════════════════════════════════════════════════
def _ssi_dot(u, v):
    """Dot product of two triples."""
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _ssi_cross(u, v):
    """Cross product of two triples."""
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def _ssi_unit(v):
    """Unit triple, or the input when degenerate."""

    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    return [v[0] / length, v[1] / length, v[2] / length] if length > 1e-300 else list(v)


def _normalize_axis(v):
    """Normalize a triple in place; false when shorter than 1e-12."""

    length = math.sqrt(_ssi_dot(v, v))

    if length < 1e-12:
        return False

    v[0] /= length
    v[1] /= length
    v[2] /= length

    return True


def _ortho_basis(n):
    """Two unit vectors spanning the plane perpendicular to unit n."""

    ax = 1.0 if (abs(n[0]) <= abs(n[1]) and abs(n[0]) <= abs(n[2])) else 0.0
    ay = 1.0 if (ax == 0.0 and abs(n[1]) <= abs(n[2])) else 0.0
    az = 1.0 if (ax == 0.0 and ay == 0.0) else 0.0
    ux = ay * n[2] - az * n[1]
    uy = az * n[0] - ax * n[2]
    uz = ax * n[1] - ay * n[0]
    ul = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux /= ul
    uy /= ul
    uz /= ul
    vx = n[1] * uz - n[2] * uy
    vy = n[2] * ux - n[0] * uz
    vz = n[0] * uy - n[1] * ux

    return [ux, uy, uz], [vx, vy, vz]


def _exact_circle(cx, cy, cz, xa, ya, radius):
    """Exact 9-CV rational NURBS circle on domain [0, 1]."""

    crv = _circle_nurbs(cx, cy, cz, xa, ya, radius)
    crv.set_domain(0.0, 1.0)

    return crv


def _exact_ellipse(cx, cy, cz, ea, eb, semi_a, semi_b):
    """Exact 9-CV rational NURBS ellipse on domain [0, 1]."""

    crv = _ellipse_nurbs(cx, cy, cz, ea, eb, semi_a, semi_b)
    crv.set_domain(0.0, 1.0)

    return crv


def _jacobi_rotate(a, v, p, q):
    """Jacobi rotation of the symmetric a that zeroes a[p][q], accumulated into the eigenvector columns of v."""

    theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
    t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
    c = 1.0 / math.sqrt(t * t + 1.0)
    s = t * c

    for k in range(3):
        akp = a[k][p]
        akq = a[k][q]
        a[k][p] = c * akp - s * akq
        a[k][q] = s * akp + c * akq

    for k in range(3):
        apk = a[p][k]
        aqk = a[q][k]
        a[p][k] = c * apk - s * aqk
        a[q][k] = s * apk + c * aqk

    for k in range(3):
        vkp = v[k][p]
        vkq = v[k][q]
        v[k][p] = c * vkp - s * vkq
        v[k][q] = s * vkp + c * vkq


def _jacobi_eig3(m):
    """Eigenvalues/vectors of a symmetric 3x3 matrix (cyclic Jacobi)."""

    a = [[0.0] * 3 for _ in range(3)]
    v = [[0.0] * 3 for _ in range(3)]

    for r in range(3):
        for c in range(3):
            a[r][c] = float(m[r][c])
            v[r][c] = 1.0 if r == c else 0.0

    for _ in range(50):
        off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])

        if off < 1e-18:
            break

        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-300:
                continue

            _jacobi_rotate(a, v, p, q)

    eigvals = [a[0][0], a[1][1], a[2][2]]
    eigvecs = []

    for k in range(3):
        eigvecs.append([v[0][k], v[1][k], v[2][k]])

    return eigvals, eigvecs


def _smallest_eigenvector(m):
    """Eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix."""

    evals, evecs = _jacobi_eig3(m)
    kmin = 0

    for k in range(1, 3):
        if evals[k] < evals[kmin]:
            kmin = k

    return evecs[kmin]


def _largest_eigenvector(m):
    """Eigenvector of the largest eigenvalue of a symmetric 3x3 matrix."""

    evals, evecs = _jacobi_eig3(m)
    kmax = 0

    for k in range(1, 3):
        if evals[k] > evals[kmax]:
            kmax = k

    return evecs[kmax]


class _RecogSurface:
    """Recognized-surface descriptor."""

    NONE = 0
    PLANE = 1
    SPHERE = 2
    CYLINDER = 3
    CONE = 4
    TORUS = 5

    def __init__(self):
        self.kind = _RecogSurface.NONE  # Recognized kind.
        self.p1 = [0.0, 0.0, 0.0]  # Origin, center or apex.
        self.p2 = [0.0, 0.0, 0.0]  # Normal or axis.
        self.r = 0.0  # Radius or major radius.
        self.r2 = 0.0  # Half angle or minor radius.


def _sample_grid(surface, n, div):
    """Points of an n x n parameter grid stepping the domain by 1 / div."""

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    pts = []

    for i in range(n):
        for j in range(n):
            p = surface.point_at(u0 + (u1 - u0) * i / div, v0 + (v1 - v0) * j / div)
            pts.append([p[0], p[1], p[2]])

    return pts


def _sample_grid_normals(surface, n, div):
    """Normals of an n x n parameter grid stepping the domain by 1 / div."""

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    nrm = []

    for i in range(n):
        for j in range(n):
            v = surface.normal_at(u0 + (u1 - u0) * i / div, v0 + (v1 - v0) * j / div)
            nrm.append([v[0], v[1], v[2]])

    return nrm


def _fit_circle_2d(xy):
    """Least-squares circle through 2D samples: center and squared radius, None when singular."""

    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3

    for p in xy:
        row = [p[0], p[1], 1.0]
        rhs = -(p[0] * p[0] + p[1] * p[1])

        for r in range(3):
            atb[r] += row[r] * rhs

            for c in range(3):
                ata[r][c] += row[r] * row[c]

    sol = _solve_gauss(ata, atb, 3)

    if sol is None:
        return None

    cx = -sol[0] / 2.0
    cy = -sol[1] / 2.0

    return cx, cy, cx * cx + cy * cy - sol[2]


def _fit_cylinder(surface, tol):
    """Recognize a cylinder from surface samples: axis point, axis direction and radius."""

    pts = _sample_grid(surface, 5, 4.0)
    nrm = _sample_grid_normals(surface, 5, 4.0)
    m = [[0.0] * 3 for _ in range(3)]

    for n in nrm:
        for r in range(3):
            for c in range(3):
                m[r][c] += n[r] * n[c]

    w = _smallest_eigenvector(m)

    if not _normalize_axis(w):
        return None

    ea, eb = _ortho_basis(w)
    p0 = pts[0]
    proj = []

    for p in pts:
        dp = [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]]
        proj.append((_ssi_dot(dp, ea), _ssi_dot(dp, eb)))

    circle = _fit_circle_2d(proj)

    if circle is None or circle[2] <= 1e-18:
        return None

    ccx, ccy, r2 = circle
    r = math.sqrt(r2)

    for pr in proj:
        if (
            abs(
                math.sqrt((pr[0] - ccx) * (pr[0] - ccx) + (pr[1] - ccy) * (pr[1] - ccy))
                - r
            )
            > tol
        ):
            return None

    axis_pt = [
        p0[0] + ccx * ea[0] + ccy * eb[0],
        p0[1] + ccx * ea[1] + ccy * eb[1],
        p0[2] + ccx * ea[2] + ccy * eb[2],
    ]

    return axis_pt, w, r


def _sample_cone(surface):
    """Cone samples on an 8 x 5 grid with the unit normals that are not degenerate."""

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    pts = []
    nrm = []

    for i in range(8):
        uu = u0 + (u1 - u0) * i / 8.0

        for j in range(5):
            vv = v0 + (v1 - v0) * j / 4.0
            p = surface.point_at(uu, vv)
            pts.append([p[0], p[1], p[2]])
            n = surface.normal_at(uu, vv)
            nl = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])

            if nl < 1e-12:
                continue

            nrm.append(([n[0] / nl, n[1] / nl, n[2] / nl], [p[0], p[1], p[2]]))

    return pts, nrm


def _cone_apex(nrm):
    """Least-squares meeting point of the tangent planes through (unit normal, point) samples."""

    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3

    for n, p in nrm:
        npd = _ssi_dot(n, p)

        for r in range(3):
            atb[r] += n[r] * npd

            for c in range(3):
                ata[r][c] += n[r] * n[c]

    return _solve_gauss(ata, atb, 3)


def _cone_axis(gs):
    """Mean generator direction of unit apex-to-sample vectors, oriented away from the apex."""

    gram = [[0.0] * 3 for _ in range(3)]

    for g in gs:
        for r in range(3):
            for c in range(3):
                gram[r][c] += g[r] * g[c]

    w = _largest_eigenvector(gram)
    sx = [0.0, 0.0, 0.0]

    for g in gs:
        sx[0] += g[0]
        sx[1] += g[1]
        sx[2] += g[2]

    if _ssi_dot(w, sx) < 0.0:
        w = [-w[0], -w[1], -w[2]]

    if not _normalize_axis(w):
        return None

    return w


def _fit_cone(surface, tol):
    """Recognize a cone from surface samples: apex, axis and half angle."""

    pts, nrm = _sample_cone(surface)
    vertex = _cone_apex(nrm) if len(nrm) >= 4 else None

    if vertex is None:
        return None

    gs = []

    for p in pts:
        d = [p[0] - vertex[0], p[1] - vertex[1], p[2] - vertex[2]]
        dl = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])

        if dl < tol:
            continue

        gs.append([d[0] / dl, d[1] / dl, d[2] / dl])

    w = _cone_axis(gs) if len(gs) >= 3 else None

    if w is None:
        return None

    sumang = 0.0

    for g in gs:
        sumang += math.acos(max(-1.0, min(1.0, _ssi_dot(g, w))))

    alpha = sumang / len(gs)

    if alpha < 1e-4 or alpha > PI / 2 - 1e-4:
        return None

    ca = math.cos(alpha)

    for p in pts:
        d = [p[0] - vertex[0], p[1] - vertex[1], p[2] - vertex[2]]
        axd = _ssi_dot(d, w)
        perp = math.sqrt(max(0.0, _ssi_dot(d, d) - axd * axd))

        if abs(perp - axd * math.tan(alpha)) * ca > tol:
            return None

    return [vertex[0], vertex[1], vertex[2]], w, alpha


def _fit_sphere(surface, tol):
    """Recognize a sphere from surface samples: center and radius."""

    pts = _sample_grid(surface, 5, 4.0)
    ata = [[0.0] * 4 for _ in range(4)]
    atb = [0.0] * 4

    for p in pts:
        row = [p[0], p[1], p[2], 1.0]
        rhs = -(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])

        for r in range(4):
            atb[r] += row[r] * rhs

            for c in range(4):
                ata[r][c] += row[r] * row[c]

    sol = _solve_gauss(ata, atb, 4)

    if sol is None:
        return None

    ccx = -sol[0] / 2.0
    ccy = -sol[1] / 2.0
    ccz = -sol[2] / 2.0
    r2 = ccx * ccx + ccy * ccy + ccz * ccz - sol[3]

    if r2 <= 0.0:
        return None

    r = math.sqrt(r2)

    for p in pts:
        d = math.sqrt(
            (p[0] - ccx) * (p[0] - ccx)
            + (p[1] - ccy) * (p[1] - ccy)
            + (p[2] - ccz) * (p[2] - ccz)
        )

        if abs(d - r) > tol:
            return None

    return ccx, ccy, ccz, r


def _principal_axis(pts):
    """Centroid of the points and the unit direction of least spread about it; None when degenerate."""

    n = len(pts)
    cen = [0.0, 0.0, 0.0]

    for p in pts:
        cen[0] += p[0]
        cen[1] += p[1]
        cen[2] += p[2]

    cen[0] /= n
    cen[1] /= n
    cen[2] /= n
    m = [[0.0] * 3 for _ in range(3)]

    for p in pts:
        d = [p[0] - cen[0], p[1] - cen[1], p[2] - cen[2]]

        for r in range(3):
            for c in range(3):
                m[r][c] += d[r] * d[c]

    w = _smallest_eigenvector(m)

    if not _normalize_axis(w):
        return None

    return cen, w


def _fit_torus(surface, tol):
    """Recognize a torus from the smallest-variance axis and a tube cross-section circle fit."""

    pts = _sample_grid(surface, 8, 8.0)
    axis = _principal_axis(pts)

    if axis is None:
        return None

    cen, w = axis
    rhoa = []

    for p in pts:
        d = [p[0] - cen[0], p[1] - cen[1], p[2] - cen[2]]
        a = _ssi_dot(d, w)
        perp = [d[0] - a * w[0], d[1] - a * w[1], d[2] - a * w[2]]
        rhoa.append((math.sqrt(_ssi_dot(perp, perp)), a))

    circle = _fit_circle_2d(rhoa)

    if circle is None or circle[2] <= 1e-18 or circle[0] <= 0.0:
        return None

    rmaj, a0, r2 = circle
    r = math.sqrt(r2)

    if rmaj <= r * 0.5:
        return None

    for ra in rhoa:
        if (
            abs(
                math.sqrt((ra[0] - rmaj) * (ra[0] - rmaj) + (ra[1] - a0) * (ra[1] - a0))
                - r
            )
            > tol
        ):
            return None

    center = [cen[0] + a0 * w[0], cen[1] + a0 * w[1], cen[2] + a0 * w[2]]

    return center, w, rmaj, r


def _surface_mid_frame(srf):
    """Point and normal at the middle of the surface domain."""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)

    return srf.point_at((u0 + u1) * 0.5, (v0 + v1) * 0.5), srf.normal_at(
        (u0 + u1) * 0.5, (v0 + v1) * 0.5
    )


def _recognize_surface(surface, tol):
    """Classify a surface as plane, cylinder, cone, sphere or torus within tol."""

    rs = _RecogSurface()

    if surface.is_planar(None, tol):
        o, n = _surface_mid_frame(surface)
        rs.kind = _RecogSurface.PLANE
        rs.p1 = [o[0], o[1], o[2]]
        rs.p2 = [n[0], n[1], n[2]]

        return rs

    sphere = _fit_sphere(surface, tol)

    if sphere is not None:
        rs.kind = _RecogSurface.SPHERE
        rs.p1 = [sphere[0], sphere[1], sphere[2]]
        rs.r = sphere[3]

        return rs

    cylinder = _fit_cylinder(surface, tol)
    cone = _fit_cone(surface, tol) if cylinder is None else None
    torus = _fit_torus(surface, tol) if cylinder is None and cone is None else None

    if cylinder is not None:
        rs.kind = _RecogSurface.CYLINDER
        rs.p1, rs.p2, rs.r = cylinder
    elif cone is not None:
        rs.kind = _RecogSurface.CONE
        rs.p1, rs.p2, rs.r = cone
    elif torus is not None:
        rs.kind = _RecogSurface.TORUS
        rs.p1, rs.p2, rs.r, rs.r2 = torus

    return rs


def _line_cone(x0, d, apex, w, alpha):
    """Parameters t where x0 + t d meets the double cone of the apex, unit axis w and half angle alpha."""

    ca2 = math.cos(alpha) * math.cos(alpha)
    e = [x0[0] - apex[0], x0[1] - apex[1], x0[2] - apex[2]]
    a = _ssi_dot(e, w)
    b = _ssi_dot(d, w)
    c = _ssi_dot(e, e)
    dd = _ssi_dot(e, d)
    ee = _ssi_dot(d, d)
    qa = b * b - ca2 * ee
    qb = 2.0 * a * b - 2.0 * ca2 * dd
    qc = a * a - ca2 * c

    if abs(qa) < 1e-14:
        return [] if abs(qb) < 1e-300 else [-qc / qb]

    disc = qb * qb - 4.0 * qa * qc

    if disc < 0.0:
        return []

    sq = math.sqrt(disc)

    return [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)]


def _ssi_plane_sphere(plane, sph):
    """Exact plane-sphere circle."""

    o = plane.p1
    nu = _ssi_unit(plane.p2)
    c = sph.p1
    r = sph.r
    d = (c[0] - o[0]) * nu[0] + (c[1] - o[1]) * nu[1] + (c[2] - o[2]) * nu[2]

    if abs(d) >= r:
        return None

    cc = [c[0] - d * nu[0], c[1] - d * nu[1], c[2] - d * nu[2]]
    rr = math.sqrt(r * r - d * d)
    xa, ya = _ortho_basis(nu)

    return _exact_circle(cc[0], cc[1], cc[2], xa, ya, rr)


def _ssi_plane_cylinder(plane, cyl):
    """Exact plane-cylinder section: an ellipse or nothing."""

    o = plane.p1
    nu = _ssi_unit(plane.p2)
    p = cyl.p1
    w = _ssi_unit(cyl.p2)
    r = cyl.r
    wn = _ssi_dot(w, nu)

    if abs(wn) < 1e-7:
        return None

    t = ((o[0] - p[0]) * nu[0] + (o[1] - p[1]) * nu[1] + (o[2] - p[2]) * nu[2]) / wn
    cc = [p[0] + t * w[0], p[1] + t * w[1], p[2] + t * w[2]]
    mraw = _ssi_cross(w, nu)

    if math.sqrt(_ssi_dot(mraw, mraw)) < 1e-9:
        xa, ya = _ortho_basis(nu)

        return _exact_circle(cc[0], cc[1], cc[2], xa, ya, r)

    minor = _ssi_unit(mraw)
    major = _ssi_unit([w[0] - wn * nu[0], w[1] - wn * nu[1], w[2] - wn * nu[2]])

    return _exact_ellipse(cc[0], cc[1], cc[2], major, minor, r / abs(wn), r)


def _axis_segment(q, w, s0, s1):
    """Degree-1 segment of the line through q along w between axial offsets s0 and s1."""

    e0 = Point(q[0] + s0 * w[0], q[1] + s0 * w[1], q[2] + s0 * w[2])
    e1 = Point(q[0] + s1 * w[0], q[1] + s1 * w[1], q[2] + s1 * w[2])

    return NurbsCurve.create(False, 1, [e0, e1])


def _cylinder_axial_range(srf, p, w):
    """Axial range of a cylinder surface over three u and both v boundaries, padded by 5%."""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    smin = 1e300
    smax = -1e300

    for uu in (u0, 0.5 * (u0 + u1), u1):
        for vv in (v0, v1):
            q = srf.point_at(uu, vv)
            s = (q[0] - p[0]) * w[0] + (q[1] - p[1]) * w[1] + (q[2] - p[2]) * w[2]
            smin = min(smin, s)
            smax = max(smax, s)

    pad = 0.05 * max(1e-9, smax - smin)

    return smin - pad, smax + pad


def _ssi_plane_cylinder_lines(plane, cyl, cyl_srf, out):
    """Ruling lines of a plane parallel to the cylinder axis."""

    o = plane.p1
    nu = _ssi_unit(plane.p2)
    p = cyl.p1
    w = _ssi_unit(cyl.p2)
    r = cyl.r
    wn = _ssi_dot(w, nu)

    if abs(wn) >= 1e-7:
        return False

    ds = (p[0] - o[0]) * nu[0] + (p[1] - o[1]) * nu[1] + (p[2] - o[2]) * nu[2]
    d = abs(ds)
    tt = r * 1e-9 + 1e-12

    if d > r + tt:
        return True

    smin, smax = _cylinder_axial_range(cyl_srf, p, w)
    foot = [p[0] - ds * nu[0], p[1] - ds * nu[1], p[2] - ds * nu[2]]
    feet = []

    if d >= r - tt:
        feet.append(foot)
    else:
        h = math.sqrt(max(0.0, r * r - d * d))
        s3 = _ssi_unit(_ssi_cross(w, nu))
        feet.append([foot[0] + h * s3[0], foot[1] + h * s3[1], foot[2] + h * s3[2]])
        feet.append([foot[0] - h * s3[0], foot[1] - h * s3[1], foot[2] - h * s3[2]])

    for q in feet:
        line = _axis_segment(q, w, smin, smax)

        if line.is_valid():
            out.append(line)

    return True


def _cone_axial_extent(srf, apex, axis):
    """Height of the surface along the cone axis from the apex."""

    w = _ssi_unit(axis)
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    um = 0.5 * (u0 + u1)
    height = 0.0

    for vv in (v0, v1):
        p = srf.point_at(um, vv)
        s = (p[0] - apex[0]) * w[0] + (p[1] - apex[1]) * w[1] + (p[2] - apex[2]) * w[2]
        height = max(height, s)

    return height


def _conic_within_cone(c, apex, w, height):
    """Whether 65 samples of a conic lie within the cone height."""

    t0, t1 = c.domain()
    pad = 1e-7 * max(1.0, height)

    for i in range(65):
        p = c.point_at(t0 + (t1 - t0) * i / 64)
        s = (p[0] - apex[0]) * w[0] + (p[1] - apex[1]) * w[1] + (p[2] - apex[2]) * w[2]

        if s < -pad or s > height + pad:
            return False

    return True


def _fit_conic_arc(pts):
    """Fit a degree-2 rational arc through sampled points."""

    m = len(pts)

    if m < 2:
        return NurbsCurve()

    if m == 2:
        return NurbsCurve.create(False, 1, pts)

    if m <= 4:
        return NurbsCurve.create_interpolated(pts, CurveNurbsKnotStyle.Chord)

    num_cvs = min(max(m // 6, 8), 64)

    if num_cvs >= m:
        num_cvs = m - 1

    c = NurbsCurve.create_fitted(pts, num_cvs, 3)

    if not c.is_valid():
        c = NurbsCurve.create_interpolated(pts, CurveNurbsKnotStyle.Chord)

    return c


def _build_exact_plane_cone_ellipse(o, nu, apex, w, alpha):
    """Exact ellipse of a plane cutting a cone away from the apex."""

    wn = _ssi_dot(w, nu)
    m = _ssi_cross(w, nu)
    ml = math.sqrt(_ssi_dot(m, m))

    if ml < 1e-12:
        return None

    m = [m[0] / ml, m[1] / ml, m[2] / ml]
    major = _ssi_unit([w[0] - wn * nu[0], w[1] - wn * nu[1], w[2] - wn * nu[2]])
    dv = (apex[0] - o[0]) * nu[0] + (apex[1] - o[1]) * nu[1] + (apex[2] - o[2]) * nu[2]
    vp = [apex[0] - dv * nu[0], apex[1] - dv * nu[1], apex[2] - dv * nu[2]]
    ts = _line_cone(vp, major, apex, w, alpha)

    if len(ts) != 2:
        return None

    pa = [vp[0] + ts[0] * major[0], vp[1] + ts[0] * major[1], vp[2] + ts[0] * major[2]]
    pb = [vp[0] + ts[1] * major[0], vp[1] + ts[1] * major[1], vp[2] + ts[1] * major[2]]
    cc = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5]
    ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]]
    semi_major = 0.5 * math.sqrt(_ssi_dot(ab, ab))
    major = _ssi_unit(ab)
    tm = _line_cone(cc, m, apex, w, alpha)

    if len(tm) != 2:
        return None

    semi_minor = 0.5 * abs(tm[1] - tm[0])

    if semi_major < 1e-12 or semi_minor < 1e-12:
        return None

    return _exact_ellipse(cc[0], cc[1], cc[2], major, m, semi_major, semi_minor)


def _conic_bezier(pa, pt, pb, wmid):
    """Single rational quadratic Bezier conic arc from pa to pb with middle control point pt."""

    crv = NurbsCurve(3, True, 3, 3)
    knots = [0, 0, 1, 1]

    for i in range(4):
        crv.set_nurbsknot(i, knots[i])

    crv.set_cv_4d(0, pa[0], pa[1], pa[2], 1.0)
    crv.set_cv_4d(1, pt[0] * wmid, pt[1] * wmid, pt[2] * wmid, wmid)
    crv.set_cv_4d(2, pb[0], pb[1], pb[2], 1.0)
    crv.set_domain(0.0, 1.0)

    return crv


class _PlaneConeFrame:
    """Frame of an open plane-cone conic: cone data and the conic's in-plane axes."""

    def __init__(self):
        self.o = [0.0, 0.0, 0.0]  # Plane origin.
        self.nu = [0.0, 0.0, 0.0]  # Unit plane normal.
        self.apex = [0.0, 0.0, 0.0]  # Cone apex.
        self.w = [0.0, 0.0, 0.0]  # Unit cone axis.
        self.height = 0.0  # Cone height.
        self.cosa = 0.0  # Cosine of the half angle.
        self.sina = 0.0  # Sine of the half angle.
        self.ta = 0.0  # Tangent of the half angle.
        self.na = 0.0  # Plane normal along the axis.
        self.cost = 0.0  # Absolute na.
        self.sint = 0.0  # Length of nu x w.
        self.axex = [0.0, 0.0, 0.0]  # In-plane axis towards the cone axis.
        self.axey = [0.0, 0.0, 0.0]  # In-plane axis across the cone axis.
        self.axw = 0.0  # axex along the cone axis.
        self.d0 = 0.0  # Signed apex distance to the plane.
        self.tol = 0.0  # On-surface tolerance.


def _plane_cone_frame(o, nu, apex, w, alpha, height):
    """Conic frame of a plane cutting a cone; None when the plane is perpendicular to or contains the axis."""

    f = _PlaneConeFrame()
    f.o = o
    f.nu = nu
    f.apex = apex
    f.w = w
    f.height = height
    f.cosa = math.cos(alpha)
    f.sina = math.sin(alpha)
    f.ta = math.tan(alpha)
    f.na = _ssi_dot(nu, w)
    f.cost = abs(f.na)
    f.axey = _ssi_cross(nu, w)
    f.sint = math.sqrt(_ssi_dot(f.axey, f.axey))

    if f.sint < 1e-12:
        return None

    f.axey = [f.axey[0] / f.sint, f.axey[1] / f.sint, f.axey[2] / f.sint]
    f.axex = _ssi_cross(f.axey, nu)
    f.axw = _ssi_dot(f.axex, w)

    if f.axw < 0:
        f.axex = [-f.axex[0], -f.axex[1], -f.axex[2]]
        f.axw = -f.axw

    if f.axw < 1e-12:
        return None

    f.d0 = (
        (apex[0] - o[0]) * nu[0] + (apex[1] - o[1]) * nu[1] + (apex[2] - o[2]) * nu[2]
    )
    f.tol = 1e-6 * max(1.0, height)

    return f


def _conic_on_plane_cone(f, c):
    """Whether 17 samples of the curve lie on both the plane and the cone within the frame tolerance."""

    for i in range(17):
        q = c.point_at(i / 16.0)
        dp = abs(
            (q[0] - f.o[0]) * f.nu[0]
            + (q[1] - f.o[1]) * f.nu[1]
            + (q[2] - f.o[2]) * f.nu[2]
        )
        zz = (
            (q[0] - f.apex[0]) * f.w[0]
            + (q[1] - f.apex[1]) * f.w[1]
            + (q[2] - f.apex[2]) * f.w[2]
        )
        wx = q[0] - f.apex[0] - zz * f.w[0]
        wy = q[1] - f.apex[1] - zz * f.w[1]
        wz = q[2] - f.apex[2] - zz * f.w[2]
        rho = math.sqrt(wx * wx + wy * wy + wz * wz)

        if dp > f.tol or abs(rho - f.ta * zz) > f.tol * (1.0 + f.ta):
            return False

        if zz < -f.tol or zz > f.height + f.tol:
            return False

    return True


def _plane_cone_parabola(f):
    """Exact plane-cone parabola arc cut at the cone height."""

    if f.cost < 1e-12:
        return None

    sax = -f.d0 / f.na
    cen = [f.apex[0] + sax * f.w[0], f.apex[1] + sax * f.w[1], f.apex[2] + sax * f.w[2]]
    distance = abs(sax)
    dc = 0.5 * distance / f.cosa
    pf = dc * f.sina * f.sina

    if pf < 1e-15:
        return None

    for cs in (-1, 1):
        c2 = [
            cen[0] + cs * dc * f.axex[0],
            cen[1] + cs * dc * f.axex[1],
            cen[2] + cs * dc * f.axex[2],
        ]
        zc = (
            (c2[0] - f.apex[0]) * f.w[0]
            + (c2[1] - f.apex[1]) * f.w[1]
            + (c2[2] - f.apex[2]) * f.w[2]
        )
        t1s = 2.0 * pf * (f.height - zc) / f.axw

        if t1s <= 0:
            continue

        t1 = math.sqrt(t1s)
        xi = t1s / (2.0 * pf)
        pa = [
            c2[0] + xi * f.axex[0] - t1 * f.axey[0],
            c2[1] + xi * f.axex[1] - t1 * f.axey[1],
            c2[2] + xi * f.axex[2] - t1 * f.axey[2],
        ]
        pb = [
            c2[0] + xi * f.axex[0] + t1 * f.axey[0],
            c2[1] + xi * f.axex[1] + t1 * f.axey[1],
            c2[2] + xi * f.axex[2] + t1 * f.axey[2],
        ]
        pt = [c2[0] - xi * f.axex[0], c2[1] - xi * f.axex[1], c2[2] - xi * f.axex[2]]
        arc = _conic_bezier(pa, pt, pb, 1.0)

        if arc.is_valid() and _conic_on_plane_cone(f, arc):
            return arc

    return None


def _hyperbola_centers(f):
    """Semi-axes and centers of the plane-cone hyperbola, one center per nappe; None when degenerate."""

    centers = []

    if f.cost < 1e-6:
        a = abs(f.d0) / f.ta
        b = abs(f.d0)
        centers.append(
            [
                f.apex[0] - f.d0 * f.nu[0],
                f.apex[1] - f.d0 * f.nu[1],
                f.apex[2] - f.d0 * f.nu[2],
            ]
        )
    else:
        dd = f.sina * f.sina - f.cost * f.cost

        if dd < 1e-12:
            return None

        sax = -f.d0 / f.na
        cen = [
            f.apex[0] + sax * f.w[0],
            f.apex[1] + sax * f.w[1],
            f.apex[2] + sax * f.w[2],
        ]
        distance = abs(sax)
        dc = f.sint * f.sina * f.sina * distance / dd
        a = f.cost * f.sina * f.cosa * distance / dd
        b = f.cost * f.sina * distance / math.sqrt(dd)
        centers.append(
            [cen[0] - dc * f.axex[0], cen[1] - dc * f.axex[1], cen[2] - dc * f.axex[2]]
        )
        centers.append(
            [cen[0] + dc * f.axex[0], cen[1] + dc * f.axex[1], cen[2] + dc * f.axex[2]]
        )

    if a < 1e-15 or b < 1e-15:
        return None

    return a, b, centers


def _plane_cone_hyperbola(f):
    """Exact plane-cone hyperbola branch cut at the cone height."""

    hyperbola = _hyperbola_centers(f)

    if hyperbola is None:
        return None

    a, b, centers = hyperbola

    for c2 in centers:
        zc = (
            (c2[0] - f.apex[0]) * f.w[0]
            + (c2[1] - f.apex[1]) * f.w[1]
            + (c2[2] - f.apex[2]) * f.w[2]
        )

        for sg in (1, -1):
            ch = (f.height - zc) / (sg * a * f.axw)

            if ch <= 1.0 + 1e-12:
                continue

            sh = math.sqrt(ch * ch - 1.0)
            xi = sg * a * ch
            xt = sg * a / ch
            pa = [
                c2[0] + xi * f.axex[0] - b * sh * f.axey[0],
                c2[1] + xi * f.axex[1] - b * sh * f.axey[1],
                c2[2] + xi * f.axex[2] - b * sh * f.axey[2],
            ]
            pb = [
                c2[0] + xi * f.axex[0] + b * sh * f.axey[0],
                c2[1] + xi * f.axex[1] + b * sh * f.axey[1],
                c2[2] + xi * f.axex[2] + b * sh * f.axey[2],
            ]
            pt = [
                c2[0] + xt * f.axex[0],
                c2[1] + xt * f.axex[1],
                c2[2] + xt * f.axex[2],
            ]
            arc = _conic_bezier(pa, pt, pb, ch)

            if arc.is_valid() and _conic_on_plane_cone(f, arc):
                return arc

    return None


def _build_exact_plane_cone_open(o, nu, apex, w, alpha, height, parabola):
    """Exact plane-cone hyperbola or parabola arc (IntAna_QuadQuadGeo.cxx:752-953 port)."""

    f = _plane_cone_frame(o, nu, apex, w, alpha, height)

    if f is None:
        return None

    if parabola:
        return _plane_cone_parabola(f)

    return _plane_cone_hyperbola(f)


class _PlaneConeSection:
    """Plane-cone section: the plane, the cone and the cone's polar frame."""

    def __init__(self):
        self.o = [0.0, 0.0, 0.0]  # Plane origin.
        self.nu = [0.0, 0.0, 0.0]  # Unit plane normal.
        self.apex = [0.0, 0.0, 0.0]  # Cone apex.
        self.w = [0.0, 0.0, 0.0]  # Unit cone axis.
        self.e1 = [0.0, 0.0, 0.0]  # First unit axis normal.
        self.e2 = [0.0, 0.0, 0.0]  # Second unit axis normal.
        self.alpha = 0.0  # Half angle.
        self.height = 0.0  # Cone height.
        self.ta = 0.0  # Tangent of the half angle.
        self.cosa = 0.0  # Cosine of the half angle.
        self.sina = 0.0  # Sine of the half angle.
        self.na = 0.0  # Plane normal along the axis.
        self.pp = 0.0  # Plane normal along e1.
        self.qp = 0.0  # Plane normal along e2.
        self.cost = 0.0  # Absolute na.
        self.sint = 0.0  # Plane normal across the axis.
        self.costa = 0.0  # Cosine of the plane-to-generator angle sum.
        self.d0 = 0.0  # Signed apex distance to the plane.

    def denom(self, phi):
        """Plane normal along the generator at polar angle phi, scaled by cos alpha."""
        return self.na + self.ta * (self.pp * math.cos(phi) + self.qp * math.sin(phi))

    def height_at(self, phi):
        """Axial height of the section point at polar angle phi."""

        d = self.denom(phi)

        return 1e308 if abs(d) < 1e-300 else -self.d0 / d

    def point(self, phi):
        """Section point at polar angle phi."""

        s = self.height_at(phi)
        rr = s * self.ta
        c = math.cos(phi)
        sn = math.sin(phi)
        apex = self.apex
        w = self.w
        e1 = self.e1
        e2 = self.e2

        return Point(
            apex[0] + s * w[0] + rr * (c * e1[0] + sn * e2[0]),
            apex[1] + s * w[1] + rr * (c * e1[1] + sn * e2[1]),
            apex[2] + s * w[2] + rr * (c * e1[2] + sn * e2[2]),
        )

    def refine_base(self, pa, pb, dtarget):
        """Polar angle in [pa, pb] where the denominator reaches dtarget, by bisection."""

        fa = self.denom(pa) - dtarget

        for _ in range(60):
            pm = 0.5 * (pa + pb)
            fm = self.denom(pm) - dtarget

            if (fm < 0) == (fa < 0):
                pa = pm
                fa = fm
            else:
                pb = pm

        return 0.5 * (pa + pb)


def _plane_cone_section(plane, cone, cone_srf):
    """Section of a recognized plane and cone; None when the cone is flat, a line or has no height."""

    s = _PlaneConeSection()
    s.o = plane.p1
    s.nu = _ssi_unit(plane.p2)
    s.apex = cone.p1
    s.w = _ssi_unit(cone.p2)
    s.alpha = cone.r

    if s.alpha < 1e-7 or s.alpha > PI / 2 - 1e-7:
        return None

    s.ta = math.tan(s.alpha)
    s.cosa = math.cos(s.alpha)
    s.sina = math.sin(s.alpha)
    s.height = _cone_axial_extent(cone_srf, s.apex, s.w)

    if s.height < 1e-12:
        return None

    s.e1, s.e2 = _ortho_basis(s.w)
    s.na = _ssi_dot(s.nu, s.w)
    s.pp = _ssi_dot(s.nu, s.e1)
    s.qp = _ssi_dot(s.nu, s.e2)
    s.cost = abs(s.na)
    s.sint = math.sqrt(max(0.0, s.pp * s.pp + s.qp * s.qp))
    s.costa = s.cost * s.cosa - s.sint * s.sina
    s.d0 = (
        (s.apex[0] - s.o[0]) * s.nu[0]
        + (s.apex[1] - s.o[1]) * s.nu[1]
        + (s.apex[2] - s.o[2]) * s.nu[2]
    )

    return s


def _collect_cone_runs(s, ok, start, dtarget, runs):
    """Runs of consecutive in-range polar samples, closed at both ends at the cone base."""

    n = len(ok)
    cur = []
    inside = False

    for i in range(n + 1):
        k = (start + i) % n
        uphi = TWO_PI * start / n + TWO_PI * i / n
        v = ok[k] != 0

        if v and not inside:
            if i > 0:
                cur.append(s.point(s.refine_base(uphi - TWO_PI / n, uphi, dtarget)))

            cur.append(s.point(uphi))
            inside = True
        elif v and inside:
            cur.append(s.point(uphi))
        elif not v and inside:
            cur.append(s.point(s.refine_base(uphi - TWO_PI / n, uphi, dtarget)))

            if len(cur) >= 2:
                runs.append(cur)

            cur = []
            inside = False

    if inside and len(cur) >= 2:
        runs.append(cur)


def _sample_plane_cone_arcs(s):
    """Sample the plane-cone section as point runs, one per branch, and whether it closes."""

    runs = []
    n = 720
    eps = 1e-9 * max(1.0, s.height)
    ok = [0] * n
    cnt = 0

    for k in range(n):
        h = s.height_at(TWO_PI * k / n)
        ok[k] = 1 if (h > eps and h < s.height + eps) else 0
        cnt += ok[k]

    if cnt == 0:
        return runs, False

    if cnt == n:
        loop = []

        for k in range(n + 1):
            loop.append(s.point(TWO_PI * (k % n) / n))

        runs.append(loop)

        return runs, True

    start = 0

    while start < n and ok[start]:
        start += 1

    dtarget = (-s.d0 / s.height) if s.height > 1e-300 else 0.0
    _collect_cone_runs(s, ok, start, dtarget, runs)

    return runs, False


def _ray_segment(q, d, length):
    """Degree-1 segment from q to q + len d."""

    e0 = Point(q[0], q[1], q[2])
    e1 = Point(q[0] + length * d[0], q[1] + length * d[1], q[2] + length * d[2])

    return NurbsCurve.create(False, 1, [e0, e1])


def _plane_cone_through_apex(s, out):
    """Plane through the cone apex: one tangent generator or two generator lines."""

    nu = s.nu
    w = s.w

    if abs(s.costa) < 1e-6:
        g = _ssi_unit([w[0] - s.na * nu[0], w[1] - s.na * nu[1], w[2] - s.na * nu[2]])
        gw = _ssi_dot(g, w)

        if gw > 1e-9:
            out.append(_ray_segment(s.apex, g, s.height / gw))

        return

    if s.cost < s.sina:
        axey = _ssi_cross(nu, w)
        axex = _ssi_cross(axey, nu)
        dh = math.sqrt(max(0.0, s.sina * s.sina - s.cost * s.cost)) / s.cosa

        for sgn in (1, -1):
            d = [
                axex[0] + sgn * dh * axey[0],
                axex[1] + sgn * dh * axey[1],
                axex[2] + sgn * dh * axey[2],
            ]
            dw = _ssi_dot(d, w)

            if dw < 1e-12:
                continue

            out.append(_ray_segment(s.apex, d, s.height / dw))


def _plane_cone_exact(s, out):
    """Exact plane-cone conic: circle, ellipse, parabola or hyperbola; false when none fits the cone."""

    ang = 1e-6
    is_circle = False
    is_parabola = False
    is_hyperbola = False
    is_ellipse = False

    if s.cost < ang:
        is_hyperbola = True
    elif abs(s.costa) < ang:
        is_parabola = True
    elif s.sint < ang:
        is_circle = True
    elif s.cost < s.sina:
        is_hyperbola = True
    else:
        is_ellipse = True

    if is_circle:
        apex = s.apex
        w = s.w
        dax = (
            (s.o[0] - apex[0]) * w[0]
            + (s.o[1] - apex[1]) * w[1]
            + (s.o[2] - apex[2]) * w[2]
        )
        rr = abs(dax) * s.ta

        if rr > 1e-12:
            cc = [apex[0] + dax * w[0], apex[1] + dax * w[1], apex[2] + dax * w[2]]
            circ = _exact_circle(cc[0], cc[1], cc[2], s.e1, s.e2, rr)

            if _conic_within_cone(circ, apex, w, s.height):
                out.append(circ)

        return True

    if is_ellipse:
        c3 = _build_exact_plane_cone_ellipse(s.o, s.nu, s.apex, s.w, s.alpha)

        if c3 is not None and _conic_within_cone(c3, s.apex, s.w, s.height):
            out.append(c3)

            return True

    if is_parabola or is_hyperbola:
        c3 = _build_exact_plane_cone_open(
            s.o, s.nu, s.apex, s.w, s.alpha, s.height, is_parabola
        )

        if c3 is not None:
            out.append(c3)

            return True

    return False


def _ssi_plane_cone(plane, cone, cone_srf, out):
    """Plane-cone section: exact lines or conic when possible, fitted arcs otherwise."""

    s = _plane_cone_section(plane, cone, cone_srf)

    if s is None:
        return False

    if abs(s.d0) < 1e-6 * max(1.0, s.height):
        _plane_cone_through_apex(s, out)

        return True

    if _plane_cone_exact(s, out):
        return True

    runs, _ = _sample_plane_cone_arcs(s)

    for r in runs:
        c = _fit_conic_arc(r)

        if c.is_valid():
            out.append(c)

    return True


def _ssi_plane_torus(plane, tor, out):
    """Exact plane-torus circles for a plane perpendicular to the axis."""

    o = plane.p1
    nu = _ssi_unit(plane.p2)
    center = tor.p1
    w = _ssi_unit(tor.p2)
    rmaj = tor.r
    r = tor.r2
    wn = _ssi_dot(w, nu)

    if abs(abs(wn) - 1.0) > 1e-7:
        return False

    d = (
        (o[0] - center[0]) * w[0]
        + (o[1] - center[1]) * w[1]
        + (o[2] - center[2]) * w[2]
    )

    if abs(d) > r:
        return True

    h = math.sqrt(max(0.0, r * r - d * d))
    cc = [center[0] + d * w[0], center[1] + d * w[1], center[2] + d * w[2]]
    xa, ya = _ortho_basis(w)

    for rr in (rmaj + h, rmaj - h):
        if rr > 1e-12:
            out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rr))

    return True


class _FaceFrame:
    """Corner frame of a bilinear face: origin, edge vectors and their Gram matrix."""

    def __init__(self, o, eu, ev):
        self.o = o  # Corner at (u0, v0).
        self.eu = eu  # Edge to (u1, v0).
        self.ev = ev  # Edge to (u0, v1).
        self.exx = _ssi_dot(eu, eu)  # eu . eu
        self.eyy = _ssi_dot(ev, ev)  # ev . ev
        self.exy = _ssi_dot(eu, ev)  # eu . ev
        self.det = self.exx * self.eyy - self.exy * self.exy  # Gram determinant.

    def fraction(self, r):
        """Face fractions (al, be) of the offset r from the corner."""

        rx = _ssi_dot(r, self.eu)
        ry = _ssi_dot(r, self.ev)

        return (self.eyy * rx - self.exy * ry) / self.det, (
            self.exx * ry - self.exy * rx
        ) / self.det


def _face_frame(s):
    """Corner frame of a surface from its corners (u0, v0), (u1, v0) and (u0, v1)."""

    u0, u1 = s.domain(0)
    v0, v1 = s.domain(1)
    o = s.point_at(u0, v0)
    pu = s.point_at(u1, v0)
    pv = s.point_at(u0, v1)

    return _FaceFrame(
        [o[0], o[1], o[2]],
        [pu[0] - o[0], pu[1] - o[1], pu[2] - o[2]],
        [pv[0] - o[0], pv[1] - o[1], pv[2] - o[2]],
    )


def _boundary_steps(s, direction):
    """Boundary samples per side in one direction: cv_count - 1 when linear, else 4 * cv_count."""
    return (
        s.cv_count(direction) - 1
        if s.degree(direction) == 1
        else 4 * s.cv_count(direction)
    )


def _cutter_boundary(cutter):
    """Surface boundary in loop order, each side split into boundary_steps pieces."""

    cu0, cu1 = cutter.domain(0)
    cv0, cv1 = cutter.domain(1)
    nu = _boundary_steps(cutter, 0)
    nv = _boundary_steps(cutter, 1)
    points = []

    for i in range(nu):
        points.append(cutter.point_at(cu0 + (cu1 - cu0) * i / nu, cv0))

    for i in range(nv):
        points.append(cutter.point_at(cu1, cv0 + (cv1 - cv0) * i / nv))

    for i in range(nu, 0, -1):
        points.append(cutter.point_at(cu0 + (cu1 - cu0) * i / nu, cv1))

    for i in range(nv, 0, -1):
        points.append(cutter.point_at(cu0, cv0 + (cv1 - cv0) * i / nv))

    return points


def _boundary_outline(s):
    """Boundary polygon of a surface in the plane through it, empty without area: (outline, frame)."""

    boundary = _cutter_boundary(s)
    normal = Vector(0.0, 0.0, 0.0)

    for i in range(1, len(boundary) - 1):
        normal += (boundary[i] - boundary[0]).cross(boundary[i + 1] - boundary[0])

    if normal.magnitude() < 1e-14:
        return Polyline(), Plane()

    frame = Plane.from_point_normal(boundary[0], normal)
    outline = Polyline()

    for p in boundary:
        d = p - frame.origin
        outline.add_point(Point(d.dot(frame.x_axis), d.dot(frame.y_axis), 0.0))

    return outline, frame


def _is_parallelogram_face(s, f):
    """Whether the surface is the parallelogram of its corner frame, mapped affinely, checked on the boundary grid."""

    if abs(f.det) < 1e-18:
        return False

    cu0, cu1 = s.domain(0)
    cv0, cv1 = s.domain(1)
    nu = _boundary_steps(s, 0)
    nv = _boundary_steps(s, 1)
    tol = 1e-9 * math.sqrt(f.exx + f.eyy)

    for i in range(nu + 1):
        a = i / nu

        for j in range(nv + 1):
            b = j / nv
            p = s.point_at(cu0 + (cu1 - cu0) * a, cv0 + (cv1 - cv0) * b)
            q = Point(
                f.o[0] + a * f.eu[0] + b * f.ev[0],
                f.o[1] + a * f.eu[1] + b * f.ev[1],
                f.o[2] + a * f.eu[2] + b * f.ev[2],
            )

            if p.distance(q) > tol:
                return False

    return True


def _clip_axis(c, d, t0, t1):
    """Narrow [t0, t1] to where c + t d lies in [0, 1]; ok is false when d is zero and c is outside."""

    if abs(d) < 1e-15:
        return (c >= -1e-9 and c <= 1.0 + 1e-9), t0, t1

    ta = (0.0 - c) / d
    tb = (1.0 - c) / d

    if ta > tb:
        ta, tb = tb, ta

    return True, max(t0, ta), min(t1, tb)


def _clip_line_to_face(f, anchor, direction, tmin, tmax):
    """Narrow [tmin, tmax] to the part of the line inside the parallelogram of the frame: (ok, tmin, tmax, empty)."""

    a0, b0 = f.fraction([anchor[0] - f.o[0], anchor[1] - f.o[1], anchor[2] - f.o[2]])
    da, db = f.fraction(direction)
    t0 = -1e300
    t1 = 1e300
    ok, t0, t1 = _clip_axis(a0, da, t0, t1)

    if ok:
        ok, t0, t1 = _clip_axis(b0, db, t0, t1)

    if not ok or t0 > t1:
        return False, tmin, tmax, True

    return True, max(tmin, t0), min(tmax, t1), False


def _clip_line_to_outline(s, anchor, direction, spans):
    """Parameter spans of the line inside the boundary polygon of the face; false when the polygon has no area."""

    outline, frame = _boundary_outline(s)

    if outline.point_count() == 0:
        return False

    offset = Point(anchor[0], anchor[1], anchor[2]) - frame.origin
    line_direction = Vector(direction[0], direction[1], direction[2])
    ax = offset.dot(frame.x_axis)
    ay = offset.dot(frame.y_axis)
    dx = line_direction.dot(frame.x_axis)
    dy = line_direction.dot(frame.y_axis)
    n = outline.point_count()
    ts = []

    for i in range(n):
        a = outline[i]
        b = outline[(i + 1) % n]
        ex = b[0] - a[0]
        ey = b[1] - a[1]
        denom = dx * ey - dy * ex

        if abs(denom) < 1e-15:
            continue

        wx = a[0] - ax
        wy = a[1] - ay
        along = (wx * dy - wy * dx) / denom

        if along >= -1e-12 and along <= 1.0 + 1e-12:
            ts.append((wx * ey - wy * ex) / denom)

    ts.sort()

    for i in range(len(ts) - 1):
        mid = 0.5 * (ts[i] + ts[i + 1])

        if ts[i + 1] - ts[i] <= 1e-9 or not outline.point_in_polygon_2d(
            Point(ax + mid * dx, ay + mid * dy, 0.0)
        ):
            continue

        if spans and ts[i] - spans[-1][1] <= 1e-9:
            spans[-1] = (spans[-1][0], ts[i + 1])
        else:
            spans.append((ts[i], ts[i + 1]))

    return True


def _clip_line_to_face_spans(s, anchor, direction, spans):
    """Parameter spans of the line inside the face, its corner parallelogram, else its boundary polygon: (ok, empty)."""

    f = _face_frame(s)

    if not _is_parallelogram_face(s, f):
        return _clip_line_to_outline(s, anchor, direction, spans), False

    ok, tmin, tmax, empty = _clip_line_to_face(f, anchor, direction, -1e300, 1e300)

    if not ok:
        return False, empty

    spans.append((tmin, tmax))

    return True, False


def _ssi_plane_plane(sa, pa, sb, pb, out):
    """Exact plane-plane line clipped to both finite faces: (hit, empty)."""

    na = _ssi_unit(pa.p2)
    nb = _ssi_unit(pb.p2)
    v = _ssi_cross(na, nb)
    vl = math.sqrt(_ssi_dot(v, v))

    if vl < 1e-9:
        return False, False

    da = _ssi_dot(na, pa.p1)
    db = _ssi_dot(nb, pb.p1)
    nb_x_v = _ssi_cross(nb, v)
    v_x_na = _ssi_cross(v, na)
    inv = 1.0 / (vl * vl)
    anchor = [
        (da * nb_x_v[0] + db * v_x_na[0]) * inv,
        (da * nb_x_v[1] + db * v_x_na[1]) * inv,
        (da * nb_x_v[2] + db * v_x_na[2]) * inv,
    ]
    direction = [v[0] / vl, v[1] / vl, v[2] / vl]
    spans_a = []
    spans_b = []
    ok, empty = _clip_line_to_face_spans(sa, anchor, direction, spans_a)

    if not ok:
        return False, empty

    ok, empty = _clip_line_to_face_spans(sb, anchor, direction, spans_b)

    if not ok:
        return False, empty

    for span_a in spans_a:
        for span_b in spans_b:
            tmin = max(span_a[0], span_b[0])
            tmax = min(span_a[1], span_b[1])

            if tmax - tmin <= 1e-9:
                continue

            start = Point(
                anchor[0] + tmin * direction[0],
                anchor[1] + tmin * direction[1],
                anchor[2] + tmin * direction[2],
            )
            end = Point(
                anchor[0] + tmax * direction[0],
                anchor[1] + tmax * direction[1],
                anchor[2] + tmax * direction[2],
            )
            c3 = NurbsCurve.create(False, 1, [start, end])
            c3.set_domain(0.0, 1.0)
            out.append(c3)

    empty = len(out) == 0

    return not empty, empty


class _AnalyticResult:
    """Tri-state analytic result: not analytic, recognised empty, or curve triples."""

    NOT_ANALYTIC = 0
    NO_HIT = 1
    HIT = 2

    def __init__(self):
        self.status = (
            _AnalyticResult.NOT_ANALYTIC
        )  # Whether both surfaces were recognized and whether they meet.
        self.triples = []  # 3D curve with both pullbacks.


def _unwrap_angle(a, prev):
    """Angle shifted by whole turns to within half a turn of prev."""

    while a - prev > PI:
        a -= TWO_PI

    while a - prev < -PI:
        a += TWO_PI

    return a


def _wrap_angle(a):
    """Angle shifted by whole turns into [-pi, pi]."""

    while a > PI:
        a -= TWO_PI

    while a < -PI:
        a += TWO_PI

    return a


def _wrap_to_range(a, lo, hi):
    """Angle shifted by whole turns into [lo - 1e-9, hi + 1e-9] when the range allows."""

    while a < lo - 1e-9:
        a += TWO_PI

    while a > hi + 1e-9:
        a -= TWO_PI

    return a


def _unwrap_period(x, prev, period):
    """Value shifted by whole periods to within half a period of prev."""

    while x - prev > period * 0.5:
        x -= period

    while x - prev < -period * 0.5:
        x += period

    return x


def _period_index(x, x0, period):
    """Index of the period cell of x counted from x0."""
    return math.floor((x - x0) / period + 1e-9)


def _axis_height(p, origin, axis):
    """Height of p along the unit axis through origin."""

    r = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]]

    return _ssi_dot(r, axis)


def _axis_radial_sq(p, origin, axis):
    """Squared distance of p from the unit axis through origin."""

    r = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]]
    h = _ssi_dot(r, axis)
    px = r[0] - h * axis[0]
    py = r[1] - h * axis[1]
    pz = r[2] - h * axis[2]

    return px * px + py * py + pz * pz


def _curve_gap(c):
    """Distance between the curve's end points."""

    t0, t1 = c.domain()

    return c.point_at(t0).distance(c.point_at(t1))


def _iso_v_line(u0, u1, vc):
    """Constant-v UV line from u0 to u1."""
    return NurbsCurve.create(False, 1, [Point(u0, vc, 0.0), Point(u1, vc, 0.0)])


def _curve_height_stats(c3d, origin, axis):
    """Height range and mean of 33 curve samples along the unit axis through origin."""

    t0, t1 = c3d.domain()
    ns = 33
    hsum = 0.0
    hmin = 1e300
    hmax = -1e300

    for i in range(ns):
        h = _axis_height(c3d.point_at(t0 + (t1 - t0) * i / 32), origin, axis)
        hmin = min(hmin, h)
        hmax = max(hmax, h)
        hsum += h

    return hmin, hmax, hsum / ns


def _bisect_height_v(srf, um, v0, v1, hc, origin, axis):
    """v on the line u = um where the axial height reaches hc, by bisection; None when hc is outside."""

    va = v0
    vb = v1
    ha = _axis_height(srf.point_at(um, va), origin, axis)
    hb = _axis_height(srf.point_at(um, vb), origin, axis)

    if (hc - ha) * (hc - hb) > 0:
        return None

    for _ in range(60):
        vm = 0.5 * (va + vb)
        hm = _axis_height(srf.point_at(um, vm), origin, axis)

        if (hm - hc) * (ha - hc) <= 0:
            vb = vm
        else:
            va = vm
            ha = hm

    return 0.5 * (va + vb)


def _inverted_plane_pcurve(srf, f, c3d):
    """Pcurve on a planar face that is not its corner parallelogram: a polyline through 65 inverted points, invalid when the curve leaves the face."""

    t0, t1 = c3d.domain()
    tol = 1e-6 * math.sqrt(f.exx + f.eyy)
    uvs = []

    for i in range(65):
        p = c3d.point_at(t0 + (t1 - t0) * i / 64.0)
        closest = Closest.surface_point(srf, p, 0.0, 0.0, 0.0, 0.0)

        if closest[2] > tol:
            return NurbsCurve()

        uvs.append(Point(closest[0], closest[1], 0.0))

    pc = NurbsCurve.create(False, 1, uvs)

    if not pc.set_domain(t0, t1):
        return NurbsCurve()

    return pc


def _plane_pcurve(srf, c3d):
    """Plane pcurve: inverted points on a face that is not its corner parallelogram, else the control points mapped to its parameters."""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    f = _face_frame(srf)

    if not _is_parallelogram_face(srf, f):
        inverted = _inverted_plane_pcurve(srf, f, c3d)

        if inverted.is_valid():
            return inverted

    if abs(f.det) < 1e-18:
        return NurbsCurve()

    pc = c3d.duplicate()

    for i in range(c3d.cv_count()):
        cv = c3d.get_cv(i)
        a, b = f.fraction([cv[0] - f.o[0], cv[1] - f.o[1], cv[2] - f.o[2]])
        u = u0 + a * (u1 - u0)
        v = v0 + b * (v1 - v0)

        if c3d.is_rational():
            w = c3d.weight(i)
            pc.set_cv_4d(i, u * w, v * w, 0.0, w)
        else:
            pc.set_cv(i, Point(u, v, 0.0))

    return pc


def _cylinder_pcurve(srf, recog, c3d):
    """Cylinder pcurve of a circle perpendicular to the axis: a constant-v line."""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    ax = list(recog.p2)

    if not _normalize_axis(ax):
        return NurbsCurve()

    um = 0.5 * (u0 + u1)
    h0 = _axis_height(srf.point_at(um, v0), recog.p1, ax)
    h1 = _axis_height(srf.point_at(um, v1), recog.p1, ax)

    if abs(h1 - h0) < 1e-12:
        return NurbsCurve()

    hmin, hmax, hc = _curve_height_stats(c3d, recog.p1, ax)

    if hmax - hmin > 1e-5 * abs(h1 - h0):
        return NurbsCurve()

    if _curve_gap(c3d) > 1e-6 * (abs(h1 - h0) + 1.0):
        return NurbsCurve()

    vc = v0 + (hc - h0) / (h1 - h0) * (v1 - v0)

    if vc < min(v0, v1) - 1e-9 or vc > max(v0, v1) + 1e-9:
        return NurbsCurve()

    return _iso_v_line(u0, u1, vc)


def _sphere_pcurve(srf, recog, c3d):
    """Sphere pcurve of a latitude circle: a constant-v line."""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    um = 0.5 * (u0 + u1)
    sp = srf.point_at(um, v0)
    np_ = srf.point_at(um, v1)
    ax = [np_[0] - sp[0], np_[1] - sp[1], np_[2] - sp[2]]

    if not _normalize_axis(ax):
        return NurbsCurve()

    hmin, hmax, hc = _curve_height_stats(c3d, recog.p1, ax)

    if hmax - hmin > recog.r * 1e-4:
        return NurbsCurve()

    if _curve_gap(c3d) > recog.r * 1e-3:
        return NurbsCurve()

    vc = _bisect_height_v(srf, um, v0, v1, hc, recog.p1, ax)

    if vc is None:
        return NurbsCurve()

    return _iso_v_line(u0, u1, vc)


def _cone_pcurve(srf, recog, c3d):
    """Cone pcurve of a circle perpendicular to the axis: a constant-v line."""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    ax = list(recog.p2)

    if not _normalize_axis(ax):
        return NurbsCurve()

    t0, t1 = c3d.domain()
    clen = c3d.point_at(t0).distance(c3d.point_at(0.5 * (t0 + t1)))
    hscale = max(clen, 1e-9)
    hmin, hmax, hc = _curve_height_stats(c3d, recog.p1, ax)

    if hmax - hmin > hscale * 1e-4:
        return NurbsCurve()

    if _curve_gap(c3d) > hscale * 1e-3:
        return NurbsCurve()

    vc = _bisect_height_v(srf, 0.5 * (u0 + u1), v0, v1, hc, recog.p1, ax)

    if vc is None:
        return NurbsCurve()

    return _iso_v_line(u0, u1, vc)


def _torus_minor_angle(p, center, w, rmaj):
    """Tube angle of p about the circle of radius rmaj around the unit axis w through center."""

    d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]]
    z = _ssi_dot(d, w)
    hx = d[0] - z * w[0]
    hy = d[1] - z * w[1]
    hz = d[2] - z * w[2]
    rho = math.sqrt(hx * hx + hy * hy + hz * hz)

    return math.atan2(z, rho - rmaj)


def _torus_angle_stats(c3d, center, w, rmaj):
    """Range and mean of the unwrapped tube angle over 33 curve samples."""

    t0, t1 = c3d.domain()
    ns = 33
    aprev = 0.0
    asum = 0.0
    amin = 1e300
    amax = -1e300

    for i in range(ns):
        a = _torus_minor_angle(c3d.point_at(t0 + (t1 - t0) * i / 32), center, w, rmaj)

        if i > 0:
            a = _unwrap_angle(a, aprev)

        aprev = a
        amin = min(amin, a)
        amax = max(amax, a)
        asum += a

    return amin, amax, asum / ns


def _torus_angle_table(srf, um, v0, v1, center, w, rmaj):
    """Unwrapped tube angle at 257 samples of the line u = um."""

    nv = 256
    tv = [0.0] * (nv + 1)
    ta = [0.0] * (nv + 1)
    ap = 0.0

    for k in range(nv + 1):
        v = v0 + (v1 - v0) * k / nv
        a = _torus_minor_angle(srf.point_at(um, v), center, w, rmaj)

        if k > 0:
            a = _unwrap_angle(a, ap)

        ap = a
        tv[k] = v
        ta[k] = a

    return tv, ta


def _inverse_table(xs, ys, y):
    """Parameter and value arrays read backwards: the x where the tabulated y reaches y."""

    nt = len(ys) - 1
    incr = ys[nt] >= ys[0]
    y = _wrap_to_range(y, min(ys[0], ys[nt]), max(ys[0], ys[nt]))
    lo = 0
    hi = nt

    while hi - lo > 1:
        mid = (lo + hi) // 2
        above = (ys[mid] < y) if incr else (ys[mid] > y)

        if above:
            lo = mid
        else:
            hi = mid

    denom = ys[hi] - ys[lo]
    f = (y - ys[lo]) / denom if abs(denom) > 1e-15 else 0.0

    return xs[lo] + (xs[hi] - xs[lo]) * f


def _torus_refine_v(srf, um, vc, a_target, v0, v1, center, w, rmaj):
    """Newton-refine v on the line u = um so the tube angle reaches a_target."""

    dv = (v1 - v0) * 1e-7
    vlo = min(v0, v1)
    vhi = max(v0, v1)

    for _ in range(3):
        g0 = _wrap_angle(
            _torus_minor_angle(
                srf.point_at(um, min(max(vc, vlo), vhi)), center, w, rmaj
            )
            - a_target
        )
        vd = min(vc + dv, vhi)
        g1 = _wrap_angle(
            _torus_minor_angle(
                srf.point_at(um, min(max(vd, vlo), vhi)), center, w, rmaj
            )
            - a_target
        )
        dg = (g1 - g0) / dv

        if abs(dg) < 1e-12:
            break

        vn = min(max(vc - g0 / dg, vlo), vhi)

        if abs(vn - vc) <= 1e-15 * max(1.0, abs(vc)):
            vc = vn
            break

        vc = vn

    return vc


def _torus_pcurve(srf, recog, c3d):
    """Torus pcurve of a circle of constant tube angle: a constant-v line."""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    w = list(recog.p2)

    if not _normalize_axis(w):
        return NurbsCurve()

    rmaj = recog.r
    rmin = recog.r2

    if rmin < 1e-12 or rmaj <= rmin:
        return NurbsCurve()

    amin, amax, a_target = _torus_angle_stats(c3d, recog.p1, w, rmaj)

    if amax - amin > 1e-4:
        return NurbsCurve()

    if _curve_gap(c3d) > rmin * 1e-3:
        return NurbsCurve()

    um = 0.5 * (u0 + u1)
    tv, ta = _torus_angle_table(srf, um, v0, v1, recog.p1, w, rmaj)
    alo = min(ta[0], ta[-1])
    ahi = max(ta[0], ta[-1])
    a_target = _wrap_to_range(a_target, alo, ahi)

    if a_target < alo - 1e-9 or a_target > ahi + 1e-9:
        return NurbsCurve()

    vc = _torus_refine_v(
        srf, um, _inverse_table(tv, ta, a_target), a_target, v0, v1, recog.p1, w, rmaj
    )

    return _iso_v_line(u0, u1, vc)


def _analytic_pcurve(srf, recog, c3d):
    """Analytic pcurve of an exact 3D intersection conic on a recognized quadric surface."""

    if recog.kind == _RecogSurface.PLANE:
        return _plane_pcurve(srf, c3d)

    if recog.kind == _RecogSurface.CYLINDER:
        return _cylinder_pcurve(srf, recog, c3d)

    if recog.kind == _RecogSurface.SPHERE:
        return _sphere_pcurve(srf, recog, c3d)

    if recog.kind == _RecogSurface.CONE:
        return _cone_pcurve(srf, recog, c3d)

    if recog.kind == _RecogSurface.TORUS:
        return _torus_pcurve(srf, recog, c3d)

    return NurbsCurve()


class _AxisFrame:
    """Orthonormal frame about a surface axis."""

    def __init__(self, o, x, y, z):
        self.o = o  # Origin on the axis.
        self.x = x  # First radial direction.
        self.y = y  # Second radial direction.
        self.z = z  # Unit axis.


def _axis_frame(origin, z, p):
    """Frame about the unit axis z through origin with x towards p; None when p lies on the axis."""

    r = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]]
    h = _ssi_dot(r, z)
    x = [r[0] - h * z[0], r[1] - h * z[1], r[2] - h * z[2]]

    if not _normalize_axis(x):
        return None

    return _AxisFrame(origin, x, _ssi_cross(z, x), z)


def _frame_longitude(f, q):
    """Longitude of q about the frame axis."""

    r = [q[0] - f.o[0], q[1] - f.o[1], q[2] - f.o[2]]

    return math.atan2(_ssi_dot(r, f.y), _ssi_dot(r, f.x))


def _frame_tube_angle(f, rmaj, rmin, q):
    """Torus tube angle of q for major radius rmaj and minor radius rmin."""

    rho = math.sqrt(_axis_radial_sq(q, f.o, f.z))

    return math.atan2(_axis_height(q, f.o, f.z) / rmin, (rho - rmaj) / rmin)


class _AngleProbe:
    """Angle of surface points along one parameter line: longitude, or the torus tube angle."""

    def __init__(self, srf, frame, fixed, x_is_u, tube, rmaj, rmin):
        self.srf = srf  # Sampled surface.
        self.frame = frame  # Frame about the surface axis.
        self.fixed = fixed  # The parameter held fixed.
        self.x_is_u = x_is_u  # Whether the free parameter is u.
        self.tube = tube  # Tube angle instead of longitude.
        self.rmaj = rmaj  # Torus major radius.
        self.rmin = rmin  # Torus minor radius.

    def point(self, x):
        """Surface point at free parameter x."""
        return (
            self.srf.point_at(x, self.fixed)
            if self.x_is_u
            else self.srf.point_at(self.fixed, x)
        )

    def angle(self, x):
        """Angle at free parameter x."""

        q = self.point(x)

        return (
            _frame_tube_angle(self.frame, self.rmaj, self.rmin, q)
            if self.tube
            else _frame_longitude(self.frame, q)
        )


class _AngleMap:
    """Tabulated angle along one parameter line."""

    def __init__(self, probe, lo, hi, xs, ys):
        self.probe = probe  # Angle along the parameter line.
        self.lo = lo  # Parameter start.
        self.hi = hi  # Parameter end.
        self.xs = xs  # Tabulated parameters.
        self.ys = ys  # Unwrapped angles at xs.


def _angle_map(probe, lo, hi):
    """Tabulate 129 unwrapped angles of the probe over [lo, hi]."""

    nt = 128
    rng = hi - lo
    xs = [0.0] * (nt + 1)
    ys = [0.0] * (nt + 1)

    for k in range(nt + 1):
        x = lo + rng * k / nt
        y = probe.angle(x)

        if k > 0:
            y = _unwrap_angle(y, ys[k - 1])

        xs[k] = x
        ys[k] = y

    return _AngleMap(probe, lo, hi, xs, ys)


def _polish_angle(probe, x, y, lo, hi):
    """Two Newton steps moving x in [lo, hi] until the probe angle reaches y."""

    dx = (hi - lo) * 1e-7

    for _ in range(2):
        xc = min(max(x, lo), hi)
        g0 = _wrap_angle(probe.angle(xc) - y)
        g1 = _wrap_angle(probe.angle(min(xc + dx, hi)) - y)
        dg = (g1 - g0) / dx

        if abs(dg) < 1e-12:
            break

        x = min(max(xc - g0 / dg, lo), hi)

    return x


def _map_parameter(m, y):
    """Parameter where the tabulated angle reaches y, Newton-polished."""
    return _polish_angle(m.probe, _inverse_table(m.xs, m.ys, y), y, m.lo, m.hi)


def _height_table(srf, f, um, v0, v1):
    """Height along the frame axis at 129 samples of the line u = um."""

    nt = 128
    tv = [0.0] * (nt + 1)
    th = [0.0] * (nt + 1)

    for k in range(nt + 1):
        v = v0 + (v1 - v0) * k / nt
        tv[k] = v
        th[k] = _axis_height(srf.point_at(um, v), f.o, f.z)

    return tv, th


def _clamped_table(xs, ys, y):
    """The x where the tabulated y reaches y, clamped to the table ends."""

    nt = len(ys) - 1
    incr = ys[nt] >= ys[0]

    if incr and y <= ys[0]:
        return xs[0]

    if incr and y >= ys[nt]:
        return xs[nt]

    if not incr and y >= ys[0]:
        return xs[0]

    if not incr and y <= ys[nt]:
        return xs[nt]

    lo = 0
    hi = nt

    while hi - lo > 1:
        mid = (lo + hi) // 2
        above = (ys[mid] < y) if incr else (ys[mid] > y)

        if above:
            lo = mid
        else:
            hi = mid

    denom = ys[hi] - ys[lo]
    f = (y - ys[lo]) / denom if abs(denom) > 1e-15 else 0.0

    return xs[lo] + (xs[hi] - xs[lo]) * f


def _sphere_refine_v(srf, f, um, v, h, v0, v1):
    """Two Newton steps moving v on the line u = um until the axial height reaches h."""

    vlo = min(v0, v1)
    vhi = max(v0, v1)

    for _ in range(2):
        dv = (v1 - v0) * 1e-7
        vc = min(max(v, vlo), vhi)
        g0 = _axis_height(srf.point_at(um, vc), f.o, f.z) - h
        g1 = _axis_height(srf.point_at(um, min(vc + dv, vhi)), f.o, f.z) - h
        dg = (g1 - g0) / dv

        if abs(dg) < 1e-12:
            break

        v = min(max(vc - g0 / dg, vlo), vhi)

    return v


def _split_pullback_u(uv, u0, range_u):
    """Degree-1 pcurves of (u, v) samples with u unwrapped, split where u crosses the seam."""

    out = []
    seg = []
    cur_k = _period_index(uv[0][0], u0, range_u)
    seg.append(Point(uv[0][0] - cur_k * range_u, uv[0][1], 0.0))

    for i in range(1, len(uv)):
        ki = _period_index(uv[i][0], u0, range_u)

        while ki != cur_k:
            step = 1 if ki > cur_k else -1
            nk = cur_k + step
            seam_cont = u0 + (nk if step > 0 else cur_k) * range_u
            denom = uv[i][0] - uv[i - 1][0]
            f = (seam_cont - uv[i - 1][0]) / denom if abs(denom) > 1e-15 else 0.0
            f = min(max(f, 0.0), 1.0)
            vc = uv[i - 1][1] + (uv[i][1] - uv[i - 1][1]) * f
            seg.append(Point(seam_cont - cur_k * range_u, vc, 0.0))

            if len(seg) >= 2:
                out.append(NurbsCurve.create(False, 1, seg))

            seg = [Point(seam_cont - nk * range_u, vc, 0.0)]
            cur_k = nk

        seg.append(Point(uv[i][0] - cur_k * range_u, uv[i][1], 0.0))

    if len(seg) >= 2:
        out.append(NurbsCurve.create(False, 1, seg))

    return out


def _analytic_sphere_pullback(srf, recog, c3d):
    """Pull a 3D curve back to sphere parameters through longitude and latitude."""

    if recog.kind != _RecogSurface.SPHERE:
        return []

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    range_u = u1 - u0

    if range_u < 1e-9:
        return []

    um = 0.5 * (u0 + u1)
    vm = 0.5 * (v0 + v1)
    sp = srf.point_at(um, v0)
    np_ = srf.point_at(um, v1)
    axis = [np_[0] - sp[0], np_[1] - sp[1], np_[2] - sp[2]]
    frame = (
        _axis_frame(recog.p1, axis, srf.point_at(u0, vm))
        if _normalize_axis(axis)
        else None
    )

    if frame is None:
        return []

    lon_map = _angle_map(_AngleProbe(srf, frame, vm, True, False, 0.0, 0.0), u0, u1)
    tv, th = _height_table(srf, frame, um, v0, v1)

    if abs(th[-1] - th[0]) < 1e-12:
        return []

    t0, t1 = c3d.domain()
    n = max(c3d.cv_count() * 8, 120)
    uv = []
    prev_u = 0.0

    for i in range(n + 1):
        p = c3d.point_at(t0 + (t1 - t0) * i / n)
        h = _axis_height(p, frame.o, frame.z)
        u = _map_parameter(lon_map, _frame_longitude(frame, p))
        v = _sphere_refine_v(srf, frame, um, _clamped_table(tv, th, h), h, v0, v1)

        if i > 0:
            u = _unwrap_period(u, prev_u, range_u)

        prev_u = u
        uv.append((u, v))

    return _split_pullback_u(uv, u0, range_u)


def _cone_pullback_samples(c3d, frame, lon_map, h0, h1, v0, v1):
    """Samples (u, v) of a curve on a cone or cylinder, u unwrapped and v linear in the axial height."""

    t0, t1 = c3d.domain()
    range_u = lon_map.hi - lon_map.lo
    n = max(c3d.cv_count() * 8, 120)
    prev_lon = 0.0
    uv = []
    prev_u = 0.0

    for i in range(n + 1):
        p = c3d.point_at(t0 + (t1 - t0) * i / n)
        r = [p[0] - frame.o[0], p[1] - frame.o[1], p[2] - frame.o[2]]
        rx = _ssi_dot(r, frame.x)
        ry = _ssi_dot(r, frame.y)
        rad = math.sqrt(max(0.0, rx * rx + ry * ry))
        lon = math.atan2(ry, rx) if rad > 1e-12 else prev_lon
        prev_lon = lon
        u = (
            _map_parameter(lon_map, lon)
            if rad > 1e-12
            else _inverse_table(lon_map.xs, lon_map.ys, lon)
        )
        v = v0 + (_ssi_dot(r, frame.z) - h0) / (h1 - h0) * (v1 - v0)

        if i > 0:
            u = _unwrap_period(u, prev_u, range_u)

        prev_u = u
        uv.append((u, v))

    return uv


def _analytic_cone_pullback(srf, recog, c3d):
    """Analytic pull-back of a 3D curve onto a recognized cone or cylinder."""

    if recog.kind != _RecogSurface.CONE and recog.kind != _RecogSurface.CYLINDER:
        return []

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    range_u = u1 - u0
    axis = list(recog.p2)

    if range_u < 1e-9 or not _normalize_axis(axis):
        return []

    um = 0.5 * (u0 + u1)
    h0 = _axis_height(srf.point_at(um, v0), recog.p1, axis)
    h1 = _axis_height(srf.point_at(um, v1), recog.p1, axis)

    if abs(h1 - h0) < 1e-12:
        return []

    v_ref = v0 if abs(h0) >= abs(h1) else v1
    frame = _axis_frame(recog.p1, axis, srf.point_at(u0, v_ref))

    if frame is None:
        return []

    lon_map = _angle_map(_AngleProbe(srf, frame, v_ref, True, False, 0.0, 0.0), u0, u1)
    uv = _cone_pullback_samples(c3d, frame, lon_map, h0, h1, v0, v1)

    return _split_pullback_u(uv, u0, range_u)


class _PeriodGrid:
    """Period cells of a torus pull-back in (a, b), swapped when a is the surface v."""

    def __init__(self, a0, range_a, b0, range_b, swapped):
        self.a0 = a0  # Start of a.
        self.range_a = range_a  # Period of a.
        self.b0 = b0  # Start of b.
        self.range_b = range_b  # Period of b.
        self.swapped = swapped  # Whether a is the surface v.


def _push_pullback_point(g, seg, a, b, ka, kb):
    """Append the point (a, b) shifted into cell (ka, kb) in surface (u, v) order."""

    uu = a - ka * g.range_a
    vv = b - kb * g.range_b
    seg.append(Point(vv, uu, 0.0) if g.swapped else Point(uu, vv, 0.0))


def _cross_period(g, p, q, ka, kb, seg, out):
    """Split the step p -> q at its first cell boundary: (crossed, p, ka, kb), false when q is in the current cell."""

    kqa = _period_index(q[0], g.a0, g.range_a)
    kqb = _period_index(q[1], g.b0, g.range_b)

    if kqa == ka and kqb == kb:
        return False, p, ka, kb

    fa = 2.0
    fb = 2.0
    sa = 0
    sb = 0

    if kqa != ka:
        sa = 1 if kqa > ka else -1
        bound = g.a0 + (ka + 1 if sa > 0 else ka) * g.range_a
        den = q[0] - p[0]
        fa = (bound - p[0]) / den if abs(den) > 1e-15 else 0.0

    if kqb != kb:
        sb = 1 if kqb > kb else -1
        bound = g.b0 + (kb + 1 if sb > 0 else kb) * g.range_b
        den = q[1] - p[1]
        fb = (bound - p[1]) / den if abs(den) > 1e-15 else 0.0

    if fa <= fb:
        c = (
            g.a0 + (ka + 1 if sa > 0 else ka) * g.range_a,
            p[1] + (q[1] - p[1]) * min(max(fa, 0.0), 1.0),
        )
    else:
        c = (
            p[0] + (q[0] - p[0]) * min(max(fb, 0.0), 1.0),
            g.b0 + (kb + 1 if sb > 0 else kb) * g.range_b,
        )

    _push_pullback_point(g, seg, c[0], c[1], ka, kb)

    if len(seg) >= 2:
        out.append(NurbsCurve.create(False, 1, seg))

    seg.clear()

    if fa <= fb:
        ka += sa
    else:
        kb += sb

    _push_pullback_point(g, seg, c[0], c[1], ka, kb)

    return True, c, ka, kb


def _split_pullback_ab(ab, g):
    """Degree-1 pcurves of (a, b) samples with a and b unwrapped, split at both seams."""

    out = []
    seg = []
    ka = _period_index(ab[0][0], g.a0, g.range_a)
    kb = _period_index(ab[0][1], g.b0, g.range_b)
    _push_pullback_point(g, seg, ab[0][0], ab[0][1], ka, kb)

    for i in range(1, len(ab)):
        p = ab[i - 1]

        for _ in range(8):
            crossed, p, ka, kb = _cross_period(g, p, ab[i], ka, kb, seg, out)

            if not crossed:
                break

        _push_pullback_point(g, seg, ab[i][0], ab[i][1], ka, kb)

    if len(seg) >= 2:
        out.append(NurbsCurve.create(False, 1, seg))

    return out


def _farthest_from_axis(srf, center, axis):
    """Sample of a 5 x 5 grid farthest from the unit axis through center."""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    pf = srf.point_at(u0, v0)
    best = -1.0

    for i in range(5):
        for j in range(5):
            q = srf.point_at(u0 + (u1 - u0) * i / 4.0, v0 + (v1 - v0) * j / 4.0)
            d = _axis_radial_sq(q, center, axis)

            if d > best:
                best = d
                pf = q

    return pf


def _torus_swapped(srf, frame):
    """Whether the torus's longitude runs along v rather than u."""

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    um = 0.5 * (u0 + u1)
    vm = 0.5 * (v0 + v1)
    lu1 = _frame_longitude(frame, srf.point_at(u0 + 0.6 * (u1 - u0), vm))
    lu0 = _frame_longitude(frame, srf.point_at(u0 + 0.3 * (u1 - u0), vm))
    lv1 = _frame_longitude(frame, srf.point_at(um, v0 + 0.6 * (v1 - v0)))
    lv0 = _frame_longitude(frame, srf.point_at(um, v0 + 0.3 * (v1 - v0)))

    return abs(_wrap_angle(lv1 - lv0)) > abs(_wrap_angle(lu1 - lu0))


def _farthest_on_line(probe, lo, hi):
    """Free parameter of 17 samples along the probe line farthest from the axis."""

    rng = hi - lo
    x_ref = lo
    best = -1.0

    for j in range(17):
        x = lo + rng * j / 16.0
        d = _axis_radial_sq(probe.point(x), probe.frame.o, probe.frame.z)

        if d > best:
            best = d
            x_ref = x

    return x_ref


def _analytic_torus_pullback(srf, recog, c3d):
    """Analytic pull-back of a 3D curve onto a recognized torus."""

    domain_u = srf.domain(0)
    domain_v = srf.domain(1)
    axis = list(recog.p2)
    rmaj = recog.r
    rmin = recog.r2

    if (
        recog.kind != _RecogSurface.TORUS
        or domain_u[1] - domain_u[0] < 1e-9
        or domain_v[1] - domain_v[0] < 1e-9
    ):
        return []

    if not _normalize_axis(axis) or rmaj < 1e-12 or rmin < 1e-12:
        return []

    frame = _axis_frame(recog.p1, axis, _farthest_from_axis(srf, recog.p1, axis))

    if frame is None:
        return []

    swapped = _torus_swapped(srf, frame)
    a0, a1 = domain_v if swapped else domain_u
    b0, b1 = domain_u if swapped else domain_v
    tube_probe = _AngleProbe(srf, frame, 0.5 * (a0 + a1), swapped, True, rmaj, rmin)
    b_ref = _farthest_on_line(tube_probe, b0, b1)
    lon_map = _angle_map(
        _AngleProbe(srf, frame, b_ref, not swapped, False, 0.0, 0.0), a0, a1
    )
    tube_map = _angle_map(tube_probe, b0, b1)
    t0, t1 = c3d.domain()
    n = max(c3d.cv_count() * 8, 4000)
    ab = []
    prev_a = 0.0
    prev_b = 0.0

    for i in range(n + 1):
        q = c3d.point_at(t0 + (t1 - t0) * i / n)
        a = _map_parameter(lon_map, _frame_longitude(frame, q))
        b = _map_parameter(tube_map, _frame_tube_angle(frame, rmaj, rmin, q))

        if i > 0:
            a = _unwrap_period(a, prev_a, a1 - a0)
            b = _unwrap_period(b, prev_b, b1 - b0)

        prev_a = a
        prev_b = b
        ab.append((a, b))

    return _split_pullback_ab(ab, _PeriodGrid(a0, a1 - a0, b0, b1 - b0, swapped))


def _analytic_pullback(srf, recog, c3d):
    """Analytic pull-back matching the recognized kind: sphere, cone or cylinder, torus."""

    if recog.kind == _RecogSurface.TORUS:
        return _analytic_torus_pullback(srf, recog, c3d)

    if recog.kind == _RecogSurface.SPHERE:
        return _analytic_sphere_pullback(srf, recog, c3d)

    if recog.kind == _RecogSurface.CONE or recog.kind == _RecogSurface.CYLINDER:
        return _analytic_cone_pullback(srf, recog, c3d)

    return []


# ═══════════════════════════════════════════════════════════════════════════
# Coaxial quadric pairs
# ═══════════════════════════════════════════════════════════════════════════
def _point_axis_dist(apt, adir, p):
    """Distance of p from the axis through apt along adir."""

    u = _ssi_unit(adir)
    dp = [p[0] - apt[0], p[1] - apt[1], p[2] - apt[2]]
    t = _ssi_dot(dp, u)
    perp = [dp[0] - t * u[0], dp[1] - t * u[1], dp[2] - t * u[2]]

    return math.sqrt(_ssi_dot(perp, perp))


def _axial_coord(apt, adir, p):
    """Coordinate of p along the axis through apt along adir."""

    u = _ssi_unit(adir)

    return (p[0] - apt[0]) * u[0] + (p[1] - apt[1]) * u[1] + (p[2] - apt[2]) * u[2]


def _axes_coaxial(p1, d1, p2, d2, tol):
    """Whether two axes coincide within tol."""

    u1 = _ssi_unit(d1)
    u2 = _ssi_unit(d2)
    cx = _ssi_cross(u1, u2)

    if math.sqrt(_ssi_dot(cx, cx)) > tol:
        return False

    return _point_axis_dist(p1, u1, p2) <= tol


def _cyl_span(srf, apt, adir):
    """Axial extent of the surface along the cylinder axis."""

    u = _ssi_unit(adir)
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    um = 0.5 * (u0 + u1)
    smin = 1e300
    smax = -1e300

    for vv in (v0, v1):
        p = srf.point_at(um, vv)
        s = (p[0] - apt[0]) * u[0] + (p[1] - apt[1]) * u[1] + (p[2] - apt[2]) * u[2]
        smin = min(smin, s)
        smax = max(smax, s)

    return smin, smax


def _lines_closest_point(p1, d1, p2, d2, tol):
    """Closest point of two lines, None when parallel or apart."""

    u = _ssi_unit(d1)
    v = _ssi_unit(d2)
    w0 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
    a = _ssi_dot(u, u)
    b = _ssi_dot(u, v)
    c = _ssi_dot(v, v)
    d = _ssi_dot(u, w0)
    e = _ssi_dot(v, w0)
    den = a * c - b * b

    if abs(den) < 1e-12:
        return None

    sc = (b * e - c * d) / den
    tc = (a * e - b * d) / den
    q1 = [p1[0] + sc * u[0], p1[1] + sc * u[1], p1[2] + sc * u[2]]
    q2 = [p2[0] + tc * v[0], p2[1] + tc * v[1], p2[2] + tc * v[2]]
    diff = [q1[0] - q2[0], q1[1] - q2[1], q1[2] - q2[2]]

    if math.sqrt(_ssi_dot(diff, diff)) > tol:
        return None

    return [0.5 * (q1[0] + q2[0]), 0.5 * (q1[1] + q2[1]), 0.5 * (q1[2] + q2[2])]


def _axis_circles(center, w, zs, rad, out):
    """Circles of radius rad around the unit axis w through center at axial offsets zs."""

    xa, ya = _ortho_basis(w)

    for z in zs:
        cc = [center[0] + z * w[0], center[1] + z * w[1], center[2] + z * w[2]]
        out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rad))


def _ssi_cylinder_sphere(cyl, sph, out):
    """Coaxial cylinder-sphere section: circles."""

    ktol = 1e-6
    p = cyl.p1
    w = _ssi_unit(cyl.p2)
    rc = cyl.r
    center = sph.p1
    rsph = sph.r

    if _point_axis_dist(p, w, center) > ktol:
        return False

    if rsph < rc - ktol:
        return True

    dist = math.sqrt(max(0.0, rsph * rsph - rc * rc))

    if dist <= ktol:
        xa, ya = _ortho_basis(w)
        out.append(_exact_circle(center[0], center[1], center[2], xa, ya, rc))

        return True

    _axis_circles(center, w, [dist, -dist], rc, out)

    return True


def _ssi_cylinder_cone(cyl, cone, out):
    """Coaxial cylinder-cone section: circles."""

    ktol = 1e-6
    pc = cyl.p1
    w = _ssi_unit(cyl.p2)
    rc = cyl.r
    apex = cone.p1
    a = _ssi_unit(cone.p2)
    alpha = cone.r

    if not _axes_coaxial(pc, w, apex, a, ktol):
        return False

    ta = math.tan(alpha)

    if ta < 1e-9:
        return False

    s = rc / ta

    if s < ktol:
        return True

    _axis_circles(apex, a, [s], rc, out)

    return True


def _ssi_cone_sphere(cone, sph, out):
    """Coaxial cone-sphere section: circles."""

    ktol = 1e-6
    apex = cone.p1
    a = _ssi_unit(cone.p2)
    alpha = cone.r
    center = sph.p1
    rsph = sph.r

    if _point_axis_dist(apex, a, center) > ktol:
        return False

    dsign = _axial_coord(apex, a, center)
    d = abs(dsign)
    direction = [-a[0], -a[1], -a[2]] if (d > ktol and dsign < 0.0) else a
    t = math.tan(alpha)
    t2 = t * t
    qa = 1.0 + t2
    qb = 2.0 * t2 * d
    qc = t2 * d * d - rsph * rsph
    disc = qb * qb - 4.0 * qa * qc

    if disc < -ktol:
        return True

    sq = math.sqrt(max(0.0, disc))

    if sq <= ktol:
        xs = [-qb / (2.0 * qa)]
    else:
        xs = [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)]

    xa, ya = _ortho_basis(a)

    for x in xs:
        sAx = d + x

        if sAx < ktol:
            continue

        rr = t * sAx

        if rr < ktol:
            continue

        cc = [
            apex[0] + sAx * direction[0],
            apex[1] + sAx * direction[1],
            apex[2] + sAx * direction[2],
        ]
        out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rr))

    return True


def _parallel_cylinder_feet(p1, w1, r1, p2, r2, d, ktol):
    """Points where the cross-section circles of two parallel cylinders at axis distance d meet, one when they touch."""

    off = _ssi_dot([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]], w1)
    p2p = [p2[0] - off * w1[0], p2[1] - off * w1[1], p2[2] - off * w1[2]]
    xdir = _ssi_unit([p2p[0] - p1[0], p2p[1] - p1[1], p2p[2] - p1[2]])
    ydir = _ssi_unit(_ssi_cross(w1, xdir))
    aa = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h = math.sqrt(max(0.0, r1 * r1 - aa * aa))
    foot = [p1[0] + aa * xdir[0], p1[1] + aa * xdir[1], p1[2] + aa * xdir[2]]
    feet = []

    if h <= ktol:
        feet.append(foot)
    else:
        feet.append(
            [foot[0] + h * ydir[0], foot[1] + h * ydir[1], foot[2] + h * ydir[2]]
        )
        feet.append(
            [foot[0] - h * ydir[0], foot[1] - h * ydir[1], foot[2] - h * ydir[2]]
        )

    return feet


def _ssi_parallel_cylinders(sa, ra, sb, rb, out):
    """Parallel cylinders: shared ruling lines, false when coaxial with equal radii."""

    ktol = 1e-6
    p1 = ra.p1
    w1 = _ssi_unit(ra.p2)
    r1 = ra.r
    p2 = rb.p1
    r2 = rb.r
    d = _point_axis_dist(p1, w1, p2)

    if d <= ktol:
        if abs(r1 - r2) <= ktol:
            return False

        return True

    if d > r1 + r2 + ktol or d < abs(r1 - r2) - ktol:
        return True

    s0a, s1a = _cyl_span(sa, p1, w1)
    s0b, s1b = _cyl_span(sb, p1, w1)
    slo = max(s0a, s0b)
    shi = min(s1a, s1b)

    if shi - slo <= ktol:
        return True

    for bp in _parallel_cylinder_feet(p1, w1, r1, p2, r2, d, ktol):
        line = _axis_segment(bp, w1, slo, shi)
        line.set_domain(0.0, 1.0)
        out.append(line)

    return True


def _ssi_cylinder_cylinder(sa, ra, sb, rb, out):
    """Cylinder-cylinder section: lines when parallel, Steinmetz ellipses when equal axes meet."""

    ktol = 1e-6
    p1 = ra.p1
    w1 = _ssi_unit(ra.p2)
    r1 = ra.r
    p2 = rb.p1
    w2 = _ssi_unit(rb.p2)
    r2 = rb.r
    cx = _ssi_cross(w1, w2)

    if math.sqrt(_ssi_dot(cx, cx)) <= ktol:
        return _ssi_parallel_cylinders(sa, ra, sb, rb, out)

    rmax = max(r1, r2)

    if rmax < 1e-12 or abs(r1 - r2) / rmax > 1e-6:
        return False

    pint = _lines_closest_point(p1, w1, p2, w2, ktol)

    if pint is None:
        return False

    r = 0.5 * (r1 + r2)
    ang = math.acos(max(-1.0, min(1.0, _ssi_dot(w1, w2))))
    sh = math.sin(0.5 * ang)
    ch = math.cos(0.5 * ang)

    if sh < 1e-9 or ch < 1e-9:
        return False

    minor = _ssi_unit(cx)
    maj1 = _ssi_unit([w1[0] + w2[0], w1[1] + w2[1], w1[2] + w2[2]])
    maj2 = _ssi_unit([w1[0] - w2[0], w1[1] - w2[1], w1[2] - w2[2]])
    out.append(_exact_ellipse(pint[0], pint[1], pint[2], maj1, minor, r / sh, r))
    out.append(_exact_ellipse(pint[0], pint[1], pint[2], maj2, minor, r / ch, r))

    return True


def _ssi_cylinder_torus(cyl, tor, out):
    """Exact circles of a coaxial cylinder-torus pair."""

    ktol = 1e-6
    p = cyl.p1
    wc = _ssi_unit(cyl.p2)
    rc = cyl.r
    center = tor.p1
    w = _ssi_unit(tor.p2)
    rmaj = tor.r
    r = tor.r2

    if r >= rmaj - ktol:
        return False

    if not _axes_coaxial(p, wc, center, w, ktol):
        return False

    dr = rc - rmaj
    h2 = r * r - dr * dr

    if h2 < -ktol:
        return True

    h = math.sqrt(max(0.0, h2))

    if h <= ktol:
        _axis_circles(center, w, [0.0], rc, out)
    else:
        _axis_circles(center, w, [h, -h], rc, out)

    return True


def _cone_torus_circles(center, w, t, za, r, rsign, out):
    """Circles of a coaxial cone and one side (rsign = +rmaj or -rmaj) of the torus tube."""

    ktol = 1e-6
    qa = t * t + 1.0
    qb = -2.0 * t * (t * za + rsign)
    qc = (t * za + rsign) * (t * za + rsign) - r * r
    disc = qb * qb - 4.0 * qa * qc

    if disc < -ktol:
        return

    sq = math.sqrt(max(0.0, disc))

    if sq <= ktol:
        zs = [-qb / (2.0 * qa)]
    else:
        zs = [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)]

    for z in zs:
        rad = t * abs(z - za)

        if rad < ktol:
            continue

        _axis_circles(center, w, [z], rad, out)


def _ssi_cone_torus(cone, tor, out):
    """Coaxial cone-torus section: circles."""

    ktol = 1e-6
    apex = cone.p1
    a = _ssi_unit(cone.p2)
    alpha = cone.r
    center = tor.p1
    w = _ssi_unit(tor.p2)
    rmaj = tor.r
    r = tor.r2

    if r >= rmaj - ktol:
        return False

    if not _axes_coaxial(apex, a, center, w, ktol):
        return False

    t = math.tan(alpha)

    if t < 1e-9:
        return False

    za = _axial_coord(center, w, apex)
    _cone_torus_circles(center, w, t, za, r, rmaj, out)
    _cone_torus_circles(center, w, t, za, r, -rmaj, out)

    return True


def _meridian_circles(center, w, rmaj, r, dx, dz, r2, out):
    """Circles where the tube circle (rmaj, 0) of radius r meets the circle of radius r2 at offset (dx, dz) from it."""

    ktol = 1e-6
    d = math.sqrt(dx * dx + dz * dz)
    aa = 0.5 * (r * r - r2 * r2 + d * d) / d
    h = math.sqrt(max(0.0, r * r - aa * aa))
    dirx = dx / d
    dirz = dz / d
    phx = rmaj + aa * dirx
    phz = aa * dirz
    perpx = -dirz
    perpz = dirx
    signs = [0] if h <= ktol else [1, -1]

    for s in signs:
        xi = phx + s * h * perpx
        z = phz + s * h * perpz
        rad = abs(xi)

        if rad < ktol:
            continue

        _axis_circles(center, w, [z], rad, out)


def _ssi_sphere_torus(sph, tor, out):
    """Coaxial sphere-torus section: circles."""

    ktol = 1e-6
    sc = sph.p1
    rsph = sph.r
    center = tor.p1
    w = _ssi_unit(tor.p2)
    rmaj = tor.r
    r = tor.r2

    if r >= rmaj - ktol:
        return False

    if _point_axis_dist(center, w, sc) > ktol:
        return False

    zs = _axial_coord(center, w, sc)
    d = math.sqrt(rmaj * rmaj + zs * zs)

    if d < ktol:
        return True

    if d - ktol > r + rsph or d + ktol < abs(r - rsph):
        return True

    _meridian_circles(center, w, rmaj, r, 0.0 - rmaj, zs - 0.0, rsph, out)

    return True


class _SpiricFrame:
    """Spiric loop frame of two equal parallel-axis tori."""

    def __init__(self, c1, ex, ey, w, rmaj, r, c, be):
        self.c1 = c1  # First torus center.
        self.ex = ex  # Unit direction between the centers.
        self.ey = ey  # Unit axis cross ex.
        self.w = w  # Unit common axis.
        self.rmaj = rmaj  # Major radius.
        self.r = r  # Minor radius.
        self.c = c  # Half the center distance.
        self.be = be  # Semi-axis of the inner loops.


def _spiric_xy(f, inner, t):
    """In-plane point (x, y) of a spiric loop at tube offset t; None when the loop does not reach t."""

    if inner:
        g = t / f.c

        if abs(g) >= 1.0:
            return None

        return f.c + f.rmaj * g, f.be * math.sqrt(1.0 - g * g)

    rho = f.rmaj + t
    y2 = rho * rho - f.c * f.c

    if y2 <= 0.0:
        return None

    return f.c, math.sqrt(y2)


def _emit_spiric_loops(f, inner, out):
    """Both mirrored spiric loops as periodic interpolants, none when either misses a sample."""

    n = 512

    for sgn in (1, -1):
        pts = []

        for k in range(n):
            phi = TWO_PI * k / n
            t = f.r * math.cos(phi)
            z = f.r * math.sin(phi)
            xy = _spiric_xy(f, inner, t)

            if xy is None:
                return

            x = xy[0]
            yy = sgn * xy[1]
            pts.append(
                Point(
                    f.c1[0] + x * f.ex[0] + yy * f.ey[0] + z * f.w[0],
                    f.c1[1] + x * f.ex[1] + yy * f.ey[1] + z * f.w[1],
                    f.c1[2] + x * f.ex[2] + yy * f.ey[2] + z * f.w[2],
                )
            )

        loop = NurbsCurve.create_interpolated(pts, CurveNurbsKnotStyle.ChordPeriodic)

        if loop.is_valid():
            loop.set_domain(0.0, 1.0)
            out.append(loop)


def _ssi_torus_torus_spiric(ta, tb, out):
    """Exact spiric loops of two equal parallel-axis tori."""

    ktol = 1e-6
    c1 = ta.p1
    w = _ssi_unit(ta.p2)
    c2 = tb.p1
    cxw = _ssi_cross(w, _ssi_unit(tb.p2))

    if (
        math.sqrt(_ssi_dot(cxw, cxw)) > ktol
        or abs(ta.r2 - tb.r2) > ktol
        or abs(ta.r - tb.r) > ktol
    ):
        return False

    if abs(_axial_coord(c1, w, c2)) > ktol:
        return False

    dp = [c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]]
    hax = _ssi_dot(dp, w)
    ex = [dp[0] - hax * w[0], dp[1] - hax * w[1], dp[2] - hax * w[2]]
    d = math.sqrt(_ssi_dot(ex, ex))

    if d <= ktol:
        return False

    ex = [ex[0] / d, ex[1] / d, ex[2] / d]
    f = _SpiricFrame(
        c1,
        ex,
        _ssi_cross(w, ex),
        w,
        0.5 * (ta.r + tb.r),
        0.5 * (ta.r2 + tb.r2),
        0.5 * d,
        0.0,
    )

    if abs(f.rmaj - f.c) <= ktol:
        return False

    lo2 = (f.rmaj - f.r) * (f.rmaj - f.r) - f.c * f.c
    hi2 = (f.rmaj + f.r) * (f.rmaj + f.r) - f.c * f.c

    if hi2 > ktol and lo2 <= ktol:
        return False

    if f.rmaj > f.c and f.r >= f.c - ktol:
        return False

    if lo2 > ktol:
        _emit_spiric_loops(f, False, out)

    if f.rmaj > f.c + ktol:
        f.be = math.sqrt(f.rmaj * f.rmaj - f.c * f.c)
        _emit_spiric_loops(f, True, out)

    return True


def _ssi_torus_torus(ta, tb, out):
    """Coaxial torus-torus section: circles."""

    ktol = 1e-6
    c1 = ta.p1
    w = _ssi_unit(ta.p2)
    rmaj1 = ta.r
    r1 = ta.r2
    c2 = tb.p1
    w2 = _ssi_unit(tb.p2)
    rmaj2 = tb.r
    r2 = tb.r2

    if r1 >= rmaj1 - ktol or r2 >= rmaj2 - ktol:
        return False

    if not _axes_coaxial(c1, w, c2, w2, ktol):
        return _ssi_torus_torus_spiric(ta, tb, out)

    z2 = _axial_coord(c1, w, c2)
    dxR = rmaj2 - rmaj1
    d = math.sqrt(dxR * dxR + z2 * z2)

    if d < ktol:
        return False

    if d - ktol > r1 + r2 or d + ktol < abs(r1 - r2):
        return True

    _meridian_circles(c1, w, rmaj1, r1, dxR, z2, r2, out)

    return True


def _ssi_sphere_sphere(ra, rb, out):
    """Exact sphere-sphere circle."""

    c1 = ra.p1
    r1 = ra.r
    c2 = rb.p1
    r2 = rb.r
    dv = [c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]]
    dist = math.sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2])
    tan_tol = (r1 + r2) * 1e-9

    if dist <= 1e-12 or dist >= r1 + r2 - tan_tol or dist <= abs(r1 - r2) + tan_tol:
        return

    nu = [dv[0] / dist, dv[1] / dist, dv[2] / dist]
    aa = (dist * dist + r1 * r1 - r2 * r2) / (2.0 * dist)
    rr2 = r1 * r1 - aa * aa

    if rr2 > 0.0:
        _axis_circles(c1, nu, [aa], math.sqrt(rr2), out)


def _plane_section_curves(plane, srf, rs, out):
    """Exact sections of a plane with a recognized surface; false when the case is not analytic."""

    if rs.kind == _RecogSurface.SPHERE:
        c3 = _ssi_plane_sphere(plane, rs)

        if c3 is not None:
            out.append(c3)

        return True

    if rs.kind == _RecogSurface.CYLINDER:
        if not _ssi_plane_cylinder_lines(plane, rs, srf, out):
            c3 = _ssi_plane_cylinder(plane, rs)

            if c3 is not None:
                out.append(c3)

        return True

    if rs.kind == _RecogSurface.CONE:
        return _ssi_plane_cone(plane, rs, srf, out)

    return _ssi_plane_torus(plane, rs, out)


def _quadric_section_curves(a, ra, b, rb, out):
    """Exact sections of two recognized curved surfaces; false when the case is not analytic."""

    ka = ra.kind
    kb = rb.kind

    if ka == _RecogSurface.SPHERE and kb == _RecogSurface.SPHERE:
        _ssi_sphere_sphere(ra, rb, out)

        return True

    if ka == _RecogSurface.CYLINDER and kb == _RecogSurface.SPHERE:
        return _ssi_cylinder_sphere(ra, rb, out)

    if ka == _RecogSurface.SPHERE and kb == _RecogSurface.CYLINDER:
        return _ssi_cylinder_sphere(rb, ra, out)

    if ka == _RecogSurface.CYLINDER and kb == _RecogSurface.CONE:
        return _ssi_cylinder_cone(ra, rb, out)

    if ka == _RecogSurface.CONE and kb == _RecogSurface.CYLINDER:
        return _ssi_cylinder_cone(rb, ra, out)

    if ka == _RecogSurface.CONE and kb == _RecogSurface.SPHERE:
        return _ssi_cone_sphere(ra, rb, out)

    if ka == _RecogSurface.SPHERE and kb == _RecogSurface.CONE:
        return _ssi_cone_sphere(rb, ra, out)

    if ka == _RecogSurface.CYLINDER and kb == _RecogSurface.CYLINDER:
        return _ssi_cylinder_cylinder(a, ra, b, rb, out)

    if ka == _RecogSurface.CYLINDER and kb == _RecogSurface.TORUS:
        return _ssi_cylinder_torus(ra, rb, out)

    if ka == _RecogSurface.TORUS and kb == _RecogSurface.CYLINDER:
        return _ssi_cylinder_torus(rb, ra, out)

    if ka == _RecogSurface.CONE and kb == _RecogSurface.TORUS:
        return _ssi_cone_torus(ra, rb, out)

    if ka == _RecogSurface.TORUS and kb == _RecogSurface.CONE:
        return _ssi_cone_torus(rb, ra, out)

    if ka == _RecogSurface.SPHERE and kb == _RecogSurface.TORUS:
        return _ssi_sphere_torus(ra, rb, out)

    if ka == _RecogSurface.TORUS and kb == _RecogSurface.SPHERE:
        return _ssi_sphere_torus(rb, ra, out)

    if ka == _RecogSurface.TORUS and kb == _RecogSurface.TORUS:
        return _ssi_torus_torus(ra, rb, out)

    return False


def _analytic_curves(a, ra, b, rb, out):
    """Exact 3D sections of two recognized surfaces; false when the pair is not analytic."""

    if ra.kind == _RecogSurface.PLANE and rb.kind == _RecogSurface.PLANE:
        hit, empty = _ssi_plane_plane(a, ra, b, rb, out)

        if hit:
            return True

        return empty

    if ra.kind == _RecogSurface.PLANE:
        return _plane_section_curves(ra, b, rb, out)

    if rb.kind == _RecogSurface.PLANE:
        return _plane_section_curves(rb, a, ra, out)

    return _quadric_section_curves(a, ra, b, rb, out)


def _analytic_side_pcurve(srf, recog, c3):
    """Pcurve of an exact section on one recognized surface: analytic, pulled back, then projected."""

    pc = _analytic_pcurve(srf, recog, c3)

    if not pc.is_valid():
        v = _analytic_pullback(srf, recog, c3)

        if v:
            pc = v[0]

    if not pc.is_valid():
        v = Closest.surface_curve(srf, c3)

        if v:
            pc = v[0]

    return pc


def _analytic_ssi(a, b, tolerance):
    """Exact section of two recognized analytic surfaces, empty when no case applies."""

    res = _AnalyticResult()
    rtol = max(tolerance, 1e-7) * 1e4
    ra = _recognize_surface(a, rtol)
    rb = _recognize_surface(b, rtol)

    if ra.kind == _RecogSurface.NONE or rb.kind == _RecogSurface.NONE:
        return res

    c3_list = []

    if not _analytic_curves(a, ra, b, rb, c3_list):
        return res

    for cc3 in c3_list:
        pa = _analytic_side_pcurve(a, ra, cc3)
        pb = _analytic_side_pcurve(b, rb, cc3)

        if pa.is_valid() and pb.is_valid():
            res.triples.append((cc3, pa, pb))

    res.status = _AnalyticResult.HIT

    return res


# ═══════════════════════════════════════════════════════════════════════════
# NURBS surfaces
# ═══════════════════════════════════════════════════════════════════════════
def _is_duplicate_curve(
    crv: NurbsCurve, curves: list[NurbsCurve], tolerance: float
) -> bool:
    """Whether crv runs through a curve of curves at a quarter, half and three quarters of its domain within tolerance."""

    ct0, ct1 = crv.domain()

    for existing in curves:
        et0, et1 = existing.domain()
        all_close = True

        for f in [0.25, 0.5, 0.75]:
            cp = crv.point_at(ct0 + (ct1 - ct0) * f)
            ep = existing.point_at(et0 + (et1 - et0) * f)
            em = existing.point_at((et0 + et1) * 0.5)
            d = min(cp.distance(ep), cp.distance(em))

            if d > tolerance:
                all_close = False
                break

        if all_close:
            return True

    return False


def surface_plane(
    surface: "NurbsSurface", plane: Plane, tolerance: float | None = None
) -> list[NurbsCurve]:
    """Surface-plane section curves."""

    if not surface.is_valid():
        return []

    if tolerance is None or tolerance <= 0.0:
        tolerance = Tolerance.ZERO_TOLERANCE

    traced = _surface_plane_traces(surface, plane, tolerance)
    step = traced.step
    uv_to_3d = traced.uv_to_3d
    uv_to_3d_min = traced.uv_to_3d_min

    result = []

    for trace in traced.traces:
        uv_trace = trace.uv_trace
        is_loop = trace.is_loop
        all_pts = []

        for uv in uv_trace:
            all_pts.append(surface.point_at(uv[0], uv[1]))

        crv = _surface_plane_fit_3d(
            all_pts, is_loop, plane, step, uv_to_3d, uv_to_3d_min
        )

        if not crv.is_valid():
            continue

        dup_tol = step * uv_to_3d * 3.0

        if not _is_duplicate_curve(crv, result, dup_tol):
            result.append(crv)

    return result


def surface_plane_uv(
    surface: "NurbsSurface", plane: Plane, tolerance: float | None = None
) -> list[tuple[NurbsCurve, NurbsCurve]]:
    """Surface-plane section curves paired with their UV pcurves."""

    if not surface.is_valid():
        return []

    if tolerance is None or tolerance <= 0.0:
        tolerance = Tolerance.ZERO_TOLERANCE

    field = _SurfacePlaneField(surface, plane, tolerance)
    traced = _surface_plane_traces(surface, plane, tolerance)
    dup_tol = traced.step * traced.uv_to_3d * 3.0

    result = []
    kept_pts3 = []

    for trace in traced.traces:
        trace_pts3 = []

        for q in trace.uv_trace:
            trace_pts3.append(field.point(q))

        if _is_duplicate_trace(trace_pts3, kept_pts3, dup_tol):
            continue

        kept_pts3.append(trace_pts3)

        for piece in _trace_pieces(field, trace):
            if len(piece.uv) < 2:
                continue

            curves = _piece_curves(field, plane, piece)

            if curves is not None:
                result.append(curves)

    return result


def _drop_point_sections(trs, tolerance):
    """Drop near-zero-length section curves."""

    min_len = max(tolerance * 10.0, 1e-9)
    kept = []

    for t in trs:
        if t[0].length() >= min_len:
            kept.append(t)

    return kept


def _surface_mid_plane(srf):
    """Plane through the middle of the surface domain."""

    po, nn = _surface_mid_frame(srf)

    return Plane.from_point_normal(po, Vector(nn[0], nn[1], nn[2]))


def _planar_section_triples(planar, other, planar_first, tolerance):
    """Plane-surface section triples, the plane's pcurve found by projection."""

    result = []

    for section in surface_plane_uv(other, _surface_mid_plane(planar), tolerance):
        c3 = section[0]
        pps = Closest.surface_curve(planar, c3)

        if len(pps) != 1:
            continue

        if planar_first:
            result.append((c3, pps[0], section[1]))
        else:
            result.append((c3, section[1], pps[0]))

    return _drop_point_sections(result, tolerance)


def _cell_box(samples, ci, cj, c0u, dcu, c0v, dcv, tolerance):
    """Bounding box of one grid cell from its 3 x 3 samples, inflated by twice its sag, and the cell center."""

    minx = math.inf
    miny = minx
    minz = minx
    maxx = -minx
    maxy = -minx
    maxz = -minx

    for i in range(2 * ci, 2 * ci + 3):
        for j in range(2 * cj, 2 * cj + 3):
            p = samples[i][j]
            minx = min(minx, p[0])
            maxx = max(maxx, p[0])
            miny = min(miny, p[1])
            maxy = max(maxy, p[1])
            minz = min(minz, p[2])
            maxz = max(maxz, p[2])

    ctr = samples[2 * ci + 1][2 * cj + 1]
    p00 = samples[2 * ci][2 * cj]
    p10 = samples[2 * ci + 2][2 * cj]
    p01 = samples[2 * ci][2 * cj + 2]
    p11 = samples[2 * ci + 2][2 * cj + 2]
    cx = (p00[0] + p10[0] + p01[0] + p11[0]) * 0.25
    cy = (p00[1] + p10[1] + p01[1] + p11[1]) * 0.25
    cz = (p00[2] + p10[2] + p01[2] + p11[2]) * 0.25
    sag = math.sqrt(
        (ctr[0] - cx) * (ctr[0] - cx)
        + (ctr[1] - cy) * (ctr[1] - cy)
        + (ctr[2] - cz) * (ctr[2] - cz)
    )
    inf = 2.0 * sag + tolerance

    return (
        minx - inf,
        miny - inf,
        minz - inf,
        maxx + inf,
        maxy + inf,
        maxz + inf,
        c0u + dcu * (ci + 0.5),
        c0v + dcv * (cj + 0.5),
    )


def _surface_cell_boxes(srf, c0u, dcu, ncu, c0v, dcv, ncv, tolerance):
    """Inflated bounding boxes and centers of an ncu x ncv grid of surface cells."""

    samples = []

    for i in range(2 * ncu + 1):
        row = []

        for j in range(2 * ncv + 1):
            row.append(srf.point_at(c0u + dcu * 0.5 * i, c0v + dcv * 0.5 * j))

        samples.append(row)

    boxes = []

    for ci in range(ncu):
        for cj in range(ncv):
            boxes.append(_cell_box(samples, ci, cj, c0u, dcu, c0v, dcv, tolerance))

    return boxes


def _cell_diagonal(boxes):
    """Smallest non-degenerate diagonal among the first 64 boxes, 1 when none."""

    best = math.inf

    for i in range(min(len(boxes), 64)):
        bx = boxes[i]
        d = math.sqrt(
            (bx[3] - bx[0]) * (bx[3] - bx[0])
            + (bx[4] - bx[1]) * (bx[4] - bx[1])
            + (bx[5] - bx[2]) * (bx[5] - bx[2])
        )

        if 1e-12 < d and d < best:
            best = d

    return best if best < math.inf else 1.0


class _SurfaceSurfaceSeed:
    """Grid seed of a surface-surface trace in joint parameters."""

    def __init__(self, u, v, s, t, used):
        self.u = u  # Seed u on a.
        self.v = v  # Seed v on a.
        self.s = s  # Seed u on b.
        self.t = t  # Seed v on b.
        self.used = used  # Whether a trace already passed the seed.


class _SurfaceSurfaceField:
    """Joint parameter space (au, av, bu, bv) of two surfaces with the marching scales."""

    def __init__(self, a, b, tolerance):
        """Sample both domains into cell boxes and derive the marching scales."""

        self.a = a  # First surface.
        self.b = b  # Second surface.
        self.tolerance = tolerance  # Section tolerance.
        self.lo = [0.0] * 4  # Domain starts.
        self.hi = [0.0] * 4  # Domain ends.
        self.range = [0.0] * 4  # Domain lengths.
        self.closed = [False] * 4  # Whether each parameter wraps around a seam.
        self.step = [0.0] * 4  # Grid cell size per parameter.
        srfs = [a, b]
        cells = [0] * 4

        for k in range(4):
            srf = srfs[k // 2]
            self.lo[k], self.hi[k] = srf.domain(k % 2)
            self.range[k] = self.hi[k] - self.lo[k]
            self.closed[k] = srf.is_closed(k % 2)
            cells[k] = max(len(srf.get_span_vector(k % 2)) - 1, 1) * 4
            self.step[k] = self.range[k] / cells[k]

        lo = self.lo
        step = self.step
        self.boxes_a = _surface_cell_boxes(
            a, lo[0], step[0], cells[0], lo[1], step[1], cells[1], tolerance
        )  # Cell boxes of a.
        self.boxes_b = _surface_cell_boxes(
            b, lo[2], step[2], cells[2], lo[3], step[3], cells[3], tolerance
        )  # Cell boxes of b.
        self.h_init = (
            min(_cell_diagonal(self.boxes_a), _cell_diagonal(self.boxes_b)) * 0.25
        )  # Initial 3D marching step.
        self.conv_tol = max(
            tolerance, self.h_init * 1e-7
        )  # Corrector convergence tolerance.
        self.seed_tol = max(
            _cell_diagonal(self.boxes_a), _cell_diagonal(self.boxes_b)
        )  # 3D distance that merges two seeds.
        self.max_steps = (
            cells[0] * cells[1] + cells[2] * cells[3]
        ) * 32  # Marching step cap per direction.
        self.close_tol = self.h_init * 3.0  # 3D distance that closes a loop.
        self.consume_tol = self.h_init * 2.0  # 3D distance that consumes a seed.

    def wrap(self, k, t):
        """Wrap parameter k across a closed seam or clamp it to the domain."""

        if self.closed[k]:
            f = math.fmod(t - self.lo[k], self.range[k])

            if f < 0:
                f += self.range[k]

            return self.lo[k] + f

        return max(self.lo[k], min(t, self.hi[k]))

    def eval_a(self, u, v):
        """Point and first derivatives (s, su, sv) of a at (u, v)."""

        d = self.a.evaluate(self.wrap(0, u), self.wrap(1, v), 1)

        return d[0], d[2], d[1]

    def eval_b(self, u, v):
        """Point and first derivatives (s, su, sv) of b at (u, v)."""

        d = self.b.evaluate(self.wrap(2, u), self.wrap(3, v), 1)

        return d[0], d[2], d[1]

    def point(self, q):
        """Point of a at the joint parameters q."""

        sa = self.eval_a(q[0], q[1])[0]

        return [sa[0], sa[1], sa[2]]

    def clamp_open(self, x):
        """Clamp the open parameters of x to their domains."""

        for k in range(4):
            if not self.closed[k]:
                x[k] = max(self.lo[k], min(x[k], self.hi[k]))

    def correct(self, x, has_pin, pd, pp):
        """Newton-project x in place onto the section, optionally pinned to the plane through pp normal to pd."""

        for _ in range(8):
            sa, sau, sav = self.eval_a(x[0], x[1])
            sb, sbu, sbv = self.eval_b(x[2], x[3])
            res = [sa[0] - sb[0], sa[1] - sb[1], sa[2] - sb[2]]

            if (
                math.sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2])
                < self.conv_tol
            ):
                return True

            jac = [[0.0] * 4 for _ in range(3)]

            for k in range(3):
                jac[k][0] = sau[k]
                jac[k][1] = sav[k]
                jac[k][2] = -sbu[k]
                jac[k][3] = -sbv[k]

            ok = (
                _newton_step_pinned(jac, res, sa, sau, sav, pd, pp, x)
                if has_pin
                else _newton_step_free(jac, res, x)
            )

            if not ok:
                return False

            self.clamp_open(x)

        sa = self.eval_a(x[0], x[1])[0]
        sb = self.eval_b(x[2], x[3])[0]
        g = math.sqrt(
            (sa[0] - sb[0]) * (sa[0] - sb[0])
            + (sa[1] - sb[1]) * (sa[1] - sb[1])
            + (sa[2] - sb[2]) * (sa[2] - sb[2])
        )

        return g < self.conv_tol * 10.0

    def tangent(self, x, dir_sign):
        """Unit 3D section tangent at x in direction dir_sign, and both surfaces' derivatives: (dir or None, sa, sau, sav, sbu, sbv)."""

        sa, sau, sav = self.eval_a(x[0], x[1])
        sb, sbu, sbv = self.eval_b(x[2], x[3])
        na = [
            sau[1] * sav[2] - sau[2] * sav[1],
            sau[2] * sav[0] - sau[0] * sav[2],
            sau[0] * sav[1] - sau[1] * sav[0],
        ]
        nb = [
            sbu[1] * sbv[2] - sbu[2] * sbv[1],
            sbu[2] * sbv[0] - sbu[0] * sbv[2],
            sbu[0] * sbv[1] - sbu[1] * sbv[0],
        ]
        d = [
            na[1] * nb[2] - na[2] * nb[1],
            na[2] * nb[0] - na[0] * nb[2],
            na[0] * nb[1] - na[1] * nb[0],
        ]
        dl = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        nal = math.sqrt(na[0] * na[0] + na[1] * na[1] + na[2] * na[2])
        nbl = math.sqrt(nb[0] * nb[0] + nb[1] * nb[1] + nb[2] * nb[2])

        if dl < 1e-4 * nal * nbl or dl < 1e-30:
            return None, sa, sau, sav, sbu, sbv

        return (
            [d[0] / dl * dir_sign, d[1] / dl * dir_sign, d[2] / dl * dir_sign],
            sa,
            sau,
            sav,
            sbu,
            sbv,
        )


def _newton_step_free(jac, res, x):
    """Minimum-norm Newton step x -= jac^T (jac jac^T)^-1 res."""

    jjt = [[0.0] * 3 for _ in range(3)]

    for r in range(3):
        for q in range(3):
            s = 0.0

            for c in range(4):
                s += jac[r][c] * jac[q][c]

            jjt[r][q] = s

    y = _solve_gauss(jjt, [res[0], res[1], res[2]], 3)

    if y is None:
        return False

    for c in range(4):
        s = 0.0

        for r in range(3):
            s += jac[r][c] * y[r]

        x[c] -= s

    return True


def _newton_step_pinned(jac, res, sa, sau, sav, pd, pp, x):
    """Newton step with a fourth row pinning a's point to the plane through pp normal to pd."""

    m = [
        [jac[0][0], jac[0][1], jac[0][2], jac[0][3]],
        [jac[1][0], jac[1][1], jac[1][2], jac[1][3]],
        [jac[2][0], jac[2][1], jac[2][2], jac[2][3]],
        [
            pd[0] * sau[0] + pd[1] * sau[1] + pd[2] * sau[2],
            pd[0] * sav[0] + pd[1] * sav[1] + pd[2] * sav[2],
            0.0,
            0.0,
        ],
    ]
    rhs = [
        res[0],
        res[1],
        res[2],
        pd[0] * (sa[0] - pp[0]) + pd[1] * (sa[1] - pp[1]) + pd[2] * (sa[2] - pp[2]),
    ]
    dx = _solve_gauss(m, rhs, 4)

    if dx is None:
        return False

    for c in range(4):
        x[c] -= dx[c]

    return True


def _triple_distance(p, q):
    """Distance between two 3D triples."""
    return math.sqrt(
        (p[0] - q[0]) * (p[0] - q[0])
        + (p[1] - q[1]) * (p[1] - q[1])
        + (p[2] - q[2]) * (p[2] - q[2])
    )


def _seed_is_duplicate(field, x, seeds):
    """Whether a's point at x lies within the seed tolerance of an existing seed."""

    p = field.point(x)

    for sd in seeds:
        if _triple_distance(p, field.point([sd.u, sd.v, 0.0, 0.0])) < field.seed_tol:
            return True

    return False


def _surface_surface_seeds(field):
    """Corrected centers of overlapping cell-box pairs, one per distinct 3D point, at most 20000 pairs."""

    seeds = []
    pair_budget = 20000
    dummy3 = [0.0, 0.0, 0.0]

    for ba in field.boxes_a:
        if pair_budget < 0:
            break

        for bb in field.boxes_b:
            if (
                bb[0] > ba[3]
                or bb[3] < ba[0]
                or bb[1] > ba[4]
                or bb[4] < ba[1]
                or bb[2] > ba[5]
                or bb[5] < ba[2]
            ):
                continue

            pair_budget -= 1

            if pair_budget < 0:
                break

            x = [ba[6], ba[7], bb[6], bb[7]]

            if not field.correct(x, False, dummy3, dummy3) or _seed_is_duplicate(
                field, x, seeds
            ):
                continue

            seeds.append(
                _SurfaceSurfaceSeed(
                    field.wrap(0, x[0]),
                    field.wrap(1, x[1]),
                    field.wrap(2, x[2]),
                    field.wrap(3, x[3]),
                    False,
                )
            )

    return seeds


class _SurfaceSurfaceMarch:
    """Marching state of one trace direction."""

    def __init__(self):
        self.x = [0.0, 0.0, 0.0, 0.0]  # Current joint parameters.
        self.d = [0.0, 0.0, 0.0]  # Current 3D direction.
        self.sa = None  # Point of a at x.
        self.sau = None  # u-derivative of a at x.
        self.sav = None  # v-derivative of a at x.
        self.sbu = None  # u-derivative of b at x.
        self.sbv = None  # v-derivative of b at x.
        self.have_prev_d = False  # Whether a previous step exists.
        self.prev_d = [0.0, 0.0, 0.0]  # Previous 3D direction.
        self.p_prev = [0.0, 0.0, 0.0]  # Previous 3D point.
        self.h = 0.0  # Current 3D step.
        self.smooth = 0  # Accepted steps since the last change of h.
        self.tang_reuse = 0  # Steps that reused the previous direction.
        self.why = "maxsteps"  # Reason the march stopped.
        self.xn = [0.0, 0.0, 0.0, 0.0]  # Accepted next parameters.
        self.p_cur = [0.0, 0.0, 0.0]  # Accepted next 3D point.
        self.step_len = 0.0  # Accepted 3D step length.
        self.hit_boundary = False  # Whether the accepted step reached an open boundary.


def _march_direction(field, m, dir_sign):
    """Direction of the next step, reusing the previous one up to three times at tangencies."""

    d, m.sa, m.sau, m.sav, m.sbu, m.sbv = field.tangent(m.x, dir_sign)

    if d is not None:
        m.d = d
        m.tang_reuse = 0

        return True

    if not m.have_prev_d or m.tang_reuse >= 3:
        m.why = "tangency"

        return False

    m.d = m.prev_d
    m.tang_reuse += 1

    return True


def _march_predict(field, m):
    """Parameters after a step h along d, cut back at open boundaries, and the predicted point; None when a surface is singular."""

    sau = m.sau
    sav = m.sav
    sbu = m.sbu
    sbv = m.sbv
    d = m.d
    ma = [
        [
            sau[0] * sau[0] + sau[1] * sau[1] + sau[2] * sau[2],
            sau[0] * sav[0] + sau[1] * sav[1] + sau[2] * sav[2],
        ],
        [
            sau[0] * sav[0] + sau[1] * sav[1] + sau[2] * sav[2],
            sav[0] * sav[0] + sav[1] * sav[1] + sav[2] * sav[2],
        ],
    ]
    ra = [
        m.h * (d[0] * sau[0] + d[1] * sau[1] + d[2] * sau[2]),
        m.h * (d[0] * sav[0] + d[1] * sav[1] + d[2] * sav[2]),
    ]
    mb = [
        [
            sbu[0] * sbu[0] + sbu[1] * sbu[1] + sbu[2] * sbu[2],
            sbu[0] * sbv[0] + sbu[1] * sbv[1] + sbu[2] * sbv[2],
        ],
        [
            sbu[0] * sbv[0] + sbu[1] * sbv[1] + sbu[2] * sbv[2],
            sbv[0] * sbv[0] + sbv[1] * sbv[1] + sbv[2] * sbv[2],
        ],
    ]
    rb = [
        m.h * (d[0] * sbu[0] + d[1] * sbu[1] + d[2] * sbu[2]),
        m.h * (d[0] * sbv[0] + d[1] * sbv[1] + d[2] * sbv[2]),
    ]
    duv_a = _solve_gauss(ma, ra, 2)
    duv_b = _solve_gauss(mb, rb, 2) if duv_a is not None else None

    if duv_a is None or duv_b is None:
        return None

    delta = [duv_a[0], duv_a[1], duv_b[0], duv_b[1]]
    tc = 1.0
    m.hit_boundary = False

    for k in range(4):
        if field.closed[k] or abs(delta[k]) < 1e-15:
            continue

        if m.x[k] + delta[k] > field.hi[k]:
            tc = min(tc, (field.hi[k] - m.x[k]) / delta[k])
            m.hit_boundary = True

        if m.x[k] + delta[k] < field.lo[k]:
            tc = min(tc, (field.lo[k] - m.x[k]) / delta[k])
            m.hit_boundary = True

    m.xn = [
        m.x[0] + tc * delta[0],
        m.x[1] + tc * delta[1],
        m.x[2] + tc * delta[2],
        m.x[3] + tc * delta[3],
    ]

    return [
        m.sa[0] + d[0] * m.h * tc,
        m.sa[1] + d[1] * m.h * tc,
        m.sa[2] + d[2] * m.h * tc,
    ]


def _march_turns_sharply(m):
    """Whether the step from p_prev to p_cur turns more than acos(0.985) from the previous direction."""

    sd0 = (m.p_cur[0] - m.p_prev[0]) / m.step_len
    sd1 = (m.p_cur[1] - m.p_prev[1]) / m.step_len
    sd2 = (m.p_cur[2] - m.p_prev[2]) / m.step_len

    return sd0 * m.prev_d[0] + sd1 * m.prev_d[1] + sd2 * m.prev_d[2] < 0.985


def _march_step(field, m):
    """Up to seven attempts at one step, halving h after a failed corrector or a sharp turn."""

    attempts = 0

    while attempts < 7:
        p_pred = _march_predict(field, m)

        if p_pred is None:
            m.why = "singular"

            return False

        if not field.correct(m.xn, True, m.d, p_pred):
            m.why = "corrector"
            m.h *= 0.5
            attempts += 1
            m.smooth = 0
            continue

        m.p_cur = field.point(m.xn)
        m.step_len = _triple_distance(m.p_cur, m.p_prev)

        if (
            m.have_prev_d
            and m.step_len > 1e-14
            and _march_turns_sharply(m)
            and attempts < 6
            and not m.hit_boundary
        ):
            m.why = "angle"
            m.h *= 0.5
            attempts += 1
            m.smooth = 0
            continue

        return True

    return False


def _consume_seeds_near(field, p, seeds):
    """Mark the unused seeds within the consume tolerance of p as used."""

    for sd in seeds:
        if (
            not sd.used
            and _triple_distance(p, field.point([sd.u, sd.v, 0.0, 0.0]))
            < field.consume_tol
        ):
            sd.used = True


def _start_march(field, x0, p_start):
    """Marching state at x0 with the initial step and no previous direction."""

    m = _SurfaceSurfaceMarch()
    m.x = list(x0)
    m.have_prev_d = False
    m.prev_d = [0.0, 0.0, 0.0]
    m.p_prev = p_start
    m.h = field.h_init
    m.smooth = 0
    m.tang_reuse = 0
    m.why = "maxsteps"
    m.xn = [0.0, 0.0, 0.0, 0.0]
    m.p_cur = [0.0, 0.0, 0.0]
    m.step_len = 0.0
    m.hit_boundary = False

    return m


def _trace_dir(field, x0, dir_sign, seeds):
    """March from x0 in direction dir_sign until it closes, leaves the domain, stalls or hits the step cap: (closed, samples, why)."""

    out = []
    p_start = field.point(x0)
    m = _start_march(field, x0, p_start)
    dist_traveled = 0.0

    for _ in range(field.max_steps):
        if not _march_direction(field, m, dir_sign) or not _march_step(field, m):
            break

        m.why = "maxsteps"
        m.prev_d = m.d
        m.have_prev_d = True
        m.smooth += 1

        if m.smooth >= 5 and m.h < field.h_init * 2.0:
            m.h *= 1.4
            m.smooth = 0

        m.x = list(m.xn)
        dist_traveled += m.step_len
        out.append(list(m.x))

        if (
            dist_traveled > field.close_tol * 3.0
            and _triple_distance(m.p_cur, p_start) < field.close_tol
        ):
            return True, out, "closed"

        m.p_prev = m.p_cur

        if m.hit_boundary:
            m.why = "boundary"
            break

        _consume_seeds_near(field, m.p_cur, seeds)

    return False, out, m.why


def _unwrap_quad(field, quad):
    """Shift closed parameters by whole periods so consecutive samples never jump more than half a period."""

    for i in range(1, len(quad)):
        for k in range(4):
            if not field.closed[k]:
                continue

            jump = quad[i][k] - quad[i - 1][k]

            if jump > field.range[k] * 0.5:
                quad[i][k] -= field.range[k]
            elif jump < -field.range[k] * 0.5:
                quad[i][k] += field.range[k]


def _trace_seed(field, x0, seeds):
    """Trace both directions from a seed into one unwrapped run: (quad, is_loop), None when it is too short."""

    fwd_closed, fwd, fwd_why = _trace_dir(field, x0, 1.0, seeds)
    bwd = []
    bwd_why = "?"

    if not fwd_closed:
        _, bwd, bwd_why = _trace_dir(field, x0, -1.0, seeds)

    quad = []

    for i in range(len(bwd) - 1, -1, -1):
        quad.append(bwd[i])

    quad.append(list(x0))

    for p in fwd:
        quad.append(p)

    min_pts = (
        2 if (not fwd_closed and fwd_why == "boundary" and bwd_why == "boundary") else 4
    )

    if len(quad) < min_pts:
        return None

    _unwrap_quad(field, quad)
    gap = _triple_distance(field.point(quad[0]), field.point(quad[-1]))
    is_loop = fwd_closed or (len(quad) >= 6 and gap < field.close_tol)

    if is_loop:
        quad.pop()

    if len(quad) < min_pts:
        return None

    return quad, is_loop


def _is_duplicate_quad(trace_pts3, kept_pts3, dup_tol):
    """Whether the quarter, half and three-quarter samples all lie within dup_tol of one kept run."""

    m = len(trace_pts3)

    for other in kept_pts3:
        all_close = True

        for f in (0.25, 0.5, 0.75):
            cp = trace_pts3[int((m - 1) * f)]
            dmin = dup_tol + 1.0

            for op in other:
                dmin = min(dmin, _triple_distance(cp, op))

            if dmin > dup_tol:
                all_close = False
                break

        if all_close:
            return True

    return False


def _densify_quad(field, quad):
    """Insert corrected midpoints into gaps longer than 1.5 median gaps, at most four passes."""

    dummy3 = [0.0, 0.0, 0.0]

    for _ in range(4):
        gg = []

        for i in range(len(quad) - 1):
            gg.append(_triple_distance(field.point(quad[i]), field.point(quad[i + 1])))

        if not gg:
            break

        gg.sort()
        med = gg[len(gg) // 2]

        if med <= 0:
            break

        changed = False
        i = 0

        while i + 1 < len(quad) and len(quad) < 4000:
            if (
                _triple_distance(field.point(quad[i]), field.point(quad[i + 1]))
                > 1.5 * med
            ):
                midq = [(quad[i][k] + quad[i + 1][k]) * 0.5 for k in range(4)]

                if field.correct(midq, False, dummy3, dummy3):
                    quad.insert(i + 1, midq)
                    changed = True
                    i += 2
                    continue

            i += 1

        if not changed:
            break


def _close_quad(field, quad, is_loop):
    """Append the loop start shifted by whole periods after the end; returns the shift."""

    closure = [0.0, 0.0, 0.0, 0.0]

    if not is_loop or len(quad) < 2:
        return closure

    virt = list(quad[0])

    for k in range(4):
        jump = quad[0][k] - quad[-1][k]

        if field.closed[k]:
            jump = _unwrap_period(jump, 0.0, field.range[k])

        virt[k] = quad[-1][k] + jump
        closure[k] = virt[k] - quad[0][k]

    quad.append(virt)

    return closure


def _quad_seam_crossings(field, pa, pb):
    """Seam crossings (t, parameter, seam value) of the step pa -> pb, sorted by t."""

    crossings = []

    for k in range(4):
        if not field.closed[k] or abs(pb[k] - pa[k]) <= 1e-15:
            continue

        k0 = math.floor((pa[k] - field.lo[k]) / field.range[k])
        k1 = math.floor((pb[k] - field.lo[k]) / field.range[k])

        for j in range(min(k0, k1) + 1, max(k0, k1) + 1):
            seam = field.lo[k] + j * field.range[k]
            t = (seam - pa[k]) / (pb[k] - pa[k])

            if 0.0 < t and t < 1.0:
                crossings.append((t, k, seam))

    crossings.sort()

    return crossings


def _snap_quad_to_seam(field, prev, p):
    """Snap closed parameters of p that sit on a seam onto it; false when none moved."""

    on_seam = False

    for k in range(4):
        if not field.closed[k]:
            continue

        j = _round_half_away((p[k] - field.lo[k]) / field.range[k])
        seam = field.lo[k] + j * field.range[k]

        if (
            abs(p[k] - seam) < field.range[k] * 1e-9
            and abs(p[k] - prev[k]) > field.range[k] * 1e-9
        ):
            p[k] = seam
            on_seam = True

    return on_seam


def _split_quad_at_seams(field, quad):
    """Insert corrected seam crossings into the run: (samples, indices of every seam sample)."""

    dummy3 = [0.0, 0.0, 0.0]
    out_pts = [list(quad[0])]
    cross_idx = []

    for i in range(1, len(quad)):
        pa = quad[i - 1]
        pb = quad[i]

        for t, idx, seam in _quad_seam_crossings(field, pa, pb):
            cp = [pa[k] + (pb[k] - pa[k]) * t for k in range(4)]
            cp[idx] = seam
            field.correct(cp, False, dummy3, dummy3)
            out_pts.append(cp)
            cross_idx.append(len(out_pts) - 1)

        out_pts.append(list(pb))

        if i < len(quad) - 1 and _snap_quad_to_seam(field, pa, out_pts[-1]):
            cross_idx.append(len(out_pts) - 1)

    return out_pts, cross_idx


def _quad_pieces(field, out_pts, cross_idx, is_loop, closure):
    """Seam-free pieces of a split run, the last loop piece wrapped past the start by the closure shift."""

    pieces = []
    wrap_drift = False

    for k in range(4):
        if abs(closure[k]) > field.range[k] * 0.5:
            wrap_drift = True

    if not cross_idx:
        pieces.append((out_pts, is_loop and not wrap_drift))

        return pieces

    if not is_loop:
        bounds = [0] + cross_idx + [len(out_pts) - 1]

        for bi in range(len(bounds) - 1):
            if bounds[bi + 1] > bounds[bi]:
                pieces.append(
                    ([list(p) for p in out_pts[bounds[bi] : bounds[bi + 1] + 1]], False)
                )

        return pieces

    for ci in range(len(cross_idx) - 1):
        pieces.append(
            ([list(p) for p in out_pts[cross_idx[ci] : cross_idx[ci + 1] + 1]], False)
        )

    wrap_piece = [list(p) for p in out_pts[cross_idx[-1] :]]

    for pi in range(1, cross_idx[0] + 1):
        wrap_piece.append([out_pts[pi][k] + closure[k] for k in range(4)])

    pieces.append((wrap_piece, False))

    return pieces


def _shift_quad_piece(field, piece_pts):
    """Shift closed parameters of a piece by whole periods so its middle sample lies in the domain."""

    mid = list(piece_pts[len(piece_pts) // 2])

    for k in range(4):
        if not field.closed[k]:
            continue

        k_s = math.floor((mid[k] - field.lo[k]) / field.range[k])

        if k_s != 0:
            for p in piece_pts:
                p[k] -= k_s * field.range[k]


def _chord_deviation(pa, pm, pb):
    """Distance of pm from the line through pa and pb, 0 for a degenerate chord."""

    ex = pb[0] - pa[0]
    ey = pb[1] - pa[1]
    ez = pb[2] - pa[2]
    l2 = ex * ex + ey * ey + ez * ez

    if l2 <= 1e-30:
        return 0.0

    tt = ((pm[0] - pa[0]) * ex + (pm[1] - pa[1]) * ey + (pm[2] - pa[2]) * ez) / l2
    c = [pa[0] + tt * ex, pa[1] + tt * ey, pa[2] + tt * ez]

    return _triple_distance(pm, c)


def _refine_quad_piece(field, piece_pts):
    """Insert corrected midpoints that deviate from their chord, at most eight passes and 3000 samples."""

    dummy3 = [0.0, 0.0, 0.0]
    refine_tol = max(field.tolerance * 100.0, 5e-6)

    for _ in range(8):
        refined = False
        new_pp = [piece_pts[0]]
        n = len(piece_pts) - 1 if len(piece_pts) < 3000 else 0

        for i in range(n):
            midq = [(piece_pts[i][k] + piece_pts[i + 1][k]) * 0.5 for k in range(4)]

            if field.correct(midq, False, dummy3, dummy3):
                dev = _chord_deviation(
                    field.point(piece_pts[i]),
                    field.point(midq),
                    field.point(piece_pts[i + 1]),
                )

                if dev > refine_tol:
                    new_pp.append(midq)
                    refined = True

            new_pp.append(piece_pts[i + 1])

        piece_pts[:] = new_pp

        if not refined:
            break


def _total_turning_3d(pts):
    """Sum of the turning angles along a 3D polyline."""

    turning = 0.0

    for i in range(1, len(pts) - 1):
        dx1 = pts[i][0] - pts[i - 1][0]
        dy1 = pts[i][1] - pts[i - 1][1]
        dz1 = pts[i][2] - pts[i - 1][2]
        dx2 = pts[i + 1][0] - pts[i][0]
        dy2 = pts[i + 1][1] - pts[i][1]
        dz2 = pts[i + 1][2] - pts[i][2]
        l1 = math.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1)
        l2 = math.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2)

        if l1 > 1e-14 and l2 > 1e-14:
            c = (dx1 * dx2 + dy1 * dy2 + dz1 * dz2) / (l1 * l2)
            c = max(-1.0, min(1.0, c))
            turning += math.acos(c)

    return turning


def _fit_track(pts, fit_tol, is_loop):
    """Cubic fitted to a traced run, CVs doubled until within fit_tol, interpolated when fitting fails."""

    mp = len(pts)
    chords = _chord_parameters(pts, is_loop)
    target_cvs = max(8, int(_total_turning_3d(pts) / 0.5) + 6)
    max_cvs = max(8, min(mp - 1, mp // 3))
    best = NurbsCurve()
    best_dev = math.inf

    while target_cvs <= max_cvs:
        crv = NurbsCurve.create_fitted(pts, target_cvs, 3, is_loop)

        if not crv.is_valid():
            break

        dev = _fitted_max_deviation(crv, pts, chords, 24)

        if dev < best_dev:
            best = crv
            best_dev = dev

        if dev < fit_tol:
            break

        target_cvs *= 2

    if best_dev >= fit_tol:
        if is_loop:
            interp = NurbsCurve.create_interpolated(
                pts, CurveNurbsKnotStyle.ChordPeriodic
            )
        else:
            interp = NurbsCurve.create_interpolated(pts)

        if interp.is_valid():
            best = interp

    if best.is_valid():
        best.set_domain(0.0, 1.0)

    return best


def _piece_triple(field, piece_pts, piece_loop):
    """Section triple of one seam-free piece: refined, then its 3D curve and both pcurves fitted; None when degenerate."""

    _shift_quad_piece(field, piece_pts)
    chord3 = 0.0

    for i in range(1, len(piece_pts)):
        chord3 += _triple_distance(
            field.point(piece_pts[i]), field.point(piece_pts[i - 1])
        )

    if chord3 < field.h_init * 0.05:
        return None

    _refine_quad_piece(field, piece_pts)
    pts3 = []
    pts_pa = []
    pts_pb = []

    for q in piece_pts:
        p = field.point(q)
        pts3.append(Point(p[0], p[1], p[2]))
        pts_pa.append(Point(q[0], q[1], 0.0))
        pts_pb.append(Point(q[2], q[3], 0.0))

    crv3 = _fit_track(pts3, max(field.tolerance * 10.0, 1e-7), piece_loop)
    pcurve_a = _fit_track(pts_pa, min(field.step[0], field.step[1]) * 1e-4, piece_loop)
    pcurve_b = _fit_track(pts_pb, min(field.step[2], field.step[3]) * 1e-4, piece_loop)

    if not crv3.is_valid() or not pcurve_a.is_valid() or not pcurve_b.is_valid():
        return None

    return crv3, pcurve_a, pcurve_b


def _marched_section_triples(a, b, tolerance):
    """Section triples of two freeform surfaces by seeding, marching and fitting every trace."""

    field = _SurfaceSurfaceField(a, b, tolerance)
    dummy3 = [0.0, 0.0, 0.0]
    seeds = _surface_surface_seeds(field)
    result = []
    kept_pts3 = []

    for seed in seeds:
        if seed.used:
            continue

        seed.used = True
        x0 = [seed.u, seed.v, seed.s, seed.t]

        if not field.correct(x0, False, dummy3, dummy3):
            continue

        traced = _trace_seed(field, x0, seeds)

        if traced is None:
            continue

        quad, is_loop = traced
        trace_pts3 = []

        for q in quad:
            trace_pts3.append(field.point(q))

        if _is_duplicate_quad(trace_pts3, kept_pts3, field.h_init * 2.0):
            continue

        kept_pts3.append(trace_pts3)
        _densify_quad(field, quad)
        closure = _close_quad(field, quad, is_loop)
        out_pts, cross_idx = _split_quad_at_seams(field, quad)

        for piece_pts, piece_loop in _quad_pieces(
            field, out_pts, cross_idx, is_loop, closure
        ):
            triple = (
                _piece_triple(field, piece_pts, piece_loop)
                if len(piece_pts) >= 2
                else None
            )

            if triple is not None:
                result.append(triple)

    return _drop_point_sections(result, tolerance)


def surface_surface(
    a: "NurbsSurface", b: "NurbsSurface", tolerance: float | None = None
) -> list[tuple[NurbsCurve, NurbsCurve, NurbsCurve]]:
    """Surface-surface section curves with their UV pcurves on both surfaces."""

    if not a.is_valid() or not b.is_valid():
        return []

    if tolerance is None or tolerance <= 0.0:
        tolerance = Tolerance.ZERO_TOLERANCE

    analytic = _analytic_ssi(a, b, tolerance)

    if analytic.status != _AnalyticResult.NOT_ANALYTIC:
        return _drop_point_sections(analytic.triples, tolerance)

    if a.is_planar(None, 1e-9):
        return _planar_section_triples(a, b, True, tolerance)

    if b.is_planar(None, 1e-9):
        return _planar_section_triples(b, a, False, tolerance)

    return _marched_section_triples(a, b, tolerance)


class _CutterGap:
    """Distance from a pcurve's lifted point to the cutter: clamped in the corner frame of a rectangle, else to the boundary polygon."""

    def __init__(self, target, pc, cutter):
        """Corner frame and boundary polygon of the cutter."""

        cu0, cu1 = cutter.domain(0)
        cv0, cv1 = cutter.domain(1)
        self.target = target  # Surface the pcurve lives on.
        self.pc = pc  # Pcurve on the target.
        self.cutter = cutter  # Cutting surface.
        self.q00 = cutter.point_at(cu0, cv0)  # Cutter corner at (u0, v0).
        q10 = cutter.point_at(cu1, cv0)
        q01 = cutter.point_at(cu0, cv1)
        q11 = cutter.point_at(cu1, cv1)
        q00 = self.q00
        self.eu = Vector(
            q10[0] - q00[0], q10[1] - q00[1], q10[2] - q00[2]
        )  # Cutter edge to (u1, v0).
        self.ev = Vector(
            q01[0] - q00[0], q01[1] - q00[1], q01[2] - q00[2]
        )  # Cutter edge to (u0, v1).
        eu = self.eu
        ev = self.ev
        self.eu2 = (
            eu[0] * eu[0] + eu[1] * eu[1] + eu[2] * eu[2]
        )  # Squared length of eu.
        self.ev2 = (
            ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2]
        )  # Squared length of ev.
        self.frame = Plane()  # Plane of the boundary polygon.
        self.outline = Polyline()  # Boundary polygon in the frame, empty without area.
        square = abs(eu.dot(ev)) <= 1e-9 * math.sqrt(self.eu2 * self.ev2)
        parallelogram = q11.distance(q00 + eu + ev) <= 1e-9 * math.sqrt(
            self.eu2 + self.ev2
        )
        bilinear = cutter.cv_count(0) == 2 and cutter.cv_count(1) == 2
        self.rectangle = (
            self.eu2 > 1e-28
            and self.ev2 > 1e-28
            and bilinear
            and square
            and parallelogram
        )  # Whether the cutter is a 2 x 2 rectangle.

        if not self.rectangle:
            self.outline, self.frame = _boundary_outline(cutter)

    def gap(self, t):
        """Distance to the cutter at pcurve parameter t."""

        uv = self.pc.point_at(t)
        p3 = self.target.point_at(uv[0], uv[1])

        if self.rectangle:
            return self.rectangle_gap(p3)

        if self.outline.point_count() == 0:
            return Closest.surface_point(self.cutter, p3, 0.0, 0.0, 0.0, 0.0)[2]

        return self.outline_gap(p3)

    def rectangle_gap(self, p3):
        """Distance from p3 to the rectangle spanned by eu and ev."""

        q00 = self.q00
        eu = self.eu
        ev = self.ev
        dx = p3[0] - q00[0]
        dy = p3[1] - q00[1]
        dz = p3[2] - q00[2]
        a = (dx * eu[0] + dy * eu[1] + dz * eu[2]) / self.eu2
        b = (dx * ev[0] + dy * ev[1] + dz * ev[2]) / self.ev2
        a = min(max(a, 0.0), 1.0)
        b = min(max(b, 0.0), 1.0)
        cx = q00[0] + a * eu[0] + b * ev[0]
        cy = q00[1] + a * eu[1] + b * ev[1]
        cz = q00[2] + a * eu[2] + b * ev[2]

        return math.sqrt(
            (p3[0] - cx) * (p3[0] - cx)
            + (p3[1] - cy) * (p3[1] - cy)
            + (p3[2] - cz) * (p3[2] - cz)
        )

    def outline_gap(self, p3):
        """Distance from p3 to the region inside the boundary polygon."""

        d = p3 - self.frame.origin
        p = Point(
            d.dot(self.frame.x_axis),
            d.dot(self.frame.y_axis),
            d.dot(self.frame.z_axis),
        )

        if self.outline.point_in_polygon_2d(p):
            return abs(p[2])

        n = self.outline.point_count()
        d2 = sys.float_info.max

        for i in range(n):
            a = self.outline[i]
            b = self.outline[(i + 1) % n]
            ex = b[0] - a[0]
            ey = b[1] - a[1]
            len2 = ex * ex + ey * ey
            s = 0.0

            if len2 > 0.0:
                s = min(max(((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / len2, 0.0), 1.0)

            dx = p[0] - a[0] - s * ex
            dy = p[1] - a[1] - s * ey
            d2 = min(d2, dx * dx + dy * dy)

        return math.sqrt(d2 + p[2] * p[2])


def _refine_footprint_edge(g, t_in, t_out, edge_tol):
    """Footprint edge between an inside and an outside parameter, by 24 bisections."""

    a = t_in
    b = t_out

    for _ in range(24):
        tm = (a + b) * 0.5

        if g.gap(tm) < edge_tol:
            a = tm
        else:
            b = tm

    return b


def _footprint_spans(g, n, on_tol, edge_tol):
    """Parameter spans of the pcurve inside the footprint from n + 1 samples, ends bisected to the edge."""

    d0, d1 = g.pc.domain()
    flags = []

    for i in range(n + 1):
        t = d0 + (d1 - d0) * i / n
        flags.append((t, g.gap(t) < on_tol))

    spans = []
    i = 0

    while i <= n:
        if not flags[i][1]:
            i += 1
            continue

        j = i

        while j + 1 <= n and flags[j + 1][1]:
            j += 1

        ta = (
            flags[i][0]
            if i == 0
            else _refine_footprint_edge(g, flags[i][0], flags[i - 1][0], edge_tol)
        )
        tb = (
            flags[j][0]
            if j == n
            else _refine_footprint_edge(g, flags[j][0], flags[j + 1][0], edge_tol)
        )

        if tb - ta > (d1 - d0) * 1e-6:
            spans.append((ta, tb))

        i = j + 1

    return spans


def _join_wrapped_spans(pc, n, spans, pieces):
    """Join the first and last spans of a closed pcurve across its start into one polyline piece."""

    d0, d1 = pc.domain()
    pc_closed = pc.point_at(d0).distance(pc.point_at(d1)) < 1e-9
    wraps = (
        pc_closed
        and len(spans) >= 2
        and spans[0][0] <= d0 + (d1 - d0) * 1e-9
        and spans[-1][1] >= d1 - (d1 - d0) * 1e-9
    )

    if not wraps:
        return

    ta = spans[-1][0]
    tb = spans[0][1]
    spans.pop()
    spans.pop(0)
    m2 = max(32, n // 2)
    pts = []
    len1 = d1 - ta
    len2 = tb - d0
    tot = len1 + len2

    for k2 in range(m2 + 1):
        f = tot * k2 / m2
        t = ta + f if f < len1 else d0 + (f - len1)
        pts.append(pc.point_at(min(t, d1)))

    joined = NurbsCurve.create(False, 1, pts)

    if joined.is_valid():
        pieces.append(joined)


def _clip_pcurve_to_cutter(target, pc, cutter):
    """Keep the pcurve sub-segments whose lifted 3D point lies inside the cutter footprint."""

    g = _CutterGap(target, pc, cutter)
    cu0, cu1 = cutter.domain(0)
    cv0, cv1 = cutter.domain(1)
    corner_diag = g.q00.distance(cutter.point_at(cu1, cv1))
    n = max(pc.cv_count() * 4, 16)
    spans = _footprint_spans(
        g, n, max(1e-6, corner_diag * 2e-3), max(1e-6, corner_diag * 2e-4)
    )
    pieces = []
    _join_wrapped_spans(pc, n, spans, pieces)

    for sp in spans:
        piece = pc.duplicate()

        if piece.trim(sp[0], sp[1]) and piece.is_valid():
            pieces.append(piece)

    return pieces


def _target_pcurves(target, rt, tr, tolerance):
    """Pcurves of one section on the target: analytic, pulled back, projected, then the traced pcurve."""

    c3d = tr[0]
    pa_tr = tr[1]
    pa_an = _analytic_pcurve(target, rt, c3d)

    if pa_an.is_valid():
        return [pa_an]

    if rt.kind == _RecogSurface.NONE or rt.kind == _RecogSurface.PLANE:
        if pa_tr.is_valid():
            return [pa_tr]

        return Closest.surface_curve(target, c3d, 0.0, 0.0, tolerance)

    pcs = _analytic_pullback(target, rt, c3d)

    if not pcs:
        pcs = Closest.surface_curve(target, c3d, 0.0, 0.0, tolerance)

    if not pcs:
        pcs = [pa_tr]

    return pcs


def cut_curves_on_surface(
    target: "NurbsSurface", cutter: "NurbsSurface", tolerance: float | None = None
) -> list[NurbsCurve]:
    """UV pcurves of the cutter's section on the target, clipped to the cutter footprint."""

    if tolerance is None:
        tolerance = Tolerance.ZERO_TOLERANCE

    out = []
    cutter_planar = cutter.is_planar(None, 1e-6)
    rt = _recognize_surface(target, max(tolerance, 1e-7) * 1e4)

    for tr in surface_surface(target, cutter, tolerance):
        for pc in _target_pcurves(target, rt, tr, tolerance):
            if not cutter_planar:
                out.append(pc)
                continue

            out.extend(_clip_pcurve_to_cutter(target, pc, cutter))

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Polylines and plane sets
# ═══════════════════════════════════════════════════════════════════════════
def _vectors_nearly_parallel(v0: Vector, v1: Vector, angle_tol: float) -> bool:
    """Whether two vectors are parallel within angle_tol."""

    m0 = math.sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2])
    m1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])

    if m0 < 1e-10 or m1 < 1e-10:
        return True

    cos_angle = abs((v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]) / (m0 * m1))

    return cos_angle > math.cos(angle_tol)


def plane_plane_plane_check(
    p0: Plane, p1: Plane, p2: Plane, angle_tol: float
) -> Point | None:
    """Three-plane intersection that rejects near-parallel pairs."""

    if _vectors_nearly_parallel(p0.z_axis, p1.z_axis, angle_tol):
        return None

    if _vectors_nearly_parallel(p0.z_axis, p2.z_axis, angle_tol):
        return None

    if _vectors_nearly_parallel(p1.z_axis, p2.z_axis, angle_tol):
        return None

    return plane_plane_plane(p0, p1, p2)


def remap(val: float, from1: float, to1: float, from2: float, to2: float) -> float:
    """Linear remap of val from [from1, to1] to [from2, to2]."""

    span = to1 - from1

    if abs(span) < 1e-14:
        return from2

    t = (val - from1) / span

    return from2 + t * (to2 - from2)


def closest_point_on_segment(pt: Point, seg: Line) -> tuple:
    """Closest point on a finite segment and its parameter in [0, 1]."""

    start = seg.start()
    end = seg.end()
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    len_sq = dx * dx + dy * dy + dz * dz

    if len_sq < 1e-20:
        return (Point(start[0], start[1], start[2]), 0.0)

    vx = pt[0] - start[0]
    vy = pt[1] - start[1]
    vz = pt[2] - start[2]
    t = _clamp_unit((vx * dx + vy * dy + vz * dz) / len_sq)

    return (Point(start[0] + t * dx, start[1] + t * dy, start[2] + t * dz), t)


def plane_4planes(main_plane: Plane, planes: list[Plane]) -> object | None:
    """Closed quad of the main plane cut by four ordered boundary planes."""

    p0 = plane_plane_plane_check(planes[0], planes[1], main_plane, 0.1)

    if p0 is None:
        return None

    p1 = plane_plane_plane_check(planes[1], planes[2], main_plane, 0.1)

    if p1 is None:
        return None

    p2 = plane_plane_plane_check(planes[2], planes[3], main_plane, 0.1)

    if p2 is None:
        return None

    p3 = plane_plane_plane_check(planes[3], planes[0], main_plane, 0.1)

    if p3 is None:
        return None

    return Polyline([p0, p1, p2, p3, p0])


def plane_4planes_open(main_plane: Plane, planes: list[Plane]) -> Polyline | None:
    """Open four-point polyline of the main plane cut by four ordered boundary planes."""

    corners = []

    for i in range(4):
        edge = plane_plane_to_line_canonical(planes[i], planes[(i + 1) % 4])

        if edge is None:
            return None

        corner = line_plane(edge, main_plane, False)

        if corner is None:
            return None

        corners.append(corner)

    return Polyline(corners)


def plane_4lines(plane: Plane, l0: Line, l1: Line, l2: Line, l3: Line) -> object | None:
    """Closed quad of a plane cut by four infinite lines."""

    p0 = line_plane(l0, plane, False)

    if p0 is None:
        return None

    p1 = line_plane(l1, plane, False)

    if p1 is None:
        return None

    p2 = line_plane(l2, plane, False)

    if p2 is None:
        return None

    p3 = line_plane(l3, plane, False)

    if p3 is None:
        return None

    return Polyline([p0, p1, p2, p3, p0])


def line_two_planes(line: Line, plane0: Plane, plane1: Plane) -> object | None:
    """Clips a segment to the two plane intersections."""

    new_start = line_plane(line, plane0, True)
    new_end = line_plane(line, plane1, True)

    if new_start is None or new_end is None:
        return None

    return Line(
        new_start[0], new_start[1], new_start[2], new_end[0], new_end[1], new_end[2]
    )


def polyline_plane(polyline: Polyline, plane: Plane) -> tuple | None:
    """Polyline edge crossings with a plane and their edge indices."""

    n = polyline.point_count()

    if n < 2:
        return None

    points = []
    edge_ids = []

    for i in range(n - 1):
        a = polyline.get_point(i)
        b = polyline.get_point(i + 1)
        va = _plane_value_at(plane, a)
        vb = _plane_value_at(plane, b)
        a_on = abs(va) < Tolerance.ZERO_TOLERANCE
        b_on = abs(vb) < Tolerance.ZERO_TOLERANCE

        if a_on and b_on:
            continue

        if a_on:
            points.append(a)
            edge_ids.append(i)
            continue

        if b_on:
            if i + 2 == n:
                front = polyline.get_point(0)
                closes = (
                    abs(b[0] - front[0]) < Tolerance.ZERO_TOLERANCE
                    and abs(b[1] - front[1]) < Tolerance.ZERO_TOLERANCE
                    and abs(b[2] - front[2]) < Tolerance.ZERO_TOLERANCE
                )

                if not closes:
                    points.append(b)
                    edge_ids.append(i)

            continue

        seg = Line(a[0], a[1], a[2], b[0], b[1], b[2])
        hit = line_plane(seg, plane, True)

        if hit is not None:
            points.append(hit)
            edge_ids.append(i)

    if not points:
        return None

    return (points, edge_ids)


def line_line_3d(cutter: Line, seg: Line) -> Point | None:
    """Closest approach point on the infinite cutter to the segment."""

    result = line_line_parameters(
        cutter, seg, 0.0, intersect_segments=False, near_parallel_as_closest=False
    )

    if result is None:
        return None

    t0, _ = result
    s = cutter.start()
    e = cutter.end()

    return Point(
        s[0] + t0 * (e[0] - s[0]), s[1] + t0 * (e[1] - s[1]), s[2] + t0 * (e[2] - s[2])
    )


def scale_vector_to_distance_of_2planes(
    direction: "Vector", plane0: Plane, plane1: Plane
) -> object | None:
    """Direction scaled to span the distance between two planes."""

    mag = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)

    if mag < 1e-14:
        return None

    ray = Line(0.0, 0.0, 0.0, direction[0], direction[1], direction[2])
    q0 = line_plane(ray, plane0, False)
    q1 = line_plane(ray, plane1, False)

    if q0 is None or q1 is None:
        return None

    output = q1 - q0
    n1 = plane1.z_axis
    n1_mag = math.sqrt(n1[0] ** 2 + n1[1] ** 2 + n1[2] ** 2)

    if n1_mag < 1e-14:
        return None

    o0 = plane0.origin
    d = (
        (o0[0] - plane1.origin[0]) * n1[0]
        + (o0[1] - plane1.origin[1]) * n1[1]
        + (o0[2] - plane1.origin[2]) * n1[2]
    ) / n1_mag
    dist_ortho_sq = d * d

    if dist_ortho_sq < 1e-28:
        return None

    dist_sq = output[0] ** 2 + output[1] ** 2 + output[2] ** 2

    if dist_sq / dist_ortho_sq >= 10.0:
        return None

    return output


# ═══════════════════════════════════════════════════════════════════════════
# Plane 2D helpers
# ═══════════════════════════════════════════════════════════════════════════
def _plane_to_2d(
    p: Point, origin: Point, xax: Vector, yax: Vector
) -> tuple[float, float]:
    """Project a point into plane coordinates."""
    d = p - origin

    return (d.dot(xax), d.dot(yax))


def _plane_to_3d(
    p: tuple[float, float], origin: Point, xax: Vector, yax: Vector
) -> Point:
    """Lift plane coordinates back to a point."""
    return origin + xax * p[0] + yax * p[1]


def _distance_sq_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Squared distance of two 2D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]

    return dx * dx + dy * dy


def _signed_area_2d(ring: list[tuple[float, float]]) -> float:
    """Signed area of a 2D ring, positive when counter-clockwise."""

    area = 0.0
    n = len(ring)

    for i in range(n):
        area += ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1]

    return area


def _polyline_to_2d(
    polyline: Polyline, origin: Point, xax: Vector, yax: Vector
) -> list[tuple[float, float]]:
    """Project a polyline into plane coordinates."""

    ring = []

    for i in range(polyline.point_count()):
        ring.append(_plane_to_2d(polyline.get_point(i), origin, xax, yax))

    if len(ring) > 1 and _distance_sq_2d(ring[0], ring[-1]) < 1e-12:
        ring.pop()

    return ring


def _polyline_to_3d(
    ring: list[tuple[float, float]], origin: Point, xax: Vector, yax: Vector
) -> Polyline:
    """Lift a 2D ring back to a polyline."""

    pts = []

    for p in ring:
        pts.append(_plane_to_3d(p, origin, xax, yax))

    pts.append(pts[0])

    return Polyline(pts)


def _point_in_polygon_2d(
    ring: list[tuple[float, float]], p: tuple[float, float]
) -> bool:
    """Even-odd point in polygon test."""

    wn = 0
    n = len(ring)

    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        e = (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])

        if a[1] <= p[1] and b[1] > p[1] and e > 0.0:
            wn += 1
        elif a[1] > p[1] and b[1] <= p[1] and e < 0.0:
            wn -= 1

    return wn != 0


def _seg_seg_2d(s0, s1, e0, e1) -> tuple[float, float] | None:
    """Segment-segment crossing with parameters on both."""

    sx = s1[0] - s0[0]
    sy = s1[1] - s0[1]
    ex = e1[0] - e0[0]
    ey = e1[1] - e0[1]
    denom = sx * ey - sy * ex

    if abs(denom) < 1e-20:
        return None

    dx = e0[0] - s0[0]
    dy = e0[1] - s0[1]

    return ((dx * ey - dy * ex) / denom, (dx * sy - dy * sx) / denom)


def _collinear_overlap_2d(s0, s1, e0, e1) -> tuple[float, float] | None:
    """Overlap range of two collinear segments on the first."""

    sx = s1[0] - s0[0]
    sy = s1[1] - s0[1]
    ex = e1[0] - e0[0]
    ey = e1[1] - e0[1]
    sl2 = sx * sx + sy * sy
    el2 = ex * ex + ey * ey

    if sl2 < 1e-20 or el2 < 1e-20:
        return None

    if abs((sx * ey - sy * ex) / math.sqrt(sl2 * el2)) > 1e-4:
        return None

    apx = s0[0] - e0[0]
    apy = s0[1] - e0[1]

    if abs((apx * ey - apy * ex) / math.sqrt(el2)) > 1e-3:
        return None

    ts0 = (apx * ex + apy * ey) / el2
    ts1 = ((s1[0] - e0[0]) * ex + (s1[1] - e0[1]) * ey) / el2
    ov_min = max(0.0, min(ts0, ts1))
    ov_max = min(1.0, max(ts0, ts1))

    if ov_max - ov_min < 1e-9:
        return None

    tsr = ts1 - ts0

    if abs(tsr) < 1e-20:
        return None

    t_enter = max(0.0, min((ov_min - ts0) / tsr, (ov_max - ts0) / tsr))
    t_exit = min(1.0, max((ov_min - ts0) / tsr, (ov_max - ts0) / tsr))

    if t_exit - t_enter <= 1e-9:
        return None

    return (t_enter, t_exit)


def _closest_param_2d(p, a, b) -> float:
    """Parameter of the closest point on segment ab to p."""

    abx = b[0] - a[0]
    aby = b[1] - a[1]
    l2 = abx * abx + aby * aby

    if l2 < 1e-20:
        return 0.0

    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / l2

    return max(0.0, min(1.0, t))


def _distance_sq_seg_2d(p, a, b) -> float:
    """Squared distance from p to segment ab."""
    t = _closest_param_2d(p, a, b)

    return _distance_sq_2d(p, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))


def _segment_plate_parameters_2d(plate, p0, p1, coll_ranges) -> list[float]:
    """Parameters along one joint segment where it crosses or overlaps the plate edges."""

    EPS = 1e-9
    ts = [0.0]

    for i in range(len(plate)):
        a = plate[i]
        b = plate[(i + 1) % len(plate)]
        crossing = _seg_seg_2d(p0, p1, a, b)

        if crossing is not None:
            t_s, t_e = crossing

            if EPS < t_s < 1.0 - EPS and -EPS <= t_e <= 1.0 + EPS:
                ts.append(t_s)

        overlap = _collinear_overlap_2d(p0, p1, a, b)

        if overlap is None:
            continue

        coll_ranges.append(overlap)

        if EPS < overlap[0] < 1.0 - EPS:
            ts.append(overlap[0])

        if EPS < overlap[1] < 1.0 - EPS:
            ts.append(overlap[1])

    ts.append(1.0)
    ts.sort()
    unique = []

    for t in ts:
        if not unique or abs(t - unique[-1]) >= EPS:
            unique.append(t)

    return unique


def _in_ranges_2d(ranges, t) -> bool:
    """Whether t falls in any of the ranges."""

    for r in ranges:
        if r[0] - 1e-9 <= t <= r[1] + 1e-9:
            return True

    return False


def _clip_open_path_2d(plate, joint) -> list[list[tuple[float, float]]]:
    """Sub-segments of the open joint path inside the plate, as separate pieces."""

    pieces = []

    for s in range(len(joint) - 1):
        p0 = joint[s]
        p1 = joint[s + 1]
        coll_ranges = []
        ts = _segment_plate_parameters_2d(plate, p0, p1, coll_ranges)
        current = []

        for i in range(len(ts) - 1):
            t_mid = 0.5 * (ts[i] + ts[i + 1])
            mid = (p0[0] + (p1[0] - p0[0]) * t_mid, p0[1] + (p1[1] - p0[1]) * t_mid)
            include = _point_in_polygon_2d(plate, mid) or _in_ranges_2d(
                coll_ranges, t_mid
            )

            if not include:
                if current:
                    pieces.append(current)

                current = []
                continue

            sub_a = (p0[0] + (p1[0] - p0[0]) * ts[i], p0[1] + (p1[1] - p0[1]) * ts[i])
            sub_b = (
                p0[0] + (p1[0] - p0[0]) * ts[i + 1],
                p0[1] + (p1[1] - p0[1]) * ts[i + 1],
            )

            if current and _distance_sq_2d(current[-1], sub_a) >= 1e-18:
                pieces.append(current)
                current = []

            if not current:
                current.append(sub_a)

            current.append(sub_b)

        if current:
            pieces.append(current)

    return pieces


def _chain_pieces_2d(pieces) -> list[tuple[float, float]]:
    """Chains clipped pieces end to end into one path."""

    DISTANCE_SQ = 0.01
    chain = []

    for piece in pieces:
        if len(piece) <= 1:
            continue

        if not chain:
            chain = list(piece)
            continue

        pts = list(piece)

        if (
            _distance_sq_2d(chain[-1], pts[0]) > DISTANCE_SQ
            and _distance_sq_2d(chain[-1], pts[-1]) > DISTANCE_SQ
        ):
            chain.reverse()

        if _distance_sq_2d(chain[-1], pts[0]) > _distance_sq_2d(chain[-1], pts[-1]):
            pts.reverse()

        for j in range(1, len(pts)):
            chain.append(pts[j])

    return chain


def _chain_plate_parameters_2d(plate, chain) -> tuple[float, float]:
    """Plate edge parameters of the chain ends, or -1 when an end is off the plate."""

    t0 = -1.0
    t1 = -1.0

    for i in range(len(plate)):
        a = plate[i]
        b = plate[(i + 1) % len(plate)]

        if _distance_sq_seg_2d(chain[0], a, b) < 1.0:
            t0 = float(i) + _closest_param_2d(chain[0], a, b)

        if _distance_sq_seg_2d(chain[-1], a, b) < 1.0:
            t1 = float(i) + _closest_param_2d(chain[-1], a, b)

        if t0 >= 0.0 and t1 >= 0.0:
            return (t0, t1)

    return (t0, t1)


def _offset_ring_2d(ring, delta, concave_notch) -> list[tuple[float, float]]:
    """Miter offset of a closed 2D ring by delta along the edge normals."""

    n = len(ring)
    normals = []

    for i in range(n):
        ex = ring[(i + 1) % n][0] - ring[i][0]
        ey = ring[(i + 1) % n][1] - ring[i][1]
        length = math.sqrt(ex * ex + ey * ey)

        if length < 1e-12:
            normals.append((0.0, 0.0))
        else:
            normals.append((ey / length, -ex / length))

    out = []

    for i in range(n):
        np_ = normals[(i + n - 1) % n]
        nn = normals[i]
        p = ring[i]
        cos_a = np_[0] * nn[0] + np_[1] * nn[1]
        sin_a = np_[0] * nn[1] - np_[1] * nn[0]
        denom = 1.0 + cos_a

        if cos_a > -0.999 and sin_a * delta < 0.0 and concave_notch:
            out.append((p[0] + np_[0] * delta, p[1] + np_[1] * delta))
            out.append(p)
            out.append((p[0] + nn[0] * delta, p[1] + nn[1] * delta))
        elif abs(denom) < 1e-9:
            bx = np_[0] + nn[0]
            by = np_[1] + nn[1]
            bl = math.sqrt(bx * bx + by * by)

            if bl < 1e-12:
                out.append((p[0] + nn[0] * delta, p[1] + nn[1] * delta))
            else:
                out.append((p[0] + (bx / bl) * delta, p[1] + (by / bl) * delta))
        else:
            k = delta / denom
            out.append((p[0] + (np_[0] + nn[0]) * k, p[1] + (np_[1] + nn[1]) * k))

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Polyline booleans
# ═══════════════════════════════════════════════════════════════════════════
def polyline_boolean(a: Polyline, b: Polyline, clip_type: int) -> list[Polyline]:
    """Boolean of two closed planar polylines, clip_type 0 intersection, 1 union, 2 difference."""
    return Polyline.boolean_op(a, b, clip_type)


def offset_in_3d(polyline: Polyline, plane: Plane, offset: float) -> bool:
    """Miter offset of a closed polyline in the plane's 2D frame, positive outward, in place."""

    if polyline.point_count() < 3:
        return False

    origin = polyline.get_point(0)
    xax = plane.base1()
    yax = plane.base2()
    ring = _polyline_to_2d(polyline, origin, xax, yax)

    if len(ring) < 3:
        return False

    delta = -offset if _signed_area_2d(ring) < 0.0 else offset
    out = _offset_ring_2d(ring, delta, offset > 0.0)

    if len(out) < 3:
        return False

    if abs(_signed_area_2d(out)) * 0.5 < 0.0001:
        return False

    cp = 0

    for i in range(1, len(out)):
        if _distance_sq_2d(out[i], ring[0]) < _distance_sq_2d(out[cp], ring[0]):
            cp = i

    out = out[cp:] + out[:cp]
    polyline.coords = _polyline_to_3d(out, origin, xax, yax).coords

    return True


def _polyline_boolean_2d(
    a2d: Polyline, b2d: Polyline, intersection_type: int
) -> list[Polyline]:
    """Boolean of two flat polylines, intersection_type 0 intersect, 1 union, 2 difference, 3 xor; empty on failure."""

    if 0 <= intersection_type <= 2:
        return BooleanPolyline.compute(a2d, b2d, intersection_type)

    if intersection_type != 3:
        return []

    u = BooleanPolyline.compute(a2d, b2d, 1)
    inter = BooleanPolyline.compute(a2d, b2d, 0)

    if not u:
        return []

    if not inter:
        return u

    return BooleanPolyline.compute(u[0], inter[0], 2)


def _collapse_close_points(
    ring: list[tuple[float, float]], eps: float
) -> list[tuple[float, float]]:
    """Ring without consecutive points closer than eps, the closing point included."""

    eps_sq = eps * eps
    collapsed = []

    for p in ring:
        if not collapsed or _distance_sq_2d(p, collapsed[-1]) >= eps_sq:
            collapsed.append(p)

    if len(collapsed) >= 2 and _distance_sq_2d(collapsed[-1], collapsed[0]) < eps_sq:
        collapsed.pop()

    return collapsed


def polyline_boolean_2d_in_plane(
    polyline0: Polyline,
    polyline1: Polyline,
    plane: Plane,
    intersection_type: int,
    include_triangles: bool = False,
    min_area: float = 0.01,
    collapse_eps: float = 0.0,
) -> Polyline | None:
    """Boolean in the plane's 2D frame, intersection_type 0 intersect, 1 union, 2 difference, 3 xor."""

    if polyline0.point_count() < 3 or polyline1.point_count() < 3:
        return None

    origin = polyline0.get_point(0)
    xax = plane.base1()
    yax = plane.base2()
    flat_origin = Point(0.0, 0.0, 0.0)
    flat_x = Vector(1.0, 0.0, 0.0)
    flat_y = Vector(0.0, 1.0, 0.0)
    a2d = _polyline_to_3d(
        _polyline_to_2d(polyline0, origin, xax, yax), flat_origin, flat_x, flat_y
    )
    b2d = _polyline_to_3d(
        _polyline_to_2d(polyline1, origin, xax, yax), flat_origin, flat_x, flat_y
    )
    result_2d = _polyline_boolean_2d(a2d, b2d, intersection_type)

    if not result_2d:
        return None

    ring = _polyline_to_2d(result_2d[0], flat_origin, flat_x, flat_y)

    if len(ring) < 3:
        return None

    if collapse_eps > 0.0:
        ring = _collapse_close_points(ring, collapse_eps)

        if len(ring) < 3:
            return None

    if len(ring) == 3 and not include_triangles:
        return None

    if abs(_signed_area_2d(ring)) * 0.5 <= min_area:
        return None

    return _polyline_to_3d(ring, origin, xax, yax)


# ═══════════════════════════════════════════════════════════════════════════
# Joints
# ═══════════════════════════════════════════════════════════════════════════
def polyline_plane_to_line(
    poly: Polyline, plane: Plane, align_start: Point
) -> Line | None:
    """Polyline-plane crossings as one line oriented from align_start."""

    result = polyline_plane(poly, plane)

    if result is None:
        return None

    pts, _ = result

    if len(pts) < 2:
        return None

    ia = 0
    ib = 1
    best = -1.0

    for p1 in range(len(pts) - 1):
        for p2 in range(p1 + 1, len(pts)):
            d = (pts[p1] - pts[p2]).magnitude_squared()

            if d > best:
                best = d
                ia = p1
                ib = p2

    a = pts[ia]
    b = pts[ib]

    if (a - align_start).magnitude_squared() <= (b - align_start).magnitude_squared():
        return Line.from_points(a, b)

    return Line.from_points(b, a)


def quad_from_line_top_bottom_planes(
    face_plane: Plane, line: Line, plane0: Plane, plane1: Plane
) -> Polyline | None:
    """Closed quad from a joint line bounded by top and bottom planes on a face plane."""

    lp0 = Plane.from_point_normal(line.start(), line.to_vector())
    lp1 = Plane.from_point_normal(line.end(), line.to_vector())
    edge0 = plane_plane(plane0, face_plane)
    edge1 = plane_plane(plane1, face_plane)

    if edge0 is None or edge1 is None:
        return None

    p0 = line_plane(edge0, lp0, False)
    p1 = line_plane(edge1, lp0, False)
    p2 = line_plane(edge1, lp1, False)
    p3 = line_plane(edge0, lp1, False)

    if p0 is None or p1 is None or p2 is None or p3 is None:
        return None

    return Polyline([p0, p1, p2, p3, p0])


def orthogonal_vector_between_two_plane_pairs(
    pp00: Plane, pp10: Plane, pp11: Plane
) -> Vector | None:
    """Vector orthogonal to the (pp00, pp10) line, anchored on the (pp00, pp11) line."""

    l0 = plane_plane_to_line_canonical(pp00, pp10)
    l1 = plane_plane_to_line_canonical(pp00, pp11)

    if l0 is None or l1 is None:
        return None

    if l0.to_vector().magnitude_squared() < 1e-20:
        return None

    return l1.start() - l0.closest_point(l1.start(), False)[1]


def closed_and_open_paths_2d(
    plate: Polyline, joint: Polyline, plane: Plane
) -> tuple[Polyline, tuple[float, float]] | None:
    """Open joint outline clipped to a closed plate polygon with the plate edge parameters."""

    origin = plate.get_point(0)
    xax = plane.base1()
    yax = plane.base2()
    plate2d = _polyline_to_2d(plate, origin, xax, yax)

    if len(plate2d) < 3:
        return None

    joint2d = []

    for i in range(joint.point_count()):
        joint2d.append(_plane_to_2d(joint.get_point(i), origin, xax, yax))

    if len(joint2d) < 2:
        return None

    c2d = _chain_pieces_2d(_clip_open_path_2d(plate2d, joint2d))

    if len(c2d) < 2:
        return None

    t0, t1 = _chain_plate_parameters_2d(plate2d, c2d)
    reverse_flag = t0 > t1

    if int(math.floor(t0)) == 0 and int(math.floor(t1)) == len(c2d) - 1:
        reverse_flag = not reverse_flag

    if reverse_flag:
        t0, t1 = t1, t0
        c2d.reverse()

    if t0 < 0.0 or t1 < 0.0:
        return None

    out_pts = []

    for p in c2d:
        out_pts.append(_plane_to_3d(p, origin, xax, yax))

    return Polyline(out_pts), (t0, t1)


# ═══════════════════════════════════════════════════════════════════════════
# Elements
# ═══════════════════════════════════════════════════════════════════════════
def _padded_face_boxes(
    polylines: list[list[Polyline]], tolerance: float
) -> list[list[list[float]]]:
    """Bounding box (min xyz, max xyz) of every face, padded by tolerance."""

    face_boxes = []

    for faces in polylines:
        boxes = []

        for f in faces:
            bx = [math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf]
            c = f.coords
            k = 0

            while k + 2 < len(c):
                bx[0] = min(bx[0], c[k])
                bx[3] = max(bx[3], c[k])
                bx[1] = min(bx[1], c[k + 1])
                bx[4] = max(bx[4], c[k + 1])
                bx[2] = min(bx[2], c[k + 2])
                bx[5] = max(bx[5], c[k + 2])
                k += 3

            for k in range(3):
                bx[k] -= tolerance
                bx[k + 3] += tolerance

            boxes.append(bx)

        face_boxes.append(boxes)

    return face_boxes


def _coplanar_face_overlap(
    face_a: Polyline, za: Vector, face_b: Polyline
) -> Polyline | None:
    """Closed overlap of two coplanar faces in the frame of face_a's first edge and normal za, None when they only touch."""

    pts_i = face_a.get_points()
    edge = Vector(
        pts_i[1][0] - pts_i[0][0],
        pts_i[1][1] - pts_i[0][1],
        pts_i[1][2] - pts_i[0][2],
    )
    edge.normalize_self()
    zax = za
    yax = zax.cross(edge)
    yax.normalize_self()
    pln = Plane.from_frame(pts_i[0], edge, yax, zax)
    bools = Polyline.boolean_op(face_a, face_b, 0, plane=pln)

    if not bools or bools[0].point_count() < 3:
        return None

    return bools[0] if bools[0].is_closed() else bools[0].closed()


def face_to_face(
    adjacency: list[int],
    polylines: list[list[Polyline]],
    planes: list[list[Plane]],
    coplanar_tolerance: float = 5.0,
) -> list[tuple[int, int, int, int, int, Polyline]]:
    """Face-to-face contacts (a, b, face_a, face_b, type, polyline) with type 0 side-side, 1 side-top, 2 top-top."""

    results = []
    face_boxes = _padded_face_boxes(polylines, coplanar_tolerance)
    idx = 0

    while idx + 1 < len(adjacency):
        a = adjacency[idx]
        b = adjacency[idx + 1]
        found = False
        i = 0

        while i < len(planes[a]) and not found:
            oa = planes[a][i].origin
            za = planes[a][i].z_axis
            ba = face_boxes[a][i]

            for j in range(len(planes[b])):
                bb = face_boxes[b][j]

                if (
                    ba[0] > bb[3]
                    or bb[0] > ba[3]
                    or ba[1] > bb[4]
                    or bb[1] > ba[4]
                    or ba[2] > bb[5]
                    or bb[2] > ba[5]
                ):
                    continue

                if not Plane.is_coplanar_from_normals(
                    oa,
                    za,
                    planes[b][j].origin,
                    planes[b][j].z_axis,
                    False,
                    coplanar_tolerance,
                ):
                    continue

                jpl = _coplanar_face_overlap(polylines[a][i], za, polylines[b][j])

                if jpl is None:
                    continue

                typ = (0 if i > 1 else 1) + (0 if j > 1 else 1)
                results.append((a, b, i, j, typ, jpl))
                found = True
                break

            i += 1

        idx += 4

    return results


def adjacency_search(elements: list[Element], inflate: float = 5.0) -> list[int]:
    """Adjacent element pairs by BVH broad phase and OBB narrow phase."""

    n = len(elements)
    obbs = []

    for element in elements:
        pts = []

        for pl in element.polylines:
            pts.extend(pl.get_points())

        obbs.append(OBB.from_points(pts, inflate))

    aabbs = []

    for obb in obbs:
        aabbs.append(obb.aabb())

    ws = 0.0

    for a in aabbs:
        ws = max(ws, abs(a.cx + a.hx))
        ws = max(ws, abs(a.cy + a.hy))
        ws = max(ws, abs(a.cz + a.hz))
        ws = max(ws, abs(a.cx - a.hx))
        ws = max(ws, abs(a.cy - a.hy))
        ws = max(ws, abs(a.cz - a.hz))

    bvh = SpatialBVH()
    bvh.build_from_aabbs(aabbs, ws * 2)
    adjacency = []

    for i in range(n):
        hits = bvh.query_aabb(aabbs[i])

        for j in hits:
            if i < j and obbs[i].collides_with(obbs[j]):
                adjacency.append(i)
                adjacency.append(j)
                adjacency.append(-1)
                adjacency.append(-1)

    return adjacency


def _line_line_frame(s0: Line, s1: Line) -> tuple:
    """Directions of two segments, their unit normal, and whether they are parallel within one degree."""

    EPS_PAR = 1.0
    v0 = s0.to_vector()
    v1 = s1.to_vector()
    normal = v0.cross(v1)
    ang = v0.angle(v1, False, True)
    is_parallel = (
        normal.magnitude_squared() < 1e-24 or (90.0 - abs(ang - 90.0)) < EPS_PAR
    )

    if is_parallel:
        normal = Plane.from_point_normal(s0.start(), v0).base1()

    normal.normalize_self()

    return v0, v1, normal, is_parallel


def _line_line_shared_end(s0: Line, s1: Line) -> tuple | None:
    """End shared by two segments as p0 and p1, with unit directions leaving it along each segment."""

    DIST_SQ = 1e-6
    ends0 = (s0.start(), s0.end())
    ends1 = (s1.start(), s1.end())

    for i in range(2):
        for j in range(2):
            if (ends0[i] - ends1[j]).magnitude_squared() < DIST_SQ:
                v0 = ends0[1 - i] - ends0[i]
                v1 = ends1[1 - j] - ends1[j]
                v0.normalize_self()
                v1.normalize_self()

                return ends0[i], ends0[i], v0, v1

    return None


def _line_line_parallel(s0: Line, s1: Line, v0: Vector, v1: Vector) -> tuple:
    """Closest points of two parallel segments at the middle of their overlap, each unit direction flipped to leave its nearer end."""

    pts = []

    for q in (s0.start(), s0.end(), s1.start(), s1.end()):
        q0 = s0.closest_point(q, False)[1]
        q1 = s1.closest_point(q, False)[1]
        pts.append(((q0 - s0.start()).dot(v0), (q1 - s1.start()).dot(v1)))

    pts.sort(key=lambda a: a[0])
    seg0_a = Point(
        s0.start()[0] + pts[1][0] * v0[0],
        s0.start()[1] + pts[1][0] * v0[1],
        s0.start()[2] + pts[1][0] * v0[2],
    )
    seg0_b = Point(
        s0.start()[0] + pts[2][0] * v0[0],
        s0.start()[1] + pts[2][0] * v0[1],
        s0.start()[2] + pts[2][0] * v0[2],
    )
    seg1_a = Point(
        s1.start()[0] + pts[1][1] * v1[0],
        s1.start()[1] + pts[1][1] * v1[1],
        s1.start()[2] + pts[1][1] * v1[2],
    )
    seg1_b = Point(
        s1.start()[0] + pts[2][1] * v1[0],
        s1.start()[1] + pts[2][1] * v1[1],
        s1.start()[2] + pts[2][1] * v1[2],
    )
    m0 = Point(
        (seg0_a[0] + seg0_b[0]) * 0.5,
        (seg0_a[1] + seg0_b[1]) * 0.5,
        (seg0_a[2] + seg0_b[2]) * 0.5,
    )
    m1 = Point(
        (seg1_a[0] + seg1_b[0]) * 0.5,
        (seg1_a[1] + seg1_b[1]) * 0.5,
        (seg1_a[2] + seg1_b[2]) * 0.5,
    )
    avg = Point((m0[0] + m1[0]) * 0.5, (m0[1] + m1[1]) * 0.5, (m0[2] + m1[2]) * 0.5)
    p0 = s0.closest_point(avg, False)[1]
    p1 = s1.closest_point(avg, False)[1]

    if s0.closest_point(p0, False)[0] > 0.5:
        v0 = -v0

    if s1.closest_point(p1, False)[0] > 0.5:
        v1 = -v1

    return p0, p1, v0, v1


def _line_line_types(
    tt0: float, tt1: float, above_closer_to_edge: float, v0: Vector, v1: Vector
) -> tuple:
    """Types (0 end, 1 side) from the positions tt0 and tt1 along the polylines, the direction of an end past the middle flipped."""

    close0 = 2.0 * abs(0.5 - tt0)
    close1 = 2.0 * abs(0.5 - tt1)

    if above_closer_to_edge < 0.0:
        type0 = 1
        type1 = 1
    elif above_closer_to_edge > 1.0:
        type0 = 0 if tt0 < tt1 else 1
        type1 = 1 if tt0 < tt1 else 0
    else:
        type0 = 0 if close0 > above_closer_to_edge else 1
        type1 = 0 if close1 > above_closer_to_edge else 1

        if close0 > close1 and type0 == 0 and type1 == 0:
            type1 = 1
        elif close0 < close1 and type0 == 0 and type1 == 0:
            type0 = 1

    if tt0 > 0.5 and type0 == 0:
        v0 = -v0

    if tt1 > 0.5 and type1 == 0:
        v1 = -v1

    return type0, type1, v0, v1


def line_line_classified(
    s0: Line,
    s1: Line,
    n_segs_0: int,
    n_segs_1: int,
    cur_seg_0: int,
    cur_seg_1: int,
    above_closer_to_edge: float,
) -> tuple | None:
    """Classifies two segments as end-to-end, side-to-end or cross with closest points and directions."""

    v0, v1, normal, is_parallel = _line_line_frame(s0, s1)
    shared = _line_line_shared_end(s0, s1)

    if shared is not None:
        p0, p1, v0, v1 = shared

        return (p0, p1, v0, v1, normal, 0, 0, is_parallel)

    v0.normalize_self()
    v1.normalize_self()

    if is_parallel:
        p0, p1, v0, v1 = _line_line_parallel(s0, s1, v0, v1)

        return (p0, p1, v0, v1, normal, 0, 0, is_parallel)

    result = line_line_parameters(s0, s1, 0.0, False, True)

    if result is None:
        return None

    t0_v, t1_v = result
    t0c = max(0.0, min(1.0, t0_v))
    t1c = max(0.0, min(1.0, t1_v))
    p0 = s0.point_at(t0c)
    p1 = s1.point_at(t1c)
    tt0 = (t0c + float(cur_seg_0)) / float(n_segs_0)
    tt1 = (t1c + float(cur_seg_1)) / float(n_segs_1)
    type0, type1, v0, v1 = _line_line_types(tt0, tt1, above_closer_to_edge, v0, v1)

    return (p0, p1, v0, v1, normal, type0, type1, is_parallel)
