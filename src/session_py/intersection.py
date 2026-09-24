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
from .tolerance import Tolerance
from .vector import Vector

if TYPE_CHECKING:
    from .element import Element
    from .nurbssurface import NurbsSurface


# ═══════════════════════════════════════════════════════════════════════════
# Lines and planes
# ═══════════════════════════════════════════════════════════════════════════


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

    n0 = plane0.z_axis
    n1 = plane1.z_axis
    n2 = plane2.z_axis

    det = n0.dot(n1.cross(n2))

    if abs(det) < 1e-10:
        return None

    d0 = plane0.d
    d1 = plane1.d
    d2 = plane2.d

    p = (n1.cross(n2) * (-d0) + n2.cross(n0) * (-d1) + n0.cross(n1) * (-d2)) * (
        1.0 / det
    )

    return Point(p[0], p[1], p[2])


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
        tm = (ta + tb) * 0.5
        pm = curve.point_at(tm)
        dist = _curve_signed_distance_to_plane(pm, plane)

        if abs(dist) < tolerance:
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

        return

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
        return

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
        tangent = field.tangent(u, v, direction)

        if tangent is None:
            if math.hypot(prev_tu, prev_tv) < 1e-14:
                break

            tangent = (prev_tu, prev_tv)

        tu, tv = tangent
        local_step = _turn_step(field, tu, tv, prev_tu, prev_tv)
        mid = field.tangent(
            u + local_step * 0.5 * tu, v + local_step * 0.5 * tv, direction
        )

        if mid is not None:
            tu, tv = mid

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


def _fitted_max_deviation(cand, pts, chords):
    """Largest distance from each point to the curve, found by ternary search around its chord parameter."""

    m = len(pts)
    ft0, ft1 = cand.domain()
    max_dev = 0.0

    for i in range(m):
        t = ft0 + (ft1 - ft0) * chords[i]
        w2 = (ft1 - ft0) * 2.0 / max(m - 1, 1)
        lo = max(ft0, t - w2)
        hi = min(ft1, t + w2)

        for _ in range(20):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3

            if cand.point_at(m1).distance(pts[i]) < cand.point_at(m2).distance(pts[i]):
                hi = m2
            else:
                lo = m1

        max_dev = max(max_dev, cand.point_at(0.5 * (lo + hi)).distance(pts[i]))

    return max_dev


def _fit_planar_freeform(all_pts, is_loop, plane, fit_tol):
    """Cubic fitted to the points in the plane's frame, CVs doubled until within fit_tol, lifted back to 3D."""

    m = len(all_pts)

    if m < 4:
        return NurbsCurve()

    pts_2d = _plane_points_2d(all_pts, plane)
    chords = _chord_parameters(pts_2d, is_loop)
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

        max_dev = _fitted_max_deviation(cand, pts_2d, chords)

        if max_dev < best_dev:
            best_dev = max_dev
            crv_2d = cand

        if max_dev < fit_tol:
            break

        target_cvs = min(target_cvs * 2, max_cvs + 1)

    if not crv_2d.is_valid():
        if is_loop:
            crv_2d = NurbsCurve.create_interpolated(
                pts_2d, CurveNurbsKnotStyle.ChordPeriodic
            )
        else:
            crv_2d = NurbsCurve.create_interpolated(pts_2d)

    if not crv_2d.is_valid():
        return NurbsCurve()

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

    return crv_2d


def _surface_plane_fit_3d(
    all_pts, is_loop, plane, step, uv_to_3d, uv_to_3d_min, allow_conics=True
):
    """Fit a 3D plane-constrained NurbsCurve to traced intersection points."""

    crv = NurbsCurve()

    if allow_conics and is_loop and len(all_pts) >= 6:
        ax = plane.x_axis
        ay = plane.y_axis
        po = plane.origin

        def to2d_circle(p):
            dx = p[0] - po[0]
            dy = p[1] - po[1]
            dz = p[2] - po[2]

            return (
                dx * ax[0] + dy * ax[1] + dz * ax[2],
                dx * ay[0] + dy * ay[1] + dz * ay[2],
            )

        n = len(all_pts)
        x1, y1 = to2d_circle(all_pts[0])
        x2, y2 = to2d_circle(all_pts[n // 3])
        x3, y3 = to2d_circle(all_pts[2 * n // 3])

        ax_ = x2 - x1
        ay_ = y2 - y1
        bx_ = x3 - x1
        by_ = y3 - y1
        D = 2.0 * (ax_ * by_ - ay_ * bx_)

        if abs(D) > 1e-10:
            a2 = ax_ * ax_ + ay_ * ay_
            b2 = bx_ * bx_ + by_ * by_
            ccx = x1 + (by_ * a2 - ay_ * b2) / D
            ccy = y1 + (ax_ * b2 - bx_ * a2) / D
            radius = math.hypot(x1 - ccx, y1 - ccy)

            max_dev = 0.0

            for p in all_pts:
                px, py = to2d_circle(p)
                max_dev = max(max_dev, abs(math.hypot(px - ccx, py - ccy) - radius))

            circle_tol = max(radius * 1e-4, 1e-6)

            if radius > 1e-10 and max_dev < circle_tol:
                cx3d = po[0] + ccx * ax[0] + ccy * ay[0]
                cy3d = po[1] + ccx * ax[1] + ccy * ay[1]
                cz3d = po[2] + ccx * ax[2] + ccy * ay[2]

                w = math.sqrt(2.0) / 2.0
                cx_ = [1, 1, 0, -1, -1, -1, 0, 1, 1]
                cy_ = [0, 1, 1, 1, 0, -1, -1, -1, 0]
                wts = [1, w, 1, w, 1, w, 1, w, 1]
                crv = NurbsCurve(3, True, 3, 9)
                nurbsknots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

                for i in range(10):
                    crv.set_nurbsknot(i, nurbsknots[i])

                for i in range(9):
                    px = cx3d + radius * (cx_[i] * ax[0] + cy_[i] * ay[0])
                    py = cy3d + radius * (cx_[i] * ax[1] + cy_[i] * ay[1])
                    pz = cz3d + radius * (cx_[i] * ax[2] + cy_[i] * ay[2])
                    crv.set_cv_4d(i, px * wts[i], py * wts[i], pz * wts[i], wts[i])

    if not crv.is_valid() and allow_conics and is_loop and len(all_pts) >= 8:
        ax = plane.x_axis
        ay = plane.y_axis
        po = plane.origin

        def to2d_ellipse(p):
            dx = p[0] - po[0]
            dy = p[1] - po[1]
            dz = p[2] - po[2]

            return (
                dx * ax[0] + dy * ax[1] + dz * ax[2],
                dx * ay[0] + dy * ay[1] + dz * ay[2],
            )

        n = len(all_pts)
        AtA = [[0.0] * 5 for _ in range(5)]
        Atb = [0.0] * 5

        for i in range(n):
            x, y = to2d_ellipse(all_pts[i])
            row = [x * x, x * y, y * y, x, y]

            for r in range(5):
                Atb[r] += row[r]

                for c in range(5):
                    AtA[r][c] += row[r] * row[c]

        M = [[0.0] * 6 for _ in range(5)]

        for r in range(5):
            for c in range(5):
                M[r][c] = AtA[r][c]

            M[r][5] = Atb[r]

        ok = True

        for col in range(5):
            if not ok:
                break

            pivot = col

            for r in range(col + 1, 5):
                if math.fabs(M[r][col]) > math.fabs(M[pivot][col]):
                    pivot = r

            if math.fabs(M[pivot][col]) < 1e-20:
                ok = False
                break

            if pivot != col:
                M[col], M[pivot] = M[pivot], M[col]

            for r in range(col + 1, 5):
                f = M[r][col] / M[col][col]

                for j in range(col, 6):
                    M[r][j] -= f * M[col][j]

        coef = [0.0] * 5

        if ok:
            for i in range(4, -1, -1):
                s = M[i][5]

                for j in range(i + 1, 5):
                    s -= M[i][j] * coef[j]

                coef[i] = s / M[i][i]

        A_c = coef[0]
        B_c = coef[1]
        C_c = coef[2]
        D_c = coef[3]
        E_c = coef[4]
        disc = B_c * B_c - 4 * A_c * C_c

        if ok and disc < -1e-10 and math.fabs(A_c) > 1e-14:
            max_conic_dev = 0.0

            for p in all_pts:
                x, y = to2d_ellipse(p)
                val = A_c * x * x + B_c * x * y + C_c * y * y + D_c * x + E_c * y - 1.0
                max_conic_dev = max(max_conic_dev, math.fabs(val))

            scale = max(math.fabs(A_c), math.fabs(C_c))
            norm_dev = max_conic_dev / max(scale, 1e-10)

            if norm_dev < 0.01:
                det = 4 * A_c * C_c - B_c * B_c
                cx = (B_c * E_c - 2 * C_c * D_c) / det
                cy = (B_c * D_c - 2 * A_c * E_c) / det

                theta = 0.5 * math.atan2(B_c, A_c - C_c)
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                A2 = A_c * cos_t * cos_t + B_c * cos_t * sin_t + C_c * sin_t * sin_t
                C2 = A_c * sin_t * sin_t - B_c * cos_t * sin_t + C_c * cos_t * cos_t
                f_val = (
                    A_c * cx * cx
                    + B_c * cx * cy
                    + C_c * cy * cy
                    + D_c * cx
                    + E_c * cy
                    - 1.0
                )
                rhs = -f_val

                if rhs > 1e-14 and A2 > 1e-14 and C2 > 1e-14:
                    semi_a = math.sqrt(rhs / A2)
                    semi_b = math.sqrt(rhs / C2)

                    cx3d = po[0] + cx * ax[0] + cy * ay[0]
                    cy3d = po[1] + cx * ax[1] + cy * ay[1]
                    cz3d = po[2] + cx * ax[2] + cy * ay[2]

                    ea = Vector(
                        cos_t * ax[0] + sin_t * ay[0],
                        cos_t * ax[1] + sin_t * ay[1],
                        cos_t * ax[2] + sin_t * ay[2],
                    )
                    eb = Vector(
                        -sin_t * ax[0] + cos_t * ay[0],
                        -sin_t * ax[1] + cos_t * ay[1],
                        -sin_t * ax[2] + cos_t * ay[2],
                    )

                    w = math.sqrt(2.0) / 2.0
                    cx_ = [1, 1, 0, -1, -1, -1, 0, 1, 1]
                    cy_ = [0, 1, 1, 1, 0, -1, -1, -1, 0]
                    wts = [1, w, 1, w, 1, w, 1, w, 1]
                    crv = NurbsCurve(3, True, 3, 9)
                    nurbsknots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

                    for i in range(10):
                        crv.set_nurbsknot(i, nurbsknots[i])

                    for i in range(9):
                        px = cx3d + semi_a * cx_[i] * ea[0] + semi_b * cy_[i] * eb[0]
                        py = cy3d + semi_a * cx_[i] * ea[1] + semi_b * cy_[i] * eb[1]
                        pz = cz3d + semi_a * cx_[i] * ea[2] + semi_b * cy_[i] * eb[2]
                        crv.set_cv_4d(i, px * wts[i], py * wts[i], pz * wts[i], wts[i])

                    et0, et1 = crv.domain()
                    max_ell_dev = 0.0

                    for p in all_pts:
                        px2, py2 = to2d_ellipse(p)
                        lx = cos_t * (px2 - cx) + sin_t * (py2 - cy)
                        ly = -sin_t * (px2 - cx) + cos_t * (py2 - cy)
                        ang = math.atan2(ly / semi_b, lx / semi_a)
                        ex = (
                            cx
                            + semi_a * math.cos(ang) * cos_t
                            - semi_b * math.sin(ang) * sin_t
                        )
                        ey = (
                            cy
                            + semi_a * math.cos(ang) * sin_t
                            + semi_b * math.sin(ang) * cos_t
                        )
                        dev = math.hypot(px2 - ex, py2 - ey)
                        max_ell_dev = max(max_ell_dev, dev)

                    ell_tol = max(semi_a, semi_b) * 5e-3

                    if max_ell_dev > ell_tol:
                        crv = NurbsCurve()

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

    return math.floor(x + 0.5) if x >= 0.0 else -math.floor(-x + 0.5)


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


def _solve_gauss(M, rhs, n):
    """Solve an n x n linear system by Gaussian elimination with partial pivoting."""

    A = [list(M[r]) + [rhs[r]] for r in range(n)]

    for col in range(n):
        pivot = col

        for r in range(col + 1, n):
            if abs(A[r][col]) > abs(A[pivot][col]):
                pivot = r

        if abs(A[pivot][col]) < 1e-20:
            return None

        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]

        for r in range(col + 1, n):
            f = A[r][col] / A[col][col]

            for j in range(col, n + 1):
                A[r][j] -= f * A[col][j]

    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        s = A[i][n]

        for j in range(i + 1, n):
            s -= A[i][j] * x[j]

        x[i] = s / A[i][i]

    return x


# ═══════════════════════════════════════════════════════════════════════════
# Analytic quadric surface intersection
# ═══════════════════════════════════════════════════════════════════════════


def _ssi_dot(u, v):
    """Dot product of two V3."""
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _ssi_cross(u, v):
    """Cross product of two V3."""

    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _ssi_unit(v):
    """Unit V3, or the input when degenerate."""
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    return (v[0] / length, v[1] / length, v[2] / length) if length > 1e-300 else v


def _ortho_basis(n):
    """Two unit vectors spanning the plane perpendicular to unit n."""

    ax = 1.0 if abs(n[0]) <= abs(n[1]) and abs(n[0]) <= abs(n[2]) else 0.0
    ay = 1.0 if ax == 0.0 and abs(n[1]) <= abs(n[2]) else 0.0
    az = 1.0 if ax == 0.0 and ay == 0.0 else 0.0
    ux = ay * n[2] - az * n[1]
    uy = az * n[0] - ax * n[2]
    uz = ax * n[1] - ay * n[0]
    ul = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / ul, uy / ul, uz / ul
    vx = n[1] * uz - n[2] * uy
    vy = n[2] * ux - n[0] * uz
    vz = n[0] * uy - n[1] * ux

    return (ux, uy, uz), (vx, vy, vz)


def _exact_circle(cx, cy, cz, xa, ya, radius):
    """Exact 9-CV rational NURBS circle."""

    w = math.sqrt(2.0) / 2.0
    px = [1, 1, 0, -1, -1, -1, 0, 1, 1]
    py = [0, 1, 1, 1, 0, -1, -1, -1, 0]
    wts = [1, w, 1, w, 1, w, 1, w, 1]
    crv = NurbsCurve(3, True, 3, 9)
    knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    for i in range(10):
        crv.set_nurbsknot(i, float(knots[i]))

    for i in range(9):
        x = cx + radius * (px[i] * xa[0] + py[i] * ya[0])
        y = cy + radius * (px[i] * xa[1] + py[i] * ya[1])
        z = cz + radius * (px[i] * xa[2] + py[i] * ya[2])
        crv.set_cv_4d(i, x * wts[i], y * wts[i], z * wts[i], wts[i])

    crv.set_domain(0.0, 1.0)

    return crv


def _exact_ellipse(cx, cy, cz, ea, eb, semi_a, semi_b):
    """Exact 9-CV rational NURBS ellipse."""

    w = math.sqrt(2.0) / 2.0
    px = [1, 1, 0, -1, -1, -1, 0, 1, 1]
    py = [0, 1, 1, 1, 0, -1, -1, -1, 0]
    wts = [1, w, 1, w, 1, w, 1, w, 1]
    crv = NurbsCurve(3, True, 3, 9)
    knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    for i in range(10):
        crv.set_nurbsknot(i, float(knots[i]))

    for i in range(9):
        x = cx + semi_a * px[i] * ea[0] + semi_b * py[i] * eb[0]
        y = cy + semi_a * px[i] * ea[1] + semi_b * py[i] * eb[1]
        z = cz + semi_a * px[i] * ea[2] + semi_b * py[i] * eb[2]
        crv.set_cv_4d(i, x * wts[i], y * wts[i], z * wts[i], wts[i])

    crv.set_domain(0.0, 1.0)

    return crv


def _jacobi_eig3(M):
    """Eigenvalues/vectors of a symmetric 3x3 matrix (cyclic Jacobi)."""

    a = [[M[r][c] for c in range(3)] for r in range(3)]
    v = [[1.0 if r == c else 0.0 for c in range(3)] for r in range(3)]

    for _ in range(50):
        off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])

        if off < 1e-18:
            break

        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-300:
                continue

            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            t = (1.0 if theta >= 0 else -1.0) / (
                abs(theta) + math.sqrt(theta * theta + 1.0)
            )
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c

            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq

            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk

            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq

    eigvals = [a[0][0], a[1][1], a[2][2]]
    eigvecs = [(v[0][k], v[1][k], v[2][k]) for k in range(3)]

    return eigvals, eigvecs


def _fit_cylinder(surface, tol):
    """Recognize a cylinder from surface samples: axis point, axis direction and radius."""

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    pts = []
    nrm = []

    for i in range(5):
        for j in range(5):
            uu = u0 + (u1 - u0) * i / 4.0
            vv = v0 + (v1 - v0) * j / 4.0
            pts.append(surface.point_at(uu, vv))
            n = surface.normal_at(uu, vv)
            nrm.append((n[0], n[1], n[2]))

    M = [[0.0] * 3 for _ in range(3)]

    for n in nrm:
        for r in range(3):
            for c in range(3):
                M[r][c] += n[r] * n[c]

    evals, evecs = _jacobi_eig3(M)
    kmin = min(range(3), key=lambda k: evals[k])
    w = evecs[kmin]
    wl = math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)

    if wl < 1e-12:
        return None

    w = (w[0] / wl, w[1] / wl, w[2] / wl)
    ea, eb = _ortho_basis(w)
    p0 = pts[0]
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    proj = []

    for p in pts:
        dp = (p[0] - p0[0], p[1] - p0[1], p[2] - p0[2])
        x = dp[0] * ea[0] + dp[1] * ea[1] + dp[2] * ea[2]
        y = dp[0] * eb[0] + dp[1] * eb[1] + dp[2] * eb[2]
        proj.append((x, y))
        row = [x, y, 1.0]
        rhs = -(x * x + y * y)

        for r in range(3):
            atb[r] += row[r] * rhs

            for c in range(3):
                ata[r][c] += row[r] * row[c]

    sol = _solve_gauss(ata, atb, 3)

    if sol is None:
        return None

    ccx, ccy = -sol[0] / 2.0, -sol[1] / 2.0
    r2 = ccx * ccx + ccy * ccy - sol[2]

    if r2 <= 1e-18:
        return None

    r = math.sqrt(r2)

    for x, y in proj:
        if abs(math.sqrt((x - ccx) ** 2 + (y - ccy) ** 2) - r) > tol:
            return None

    axis_pt = (
        p0[0] + ccx * ea[0] + ccy * eb[0],
        p0[1] + ccx * ea[1] + ccy * eb[1],
        p0[2] + ccx * ea[2] + ccy * eb[2],
    )

    return (axis_pt, w, r)


def _fit_cone(surface, tol):
    """Recognize a cone from surface samples: apex, axis and half angle."""

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    pts = []
    nrm = []
    nu_s = 8

    for i in range(nu_s):
        uu = u0 + (u1 - u0) * i / nu_s

        for j in range(5):
            vv = v0 + (v1 - v0) * j / 4.0
            pts.append(surface.point_at(uu, vv))
            n = surface.normal_at(uu, vv)
            nl = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)

            if nl < 1e-12:
                continue

            nrm.append(((n[0] / nl, n[1] / nl, n[2] / nl), surface.point_at(uu, vv)))

    if len(nrm) < 4:
        return None

    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3

    for n, p in nrm:
        npd = n[0] * p[0] + n[1] * p[1] + n[2] * p[2]

        for r in range(3):
            atb[r] += n[r] * npd

            for c in range(3):
                ata[r][c] += n[r] * n[c]

    V = _solve_gauss(ata, atb, 3)

    if V is None:
        return None

    gs = []

    for p in pts:
        d = (p[0] - V[0], p[1] - V[1], p[2] - V[2])
        dl = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

        if dl < tol:
            continue

        gs.append((d[0] / dl, d[1] / dl, d[2] / dl))

    if len(gs) < 3:
        return None

    G = [[0.0] * 3 for _ in range(3)]

    for g in gs:
        for r in range(3):
            for c in range(3):
                G[r][c] += g[r] * g[c]

    gevals, gevecs = _jacobi_eig3(G)
    kmax = max(range(3), key=lambda k: gevals[k])
    w = gevecs[kmax]
    sx = (sum(g[0] for g in gs), sum(g[1] for g in gs), sum(g[2] for g in gs))

    if w[0] * sx[0] + w[1] * sx[1] + w[2] * sx[2] < 0.0:
        w = (-w[0], -w[1], -w[2])

    wl = math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)

    if wl < 1e-12:
        return None

    w = (w[0] / wl, w[1] / wl, w[2] / wl)
    angs = [
        math.acos(max(-1.0, min(1.0, g[0] * w[0] + g[1] * w[1] + g[2] * w[2])))
        for g in gs
    ]
    alpha = sum(angs) / len(angs)

    if alpha < 1e-4 or alpha > math.pi / 2 - 1e-4:
        return None

    ca = math.cos(alpha)

    for p in pts:
        d = (p[0] - V[0], p[1] - V[1], p[2] - V[2])
        axd = d[0] * w[0] + d[1] * w[1] + d[2] * w[2]
        perp = math.sqrt(max(0.0, (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) - axd * axd))

        if abs(perp - axd * math.tan(alpha)) * ca > tol:
            return None

    return ((V[0], V[1], V[2]), w, alpha)


def _fit_sphere(surface, tol):
    """Recognize a sphere from surface samples: center and radius."""

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    pts = []

    for i in range(5):
        for j in range(5):
            uu = u0 + (u1 - u0) * i / 4.0
            vv = v0 + (v1 - v0) * j / 4.0
            pts.append(surface.point_at(uu, vv))

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

    cx, cy, cz = -sol[0] / 2.0, -sol[1] / 2.0, -sol[2] / 2.0
    r2 = cx * cx + cy * cy + cz * cz - sol[3]

    if r2 <= 0.0:
        return None

    r = math.sqrt(r2)

    for p in pts:
        d = math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2)

        if abs(d - r) > tol:
            return None

    return (cx, cy, cz, r)


def _fit_torus(surface, tol):
    """Recognize a torus from the smallest-variance axis and a tube cross-section circle fit."""

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    pts = []

    for i in range(8):
        for j in range(8):
            pts.append(
                surface.point_at(u0 + (u1 - u0) * i / 8.0, v0 + (v1 - v0) * j / 8.0)
            )

    n = len(pts)
    cen = [sum(p[k] for p in pts) / n for k in range(3)]
    M = [[0.0] * 3 for _ in range(3)]

    for p in pts:
        d = (p[0] - cen[0], p[1] - cen[1], p[2] - cen[2])

        for r in range(3):
            for c in range(3):
                M[r][c] += d[r] * d[c]

    evals, evecs = _jacobi_eig3(M)
    kmin = min(range(3), key=lambda k: evals[k])
    w = evecs[kmin]
    wl = math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)

    if wl < 1e-12:
        return None

    w = (w[0] / wl, w[1] / wl, w[2] / wl)
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    rhoa = []

    for p in pts:
        d = (p[0] - cen[0], p[1] - cen[1], p[2] - cen[2])
        a = d[0] * w[0] + d[1] * w[1] + d[2] * w[2]
        perp = (d[0] - a * w[0], d[1] - a * w[1], d[2] - a * w[2])
        rho = math.sqrt(perp[0] ** 2 + perp[1] ** 2 + perp[2] ** 2)
        rhoa.append((rho, a))
        row = [rho, a, 1.0]
        rhs = -(rho * rho + a * a)

        for r in range(3):
            atb[r] += row[r] * rhs

            for c in range(3):
                ata[r][c] += row[r] * row[c]

    sol = _solve_gauss(ata, atb, 3)

    if sol is None:
        return None

    R = -sol[0] / 2.0
    a0 = -sol[1] / 2.0
    r2 = R * R + a0 * a0 - sol[2]

    if r2 <= 1e-18 or R <= 0.0:
        return None

    r = math.sqrt(r2)

    if R <= r * 0.5:
        return None

    for rho, a in rhoa:
        if abs(math.sqrt((rho - R) ** 2 + (a - a0) ** 2) - r) > tol:
            return None

    center = (cen[0] + a0 * w[0], cen[1] + a0 * w[1], cen[2] + a0 * w[2])

    return (center, w, R, r)


def _recognize_surface(surface, tol):
    """Classify a surface as plane, cylinder, cone, sphere or torus within tol."""

    if surface.is_planar(None, tol):
        u0, u1 = surface.domain(0)
        v0, v1 = surface.domain(1)
        o = surface.point_at((u0 + u1) * 0.5, (v0 + v1) * 0.5)
        n = surface.normal_at((u0 + u1) * 0.5, (v0 + v1) * 0.5)

        return ("plane", (o[0], o[1], o[2]), (n[0], n[1], n[2]))

    sph = _fit_sphere(surface, tol)

    if sph is not None:
        return ("sphere", (sph[0], sph[1], sph[2]), sph[3])

    cyl = _fit_cylinder(surface, tol)

    if cyl is not None:
        return ("cylinder", cyl[0], cyl[1], cyl[2])

    cone = _fit_cone(surface, tol)

    if cone is not None:
        return ("cone", cone[0], cone[1], cone[2])

    tor = _fit_torus(surface, tol)

    if tor is not None:
        return ("torus", tor[0], tor[1], tor[2], tor[3])

    return None


def _analytic_pcurve(srf, recog, c3d):
    """Analytic pcurve of an exact 3D intersection conic on a recognized quadric surface."""

    if recog is None:
        return None

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)

    def dot(p, q):
        return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]

    if recog[0] == "cylinder":
        ap = recog[1]
        ax = recog[2]
        an = math.sqrt(dot(ax, ax))

        if an < 1e-12:
            return None

        ax = (ax[0] / an, ax[1] / an, ax[2] / an)

        def height(p):
            return (
                (p[0] - ap[0]) * ax[0] + (p[1] - ap[1]) * ax[1] + (p[2] - ap[2]) * ax[2]
            )

        um = 0.5 * (u0 + u1)
        h0 = height(srf.point_at(um, v0))
        h1 = height(srf.point_at(um, v1))

        if abs(h1 - h0) < 1e-12:
            return None

        hmin = 1e300
        hmax = -1e300
        hsum = 0.0
        ns = 0
        t0, t1 = c3d.domain()

        for i in range(33):
            h = height(c3d.point_at(t0 + (t1 - t0) * i / 32))
            hmin = min(hmin, h)
            hmax = max(hmax, h)
            hsum += h
            ns += 1

        if hmax - hmin > 1e-5 * abs(h1 - h0):
            return None

        if c3d.point_at(t0).distance(c3d.point_at(t1)) > 1e-6 * (abs(h1 - h0) + 1.0):
            return None

        hc = hsum / ns
        vc = v0 + (hc - h0) / (h1 - h0) * (v1 - v0)

        if vc < min(v0, v1) - 1e-9 or vc > max(v0, v1) + 1e-9:
            return None

        return NurbsCurve.create(
            False,
            1,
            [
                Point(u0, vc, 0.0),
                Point(u1, vc, 0.0),
            ],
        )

    if recog[0] == "cone":
        ax = recog[2]
        an = math.sqrt(dot(ax, ax))

        if an < 1e-12:
            return None

        ax = (ax[0] / an, ax[1] / an, ax[2] / an)
        A = recog[1]

        def height(p):
            return (p[0] - A[0]) * ax[0] + (p[1] - A[1]) * ax[1] + (p[2] - A[2]) * ax[2]

        t0, t1 = c3d.domain()
        clen = c3d.point_at(t0).distance(c3d.point_at(0.5 * (t0 + t1)))
        hscale = max(clen, 1e-9)
        hmin = 1e300
        hmax = -1e300
        hsum = 0.0
        ns = 0

        for i in range(33):
            h = height(c3d.point_at(t0 + (t1 - t0) * i / 32))
            hmin = min(hmin, h)
            hmax = max(hmax, h)
            hsum += h
            ns += 1

        if hmax - hmin > hscale * 1e-4:
            return None

        if c3d.point_at(t0).distance(c3d.point_at(t1)) > hscale * 1e-3:
            return None

        hc = hsum / ns
        um2 = 0.5 * (u0 + u1)
        va = v0
        vb = v1
        ha = height(srf.point_at(um2, va))
        hb = height(srf.point_at(um2, vb))

        if (hc - ha) * (hc - hb) > 0:
            return None

        for _ in range(60):
            vmid = 0.5 * (va + vb)
            hm = height(srf.point_at(um2, vmid))

            if (hm - hc) * (ha - hc) <= 0:
                vb = vmid
            else:
                va = vmid
                ha = hm

        vc = 0.5 * (va + vb)

        return NurbsCurve.create(
            False,
            1,
            [
                Point(u0, vc, 0.0),
                Point(u1, vc, 0.0),
            ],
        )

    return None


def _emit_pullback_curve(nodes):
    """Degree-1 UV polyline through pull-back nodes."""

    pts = []

    for node in nodes:
        pts.append(Point(node[0], node[1], 0.0))

    return NurbsCurve.create(False, 1, pts)


def _analytic_sphere_pullback(srf, recog, c3d):
    """Pull a 3D curve back to sphere parameters through longitude and latitude."""

    if recog is None or recog[0] != "sphere":
        return []

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    range_u = u1 - u0

    if range_u < 1e-9:
        return []

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    C = recog[1]
    um = 0.5 * (u0 + u1)
    vm = 0.5 * (v0 + v1)
    sp = srf.point_at(um, v0)
    np = srf.point_at(um, v1)
    Zs = [np[0] - sp[0], np[1] - sp[1], np[2] - sp[2]]
    zn = math.sqrt(dot(Zs, Zs))

    if zn < 1e-12:
        return []

    Zs = [Zs[0] / zn, Zs[1] / zn, Zs[2] / zn]
    P0 = srf.point_at(u0, vm)
    x0 = [P0[0] - C[0], P0[1] - C[1], P0[2] - C[2]]
    h0 = dot(x0, Zs)
    Xs = [x0[0] - h0 * Zs[0], x0[1] - h0 * Zs[1], x0[2] - h0 * Zs[2]]
    xn = math.sqrt(dot(Xs, Xs))

    if xn < 1e-12:
        return []

    Xs = [Xs[0] / xn, Xs[1] / xn, Xs[2] / xn]
    Ys = [
        Zs[1] * Xs[2] - Zs[2] * Xs[1],
        Zs[2] * Xs[0] - Zs[0] * Xs[2],
        Zs[0] * Xs[1] - Zs[1] * Xs[0],
    ]
    PI = math.pi
    TWO_PI = 2.0 * PI
    NT = 128
    tu = [0.0] * (NT + 1)
    tlon = [0.0] * (NT + 1)

    for k in range(NT + 1):
        u = u0 + range_u * k / NT
        p = srf.point_at(u, vm)
        r = [p[0] - C[0], p[1] - C[1], p[2] - C[2]]
        lon = math.atan2(dot(r, Ys), dot(r, Xs))

        if k > 0:
            while lon - tlon[k - 1] > PI:
                lon -= TWO_PI

            while lon - tlon[k - 1] < -PI:
                lon += TWO_PI

        tu[k] = u
        tlon[k] = lon

    lon_incr = tlon[NT] >= tlon[0]
    lon_lo = min(tlon[0], tlon[NT])
    lon_hi = max(tlon[0], tlon[NT])

    def u_from_lon(lon):
        while lon < lon_lo - 1e-9:
            lon += TWO_PI

        while lon > lon_hi + 1e-9:
            lon -= TWO_PI

        lo = 0
        hi = NT

        while hi - lo > 1:
            mid = (lo + hi) // 2
            above = (tlon[mid] < lon) if lon_incr else (tlon[mid] > lon)

            if above:
                lo = mid
            else:
                hi = mid

        denom = tlon[hi] - tlon[lo]
        f = (lon - tlon[lo]) / denom if abs(denom) > 1e-15 else 0.0

        return tu[lo] + (tu[hi] - tu[lo]) * f

    tv = [0.0] * (NT + 1)
    th = [0.0] * (NT + 1)

    for k in range(NT + 1):
        v = v0 + (v1 - v0) * k / NT
        p = srf.point_at(um, v)
        r = [p[0] - C[0], p[1] - C[1], p[2] - C[2]]
        tv[k] = v
        th[k] = dot(r, Zs)

    incr = th[NT] >= th[0]

    if abs(th[NT] - th[0]) < 1e-12:
        return []

    def v_from_height(h):
        if incr:
            if h <= th[0]:
                return tv[0]

            if h >= th[NT]:
                return tv[NT]
        else:
            if h >= th[0]:
                return tv[0]

            if h <= th[NT]:
                return tv[NT]

        lo = 0
        hi = NT

        while hi - lo > 1:
            mid = (lo + hi) // 2
            above = (th[mid] < h) if incr else (th[mid] > h)

            if above:
                lo = mid
            else:
                hi = mid

        denom = th[hi] - th[lo]
        f = (h - th[lo]) / denom if abs(denom) > 1e-15 else 0.0

        return tv[lo] + (tv[hi] - tv[lo]) * f

    t0, t1 = c3d.domain()

    def project_t(t):
        p = c3d.point_at(t)
        r = [p[0] - C[0], p[1] - C[1], p[2] - C[2]]
        lon = math.atan2(dot(r, Ys), dot(r, Xs))
        h = dot(r, Zs)
        u = u_from_lon(lon)

        for _ in range(2):
            du_ = range_u * 1e-7
            uc = min(max(u, u0), u1)
            pc0 = srf.point_at(uc, vm)
            rc0 = [pc0[0] - C[0], pc0[1] - C[1], pc0[2] - C[2]]
            g0 = math.atan2(dot(rc0, Ys), dot(rc0, Xs)) - lon

            while g0 > PI:
                g0 -= TWO_PI

            while g0 < -PI:
                g0 += TWO_PI

            pc1 = srf.point_at(min(uc + du_, u1), vm)
            rc1 = [pc1[0] - C[0], pc1[1] - C[1], pc1[2] - C[2]]
            g1 = math.atan2(dot(rc1, Ys), dot(rc1, Xs)) - lon

            while g1 > PI:
                g1 -= TWO_PI

            while g1 < -PI:
                g1 += TWO_PI

            dg = (g1 - g0) / du_

            if abs(dg) < 1e-12:
                break

            u = min(max(uc - g0 / dg, u0), u1)

        v = v_from_height(h)

        for _ in range(2):
            dv_ = (v1 - v0) * 1e-7
            vc2 = min(max(v, min(v0, v1)), max(v0, v1))
            qc0 = srf.point_at(um, vc2)
            g0 = (
                (qc0[0] - C[0]) * Zs[0]
                + (qc0[1] - C[1]) * Zs[1]
                + (qc0[2] - C[2]) * Zs[2]
                - h
            )
            qc1 = srf.point_at(um, min(vc2 + dv_, max(v0, v1)))
            g1 = (
                (qc1[0] - C[0]) * Zs[0]
                + (qc1[1] - C[1]) * Zs[1]
                + (qc1[2] - C[2]) * Zs[2]
                - h
            )
            dg = (g1 - g0) / dv_

            if abs(dg) < 1e-12:
                break

            v = min(max(vc2 - g0 / dg, min(v0, v1)), max(v0, v1))

        return u, v

    n = max(c3d.cv_count() * 8, 120)
    uv = []
    prev_u = 0.0

    for i in range(n + 1):
        t = t0 + (t1 - t0) * i / n
        u, v = project_t(t)

        if i > 0:
            while u - prev_u > range_u * 0.5:
                u -= range_u

            while u - prev_u < -range_u * 0.5:
                u += range_u

        prev_u = u
        uv.append((u, v))

    if len(uv) < 2:
        return []

    out = []
    seg = []

    def kof(u):
        return math.floor((u - u0) / range_u + 1e-9)

    cur_k = kof(uv[0][0])
    seg.append((uv[0][0] - cur_k * range_u, uv[0][1]))

    for i in range(1, len(uv)):
        ki = kof(uv[i][0])

        while ki != cur_k:
            step = 1 if ki > cur_k else -1
            nk = cur_k + step
            seam_cont = u0 + (nk if step > 0 else cur_k) * range_u
            denom = uv[i][0] - uv[i - 1][0]
            f = (seam_cont - uv[i - 1][0]) / denom if abs(denom) > 1e-15 else 0.0
            f = min(max(f, 0.0), 1.0)
            vc = uv[i - 1][1] + (uv[i][1] - uv[i - 1][1]) * f
            seg.append((seam_cont - cur_k * range_u, vc))

            if len(seg) >= 2:
                out.append(_emit_pullback_curve(seg))

            seg = []
            seg.append((seam_cont - nk * range_u, vc))
            cur_k = nk

        seg.append((uv[i][0] - cur_k * range_u, uv[i][1]))

    if len(seg) >= 2:
        out.append(_emit_pullback_curve(seg))

    return out


def _analytic_cone_pullback(srf, recog, c3d):
    """Analytic pull-back of a 3D curve onto a recognized cone or cylinder."""

    if recog is None or (recog[0] != "cone" and recog[0] != "cylinder"):
        return []

    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    range_u = u1 - u0

    if range_u < 1e-9:
        return []

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    A = recog[1]
    Zc = [recog[2][0], recog[2][1], recog[2][2]]
    zn = math.sqrt(dot(Zc, Zc))

    if zn < 1e-12:
        return []

    Zc = [Zc[0] / zn, Zc[1] / zn, Zc[2] / zn]

    def height(p):
        r = [p[0] - A[0], p[1] - A[1], p[2] - A[2]]

        return dot(r, Zc)

    um = 0.5 * (u0 + u1)
    h0 = height(srf.point_at(um, v0))
    h1 = height(srf.point_at(um, v1))

    if abs(h1 - h0) < 1e-12:
        return []

    def v_from_height(h):
        return v0 + (h - h0) / (h1 - h0) * (v1 - v0)

    v_ref = v0 if abs(h0) >= abs(h1) else v1
    P0 = srf.point_at(u0, v_ref)
    x0 = [P0[0] - A[0], P0[1] - A[1], P0[2] - A[2]]
    hp = dot(x0, Zc)
    Xc = [x0[0] - hp * Zc[0], x0[1] - hp * Zc[1], x0[2] - hp * Zc[2]]
    xn = math.sqrt(dot(Xc, Xc))

    if xn < 1e-12:
        return []

    Xc = [Xc[0] / xn, Xc[1] / xn, Xc[2] / xn]
    Yc = [
        Zc[1] * Xc[2] - Zc[2] * Xc[1],
        Zc[2] * Xc[0] - Zc[0] * Xc[2],
        Zc[0] * Xc[1] - Zc[1] * Xc[0],
    ]
    PI = math.pi
    TWO_PI = 2.0 * PI
    NT = 128
    tu = [0.0] * (NT + 1)
    tlon = [0.0] * (NT + 1)

    for k in range(NT + 1):
        u = u0 + range_u * k / NT
        p = srf.point_at(u, v_ref)
        r = [p[0] - A[0], p[1] - A[1], p[2] - A[2]]
        lon = math.atan2(dot(r, Yc), dot(r, Xc))

        if k > 0:
            while lon - tlon[k - 1] > PI:
                lon -= TWO_PI

            while lon - tlon[k - 1] < -PI:
                lon += TWO_PI

        tu[k] = u
        tlon[k] = lon

    lon_incr = tlon[NT] >= tlon[0]
    lon_lo = min(tlon[0], tlon[NT])
    lon_hi = max(tlon[0], tlon[NT])

    def u_from_lon(lon):
        while lon < lon_lo - 1e-9:
            lon += TWO_PI

        while lon > lon_hi + 1e-9:
            lon -= TWO_PI

        lo = 0
        hi = NT

        while hi - lo > 1:
            mid = (lo + hi) // 2
            above = (tlon[mid] < lon) if lon_incr else (tlon[mid] > lon)

            if above:
                lo = mid
            else:
                hi = mid

        denom = tlon[hi] - tlon[lo]
        f = (lon - tlon[lo]) / denom if abs(denom) > 1e-15 else 0.0

        return tu[lo] + (tu[hi] - tu[lo]) * f

    t0, t1 = c3d.domain()
    n = max(c3d.cv_count() * 8, 120)
    prev_lon_s = [0.0]

    def project_t(tq):
        p = c3d.point_at(tq)
        r = [p[0] - A[0], p[1] - A[1], p[2] - A[2]]
        rad = math.sqrt(max(0.0, dot(r, Xc) * dot(r, Xc) + dot(r, Yc) * dot(r, Yc)))
        lon = math.atan2(dot(r, Yc), dot(r, Xc)) if rad > 1e-12 else prev_lon_s[0]
        prev_lon_s[0] = lon
        u = u_from_lon(lon)

        if rad > 1e-12:
            for _ in range(2):
                du_ = range_u * 1e-7
                uc = min(max(u, u0), u1)
                pc0 = srf.point_at(uc, v_ref)
                rc0 = [pc0[0] - A[0], pc0[1] - A[1], pc0[2] - A[2]]
                g0 = math.atan2(dot(rc0, Yc), dot(rc0, Xc)) - lon

                while g0 > PI:
                    g0 -= TWO_PI

                while g0 < -PI:
                    g0 += TWO_PI

                pc1 = srf.point_at(min(uc + du_, u1), v_ref)
                rc1 = [pc1[0] - A[0], pc1[1] - A[1], pc1[2] - A[2]]
                g1 = math.atan2(dot(rc1, Yc), dot(rc1, Xc)) - lon

                while g1 > PI:
                    g1 -= TWO_PI

                while g1 < -PI:
                    g1 += TWO_PI

                dg = (g1 - g0) / du_

                if abs(dg) < 1e-12:
                    break

                u = min(max(uc - g0 / dg, u0), u1)

        return u, v_from_height(dot(r, Zc))

    uv = []
    prev_u = 0.0

    for i in range(n + 1):
        tq = t0 + (t1 - t0) * i / n
        u, v = project_t(tq)

        if i > 0:
            while u - prev_u > range_u * 0.5:
                u -= range_u

            while u - prev_u < -range_u * 0.5:
                u += range_u

        prev_u = u
        uv.append((u, v))

    if len(uv) < 2:
        return []

    out = []
    seg = []

    def kof(u):
        return math.floor((u - u0) / range_u + 1e-9)

    cur_k = kof(uv[0][0])
    seg.append((uv[0][0] - cur_k * range_u, uv[0][1]))

    for i in range(1, len(uv)):
        ki = kof(uv[i][0])

        while ki != cur_k:
            step = 1 if ki > cur_k else -1
            nk = cur_k + step
            seam_cont = u0 + (nk if step > 0 else cur_k) * range_u
            denom = uv[i][0] - uv[i - 1][0]
            f = (seam_cont - uv[i - 1][0]) / denom if abs(denom) > 1e-15 else 0.0
            f = min(max(f, 0.0), 1.0)
            vc = uv[i - 1][1] + (uv[i][1] - uv[i - 1][1]) * f
            seg.append((seam_cont - cur_k * range_u, vc))

            if len(seg) >= 2:
                out.append(_emit_pullback_curve(seg))

            seg = []
            seg.append((seam_cont - nk * range_u, vc))
            cur_k = nk

        seg.append((uv[i][0] - cur_k * range_u, uv[i][1]))

    if len(seg) >= 2:
        out.append(_emit_pullback_curve(seg))

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Coaxial quadric pairs
# ═══════════════════════════════════════════════════════════════════════════


def _point_axis_dist(apt, adir, P):
    """Distance of P from the axis through apt along adir."""

    u = _ssi_unit(adir)
    dp = (P[0] - apt[0], P[1] - apt[1], P[2] - apt[2])
    t = _ssi_dot(dp, u)
    perp = (dp[0] - t * u[0], dp[1] - t * u[1], dp[2] - t * u[2])

    return math.sqrt(_ssi_dot(perp, perp))


def _axial_coord(apt, adir, P):
    """Coordinate of P along the axis through apt along adir."""
    u = _ssi_unit(adir)

    return (P[0] - apt[0]) * u[0] + (P[1] - apt[1]) * u[1] + (P[2] - apt[2]) * u[2]


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
    """Closest point of two lines, false when parallel."""

    u = _ssi_unit(d1)
    v = _ssi_unit(d2)
    w0 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
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
    q1 = (p1[0] + sc * u[0], p1[1] + sc * u[1], p1[2] + sc * u[2])
    q2 = (p2[0] + tc * v[0], p2[1] + tc * v[1], p2[2] + tc * v[2])
    diff = (q1[0] - q2[0], q1[1] - q2[1], q1[2] - q2[2])

    if math.sqrt(_ssi_dot(diff, diff)) > tol:
        return None

    return (0.5 * (q1[0] + q2[0]), 0.5 * (q1[1] + q2[1]), 0.5 * (q1[2] + q2[2]))


def _ssi_cylinder_sphere(cyl, sph):
    """Coaxial cylinder-sphere section: circles."""

    kTol = 1e-6
    P = cyl[1]
    w = _ssi_unit(cyl[2])
    rc = cyl[3]
    C = sph[1]
    R = sph[2]

    if _point_axis_dist(P, w, C) > kTol:
        return None

    out = []

    if R < rc - kTol:
        return out

    dist = math.sqrt(max(0.0, R * R - rc * rc))
    xa, ya = _ortho_basis(w)

    if dist <= kTol:
        out.append(_exact_circle(C[0], C[1], C[2], xa, ya, rc))

        return out

    for s in (dist, -dist):
        cc = (C[0] + s * w[0], C[1] + s * w[1], C[2] + s * w[2])
        out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rc))

    return out


def _ssi_cylinder_cone(cyl, cone):
    """Coaxial cylinder-cone section: circles."""

    kTol = 1e-6
    Pc = cyl[1]
    w = _ssi_unit(cyl[2])
    rc = cyl[3]
    apex = cone[1]
    a = _ssi_unit(cone[2])
    alpha = cone[3]

    if not _axes_coaxial(Pc, w, apex, a, kTol):
        return None

    ta = math.tan(alpha)

    if ta < 1e-9:
        return None

    s = rc / ta
    out = []

    if s < kTol:
        return out

    cc = (apex[0] + s * a[0], apex[1] + s * a[1], apex[2] + s * a[2])
    xa, ya = _ortho_basis(a)
    out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rc))

    return out


def _ssi_cone_sphere(cone, sph):
    """Coaxial cone-sphere section: circles."""

    kTol = 1e-6
    apex = cone[1]
    a = _ssi_unit(cone[2])
    alpha = cone[3]
    C = sph[1]
    R = sph[2]

    if _point_axis_dist(apex, a, C) > kTol:
        return None

    dsign = _axial_coord(apex, a, C)
    d = abs(dsign)
    dir = (-a[0], -a[1], -a[2]) if (d > kTol and dsign < 0.0) else a
    t = math.tan(alpha)
    t2 = t * t
    A = 1.0 + t2
    B = 2.0 * t2 * d
    Cq = t2 * d * d - R * R
    disc = B * B - 4.0 * A * Cq
    out = []

    if disc < -kTol:
        return out

    sq = math.sqrt(max(0.0, disc))

    if sq <= kTol:
        xs = [-B / (2.0 * A)]
    else:
        xs = [(-B - sq) / (2.0 * A), (-B + sq) / (2.0 * A)]

    xa, ya = _ortho_basis(a)

    for x in xs:
        sAx = d + x

        if sAx < kTol:
            continue

        rr = t * sAx

        if rr < kTol:
            continue

        cc = (apex[0] + sAx * dir[0], apex[1] + sAx * dir[1], apex[2] + sAx * dir[2])
        out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rr))

    return out


def _ssi_cylinder_cylinder(sa, A, sb, B):
    """Cylinder-cylinder section: circles when coaxial, Steinmetz curves when the axes meet."""

    kTol = 1e-6
    P1 = A[1]
    w1 = _ssi_unit(A[2])
    R1 = A[3]
    P2 = B[1]
    w2 = _ssi_unit(B[2])
    R2 = B[3]
    cx = _ssi_cross(w1, w2)
    sinmag = math.sqrt(_ssi_dot(cx, cx))
    out = []

    if sinmag <= kTol:
        dline = _point_axis_dist(P1, w1, P2)

        if dline <= kTol:
            if abs(R1 - R2) <= kTol:
                return None

            return out

        off = _ssi_dot((P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]), w1)
        P2p = (P2[0] - off * w1[0], P2[1] - off * w1[1], P2[2] - off * w1[2])
        d = dline

        if d > R1 + R2 + kTol:
            return out

        if d < abs(R1 - R2) - kTol:
            return out

        xdir = _ssi_unit((P2p[0] - P1[0], P2p[1] - P1[1], P2p[2] - P1[2]))
        ydir = _ssi_unit(_ssi_cross(w1, xdir))
        aa = (R1 * R1 - R2 * R2 + d * d) / (2.0 * d)
        h = math.sqrt(max(0.0, R1 * R1 - aa * aa))
        foot = (P1[0] + aa * xdir[0], P1[1] + aa * xdir[1], P1[2] + aa * xdir[2])
        s0a, s1a = _cyl_span(sa, P1, w1)
        s0b, s1b = _cyl_span(sb, P1, w1)
        slo = max(s0a, s0b)
        shi = min(s1a, s1b)

        if shi - slo <= kTol:
            return out

        def emit(bp):
            e0 = (bp[0] + slo * w1[0], bp[1] + slo * w1[1], bp[2] + slo * w1[2])
            e1 = (bp[0] + shi * w1[0], bp[1] + shi * w1[1], bp[2] + shi * w1[2])
            ln = NurbsCurve.create(
                False,
                1,
                [
                    Point(e0[0], e0[1], e0[2]),
                    Point(e1[0], e1[1], e1[2]),
                ],
            )
            ln.set_domain(0.0, 1.0)
            out.append(ln)

        if h <= kTol:
            emit(foot)
        else:
            emit((foot[0] + h * ydir[0], foot[1] + h * ydir[1], foot[2] + h * ydir[2]))
            emit((foot[0] - h * ydir[0], foot[1] - h * ydir[1], foot[2] - h * ydir[2]))

        return out

    Rmax = max(R1, R2)

    if Rmax < 1e-12 or abs(R1 - R2) / Rmax > 1e-6:
        return None

    Pint = _lines_closest_point(P1, w1, P2, w2, kTol)

    if Pint is None:
        return None

    R = 0.5 * (R1 + R2)
    ang = math.acos(max(-1.0, min(1.0, _ssi_dot(w1, w2))))
    sh = math.sin(0.5 * ang)
    ch = math.cos(0.5 * ang)

    if sh < 1e-9 or ch < 1e-9:
        return None

    minor = _ssi_unit(cx)
    maj1 = _ssi_unit((w1[0] + w2[0], w1[1] + w2[1], w1[2] + w2[2]))
    maj2 = _ssi_unit((w1[0] - w2[0], w1[1] - w2[1], w1[2] - w2[2]))
    out.append(_exact_ellipse(Pint[0], Pint[1], Pint[2], maj1, minor, R / sh, R))
    out.append(_exact_ellipse(Pint[0], Pint[1], Pint[2], maj2, minor, R / ch, R))

    return out


def _analytic_ssi(a, b, tolerance):
    """Exact section of two recognized analytic surfaces, empty when no case applies."""

    ra = _recognize_surface(a, max(tolerance, 1e-7) * 1e4)
    rb = _recognize_surface(b, max(tolerance, 1e-7) * 1e4)

    if ra is None or rb is None:
        return None

    def unit(v):
        length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

        return (v[0] / length, v[1] / length, v[2] / length) if length > 1e-300 else v

    def cross(u, v):
        return (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )

    def plane_sphere(plane, sph):
        o, nu = plane[1], unit(plane[2])
        c, r = sph[1], sph[2]
        d = (c[0] - o[0]) * nu[0] + (c[1] - o[1]) * nu[1] + (c[2] - o[2]) * nu[2]

        if abs(d) >= r:
            return None

        cc = (c[0] - d * nu[0], c[1] - d * nu[1], c[2] - d * nu[2])
        rr = math.sqrt(r * r - d * d)
        xa, ya = _ortho_basis(nu)

        return _exact_circle(cc[0], cc[1], cc[2], xa, ya, rr)

    def plane_cylinder(plane, cyl):
        o, nu = plane[1], unit(plane[2])
        P, w, r = cyl[1], unit(cyl[2]), cyl[3]
        wn = w[0] * nu[0] + w[1] * nu[1] + w[2] * nu[2]

        if abs(wn) < 1e-7:
            return None

        t = ((o[0] - P[0]) * nu[0] + (o[1] - P[1]) * nu[1] + (o[2] - P[2]) * nu[2]) / wn
        cc = (P[0] + t * w[0], P[1] + t * w[1], P[2] + t * w[2])
        mraw = cross(w, nu)

        if math.sqrt(mraw[0] ** 2 + mraw[1] ** 2 + mraw[2] ** 2) < 1e-9:
            xa, ya = _ortho_basis(nu)

            return _exact_circle(cc[0], cc[1], cc[2], xa, ya, r)

        minor = unit(mraw)
        major = unit((w[0] - wn * nu[0], w[1] - wn * nu[1], w[2] - wn * nu[2]))

        return _exact_ellipse(cc[0], cc[1], cc[2], major, minor, r / abs(wn), r)

    def line_cone(x0, d, V, w, alpha):
        """Solve ((X-V).w)^2 - cos^2a |X-V|^2 = 0 along X = x0 + t d. Returns roots."""

        ca2 = math.cos(alpha) ** 2
        e = (x0[0] - V[0], x0[1] - V[1], x0[2] - V[2])
        A = e[0] * w[0] + e[1] * w[1] + e[2] * w[2]
        B = d[0] * w[0] + d[1] * w[1] + d[2] * w[2]
        C = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
        D = e[0] * d[0] + e[1] * d[1] + e[2] * d[2]
        E = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        qa = B * B - ca2 * E
        qb = 2.0 * A * B - 2.0 * ca2 * D
        qc = A * A - ca2 * C

        if abs(qa) < 1e-14:
            return [] if abs(qb) < 1e-300 else [-qc / qb]

        disc = qb * qb - 4.0 * qa * qc

        if disc < 0.0:
            return []

        sq = math.sqrt(disc)

        return [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)]

    def plane_cone(plane, cone):
        o, nu = plane[1], unit(plane[2])
        V, w, alpha = cone[1], unit(cone[2]), cone[3]
        wn = w[0] * nu[0] + w[1] * nu[1] + w[2] * nu[2]

        if abs(abs(wn) - 1.0) < 1e-9:
            dax = (o[0] - V[0]) * w[0] + (o[1] - V[1]) * w[1] + (o[2] - V[2]) * w[2]
            rr = abs(dax) * math.tan(alpha)
            cc = (V[0] + dax * w[0], V[1] + dax * w[1], V[2] + dax * w[2])

            if rr < 1e-12:
                return None

            xa, ya = _ortho_basis(nu)

            return _exact_circle(cc[0], cc[1], cc[2], xa, ya, rr)

        m = cross(w, nu)
        ml = math.sqrt(m[0] ** 2 + m[1] ** 2 + m[2] ** 2)

        if ml < 1e-12:
            return None

        m = (m[0] / ml, m[1] / ml, m[2] / ml)
        major = unit((w[0] - wn * nu[0], w[1] - wn * nu[1], w[2] - wn * nu[2]))
        dV = (V[0] - o[0]) * nu[0] + (V[1] - o[1]) * nu[1] + (V[2] - o[2]) * nu[2]
        Vp = (V[0] - dV * nu[0], V[1] - dV * nu[1], V[2] - dV * nu[2])
        ts = line_cone(Vp, major, V, w, alpha)

        if len(ts) != 2:
            return None

        A = (
            Vp[0] + ts[0] * major[0],
            Vp[1] + ts[0] * major[1],
            Vp[2] + ts[0] * major[2],
        )
        Bp = (
            Vp[0] + ts[1] * major[0],
            Vp[1] + ts[1] * major[1],
            Vp[2] + ts[1] * major[2],
        )
        cc = ((A[0] + Bp[0]) * 0.5, (A[1] + Bp[1]) * 0.5, (A[2] + Bp[2]) * 0.5)
        semi_major = 0.5 * math.sqrt(
            (Bp[0] - A[0]) ** 2 + (Bp[1] - A[1]) ** 2 + (Bp[2] - A[2]) ** 2
        )
        major = unit((Bp[0] - A[0], Bp[1] - A[1], Bp[2] - A[2]))
        tm = line_cone(cc, m, V, w, alpha)

        if len(tm) != 2:
            return None

        semi_minor = 0.5 * abs(tm[1] - tm[0])

        if semi_major < 1e-12 or semi_minor < 1e-12:
            return None

        return _exact_ellipse(cc[0], cc[1], cc[2], major, m, semi_major, semi_minor)

    def plane_torus(plane, tor):
        o, nu = plane[1], unit(plane[2])
        C, w, R, r = tor[1], unit(tor[2]), tor[3], tor[4]
        wn = w[0] * nu[0] + w[1] * nu[1] + w[2] * nu[2]

        if abs(abs(wn) - 1.0) > 1e-7:
            return None

        d = (o[0] - C[0]) * w[0] + (o[1] - C[1]) * w[1] + (o[2] - C[2]) * w[2]

        if abs(d) > r:
            return []

        h = math.sqrt(max(0.0, r * r - d * d))
        cc = (C[0] + d * w[0], C[1] + d * w[1], C[2] + d * w[2])
        xa, ya = _ortho_basis(w)
        out = []

        for rr in (R + h, R - h):
            if rr > 1e-12:
                out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rr))

        return out

    def single(c):
        return [c] if c is not None else []

    c3_list = None

    if ra[0] == "plane" and rb[0] == "sphere":
        c3_list = single(plane_sphere(ra, rb))
    elif ra[0] == "sphere" and rb[0] == "plane":
        c3_list = single(plane_sphere(rb, ra))
    elif ra[0] == "plane" and rb[0] == "cylinder":
        c3_list = single(plane_cylinder(ra, rb))
    elif ra[0] == "cylinder" and rb[0] == "plane":
        c3_list = single(plane_cylinder(rb, ra))
    elif ra[0] == "plane" and rb[0] == "cone":
        c3_list = single(plane_cone(ra, rb))
    elif ra[0] == "cone" and rb[0] == "plane":
        c3_list = single(plane_cone(rb, ra))
    elif ra[0] == "plane" and rb[0] == "torus":
        c3_list = plane_torus(ra, rb)
    elif ra[0] == "torus" and rb[0] == "plane":
        c3_list = plane_torus(rb, ra)
    elif ra[0] == "sphere" and rb[0] == "sphere":
        c1, r1 = ra[1], ra[2]
        c2, r2 = rb[1], rb[2]
        dv = (c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2])
        dist = math.sqrt(dv[0] ** 2 + dv[1] ** 2 + dv[2] ** 2)
        c3 = None

        if 1e-12 < dist < r1 + r2 and dist > abs(r1 - r2):
            nu = (dv[0] / dist, dv[1] / dist, dv[2] / dist)
            aa = (dist * dist + r1 * r1 - r2 * r2) / (2.0 * dist)
            rr2 = r1 * r1 - aa * aa

            if rr2 > 0.0:
                cc = (c1[0] + aa * nu[0], c1[1] + aa * nu[1], c1[2] + aa * nu[2])
                xa, ya = _ortho_basis(nu)
                c3 = _exact_circle(cc[0], cc[1], cc[2], xa, ya, math.sqrt(rr2))

        c3_list = single(c3)
    elif ra[0] == "cylinder" and rb[0] == "sphere":
        c3_list = _ssi_cylinder_sphere(ra, rb)
    elif ra[0] == "sphere" and rb[0] == "cylinder":
        c3_list = _ssi_cylinder_sphere(rb, ra)
    elif ra[0] == "cylinder" and rb[0] == "cone":
        c3_list = _ssi_cylinder_cone(ra, rb)
    elif ra[0] == "cone" and rb[0] == "cylinder":
        c3_list = _ssi_cylinder_cone(rb, ra)
    elif ra[0] == "cone" and rb[0] == "sphere":
        c3_list = _ssi_cone_sphere(ra, rb)
    elif ra[0] == "sphere" and rb[0] == "cone":
        c3_list = _ssi_cone_sphere(rb, ra)
    elif ra[0] == "cylinder" and rb[0] == "cylinder":
        c3_list = _ssi_cylinder_cylinder(a, ra, b, rb)
    else:
        return None

    if c3_list is None:
        return None

    triples = []

    for cc3 in c3_list:
        pa = _analytic_pcurve(a, ra, cc3)
        pb = _analytic_pcurve(b, rb, cc3)

        if pa is None and ra[0] == "sphere":
            v = _analytic_sphere_pullback(a, ra, cc3)

            if v:
                pa = v[0]

        if pb is None and rb[0] == "sphere":
            v = _analytic_sphere_pullback(b, rb, cc3)

            if v:
                pb = v[0]

        if pa is None and (ra[0] == "cone" or ra[0] == "cylinder"):
            v = _analytic_cone_pullback(a, ra, cc3)

            if v:
                pa = v[0]

        if pb is None and (rb[0] == "cone" or rb[0] == "cylinder"):
            v = _analytic_cone_pullback(b, rb, cc3)

            if v:
                pb = v[0]

        if pa is None:
            v = Closest.surface_curve(a, cc3)

            if v:
                pa = v[0]

        if pb is None:
            v = Closest.surface_curve(b, cc3)

            if v:
                pb = v[0]

        if pa is not None and pa.is_valid() and pb is not None and pb.is_valid():
            triples.append((cc3, pa, pb))

    return triples


# ═══════════════════════════════════════════════════════════════════════════
# NURBS surfaces
# ═══════════════════════════════════════════════════════════════════════════


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
        all_pts = [surface.point_at(uv[0], uv[1]) for uv in uv_trace]
        crv = _surface_plane_fit_3d(
            all_pts, is_loop, plane, step, uv_to_3d, uv_to_3d_min
        )

        if not crv.is_valid():
            continue

        ct0, ct1 = crv.domain()
        dup_tol = step * uv_to_3d * 3.0
        dup = False

        for existing in result:
            et0, et1 = existing.domain()
            all_close = True

            for f in [0.25, 0.5, 0.75]:
                cp = crv.point_at(ct0 + (ct1 - ct0) * f)
                ep = existing.point_at(et0 + (et1 - et0) * f)
                em = existing.point_at((et0 + et1) * 0.5)
                d = min(cp.distance(ep), cp.distance(em))

                if d > dup_tol:
                    all_close = False
                    break

            if all_close:
                dup = True
                break

        if not dup:
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


def surface_surface(
    a: "NurbsSurface", b: "NurbsSurface", tolerance: float | None = None
) -> list[tuple[NurbsCurve, NurbsCurve, NurbsCurve]]:
    """Surface-surface section curves with their UV pcurves on both surfaces."""

    if a.is_valid() and b.is_valid():
        _ana = _analytic_ssi(
            a,
            b,
            tolerance if (tolerance and tolerance > 0) else Tolerance.ZERO_TOLERANCE,
        )

        if _ana is not None:
            return _ana

    if not a.is_valid() or not b.is_valid():
        return []

    if tolerance is None or tolerance <= 0.0:
        tolerance = Tolerance.ZERO_TOLERANCE

    def plane_from(srf):
        s0, s1 = srf.domain(0)
        t0, t1 = srf.domain(1)
        po = srf.point_at((s0 + s1) * 0.5, (t0 + t1) * 0.5)
        nn = srf.normal_at((s0 + s1) * 0.5, (t0 + t1) * 0.5)

        return Plane.from_point_normal(po, Vector(nn[0], nn[1], nn[2]))

    if a.is_planar(None, 1e-9):
        plane = plane_from(a)
        result = []

        for c3, pb in surface_plane_uv(b, plane, tolerance):
            pas = Closest.surface_curve(a, c3)

            if len(pas) == 1:
                result.append((c3, pas[0], pb))

        return result

    if b.is_planar(None, 1e-9):
        plane = plane_from(b)
        result = []

        for c3, pa in surface_plane_uv(a, plane, tolerance):
            pbs = Closest.surface_curve(b, c3)

            if len(pbs) == 1:
                result.append((c3, pa, pbs[0]))

        return result

    au0, au1 = a.domain(0)
    av0, av1 = a.domain(1)
    bu0, bu1 = b.domain(0)
    bv0, bv1 = b.domain(1)
    a_range_u = au1 - au0
    a_range_v = av1 - av0
    b_range_u = bu1 - bu0
    b_range_v = bv1 - bv0
    a_closed_u = a.is_closed(0)
    a_closed_v = a.is_closed(1)
    b_closed_u = b.is_closed(0)
    b_closed_v = b.is_closed(1)

    def make_wrap(c0, c1, rng, closed):
        def w(t):
            if closed:
                f = math.fmod(t - c0, rng)

                if f < 0:
                    f += rng

                return c0 + f

            return max(c0, min(t, c1))

        return w

    a_wrap_u = make_wrap(au0, au1, a_range_u, a_closed_u)
    a_wrap_v = make_wrap(av0, av1, a_range_v, a_closed_v)
    b_wrap_u = make_wrap(bu0, bu1, b_range_u, b_closed_u)
    b_wrap_v = make_wrap(bv0, bv1, b_range_v, b_closed_v)

    def eval_a(u, v):
        d = a.evaluate(a_wrap_u(u), a_wrap_v(v), 1)

        return d[0], d[2], d[1]

    def eval_b(u, v):
        d = b.evaluate(b_wrap_u(u), b_wrap_v(v), 1)

        return d[0], d[2], d[1]

    spans_au = a.get_span_vector(0)
    spans_av = a.get_span_vector(1)
    spans_bu = b.get_span_vector(0)
    spans_bv = b.get_span_vector(1)
    a_nu = max(len(spans_au) - 1, 1) * 4
    a_nv = max(len(spans_av) - 1, 1) * 4
    b_nu = max(len(spans_bu) - 1, 1) * 4
    b_nv = max(len(spans_bv) - 1, 1) * 4
    a_du = a_range_u / a_nu
    a_dv = a_range_v / a_nv
    b_du = b_range_u / b_nu
    b_dv = b_range_v / b_nv

    def cell_boxes(srf, c0u, dcu, ncu, c0v, dcv, ncv):
        S = []

        for i in range(2 * ncu + 1):
            row = []

            for j in range(2 * ncv + 1):
                row.append(srf.point_at(c0u + dcu * 0.5 * i, c0v + dcv * 0.5 * j))

            S.append(row)

        boxes = []

        for ci in range(ncu):
            for cj in range(ncv):
                xs = []
                ys = []
                zs = []

                for i in range(2 * ci, 2 * ci + 3):
                    for j in range(2 * cj, 2 * cj + 3):
                        p = S[i][j]
                        xs.append(p[0])
                        ys.append(p[1])
                        zs.append(p[2])

                ctr = S[2 * ci + 1][2 * cj + 1]
                cx = (
                    S[2 * ci][2 * cj][0]
                    + S[2 * ci + 2][2 * cj][0]
                    + S[2 * ci][2 * cj + 2][0]
                    + S[2 * ci + 2][2 * cj + 2][0]
                ) * 0.25
                cy = (
                    S[2 * ci][2 * cj][1]
                    + S[2 * ci + 2][2 * cj][1]
                    + S[2 * ci][2 * cj + 2][1]
                    + S[2 * ci + 2][2 * cj + 2][1]
                ) * 0.25
                cz = (
                    S[2 * ci][2 * cj][2]
                    + S[2 * ci + 2][2 * cj][2]
                    + S[2 * ci][2 * cj + 2][2]
                    + S[2 * ci + 2][2 * cj + 2][2]
                ) * 0.25
                sag = math.sqrt(
                    (ctr[0] - cx) ** 2 + (ctr[1] - cy) ** 2 + (ctr[2] - cz) ** 2
                )
                inf = 2.0 * sag + tolerance
                boxes.append(
                    (
                        min(xs) - inf,
                        min(ys) - inf,
                        min(zs) - inf,
                        max(xs) + inf,
                        max(ys) + inf,
                        max(zs) + inf,
                        c0u + dcu * (ci + 0.5),
                        c0v + dcv * (cj + 0.5),
                    )
                )

        return boxes

    boxes_a = cell_boxes(a, au0, a_du, a_nu, av0, a_dv, a_nv)
    boxes_b = cell_boxes(b, bu0, b_du, b_nu, bv0, b_dv, b_nv)

    def cell_3d(boxes):
        best = float("inf")

        for bx in boxes[:64]:
            d = math.sqrt(
                (bx[3] - bx[0]) ** 2 + (bx[4] - bx[1]) ** 2 + (bx[5] - bx[2]) ** 2
            )

            if 1e-12 < d < best:
                best = d

        return best if best < float("inf") else 1.0

    h_init = min(cell_3d(boxes_a), cell_3d(boxes_b)) * 0.25
    conv_tol = max(tolerance, h_init * 1e-7)

    def clamp_open(x):
        if not a_closed_u:
            x[0] = max(au0, min(x[0], au1))

        if not a_closed_v:
            x[1] = max(av0, min(x[1], av1))

        if not b_closed_u:
            x[2] = max(bu0, min(x[2], bu1))

        if not b_closed_v:
            x[3] = max(bv0, min(x[3], bv1))

    def correct(x, pin=None):
        for _ in range(8):
            Sa, Sau, Sav = eval_a(x[0], x[1])
            Sb, Sbu, Sbv = eval_b(x[2], x[3])
            F = [Sa[0] - Sb[0], Sa[1] - Sb[1], Sa[2] - Sb[2]]

            if math.sqrt(F[0] ** 2 + F[1] ** 2 + F[2] ** 2) < conv_tol:
                return True

            J = [[Sau[k], Sav[k], -Sbu[k], -Sbv[k]] for k in range(3)]

            if pin is None:
                JJt = [
                    [sum(J[r][c] * J[q][c] for c in range(4)) for q in range(3)]
                    for r in range(3)
                ]
                y = _solve_gauss(JJt, F, 3)

                if y is None:
                    return False

                for c in range(4):
                    x[c] -= sum(J[r][c] * y[r] for r in range(3))
            else:
                d, pp = pin
                M = [
                    J[0],
                    J[1],
                    J[2],
                    [
                        d[0] * Sau[0] + d[1] * Sau[1] + d[2] * Sau[2],
                        d[0] * Sav[0] + d[1] * Sav[1] + d[2] * Sav[2],
                        0.0,
                        0.0,
                    ],
                ]
                rhs = [
                    F[0],
                    F[1],
                    F[2],
                    d[0] * (Sa[0] - pp[0])
                    + d[1] * (Sa[1] - pp[1])
                    + d[2] * (Sa[2] - pp[2]),
                ]
                dx = _solve_gauss(M, rhs, 4)

                if dx is None:
                    return False

                for c in range(4):
                    x[c] -= dx[c]

            clamp_open(x)

        Sa, _, _ = eval_a(x[0], x[1])
        Sb, _, _ = eval_b(x[2], x[3])
        g = math.sqrt(
            (Sa[0] - Sb[0]) ** 2 + (Sa[1] - Sb[1]) ** 2 + (Sa[2] - Sb[2]) ** 2
        )

        return g < conv_tol * 10.0

    seeds = []
    seed_tol_3d = max(cell_3d(boxes_a), cell_3d(boxes_b))
    pair_budget = 20000

    for ba in boxes_a:
        if pair_budget < 0:
            break

        for bb in boxes_b:
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

            if not correct(x):
                continue

            Sa, _, _ = eval_a(x[0], x[1])
            dup = False

            for sd in seeds:
                So, _, _ = eval_a(sd[0], sd[1])

                if (
                    math.sqrt(
                        (Sa[0] - So[0]) ** 2
                        + (Sa[1] - So[1]) ** 2
                        + (Sa[2] - So[2]) ** 2
                    )
                    < seed_tol_3d
                ):
                    dup = True
                    break

            if not dup:
                seeds.append(
                    [
                        a_wrap_u(x[0]),
                        a_wrap_v(x[1]),
                        b_wrap_u(x[2]),
                        b_wrap_v(x[3]),
                        False,
                    ]
                )

    max_steps = (a_nu * a_nv + b_nu * b_nv) * 32
    close_tol = h_init * 3.0
    consume_tol = h_init * 2.0

    def tangent_3d(x, dir_sign):
        Sa, Sau, Sav = eval_a(x[0], x[1])
        Sb, Sbu, Sbv = eval_b(x[2], x[3])
        na = (
            Sau[1] * Sav[2] - Sau[2] * Sav[1],
            Sau[2] * Sav[0] - Sau[0] * Sav[2],
            Sau[0] * Sav[1] - Sau[1] * Sav[0],
        )
        nb = (
            Sbu[1] * Sbv[2] - Sbu[2] * Sbv[1],
            Sbu[2] * Sbv[0] - Sbu[0] * Sbv[2],
            Sbu[0] * Sbv[1] - Sbu[1] * Sbv[0],
        )
        d = (
            na[1] * nb[2] - na[2] * nb[1],
            na[2] * nb[0] - na[0] * nb[2],
            na[0] * nb[1] - na[1] * nb[0],
        )
        dl = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        nal = math.sqrt(na[0] ** 2 + na[1] ** 2 + na[2] ** 2)
        nbl = math.sqrt(nb[0] ** 2 + nb[1] ** 2 + nb[2] ** 2)

        if dl < 1e-4 * nal * nbl or dl < 1e-30:
            return None

        return (d[0] / dl * dir_sign, d[1] / dl * dir_sign, d[2] / dl * dir_sign), (
            Sa,
            Sau,
            Sav,
            Sbu,
            Sbv,
        )

    def trace_dir(x0, dir_sign):
        out = []
        x = list(x0)
        prev_d = None
        Sa0, _, _ = eval_a(x[0], x[1])
        p_start = (Sa0[0], Sa0[1], Sa0[2])
        p_prev = p_start
        dist_traveled = 0.0
        h = h_init
        smooth = 0

        for _step in range(max_steps):
            tng = tangent_3d(x, dir_sign)

            if tng is None:
                break

            d, (Sa, Sau, Sav, Sbu, Sbv) = tng
            accepted = False
            attempts = 0
            xn = None
            p_cur = None
            step_len = 0.0
            hit_boundary = False

            while attempts < 7 and not accepted:
                duv_a = _solve_gauss(
                    [
                        [
                            Sau[0] ** 2 + Sau[1] ** 2 + Sau[2] ** 2,
                            Sau[0] * Sav[0] + Sau[1] * Sav[1] + Sau[2] * Sav[2],
                        ],
                        [
                            Sau[0] * Sav[0] + Sau[1] * Sav[1] + Sau[2] * Sav[2],
                            Sav[0] ** 2 + Sav[1] ** 2 + Sav[2] ** 2,
                        ],
                    ],
                    [
                        h * (d[0] * Sau[0] + d[1] * Sau[1] + d[2] * Sau[2]),
                        h * (d[0] * Sav[0] + d[1] * Sav[1] + d[2] * Sav[2]),
                    ],
                    2,
                )
                duv_b = _solve_gauss(
                    [
                        [
                            Sbu[0] ** 2 + Sbu[1] ** 2 + Sbu[2] ** 2,
                            Sbu[0] * Sbv[0] + Sbu[1] * Sbv[1] + Sbu[2] * Sbv[2],
                        ],
                        [
                            Sbu[0] * Sbv[0] + Sbu[1] * Sbv[1] + Sbu[2] * Sbv[2],
                            Sbv[0] ** 2 + Sbv[1] ** 2 + Sbv[2] ** 2,
                        ],
                    ],
                    [
                        h * (d[0] * Sbu[0] + d[1] * Sbu[1] + d[2] * Sbu[2]),
                        h * (d[0] * Sbv[0] + d[1] * Sbv[1] + d[2] * Sbv[2]),
                    ],
                    2,
                )

                if duv_a is None or duv_b is None:
                    return out, False

                delta = [duv_a[0], duv_a[1], duv_b[0], duv_b[1]]
                tc = 1.0
                hit_boundary = False

                for idx, lo, hi, closed in (
                    (0, au0, au1, a_closed_u),
                    (1, av0, av1, a_closed_v),
                    (2, bu0, bu1, b_closed_u),
                    (3, bv0, bv1, b_closed_v),
                ):
                    if closed or abs(delta[idx]) < 1e-15:
                        continue

                    if x[idx] + delta[idx] > hi:
                        tc = min(tc, (hi - x[idx]) / delta[idx])
                        hit_boundary = True

                    if x[idx] + delta[idx] < lo:
                        tc = min(tc, (lo - x[idx]) / delta[idx])
                        hit_boundary = True

                xn = [x[k] + tc * delta[k] for k in range(4)]
                p_pred = (
                    Sa[0] + d[0] * h * tc,
                    Sa[1] + d[1] * h * tc,
                    Sa[2] + d[2] * h * tc,
                )

                if not correct(xn, (d, p_pred)):
                    return out, False

                San, _, _ = eval_a(xn[0], xn[1])
                p_cur = (San[0], San[1], San[2])
                step_len = math.sqrt(
                    (p_cur[0] - p_prev[0]) ** 2
                    + (p_cur[1] - p_prev[1]) ** 2
                    + (p_cur[2] - p_prev[2]) ** 2
                )

                if prev_d is not None and step_len > 1e-14:
                    sd_ = (
                        (p_cur[0] - p_prev[0]) / step_len,
                        (p_cur[1] - p_prev[1]) / step_len,
                        (p_cur[2] - p_prev[2]) / step_len,
                    )
                    ddot = sd_[0] * prev_d[0] + sd_[1] * prev_d[1] + sd_[2] * prev_d[2]

                    if ddot < 0.985 and attempts < 6 and not hit_boundary:
                        h *= 0.5
                        attempts += 1
                        smooth = 0
                        continue

                accepted = True

            if not accepted:
                break

            prev_d = d
            smooth += 1

            if smooth >= 5 and h < h_init * 2.0:
                h *= 1.4
                smooth = 0

            x = xn
            dist_traveled += step_len

            if (
                dist_traveled > close_tol * 3.0
                and math.sqrt(
                    (p_cur[0] - p_start[0]) ** 2
                    + (p_cur[1] - p_start[1]) ** 2
                    + (p_cur[2] - p_start[2]) ** 2
                )
                < close_tol
            ):
                out.append(list(x))

                return out, True

            out.append(list(x))
            p_prev = p_cur

            if hit_boundary:
                break

            for sd in seeds:
                if not sd[4]:
                    So, _, _ = eval_a(sd[0], sd[1])

                    if (
                        math.sqrt(
                            (p_cur[0] - So[0]) ** 2
                            + (p_cur[1] - So[1]) ** 2
                            + (p_cur[2] - So[2]) ** 2
                        )
                        < consume_tol
                    ):
                        sd[4] = True

        return out, False

    axes = (
        (0, au0, a_range_u, a_closed_u),
        (1, av0, a_range_v, a_closed_v),
        (2, bu0, b_range_u, b_closed_u),
        (3, bv0, b_range_v, b_closed_v),
    )

    result = []
    kept_pts3 = []

    for seed in seeds:
        if seed[4]:
            continue

        seed[4] = True
        x0 = [seed[0], seed[1], seed[2], seed[3]]

        if not correct(x0):
            continue

        fwd, fwd_closed = trace_dir(x0, +1)

        if not fwd_closed:
            bwd, _ = trace_dir(x0, -1)
        else:
            bwd = []

        quad = []

        for i in range(len(bwd) - 1, -1, -1):
            quad.append(list(bwd[i]))

        quad.append(list(x0))

        for p in fwd:
            quad.append(list(p))

        if len(quad) < 4:
            continue

        for i in range(1, len(quad)):
            for idx, c0, rng, closed in axes:
                if not closed:
                    continue

                jump = quad[i][idx] - quad[i - 1][idx]

                if jump > rng * 0.5:
                    quad[i][idx] -= rng
                elif jump < -rng * 0.5:
                    quad[i][idx] += rng

        def eval3_q(q):
            Sa, _, _ = eval_a(q[0], q[1])

            return Sa

        p_first = eval3_q(quad[0])
        p_last = eval3_q(quad[-1])
        gap2 = math.sqrt(
            (p_first[0] - p_last[0]) ** 2
            + (p_first[1] - p_last[1]) ** 2
            + (p_first[2] - p_last[2]) ** 2
        )
        is_loop = fwd_closed or (len(quad) >= 6 and gap2 < close_tol)

        if is_loop:
            quad.pop()

        if len(quad) < 4:
            continue

        m = len(quad)
        trace_pts3 = [eval3_q(q) for q in quad]
        dup_tol = h_init * 2.0
        dup = False

        for other in kept_pts3:
            all_close = True

            for f in [0.25, 0.5, 0.75]:
                cp = trace_pts3[int((m - 1) * f)]
                dmin = dup_tol + 1.0

                for k in range(0, len(other), 1):
                    op = other[k]
                    dmin = min(
                        dmin,
                        math.sqrt(
                            (cp[0] - op[0]) ** 2
                            + (cp[1] - op[1]) ** 2
                            + (cp[2] - op[2]) ** 2
                        ),
                    )

                if dmin > dup_tol:
                    all_close = False
                    break

            if all_close:
                dup = True
                break

        if dup:
            continue

        kept_pts3.append(trace_pts3)

        def _gap3(qi, qj):
            pi = eval3_q(qi)
            pj = eval3_q(qj)

            return math.sqrt(
                (pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2 + (pi[2] - pj[2]) ** 2
            )

        for _gp in range(4):
            gg = [_gap3(quad[i], quad[i + 1]) for i in range(len(quad) - 1)]

            if not gg:
                break

            med = sorted(gg)[len(gg) // 2]

            if med <= 0:
                break

            changed = False
            i = 0

            while i < len(quad) - 1 and len(quad) < 4000:
                if _gap3(quad[i], quad[i + 1]) > 1.5 * med:
                    mid = [(quad[i][k] + quad[i + 1][k]) * 0.5 for k in range(4)]

                    if correct(mid):
                        quad.insert(i + 1, mid)
                        changed = True
                        i += 2
                        continue

                i += 1

            if not changed:
                break

        closure = [0.0, 0.0, 0.0, 0.0]

        if is_loop and len(quad) >= 2:
            virt = list(quad[0])

            for idx, c0, rng, closed in axes:
                jump = quad[0][idx] - quad[-1][idx]

                if closed:
                    while jump > rng * 0.5:
                        jump -= rng

                    while jump < -rng * 0.5:
                        jump += rng

                virt[idx] = quad[-1][idx] + jump
                closure[idx] = virt[idx] - quad[0][idx]

            quad.append(virt)

        out_pts = [quad[0]]
        cross_idx = []

        for i in range(1, len(quad)):
            pa_ = quad[i - 1]
            pb_ = quad[i]
            crossings = []

            for idx, c0, rng, closed in axes:
                if not closed or abs(pb_[idx] - pa_[idx]) <= 1e-15:
                    continue

                k0 = math.floor((pa_[idx] - c0) / rng)
                k1 = math.floor((pb_[idx] - c0) / rng)

                for k in range(min(k0, k1) + 1, max(k0, k1) + 1):
                    L = c0 + k * rng
                    t = (L - pa_[idx]) / (pb_[idx] - pa_[idx])

                    if 0.0 < t < 1.0:
                        crossings.append((t, idx, L))

            crossings.sort()

            for t, idx, L in crossings:
                cp = [pa_[k] + (pb_[k] - pa_[k]) * t for k in range(4)]
                cp[idx] = L
                correct(cp)
                out_pts.append(cp)
                cross_idx.append(len(out_pts) - 1)

            out_pts.append(list(pb_))

            if i < len(quad) - 1:
                on_seam = False

                for idx, c0, rng, closed in axes:
                    if not closed:
                        continue

                    k = round((pb_[idx] - c0) / rng)
                    L = c0 + k * rng

                    if (
                        abs(pb_[idx] - L) < rng * 1e-9
                        and abs(pb_[idx] - pa_[idx]) > rng * 1e-9
                    ):
                        out_pts[-1][idx] = L
                        on_seam = True

                if on_seam:
                    cross_idx.append(len(out_pts) - 1)

        wrap_drift = False

        for idx, c0, rng, closed in axes:
            if abs(closure[idx]) > rng * 0.5:
                wrap_drift = True

        if len(cross_idx) == 0:
            pieces = [(out_pts, is_loop and not wrap_drift)]
        else:
            pieces = []

            if is_loop:
                for ia, ib in zip(cross_idx, cross_idx[1:]):
                    pieces.append((out_pts[ia : ib + 1], False))

                wrap_piece = [list(p) for p in out_pts[cross_idx[-1] :]]

                for p in out_pts[1 : cross_idx[0] + 1]:
                    wrap_piece.append([p[k] + closure[k] for k in range(4)])

                pieces.append((wrap_piece, False))
            else:
                bounds = [0] + cross_idx + [len(out_pts) - 1]

                for ia, ib in zip(bounds, bounds[1:]):
                    if ib > ia:
                        pieces.append((out_pts[ia : ib + 1], False))

        for piece_pts, piece_loop in pieces:
            if len(piece_pts) < 2:
                continue

            mid = piece_pts[len(piece_pts) // 2]

            for idx, c0, rng, closed in axes:
                if not closed:
                    continue

                k_s = math.floor((mid[idx] - c0) / rng)

                if k_s != 0:
                    for p in piece_pts:
                        p[idx] -= k_s * rng

            pts3 = [eval3_q(p) for p in piece_pts]
            chord3 = 0.0

            for i in range(1, len(pts3)):
                chord3 += math.sqrt(
                    (pts3[i][0] - pts3[i - 1][0]) ** 2
                    + (pts3[i][1] - pts3[i - 1][1]) ** 2
                    + (pts3[i][2] - pts3[i - 1][2]) ** 2
                )

            if chord3 < h_init * 0.5:
                continue

            refine_tol = max(tolerance * 100.0, 5e-6)

            for _dp in range(8):
                refined = False
                new_pp = [piece_pts[0]]
                i = 0

                while i < len(piece_pts) - 1 and len(piece_pts) < 3000:
                    pa2 = piece_pts[i]
                    pb2 = piece_pts[i + 1]
                    p3a = eval3_q(pa2)
                    p3b = eval3_q(pb2)
                    mid = [(pa2[k] + pb2[k]) * 0.5 for k in range(4)]

                    if correct(mid):
                        p3m = eval3_q(mid)
                        ex = p3b[0] - p3a[0]
                        ey = p3b[1] - p3a[1]
                        ez = p3b[2] - p3a[2]
                        l2 = ex * ex + ey * ey + ez * ez

                        if l2 > 1e-30:
                            tt = (
                                (p3m[0] - p3a[0]) * ex
                                + (p3m[1] - p3a[1]) * ey
                                + (p3m[2] - p3a[2]) * ez
                            ) / l2
                            cx = p3a[0] + tt * ex
                            cy = p3a[1] + tt * ey
                            cz = p3a[2] + tt * ez
                            dev = math.sqrt(
                                (p3m[0] - cx) ** 2
                                + (p3m[1] - cy) ** 2
                                + (p3m[2] - cz) ** 2
                            )
                        else:
                            dev = 0.0

                        if dev > refine_tol:
                            new_pp.append(mid)
                            refined = True

                    new_pp.append(pb2)
                    i += 1

                piece_pts = new_pp

                if not refined:
                    break

            pts3 = [eval3_q(p) for p in piece_pts]

            def fit_track(pts2, fit_tol_track):
                mp = len(pts2)
                total_turning = 0.0

                for i in range(1, mp - 1):
                    dx1 = pts2[i][0] - pts2[i - 1][0]
                    dy1 = pts2[i][1] - pts2[i - 1][1]
                    dz1 = pts2[i][2] - pts2[i - 1][2]
                    dx2 = pts2[i + 1][0] - pts2[i][0]
                    dy2 = pts2[i + 1][1] - pts2[i][1]
                    dz2 = pts2[i + 1][2] - pts2[i][2]
                    l1 = math.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1)
                    l2 = math.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2)

                    if l1 > 1e-14 and l2 > 1e-14:
                        c = max(
                            -1.0,
                            min(1.0, (dx1 * dx2 + dy1 * dy2 + dz1 * dz2) / (l1 * l2)),
                        )
                        total_turning += math.acos(c)

                chords = [0.0] * mp
                total_len = 0.0

                for i in range(1, mp):
                    total_len += pts2[i].distance(pts2[i - 1])
                    chords[i] = total_len

                if piece_loop and mp > 1:
                    total_len += pts2[0].distance(pts2[mp - 1])

                if total_len > 1e-14:
                    for i in range(1, mp):
                        chords[i] /= total_len

                target_cvs = max(8, int(total_turning / 0.5) + 6)
                max_cvs = max(8, min(mp - 1, mp // 3))
                best = NurbsCurve()
                best_dev = float("inf")

                while target_cvs <= max_cvs:
                    crv = NurbsCurve.create_fitted(pts2, target_cvs, 3, piece_loop)

                    if not crv.is_valid():
                        break

                    ft0, ft1 = crv.domain()
                    dev = 0.0

                    for i in range(mp):
                        dev = max(
                            dev,
                            crv.point_at(ft0 + (ft1 - ft0) * chords[i]).distance(
                                pts2[i]
                            ),
                        )

                    if dev < best_dev:
                        best, best_dev = crv, dev

                    if dev < fit_tol_track:
                        break

                    target_cvs *= 2

                if best_dev >= fit_tol_track:
                    interp = (
                        NurbsCurve.create_interpolated(
                            pts2, CurveNurbsKnotStyle.ChordPeriodic
                        )
                        if piece_loop
                        else NurbsCurve.create_interpolated(pts2)
                    )

                    if interp.is_valid():
                        best = interp

                if best.is_valid():
                    best.set_domain(0.0, 1.0)

                return best

            pts3_p = [Point(p[0], p[1], p[2]) for p in pts3]
            pts_pa = [Point(p[0], p[1], 0.0) for p in piece_pts]
            pts_pb = [Point(p[2], p[3], 0.0) for p in piece_pts]
            crv3 = fit_track(pts3_p, max(tolerance * 10.0, 1e-7))
            pcurve_a = fit_track(pts_pa, min(a_du, a_dv) * 1e-4)
            pcurve_b = fit_track(pts_pb, min(b_du, b_dv) * 1e-4)

            if (
                not crv3.is_valid()
                or not pcurve_a.is_valid()
                or not pcurve_b.is_valid()
            ):
                continue

            result.append((crv3, pcurve_a, pcurve_b))

    return result


def _clip_pcurve_to_cutter(target, pc, cutter):
    """Keep the pcurve sub-segments whose lifted 3D point lies inside the cutter footprint."""

    n = max(pc.cv_count() * 4, 16)
    d0, d1 = pc.domain()
    cu0, cu1 = cutter.domain(0)
    cv0, cv1 = cutter.domain(1)
    corner_diag = cutter.point_at(cu0, cv0).distance(cutter.point_at(cu1, cv1))
    on_tol = max(1e-7, corner_diag * 1e-4)

    q00 = cutter.point_at(cu0, cv0)
    q10 = cutter.point_at(cu1, cv0)
    q01 = cutter.point_at(cu0, cv1)
    eu0, eu1, eu2_ = q10[0] - q00[0], q10[1] - q00[1], q10[2] - q00[2]
    ev0, ev1, ev2_ = q01[0] - q00[0], q01[1] - q00[1], q01[2] - q00[2]
    eu_sq = eu0 * eu0 + eu1 * eu1 + eu2_ * eu2_
    ev_sq = ev0 * ev0 + ev1 * ev1 + ev2_ * ev2_
    q00x, q00y, q00z = q00[0], q00[1], q00[2]
    fast_planar = eu_sq > 1e-28 and ev_sq > 1e-28

    def gap(t):
        uv = pc.point_at(t)
        p3 = target.point_at(uv[0], uv[1])

        if fast_planar:
            dx = p3[0] - q00x
            dy = p3[1] - q00y
            dz = p3[2] - q00z
            a = (dx * eu0 + dy * eu1 + dz * eu2_) / eu_sq
            b = (dx * ev0 + dy * ev1 + dz * ev2_) / ev_sq

            if a < 0.0:
                a = 0.0
            elif a > 1.0:
                a = 1.0

            if b < 0.0:
                b = 0.0
            elif b > 1.0:
                b = 1.0

            cx = q00x + a * eu0 + b * ev0
            cy = q00y + a * eu1 + b * ev1
            cz = q00z + a * eu2_ + b * ev2_

            return ((p3[0] - cx) ** 2 + (p3[1] - cy) ** 2 + (p3[2] - cz) ** 2) ** 0.5

        return Closest.surface_point(cutter, p3, 0.0, 0.0, 0.0, 0.0)[2]

    def refine(t_in, t_out):
        for _ in range(20):
            tm = (t_in + t_out) * 0.5

            if gap(tm) < on_tol:
                t_in = tm
            else:
                t_out = tm

        return t_out

    flags = []

    for i in range(n + 1):
        t = d0 + (d1 - d0) * i / n
        flags.append((t, gap(t) < on_tol))

    pieces = []
    i = 0

    while i <= n:
        if flags[i][1]:
            j = i

            while j + 1 <= n and flags[j + 1][1]:
                j += 1

            ta = flags[i][0] if i == 0 else refine(flags[i][0], flags[i - 1][0])
            tb = flags[j][0] if j == n else refine(flags[j][0], flags[j + 1][0])

            if tb - ta > (d1 - d0) * 1e-6:
                piece = pc.duplicate()

                if piece.trim(ta, tb) and piece.is_valid():
                    pieces.append(piece)

            i = j + 1
        else:
            i += 1

    return pieces


def cut_curves_on_surface(
    target: "NurbsSurface", cutter: "NurbsSurface", tolerance: float | None = None
) -> list[NurbsCurve]:
    """UV pcurves of the cutter's section on the target, clipped to the cutter footprint."""

    rtol = max(tolerance if (tolerance and tolerance > 0) else 1e-7, 1e-7) * 1e4
    rt = _recognize_surface(target, rtol)

    if rt is not None and rt[0] == "sphere":
        cutter_planar = cutter.is_planar(None, 1e-6)
        out = []

        for triple in surface_surface(target, cutter, tolerance):
            c3d = triple[0]
            pcs = _analytic_sphere_pullback(target, rt, c3d)

            if not pcs:
                pcs = Closest.surface_curve(target, c3d, 0.0, 0.0, tolerance or 0.0)

            if not pcs:
                pcs = [triple[1]]

            for pc in pcs:
                if cutter_planar:
                    out.extend(_clip_pcurve_to_cutter(target, pc, cutter))
                else:
                    out.append(pc)

        return out

    if cutter.is_planar(None, 1e-6):
        cu0, cu1 = cutter.domain(0)
        cv0, cv1 = cutter.domain(1)
        mu = (cu0 + cu1) * 0.5
        mv = (cv0 + cv1) * 0.5
        origin = cutter.point_at(mu, mv)
        normal = cutter.normal_at(mu, mv)
        plane = Plane.from_point_normal(origin, normal)
        out = []

        for pair in surface_plane_uv(target, plane, tolerance):
            out.extend(_clip_pcurve_to_cutter(target, pair[1], cutter))

        return out

    return [triple[1] for triple in surface_surface(target, cutter, tolerance)]


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


def line_two_planes(line: Line, p0: Plane, p1: Plane) -> object | None:
    """Clips a segment to the two plane intersections."""

    new_start = line_plane(line, p0, True)
    new_end = line_plane(line, p1, True)

    if new_start is None or new_end is None:
        return None

    return Line(
        new_start[0], new_start[1], new_start[2], new_end[0], new_end[1], new_end[2]
    )


def polyline_plane(poly: Polyline, plane: Plane) -> tuple | None:
    """Polyline edge crossings with a plane and their edge indices."""

    n = poly.point_count()

    if n < 2:
        return None

    points = []
    edge_ids = []

    for i in range(n - 1):
        a = poly.get_point(i)
        b = poly.get_point(i + 1)
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
                front = poly.get_point(0)
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
    direction: "Vector", p0: Plane, p1: Plane
) -> object | None:
    """Direction scaled to span the distance between two planes."""

    mag = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)

    if mag < 1e-14:
        return None

    ray = Line(0.0, 0.0, 0.0, direction[0], direction[1], direction[2])
    q0 = line_plane(ray, p0, False)
    q1 = line_plane(ray, p1, False)

    if q0 is None or q1 is None:
        return None

    output = q1 - q0
    n1 = p1.z_axis
    n1_mag = math.sqrt(n1[0] ** 2 + n1[1] ** 2 + n1[2] ** 2)

    if n1_mag < 1e-14:
        return None

    o0 = p0.origin
    d = (
        (o0[0] - p1.origin[0]) * n1[0]
        + (o0[1] - p1.origin[1]) * n1[1]
        + (o0[2] - p1.origin[2]) * n1[2]
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

    if 0 <= intersection_type <= 2:
        result_2d = BooleanPolyline.compute(a2d, b2d, intersection_type)
    elif intersection_type == 3:
        u = BooleanPolyline.compute(a2d, b2d, 1)
        inter = BooleanPolyline.compute(a2d, b2d, 0)

        if not u:
            return None

        result_2d = u if not inter else BooleanPolyline.compute(u[0], inter[0], 2)
    else:
        return None

    if not result_2d:
        return None

    ring = _polyline_to_2d(result_2d[0], flat_origin, flat_x, flat_y)

    if len(ring) < 3:
        return None

    if collapse_eps > 0.0:
        eps_sq = collapse_eps * collapse_eps
        collapsed = []

        for p in ring:
            if not collapsed or _distance_sq_2d(p, collapsed[-1]) >= eps_sq:
                collapsed.append(p)

        if (
            len(collapsed) >= 2
            and _distance_sq_2d(collapsed[-1], collapsed[0]) < eps_sq
        ):
            collapsed.pop()

        ring = collapsed

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


def face_to_face(
    adjacency: list[int],
    polylines: list[list[Polyline]],
    planes: list[list[Plane]],
    coplanar_tolerance: float = 5.0,
) -> list[tuple[int, int, int, int, int, Polyline]]:
    """Face-to-face contacts (a, b, face_a, face_b, type, polyline) with type 0 side-side, 1 side-top, 2 top-top."""

    results = []
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
                bx[k] -= coplanar_tolerance
                bx[k + 3] += coplanar_tolerance

            boxes.append(bx)

        face_boxes.append(boxes)

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

                pts_i = polylines[a][i].get_points()
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
                bools = Polyline.boolean_op(
                    polylines[a][i], polylines[b][j], 0, plane=pln
                )

                if not bools or bools[0].point_count() < 3:
                    continue

                typ = (0 if i > 1 else 1) + (0 if j > 1 else 1)
                jpl = bools[0] if bools[0].is_closed() else bools[0].closed()
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

    DIST_SQ = 1e-6
    EPS_PAR = 1.0

    v0 = s0.to_vector()
    v1 = s1.to_vector()
    normal = v0.cross(v1)
    nmag2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
    ang = v0.angle(v1, False, True)
    is_parallel = (nmag2 < 1e-24) or ((90.0 - abs(ang - 90.0)) < EPS_PAR)

    if is_parallel:
        tmp_origin = s0.start()
        tmp_normal = v0
        pl_tmp = Plane.from_point_normal(tmp_origin, tmp_normal)
        normal = pl_tmp.base1()

    normal.normalize_self()

    def eq(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]

        return dx * dx + dy * dy + dz * dz < DIST_SQ

    def endcase(pp, dv0, dv1):
        p0 = pp
        p1 = pp
        v0 = dv0
        v0.normalize_self()
        v1 = dv1
        v1.normalize_self()

        return (p0, p1, v0, v1, normal, 0, 0, is_parallel)

    if eq(s0.start(), s1.start()):
        return endcase(s0.start(), s0.end() - s0.start(), s1.end() - s1.start())

    if eq(s0.start(), s1.end()):
        return endcase(s0.start(), s0.end() - s0.start(), s1.start() - s1.end())

    if eq(s0.end(), s1.start()):
        return endcase(s0.end(), s0.start() - s0.end(), s1.end() - s1.start())

    if eq(s0.end(), s1.end()):
        return endcase(s0.end(), s0.start() - s0.end(), s1.start() - s1.end())

    if is_parallel:
        v0.normalize_self()
        v1.normalize_self()

        def signed_t(src, unit, q):
            return (
                (q[0] - src[0]) * unit[0]
                + (q[1] - src[1]) * unit[1]
                + (q[2] - src[2]) * unit[2]
            )

        def proj_onto_line(L, q):
            return L.closest_point(q, False)[1]

        pts = []

        def push(q):
            q0 = proj_onto_line(s0, q)
            q1 = proj_onto_line(s1, q)
            pts.append((signed_t(s0.start(), v0, q0), signed_t(s1.start(), v1, q1)))

        push(s0.start())
        push(s0.end())
        push(s1.start())
        push(s1.end())
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
        p0 = proj_onto_line(s0, avg)
        p1 = proj_onto_line(s1, avg)

        def t_of(L, q):
            return L.closest_point(q, False)[0]

        t0_v = t_of(s0, p0)
        t1_v = t_of(s1, p1)

        if t0_v > 0.5:
            v0 = Vector(-v0[0], -v0[1], -v0[2])

        if t1_v > 0.5:
            v1 = Vector(-v1[0], -v1[1], -v1[2])

        return (p0, p1, v0, v1, normal, 0, 0, is_parallel)

    v0.normalize_self()
    v1.normalize_self()
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
            type0 = 0
            type1 = 1
        elif close0 < close1 and type0 == 0 and type1 == 0:
            type0 = 1
            type1 = 0

    if tt0 > 0.5 and type0 == 0:
        v0 = Vector(-v0[0], -v0[1], -v0[2])

    if tt1 > 0.5 and type1 == 0:
        v1 = Vector(-v1[0], -v1[1], -v1[2])

    return (p0, p1, v0, v1, normal, type0, type1, is_parallel)
