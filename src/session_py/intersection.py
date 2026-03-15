"""
Intersection functions for geometric primitives.

This module provides intersection calculations between various geometric objects
including lines, planes, rays, boxes, spheres, triangles, and meshes.
"""

from typing import Optional, Tuple, List
from .line import Line
from .point import Point
from .boundingbox import BoundingBox
from .mesh import Mesh
from .bvh import BVH
from .closest import Closest


def line_line_parameters(
    line0: Line,
    line1: Line,
    tolerance: float,
    intersect_segments: bool = True,
    near_parallel_as_closest: bool = False,
) -> Optional[Tuple[float, float]]:
    """
    Find parametric values where two lines are closest.

    Args:
        line0: First line
        line1: Second line
        tolerance: Maximum distance to consider intersection
        intersect_segments: If True, clamp parameters to [0,1]; if False, treat as infinite lines
        near_parallel_as_closest: If True, return closest point for near-parallel lines

    Returns:
        Tuple of (t0, t1) parameters if intersection found, None otherwise
        t0 is parameter on line0 (0=start, 1=end)
        t1 is parameter on line1 (0=start, 1=end)
    """
    p0_start = line0.start()
    p0_end = line0.end()
    p1_start = line1.start()
    p1_end = line1.end()

    if p0_start == p1_start:
        return (0.0, 0.0)
    if p0_start == p1_end:
        return (0.0, 1.0)
    if p0_end == p1_start:
        return (1.0, 0.0)
    if p0_end == p1_end:
        return (1.0, 1.0)

    A = line0.to_vector()
    B = line1.to_vector()
    C = p1_start - p0_start

    AA = A.dot(A)
    BB = B.dot(B)
    AB = A.dot(B)
    AC = A.dot(C)
    BC = B.dot(C)

    det = AA * BB - AB * AB

    zero_tol = max(AA, BB) * 1e-15
    if abs(det) < zero_tol:
        if not near_parallel_as_closest:
            return None
        t0 = (AC / AA) if AA > 0.0 else 0.0
        t1 = ((BC + t0 * AB) / BB) if BB > 0.0 else 0.0

        if intersect_segments:
            t0 = max(0.0, min(1.0, t0))
            t1 = max(0.0, min(1.0, t1))

        if tolerance > 0.0:
            pt0 = line0.point_at(t0)
            pt1 = line1.point_at(t1)
            if pt0.distance(pt1) > tolerance:
                return None
        return (t0, t1)

    inv_det = 1.0 / det
    t0 = (BB * AC - AB * BC) * inv_det
    t1 = (AB * AC - AA * BC) * inv_det

    if intersect_segments:
        t0 = max(0.0, min(1.0, t0))
        t1 = max(0.0, min(1.0, t1))

    if tolerance > 0.0:
        pt0 = line0.point_at(t0)
        pt1 = line1.point_at(t1)
        if pt0.distance(pt1) > tolerance:
            return None

    return (t0, t1)


def line_line(line0: Line, line1: Line, tolerance: float) -> Optional[Point]:
    """
    Find intersection point between two 3D lines.

    Args:
        line0: First line
        line1: Second line
        tolerance: Maximum distance between lines to consider them intersecting

    Returns:
        Intersection point (midpoint of closest approach for skew lines) if within tolerance,
        None otherwise
    """
    result = line_line_parameters(line0, line1, tolerance, True, False)

    if result is None:
        return None

    t0, t1 = result
    p0 = line0.point_at(t0)
    p1 = line1.point_at(t1)

    return Point((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, (p0[2] + p1[2]) * 0.5)


def plane_plane(plane0, plane1) -> Optional[Line]:
    from .plane import Plane

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


def plane_value_at(plane, point: Point) -> float:
    """Calculate the plane equation value at a point"""
    return plane.a * point[0] + plane.b * point[1] + plane.c * point[2] + plane.d


def line_plane(line: Line, plane, is_finite: bool = True) -> Optional[Point]:
    """
    Find intersection point between a line and a plane.

    Args:
        line: Line to intersect
        plane: Plane to intersect
        is_finite: If True, treat line as finite segment; if False, treat as infinite

    Returns:
        Intersection point if exists, None if line is parallel to plane
    """
    pt0 = line.start()
    pt1 = line.end()

    a = plane_value_at(plane, pt0)
    b = plane_value_at(plane, pt1)
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


def plane_plane_plane(plane0, plane1, plane2) -> Optional[Point]:
    """
    Find intersection point of three planes.

    Args:
        plane0: First plane
        plane1: Second plane
        plane2: Third plane

    Returns:
        Intersection point if planes intersect at a point, None if parallel or degenerate
    """
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


def ray_box(
    line: Line, box: BoundingBox, t0: float, t1: float
) -> Optional[List[Point]]:
    """
    Find intersection points between a line and an axis-aligned bounding box.

    Args:
        line: Line to intersect
        box: Axis-aligned bounding box
        t0: Minimum parameter value to consider (e.g., 0.0 for ray origin)
        t1: Maximum parameter value to consider (e.g., 1000.0 for max distance)

    Returns:
        List of 2 points [entry, exit] if intersection exists, None otherwise
        Points are sorted from line start (entry first, exit second)
    """
    origin = line.start()
    direction = line.to_vector()

    box_min = box.min_point()
    box_max = box.max_point()

    # Calculate inverse direction (avoid division by zero)
    inv_dir_x = 1.0 / direction[0] if direction[0] != 0.0 else float("inf")
    inv_dir_y = 1.0 / direction[1] if direction[1] != 0.0 else float("inf")
    inv_dir_z = 1.0 / direction[2] if direction[2] != 0.0 else float("inf")

    # Calculate intersections with X slabs
    tx1 = (box_min[0] - origin[0]) * inv_dir_x
    tx2 = (box_max[0] - origin[0]) * inv_dir_x

    tmin = min(tx1, tx2)
    tmax = max(tx1, tx2)

    # Calculate intersections with Y slabs
    ty1 = (box_min[1] - origin[1]) * inv_dir_y
    ty2 = (box_max[1] - origin[1]) * inv_dir_y

    tmin = max(tmin, min(ty1, ty2))
    tmax = min(tmax, max(ty1, ty2))

    # Calculate intersections with Z slabs
    tz1 = (box_min[2] - origin[2]) * inv_dir_z
    tz2 = (box_max[2] - origin[2]) * inv_dir_z

    tmin = max(tmin, min(tz1, tz2))
    tmax = min(tmax, max(tz1, tz2))

    # Clip to valid range
    tmin = max(tmin, t0)
    tmax = min(tmax, t1)

    # Check if intersection exists
    if tmax < tmin:
        return None

    # Calculate actual intersection points
    entry = Point(
        origin[0] + direction[0] * tmin,
        origin[1] + direction[1] * tmin,
        origin[2] + direction[2] * tmin,
    )

    exit_point = Point(
        origin[0] + direction[0] * tmax,
        origin[1] + direction[1] * tmax,
        origin[2] + direction[2] * tmax,
    )

    return [entry, exit_point]


def ray_sphere(line: Line, center: Point, radius: float) -> Optional[List[Point]]:
    """
    Find intersection points between a line and a sphere.

    Args:
        line: Line to intersect
        center: Sphere center point
        radius: Sphere radius

    Returns:
        List of 1 point (tangent) or 2 points (entry/exit) if intersection exists,
        None otherwise. Points are sorted from line start.
    """
    origin = line.start()
    direction = line.to_vector()

    # Vector from origin to center
    o_x = origin[0] - center[0]
    o_y = origin[1] - center[1]
    o_z = origin[2] - center[2]

    # Quadratic equation coefficients
    a = (
        direction[0] * direction[0]
        + direction[1] * direction[1]
        + direction[2] * direction[2]
    )
    b = 2.0 * (direction[0] * o_x + direction[1] * o_y + direction[2] * o_z)
    c = o_x * o_x + o_y * o_y + o_z * o_z - radius * radius

    # Discriminant
    disc = b * b - 4.0 * a * c

    if disc < 0.0:
        return None

    # Calculate intersection parameters
    dist_sqrt = disc**0.5
    if b < 0.0:
        q = (-b - dist_sqrt) / 2.0
    else:
        q = (-b + dist_sqrt) / 2.0

    t0 = q / a
    t1 = c / q

    # Sort parameters
    if t0 > t1:
        t0, t1 = t1, t0

    # Calculate intersection points
    points = []

    # First intersection
    p0 = Point(
        origin[0] + direction[0] * t0,
        origin[1] + direction[1] * t0,
        origin[2] + direction[2] * t0,
    )
    points.append(p0)

    # Second intersection (if different from first)
    if abs(t1 - t0) > 1e-10:
        p1 = Point(
            origin[0] + direction[0] * t1,
            origin[1] + direction[1] * t1,
            origin[2] + direction[2] * t1,
        )
        points.append(p1)

    return points


def ray_triangle(
    line: Line, v0: Point, v1: Point, v2: Point, epsilon: float
) -> Optional[Point]:
    """
    Find intersection point between a line and a triangle.

    Args:
        line: Line to intersect (start point used as origin, direction computed internally)
        v0: First vertex of triangle
        v1: Second vertex of triangle
        v2: Third vertex of triangle
        epsilon: Tolerance for parallel detection

    Returns:
        Intersection point if exists, None otherwise
    """
    origin = line.start()
    direction = line.to_vector()

    # Möller-Trumbore algorithm
    edge1_x = v1[0] - v0[0]
    edge1_y = v1[1] - v0[1]
    edge1_z = v1[2] - v0[2]

    edge2_x = v2[0] - v0[0]
    edge2_y = v2[1] - v0[1]
    edge2_z = v2[2] - v0[2]

    # pvec = direction.cross(edge2)
    pvec_x = direction[1] * edge2_z - direction[2] * edge2_y
    pvec_y = direction[2] * edge2_x - direction[0] * edge2_z
    pvec_z = direction[0] * edge2_y - direction[1] * edge2_x

    # det = edge1.dot(pvec)
    det = edge1_x * pvec_x + edge1_y * pvec_y + edge1_z * pvec_z

    if -epsilon < det < epsilon:
        return None  # Parallel

    inv_det = 1.0 / det

    # tvec = origin - v0
    tvec_x = origin[0] - v0[0]
    tvec_y = origin[1] - v0[1]
    tvec_z = origin[2] - v0[2]

    # u = tvec.dot(pvec) * inv_det
    u = (tvec_x * pvec_x + tvec_y * pvec_y + tvec_z * pvec_z) * inv_det

    if u < -epsilon or u > 1.0 + epsilon:
        return None

    # qvec = tvec.cross(edge1)
    qvec_x = tvec_y * edge1_z - tvec_z * edge1_y
    qvec_y = tvec_z * edge1_x - tvec_x * edge1_z
    qvec_z = tvec_x * edge1_y - tvec_y * edge1_x

    # v = direction.dot(qvec) * inv_det
    v = (direction[0] * qvec_x + direction[1] * qvec_y + direction[2] * qvec_z) * inv_det

    if v < -epsilon or u + v > 1.0 + epsilon:
        return None

    # t = edge2.dot(qvec) * inv_det
    t = (edge2_x * qvec_x + edge2_y * qvec_y + edge2_z * qvec_z) * inv_det

    # Calculate intersection point: origin + t * direction
    return Point(
        origin[0] + t * direction[0],
        origin[1] + t * direction[1],
        origin[2] + t * direction[2],
    )


def _mesh_triangles(mesh: Mesh) -> List[Tuple[Point, Point, Point]]:
    vertices, faces = mesh.to_vertices_and_faces()
    tris: List[Tuple[Point, Point, Point]] = []
    for face in faces:
        if len(face) < 3:
            continue
        v0 = vertices[face[0]]
        for i in range(1, len(face) - 1):
            v1 = vertices[face[i]]
            v2 = vertices[face[i + 1]]
            tris.append((v0, v1, v2))
    return tris


def ray_mesh(
    line: Line, mesh: Mesh, epsilon: float = 1e-6, find_all: bool = True
) -> Optional[List[Point]]:
    tris = _mesh_triangles(mesh)
    if not tris:
        return None

    hits: List[Tuple[float, Point]] = []
    origin = line.start()
    direction = line.to_vector().normalize()

    for v0, v1, v2 in tris:
        p = ray_triangle(line, v0, v1, v2, epsilon)
        if p is None:
            continue
        t = (
            (p[0] - origin[0]) * direction[0]
            + (p[1] - origin[1]) * direction[1]
            + (p[2] - origin[2]) * direction[2]
        )
        if t >= 0.0:
            hits.append((t, p))

    if not hits:
        return None

    hits.sort(key=lambda tp: tp[0])
    if find_all:
        return [p for _, p in hits]
    else:
        return [hits[0][1]]


def ray_mesh_bvh(
    line: Line, mesh: Mesh, epsilon: float = 1e-6, find_all: bool = True
) -> Optional[List[Point]]:
    tris = _mesh_triangles(mesh)
    if not tris:
        return None

    # Build AABBs for triangles
    tri_boxes: List[BoundingBox] = []
    for v0, v1, v2 in tris:
        tri_boxes.append(BoundingBox.from_points([v0, v1, v2]))

    world_size = BVH.compute_world_size(tri_boxes)
    bvh = BVH.from_boxes(tri_boxes, world_size)

    origin = line.start()
    direction = line.to_vector().normalize()
    candidate_ids: List[int] = []
    found = bvh.ray_cast(origin, direction, candidate_ids, True)
    if not found:
        return None

    hits: List[Tuple[float, Point]] = []
    for idx in candidate_ids:
        if 0 <= idx < len(tris):
            v0, v1, v2 = tris[idx]
            p = ray_triangle(line, v0, v1, v2, epsilon)
            if p is None:
                continue
            t = (
                (p[0] - origin[0]) * direction[0]
                + (p[1] - origin[1]) * direction[1]
                + (p[2] - origin[2]) * direction[2]
            )
            if t >= 0.0:
                hits.append((t, p))

    if not hits:
        return None

    hits.sort(key=lambda tp: tp[0])
    if find_all:
        return [p for _, p in hits]
    else:
        return [hits[0][1]]


#==========================================================================================
# NURBS Curve Intersection Functions
#==========================================================================================

from .vector import Vector
from .tolerance import Tolerance


def _curve_signed_distance_to_plane(pt, plane):
    """Signed distance from point to plane."""
    v = Vector(pt[0] - plane.origin[0], pt[1] - plane.origin[1], pt[2] - plane.origin[2])
    return v.dot(plane.z_axis)


def curve_plane(curve, plane, tolerance=None):
    """Find all intersections between NURBS curve and plane."""
    if tolerance is None:
        tolerance = Tolerance.ZERO_TOLERANCE

    if not curve.is_valid():
        return []

    results = []
    t_start, t_end = curve.domain()
    span_params = curve.get_span_vector()

    for i in range(len(span_params) - 1):
        t0 = span_params[i]
        t1 = span_params[i + 1]

        if abs(t1 - t0) < tolerance:
            continue

        d0 = _curve_signed_distance_to_plane(curve.point_at(t0), plane)
        d1 = _curve_signed_distance_to_plane(curve.point_at(t1), plane)

        if d0 * d1 < 0:
            ta, tb = t0, t1
            tm = (ta + tb) * 0.5
            for _ in range(50):
                tm = (ta + tb) * 0.5
                dm = _curve_signed_distance_to_plane(curve.point_at(tm), plane)
                if abs(dm) < tolerance:
                    break
                if dm * d0 < 0:
                    tb = tm
                else:
                    ta = tm
            results.append(tm)
        elif abs(d0) < tolerance:
            if not results or abs(results[-1] - t0) >= tolerance:
                results.append(t0)

    d_end = _curve_signed_distance_to_plane(curve.point_at(t_end), plane)
    if abs(d_end) < tolerance:
        if not results or abs(results[-1] - t_end) >= tolerance:
            results.append(t_end)

    results.sort()
    if len(results) > 1:
        unique_results = [results[0]]
        for i in range(1, len(results)):
            if abs(results[i] - unique_results[-1]) >= tolerance * 2.0:
                unique_results.append(results[i])
        results = unique_results

    return results


def curve_plane_points(curve, plane, tolerance=None):
    """Find all intersection points between NURBS curve and plane."""
    params = curve_plane(curve, plane, tolerance)
    return [curve.point_at(t) for t in params]


def curve_plane_bezier_clipping(curve, plane, tolerance=None):
    """Curve-plane intersection using Bézier clipping (advanced method)."""
    if tolerance is None:
        tolerance = Tolerance.ZERO_TOLERANCE

    if not curve.is_valid():
        return []

    results = []
    t0, t1 = curve.domain()

    def clip_recursive(ta, tb, depth):
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
                    tan = curve.tangent_at(t)
                    f = _curve_signed_distance_to_plane(pt, plane)
                    df = tan.dot(plane.z_axis)
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
                if abs(_curve_signed_distance_to_plane(pt_final, plane)) < tolerance and ta <= t <= tb:
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
            clip_recursive(ta, tm, depth + 1)
            clip_recursive(tm, tb, depth + 1)
        else:
            clip_recursive(t_min, t_max, depth + 1)

    clip_recursive(t0, t1, 0)

    results.sort()
    if len(results) > 1:
        unique_results = [results[0]]
        for i in range(1, len(results)):
            if abs(results[i] - results[i-1]) > tolerance * 2.0:
                unique_results.append(results[i])
        results = unique_results

    return results


def curve_plane_algebraic(curve, plane, tolerance=None):
    """Curve-plane intersection using algebraic/hodograph method."""
    if tolerance is None:
        tolerance = Tolerance.ZERO_TOLERANCE

    if not curve.is_valid():
        return []

    results = []
    spans = curve.get_span_vector()

    for span_idx in range(len(spans) - 1):
        span_t0 = spans[span_idx]
        span_t1 = spans[span_idx + 1]

        if abs(span_t1 - span_t0) < tolerance:
            continue

        d0 = _curve_signed_distance_to_plane(curve.point_at(span_t0), plane)
        d1 = _curve_signed_distance_to_plane(curve.point_at(span_t1), plane)

        if d0 * d1 > tolerance * tolerance:
            continue

        ta, tb = span_t0, span_t1
        da, db = d0, d1

        for _ in range(20):
            if abs(tb - ta) < tolerance * 0.1:
                break
            tm = (ta + tb) * 0.5
            pt_m = curve.point_at(tm)
            dm = _curve_signed_distance_to_plane(pt_m, plane)
            if abs(dm) < tolerance:
                ta = tb = tm
                break
            if da * dm < 0:
                tb, db = tm, dm
            else:
                ta, da = tm, dm

        t = (ta + tb) * 0.5

        for iteration in range(15):
            pt = curve.point_at(t)
            f = _curve_signed_distance_to_plane(pt, plane)
            if abs(f) < tolerance:
                break
            tan = curve.tangent_at(t)
            df = plane.z_axis.dot(tan)
            if abs(df) < 1e-10:
                if f * da < 0:
                    t = (ta + t) * 0.5
                else:
                    t = (t + tb) * 0.5
                continue
            dt = -f / df
            t_new = t + dt
            t_new = max(span_t0, min(span_t1, t_new))
            if abs(dt) < tolerance * 0.01:
                t = t_new
                break
            t = t_new

        pt_final = curve.point_at(t)
        if abs(_curve_signed_distance_to_plane(pt_final, plane)) < tolerance:
            is_duplicate = False
            for existing_t in results:
                if abs(t - existing_t) < tolerance * 2.0:
                    is_duplicate = True
                    break
            if not is_duplicate:
                results.append(t)

    return sorted(results)


def curve_plane_production(curve, plane, tolerance=None):
    """Curve-plane intersection using production CAD kernel method."""
    if tolerance is None:
        tolerance = Tolerance.ZERO_TOLERANCE

    if not curve.is_valid():
        return []

    def signed_distance_derivative(t):
        tan = curve.tangent_at(t)
        return plane.z_axis.dot(tan)

    results = []
    spans = curve.get_span_vector()

    for span_idx in range(len(spans) - 1):
        span_t0 = spans[span_idx]
        span_t1 = spans[span_idx + 1]

        if abs(span_t1 - span_t0) < tolerance:
            continue

        bezier_cvs = curve.convert_span_to_bezier(span_idx)
        if not bezier_cvs:
            continue

        def subdivide_and_solve(ta, tb, depth):
            MAX_DEPTH = 30
            if depth > MAX_DEPTH:
                return

            pa = curve.point_at(ta)
            pb = curve.point_at(tb)
            da = _curve_signed_distance_to_plane(pa, plane)
            db = _curve_signed_distance_to_plane(pb, plane)

            if da * db > tolerance * tolerance:
                return

            segment_length = pa.distance(pb)
            if segment_length < tolerance * 10.0 or abs(tb - ta) < tolerance * 0.001:
                if abs(db - da) > tolerance:
                    t_init = ta - da * (tb - ta) / (db - da)
                else:
                    t_init = (ta + tb) * 0.5
                t_init = max(ta, min(tb, t_init))

                t = t_init
                for newton_iter in range(5):
                    pt = curve.point_at(t)
                    f = _curve_signed_distance_to_plane(pt, plane)
                    if abs(f) < tolerance:
                        if ta <= t <= tb:
                            is_duplicate = False
                            for existing_t in results:
                                if abs(t - existing_t) < tolerance * 2.0:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                results.append(t)
                        return
                    df = signed_distance_derivative(t)
                    if abs(df) < 1e-10:
                        t = (ta + tb) * 0.5
                        break
                    dt = -f / df
                    t_new = t + dt
                    t_new = max(ta, min(tb, t_new))
                    if abs(dt) < tolerance * 0.001:
                        t = t_new
                        break
                    t = t_new

                pt_final = curve.point_at(t)
                if abs(_curve_signed_distance_to_plane(pt_final, plane)) < tolerance and ta <= t <= tb:
                    is_duplicate = False
                    for existing_t in results:
                        if abs(t - existing_t) < tolerance * 2.0:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        results.append(t)
                return

            tm = (ta + tb) * 0.5
            pm = curve.point_at(tm)

            v = Vector(pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
            w = Vector(pm[0] - pa[0], pm[1] - pa[1], pm[2] - pa[2])

            if v.magnitude() > Tolerance.ZERO_TOLERANCE:
                t_proj = w.dot(v) / v.dot(v)
                p_proj = Point(pa[0] + t_proj * v[0], pa[1] + t_proj * v[1], pa[2] + t_proj * v[2])
                deviation = pm.distance(p_proj)

                if deviation < tolerance * 10.0:
                    if abs(db - da) > tolerance:
                        t_root = ta - da * (tb - ta) / (db - da)
                        t_root = max(ta, min(tb, t_root))
                        for _ in range(3):
                            pt = curve.point_at(t_root)
                            f = _curve_signed_distance_to_plane(pt, plane)
                            if abs(f) < tolerance:
                                break
                            df = signed_distance_derivative(t_root)
                            if abs(df) > 1e-10:
                                t_root -= f / df
                                t_root = max(ta, min(tb, t_root))

                        if abs(_curve_signed_distance_to_plane(curve.point_at(t_root), plane)) < tolerance:
                            is_duplicate = False
                            for existing_t in results:
                                if abs(t_root - existing_t) < tolerance * 2.0:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                results.append(t_root)
                    return

            subdivide_and_solve(ta, tm, depth + 1)
            subdivide_and_solve(tm, tb, depth + 1)

        subdivide_and_solve(span_t0, span_t1, 0)

    return sorted(results)


def curve_closest_point(curve, test_point, t0=0.0, t1=0.0):
    """Find closest point on NURBS curve to test point."""
    return Closest.curve_point(curve, test_point, t0, t1)


def surface_plane(surface, plane, tolerance=None):
    """Find intersection curves between a NURBS surface and a plane."""
    import math
    from .nurbscurve import NurbsCurve
    from .nurbssurface import NurbsSurface
    from .knot import CurveKnotStyle

    if not surface.is_valid():
        return []
    if tolerance is None or tolerance <= 0.0:
        tolerance = Tolerance.ZERO_TOLERANCE

    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    range_u = u1 - u0
    range_v = v1 - v0
    closed_u = surface.is_closed(0)
    closed_v = surface.is_closed(1)

    def wrap_u(u):
        if closed_u:
            t = math.fmod(u - u0, range_u)
            if t < 0:
                t += range_u
            return u0 + t
        return max(u0, min(u, u1))

    def wrap_v(v):
        if closed_v:
            t = math.fmod(v - v0, range_v)
            if t < 0:
                t += range_v
            return v0 + t
        return max(v0, min(v, v1))

    pn = plane.z_axis
    p0 = plane.origin

    def g(u, v):
        p = surface.point_at(wrap_u(u), wrap_v(v))
        return (p[0]-p0[0])*pn[0] + (p[1]-p0[1])*pn[1] + (p[2]-p0[2])*pn[2]

    def g_and_grad(u, v):
        derivs = surface.evaluate(wrap_u(u), wrap_v(v), 1)
        S = derivs[0]
        Su = derivs[2]
        Sv = derivs[1]
        val = (S[0]-p0[0])*pn[0] + (S[1]-p0[1])*pn[1] + (S[2]-p0[2])*pn[2]
        gu = Su[0]*pn[0] + Su[1]*pn[1] + Su[2]*pn[2]
        gv = Sv[0]*pn[0] + Sv[1]*pn[1] + Sv[2]*pn[2]
        return val, gu, gv

    def newton_correct(uv):
        u, v = uv
        for _ in range(10):
            val, gu, gv = g_and_grad(u, v)
            if abs(val) < tolerance:
                uv[0], uv[1] = u, v
                return True
            mag2 = gu * gu + gv * gv
            if mag2 < 1e-28:
                uv[0], uv[1] = u, v
                return False
            u -= val * gu / mag2
            v -= val * gv / mag2
            u = wrap_u(u)
            v = wrap_v(v)
        uv[0], uv[1] = u, v
        return abs(g(u, v)) < tolerance * 10.0

    # 1. Find seeds: coarse UV grid, detect sign changes, Newton-refine
    spans_u = surface.get_span_vector(0)
    spans_v = surface.get_span_vector(1)
    nu = max(len(spans_u) - 1, 1) * 4
    nv = max(len(spans_v) - 1, 1) * 4
    du = range_u / nu
    dv = range_v / nv

    mu = (u0 + u1) * 0.5
    mv = (v0 + v1) * 0.5
    pmid = surface.point_at(mu, mv)
    uv_to_3d_u = pmid.distance(surface.point_at(wrap_u(mu + du), mv)) / du
    uv_to_3d_v = pmid.distance(surface.point_at(mu, wrap_v(mv + dv))) / dv
    uv_to_3d = max(uv_to_3d_u, uv_to_3d_v)
    uv_to_3d_min = min(uv_to_3d_u, uv_to_3d_v)
    if uv_to_3d < 1e-10:
        uv_to_3d = 1.0
    if uv_to_3d_min < 1e-10:
        uv_to_3d_min = 1.0

    cols = nv + 1
    dist = [0.0] * ((nu + 1) * cols)
    for i in range(nu + 1):
        u = u0 + du * i
        for j in range(nv + 1):
            v = v0 + dv * j
            d = g(u, v)
            if d == 0.0:
                d = -1e-14
            dist[i * cols + j] = d

    seeds = []  # list of [u, v, used]

    h_jmax = nv - 1 if closed_v else nv
    for i in range(nu):
        for j in range(h_jmax + 1):
            d0 = dist[i * cols + j]
            d1 = dist[(i + 1) * cols + j]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                su = u0 + du * (i + t)
                sv = v0 + dv * j
                uv = [su, sv]
                if newton_correct(uv):
                    seeds.append([uv[0], uv[1], False])

    v_imax = nu - 1 if closed_u else nu
    for i in range(v_imax + 1):
        for j in range(nv):
            d0 = dist[i * cols + j]
            d1 = dist[i * cols + j + 1]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                su = u0 + du * i
                sv = v0 + dv * (j + t)
                uv = [su, sv]
                if newton_correct(uv):
                    seeds.append([uv[0], uv[1], False])

    # Deduplicate seeds (3D distance)
    seed_tol_3d = max(du, dv) * uv_to_3d
    for i in range(len(seeds)):
        if seeds[i][2]:
            continue
        pi = surface.point_at(seeds[i][0], seeds[i][1])
        for j in range(i + 1, len(seeds)):
            if seeds[j][2]:
                continue
            if pi.distance(surface.point_at(seeds[j][0], seeds[j][1])) < seed_tol_3d:
                seeds[j][2] = True

    # 2. Trace intersection curves via predictor-corrector marching
    step = min(du, dv) * 0.25
    max_steps = nu * nv * 32
    close_tol_3d = step * 4.0 * uv_to_3d_min
    consume_tol_3d = step * uv_to_3d * 2.0

    result = []

    for seed in seeds:
        if seed[2]:
            continue
        seed[2] = True

        def tangent_at_uv(u, v, dir_sign):
            val, gu, gv = g_and_grad(u, v)
            mag = math.hypot(gu, gv)
            if mag < 1e-14:
                return None
            return (-gv / mag * dir_sign, gu / mag * dir_sign)

        def trace_dir(su, sv, dir_sign):
            out = []
            u, v = su, sv
            prev_tu, prev_tv = 0.0, 0.0
            p_start = surface.point_at(su, sv)
            p_prev = p_start
            dist_traveled = 0.0
            for s in range(max_steps):
                tang = tangent_at_uv(u, v, dir_sign)
                if tang is None:
                    if math.hypot(prev_tu, prev_tv) < 1e-14:
                        break
                    tu, tv = prev_tu, prev_tv
                else:
                    tu, tv = tang

                local_step = step
                if math.hypot(prev_tu, prev_tv) > 1e-14:
                    dot = tu * prev_tu + tv * prev_tv
                    dot = max(-1.0, min(1.0, dot))
                    if dot < 0.95:
                        local_step = step * 0.25
                    elif dot < 0.985:
                        local_step = step * 0.5

                u_mid = u + local_step * 0.5 * tu
                v_mid = v + local_step * 0.5 * tv
                tang2 = tangent_at_uv(u_mid, v_mid, dir_sign)
                if tang2 is not None:
                    tu, tv = tang2
                prev_tu, prev_tv = tu, tv

                un = u + local_step * tu
                vn = v + local_step * tv

                hit_boundary = False
                if (not closed_u and (un < u0 or un > u1)) or \
                   (not closed_v and (vn < v0 or vn > v1)):
                    tc = 1.0
                    if not closed_u and tu > 0 and un > u1:
                        tc = min(tc, (u1 - u) / (local_step * tu))
                    if not closed_u and tu < 0 and un < u0:
                        tc = min(tc, (u0 - u) / (local_step * tu))
                    if not closed_v and tv > 0 and vn > v1:
                        tc = min(tc, (v1 - v) / (local_step * tv))
                    if not closed_v and tv < 0 and vn < v0:
                        tc = min(tc, (v0 - v) / (local_step * tv))
                    un = u + tc * local_step * tu
                    vn = v + tc * local_step * tv
                    hit_boundary = True
                un = wrap_u(un)
                vn = wrap_v(vn)

                uv = [un, vn]
                if not newton_correct(uv):
                    break
                un, vn = uv[0], uv[1]

                p_cur = surface.point_at(un, vn)
                dist_traveled += p_prev.distance(p_cur)

                if dist_traveled > close_tol_3d * 3.0 and \
                   p_start.distance(p_cur) < close_tol_3d:
                    out.append((un, vn))
                    return out, True

                out.append((un, vn))
                u, v = un, vn
                p_prev = p_cur

                if hit_boundary:
                    break

                for other in seeds:
                    if not other[2]:
                        if p_cur.distance(surface.point_at(other[0], other[1])) < consume_tol_3d:
                            other[2] = True

            return out, False

        fwd, fwd_closed = trace_dir(seed[0], seed[1], +1)
        if not fwd_closed:
            bwd, _ = trace_dir(seed[0], seed[1], -1)
        else:
            bwd = []

        # Assemble UV trace: reverse(bwd) + seed + fwd
        uv_trace = []
        for i in range(len(bwd) - 1, -1, -1):
            uv_trace.append(bwd[i])
        uv_trace.append((seed[0], seed[1]))
        for p in fwd:
            uv_trace.append(p)

        if len(uv_trace) < 4:
            continue

        p_first = surface.point_at(uv_trace[0][0], uv_trace[0][1])
        p_last = surface.point_at(uv_trace[-1][0], uv_trace[-1][1])
        is_loop = fwd_closed or (len(uv_trace) >= 6 and p_first.distance(p_last) < close_tol_3d)
        if is_loop:
            uv_trace.pop()
        if len(uv_trace) < 4:
            continue

        # Unwrap UV trace for smooth interpolation across seams
        uv_unwrapped = [list(p) for p in uv_trace]
        for i in range(1, len(uv_unwrapped)):
            du_jump = uv_unwrapped[i][0] - uv_unwrapped[i - 1][0]
            dv_jump = uv_unwrapped[i][1] - uv_unwrapped[i - 1][1]
            if closed_u:
                if du_jump > range_u * 0.5:
                    uv_unwrapped[i][0] -= range_u
                elif du_jump < -range_u * 0.5:
                    uv_unwrapped[i][0] += range_u
            if closed_v:
                if dv_jump > range_v * 0.5:
                    uv_unwrapped[i][1] -= range_v
                elif dv_jump < -range_v * 0.5:
                    uv_unwrapped[i][1] += range_v

        # 3. Evaluate all trace points to 3D
        all_pts = [surface.point_at(uv[0], uv[1]) for uv in uv_trace]

        # 4. Circle detection: if points lie on a circle -> exact rational NURBS
        crv = NurbsCurve()
        if is_loop and len(all_pts) >= 6:
            ax = plane.x_axis
            ay = plane.y_axis
            po = plane.origin

            def to2d_circle(p):
                dx = p[0] - po[0]
                dy = p[1] - po[1]
                dz = p[2] - po[2]
                return (dx*ax[0] + dy*ax[1] + dz*ax[2],
                        dx*ay[0] + dy*ay[1] + dz*ay[2])

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
                    knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
                    for i in range(10):
                        crv.set_knot(i, knots[i])
                    for i in range(9):
                        px = cx3d + radius * (cx_[i] * ax[0] + cy_[i] * ay[0])
                        py = cy3d + radius * (cx_[i] * ax[1] + cy_[i] * ay[1])
                        pz = cz3d + radius * (cx_[i] * ax[2] + cy_[i] * ay[2])
                        crv.set_cv_4d(i, px * wts[i], py * wts[i], pz * wts[i], wts[i])

        # 4b. Ellipse (conic) detection for non-circular closed curves
        if not crv.is_valid() and is_loop and len(all_pts) >= 8:
            ax = plane.x_axis
            ay = plane.y_axis
            po = plane.origin

            def to2d_ellipse(p):
                dx = p[0] - po[0]
                dy = p[1] - po[1]
                dz = p[2] - po[2]
                return (dx*ax[0] + dy*ax[1] + dz*ax[2],
                        dx*ay[0] + dy*ay[1] + dz*ay[2])

            n = len(all_pts)
            AtA = [[0.0]*5 for _ in range(5)]
            Atb = [0.0]*5
            for i in range(n):
                x, y = to2d_ellipse(all_pts[i])
                row = [x*x, x*y, y*y, x, y]
                for r in range(5):
                    Atb[r] += row[r]
                    for c in range(5):
                        AtA[r][c] += row[r] * row[c]

            M = [[0.0]*6 for _ in range(5)]
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

            coef = [0.0]*5
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
                    val = A_c*x*x + B_c*x*y + C_c*y*y + D_c*x + E_c*y - 1.0
                    max_conic_dev = max(max_conic_dev, math.fabs(val))

                scale = max(math.fabs(A_c), math.fabs(C_c))
                norm_dev = max_conic_dev / max(scale, 1e-10)

                if norm_dev < 0.01:
                    det = 4*A_c*C_c - B_c*B_c
                    cx = (B_c*E_c - 2*C_c*D_c) / det
                    cy = (B_c*D_c - 2*A_c*E_c) / det

                    theta = 0.5 * math.atan2(B_c, A_c - C_c)
                    cos_t = math.cos(theta)
                    sin_t = math.sin(theta)
                    A2 = A_c*cos_t*cos_t + B_c*cos_t*sin_t + C_c*sin_t*sin_t
                    C2 = A_c*sin_t*sin_t - B_c*cos_t*sin_t + C_c*cos_t*cos_t
                    f_val = A_c*cx*cx + B_c*cx*cy + C_c*cy*cy + D_c*cx + E_c*cy - 1.0
                    rhs = -f_val

                    if rhs > 1e-14 and A2 > 1e-14 and C2 > 1e-14:
                        semi_a = math.sqrt(rhs / A2)
                        semi_b = math.sqrt(rhs / C2)

                        cx3d = po[0] + cx*ax[0] + cy*ay[0]
                        cy3d = po[1] + cx*ax[1] + cy*ay[1]
                        cz3d = po[2] + cx*ax[2] + cy*ay[2]

                        ea = Vector(cos_t*ax[0]+sin_t*ay[0], cos_t*ax[1]+sin_t*ay[1], cos_t*ax[2]+sin_t*ay[2])
                        eb = Vector(-sin_t*ax[0]+cos_t*ay[0], -sin_t*ax[1]+cos_t*ay[1], -sin_t*ax[2]+cos_t*ay[2])

                        w = math.sqrt(2.0) / 2.0
                        cx_ = [1, 1, 0, -1, -1, -1, 0, 1, 1]
                        cy_ = [0, 1, 1, 1, 0, -1, -1, -1, 0]
                        wts = [1, w, 1, w, 1, w, 1, w, 1]
                        crv = NurbsCurve(3, True, 3, 9)
                        knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
                        for i in range(10):
                            crv.set_knot(i, knots[i])
                        for i in range(9):
                            px = cx3d + semi_a*cx_[i]*ea[0] + semi_b*cy_[i]*eb[0]
                            py = cy3d + semi_a*cx_[i]*ea[1] + semi_b*cy_[i]*eb[1]
                            pz = cz3d + semi_a*cx_[i]*ea[2] + semi_b*cy_[i]*eb[2]
                            crv.set_cv_4d(i, px*wts[i], py*wts[i], pz*wts[i], wts[i])

                        et0, et1 = crv.domain()
                        max_ell_dev = 0.0
                        for p in all_pts:
                            px2, py2 = to2d_ellipse(p)
                            lx = cos_t*(px2-cx) + sin_t*(py2-cy)
                            ly = -sin_t*(px2-cx) + cos_t*(py2-cy)
                            ang = math.atan2(ly/semi_b, lx/semi_a)
                            ex = cx + semi_a*math.cos(ang)*cos_t - semi_b*math.sin(ang)*sin_t
                            ey = cy + semi_a*math.cos(ang)*sin_t + semi_b*math.sin(ang)*cos_t
                            dev = math.hypot(px2-ex, py2-ey)
                            max_ell_dev = max(max_ell_dev, dev)
                        ell_tol = max(semi_a, semi_b) * 5e-3
                        if max_ell_dev > ell_tol:
                            crv = NurbsCurve()

        # 5. 2D plane-constrained fitting for non-circular/elliptical curves
        if not crv.is_valid():
            m = len(all_pts)
            if m < 4:
                continue

            ax = plane.x_axis
            ay = plane.y_axis
            po = plane.origin
            pts_2d = []
            for i in range(m):
                dx = all_pts[i][0]-po[0]
                dy = all_pts[i][1]-po[1]
                dz = all_pts[i][2]-po[2]
                px = dx*ax[0] + dy*ax[1] + dz*ax[2]
                py = dx*ay[0] + dy*ay[1] + dz*ay[2]
                pts_2d.append(Point(px, py, 0))

            chords = [0.0]*m
            total_len = 0.0
            for i in range(1, m):
                total_len += pts_2d[i].distance(pts_2d[i-1])
                chords[i] = total_len
            if is_loop and m > 1:
                total_len += pts_2d[0].distance(pts_2d[m-1])
            if total_len > 1e-14:
                for i in range(1, m):
                    chords[i] /= total_len

            fit_tol = step * (uv_to_3d + uv_to_3d_min) * 0.5
            total_turning = 0.0
            for i in range(1, m - 1):
                dx1 = pts_2d[i][0]-pts_2d[i-1][0]
                dy1 = pts_2d[i][1]-pts_2d[i-1][1]
                dx2 = pts_2d[i+1][0]-pts_2d[i][0]
                dy2 = pts_2d[i+1][1]-pts_2d[i][1]
                l1 = math.hypot(dx1, dy1)
                l2 = math.hypot(dx2, dy2)
                if l1 > 1e-14 and l2 > 1e-14:
                    c = (dx1*dx2+dy1*dy2) / (l1*l2)
                    c = max(-1.0, min(1.0, c))
                    total_turning += math.acos(c)

            target_cvs = max(8, int(total_turning / 0.5) + 6)
            max_cvs = m - 1
            crv_2d = NurbsCurve()
            for attempt in range(5):
                if target_cvs > max_cvs:
                    break
                crv_2d = NurbsCurve.create_fitted(pts_2d, target_cvs, 3, is_loop)
                if not crv_2d.is_valid():
                    break
                ft0, ft1 = crv_2d.domain()
                max_dev = 0.0
                for i in range(m):
                    t = ft0 + (ft1 - ft0) * chords[i]
                    max_dev = max(max_dev, crv_2d.point_at(t).distance(pts_2d[i]))
                if max_dev < fit_tol:
                    break
                target_cvs = min(target_cvs * 2, max_cvs)

            if not crv_2d.is_valid():
                if is_loop:
                    crv_2d = NurbsCurve.create_interpolated(pts_2d, CurveKnotStyle.ChordPeriodic)
                else:
                    crv_2d = NurbsCurve.create_interpolated(pts_2d)

            if crv_2d.is_valid():
                crv = crv_2d
                for i in range(crv.cv_count()):
                    cv2 = crv.get_cv(i)
                    cx_l = cv2[0]
                    cy_l = cv2[1]
                    crv.set_cv(i, Point(po[0] + cx_l*ax[0] + cy_l*ay[0],
                                        po[1] + cx_l*ax[1] + cy_l*ay[1],
                                        po[2] + cx_l*ax[2] + cy_l*ay[2]))

        if not crv.is_valid():
            continue

        # Deduplicate: skip if ALL sample points are close to an existing curve
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
