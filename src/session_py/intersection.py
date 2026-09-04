from __future__ import annotations
"""
Intersection functions for geometric primitives.

This module provides intersection calculations between various geometric objects
including lines, planes, rays, boxes, spheres, triangles, and meshes.
"""

from typing import Optional
from typing import Tuple
from typing import List
from typing import TYPE_CHECKING
from .line import Line
from .point import Point
from .obb import OBB
from .mesh import Mesh
from .spatial_bvh import SpatialBVH
from .closest import Closest

if TYPE_CHECKING:
    from .element import Element
    from .nurbssurface import NurbsSurface
    from .nurbscurve import NurbsCurve
    from .plane import Plane
    from .polyline import Polyline


def line_line_parameters(
    line0: Line,
    line1: Line,
    tolerance: float,
    intersect_segments: bool = True,
    near_parallel_as_closest: bool = False,
) -> tuple[float, float] | None:
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


def line_line(line0: Line, line1: Line, tolerance: float) -> Point | None:
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


def plane_plane(plane0: "Plane", plane1: "Plane") -> Line | None:
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


def plane_plane_to_line_canonical(plane0: "Plane", plane1: "Plane") -> Line | None:
    # CGAL-canonical anchor (foot-of-perpendicular from world origin) used by
    # wood's cgal::intersection_util::plane_plane. Independent of input-plane
    # origin choice, giving bit-exact match to wood for parallel input planes.
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


def plane_value_at(plane: "Plane", point: Point) -> float:
    """Calculate the plane equation value at a point"""
    return plane.a * point[0] + plane.b * point[1] + plane.c * point[2] + plane.d


def line_plane(line: Line, plane: "Plane", is_finite: bool = True) -> Point | None:
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


def plane_plane_plane(plane0: "Plane", plane1: "Plane", plane2: "Plane") -> Point | None:
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
    line: Line, box: OBB, t0: float, t1: float
) -> list[Point] | None:
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


def ray_sphere(line: Line, center: Point, radius: float) -> list[Point] | None:
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
) -> Point | None:
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


def _mesh_triangles(mesh: Mesh) -> list[tuple[Point, Point, Point]]:
    vertices, faces = mesh.to_vertices_and_faces()
    tris: list[tuple[Point, Point, Point]] = []
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
) -> list[Point] | None:
    tris = _mesh_triangles(mesh)
    if not tris:
        return None

    hits: list[tuple[float, Point]] = []
    origin = line.start()
    direction = line.to_vector().normalized()

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
) -> list[Point] | None:
    tris = _mesh_triangles(mesh)
    if not tris:
        return None

    # Build AABBs for triangles
    tri_boxes: list[OBB] = []
    for v0, v1, v2 in tris:
        tri_boxes.append(OBB.from_points([v0, v1, v2]))

    world_size = SpatialBVH.compute_world_size(tri_boxes)
    bvh = SpatialBVH.from_boxes(tri_boxes, world_size)

    origin = line.start()
    direction = line.to_vector().normalized()
    candidate_ids: list[int] = []
    found = bvh.ray_cast(origin, direction, candidate_ids, True)
    if not found:
        return None

    hits: list[tuple[float, Point]] = []
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


# ═══════════════════════════════════════════════════════════════════════════
# NURBS Curve Intersection Functions
# ═══════════════════════════════════════════════════════════════════════════

from .vector import Vector
from .tolerance import Tolerance


def _curve_signed_distance_to_plane(pt, plane):
    """Signed distance from point to plane."""
    v = Vector(pt[0] - plane.origin[0], pt[1] - plane.origin[1], pt[2] - plane.origin[2])
    return v.dot(plane.z_axis)


def _curve_find_root_bisection(curve, plane, t0, t1, tolerance):
    """Bisection root find; returns (found, t_result)."""
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
    return abs(_curve_signed_distance_to_plane(curve.point_at(t_result), plane)) < tolerance * 10.0, t_result


def _curve_refine_intersection_newton(curve, plane, t, tolerance):
    """Newton polish of a root; returns the refined t."""
    max_iterations = 10
    step_tolerance = tolerance * 0.01

    for _ in range(max_iterations):
        pt = curve.point_at(t)
        tangent = curve.tangent_at(t)

        f = _curve_signed_distance_to_plane(pt, plane)
        df = tangent.dot(plane.z_axis)

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


def curve_plane(curve: "NurbsCurve", plane: "Plane", tolerance: float | None=None) -> list[float]:
    """Find all intersections between NURBS curve and plane."""
    intersections = []

    if not curve.is_valid():
        return intersections
    if tolerance is None or tolerance <= 0.0:
        tolerance = Tolerance.ZERO_TOLERANCE

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
            found, t_intersection = _curve_find_root_bisection(curve, plane, t0, t1, tolerance)
            if found:
                t_intersection = _curve_refine_intersection_newton(curve, plane, t_intersection, tolerance)
                intersections.append(t_intersection)
        elif abs(d0) < tolerance:
            add = True
            if intersections and abs(intersections[-1] - t0) < tolerance:
                add = False
            if add:
                intersections.append(t0)

    d_end = _curve_signed_distance_to_plane(curve.point_at(t_end), plane)
    if abs(d_end) < tolerance:
        add = True
        if intersections and abs(intersections[-1] - t_end) < tolerance:
            add = False
        if add:
            intersections.append(t_end)

    if curve.degree() > 3 and len(intersections) < curve.degree():
        num_samples = curve.degree() * 4
        dt = (t_end - t_start) / num_samples

        for i in range(num_samples):
            t0 = t_start + i * dt
            t1 = t_start + (i + 1) * dt

            d0 = _curve_signed_distance_to_plane(curve.point_at(t0), plane)
            d1 = _curve_signed_distance_to_plane(curve.point_at(t1), plane)

            if d0 * d1 < 0:
                found, t_intersection = _curve_find_root_bisection(curve, plane, t0, t1, tolerance)
                if found:
                    is_new = True
                    for existing in intersections:
                        if abs(existing - t_intersection) < tolerance * 2.0:
                            is_new = False
                            break
                    if is_new:
                        t_intersection = _curve_refine_intersection_newton(curve, plane, t_intersection, tolerance)
                        intersections.append(t_intersection)

    intersections.sort()
    if len(intersections) > 1:
        unique_results = [intersections[0]]
        for i in range(1, len(intersections)):
            if abs(intersections[i] - unique_results[-1]) >= tolerance * 2.0:
                unique_results.append(intersections[i])
        intersections = unique_results

    return intersections


def curve_plane_points(curve: "NurbsCurve", plane: "Plane", tolerance: float | None=None) -> list[Point]:
    """Find all intersection points between NURBS curve and plane."""
    params = curve_plane(curve, plane, tolerance)
    return [curve.point_at(t) for t in params]


def curve_plane_bezier_clipping(curve: "NurbsCurve", plane: "Plane", tolerance: float | None=None) -> list[float]:
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


def curve_plane_algebraic(curve: "NurbsCurve", plane: "Plane", tolerance: float | None=None) -> list[float]:
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


def curve_plane_production(curve: "NurbsCurve", plane: "Plane", tolerance: float | None=None) -> list[float]:
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


def curve_closest_point(curve: "NurbsCurve", test_point: Point, t0: float = 0.0, t1: float = 0.0) -> tuple[float, float]:
    """Find closest point on NURBS curve to test point."""
    return Closest.curve_point(curve, test_point, t0, t1)


def _surface_plane_traces(surface, plane, tolerance):
    """Seed and trace surface/plane intersection curves in UV space.

    Returns (traces, step, uv_to_3d, uv_to_3d_min) where traces is a list of
    (uv_trace, uv_unwrapped, is_loop): wrapped UV samples, seam-unwrapped UV
    samples, and whether the trace is a closed loop.
    """
    import math

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

    traces = []

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

        traces.append((uv_trace, uv_unwrapped, is_loop))

    return traces, step, uv_to_3d, uv_to_3d_min


def _surface_plane_fit_3d(all_pts, is_loop, plane, step, uv_to_3d, uv_to_3d_min, allow_conics=True):
    """Fit a 3D plane-constrained NurbsCurve to traced intersection points.

    Tries exact circle recognition, then ellipse recognition for closed loops
    (when allow_conics), then adaptive plane-constrained least-squares fitting.
    Returns an invalid curve on failure.
    """
    import math
    from .nurbscurve import NurbsCurve
    from .nurbsknot import CurveNurbsKnotStyle

    # 4. Circle detection: if points lie on a circle -> exact rational NURBS
    crv = NurbsCurve()
    if allow_conics and is_loop and len(all_pts) >= 6:
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
                nurbsknots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
                for i in range(10):
                    crv.set_nurbsknot(i, nurbsknots[i])
                for i in range(9):
                    px = cx3d + radius * (cx_[i] * ax[0] + cy_[i] * ay[0])
                    py = cy3d + radius * (cx_[i] * ax[1] + cy_[i] * ay[1])
                    pz = cz3d + radius * (cx_[i] * ax[2] + cy_[i] * ay[2])
                    crv.set_cv_4d(i, px * wts[i], py * wts[i], pz * wts[i], wts[i])

    # 4b. Ellipse (conic) detection for non-circular closed curves
    if not crv.is_valid() and allow_conics and is_loop and len(all_pts) >= 8:
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
                    nurbsknots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
                    for i in range(10):
                        crv.set_nurbsknot(i, nurbsknots[i])
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
            return NurbsCurve()

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
                crv_2d = NurbsCurve.create_interpolated(pts_2d, CurveNurbsKnotStyle.ChordPeriodic)
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

    return crv


def surface_plane(surface: "NurbsSurface", plane: "Plane", tolerance: float | None=None) -> list["NurbsCurve"]:
    """Find intersection curves between a NURBS surface and a plane."""
    if not surface.is_valid():
        return []
    if tolerance is None or tolerance <= 0.0:
        tolerance = Tolerance.ZERO_TOLERANCE

    traces, step, uv_to_3d, uv_to_3d_min = _surface_plane_traces(surface, plane, tolerance)

    result = []
    for uv_trace, uv_unwrapped, is_loop in traces:
        all_pts = [surface.point_at(uv[0], uv[1]) for uv in uv_trace]
        crv = _surface_plane_fit_3d(all_pts, is_loop, plane, step, uv_to_3d, uv_to_3d_min)
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


def _clip_pcurve_to_cutter(target, pc, cutter):
    """Keep only the sub-segments of a UV pcurve on `target` whose lifted 3D
    point projects within the cutter surface's domain (the cutter footprint)."""
    from .closest import Closest
    n = max(pc.cv_count() * 4, 16)
    d0, d1 = pc.domain()
    cu0, cu1 = cutter.domain(0)
    cv0, cv1 = cutter.domain(1)
    # A cut point lies within the (bounded) cutter footprint iff its closest
    # point on the cutter is essentially coincident: surface_point clamps the
    # projection to the domain, so an out-of-footprint point projects to the
    # boundary with a non-zero gap. Gate on that gap, not on the clamped (u,v).
    corner_diag = cutter.point_at(cu0, cv0).distance(cutter.point_at(cu1, cv1))
    on_tol = max(1e-7, corner_diag * 1e-4)

    # _clip_pcurve_to_cutter is only ever called for a PLANAR cutter (the caller guards on
    # cutter.is_planar), so the closest-point gap to the finite cutter face is the analytic
    # point-to-rectangle distance: project p3 into the face's (eu,ev) frame, clamp the
    # parameters to the face rect, measure the 3D residual. This replaces a per-sample grid
    # search (Closest.surface_point) that dominated SSI time, with an O(1) projection.
    q00 = cutter.point_at(cu0, cv0)
    q10 = cutter.point_at(cu1, cv0)
    q01 = cutter.point_at(cu0, cv1)
    eu0, eu1, eu2_ = q10[0]-q00[0], q10[1]-q00[1], q10[2]-q00[2]
    ev0, ev1, ev2_ = q01[0]-q00[0], q01[1]-q00[1], q01[2]-q00[2]
    eu_sq = eu0*eu0 + eu1*eu1 + eu2_*eu2_
    ev_sq = ev0*ev0 + ev1*ev1 + ev2_*ev2_
    q00x, q00y, q00z = q00[0], q00[1], q00[2]
    fast_planar = eu_sq > 1e-28 and ev_sq > 1e-28

    def gap(t):
        uv = pc.point_at(t)
        p3 = target.point_at(uv[0], uv[1])
        if fast_planar:
            dx = p3[0]-q00x; dy = p3[1]-q00y; dz = p3[2]-q00z
            a = (dx*eu0 + dy*eu1 + dz*eu2_) / eu_sq
            b = (dx*ev0 + dy*ev1 + dz*ev2_) / ev_sq
            if a < 0.0: a = 0.0
            elif a > 1.0: a = 1.0
            if b < 0.0: b = 0.0
            elif b > 1.0: b = 1.0
            cx = q00x + a*eu0 + b*ev0
            cy = q00y + a*eu1 + b*ev1
            cz = q00z + a*eu2_ + b*ev2_
            return ((p3[0]-cx)**2 + (p3[1]-cy)**2 + (p3[2]-cz)**2) ** 0.5
        return Closest.surface_point(cutter, p3, 0.0, 0.0, 0.0, 0.0)[2]

    def refine(t_in, t_out):
        # Bisect for the footprint boundary (gap crosses on_tol) so adjacent
        # clipped segments meet exactly at the cutter corner.
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


def _analytic_sphere_pullback(srf, recog, c3d):
    """Analytic pull-back of a 3D curve onto a recognized SPHERE, replicating OCCT
    ProjLib_Sphere's per-point inverse (EvalPnt2d): in the sphere's local frame the
    longitude = atan2(y, x) is EXACT (so a seam-straddling circle crosses the u-seam
    EXACTLY on u=u0/u=u1 -- the thing the iterative projector got ~0.18 wrong), and the
    nonlinear meridian v is found by table inversion. Returns the seam-split arcs (a
    circle straddling the seam -> 2 arcs, each anchored on the seam) as exact-endpoint
    degree-1 polylines. Empty if not a usable sphere/circle.
    """
    import math
    from .nurbscurve import NurbsCurve
    from .point import Point

    if recog is None or recog[0] != 'sphere':
        return []
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    range_u = u1 - u0
    if range_u < 1e-9:
        return []

    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    C = recog[1]
    um = 0.5*(u0 + u1); vm = 0.5*(v0 + v1)
    # Polar axis Zs (south->north pole), and the equatorial frame Xs (u=u0 meridian dir), Ys.
    sp = srf.point_at(um, v0); npp = srf.point_at(um, v1)
    Zs = [npp[0]-sp[0], npp[1]-sp[1], npp[2]-sp[2]]
    zn = math.sqrt(dot(Zs, Zs))
    if zn < 1e-12:
        return []
    Zs = [Zs[0]/zn, Zs[1]/zn, Zs[2]/zn]
    P0 = srf.point_at(u0, vm)
    x0 = [P0[0]-C[0], P0[1]-C[1], P0[2]-C[2]]
    h0 = dot(x0, Zs)
    Xs = [x0[0]-h0*Zs[0], x0[1]-h0*Zs[1], x0[2]-h0*Zs[2]]
    xn = math.sqrt(dot(Xs, Xs))
    if xn < 1e-12:
        return []
    Xs = [Xs[0]/xn, Xs[1]/xn, Xs[2]/xn]
    Ys = [Zs[1]*Xs[2]-Zs[2]*Xs[1], Zs[2]*Xs[0]-Zs[0]*Xs[2], Zs[0]*Xs[1]-Zs[1]*Xs[0]]
    PI = math.pi; TWO_PI = 2.0*PI
    # (u -> longitude) table along the equator. The NURBS sphere's u is the RATIONAL-quadratic
    # circle parameter, which is NOT linear in longitude (only correct at 45-deg multiples) -- a
    # linear u = u0 + (lon/2pi)*range_u approximation distorts the pulled-back circle so it bounds
    # ~2% too little flux (wrong volume). Invert the true parametrization: tabulate longitude(u)
    # on the equator (v-independent for a surface of revolution), then binary-search per point.
    NT = 128
    tu = [0.0]*(NT+1); tlon = [0.0]*(NT+1)
    for k in range(NT+1):
        u = u0 + range_u*k/NT
        p = srf.point_at(u, vm)
        r = [p[0]-C[0], p[1]-C[1], p[2]-C[2]]
        lon = math.atan2(dot(r, Ys), dot(r, Xs))
        if k > 0:
            while lon - tlon[k-1] > PI:
                lon -= TWO_PI
            while lon - tlon[k-1] < -PI:
                lon += TWO_PI
        tu[k] = u; tlon[k] = lon
    lon_incr = tlon[NT] >= tlon[0]
    lon_lo = min(tlon[0], tlon[NT]); lon_hi = max(tlon[0], tlon[NT])

    def u_from_lon(lon):
        while lon < lon_lo - 1e-9:
            lon += TWO_PI
        while lon > lon_hi + 1e-9:
            lon -= TWO_PI
        lo, hi = 0, NT
        while hi - lo > 1:
            mid = (lo + hi) // 2
            above = (tlon[mid] < lon) if lon_incr else (tlon[mid] > lon)
            if above:
                lo = mid
            else:
                hi = mid
        denom = tlon[hi] - tlon[lo]
        f = (lon - tlon[lo]) / denom if abs(denom) > 1e-15 else 0.0
        return tu[lo] + (tu[hi] - tu[lo])*f

    # height(v) along a meridian (independent of u by sphere symmetry) is MONOTONE pole-to-pole.
    tv = [0.0]*(NT+1); th = [0.0]*(NT+1)
    for k in range(NT+1):
        v = v0 + (v1-v0)*k/NT
        p = srf.point_at(um, v)
        r = [p[0]-C[0], p[1]-C[1], p[2]-C[2]]
        tv[k] = v; th[k] = dot(r, Zs)
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
        lo, hi = 0, NT
        while hi - lo > 1:
            mid = (lo + hi) // 2
            above = (th[mid] < h) if incr else (th[mid] > h)
            if above:
                lo = mid
            else:
                hi = mid
        denom = th[hi] - th[lo]
        f = (h - th[lo]) / denom if abs(denom) > 1e-15 else 0.0
        return tv[lo] + (tv[hi] - tv[lo])*f

    # Sample the 3D curve; project each point analytically -> (u_unwrapped, v).
    t0, t1 = c3d.domain()
    n = max(c3d.cv_count() * 8, 120)
    uv = []
    prev_u = 0.0
    for i in range(n+1):
        p = c3d.point_at(t0 + (t1-t0)*i/n)
        r = [p[0]-C[0], p[1]-C[1], p[2]-C[2]]
        lon = math.atan2(dot(r, Ys), dot(r, Xs))   # (-pi, pi], exact
        h = dot(r, Zs)
        u = u_from_lon(lon)                         # exact NURBS u (not the linear approx)
        if i > 0:
            while u - prev_u > range_u*0.5:
                u -= range_u
            while u - prev_u < -range_u*0.5:
                u += range_u
        prev_u = u
        uv.append((u, v_from_height(h)))
    if len(uv) < 2:
        return []
    # Split the continuous (u,v) polyline into arcs by "domain copy" index k = floor((u-u0)/range).
    # When k changes between consecutive samples the curve crosses a seam: end the current arc
    # EXACTLY on the seam (u0 or u1) and start the next on the opposite seam, each shifted into
    # [u0,u1]. So a circle straddling the seam -> two arcs anchored exactly on u0 and u1.
    out = []
    seg = []

    def kof(u):
        return int(math.floor((u - u0) / range_u + 1e-9))

    cur_k = kof(uv[0][0])
    seg.append(Point(uv[0][0] - cur_k*range_u, uv[0][1], 0.0))
    for i in range(1, len(uv)):
        ki = kof(uv[i][0])
        while ki != cur_k:
            step = 1 if ki > cur_k else -1
            nk = cur_k + step
            seam_cont = u0 + (nk if step > 0 else cur_k) * range_u  # the boundary being crossed
            denom = uv[i][0] - uv[i-1][0]
            f = (seam_cont - uv[i-1][0]) / denom if abs(denom) > 1e-15 else 0.0
            f = min(max(f, 0.0), 1.0)
            vc = uv[i-1][1] + (uv[i][1] - uv[i-1][1])*f
            seg.append(Point(seam_cont - cur_k*range_u, vc, 0.0))   # end at u1 (step>0) or u0
            if len(seg) >= 2:
                out.append(NurbsCurve.create(False, 1, seg))
            seg = []
            seg.append(Point(seam_cont - nk*range_u, vc, 0.0))      # start at u0 (step>0) or u1
            cur_k = nk
        seg.append(Point(uv[i][0] - cur_k*range_u, uv[i][1], 0.0))
    if len(seg) >= 2:
        out.append(NurbsCurve.create(False, 1, seg))
    return out


def cut_curves_on_surface(target: "NurbsSurface", cutter: "NurbsSurface", tolerance: float | None=None) -> list["NurbsCurve"]:
    """Return the cutter surface's UV pcurves on the target surface.

    Fast path: if the cutter is planar, intersect the target with the cutter's
    plane (surface_plane_uv) and clip the result to the cutter footprint.
    Otherwise use the surface/surface intersection (already domain-clipped).

    SPHERE target: mirror the C++ dispatch (intersection.cpp cut_curves_on_surface).
    Pull the FULL analytic 3D intersection curve back via the OCCT-style per-point
    inverse (analytic_sphere_pullback -> exact seam crossings, so a seam-straddling cut
    circle splits into two arcs anchored on u0/u1 instead of being dropped). Fall back to
    the iterative surface_curve projection, then the default pcurve, when that is empty.
    """
    from .plane import Plane
    from .closest import Closest

    rtol = max(tolerance if (tolerance and tolerance > 0) else 1e-7, 1e-7) * 1e4
    rt = _recognize_surface(target, rtol)
    if rt is not None and rt[0] == 'sphere':
        cutter_planar = cutter.is_planar(None, 1e-6)
        out = []
        for triple in surface_surface(target, cutter, tolerance):
            c3d = triple[0]
            pcs = _analytic_sphere_pullback(target, rt, c3d)
            if not pcs:
                pcs = Closest.surface_curve(target, c3d, 0.0, 0.0, tolerance)
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


def surface_plane_uv(surface: "NurbsSurface", plane: "Plane", tolerance: float | None=None) -> list[tuple["NurbsCurve", "NurbsCurve"]]:
    """Find surface/plane intersection curves with their UV pcurves.

    Returns a list of (curve_3d, pcurve) pairs. Pcurves are NurbsCurves in
    parameter space (x=u, y=v, z=0), seam-split so each pcurve is continuous
    inside the surface domain. Both curves are reparameterized to [0, 1] by
    chord length; the pcurve is a tolerance companion of the 3D curve, not an
    exact reparameterization.
    """
    import math
    from .nurbscurve import NurbsCurve
    from .nurbsknot import CurveNurbsKnotStyle

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

    def g_and_grad(u, v):
        derivs = surface.evaluate(wrap_u(u), wrap_v(v), 1)
        S = derivs[0]
        Su = derivs[2]
        Sv = derivs[1]
        val = (S[0]-p0[0])*pn[0] + (S[1]-p0[1])*pn[1] + (S[2]-p0[2])*pn[2]
        gu = Su[0]*pn[0] + Su[1]*pn[1] + Su[2]*pn[2]
        gv = Sv[0]*pn[0] + Sv[1]*pn[1] + Sv[2]*pn[2]
        return val, gu, gv

    def seam_newton(cu, cv_, axis):
        # Refine the free coordinate along a fixed seam iso-line so g = 0
        for _ in range(10):
            val, gu, gv = g_and_grad(cu, cv_)
            if abs(val) < tolerance:
                break
            if axis == 0:
                if abs(gv) < 1e-14:
                    break
                cv_ = cv_ - val / gv
            else:
                if abs(gu) < 1e-14:
                    break
                cu = cu - val / gu
        return cu, cv_

    traces, step, uv_to_3d, uv_to_3d_min = _surface_plane_traces(surface, plane, tolerance)

    fit_tol = step * (uv_to_3d + uv_to_3d_min) * 0.5
    dup_tol = step * uv_to_3d * 3.0

    result = []
    kept_pts3 = []
    for uv_trace, uv_unwrapped, is_loop in traces:
        # Trace-level dedup against already kept traces (3-sample proximity)
        m = len(uv_trace)
        trace_pts3 = [surface.point_at(uv[0], uv[1]) for uv in uv_trace]
        dup = False
        for other in kept_pts3:
            all_close = True
            for f in [0.25, 0.5, 0.75]:
                cp = trace_pts3[int((m - 1) * f)]
                dmin = dup_tol + 1.0
                for k in range(0, len(other), 5):
                    dmin = min(dmin, cp.distance(other[k]))
                if dmin > dup_tol:
                    all_close = False
                    break
            if all_close:
                dup = True
                break
        if dup:
            continue
        kept_pts3.append(trace_pts3)

        # Extend closed loops with a virtual copy of the first point
        pts = [list(p) for p in uv_unwrapped]
        closure_du = 0.0
        closure_dv = 0.0
        if is_loop and len(pts) >= 2:
            du_j = pts[0][0] - pts[-1][0]
            dv_j = pts[0][1] - pts[-1][1]
            if closed_u:
                while du_j > range_u * 0.5:
                    du_j -= range_u
                while du_j < -range_u * 0.5:
                    du_j += range_u
            if closed_v:
                while dv_j > range_v * 0.5:
                    dv_j -= range_v
                while dv_j < -range_v * 0.5:
                    dv_j += range_v
            closure_du = (pts[-1][0] + du_j) - pts[0][0]
            closure_dv = (pts[-1][1] + dv_j) - pts[0][1]
            pts.append([pts[0][0] + closure_du, pts[0][1] + closure_dv])

        # Insert seam crossings (Newton-refined onto the seam iso-line)
        out_pts = [pts[0]]
        cross_idx = []
        for i in range(1, len(pts)):
            pa = pts[i - 1]
            pb = pts[i]
            crossings = []
            if closed_u and abs(pb[0] - pa[0]) > 1e-15:
                k0 = math.floor((pa[0] - u0) / range_u)
                k1 = math.floor((pb[0] - u0) / range_u)
                for k in range(min(k0, k1) + 1, max(k0, k1) + 1):
                    L = u0 + k * range_u
                    t = (L - pa[0]) / (pb[0] - pa[0])
                    if 0.0 < t < 1.0:
                        crossings.append((t, 0, L))
            if closed_v and abs(pb[1] - pa[1]) > 1e-15:
                k0 = math.floor((pa[1] - v0) / range_v)
                k1 = math.floor((pb[1] - v0) / range_v)
                for k in range(min(k0, k1) + 1, max(k0, k1) + 1):
                    L = v0 + k * range_v
                    t = (L - pa[1]) / (pb[1] - pa[1])
                    if 0.0 < t < 1.0:
                        crossings.append((t, 1, L))
            crossings.sort()
            for t, axis, L in crossings:
                cu = pa[0] + (pb[0] - pa[0]) * t
                cv_ = pa[1] + (pb[1] - pa[1]) * t
                if axis == 0:
                    cu_r, cv_r = seam_newton(L, cv_, 0)
                    cu = L
                    cv_ = cv_r
                else:
                    cu_r, cv_r = seam_newton(cu, L, 1)
                    cu = cu_r
                    cv_ = L
                out_pts.append([cu, cv_])
                cross_idx.append(len(out_pts) - 1)
            out_pts.append([pb[0], pb[1]])
            # An interior sample sitting exactly on a seam level is a crossing
            if i < len(pts) - 1:
                on_seam = False
                if closed_u:
                    k = round((pb[0] - u0) / range_u)
                    L = u0 + k * range_u
                    if abs(pb[0] - L) < range_u * 1e-9 and abs(pb[0] - pa[0]) > range_u * 1e-9:
                        out_pts[-1][0] = L
                        on_seam = True
                if closed_v:
                    k = round((pb[1] - v0) / range_v)
                    L = v0 + k * range_v
                    if abs(pb[1] - L) < range_v * 1e-9 and abs(pb[1] - pa[1]) > range_v * 1e-9:
                        out_pts[-1][1] = L
                        on_seam = True
                if on_seam:
                    cross_idx.append(len(out_pts) - 1)

        # Split at seam crossings into continuous UV pieces
        wrap_drift = abs(closure_du) > range_u * 0.5 or abs(closure_dv) > range_v * 0.5
        if len(cross_idx) == 0:
            # A loop with net unwrap drift wraps the seam with endpoints on it:
            # emit as one open piece spanning the full period
            pieces = [(out_pts, is_loop and not wrap_drift)]
        else:
            pieces = []
            if is_loop:
                for a, b in zip(cross_idx, cross_idx[1:]):
                    pieces.append((out_pts[a:b + 1], False))
                wrap_piece = [list(p) for p in out_pts[cross_idx[-1]:]]
                for p in out_pts[1:cross_idx[0] + 1]:
                    wrap_piece.append([p[0] + closure_du, p[1] + closure_dv])
                pieces.append((wrap_piece, False))
            else:
                bounds = [0] + cross_idx + [len(out_pts) - 1]
                for a, b in zip(bounds, bounds[1:]):
                    if b > a:
                        pieces.append((out_pts[a:b + 1], False))

        for piece_pts, piece_loop in pieces:
            if len(piece_pts) < 2:
                continue
            # Shift the piece into the base domain
            mid = piece_pts[len(piece_pts) // 2]
            if closed_u:
                k_u = math.floor((mid[0] - u0) / range_u)
                if k_u != 0:
                    for p in piece_pts:
                        p[0] -= k_u * range_u
            if closed_v:
                k_v = math.floor((mid[1] - v0) / range_v)
                if k_v != 0:
                    for p in piece_pts:
                        p[1] -= k_v * range_v

            pts3 = [surface.point_at(wrap_u(p[0]), wrap_v(p[1])) for p in piece_pts]

            # Fit the 3D curve (plane-constrained; circle/ellipse for full loops)
            crv3 = _surface_plane_fit_3d(pts3, piece_loop, plane, step, uv_to_3d, uv_to_3d_min, False)
            if not crv3.is_valid():
                if piece_loop:
                    crv3 = NurbsCurve.create_interpolated(pts3, CurveNurbsKnotStyle.ChordPeriodic)
                else:
                    crv3 = NurbsCurve.create_interpolated(pts3)
            if not crv3.is_valid():
                continue

            # Fit the UV pcurve
            pts_uv = [Point(p[0], p[1], 0.0) for p in piece_pts]
            mp = len(pts_uv)
            fit_tol_uv = step
            total_turning = 0.0
            for i in range(1, mp - 1):
                dx1 = pts_uv[i][0] - pts_uv[i-1][0]
                dy1 = pts_uv[i][1] - pts_uv[i-1][1]
                dx2 = pts_uv[i+1][0] - pts_uv[i][0]
                dy2 = pts_uv[i+1][1] - pts_uv[i][1]
                l1 = math.hypot(dx1, dy1)
                l2 = math.hypot(dx2, dy2)
                if l1 > 1e-14 and l2 > 1e-14:
                    c = (dx1*dx2 + dy1*dy2) / (l1*l2)
                    c = max(-1.0, min(1.0, c))
                    total_turning += math.acos(c)

            chords = [0.0] * mp
            total_len = 0.0
            for i in range(1, mp):
                total_len += pts_uv[i].distance(pts_uv[i-1])
                chords[i] = total_len
            if piece_loop and mp > 1:
                total_len += pts_uv[0].distance(pts_uv[mp-1])
            if total_len > 1e-14:
                for i in range(1, mp):
                    chords[i] /= total_len

            target_cvs = max(8, int(total_turning / 0.5) + 6)
            max_cvs = mp - 1
            pcurve = NurbsCurve()
            for attempt in range(5):
                if target_cvs > max_cvs:
                    break
                pcurve = NurbsCurve.create_fitted(pts_uv, target_cvs, 3, piece_loop)
                if not pcurve.is_valid():
                    break
                ft0, ft1 = pcurve.domain()
                max_dev = 0.0
                for i in range(mp):
                    t = ft0 + (ft1 - ft0) * chords[i]
                    max_dev = max(max_dev, pcurve.point_at(t).distance(pts_uv[i]))
                if max_dev < fit_tol_uv:
                    break
                target_cvs = min(target_cvs * 2, max_cvs)

            if not pcurve.is_valid():
                if piece_loop:
                    pcurve = NurbsCurve.create_interpolated(pts_uv, CurveNurbsKnotStyle.ChordPeriodic)
                else:
                    pcurve = NurbsCurve.create_interpolated(pts_uv)
            if not pcurve.is_valid():
                continue

            crv3.set_domain(0.0, 1.0)
            pcurve.set_domain(0.0, 1.0)

            # Validate: lifted pcurve must stay on the plane within the fit budget
            vali_tol = max(10.0 * tolerance, fit_tol * 2.0)
            max_off = 0.0
            for i in range(17):
                t = i / 16.0
                pc = pcurve.point_at(t)
                val, gu, gv = g_and_grad(pc[0], pc[1])
                max_off = max(max_off, abs(val))
            if max_off > vali_tol and target_cvs * 2 <= max_cvs:
                refit = NurbsCurve.create_fitted(pts_uv, target_cvs * 2, 3, piece_loop)
                if refit.is_valid():
                    refit.set_domain(0.0, 1.0)
                    pcurve = refit

            result.append((crv3, pcurve))

    return result


def _solve_gauss(M, rhs, n):
    """Solve an n x n linear system by Gaussian elimination with pivoting."""
    import math
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


def _ortho_basis(n):
    """Return two unit vectors spanning the plane perpendicular to unit n."""
    import math
    ax = 1.0 if abs(n[0]) <= abs(n[1]) and abs(n[0]) <= abs(n[2]) else 0.0
    ay = 1.0 if ax == 0.0 and abs(n[1]) <= abs(n[2]) else 0.0
    az = 1.0 if ax == 0.0 and ay == 0.0 else 0.0
    # u = (ax,ay,az) x n, then v = n x u
    ux = ay*n[2] - az*n[1]
    uy = az*n[0] - ax*n[2]
    uz = ax*n[1] - ay*n[0]
    ul = math.sqrt(ux*ux + uy*uy + uz*uz)
    ux, uy, uz = ux/ul, uy/ul, uz/ul
    vx = n[1]*uz - n[2]*uy
    vy = n[2]*ux - n[0]*uz
    vz = n[0]*uy - n[1]*ux
    return (ux, uy, uz), (vx, vy, vz)


def _exact_circle(cx, cy, cz, xa, ya, radius):
    """Exact 9-CV rational NURBS circle: center (cx,cy,cz), in-plane orthonormal
    axes xa, ya, given radius. Geometrically exact (not a fit)."""
    import math
    from .nurbscurve import NurbsCurve
    w = math.sqrt(2.0) / 2.0
    px = [1, 1, 0, -1, -1, -1, 0, 1, 1]
    py = [0, 1, 1, 1, 0, -1, -1, -1, 0]
    wts = [1, w, 1, w, 1, w, 1, w, 1]
    crv = NurbsCurve(3, True, 3, 9)
    knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    for i in range(10):
        crv.set_nurbsknot(i, float(knots[i]))
    for i in range(9):
        x = cx + radius * (px[i]*xa[0] + py[i]*ya[0])
        y = cy + radius * (px[i]*xa[1] + py[i]*ya[1])
        z = cz + radius * (px[i]*xa[2] + py[i]*ya[2])
        crv.set_cv_4d(i, x*wts[i], y*wts[i], z*wts[i], wts[i])
    crv.set_domain(0.0, 1.0)
    return crv


def _jacobi_eig3(M):
    """Eigenvalues/vectors of a symmetric 3x3 matrix (cyclic Jacobi).
    Returns (eigvals, eigvecs) with eigvecs[k] the unit vector for eigvals[k]."""
    import math
    a = [[M[r][c] for c in range(3)] for r in range(3)]
    v = [[1.0 if r == c else 0.0 for c in range(3)] for r in range(3)]
    for _ in range(50):
        off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])
        if off < 1e-18:
            break
        for (p, q) in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-300:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta*theta + 1.0))
            c = 1.0 / math.sqrt(t*t + 1.0)
            s = t * c
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = c*akp - s*akq
                a[k][q] = s*akp + c*akq
            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k] = c*apk - s*aqk
                a[q][k] = s*apk + c*aqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c*vkp - s*vkq
                v[k][q] = s*vkp + c*vkq
    eigvals = [a[0][0], a[1][1], a[2][2]]
    eigvecs = [(v[0][k], v[1][k], v[2][k]) for k in range(3)]
    return eigvals, eigvecs


def _exact_ellipse(cx, cy, cz, ea, eb, semi_a, semi_b):
    """Exact 9-CV rational NURBS ellipse: center, in-plane orthonormal axes
    ea/eb, semi-axes semi_a/semi_b. Geometrically exact."""
    import math
    from .nurbscurve import NurbsCurve
    w = math.sqrt(2.0) / 2.0
    px = [1, 1, 0, -1, -1, -1, 0, 1, 1]
    py = [0, 1, 1, 1, 0, -1, -1, -1, 0]
    wts = [1, w, 1, w, 1, w, 1, w, 1]
    crv = NurbsCurve(3, True, 3, 9)
    knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    for i in range(10):
        crv.set_nurbsknot(i, float(knots[i]))
    for i in range(9):
        x = cx + semi_a*px[i]*ea[0] + semi_b*py[i]*eb[0]
        y = cy + semi_a*px[i]*ea[1] + semi_b*py[i]*eb[1]
        z = cz + semi_a*px[i]*ea[2] + semi_b*py[i]*eb[2]
        crv.set_cv_4d(i, x*wts[i], y*wts[i], z*wts[i], wts[i])
    crv.set_domain(0.0, 1.0)
    return crv


def _fit_cylinder(surface, tol):
    """Recognize a circular cylinder: axis via the min-variance direction of the
    surface normals, radius via distance of points to the axis line. Returns
    (axis_pt, axis_dir, radius) or None."""
    import math
    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    pts = []
    nrm = []
    for i in range(5):
        for j in range(5):
            uu = u0 + (u1-u0)*i/4.0
            vv = v0 + (v1-v0)*j/4.0
            pts.append(surface.point_at(uu, vv))
            n = surface.normal_at(uu, vv)
            nrm.append((n[0], n[1], n[2]))
    # Axis = eigenvector of the smallest eigenvalue of sum(n n^T) (normals are
    # perpendicular to the axis, so variance along the axis is ~0).
    M = [[0.0]*3 for _ in range(3)]
    for n in nrm:
        for r in range(3):
            for c in range(3):
                M[r][c] += n[r]*n[c]
    evals, evecs = _jacobi_eig3(M)
    kmin = min(range(3), key=lambda k: evals[k])
    w = evecs[kmin]
    wl = math.sqrt(w[0]**2 + w[1]**2 + w[2]**2)
    if wl < 1e-12:
        return None
    w = (w[0]/wl, w[1]/wl, w[2]/wl)
    # Project points onto the plane perpendicular to the axis and fit a 2D
    # circle there (unbiased; a centroid is skewed by non-uniform/seam sampling).
    ea, eb = _ortho_basis(w)
    p0 = pts[0]
    ata = [[0.0]*3 for _ in range(3)]
    atb = [0.0]*3
    proj = []
    for p in pts:
        dp = (p[0]-p0[0], p[1]-p0[1], p[2]-p0[2])
        x = dp[0]*ea[0] + dp[1]*ea[1] + dp[2]*ea[2]
        y = dp[0]*eb[0] + dp[1]*eb[1] + dp[2]*eb[2]
        proj.append((x, y))
        row = [x, y, 1.0]
        rhs = -(x*x + y*y)
        for r in range(3):
            atb[r] += row[r]*rhs
            for c in range(3):
                ata[r][c] += row[r]*row[c]
    sol = _solve_gauss(ata, atb, 3)
    if sol is None:
        return None
    ccx, ccy = -sol[0]/2.0, -sol[1]/2.0
    r2 = ccx*ccx + ccy*ccy - sol[2]
    if r2 <= 1e-18:
        return None
    r = math.sqrt(r2)
    for (x, y) in proj:
        if abs(math.sqrt((x-ccx)**2 + (y-ccy)**2) - r) > tol:
            return None
    axis_pt = (p0[0] + ccx*ea[0] + ccy*eb[0],
               p0[1] + ccx*ea[1] + ccy*eb[1],
               p0[2] + ccx*ea[2] + ccy*eb[2])
    return (axis_pt, w, r)


def _fit_cone(surface, tol):
    """Recognize a circular cone. Apex from the tangent-plane condition
    n.(V-p)=0 (least squares), axis from the mean generator, half-angle from the
    mean apex-to-point angle. Returns (apex, axis, half_angle) or None."""
    import math
    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    # Sample the closed u-direction over [u0,u1) (8 distinct angles, no seam
    # duplicate, which would bias the covariance-based axis estimate).
    pts = []
    nrm = []
    nu_s = 8
    for i in range(nu_s):
        uu = u0 + (u1-u0)*i/nu_s
        for j in range(5):
            vv = v0 + (v1-v0)*j/4.0
            pts.append(surface.point_at(uu, vv))
            n = surface.normal_at(uu, vv)
            nl = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
            if nl < 1e-12:
                continue
            nrm.append(((n[0]/nl, n[1]/nl, n[2]/nl), surface.point_at(uu, vv)))
    if len(nrm) < 4:
        return None
    ata = [[0.0]*3 for _ in range(3)]
    atb = [0.0]*3
    for n, p in nrm:
        npd = n[0]*p[0] + n[1]*p[1] + n[2]*p[2]
        for r in range(3):
            atb[r] += n[r]*npd
            for c in range(3):
                ata[r][c] += n[r]*n[c]
    V = _solve_gauss(ata, atb, 3)
    if V is None:
        return None
    gs = []
    for p in pts:
        d = (p[0]-V[0], p[1]-V[1], p[2]-V[2])
        dl = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
        if dl < tol:
            continue
        gs.append((d[0]/dl, d[1]/dl, d[2]/dl))
    if len(gs) < 3:
        return None
    # Axis = principal (largest-eigenvalue) direction of the generator
    # covariance — generators cluster tightly around the axis. (The mean
    # generator is biased by non-uniform/seam sampling.)
    G = [[0.0]*3 for _ in range(3)]
    for g in gs:
        for r in range(3):
            for c in range(3):
                G[r][c] += g[r]*g[c]
    gevals, gevecs = _jacobi_eig3(G)
    kmax = max(range(3), key=lambda k: gevals[k])
    w = gevecs[kmax]
    sx = (sum(g[0] for g in gs), sum(g[1] for g in gs), sum(g[2] for g in gs))
    if w[0]*sx[0] + w[1]*sx[1] + w[2]*sx[2] < 0.0:   # orient toward the generators
        w = (-w[0], -w[1], -w[2])
    wl = math.sqrt(w[0]**2 + w[1]**2 + w[2]**2)
    if wl < 1e-12:
        return None
    w = (w[0]/wl, w[1]/wl, w[2]/wl)
    angs = [math.acos(max(-1.0, min(1.0, g[0]*w[0]+g[1]*w[1]+g[2]*w[2]))) for g in gs]
    alpha = sum(angs)/len(angs)
    if alpha < 1e-4 or alpha > math.pi/2 - 1e-4:
        return None
    ca = math.cos(alpha)
    for p in pts:
        d = (p[0]-V[0], p[1]-V[1], p[2]-V[2])
        axd = d[0]*w[0] + d[1]*w[1] + d[2]*w[2]
        perp = math.sqrt(max(0.0, (d[0]**2+d[1]**2+d[2]**2) - axd*axd))
        if abs(perp - axd*math.tan(alpha)) * ca > tol:
            return None
    return ((V[0], V[1], V[2]), w, alpha)


def _fit_sphere(surface, tol):
    """Algebraic sphere fit on sampled surface points. Returns (cx,cy,cz,r) if
    the surface is a sphere within tol, else None."""
    import math
    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    pts = []
    for i in range(5):
        for j in range(5):
            uu = u0 + (u1-u0)*i/4.0
            vv = v0 + (v1-v0)*j/4.0
            pts.append(surface.point_at(uu, vv))
    # Solve x^2+y^2+z^2 + D x + E y + F z + G = 0  (least squares, normal eqs)
    ata = [[0.0]*4 for _ in range(4)]
    atb = [0.0]*4
    for p in pts:
        row = [p[0], p[1], p[2], 1.0]
        rhs = -(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
        for r in range(4):
            atb[r] += row[r]*rhs
            for c in range(4):
                ata[r][c] += row[r]*row[c]
    sol = _solve_gauss(ata, atb, 4)
    if sol is None:
        return None
    cx, cy, cz = -sol[0]/2.0, -sol[1]/2.0, -sol[2]/2.0
    r2 = cx*cx + cy*cy + cz*cz - sol[3]
    if r2 <= 0.0:
        return None
    r = math.sqrt(r2)
    # Verify: all sampled points within tol of the sphere.
    for p in pts:
        d = math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2)
        if abs(d - r) > tol:
            return None
    return (cx, cy, cz, r)


def _fit_torus(surface, tol):
    """Recognize a torus. Axis = smallest-variance direction of the surface
    points (a torus is flattest along its axis, for major > minor). Then fit a
    2D circle (rho, axial) of the tube cross-section. Returns
    (center, axis, major_radius, minor_radius) or None."""
    import math
    u0, u1 = surface.domain(0)
    v0, v1 = surface.domain(1)
    pts = []
    for i in range(8):
        for j in range(8):
            pts.append(surface.point_at(u0 + (u1-u0)*i/8.0, v0 + (v1-v0)*j/8.0))
    n = len(pts)
    cen = [sum(p[k] for p in pts)/n for k in range(3)]
    M = [[0.0]*3 for _ in range(3)]
    for p in pts:
        d = (p[0]-cen[0], p[1]-cen[1], p[2]-cen[2])
        for r in range(3):
            for c in range(3):
                M[r][c] += d[r]*d[c]
    evals, evecs = _jacobi_eig3(M)
    kmin = min(range(3), key=lambda k: evals[k])
    w = evecs[kmin]
    wl = math.sqrt(w[0]**2 + w[1]**2 + w[2]**2)
    if wl < 1e-12:
        return None
    w = (w[0]/wl, w[1]/wl, w[2]/wl)
    # Fit circle (rho-R)^2 + (a-a0)^2 = r^2 in (rho, axial) coords.
    ata = [[0.0]*3 for _ in range(3)]
    atb = [0.0]*3
    rhoa = []
    for p in pts:
        d = (p[0]-cen[0], p[1]-cen[1], p[2]-cen[2])
        a = d[0]*w[0] + d[1]*w[1] + d[2]*w[2]
        perp = (d[0]-a*w[0], d[1]-a*w[1], d[2]-a*w[2])
        rho = math.sqrt(perp[0]**2 + perp[1]**2 + perp[2]**2)
        rhoa.append((rho, a))
        row = [rho, a, 1.0]
        rhs = -(rho*rho + a*a)
        for r in range(3):
            atb[r] += row[r]*rhs
            for c in range(3):
                ata[r][c] += row[r]*row[c]
    sol = _solve_gauss(ata, atb, 3)
    if sol is None:
        return None
    R = -sol[0]/2.0
    a0 = -sol[1]/2.0
    r2 = R*R + a0*a0 - sol[2]
    if r2 <= 1e-18 or R <= 0.0:
        return None
    r = math.sqrt(r2)
    if R <= r * 0.5:
        return None  # not a clear ring torus
    for (rho, a) in rhoa:
        if abs(math.sqrt((rho-R)**2 + (a-a0)**2) - r) > tol:
            return None
    center = (cen[0]+a0*w[0], cen[1]+a0*w[1], cen[2]+a0*w[2])
    return (center, w, R, r)


def _recognize_surface(surface, tol):
    """Classify a surface as plane / sphere / cylinder / cone / torus, else None."""
    if surface.is_planar(None, tol):
        u0, u1 = surface.domain(0)
        v0, v1 = surface.domain(1)
        o = surface.point_at((u0+u1)*0.5, (v0+v1)*0.5)
        n = surface.normal_at((u0+u1)*0.5, (v0+v1)*0.5)
        return ('plane', (o[0], o[1], o[2]), (n[0], n[1], n[2]))
    sph = _fit_sphere(surface, tol)
    if sph is not None:
        return ('sphere', (sph[0], sph[1], sph[2]), sph[3])
    cyl = _fit_cylinder(surface, tol)
    if cyl is not None:
        return ('cylinder', cyl[0], cyl[1], cyl[2])
    cone = _fit_cone(surface, tol)
    if cone is not None:
        return ('cone', cone[0], cone[1], cone[2])
    tor = _fit_torus(surface, tol)
    if tor is not None:
        return ('torus', tor[0], tor[1], tor[2], tor[3])
    return None


def _analytic_pcurve(srf, recog, c3d):
    """OCCT ProjLib-style closed-form pull-back of a 3D curve onto a recognized surface.
    Currently handles the CONE case (a coaxial circle perpendicular to the axis -> an exact
    v=const line); returns None otherwise (the caller falls back to iterative projection)."""
    import math
    from .nurbscurve import NurbsCurve
    if recog is None:
        return None
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)

    def dot(p, q):
        return p[0]*q[0] + p[1]*q[1] + p[2]*q[2]

    if recog[0] == 'cone':
        # A circle perpendicular to the cone axis (a coaxial "parallel") -> exact v=const line.
        # (recog[3] is the cone HALF-ANGLE, not a length, so use a curve-length scale for tols.)
        ax = recog[2]
        an = math.sqrt(dot(ax, ax))
        if an < 1e-12:
            return None
        ax = (ax[0]/an, ax[1]/an, ax[2]/an)
        A = recog[1]   # apex

        def height(p):
            return (p[0]-A[0])*ax[0] + (p[1]-A[1])*ax[1] + (p[2]-A[2])*ax[2]

        t0, t1 = c3d.domain()
        clen = c3d.point_at(t0).distance(c3d.point_at(0.5*(t0+t1)))
        hscale = max(clen, 1e-9)
        hmin = 1e300; hmax = -1e300; hsum = 0.0; ns = 0
        for i in range(33):
            h = height(c3d.point_at(t0 + (t1-t0)*i/32))
            hmin = min(hmin, h); hmax = max(hmax, h); hsum += h; ns += 1
        if hmax - hmin > hscale * 1e-4:
            return None   # oblique conic -> projection
        if c3d.point_at(t0).distance(c3d.point_at(t1)) > hscale * 1e-3:
            return None   # not a full wrap
        hc = hsum / ns
        um2 = 0.5*(u0+u1)
        va = v0; vb = v1
        ha = height(srf.point_at(um2, va)); hb = height(srf.point_at(um2, vb))
        if (hc-ha)*(hc-hb) > 0:
            return None   # height out of v-range
        for _ in range(60):
            vmid = 0.5*(va+vb)
            hm = height(srf.point_at(um2, vmid))
            if (hm-hc)*(ha-hc) <= 0:
                vb = vmid
            else:
                va = vmid; ha = hm
        vc = 0.5*(va+vb)
        return NurbsCurve.create(False, 1, [Point(u0, vc, 0.0), Point(u1, vc, 0.0)])

    return None


# --- analytic SSI helpers (V3 = plain (x,y,z) tuples) -------------------------------------------
def _ssi_unit(v):
    import math
    l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    return (v[0]/l, v[1]/l, v[2]/l) if l > 1e-300 else v


def _ssi_dot(u, v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]


def _ssi_cross(u, v):
    return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])


def _point_axis_dist(apt, adir, P):
    import math
    u = _ssi_unit(adir)
    dp = (P[0]-apt[0], P[1]-apt[1], P[2]-apt[2])
    t = _ssi_dot(dp, u)
    perp = (dp[0]-t*u[0], dp[1]-t*u[1], dp[2]-t*u[2])
    return math.sqrt(_ssi_dot(perp, perp))


def _axial_coord(apt, adir, P):
    u = _ssi_unit(adir)
    return (P[0]-apt[0])*u[0] + (P[1]-apt[1])*u[1] + (P[2]-apt[2])*u[2]


def _axes_coaxial(p1, d1, p2, d2, tol):
    import math
    u1 = _ssi_unit(d1); u2 = _ssi_unit(d2)
    cx = _ssi_cross(u1, u2)
    if math.sqrt(_ssi_dot(cx, cx)) > tol:
        return False
    return _point_axis_dist(p1, u1, p2) <= tol


def _cyl_span(srf, apt, adir):
    u = _ssi_unit(adir)
    u0, u1 = srf.domain(0); v0, v1 = srf.domain(1)
    um = 0.5 * (u0 + u1)
    smin = 1e300; smax = -1e300
    for vv in (v0, v1):
        p = srf.point_at(um, vv)
        s = (p[0]-apt[0])*u[0] + (p[1]-apt[1])*u[1] + (p[2]-apt[2])*u[2]
        smin = min(smin, s); smax = max(smax, s)
    return smin, smax


def _lines_closest_point(p1, d1, p2, d2, tol):
    import math
    u = _ssi_unit(d1); v = _ssi_unit(d2)
    w0 = (p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2])
    a = _ssi_dot(u, u); b = _ssi_dot(u, v); c = _ssi_dot(v, v)
    d = _ssi_dot(u, w0); e = _ssi_dot(v, w0)
    den = a*c - b*b
    if abs(den) < 1e-12:
        return None
    sc = (b*e - c*d) / den; tc = (a*e - b*d) / den
    q1 = (p1[0]+sc*u[0], p1[1]+sc*u[1], p1[2]+sc*u[2])
    q2 = (p2[0]+tc*v[0], p2[1]+tc*v[1], p2[2]+tc*v[2])
    diff = (q1[0]-q2[0], q1[1]-q2[1], q1[2]-q2[2])
    if math.sqrt(_ssi_dot(diff, diff)) > tol:
        return None
    return (0.5*(q1[0]+q2[0]), 0.5*(q1[1]+q2[1]), 0.5*(q1[2]+q2[2]))


def _ssi_cylinder_sphere(cyl, sph):
    """Coaxial cylinder/sphere -> 0, 1, or 2 exact circles. Returns None if the axes are not
    coaxial (caller marches), else a list (possibly empty)."""
    import math
    kTol = 1e-6
    P = cyl[1]; w = _ssi_unit(cyl[2]); rc = cyl[3]
    C = sph[1]; R = sph[2]
    if _point_axis_dist(P, w, C) > kTol:
        return None
    out = []
    if R < rc - kTol:
        return out
    dist = math.sqrt(max(0.0, R*R - rc*rc))
    xa, ya = _ortho_basis(w)
    if dist <= kTol:
        out.append(_exact_circle(C[0], C[1], C[2], xa, ya, rc))
        return out
    for s in (dist, -dist):
        cc = (C[0]+s*w[0], C[1]+s*w[1], C[2]+s*w[2])
        out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rc))
    return out


def _ssi_cylinder_cone(cyl, cone):
    """Coaxial cylinder/cone -> 1 exact circle. Returns None if not coaxial (caller marches)."""
    import math
    kTol = 1e-6
    Pc = cyl[1]; w = _ssi_unit(cyl[2]); rc = cyl[3]
    apex = cone[1]; a = _ssi_unit(cone[2]); alpha = cone[3]
    if not _axes_coaxial(Pc, w, apex, a, kTol):
        return None
    ta = math.tan(alpha)
    if ta < 1e-9:
        return None
    s = rc / ta
    out = []
    if s < kTol:
        return out
    cc = (apex[0]+s*a[0], apex[1]+s*a[1], apex[2]+s*a[2])
    xa, ya = _ortho_basis(a)
    out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rc))
    return out


def _ssi_cone_sphere(cone, sph):
    """Coaxial cone/sphere -> 1 or 2 exact circles. Returns None if not coaxial (caller marches)."""
    import math
    kTol = 1e-6
    apex = cone[1]; a = _ssi_unit(cone[2]); alpha = cone[3]
    C = sph[1]; R = sph[2]
    if _point_axis_dist(apex, a, C) > kTol:
        return None
    dsign = _axial_coord(apex, a, C)
    d = abs(dsign)
    dir = (-a[0], -a[1], -a[2]) if (d > kTol and dsign < 0.0) else a
    t = math.tan(alpha); t2 = t*t
    A = 1.0 + t2; B = 2.0*t2*d; Cq = t2*d*d - R*R
    disc = B*B - 4.0*A*Cq
    out = []
    if disc < -kTol:
        return out
    sq = math.sqrt(max(0.0, disc))
    if sq <= kTol:
        xs = [-B/(2.0*A)]
    else:
        xs = [(-B - sq)/(2.0*A), (-B + sq)/(2.0*A)]
    xa, ya = _ortho_basis(a)
    for x in xs:
        sAx = d + x
        if sAx < kTol:
            continue
        rr = t * sAx
        if rr < kTol:
            continue
        cc = (apex[0]+sAx*dir[0], apex[1]+sAx*dir[1], apex[2]+sAx*dir[2])
        out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rr))
    return out


def _ssi_cylinder_cylinder(sa, A, sb, B):
    """Two cylinders: parallel -> 0/1/2 line segments (clipped to the shared axial span);
    equal-radius skew/intersecting axes -> 2 exact ellipses. Returns None if not analytically
    handled (caller marches)."""
    import math
    from .nurbscurve import NurbsCurve
    kTol = 1e-6
    P1 = A[1]; w1 = _ssi_unit(A[2]); R1 = A[3]
    P2 = B[1]; w2 = _ssi_unit(B[2]); R2 = B[3]
    cx = _ssi_cross(w1, w2)
    sinmag = math.sqrt(_ssi_dot(cx, cx))
    out = []
    if sinmag <= kTol:
        dline = _point_axis_dist(P1, w1, P2)
        if dline <= kTol:
            if abs(R1 - R2) <= kTol:
                return None   # coincident axes, equal radius -> degenerate -> marcher
            return out         # coaxial, different radius -> no intersection
        off = _ssi_dot((P2[0]-P1[0], P2[1]-P1[1], P2[2]-P1[2]), w1)
        P2p = (P2[0]-off*w1[0], P2[1]-off*w1[1], P2[2]-off*w1[2])
        d = dline
        if d > R1 + R2 + kTol:
            return out
        if d < abs(R1 - R2) - kTol:
            return out
        xdir = _ssi_unit((P2p[0]-P1[0], P2p[1]-P1[1], P2p[2]-P1[2]))
        ydir = _ssi_unit(_ssi_cross(w1, xdir))
        aa = (R1*R1 - R2*R2 + d*d) / (2.0*d)
        h = math.sqrt(max(0.0, R1*R1 - aa*aa))
        foot = (P1[0]+aa*xdir[0], P1[1]+aa*xdir[1], P1[2]+aa*xdir[2])
        s0a, s1a = _cyl_span(sa, P1, w1)
        s0b, s1b = _cyl_span(sb, P1, w1)
        slo = max(s0a, s0b); shi = min(s1a, s1b)
        if shi - slo <= kTol:
            return out

        def emit(bp):
            e0 = (bp[0]+slo*w1[0], bp[1]+slo*w1[1], bp[2]+slo*w1[2])
            e1 = (bp[0]+shi*w1[0], bp[1]+shi*w1[1], bp[2]+shi*w1[2])
            ln = NurbsCurve.create(False, 1, [Point(e0[0], e0[1], e0[2]), Point(e1[0], e1[1], e1[2])])
            ln.set_domain(0.0, 1.0)
            out.append(ln)

        if h <= kTol:
            emit(foot)
        else:
            emit((foot[0]+h*ydir[0], foot[1]+h*ydir[1], foot[2]+h*ydir[2]))
            emit((foot[0]-h*ydir[0], foot[1]-h*ydir[1], foot[2]-h*ydir[2]))
        return out
    Rmax = max(R1, R2)
    if Rmax < 1e-12 or abs(R1 - R2) / Rmax > 1e-6:
        return None
    Pint = _lines_closest_point(P1, w1, P2, w2, kTol)
    if Pint is None:
        return None
    R = 0.5 * (R1 + R2)
    ang = math.acos(max(-1.0, min(1.0, _ssi_dot(w1, w2))))
    sh = math.sin(0.5*ang); ch = math.cos(0.5*ang)
    if sh < 1e-9 or ch < 1e-9:
        return None
    minor = _ssi_unit(cx)
    maj1 = _ssi_unit((w1[0]+w2[0], w1[1]+w2[1], w1[2]+w2[2]))
    maj2 = _ssi_unit((w1[0]-w2[0], w1[1]-w2[1], w1[2]-w2[2]))
    out.append(_exact_ellipse(Pint[0], Pint[1], Pint[2], maj1, minor, R/sh, R))
    out.append(_exact_ellipse(Pint[0], Pint[1], Pint[2], maj2, minor, R/ch, R))
    return out


def _analytic_ssi(a, b, tolerance):
    """Closed-form intersection for recognized quadric pairs (exact conics).
    Returns a list of (curve_3d, pcurve_a, pcurve_b), or None if the pair is
    not an analytically-exact case (caller falls back to marching)."""
    import math
    from .closest import Closest
    ra = _recognize_surface(a, max(tolerance, 1e-7) * 1e4)
    rb = _recognize_surface(b, max(tolerance, 1e-7) * 1e4)
    if ra is None or rb is None:
        return None

    def unit(v):
        l = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        return (v[0]/l, v[1]/l, v[2]/l) if l > 1e-300 else v

    def cross(u, v):
        return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])

    def plane_sphere(plane, sph):
        o, nu = plane[1], unit(plane[2])
        c, r = sph[1], sph[2]
        d = (c[0]-o[0])*nu[0] + (c[1]-o[1])*nu[1] + (c[2]-o[2])*nu[2]
        if abs(d) >= r:
            return None
        cc = (c[0]-d*nu[0], c[1]-d*nu[1], c[2]-d*nu[2])
        rr = math.sqrt(r*r - d*d)
        xa, ya = _ortho_basis(nu)
        return _exact_circle(cc[0], cc[1], cc[2], xa, ya, rr)

    def plane_cylinder(plane, cyl):
        o, nu = plane[1], unit(plane[2])
        P, w, r = cyl[1], unit(cyl[2]), cyl[3]
        wn = w[0]*nu[0] + w[1]*nu[1] + w[2]*nu[2]
        if abs(wn) < 1e-7:
            return None  # plane parallel to axis -> lines (degenerate); marcher
        t = ((o[0]-P[0])*nu[0] + (o[1]-P[1])*nu[1] + (o[2]-P[2])*nu[2]) / wn
        cc = (P[0]+t*w[0], P[1]+t*w[1], P[2]+t*w[2])
        mraw = cross(w, nu)                              # in plane, perp to axis
        if math.sqrt(mraw[0]**2 + mraw[1]**2 + mraw[2]**2) < 1e-9:
            # Plane perpendicular to the axis -> the section is a circle.
            xa, ya = _ortho_basis(nu)
            return _exact_circle(cc[0], cc[1], cc[2], xa, ya, r)
        minor = unit(mraw)
        major = unit((w[0]-wn*nu[0], w[1]-wn*nu[1], w[2]-wn*nu[2]))
        return _exact_ellipse(cc[0], cc[1], cc[2], major, minor, r/abs(wn), r)

    def line_cone(x0, d, V, w, alpha):
        # Solve ((X-V).w)^2 - cos^2a |X-V|^2 = 0 along X=x0+t d. Returns [t...].
        ca2 = math.cos(alpha)**2
        e = (x0[0]-V[0], x0[1]-V[1], x0[2]-V[2])
        A = e[0]*w[0]+e[1]*w[1]+e[2]*w[2]
        B = d[0]*w[0]+d[1]*w[1]+d[2]*w[2]
        C = e[0]*e[0]+e[1]*e[1]+e[2]*e[2]
        D = e[0]*d[0]+e[1]*d[1]+e[2]*d[2]
        E = d[0]*d[0]+d[1]*d[1]+d[2]*d[2]
        qa = B*B - ca2*E
        qb = 2.0*A*B - 2.0*ca2*D
        qc = A*A - ca2*C
        if abs(qa) < 1e-14:
            return [] if abs(qb) < 1e-300 else [-qc/qb]
        disc = qb*qb - 4.0*qa*qc
        if disc < 0.0:
            return []
        sq = math.sqrt(disc)
        return [(-qb - sq)/(2.0*qa), (-qb + sq)/(2.0*qa)]

    def plane_cone(plane, cone):
        o, nu = plane[1], unit(plane[2])
        V, w, alpha = cone[1], unit(cone[2]), cone[3]
        wn = w[0]*nu[0] + w[1]*nu[1] + w[2]*nu[2]
        if abs(abs(wn) - 1.0) < 1e-9:
            # Plane perpendicular to axis -> circle at axial distance from apex.
            dax = (o[0]-V[0])*w[0] + (o[1]-V[1])*w[1] + (o[2]-V[2])*w[2]
            rr = abs(dax) * math.tan(alpha)
            cc = (V[0]+dax*w[0], V[1]+dax*w[1], V[2]+dax*w[2])
            if rr < 1e-12:
                return None
            xa, ya = _ortho_basis(nu)
            return _exact_circle(cc[0], cc[1], cc[2], xa, ya, rr)
        # General: major axis = cutting plane ∩ symmetry plane (span of w,n).
        m = cross(w, nu)
        ml = math.sqrt(m[0]**2 + m[1]**2 + m[2]**2)
        if ml < 1e-12:
            return None
        m = (m[0]/ml, m[1]/ml, m[2]/ml)               # minor direction
        major = unit((w[0]-wn*nu[0], w[1]-wn*nu[1], w[2]-wn*nu[2]))
        # Apex projected into the cutting plane lies on the major-axis line.
        dV = (V[0]-o[0])*nu[0] + (V[1]-o[1])*nu[1] + (V[2]-o[2])*nu[2]
        Vp = (V[0]-dV*nu[0], V[1]-dV*nu[1], V[2]-dV*nu[2])
        ts = line_cone(Vp, major, V, w, alpha)
        if len(ts) != 2:
            return None  # parabola/hyperbola (unbounded) -> not an ellipse
        A = (Vp[0]+ts[0]*major[0], Vp[1]+ts[0]*major[1], Vp[2]+ts[0]*major[2])
        Bp = (Vp[0]+ts[1]*major[0], Vp[1]+ts[1]*major[1], Vp[2]+ts[1]*major[2])
        cc = ((A[0]+Bp[0])*0.5, (A[1]+Bp[1])*0.5, (A[2]+Bp[2])*0.5)
        semi_major = 0.5*math.sqrt((Bp[0]-A[0])**2 + (Bp[1]-A[1])**2 + (Bp[2]-A[2])**2)
        major = unit((Bp[0]-A[0], Bp[1]-A[1], Bp[2]-A[2]))
        tm = line_cone(cc, m, V, w, alpha)
        if len(tm) != 2:
            return None
        semi_minor = 0.5*abs(tm[1]-tm[0])
        if semi_major < 1e-12 or semi_minor < 1e-12:
            return None
        return _exact_ellipse(cc[0], cc[1], cc[2], major, m, semi_major, semi_minor)

    def plane_torus(plane, tor):
        o, nu = plane[1], unit(plane[2])
        C, w, R, r = tor[1], unit(tor[2]), tor[3], tor[4]
        wn = w[0]*nu[0] + w[1]*nu[1] + w[2]*nu[2]
        if abs(abs(wn) - 1.0) > 1e-7:
            return None  # non-perpendicular -> quartic, not conic -> marcher
        d = (o[0]-C[0])*w[0] + (o[1]-C[1])*w[1] + (o[2]-C[2])*w[2]
        if abs(d) > r:
            return []    # plane misses the tube
        h = math.sqrt(max(0.0, r*r - d*d))
        cc = (C[0]+d*w[0], C[1]+d*w[1], C[2]+d*w[2])
        xa, ya = _ortho_basis(w)
        out = []
        for rr in (R + h, R - h):
            if rr > 1e-12:
                out.append(_exact_circle(cc[0], cc[1], cc[2], xa, ya, rr))
        return out

    # Each handler returns a list of exact 3D curves (empty = recognized but no
    # intersection), or None = not analytically handled (caller marches).
    def single(c):
        return [c] if c is not None else []

    c3_list = None
    if ra[0] == 'plane' and rb[0] == 'sphere':
        c3_list = single(plane_sphere(ra, rb))
    elif ra[0] == 'sphere' and rb[0] == 'plane':
        c3_list = single(plane_sphere(rb, ra))
    elif ra[0] == 'plane' and rb[0] == 'cylinder':
        c3_list = single(plane_cylinder(ra, rb))
    elif ra[0] == 'cylinder' and rb[0] == 'plane':
        c3_list = single(plane_cylinder(rb, ra))
    elif ra[0] == 'plane' and rb[0] == 'cone':
        c3_list = single(plane_cone(ra, rb))
    elif ra[0] == 'cone' and rb[0] == 'plane':
        c3_list = single(plane_cone(rb, ra))
    elif ra[0] == 'plane' and rb[0] == 'torus':
        c3_list = plane_torus(ra, rb)
    elif ra[0] == 'torus' and rb[0] == 'plane':
        c3_list = plane_torus(rb, ra)
    elif ra[0] == 'sphere' and rb[0] == 'sphere':
        c1, r1 = ra[1], ra[2]
        c2, r2 = rb[1], rb[2]
        dv = (c2[0]-c1[0], c2[1]-c1[1], c2[2]-c1[2])
        dist = math.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
        c3 = None
        if 1e-12 < dist < r1 + r2 and dist > abs(r1 - r2):
            nu = (dv[0]/dist, dv[1]/dist, dv[2]/dist)
            aa = (dist*dist + r1*r1 - r2*r2) / (2.0*dist)
            rr2 = r1*r1 - aa*aa
            if rr2 > 0.0:
                cc = (c1[0]+aa*nu[0], c1[1]+aa*nu[1], c1[2]+aa*nu[2])
                xa, ya = _ortho_basis(nu)
                c3 = _exact_circle(cc[0], cc[1], cc[2], xa, ya, math.sqrt(rr2))
        c3_list = single(c3)
    # Coaxial / canonical quadric pairs (exact conics from IntAna_QuadQuadGeo). Each helper
    # returns None when the pair is recognized but not analytically handled (-> marcher), or a
    # list (possibly empty = recognized, no intersection).
    elif ra[0] == 'cylinder' and rb[0] == 'sphere':
        c3_list = _ssi_cylinder_sphere(ra, rb)
    elif ra[0] == 'sphere' and rb[0] == 'cylinder':
        c3_list = _ssi_cylinder_sphere(rb, ra)
    elif ra[0] == 'cylinder' and rb[0] == 'cone':
        c3_list = _ssi_cylinder_cone(ra, rb)
    elif ra[0] == 'cone' and rb[0] == 'cylinder':
        c3_list = _ssi_cylinder_cone(rb, ra)
    elif ra[0] == 'cone' and rb[0] == 'sphere':
        c3_list = _ssi_cone_sphere(ra, rb)
    elif ra[0] == 'sphere' and rb[0] == 'cone':
        c3_list = _ssi_cone_sphere(rb, ra)
    elif ra[0] == 'cylinder' and rb[0] == 'cylinder':
        c3_list = _ssi_cylinder_cylinder(a, ra, b, rb)
    else:
        return None   # not an analytically-exact pair -> marcher

    if c3_list is None:
        return None   # recognized but not analytically handled -> marcher

    # The 3D curves are exact; pull back onto each surface. Try the closed-form analytic pcurve
    # first (OCCT ProjLib-style, no sampling -- e.g. a coaxial circle on a cone -> exact v=const
    # line); fall back to iterative projection otherwise. A curve crossing a surface seam pulls
    # back to several UV pieces (c3 stays whole); skip a curve whose pullback fails entirely.
    triples = []
    for c3 in c3_list:
        pa = _analytic_pcurve(a, ra, c3)
        pb = _analytic_pcurve(b, rb, c3)
        if pa is None or not pa.is_valid():
            v = Closest.surface_curve(a, c3); pa = v[0] if v else None
        if pb is None or not pb.is_valid():
            v = Closest.surface_curve(b, c3); pb = v[0] if v else None
        if pa is not None and pa.is_valid() and pb is not None and pb.is_valid():
            triples.append((c3, pa, pb))
    return triples


def surface_surface(a: "NurbsSurface", b: "NurbsSurface", tolerance: float | None=None) -> list[tuple["NurbsCurve", "NurbsCurve", "NurbsCurve"]]:
    """Find surface/surface intersection curves with UV pcurves on both.

    Returns a list of (curve_3d, pcurve_a, pcurve_b) triples. Recognized quadric
    pairs (plane/sphere ...) are solved in closed form (exact conics); other
    pairs fall back to predictor-corrector marching. Pcurves are NurbsCurves in
    each surface's parameter space (x=u, y=v, z=0). Marching terminates at
    tangencies (n_a parallel n_b); tangential intersections are unsupported.
    """
    import math
    from .nurbscurve import NurbsCurve
    from .nurbsknot import CurveNurbsKnotStyle
    from .plane import Plane
    from .closest import Closest

    if a.is_valid() and b.is_valid():
        _ana = _analytic_ssi(a, b, tolerance if (tolerance and tolerance > 0) else Tolerance.ZERO_TOLERANCE)
        if _ana is not None:
            return _ana

    if not a.is_valid() or not b.is_valid():
        return []
    if tolerance is None or tolerance <= 0.0:
        tolerance = Tolerance.ZERO_TOLERANCE

    # Planar dispatch: reuse the plane tracer when either surface is planar
    def plane_from(srf):
        s0, s1 = srf.domain(0)
        t0, t1 = srf.domain(1)
        po = srf.point_at((s0+s1)*0.5, (t0+t1)*0.5)
        nn = srf.normal_at((s0+s1)*0.5, (t0+t1)*0.5)
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

    # ---- Per-surface context ----
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

    # ---- Seed cells: half-resolution sample grids + sag-inflated AABBs ----
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
                for i in range(2*ci, 2*ci + 3):
                    for j in range(2*cj, 2*cj + 3):
                        p = S[i][j]
                        xs.append(p[0])
                        ys.append(p[1])
                        zs.append(p[2])
                ctr = S[2*ci + 1][2*cj + 1]
                cx = (S[2*ci][2*cj][0] + S[2*ci+2][2*cj][0] + S[2*ci][2*cj+2][0] + S[2*ci+2][2*cj+2][0]) * 0.25
                cy = (S[2*ci][2*cj][1] + S[2*ci+2][2*cj][1] + S[2*ci][2*cj+2][1] + S[2*ci+2][2*cj+2][1]) * 0.25
                cz = (S[2*ci][2*cj][2] + S[2*ci+2][2*cj][2] + S[2*ci][2*cj+2][2] + S[2*ci+2][2*cj+2][2]) * 0.25
                sag = math.sqrt((ctr[0]-cx)**2 + (ctr[1]-cy)**2 + (ctr[2]-cz)**2)
                inf = 2.0 * sag + tolerance
                boxes.append((min(xs)-inf, min(ys)-inf, min(zs)-inf,
                              max(xs)+inf, max(ys)+inf, max(zs)+inf,
                              c0u + dcu * (ci + 0.5), c0v + dcv * (cj + 0.5)))
        return boxes

    boxes_a = cell_boxes(a, au0, a_du, a_nu, av0, a_dv, a_nv)
    boxes_b = cell_boxes(b, bu0, b_du, b_nu, bv0, b_dv, b_nv)

    def cell_3d(boxes):
        best = float('inf')
        for bx in boxes[:64]:
            d = math.sqrt((bx[3]-bx[0])**2 + (bx[4]-bx[1])**2 + (bx[5]-bx[2])**2)
            if 1e-12 < d < best:
                best = d
        return best if best < float('inf') else 1.0

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
        # Newton on Sa(u,v) - Sb(s,t) = 0; minimum-norm or tangent-pinned
        for _ in range(8):
            Sa, Sau, Sav = eval_a(x[0], x[1])
            Sb, Sbu, Sbv = eval_b(x[2], x[3])
            F = [Sa[0]-Sb[0], Sa[1]-Sb[1], Sa[2]-Sb[2]]
            if math.sqrt(F[0]**2 + F[1]**2 + F[2]**2) < conv_tol:
                return True
            J = [[Sau[k], Sav[k], -Sbu[k], -Sbv[k]] for k in range(3)]
            if pin is None:
                JJt = [[sum(J[r][c]*J[q][c] for c in range(4)) for q in range(3)] for r in range(3)]
                y = _solve_gauss(JJt, F, 3)
                if y is None:
                    return False
                for c in range(4):
                    x[c] -= sum(J[r][c]*y[r] for r in range(3))
            else:
                d, pp = pin
                M = [J[0], J[1], J[2],
                     [d[0]*Sau[0]+d[1]*Sau[1]+d[2]*Sau[2],
                      d[0]*Sav[0]+d[1]*Sav[1]+d[2]*Sav[2], 0.0, 0.0]]
                rhs = [F[0], F[1], F[2],
                       d[0]*(Sa[0]-pp[0]) + d[1]*(Sa[1]-pp[1]) + d[2]*(Sa[2]-pp[2])]
                dx = _solve_gauss(M, rhs, 4)
                if dx is None:
                    return False
                for c in range(4):
                    x[c] -= dx[c]
            clamp_open(x)
        Sa, _, _ = eval_a(x[0], x[1])
        Sb, _, _ = eval_b(x[2], x[3])
        g = math.sqrt((Sa[0]-Sb[0])**2 + (Sa[1]-Sb[1])**2 + (Sa[2]-Sb[2])**2)
        return g < conv_tol * 10.0

    # ---- Seeds from overlapping cell pairs (minimum-norm Gauss-Newton) ----
    seeds = []  # [u, v, s, t, used]
    seed_tol_3d = max(cell_3d(boxes_a), cell_3d(boxes_b))
    pair_budget = 20000
    for ba in boxes_a:
        if pair_budget < 0:
            break
        for bb in boxes_b:
            if bb[0] > ba[3] or bb[3] < ba[0] or bb[1] > ba[4] or bb[4] < ba[1] or bb[2] > ba[5] or bb[5] < ba[2]:
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
                if math.sqrt((Sa[0]-So[0])**2 + (Sa[1]-So[1])**2 + (Sa[2]-So[2])**2) < seed_tol_3d:
                    dup = True
                    break
            if not dup:
                seeds.append([a_wrap_u(x[0]), a_wrap_v(x[1]), b_wrap_u(x[2]), b_wrap_v(x[3]), False])

    # ---- Trace each branch with predictor-corrector marching ----
    max_steps = (a_nu * a_nv + b_nu * b_nv) * 32
    close_tol = h_init * 3.0
    consume_tol = h_init * 2.0

    def tangent_3d(x, dir_sign):
        Sa, Sau, Sav = eval_a(x[0], x[1])
        Sb, Sbu, Sbv = eval_b(x[2], x[3])
        na = (Sau[1]*Sav[2]-Sau[2]*Sav[1], Sau[2]*Sav[0]-Sau[0]*Sav[2], Sau[0]*Sav[1]-Sau[1]*Sav[0])
        nb = (Sbu[1]*Sbv[2]-Sbu[2]*Sbv[1], Sbu[2]*Sbv[0]-Sbu[0]*Sbv[2], Sbu[0]*Sbv[1]-Sbu[1]*Sbv[0])
        d = (na[1]*nb[2]-na[2]*nb[1], na[2]*nb[0]-na[0]*nb[2], na[0]*nb[1]-na[1]*nb[0])
        dl = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
        nal = math.sqrt(na[0]**2 + na[1]**2 + na[2]**2)
        nbl = math.sqrt(nb[0]**2 + nb[1]**2 + nb[2]**2)
        if dl < 1e-4 * nal * nbl or dl < 1e-30:
            return None
        return (d[0]/dl*dir_sign, d[1]/dl*dir_sign, d[2]/dl*dir_sign), (Sa, Sau, Sav, Sbu, Sbv)

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
                    [[Sau[0]**2+Sau[1]**2+Sau[2]**2, Sau[0]*Sav[0]+Sau[1]*Sav[1]+Sau[2]*Sav[2]],
                     [Sau[0]*Sav[0]+Sau[1]*Sav[1]+Sau[2]*Sav[2], Sav[0]**2+Sav[1]**2+Sav[2]**2]],
                    [h*(d[0]*Sau[0]+d[1]*Sau[1]+d[2]*Sau[2]), h*(d[0]*Sav[0]+d[1]*Sav[1]+d[2]*Sav[2])], 2)
                duv_b = _solve_gauss(
                    [[Sbu[0]**2+Sbu[1]**2+Sbu[2]**2, Sbu[0]*Sbv[0]+Sbu[1]*Sbv[1]+Sbu[2]*Sbv[2]],
                     [Sbu[0]*Sbv[0]+Sbu[1]*Sbv[1]+Sbu[2]*Sbv[2], Sbv[0]**2+Sbv[1]**2+Sbv[2]**2]],
                    [h*(d[0]*Sbu[0]+d[1]*Sbu[1]+d[2]*Sbu[2]), h*(d[0]*Sbv[0]+d[1]*Sbv[1]+d[2]*Sbv[2])], 2)
                if duv_a is None or duv_b is None:
                    return out, False
                delta = [duv_a[0], duv_a[1], duv_b[0], duv_b[1]]
                tc = 1.0
                hit_boundary = False
                for idx, lo, hi, closed in ((0, au0, au1, a_closed_u), (1, av0, av1, a_closed_v),
                                            (2, bu0, bu1, b_closed_u), (3, bv0, bv1, b_closed_v)):
                    if closed or abs(delta[idx]) < 1e-15:
                        continue
                    if x[idx] + delta[idx] > hi:
                        tc = min(tc, (hi - x[idx]) / delta[idx])
                        hit_boundary = True
                    if x[idx] + delta[idx] < lo:
                        tc = min(tc, (lo - x[idx]) / delta[idx])
                        hit_boundary = True
                xn = [x[k] + tc * delta[k] for k in range(4)]
                p_pred = (Sa[0] + d[0]*h*tc, Sa[1] + d[1]*h*tc, Sa[2] + d[2]*h*tc)
                if not correct(xn, (d, p_pred)):
                    return out, False
                San, _, _ = eval_a(xn[0], xn[1])
                p_cur = (San[0], San[1], San[2])
                step_len = math.sqrt((p_cur[0]-p_prev[0])**2 + (p_cur[1]-p_prev[1])**2 + (p_cur[2]-p_prev[2])**2)
                if prev_d is not None and step_len > 1e-14:
                    sd_ = ((p_cur[0]-p_prev[0])/step_len, (p_cur[1]-p_prev[1])/step_len, (p_cur[2]-p_prev[2])/step_len)
                    ddot = sd_[0]*prev_d[0] + sd_[1]*prev_d[1] + sd_[2]*prev_d[2]
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
            if dist_traveled > close_tol * 3.0 and \
               math.sqrt((p_cur[0]-p_start[0])**2 + (p_cur[1]-p_start[1])**2 + (p_cur[2]-p_start[2])**2) < close_tol:
                out.append(list(x))
                return out, True
            out.append(list(x))
            p_prev = p_cur
            if hit_boundary:
                break
            for sd in seeds:
                if not sd[4]:
                    So, _, _ = eval_a(sd[0], sd[1])
                    if math.sqrt((p_cur[0]-So[0])**2 + (p_cur[1]-So[1])**2 + (p_cur[2]-So[2])**2) < consume_tol:
                        sd[4] = True
        return out, False

    axes = ((0, au0, a_range_u, a_closed_u), (1, av0, a_range_v, a_closed_v),
            (2, bu0, b_range_u, b_closed_u), (3, bv0, b_range_v, b_closed_v))

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

        # Unwrap all four parameters along the trace
        for i in range(1, len(quad)):
            for idx, c0, rng, closed in axes:
                if not closed:
                    continue
                jump = quad[i][idx] - quad[i-1][idx]
                if jump > rng * 0.5:
                    quad[i][idx] -= rng
                elif jump < -rng * 0.5:
                    quad[i][idx] += rng

        def eval3_q(q):
            Sa, _, _ = eval_a(q[0], q[1])
            return Sa

        p_first = eval3_q(quad[0])
        p_last = eval3_q(quad[-1])
        gap2 = math.sqrt((p_first[0]-p_last[0])**2 + (p_first[1]-p_last[1])**2 + (p_first[2]-p_last[2])**2)
        is_loop = fwd_closed or (len(quad) >= 6 and gap2 < close_tol)
        if is_loop:
            quad.pop()
        if len(quad) < 4:
            continue

        # Trace-level dedup against already kept traces. The tolerance must be tight relative to the
        # spacing between DISTINCT intersection branches: two perpendicular cylinders (Steinmetz) have
        # 4 arcs only ~1.4 apart at the 0.25/0.75 samples, so the old h_init*6 (~2.3) wrongly merged
        # 3 of the 4 arcs into one. Use h_init*2 and scan EVERY kept point (a true duplicate lies
        # within ~1 marching step everywhere; distinct branches do not).
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
                    dmin = min(dmin, math.sqrt((cp[0]-op[0])**2 + (cp[1]-op[1])**2 + (cp[2]-op[2])**2))
                if dmin > dup_tol:
                    all_close = False
                    break
            if all_close:
                dup = True
                break
        if dup:
            continue
        kept_pts3.append(trace_pts3)

        # Densify: fill large 3D gaps (grown steps / fwd-bwd junction) with
        # Newton-corrected midpoints so per-piece interpolation reaches 1e-6.
        def _gap3(qi, qj):
            pi = eval3_q(qi); pj = eval3_q(qj)
            return math.sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2 + (pi[2]-pj[2])**2)
        for _gp in range(4):
            gg = [_gap3(quad[i], quad[i+1]) for i in range(len(quad)-1)]
            if not gg:
                break
            med = sorted(gg)[len(gg)//2]
            if med <= 0:
                break
            changed = False
            i = 0
            while i < len(quad)-1 and len(quad) < 4000:
                if _gap3(quad[i], quad[i+1]) > 1.5*med:
                    mid = [(quad[i][k]+quad[i+1][k])*0.5 for k in range(4)]
                    if correct(mid):
                        quad.insert(i+1, mid)
                        changed = True
                        i += 2
                        continue
                i += 1
            if not changed:
                break

        # Closed-loop virtual closure point across all four parameters
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

        # Seam-split into pieces so pcurves stay inside each surface's domain.
        # (A whole closed 3D loop crossing a seam needs a richer return type to
        # keep c3 whole AND pcurves domain-valid — that is the remaining design
        # work for 1e-6 on seam-crossing quartics.)
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
                # The crossing was linearly interpolated; Newton-correct it onto
                # both surfaces so the piece boundary is accurate (1e-6), not off
                # by the chord error.
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
                    if abs(pb_[idx] - L) < rng * 1e-9 and abs(pb_[idx] - pa_[idx]) > rng * 1e-9:
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
                    pieces.append((out_pts[ia:ib + 1], False))
                wrap_piece = [list(p) for p in out_pts[cross_idx[-1]:]]
                for p in out_pts[1:cross_idx[0] + 1]:
                    wrap_piece.append([p[k] + closure[k] for k in range(4)])
                pieces.append((wrap_piece, False))
            else:
                bounds = [0] + cross_idx + [len(out_pts) - 1]
                for ia, ib in zip(bounds, bounds[1:]):
                    if ib > ia:
                        pieces.append((out_pts[ia:ib + 1], False))

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
                chord3 += math.sqrt((pts3[i][0]-pts3[i-1][0])**2 + (pts3[i][1]-pts3[i-1][1])**2 + (pts3[i][2]-pts3[i-1][2])**2)
            if chord3 < h_init * 0.5:
                continue

            # Deflection-refine this piece: insert Newton-corrected midpoints
            # wherever the 3D curve deviates from its chord by more than the
            # target, so the per-piece interpolation reaches 1e-6 even in
            # high-curvature regions (the global gap-fill misses locally-curved
            # pieces because it uses the whole-curve median spacing).
            refine_tol = max(tolerance * 100.0, 5e-6)
            for _dp in range(8):
                refined = False
                new_pp = [piece_pts[0]]
                i = 0
                while i < len(piece_pts) - 1 and len(piece_pts) < 3000:
                    pa2 = piece_pts[i]; pb2 = piece_pts[i + 1]
                    p3a = eval3_q(pa2); p3b = eval3_q(pb2)
                    mid = [(pa2[k] + pb2[k]) * 0.5 for k in range(4)]
                    if correct(mid):
                        p3m = eval3_q(mid)
                        ex = p3b[0]-p3a[0]; ey = p3b[1]-p3a[1]; ez = p3b[2]-p3a[2]
                        l2 = ex*ex + ey*ey + ez*ez
                        if l2 > 1e-30:
                            tt = ((p3m[0]-p3a[0])*ex + (p3m[1]-p3a[1])*ey + (p3m[2]-p3a[2])*ez) / l2
                            cx = p3a[0]+tt*ex; cy = p3a[1]+tt*ey; cz = p3a[2]+tt*ez
                            dev = math.sqrt((p3m[0]-cx)**2 + (p3m[1]-cy)**2 + (p3m[2]-cz)**2)
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
                    dx1 = pts2[i][0]-pts2[i-1][0]; dy1 = pts2[i][1]-pts2[i-1][1]; dz1 = pts2[i][2]-pts2[i-1][2]
                    dx2 = pts2[i+1][0]-pts2[i][0]; dy2 = pts2[i+1][1]-pts2[i][1]; dz2 = pts2[i+1][2]-pts2[i][2]
                    l1 = math.sqrt(dx1*dx1+dy1*dy1+dz1*dz1); l2 = math.sqrt(dx2*dx2+dy2*dy2+dz2*dz2)
                    if l1 > 1e-14 and l2 > 1e-14:
                        c = max(-1.0, min(1.0, (dx1*dx2+dy1*dy2+dz1*dz2)/(l1*l2)))
                        total_turning += math.acos(c)
                chords = [0.0] * mp
                total_len = 0.0
                for i in range(1, mp):
                    total_len += pts2[i].distance(pts2[i-1])
                    chords[i] = total_len
                if piece_loop and mp > 1:
                    total_len += pts2[0].distance(pts2[mp-1])
                if total_len > 1e-14:
                    for i in range(1, mp):
                        chords[i] /= total_len
                # Compact least-squares first (keep best valid); if it cannot
                # reach the tolerance, interpolate EXACTLY through the dense,
                # high-precision (on-surface) samples to reach 1e-6.
                target_cvs = max(8, int(total_turning / 0.5) + 6)
                max_cvs = max(8, min(mp - 1, mp // 3))
                best = NurbsCurve()
                best_dev = float('inf')
                while target_cvs <= max_cvs:
                    crv = NurbsCurve.create_fitted(pts2, target_cvs, 3, piece_loop)
                    if not crv.is_valid():
                        break
                    ft0, ft1 = crv.domain()
                    dev = 0.0
                    for i in range(mp):
                        dev = max(dev, crv.point_at(ft0 + (ft1-ft0)*chords[i]).distance(pts2[i]))
                    if dev < best_dev:
                        best, best_dev = crv, dev
                    if dev < fit_tol_track:
                        break
                    target_cvs *= 2
                if best_dev >= fit_tol_track:
                    interp = (NurbsCurve.create_interpolated(pts2, CurveNurbsKnotStyle.ChordPeriodic)
                              if piece_loop else NurbsCurve.create_interpolated(pts2))
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
            if not crv3.is_valid() or not pcurve_a.is_valid() or not pcurve_b.is_valid():
                continue
            result.append((crv3, pcurve_a, pcurve_b))

    return result


# ---------------------------------------------------------------------------
# Group A: CGAL-equivalent intersection utilities
# ---------------------------------------------------------------------------

def _vectors_nearly_parallel(v0, v1, angle_tol: float = 0.1) -> bool:
    import math
    m0 = math.sqrt(v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2])
    m1 = math.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2])
    if m0 < 1e-10 or m1 < 1e-10:
        return True
    cos_angle = abs((v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2]) / (m0 * m1))
    return cos_angle > math.cos(angle_tol)


def remap(val: float, from1: float, to1: float, from2: float, to2: float) -> float:
    """Linear remap: map val from [from1,to1] to [from2,to2]."""
    span = to1 - from1
    if abs(span) < 1e-14:
        return from2
    t = (val - from1) / span
    return from2 + t * (to2 - from2)


def closest_point_on_segment(pt: Point, seg: Line) -> tuple:
    """Project point onto finite segment; returns (closest_point, t in [0,1])."""
    start = seg.start()
    end = seg.end()
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    len_sq = dx*dx + dy*dy + dz*dz
    if len_sq < 1e-20:
        return (Point(start[0], start[1], start[2]), 0.0)
    vx = pt[0] - start[0]
    vy = pt[1] - start[1]
    vz = pt[2] - start[2]
    t = (vx*dx + vy*dy + vz*dz) / len_sq
    t = max(0.0, min(1.0, t))
    return (Point(start[0] + t*dx, start[1] + t*dy, start[2] + t*dz), t)


def plane_plane_plane_check(p0: "Plane", p1: "Plane", p2: "Plane", angle_tol: float = 0.1) -> Point | None:
    """3-plane intersection with angle-tolerance parallelism guard."""
    if _vectors_nearly_parallel(p0.z_axis, p1.z_axis, angle_tol):
        return None
    if _vectors_nearly_parallel(p0.z_axis, p2.z_axis, angle_tol):
        return None
    if _vectors_nearly_parallel(p1.z_axis, p2.z_axis, angle_tol):
        return None
    return plane_plane_plane(p0, p1, p2)


def plane_4planes(main_plane: "Plane", planes: list["Plane"]) -> object | None:
    """Intersect main plane with 4 ordered boundary planes → closed quad (5 pts)."""
    from .polyline import Polyline
    p0 = plane_plane_plane_check(planes[0], planes[1], main_plane)
    if p0 is None:
        return None
    p1 = plane_plane_plane_check(planes[1], planes[2], main_plane)
    if p1 is None:
        return None
    p2 = plane_plane_plane_check(planes[2], planes[3], main_plane)
    if p2 is None:
        return None
    p3 = plane_plane_plane_check(planes[3], planes[0], main_plane)
    if p3 is None:
        return None
    return Polyline([p0, p1, p2, p3, p0])


def plane_4planes_open(main_plane: "Plane", planes: list["Plane"]) -> object | None:
    """Same as plane_4planes but open (4 pts, last != first)."""
    from .polyline import Polyline
    p0 = plane_plane_plane_check(planes[0], planes[1], main_plane)
    if p0 is None:
        return None
    p1 = plane_plane_plane_check(planes[1], planes[2], main_plane)
    if p1 is None:
        return None
    p2 = plane_plane_plane_check(planes[2], planes[3], main_plane)
    if p2 is None:
        return None
    p3 = plane_plane_plane_check(planes[3], planes[0], main_plane)
    if p3 is None:
        return None
    return Polyline([p0, p1, p2, p3])


def plane_4lines(plane: "Plane", l0: Line, l1: Line, l2: Line, l3: Line) -> object | None:
    """Intersect plane with 4 line segments → closed quad (5 pts)."""
    from .polyline import Polyline
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


def get_quad_from_line_topbottomplanes(face_plane: "Plane", line: Line, plane0: "Plane", plane1: "Plane") -> object | None:
    """Build joint quad from collision face-plane and two bounding planes."""
    from .polyline import Polyline
    from .plane import Plane
    l0 = plane_plane(face_plane, plane0)
    l1 = plane_plane(face_plane, plane1)
    if l0 is None or l1 is None:
        return None
    # Build side planes perpendicular to the line direction
    seg_dir = line.to_vector()
    seg_start = line.start()
    seg_end = line.end()
    side0 = Plane.from_point_normal(seg_start, seg_dir)
    side1 = Plane.from_point_normal(seg_end, seg_dir)
    p0 = line_plane(l0, side0, False)
    p1 = line_plane(l0, side1, False)
    p2 = line_plane(l1, side1, False)
    p3 = line_plane(l1, side0, False)
    if any(p is None for p in (p0, p1, p2, p3)):
        return None
    return Polyline([p0, p1, p2, p3, p0])


def scale_vector_to_distance_of_2planes(direction: "Vector", p0: "Plane", p1: "Plane") -> object | None:
    """Scale direction vector so it spans the distance between two parallel planes."""
    import math
    from .vector import Vector
    mag = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
    if mag < 1e-14:
        return None
    ray = Line(0.0, 0.0, 0.0, direction[0], direction[1], direction[2])
    q0 = line_plane(ray, p0, False)
    q1 = line_plane(ray, p1, False)
    if q0 is None or q1 is None:
        return None
    output = Vector(q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2])
    # Validity: squared-distance ratio < 10 (mirrors CGAL)
    n1 = p1.z_axis
    n1_mag = math.sqrt(n1[0]**2 + n1[1]**2 + n1[2]**2)
    if n1_mag < 1e-14:
        return None
    o0 = p0.origin
    d = ((o0[0] - p1.origin[0]) * n1[0]
       + (o0[1] - p1.origin[1]) * n1[1]
       + (o0[2] - p1.origin[2]) * n1[2]) / n1_mag
    dist_ortho_sq = d * d
    if dist_ortho_sq < 1e-28:
        return None
    dist_sq = output[0]**2 + output[1]**2 + output[2]**2
    if dist_sq / dist_ortho_sq >= 10.0:
        return None
    return output


def get_orthogonal_vector_between_two_plane_pairs(pp0_0: "Plane", pp1_0: "Plane", pp1_1: "Plane") -> object | None:
    """Shortest orthogonal vector between two infinite lines defined by plane pairs."""
    from .vector import Vector
    l0 = plane_plane(pp0_0, pp1_0)
    l1 = plane_plane(pp0_0, pp1_1)
    if l0 is None or l1 is None:
        return None
    result = line_line_parameters(l0, l1, 0.0, intersect_segments=False, near_parallel_as_closest=True)
    if result is None:
        return None
    t0, t1 = result
    s0 = l0.start()
    s1 = l0.end()
    p0 = Point(s0[0] + t0*(s1[0]-s0[0]), s0[1] + t0*(s1[1]-s0[1]), s0[2] + t0*(s1[2]-s0[2]))
    r0 = l1.start()
    r1 = l1.end()
    p1 = Point(r0[0] + t1*(r1[0]-r0[0]), r0[1] + t1*(r1[1]-r0[1]), r0[2] + t1*(r1[2]-r0[2]))
    return Vector(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])


def line_two_planes(line: Line, p0: "Plane", p1: "Plane") -> object | None:
    """Clip finite segment endpoints to intersections with two planes; returns new Line or None."""
    new_start = line_plane(line, p0, True)
    new_end = line_plane(line, p1, True)
    if new_start is None or new_end is None:
        return None
    return Line(new_start[0], new_start[1], new_start[2],
                new_end[0], new_end[1], new_end[2])


def polyline_plane(poly: "Polyline", plane: "Plane") -> tuple | None:
    """Find all perimeter edge-plane intersections; returns (points, edge_indices)."""
    n = poly.point_count()
    if n < 2:
        return None
    pts_out = []
    idx_out = []
    for i in range(n - 1):
        pa = poly.get_point(i)
        pb = poly.get_point(i + 1)
        if pa is None or pb is None:
            continue
        seg = Line(pa[0], pa[1], pa[2], pb[0], pb[1], pb[2])
        pt = line_plane(seg, plane, True)
        if pt is not None:
            pts_out.append(pt)
            idx_out.append(i)
    if not pts_out:
        return None
    return (pts_out, idx_out)


def polyline_plane_to_line(poly: "Polyline", plane: "Plane", align_start: bool) -> object | None:
    """Intersect polyline perimeter with plane → single segment aligned to edge direction."""
    result = polyline_plane(poly, plane)
    if result is None:
        return None
    pts, _ = result
    if len(pts) < 2:
        return None
    p0, p1 = pts[0], pts[1]
    # Align so that p0 is closer to align_start
    d0 = (p0[0]-align_start[0])**2 + (p0[1]-align_start[1])**2 + (p0[2]-align_start[2])**2
    d1 = (p1[0]-align_start[0])**2 + (p1[1]-align_start[1])**2 + (p1[2]-align_start[2])**2
    if d1 < d0:
        p0, p1 = p1, p0
    return Line(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2])


def quad_from_line_top_bottom_planes(face_plane: "Plane", line: Line, plane0: "Plane", plane1: "Plane") -> Optional["Polyline"]:
    """Build a closed quad polyline from a joint line plus two side planes.

    End-cap planes are perpendicular to the joint line at each endpoint;
    the four corners are 3-plane intersections.

    Parameters
    ----------
    face_plane, plane0, plane1 : :class:`Plane`
    line : :class:`Line`

    Returns
    -------
    Optional[:class:`Polyline`]
        ``None`` if any of the 3-plane intersections is degenerate.
    """
    from .plane import Plane
    from .polyline import Polyline
    direction = line.to_vector()
    s = line.start()
    lp0 = Plane.from_point_normal(s, direction)
    e = line.end()
    lp1 = Plane.from_point_normal(e, direction)
    p0 = plane_plane_plane(lp0, plane0, face_plane)
    if p0 is None:
        return None
    p1 = plane_plane_plane(lp0, plane1, face_plane)
    if p1 is None:
        return None
    p2 = plane_plane_plane(lp1, plane1, face_plane)
    if p2 is None:
        return None
    p3 = plane_plane_plane(lp1, plane0, face_plane)
    if p3 is None:
        return None
    return Polyline([p0, p1, p2, p3, p0])


def orthogonal_vector_between_two_plane_pairs(pp00: "Plane", pp10: "Plane", pp11: "Plane") -> Vector | None:
    """Vector orthogonal to the (pp00, pp10) intersection line, anchored on (pp00, pp11).

    Verbatim port of wood ``cgal_intersection_util.cpp:619-628``::

        plane_plane(pp00, pp10, l0);
        plane_plane(pp00, pp11, l1);
        output = l1.point() - l0.projection(l1.point());

    Parameters
    ----------
    pp00, pp10, pp11 : :class:`Plane`

    Returns
    -------
    Optional[:class:`Vector`]
    """
    l0 = plane_plane(pp00, pp10)
    if l0 is None:
        return None
    l1 = plane_plane(pp00, pp11)
    if l1 is None:
        return None
    p1 = l1.start()
    ldir = l0.to_vector()
    len_sq = ldir[0]*ldir[0] + ldir[1]*ldir[1] + ldir[2]*ldir[2]
    if len_sq < 1e-20:
        return None
    l0s = l0.start()
    vx = p1[0] - l0s[0]
    vy = p1[1] - l0s[1]
    vz = p1[2] - l0s[2]
    t = (vx*ldir[0] + vy*ldir[1] + vz*ldir[2]) / len_sq
    px = l0s[0] + ldir[0]*t
    py = l0s[1] + ldir[1]*t
    pz = l0s[2] + ldir[2]*t
    return Vector(p1[0]-px, p1[1]-py, p1[2]-pz)


def closed_and_open_paths_2d(plate: "Polyline", joint: "Polyline", plane: "Plane") -> tuple["Polyline", tuple[float, float]] | None:
    """Clip an open joint outline against a closed plate polygon in 2D.

    Port of the wood ``wood_element.cpp:438-651`` helper. Returns the clipped
    3D polyline plus parametric positions ``(t0, t1)`` on the plate edges, or
    ``None`` if the joint outline does not intersect the plate polygon.

    Parameters
    ----------
    plate : :class:`Polyline`
    joint : :class:`Polyline`
    plane : :class:`Plane`

    Returns
    -------
    Optional[Tuple[:class:`Polyline`, Tuple[float, float]]]
    """
    from .polyline import Polyline
    import math as _math

    origin = plate.get_point(0)
    xax = plane.x_axis
    yax = plane.y_axis
    xax = Vector(xax[0], xax[1], xax[2]); xax.normalize_self()
    yax = Vector(yax[0], yax[1], yax[2]); yax.normalize_self()

    def to_2d(pp):
        dx = pp[0]-origin[0]
        dy = pp[1]-origin[1]
        dz = pp[2]-origin[2]
        return (dx*xax[0]+dy*xax[1]+dz*xax[2],
                dx*yax[0]+dy*yax[1]+dz*yax[2])

    def to_3d(u, v):
        return Point(origin[0] + u*xax[0] + v*yax[0],
                     origin[1] + u*xax[1] + v*yax[1],
                     origin[2] + u*xax[2] + v*yax[2])

    # Plate outline (2D), strip closing duplicate.
    plate_n = plate.point_count()
    if plate_n > 1:
        f = plate.get_point(0)
        l = plate.get_point(plate_n-1)
        if abs(f[0]-l[0]) < 1e-6 and abs(f[1]-l[1]) < 1e-6 and abs(f[2]-l[2]) < 1e-6:
            plate_n -= 1
    plate2d = [to_2d(plate.get_point(i)) for i in range(plate_n)]
    if len(plate2d) < 3:
        return None

    joint2d = [to_2d(joint.get_point(i)) for i in range(joint.point_count())]
    if len(joint2d) < 2:
        return None

    def pip(px, py):
        wn = 0
        n = len(plate2d)
        for i in range(n):
            ax, ay = plate2d[i]
            bx, by = plate2d[(i+1) % n]
            if ay <= py:
                if by > py:
                    e = (bx-ax)*(py-ay) - (px-ax)*(by-ay)
                    if e > 0.0:
                        wn += 1
            else:
                if by <= py:
                    e = (bx-ax)*(py-ay) - (px-ax)*(by-ay)
                    if e < 0.0:
                        wn -= 1
        return wn != 0

    def seg_seg_2d(s0, s1, e0, e1):
        sx = s1[0]-s0[0]; sy = s1[1]-s0[1]
        ex = e1[0]-e0[0]; ey = e1[1]-e0[1]
        denom = sx*ey - sy*ex
        if abs(denom) < 1e-20:
            return None
        dx = e0[0]-s0[0]; dy = e0[1]-s0[1]
        t_s = (dx*ey - dy*ex) / denom
        t_e = (dx*sy - dy*sx) / denom
        return t_s, t_e

    EPS = 1e-9
    pieces = []
    for s in range(len(joint2d)-1):
        p0 = joint2d[s]
        p1 = joint2d[s+1]
        ts = [0.0]
        for i in range(len(plate2d)):
            a = plate2d[i]
            b = plate2d[(i+1) % len(plate2d)]
            r = seg_seg_2d(p0, p1, a, b)
            if r is None:
                continue
            t_s, t_e = r
            if EPS < t_s < 1.0 - EPS and -EPS <= t_e <= 1.0 + EPS:
                ts.append(t_s)
        ts.append(1.0)
        ts.sort()
        # Deduplicate close-to-equal values.
        deduped = [ts[0]]
        for v in ts[1:]:
            if abs(v - deduped[-1]) > EPS:
                deduped.append(v)
        ts = deduped

        current = []
        for i in range(len(ts) - 1):
            t_mid = 0.5 * (ts[i] + ts[i+1])
            mx = p0[0] + (p1[0]-p0[0]) * t_mid
            my = p0[1] + (p1[1]-p0[1]) * t_mid
            if pip(mx, my):
                sub_a = (p0[0] + (p1[0]-p0[0])*ts[i],   p0[1] + (p1[1]-p0[1])*ts[i])
                sub_b = (p0[0] + (p1[0]-p0[0])*ts[i+1], p0[1] + (p1[1]-p0[1])*ts[i+1])
                if not current:
                    current.append(sub_a)
                    current.append(sub_b)
                else:
                    dx = current[-1][0] - sub_a[0]
                    dy = current[-1][1] - sub_a[1]
                    if dx*dx + dy*dy < 1e-18:
                        current.append(sub_b)
                    else:
                        pieces.append(current)
                        current = [sub_a, sub_b]
            else:
                if current:
                    pieces.append(current)
                    current = []
        if current:
            pieces.append(current)

    def sq2(a, b):
        dx = a[0]-b[0]; dy = a[1]-b[1]
        return dx*dx + dy*dy

    DISTANCE_SQ = 0.01
    c2d = []
    count = 0
    for piece in pieces:
        if len(piece) <= 1:
            continue
        if count == 0:
            c2d = list(piece)
        else:
            pts = list(piece)
            if sq2(c2d[-1], pts[0]) > DISTANCE_SQ and sq2(c2d[-1], pts[-1]) > DISTANCE_SQ:
                c2d.reverse()
            if sq2(c2d[-1], pts[0]) > sq2(c2d[-1], pts[-1]):
                pts.reverse()
            for j in range(1, len(pts)):
                c2d.append(pts[j])
        count += 1

    if len(c2d) < 2:
        return None

    def closest_param(p, a, b):
        abx = b[0]-a[0]; aby = b[1]-a[1]
        l2 = abx*abx + aby*aby
        if l2 < 1e-20:
            return 0.0
        apx = p[0]-a[0]; apy = p[1]-a[1]
        t = (apx*abx + apy*aby) / l2
        if t < 0.0: t = 0.0
        if t > 1.0: t = 1.0
        return t

    def sq_dist_seg(p, a, b):
        abx = b[0]-a[0]; aby = b[1]-a[1]
        l2 = abx*abx + aby*aby
        if l2 < 1e-20:
            dx = p[0]-a[0]; dy = p[1]-a[1]
            return dx*dx + dy*dy
        apx = p[0]-a[0]; apy = p[1]-a[1]
        t = (apx*abx + apy*aby) / l2
        if t < 0.0: t = 0.0
        if t > 1.0: t = 1.0
        px = a[0] + t*abx
        py = a[1] + t*aby
        dx = p[0]-px
        dy = p[1]-py
        return dx*dx + dy*dy

    t0 = -1.0
    t1 = -1.0
    for i in range(len(plate2d)):
        a = plate2d[i]
        b = plate2d[(i+1) % len(plate2d)]
        for jj in range(2):
            idx = 0 if jj == 0 else len(c2d) - 1
            d = sq_dist_seg(c2d[idx], a, b)
            if jj == 0 and d < 1.0:
                t0 = float(i) + closest_param(c2d[0], a, b)
            elif jj == 1 and d < 1.0:
                t1 = float(i) + closest_param(c2d[-1], a, b)
        if t0 >= 0.0 and t1 >= 0.0:
            break

    reverse_flag = (t0 > t1)
    if int(_math.floor(t0)) == 0 and int(_math.floor(t1)) == len(c2d) - 1:
        reverse_flag = not reverse_flag
    if reverse_flag:
        t0, t1 = t1, t0
        c2d.reverse()

    if t0 < 0.0 or t1 < 0.0:
        return None

    out_pts = [to_3d(p[0], p[1]) for p in c2d]
    return Polyline(out_pts), (t0, t1)


def line_line_3d(cutter: Line, seg: Line) -> Point | None:
    """3D skew-line intersection via infinite line_line_parameters."""
    result = line_line_parameters(cutter, seg, 0.0,
                                  intersect_segments=False,
                                  near_parallel_as_closest=False)
    if result is None:
        return None
    t0, _ = result
    s = cutter.start()
    e = cutter.end()
    return Point(s[0] + t0*(e[0]-s[0]),
                 s[1] + t0*(e[1]-s[1]),
                 s[2] + t0*(e[2]-s[2]))


def face_to_face(adjacency: list[int], polylines_list: list[list["Polyline"]], planes_list: list[list["Plane"]], coplanar_tolerance: float = 5.0) -> list[tuple[int, int, int, int, int, "Polyline"]]:
    """Face-to-face joint detection between elements.
    Returns list of (a, b, face_a, face_b, type, joint_polyline).

    Optimization: pre-pack all face origins/normals into numpy arrays once and
    do a vectorized coplanar test per adjacency pair, replacing ~166k Python
    calls to Plane.is_coplanar_from_normals (the dominant ~38% hot spot in
    `compute_face_to_face` profiling).
    """
    import numpy as np
    import math
    from .polyline import Polyline
    from .plane import Plane
    from .vector import Vector
    from .tolerance import Tolerance
    from .tolerance import TO_RADIANS

    # Pre-pack all face planes into flat (M, 3) ndarrays. face_starts[i] gives
    # the index into the flat arrays where element i's faces begin.
    n_elems = len(planes_list)
    face_starts = np.zeros(n_elems + 1, dtype=np.int64)
    for i, planes in enumerate(planes_list):
        face_starts[i + 1] = face_starts[i] + len(planes)
    total_faces = int(face_starts[-1])
    face_origins = np.empty((total_faces, 3), dtype=np.float64)
    face_normals = np.empty((total_faces, 3), dtype=np.float64)
    k = 0
    for planes in planes_list:
        for p in planes:
            o = p.origin
            n = p.z_axis
            face_origins[k, 0] = o[0]; face_origins[k, 1] = o[1]; face_origins[k, 2] = o[2]
            face_normals[k, 0] = n[0]; face_normals[k, 1] = n[1]; face_normals[k, 2] = n[2]
            k += 1

    # Per-face inflated AABBs, packed once - two faces can only yield a NON-EMPTY contact
    # area if their inflated boxes overlap, so each pair rejects on a broadcast box test
    # before any plane math or polygon boolean runs. This kills the coplanar-but-distant
    # pairs (same infinite plane, far apart) that used to reach boolean_op and return empty.
    # Conservative by the coplanar tolerance, so accepted pairs are unchanged.
    face_lo = np.empty((total_faces, 3), dtype=np.float64)
    face_hi = np.empty((total_faces, 3), dtype=np.float64)
    k = 0
    for faces in polylines_list:
        for f in faces:
            c = np.asarray(f.coords, dtype=np.float64).reshape(-1, 3)
            face_lo[k] = c.min(axis=0) - coplanar_tolerance
            face_hi[k] = c.max(axis=0) + coplanar_tolerance
            k += 1

    # Vectorized coplanar test, matching Plane.is_coplanar_from_normals with
    # can_be_flipped=False (only antiparallel normals accepted, i.e. faces
    # touching back-to-back).
    cos_tol = math.cos(Tolerance.ANGLE_TOLERANCE_DEGREES * TO_RADIANS)

    results = []
    for idx in range(0, len(adjacency), 4):
        a, b = adjacency[idx], adjacency[idx + 1]

        a0, a1 = int(face_starts[a]), int(face_starts[a + 1])
        b0, b1 = int(face_starts[b]), int(face_starts[b + 1])
        oa = face_origins[a0:a1]                # (na, 3)
        na_ = face_normals[a0:a1]               # (na, 3)
        ob = face_origins[b0:b1]                # (nb, 3)
        nb_ = face_normals[b0:b1]               # (nb, 3)

        # Box gate first: overlap[i, j] on all three axes, broadcast (na, nb).
        alo = face_lo[a0:a1]; ahi = face_hi[a0:a1]
        blo = face_lo[b0:b1]; bhi = face_hi[b0:b1]
        overlap = np.all(
            (alo[:, None, :] <= bhi[None, :, :]) & (blo[None, :, :] <= ahi[:, None, :]),
            axis=2,
        )
        if not overlap.any():
            continue

        # Antiparallel check: dot(na, nb) <= -cos_tol  → mask shape (na, nb)
        dots = na_ @ nb_.T                                          # (na, nb)
        antiparallel = dots <= -cos_tol

        # Plane-distance check: |na · (ob - oa)| < tol  AND  |nb · (oa - ob)| < tol
        # Using na_dot_oa[i] = na_[i] · oa[i], na_dot_ob[i,j] = na_[i] · ob[j].
        na_dot_oa = np.einsum('ij,ij->i', na_, oa)                  # (na,)
        nb_dot_ob = np.einsum('ij,ij->i', nb_, ob)                  # (nb,)
        na_dot_ob = na_ @ ob.T                                      # (na, nb)
        nb_dot_oa = nb_ @ oa.T                                      # (nb, na) → transpose
        dist0 = np.abs(na_dot_ob - na_dot_oa[:, None])              # (na, nb)
        dist1 = np.abs(nb_dot_oa - nb_dot_ob[:, None]).T            # (na, nb)
        coplanar_mask = overlap & antiparallel & (dist0 < coplanar_tolerance) & (dist1 < coplanar_tolerance)

        if not coplanar_mask.any():
            continue

        # Walk row-major to preserve the original loop order: first matching
        # (i, j) for which the boolean op produces a valid polygon wins.
        # np.nonzero on a 2D array returns indices in row-major (C) order.
        ii_arr, jj_arr = np.nonzero(coplanar_mask)
        for k in range(len(ii_arr)):
            i = int(ii_arr[k])
            j = int(jj_arr[k])
            pts_i = polylines_list[a][i].get_points()
            if len(pts_i) < 2:
                continue
            edge = Vector(pts_i[1][0] - pts_i[0][0], pts_i[1][1] - pts_i[0][1], pts_i[1][2] - pts_i[0][2])
            edge.normalize_self()
            zax = planes_list[a][i].z_axis
            yax = zax.cross(edge)
            yax.normalize_self()
            pln = Plane(pts_i[0], edge, yax)
            bools = Polyline.boolean_op(polylines_list[a][i], polylines_list[b][j], 0, plane=pln)
            if not bools or bools[0].point_count() < 3:
                continue
            type_val = (0 if i > 1 else 1) + (0 if j > 1 else 1)
            jpl = bools[0] if bools[0].is_closed() else bools[0].closed()
            results.append((a, b, i, j, type_val, jpl))
            break
    return results


def polyline_boolean(a: "Polyline", b: "Polyline", clip_type: int) -> list["Polyline"]:
    """Thin wrapper over Polyline.boolean_op mirroring C++ Intersection::polyline_boolean."""
    from .polyline import Polyline
    return Polyline.boolean_op(a, b, clip_type)


def polyline_boolean_2d_in_plane(
    polyline0: "Polyline",
    polyline1: "Polyline",
    plane: "Plane",
    intersection_type: int,
    include_triangles: bool = False,
    min_area: float = 0.01,
    collapse_eps: float = 0.0,
) -> Optional["Polyline"]:
    # 2D boolean between two closed planar polylines, projected into the plane's
    # canonical 2D frame (base1/base2). intersection_type: 0=Intersect, 1=Union,
    # 2=Difference, 3=Xor. Returns the result polyline (closed, 3D) on success,
    # or None on empty/degenerate/triangle-reject/sub-min_area. Verbatim port of
    # C++ Intersection::polyline_boolean_2d_in_plane.
    from .polyline import Polyline
    from .boolean_polyline import BooleanPolyline

    n0 = polyline0.point_count()
    n1 = polyline1.point_count()
    if n0 < 3 or n1 < 3:
        return None

    origin = polyline0.get_point(0)
    xax = plane.base1()
    yax = plane.base2()

    def to_2d(pl):
        n = pl.point_count()
        pts2d = []
        for i in range(n):
            p = pl.get_point(i)
            dx = p[0]-origin[0]; dy = p[1]-origin[1]; dz = p[2]-origin[2]
            u = dx*xax[0] + dy*xax[1] + dz*xax[2]
            v = dx*yax[0] + dy*yax[1] + dz*yax[2]
            pts2d.append(Point(u, v, 0.0))
        if len(pts2d) > 1:
            f = pts2d[0]; l = pts2d[-1]
            dx = f[0]-l[0]; dy = f[1]-l[1]
            if dx*dx + dy*dy > 1e-12:
                pts2d.append(Point(f[0], f[1], 0.0))
        return Polyline(pts2d)

    a2d = to_2d(polyline0)
    b2d = to_2d(polyline1)

    if 0 <= intersection_type <= 2:
        result_2d = BooleanPolyline.compute(a2d, b2d, intersection_type)
    elif intersection_type == 3:
        # A XOR B = (A ∪ B) − (A ∩ B). Session's BooleanPolyline lacks Xor.
        u = BooleanPolyline.compute(a2d, b2d, 1)
        inter = BooleanPolyline.compute(a2d, b2d, 0)
        if not u:
            return None
        result_2d = u if not inter else BooleanPolyline.compute(u[0], inter[0], 2)
    else:
        return None
    if not result_2d:
        return None

    C = result_2d[0]
    nc = C.point_count()
    if nc > 1:
        f = C.get_point(0); l = C.get_point(nc-1)
        dx = f[0]-l[0]; dy = f[1]-l[1]
        if dx*dx + dy*dy < 1e-12:
            nc -= 1
    if nc < 3:
        return None

    src2d = [C.get_point(i) for i in range(nc)]
    if collapse_eps > 0.0:
        eps_sq = collapse_eps * collapse_eps
        collapsed = []
        for p in src2d:
            if collapsed:
                dx = p[0] - collapsed[-1][0]
                dy = p[1] - collapsed[-1][1]
                if dx*dx + dy*dy < eps_sq:
                    continue
            collapsed.append(p)
        if len(collapsed) >= 2:
            dx = collapsed[-1][0] - collapsed[0][0]
            dy = collapsed[-1][1] - collapsed[0][1]
            if dx*dx + dy*dy < eps_sq:
                collapsed.pop()
        src2d = collapsed
        nc = len(src2d)
        if nc < 3:
            return None
    if nc == 3 and not include_triangles:
        return None

    area = 0.0
    for i in range(nc):
        p0 = src2d[i]
        p1 = src2d[(i+1) % nc]
        area += p0[0]*p1[1] - p1[0]*p0[1]
    if abs(area) * 0.5 <= min_area:
        return None

    pts = []
    for i in range(nc):
        u = src2d[i][0]; v = src2d[i][1]
        pts.append(Point(
            origin[0] + u*xax[0] + v*yax[0],
            origin[1] + u*xax[1] + v*yax[1],
            origin[2] + u*xax[2] + v*yax[2],
        ))
    pts.append(pts[0])
    return Polyline(pts)


def offset_in_3d(polyline: "Polyline", plane: "Plane", offset: float) -> bool:
    # Native miter-join polygon offset in plane-space 2D. Verbatim port of
    # Intersection::offset_in_3d. Mutates polyline in place and returns True
    # on success. Uses plane.base1()/base2() canonical axes so the output is
    # deterministic across plane constructions.
    from .polyline import Polyline
    n_raw = polyline.point_count()
    if n_raw < 3:
        return False

    origin = polyline.get_point(0)
    xax = plane.base1()
    yax = plane.base2()

    path = []
    for i in range(n_raw):
        p = polyline.get_point(i)
        dx = p[0] - origin[0]; dy = p[1] - origin[1]; dz = p[2] - origin[2]
        u = dx*xax[0] + dy*xax[1] + dz*xax[2]
        v = dx*yax[0] + dy*yax[1] + dz*yax[2]
        path.append((u, v))
    if len(path) >= 2:
        dx = path[-1][0] - path[0][0]; dy = path[-1][1] - path[0][1]
        if dx*dx + dy*dy < 1e-12:
            path.pop()
    n = len(path)
    if n < 3:
        return False

    signed_area = 0.0
    for i in range(n):
        ax, ay = path[i]
        bx, by = path[(i+1) % n]
        signed_area += ax * by - bx * ay
    delta = -offset if signed_area < 0.0 else offset

    normals = []
    for i in range(n):
        ax, ay = path[i]
        bx, by = path[(i+1) % n]
        ex = bx - ax; ey = by - ay
        length = (ex*ex + ey*ey) ** 0.5
        if length < 1e-12:
            normals.append((0.0, 0.0))
        else:
            normals.append((ey/length, -ex/length))

    out = []
    for i in range(n):
        npx, npy = normals[(i + n - 1) % n]
        nnx, nny = normals[i]
        cos_a = npx*nnx + npy*nny
        sin_a = npx*nny - npy*nnx
        denom = 1.0 + cos_a
        concave = (cos_a > -0.999) and (sin_a * delta < 0.0) and (offset > 0.0)
        px, py = path[i]
        if concave:
            out.append((px + npx * delta, py + npy * delta))
            out.append((px, py))
            out.append((px + nnx * delta, py + nny * delta))
        elif abs(denom) < 1e-9:
            bx = npx + nnx; by = npy + nny
            bl = (bx*bx + by*by) ** 0.5
            if bl < 1e-12:
                out.append((px + nnx * delta, py + nny * delta))
            else:
                out.append((px + (bx/bl) * delta, py + (by/bl) * delta))
        else:
            k = delta / denom
            out.append((px + (npx + nnx) * k, py + (npy + nny) * k))

    nout = len(out)
    if nout < 3:
        return False

    out_area = 0.0
    for i in range(nout):
        ax, ay = out[i]
        bx, by = out[(i+1) % nout]
        out_area += ax * by - bx * ay
    if abs(out_area) * 0.5 < 0.0001:
        return False

    cp = 0
    cd = (out[0][0] - path[0][0])**2 + (out[0][1] - path[0][1])**2
    for i in range(1, nout):
        d = (out[i][0] - path[0][0])**2 + (out[i][1] - path[0][1])**2
        if d < cd:
            cd = d; cp = i
    if cp != 0:
        out = out[cp:] + out[:cp]

    pts = []
    for u, v in out:
        pts.append(Point(
            origin[0] + u*xax[0] + v*yax[0],
            origin[1] + u*xax[1] + v*yax[1],
            origin[2] + u*xax[2] + v*yax[2],
        ))
    pts.append(pts[0])
    new_poly = Polyline(pts)
    # Replace coords in-place so caller references remain valid.
    polyline.coords = new_poly.coords
    return True


def adjacency_search(elements: list["Element"], inflate: float = 5.0) -> list[int]:
    """SpatialBVH/brute-force adjacency search. Returns flat list [a, b, -1, -1, ...]."""
    from .aabb import AABB
    N = len(elements)
    aabbs = []
    for elem in elements:
        polys = elem.polylines if hasattr(elem, "polylines") and callable(getattr(elem, "polylines", None)) else []
        if callable(getattr(elem, "compute_polylines", None)):
            polys = elem.compute_polylines()
        pts = []
        for pl in polys:
            pts.extend(pl.get_points())
        if pts:
            aabbs.append(AABB.from_points(pts, inflate))
        else:
            aabbs.append(AABB.from_point(Point(0,0,0), inflate))
    adjacency = []
    for i in range(N):
        for j in range(i+1, N):
            if aabbs[i].intersects(aabbs[j]):
                adjacency.extend([i, j, -1, -1])
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
    """Port of cgal_box_search.h:252-496. Classifies two finite segments s0/s1
    as end-to-end, side-to-end, or cross based on above_closer_to_edge in [0,1].
    n_segs_*/cur_seg_* give the segment's position within its parent polyline
    (used for full-polyline parameter remapping that determines type0/type1).
    Returns (p0, p1, v0, v1, normal, type0, type1, is_parallel):
      p0/p1 = closest approach points (clamped to [0,1] on each segment);
      v0/v1 = unit directions, flipped away from the far end when type=0;
      normal = unit perpendicular to both segments (Plane.base1 if parallel);
      type0/type1 = 0 means segment-end, 1 means segment-interior;
      is_parallel = True when v0 x v1 is near-zero.
    Returns None for degenerate cases."""
    from .plane import Plane
    DIST_SQ = 1e-6
    EPS_PAR = 1.0  # degrees

    v0 = s0.to_vector()
    v1 = s1.to_vector()
    normal = v0.cross(v1)
    nmag2 = normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]
    ang = v0.angle(v1, False, True)
    is_parallel = (nmag2 < 1e-24) or ((90.0 - abs(ang - 90.0)) < EPS_PAR)
    if is_parallel:
        tmp_origin = s0.start()
        tmp_normal = v0
        pl_tmp = Plane.from_point_normal(tmp_origin, tmp_normal)
        normal = pl_tmp.base1()
    normal.normalize_self()

    def eq(a, b):
        dx = a[0]-b[0]; dy = a[1]-b[1]; dz = a[2]-b[2]
        return dx*dx+dy*dy+dz*dz < DIST_SQ

    def endcase(pp, dv0, dv1):
        p0 = pp; p1 = pp
        v0 = dv0; v0.normalize_self()
        v1 = dv1; v1.normalize_self()
        return (p0, p1, v0, v1, normal, 0, 0, is_parallel)

    if eq(s0.start(), s1.start()):
        return endcase(s0.start(), s0.end()-s0.start(), s1.end()-s1.start())
    if eq(s0.start(), s1.end()):
        return endcase(s0.start(), s0.end()-s0.start(), s1.start()-s1.end())
    if eq(s0.end(), s1.start()):
        return endcase(s0.end(), s0.start()-s0.end(), s1.end()-s1.start())
    if eq(s0.end(), s1.end()):
        return endcase(s0.end(), s0.start()-s0.end(), s1.start()-s1.end())

    if is_parallel:
        v0.normalize_self()
        v1.normalize_self()

        def signed_t(src, unit, q):
            return (q[0]-src[0])*unit[0] + (q[1]-src[1])*unit[1] + (q[2]-src[2])*unit[2]

        def proj_onto_line(L, q):
            return L.closest_point(q, False)[1]

        pts = []

        def push(q):
            q0 = proj_onto_line(s0, q)
            q1 = proj_onto_line(s1, q)
            pts.append((signed_t(s0.start(), v0, q0),
                        signed_t(s1.start(), v1, q1)))

        push(s0.start()); push(s0.end())
        push(s1.start()); push(s1.end())
        pts.sort(key=lambda a: a[0])
        seg0_a = Point(s0.start()[0]+pts[1][0]*v0[0],
                       s0.start()[1]+pts[1][0]*v0[1],
                       s0.start()[2]+pts[1][0]*v0[2])
        seg0_b = Point(s0.start()[0]+pts[2][0]*v0[0],
                       s0.start()[1]+pts[2][0]*v0[1],
                       s0.start()[2]+pts[2][0]*v0[2])
        seg1_a = Point(s1.start()[0]+pts[1][1]*v1[0],
                       s1.start()[1]+pts[1][1]*v1[1],
                       s1.start()[2]+pts[1][1]*v1[2])
        seg1_b = Point(s1.start()[0]+pts[2][1]*v1[0],
                       s1.start()[1]+pts[2][1]*v1[1],
                       s1.start()[2]+pts[2][1]*v1[2])
        m0 = Point((seg0_a[0]+seg0_b[0])*0.5, (seg0_a[1]+seg0_b[1])*0.5, (seg0_a[2]+seg0_b[2])*0.5)
        m1 = Point((seg1_a[0]+seg1_b[0])*0.5, (seg1_a[1]+seg1_b[1])*0.5, (seg1_a[2]+seg1_b[2])*0.5)
        avg = Point((m0[0]+m1[0])*0.5, (m0[1]+m1[1])*0.5, (m0[2]+m1[2])*0.5)
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
        type0 = 1; type1 = 1
    elif above_closer_to_edge > 1.0:
        type0 = 0 if tt0 < tt1 else 1
        type1 = 1 if tt0 < tt1 else 0
    else:
        type0 = 0 if close0 > above_closer_to_edge else 1
        type1 = 0 if close1 > above_closer_to_edge else 1
        if close0 > close1 and type0 == 0 and type1 == 0:
            type0 = 0; type1 = 1
        elif close0 < close1 and type0 == 0 and type1 == 0:
            type0 = 1; type1 = 0

    if tt0 > 0.5 and type0 == 0:
        v0 = Vector(-v0[0], -v0[1], -v0[2])
    if tt1 > 0.5 and type1 == 0:
        v1 = Vector(-v1[0], -v1[1], -v1[2])
    return (p0, p1, v0, v1, normal, type0, type1, is_parallel)

