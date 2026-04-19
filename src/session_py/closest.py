"""Closest point operations for geometry classes."""

from typing import Tuple
import math

from session_py.point import Point
from session_py.vector import Vector
from session_py.line import Line
from session_py.polyline import Polyline


class Closest:
    """Static methods for finding closest points between geometry objects."""

    @staticmethod
    def curve_point(curve, test_point: Point, t0: float = 0.0, t1: float = 0.0) -> Tuple[float, float]:
        """Find closest point on NURBS curve to test point.

        Parameters
        ----------
        curve : NurbsCurve
            The NURBS curve.
        test_point : Point
            Point to find closest curve point to.
        t0 : float
            Start of search interval. 0 means curve start.
        t1 : float
            End of search interval. 0 means curve end.

        Returns
        -------
        tuple of (float, float)
            (parameter, distance) of closest point.
        """
        if not curve.is_valid():
            return (0.0, float('inf'))

        domain_start, domain_end = curve.domain()
        if t0 <= 0.0:
            t0 = domain_start
        if t1 <= 0.0:
            t1 = domain_end

        t0 = max(t0, domain_start)
        t1 = min(t1, domain_end)

        num_samples = max(10, curve.degree() * 2)
        dt = (t1 - t0) / num_samples

        best_t = t0
        best_dist = curve.point_at(t0).distance(test_point)

        for i in range(num_samples + 1):
            t = t0 + i * dt
            dist = curve.point_at(t).distance(test_point)
            if dist < best_dist:
                best_dist = dist
                best_t = t

        max_iterations = 20
        step_tolerance = (t1 - t0) * 1e-10

        t = best_t

        for _ in range(max_iterations):
            pt = curve.point_at(t)
            tangent = curve.tangent_at(t)

            delta = Vector(
                test_point[0] - pt[0],
                test_point[1] - pt[1],
                test_point[2] - pt[2]
            )

            f = -delta.dot(tangent)

            if abs(f) < step_tolerance:
                break

            derivs = curve.evaluate(t, 2)
            if len(derivs) < 3:
                break

            d2 = Vector(derivs[2][0], derivs[2][1], derivs[2][2])
            tangent_mag = tangent.magnitude()
            df = delta.dot(d2) - tangent_mag * tangent_mag

            if abs(df) < 1e-12:
                break

            dt_step = f / df

            if abs(dt_step) > (t1 - t0) * 0.5:
                dt_step = math.copysign((t1 - t0) * 0.5, dt_step)

            t += dt_step

            if t < t0:
                t = t0
            if t > t1:
                t = t1

            if abs(dt_step) < step_tolerance:
                break

        final_dist = curve.point_at(t).distance(test_point)

        dist_start = curve.point_at(t0).distance(test_point)
        dist_end = curve.point_at(t1).distance(test_point)

        if dist_start < final_dist:
            t = t0
            final_dist = dist_start
        if dist_end < final_dist:
            t = t1
            final_dist = dist_end

        return (t, final_dist)

    @staticmethod
    def line_point(line: Line, test_point: Point) -> Tuple[Point, float, float]:
        """Find closest point on line to test point.

        Parameters
        ----------
        line : Line
            The line segment.
        test_point : Point
            Point to find closest line point to.

        Returns
        -------
        tuple of (Point, float, float)
            (closest_point, parameter, distance).
        """
        start = line.start()
        end = line.end()

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]

        len_sq = dx * dx + dy * dy + dz * dz

        if len_sq < 1e-20:
            dist = start.distance(test_point)
            return (start, 0.0, dist)

        t = ((test_point[0] - start[0]) * dx +
             (test_point[1] - start[1]) * dy +
             (test_point[2] - start[2]) * dz) / len_sq

        t = max(0.0, min(1.0, t))

        closest = Point(
            start[0] + t * dx,
            start[1] + t * dy,
            start[2] + t * dz
        )

        dist = closest.distance(test_point)

        return (closest, t, dist)

    @staticmethod
    def polyline_point(polyline: Polyline, test_point: Point) -> Tuple[Point, float, float]:
        """Find closest point on polyline to test point.

        Parameters
        ----------
        polyline : Polyline
            The polyline.
        test_point : Point
            Point to find closest polyline point to.

        Returns
        -------
        tuple of (Point, float, float)
            (closest_point, parameter, distance).
        """
        points = polyline.get_points()

        if not points:
            return (Point(0, 0, 0), 0.0, float('inf'))

        if len(points) == 1:
            dist = points[0].distance(test_point)
            return (points[0], 0.0, dist)

        best_point = points[0]
        best_param = 0.0
        best_dist = float('inf')

        cumulative_length = 0.0
        total_length = polyline.length()

        for i in range(len(points) - 1):
            segment = Line.from_points(points[i], points[i + 1])
            closest, t, dist = Closest.line_point(segment, test_point)

            if dist < best_dist:
                best_dist = dist
                best_point = closest
                segment_length = segment.length()
                if total_length > 1e-20:
                    best_param = (cumulative_length + t * segment_length) / total_length
                else:
                    best_param = float(i) / (len(points) - 1)

            cumulative_length += segment.length()

        return (best_point, best_param, best_dist)

    @staticmethod
    def surface_point(surface, test_point: Point, u0: float = 0.0, u1: float = 0.0,
                      v0: float = 0.0, v1: float = 0.0) -> Tuple[float, float, float]:
        """Find closest point on NURBS surface to test point.

        Parameters
        ----------
        surface : NurbsSurface
            The NURBS surface.
        test_point : Point
            Point to find closest surface point to.
        u0, u1 : float
            U parameter search interval. 0 means use surface domain.
        v0, v1 : float
            V parameter search interval. 0 means use surface domain.

        Returns
        -------
        tuple of (float, float, float)
            (u_param, v_param, distance).
        """
        if not surface.is_valid():
            return (0.0, 0.0, float('inf'))

        domain_u0, domain_u1 = surface.domain(0)
        domain_v0, domain_v1 = surface.domain(1)

        if u0 <= 0.0:
            u0 = domain_u0
        if u1 <= 0.0:
            u1 = domain_u1
        if v0 <= 0.0:
            v0 = domain_v0
        if v1 <= 0.0:
            v1 = domain_v1

        u0 = max(u0, domain_u0)
        u1 = min(u1, domain_u1)
        v0 = max(v0, domain_v0)
        v1 = min(v1, domain_v1)

        u_samples = max(10, surface.order(0))
        v_samples = max(10, surface.order(1))

        du_param = (u1 - u0) / u_samples
        dv_param = (v1 - v0) / v_samples

        best_u = u0
        best_v = v0
        best_dist = float('inf')

        for i in range(u_samples + 1):
            for j in range(v_samples + 1):
                uu = u0 + i * du_param
                vv = v0 + j * dv_param
                pt = surface.point_at(uu, vv)
                dist = pt.distance(test_point)
                if dist < best_dist:
                    best_dist = dist
                    best_u = uu
                    best_v = vv

        max_iterations = 20
        step_tolerance = min(u1 - u0, v1 - v0) * 1e-10

        u = best_u
        v = best_v

        for _ in range(max_iterations):
            derivs = surface.evaluate(u, v, 1)
            if len(derivs) < 3:
                break

            pt = surface.point_at(u, v)
            du_vec = derivs[2]  # evaluate returns [S, Sv, Su, ...]
            dv_vec = derivs[1]

            delta = Vector(
                test_point[0] - pt[0],
                test_point[1] - pt[1],
                test_point[2] - pt[2]
            )

            fu = -delta.dot(du_vec)
            fv = -delta.dot(dv_vec)

            if abs(fu) < step_tolerance and abs(fv) < step_tolerance:
                break

            duu = du_vec.dot(du_vec)
            dvv = dv_vec.dot(dv_vec)
            duv = du_vec.dot(dv_vec)

            det = duu * dvv - duv * duv
            if abs(det) < 1e-12:
                break

            du_step = (dvv * fu - duv * fv) / det
            dv_step = (duu * fv - duv * fu) / det

            max_step = min(u1 - u0, v1 - v0) * 0.5
            if abs(du_step) > max_step:
                du_step = math.copysign(max_step, du_step)
            if abs(dv_step) > max_step:
                dv_step = math.copysign(max_step, dv_step)

            u -= du_step
            v -= dv_step

            u = max(u0, min(u1, u))
            v = max(v0, min(v1, v))

            if abs(du_step) < step_tolerance and abs(dv_step) < step_tolerance:
                break

        final_dist = surface.point_at(u, v).distance(test_point)

        return (u, v, final_dist)

    @staticmethod
    def _closest_point_on_triangle(p, a, b, c):
        abx, aby, abz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
        acx, acy, acz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
        apx, apy, apz = p[0]-a[0], p[1]-a[1], p[2]-a[2]

        d1 = abx*apx + aby*apy + abz*apz
        d2 = acx*apx + acy*apy + acz*apz
        if d1 <= 0.0 and d2 <= 0.0:
            return a

        bpx, bpy, bpz = p[0]-b[0], p[1]-b[1], p[2]-b[2]
        d3 = abx*bpx + aby*bpy + abz*bpz
        d4 = acx*bpx + acy*bpy + acz*bpz
        if d3 >= 0.0 and d4 <= d3:
            return b

        vc = d1*d4 - d3*d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return Point(a[0] + v*abx, a[1] + v*aby, a[2] + v*abz)

        cpx, cpy, cpz = p[0]-c[0], p[1]-c[1], p[2]-c[2]
        d5 = abx*cpx + aby*cpy + abz*cpz
        d6 = acx*cpx + acy*cpy + acz*cpz
        if d6 >= 0.0 and d5 <= d6:
            return c

        vb = d5*d2 - d1*d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return Point(a[0] + w*acx, a[1] + w*acy, a[2] + w*acz)

        va = d3*d6 - d5*d4
        if va <= 0.0 and (d4-d3) >= 0.0 and (d5-d6) >= 0.0:
            w = (d4-d3) / ((d4-d3) + (d5-d6))
            return Point(b[0] + w*(c[0]-b[0]), b[1] + w*(c[1]-b[1]), b[2] + w*(c[2]-b[2]))

        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return Point(a[0] + abx*v + acx*w, a[1] + aby*v + acy*w, a[2] + abz*v + acz*w)

    @staticmethod
    def mesh_point(mesh, test_point):
        if mesh.number_of_faces() == 0:
            return (Point(0, 0, 0), 0, float('inf'))

        vertices, faces = mesh.to_vertices_and_faces()
        sorted_face_keys = sorted(mesh.face.keys())

        best_point = Point(0, 0, 0)
        best_face_key = 0
        best_dist = float('inf')

        for fi, fv in enumerate(faces):
            if len(fv) < 3:
                continue
            v0 = vertices[fv[0]]
            for j in range(1, len(fv) - 1):
                v1 = vertices[fv[j]]
                v2 = vertices[fv[j + 1]]
                cp = Closest._closest_point_on_triangle(test_point, v0, v1, v2)
                dist = cp.distance(test_point)
                if dist < best_dist:
                    best_dist = dist
                    best_point = cp
                    best_face_key = sorted_face_keys[fi]

        return (best_point, best_face_key, best_dist)

    @staticmethod
    def mesh_point_aabb(mesh, test_point):
        if mesh.number_of_faces() == 0:
            return (Point(0, 0, 0), 0, float('inf'))

        vertices, faces = mesh.to_vertices_and_faces()
        sorted_face_keys = sorted(mesh.face.keys())

        tris = []
        tri_face_idx = []
        for fi, fv in enumerate(faces):
            if len(fv) < 3:
                continue
            v0 = vertices[fv[0]]
            for j in range(1, len(fv) - 1):
                tris.append((v0, vertices[fv[j]], vertices[fv[j + 1]]))
                tri_face_idx.append(fi)

        if not tris:
            return (Point(0, 0, 0), 0, float('inf'))

        boxes = []
        for v0, v1, v2 in tris:
            lx = min(v0[0], v1[0], v2[0])
            ly = min(v0[1], v1[1], v2[1])
            lz = min(v0[2], v1[2], v2[2])
            hx = max(v0[0], v1[0], v2[0])
            hy = max(v0[1], v1[1], v2[1])
            hz = max(v0[2], v1[2], v2[2])
            boxes.append(((lx+hx)*0.5, (ly+hy)*0.5, (lz+hz)*0.5,
                          (hx-lx)*0.5, (hy-ly)*0.5, (hz-lz)*0.5))

        nodes = []

        def build_node(ids):
            ni = len(nodes)
            nodes.append([None, -1, -1])
            lx = ly = lz = 1e308
            hx = hy = hz = -1e308
            for i in ids:
                b = boxes[i]
                lx = min(lx, b[0]-b[3]); hx = max(hx, b[0]+b[3])
                ly = min(ly, b[1]-b[4]); hy = max(hy, b[1]+b[4])
                lz = min(lz, b[2]-b[5]); hz = max(hz, b[2]+b[5])
            nodes[ni][0] = ((lx+hx)*0.5, (ly+hy)*0.5, (lz+hz)*0.5,
                            (hx-lx)*0.5, (hy-ly)*0.5, (hz-lz)*0.5)
            if len(ids) == 1:
                nodes[ni][2] = ids[0]
                return
            dx, dy, dz = hx-lx, hy-ly, hz-lz
            axis = 0 if dx >= dy and dx >= dz else (1 if dy >= dz else 2)
            ids.sort(key=lambda i: boxes[i][axis])
            mid = len(ids) // 2
            build_node(ids[:mid])
            nodes[ni][1] = len(nodes)
            build_node(ids[mid:])

        build_node(list(range(len(tris))))

        def aabb_min_dist(aabb, pt):
            dx = max(0.0, abs(pt[0] - aabb[0]) - aabb[3])
            dy = max(0.0, abs(pt[1] - aabb[1]) - aabb[4])
            dz = max(0.0, abs(pt[2] - aabb[2]) - aabb[5])
            return (dx*dx + dy*dy + dz*dz) ** 0.5

        best = [Point(0, 0, 0), 0, float('inf')]

        def dfs(ni):
            aabb, right, obj = nodes[ni]
            if aabb_min_dist(aabb, test_point) >= best[2]:
                return
            if obj >= 0:
                v0, v1, v2 = tris[obj]
                cp = Closest._closest_point_on_triangle(test_point, v0, v1, v2)
                d = cp.distance(test_point)
                if d < best[2]:
                    best[0] = cp
                    best[1] = sorted_face_keys[tri_face_idx[obj]]
                    best[2] = d
                return
            left = ni + 1
            ld = aabb_min_dist(nodes[left][0], test_point)
            rd = aabb_min_dist(nodes[right][0], test_point)
            if ld <= rd:
                if ld < best[2]: dfs(left)
                if rd < best[2]: dfs(right)
            else:
                if rd < best[2]: dfs(right)
                if ld < best[2]: dfs(left)

        dfs(0)
        return tuple(best)

    @staticmethod
    def pointcloud_point(cloud, test_point):
        if cloud.point_count() == 0:
            return (Point(0, 0, 0), 0, float('inf'))

        best_point = cloud.get_point(0)
        best_index = 0
        best_dist = best_point.distance(test_point)

        for i in range(1, cloud.point_count()):
            p = cloud.get_point(i)
            dist = p.distance(test_point)
            if dist < best_dist:
                best_dist = dist
                best_point = p
                best_index = i

        return (best_point, best_index, best_dist)

    @staticmethod
    def pointcloud_point_kdtree(cloud, test_point):
        if cloud.point_count() == 0:
            return (Point(0, 0, 0), 0, float('inf'))
        from session_py import KDTree
        pts = [cloud.get_point(i) for i in range(cloud.point_count())]
        tree = KDTree(pts)
        idx, dist = tree.nearest(test_point)
        return (cloud.get_point(idx), idx, dist)

    @staticmethod
    def _build_raw_boxes(aabbs):
        return [(b.cx, b.cy, b.cz, b.hx, b.hy, b.hz) for b in aabbs]

    @staticmethod
    def _build_aabb_nodes(raw_boxes):
        nodes = []

        def build(ids):
            ni = len(nodes)
            nodes.append([None, -1, -1])
            lx = ly = lz = 1e308
            hx = hy = hz = -1e308
            for i in ids:
                b = raw_boxes[i]
                lx = min(lx, b[0]-b[3]); hx = max(hx, b[0]+b[3])
                ly = min(ly, b[1]-b[4]); hy = max(hy, b[1]+b[4])
                lz = min(lz, b[2]-b[5]); hz = max(hz, b[2]+b[5])
            nodes[ni][0] = ((lx+hx)*0.5, (ly+hy)*0.5, (lz+hz)*0.5,
                            (hx-lx)*0.5, (hy-ly)*0.5, (hz-lz)*0.5)
            if len(ids) == 1:
                nodes[ni][2] = ids[0]
                return
            dx = hx-lx; dy = hy-ly; dz = hz-lz
            axis = 0 if dx >= dy and dx >= dz else (1 if dy >= dz else 2)
            ids.sort(key=lambda i: raw_boxes[i][axis])
            mid = len(ids) // 2
            build(ids[:mid])
            nodes[ni][1] = len(nodes)
            build(ids[mid:])

        build(list(range(len(raw_boxes))))
        return nodes

    @staticmethod
    def _query_aabb_nodes(nodes, query, result):
        def overlaps(a, b):
            return (abs(a[0]-b[0]) <= a[3]+b[3] and
                    abs(a[1]-b[1]) <= a[4]+b[4] and
                    abs(a[2]-b[2]) <= a[5]+b[5])

        def dfs(ni):
            aabb, right, obj = nodes[ni]
            if not overlaps(aabb, query):
                return
            if obj >= 0:
                result.append(obj)
                return
            dfs(ni + 1)
            dfs(right)

        dfs(0)

    @staticmethod
    def _aabb_to_aabb_min_dist(a, b):
        dx = max(0.0, abs(a[0]-b[0]) - a[3] - b[3])
        dy = max(0.0, abs(a[1]-b[1]) - a[4] - b[4])
        dz = max(0.0, abs(a[2]-b[2]) - a[5] - b[5])
        return (dx*dx + dy*dy + dz*dz) ** 0.5

    @staticmethod
    def lines_closest(lines, threshold=0.0):
        if len(lines) < 2:
            return []
        from session_py import AABB
        raw = Closest._build_raw_boxes([AABB.from_line(ln, threshold) for ln in lines])
        nodes = Closest._build_aabb_nodes(raw)
        pairs = []
        for i in range(len(lines)):
            candidates = []
            Closest._query_aabb_nodes(nodes, raw[i], candidates)
            for j in candidates:
                if j <= i:
                    continue
                _, _, d_a = Closest.line_point(lines[j], lines[i].start())
                _, _, d_b = Closest.line_point(lines[j], lines[i].end())
                _, _, d_c = Closest.line_point(lines[i], lines[j].start())
                _, _, d_d = Closest.line_point(lines[i], lines[j].end())
                if min(d_a, d_b, d_c, d_d) <= threshold:
                    pairs.append((i, j))
        return pairs

    @staticmethod
    def polylines_closest(polylines, threshold=0.0):
        if len(polylines) < 2:
            return []
        from session_py import AABB
        raw = Closest._build_raw_boxes([AABB.from_polyline(pl, threshold) for pl in polylines])
        nodes = Closest._build_aabb_nodes(raw)
        pairs = []
        for i in range(len(polylines)):
            candidates = []
            Closest._query_aabb_nodes(nodes, raw[i], candidates)
            for j in candidates:
                if j <= i:
                    continue
                pts_a = polylines[i].get_points()
                dist = min(Closest.polyline_point(polylines[j], pt)[2] for pt in pts_a)
                if dist <= threshold:
                    pairs.append((i, j))
        return pairs

    @staticmethod
    def nurbscurves_closest(curves, threshold=0.0):
        if len(curves) < 2:
            return []
        from session_py import AABB
        raw = Closest._build_raw_boxes([AABB.from_nurbscurve(crv, threshold, False) for crv in curves])
        nodes = Closest._build_aabb_nodes(raw)
        pairs = []
        for i in range(len(curves)):
            candidates = []
            Closest._query_aabb_nodes(nodes, raw[i], candidates)
            for j in candidates:
                if j <= i:
                    continue
                t0, t1 = curves[i].domain()
                p_start = curves[i].point_at(t0)
                p_end = curves[i].point_at(t1)
                _, d_a = Closest.curve_point(curves[j], p_start)
                _, d_b = Closest.curve_point(curves[j], p_end)
                if min(d_a, d_b) <= threshold:
                    pairs.append((i, j))
        return pairs

    @staticmethod
    def boxes_closest(boxes, threshold=0.0):
        if len(boxes) < 2:
            return []
        inflated = []
        for b in boxes:
            from session_py import AABB
            inf = AABB(b.cx, b.cy, b.cz, b.hx + threshold, b.hy + threshold, b.hz + threshold)
            inflated.append(inf)
        raw = Closest._build_raw_boxes(inflated)
        nodes = Closest._build_aabb_nodes(raw)
        raw_orig = Closest._build_raw_boxes(boxes)
        pairs = []
        for i in range(len(boxes)):
            candidates = []
            Closest._query_aabb_nodes(nodes, raw[i], candidates)
            for j in candidates:
                if j <= i:
                    continue
                dist = Closest._aabb_to_aabb_min_dist(raw_orig[i], raw_orig[j])
                if dist <= threshold:
                    pairs.append((i, j))
        return pairs
