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

            dt_step = -f / df

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
        points = polyline.points()

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
            du_vec = derivs[1]
            dv_vec = derivs[2]

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
