import numpy as np
import math
import copy
import os

from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .plane import Plane
from .point import Point
from .vector import Vector
from .line import Line
from .polyline import Polyline
from .xform import Xform
from .mesh import Mesh
from .tolerance import Tolerance
from . import knot
from . import intersection


class Primitives:
    """Static factory methods for creating NURBS curve primitives."""

    @staticmethod
    def circle(cx: float, cy: float, cz: float, radius: float) -> NurbsCurve:
        """Create a circle as a rational NURBS curve (9 control points)."""
        w = math.sqrt(2.0) / 2.0

        cx_pat = [1, 1, 0, -1, -1, -1, 0, 1, 1]
        cy_pat = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        weights = [1, w, 1, w, 1, w, 1, w, 1]

        curve = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=9)
        curve.m_knot = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float64)
        curve.m_cv = np.zeros(9 * 4, dtype=np.float64)

        for i in range(9):
            px = cx + radius * cx_pat[i]
            py = cy + radius * cy_pat[i]
            curve.set_cv_4d(i, px * weights[i], py * weights[i], cz * weights[i], weights[i])

        return curve

    @staticmethod
    def ellipse(cx: float, cy: float, cz: float, major_radius: float, minor_radius: float) -> NurbsCurve:
        """Create an ellipse as a rational NURBS curve."""
        w = math.sqrt(2.0) / 2.0
        ex = [1, 1, 0, -1, -1, -1, 0, 1, 1]
        ey = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        weights = [1, w, 1, w, 1, w, 1, w, 1]

        curve = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=9)
        curve.m_knot = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float64)
        curve.m_cv = np.zeros(9 * 4, dtype=np.float64)

        for i in range(9):
            px = cx + major_radius * ex[i]
            py = cy + minor_radius * ey[i]
            curve.set_cv_4d(i, px * weights[i], py * weights[i], cz * weights[i], weights[i])

        return curve

    @staticmethod
    def arc(start: Point, mid: Point, end: Point) -> NurbsCurve:
        """Create an arc through three points as a rational NURBS curve."""
        d1 = [mid[0] - start[0], mid[1] - start[1], mid[2] - start[2]]
        d2 = [end[0] - mid[0], end[1] - mid[1], end[2] - mid[2]]

        m1 = [(start[0] + mid[0]) / 2, (start[1] + mid[1]) / 2, (start[2] + mid[2]) / 2]
        m2 = [(mid[0] + end[0]) / 2, (mid[1] + end[1]) / 2, (mid[2] + end[2]) / 2]

        normal = [d1[1]*d2[2] - d1[2]*d2[1],
                  d1[2]*d2[0] - d1[0]*d2[2],
                  d1[0]*d2[1] - d1[1]*d2[0]]
        normal_len = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

        if normal_len < Tolerance.ZERO_TOLERANCE:
            return NurbsCurve.create(periodic=False, degree=1, points=[start, end])

        # Calculate weight from arc geometry
        chord_mid = Point((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2)
        sagitta = chord_mid.distance(mid)
        chord_len = start.distance(end)

        if sagitta < Tolerance.ZERO_TOLERANCE:
            return NurbsCurve.create(periodic=False, degree=1, points=[start, end])

        # w = cos(theta/2) where theta is the arc angle
        # For a circular arc: sagitta = r(1 - cos(theta/2))
        # Using the relation: w relates to how much the shoulder point is pushed out
        half_chord = chord_len / 2
        r_approx = (half_chord**2 + sagitta**2) / (2 * sagitta) if sagitta > 0 else float('inf')

        if r_approx > 0:
            cos_half = (r_approx - sagitta) / r_approx
            cos_half = max(-1.0, min(1.0, cos_half))
            w = abs(cos_half) if cos_half > 0 else 0.5
        else:
            w = 0.5

        w = max(0.1, min(1.0, w))

        curve = NurbsCurve(dimension=3, is_rational=True, order=3, cv_count=3)
        curve.m_knot = np.array([0, 0, 1, 1], dtype=np.float64)
        curve.m_cv = np.zeros(3 * 4, dtype=np.float64)

        shoulder = Point(
            (start[0] + end[0]) / 2 + (mid[0] - (start[0] + end[0]) / 2) / w,
            (start[1] + end[1]) / 2 + (mid[1] - (start[1] + end[1]) / 2) / w,
            (start[2] + end[2]) / 2 + (mid[2] - (start[2] + end[2]) / 2) / w
        )

        curve.set_cv_4d(0, start[0], start[1], start[2], 1.0)
        curve.set_cv_4d(1, shoulder[0] * w, shoulder[1] * w, shoulder[2] * w, w)
        curve.set_cv_4d(2, end[0], end[1], end[2], 1.0)

        return curve

    @staticmethod
    def parabola(p0: Point, p1: Point, p2: Point) -> NurbsCurve:
        """Create a parabola through 3 points as a non-rational quadratic NURBS."""
        curve = NurbsCurve(dimension=3, is_rational=False, order=3, cv_count=3)
        curve.m_knot = np.array([0, 0, 1, 1], dtype=np.float64)
        curve.m_cv = np.zeros(3 * 3, dtype=np.float64)

        cv1 = Point(
            2 * p1[0] - (p0[0] + p2[0]) / 2,
            2 * p1[1] - (p0[1] + p2[1]) / 2,
            2 * p1[2] - (p0[2] + p2[2]) / 2
        )

        curve.set_cv(0, p0)
        curve.set_cv(1, cv1)
        curve.set_cv(2, p2)

        return curve

    @staticmethod
    def hyperbola(center: Point, a: float, b: float, extent: float) -> NurbsCurve:
        """Create a hyperbola segment as a NURBS curve."""
        num_segments = 8
        cv_count = num_segments + 1

        curve = NurbsCurve(dimension=3, is_rational=False, order=4, cv_count=cv_count)
        curve.m_cv = np.zeros(cv_count * 3, dtype=np.float64)

        for i in range(cv_count):
            t = -extent + 2 * extent * i / num_segments
            x = center[0] + a * math.cosh(t)
            y = center[1] + b * math.sinh(t)
            z = center[2]
            curve.set_cv(i, Point(x, y, z))

        curve.m_knot = knot.make_clamped_uniform(curve.m_order, curve.m_cv_count, 1.0)
        return curve

    @staticmethod
    def spiral(start_radius: float, end_radius: float, pitch: float, turns: float) -> NurbsCurve:
        """Create a spiral (helix with varying radius)."""
        segments_per_turn = 8
        total_segments = max(4, int(turns * segments_per_turn))
        cv_count = total_segments + 1

        curve = NurbsCurve(dimension=3, is_rational=False, order=4, cv_count=cv_count)
        curve.m_cv = np.zeros(cv_count * 3, dtype=np.float64)

        total_angle = turns * 2 * math.pi

        for i in range(cv_count):
            t = i / total_segments
            angle = t * total_angle
            r = start_radius + t * (end_radius - start_radius)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = t * turns * pitch
            curve.set_cv(i, Point(x, y, z))

        curve.m_knot = knot.make_clamped_uniform(curve.m_order, curve.m_cv_count, 1.0)
        return curve

    @staticmethod
    def _unit_cylinder_geometry():
        vertices = [
            Point(0.5, 0.0, -0.5), Point(0.404508, 0.293893, -0.5),
            Point(0.154508, 0.475528, -0.5), Point(-0.154508, 0.475528, -0.5),
            Point(-0.404508, 0.293893, -0.5), Point(-0.5, 0.0, -0.5),
            Point(-0.404508, -0.293893, -0.5), Point(-0.154508, -0.475528, -0.5),
            Point(0.154508, -0.475528, -0.5), Point(0.404508, -0.293893, -0.5),
            Point(0.5, 0.0, 0.5), Point(0.404508, 0.293893, 0.5),
            Point(0.154508, 0.475528, 0.5), Point(-0.154508, 0.475528, 0.5),
            Point(-0.404508, 0.293893, 0.5), Point(-0.5, 0.0, 0.5),
            Point(-0.404508, -0.293893, 0.5), Point(-0.154508, -0.475528, 0.5),
            Point(0.154508, -0.475528, 0.5), Point(0.404508, -0.293893, 0.5),
        ]
        triangles = [
            [0, 1, 11], [0, 11, 10], [1, 2, 12], [1, 12, 11],
            [2, 3, 13], [2, 13, 12], [3, 4, 14], [3, 14, 13],
            [4, 5, 15], [4, 15, 14], [5, 6, 16], [5, 16, 15],
            [6, 7, 17], [6, 17, 16], [7, 8, 18], [7, 18, 17],
            [8, 9, 19], [8, 19, 18], [9, 0, 10], [9, 10, 19],
        ]
        return vertices, triangles

    @staticmethod
    def _unit_cone_geometry():
        vertices = [
            Point(0.0, 0.0, 0.5),
            Point(0.5, 0.0, -0.5), Point(0.353553, -0.353553, -0.5),
            Point(0.0, -0.5, -0.5), Point(-0.353553, -0.353553, -0.5),
            Point(-0.5, 0.0, -0.5), Point(-0.353553, 0.353553, -0.5),
            Point(0.0, 0.5, -0.5), Point(0.353553, 0.353553, -0.5),
        ]
        triangles = [
            [0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 5, 4],
            [0, 6, 5], [0, 7, 6], [0, 8, 7], [0, 1, 8],
        ]
        return vertices, triangles

    @staticmethod
    def _line_to_cylinder_transform(line, radius):
        start = line.start()
        end = line.end()
        line_vec = line.to_vector()
        length = line.length()

        z_axis = line_vec.normalize()
        if abs(z_axis[2]) < 0.9:
            x_axis = Vector(0.0, 0.0, 1.0).cross(z_axis).normalize()
        else:
            x_axis = Vector(1.0, 0.0, 0.0).cross(z_axis).normalize()
        y_axis = z_axis.cross(x_axis).normalize()

        scale = Xform.scale_xyz(radius * 2.0, radius * 2.0, length)
        rotation = Xform()
        rotation.m[0] = x_axis[0]; rotation.m[1] = x_axis[1]; rotation.m[2] = x_axis[2]
        rotation.m[4] = y_axis[0]; rotation.m[5] = y_axis[1]; rotation.m[6] = y_axis[2]
        rotation.m[8] = z_axis[0]; rotation.m[9] = z_axis[1]; rotation.m[10] = z_axis[2]

        center = Point(
            (start.x + end.x) * 0.5, (start.y + end.y) * 0.5, (start.z + end.z) * 0.5
        )
        translation = Xform.translation(center.x, center.y, center.z)
        return translation * rotation * scale

    @staticmethod
    def _transform_geometry(geometry, xform):
        vertices, triangles = geometry
        mesh = Mesh()
        vertex_keys = []
        for v in vertices:
            transformed = xform.transformed_point(v)
            vertex_keys.append(mesh.add_vertex(transformed))
        for tri in triangles:
            face_vertices = [vertex_keys[tri[0]], vertex_keys[tri[1]], vertex_keys[tri[2]]]
            mesh.add_face(face_vertices)
        return mesh

    @staticmethod
    def cylinder_mesh(line, radius):
        unit_cyl = Primitives._unit_cylinder_geometry()
        xform = Primitives._line_to_cylinder_transform(line, radius)
        return Primitives._transform_geometry(unit_cyl, xform)

    @staticmethod
    def arrow_mesh(line, radius):
        start = line.start()
        line_vec = line.to_vector()
        length = line.length()

        z_axis = line_vec.normalize()
        if abs(z_axis[2]) < 0.9:
            x_axis = Vector(0.0, 0.0, 1.0).cross(z_axis).normalize()
        else:
            x_axis = Vector(1.0, 0.0, 0.0).cross(z_axis).normalize()
        y_axis = z_axis.cross(x_axis).normalize()

        cone_length = length * 0.2
        body_length = length * 0.8

        body_center = Point(
            start.x + line_vec[0] * 0.4,
            start.y + line_vec[1] * 0.4,
            start.z + line_vec[2] * 0.4,
        )
        cone_base_center = Point(
            start.x + line_vec[0] * 0.9,
            start.y + line_vec[1] * 0.9,
            start.z + line_vec[2] * 0.9,
        )

        body_scale = Xform.scale_xyz(radius * 2.0, radius * 2.0, body_length)
        rotation = Xform()
        rotation.m[0] = x_axis[0]; rotation.m[1] = x_axis[1]; rotation.m[2] = x_axis[2]
        rotation.m[4] = y_axis[0]; rotation.m[5] = y_axis[1]; rotation.m[6] = y_axis[2]
        rotation.m[8] = z_axis[0]; rotation.m[9] = z_axis[1]; rotation.m[10] = z_axis[2]
        body_translation = Xform.translation(body_center.x, body_center.y, body_center.z)
        body_xform = body_translation * rotation * body_scale

        cone_scale = Xform.scale_xyz(radius * 3.0, radius * 3.0, cone_length)
        cone_translation = Xform.translation(
            cone_base_center.x, cone_base_center.y, cone_base_center.z
        )
        cone_xform = cone_translation * rotation * cone_scale

        body_geometry = Primitives._unit_cylinder_geometry()
        cone_geometry = Primitives._unit_cone_geometry()

        mesh = Mesh()

        body_vertex_map = []
        for v in body_geometry[0]:
            transformed = body_xform.transformed_point(v)
            body_vertex_map.append(mesh.add_vertex(transformed))
        for tri in body_geometry[1]:
            face_vertices = [body_vertex_map[tri[0]], body_vertex_map[tri[1]], body_vertex_map[tri[2]]]
            mesh.add_face(face_vertices)

        cone_vertex_map = []
        for v in cone_geometry[0]:
            transformed = cone_xform.transformed_point(v)
            cone_vertex_map.append(mesh.add_vertex(transformed))
        for tri in cone_geometry[1]:
            face_vertices = [cone_vertex_map[tri[0]], cone_vertex_map[tri[1]], cone_vertex_map[tri[2]]]
            mesh.add_face(face_vertices)

        return mesh

    ###########################################################################
    # Surface Factory Methods
    ###########################################################################

    @staticmethod
    def _merge_knot_vectors(a, b, tol=1e-10):
        merged = []
        i, j = 0, 0
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

    @staticmethod
    def _knot_vectors_equal(a, b, tol=1e-10):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if abs(a[i] - b[i]) > tol:
                return False
        return True

    @staticmethod
    def _make_curves_compatible(curves):
        if len(curves) < 2:
            return
        max_deg = max(c.degree() for c in curves)
        for c in curves:
            if c.degree() < max_deg:
                c.increase_degree(max_deg)
        any_rational = any(c.is_rational() for c in curves)
        if any_rational:
            for c in curves:
                c.make_rational()
        already_compatible = True
        for i in range(1, len(curves)):
            if curves[i].cv_count() != curves[0].cv_count():
                already_compatible = False
                break
            if not Primitives._knot_vectors_equal(list(curves[i].get_knots()), list(curves[0].get_knots())):
                already_compatible = False
                break
        if already_compatible:
            return
        for c in curves:
            c.set_domain(0.0, 1.0)
        unified = list(curves[0].get_knots())
        for i in range(1, len(curves)):
            unified = Primitives._merge_knot_vectors(unified, list(curves[i].get_knots()))
        tol = 1e-10
        for c in curves:
            cur_knots = list(c.get_knots())
            ci = 0
            for ui in range(len(unified)):
                if ci < len(cur_knots) and abs(cur_knots[ci] - unified[ui]) < tol:
                    ci += 1
                else:
                    c.insert_knot(unified[ui], 1)

    @staticmethod
    def cylinder_surface(cx, cy, cz, radius, height):
        w = math.sqrt(2.0) / 2.0
        circle_weights = [1, w, 1, w, 1, w, 1, w, 1]
        circle_x = [1, 1, 0, -1, -1, -1, 0, 1, 1]
        circle_y = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        u_knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        v_knots = [0, 1]

        srf = NurbsSurface.create_raw(3, True, 3, 2, 9, 2)
        for i in range(10):
            srf.set_knot(0, i, u_knots[i])
        for i in range(2):
            srf.set_knot(1, i, v_knots[i])

        for i in range(9):
            wi = circle_weights[i]
            px = cx + radius * circle_x[i]
            py = cy + radius * circle_y[i]
            srf.set_cv_4d(i, 0, px * wi, py * wi, cz * wi, wi)
            srf.set_cv_4d(i, 1, px * wi, py * wi, (cz + height) * wi, wi)

        return srf

    @staticmethod
    def cone_surface(cx, cy, cz, radius, height):
        w = math.sqrt(2.0) / 2.0
        circle_weights = [1, w, 1, w, 1, w, 1, w, 1]
        circle_x = [1, 1, 0, -1, -1, -1, 0, 1, 1]
        circle_y = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        u_knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        v_knots = [0, 1]

        srf = NurbsSurface.create_raw(3, True, 3, 2, 9, 2)
        for i in range(10):
            srf.set_knot(0, i, u_knots[i])
        for i in range(2):
            srf.set_knot(1, i, v_knots[i])

        apex_z = cz + height
        for i in range(9):
            wi = circle_weights[i]
            px = cx + radius * circle_x[i]
            py = cy + radius * circle_y[i]
            srf.set_cv_4d(i, 0, px * wi, py * wi, cz * wi, wi)
            srf.set_cv_4d(i, 1, cx * wi, cy * wi, apex_z * wi, wi)

        return srf

    @staticmethod
    def torus_surface(cx, cy, cz, major_radius, minor_radius):
        w = math.sqrt(2.0) / 2.0
        cw = [1, w, 1, w, 1, w, 1, w, 1]
        cos_a = [1, 1, 0, -1, -1, -1, 0, 1, 1]
        sin_a = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        u_knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

        srf = NurbsSurface.create_raw(3, True, 3, 3, 9, 9)
        for d in range(2):
            for i in range(10):
                srf.set_knot(d, i, u_knots[i])

        for i in range(9):
            ca = cos_a[i]
            sa = sin_a[i]
            for j in range(9):
                cb = cos_a[j]
                sb = sin_a[j]
                r = major_radius + minor_radius * cb
                px = cx + r * ca
                py = cy + r * sa
                pz = cz + minor_radius * sb
                wij = cw[i] * cw[j]
                srf.set_cv_4d(i, j, px * wij, py * wij, pz * wij, wij)

        return srf

    @staticmethod
    def sphere_surface(cx, cy, cz, radius):
        w = math.sqrt(2.0) / 2.0
        cw = [1, w, 1, w, 1, w, 1, w, 1]
        cos_a = [1, 1, 0, -1, -1, -1, 0, 1, 1]
        sin_a = [0, 1, 1, 1, 0, -1, -1, -1, 0]
        u_knots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        v_knots = [0, 0, 1, 1, 2, 2]
        lat_r = [0, 1, 1, 1, 0]
        lat_z = [-1, -1, 0, 1, 1]
        lat_w = [1, w, 1, w, 1]

        srf = NurbsSurface.create_raw(3, True, 3, 3, 9, 5)
        for i in range(10):
            srf.set_knot(0, i, u_knots[i])
        for i in range(6):
            srf.set_knot(1, i, v_knots[i])

        for j in range(5):
            r = radius * lat_r[j]
            pz = cz + radius * lat_z[j]
            wj = lat_w[j]
            for i in range(9):
                px = cx + r * cos_a[i]
                py = cy + r * sin_a[i]
                wij = cw[i] * wj
                srf.set_cv_4d(i, j, px * wij, py * wij, pz * wij, wij)

        return srf

    @staticmethod
    def quad_sphere(cx, cy, cz, radius):
        R = radius
        a = R / math.sqrt(3.0)
        e = R * math.sqrt(3.0) / 2.0
        wk = math.sqrt(2.0 / 3.0)
        wc = (-72.0 - 32.0*math.sqrt(6.0) + 48.0*math.sqrt(3.0) + 56.0*math.sqrt(2.0)) \
           / (48.0*(1.0 + math.sqrt(2.0/3.0) - 1.0/math.sqrt(3.0) - 1.0/math.sqrt(2.0)))
        K = R * (1.0 - 1.0/math.sqrt(3.0) + 2.0*math.sqrt(2.0/3.0) - math.sqrt(2.0))
        h = R + K / wc

        zf = [
            [(-a,-a, a, 1), (-e, 0, e, wk), (-a, a, a, 1)],
            [( 0,-e, e, wk),( 0, 0, h, wc), ( 0, e, e, wk)],
            [( a,-a, a, 1), ( e, 0, e, wk), ( a, a, a, 1)]
        ]

        rot = [
            [[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]],
            [[ 1, 0, 0],[ 0,-1, 0],[ 0, 0,-1]],
            [[ 0, 0, 1],[ 0, 1, 0],[-1, 0, 0]],
            [[ 0, 0,-1],[ 0, 1, 0],[ 1, 0, 0]],
            [[ 1, 0, 0],[ 0, 0, 1],[ 0,-1, 0]],
            [[ 1, 0, 0],[ 0, 0,-1],[ 0, 1, 0]]
        ]

        faces = []
        for f in range(6):
            srf = NurbsSurface.create_raw(3, True, 3, 3, 3, 3)
            for i in range(3):
                for j in range(3):
                    p = zf[i][j]
                    rx = rot[f][0][0]*p[0] + rot[f][0][1]*p[1] + rot[f][0][2]*p[2] + cx
                    ry = rot[f][1][0]*p[0] + rot[f][1][1]*p[1] + rot[f][1][2]*p[2] + cy
                    rz = rot[f][2][0]*p[0] + rot[f][2][1]*p[1] + rot[f][2][2]*p[2] + cz
                    srf.set_cv_4d(i, j, rx*p[3], ry*p[3], rz*p[3], p[3])
            faces.append(srf)
        return faces

    @staticmethod
    def create_ruled(curveA, curveB):
        if not curveA.is_valid() or not curveB.is_valid():
            return NurbsSurface()

        cA = curveA.duplicate()
        cB = curveB.duplicate()

        cA.set_domain(0.0, 1.0)
        cB.set_domain(0.0, 1.0)

        if cA.degree() < cB.degree():
            cA.increase_degree(cB.degree())
        elif cB.degree() < cA.degree():
            cB.increase_degree(cA.degree())

        if cA.is_rational() or cB.is_rational():
            cA.make_rational()
            cB.make_rational()

        knots_a = list(cA.get_knots())
        knots_b = list(cB.get_knots())
        tol = 1e-10

        for k in knots_b:
            found = any(abs(ka - k) < tol for ka in knots_a)
            if not found:
                cA.insert_knot(k, 1)

        knots_a = list(cA.get_knots())
        for k in knots_a:
            found = any(abs(kb - k) < tol for kb in knots_b)
            if not found:
                cB.insert_knot(k, 1)

        order_u = cA.order()
        cv_count_u = cA.cv_count()
        is_rat = cA.is_rational()

        surface = NurbsSurface.create_raw(3, is_rat, order_u, 2, cv_count_u, 2)
        if surface is None:
            return NurbsSurface()

        for i in range(cA.knot_count()):
            surface.set_knot(0, i, cA.knot(i))

        surface.set_knot(1, 0, 0.0)
        surface.set_knot(1, 1, 1.0)

        if is_rat:
            for i in range(cv_count_u):
                ax, ay, az, aw = cA.get_cv_4d(i)
                surface.set_cv_4d(i, 0, ax, ay, az, aw)
                bx, by, bz, bw = cB.get_cv_4d(i)
                surface.set_cv_4d(i, 1, bx, by, bz, bw)
        else:
            for i in range(cv_count_u):
                surface.set_cv(i, 0, cA.get_cv(i))
                surface.set_cv(i, 1, cB.get_cv(i))

        return surface

    @staticmethod
    def create_extrusion(curve, direction):
        if not curve.is_valid():
            return NurbsSurface()
        translated = curve.duplicate()
        t = Xform.translation(direction[0], direction[1], direction[2])
        translated.transform(t)
        return Primitives.create_ruled(curve, translated)

    @staticmethod
    def create_planar(boundary):
        if not boundary.is_valid():
            return NurbsSurface()

        all_pts = []
        for i in range(boundary.cv_count()):
            pt = boundary.get_cv(i)
            if pt is not None:
                all_pts.append(pt)

        unique_pts = list(all_pts)
        if len(unique_pts) >= 2:
            f = unique_pts[0]
            l = unique_pts[-1]
            d2 = (f[0]-l[0])**2 + (f[1]-l[1])**2 + (f[2]-l[2])**2
            if d2 < 1e-20:
                unique_pts.pop()
        if len(unique_pts) < 3:
            return NurbsSurface()

        def make_bilinear(orig, xax, yax, min_u, max_u, min_v, max_v):
            srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2)
            srf.set_knot(0, 0, 0.0)
            srf.set_knot(0, 1, 1.0)
            srf.set_knot(1, 0, 0.0)
            srf.set_knot(1, 1, 1.0)
            def pt(u, v):
                return Point(orig[0] + u*xax[0] + v*yax[0],
                             orig[1] + u*xax[1] + v*yax[1],
                             orig[2] + u*xax[2] + v*yax[2])
            srf.set_cv(0, 0, pt(min_u, min_v))
            srf.set_cv(1, 0, pt(max_u, min_v))
            srf.set_cv(1, 1, pt(max_u, max_v))
            srf.set_cv(0, 1, pt(min_u, max_v))
            return srf

        def longest_edge_dir(pts):
            best_d2 = 0.0
            best_i = 0
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                dx = pts[j][0]-pts[i][0]
                dy = pts[j][1]-pts[i][1]
                dz = pts[j][2]-pts[i][2]
                d2 = dx*dx + dy*dy + dz*dz
                if d2 > best_d2:
                    best_d2 = d2
                    best_i = i
            j = (best_i + 1) % len(pts)
            dx = pts[j][0]-pts[best_i][0]
            dy = pts[j][1]-pts[best_i][1]
            dz = pts[j][2]-pts[best_i][2]
            length = math.sqrt(dx*dx + dy*dy + dz*dz)
            return Vector(dx/length, dy/length, dz/length)

        if len(unique_pts) == 3 and boundary.degree() <= 1:
            srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2)
            srf.set_knot(0, 0, 0.0)
            srf.set_knot(0, 1, 1.0)
            srf.set_knot(1, 0, 0.0)
            srf.set_knot(1, 1, 1.0)
            srf.set_cv(0, 0, unique_pts[0])
            srf.set_cv(1, 0, unique_pts[1])
            srf.set_cv(1, 1, unique_pts[2])
            srf.set_cv(0, 1, unique_pts[0])
            return srf

        if len(unique_pts) == 4 and boundary.degree() <= 1:
            srf = NurbsSurface.create_raw(3, False, 2, 2, 2, 2)
            srf.set_knot(0, 0, 0.0)
            srf.set_knot(0, 1, 1.0)
            srf.set_knot(1, 0, 0.0)
            srf.set_knot(1, 1, 1.0)
            srf.set_cv(0, 0, unique_pts[0])
            srf.set_cv(1, 0, unique_pts[1])
            srf.set_cv(1, 1, unique_pts[2])
            srf.set_cv(0, 1, unique_pts[3])
            return srf

        if boundary.degree() <= 1:
            e1 = Vector(unique_pts[1][0]-unique_pts[0][0], unique_pts[1][1]-unique_pts[0][1], unique_pts[1][2]-unique_pts[0][2])
            e2 = Vector(unique_pts[2][0]-unique_pts[0][0], unique_pts[2][1]-unique_pts[0][1], unique_pts[2][2]-unique_pts[0][2])
            normal = e1.cross(e2)
            nlen = normal.magnitude()
            if nlen < 1e-14:
                return NurbsSurface()
            normal = normal * (1.0 / nlen)

            xax = longest_edge_dir(unique_pts)
            yax = normal.cross(xax)
            ylen = yax.magnitude()
            if ylen < 1e-14:
                return NurbsSurface()
            yax = yax * (1.0 / ylen)

            orig = unique_pts[0]
            min_u, max_u, min_v, max_v = 0.0, 0.0, 0.0, 0.0
            for pt in unique_pts:
                dx = pt[0]-orig[0]
                dy = pt[1]-orig[1]
                dz = pt[2]-orig[2]
                u = dx*xax[0] + dy*xax[1] + dz*xax[2]
                v = dx*yax[0] + dy*yax[1] + dz*yax[2]
                if u < min_u: min_u = u
                if u > max_u: max_u = u
                if v < min_v: min_v = v
                if v > max_v: max_v = v

            pad = max(max_u - min_u, max_v - min_v) * 0.05
            if pad < 1e-6:
                pad = 1.0
            min_u -= pad
            max_u += pad
            min_v -= pad
            max_v += pad
            return make_bilinear(orig, xax, yax, min_u, max_u, min_v, max_v)

        n_samples = max(20, boundary.cv_count() * 4)
        sample_pts, _sample_params = boundary.divide_by_count(n_samples, True)
        plane = Plane.from_points_pca(sample_pts)
        if plane.z_axis.magnitude() < 1e-10:
            return NurbsSurface()

        xax = plane.x_axis
        yax = plane.y_axis
        orig = plane.origin

        min_u, max_u = 1e30, -1e30
        min_v, max_v = 1e30, -1e30
        for pt in sample_pts:
            dx = pt[0]-orig[0]
            dy = pt[1]-orig[1]
            dz = pt[2]-orig[2]
            u = dx*xax[0] + dy*xax[1] + dz*xax[2]
            v = dx*yax[0] + dy*yax[1] + dz*yax[2]
            if u < min_u: min_u = u
            if u > max_u: max_u = u
            if v < min_v: min_v = v
            if v > max_v: max_v = v

        pad = max(max_u - min_u, max_v - min_v) * 0.05
        if pad < 1e-6:
            pad = 1.0
        min_u -= pad
        max_u += pad
        min_v -= pad
        max_v += pad
        return make_bilinear(orig, xax, yax, min_u, max_u, min_v, max_v)

    @staticmethod
    def create_loft(input_curves, degree_v=3):
        if len(input_curves) < 2:
            return NurbsSurface()
        for c in input_curves:
            if not c.is_valid():
                return NurbsSurface()

        curves = [c.duplicate() for c in input_curves]
        Primitives._make_curves_compatible(curves)
        Primitives._make_curves_compatible(curves)

        n_sections = len(curves)
        cv_count_u = curves[0].cv_count()
        order_u = curves[0].order()
        is_rat = curves[0].is_rational()

        if degree_v >= n_sections:
            degree_v = n_sections - 1
        if degree_v < 1:
            degree_v = 1
        order_v = degree_v + 1

        v_params = [0.0] * n_sections
        for k in range(1, n_sections):
            pk_prev = curves[k - 1].point_at_middle()
            pk_curr = curves[k].point_at_middle()
            dx = pk_curr[0] - pk_prev[0]
            dy = pk_curr[1] - pk_prev[1]
            dz = pk_curr[2] - pk_prev[2]
            v_params[k] = v_params[k - 1] + math.sqrt(dx * dx + dy * dy + dz * dz)

        total_len = v_params[-1]
        if total_len > 1e-14:
            for k in range(n_sections):
                v_params[k] /= total_len
        else:
            for k in range(n_sections):
                v_params[k] = float(k) / (n_sections - 1)

        cv_count_v = n_sections
        knot_count_v = order_v + cv_count_v - 2
        knots_v = [0.0] * knot_count_v

        if degree_v >= n_sections - 1:
            d = degree_v
            for i in range(d):
                knots_v[i] = 0.0
            for i in range(d, knot_count_v):
                knots_v[i] = 1.0
        else:
            for i in range(order_v - 1):
                knots_v[i] = v_params[0]
            for j in range(1, n_sections - order_v + 1):
                s = 0.0
                for i in range(j, j + degree_v):
                    s += v_params[i]
                knots_v[order_v - 2 + j] = s / degree_v
            for i in range(knot_count_v - order_v + 1, knot_count_v):
                knots_v[i] = v_params[n_sections - 1]

        surface = NurbsSurface.create_raw(3, is_rat, order_u, order_v, cv_count_u, cv_count_v)
        if surface is None:
            return NurbsSurface()

        for i in range(surface.knot_count(0)):
            surface.set_knot(0, i, curves[0].knot(i))
        for i in range(len(knots_v)):
            if i < surface.knot_count(1):
                surface.set_knot(1, i, knots_v[i])

        n = n_sections
        N_matrix = [[0.0] * n for _ in range(n)]
        knots_v_arr = np.array(knots_v)

        for k in range(n):
            t = v_params[k]
            t0 = knots_v[order_v - 2]
            t1 = knots_v[knot_count_v - order_v + 1]
            if t < t0:
                t = t0
            if t > t1:
                t = t1

            span = knot.find_span(order_v, cv_count_v, knots_v_arr, t)
            d = order_v - 1
            knot_base = span + d

            if knots_v[knot_base - 1] == knots_v[knot_base]:
                if t <= knots_v[knot_base]:
                    N_matrix[k][span] = 1.0
                else:
                    N_matrix[k][span + order_v - 1] = 1.0
                continue

            Nvals = [0.0] * (order_v * order_v)
            Nvals[order_v * order_v - 1] = 1.0
            left = [0.0] * d
            right = [0.0] * d
            N_idx = order_v * order_v - 1
            k_right = knot_base
            k_left = knot_base - 1

            for j in range(d):
                N0_idx = N_idx
                N_idx -= (order_v + 1)
                left[j] = t - knots_v[k_left]
                right[j] = knots_v[k_right] - t
                k_left -= 1
                k_right += 1

                x = 0.0
                for r in range(j + 1):
                    a0 = left[j - r]
                    a1 = right[r]
                    denom = a0 + a1
                    y = Nvals[N0_idx + r] / denom if denom != 0.0 else 0.0
                    Nvals[N_idx + r] = x + a1 * y
                    x = a0 * y
                Nvals[N_idx + j + 1] = x

            for j in range(order_v):
                col = span + j
                if 0 <= col < n:
                    N_matrix[k][col] = Nvals[j]

        dim = 4 if is_rat else 3
        for i in range(cv_count_u):
            rhs = [[0.0] * dim for _ in range(n)]
            for k in range(n):
                if is_rat:
                    cx_v, cy_v, cz_v, cw_v = curves[k].get_cv_4d(i)
                    rhs[k] = [cx_v, cy_v, cz_v, cw_v]
                else:
                    p = curves[k].get_cv(i)
                    rhs[k] = [p[0], p[1], p[2]]

            A = [row[:] for row in N_matrix]
            b = [row[:] for row in rhs]

            for col in range(n):
                max_row = col
                max_val = abs(A[col][col])
                for row in range(col + 1, n):
                    if abs(A[row][col]) > max_val:
                        max_val = abs(A[row][col])
                        max_row = row
                if max_val < 1e-14:
                    continue
                A[col], A[max_row] = A[max_row], A[col]
                b[col], b[max_row] = b[max_row], b[col]
                for row in range(col + 1, n):
                    factor = A[row][col] / A[col][col]
                    for c in range(col, n):
                        A[row][c] -= factor * A[col][c]
                    for d2 in range(dim):
                        b[row][d2] -= factor * b[col][d2]

            Q = [[0.0] * dim for _ in range(n)]
            for row in range(n - 1, -1, -1):
                for d2 in range(dim):
                    Q[row][d2] = b[row][d2]
                    for c in range(row + 1, n):
                        Q[row][d2] -= A[row][c] * Q[c][d2]
                    if abs(A[row][row]) > 1e-14:
                        Q[row][d2] /= A[row][row]

            for j in range(n):
                if is_rat:
                    surface.set_cv_4d(i, j, Q[j][0], Q[j][1], Q[j][2], Q[j][3])
                else:
                    surface.set_cv(i, j, Point(Q[j][0], Q[j][1], Q[j][2]))

        return surface

    @staticmethod
    def create_revolve(profile, axis_origin, axis_direction, angle):
        if not profile.is_valid():
            return NurbsSurface()
        ax_len = axis_direction.magnitude()
        if ax_len < 1e-14:
            return NurbsSurface()
        axis_dir = axis_direction * (1.0 / ax_len)

        angle = abs(angle)
        PI = Tolerance.PI
        if angle > 2.0 * PI:
            angle = 2.0 * PI
        if angle < 1e-14:
            return NurbsSurface()

        if angle <= PI / 2.0 + 1e-10:
            n_arcs = 1
        elif angle <= PI + 1e-10:
            n_arcs = 2
        elif angle <= 3.0 * PI / 2.0 + 1e-10:
            n_arcs = 3
        else:
            n_arcs = 4

        d_theta = angle / n_arcs
        w_mid = math.cos(d_theta / 2.0)
        n_u = 2 * n_arcs + 1

        knot_count_u = n_u + 1
        knots_u = [0.0] * knot_count_u
        knots_u[0] = 0.0
        knots_u[1] = 0.0
        for i in range(1, n_arcs + 1):
            kv = i * d_theta
            knots_u[2 * i] = kv
            knots_u[2 * i + 1] = kv
        knots_u[knot_count_u - 1] = angle
        knots_u[knot_count_u - 2] = angle

        cv_count_v = profile.cv_count()
        order_v = profile.order()
        profile_rational = profile.is_rational()

        surface = NurbsSurface.create_raw(3, True, 3, order_v, n_u, cv_count_v)
        if surface is None:
            return NurbsSurface()

        for i in range(min(knot_count_u, surface.knot_count(0))):
            surface.set_knot(0, i, knots_u[i])
        for i in range(min(profile.knot_count(), surface.knot_count(1))):
            surface.set_knot(1, i, profile.knot(i))

        u_angles = [0.0] * n_u
        u_weights = [0.0] * n_u
        for i in range(n_u):
            if i % 2 == 0:
                u_angles[i] = (i // 2) * d_theta
                u_weights[i] = 1.0
            else:
                u_angles[i] = (i // 2) * d_theta + d_theta / 2.0
                u_weights[i] = w_mid

        for j in range(cv_count_v):
            p_j = profile.get_cv(j)
            if p_j is None:
                p_j = Point(0.0, 0.0, 0.0)
            profile_w = profile.weight(j) if profile_rational else 1.0

            dx = p_j[0] - axis_origin[0]
            dy = p_j[1] - axis_origin[1]
            dz = p_j[2] - axis_origin[2]
            proj = dx * axis_dir[0] + dy * axis_dir[1] + dz * axis_dir[2]
            o_j = Point(
                axis_origin[0] + proj * axis_dir[0],
                axis_origin[1] + proj * axis_dir[1],
                axis_origin[2] + proj * axis_dir[2]
            )

            rx = p_j[0] - o_j[0]
            ry = p_j[1] - o_j[1]
            rz = p_j[2] - o_j[2]
            r_j = math.sqrt(rx * rx + ry * ry + rz * rz)

            if r_j < 1e-14:
                for i in range(n_u):
                    combined_w = u_weights[i] * profile_w
                    surface.set_cv(i, j, o_j)
                    surface.set_weight(i, j, combined_w)
            else:
                x_local = Vector(rx / r_j, ry / r_j, rz / r_j)
                y_local = axis_dir.cross(x_local)
                y_len = y_local.magnitude()
                if y_len > 1e-14:
                    y_local = y_local * (1.0 / y_len)

                for i in range(n_u):
                    theta = u_angles[i]
                    cos_t = math.cos(theta)
                    sin_t = math.sin(theta)

                    effective_r = r_j / w_mid if i % 2 == 1 else r_j

                    px = o_j[0] + effective_r * (cos_t * x_local[0] + sin_t * y_local[0])
                    py = o_j[1] + effective_r * (cos_t * x_local[1] + sin_t * y_local[1])
                    pz = o_j[2] + effective_r * (cos_t * x_local[2] + sin_t * y_local[2])

                    combined_w = u_weights[i] * profile_w
                    surface.set_cv_4d(i, j, px * combined_w, py * combined_w, pz * combined_w, combined_w)

        return surface

    @staticmethod
    def create_revolve_full(profile, axis_origin, axis_direction):
        return Primitives.create_revolve(profile, axis_origin, axis_direction, 2.0 * Tolerance.PI)

    @staticmethod
    def create_sweep1(rail, profile):
        if not rail.is_valid() or not profile.is_valid():
            return NurbsSurface()

        working_profile = profile.duplicate()

        n = min(max(rail.span_count() * 2 + 1, 5), 20)
        frames = rail.get_perpendicular_planes(n)
        if not frames:
            return NurbsSurface()

        nc = working_profile.cv_count()
        cx, cy, cz = 0.0, 0.0, 0.0
        for k in range(nc):
            cv = working_profile.get_cv(k)
            if cv is not None:
                cx += cv[0]
                cy += cv[1]
                cz += cv[2]
        cx /= nc
        cy /= nc
        cz /= nc

        t0, t1 = working_profile.domain()
        pa = working_profile.point_at(t0)
        pb = working_profile.point_at(t0 + (t1 - t0) / 3.0)
        pc = working_profile.point_at(t0 + 2.0 * (t1 - t0) / 3.0)
        v1 = Vector(pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        v2 = Vector(pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
        prof_normal = v1.cross(v2)
        nlen = prof_normal.magnitude()
        if nlen < 1e-14:
            prof_normal = Vector(1.0, 0.0, 0.0)
        else:
            prof_normal = prof_normal * (1.0 / nlen)

        prof_x = Vector(pa[0] - cx, pa[1] - cy, pa[2] - cz)
        pxlen = prof_x.magnitude()
        if pxlen < 1e-14:
            prof_x = Vector(0.0, 1.0, 0.0)
        else:
            prof_x = prof_x * (1.0 / pxlen)
        dot = prof_x[0] * prof_normal[0] + prof_x[1] * prof_normal[1] + prof_x[2] * prof_normal[2]
        prof_x = Vector(prof_x[0] - dot * prof_normal[0], prof_x[1] - dot * prof_normal[1], prof_x[2] - dot * prof_normal[2])
        pxlen = prof_x.magnitude()
        if pxlen < 1e-14:
            prof_x = Vector(0.0, 1.0, 0.0)
        else:
            prof_x = prof_x * (1.0 / pxlen)
        prof_y = prof_normal.cross(prof_x)
        pylen = prof_y.magnitude()
        if pylen > 1e-14:
            prof_y = prof_y * (1.0 / pylen)

        positioned_profiles = []
        for i in range(len(frames)):
            prof_copy = working_profile.duplicate()
            fo = frames[i].origin
            fx = frames[i].x_axis
            fy = frames[i].y_axis
            fz = frames[i].z_axis

            t1x = Xform.translation(-cx, -cy, -cz)

            rot = Xform.identity()
            rot.m[0]  = fx[0]*prof_x[0] + fy[0]*prof_y[0] + fz[0]*prof_normal[0]
            rot.m[1]  = fx[1]*prof_x[0] + fy[1]*prof_y[0] + fz[1]*prof_normal[0]
            rot.m[2]  = fx[2]*prof_x[0] + fy[2]*prof_y[0] + fz[2]*prof_normal[0]
            rot.m[4]  = fx[0]*prof_x[1] + fy[0]*prof_y[1] + fz[0]*prof_normal[1]
            rot.m[5]  = fx[1]*prof_x[1] + fy[1]*prof_y[1] + fz[1]*prof_normal[1]
            rot.m[6]  = fx[2]*prof_x[1] + fy[2]*prof_y[1] + fz[2]*prof_normal[1]
            rot.m[8]  = fx[0]*prof_x[2] + fy[0]*prof_y[2] + fz[0]*prof_normal[2]
            rot.m[9]  = fx[1]*prof_x[2] + fy[1]*prof_y[2] + fz[1]*prof_normal[2]
            rot.m[10] = fx[2]*prof_x[2] + fy[2]*prof_y[2] + fz[2]*prof_normal[2]
            rot.m[12] = fo[0]
            rot.m[13] = fo[1]
            rot.m[14] = fo[2]

            prof_copy.transform(t1x)
            prof_copy.transform(rot)
            positioned_profiles.append(prof_copy)

        loft_degree = min(3, len(positioned_profiles) - 1)
        return Primitives.create_loft(positioned_profiles, loft_degree)

    @staticmethod
    def create_sweep2(rail1, rail2, shapes):
        if not rail1.is_valid() or not rail2.is_valid() or not shapes:
            return NurbsSurface()
        for s in shapes:
            if not s.is_valid():
                return NurbsSurface()

        compat_shapes = [s.duplicate() for s in shapes]
        if len(compat_shapes) >= 2:
            Primitives._make_curves_compatible(compat_shapes)

        n_shapes = len(compat_shapes)
        shape_params = [0.0 if n_shapes == 1 else float(k) / (n_shapes - 1) for k in range(n_shapes)]

        n = min(max(max(rail1.span_count(), rail2.span_count()) * 2 + 1, 5), 20)

        pts1, _params1 = rail1.divide_by_count(n + 1, True)
        pts2, _params2 = rail2.divide_by_count(n + 1, True)

        frames1 = rail1.get_perpendicular_planes(n)
        if not frames1:
            return NurbsSurface()

        class ShapeInfo:
            pass

        sinfo = []
        for k in range(n_shapes):
            si = ShapeInfo()
            si.start = compat_shapes[k].point_at_start()
            si.end = compat_shapes[k].point_at_end()
            span = Vector(si.end[0]-si.start[0], si.end[1]-si.start[1], si.end[2]-si.start[2])
            si.width = span.magnitude()
            if si.width < 1e-14:
                si.width = 1.0
            si.dir = span * (1.0 / si.width)
            up_try = Vector(0.0, 0.0, 1.0)
            si.side = si.dir.cross(up_try)
            if si.side.magnitude() < 1e-10:
                up_try = Vector(0.0, 1.0, 0.0)
                si.side = si.dir.cross(up_try)
            si.side = si.side * (1.0 / si.side.magnitude())
            si.up = si.side.cross(si.dir)
            ulen = si.up.magnitude()
            if ulen > 1e-14:
                si.up = si.up * (1.0 / ulen)
            sinfo.append(si)

        positioned_profiles = []
        for i in range(min(len(frames1), len(pts1), len(pts2))):
            t = 0.0 if len(frames1) <= 1 else float(i) / (len(frames1) - 1)

            j = 0
            s = 0.0
            if n_shapes == 1:
                j = 0
                s = 0.0
            else:
                for k in range(n_shapes - 1):
                    if t <= shape_params[k + 1] + 1e-14:
                        j = k
                        break
                    j = k
                denom = shape_params[j + 1] - shape_params[j]
                s = (t - shape_params[j]) / denom if denom > 1e-14 else 0.0
                s = max(0.0, min(1.0, s))

            interp_shape = compat_shapes[j].duplicate()
            if n_shapes > 1 and j + 1 < n_shapes:
                nc = compat_shapes[j].cv_count()
                for c in range(nc):
                    cv0 = compat_shapes[j].get_cv(c)
                    cv1 = compat_shapes[j + 1].get_cv(c)
                    if cv0 is None:
                        cv0 = Point(0.0, 0.0, 0.0)
                    if cv1 is None:
                        cv1 = Point(0.0, 0.0, 0.0)
                    lerped = Point(cv0[0]*(1-s) + cv1[0]*s, cv0[1]*(1-s) + cv1[1]*s, cv0[2]*(1-s) + cv1[2]*s)
                    interp_shape.set_cv(c, lerped)

            if n_shapes == 1:
                shape_width = sinfo[0].width
            else:
                shape_width = sinfo[j].width * (1 - s) + (sinfo[j+1].width * s if j + 1 < n_shapes else 0.0)

            def lerp_vec(a, b):
                return Vector(a[0]*(1-s)+b[0]*s, a[1]*(1-s)+b[1]*s, a[2]*(1-s)+b[2]*s)

            if n_shapes > 1 and j + 1 < n_shapes:
                prof_dir = lerp_vec(sinfo[j].dir, sinfo[j+1].dir)
                prof_side = lerp_vec(sinfo[j].side, sinfo[j+1].side)
                prof_up = lerp_vec(sinfo[j].up, sinfo[j+1].up)
            else:
                prof_dir = Vector(sinfo[j].dir[0], sinfo[j].dir[1], sinfo[j].dir[2])
                prof_side = Vector(sinfo[j].side[0], sinfo[j].side[1], sinfo[j].side[2])
                prof_up = Vector(sinfo[j].up[0], sinfo[j].up[1], sinfo[j].up[2])

            pdlen = prof_dir.magnitude()
            if pdlen > 1e-14:
                prof_dir = prof_dir * (1.0 / pdlen)
            pslen = prof_side.magnitude()
            if pslen > 1e-14:
                prof_side = prof_side * (1.0 / pslen)
            pulen = prof_up.magnitude()
            if pulen > 1e-14:
                prof_up = prof_up * (1.0 / pulen)

            if n_shapes == 1:
                interp_start = sinfo[0].start
            elif j + 1 < n_shapes:
                interp_start = Point(
                    sinfo[j].start[0]*(1-s) + sinfo[j+1].start[0]*s,
                    sinfo[j].start[1]*(1-s) + sinfo[j+1].start[1]*s,
                    sinfo[j].start[2]*(1-s) + sinfo[j+1].start[2]*s)
            else:
                interp_start = sinfo[j].start

            p1 = pts1[i]
            p2 = pts2[i]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]
            rail_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            scale_factor = rail_dist / shape_width if rail_dist > 1e-14 and shape_width > 1e-14 else 1.0

            prof_copy = interp_shape.duplicate()
            t1_xf = Xform.translation(-interp_start[0], -interp_start[1], -interp_start[2])
            prof_copy.transform(t1_xf)

            sc = Xform.scale_xyz(scale_factor, scale_factor, scale_factor)
            prof_copy.transform(sc)

            tangent_orig = frames1[i].z_axis
            x_dir = Vector(dx, dy, dz)
            x_len = x_dir.magnitude()
            if x_len > 1e-14:
                x_dir = x_dir * (1.0 / x_len)
            else:
                x_dir = frames1[i].x_axis
            y_dir = tangent_orig.cross(x_dir)
            y_len = y_dir.magnitude()
            if y_len > 1e-14:
                y_dir = y_dir * (1.0 / y_len)
            else:
                y_dir = frames1[i].y_axis
            dot_up = y_dir[0]*prof_up[0] + y_dir[1]*prof_up[1] + y_dir[2]*prof_up[2]
            if dot_up < 0:
                y_dir = Vector(-y_dir[0], -y_dir[1], -y_dir[2])
            tangent = x_dir.cross(y_dir)
            tz = tangent.magnitude()
            if tz > 1e-14:
                tangent = tangent * (1.0 / tz)

            rot = Xform.identity()
            rot.m[0]  = tangent[0]*prof_side[0] + x_dir[0]*prof_dir[0] + y_dir[0]*prof_up[0]
            rot.m[1]  = tangent[1]*prof_side[0] + x_dir[1]*prof_dir[0] + y_dir[1]*prof_up[0]
            rot.m[2]  = tangent[2]*prof_side[0] + x_dir[2]*prof_dir[0] + y_dir[2]*prof_up[0]
            rot.m[4]  = tangent[0]*prof_side[1] + x_dir[0]*prof_dir[1] + y_dir[0]*prof_up[1]
            rot.m[5]  = tangent[1]*prof_side[1] + x_dir[1]*prof_dir[1] + y_dir[1]*prof_up[1]
            rot.m[6]  = tangent[2]*prof_side[1] + x_dir[2]*prof_dir[1] + y_dir[2]*prof_up[1]
            rot.m[8]  = tangent[0]*prof_side[2] + x_dir[0]*prof_dir[2] + y_dir[0]*prof_up[2]
            rot.m[9]  = tangent[1]*prof_side[2] + x_dir[1]*prof_dir[2] + y_dir[1]*prof_up[2]
            rot.m[10] = tangent[2]*prof_side[2] + x_dir[2]*prof_dir[2] + y_dir[2]*prof_up[2]
            rot.m[12] = p1[0]
            rot.m[13] = p1[1]
            rot.m[14] = p1[2]

            prof_copy.transform(rot)
            positioned_profiles.append(prof_copy)

        loft_degree = min(3, len(positioned_profiles) - 1)
        return Primitives.create_loft(positioned_profiles, loft_degree)

    @staticmethod
    def create_edge(c0, c1, c2, c3):
        if not c0.is_valid() or not c1.is_valid() or not c2.is_valid() or not c3.is_valid():
            return NurbsSurface()

        input_curves = [c0.duplicate(), c1.duplicate(), c2.duplicate(), c3.duplicate()]
        loop = [input_curves[0].duplicate()]
        used = [True, False, False, False]
        tol = 1e-6

        for _step in range(3):
            tail = loop[-1].point_at_end()
            found = False
            for idx in range(4):
                if used[idx]:
                    continue
                s = input_curves[idx].point_at_start()
                e = input_curves[idx].point_at_end()
                if s.distance(tail) < tol:
                    loop.append(input_curves[idx].duplicate())
                    used[idx] = True
                    found = True
                    break
                if e.distance(tail) < tol:
                    rev = input_curves[idx].duplicate()
                    rev.reverse()
                    loop.append(rev)
                    used[idx] = True
                    found = True
                    break
            if not found:
                return NurbsSurface()

        if loop[3].point_at_end().distance(loop[0].point_at_start()) > tol:
            return NurbsSurface()

        south = loop[0].duplicate()
        east = loop[1].duplicate()
        north = loop[2].duplicate()
        north.reverse()
        west = loop[3].duplicate()
        west.reverse()

        v_pair = [south.duplicate(), north.duplicate()]
        Primitives._make_curves_compatible(v_pair)
        south = v_pair[0]
        north = v_pair[1]

        u_pair = [west.duplicate(), east.duplicate()]
        Primitives._make_curves_compatible(u_pair)
        west = u_pair[0]
        east = u_pair[1]

        order_v = south.order()
        cv_count_v = south.cv_count()
        order_u = west.order()
        cv_count_u = west.cv_count()
        is_rat = south.is_rational() or west.is_rational()

        surface = NurbsSurface.create_raw(3, is_rat, order_u, order_v, cv_count_u, cv_count_v)
        if surface is None:
            return NurbsSurface()

        for i in range(surface.knot_count(0)):
            surface.set_knot(0, i, west.knot(i))
        for i in range(surface.knot_count(1)):
            surface.set_knot(1, i, south.knot(i))

        u_grev = west.get_greville_abcissae()
        v_grev = south.get_greville_abcissae()

        u0, u1 = west.domain()
        v0, v1 = south.domain()
        u_grev = [(g - u0) / (u1 - u0) if u1 > u0 else 0.0 for g in u_grev]
        v_grev = [(g - v0) / (v1 - v0) if v1 > v0 else 0.0 for g in v_grev]

        c00 = south.get_cv(0) or Point(0.0, 0.0, 0.0)
        c01 = south.get_cv(cv_count_v - 1) or Point(0.0, 0.0, 0.0)
        c10 = north.get_cv(0) or Point(0.0, 0.0, 0.0)
        c11 = north.get_cv(cv_count_v - 1) or Point(0.0, 0.0, 0.0)

        for i in range(cv_count_u):
            ui = u_grev[i]
            wi = west.get_cv(i) or Point(0.0, 0.0, 0.0)
            ei = east.get_cv(i) or Point(0.0, 0.0, 0.0)
            for j in range(cv_count_v):
                vj = v_grev[j]
                sj = south.get_cv(j) or Point(0.0, 0.0, 0.0)
                nj = north.get_cv(j) or Point(0.0, 0.0, 0.0)

                x = ((1-ui)*sj[0] + ui*nj[0] + (1-vj)*wi[0] + vj*ei[0]
                     - (1-ui)*(1-vj)*c00[0] - (1-ui)*vj*c01[0]
                     - ui*(1-vj)*c10[0] - ui*vj*c11[0])
                y = ((1-ui)*sj[1] + ui*nj[1] + (1-vj)*wi[1] + vj*ei[1]
                     - (1-ui)*(1-vj)*c00[1] - (1-ui)*vj*c01[1]
                     - ui*(1-vj)*c10[1] - ui*vj*c11[1])
                z = ((1-ui)*sj[2] + ui*nj[2] + (1-vj)*wi[2] + vj*ei[2]
                     - (1-ui)*(1-vj)*c00[2] - (1-ui)*vj*c01[2]
                     - ui*(1-vj)*c10[2] - ui*vj*c11[2])

                surface.set_cv(i, j, Point(x, y, z))

        return surface

    @staticmethod
    def create_interpolated(points, parameterization=knot.CurveKnotStyle.Chord):
        return NurbsCurve.create_interpolated(points, parameterization)

    @staticmethod
    def quad_mesh(surface, u_count, v_count):
        mesh = Mesh()
        du = surface.domain(0)
        dv = surface.domain(1)
        nu, nv = u_count + 1, v_count + 1
        closed_u = surface.is_closed(0)
        singular_south = surface.is_singular(0)
        singular_north = surface.is_singular(2)

        vkeys = [[0]*nv for _ in range(nu)]
        for i in range(nu):
            u = du[0] + (du[1] - du[0]) * i / u_count
            for j in range(nv):
                if closed_u and i == u_count:
                    vkeys[i][j] = vkeys[0][j]; continue
                if singular_south and j == 0 and i > 0:
                    vkeys[i][j] = vkeys[0][0]; continue
                if singular_north and j == v_count and i > 0:
                    vkeys[i][j] = vkeys[0][v_count]; continue
                v = dv[0] + (dv[1] - dv[0]) * j / v_count
                vkeys[i][j] = mesh.add_vertex(surface.point_at(u, v))

        if singular_south:
            for i in range(u_count):
                mesh.add_face([vkeys[0][0], vkeys[i+1][1], vkeys[i][1]])
        if singular_north:
            for i in range(u_count):
                mesh.add_face([vkeys[0][v_count], vkeys[i][v_count-1], vkeys[i+1][v_count-1]])

        j0 = 1 if singular_south else 0
        j1 = v_count - 1 if singular_north else v_count
        for i in range(u_count):
            for j in range(j0, j1):
                mesh.add_face([vkeys[i][j], vkeys[i+1][j], vkeys[i+1][j+1], vkeys[i][j+1]])
        return mesh

    @staticmethod
    def diamond_mesh(surface, u_count, v_count):
        mesh = Mesh()
        du = surface.domain(0)
        dv = surface.domain(1)
        su = (du[1] - du[0]) / u_count
        sv = (dv[1] - dv[0]) / v_count
        nu, nv = u_count + 1, v_count + 1
        closed_u = surface.is_closed(0)
        singular_south = surface.is_singular(0)
        singular_north = surface.is_singular(2)

        grid = [[0]*nv for _ in range(nu)]
        for i in range(nu):
            u = du[0] + su * i
            for j in range(nv):
                if closed_u and i == u_count:
                    grid[i][j] = grid[0][j]; continue
                if singular_south and j == 0 and i > 0:
                    grid[i][j] = grid[0][0]; continue
                if singular_north and j == v_count and i > 0:
                    grid[i][j] = grid[0][v_count]; continue
                v = dv[0] + sv * j
                grid[i][j] = mesh.add_vertex(surface.point_at(u, v))

        u_end = u_count - 1 if closed_u else u_count
        for i in range(u_end + 1):
            for j in range(nv):
                if (i + j) % 2 != 0:
                    continue
                center = grid[i][j]
                il = i - 1 if i > 0 else (u_count - 1 if closed_u else -1)
                left   = grid[il][j] if il >= 0 else center
                bottom = grid[i][j-1] if j > 0 else center
                right  = grid[i+1][j] if i < u_count else center
                top    = grid[i][j+1] if j < v_count else center
                verts = [left, bottom, right, top]
                unique = []
                for k in range(4):
                    if verts[k] != verts[(k + 1) % 4]:
                        unique.append(verts[k])
                if len(unique) >= 3:
                    mesh.add_face(unique)
        return mesh

    @staticmethod
    def hex_mesh(surface, u_count, v_count, t=1.0/3.0):
        mesh = Mesh()
        du = surface.domain(0)
        dv = surface.domain(1)
        su = (du[1] - du[0]) / u_count
        sv = (dv[1] - dv[0]) / v_count

        nu, nv = u_count + 1, v_count + 1
        closed_u = surface.is_closed(0)
        singular_south = surface.is_singular(0)
        singular_north = surface.is_singular(2)

        grid = [[0]*nv for _ in range(nu)]
        for i in range(nu):
            u = du[0] + su * i
            for j in range(nv):
                if closed_u and i == u_count:
                    grid[i][j] = grid[0][j]; continue
                if singular_south and j == 0 and i > 0:
                    grid[i][j] = grid[0][0]; continue
                if singular_north and j == v_count and i > 0:
                    grid[i][j] = grid[0][v_count]; continue
                v = dv[0] + sv * j
                grid[i][j] = mesh.add_vertex(surface.point_at(u, v))

        mid_a = [[0]*v_count for _ in range(nu)]
        for i in range(nu):
            u = du[0] + su * i
            for j in range(v_count):
                if closed_u and i == u_count:
                    mid_a[i][j] = mid_a[0][j]; continue
                v = dv[0] + sv * (j + t)
                mid_a[i][j] = mesh.add_vertex(surface.point_at(u, v))

        mid_b = [[0]*v_count for _ in range(nu)]
        for i in range(nu):
            u = du[0] + su * i
            for j in range(v_count):
                if closed_u and i == u_count:
                    mid_b[i][j] = mid_b[0][j]; continue
                v = dv[0] + sv * (j + (1.0 - t))
                mid_b[i][j] = mesh.add_vertex(surface.point_at(u, v))

        def dedup_face(v):
            r = []
            n = len(v)
            for k in range(n):
                if v[k] != v[(k + 1) % n]:
                    r.append(v[k])
            return r

        u_end = u_count - 1 if closed_u else u_count
        for i in range(u_end + 1):
            for j in range(nv):
                if (i + j) % 2 != 0:
                    continue
                center = grid[i][j]
                il = i - 1 if i > 0 else (u_count - 1 if closed_u else -1)
                ul = mid_a[il][j]   if (il >= 0 and j < v_count)         else (grid[il][j] if il >= 0 else center)
                ll = mid_b[il][j-1] if (il >= 0 and j > 0)               else (grid[il][j] if il >= 0 else center)
                bt = mid_a[i][j-1]  if j > 0                             else center
                lr = mid_b[i+1][j-1] if (i < u_count and j > 0)          else (grid[i+1][j] if i < u_count else center)
                ur = mid_a[i+1][j]   if (i < u_count and j < v_count)    else (grid[i+1][j] if i < u_count else center)
                tp = mid_b[i][j]     if j < v_count                      else center

                face = dedup_face([ul, ll, bt, lr, ur, tp])
                if len(face) >= 3:
                    mesh.add_face(face)
        return mesh

    @staticmethod
    def tetrahedron(edge=2.0):
        a = edge / 2.0
        h = edge * math.sqrt(2.0 / 3.0)
        r = edge / math.sqrt(3.0)
        z0 = -h / 4.0
        z1 = 3.0 * h / 4.0
        faces = [
            [Point(a, -r/2.0, z0), Point(-a, -r/2.0, z0), Point(0, r, z0)],
            [Point(0, 0, z1), Point(-a, -r/2.0, z0), Point(a, -r/2.0, z0)],
            [Point(0, 0, z1), Point(0, r, z0), Point(-a, -r/2.0, z0)],
            [Point(0, 0, z1), Point(a, -r/2.0, z0), Point(0, r, z0)],
        ]
        return Mesh.from_polylines(faces, 1e-10)

    @staticmethod
    def cube(edge=2.0):
        a = edge / 2.0
        v0, v1, v2, v3 = Point(-a,-a,-a), Point(a,-a,-a), Point(a,a,-a), Point(-a,a,-a)
        v4, v5, v6, v7 = Point(-a,-a,a), Point(a,-a,a), Point(a,a,a), Point(-a,a,a)
        faces = [
            [v3, v2, v1, v0], [v4, v5, v6, v7],
            [v0, v1, v5, v4], [v2, v3, v7, v6],
            [v0, v4, v7, v3], [v1, v2, v6, v5],
        ]
        return Mesh.from_polylines(faces, 1e-10)

    @staticmethod
    def octahedron(edge=2.0):
        a = edge / math.sqrt(2.0)
        px, nx = Point(a,0,0), Point(-a,0,0)
        py, ny = Point(0,a,0), Point(0,-a,0)
        pz, nz = Point(0,0,a), Point(0,0,-a)
        faces = [
            [pz, px, py], [pz, py, nx], [pz, nx, ny], [pz, ny, px],
            [nz, py, px], [nz, nx, py], [nz, ny, nx], [nz, px, ny],
        ]
        return Mesh.from_polylines(faces, 1e-10)

    @staticmethod
    def icosahedron(edge=2.0):
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        s = edge / 2.0
        sp = s * phi
        verts = [
            Point(-s, sp, 0), Point(s, sp, 0), Point(-s,-sp, 0), Point(s,-sp, 0),
            Point(0,-s, sp), Point(0, s, sp), Point(0,-s,-sp), Point(0, s,-sp),
            Point(sp, 0,-s), Point(sp, 0, s), Point(-sp, 0,-s), Point(-sp, 0, s),
        ]
        idx = [
            [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
            [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
            [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
            [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
        ]
        faces = [[verts[f[0]], verts[f[1]], verts[f[2]]] for f in idx]
        return Mesh.from_polylines(faces, 1e-10)

    @staticmethod
    def dodecahedron(edge=2.0):
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        ip = 1.0 / phi
        s = edge / (2.0 * ip)
        verts = [
            Point( s,  s,  s), Point( s,  s, -s), Point( s, -s,  s), Point( s, -s, -s),
            Point(-s,  s,  s), Point(-s,  s, -s), Point(-s, -s,  s), Point(-s, -s, -s),
            Point(0,  s*ip,  s*phi), Point(0,  s*ip, -s*phi),
            Point(0, -s*ip,  s*phi), Point(0, -s*ip, -s*phi),
            Point( s*ip,  s*phi, 0), Point( s*ip, -s*phi, 0),
            Point(-s*ip,  s*phi, 0), Point(-s*ip, -s*phi, 0),
            Point( s*phi, 0,  s*ip), Point( s*phi, 0, -s*ip),
            Point(-s*phi, 0,  s*ip), Point(-s*phi, 0, -s*ip),
        ]
        idx = [
            [0, 8,10, 2,16], [0,16,17, 1,12], [0,12,14, 4, 8],
            [1,17, 3,11, 9], [1, 9, 5,14,12], [2,10, 6,15,13],
            [2,13, 3,17,16], [3,13,15, 7,11], [4,14, 5,19,18],
            [4,18, 6,10, 8], [5, 9,11, 7,19], [6,18,19, 7,15],
        ]
        faces = [[verts[f[0]], verts[f[1]], verts[f[2]], verts[f[3]], verts[f[4]]] for f in idx]
        return Mesh.from_polylines(faces, 1e-10)

    @staticmethod
    def wave_surface(size, amplitude):
        n = 13
        PI2 = 2.0 * math.pi
        pts = []
        for i in range(n):
            u = i / (n - 1)
            x = size * u
            for j in range(n):
                v = j / (n - 1)
                y = size * v
                z = amplitude * math.sin(PI2 * u) * math.sin(PI2 * v)
                pts.append(Point(x, y, z))
        return NurbsSurface.create(False, False, 3, 3, n, n, pts)

    @staticmethod
    def chevron_mesh(surface, u_divisions=4, v_division_dist=900.0, shift=0.5, scale=0.05799):
        srf = copy.deepcopy(surface)

        du0 = srf.domain(0)
        dv0 = srf.domain(1)
        nsamp = 50
        u_arc = 0.0
        v_arc = 0.0

        v_mid = (dv0[0] + dv0[1]) / 2.0
        prev = srf.point_at(du0[0], v_mid)
        for i in range(1, nsamp + 1):
            curr = srf.point_at(du0[0] + (du0[1] - du0[0]) * i / nsamp, v_mid)
            u_arc += prev.distance(curr)
            prev = curr

        u_mid = (du0[0] + du0[1]) / 2.0
        prev = srf.point_at(u_mid, dv0[0])
        for i in range(1, nsamp + 1):
            curr = srf.point_at(u_mid, dv0[0] + (dv0[1] - dv0[0]) * i / nsamp)
            v_arc += prev.distance(curr)
            prev = curr

        if u_arc > v_arc:
            srf.transpose()
            u_arc, v_arc = v_arc, u_arc

        du = srf.domain(0)
        dv = srf.domain(1)

        param_v = dv[1] - dv[0]
        half_v = dv[1] * 0.5
        StepU = (du[1] - du[0]) / u_divisions
        totalV = param_v
        baseStepV = v_division_dist * param_v / v_arc if v_arc > 1e-10 else v_division_dist

        polygons = []
        ctU = 0.0
        for j in range(u_divisions):
            ctV = 0.0
            thresh = totalV / 2.0
            StepV1 = baseStepV
            running = True
            ListV = []
            iterations = 0

            p0 = p1 = p2 = p6 = p7 = p8 = None
            savept6 = savept7 = savept8 = None

            while running and iterations < 1000:
                iterations += 1
                ListV.append(StepV1)

                if iterations == 1:
                    p0 = srf.point_at(ctU, ctV)
                    p1 = srf.point_at(ctU + StepU * 0.5, ctV)
                    p2 = srf.point_at(ctU + StepU, ctV)
                    p6 = srf.point_at(ctU, ctV + StepV1 * (1.0 - shift / 2.0))
                    p7 = srf.point_at(ctU + StepU * 0.5, ctV + StepV1 * (1.0 + shift / 2.0))
                    p8 = srf.point_at(ctU + StepU, ctV + StepV1 * (1.0 - shift / 2.0))
                    savept6, savept7, savept8 = p6, p7, p8
                else:
                    p0, p1, p2 = savept6, savept7, savept8
                    p6 = srf.point_at(ctU, ctV + StepV1 * (1.0 - shift / 2.0))
                    p7 = srf.point_at(ctU + StepU * 0.5, ctV + StepV1 * (1.0 + shift / 2.0))
                    p8 = srf.point_at(ctU + StepU, ctV + StepV1 * (1.0 - shift / 2.0))
                    savept6, savept7, savept8 = p6, p7, p8

                polygons.append([p0, p6, p7, p1])
                polygons.append([p1, p7, p8, p2])

                ctV += StepV1
                thresh -= StepV1
                StepV1 += StepV1 * scale

                if ctV + StepV1 > half_v:
                    ListV.append(thresh)
                    ListV.reverse()
                    revCt = totalV / 2.0

                    for i in range(len(ListV) - 1):
                        revCt += ListV[i]

                        if i == 0:
                            p0 = srf.point_at(ctU, revCt - ListV[i + 1] * shift / 2.0)
                            p1 = srf.point_at(ctU + StepU * 0.5, revCt + ListV[i + 1] * shift / 2.0)
                            p2 = srf.point_at(ctU + StepU, revCt - ListV[i + 1] * shift / 2.0)

                            polygons.append([p6, p0, p1, p7])
                            polygons.append([p7, p1, p2, p8])

                            p6 = srf.point_at(ctU, revCt + ListV[i + 1] * (1.0 - shift / 2.0))
                            p7 = srf.point_at(ctU + StepU * 0.5, revCt + ListV[i + 1] * (1.0 + shift / 2.0))
                            p8 = srf.point_at(ctU + StepU, revCt + ListV[i + 1] * (1.0 - shift / 2.0))
                            savept6, savept7, savept8 = p6, p7, p8
                        elif i == len(ListV) - 2:
                            p0, p1, p2 = savept6, savept7, savept8
                            p6 = srf.point_at(ctU, revCt + ListV[i + 1])
                            p7 = srf.point_at(ctU + StepU * 0.5, revCt + ListV[i + 1])
                            p8 = srf.point_at(ctU + StepU, revCt + ListV[i + 1])
                        else:
                            p0, p1, p2 = savept6, savept7, savept8
                            p6 = srf.point_at(ctU, revCt + ListV[i + 1] * (1.0 - shift / 2.0))
                            p7 = srf.point_at(ctU + StepU * 0.5, revCt + ListV[i + 1] * (1.0 + shift / 2.0))
                            p8 = srf.point_at(ctU + StepU, revCt + ListV[i + 1] * (1.0 - shift / 2.0))
                            savept6, savept7, savept8 = p6, p7, p8

                        polygons.append([p1, p7, p8, p2])
                        polygons.append([p0, p6, p7, p1])

                    running = False

            ctU += StepU

        return Mesh.from_polylines(polygons, 0.01)

    @staticmethod
    def annen_surfaces():
        surfaces = []
        prefixes = ["session_data/annen_surfaces/", "../session_data/annen_surfaces/"]
        for i in range(23):
            fname = f"surface_{i}.json"
            for prefix in prefixes:
                path = prefix + fname
                if not os.path.exists(path):
                    continue
                srf = NurbsSurface.json_load(path)
                if srf.is_valid():
                    surfaces.append(srf)
                    break
        return surfaces


class FoldedPlates:
    def __init__(self, surface, u_divisions, v_divisions, thickness, chamfer=0.0, base_planes=None, face_positions=None):
        self._srf = surface
        self._udiv = max(1, u_divisions)
        self._vdiv = max(1, v_divisions)
        self._thick = thickness
        self._cham = chamfer
        self._base_planes = base_planes if base_planes is not None else []
        self._face_pos = face_positions if face_positions is not None else [0.0]
        self._f = 0
        self.mesh = Mesh()
        self.flags = []
        self.adjacency = []
        self.polylines = []
        self.insertion_lines = []
        self._fkeys = []
        self._fv = []
        self._eidx = {}
        self._ef = []
        self._fplanes = []
        self._eplanes = []
        self._fe_planes = []

        self._diamond_subdivision()
        self._build_topology()
        self._compute_face_planes()
        self._compute_edge_planes()
        self._compute_face_edge_planes()
        self._compute_face_polylines()
        self._compute_insertion_vectors()

    @staticmethod
    def closest_on_line(pt, a, b):
        dx = b[0]-a[0]; dy = b[1]-a[1]; dz = b[2]-a[2]
        len2 = dx*dx + dy*dy + dz*dz
        if len2 < 1e-20:
            return a
        t = ((pt[0]-a[0])*dx + (pt[1]-a[1])*dy + (pt[2]-a[2])*dz) / len2
        return Point(a[0]+t*dx, a[1]+t*dy, a[2]+t*dz)

    @staticmethod
    def line_plane_t(a, b, pl):
        n = pl.z_axis
        o = pl.origin
        dx = b[0]-a[0]; dy = b[1]-a[1]; dz = b[2]-a[2]
        denom = n[0]*dx + n[1]*dy + n[2]*dz
        if abs(denom) < 1e-12:
            return None
        t = (n[0]*(o[0]-a[0]) + n[1]*(o[1]-a[1]) + n[2]*(o[2]-a[2])) / denom
        return t

    @staticmethod
    def intersect_3_planes(p0, p1, p2):
        n0 = p0.z_axis; n1 = p1.z_axis; n2 = p2.z_axis
        d0 = n0[0]*p0.origin[0] + n0[1]*p0.origin[1] + n0[2]*p0.origin[2]
        d1 = n1[0]*p1.origin[0] + n1[1]*p1.origin[1] + n1[2]*p1.origin[2]
        d2 = n2[0]*p2.origin[0] + n2[1]*p2.origin[1] + n2[2]*p2.origin[2]
        c12 = n1.cross(n2); c20 = n2.cross(n0); c01 = n0.cross(n1)
        det = n0.dot(c12)
        if abs(det) < 1e-12:
            return None
        inv = 1.0 / det
        return Point((d0*c12[0] + d1*c20[0] + d2*c01[0]) * inv,
                      (d0*c12[1] + d1*c20[1] + d2*c01[1]) * inv,
                      (d0*c12[2] + d1*c20[2] + d2*c01[2]) * inv)

    @staticmethod
    def chamfer_polyline(pl, value):
        if abs(value) < 1e-10:
            return pl
        n = pl.point_count()
        if n < 2:
            return pl
        segs = n - 1
        result = Polyline()
        for i in range(segs):
            a = pl.get_point(i); b = pl.get_point(i + 1)
            dx = b[0]-a[0]; dy = b[1]-a[1]; dz = b[2]-a[2]
            length = math.sqrt(dx*dx + dy*dy + dz*dz)
            if length < 1e-10:
                continue
            if value < 0:
                r = abs(value) / length
                result.add_point(Point(a[0]+r*dx, a[1]+r*dy, a[2]+r*dz))
                result.add_point(Point(b[0]-r*dx, b[1]-r*dy, b[2]-r*dz))
            else:
                result.add_point(Point(a[0]+value*dx, a[1]+value*dy, a[2]+value*dz))
                omt = 1.0 - value
                result.add_point(Point(a[0]+omt*dx, a[1]+omt*dy, a[2]+omt*dz))
        if result.point_count() > 0:
            result.add_point(result.get_point(0))
        return result

    def _diamond_subdivision(self):
        du = self._srf.domain(0)
        dv = self._srf.domain(1)
        su = (du[1] - du[0]) / self._udiv
        sv = (dv[1] - dv[0]) / self._vdiv
        tris = []
        fc = 0

        def plane_valid(p):
            z = p.z_axis
            return abs(z[0]) + abs(z[1]) + abs(z[2]) > 0.01

        uu = du[0]
        for i in range(self._udiv):
            a = 1 if (i % 2 == 0) else 0
            b = 2 if (i % 2 == 0) else 3

            vv = dv[0]
            for j in range(self._vdiv):
                p0 = self._srf.point_at(uu, vv)
                p1 = self._srf.point_at(uu + su, vv)
                p2 = self._srf.point_at(uu, vv + sv)
                p3 = self._srf.point_at(uu + su, vv + sv)
                p4 = self._srf.point_at(uu + su * 0.5, vv + sv * 1.5)
                p5 = self._srf.point_at(uu + su * 0.5, vv + sv * 0.5)
                p6 = self._srf.point_at(uu + su * 0.5, vv - sv * 0.5)

                if j == 0:
                    p9 = self._srf.point_at(uu + su * 0.5, dv[0])
                    cp = FoldedPlates.closest_on_line(p9, p5, p6)
                    if self._base_planes and plane_valid(self._base_planes[0]):
                        t = FoldedPlates.line_plane_t(p5, p6, self._base_planes[0])
                        if t is not None:
                            cp = Point(p5[0]+t*(p6[0]-p5[0]), p5[1]+t*(p6[1]-p5[1]), p5[2]+t*(p6[2]-p5[2]))
                    self.flags.append(fc % 4 == a or fc % 4 == b); tris.append([p5, cp, p1]); fc += 1
                    self.flags.append(fc % 4 == a or fc % 4 == b); tris.append([cp, p5, p0]); fc += 1

                self.flags.append(fc % 4 == a or fc % 4 == b); tris.append([p1, p3, p5]); fc += 1
                self.flags.append(fc % 4 == a or fc % 4 == b); tris.append([p2, p0, p5]); fc += 1

                if j < self._vdiv - 1:
                    self.flags.append(fc % 4 == a or fc % 4 == b); tris.append([p4, p5, p3]); fc += 1
                    self.flags.append(fc % 4 == a or fc % 4 == b); tris.append([p5, p4, p2]); fc += 1
                else:
                    p9 = self._srf.point_at(uu + su * 0.5, dv[1])
                    cp = FoldedPlates.closest_on_line(p9, p4, p5)
                    if self._base_planes and plane_valid(self._base_planes[-1]):
                        t = FoldedPlates.line_plane_t(p4, p5, self._base_planes[-1])
                        if t is not None:
                            cp = Point(p4[0]+t*(p5[0]-p4[0]), p4[1]+t*(p5[1]-p4[1]), p4[2]+t*(p5[2]-p4[2]))
                    self.flags.append(fc % 4 == a or fc % 4 == b); tris.append([cp, p5, p3]); fc += 1
                    self.flags.append(fc % 4 == a or fc % 4 == b); tris.append([p5, cp, p2]); fc += 1

                vv += sv
            uu += su

        self.mesh = Mesh.from_polylines(tris, 1e-6)

    def _build_topology(self):
        self._fkeys = sorted(self.mesh.face.keys())
        self._f = len(self._fkeys)

        self._fv = [list(self.mesh.face[fk]) for fk in self._fkeys]

        ef_map = {}
        for i in range(self._f):
            v = self._fv[i]
            n = len(v)
            for j in range(n):
                key = (min(v[j], v[(j+1)%n]), max(v[j], v[(j+1)%n]))
                if key not in ef_map:
                    ef_map[key] = []
                ef_map[key].append(i)

        ei = 0
        self._eidx = {}
        self._ef = []
        for ep, fl in sorted(ef_map.items()):
            self._eidx[ep] = ei
            self._ef.append(fl)
            ei += 1

        self.adjacency = []
        for i in range(self._f):
            if not self.flags[i]:
                continue
            v = self._fv[i]
            n = len(v)
            neighbors = set()
            for j in range(n):
                key = (min(v[j], v[(j+1)%n]), max(v[j], v[(j+1)%n]))
                for fi in self._ef[self._eidx[key]]:
                    if fi != i:
                        neighbors.add(fi)
            for ni in neighbors:
                self.adjacency.append([i, ni, -1, -1])

    def _compute_face_planes(self):
        self._fplanes = [None] * self._f
        for i in range(self._f):
            verts = self._fv[i]
            n = len(verts)
            cx = cy = cz = w = 0.0
            for j in range(n):
                pa = self.mesh.vertex_position(verts[j])
                pb = self.mesh.vertex_position(verts[(j+1)%n])
                d = math.sqrt((pb[0]-pa[0])**2 + (pb[1]-pa[1])**2 + (pb[2]-pa[2])**2)
                cx += d * (pa[0]+pb[0]) * 0.5
                cy += d * (pa[1]+pb[1]) * 0.5
                cz += d * (pa[2]+pb[2]) * 0.5
                w += d
            if w > 1e-10:
                cx /= w; cy /= w; cz /= w
            center = Point(cx, cy, cz)
            normal = self.mesh.face_normal(self._fkeys[i])
            if normal is None:
                normal = Vector(0, 0, 1)
            self._fplanes[i] = Plane.from_point_normal(center, normal)

    def _compute_edge_planes(self):
        self._eplanes = [None] * len(self._ef)
        for ep, ei in self._eidx.items():
            v1 = self.mesh.vertex_position(ep[0])
            v2 = self.mesh.vertex_position(ep[1])
            mid = Point((v1[0]+v2[0])*0.5, (v1[1]+v2[1])*0.5, (v1[2]+v2[2])*0.5)
            edir = Vector(v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2])
            edir.normalize_self()

            cf = self._ef[ei]
            if len(cf) == 2:
                fn0 = self.mesh.face_normal(self._fkeys[cf[0]])
                fn1 = self.mesh.face_normal(self._fkeys[cf[1]])
                avg = Vector((fn0[0]+fn1[0])*0.5, (fn0[1]+fn1[1])*0.5, (fn0[2]+fn1[2])*0.5)
                avg.normalize_self()
                z = edir.cross(avg)
            else:
                fn0 = self.mesh.face_normal(self._fkeys[cf[0]])
                z = fn0.cross(edir)

            self._eplanes[ei] = Plane.from_point_normal(mid, z)

    def _compute_face_edge_planes(self):
        self._fe_planes = [None] * self._f
        for i in range(self._f):
            v = self._fv[i]
            n = len(v)
            self._fe_planes[i] = [None] * n
            for j in range(n):
                v1 = v[j]; v2 = v[(j+1)%n]
                key = (min(v1, v2), max(v1, v2))
                cf = self._ef[self._eidx[key]]

                if len(cf) == 2:
                    self._fe_planes[i][j] = self._eplanes[self._eidx[key]]
                else:
                    p1 = self.mesh.vertex_position(v1)
                    p2 = self.mesh.vertex_position(v2)
                    mid = Point((p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5, (p1[2]+p2[2])*0.5)
                    sx = sy = sz = 0.0
                    for fi in cf:
                        fn = self.mesh.face_normal(self._fkeys[fi])
                        sx += fn[0]; sy += fn[1]; sz += fn[2]
                    inv = 1.0 / len(cf)
                    s = Vector(sx*inv, sy*inv, sz*inv)
                    xdir = Vector(p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2])
                    self._fe_planes[i][j] = Plane(mid, xdir, s)

    def _compute_face_polylines(self):
        self.polylines = [[] for _ in range(self._f)]
        for i in range(self._f):
            sides = self._fe_planes[i]
            n = len(sides)

            for j in range(len(self._face_pos)):
                base0 = self._fplanes[i].translate_by_normal(-self._face_pos[j])
                base1 = self._fplanes[i].translate_by_normal(-(self._face_pos[j] + self._thick))

                pl0 = Polyline()
                pl1 = Polyline()
                for k in range(n):
                    pt = FoldedPlates.intersect_3_planes(base0, sides[k], sides[(k+1)%n])
                    if pt is not None:
                        pl0.add_point(pt)
                    pt = FoldedPlates.intersect_3_planes(base1, sides[k], sides[(k+1)%n])
                    if pt is not None:
                        pl1.add_point(pt)
                if pl0.point_count() > 0:
                    pl0.add_point(pl0.get_point(0))
                if pl1.point_count() > 0:
                    pl1.add_point(pl1.get_point(0))

                self.polylines[i].append(pl0)
                self.polylines[i].append(pl1)

    def _compute_insertion_vectors(self):
        for i in range(self._f):
            if self.flags[i]:
                continue
            if not self.polylines[i] or self.polylines[i][0].point_count() < 4:
                continue

            pl = self.polylines[i][0]
            vec = Vector(pl.get_point(2)[0] - pl.get_point(0)[0],
                         pl.get_point(2)[1] - pl.get_point(0)[1],
                         pl.get_point(2)[2] - pl.get_point(0)[2])
            vec.normalize_self()
            vec = Vector(vec[0]*0.5, vec[1]*0.5, vec[2]*0.5)

            for j in range(3):
                a = pl.get_point(j)
                b = pl.get_point((j + 1) % 3)
                mid = Point((a[0]+b[0])*0.5, (a[1]+b[1])*0.5, (a[2]+b[2])*0.5)
                self.insertion_lines.append(Line.from_point_and_vector(mid, vec))

        if self._cham > 1e-6:
            for i in range(self._f):
                for j in range(len(self.polylines[i])):
                    self.polylines[i][j] = FoldedPlates.chamfer_polyline(self.polylines[i][j], -self._cham)


class CrossConnectors:
    def __init__(self, mesh, face_thickness, face_positions=None, edge_divisions=2,
                 rect_width=10.0, rect_height=10.0, rect_thickness=1.0, chamfer=0.0):
        self.mesh = mesh
        self._f = 0
        self._thick = face_thickness
        self._rect_w = rect_width
        self._rect_h = rect_height
        self._rect_t = rect_thickness
        self._cham = chamfer
        self._edge_div = max(1, edge_divisions)
        self._face_pos = list(face_positions) if face_positions is not None else [0.0]
        if not self._face_pos:
            self._face_pos = [0.0]
        self._face_pos.sort()

        self._fkeys = []
        self._fv = []
        self._eidx = {}
        self.face_planes = []
        self.fe_planes = []
        self.bisector_planes = []
        self.face_polylines = []
        self.edges = []
        self.edge_faces = []
        self.edge_planes = []
        self.edge_polylines = []
        self._e90_multiple_planes = []

        self._build_topology()
        self._compute_face_planes()
        self._compute_face_edge_planes()
        self._compute_bisector_planes()
        self._compute_face_polylines()
        self._compute_edges()
        self._compute_edge_faces()
        self._compute_edge_planes_method()
        self._compute_connectors()

    def _build_topology(self):
        self._fkeys = sorted(self.mesh.face.keys())
        self._f = len(self._fkeys)
        self._fv = [list(self.mesh.face[fk]) for fk in self._fkeys]

        self._eidx = {}
        ei = 0
        for i in range(self._f):
            v = self._fv[i]
            n = len(v)
            for j in range(n):
                key = (min(v[j], v[(j+1)%n]), max(v[j], v[(j+1)%n]))
                if key not in self._eidx:
                    self._eidx[key] = ei
                    ei += 1

    def _compute_face_planes(self):
        self.face_planes = [None] * self._f
        for i in range(self._f):
            verts = self._fv[i]
            n = len(verts)
            cx = cy = cz = w = 0.0
            for j in range(n):
                pa = self.mesh.vertex_position(verts[j])
                pb = self.mesh.vertex_position(verts[(j+1)%n])
                d = pa.distance(pb)
                cx += d * (pa[0]+pb[0]) * 0.5
                cy += d * (pa[1]+pb[1]) * 0.5
                cz += d * (pa[2]+pb[2]) * 0.5
                w += d
            if w > 1e-10:
                cx /= w; cy /= w; cz /= w
            center = Point(cx, cy, cz)
            normal = self.mesh.face_normal(self._fkeys[i])
            if normal is None:
                normal = Vector(0, 0, 1)
            self.face_planes[i] = Plane.from_point_normal(center, normal)

    def _compute_face_edge_planes(self):
        self.fe_planes = [None] * self._f
        for i in range(self._f):
            v = self._fv[i]
            n = len(v)
            self.fe_planes[i] = [None] * n
            for j in range(n):
                v1 = v[j]; v2 = v[(j+1)%n]
                p1 = self.mesh.vertex_position(v1)
                p2 = self.mesh.vertex_position(v2)
                mid = Point((p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5, (p1[2]+p2[2])*0.5)

                key = (min(v1, v2), max(v1, v2))

                sx = sy = sz = 0.0
                count = 0
                for fi in range(self._f):
                    fv = self._fv[fi]
                    fn = len(fv)
                    for k in range(fn):
                        ek = (min(fv[k], fv[(k+1)%fn]), max(fv[k], fv[(k+1)%fn]))
                        if ek == key:
                            fn_vec = self.mesh.face_normal(self._fkeys[fi])
                            if fn_vec is None:
                                fn_vec = Vector(0, 0, 1)
                            sx += fn_vec[0]; sy += fn_vec[1]; sz += fn_vec[2]
                            count += 1
                            break
                if count > 0:
                    inv = 1.0 / count
                    sx *= inv; sy *= inv; sz *= inv

                xaxis = Vector(p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2])
                xaxis.normalize_self()
                s = Vector(sx, sy, sz)
                s.normalize_self()
                zaxis = xaxis.cross(s)
                zaxis.normalize_self()
                self.fe_planes[i][j] = Plane.from_frame(mid, xaxis, s, zaxis)

    @staticmethod
    def dihedral_plane(p0, p1):
        isect_line = intersection.plane_plane(p0, p1)
        if isect_line is None:
            return p0

        centerDihedral = intersection.line_plane(isect_line, p0, False)
        if centerDihedral is None:
            a = isect_line.start(); b = isect_line.end()
            centerDihedral = Point((a[0]+b[0])*0.5, (a[1]+b[1])*0.5, (a[2]+b[2])*0.5)

        line_dir = Vector(isect_line.end()[0]-isect_line.start()[0],
                          isect_line.end()[1]-isect_line.start()[1],
                          isect_line.end()[2]-isect_line.start()[2])
        line_dir.normalize_self()

        z0 = p0.z_axis; z1 = p1.z_axis
        o0 = p0.origin; o1 = p1.origin
        ray0 = Line(o0[0], o0[1], o0[2], o0[0]+z0[0], o0[1]+z0[1], o0[2]+z0[2])
        ray1 = Line(o1[0], o1[1], o1[2], o1[0]+z1[0], o1[1]+z1[1], o1[2]+z1[2])

        parallel = abs(z0.dot(z1)) > 0.9999
        dist2 = (o0[0]-o1[0])**2 + (o0[1]-o1[1])**2 + (o0[2]-o1[2])**2

        if not parallel and dist2 > 0.001:
            result = intersection.line_line_parameters(ray0, ray1, 0.01, False, False)
            if result is not None:
                t0, t1 = result
                center = Point(o0[0]+t0*z0[0], o0[1]+t0*z0[1], o0[2]+t0*z0[2])
                v0 = Vector(o0[0]-center[0], o0[1]-center[1], o0[2]-center[2])
                v1 = Vector(o1[0]-center[0], o1[1]-center[1], o1[2]-center[2])
                v0.normalize_self()
                v1.normalize_self()
                bisector = Vector(v0[0]+v1[0], v0[1]+v1[1], v0[2]+v1[2])
                return Plane(centerDihedral, line_dir, bisector)

        bisector = Vector(z0[0]+z1[0], z0[1]+z1[1], z0[2]+z1[2])
        return Plane(centerDihedral, line_dir, bisector)

    def _compute_bisector_planes(self):
        self.bisector_planes = [None] * self._f
        for i in range(self._f):
            n = len(self.fe_planes[i])
            self.bisector_planes[i] = [None] * n
            for j in range(n):
                self.bisector_planes[i][j] = CrossConnectors.dihedral_plane(
                    self.fe_planes[i][(j+1)%n], self.fe_planes[i][j])

    @staticmethod
    def move_plane_by_axis(pl, dist, axis=2):
        if axis == 0:
            d = pl.x_axis
        elif axis == 1:
            d = pl.y_axis
        else:
            d = pl.z_axis
        o = pl.origin
        new_o = Point(o[0]+d[0]*dist, o[1]+d[1]*dist, o[2]+d[2]*dist)
        return Plane.from_frame(new_o, pl.x_axis, pl.y_axis, pl.z_axis)

    @staticmethod
    def outline_from_planes(face_plane, edge_planes, bise_planes):
        pl = Polyline()
        n = len(edge_planes)
        for i in range(n):
            plane = edge_planes[i]
            plane1 = edge_planes[(i+1)%n]

            angle = math.acos(min(1.0, max(-1.0, plane.z_axis.dot(plane1.z_axis))))
            if angle < 0.01 and angle > -0.01:
                continue

            line = intersection.plane_plane(plane, plane1)
            if line is None:
                continue
            pt = intersection.line_plane(line, face_plane, False)
            if pt is None:
                continue
            pl.add_point(pt)
        if pl.point_count() > 0:
            pl.add_point(pl.get_point(0))
        return pl

    def _compute_face_polylines(self):
        self.face_polylines = [[] for _ in range(self._f)]
        for i in range(self._f):
            for j in range(len(self._face_pos)):
                base_bot = CrossConnectors.move_plane_by_axis(self.face_planes[i], self._face_pos[j] + self._thick * -0.5)
                base_top = CrossConnectors.move_plane_by_axis(self.face_planes[i], self._face_pos[j] + self._thick * 0.5)
                pline0 = CrossConnectors.outline_from_planes(base_bot, self.fe_planes[i], self.bisector_planes[i])
                pline1 = CrossConnectors.outline_from_planes(base_top, self.fe_planes[i], self.bisector_planes[i])
                self.face_polylines[i].append(pline0)
                self.face_polylines[i].append(pline1)

    def _compute_edges(self):
        self.edges = []
        seen = set()
        for i in range(self._f):
            v = self._fv[i]
            n = len(v)
            for j in range(n):
                key = (min(v[j], v[(j+1)%n]), max(v[j], v[(j+1)%n]))
                if key not in seen:
                    seen.add(key)
                    self.edges.append(key)

    def _compute_edge_faces(self):
        self.edge_faces = [[] for _ in range(len(self.edges))]
        edge_idx = {}
        for i, e in enumerate(self.edges):
            edge_idx[e] = i

        for i in range(self._f):
            v = self._fv[i]
            n = len(v)
            for j in range(n):
                key = (min(v[j], v[(j+1)%n]), max(v[j], v[(j+1)%n]))
                if key in edge_idx:
                    self.edge_faces[edge_idx[key]].append(i)

    def _compute_edge_planes_method(self):
        self.edge_planes = [None] * len(self.edges)
        self._e90_multiple_planes = [[] for _ in range(len(self.edges))]

        for i in range(len(self.edges)):
            if len(self.edge_faces[i]) < 2:
                self.edge_planes[i] = Plane()
                self._e90_multiple_planes[i] = [Plane()]
                continue

            v1 = self.mesh.vertex_position(self.edges[i][0])
            v2 = self.mesh.vertex_position(self.edges[i][1])
            mid = Point((v1[0]+v2[0])*0.5, (v1[1]+v2[1])*0.5, (v1[2]+v2[2])*0.5)
            zaxis = Vector(v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2])
            zaxis.normalize_self()

            z0 = self.face_planes[self.edge_faces[i][0]].z_axis
            z1 = self.face_planes[self.edge_faces[i][1]].z_axis
            yaxis = Vector((z0[0]+z1[0])*0.5, (z0[1]+z1[1])*0.5, (z0[2]+z1[2])*0.5)
            yaxis.normalize_self()

            xaxis = Vector(zaxis[1]*yaxis[2]-zaxis[2]*yaxis[1],
                           zaxis[2]*yaxis[0]-zaxis[0]*yaxis[2],
                           zaxis[0]*yaxis[1]-zaxis[1]*yaxis[0])
            xaxis.normalize_self()

            f0_o = self.face_planes[self.edge_faces[i][0]].origin
            d_pos = ((mid[0]+xaxis[0]-f0_o[0])**2 + (mid[1]+xaxis[1]-f0_o[1])**2 + (mid[2]+xaxis[2]-f0_o[2])**2)
            d_neg = ((mid[0]-xaxis[0]-f0_o[0])**2 + (mid[1]-xaxis[1]-f0_o[1])**2 + (mid[2]-xaxis[2]-f0_o[2])**2)
            if d_pos > d_neg:
                xaxis = Vector(-xaxis[0], -xaxis[1], -xaxis[2])

            self.edge_planes[i] = Plane.from_frame(mid, xaxis, yaxis, zaxis)

            self._e90_multiple_planes[i] = []
            for d in range(self._edge_div):
                t = (d + 1.0) / (self._edge_div + 1.0)
                pt = Point(v1[0]+t*(v2[0]-v1[0]), v1[1]+t*(v2[1]-v1[1]), v1[2]+t*(v2[2]-v1[2]))
                self._e90_multiple_planes[i].append(Plane.from_frame(pt, xaxis, yaxis, zaxis))

    @staticmethod
    def make_rectangle(pl, w, h):
        x = pl.x_axis; y = pl.y_axis; o = pl.origin
        hw = w * 0.5; hh = h * 0.5
        rect = Polyline()
        rect.add_point(Point(o[0]-hw*x[0]-hh*y[0], o[1]-hw*x[1]-hh*y[1], o[2]-hw*x[2]-hh*y[2]))
        rect.add_point(Point(o[0]+hw*x[0]-hh*y[0], o[1]+hw*x[1]-hh*y[1], o[2]+hw*x[2]-hh*y[2]))
        rect.add_point(Point(o[0]+hw*x[0]+hh*y[0], o[1]+hw*x[1]+hh*y[1], o[2]+hw*x[2]+hh*y[2]))
        rect.add_point(Point(o[0]-hw*x[0]+hh*y[0], o[1]-hw*x[1]+hh*y[1], o[2]-hw*x[2]+hh*y[2]))
        rect.add_point(rect.get_point(0))
        return rect

    def _compute_connectors(self):
        self.edge_polylines = [[] for _ in range(len(self.edges))]
        for i in range(len(self.edges)):
            if len(self.edge_faces[i]) < 2:
                continue

            for j in range(len(self._e90_multiple_planes[i])):
                epl = self._e90_multiple_planes[i][j]
                if self._rect_w > 0 and self._rect_h > 0:
                    pl0 = CrossConnectors.move_plane_by_axis(epl, self._rect_t * 0.5)
                    pl1 = CrossConnectors.move_plane_by_axis(epl, self._rect_t * -0.5)
                    self.edge_polylines[i].append(CrossConnectors.make_rectangle(pl0, self._rect_w, self._rect_h))
                    self.edge_polylines[i].append(CrossConnectors.make_rectangle(pl1, self._rect_w, self._rect_h))
                else:
                    w = abs(self._rect_h)
                    h = abs(self._rect_w)

                    new_x = epl.z_axis
                    ey = epl.y_axis
                    new_z = Vector(-epl.x_axis[0], -epl.x_axis[1], -epl.x_axis[2])
                    e_plane = Plane.from_frame(epl.origin, new_x, ey, new_z)

                    top0 = CrossConnectors.move_plane_by_axis(
                        CrossConnectors.move_plane_by_axis(self.face_planes[self.edge_faces[i][0]], self._face_pos[-1]),
                        self._thick * 0.5 + h)
                    top1 = CrossConnectors.move_plane_by_axis(
                        CrossConnectors.move_plane_by_axis(self.face_planes[self.edge_faces[i][1]], self._face_pos[-1]),
                        self._thick * 0.5 + h)
                    bot0 = CrossConnectors.move_plane_by_axis(
                        CrossConnectors.move_plane_by_axis(self.face_planes[self.edge_faces[i][0]], self._face_pos[0]),
                        self._thick * -0.5 - h)
                    bot1 = CrossConnectors.move_plane_by_axis(
                        CrossConnectors.move_plane_by_axis(self.face_planes[self.edge_faces[i][1]], self._face_pos[0]),
                        self._thick * -0.5 - h)

                    emx = self.edge_planes[i].z_axis
                    emy = self.edge_planes[i].y_axis
                    emz = Vector(-self.edge_planes[i].x_axis[0], -self.edge_planes[i].x_axis[1], -self.edge_planes[i].x_axis[2])
                    e_plane_main = Plane.from_frame(self.edge_planes[i].origin, emx, emy, emz)

                    sides = [
                        top0, e_plane, top1,
                        CrossConnectors.move_plane_by_axis(e_plane_main, w * 0.5),
                        bot1, e_plane, bot0,
                        CrossConnectors.move_plane_by_axis(e_plane_main, w * -0.5)
                    ]

                    base0 = CrossConnectors.move_plane_by_axis(epl, -self._rect_t * 0.5)
                    type1_0 = Polyline()
                    ns = len(sides)
                    for s in range(ns):
                        line = intersection.plane_plane(sides[s], sides[(s+1)%ns])
                        if line is None:
                            continue
                        pt = intersection.line_plane(line, base0, False)
                        if pt is None:
                            continue
                        type1_0.add_point(pt)
                    if type1_0.point_count() > 0:
                        type1_0.add_point(type1_0.get_point(0))

                    type1_1 = Polyline()
                    zdir = epl.z_axis
                    for p in range(type1_0.point_count()):
                        pp = type1_0.get_point(p)
                        type1_1.add_point(Point(pp[0]+zdir[0]*self._rect_t, pp[1]+zdir[1]*self._rect_t, pp[2]+zdir[2]*self._rect_t))

                    self.edge_polylines[i].append(type1_0)
                    self.edge_polylines[i].append(type1_1)
