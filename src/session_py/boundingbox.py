import uuid
from typing import List
from .point import Point
from .vector import Vector
from .plane import Plane


class BoundingBox:
    def __init__(
        self,
        center: Point = None,
        x_axis: Vector = None,
        y_axis: Vector = None,
        z_axis: Vector = None,
        half_size: Vector = None,
    ):
        self.center = center if center is not None else Point(0.0, 0.0, 0.0)
        self.x_axis = x_axis if x_axis is not None else Vector(1.0, 0.0, 0.0)
        self.y_axis = y_axis if y_axis is not None else Vector(0.0, 1.0, 0.0)
        self.z_axis = z_axis if z_axis is not None else Vector(0.0, 0.0, 1.0)
        self.half_size = half_size if half_size is not None else Vector(0.5, 0.5, 0.5)
        self.guid = str(uuid.uuid4())
        self.name = "my_boundingbox"

    @classmethod
    def from_plane(cls, plane: Plane, dx: float, dy: float, dz: float):
        return cls(
            center=plane.origin,
            x_axis=plane.x_axis,
            y_axis=plane.y_axis,
            z_axis=plane.z_axis,
            half_size=Vector(dx * 0.5, dy * 0.5, dz * 0.5),
        )

    @classmethod
    def from_point(cls, point: Point, inflate: float = 0.0):
        return cls(
            center=point,
            x_axis=Vector(1.0, 0.0, 0.0),
            y_axis=Vector(0.0, 1.0, 0.0),
            z_axis=Vector(0.0, 0.0, 1.0),
            half_size=Vector(inflate, inflate, inflate),
        )

    @classmethod
    def from_points(cls, points: List[Point], inflate: float = 0.0):
        if not points:
            return cls()

        min_x = min(p.x for p in points)
        min_y = min(p.y for p in points)
        min_z = min(p.z for p in points)
        max_x = max(p.x for p in points)
        max_y = max(p.y for p in points)
        max_z = max(p.z for p in points)

        center = Point(
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
        )
        half_size = Vector(
            (max_x - min_x) * 0.5 + inflate,
            (max_y - min_y) * 0.5 + inflate,
            (max_z - min_z) * 0.5 + inflate,
        )

        return cls(
            center=center,
            x_axis=Vector(1.0, 0.0, 0.0),
            y_axis=Vector(0.0, 1.0, 0.0),
            z_axis=Vector(0.0, 0.0, 1.0),
            half_size=half_size,
        )

    @classmethod
    def from_line(cls, line, inflate: float = 0.0):
        points = [line.start(), line.end()]
        return cls.from_points(points, inflate)

    @classmethod
    def from_polyline(cls, polyline, inflate: float = 0.0, plane=None):
        """Create bounding box from polyline.

        Parameters
        ----------
        polyline : Polyline
            The polyline to bound.
        inflate : float, optional
            Amount to inflate the bounding box (default 0.0).
        plane : Plane, optional
            If provided, creates an OOBB aligned to the plane.

        Returns
        -------
        BoundingBox
            Axis-aligned or oriented bounding box containing the polyline.
        """
        if plane is not None:
            return cls.from_points(polyline.points, plane, inflate)
        return cls.from_points(polyline.points, inflate)

    @classmethod
    def from_mesh(cls, mesh, inflate: float = 0.0, plane=None):
        """Create bounding box from mesh.

        Parameters
        ----------
        mesh : Mesh
            The mesh to bound.
        inflate : float, optional
            Amount to inflate the bounding box (default 0.0).
        plane : Plane, optional
            If provided, creates an OOBB aligned to the plane.

        Returns
        -------
        BoundingBox
            Axis-aligned or oriented bounding box containing the mesh.
        """
        vertices, faces = mesh.to_vertices_and_faces()
        if plane is not None:
            return cls.from_points(vertices, plane, inflate)
        return cls.from_points(vertices, inflate)

    @classmethod
    def from_pointcloud(cls, pointcloud, inflate: float = 0.0, plane=None):
        points = pointcloud.get_points()
        if plane is not None:
            return cls.from_points_with_plane(points, plane, inflate)
        return cls.from_points(points, inflate)

    @classmethod
    def from_nurbssurface(cls, surface, inflate: float = 0.0, plane=None):
        if not surface.is_valid() or surface.cv_count(0) == 0 or surface.cv_count(1) == 0:
            return cls()
        points = []
        for i in range(surface.cv_count(0)):
            for j in range(surface.cv_count(1)):
                points.append(surface.get_cv(i, j))
        if plane is not None:
            return cls.from_points_with_plane(points, plane, inflate)
        return cls.from_points(points, inflate)

    @classmethod
    def from_nurbscurve(cls, curve, inflate: float = 0.0, tight: bool = False, plane=None):
        if not curve.is_valid() or curve.cv_count() == 0:
            return cls()

        if not tight:
            points = [curve.get_cv(i) for i in range(curve.cv_count())]
            if plane is not None:
                return cls.from_points_with_plane(points, plane, inflate)
            return cls.from_points(points, inflate)

        t0, t1 = curve.domain()
        extrema_points = [curve.point_at(t0), curve.point_at(t1)]

        spans = curve.get_span_vector()
        for t in spans:
            if t > t0 and t < t1:
                extrema_points.append(curve.point_at(t))

        if plane is not None:
            axes = [plane.x_axis, plane.y_axis, plane.z_axis]
        else:
            axes = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

        NUM_SAMPLES = 20
        dt = (t1 - t0) / NUM_SAMPLES

        for axis_idx, axis in enumerate(axes):
            for i in range(NUM_SAMPLES):
                t_start = t0 + i * dt
                t_end = t_start + dt

                deriv_start = curve.evaluate(t_start, 1)
                deriv_end = curve.evaluate(t_end, 1)
                if len(deriv_start) < 2 or len(deriv_end) < 2:
                    continue

                if plane is not None:
                    d_start = deriv_start[1].dot(axis)
                    d_end = deriv_end[1].dot(axis)
                else:
                    d_start = deriv_start[1][axis_idx]
                    d_end = deriv_end[1][axis_idx]

                if d_start * d_end < 0:
                    t_lo, t_hi = t_start, t_end
                    t_root = (t_lo + t_hi) * 0.5

                    for _ in range(20):
                        deriv = curve.evaluate(t_root, 2)
                        if len(deriv) < 3:
                            break

                        if plane is not None:
                            f = deriv[1].dot(axis)
                            fp = deriv[2].dot(axis)
                        else:
                            f = deriv[1][axis_idx]
                            fp = deriv[2][axis_idx]

                        if abs(f) < 1e-12:
                            break

                        if abs(fp) > 1e-14:
                            t_new = t_root - f / fp
                            if t_lo <= t_new <= t_hi:
                                t_root = t_new
                            else:
                                if f * d_start < 0:
                                    t_hi = t_root
                                else:
                                    t_lo = t_root
                                t_root = (t_lo + t_hi) * 0.5
                        else:
                            t_root = (t_lo + t_hi) * 0.5

                        deriv_check = curve.evaluate(t_root, 1)
                        if len(deriv_check) >= 2:
                            if plane is not None:
                                f_check = deriv_check[1].dot(axis)
                            else:
                                f_check = deriv_check[1][axis_idx]
                            if f_check * d_start < 0:
                                t_hi = t_root
                                d_end = f_check
                            else:
                                t_lo = t_root
                                d_start = f_check

                    extrema_points.append(curve.point_at(t_root))

        if plane is not None:
            return cls.from_points_with_plane(extrema_points, plane, inflate)
        return cls.from_points(extrema_points, inflate)

    @classmethod
    def from_points_with_plane(cls, points: List[Point], plane, inflate: float = 0.0):
        if not points:
            return cls()

        from .xform import Xform
        origin = plane.origin
        x_axis = plane.x_axis
        y_axis = plane.y_axis
        z_axis = plane.z_axis
        plane_to_xy = Xform.plane_to_xy(origin, x_axis, y_axis, z_axis)

        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for pt in points:
            local_pt = plane_to_xy.transformed_point(pt)
            min_x = min(min_x, local_pt.x)
            min_y = min(min_y, local_pt.y)
            min_z = min(min_z, local_pt.z)
            max_x = max(max_x, local_pt.x)
            max_y = max(max_y, local_pt.y)
            max_z = max(max_z, local_pt.z)

        local_center = Point((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5)
        half_size = Vector(
            (max_x - min_x) * 0.5 + inflate,
            (max_y - min_y) * 0.5 + inflate,
            (max_z - min_z) * 0.5 + inflate
        )

        xy_to_plane = Xform.xy_to_plane(origin, x_axis, y_axis, z_axis)
        world_center = xy_to_plane.transformed_point(local_center)

        return cls(world_center, x_axis, y_axis, z_axis, half_size)

    def aabb(self):
        ex, ey, ez = self.half_size[0], self.half_size[1], self.half_size[2]
        hx = abs(self.x_axis[0]) * ex + abs(self.y_axis[0]) * ey + abs(self.z_axis[0]) * ez
        hy = abs(self.x_axis[1]) * ex + abs(self.y_axis[1]) * ey + abs(self.z_axis[1]) * ez
        hz = abs(self.x_axis[2]) * ex + abs(self.y_axis[2]) * ey + abs(self.z_axis[2]) * ez
        return BoundingBox(self.center, Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1), Vector(hx, hy, hz))

    def point_at(self, x: float, y: float, z: float) -> Point:
        return Point(
            self.center.x + x * self.x_axis[0] + y * self.y_axis[0] + z * self.z_axis[0],
            self.center.y + x * self.x_axis[1] + y * self.y_axis[1] + z * self.z_axis[1],
            self.center.z + x * self.x_axis[2] + y * self.y_axis[2] + z * self.z_axis[2],
        )

    def min_point(self) -> Point:
        """Get the minimum corner point of the axis-aligned bounding box.

        Returns
        -------
        Point
            The point with minimum x, y, z coordinates.
        """
        return Point(
            self.center.x - self.half_size[0],
            self.center.y - self.half_size[1],
            self.center.z - self.half_size[2],
        )

    def max_point(self) -> Point:
        """Get the maximum corner point of the axis-aligned bounding box.

        Returns
        -------
        Point
            The point with maximum x, y, z coordinates.
        """
        return Point(
            self.center.x + self.half_size[0],
            self.center.y + self.half_size[1],
            self.center.z + self.half_size[2],
        )

    def corners(self) -> List[Point]:
        """Get all 8 corner points of the bounding box.

        Returns
        -------
        List[Point]
            List of 8 corner points in a specific order.
        """
        return [
            self.point_at(self.half_size[0], self.half_size[1], -self.half_size[2]),
            self.point_at(-self.half_size[0], self.half_size[1], -self.half_size[2]),
            self.point_at(-self.half_size[0], -self.half_size[1], -self.half_size[2]),
            self.point_at(self.half_size[0], -self.half_size[1], -self.half_size[2]),
            self.point_at(self.half_size[0], self.half_size[1], self.half_size[2]),
            self.point_at(-self.half_size[0], self.half_size[1], self.half_size[2]),
            self.point_at(-self.half_size[0], -self.half_size[1], self.half_size[2]),
            self.point_at(self.half_size[0], -self.half_size[1], self.half_size[2]),
        ]

    def two_rectangles(self) -> List[Point]:
        return [
            self.point_at(self.half_size[0], self.half_size[1], -self.half_size[2]),
            self.point_at(-self.half_size[0], self.half_size[1], -self.half_size[2]),
            self.point_at(-self.half_size[0], -self.half_size[1], -self.half_size[2]),
            self.point_at(self.half_size[0], -self.half_size[1], -self.half_size[2]),
            self.point_at(self.half_size[0], self.half_size[1], -self.half_size[2]),
            self.point_at(self.half_size[0], self.half_size[1], self.half_size[2]),
            self.point_at(-self.half_size[0], self.half_size[1], self.half_size[2]),
            self.point_at(-self.half_size[0], -self.half_size[1], self.half_size[2]),
            self.point_at(self.half_size[0], -self.half_size[1], self.half_size[2]),
            self.point_at(self.half_size[0], self.half_size[1], self.half_size[2]),
        ]

    def inflate(self, amount: float):
        self.half_size = Vector(
            self.half_size[0] + amount,
            self.half_size[1] + amount,
            self.half_size[2] + amount,
        )

    @staticmethod
    def _separating_plane_exists(
        relative_position: Vector,
        axis: Vector,
        box1: "BoundingBox",
        box2: "BoundingBox",
    ) -> bool:
        dot_rp = abs(relative_position.dot(axis))

        v1 = box1.x_axis * box1.half_size[0]
        v2 = box1.y_axis * box1.half_size[1]
        v3 = box1.z_axis * box1.half_size[2]
        proj1 = abs(v1.dot(axis)) + abs(v2.dot(axis)) + abs(v3.dot(axis))

        v4 = box2.x_axis * box2.half_size[0]
        v5 = box2.y_axis * box2.half_size[1]
        v6 = box2.z_axis * box2.half_size[2]
        proj2 = abs(v4.dot(axis)) + abs(v5.dot(axis)) + abs(v6.dot(axis))

        return dot_rp > (proj1 + proj2)

    def collides_with(self, other: "BoundingBox") -> bool:
        center_vec = Vector(self.center.x, self.center.y, self.center.z)
        other_center_vec = Vector(other.center.x, other.center.y, other.center.z)
        relative_position = Vector.from_points(center_vec, other_center_vec)

        return not (
            self._separating_plane_exists(relative_position, self.x_axis, self, other)
            or self._separating_plane_exists(
                relative_position, self.y_axis, self, other
            )
            or self._separating_plane_exists(
                relative_position, self.z_axis, self, other
            )
            or self._separating_plane_exists(
                relative_position, other.x_axis, self, other
            )
            or self._separating_plane_exists(
                relative_position, other.y_axis, self, other
            )
            or self._separating_plane_exists(
                relative_position, other.z_axis, self, other
            )
            or self._separating_plane_exists(
                relative_position, self.x_axis.cross(other.x_axis), self, other
            )
            or self._separating_plane_exists(
                relative_position, self.x_axis.cross(other.y_axis), self, other
            )
            or self._separating_plane_exists(
                relative_position, self.x_axis.cross(other.z_axis), self, other
            )
            or self._separating_plane_exists(
                relative_position, self.y_axis.cross(other.x_axis), self, other
            )
            or self._separating_plane_exists(
                relative_position, self.y_axis.cross(other.y_axis), self, other
            )
            or self._separating_plane_exists(
                relative_position, self.y_axis.cross(other.z_axis), self, other
            )
            or self._separating_plane_exists(
                relative_position, self.z_axis.cross(other.x_axis), self, other
            )
            or self._separating_plane_exists(
                relative_position, self.z_axis.cross(other.y_axis), self, other
            )
            or self._separating_plane_exists(
                relative_position, self.z_axis.cross(other.z_axis), self, other
            )
        )

    ###########################################################################################
    # Transformation
    ###########################################################################################

    def transform(self):
        """Apply the stored xform transformation to the bounding box.

        Transforms the bounding box in-place and resets xform to identity.
        """
        from .xform import Xform

        self.xform.transform_point(self.center)
        self.xform.transform_vector(self.x_axis)
        self.xform.transform_vector(self.y_axis)
        self.xform.transform_vector(self.z_axis)
        self.xform = Xform.identity()

    def transformed(self):
        """Return a transformed copy of the bounding box."""
        import copy

        result = copy.deepcopy(self)
        result.transform()
        return result

    ###########################################################################################
    # Polymorphic JSON Serialization (COMPAS-style)
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field."""
        return {
            "type": f"{self.__class__.__name__}",
            "guid": self.guid,
            "name": self.name,
            "center": self.center.__jsondump__(),
            "x_axis": self.x_axis.__jsondump__(),
            "y_axis": self.y_axis.__jsondump__(),
            "z_axis": self.z_axis.__jsondump__(),
            "half_size": self.half_size.__jsondump__(),
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format."""
        from .encoders import decode_node

        center = decode_node(data["center"])
        x_axis = decode_node(data["x_axis"])
        y_axis = decode_node(data["y_axis"])
        z_axis = decode_node(data["z_axis"])
        half_size = decode_node(data["half_size"])

        bbox = cls(center, x_axis, y_axis, z_axis, half_size)
        bbox.guid = guid
        bbox.name = name

        if "xform" in data:
            bbox.xform = decode_node(data["xform"])

        return bbox

    def json_dumps(self):
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, s):
        import json
        return cls.__jsonload__(json.loads(s))

    def json_dump(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        import json
        with open(filepath, 'r') as f:
            return cls.__jsonload__(json.load(f))

    def pb_dumps(self):
        from .proto import boundingbox_pb2
        proto = boundingbox_pb2.BoundingBox()
        proto.center.ParseFromString(self.center.pb_dumps())
        proto.x_axis.ParseFromString(self.x_axis.pb_dumps())
        proto.y_axis.ParseFromString(self.y_axis.pb_dumps())
        proto.z_axis.ParseFromString(self.z_axis.pb_dumps())
        proto.half_size.ParseFromString(self.half_size.pb_dumps())
        proto.guid = self.guid
        proto.name = self.name
        if hasattr(self, 'xform'):
            proto.xform.guid = self.xform.guid
            proto.xform.name = self.xform.name
            proto.xform.matrix.extend(self.xform.m)
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        from .proto import boundingbox_pb2
        proto = boundingbox_pb2.BoundingBox()
        proto.ParseFromString(data)
        center = Point.pb_loads(proto.center.SerializeToString())
        x_axis = Vector.pb_loads(proto.x_axis.SerializeToString())
        y_axis = Vector.pb_loads(proto.y_axis.SerializeToString())
        z_axis = Vector.pb_loads(proto.z_axis.SerializeToString())
        half_size = Vector.pb_loads(proto.half_size.SerializeToString())
        bbox = cls(center, x_axis, y_axis, z_axis, half_size)
        bbox.guid = proto.guid
        bbox.name = proto.name
        if proto.HasField('xform'):
            from .xform import Xform
            bbox.xform = Xform.pb_loads(proto.xform.SerializeToString())
        return bbox

    def pb_dump(self, filepath):
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath):
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())
