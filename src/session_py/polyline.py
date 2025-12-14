import uuid
import copy
from typing import List, Optional, Tuple, Union

from .color import Color
from .xform import Xform
from .plane import Plane
from .point import Point
from .tolerance import Tolerance
from .vector import Vector


class Polyline:
    """A polyline defined by a collection of coordinates with an associated plane.

    Internally stores coordinates as a flat array [x0, y0, z0, x1, y1, z1, ...] for
    efficient serialization. Provides Point-based API for compatibility.
    """

    def __init__(self, points: Optional[List[Point]] = None):
        """Creates a new Polyline with default guid and name.

        Args:
            points: The collection of points (converted to flat coords internally).
        """
        self.guid = str(uuid.uuid4())
        self.name = "my_polyline"
        self.width = 1.0
        self.linecolor = Color.white()
        self.xform = Xform.identity()

        # Store coordinates as flat array [x0, y0, z0, x1, y1, z1, ...]
        self._coords: List[float] = []
        if points is not None:
            for p in points:
                self._coords.extend([p[0], p[1], p[2]])

        # Delegate plane computation to Plane.from_points
        if self.point_count() >= 3:
            self.plane = Plane.from_points(self.get_points())
        else:
            self.plane = Plane()

    @classmethod
    def from_coords(cls, coords: List[float]) -> "Polyline":
        """Create a Polyline from a flat coordinate array.

        Args:
            coords: Flat array [x0, y0, z0, x1, y1, z1, ...]

        Returns:
            New Polyline instance.
        """
        pl = cls()
        pl._coords = list(coords)
        if pl.point_count() >= 3:
            pl.plane = Plane.from_points(pl.get_points())
        return pl

    ###########################################################################################
    # Point Access (compatibility layer)
    ###########################################################################################

    def point_count(self) -> int:
        """Returns the number of points."""
        return len(self._coords) // 3

    def get_points(self) -> List[Point]:
        """Returns all points as Point objects."""
        points = []
        for i in range(self.point_count()):
            idx = i * 3
            points.append(Point(self._coords[idx], self._coords[idx + 1], self._coords[idx + 2]))
        return points

    @property
    def points(self) -> List[Point]:
        """Property for backward compatibility - returns list of Point objects."""
        return self.get_points()

    @points.setter
    def points(self, value: List[Point]) -> None:
        """Set points from a list of Point objects."""
        self._coords = []
        for p in value:
            self._coords.extend([p[0], p[1], p[2]])

    def __len__(self) -> int:
        """Returns the number of points in the polyline."""
        return self.point_count()

    def is_empty(self) -> bool:
        """Returns true if the polyline has no points."""
        return self.point_count() == 0

    def segment_count(self) -> int:
        """Returns the number of segments (n-1 for n points)."""
        n = self.point_count()
        return n - 1 if n > 1 else 0

    def length(self) -> float:
        """Calculates the total length of the polyline."""
        total_length = 0.0
        for i in range(self.segment_count()):
            idx0 = i * 3
            idx1 = (i + 1) * 3
            dx = self._coords[idx1] - self._coords[idx0]
            dy = self._coords[idx1 + 1] - self._coords[idx0 + 1]
            dz = self._coords[idx1 + 2] - self._coords[idx0 + 2]
            total_length += (dx * dx + dy * dy + dz * dz) ** 0.5
        return total_length

    def get_point(self, index: int) -> Optional[Point]:
        """Returns the point at the given index, or None if out of bounds."""
        if 0 <= index < self.point_count():
            idx = index * 3
            return Point(self._coords[idx], self._coords[idx + 1], self._coords[idx + 2])
        return None

    def set_point(self, index: int, point: Point) -> None:
        """Sets the point at the given index."""
        if 0 <= index < self.point_count():
            idx = index * 3
            self._coords[idx] = point[0]
            self._coords[idx + 1] = point[1]
            self._coords[idx + 2] = point[2]

    def add_point(self, point: Point) -> None:
        """Adds a point to the end of the polyline."""
        self._coords.extend([point[0], point[1], point[2]])
        if self.point_count() == 3:
            self._recompute_plane()

    def insert_point(self, index: int, point: Point) -> None:
        """Inserts a point at the specified index."""
        idx = index * 3
        self._coords[idx:idx] = [point[0], point[1], point[2]]
        if self.point_count() == 3:
            self._recompute_plane()

    def remove_point(self, index: int) -> Optional[Point]:
        """Removes and returns the point at the specified index."""
        if 0 <= index < self.point_count():
            idx = index * 3
            point = Point(self._coords[idx], self._coords[idx + 1], self._coords[idx + 2])
            del self._coords[idx:idx + 3]
            if self.point_count() == 3:
                self._recompute_plane()
            return point
        return None

    def reverse(self) -> None:
        """Reverses the order of points in the polyline."""
        # Reverse coords in groups of 3
        n = self.point_count()
        new_coords = []
        for i in range(n - 1, -1, -1):
            idx = i * 3
            new_coords.extend([self._coords[idx], self._coords[idx + 1], self._coords[idx + 2]])
        self._coords = new_coords
        self.plane.reverse()

    def reversed(self) -> "Polyline":
        """Returns a new polyline with reversed point order."""
        result = Polyline.from_coords(self._coords[:])
        result.guid = self.guid
        result.name = self.name
        result.width = self.width
        result.linecolor = copy.deepcopy(self.linecolor)
        result.xform = copy.deepcopy(self.xform)
        result.plane = copy.deepcopy(self.plane)
        result.reverse()
        return result

    def _recompute_plane(self) -> None:
        """Helper to recompute plane when points change."""
        if self.point_count() >= 3:
            self.plane = Plane.from_points(self.get_points())

    ###########################################################################################
    # Core Methods
    ###########################################################################################

    def duplicate(self) -> "Polyline":
        """Create a deep copy with a new GUID."""
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def __iadd__(self, vector: Vector) -> "Polyline":
        """Translates all points in the polyline by a vector (+=)."""
        for i in range(self.point_count()):
            idx = i * 3
            self._coords[idx] += vector[0]
            self._coords[idx + 1] += vector[1]
            self._coords[idx + 2] += vector[2]
        # Update plane origin
        self.plane = Plane(
            self.plane.origin + vector, self.plane.x_axis, self.plane.y_axis
        )
        return self

    def __add__(self, vector: Vector) -> "Polyline":
        """Translates the polyline by a vector and returns a new polyline (+)."""
        result = Polyline.from_coords(self._coords[:])
        result.guid = self.guid
        result.name = self.name
        result.width = self.width
        result.linecolor = copy.deepcopy(self.linecolor)
        result.xform = copy.deepcopy(self.xform)
        result.plane = copy.deepcopy(self.plane)
        result += vector
        return result

    def __isub__(self, vector: Vector) -> "Polyline":
        """Translates all points by the negative of a vector (-=)."""
        for i in range(self.point_count()):
            idx = i * 3
            self._coords[idx] -= vector[0]
            self._coords[idx + 1] -= vector[1]
            self._coords[idx + 2] -= vector[2]
        # Update plane origin
        self.plane = Plane(
            self.plane.origin - vector, self.plane.x_axis, self.plane.y_axis
        )
        return self

    def __sub__(self, vector: Vector) -> "Polyline":
        """Translates the polyline by the negative of a vector and returns a new polyline (-)."""
        result = Polyline.from_coords(self._coords[:])
        result.guid = self.guid
        result.name = self.name
        result.width = self.width
        result.linecolor = copy.deepcopy(self.linecolor)
        result.xform = copy.deepcopy(self.xform)
        result.plane = copy.deepcopy(self.plane)
        result -= vector
        return result

    def transform(self):
        """Apply the stored xform transformation to the polyline.

        Transforms all points in-place and resets xform to identity.
        """
        for i in range(self.point_count()):
            idx = i * 3
            pt = Point(self._coords[idx], self._coords[idx + 1], self._coords[idx + 2])
            self.xform.transform_point(pt)
            self._coords[idx] = pt[0]
            self._coords[idx + 1] = pt[1]
            self._coords[idx + 2] = pt[2]
        self.xform = Xform.identity()

    def transformed(self):
        """Return a transformed copy of the polyline."""
        result = copy.deepcopy(self)
        result.transform()
        return result

    # ===========================================================================================
    # Geometric Utilities
    # ===========================================================================================

    def shift(self, times: int) -> None:
        """Shift polyline points by specified number of positions."""
        if not self.points:
            return
        n = len(self.points)
        shift_amount = times % n
        self.points = self.points[shift_amount:] + self.points[:shift_amount]

    def magnitude_squared(self) -> float:
        """Calculate squared magnitude of polyline (faster, no sqrt)."""
        mag = 0.0
        for i in range(self.segment_count()):
            segment = self.points[i + 1] - self.points[i]
            mag += segment.magnitude_squared()
        return mag

    @staticmethod
    def point_at_parameter(start: Point, end: Point, t: float) -> Point:
        """Get point at parameter t along a line segment (t=0 is start, t=1 is end)."""
        s = 1.0 - t
        return Point(
            start.x if start.x == end.x else s * start.x + t * end.x,
            start.y if start.y == end.y else s * start.y + t * end.y,
            start.z if start.z == end.z else s * start.z + t * end.z,
        )

    @staticmethod
    def closest_point_to_line(
        point: Point, line_start: Point, line_end: Point
    ) -> float:
        """Find closest point on line segment to given point, returns parameter t."""
        d = line_end - line_start
        dod = d.magnitude_squared()

        if dod > 0.0:
            if (point - line_start).magnitude_squared() <= (
                point - line_end
            ).magnitude_squared():
                t = (point - line_start).dot(d) / dod
            else:
                t = 1.0 + (point - line_end).dot(d) / dod
            return t
        else:
            return 0.0

    @staticmethod
    def line_line_overlap(
        line0_start: Point,
        line0_end: Point,
        line1_start: Point,
        line1_end: Point,
    ) -> Optional[Tuple[Point, Point]]:
        """Check if two line segments overlap and return the overlapping segment."""
        t = [0.0, 1.0, 0.0, 0.0]
        t[2] = Polyline.closest_point_to_line(line1_start, line0_start, line0_end)
        t[3] = Polyline.closest_point_to_line(line1_end, line0_start, line0_end)

        do_overlap = not ((t[2] < 0.0 and t[3] < 0.0) or (t[2] > 1.0 and t[3] > 1.0))
        t.sort()

        overlap_valid = abs(t[2] - t[1]) > Tolerance.ZERO_TOLERANCE

        if do_overlap and overlap_valid:
            return (
                Polyline.point_at_parameter(line0_start, line0_end, t[1]),
                Polyline.point_at_parameter(line0_start, line0_end, t[2]),
            )
        else:
            return None

    @staticmethod
    def line_line_average(
        line0_start: Point,
        line0_end: Point,
        line1_start: Point,
        line1_end: Point,
    ) -> Tuple[Point, Point]:
        """Calculate average of two line segments."""
        output_start = Point(
            (line0_start.x + line1_start.x) * 0.5,
            (line0_start.y + line1_start.y) * 0.5,
            (line0_start.z + line1_start.z) * 0.5,
        )
        output_end = Point(
            (line0_end.x + line1_end.x) * 0.5,
            (line0_end.y + line1_end.y) * 0.5,
            (line0_end.z + line1_end.z) * 0.5,
        )
        return output_start, output_end

    @staticmethod
    def line_line_overlap_average(
        line0_start: Point,
        line0_end: Point,
        line1_start: Point,
        line1_end: Point,
    ) -> Tuple[Point, Point]:
        """Calculate overlap average of two line segments."""
        line_a = Polyline.line_line_overlap(
            line0_start, line0_end, line1_start, line1_end
        )
        line_b = Polyline.line_line_overlap(
            line1_start, line1_end, line0_start, line0_end
        )

        if line_a and line_b:
            line_a_start, line_a_end = line_a
            line_b_start, line_b_end = line_b

            mid_line0_start = Point(
                (line_a_start.x + line_b_start.x) * 0.5,
                (line_a_start.y + line_b_start.y) * 0.5,
                (line_a_start.z + line_b_start.z) * 0.5,
            )
            mid_line0_end = Point(
                (line_a_end.x + line_b_end.x) * 0.5,
                (line_a_end.y + line_b_end.y) * 0.5,
                (line_a_end.z + line_b_end.z) * 0.5,
            )
            mid_line1_start = Point(
                (line_a_start.x + line_b_end.x) * 0.5,
                (line_a_start.y + line_b_end.y) * 0.5,
                (line_a_start.z + line_b_end.z) * 0.5,
            )
            mid_line1_end = Point(
                (line_a_end.x + line_b_start.x) * 0.5,
                (line_a_end.y + line_b_start.y) * 0.5,
                (line_a_end.z + line_b_start.z) * 0.5,
            )

            mid0_vec = mid_line0_end - mid_line0_start
            mid1_vec = mid_line1_end - mid_line1_start

            if mid0_vec.magnitude_squared() > mid1_vec.magnitude_squared():
                return mid_line0_start, mid_line0_end
            else:
                return mid_line1_start, mid_line1_end
        else:
            return Polyline.line_line_average(
                line0_start, line0_end, line1_start, line1_end
            )

    @staticmethod
    def line_from_projected_points(
        line_start: Point,
        line_end: Point,
        points: List[Point],
    ) -> Optional[Tuple[Point, Point]]:
        """Create line from projected points onto a base line."""
        if not points:
            return None

        t_values = [
            Polyline.closest_point_to_line(p, line_start, line_end) for p in points
        ]
        t_values.sort()

        output_start = Polyline.point_at_parameter(line_start, line_end, t_values[0])
        output_end = Polyline.point_at_parameter(line_start, line_end, t_values[-1])

        if abs(t_values[0] - t_values[-1]) > Tolerance.ZERO_TOLERANCE:
            return output_start, output_end
        else:
            return None

    def closest_distance_and_point(self, point: Point) -> Tuple[float, int, Point]:
        """Find closest distance and point from a point to this polyline."""
        edge_id = 0
        closest_distance = float("inf")
        best_t = 0.0

        for i in range(self.segment_count()):
            t = self.closest_point_to_line(point, self.points[i], self.points[i + 1])
            point_on_segment = self.point_at_parameter(
                self.points[i], self.points[i + 1], t
            )
            distance = point.distance(point_on_segment)

            if distance < closest_distance:
                closest_distance = distance
                edge_id = i
                best_t = t

            if closest_distance < Tolerance.ZERO_TOLERANCE:
                break

        closest_point = self.point_at_parameter(
            self.points[edge_id], self.points[edge_id + 1], best_t
        )
        return closest_distance, edge_id, closest_point

    def is_closed(self) -> bool:
        """Check if polyline is closed (first and last points are the same)."""
        if len(self.points) < 2:
            return False
        return self.points[0].distance(self.points[-1]) < Tolerance.ZERO_TOLERANCE

    def center(self) -> Point:
        """Calculate center point of polyline."""
        if not self.points:
            return Point(0.0, 0.0, 0.0)

        n = (
            len(self.points) - 1
            if self.is_closed() and len(self.points) > 1
            else len(self.points)
        )

        sum_x = sum(self.points[i].x for i in range(n))
        sum_y = sum(self.points[i].y for i in range(n))
        sum_z = sum(self.points[i].z for i in range(n))

        return Point(sum_x / n, sum_y / n, sum_z / n)

    def center_vec(self) -> Vector:
        """Calculate center as vector."""
        center = self.center()
        return Vector(center.x, center.y, center.z)

    def get_average_plane(self) -> Tuple[Point, Vector, Vector, Vector]:
        """Get average plane from polyline points."""
        origin = self.center()

        if len(self.points) >= 2:
            x_axis = (self.points[1] - self.points[0]).normalize()
        else:
            x_axis = Vector(1.0, 0.0, 0.0)

        z_axis = self._average_normal()
        y_axis = z_axis.cross(x_axis).normalize()

        return origin, x_axis, y_axis, z_axis

    def get_fast_plane(self) -> Tuple[Point, Plane]:
        """Get fast plane calculation from polyline."""
        origin = self.points[0] if self.points else Point(0.0, 0.0, 0.0)
        average_normal = self._average_normal()
        plane = Plane.from_point_normal(origin, average_normal)
        return origin, plane

    @staticmethod
    def get_middle_line(
        line0_start: Point,
        line0_end: Point,
        line1_start: Point,
        line1_end: Point,
    ) -> Tuple[Point, Point]:
        """Calculate middle line between two line segments."""
        p0 = Point(
            (line0_start.x + line1_start.x) * 0.5,
            (line0_start.y + line1_start.y) * 0.5,
            (line0_start.z + line1_start.z) * 0.5,
        )
        p1 = Point(
            (line0_end.x + line1_end.x) * 0.5,
            (line0_end.y + line1_end.y) * 0.5,
            (line0_end.z + line1_end.z) * 0.5,
        )
        return p0, p1

    @staticmethod
    def extend_line(
        line_start: Point, line_end: Point, distance0: float, distance1: float
    ) -> None:
        """Extend line segment by specified distances at both ends."""
        v = (line_end - line_start).normalize()
        line_start -= v * distance0
        line_end += v * distance1

    @staticmethod
    def scale_line(line_start: Point, line_end: Point, distance: float) -> None:
        """Scale line segment inward by specified distance."""
        v = line_end - line_start
        line_start += v * distance
        line_end -= v * distance

    def extend_segment(
        self,
        segment_id: int,
        dist0: float,
        dist1: float,
        proportion0: float = 0.0,
        proportion1: float = 0.0,
    ) -> None:
        """Extend polyline segment."""
        if segment_id < 0 or segment_id >= self.segment_count():
            return

        p0 = self.points[segment_id]
        p1 = self.points[segment_id + 1]
        v = p1 - p0

        if proportion0 != 0.0 or proportion1 != 0.0:
            p0 -= v * proportion0
            p1 += v * proportion1
        else:
            v_norm = v.normalize()
            p0 -= v_norm * dist0
            p1 += v_norm * dist1

        self.points[segment_id] = p0
        self.points[segment_id + 1] = p1

        if self.is_closed():
            if segment_id == 0:
                self.points[-1] = self.points[0]
            elif segment_id + 1 == len(self.points) - 1:
                self.points[0] = self.points[-1]

    @staticmethod
    def extend_segment_equally_static(
        segment_start: Point, segment_end: Point, dist: float, proportion: float = 0.0
    ) -> None:
        """Extend segment equally on both ends (static utility)."""
        if dist == 0.0 and proportion == 0.0:
            return

        v = segment_end - segment_start

        if proportion != 0.0:
            segment_start -= v * proportion
            segment_end += v * proportion
        else:
            v_norm = v.normalize()
            segment_start -= v_norm * dist
            segment_end += v_norm * dist

    def extend_segment_equally(
        self, segment_id: int, dist: float, proportion: float = 0.0
    ) -> None:
        """Extend polyline segment equally."""
        if segment_id < 0 or segment_id >= self.segment_count():
            return

        start = self.points[segment_id]
        end = self.points[segment_id + 1]
        self.extend_segment_equally_static(start, end, dist, proportion)
        self.points[segment_id] = start
        self.points[segment_id + 1] = end

        if len(self.points) > 2 and self.is_closed():
            if segment_id == 0:
                self.points[-1] = self.points[0]
            elif segment_id + 1 == len(self.points) - 1:
                self.points[0] = self.points[-1]

    def move_by(self, direction: Vector) -> None:
        """Move polyline by direction vector."""
        for point in self.points:
            point += direction

    def is_clockwise(self, plane: Plane) -> bool:
        """Check if polyline is clockwise oriented."""
        if len(self.points) < 3:
            return False

        sum_val = 0.0
        n = len(self.points) - 1 if self.is_closed() else len(self.points)

        for i in range(n):
            current = self.points[i]
            next_pt = self.points[(i + 1) % n]
            sum_val += (next_pt.x - current.x) * (next_pt.y + current.y)

        return sum_val > 0.0

    def flip(self) -> None:
        """Flip polyline direction (reverse point order)."""
        self.points.reverse()

    def get_convex_corners(self) -> List[bool]:
        """Get convex/concave corners of polyline."""
        if len(self.points) < 3:
            return []

        closed = self.is_closed()
        normal = self._average_normal()
        n = len(self.points) - 1 if closed else len(self.points)
        convex_corners = []

        for current in range(n):
            prev = n - 1 if current == 0 else current - 1
            next_pt = 0 if current == n - 1 else current + 1

            dir0 = (self.points[current] - self.points[prev]).normalize()
            dir1 = (self.points[next_pt] - self.points[current]).normalize()

            cross = dir0.cross(dir1).normalize()
            dot = cross.dot(normal)
            is_convex = not (dot < 0.0)
            convex_corners.append(is_convex)

        return convex_corners

    @staticmethod
    def tween_two_polylines(
        polyline0: "Polyline", polyline1: "Polyline", weight: float
    ) -> "Polyline":
        """Interpolate between two polylines."""
        if len(polyline0.points) != len(polyline1.points):
            return Polyline(polyline0.points[:])

        result_points = []
        for i in range(len(polyline0.points)):
            diff = polyline1.points[i] - polyline0.points[i]
            interpolated = polyline0.points[i] + diff * weight
            result_points.append(interpolated)

        return Polyline(result_points)

    def _average_normal(self) -> Vector:
        """Calculate average normal from polyline points."""
        if len(self.points) < 3:
            return Vector(0.0, 0.0, 1.0)

        closed = self.is_closed()
        n = (
            len(self.points) - 1
            if closed and len(self.points) > 1
            else len(self.points)
        )

        average_normal = Vector(0.0, 0.0, 0.0)

        for i in range(n):
            prev = n - 1 if i == 0 else i - 1
            next_pt = (i + 1) % n

            v1 = self.points[prev] - self.points[i]
            v2 = self.points[i] - self.points[next_pt]
            cross = v1.cross(v2)
            average_normal += cross

        return average_normal.normalize()

    ###########################################################################################
    # Polymorphic JSON Serialization
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field.

        Uses compact coords array format: [x0, y0, z0, x1, y1, z1, ...]

        Returns
        -------
        dict
            Dictionary with 'type', 'guid', 'name', and object fields.

        """
        return {
            "type": f"{self.__class__.__name__}",
            "guid": self.guid,
            "name": self.name,
            "coords": self._coords,
            "width": self.width,
            "linecolor": self.linecolor.__jsondump__(),
            "xform": self.xform.__jsondump__(),
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format.

        Supports both compact coords format and legacy points format.

        Parameters
        ----------
        data : dict
            Dictionary containing polyline data.
        guid : str, optional
            GUID for the polyline.
        name : str, optional
            Name for the polyline.

        Returns
        -------
        :class:`Polyline`
            Reconstructed polyline instance.

        """
        from .encoders import decode_node

        # Support both new coords format and legacy points format
        if "coords" in data:
            polyline = cls.from_coords(data["coords"])
        else:
            # Legacy format with full Point objects
            points = [decode_node(p) for p in data["points"]]
            polyline = cls(points)

        polyline.guid = guid if guid is not None else data.get("guid", polyline.guid)
        polyline.name = name if name is not None else data.get("name", polyline.name)

        if "width" in data:
            polyline.width = data["width"]
        if "linecolor" in data:
            polyline.linecolor = decode_node(data["linecolor"])
        if "xform" in data:
            polyline.xform = decode_node(data["xform"])

        return polyline

    def json_dump(self, filepath):
        """Write JSON to file.

        Parameters
        ----------
        filepath : str or Path
            Path to the output file.

        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        """Read JSON from file.

        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file.

        Returns
        -------
        :class:`Polyline`
            The deserialized Polyline.

        """
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    ###########################################################################################
    # Protobuf Serialization
    ###########################################################################################

    def to_protobuf(self):
        """Convert to protobuf binary format.

        Returns
        -------
        bytes
            Serialized protobuf data.

        """
        from .proto import polyline_pb2

        proto = polyline_pb2.Polyline()
        proto.guid = self.guid
        proto.name = self.name
        proto.coords.extend(self._coords)
        proto.width = self.width

        # Set linecolor
        proto.linecolor.name = self.linecolor.name
        proto.linecolor.r = self.linecolor[0]
        proto.linecolor.g = self.linecolor[1]
        proto.linecolor.b = self.linecolor[2]
        proto.linecolor.a = self.linecolor[3]

        # Set xform
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)

        return proto.SerializeToString()

    @classmethod
    def from_protobuf(cls, data):
        """Create Polyline from protobuf binary data.

        Parameters
        ----------
        data : bytes
            Protobuf-encoded polyline data.

        Returns
        -------
        :class:`Polyline`
            The deserialized Polyline.

        """
        from .proto import polyline_pb2

        proto = polyline_pb2.Polyline()
        proto.ParseFromString(data)

        polyline = cls.from_coords(list(proto.coords))
        polyline.guid = proto.guid
        polyline.name = proto.name
        polyline.width = proto.width

        # Load linecolor
        polyline.linecolor = Color(
            proto.linecolor.r,
            proto.linecolor.g,
            proto.linecolor.b,
            proto.linecolor.a
        )
        polyline.linecolor.name = proto.linecolor.name

        # Load xform
        polyline.xform = Xform()
        polyline.xform.name = proto.xform.name
        polyline.xform.m = list(proto.xform.matrix)

        return polyline

    def protobuf_dump(self, filepath):
        """Write protobuf to file.

        Parameters
        ----------
        filepath : str
            Path to the output file.

        """
        data = self.to_protobuf()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def protobuf_load(cls, filepath):
        """Read protobuf from file.

        Parameters
        ----------
        filepath : str
            Path to the protobuf file.

        Returns
        -------
        :class:`Polyline`
            The deserialized Polyline.

        """
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.from_protobuf(data)

    def __str__(self) -> str:
        """Returns a minimal string representation of the polyline."""
        pts = []
        for i in range(self.point_count()):
            idx = i * 3
            pts.append(f"({self._coords[idx]}, {self._coords[idx + 1]}, {self._coords[idx + 2]})")
        return "[" + ", ".join(pts) + "]"

    def __repr__(self) -> str:
        """Returns a detailed string representation."""
        return f"Polyline({self.name}, {self.point_count()} points)"

    def __eq__(self, other) -> bool:
        """Compare polylines by value (ignoring GUIDs)."""
        if not isinstance(other, Polyline):
            return False
        if self.name != other.name:
            return False
        if self.point_count() != other.point_count():
            return False
        for i in range(len(self._coords)):
            if round(self._coords[i], Tolerance.ROUNDING) != round(other._coords[i], Tolerance.ROUNDING):
                return False
        if round(self.width, Tolerance.ROUNDING) != round(other.width, Tolerance.ROUNDING):
            return False
        if self.linecolor != other.linecolor:
            return False
        return True

    def __ne__(self, other) -> bool:
        return not self == other
