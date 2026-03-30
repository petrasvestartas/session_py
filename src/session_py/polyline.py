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
        self._guid = None
        self.name = "my_polyline"
        self.width = 1.0
        self._linecolor = None
        self._xform = None

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

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

    @property
    def linecolor(self):
        if self._linecolor is None:
            self._linecolor = Color.black()
        return self._linecolor

    @linecolor.setter
    def linecolor(self, value):
        self._linecolor = value

    @property
    def xform(self):
        if getattr(self, '_xform', None) is None:
            self._xform = Xform.identity()
        return self._xform

    @xform.setter
    def xform(self, value):
        self._xform = value

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

    @classmethod
    def from_sides(cls, sides: int, radius: float = 1.0, close: bool = False) -> "Polyline":
        """Create a regular polygon with given number of sides and radius."""
        import math
        pts = []
        for i in range(sides):
            angle = 2.0 * Tolerance.PI * i / sides
            pts.append(Point(radius * math.cos(angle), radius * math.sin(angle), 0.0))
        if close:
            pts.append(pts[0])
        return cls(pts)

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

    def __getitem__(self, index: int) -> Point:
        if 0 <= index < self.point_count():
            idx = index * 3
            return Point(self._coords[idx], self._coords[idx + 1], self._coords[idx + 2])
        raise IndexError("Index out of range")

    def __setitem__(self, index: int, point: Point) -> None:
        if 0 <= index < self.point_count():
            idx = index * 3
            self._coords[idx] = point[0]
            self._coords[idx + 1] = point[1]
            self._coords[idx + 2] = point[2]
        else:
            raise IndexError("Index out of range")

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

    def __imul__(self, factor: float) -> "Polyline":
        """Multiply all coordinates by scalar in place (*=)."""
        for i in range(len(self._coords)):
            self._coords[i] *= factor
        return self

    def __mul__(self, factor: float) -> "Polyline":
        """Multiply polyline by scalar and return new polyline (*)."""
        result = Polyline.from_coords([c * factor for c in self._coords])
        result.name = self.name
        result.width = self.width
        result.linecolor = copy.deepcopy(self.linecolor)
        result.xform = copy.deepcopy(self.xform)
        result.plane = copy.deepcopy(self.plane)
        return result

    def __itruediv__(self, factor: float) -> "Polyline":
        """Divide all coordinates by scalar in place (/=)."""
        for i in range(len(self._coords)):
            self._coords[i] /= factor
        return self

    def __truediv__(self, factor: float) -> "Polyline":
        """Divide polyline by scalar and return new polyline (/)."""
        result = Polyline.from_coords([c / factor for c in self._coords])
        result.name = self.name
        result.width = self.width
        result.linecolor = copy.deepcopy(self.linecolor)
        result.xform = copy.deepcopy(self.xform)
        result.plane = copy.deepcopy(self.plane)
        return result

    def __neg__(self) -> "Polyline":
        """Negate polyline (reverse point order)."""
        return self.reversed()

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
    def point_at(start: Point, end: Point, t: float) -> Point:
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
                Polyline.point_at(line0_start, line0_end, t[1]),
                Polyline.point_at(line0_start, line0_end, t[2]),
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

        output_start = Polyline.point_at(line_start, line_end, t_values[0])
        output_end = Polyline.point_at(line_start, line_end, t_values[-1])

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
            point_on_segment = Polyline.point_at(
                self.points[i], self.points[i + 1], t
            )
            distance = point.distance(point_on_segment)

            if distance < closest_distance:
                closest_distance = distance
                edge_id = i
                best_t = t

            if closest_distance < Tolerance.ZERO_TOLERANCE:
                break

        closest_point = Polyline.point_at(
            self.points[edge_id], self.points[edge_id + 1], best_t
        )
        return closest_distance, edge_id, closest_point

    def is_closed(self) -> bool:
        """Check if polyline is closed (first and last points are the same)."""
        if len(self.points) < 2:
            return False
        return self.points[0].distance(self.points[-1]) < Tolerance.ZERO_TOLERANCE

    def merge_collinear(self, tol: float = Tolerance.APPROXIMATION) -> None:
        """Merge consecutive collinear segments in-place; closed polyline wraps around."""
        closed = self.is_closed()
        pts = self.get_points()
        if closed and len(pts) > 1:
            pts.pop()
        zt2 = Tolerance.ZERO_TOLERANCE ** 2
        changed = True
        while changed:
            changed = False
            m = len(pts)
            if m < 3:
                break
            out = []
            for i in range(m):
                p, nx = (i - 1) % m, (i + 1) % m
                if not closed and (i == 0 or i == m - 1):
                    out.append(pts[i])
                    continue
                ax, ay, az = pts[i][0]-pts[p][0], pts[i][1]-pts[p][1], pts[i][2]-pts[p][2]
                bx, by, bz = pts[nx][0]-pts[i][0], pts[nx][1]-pts[i][1], pts[nx][2]-pts[i][2]
                cx, cy, cz = ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx
                a2, b2 = ax*ax+ay*ay+az*az, bx*bx+by*by+bz*bz
                if a2 < zt2 or b2 < zt2 or cx*cx+cy*cy+cz*cz < tol*tol*a2*b2:
                    changed = True
                else:
                    out.append(pts[i])
            pts = out
        self._coords = []
        for p in pts:
            self._coords.extend([p[0], p[1], p[2]])
        if closed and pts:
            self._coords.extend([pts[0][0], pts[0][1], pts[0][2]])
        if self.point_count() >= 3:
            self.plane = Plane.from_points(self.get_points())

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

    def point_in_polygon_2d(self, p) -> bool:
        """Winding-number point-in-polygon test. p.x/y tested; polygon vertex z ignored."""
        px, py = p[0], p[1]
        coords = self.points
        winding = 0
        n = len(coords)
        for i in range(n):
            j = (i + 1) % n
            y0, y1 = coords[i][1], coords[j][1]
            if y0 <= py:
                if y1 > py:
                    x0, x1 = coords[i][0], coords[j][0]
                    if (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0) > 0:
                        winding += 1
            else:
                if y1 <= py:
                    x0, x1 = coords[i][0], coords[j][0]
                    if (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0) < 0:
                        winding -= 1
        return winding != 0

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

        p0 = self.get_point(segment_id)
        p1 = self.get_point(segment_id + 1)
        v = p1 - p0

        if proportion0 != 0.0 or proportion1 != 0.0:
            p0 -= v * proportion0
            p1 += v * proportion1
        else:
            v_norm = v.normalize()
            p0 -= v_norm * dist0
            p1 += v_norm * dist1

        self.set_point(segment_id, p0)
        self.set_point(segment_id + 1, p1)

        if self.is_closed():
            if segment_id == 0:
                self.set_point(self.point_count() - 1, self.get_point(0))
            elif segment_id + 1 == self.point_count() - 1:
                self.set_point(0, self.get_point(self.point_count() - 1))

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

        start = self.get_point(segment_id)
        end = self.get_point(segment_id + 1)
        self.extend_segment_equally_static(start, end, dist, proportion)
        self.set_point(segment_id, start)
        self.set_point(segment_id + 1, end)

        if self.point_count() > 2 and self.is_closed():
            if segment_id == 0:
                self.set_point(self.point_count() - 1, self.get_point(0))
            elif segment_id + 1 == self.point_count() - 1:
                self.set_point(0, self.get_point(self.point_count() - 1))

    @staticmethod
    def extend_line_static(start: Point, end: Point, d0: float, d1: float) -> None:
        """Extend a line segment independently at each end by a real (normalized) distance."""
        v = end - start
        v_norm = v.normalize()
        start -= v_norm * d0
        end += v_norm * d1

    @staticmethod
    def scale_line_static(start: Point, end: Point, dist: float) -> None:
        """Shrink a line segment equally from both ends by a fraction of its length (not normalized)."""
        v = end - start
        start += v * dist
        end -= v * dist

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

    @staticmethod
    def interpolate_points(
        from_pt: Point, to_pt: Point, steps: int, kind: int = 0
    ) -> List[Point]:
        """Linear interpolation between two points.

        kind: 0=no endpoints, 1=both endpoints, 2=start only
        """
        result = []
        for i in range(1, steps + 1):
            t = float(i) / float(steps + 1)
            result.append(Point(
                from_pt[0] + t * (to_pt[0] - from_pt[0]),
                from_pt[1] + t * (to_pt[1] - from_pt[1]),
                from_pt[2] + t * (to_pt[2] - from_pt[2]),
            ))
        if kind == 1:
            result.insert(0, Point(from_pt[0], from_pt[1], from_pt[2]))
            result.append(Point(to_pt[0], to_pt[1], to_pt[2]))
        elif kind == 2:
            result.insert(0, Point(from_pt[0], from_pt[1], from_pt[2]))
        return result

    @staticmethod
    def quick_hull(polygon: "Polyline") -> "Polyline":
        """2D convex hull via quickhull in the polygon's local plane."""
        pts = polygon.get_points()
        if len(pts) < 3:
            return Polyline(pts[:])

        origin, x_axis, y_axis, _ = polygon.get_average_plane()

        def proj2d(p):
            d = Vector(p[0] - origin[0], p[1] - origin[1], p[2] - origin[2])
            return (d.dot(x_axis), d.dot(y_axis))

        def unproj(u, v):
            return origin + x_axis * u + y_axis * v

        def cross2d(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def qh_upper(a, b, points):
            if not points:
                return []
            apex = max(points, key=lambda p: cross2d(a, b, p))
            if cross2d(a, b, apex) <= 0.0:
                return []
            left = [p for p in points if cross2d(a, apex, p) > 0.0]
            right = [p for p in points if cross2d(apex, b, p) > 0.0]
            return qh_upper(a, apex, left) + [apex] + qh_upper(apex, b, right)

        pts2d = [proj2d(p) for p in pts]
        min_x = min(pts2d, key=lambda p: p[0])
        max_x = max(pts2d, key=lambda p: p[0])

        upper = [p for p in pts2d if cross2d(min_x, max_x, p) > 0.0]
        lower = [p for p in pts2d if cross2d(max_x, min_x, p) > 0.0]

        hull2d = (
            [min_x]
            + qh_upper(min_x, max_x, upper)
            + [max_x]
            + qh_upper(max_x, min_x, lower)
        )

        return Polyline([unproj(u, v) for u, v in hull2d])

    @staticmethod
    def bounding_rectangle(polygon: "Polyline") -> Optional["Polyline"]:
        """Minimum area bounding rectangle via rotating calipers; returns closed 5-point Polyline."""
        import math

        hull = Polyline.quick_hull(polygon)
        hull_pts = hull.get_points()
        n = len(hull_pts)
        if n < 3:
            return None

        origin, x_axis, y_axis, _ = polygon.get_average_plane()

        def proj2d(p):
            d = Vector(p[0] - origin[0], p[1] - origin[1], p[2] - origin[2])
            return (d.dot(x_axis), d.dot(y_axis))

        def unproj(u, v):
            return origin + x_axis * u + y_axis * v

        hull2d = [proj2d(p) for p in hull_pts]
        best_area = float("inf")
        best_corners = None

        for i in range(n):
            ax, ay = hull2d[i]
            bx, by = hull2d[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            length = math.sqrt(ex * ex + ey * ey)
            if length < 1e-12:
                continue
            ex /= length
            ey /= length

            min_u = min_v = float("inf")
            max_u = max_v = float("-inf")
            for px, py in hull2d:
                u = px * ex + py * ey
                v = -px * ey + py * ex
                if u < min_u:
                    min_u = u
                if u > max_u:
                    max_u = u
                if v < min_v:
                    min_v = v
                if v > max_v:
                    max_v = v

            area = (max_u - min_u) * (max_v - min_v)
            if area < best_area:
                best_area = area
                best_corners = [
                    (min_u * ex - min_v * ey, min_u * ey + min_v * ex),
                    (max_u * ex - min_v * ey, max_u * ey + min_v * ex),
                    (max_u * ex - max_v * ey, max_u * ey + max_v * ex),
                    (min_u * ex - max_v * ey, min_u * ey + max_v * ex),
                ]

        if best_corners is None:
            return None

        pts3d = [unproj(u, v) for u, v in best_corners]
        pts3d.append(pts3d[0])
        return Polyline(pts3d)

    @staticmethod
    def grid_of_points_in_polygon(
        polygon: "Polyline",
        offset_dist: float,
        div_dist: float,
        max_pts: int = 100,
    ) -> List[Point]:
        """Grid of interior points; offset_dist is ignored (requires Clipper2)."""
        pts = polygon.get_points()
        if len(pts) < 3:
            return []

        origin, x_axis, y_axis, _ = polygon.get_average_plane()

        def proj2d(p):
            d = Vector(p[0] - origin[0], p[1] - origin[1], p[2] - origin[2])
            return (d.dot(x_axis), d.dot(y_axis))

        def unproj(u, v):
            return origin + x_axis * u + y_axis * v

        poly2d = [proj2d(p) for p in pts]
        nv = len(poly2d)

        def pt_in_poly(pu, pv):
            inside = False
            j = nv - 1
            for i in range(nv):
                xi, yi = poly2d[i]
                xj, yj = poly2d[j]
                if (yi > pv) != (yj > pv):
                    t = (xj - xi) * (pv - yi) / (yj - yi + 1e-300) + xi
                    if pu < t:
                        inside = not inside
                j = i
            return inside

        min_u = min(p[0] for p in poly2d)
        max_u = max(p[0] for p in poly2d)
        min_v = min(p[1] for p in poly2d)
        max_v = max(p[1] for p in poly2d)

        result = []
        u = min_u + div_dist * 0.5
        while u <= max_u and len(result) < max_pts:
            v = min_v + div_dist * 0.5
            while v <= max_v and len(result) < max_pts:
                if pt_in_poly(u, v):
                    result.append(unproj(u, v))
                v += div_dist
            u += div_dist

        return result

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
        # Alphabetical order to match Rust's serde_json
        return {
            "coords": self._coords,
            "guid": self.guid,
            "linecolor": self.linecolor.__jsondump__(),
            "name": self.name,
            "type": f"{self.__class__.__name__}",
            "width": self.width,
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

    def json_dumps(self):
        """Convert to JSON string."""
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, json_string):
        """Load from JSON string."""
        import json
        return cls.__jsonload__(json.loads(json_string))

    ###########################################################################################
    # Protobuf Serialization
    ###########################################################################################

    def pb_dumps(self):
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

    def pb_fill(self, proto):
        """Fill an existing Polyline proto message directly (avoids serialize/deserialize cycle)."""
        proto.guid = self.guid
        proto.name = self.name
        proto.coords.extend(self._coords)
        proto.width = self.width
        proto.linecolor.name = self.linecolor.name
        proto.linecolor.r = self.linecolor[0]
        proto.linecolor.g = self.linecolor[1]
        proto.linecolor.b = self.linecolor[2]
        proto.linecolor.a = self.linecolor[3]
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)

    @classmethod
    def pb_loads(cls, data):
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

    def pb_dump(self, filepath):
        """Write protobuf to file.

        Parameters
        ----------
        filepath : str
            Path to the output file.

        """
        data = self.pb_dumps()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath):
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
        return cls.pb_loads(data)

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

    @staticmethod
    def _simplify_perp_dist(pt, line_start, line_end):
        import math
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        dz = line_end[2] - line_start[2]
        len_sq = dx * dx + dy * dy + dz * dz
        if len_sq == 0.0:
            ex = pt[0] - line_start[0]
            ey = pt[1] - line_start[1]
            ez = pt[2] - line_start[2]
            return math.sqrt(ex * ex + ey * ey + ez * ez)
        t = ((pt[0] - line_start[0]) * dx + (pt[1] - line_start[1]) * dy + (pt[2] - line_start[2]) * dz) / len_sq
        t = max(0.0, min(1.0, t))
        cx = line_start[0] + t * dx
        cy = line_start[1] + t * dy
        cz = line_start[2] + t * dz
        ex = pt[0] - cx
        ey = pt[1] - cy
        ez = pt[2] - cz
        return math.sqrt(ex * ex + ey * ey + ez * ez)

    @staticmethod
    def _simplify_rdp(points, start, end, tolerance, keep):
        if end <= start + 1:
            return
        max_dist = 0.0
        max_idx = start
        for i in range(start + 1, end):
            d = Polyline._simplify_perp_dist(points[i], points[start], points[end])
            if d > max_dist:
                max_dist = d
                max_idx = i
        if max_dist > tolerance:
            keep[max_idx] = True
            Polyline._simplify_rdp(points, start, max_idx, tolerance, keep)
            Polyline._simplify_rdp(points, max_idx, end, tolerance, keep)

    @staticmethod
    def simplify_points(points, tolerance):
        n = len(points)
        if n < 3:
            return list(points)
        keep = [False] * n
        keep[0] = True
        keep[n - 1] = True
        Polyline._simplify_rdp(points, 0, n - 1, tolerance, keep)
        return [points[i] for i in range(n) if keep[i]]

    def simplify(self, tolerance):
        pts = Polyline.simplify_points(self.points, tolerance)
        return Polyline(pts)
