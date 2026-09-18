from __future__ import annotations
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
import copy
import heapq
import json
import math
import uuid
from .color import Color
from .plane import Plane
from .point import Point
from .tolerance import Tolerance
from .vector import Vector

if TYPE_CHECKING:
    from pathlib import Path
    from .line import Line
    from .proto import polyline_pb2
    from .xform import Xform


# ═══════════════════════════════════════════════════════════════════════════
# 2D helpers
# ═══════════════════════════════════════════════════════════════════════════
def _ccw_2d(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    """Return the cross product sign of (b - a) x (p - a)."""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _seg_dist_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Return the squared distance from (px, py) to the segment (a, b)."""

    x = ax
    y = ay
    dx = bx - x
    dy = by - y

    if dx != 0.0 or dy != 0.0:
        t = ((px - x) * dx + (py - y) * dy) / (dx * dx + dy * dy)

        if t > 1.0:
            x = bx
            y = by
        elif t > 0.0:
            x += dx * t
            y += dy * t

    dx = px - x
    dy = py - y

    return dx * dx + dy * dy


def _point_to_polygon_dist(
    px: float, py: float, polygon: list[list[tuple[float, float]]]
) -> float:
    """Return the signed distance to the polygon rings, positive inside."""

    inside = False
    min_dist_sq = float("inf")

    for ring in polygon:
        length = len(ring)
        j = length - 1

        for i in range(length):
            ax = ring[i][0]
            ay = ring[i][1]
            bx = ring[j][0]
            by = ring[j][1]

            if (ay > py) != (by > py) and px < (bx - ax) * (py - ay) / (by - ay) + ax:
                inside = not inside

            min_dist_sq = min(min_dist_sq, _seg_dist_sq(px, py, ax, ay, bx, by))
            j = i

    return (1.0 if inside else -1.0) * math.sqrt(min_dist_sq)


class _PCell:
    """Quadtree cell of the polylabel search: center, half size, distance and its upper bound."""

    def __init__(
        self, cx: float, cy: float, h: float, polygon: list[list[tuple[float, float]]]
    ):
        """Construct the cell at (cx, cy) with half size h against polygon."""

        self.cx = cx
        self.cy = cy
        self.h = h
        self.d = _point_to_polygon_dist(cx, cy, polygon)
        self.mx = self.d + h * math.sqrt(2.0)

    def __lt__(self, o: "_PCell") -> bool:
        """Order by the upper bound so the heap pops the most promising cell."""
        return self.mx > o.mx


def _centroid_cell(polygon: list[list[tuple[float, float]]]) -> _PCell:
    """Return the cell at the centroid of the outer ring."""

    area = 0.0
    cx = 0.0
    cy = 0.0
    ring = polygon[0]
    length = len(ring)
    j = length - 1

    for i in range(length):
        ax = ring[i][0]
        ay = ring[i][1]
        bx = ring[j][0]
        by = ring[j][1]
        f = ax * by - bx * ay
        cx += (ax + bx) * f
        cy += (ay + by) * f
        area += f * 3.0
        j = i

    if area == 0.0:
        return _PCell(ring[0][0], ring[0][1], 0.0, polygon)

    return _PCell(cx / area, cy / area, 0.0, polygon)


def _mapbox_polylabel(
    polygon: list[list[tuple[float, float]]], precision: float
) -> tuple[float, float, float]:
    """Return the center and radius of the largest inscribed circle in 2D (Mapbox polylabel)."""

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for p in polygon[0]:
        min_x = min(min_x, p[0])
        max_x = max(max_x, p[0])
        min_y = min(min_y, p[1])
        max_y = max(max_y, p[1])

    size_x = max_x - min_x
    size_y = max_y - min_y
    cell_size = min(size_x, size_y)
    h = cell_size / 2.0

    if cell_size == 0.0:
        return (min_x, min_y, 0.0)

    queue = []
    x = min_x

    while x < max_x:
        y = min_y

        while y < max_y:
            heapq.heappush(queue, _PCell(x + h, y + h, h, polygon))
            y += cell_size

        x += cell_size

    best = _centroid_cell(polygon)
    bbox_cell = _PCell(min_x + size_x / 2.0, min_y + size_y / 2.0, 0.0, polygon)

    if bbox_cell.d > best.d:
        best = bbox_cell

    max_iter = 1000000

    for _ in range(max_iter):
        if not queue:
            break

        cell = heapq.heappop(queue)

        if cell.d > best.d:
            best = cell

        if cell.mx - best.d <= precision:
            continue

        nh = cell.h / 2.0
        heapq.heappush(queue, _PCell(cell.cx - nh, cell.cy - nh, nh, polygon))
        heapq.heappush(queue, _PCell(cell.cx + nh, cell.cy - nh, nh, polygon))
        heapq.heappush(queue, _PCell(cell.cx - nh, cell.cy + nh, nh, polygon))
        heapq.heappush(queue, _PCell(cell.cx + nh, cell.cy + nh, nh, polygon))

    return (best.cx, best.cy, best.d)


class Polyline:
    """A polyline stored as flat coordinates [x0, y0, z0, x1, y1, z1, ...] with a lazily computed plane."""

    def __init__(self, points: list[Point] | None = None):
        """Construct from points."""

        self._guid = None
        self.name = "my_polyline"
        self.coords: list[float] = []
        self.plane = Plane()
        self._plane_dirty = True
        self.width = 1.0
        self.dash = []
        self.linecolor = Color.black()

        if points is not None:
            for p in points:
                self.coords.extend([p[0], p[1], p[2]])

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = Polyline()
        result.name = self.name
        result.coords = list(self.coords)
        result.plane = copy.deepcopy(self.plane, memo)
        result._plane_dirty = self._plane_dirty
        result.width = self.width
        result.dash = list(self.dash)
        result.linecolor = copy.deepcopy(self.linecolor, memo)
        memo[id(self)] = result

        return result

    def duplicate(self) -> "Polyline":
        """Copy (new guid, same data)"""
        return copy.deepcopy(self)

    def has_guid(self) -> bool:
        """Return whether the lazy guid has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the guid, creating it on first access."""
        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        """Set the guid."""
        self._guid = value

    def refresh_guid(self) -> None:
        """Clear the guid so a fresh one mints lazily on the next read."""
        self._guid = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def from_coords(coords: list[float]) -> "Polyline":
        """Construct from flat [x, y, z, ...] coordinates."""

        pl = Polyline()
        pl.coords = list(coords)
        pl._recompute_plane_if_needed()

        return pl

    @staticmethod
    def from_sides(sides: int, radius: float = 1.0, close: bool = False) -> "Polyline":
        """Construct a regular polygon of sides around the origin in the XY plane."""

        pts = []

        for i in range(sides):
            angle = 2.0 * Tolerance.PI * i / sides
            pts.append(Point(radius * math.cos(angle), radius * math.sin(angle), 0.0))

        if close:
            pts.append(pts[0])

        return Polyline(pts)

    @staticmethod
    def rectangle(
        origin: Point,
        x_axis: Vector,
        y_axis: Vector,
        width: float,
        height: float,
        close: bool = True,
    ) -> "Polyline":
        """Construct a rectangle with its corner at origin, sides along x_axis and y_axis."""

        plane = Plane(origin, x_axis, y_axis)
        o = plane.origin
        x = plane.x_axis * width
        y = plane.y_axis * height
        pts = [o, o + x, o + x + y, o + y]

        if close:
            pts.append(pts[0])

        return Polyline(pts)

    @staticmethod
    def quadratic_points(
        p0: Point, p1: Point, p2: Point, divisions: int = 7
    ) -> "Polyline":
        """Construct the quadratic Bezier through p0, p1, p2 sampled at divisions points."""

        n = max(divisions, 2)
        d = float(n - 1)
        pts = []

        for k in range(n):
            t = k / d
            s = 1.0 - t
            s2 = s * s
            ts = 2.0 * s * t
            t2 = t * t
            pts.append(
                Point(
                    s2 * p0[0] + ts * p1[0] + t2 * p2[0],
                    s2 * p0[1] + ts * p1[1] + t2 * p2[1],
                    s2 * p0[2] + ts * p1[2] + t2 * p2[2],
                )
            )
        return Polyline(pts)

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════

    def point_count(self) -> int:
        """Return the number of points."""
        return len(self.coords) // 3

    def __len__(self) -> int:
        """Return the number of points."""
        return self.point_count()

    def is_empty(self) -> bool:
        """Return whether the polyline has no points."""
        return len(self.coords) == 0

    def segment_count(self) -> int:
        """Return the number of segments."""
        n = self.point_count()

        return n - 1 if n > 1 else 0

    def get_point(self, index: int) -> Point | None:
        """Return the point at index, or the origin when out of range."""

        if index < 0 or index >= self.point_count():
            return None

        idx = index * 3

        return Point(self.coords[idx], self.coords[idx + 1], self.coords[idx + 2])

    def get_points(self) -> list[Point]:
        """Return all points."""

        points = []

        for i in range(self.point_count()):
            points.append(
                Point(
                    self.coords[i * 3], self.coords[i * 3 + 1], self.coords[i * 3 + 2]
                )
            )
        return points

    def get_lines(self) -> list["Line"]:
        """Return one line per segment."""

        from .line import Line

        lines = []

        for i in range(self.segment_count()):
            idx0 = i * 3
            idx1 = (i + 1) * 3
            lines.append(
                Line(
                    self.coords[idx0],
                    self.coords[idx0 + 1],
                    self.coords[idx0 + 2],
                    self.coords[idx1],
                    self.coords[idx1 + 1],
                    self.coords[idx1 + 2],
                )
            )
        return lines

    def get_plane(self) -> Plane:
        """Return the plane from the first non-collinear triple, computed on first access."""

        if not self._plane_dirty or self.point_count() < 3:
            return self.plane

        n = self.point_count()
        p0 = self.get_point(0)
        found = False

        for i in range(1, n):
            if found:
                break

            v1 = self.get_point(i) - p0

            if v1.magnitude_squared() < 1e-20:
                continue

            for j in range(i + 1, n):
                if found:
                    break

                normal = v1.cross(self.get_point(j) - p0)

                if normal.magnitude_squared() < 1e-20:
                    continue

                normal.normalize_self()
                v1.normalize_self()
                yax = normal.cross(v1)
                yax.normalize_self()
                self.plane = Plane.from_frame(p0, v1, yax, normal)
                found = True

        if not found:
            self.plane = Plane()

        self._plane_dirty = False

        return self.plane

    def length(self) -> float:
        """Return the total length."""

        total = 0.0

        for i in range(self.segment_count()):
            total += math.sqrt(
                (self.get_point(i + 1) - self.get_point(i)).magnitude_squared()
            )
        return total

    def length_squared(self) -> float:
        """Return the sum of squared segment lengths."""

        total = 0.0

        for i in range(self.segment_count()):
            total += (self.get_point(i + 1) - self.get_point(i)).magnitude_squared()

        return total

    def is_closed(self) -> bool:
        """Return whether the first and last points coincide."""

        if self.point_count() < 2:
            return False

        return (
            self.get_point(0).distance(self.get_point(self.point_count() - 1))
            < Tolerance.ZERO_TOLERANCE
        )

    def closed(self) -> "Polyline":
        """Return a copy with the first point appended when open."""

        if self.is_closed():
            return Polyline.from_coords(self.coords)

        coords = list(self.coords)
        coords.append(self.coords[0])
        coords.append(self.coords[1])
        coords.append(self.coords[2])

        return Polyline.from_coords(coords)

    def center(self) -> Point:
        """Return the average of the points, closing duplicate excluded."""

        if not self.coords:
            return Point(0.0, 0.0, 0.0)

        n = self.point_count() - 1 if self.is_closed() else self.point_count()
        x = 0.0
        y = 0.0
        z = 0.0

        for i in range(n):
            x += self.coords[i * 3]
            y += self.coords[i * 3 + 1]
            z += self.coords[i * 3 + 2]

        return Point(x / n, y / n, z / n)

    def get_average_plane(self) -> tuple[Point, Vector, Vector, Vector]:
        """Compute the frame with origin at center, x along the first segment, z the average normal."""

        origin = self.center()
        x_axis = (
            self.get_point(1) - self.get_point(0)
            if self.point_count() >= 2
            else Vector(1.0, 0.0, 0.0)
        )
        x_axis.normalize_self()
        z_axis = self._average_normal()
        y_axis = z_axis.cross(x_axis)
        y_axis.normalize_self()

        return (origin, x_axis, y_axis, z_axis)

    def get_fast_plane(self) -> tuple[Point, Plane]:
        """Compute the plane with origin at the first point, normal the average normal."""

        if not self.coords:
            return (Point(0.0, 0.0, 0.0), Plane())

        origin = self.get_point(0)
        normal = self._average_normal()

        return (origin, Plane.from_point_normal(origin, normal))

    def get_convex_corners(self) -> list[bool]:
        """Compute one flag per corner, true when convex against the average normal."""

        if self.point_count() < 3:
            return []

        n = self.point_count() - 1 if self.is_closed() else self.point_count()
        normal = self._average_normal()
        convex_or_concave = []

        for current in range(n):
            prev = n - 1 if current == 0 else current - 1
            next = 0 if current == n - 1 else current + 1
            dir0 = self.get_point(current) - self.get_point(prev)
            dir0.normalize_self()
            dir1 = self.get_point(next) - self.get_point(current)
            dir1.normalize_self()
            cross = dir0.cross(dir1)
            cross.normalize_self()
            convex_or_concave.append(cross.dot(normal) >= 0.0)

        return convex_or_concave

    def is_clockwise(self, pln: Plane) -> bool:
        """Return the shoelace sign of the points projected onto pln."""

        n = self.point_count()

        if n < 3:
            return False

        xv = pln.x_axis
        yv = pln.y_axis
        orig = pln.origin
        lim = n - 1 if self.is_closed() else n
        area = 0.0

        for i in range(lim):
            d0 = self.get_point(i) - orig
            d1 = self.get_point((i + 1) % lim) - orig
            u0 = d0.dot(xv)
            v0 = d0.dot(yv)
            u1 = d1.dot(xv)
            v1 = d1.dot(yv)
            area += (u1 - u0) * (v1 + v0)

        return area > 0

    def point_in_polygon_2d(self, p: Point) -> bool:
        """Return the winding-number test on x and y."""

        px = p[0]
        py = p[1]
        n = self.point_count()
        winding = 0

        for i in range(n):
            j = (i + 1) % n
            x0 = self.coords[i * 3]
            y0 = self.coords[i * 3 + 1]
            x1 = self.coords[j * 3]
            y1 = self.coords[j * 3 + 1]
            side = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)

            if y0 <= py and y1 > py and side > 0.0:
                winding += 1
            elif y0 > py and y1 <= py and side < 0.0:
                winding -= 1

        return winding != 0

    def closest_distance_and_point(self, point: Point) -> tuple[float, int, Point]:
        """Return the distance to the nearest segment, with its index and the closest point."""

        edge_id = 0
        closest_distance = float("inf")
        best_t = 0.0

        for i in range(self.segment_count()):
            t = Polyline.closest_point_to_line(
                point, self.get_point(i), self.get_point(i + 1)
            )
            distance = point.distance(
                Polyline.point_at(self.get_point(i), self.get_point(i + 1), t)
            )

            if distance < closest_distance:
                closest_distance = distance
                edge_id = i
                best_t = t

            if closest_distance < Tolerance.ZERO_TOLERANCE:
                break

        closest_point = Polyline.point_at(
            self.get_point(edge_id), self.get_point(edge_id + 1), best_t
        )

        return (closest_distance, edge_id, closest_point)

    # ═══════════════════════════════════════════════════════════════════════════
    # Mutators
    # ═══════════════════════════════════════════════════════════════════════════

    def set_point(self, index: int, point: Point) -> None:
        """Set the point at index."""

        if index < 0 or index >= self.point_count():
            return

        idx = index * 3
        self.coords[idx] = point[0]
        self.coords[idx + 1] = point[1]
        self.coords[idx + 2] = point[2]

    def add_point(self, point: Point) -> None:
        """Append a point."""
        self.coords.extend([point[0], point[1], point[2]])

        if self.point_count() == 3:
            self._recompute_plane_if_needed()

    def insert_point(self, index: int, point: Point) -> None:
        """Insert a point at index."""

        if index < 0 or index > self.point_count():
            return

        idx = index * 3
        self.coords[idx:idx] = [point[0], point[1], point[2]]

        if self.point_count() == 3:
            self._recompute_plane_if_needed()

    def remove_point(self, index: int) -> Point | None:
        """Remove the point at index into out_point; false when out of range."""

        if index < 0 or index >= self.point_count():
            return None

        idx = index * 3
        out_point = Point(self.coords[idx], self.coords[idx + 1], self.coords[idx + 2])
        del self.coords[idx : idx + 3]

        if self.point_count() == 3:
            self._recompute_plane_if_needed()

        return out_point

    def reverse(self) -> None:
        """Reverse the point order in place."""

        n = self.point_count()
        coords = []

        for i in range(n, 0, -1):
            idx = (i - 1) * 3
            coords.extend(
                [self.coords[idx], self.coords[idx + 1], self.coords[idx + 2]]
            )
        self.coords = coords
        self.plane.reverse()

    def reversed(self) -> "Polyline":
        """Return a reversed copy."""
        result = self.duplicate()
        result.reverse()

        return result

    def shift(self, times: int) -> None:
        """Rotate the points by times positions, keeping the closing duplicate."""

        if not self.coords:
            return

        was_closed = self.is_closed()

        if was_closed:
            del self.coords[-3:]

        n = self.point_count()

        if n > 0 and times != 0:
            offset = times % n
            coords = []

            for i in range(n):
                src = ((i + offset) % n) * 3
                coords.extend(
                    [self.coords[src], self.coords[src + 1], self.coords[src + 2]]
                )
            self.coords = coords

        if was_closed and n > 0:
            self.coords.extend([self.coords[0], self.coords[1], self.coords[2]])

    def translate(self, v: Vector) -> None:
        """Translate in place."""
        self += v

    def translated(self, v: Vector) -> "Polyline":
        """Return a translated copy."""
        result = self.duplicate()
        result.translate(v)

        return result

    def extend_segment(
        self,
        segment_id: int,
        dist0: float,
        dist1: float,
        proportion0: float = 0.0,
        proportion1: float = 0.0,
    ) -> None:
        """Move the segment ends by dist0 and dist1, or by proportions of its length when non-zero."""

        if segment_id < 0 or segment_id >= self.segment_count():
            return

        if dist0 == 0 and dist1 == 0 and proportion0 == 0 and proportion1 == 0:
            return

        was_closed = self.is_closed()
        p0 = self.get_point(segment_id)
        p1 = self.get_point(segment_id + 1)
        v = p1 - p0

        if proportion0 != 0 or proportion1 != 0:
            p0 = p0 - v * proportion0
            p1 = p1 + v * proportion1
        else:
            v.normalize_self()
            p0 = p0 - v * dist0
            p1 = p1 + v * dist1

        self.set_point(segment_id, p0)
        self.set_point(segment_id + 1, p1)

        if not was_closed:
            return

        if segment_id == 0:
            self.set_point(self.point_count() - 1, self.get_point(0))
        elif segment_id + 1 == self.point_count() - 1:
            self.set_point(0, self.get_point(self.point_count() - 1))

    def extend_segment_equally(
        self, segment_id: int, dist: float, proportion: float = 0.0
    ) -> None:
        """Move both segment ends by dist, or by proportion of its length when non-zero."""

        if segment_id < 0 or segment_id >= self.segment_count():
            return

        p0 = self.get_point(segment_id)
        p1 = self.get_point(segment_id + 1)
        Polyline.extend_segment_equally_static(p0, p1, dist, proportion)
        self.set_point(segment_id, p0)
        self.set_point(segment_id + 1, p1)

        if self.point_count() <= 2 or not self.is_closed():
            return

        if segment_id == 0:
            self.set_point(self.point_count() - 1, self.get_point(0))
        elif segment_id + 1 == self.point_count() - 1:
            self.set_point(0, self.get_point(self.point_count() - 1))

    def extend_edge_equally(self, edge_idx: int, distance: float) -> None:
        """Slide both ends of edge edge_idx outward by distance, keeping the closing duplicate in sync."""

        n = self.point_count()

        if n < 2 or edge_idx + 1 >= n:
            return

        i = edge_idx
        j = edge_idx + 1
        pi = self.get_point(i)
        pj = self.get_point(j)
        dir = pj - pi
        length = math.sqrt(dir.magnitude_squared())

        if length < 1e-12:
            return

        dir = dir * (distance / length)
        new_pi = pi - dir
        new_pj = pj + dir
        self.set_point(i, new_pi)
        self.set_point(j, new_pj)

        if i == 0:
            self.set_point(n - 1, new_pi)

        if j == n - 1:
            self.set_point(0, new_pj)

    def merge_collinear(self, tol: float = Tolerance.APPROXIMATION) -> None:
        """Drop points whose neighbours are collinear within tol; closed polylines wrap around."""

        closed = self.is_closed()
        pts = self.get_points()

        if closed and len(pts) > 1:
            pts.pop()

        zt2 = Tolerance.ZERO_TOLERANCE * Tolerance.ZERO_TOLERANCE
        max_pass = len(pts)
        changed = True

        for _ in range(max_pass):
            if not changed or len(pts) < 3:
                break

            changed = False
            m = len(pts)
            out = []

            for i in range(m):
                p = (i + m - 1) % m
                nx = (i + 1) % m

                if not closed and (i == 0 or i == m - 1):
                    out.append(pts[i])
                    continue

                a = pts[i] - pts[p]
                b = pts[nx] - pts[i]
                a2 = a.magnitude_squared()
                b2 = b.magnitude_squared()

                if (
                    a2 < zt2
                    or b2 < zt2
                    or a.cross(b).magnitude_squared() < tol * tol * a2 * b2
                ):
                    changed = True
                else:
                    out.append(pts[i])

            pts = out

        if closed and pts:
            pts.append(pts[0])

        self.coords = Polyline(pts).coords
        self._recompute_plane_if_needed()

    def remove_consecutive_duplicates(
        self, tol: float = Tolerance.APPROXIMATION
    ) -> None:
        """Drop consecutive points closer than tol."""

        tol_sq = tol * tol
        cleaned = []

        for p in self.get_points():
            if not cleaned or (p - cleaned[-1]).magnitude_squared() >= tol_sq:
                cleaned.append(p)

        self.coords = Polyline(cleaned).coords
        self._recompute_plane_if_needed()

    def simplify(self, tolerance: float) -> "Polyline":
        """Return a Ramer-Douglas-Peucker copy."""
        return Polyline(Polyline.simplify_points(self.get_points(), tolerance))

    def cut_by_plane(self, plane: Plane, flip: bool | None = None) -> "Polyline":
        """Return the part on one side of plane; flip picks the normal side, unset keeps the arc-length midpoint side."""

        n = self.point_count()

        if n < 2:
            return self.duplicate()

        normal = plane.z_axis
        origin = plane.origin
        keep_sign = 1.0

        if flip is not None:
            keep_sign = 1.0 if flip else -1.0
        else:
            keep_sign = (
                1.0
                if normal.dot(self._point_at_length(self.length() * 0.5) - origin)
                >= 0.0
                else -1.0
            )
        result = []

        for i in range(n - 1):
            a = self.get_point(i)
            b = self.get_point(i + 1)
            da = normal.dot(a - origin)
            db = normal.dot(b - origin)

            if da * keep_sign >= 0.0:
                result.append(a)

            if (da > 0.0) != (db > 0.0):
                result.append(a + (b - a) * (da / (da - db)))

        last = self.get_point(n - 1)

        if normal.dot(last - origin) * keep_sign >= 0.0:
            result.append(last)

        cut = Polyline(result)
        cut.remove_consecutive_duplicates(1e-6)

        return cut

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════

    def __eq__(self, other) -> bool:
        """Compare name, coordinates to 1e-6, width and linecolor; guid ignored."""

        if not isinstance(other, Polyline):
            return False

        if self.name != other.name:
            return False

        if self.point_count() != other.point_count():
            return False

        for i in range(len(self.coords)):
            if round(self.coords[i], Tolerance.ROUNDING) != round(
                other.coords[i], Tolerance.ROUNDING
            ):
                return False

        if round(self.width, Tolerance.ROUNDING) != round(
            other.width, Tolerance.ROUNDING
        ):
            return False

        return self.linecolor == other.linecolor

    def __ne__(self, other) -> bool:
        """Compare name, coordinates to 1e-6, width and linecolor; guid ignored."""
        return not self == other

    def __getitem__(self, index: int) -> Point:
        """Return the point at index."""
        if index < 0 or index >= self.point_count():
            raise IndexError("Index out of range")

        return self.get_point(index)

    def __setitem__(self, index: int, point: Point) -> None:
        """Set the point at index."""
        if index < 0 or index >= self.point_count():
            raise IndexError("Index out of range")

        self.set_point(index, point)

    def __iadd__(self, v: Vector) -> "Polyline":
        """Translate in place."""

        for i in range(self.point_count()):
            self.coords[i * 3] += v[0]
            self.coords[i * 3 + 1] += v[1]
            self.coords[i * 3 + 2] += v[2]

        self.plane = Plane(self.plane.origin + v, self.plane.x_axis, self.plane.y_axis)

        return self

    def __isub__(self, v: Vector) -> "Polyline":
        """Translate back in place."""

        for i in range(self.point_count()):
            self.coords[i * 3] -= v[0]
            self.coords[i * 3 + 1] -= v[1]
            self.coords[i * 3 + 2] -= v[2]

        self.plane = Plane(self.plane.origin - v, self.plane.x_axis, self.plane.y_axis)

        return self

    def __imul__(self, factor: float) -> "Polyline":
        """Scale every point in place."""
        for i in range(len(self.coords)):
            self.coords[i] *= factor

        return self

    def __itruediv__(self, factor: float) -> "Polyline":
        """Divide every point in place."""
        for i in range(len(self.coords)):
            self.coords[i] /= factor

        return self

    def __add__(self, v: Vector) -> "Polyline":
        """Return a translated copy."""
        result = self.duplicate()
        result += v

        return result

    def __sub__(self, v: Vector) -> "Polyline":
        """Return a copy translated back."""
        result = self.duplicate()
        result -= v

        return result

    def __mul__(self, factor: float) -> "Polyline":
        """Return a scaled copy."""
        result = self.duplicate()
        result *= factor

        return result

    def __truediv__(self, factor: float) -> "Polyline":
        """Return a divided copy."""
        result = self.duplicate()
        result /= factor

        return result

    def __neg__(self) -> "Polyline":
        """Return a reversed copy."""
        return self.reversed()

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════

    def transform(self, xform: "Xform") -> None:
        """Transform in place."""

        for i in range(self.point_count()):
            pt = self.get_point(i)
            pt.transform(xform)
            self.set_point(i, pt)

    def transformed(self, xform: "Xform") -> "Polyline":
        """Return a transformed copy."""
        result = self.duplicate()
        result.transform(xform)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Segment utilities
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def point_at(start: Point, end: Point, t: float) -> Point:
        """Return the point at parameter t (0 = start, 1 = end)."""

        s = 1.0 - t

        return Point(
            start[0] if start[0] == end[0] else s * start[0] + t * end[0],
            start[1] if start[1] == end[1] else s * start[1] + t * end[1],
            start[2] if start[2] == end[2] else s * start[2] + t * end[2],
        )

    @staticmethod
    def closest_point_to_line(
        point: Point, line_start: Point, line_end: Point
    ) -> float:
        """Compute the parameter t of the closest point on the line through line_start and line_end."""

        d = line_end - line_start
        dod = d.magnitude_squared()

        if dod <= 0.0:
            return 0.0

        to_start = point - line_start
        to_end = point - line_end

        if to_start.magnitude_squared() <= to_end.magnitude_squared():
            return to_start.dot(d) / dod

        return 1.0 + to_end.dot(d) / dod

    @staticmethod
    def line_line_overlap(
        line0_start: Point, line0_end: Point, line1_start: Point, line1_end: Point
    ) -> tuple[Point, Point] | None:
        """Compute the collinear overlap of two segments; false when none or a single point."""

        do_overlap, overlap_start, overlap_end = Polyline._line_line_overlap_points(
            line0_start, line0_end, line1_start, line1_end
        )

        if not do_overlap:
            return None

        return (overlap_start, overlap_end)

    @staticmethod
    def line_line_average(
        line0_start: Point, line0_end: Point, line1_start: Point, line1_end: Point
    ) -> tuple[Point, Point]:
        """Compute the midpoints of the paired starts and ends."""

        output_start = Point(
            (line0_start[0] + line1_start[0]) * 0.5,
            (line0_start[1] + line1_start[1]) * 0.5,
            (line0_start[2] + line1_start[2]) * 0.5,
        )
        output_end = Point(
            (line0_end[0] + line1_end[0]) * 0.5,
            (line0_end[1] + line1_end[1]) * 0.5,
            (line0_end[2] + line1_end[2]) * 0.5,
        )

        return (output_start, output_end)

    @staticmethod
    def line_line_overlap_average(
        line0_start: Point, line0_end: Point, line1_start: Point, line1_end: Point
    ) -> tuple[Point, Point]:
        """Compute the longer of the two midpoint pairings of the mutual overlaps."""

        _, line_a_start, line_a_end = Polyline._line_line_overlap_points(
            line0_start, line0_end, line1_start, line1_end
        )
        _, line_b_start, line_b_end = Polyline._line_line_overlap_points(
            line1_start, line1_end, line0_start, line0_end
        )
        mid_line0_start, mid_line0_end = Polyline.line_line_average(
            line_a_start, line_a_end, line_b_start, line_b_end
        )
        mid_line1_start, mid_line1_end = Polyline.line_line_average(
            line_a_start, line_a_end, line_b_end, line_b_start
        )

        if (mid_line0_end - mid_line0_start).magnitude_squared() > (
            mid_line1_end - mid_line1_start
        ).magnitude_squared():
            return (mid_line0_start, mid_line0_end)

        return (mid_line1_start, mid_line1_end)

    @staticmethod
    def line_from_projected_points(
        line_start: Point, line_end: Point, points: list[Point]
    ) -> tuple[Point, Point] | None:
        """Compute the extreme sub-segment of the line spanned by the projected points; false when a single point."""

        if not points:
            return None

        t_values = []

        for point in points:
            t_values.append(Polyline.closest_point_to_line(point, line_start, line_end))

        t_values.sort()
        output_start = Polyline.point_at(line_start, line_end, t_values[0])
        output_end = Polyline.point_at(line_start, line_end, t_values[-1])

        if abs(t_values[0] - t_values[-1]) <= Tolerance.ZERO_TOLERANCE:
            return None

        return (output_start, output_end)

    @staticmethod
    def extend_segment_equally_static(
        segment_start: Point, segment_end: Point, dist: float, proportion: float = 0.0
    ) -> None:
        """Move both ends by dist, or by proportion of the length when non-zero."""

        if dist == 0 and proportion == 0:
            return

        v = segment_end - segment_start

        if proportion != 0:
            segment_start -= v * proportion
            segment_end += v * proportion
        else:
            v.normalize_self()
            segment_start -= v * dist
            segment_end += v * dist

    @staticmethod
    def extend_line_segment(start: Point, end: Point, d0: float, d1: float) -> None:
        """Move start by d0 and end by d1 along the unit direction."""

        v = end - start
        v.normalize_self()
        start -= v * d0
        end += v * d1

    @staticmethod
    def shrink_line_segment(start: Point, end: Point, dist: float) -> None:
        """Move both ends inward by dist as a fraction of the length."""
        v = end - start
        start += v * dist
        end -= v * dist

    # ═══════════════════════════════════════════════════════════════════════════
    # Polygon utilities
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def tween_two_polylines(
        polyline0: "Polyline", polyline1: "Polyline", weight: float
    ) -> "Polyline":
        """Return the pointwise blend; polyline0 when the counts differ."""

        if polyline0.point_count() != polyline1.point_count():
            return polyline0.duplicate()

        result = Polyline()

        for i in range(polyline0.point_count()):
            p0 = polyline0.get_point(i)
            p1 = polyline1.get_point(i)
            result.add_point(p0 + (p1 - p0) * weight)

        return result

    @staticmethod
    def interpolate_points(
        from_pt: Point, to_pt: Point, steps: int, kind: int = 0
    ) -> list[Point]:
        """Return steps points between from and to; kind 0 none, 1 both, 2 start endpoint."""
        return Point.interpolate(from_pt, to_pt, steps, kind)

    @staticmethod
    def quick_hull(polygon: "Polyline") -> "Polyline":
        """Return the convex hull in the polygon's average plane."""

        origin, xa, ya, _ = polygon.get_average_plane()
        pts2d = polygon._project_to_plane(origin, xa, ya)
        ai = 0
        bi = 0

        for i in range(1, len(pts2d)):
            if pts2d[i][0] < pts2d[ai][0]:
                ai = i

            if pts2d[i][0] >= pts2d[bi][0]:
                bi = i

        ax = pts2d[ai][0]
        ay = pts2d[ai][1]
        bx = pts2d[bi][0]
        by = pts2d[bi][1]
        left = []
        right = []

        for p in pts2d:
            if _ccw_2d(ax, ay, bx, by, p[0], p[1]) > 0.0:
                left.append(p)
            else:
                right.append(p)

        hull = [(ax, ay)]
        Polyline._quick_hull_recurse(left, ax, ay, bx, by, hull)
        hull.append((bx, by))
        Polyline._quick_hull_recurse(right, bx, by, ax, ay, hull)
        pts3d = []

        for h in hull:
            pts3d.append(Polyline._unproject(origin, xa, ya, h[0], h[1]))

        return Polyline(pts3d)

    @staticmethod
    def bounding_rectangle(polygon: "Polyline") -> Optional["Polyline"]:
        """Return the minimum-area rectangle of the hull as a closed 5-point polyline."""

        hull = Polyline.quick_hull(polygon)

        if hull.point_count() <= 2:
            return None

        origin, xa, ya, _ = polygon.get_average_plane()
        hull2d = hull._project_to_plane(origin, xa, ya)
        best_area = float("inf")
        best_min_u = 0.0
        best_max_u = 0.0
        best_min_v = 0.0
        best_max_v = 0.0
        best_angle = 0.0
        hn = len(hull2d)

        for i in range(hn):
            j = (i + 1) % hn
            ex = hull2d[j][0] - hull2d[i][0]
            ey = hull2d[j][1] - hull2d[i][1]
            length = math.sqrt(ex * ex + ey * ey)

            if length < 1e-12:
                continue

            ca = ex / length
            sa = ey / length
            min_u = float("inf")
            max_u = float("-inf")
            min_v = float("inf")
            max_v = float("-inf")

            for h in hull2d:
                u = h[0] * ca + h[1] * sa
                v = -h[0] * sa + h[1] * ca
                min_u = min(min_u, u)
                max_u = max(max_u, u)
                min_v = min(min_v, v)
                max_v = max(max_v, v)

            area = (max_u - min_u) * (max_v - min_v)

            if area < best_area:
                best_area = area
                best_min_u = min_u
                best_max_u = max_u
                best_min_v = min_v
                best_max_v = max_v
                best_angle = math.atan2(ey, ex)

        ca = math.cos(best_angle)
        sa = math.sin(best_angle)
        uv = [
            (best_min_u, best_min_v),
            (best_min_u, best_max_v),
            (best_max_u, best_max_v),
            (best_max_u, best_min_v),
        ]
        pts3d = []

        for c in uv:
            pts3d.append(
                Polyline._unproject(
                    origin, xa, ya, c[0] * ca - c[1] * sa, c[0] * sa + c[1] * ca
                )
            )
        pts3d.append(pts3d[0])

        return Polyline(pts3d)

    @staticmethod
    def grid_of_points_in_polygon(
        polygon: "Polyline", offset_dist: float, div_dist: float, max_pts: int = 100
    ) -> list[Point]:
        """Return a grid of interior points spaced div_dist, on the polygon miter-offset by offset_dist."""

        if div_dist < 1e-12:
            return []

        origin, xa, ya, _ = polygon.get_average_plane()
        poly2d = polygon._project_to_plane(origin, xa, ya)

        if (
            len(poly2d) > 1
            and polygon.get_point(0).distance(
                polygon.get_point(polygon.point_count() - 1)
            )
            < 1e-10
        ):
            poly2d.pop()

        if not poly2d:
            return []

        poly2d = Polyline._offset_polygon_2d(poly2d, offset_dist)
        x_min = float("inf")
        x_max = float("-inf")
        y_min = float("inf")
        y_max = float("-inf")

        for p in poly2d:
            x_min = min(x_min, p[0])
            x_max = max(x_max, p[0])
            y_min = min(y_min, p[1])
            y_max = max(y_max, p[1])

        result = []
        u = x_min

        while u <= x_max + 1e-10 and len(result) < max_pts:
            v = y_min

            while v <= y_max + 1e-10 and len(result) < max_pts:
                if Polyline._point_in_polygon(poly2d, u, v):
                    result.append(Polyline._unproject(origin, xa, ya, u, v))

                v += div_dist

            u += div_dist

        return result

    @staticmethod
    def polylabel(
        polylines: list["Polyline"], precision: float = 1.0
    ) -> tuple[Point, Plane, float]:
        """Return the largest inscribed circle of polylines[0] minus the holes polylines[1..]: center, plane, radius."""

        if not polylines:
            return (Point(0.0, 0.0, 0.0), Plane(), 0.0)

        origin, xa, ya, za = polylines[0].get_average_plane()
        rings2d = []
        sizes = []

        for pl in polylines:
            ring = pl._project_to_plane(origin, xa, ya)

            if (
                len(ring) > 1
                and pl.get_point(0).distance(pl.get_point(pl.point_count() - 1)) < 1e-10
            ):
                ring.pop()

            mnx = float("inf")
            mny = float("inf")
            mxx = float("-inf")
            mxy = float("-inf")

            for uv in ring:
                mnx = min(mnx, uv[0])
                mxx = max(mxx, uv[0])
                mny = min(mny, uv[1])
                mxy = max(mxy, uv[1])

            rings2d.append(ring)
            sizes.append((mxx - mnx) * (mxx - mnx) + (mxy - mny) * (mxy - mny))

        ids = sorted(range(len(rings2d)), key=lambda i: -sizes[i])
        polygon = []

        for id in ids:
            polygon.append(rings2d[id])

        cr = _mapbox_polylabel(polygon, precision)
        center = Polyline._unproject(origin, xa, ya, cr[0], cr[1])

        return (center, Plane.from_frame(origin, xa, ya, za), cr[2])

    @staticmethod
    def polylabel_circle_division_points(
        division_direction_in_3d: Vector,
        polylines: list["Polyline"],
        division: int = 4,
        scale: float = 0.75,
        precision: float = 1.0,
        orient_to_closest_edge: bool = True,
    ) -> list[Point]:
        """Return division points on the polylabel circle scaled by scale, oriented to the closest edge or division_direction_in_3d."""

        center, plane, r = Polyline.polylabel(polylines, precision)
        radius = r * scale
        is_direction_valid = (
            division_direction_in_3d[0] != 0.0
            or division_direction_in_3d[1] != 0.0
            or division_direction_in_3d[2] != 0.0
        )
        found, edge_i, edge_j = Polyline._closest_edge(center, polylines)
        found = orient_to_closest_edge and found
        x_axis = plane.x_axis
        y_axis = plane.y_axis
        z_axis = plane.z_axis

        if is_direction_valid or orient_to_closest_edge:
            dir = (
                polylines[edge_i].get_point(edge_j + 1)
                - polylines[edge_i].get_point(edge_j)
                if found
                else division_direction_in_3d
            )
            x_axis = Vector(dir[0], dir[1], dir[2])
            y_axis = dir.cross(z_axis)

        x_axis.normalize_self()
        y_axis.normalize_self()
        z_axis.normalize_self()
        points = []
        chunk = 360.0 / division

        for i in range(division):
            rad = (45.0 + i * chunk) * Tolerance.PI / 180.0
            points.append(
                Polyline._unproject(
                    center,
                    x_axis,
                    y_axis,
                    radius * math.cos(rad),
                    radius * math.sin(rad),
                )
            )
        return points

    @staticmethod
    def boolean_op(
        a: "Polyline", b: "Polyline", clip_type: int, plane: Plane | None = None
    ) -> list["Polyline"]:
        """Return the Vatti boolean of two closed polylines; plane given: in its local frame, else on x and y; clip_type 0 intersection, 1 union, 2 a minus b."""

        from .boolean_polyline import BooleanPolyline

        if plane is None:
            return BooleanPolyline.compute(a, b, clip_type)

        pa2d = Polyline._boolean_project(a, plane)
        pb2d = Polyline._boolean_project(b, plane)
        Polyline._ensure_ccw(pa2d)
        Polyline._ensure_ccw(pb2d)
        results = BooleanPolyline.compute(pa2d, pb2d, clip_type)
        o = plane.origin
        x = plane.x_axis
        y = plane.y_axis

        for r in results:
            for i in range(r.point_count()):
                r.set_point(
                    i,
                    Polyline._unproject(o, x, y, r.coords[i * 3], r.coords[i * 3 + 1]),
                )
        return results

    @staticmethod
    def simplify_points(points: list[Point], tolerance: float) -> list[Point]:
        """Return the Ramer-Douglas-Peucker simplification of a point list."""

        n = len(points)

        if n < 3:
            return list(points)

        keep = [False] * n
        keep[0] = True
        keep[n - 1] = True
        Polyline._simplify_rdp(points, 0, n - 1, tolerance, keep)
        result = []

        for i in range(n):
            if keep[i]:
                result.append(points[i])

        return result

    @staticmethod
    def two_rects_from_frame(
        p: Point,
        segment_vector: Vector,
        zaxis: Vector,
        middle: bool,
        radius: float,
        length: float,
        flip_male: int,
    ) -> tuple["Polyline", "Polyline"]:
        """Compute male rect0 and female rect1 cross-sections of radius about p along segment_vector; flip_male rotates the corners."""

        y_axis = zaxis.cross(segment_vector)
        x_axis = y_axis.cross(segment_vector)
        x_axis.normalize_self()
        y_axis.normalize_self()
        x_axis = x_axis * radius
        y_axis = y_axis * radius
        sv0 = segment_vector * (length * -0.5)
        sv1 = segment_vector * (length * 0.5)
        v = [-x_axis - y_axis, x_axis - y_axis, x_axis + y_axis, -x_axis + y_axis]

        if not middle and flip_male == 1:
            v = v[1:] + v[:1]
        elif not middle and flip_male == -1:
            v = v[-1:] + v[:-1]

        rect0 = Polyline(
            [
                p + sv0 + v[1],
                p + sv1 + v[1],
                p + sv1 + v[0],
                p + sv0 + v[0],
                p + sv0 + v[1],
            ]
        )
        rect1 = Polyline(
            [
                p + sv0 + v[2],
                p + sv1 + v[2],
                p + sv1 + v[3],
                p + sv0 + v[3],
                p + sv0 + v[2],
            ]
        )

        return (rect0, rect1)

    @staticmethod
    def trim_rectangles_by_plane(first: Polyline, second: Polyline, plane: Plane) -> bool:
        """Cut two closed 5-point rectangles at plane, keeping the side on the positive half; false when a long edge misses the plane."""
        from .intersection import line_plane
        from .line import Line

        if first.point_count() != 5 or second.point_count() != 5:
            return False

        points = [
            line_plane(Line.from_points(first[0], first[1]), plane, False),
            line_plane(Line.from_points(first[3], first[2]), plane, False),
            line_plane(Line.from_points(second[0], second[1]), plane, False),
            line_plane(Line.from_points(second[3], second[2]), plane, False),
        ]
        for point in points:
            if point is None or not all(math.isfinite(point[i]) for i in range(3)):
                return False

        if plane.has_on_negative_side(first[0]):
            first.set_point(0, points[0])
            first.set_point(3, points[1])
            first.set_point(4, points[0])
            second.set_point(0, points[2])
            second.set_point(3, points[3])
            second.set_point(4, points[2])
        else:
            first.set_point(1, points[0])
            first.set_point(2, points[1])
            second.set_point(1, points[2])
            second.set_point(2, points[3])

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        return {
            "coords": self.coords,
            "dash": list(self.dash),
            "guid": self.guid,
            "linecolor": self.linecolor.__jsondump__(),
            "name": self.name,
            "type": "Polyline",
            "width": self.width,
        }

    @classmethod
    def __jsonload__(cls, data: dict, guid: str = None, name: str = None) -> "Polyline":
        """Deserialize from a JSON object."""

        from .file_encoders import file_decode_node

        polyline = cls()
        polyline.guid = guid if guid is not None else data.get("guid", polyline.guid)
        polyline.name = name if name is not None else data.get("name", polyline.name)

        if "coords" in data:
            polyline.coords = list(data["coords"])
        elif "points" in data:
            for pt_json in data["points"]:
                polyline.add_point(file_decode_node(pt_json))

        if "width" in data:
            polyline.width = data["width"]

        if "dash" in data:
            polyline.dash = list(data["dash"])

        if "linecolor" in data:
            polyline.linecolor = file_decode_node(data["linecolor"])

        polyline._recompute_plane_if_needed()

        return polyline

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Polyline":
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "Polyline":
        """Read from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        return cls.__jsonload__(data)

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""

        from .proto import polyline_pb2

        proto = polyline_pb2.Polyline()
        self.pb_fill(proto)

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Polyline":
        """Deserialize from protobuf bytes."""

        from .proto import polyline_pb2

        proto = polyline_pb2.Polyline()
        proto.ParseFromString(data)
        pl = cls.from_coords(list(proto.coords))

        if proto.guid:
            pl.guid = proto.guid

        pl.name = proto.name
        pl.width = proto.width
        pl.dash = list(proto.dash)

        if proto.HasField("linecolor"):
            c = proto.linecolor
            pl.linecolor = Color(c.r, c.g, c.b, c.a, c.name)

        return pl

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        """Write to a protobuf file."""
        data = self.pb_dumps()

        with open(filepath, "wb") as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "Polyline":
        """Read from a protobuf file."""
        with open(filepath, "rb") as f:
            data = f.read()

        return cls.pb_loads(data)

    def pb_fill(self, proto: "polyline_pb2.Polyline") -> None:
        """Fill a Polyline proto in place."""

        if self.has_guid():
            proto.guid = self._guid

        proto.name = self.name
        proto.width = self.width
        proto.dash.extend(self.dash)
        proto.coords.extend(self.coords)
        proto.linecolor.r = self.linecolor.r
        proto.linecolor.g = self.linecolor.g
        proto.linecolor.b = self.linecolor.b
        proto.linecolor.a = self.linecolor.a
        proto.linecolor.name = self.linecolor.name

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════

    def __str__(self) -> str:
        """Return "[(x0, y0, z0), (x1, y1, z1), ...]"."""

        pts = []

        for i in range(self.point_count()):
            pts.append(
                f"({self.coords[i * 3]}, {self.coords[i * 3 + 1]}, {self.coords[i * 3 + 2]})"
            )
        return "[" + ", ".join(pts) + "]"

    def __repr__(self) -> str:
        """Return "Polyline(name, N points)"."""
        return f"Polyline({self.name}, {self.point_count()} points)"

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _recompute_plane_if_needed(self) -> None:
        """Recompute plane when _plane_dirty."""
        self._plane_dirty = True

    def _average_normal(self) -> Vector:
        """Compute the average normal by Newell's method."""

        if self.point_count() < 3:
            return Vector(0.0, 0.0, 1.0)

        n = self.point_count() - 1 if self.is_closed() else self.point_count()
        avg_normal = Vector(0.0, 0.0, 0.0)

        for i in range(n):
            prev = n - 1 if i == 0 else i - 1
            next = (i + 1) % n
            v1 = self.get_point(i) - self.get_point(prev)
            v2 = self.get_point(next) - self.get_point(i)
            avg_normal += v1.cross(v2)

        avg_normal.normalize_self()

        return avg_normal

    def _point_at_length(self, distance: float) -> Point:
        """Return the point at arc length distance from the start."""

        acc = 0.0

        for i in range(self.point_count() - 1):
            a = self.get_point(i)
            b = self.get_point(i + 1)
            seg_len = (b - a).magnitude()

            if acc + seg_len >= distance:
                t = (distance - acc) / seg_len if seg_len > 1e-14 else 0.0

                return a + (b - a) * t

            acc += seg_len

        return self.get_point(0)

    def _project_to_plane(
        self, origin: Point, x_axis: Vector, y_axis: Vector
    ) -> list[tuple[float, float]]:
        """Compute the 2D coordinates of the points in the frame (origin, x_axis, y_axis)."""

        pts2d = []

        for i in range(self.point_count()):
            d = self.get_point(i) - origin
            pts2d.append((d.dot(x_axis), d.dot(y_axis)))

        return pts2d

    @staticmethod
    def _unproject(
        origin: Point, x_axis: Vector, y_axis: Vector, u: float, v: float
    ) -> Point:
        """Return the 3D point of (u, v) in the frame (origin, x_axis, y_axis)."""

        return Point(
            origin[0] + u * x_axis[0] + v * y_axis[0],
            origin[1] + u * x_axis[1] + v * y_axis[1],
            origin[2] + u * x_axis[2] + v * y_axis[2],
        )

    @staticmethod
    def _line_line_overlap_points(
        line0_start: Point, line0_end: Point, line1_start: Point, line1_end: Point
    ) -> tuple[bool, Point, Point]:
        """Compute the collinear overlap of two segments as points; False when none or a single point."""

        t = [0.0, 1.0, 0.0, 0.0]
        t[2] = Polyline.closest_point_to_line(line1_start, line0_start, line0_end)
        t[3] = Polyline.closest_point_to_line(line1_end, line0_start, line0_end)
        do_overlap = not ((t[2] < 0 and t[3] < 0) or (t[2] > 1 and t[3] > 1))
        t.sort()
        do_overlap = do_overlap and abs(t[2] - t[1]) > Tolerance.ZERO_TOLERANCE
        overlap_start = Polyline.point_at(line0_start, line0_end, t[1])
        overlap_end = Polyline.point_at(line0_start, line0_end, t[2])

        return (do_overlap, overlap_start, overlap_end)

    @staticmethod
    def _quick_hull_recurse(
        pts: list[tuple[float, float]],
        ax: float,
        ay: float,
        bx: float,
        by: float,
        hull: list[tuple[float, float]],
    ) -> None:
        """Add the hull points of pts right of the segment (a, b) to hull."""

        if not pts:
            return

        fi = 0
        best = float("-inf")

        for i in range(len(pts)):
            val = _ccw_2d(ax, ay, bx, by, pts[i][0], pts[i][1])

            if val >= best:
                best = val
                fi = i

        fx = pts[fi][0]
        fy = pts[fi][1]
        left = []
        right = []

        for p in pts:
            if _ccw_2d(ax, ay, fx, fy, p[0], p[1]) > 0.0:
                left.append(p)

            if _ccw_2d(fx, fy, bx, by, p[0], p[1]) > 0.0:
                right.append(p)

        Polyline._quick_hull_recurse(left, ax, ay, fx, fy, hull)
        hull.append((fx, fy))
        Polyline._quick_hull_recurse(right, fx, fy, bx, by, hull)

    @staticmethod
    def _offset_polygon_2d(
        poly2d: list[tuple[float, float]], offset_dist: float
    ) -> list[tuple[float, float]]:
        """Miter-offset a 2D polygon in place by offset_dist."""

        n = len(poly2d)

        if offset_dist == 0.0 or n < 3:
            return poly2d

        signed_area = 0.0

        for i in range(n):
            a = poly2d[i]
            b = poly2d[(i + 1) % n]
            signed_area += a[0] * b[1] - b[0] * a[1]

        delta = -offset_dist if signed_area < 0.0 else offset_dist
        normals = []

        for i in range(n):
            a = poly2d[i]
            b = poly2d[(i + 1) % n]
            ex = b[0] - a[0]
            ey = b[1] - a[1]
            length = math.sqrt(ex * ex + ey * ey)

            if length < 1e-12:
                normals.append((0.0, 0.0))
            else:
                normals.append((ey / length, -ex / length))

        out = []

        for i in range(n):
            np = normals[(i + n - 1) % n]
            nn = normals[i]
            cos_a = np[0] * nn[0] + np[1] * nn[1]
            sin_a = np[0] * nn[1] - np[1] * nn[0]
            denom = 1.0 + cos_a
            concave = cos_a > -0.999 and sin_a * delta < 0.0 and offset_dist > 0.0

            if concave:
                out.append((poly2d[i][0] + np[0] * delta, poly2d[i][1] + np[1] * delta))
                out.append((poly2d[i][0], poly2d[i][1]))
                out.append((poly2d[i][0] + nn[0] * delta, poly2d[i][1] + nn[1] * delta))
            elif abs(denom) < 1e-9:
                out.append(
                    (
                        poly2d[i][0] + (np[0] + nn[0]) * 0.5 * delta,
                        poly2d[i][1] + (np[1] + nn[1]) * 0.5 * delta,
                    )
                )
            else:
                out.append(
                    (
                        poly2d[i][0] + (np[0] + nn[0]) / denom * delta,
                        poly2d[i][1] + (np[1] + nn[1]) / denom * delta,
                    )
                )
        out_area = 0.0

        for i in range(len(out)):
            a = out[i]
            b = out[(i + 1) % len(out)]
            out_area += a[0] * b[1] - b[0] * a[1]

        if len(out) >= 3 and abs(out_area) > 1e-4:
            return out

        return poly2d

    @staticmethod
    def _point_in_polygon(
        poly2d: list[tuple[float, float]], px: float, py: float
    ) -> bool:
        """Return whether (px, py) is inside poly2d by ray crossing."""

        n = len(poly2d)
        inside = False
        j = n - 1

        for i in range(n):
            xi = poly2d[i][0]
            yi = poly2d[i][1]
            xj = poly2d[j][0]
            yj = poly2d[j][1]

            if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                inside = not inside

            j = i

        return inside

    @staticmethod
    def _closest_edge(
        center: Point, polylines: list["Polyline"]
    ) -> tuple[bool, int, int]:
        """Find the polyline and edge closest to center."""

        edge_i = 0
        edge_j = 0
        best_sq = float("inf")

        for i in range(len(polylines)):
            for j in range(polylines[i].point_count() - 1):
                a = polylines[i].get_point(j)
                e = polylines[i].get_point(j + 1) - a
                len2 = e.magnitude_squared()

                if len2 <= 0.0:
                    continue

                t = (center - a).dot(e) / len2

                if t < 0.0 or t > 1.0:
                    continue

                d2 = (center - (a + e * t)).magnitude_squared()

                if d2 < best_sq:
                    best_sq = d2
                    edge_i = i
                    edge_j = j

        return (best_sq < float("inf"), edge_i, edge_j)

    @staticmethod
    def _boolean_project(pl: "Polyline", plane: Plane) -> "Polyline":
        """Return pl projected into plane's local frame as 2D."""

        o = plane.origin
        x = plane.x_axis
        y = plane.y_axis
        n = pl.point_count()
        p2d = Polyline()
        p2d.coords = [0.0] * (n * 3)

        for i in range(n):
            d = pl.get_point(i) - o
            p2d.coords[i * 3] = d.dot(x)
            p2d.coords[i * 3 + 1] = d.dot(y)
            p2d.coords[i * 3 + 2] = 0.0

        if n >= 4:
            dx = p2d.coords[(n - 1) * 3] - p2d.coords[0]
            dy = p2d.coords[(n - 1) * 3 + 1] - p2d.coords[1]

            if dx * dx + dy * dy < 1.0:
                p2d.coords[(n - 1) * 3] = p2d.coords[0]
                p2d.coords[(n - 1) * 3 + 1] = p2d.coords[1]

        return p2d

    @staticmethod
    def _ensure_ccw(p2d: "Polyline") -> None:
        """Reverse p2d in place when it winds clockwise."""

        n = p2d.point_count()
        m = n

        if m >= 4:
            dx = p2d.coords[(m - 1) * 3] - p2d.coords[0]
            dy = p2d.coords[(m - 1) * 3 + 1] - p2d.coords[1]

            if dx * dx + dy * dy < 1e-10:
                m -= 1

        if m < 3:
            return

        area = 0.0

        for i in range(m):
            j = (i + 1) % m
            area += (
                p2d.coords[i * 3] * p2d.coords[j * 3 + 1]
                - p2d.coords[j * 3] * p2d.coords[i * 3 + 1]
            )
        if area < 0.0:
            p2d.reverse()

    @staticmethod
    def _simplify_perp_dist(pt: Point, line_start: Point, line_end: Point) -> float:
        """Return the perpendicular distance from pt to the line through line_start and line_end."""

        d = line_end - line_start
        len_sq = d.magnitude_squared()

        if len_sq == 0.0:
            return math.sqrt((pt - line_start).magnitude_squared())

        t = (pt - line_start).dot(d) / len_sq
        t = max(0.0, min(1.0, t))

        return math.sqrt((pt - (line_start + d * t)).magnitude_squared())

    @staticmethod
    def _simplify_rdp(
        points: list[Point], start: int, end: int, tolerance: float, keep: list[bool]
    ) -> None:
        """Mark the points to keep between start and end by recursive Ramer-Douglas-Peucker."""

        if end <= start + 1:
            return

        max_dist = 0.0
        max_idx = start

        for i in range(start + 1, end):
            d = Polyline._simplify_perp_dist(points[i], points[start], points[end])

            if d > max_dist:
                max_dist = d
                max_idx = i

        if max_dist <= tolerance:
            return

        keep[max_idx] = True
        Polyline._simplify_rdp(points, start, max_idx, tolerance, keep)
        Polyline._simplify_rdp(points, max_idx, end, tolerance, keep)
