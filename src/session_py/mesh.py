from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Callable
from typing import Optional
from typing import Union
from enum import Enum
import copy
import json
import math
import struct
import uuid

if TYPE_CHECKING:
    from .proto import mesh_pb2
    from pathlib import Path
    from .xform import Xform
    from .spatial_aabbtree import SpatialAABBTree

from .point import Point
from .vector import Vector
from .tolerance import Tolerance
from .tolerance import PI
from .color import Color
from .obb import OBB
from .aabb import AABB
from .line import Line
from .plane import Plane
from .polyline import Polyline
from .spatial_bvh import SpatialBVH
from .remesh_cdt import cdt_triangulate


class ColorMode(Enum):
    """Which stored colors a mesh renders with."""

    OBJECTCOLOR = "objectcolor"
    POINTCOLORS = "pointcolors"
    FACECOLORS = "facecolors"
    NONE = "none"


class NormalWeighting(Enum):
    """Weighting scheme for vertex normals."""

    AREA = "area"
    ANGLE = "angle"
    UNIFORM = "uniform"


class Attributes:
    """A vertex's attribute map, allocated only once something is stored in it."""

    __slots__ = ("_m",)

    def __init__(self, other: Attributes | dict[str, float] | None = None):
        """Construct from another map, allocating only when it is not empty."""
        self._m: dict[str, float] | None = dict(other) if other else None

    def get(self, key: str, default: float | None = None) -> float | None:
        """Return the value named key, or default when missing."""
        return self._m.get(key, default) if self._m is not None else default

    def items(self):
        """Return the (name, value) pairs."""
        return self._m.items() if self._m is not None else {}.items()

    def keys(self):
        """Return the names."""
        return self._m.keys() if self._m is not None else {}.keys()

    def values(self):
        """Return the values."""
        return self._m.values() if self._m is not None else {}.values()

    def update(self, other: Attributes | dict[str, float]) -> None:
        """Store every entry of other."""

        for k, v in other.items():
            self[k] = v

    def pop(self, key: str, *default: float) -> float:
        """Remove key and free the map when it becomes empty; returns the value or default."""

        if self._m is None:
            if default:
                return default[0]

            raise KeyError(key)

        out = self._m.pop(key, *default)

        if not self._m:
            self._m = None

        return out

    def clear(self) -> None:
        """Free the map."""
        self._m = None

    def __getitem__(self, key: str) -> float:
        """Return the value named key; raises when missing."""

        if self._m is None:
            raise KeyError(key)

        return self._m[key]

    def __setitem__(self, key: str, value: float) -> None:
        """Store a value, allocating the map on first use."""

        if self._m is None:
            self._m = {}

        self._m[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove key."""
        self.pop(key)

    def __contains__(self, key: str) -> bool:
        """Return whether key is stored."""
        return self._m is not None and key in self._m

    def __iter__(self):
        """Iterate the names."""
        return iter(self._m) if self._m is not None else iter(())

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._m) if self._m is not None else 0

    def __bool__(self) -> bool:
        """Return whether anything is stored."""
        return bool(self._m)

    def __eq__(self, other: object) -> bool:
        """Compare the stored maps."""

        mine = self._m if self._m is not None else {}

        if isinstance(other, Attributes):
            return mine == (other._m if other._m is not None else {})

        if isinstance(other, dict):
            return mine == other

        return NotImplemented

    def __ne__(self, other: object) -> bool:
        """Compare the stored maps."""

        eq = self.__eq__(other)

        return eq if eq is NotImplemented else not eq

    def __repr__(self) -> str:
        """Return the stored map as a string."""
        return repr(self._m if self._m is not None else {})


class VertexData:
    """Vertex position and attributes."""

    def __init__(self, point: Point | None = None):
        """Construct at a point, the origin by default."""

        if point is None:
            point = Point(0.0, 0.0, 0.0)

        self.x = point[0]
        self.y = point[1]
        self.z = point[2]
        self.attributes = Attributes()

    def __eq__(self, other):
        """Compare position and attributes exactly."""

        if not isinstance(other, VertexData):
            return NotImplemented

        return (
            self.x == other.x
            and self.y == other.y
            and self.z == other.z
            and self.attributes == other.attributes
        )

    def __ne__(self, other):
        """Compare position and attributes exactly."""
        return not self.__eq__(other)

    def position(self) -> Point:
        """Return the position as a Point."""
        return Point(self.x, self.y, self.z)

    def set_position(self, point: Point) -> None:
        """Set the position from a Point."""

        self.x = point[0]
        self.y = point[1]
        self.z = point[2]

    def color(self) -> list[float]:
        """Return the vertex color as RGB, 0.5 grey when unset."""

        return [
            self.attributes.get("r", 0.5),
            self.attributes.get("g", 0.5),
            self.attributes.get("b", 0.5),
        ]

    def set_color(self, r: float, g: float, b: float) -> None:
        """Set the vertex color."""

        self.attributes["r"] = r
        self.attributes["g"] = g
        self.attributes["b"] = b

    def normal(self) -> list[float] | None:
        """Return the vertex normal when set."""

        if (
            "nx" in self.attributes
            and "ny" in self.attributes
            and "nz" in self.attributes
        ):
            return [self.attributes["nx"], self.attributes["ny"], self.attributes["nz"]]

        return None

    def set_normal(self, nx: float, ny: float, nz: float) -> None:
        """Set the vertex normal."""

        self.attributes["nx"] = nx
        self.attributes["ny"] = ny
        self.attributes["nz"] = nz


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _newell_normal(pts: list[Point]) -> Vector:
    """Unit Newell normal of a closed ring."""

    n = len(pts)
    nx = 0.0
    ny = 0.0
    nz = 0.0

    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])

    normal = Vector(nx, ny, nz)

    if not normal.normalize_self():
        return Vector(0.0, 0.0, 0.0)

    return normal


def _round_half_away(x: float) -> int:
    """Round to the nearest integer, halves away from zero like std::round."""

    r = math.floor(abs(x))

    if abs(x) - r >= 0.5:
        r += 1

    return int(math.copysign(r, x))


def _ring_centroid(pts: list[Point]) -> Point:
    """Average of a point ring."""

    x = 0.0
    y = 0.0
    z = 0.0

    for p in pts:
        x += p[0]
        y += p[1]
        z += p[2]

    n = float(len(pts))

    return Point(x / n, y / n, z / n)


def _planar_cdt(pts: list[Point]) -> list:
    """CDT of a planar ring projected onto its own plane; indices into pts, empty when degenerate."""

    n = len(pts)
    nx = 0.0
    ny = 0.0
    nz = 0.0

    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])

    nlen = math.sqrt(nx * nx + ny * ny + nz * nz)

    if nlen <= 1e-12:
        return []

    nx /= nlen
    ny /= nlen
    nz /= nlen
    ux = 1.0
    uy = 0.0
    uz = 0.0

    if abs(nx) > 0.9:
        ux = 0.0
        uy = 1.0

    dot = ux * nx + uy * ny + uz * nz
    ux -= dot * nx
    uy -= dot * ny
    uz -= dot * nz
    um = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux /= um
    uy /= um
    uz /= um
    vx = ny * uz - nz * uy
    vy = nz * ux - nx * uz
    vz = nx * uy - ny * ux
    bpts = []

    for p in pts:
        bpts.append(
            Point(
                p[0] * ux + p[1] * uy + p[2] * uz,
                p[0] * vx + p[1] * vy + p[2] * vz,
                0.0,
            )
        )

    return cdt_triangulate(bpts, [])


def _signed_area_2d(pts: list[tuple[float, float]]) -> float:
    """Twice the signed 2D area of a ring."""

    area = 0.0
    n = len(pts)

    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]

    return area


# ═══════════════════════════════════════════════════════════════════════════
# Loft
# ═══════════════════════════════════════════════════════════════════════════


class LoftFaceRole(Enum):
    """Role of a face inside a loft panel."""

    TopCap = "TopCap"
    BotCap = "BotCap"
    QuadWall = "QuadWall"
    TriWall = "TriWall"


class LoftWallFace:
    """One wall face of a loft panel and the original vertices it spans."""

    def __init__(self):
        """Construct an empty wall face."""

        self.face_key = 0
        self.face_index = 0
        self.is_quad = False
        self.top_v0 = 0
        self.top_v1 = 0
        self.bot_v0 = 0
        self.bot_v1 = 0


class LoftPanel:
    """One lofted panel with its cap faces, walls and vertex maps."""

    def __init__(self):
        """Construct an empty panel."""

        self.mesh = Mesh()
        self.top_face_key: int | None = None
        self.bot_face_key: int | None = None
        self.wall_faces: list[LoftWallFace] = []
        self.face_roles: dict[int, LoftFaceRole] = {}
        self.orig_top_to_local: dict[int, int] = {}
        self.orig_bot_to_local: dict[int, int] = {}
        self.top_vertices: list[int] = []
        self.bot_vertices: list[int] = []


class LoftAdjPair:
    """Two panel walls that face each other."""

    def __init__(self, pi: int, wi: int, pj: int, wj: int):
        """Construct from panel and wall indices of both sides."""

        self.pi = pi
        self.wi = wi
        self.pj = pj
        self.wj = wj


class LoftResult:
    """Panels of loft_panels with their wall adjacency."""

    def __init__(
        self,
        panels: list[LoftPanel],
        adjacency: list[LoftAdjPair],
        top_mesh: "Mesh",
        bot_mesh: "Mesh",
    ):
        """Construct from panels, adjacency and the two cap meshes."""

        self.panels = panels
        self.adjacency = adjacency
        self.top_mesh = top_mesh
        self.bot_mesh = bot_mesh

    def __iter__(self):
        """Unpack as (panels, adjacency, top_mesh, bot_mesh)."""
        return iter((self.panels, self.adjacency, self.top_mesh, self.bot_mesh))


class _LoftFrame:
    """Planar frame of a loft: origin and two in-plane axes."""

    def __init__(self, origin: Point, xaxis: Vector, yaxis: Vector):
        """Construct from origin and axes."""

        self.origin = origin
        self.xaxis = xaxis
        self.yaxis = yaxis


class _LoftRing:
    """Offset and length of one ring inside a flat point list."""

    def __init__(self, off: int, n: int):
        """Construct from offset and length."""

        self.off = off
        self.n = n


class _LoftPoly:
    """Bottom and top rings of one lofted polygon."""

    def __init__(self, bot: _LoftRing, top: _LoftRing):
        """Construct from bottom and top rings."""

        self.bot = bot
        self.top = top


def _loft_project(frame: _LoftFrame, p: Point) -> tuple[float, float]:
    """Return the 2D coordinates of p in the frame."""

    d = p - frame.origin

    return (d.dot(frame.xaxis), d.dot(frame.yaxis))


def _loft_open_points(pl: Polyline) -> list[Point]:
    """Polyline points without the closing duplicate."""

    pts = pl.get_points()

    if len(pts) > 1:
        f = pts[0]
        b = pts[-1]

        if (
            abs(f[0] - b[0]) < 1e-12
            and abs(f[1] - b[1]) < 1e-12
            and abs(f[2] - b[2]) < 1e-12
        ):
            pts.pop()

    return pts


def _loft_signed_area(frame: _LoftFrame, pts: list[Point]) -> float:
    """Return the signed area of pts projected on the frame."""

    pts2d = []

    for p in pts:
        pts2d.append(_loft_project(frame, p))

    return _signed_area_2d(pts2d) * 0.5


def _loft_border_index(polylines: list[Polyline]) -> int:
    """Index of the polyline with the largest bbox diagonal."""

    border_idx = 0
    max_diag = 0.0

    for i in range(len(polylines)):
        pts = polylines[i].get_points()

        if not pts:
            continue

        minx = pts[0][0]
        miny = pts[0][1]
        minz = pts[0][2]
        maxx = minx
        maxy = miny
        maxz = minz

        for p in pts:
            minx = min(minx, p[0])
            maxx = max(maxx, p[0])
            miny = min(miny, p[1])
            maxy = max(maxy, p[1])
            minz = min(minz, p[2])
            maxz = max(maxz, p[2])

        dx = maxx - minx
        dy = maxy - miny
        dz = maxz - minz
        diag = math.sqrt(dx * dx + dy * dy + dz * dz)

        if diag > max_diag:
            max_diag = diag
            border_idx = i

    return border_idx


def _loft_frame(bottom: Polyline, top: Polyline) -> _LoftFrame:
    """Projection frame of the border polyline, y flipped so z points from bottom to top."""

    origin, xaxis, yaxis, zaxis = bottom.get_average_plane()
    c0 = bottom.center()
    c1 = top.center()
    bottom_to_top = c1 - c0

    if zaxis.dot(bottom_to_top) < 0:
        yaxis = -yaxis

    return _LoftFrame(origin, xaxis, yaxis)


def _loft_add_vkeys(mesh: "Mesh", pts: list[Point]) -> list[int]:
    """Vertex keys of pts; consecutive same-position points share one key."""

    keys = []

    for i in range(len(pts)):
        if i > 0:
            prev = pts[i - 1]
            curr = pts[i]
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            dz = curr[2] - prev[2]

            if dx * dx + dy * dy + dz * dz < 1e-20:
                keys.append(keys[-1])
                continue

        keys.append(mesh.add_vertex(pts[i]))

    return keys


def _loft_split_triangle(tris: list[list[int]], j: int, a: int, c: int, b: int) -> None:
    """Split triangle j, which spans corners a and c, at boundary vertex b."""

    ft = tris[j]

    if (ft[0] == a or ft[0] == c) and (ft[1] == a or ft[1] == c):
        t1 = [ft[0], b, ft[2]]
        t2 = [b, ft[1], ft[2]]
    elif (ft[1] == a or ft[1] == c) and (ft[2] == a or ft[2] == c):
        t1 = [ft[0], ft[1], b]
        t2 = [ft[0], b, ft[2]]
    else:
        t1 = [ft[0], ft[1], b]
        t2 = [b, ft[1], ft[2]]

    tris[j] = t1
    tris.append(t2)


def _loft_fix_collinear(tris: list[list[int]], fvkeys: list[int]) -> None:
    """Insert boundary vertices the CDT skipped as collinear, one per pass."""

    n = len(fvkeys)

    for _pass in range(n):
        used = set()

        for t in tris:
            for v in t:
                used.add(v)

        changed = False

        for k in range(n):
            if changed:
                break

            b = fvkeys[k]

            if b in used:
                continue

            a = fvkeys[(k + n - 1) % n]
            c = fvkeys[(k + 1) % n]

            for j in range(len(tris)):
                has_a = tris[j][0] == a or tris[j][1] == a or tris[j][2] == a
                has_c = tris[j][0] == c or tris[j][1] == c or tris[j][2] == c

                if not has_a or not has_c:
                    continue

                _loft_split_triangle(tris, j, a, c, b)
                changed = True
                break

        if not changed:
            return


def _loft_drop_degenerate(
    tris: list[list[int]], mesh: "Mesh", frame: _LoftFrame
) -> list[list[int]]:
    """Drop triangles with zero area in the projected integer grid."""

    sc = 1e6
    kept = []

    for t in tris:
        u0, v0 = _loft_project(frame, mesh.vertex[t[0]].position())
        u1, v1 = _loft_project(frame, mesh.vertex[t[1]].position())
        u2, v2 = _loft_project(frame, mesh.vertex[t[2]].position())
        iu0 = _round_half_away(u0 * sc)
        iv0 = _round_half_away(v0 * sc)
        iu1 = _round_half_away(u1 * sc)
        iv1 = _round_half_away(v1 * sc)
        iu2 = _round_half_away(u2 * sc)
        iv2 = _round_half_away(v2 * sc)

        if (iu1 - iu0) * (iv2 - iv0) - (iv1 - iv0) * (iu2 - iu0) != 0:
            kept.append(t)

    return kept


def _loft_cap_outer(ring: _LoftRing, vkeys: list[int]) -> list[int]:
    """Point indices of the outer ring, skipping consecutive points that share a vertex key."""

    outer = []

    for i in range(ring.n):
        vi = ring.off + i

        if outer and vkeys[vi] == vkeys[outer[-1]]:
            continue

        outer.append(vi)

    return outer


def _loft_cap_triangles(
    frame: _LoftFrame,
    rings: list[_LoftRing],
    pts: list[Point],
    vkeys: list[int],
    outer: list[int],
    reverse: bool,
) -> list[list[int]]:
    """CDT triangles of the outer ring with its holes as vertex keys, reversed for the bottom."""

    border_2d = []

    for vi in outer:
        u, v = _loft_project(frame, pts[vi])
        border_2d.append(Point(u, v, 0.0))

    flat = list(outer)
    holes_2d = []

    for h in range(1, len(rings)):
        hole = []

        for i in range(rings[h].off, rings[h].off + rings[h].n):
            u, v = _loft_project(frame, pts[i])
            hole.append(Point(u, v, 0.0))
            flat.append(i)

        holes_2d.append(hole)

    tris = cdt_triangulate(border_2d, holes_2d)
    tri_list = []

    for t in tris:
        if reverse:
            tri_list.append([vkeys[flat[t[0]]], vkeys[flat[t[2]]], vkeys[flat[t[1]]]])
        else:
            tri_list.append([vkeys[flat[t[0]]], vkeys[flat[t[1]]], vkeys[flat[t[2]]]])

    return tri_list


def _loft_cap_holes(rings: list[_LoftRing], vkeys: list[int]) -> list[list[int]]:
    """Vertex keys of the hole rings, every ring after the first."""

    hole_rings = []

    for h in range(1, len(rings)):
        ring = []

        for i in range(rings[h].off, rings[h].off + rings[h].n):
            ring.append(vkeys[i])

        hole_rings.append(ring)

    return hole_rings


def _loft_cap(
    mesh: "Mesh",
    frame: _LoftFrame,
    rings: list[_LoftRing],
    pts: list[Point],
    vkeys: list[int],
    reverse: bool,
    fix_collinear: bool,
) -> None:
    """One n-gon cap with stored CDT triangulation and hole rings; reversed for the bottom."""

    outer = _loft_cap_outer(rings[0], vkeys)
    tri_list = _loft_cap_triangles(frame, rings, pts, vkeys, outer, reverse)
    fvkeys = []

    for i in range(len(outer)):
        fvkeys.append(vkeys[outer[len(outer) - 1 - i if reverse else i]])

    fk = mesh.add_face(fvkeys)

    if fk is None:
        return

    if fix_collinear:
        _loft_fix_collinear(tri_list, fvkeys)
        tri_list = _loft_drop_degenerate(tri_list, mesh, frame)

    hole_rings = _loft_cap_holes(rings, vkeys)
    mesh.set_face_triangulation(fk, tri_list)

    if hole_rings and tri_list:
        mesh.set_face_holes(fk, hole_rings)


def _loft_edge_sq_2d(frame: _LoftFrame, pts: list[Point], i: int) -> float:
    """Squared 2D length of edge i of a ring."""

    j = (i + 1) % len(pts)
    xi, yi = _loft_project(frame, pts[i])
    xj, yj = _loft_project(frame, pts[j])
    dx = xj - xi
    dy = yj - yi

    return dx * dx + dy * dy


def _loft_wall_start(
    frame: _LoftFrame, bpts: list[Point], tpts: list[Point]
) -> tuple[int, int]:
    """Start offsets (ia, ib): the longest bottom edge, and the top offset that minimizes the projected gap."""

    bot_n = len(bpts)
    top_n = len(tpts)
    ia = 0
    ib = 0
    max_b = 0.0

    for k in range(bot_n):
        v = _loft_edge_sq_2d(frame, bpts, k)

        if v > max_b:
            max_b = v
            ia = k

    if bot_n != top_n:
        return (ia, ib)

    min_total = math.inf

    for cand in range(top_n):
        total = 0.0

        for k in range(bot_n):
            xb, yb = _loft_project(frame, bpts[(ia + k) % bot_n])
            xt, yt = _loft_project(frame, tpts[(cand + k) % top_n])
            total += (xt - xb) * (xt - xb) + (yt - yb) * (yt - yb)

        if total < min_total:
            min_total = total
            ib = cand

    return (ia, ib)


def _loft_same_point(a: Point, b: Point) -> bool:
    """Return whether a and b coincide within 1e-10."""

    return (
        abs(a[0] - b[0]) < 1e-10
        and abs(a[1] - b[1]) < 1e-10
        and abs(a[2] - b[2]) < 1e-10
    )


def _loft_walls_quads(
    mesh: "Mesh",
    poly: _LoftPoly,
    ia: int,
    ib: int,
    bpts: list[Point],
    tpts: list[Point],
    bot_vkeys: list[int],
    top_vkeys: list[int],
) -> None:
    """Quad walls for equal counts; a collapsed bottom or top edge gives a triangle."""

    bot_n = poly.bot.n
    top_n = poly.top.n

    for k in range(bot_n):
        cb = poly.bot.off + (ia + k) % bot_n
        ct = poly.top.off + (ib + k) % top_n
        nb = poly.bot.off + (ia + k + 1) % bot_n
        nt = poly.top.off + (ib + k + 1) % top_n
        bot_col = _loft_same_point(bpts[(ia + k) % bot_n], bpts[(ia + k + 1) % bot_n])
        top_col = _loft_same_point(tpts[(ib + k) % top_n], tpts[(ib + k + 1) % top_n])

        if bot_col and top_col:
            continue
        elif bot_col:
            mesh.add_face([bot_vkeys[cb], top_vkeys[nt], top_vkeys[ct]])
        elif top_col:
            mesh.add_face([bot_vkeys[cb], bot_vkeys[nb], top_vkeys[ct]])
        else:
            mesh.add_face([bot_vkeys[cb], bot_vkeys[nb], top_vkeys[nt], top_vkeys[ct]])


def _loft_arcs(pts: list[Point], start: int) -> list[float]:
    """Normalized arc lengths of a ring starting at offset start."""

    n = len(pts)
    arcs = [0.0] * (n + 1)

    for k in range(n):
        i = (start + k) % n
        j = (start + k + 1) % n
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        dz = pts[j][2] - pts[i][2]
        arcs[k + 1] = arcs[k] + math.sqrt(dx * dx + dy * dy + dz * dz)

    inv = 1.0 / arcs[n] if arcs[n] > 0 else 1.0

    for k in range(n + 1):
        arcs[k] *= inv

    return arcs


def _loft_walls_zipper(
    mesh: "Mesh",
    poly: _LoftPoly,
    ia: int,
    ib: int,
    bpts: list[Point],
    tpts: list[Point],
    bot_vkeys: list[int],
    top_vkeys: list[int],
) -> None:
    """Zipper walls for unequal counts: quads where arc lengths meet, triangles elsewhere."""

    bot_n = poly.bot.n
    top_n = poly.top.n
    b_arcs = _loft_arcs(bpts, ia)
    t_arcs = _loft_arcs(tpts, ib)
    bi = 0
    ti = 0

    while bi < bot_n or ti < top_n:
        cb = poly.bot.off + (ia + bi) % bot_n
        ct = poly.top.off + (ib + ti) % top_n
        nb = poly.bot.off + (ia + bi + 1) % bot_n
        nt = poly.top.off + (ib + ti + 1) % top_n

        if bi >= bot_n:
            mesh.add_face([bot_vkeys[cb], top_vkeys[ct], top_vkeys[nt]])
            ti += 1
        elif ti >= top_n:
            mesh.add_face([bot_vkeys[cb], bot_vkeys[nb], top_vkeys[ct]])
            bi += 1
        elif abs(b_arcs[bi + 1] - t_arcs[ti + 1]) < 1e-9:
            mesh.add_face([bot_vkeys[cb], bot_vkeys[nb], top_vkeys[nt], top_vkeys[ct]])
            bi += 1
            ti += 1
        elif b_arcs[bi + 1] < t_arcs[ti + 1]:
            mesh.add_face([bot_vkeys[cb], bot_vkeys[nb], top_vkeys[ct]])
            bi += 1
        else:
            mesh.add_face([bot_vkeys[cb], top_vkeys[ct], top_vkeys[nt]])
            ti += 1


def _loft_walls(
    mesh: "Mesh",
    frame: _LoftFrame,
    poly: _LoftPoly,
    all_bot: list[Point],
    all_top: list[Point],
    bot_vkeys: list[int],
    top_vkeys: list[int],
) -> None:
    """Add the wall faces of one polygon, as quads when the rings match in size, zipped otherwise."""

    bpts = all_bot[poly.bot.off : poly.bot.off + poly.bot.n]
    tpts = all_top[poly.top.off : poly.top.off + poly.top.n]
    ia, ib = _loft_wall_start(frame, bpts, tpts)

    if poly.bot.n == poly.top.n:
        _loft_walls_quads(mesh, poly, ia, ib, bpts, tpts, bot_vkeys, top_vkeys)
    else:
        _loft_walls_zipper(mesh, poly, ia, ib, bpts, tpts, bot_vkeys, top_vkeys)


def _loft_order(n: int, border_idx: int) -> list[int]:
    """Polygon order with the border polygon first and the rest in input order."""

    order = [border_idx]

    for i in range(n):
        if i != border_idx:
            order.append(i)

    return order


def _loft_rings(polys: list[_LoftPoly], top: bool) -> list[_LoftRing]:
    """Top or bottom rings of every lofted polygon."""

    rings = []

    for poly in polys:
        rings.append(poly.top if top else poly.bot)

    return rings


# ═══════════════════════════════════════════════════════════════════════════
# Loft panels
# ═══════════════════════════════════════════════════════════════════════════


def _lp_merge_collinear(
    pts: list[Point], vkeys: list[int]
) -> tuple[list[Point], list[int]]:
    """Drop ring points collinear with their neighbors, until none is left."""

    tol = Tolerance.APPROXIMATION
    zt2 = Tolerance.ZERO_TOLERANCE * Tolerance.ZERO_TOLERANCE
    bound = len(pts)

    for _pass in range(bound):
        m = len(pts)

        if m < 3:
            return pts, vkeys

        changed = False
        np_ = []
        nk = []

        for i in range(m):
            p = (i + m - 1) % m
            nx = (i + 1) % m
            ax = pts[i][0] - pts[p][0]
            ay = pts[i][1] - pts[p][1]
            az = pts[i][2] - pts[p][2]
            bx = pts[nx][0] - pts[i][0]
            by = pts[nx][1] - pts[i][1]
            bz = pts[nx][2] - pts[i][2]
            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx
            a2 = ax * ax + ay * ay + az * az
            b2 = bx * bx + by * by + bz * bz

            if (
                a2 < zt2
                or b2 < zt2
                or cx * cx + cy * cy + cz * cz < tol * tol * a2 * b2
            ):
                changed = True
            else:
                np_.append(pts[i])
                nk.append(vkeys[i])

        pts = np_
        vkeys = nk

        if not changed:
            return pts, vkeys

    return pts, vkeys


def _lp_merge_close(
    pts: list[Point], vkeys: list[int]
) -> tuple[list[Point], list[int]]:
    """Drop ring points closer than a thousandth of the longest edge to their predecessor."""

    sz = len(pts)
    max_edge = 0.0

    for i in range(sz):
        max_edge = max(max_edge, pts[i].distance(pts[(i + 1) % sz]))

    stol = max_edge * 0.001
    tp = []
    tk = []

    for i in range(sz):
        if not tp or tp[-1].distance(pts[i]) > stol:
            tp.append(pts[i])
            tk.append(vkeys[i])

    while len(tp) >= 3 and tp[-1].distance(tp[0]) <= stol:
        tp.pop()
        tk.pop()

    if len(tp) < 3:
        return pts, vkeys

    return tp, tk


def _lp_offset_toward(p: Point, cx: float, cy: float, cz: float, gap: float) -> Point:
    """Return p moved by gap toward (cx, cy, cz)."""

    dx = cx - p[0]
    dy = cy - p[1]
    dz = cz - p[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)

    if length > 1e-10:
        dx *= gap / length
        dy *= gap / length
        dz *= gap / length

    return p + Vector(dx, dy, dz)


def _lp_face_centroid(m: "Mesh", fk: int) -> Point:
    """Return the average of the face vertex positions."""

    vkeys = m.face_vertices(fk)
    cx = 0.0
    cy = 0.0
    cz = 0.0

    for vk in vkeys:
        p = m.vertex_point(vk)
        cx += p[0]
        cy += p[1]
        cz += p[2]

    return Point(cx / len(vkeys), cy / len(vkeys), cz / len(vkeys))


def _lp_match_faces(top_mesh: "Mesh", bot_mesh: "Mesh") -> list[tuple[int, int]]:
    """Greedy top/bottom face pairs by centroid distance, sorted by key."""

    tfks = top_mesh.faces()
    bfks = bot_mesh.faces()
    dists = []

    for ti in range(len(tfks)):
        for bi in range(len(bfks)):
            dists.append(
                (
                    _lp_face_centroid(top_mesh, tfks[ti]).distance(
                        _lp_face_centroid(bot_mesh, bfks[bi])
                    ),
                    ti,
                    bi,
                )
            )

    dists.sort()
    top_used = [False] * len(tfks)
    bot_used = [False] * len(bfks)
    face_match = []

    for _d, ti, bi in dists:
        if top_used[ti] or bot_used[bi]:
            continue

        face_match.append((tfks[ti], bfks[bi]))
        top_used[ti] = True
        bot_used[bi] = True

    face_match.sort()

    return face_match


def _lp_orient_rings(
    top_pts: list[Point],
    top_vkeys: list[int],
    bot_pts: list[Point],
    bot_vkeys: list[int],
) -> None:
    """Reverse top and bottom rings so the top normal points from bottom to top and the bottom normal away."""

    tc = _ring_centroid(top_pts)
    bc = _ring_centroid(bot_pts)
    axis = tc - bc

    if not axis.normalize_self():
        return

    if _newell_normal(top_pts).dot(axis) < 0:
        top_pts.reverse()
        top_vkeys.reverse()

    if _newell_normal(bot_pts).dot(axis) > 0:
        bot_pts.reverse()
        bot_vkeys.reverse()


def _lp_add_cap(mesh: "Mesh", cap: list[int], pts: list[Point]) -> int | None:
    """Cap face over local keys with its planar CDT stored."""

    fk = mesh.add_face(cap)

    if fk is None or len(cap) < 3:
        return fk

    tris = _planar_cdt(pts)

    if not tris:
        return fk

    tri_list = []

    for t in tris:
        tri_list.append([cap[t[0]], cap[t[1]], cap[t[2]]])

    mesh.set_face_triangulation(fk, tri_list)

    return fk


def _lp_midpoints(pts: list[Point]) -> list[Point]:
    """Edge midpoints of a ring."""

    n = len(pts)
    mids = []

    for i in range(n):
        mids.append(
            Point(
                (pts[i][0] + pts[(i + 1) % n][0]) * 0.5,
                (pts[i][1] + pts[(i + 1) % n][1]) * 0.5,
                (pts[i][2] + pts[(i + 1) % n][2]) * 0.5,
            )
        )

    return mids


def _lp_nearest(p: Point, pts: list[Point]) -> int:
    """Index of the point in pts nearest to p."""

    best_d = math.inf
    best = 0

    for i in range(len(pts)):
        d = p.distance(pts[i])

        if d < best_d:
            best_d = d
            best = i

    return best


def _lp_add_quad(
    panel: LoftPanel, b0: int, b1: int, t0: int, t1: int, edge_gap: float
) -> int | None:
    """Quad wall over matched edge j of the bottom and ti of the top, inset by edge_gap."""

    if edge_gap <= 0.0:
        return panel.mesh.add_face([b0, t1, t0, b1])

    pb0 = panel.mesh.vertex_point(b0)
    pb1 = panel.mesh.vertex_point(b1)
    pt0 = panel.mesh.vertex_point(t0)
    pt1 = panel.mesh.vertex_point(t1)
    cx = (pb0[0] + pb1[0] + pt0[0] + pt1[0]) * 0.25
    cy = (pb0[1] + pb1[1] + pt0[1] + pt1[1]) * 0.25
    cz = (pb0[2] + pb1[2] + pt0[2] + pt1[2]) * 0.25
    nb0 = panel.mesh.add_vertex(_lp_offset_toward(pb0, cx, cy, cz, edge_gap))
    nb1 = panel.mesh.add_vertex(_lp_offset_toward(pb1, cx, cy, cz, edge_gap))

    return panel.mesh.add_face([nb0, t1, t0, nb1])


def _lp_add_top_triangles(
    panel: LoftPanel,
    top_mids: list[Point],
    top_vkeys: list[int],
    bot_pts: list[Point],
    bot_vkeys: list[int],
    top_used: list[bool],
) -> None:
    """A triangle from every unmatched top edge to the nearest bottom vertex."""

    n = len(top_vkeys)

    for i in range(n):
        if top_used[i]:
            continue

        t0 = panel.orig_top_to_local[top_vkeys[i]]
        t1 = panel.orig_top_to_local[top_vkeys[(i + 1) % n]]
        bv = panel.orig_bot_to_local[bot_vkeys[_lp_nearest(top_mids[i], bot_pts)]]
        fk = panel.mesh.add_face([t1, t0, bv])

        if fk is not None:
            w = LoftWallFace()
            w.face_key = fk
            panel.wall_faces.append(w)


def _lp_add_quad_wall(
    panel: LoftPanel,
    top_vkeys: list[int],
    bot_vkeys: list[int],
    j: int,
    ti: int,
    edge_gap: float,
) -> None:
    """Quad wall between bottom edge j and its matched top edge ti, recorded with the original keys it spans."""

    n = len(top_vkeys)
    m = len(bot_vkeys)
    b0 = panel.orig_bot_to_local[bot_vkeys[j]]
    b1 = panel.orig_bot_to_local[bot_vkeys[(j + 1) % m]]
    t0 = panel.orig_top_to_local[top_vkeys[ti]]
    t1 = panel.orig_top_to_local[top_vkeys[(ti + 1) % n]]
    fk = _lp_add_quad(panel, b0, b1, t0, t1, edge_gap)

    if fk is None:
        return

    w = LoftWallFace()
    w.face_key = fk
    w.is_quad = True
    w.top_v0 = top_vkeys[ti]
    w.top_v1 = top_vkeys[(ti + 1) % n]
    w.bot_v0 = bot_vkeys[(j + 1) % m]
    w.bot_v1 = bot_vkeys[j]
    panel.wall_faces.append(w)


def _lp_add_walls(
    panel: LoftPanel,
    top_pts: list[Point],
    top_vkeys: list[int],
    bot_pts: list[Point],
    bot_vkeys: list[int],
    edge_gap: float,
    edge_match_threshold: float,
    skip_triangles: bool,
) -> None:
    """Walls of one panel: a quad per mutually nearest edge pair, a triangle for every unmatched edge."""

    n = len(top_pts)
    m = len(bot_pts)
    top_mids = _lp_midpoints(top_pts)
    bot_mids = _lp_midpoints(bot_pts)
    bot_to_top = [0] * m
    top_to_bot = [0] * n
    bot_dist = [0.0] * m

    for j in range(m):
        bot_to_top[j] = _lp_nearest(bot_mids[j], top_mids)
        bot_dist[j] = bot_mids[j].distance(top_mids[bot_to_top[j]])

    for i in range(n):
        top_to_bot[i] = _lp_nearest(top_mids[i], bot_mids)

    avg = 0.0

    for j in range(m):
        avg += bot_dist[j]

    threshold = avg / m * edge_match_threshold
    top_used = [False] * n

    for j in range(m):
        b0 = panel.orig_bot_to_local[bot_vkeys[j]]
        b1 = panel.orig_bot_to_local[bot_vkeys[(j + 1) % m]]
        ti = bot_to_top[j]

        if bot_dist[j] <= threshold and top_to_bot[ti] == j:
            _lp_add_quad_wall(panel, top_vkeys, bot_vkeys, j, ti, edge_gap)
            top_used[ti] = True
        elif not skip_triangles:
            tv = panel.orig_top_to_local[top_vkeys[_lp_nearest(bot_mids[j], top_pts)]]
            fk = panel.mesh.add_face([b0, tv, b1])

            if fk is not None:
                w = LoftWallFace()
                w.face_key = fk
                panel.wall_faces.append(w)

    if not skip_triangles:
        _lp_add_top_triangles(panel, top_mids, top_vkeys, bot_pts, bot_vkeys, top_used)


def _lp_assign_roles(panel: LoftPanel) -> None:
    """Face index of every wall and the role of every wall and cap face."""

    fkey_to_idx = {}
    fi = 0

    for fk in panel.mesh.faces():
        fkey_to_idx[fk] = fi
        fi += 1

    for w in panel.wall_faces:
        w.face_index = fkey_to_idx[w.face_key]
        panel.face_roles[w.face_key] = (
            LoftFaceRole.QuadWall if w.is_quad else LoftFaceRole.TriWall
        )

    if panel.top_face_key is not None:
        panel.face_roles[panel.top_face_key] = LoftFaceRole.TopCap

    if panel.bot_face_key is not None:
        panel.face_roles[panel.bot_face_key] = LoftFaceRole.BotCap


def _lp_build_panel(
    top_mesh: "Mesh",
    bot_mesh: "Mesh",
    tfk: int,
    bfk: int,
    edge_gap: float,
    edge_match_threshold: float,
    add_caps: bool,
    skip_triangles: bool,
) -> LoftPanel:
    """One panel between matched faces tfk of the top mesh and bfk of the bottom mesh."""

    panel = LoftPanel()
    top_vkeys = list(top_mesh.face_vertices(tfk))
    bot_vkeys = list(bot_mesh.face_vertices(bfk))
    top_pts = []
    bot_pts = []

    for vk in top_vkeys:
        top_pts.append(top_mesh.vertex_point(vk))

    for vk in bot_vkeys:
        bot_pts.append(bot_mesh.vertex_point(vk))

    top_pts, top_vkeys = _lp_merge_collinear(top_pts, top_vkeys)
    bot_pts, bot_vkeys = _lp_merge_collinear(bot_pts, bot_vkeys)
    top_pts, top_vkeys = _lp_merge_close(top_pts, top_vkeys)
    _lp_orient_rings(top_pts, top_vkeys, bot_pts, bot_vkeys)

    for i in range(len(top_pts)):
        lk = panel.mesh.add_vertex(top_pts[i])
        panel.orig_top_to_local[top_vkeys[i]] = lk
        panel.top_vertices.append(lk)

    for j in range(len(bot_pts)):
        lk = panel.mesh.add_vertex(bot_pts[j])
        panel.orig_bot_to_local[bot_vkeys[j]] = lk
        panel.bot_vertices.append(lk)

    if add_caps:
        panel.top_face_key = _lp_add_cap(panel.mesh, panel.top_vertices, top_pts)

    _lp_add_walls(
        panel,
        top_pts,
        top_vkeys,
        bot_pts,
        bot_vkeys,
        edge_gap,
        edge_match_threshold,
        skip_triangles,
    )

    if add_caps:
        panel.bot_face_key = _lp_add_cap(panel.mesh, panel.bot_vertices, bot_pts)

    _lp_assign_roles(panel)

    return panel


def _lp_adjacency(panels: list[LoftPanel]) -> list[LoftAdjPair]:
    """Quad walls of different panels that share a top edge, once per pair."""

    edge_to_wall = {}

    for pi in range(len(panels)):
        for wi in range(len(panels[pi].wall_faces)):
            w = panels[pi].wall_faces[wi]

            if w.is_quad:
                edge_to_wall[(w.top_v0, w.top_v1)] = (pi, wi)

    adjacency = []

    for pi in range(len(panels)):
        for wi in range(len(panels[pi].wall_faces)):
            w = panels[pi].wall_faces[wi]

            if not w.is_quad:
                continue

            key = (w.top_v1, w.top_v0)

            if key in edge_to_wall and edge_to_wall[key][0] > pi:
                adjacency.append(
                    LoftAdjPair(pi, wi, edge_to_wall[key][0], edge_to_wall[key][1])
                )

    return adjacency


def _lp_ordered_mesh(panels: list[LoftPanel], top: bool) -> "Mesh":
    """One face per panel, from its local top or bottom ring."""

    ordered = Mesh()

    for i in range(len(panels)):
        ring = panels[i].top_vertices if top else panels[i].bot_vertices
        vks = []

        for lk in ring:
            vks.append(ordered.add_vertex(panels[i].mesh.vertex_point(lk)))

        ordered.add_face(vks, i)

    return ordered


# ═══════════════════════════════════════════════════════════════════════════
# Miter contours
# ═══════════════════════════════════════════════════════════════════════════


def _fold_chamfer_mask(pts: list[Point], max_angle_deg: float) -> list[bool]:
    """Corners whose interior angle is below max_angle_deg."""

    n = len(pts)
    mask = [False] * n

    for i in range(n):
        prev = (i + n - 1) % n
        nxt = (i + 1) % n
        dp = pts[prev] - pts[i]
        dn = pts[nxt] - pts[i]
        lp = dp.magnitude()
        ln = dn.magnitude()

        if lp < 1e-12 or ln < 1e-12:
            continue

        cos_a = max(-1.0, min(1.0, dp.dot(dn) / (lp * ln)))
        mask[i] = math.acos(cos_a) * Tolerance.TO_DEGREES < max_angle_deg

    return mask


def _fold_chamfer(pts: list[Point], s: float, mask: list[bool]) -> list[Point]:
    """Ring with every masked corner cut back by s, capped at a third of the shortest edge."""

    n = len(pts)

    if s <= 0.0:
        return list(pts)

    min_edge = math.inf

    for i in range(n):
        j = (i + 1) % n
        d = pts[j] - pts[i]
        min_edge = min(min_edge, d.magnitude())

    sc = min(s, min_edge / 3.0)
    result = []

    for i in range(n):
        if not mask[i]:
            result.append(pts[i])
            continue

        prev = (i + n - 1) % n
        nxt = (i + 1) % n
        dp = pts[prev] - pts[i]
        dn = pts[nxt] - pts[i]
        lp = dp.magnitude()
        ln = dn.magnitude()
        sp = sc / lp if lp > 1e-12 else 0.0
        sn = sc / ln if ln > 1e-12 else 0.0
        result.append((pts[i] + dp * sp))
        result.append((pts[i] + dn * sn))

    return result


def _fold_face_normal(mesh: "Mesh", fk: int) -> Vector:
    """Newell normal of a face, +z when the face is missing."""

    pts = mesh.face_points(fk)

    if pts is None:
        return Vector(0.0, 0.0, 1.0)

    return _newell_normal(pts)


def _miter_planes(
    shell: "Mesh", efm: dict, fverts: list[int], pts: list[Point], fn: Vector
) -> list[Plane]:
    """One miter plane per edge, through the edge midpoint along the averaged neighbor normal."""

    n = len(fverts)
    planes = []

    for i in range(n):
        j = (i + 1) % n
        avg_n = fn
        adj_fk = efm.get((fverts[j], fverts[i]))

        if adj_fk is not None:
            adj = _fold_face_normal(shell, adj_fk)
            total = fn + adj

            if total.magnitude() > 0.1 and total.normalize_self():
                avg_n = total

        edge_dir = pts[j] - pts[i]

        if not edge_dir.normalize_self():
            return []

        mn = avg_n.cross(edge_dir)

        if not mn.normalize_self():
            return []

        mid = Point(
            (pts[i][0] + pts[j][0]) / 2,
            (pts[i][1] + pts[j][1]) / 2,
            (pts[i][2] + pts[j][2]) / 2,
        )
        planes.append(Plane.from_point_normal(mid, mn))

    return planes


def _miter_contour(corner_lines: list[Line], plane: Plane) -> list[Point]:
    """Where the corner lines pierce a plane, empty when any misses."""

    from .intersection import line_plane

    contour = []

    for line in corner_lines:
        p = line_plane(line, plane, False)

        if p is None:
            return []

        contour.append(p)

    return contour


# ═══════════════════════════════════════════════════════════════════════════
# Cutting
# ═══════════════════════════════════════════════════════════════════════════


class _CutFace:
    """One face of a cut and the input face it came from."""

    def __init__(self, rings: list[list[int]], parent: int | None):
        """Construct from rings and parent."""

        self.rings = rings  # Outer ring, then hole rings.
        self.parent = parent  # Input face key, none for a cap.


def _cut_points(ring: list[int], points: dict[int, Point]) -> list[Point]:
    """Positions of a key ring."""

    result = []

    for key in ring:
        result.append(points[key])

    return result


def _cut_area(ring: list[int], uv: dict[int, tuple[float, float]]) -> float:
    """Twice the signed area of a key ring in plane coordinates."""

    area = 0.0

    for i in range(len(ring)):
        a = uv[ring[i]]
        b = uv[ring[(i + 1) % len(ring)]]
        area += a[0] * b[1] - b[0] * a[1]

    return area


def _cut_inside(
    p: tuple[float, float],
    rings: list[list[int]],
    uv: dict[int, tuple[float, float]],
) -> bool:
    """Whether p lies inside the key rings by the even-odd rule."""

    inside = False

    for ring in rings:
        for i in range(len(ring)):
            a = uv[ring[i]]
            b = uv[ring[(i + 1) % len(ring)]]

            if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < a[0] + (p[1] - a[1]) * (
                b[0] - a[0]
            ) / (b[1] - a[1]):
                inside = not inside

    return inside


def _cut_split(walk: list[int], loops: list[list[int]]) -> None:
    """Split a closed walk into simple loops where it revisits a vertex; loops under three vertices are dropped."""

    stack = []
    index = {}

    for key in walk:
        start = index.get(key)

        if start is None:
            index[key] = len(stack)
            stack.append(key)
            continue

        if len(stack) - start > 2:
            loops.append(stack[start:])

        for k in range(start + 1, len(stack)):
            del index[stack[k]]

        del stack[start + 1 :]

    if len(stack) > 2:
        loops.append(stack)


def _cut_loops(
    edges: dict[int, list[int]], uv: dict[int, tuple[float, float]]
) -> list[list[int]]:
    """Closed loops of directed edges, turning sharpest left where loops meet; open chains are dropped."""

    loops = []

    while edges:
        first = min(edges)
        walk = [first]
        prev = first
        cur = edges[first].pop()

        if not edges[first]:
            del edges[first]

        while cur != walk[0]:
            targets = edges.get(cur)

            if targets is None:
                walk.clear()
                break

            walk.append(cur)
            ax = uv[cur][0] - uv[prev][0]
            ay = uv[cur][1] - uv[prev][1]
            best = 0
            turn = -4.0

            for j in range(len(targets)):
                bx = uv[targets[j]][0] - uv[cur][0]
                by = uv[targets[j]][1] - uv[cur][1]
                angle = math.atan2(ax * by - ay * bx, ax * bx + ay * by)

                if angle > turn:
                    turn = angle
                    best = j

            prev = cur
            cur = targets.pop(best)

            if not targets:
                del edges[prev]

        _cut_split(walk, loops)

    return loops


def _cut_regions(
    loops: list[list[int]], uv: dict[int, tuple[float, float]]
) -> list[_CutFace]:
    """Loops wound like the largest one become outer rings, each taking the opposite-wound loops inside it as holes."""

    areas = []
    largest = 0.0

    for loop in loops:
        areas.append(_cut_area(loop, uv))

        if abs(areas[-1]) > abs(largest):
            largest = areas[-1]

    regions = []
    sizes = []

    for i in range(len(loops)):
        if areas[i] * largest > 0.0:
            regions.append(_CutFace([loops[i]], None))
            sizes.append(abs(areas[i]))

    for i in range(len(loops)):
        if areas[i] * largest >= 0.0:
            continue

        a = uv[loops[i][0]]
        b = uv[loops[i][1]]
        mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        owner = None

        for r in range(len(regions)):
            if _cut_inside(mid, [regions[r].rings[0]], uv) and (
                owner is None or sizes[r] < sizes[owner]
            ):
                owner = r

        if owner is not None:
            regions[owner].rings.append(loops[i])

    return regions


def _cut_triangulation(piece: _CutFace, points: dict[int, Point]) -> list[list[int]]:
    """CDT of a face with hole rings, wound like its outer ring; empty when degenerate."""

    normal = _newell_normal(_cut_points(piece.rings[0], points))

    if normal.magnitude() == 0.0:
        return []

    plane = Plane.from_point_normal(points[piece.rings[0][0]], normal)
    frame = _LoftFrame(plane.origin, plane.x_axis, plane.y_axis)
    rings_2d = []
    flat = []

    for ring in piece.rings:
        rings_2d.append([])

        for key in ring:
            u, v = _loft_project(frame, points[key])
            rings_2d[-1].append(Point(u, v, 0.0))
            flat.append(key)

    triangles = []

    for t in cdt_triangulate(rings_2d[0], rings_2d[1:]):
        triangles.append([flat[t[0]], flat[t[1]], flat[t[2]]])

    _loft_fix_collinear(triangles, piece.rings[0])

    return triangles


def _cut_pieces(
    rings: list[list[int]],
    normal: Vector,
    xaxis: Vector,
    distance: dict[int, float],
    points: dict[int, Point],
    tolerance: float,
) -> list[_CutFace]:
    """Kept pieces of one crossed face from its rings with crossing vertices inserted, split along the plane in the face frame."""

    frame = _LoftFrame(points[rings[0][0]], xaxis, normal.cross(xaxis))
    uv = {}
    edges = {}
    lines = set()
    events = set()

    for ring in rings:
        for key in ring:
            uv[key] = _loft_project(frame, points[key])

    for ring in rings:
        for i in range(len(ring)):
            a = ring[i]
            b = ring[(i + 1) % len(ring)]

            if distance[a] == 0.0:
                events.add(a)

            if distance[a] == 0.0 and distance[b] == 0.0:
                lines.add((min(a, b), max(a, b)))

            if (
                distance[a] >= 0.0
                and distance[b] >= 0.0
                and (distance[a] + distance[b] > 0.0 or uv[b][1] < uv[a][1])
            ):
                edges.setdefault(a, []).append(b)

    order = sorted(events, key=lambda key: uv[key][1])

    for i in range(len(order) - 1):
        a = uv[order[i]]
        b = uv[order[i + 1]]
        mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

        if (
            b[1] - a[1] > tolerance
            and (min(order[i], order[i + 1]), max(order[i], order[i + 1])) not in lines
            and _cut_inside(mid, rings, uv)
        ):
            edges.setdefault(order[i + 1], []).append(order[i])

    return _cut_regions(_cut_loops(edges, uv), uv)


def _cut_crossing(
    edge: tuple[int, int],
    crossings: dict[tuple[int, int], int],
    distance: dict[int, float],
    points: dict[int, Point],
    first: int,
) -> int:
    """Key of the vertex where edge crosses the plane, added on first use as first plus the number of crossings so far."""

    edge = (min(edge[0], edge[1]), max(edge[0], edge[1]))

    if edge in crossings:
        return crossings[edge]

    key = first + len(crossings)
    t = distance[edge[0]] / (distance[edge[0]] - distance[edge[1]])
    points[key] = points[edge[0]] + (points[edge[1]] - points[edge[0]]) * t
    distance[key] = 0.0
    crossings[edge] = key

    return key


def _cut_caps(
    faces: dict[int, _CutFace],
    distance: dict[int, float],
    points: dict[int, Point],
    plane: Plane,
) -> list[_CutFace]:
    """Caps closing the loops of unpaired half-edges that lie on the plane."""

    frame = _LoftFrame(plane.origin, plane.y_axis, plane.x_axis)
    halfedges = set()
    section = {}
    uv = {}

    for piece in faces.values():
        for ring in piece.rings:
            for i in range(len(ring)):
                halfedges.add((ring[i], ring[(i + 1) % len(ring)]))

    for edge in sorted(halfedges):
        if (
            distance[edge[0]] == 0.0
            and distance[edge[1]] == 0.0
            and (edge[1], edge[0]) not in halfedges
        ):
            section.setdefault(edge[1], []).append(edge[0])

    for vk, d in distance.items():
        if d == 0.0:
            uv[vk] = _loft_project(frame, points[vk])

    return _cut_regions(_cut_loops(section, uv), uv)


def _cut_rings(
    fk: int,
    ring: list[int],
    face_holes: dict[int, list[list[int]]],
    normal: Vector,
    points: dict[int, Point],
) -> list[list[int]]:
    """Face ring followed by its hole rings, every hole turned against the face normal."""

    rings = [list(ring)]

    for hole in face_holes.get(fk, []):
        rings.append(list(hole))

        if _newell_normal(_cut_points(hole, points)).dot(normal) > 0.0:
            rings[-1].reverse()

    return rings


def _cut_split_rings(
    rings: list[list[int]],
    crossings: dict[tuple[int, int], int],
    distance: dict[int, float],
    points: dict[int, Point],
    first: int,
) -> list[list[int]]:
    """Rings with the crossing vertex inserted after every edge that crosses the plane."""

    split = []

    for r in rings:
        split.append([])

        for i in range(len(r)):
            split[-1].append(r[i])

            if distance[r[i]] * distance[r[(i + 1) % len(r)]] < 0.0:
                split[-1].append(
                    _cut_crossing(
                        (r[i], r[(i + 1) % len(r)]), crossings, distance, points, first
                    )
                )

    return split


def _cut_face(
    fk: int,
    rings: list[list[int]],
    normal: Vector,
    plane: Plane,
    crossings: dict[tuple[int, int], int],
    distance: dict[int, float],
    points: dict[int, Point],
    first: int,
    tolerance: float,
) -> list[_CutFace]:
    """Kept pieces of face fk: none below the plane, the whole face above it, the split pieces when it crosses."""

    above = False
    below = False

    for r in rings:
        for key in r:
            above = above or distance[key] > 0.0
            below = below or distance[key] < 0.0

    if not above:
        return []

    if not below:
        return [_CutFace(rings, fk)]

    xaxis = plane.z_axis - normal * plane.z_axis.dot(normal)

    if not xaxis.normalize_self():
        return []

    split = _cut_split_rings(rings, crossings, distance, points, first)
    pieces = _cut_pieces(split, normal, xaxis, distance, points, tolerance)

    for piece in pieces:
        piece.parent = fk

    return pieces


def _cut_result(
    output: dict[int, _CutFace],
    points: dict[int, Point],
    face: dict[int, list[int]],
    facedata: dict[int, dict[str, float]],
    triangulation: dict[int, list[list[int]]],
) -> "Mesh":
    """Mesh of the kept pieces with the parent face data and the triangulation of every untouched face."""

    result = Mesh()
    used = set()

    for piece in output.values():
        for r in piece.rings:
            used.update(r)

    for vk in sorted(used):
        result.add_vertex(points[vk], vk)

    for fk, piece in sorted(output.items()):
        if result.add_face(piece.rings[0], fk) is None:
            continue

        whole = piece.parent is not None and piece.rings[0] == face[piece.parent]

        if len(piece.rings) > 1:
            result.set_face_holes(fk, piece.rings[1:])

        if piece.parent in facedata:
            result.facedata[fk] = dict(facedata[piece.parent])

        if whole and fk in triangulation:
            result.set_face_triangulation(fk, [list(t) for t in triangulation[fk]])

        if not whole and (len(piece.rings) > 1 or len(piece.rings[0]) > 3):
            result.set_face_triangulation(fk, _cut_triangulation(piece, points))

    return result


def _cut_tolerance(points: dict[int, Point]) -> float:
    """Snap distance of the plane test, 1e-9 of the bounding box diagonal."""

    big = math.inf
    low = Point(big, big, big)
    high = Point(-big, -big, -big)

    for p in points.values():
        for k in range(3):
            low[k] = min(low[k], p[k])
            high[k] = max(high[k], p[k])

    return 1e-9 * low.distance(high)


class Mesh:
    """A halfedge mesh data structure for representing polygonal surfaces."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self):
        """Construct an empty mesh."""

        self.halfedge: dict[int, dict[int, int | None]] = {}
        self.vertex: dict[int, VertexData] = {}
        self.face: dict[int, list[int]] = {}
        self.face_holes: dict[int, list[list[int]]] = {}
        self.facedata: dict[int, dict[str, float]] = {}
        self.edgedata: dict[tuple[int, int], dict[str, float]] = {}
        self.default_vertex_attributes: dict[str, float] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
        self.default_face_attributes: dict[str, float] = {}
        self.default_edge_attributes: dict[str, float] = {}
        self._guid: str | None = None
        self.name = "my_mesh"
        self.color_mode = ColorMode.OBJECTCOLOR
        self._pointcolors: list[Color] = []
        self._facecolors: list[Color] = []
        self._linecolors: list[Color] = []
        self._widths: list[float] = []
        self._objectcolor = Color.lightgrey()
        self._max_vertex = 0
        self._max_face = 0
        self.triangulation: dict[int, list[list[int]]] = {}
        self._triangle_bvh_built = False
        self._triangle_bvh: SpatialBVH | None = None
        self._triangle_aabbs_cache: list[AABB] = []
        self._triangle_indices_cache: list[tuple[int, int, int]] = []
        self._triangle_face_subidx_cache: list[tuple[int, int]] = []
        self._vertices_cache: list[Point] = []
        self._triangle_aabb_tree: SpatialAABBTree | None = None

    def __deepcopy__(self, memo):
        """Copy (same guid, same data)."""

        m = Mesh()
        m._guid = self._guid
        m.name = self.name
        m.halfedge = {u: dict(v) for u, v in self.halfedge.items()}

        for k, v in self.vertex.items():
            m.vertex[k] = VertexData(v.position())
            m.vertex[k].attributes = Attributes(v.attributes)

        m.face = {k: list(v) for k, v in self.face.items()}
        m.face_holes = {k: [list(r) for r in v] for k, v in self.face_holes.items()}
        m.facedata = {k: dict(v) for k, v in self.facedata.items()}
        m.edgedata = {k: dict(v) for k, v in self.edgedata.items()}
        m.default_vertex_attributes = dict(self.default_vertex_attributes)
        m.default_face_attributes = dict(self.default_face_attributes)
        m.default_edge_attributes = dict(self.default_edge_attributes)
        m._pointcolors = list(self._pointcolors)
        m._facecolors = list(self._facecolors)
        m._linecolors = list(self._linecolors)
        m._widths = list(self._widths)
        m._objectcolor = self._objectcolor
        m.color_mode = self.color_mode
        m._max_vertex = self._max_vertex
        m._max_face = self._max_face
        m.triangulation = {
            k: [list(t) for t in v] for k, v in self.triangulation.items()
        }
        memo[id(self)] = m

        return m

    def duplicate(self) -> "Mesh":
        """Copy (new guid, same data)."""

        result = copy.deepcopy(self)
        result.refresh_guid()

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def from_vertices_and_faces(
        vertices: list[Point], faces: list[list[int]]
    ) -> "Mesh":
        """Construct from a list of vertices and faces."""

        mesh = Mesh()

        for pt in vertices:
            mesh.add_vertex(pt)

        for f in faces:
            mesh.add_face(f)

        return mesh

    @staticmethod
    def _polylines_vertex_key(
        mesh: "Mesh", p: Point, precision: float | None, map_eps: dict, map_exact: dict
    ) -> int:
        """Vertex key of p, merged by precision grid when given and by exact bits otherwise."""

        if precision is not None:
            key = (
                _round_half_away(p[0] / precision),
                _round_half_away(p[1] / precision),
                _round_half_away(p[2] / precision),
            )

            if key in map_eps:
                return map_eps[key]

            vk = mesh.add_vertex(p)
            map_eps[key] = vk

            return vk

        key = struct.pack("<3d", p[0], p[1], p[2])

        if key in map_exact:
            return map_exact[key]

        vk = mesh.add_vertex(p)
        map_exact[key] = vk

        return vk

    @staticmethod
    def from_polylines(
        polygons: list[list[Point]], precision: float | None = None
    ) -> "Mesh":
        """Construct from a list of polygons, merging vertices within precision when given."""

        mesh = Mesh()
        map_eps = {}
        map_exact = {}

        for poly in polygons:
            if len(poly) < 3:
                continue

            vkeys = []

            for p in poly:
                vkeys.append(
                    Mesh._polylines_vertex_key(mesh, p, precision, map_eps, map_exact)
                )

            if len(vkeys) > 1 and vkeys[-1] == vkeys[0]:
                vkeys.pop()

            if len(vkeys) < 3:
                continue

            fk = mesh.add_face(vkeys)

            if fk is None or len(vkeys) < 4:
                continue

            tris = _planar_cdt(poly[: len(vkeys)])

            if not tris:
                continue

            tri_list = []

            for t in tris:
                tri_list.append([vkeys[t[0]], vkeys[t[1]], vkeys[t[2]]])

            mesh.triangulation[fk] = tri_list

        return mesh

    @staticmethod
    def from_polylines_polyline(
        polylines: list[Polyline], precision: float | None = None
    ) -> "Mesh":
        """Construct from a list of polygons, merging vertices within precision when given."""

        polygons = []

        for polyline in polylines:
            polygons.append(polyline.get_points())

        return Mesh.from_polylines(polygons, precision)

    @staticmethod
    def _lines_precision(pts: list[Point], precision: float | None) -> float:
        """Grid spacing for merging line endpoints: the given precision or a millionth of the bbox diagonal."""

        eps = precision if precision is not None else 0.0

        if eps > 0.0:
            return eps

        minx = pts[0][0]
        miny = pts[0][1]
        minz = pts[0][2]
        maxx = minx
        maxy = miny
        maxz = minz

        for p in pts:
            minx = min(minx, p[0])
            maxx = max(maxx, p[0])
            miny = min(miny, p[1])
            maxy = max(maxy, p[1])
            minz = min(minz, p[2])
            maxz = max(maxz, p[2])

        diag = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2 + (maxz - minz) ** 2)

        return max(diag * 1e-6, 1e-12)

    @staticmethod
    def _lines_vertex_id(p: Point, eps: float, vmap: dict, verts: list[Point]) -> int:
        """Index of p in verts, appending it when its grid cell is new."""

        key = (
            _round_half_away(p[0] / eps),
            _round_half_away(p[1] / eps),
            _round_half_away(p[2] / eps),
        )

        if key in vmap:
            return vmap[key]

        vid = len(verts)
        verts.append(p)
        vmap[key] = vid

        return vid

    @staticmethod
    def _lines_face_cycles(adj: dict[int, list[int]], nv: int) -> list[list[int]]:
        """Face cycles of a planar graph: from u->v the next edge turns to the CW predecessor of u around v."""

        visited = set()
        cycles = []

        for u in sorted(adj):
            for v in adj[u]:
                if (u, v) in visited:
                    continue

                cycle = []
                cu = u
                cv = v
                valid = True

                while len(cycle) <= nv * 2:
                    if (cu, cv) in visited:
                        break

                    visited.add((cu, cv))
                    cycle.append(cu)
                    cv_nbrs = adj.get(cv, [])

                    if cu not in cv_nbrs:
                        valid = False
                        break

                    idx = cv_nbrs.index(cu)
                    prev_idx = len(cv_nbrs) - 1 if idx == 0 else idx - 1
                    cu = cv
                    cv = cv_nbrs[prev_idx]

                if len(cycle) > nv * 2:
                    valid = False

                if valid and len(cycle) >= 3:
                    cycles.append(cycle)

        return cycles

    @staticmethod
    def _lines_outer_cycle(cycles: list[list[int]], verts: list[Point]) -> int:
        """Index of the cycle with the most negative signed area: the outer boundary."""

        min_idx = 0
        min_area = math.inf

        for i in range(len(cycles)):
            pts = []

            for vid in cycles[i]:
                pts.append((verts[vid][0], verts[vid][1]))

            area = _signed_area_2d(pts) * 0.5

            if area < min_area:
                min_area = area
                min_idx = i

        return min_idx

    @staticmethod
    def _lines_sort_neighbors(adj: dict[int, list[int]], verts: list[Point]) -> None:
        """Sort and dedupe every neighbor list counter-clockwise by angle around its vertex."""

        for v in adj:
            nbrs = sorted(set(adj[v]))
            vx = verts[v][0]
            vy = verts[v][1]
            nbrs.sort(key=lambda n: math.atan2(verts[n][1] - vy, verts[n][0] - vx))
            adj[v] = nbrs

    @staticmethod
    def _lines_cycle_triangles(
        cycle: list[int], verts: list[Point], vkeys: list[int]
    ) -> list[list[int]]:
        """CDT triangles of one face cycle as mesh vertex keys, counter-clockwise in xy."""

        ordered = list(cycle)
        bpts = []

        for vid in ordered:
            bpts.append((verts[vid][0], verts[vid][1]))

        if _signed_area_2d(bpts) < 0.0:
            bpts.reverse()
            ordered.reverse()

        bpts2d = []

        for b in bpts:
            bpts2d.append(Point(b[0], b[1], 0.0))

        tris = cdt_triangulate(bpts2d, [])
        tri_list = []

        for t in tris:
            tri_list.append(
                [vkeys[ordered[t[0]]], vkeys[ordered[t[1]]], vkeys[ordered[t[2]]]]
            )

        return tri_list

    @staticmethod
    def from_lines(
        lines: list[Line],
        delete_boundary_face: bool = False,
        precision: float | None = None,
    ) -> "Mesh":
        """Construct a planar mesh from a line network, optionally without its outer boundary face."""

        if not lines:
            return Mesh()

        all_pts = []

        for ln in lines:
            all_pts.append(ln.start())
            all_pts.append(ln.end())

        eps = Mesh._lines_precision(all_pts, precision)
        vmap = {}
        verts = []
        adj = {}

        for ln in lines:
            a = Mesh._lines_vertex_id(ln.start(), eps, vmap, verts)
            b = Mesh._lines_vertex_id(ln.end(), eps, vmap, verts)

            if a == b:
                continue

            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        Mesh._lines_sort_neighbors(adj, verts)

        cycles = Mesh._lines_face_cycles(adj, len(verts))

        if delete_boundary_face and cycles:
            cycles.pop(Mesh._lines_outer_cycle(cycles, verts))

        mesh = Mesh()
        vkeys = []

        for pt in verts:
            vkeys.append(mesh.add_vertex(pt))

        for cycle in cycles:
            fvkeys = []

            for vid in cycle:
                fvkeys.append(vkeys[vid])

            fk = mesh.add_face(fvkeys)

            if fk is not None:
                mesh.triangulation[fk] = Mesh._lines_cycle_triangles(
                    cycle, verts, vkeys
                )

        return mesh

    @staticmethod
    def from_polygon_with_holes(
        polylines: list[list[Point]], sort_by_bbox: bool = False
    ) -> "Mesh":
        """Construct from a polygon boundary with optional holes; sort_by_bbox picks the largest polyline as boundary."""

        from .remesh_cdt import RemeshCDT

        if not polylines:
            return Mesh()

        pls = []

        for v in polylines:
            pls.append(Polyline(v))

        return RemeshCDT.from_polylines(pls, False, not sort_by_bbox)

    @staticmethod
    def loft(
        polylines0: list[Polyline],
        polylines1: list[Polyline],
        cap: bool = True,
        fix_collinear: bool = True,
    ) -> "Mesh":
        """Construct a loft between two sets of polylines into a mesh volume, capped when cap is true."""

        if not polylines0 or not polylines1:
            return Mesh()

        if len(polylines0) != len(polylines1):
            return Mesh()

        border_idx = _loft_border_index(polylines0)
        frame = _loft_frame(polylines0[border_idx], polylines1[border_idx])
        order = _loft_order(len(polylines0), border_idx)
        polys = []
        all_bot = []
        all_top = []

        for oi in range(len(order)):
            bot = _loft_open_points(polylines0[order[oi]])
            top = _loft_open_points(polylines1[order[oi]])
            area = _loft_signed_area(frame, bot)

            if (area < 0) if oi == 0 else (area > 0):
                bot.reverse()
                top.reverse()

            polys.append(
                _LoftPoly(
                    _LoftRing(len(all_bot), len(bot)), _LoftRing(len(all_top), len(top))
                )
            )
            all_bot.extend(bot)
            all_top.extend(top)

        mesh = Mesh()
        bot_vkeys = _loft_add_vkeys(mesh, all_bot)
        top_vkeys = _loft_add_vkeys(mesh, all_top)

        if cap:
            _loft_cap(
                mesh,
                frame,
                _loft_rings(polys, False),
                all_bot,
                bot_vkeys,
                True,
                fix_collinear,
            )
            _loft_cap(
                mesh,
                frame,
                _loft_rings(polys, True),
                all_top,
                top_vkeys,
                False,
                fix_collinear,
            )

        for poly in polys:
            _loft_walls(mesh, frame, poly, all_bot, all_top, bot_vkeys, top_vkeys)

        return mesh

    @staticmethod
    def from_polygon_with_holes_many(
        inputs: list, sort_by_bbox: bool = False, parallel: bool = True
    ) -> list["Mesh"]:
        """Construct a batch of from_polygon_with_holes, parallel when asked."""

        if parallel and len(inputs) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as pool:
                return list(
                    pool.map(
                        lambda x: Mesh.from_polygon_with_holes(x, sort_by_bbox), inputs
                    )
                )

        results = []

        for x in inputs:
            results.append(Mesh.from_polygon_with_holes(x, sort_by_bbox))

        return results

    @staticmethod
    def loft_many(
        pairs: list, cap: bool = True, parallel: bool = True, fix_collinear: bool = True
    ) -> list["Mesh"]:
        """Construct a batch of loft, parallel when asked."""

        if parallel and len(pairs) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as pool:
                return list(
                    pool.map(lambda p: Mesh.loft(p[0], p[1], cap, fix_collinear), pairs)
                )

        results = []

        for p in pairs:
            results.append(Mesh.loft(p[0], p[1], cap, fix_collinear))

        return results

    @staticmethod
    def loft_panels(
        top_polygons: list[list[Point]],
        bot_polygons: list[list[Point]],
        merge_precision: float = 0.001,
        edge_gap: float = 0.0,
        edge_match_threshold: float = 2.0,
        add_caps: bool = True,
        skip_triangles: bool = False,
    ) -> LoftResult:
        """Construct a loft of matched top/bottom polygon pairs into one panel each, with matched quad walls and triangle fill."""

        top_mesh = Mesh.from_polylines(top_polygons, merge_precision)
        bot_mesh = Mesh.from_polylines(bot_polygons, merge_precision)
        face_match = _lp_match_faces(top_mesh, bot_mesh)
        panels = []

        for tfk, bfk in face_match:
            panels.append(
                _lp_build_panel(
                    top_mesh,
                    bot_mesh,
                    tfk,
                    bfk,
                    edge_gap,
                    edge_match_threshold,
                    add_caps,
                    skip_triangles,
                )
            )

        adjacency = _lp_adjacency(panels)
        top_ordered = _lp_ordered_mesh(panels, True)
        bot_ordered = _lp_ordered_mesh(panels, False)

        return LoftResult(panels, adjacency, top_ordered, bot_ordered)

    @staticmethod
    def create_box(x: float, y: float, z: float) -> "Mesh":
        """Construct a closed box centered at the origin: 8 vertices, 6 quads."""

        hx = x * 0.5
        hy = y * 0.5
        hz = z * 0.5
        vertices = [
            Point(-hx, -hy, -hz),
            Point(hx, -hy, -hz),
            Point(hx, hy, -hz),
            Point(-hx, hy, -hz),
            Point(-hx, -hy, hz),
            Point(hx, -hy, hz),
            Point(hx, hy, hz),
            Point(-hx, hy, hz),
        ]
        faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ]

        return Mesh.from_vertices_and_faces(vertices, faces)

    @staticmethod
    def create_dodecahedron(edge: float = 2.0) -> "Mesh":
        """Construct a dodecahedron with the given edge length."""

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        ip = 1.0 / phi
        s = edge / (2.0 * ip)
        verts = [
            Point(s, s, s),
            Point(s, s, -s),
            Point(s, -s, s),
            Point(s, -s, -s),
            Point(-s, s, s),
            Point(-s, s, -s),
            Point(-s, -s, s),
            Point(-s, -s, -s),
            Point(0, s * ip, s * phi),
            Point(0, s * ip, -s * phi),
            Point(0, -s * ip, s * phi),
            Point(0, -s * ip, -s * phi),
            Point(s * ip, s * phi, 0),
            Point(s * ip, -s * phi, 0),
            Point(-s * ip, s * phi, 0),
            Point(-s * ip, -s * phi, 0),
            Point(s * phi, 0, s * ip),
            Point(s * phi, 0, -s * ip),
            Point(-s * phi, 0, s * ip),
            Point(-s * phi, 0, -s * ip),
        ]
        idx = [
            [0, 8, 10, 2, 16],
            [0, 16, 17, 1, 12],
            [0, 12, 14, 4, 8],
            [1, 17, 3, 11, 9],
            [1, 9, 5, 14, 12],
            [2, 10, 6, 15, 13],
            [2, 13, 3, 17, 16],
            [3, 13, 15, 7, 11],
            [4, 14, 5, 19, 18],
            [4, 18, 6, 10, 8],
            [5, 9, 11, 7, 19],
            [6, 18, 19, 7, 15],
        ]
        faces = []

        for f in idx:
            faces.append(
                [verts[f[0]], verts[f[1]], verts[f[2]], verts[f[3]], verts[f[4]]]
            )

        return Mesh.from_polylines(faces, 1e-6)

    @staticmethod
    def _pairs_open_count(pl: Polyline) -> int:
        """Number of points of a polyline without its closing duplicate."""

        n = pl.point_count()

        if n > 1 and pl.is_closed():
            return n - 1

        return n

    @staticmethod
    def _pairs_scaled_open(src: Polyline, scale: float) -> Polyline:
        """Open polyline with every coordinate divided by scale."""

        limit = Mesh._pairs_open_count(src)
        pts = []

        for j in range(limit):
            pts.append(src.get_point(j) / scale)

        return Polyline(pts)

    @staticmethod
    def from_polyline_pairs(pairs: list[Polyline], scale: float = 1.0) -> "Mesh":
        """Construct a closed mesh from interleaved top/bottom polyline pairs [top0, bot0, ...], coordinates divided by scale."""

        if not pairs or len(pairs) % 2 != 0:
            return Mesh()

        for i in range(0, len(pairs), 2):
            a = Mesh._pairs_open_count(pairs[i])
            b = Mesh._pairs_open_count(pairs[i + 1])

            if a != b or a < 3:
                return Mesh()

        top_polys = []
        bot_polys = []

        for i in range(0, len(pairs), 2):
            top_polys.append(Mesh._pairs_scaled_open(pairs[i], scale))
            bot_polys.append(Mesh._pairs_scaled_open(pairs[i + 1], scale))

        return Mesh.loft(top_polys, bot_polys, True)

    @staticmethod
    def from_polyline_pairs_vnf(
        pairs: list[Polyline], scale: float = 1.0
    ) -> tuple[list[float], list[float], list[int]]:
        """Write the flat vertex, normal and triangle arrays of a closed mesh from interleaved top/bottom polyline pairs."""

        out_vertices = []
        out_normals = []
        out_triangles = []
        m = Mesh.from_polyline_pairs(pairs, scale)

        if m.is_empty():
            return out_vertices, out_normals, out_triangles

        face_nrms = m.face_normals()

        for fk in m.faces():
            fpts = m.face_points(fk)

            if fpts is None or len(fpts) < 3:
                continue

            nrm = face_nrms.get(fk, Vector(0.0, 0.0, 1.0))

            for i in range(1, len(fpts) - 1):
                for p in (fpts[0], fpts[i], fpts[i + 1]):
                    out_triangles.append(len(out_triangles))
                    out_vertices.append(p[0])
                    out_vertices.append(p[1])
                    out_vertices.append(p[2])
                    out_normals.append(nrm[0])
                    out_normals.append(nrm[1])
                    out_normals.append(nrm[2])

        return out_vertices, out_normals, out_triangles

    @staticmethod
    def reflex_fold(cross_section: Polyline, profile: Polyline) -> "Mesh":
        """Construct a ruled quad mesh by projecting profile onto planes perpendicular to cross_section."""

        n_cs = cross_section.point_count()
        n_p = profile.point_count()
        planes = []

        for i in range(n_cs):
            normal = Vector(0.0, 0.0, 1.0)

            if i > 0 and i < n_cs - 1:
                ci = cross_section[i]
                cp = cross_section[i - 1]
                cn = cross_section[i + 1]
                v1 = (cp - ci).normalized()
                v2 = (cn - ci).normalized()
                normal = v1 + v2

                if not normal.normalize_self():
                    normal = Vector(0.0, 0.0, 1.0)

            planes.append(Plane.from_point_normal(cross_section[i], normal))

        all_pts = []

        for j in range(n_p):
            all_pts.append(profile[j])

        faces = []

        for i in range(1, n_cs):
            po = planes[i].origin
            pp = planes[i - 1].origin
            n1 = po - pp
            n2 = planes[i].z_axis
            row_start = len(all_pts)

            for j in range(n_p):
                pvrt = all_pts[row_start - n_p + j]
                diff = po - pvrt
                denom = n2.dot(n1)
                t = n2.dot(diff) / denom if abs(denom) > 1e-12 else 0.0
                all_pts.append(pvrt + n1 * t)

            for j in range(n_p - 1):
                new_j = row_start + j
                old_j = row_start - n_p + j
                faces.append([new_j, old_j, old_j + 1, new_j + 1])

        return Mesh.from_vertices_and_faces(all_pts, faces)

    @staticmethod
    def miter_contours(
        shell: "Mesh",
        thickness: float,
        chamfer_bot: float,
        chamfer_top: float,
        flatter: bool,
        chamfer_angle_deg: float = 90.0,
    ) -> list[tuple[list[Point], list[Point], list[Point], list[Point], Vector]]:
        """Compute the per-face miter plate contours of a shell: (top_chamfered, bot_chamfered, top_raw, bot_raw, face_normal)."""

        from .intersection import plane_plane

        result = []
        efm = shell.edge_face_map()

        for fk in shell.faces():
            fverts = shell.face_vertices(fk)
            n = len(fverts)
            pts = shell.face_points(fk)

            if pts is None or len(pts) != n:
                continue

            fn = _newell_normal(pts)
            cen = _ring_centroid(pts)
            planes = _miter_planes(shell, efm, fverts, pts, fn)

            if len(planes) != n:
                continue

            corner_lines = []

            for i in range(n):
                line = plane_plane(planes[i], planes[(i + 1) % n])

                if line is None:
                    break

                corner_lines.append(line)

            if len(corner_lines) != n:
                continue

            bot_origin = cen + fn * 2.0 * thickness
            top_contour = _miter_contour(corner_lines, Plane.from_point_normal(cen, fn))
            bot_contour = _miter_contour(
                corner_lines, Plane.from_point_normal(bot_origin, fn)
            )

            if len(top_contour) != n or len(bot_contour) != n:
                continue

            top_mask = _fold_chamfer_mask(top_contour, chamfer_angle_deg)
            bot_mask = _fold_chamfer_mask(bot_contour, chamfer_angle_deg)
            top_ch = _fold_chamfer(top_contour, chamfer_bot, top_mask)
            bot_ch = _fold_chamfer(bot_contour, chamfer_top, bot_mask)
            result.append((top_ch, bot_ch, top_contour, bot_contour, fn))

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
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

    def set_pointcolors(self, colors: list[Color]) -> None:
        """Store vertex colors and render with them."""

        self._pointcolors = list(colors)
        self.color_mode = ColorMode.POINTCOLORS

    def set_facecolors(self, colors: list[Color]) -> None:
        """Store face colors and render with them."""

        self._facecolors = list(colors)
        self.color_mode = ColorMode.FACECOLORS

    def set_linecolors(
        self, colors: list[Color], line_widths: list[float] | None = None
    ) -> None:
        """Store edge colors and, when given, edge widths."""

        self._linecolors = list(colors)

        if line_widths:
            self._widths = list(line_widths)

    def set_objectcolor(self, color: Color) -> None:
        """Store the object color."""
        self._objectcolor = color

    def clear_pointcolors(self) -> None:
        """Drop vertex colors, falling back to the object color when they were active."""

        self._pointcolors.clear()

        if self.color_mode == ColorMode.POINTCOLORS:
            self.color_mode = ColorMode.OBJECTCOLOR

    def clear_facecolors(self) -> None:
        """Drop face colors, falling back to the object color when they were active."""

        self._facecolors.clear()

        if self.color_mode == ColorMode.FACECOLORS:
            self.color_mode = ColorMode.OBJECTCOLOR

    def clear_linecolors(self) -> None:
        """Drop edge colors and widths."""

        self._linecolors.clear()
        self._widths.clear()

    def get_pointcolors(self) -> list[Color]:
        """Return the vertex colors."""
        return self._pointcolors

    def get_facecolors(self) -> list[Color]:
        """Return the face colors."""
        return self._facecolors

    def get_linecolors(self) -> list[Color]:
        """Return the edge colors."""
        return self._linecolors

    def get_widths(self) -> list[float]:
        """Return the edge widths."""
        return self._widths

    def get_objectcolor(self) -> Color:
        """Return the object color."""
        return self._objectcolor

    def get_triangulation(self) -> dict[int, list[list[int]]]:
        """Return the cached triangulation per face."""
        return self.triangulation

    def set_face_triangulation(self, fk: int, tris: list[list[int]]) -> None:
        """Cache the triangles of face fk."""
        self.triangulation[fk] = tris

    def get_face_holes(self) -> dict[int, list[list[int]]]:
        """Return the hole rings per face."""
        return self.face_holes

    def set_face_holes(self, fkey: int, rings: list[list[int]]) -> None:
        """Store the hole rings of face fkey."""
        self.face_holes[fkey] = rings

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __eq__(self, other):
        """Compare name, vertices and faces; guid ignored."""

        if not isinstance(other, Mesh):
            return NotImplemented

        if self.name != other.name:
            return False

        if self.vertex != other.vertex:
            return False

        if self.face != other.face:
            return False

        return True

    def __ne__(self, other):
        """Compare name, vertices and faces; guid ignored."""
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # Boolean Queries
    # ═══════════════════════════════════════════════════════════════════════════
    def is_empty(self) -> bool:
        """Return whether the mesh has no vertices."""
        return len(self.vertex) == 0

    def is_valid(self) -> bool:
        """Return whether every face has at least three existing vertices."""

        if not self.vertex or not self.face:
            return False

        for vkeys in self.face.values():
            if len(vkeys) < 3:
                return False

            for vk in vkeys:
                if vk not in self.vertex:
                    return False

        return True

    def is_closed(self) -> bool:
        """Return whether every face edge has a twin face or a declared hole ring."""

        hole_edges = set()

        for rings in self.face_holes.values():
            for ring in rings:
                n = len(ring)

                for i in range(n):
                    hole_edges.add((ring[i], ring[(i + 1) % n]))
                    hole_edges.add((ring[(i + 1) % n], ring[i]))

        dfe = self._directed_face_edges()

        for u, v in dfe:
            if (v, u) not in dfe and (v, u) not in hole_edges:
                return False

        return bool(self.vertex)

    def is_vertex_on_boundary(self, vertex_key: int) -> bool:
        """Return whether the vertex touches a boundary edge."""

        dfe = self._directed_face_edges()

        for u, v in dfe:
            if (v, u) not in dfe and (u == vertex_key or v == vertex_key):
                return True

        return False

    def is_edge_on_boundary(self, u: int, v: int) -> bool:
        """Return whether the edge has a face on one side only."""

        dfe = self._directed_face_edges()

        return not ((u, v) in dfe and (v, u) in dfe)

    def is_face_on_boundary(self, face_key: int) -> bool:
        """Return whether the face has a boundary edge."""

        fe = self.face_edges(face_key)

        if fe is None:
            return False

        for u, v in fe:
            if self.is_edge_on_boundary(u, v):
                return True

        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # Attributes
    # ═══════════════════════════════════════════════════════════════════════════
    def number_of_vertices(self) -> int:
        """Return the vertex count."""
        return len(self.vertex)

    def number_of_faces(self) -> int:
        """Return the face count."""
        return len(self.face)

    def number_of_edges(self) -> int:
        """Return the undirected edge count."""

        dfe = self._directed_face_edges()
        count = 0

        for u, v in dfe:
            if u < v or (v, u) not in dfe:
                count += 1

        return count

    def euler(self) -> int:
        """Return the Euler characteristic V - E + F."""

        return (
            self.number_of_vertices() - self.number_of_edges() + self.number_of_faces()
        )

    def vertices(self) -> list[int]:
        """Return the sorted vertex keys."""
        return sorted(self.vertex.keys())

    def faces(self) -> list[int]:
        """Return the sorted face keys."""
        return sorted(self.face.keys())

    def edges(self) -> list[tuple[int, int]]:
        """Return the undirected edges as sorted (u, v) pairs."""

        seen = set()

        for u, v in self._directed_face_edges():
            seen.add((min(u, v), max(u, v)))

        return sorted(seen)

    def to_vertices_and_faces(self) -> tuple[list[Point], list[list[int]]]:
        """Return the vertices and faces with sequential 0-based indices."""

        vertex_idx = self.vertex_index()
        vertices = [None] * len(self.vertex)

        for key, vdata in self.vertex.items():
            vertices[vertex_idx[key]] = vdata.position()

        faces = []

        for key in self.faces():
            remapped = []

            for v in self.face[key]:
                remapped.append(vertex_idx[v])

            faces.append(remapped)

        return vertices, faces

    def vertex_index(self) -> dict[int, int]:
        """Return the map from sparse vertex key to sequential index."""

        index_map = {}
        index = 0

        for key in self.vertices():
            index_map[key] = index
            index += 1

        return index_map

    def naked_edges(self, boundary: bool = True) -> list[tuple[int, int]]:
        """Return the boundary (true) or interior (false) edges."""

        dfe = self._directed_face_edges()
        seen = set()

        for u, v in dfe:
            seen.add((min(u, v), max(u, v)))

        result = []

        for u, v in sorted(seen):
            naked = not ((u, v) in dfe and (v, u) in dfe)

            if naked == boundary:
                result.append((u, v))

        return result

    def naked_vertices(self, boundary: bool = True) -> list[int]:
        """Return the boundary (true) or interior (false) vertices."""

        result = []

        for vk in self.vertices():
            if self.is_vertex_on_boundary(vk) == boundary:
                result.append(vk)

        return result

    def naked_faces(self, boundary: bool = True) -> list[int]:
        """Return the boundary (true) or interior (false) faces."""

        result = []

        for fk in self.faces():
            if self.is_face_on_boundary(fk) == boundary:
                result.append(fk)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Vertex and Face Operations
    # ═══════════════════════════════════════════════════════════════════════════
    def add_vertex(self, position: Point, vkey: int | None = None) -> int:
        """Add a vertex, with an explicit key when given; returns the key."""

        self.ensure_halfedges()
        vertex_key = self._max_vertex if vkey is None else vkey

        if vertex_key >= self._max_vertex:
            self._max_vertex = vertex_key + 1

        self.vertex[vertex_key] = VertexData(position)
        self.halfedge[vertex_key] = {}
        self._pointcolors.append(Color.white())
        self.clear_triangle_bvh()

        return vertex_key

    def add_face(self, vertices: list[int], fkey: int | None = None) -> int | None:
        """Add a face, with an explicit key when given; returns the key or nullopt when invalid."""

        self.ensure_halfedges()

        if len(vertices) < 3:
            return None

        for v in vertices:
            if v not in self.vertex:
                return None

        if len(set(vertices)) != len(vertices):
            return None

        face_key = self._max_face if fkey is None else fkey

        if face_key >= self._max_face:
            self._max_face = face_key + 1

        self.face[face_key] = list(vertices)
        self.triangulation.pop(face_key, None)
        self._facecolors.append(Color.white())

        for i in range(len(vertices)):
            u = vertices[i]
            v = vertices[(i + 1) % len(vertices)]
            is_new_edge = u not in self.halfedge.setdefault(v, {})
            self.halfedge.setdefault(u, {})[v] = face_key

            if is_new_edge:
                self.halfedge[v][u] = None
                self._linecolors.append(Color.black())
                self._widths.append(1.0)

        self.clear_triangle_bvh()

        return face_key

    def remove_vertex(self, vkey: int) -> None:
        """Remove a vertex and every face that uses it."""

        self.ensure_halfedges()

        if vkey not in self.vertex:
            return

        faces_to_remove = []

        for fk, verts in self.face.items():
            if vkey in verts:
                faces_to_remove.append(fk)

        for fk in faces_to_remove:
            self.remove_face(fk)

        if vkey in self.halfedge:
            for v in list(self.halfedge[vkey].keys()):
                if v in self.halfedge:
                    self.halfedge[v].pop(vkey, None)

            del self.halfedge[vkey]

        for key in list(self.edgedata.keys()):
            if key[0] == vkey or key[1] == vkey:
                del self.edgedata[key]

        del self.vertex[vkey]

        if len(self._pointcolors) > len(self.vertex):
            del self._pointcolors[len(self.vertex) :]

        self.clear_triangle_bvh()

    def remove_face(self, fkey: int) -> None:
        """Remove a face and its orphaned halfedges."""

        self.ensure_halfedges()

        if fkey not in self.face:
            return

        verts = self.face[fkey]
        n = len(verts)

        for i in range(n):
            u = verts[i]
            v = verts[(i + 1) % n]

            if u not in self.halfedge or v not in self.halfedge[u]:
                continue

            self.halfedge[u][v] = None

            if v not in self.halfedge or u not in self.halfedge[v]:
                continue

            if self.halfedge[v][u] is None:
                del self.halfedge[u][v]
                del self.halfedge[v][u]

        del self.face[fkey]
        self.triangulation.pop(fkey, None)
        self.facedata.pop(fkey, None)
        self.face_holes.pop(fkey, None)
        n_edges = self.number_of_edges()

        if len(self._linecolors) > n_edges:
            del self._linecolors[n_edges:]

        if len(self._widths) > n_edges:
            del self._widths[n_edges:]

        if len(self._facecolors) > len(self.face):
            del self._facecolors[len(self.face) :]

        self.clear_triangle_bvh()

    def remove_edge(self, u: int, v: int) -> None:
        """Remove an edge, its adjacent faces and its halfedges."""

        self.ensure_halfedges()
        faces_to_remove = []
        f_uv = self.halfedge_face((u, v))
        f_vu = self.halfedge_face((v, u))

        if f_uv is not None:
            faces_to_remove.append(f_uv)

        if f_vu is not None and f_vu != f_uv:
            faces_to_remove.append(f_vu)

        for fk in faces_to_remove:
            self.remove_face(fk)

        if u in self.halfedge:
            self.halfedge[u].pop(v, None)

        if v in self.halfedge:
            self.halfedge[v].pop(u, None)

        self.edgedata.pop((u, v), None)
        self.edgedata.pop((v, u), None)
        n_edges = self.number_of_edges()

        if len(self._linecolors) > n_edges:
            del self._linecolors[n_edges:]

        if len(self._widths) > n_edges:
            del self._widths[n_edges:]

        self.clear_triangle_bvh()

    def flip_face(self, fkey: int) -> None:
        """Reverse the winding of one face in place."""

        self.ensure_halfedges()

        if fkey not in self.face:
            return

        fv = list(self.face[fkey])
        self.remove_face(fkey)
        fv.reverse()
        self.add_face(fv, fkey)

    def flip(self) -> None:
        """Reverse the winding of every face."""

        for fkey in self.face:
            self.face[fkey].reverse()

        self.rebuild_halfedges()

    def clear(self) -> None:
        """Clear all mesh data."""

        self.halfedge.clear()
        self.vertex.clear()
        self.face.clear()
        self.facedata.clear()
        self.edgedata.clear()
        self.triangulation.clear()
        self.face_holes.clear()
        self._max_vertex = 0
        self._max_face = 0
        self._pointcolors.clear()
        self._facecolors.clear()
        self._linecolors.clear()
        self._widths.clear()
        self._objectcolor = Color.lightgrey()
        self.color_mode = ColorMode.OBJECTCOLOR
        self.clear_triangle_bvh()

    def unweld(self) -> "Mesh":
        """Copy where every face owns its own vertices."""

        m = Mesh()

        for fkey in self.faces():
            new_vkeys = []

            for vk in self.face[fkey]:
                new_vkeys.append(m.add_vertex(self.vertex[vk].position()))

            m.add_face(new_vkeys)

        return m

    @staticmethod
    def _weld_find(parent: list[int], x: int) -> int:
        """Root of x in a union-find forest, halving the path on the way."""

        bound = len(parent)

        for _step in range(bound):
            if parent[x] == x:
                break

            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    @staticmethod
    def _weld_union(
        parent: list[int], positions: list[Point], tolerance: float
    ) -> None:
        """Join in the union-find forest every pair of positions closer than tolerance."""

        boxes = []

        for p in positions:
            boxes.append(OBB.from_point(p, tolerance))

        ws = SpatialBVH.compute_world_size(boxes)
        bvh = SpatialBVH.from_boxes(boxes, ws)
        pairs, _ignore1, _ignore2 = bvh.check_all_collisions(boxes)

        for i, j in pairs:
            if positions[i].distance(positions[j]) > tolerance:
                continue

            ri = Mesh._weld_find(parent, i)
            rj = Mesh._weld_find(parent, j)

            if ri != rj:
                parent[ri] = rj

    def weld(self, tolerance: float = 0.001) -> "Mesh":
        """Copy with vertices closer than tolerance merged; degenerate faces are dropped."""

        if not self.vertex:
            return Mesh()

        vkeys = self.vertices()
        positions = []

        for vk in vkeys:
            positions.append(self.vertex[vk].position())

        n = len(vkeys)
        parent = list(range(n))

        if tolerance > 0.0:
            Mesh._weld_union(parent, positions, tolerance)

        root_to_rep = {}

        for i in range(n):
            root = Mesh._weld_find(parent, i)

            if root not in root_to_rep or vkeys[i] < root_to_rep[root]:
                root_to_rep[root] = vkeys[i]

        vkey_to_rep = {}

        for i in range(n):
            vkey_to_rep[vkeys[i]] = root_to_rep[Mesh._weld_find(parent, i)]

        m = Mesh()
        added = set()

        for i in range(n):
            rep = vkey_to_rep[vkeys[i]]

            if rep not in added:
                added.add(rep)
                m.add_vertex(self.vertex[rep].position(), rep)

        for fk in self.faces():
            new_vkeys = []

            for vk in self.face[fk]:
                new_vkeys.append(vkey_to_rep[vk])

            m.add_face(new_vkeys, fk)

        return m

    @staticmethod
    def _winding_edge_faces(
        face: dict[int, list[int]],
    ) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
        """Faces on every undirected edge as (face key, u, v) in ring direction."""

        edge_faces = {}

        for fkey, verts in face.items():
            n = len(verts)

            for i in range(n):
                u = verts[i]
                v = verts[(i + 1) % n]
                edge_faces.setdefault((min(u, v), max(u, v)), []).append((fkey, u, v))

        return edge_faces

    @staticmethod
    def _winding_flipped(
        face: dict[int, list[int]],
        edge_faces: dict[tuple[int, int], list[tuple[int, int, int]]],
    ) -> set[int]:
        """Faces to reverse so every face agrees with the neighbor it was first reached from."""

        visited = set()
        flipped = set()

        for seed in sorted(face.keys()):
            if seed in visited:
                continue

            visited.add(seed)
            queue = [seed]

            while queue:
                f = queue.pop()
                is_flipped = f in flipped
                verts = face[f]
                n = len(verts)

                for i in range(n):
                    u_orig = verts[i]
                    v_orig = verts[(i + 1) % n]
                    eff_u = v_orig if is_flipped else u_orig
                    eff_v = u_orig if is_flipped else v_orig

                    for adj_key, adj_u, adj_v in edge_faces[
                        (min(u_orig, v_orig), max(u_orig, v_orig))
                    ]:
                        if adj_key == f or adj_key in visited:
                            continue

                        if not (adj_u == eff_v and adj_v == eff_u):
                            flipped.add(adj_key)

                        visited.add(adj_key)
                        queue.append(adj_key)

        return flipped

    def unify_winding(self) -> bool:
        """Unify face winding by BFS; returns true when any face was flipped."""

        if len(self.face) < 2:
            return False

        flipped = Mesh._winding_flipped(self.face, Mesh._winding_edge_faces(self.face))

        if not flipped:
            return False

        for fkey in flipped:
            self.face[fkey].reverse()

        self.rebuild_halfedges()
        self.orient_outward()

        return True

    def orient_outward(self) -> bool:
        """Flip a closed mesh whose normals point inward; returns true when flipped."""

        self.ensure_halfedges()

        if not self.face or self.naked_edges(True):
            return False

        vol = 0.0

        for verts in self.face.values():
            p0 = self.vertex_point(verts[0])

            for i in range(1, len(verts) - 1):
                p1 = self.vertex_point(verts[i])
                p2 = self.vertex_point(verts[i + 1])
                vol += (
                    p0[0] * (p1[1] * p2[2] - p1[2] * p2[1])
                    + p0[1] * (p1[2] * p2[0] - p1[0] * p2[2])
                    + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0])
                )

        if vol >= 0.0:
            return False

        for fk in self.face:
            self.face[fk].reverse()

        self.rebuild_halfedges()

        return True

    def rebuild_halfedges(self) -> None:
        """Recreate halfedge from vertex and face alone."""
        self.halfedge = self._compute_halfedges()

    def ensure_halfedges(self) -> None:
        """Build the lazy halfedge map when it is empty and faces exist."""

        if not self.halfedge and self.face:
            self.rebuild_halfedges()

    # ═══════════════════════════════════════════════════════════════════════════
    # Connectivity Queries
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _edge_ends(dfe: set[tuple[int, int]], x: int) -> list[int]:
        """Sorted neighbors of x over a directed edge set."""

        keys = set()

        for a, b in dfe:
            if a == x:
                keys.add(b)
            elif b == x:
                keys.add(a)

        return sorted(keys)

    def edge_edges(self, u: int, v: int) -> list[tuple[int, int]] | None:
        """Return the edges sharing a vertex with (u, v), excluding (u, v) and (v, u)."""

        dfe = self._directed_face_edges()

        if (u, v) not in dfe and (v, u) not in dfe:
            return None

        edges = []

        for w in Mesh._edge_ends(dfe, u):
            if w != v:
                edges.append((u, w))

        for w in Mesh._edge_ends(dfe, v):
            if w != u:
                edges.append((v, w))

        return edges

    def edge_faces(self, u: int, v: int) -> list[int] | None:
        """Return the faces on each side of an edge."""

        result = []

        for fkey in self.faces():
            verts = self.face[fkey]
            n = len(verts)

            for i in range(n):
                a = verts[i]
                b = verts[(i + 1) % n]

                if not ((a == u and b == v) or (a == v and b == u)):
                    continue

                if fkey not in result:
                    result.append(fkey)

        if not result:
            return None

        return result

    def edge_face_map(self) -> dict[tuple[int, int], int]:
        """Return every directed face edge mapped to its face key, in one face walk."""

        m = {}

        for fkey in self.faces():
            verts = self.face[fkey]
            n = len(verts)

            for i in range(n):
                m[(verts[i], verts[(i + 1) % n])] = fkey

        return m

    def edge_line(self, u: int, v: int) -> Line | None:
        """Return the edge as a Line."""

        dfe = self._directed_face_edges()

        if (u, v) not in dfe and (v, u) not in dfe:
            return None

        pu = self.vertex_point(u)
        pv = self.vertex_point(v)

        if pu is None or pv is None:
            return None

        return Line.from_points(pu, pv)

    def face_edges(self, face_key: int) -> list[tuple[int, int]] | None:
        """Return the edges of a face as (vi, vi+1) pairs."""

        verts = self.face.get(face_key)

        if verts is None:
            return None

        n = len(verts)
        edges = []

        for i in range(n):
            edges.append((verts[i], verts[(i + 1) % n]))

        return edges

    def face_faces(self, face_key: int) -> list[int] | None:
        """Return the faces sharing an edge with a face."""

        fe = self.face_edges(face_key)

        if fe is None:
            return None

        efm = self.edge_face_map()
        neighbors = []

        for u, v in fe:
            if (v, u) in efm:
                neighbors.append(efm[(v, u)])

        return neighbors

    def face_points(self, face_key: int) -> list[Point] | None:
        """Return the points of a face."""

        fv = self.face_vertices(face_key)

        if fv is None:
            return None

        pts = []

        for vk in fv:
            p = self.vertex_point(vk)

            if p is None:
                return None

            pts.append(p)

        return pts

    def face_polyline(self, face_key: int) -> Polyline | None:
        """Return the face as a Polyline."""

        pts = self.face_points(face_key)

        if pts is None:
            return None

        return Polyline(pts)

    def face_vertices(self, face_key: int) -> list[int] | None:
        """Return the vertex keys of a face."""
        return self.face.get(face_key)

    def vertex_edges(self, vertex_key: int) -> list[tuple[int, int]] | None:
        """Return the edges incident to a vertex as (vertex_key, neighbor) pairs."""

        keys = self.vertex_vertices(vertex_key)

        if keys is None:
            return None

        edges = []

        for u in keys:
            edges.append((vertex_key, u))

        return edges

    def vertex_faces(self, vertex_key: int) -> list[int] | None:
        """Return the faces incident to a vertex."""

        keys = self.vertex_vertices(vertex_key)

        if keys is None:
            return None

        efm = self.edge_face_map()
        faces = []

        for u in keys:
            if (vertex_key, u) in efm:
                faces.append(efm[(vertex_key, u)])

        return faces

    def vertex_point(self, vertex_key: int) -> Point | None:
        """Return the position of a vertex."""

        vd = self.vertex.get(vertex_key)

        if vd is None:
            return None

        return vd.position()

    def vertex_vertices(self, vertex_key: int) -> list[int] | None:
        """Return the neighboring vertices of a vertex."""

        if vertex_key not in self.vertex:
            return None

        return Mesh._edge_ends(self._directed_face_edges(), vertex_key)

    def vertex_neighbors(
        self, vertex_key: int, ordered: bool = False
    ) -> list[int] | None:
        """Return the neighbors of a vertex, in face-cycle order when ordered is true."""

        nbrs_map = self.halfedge.get(vertex_key)

        if nbrs_map is None:
            return None

        nbrs = sorted(nbrs_map.keys())

        if not ordered or len(nbrs) <= 1:
            return nbrs

        start = nbrs[0]

        for n in nbrs:
            if nbrs_map[n] is None:
                start = n
                break

        fkey = self.halfedge_face((start, vertex_key))
        out = [start]

        for _step in range(len(self.face)):
            if fkey is None:
                break

            verts = self.face.get(fkey)

            if verts is None or vertex_key not in verts:
                break

            nbr = verts[(verts.index(vertex_key) + 1) % len(verts)]

            if nbr == start:
                break

            out.append(nbr)
            fkey = self.halfedge_face((nbr, vertex_key))

        return out

    # ═══════════════════════════════════════════════════════════════════════════
    # Boundary
    # ═══════════════════════════════════════════════════════════════════════════
    def vertices_on_boundary(self) -> list[int]:
        """Return the vertices touching a boundary edge."""

        out = []

        for v in self.vertices():
            if self.is_vertex_on_boundary(v):
                out.append(v)

        return out

    def edges_on_boundary(self) -> list[tuple[int, int]]:
        """Return the edges with a face on one side only."""

        out = []

        for u, nbrs in self.halfedge.items():
            for v, f in nbrs.items():
                if f is None:
                    out.append((u, v))

        return sorted(out)

    def faces_on_boundary(self) -> list[int]:
        """Return the faces with a boundary edge."""

        out = []

        for f in self.faces():
            if self.is_face_on_boundary(f):
                out.append(f)

        return out

    # ═══════════════════════════════════════════════════════════════════════════
    # Halfedge Navigation
    # ═══════════════════════════════════════════════════════════════════════════
    def halfedge_face(self, edge: tuple[int, int]) -> int | None:
        """Return the face of a directed edge, nullopt when unknown or on the boundary."""

        nbrs = self.halfedge.get(edge[0])

        if nbrs is None:
            return None

        return nbrs.get(edge[1])

    def halfedge_after(self, edge: tuple[int, int]) -> tuple[int, int] | None:
        """Return the next directed edge around the face of edge."""

        u, v = edge
        f = self.halfedge_face(edge)

        if f is not None:
            verts = self.face.get(f)

            if verts is None or v not in verts:
                return None

            return (v, verts[(verts.index(v) + 1) % len(verts)])

        nbrs = self.halfedge.get(v)

        if nbrs is None:
            return None

        for w in sorted(nbrs.keys()):
            if w != u and nbrs[w] is None:
                return (v, w)

        return None

    def halfedge_before(self, edge: tuple[int, int]) -> tuple[int, int] | None:
        """Return the previous directed edge around the face of edge."""

        u, v = edge
        f = self.halfedge_face(edge)

        if f is not None:
            verts = self.face.get(f)

            if verts is None or u not in verts:
                return None

            n = len(verts)

            return (verts[(verts.index(u) + n - 1) % n], u)

        nbrs = self.halfedge.get(u)

        if nbrs is None:
            return None

        for w in sorted(nbrs.keys()):
            if w == v:
                continue

            if (
                w in self.halfedge
                and u in self.halfedge[w]
                and self.halfedge[w][u] is None
            ):
                return (w, u)

        return None

    def _halfedge_loop_boundary(self, edge: tuple[int, int]) -> list[tuple[int, int]]:
        """Boundary loop from edge: at each vertex continue along the other boundary edge."""

        edges = [edge]
        u, v = edge

        for _step in range(len(self.vertex)):
            nbrs = self.vertex_neighbors(v, False)

            if nbrs is None or len(nbrs) == 2:
                break

            nbr = None

            for temp in nbrs:
                if temp == u:
                    continue

                if self.is_edge_on_boundary(v, temp):
                    nbr = temp
                    break

            if nbr is None:
                break

            u = v
            v = nbr
            edges.append((u, v))

            if v == edges[0][0]:
                break

        return edges

    def halfedge_loop(self, edge: tuple[int, int]) -> list[tuple[int, int]]:
        """Return the directed edges around the face of edge, starting at edge."""

        if self.is_edge_on_boundary(edge[0], edge[1]):
            return self._halfedge_loop_boundary(edge)

        edges = [edge]
        u, v = edge

        for _step in range(len(self.vertex)):
            nbrs = self.vertex_neighbors(v, True)

            if nbrs is None or len(nbrs) != 4 or u not in nbrs:
                break

            i = nbrs.index(u)
            u = v
            v = nbrs[(i + 2) % 4]
            edges.append((u, v))

            if v == edges[0][0]:
                break

        return edges

    def halfedge_strip(self, edge: tuple[int, int]) -> list[tuple[int, int]]:
        """Return the directed edges straight across quads from edge until a boundary or a non-quad."""

        u, v = edge
        edges = [edge]

        for _step in range(len(self.face)):
            f = self.halfedge_face((u, v))

            if f is None:
                break

            verts = self.face.get(f)

            if verts is None or len(verts) != 4 or u not in verts:
                break

            i = verts.index(u)
            u = verts[(i + 3) % 4]
            v = verts[(i + 2) % 4]
            edges.append((u, v))

            if (u, v) == edge:
                break

        return edges

    # ═══════════════════════════════════════════════════════════════════════════
    # Sampling
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _lcg_sample(keys: list, size: int, seed: int) -> list:
        """Return size keys; seed 0 takes the first keys, any other seed drives a deterministic LCG."""

        if not keys or size == 0:
            return []

        n = len(keys)
        take = min(size, n)

        if seed == 0:
            return keys[:take]

        s = seed & 0x7FFFFFFF

        if s == 0:
            s = 1

        period = 1 << 31
        used = set()
        out = []

        for _step in range(period):
            if len(out) >= take:
                break

            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            i = s % n

            if i not in used:
                used.add(i)
                out.append(keys[i])

        return out

    def vertex_sample(self, size: int, seed: int = 0) -> list[int]:
        """Return size vertex keys; seed 0 takes the first keys, any other seed drives a deterministic LCG."""
        return Mesh._lcg_sample(self.vertices(), size, seed)

    def edge_sample(self, size: int, seed: int = 0) -> list[tuple[int, int]]:
        """Return size edges; seed 0 takes the first edges, any other seed drives a deterministic LCG."""
        return Mesh._lcg_sample(self.edges(), size, seed)

    def face_sample(self, size: int, seed: int = 0) -> list[int]:
        """Return size face keys; seed 0 takes the first keys, any other seed drives a deterministic LCG."""
        return Mesh._lcg_sample(self.faces(), size, seed)

    # ═══════════════════════════════════════════════════════════════════════════
    # Aliases
    # ═══════════════════════════════════════════════════════════════════════════
    def face_center(self, face_key: int) -> Point | None:
        """Return the average of a face's vertex positions."""
        return self.face_centroid(face_key)

    def face_polygon(self, face_key: int) -> Polyline | None:
        """Return the face as a Polyline."""

        pts = self.face_points(face_key)

        if pts is None:
            return None

        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])

        return Polyline(pts)

    def face_outlines(self) -> list[Polyline]:
        """Return every face as a closed outline in face-key order; faces under three vertices are skipped."""

        outlines = []

        for face_key in self.faces():
            outline = self.face_polygon(face_key)

            if outline is not None and outline.point_count() >= 4:
                outlines.append(outline)

        return outlines

    def flip_cycles(self) -> None:
        """Reverse the winding of every face."""
        self.flip()

    # ═══════════════════════════════════════════════════════════════════════════
    # Attribute API
    # ═══════════════════════════════════════════════════════════════════════════
    def update_default_vertex_attributes(self, attrs: dict[str, float]) -> None:
        """Merge attrs into the default vertex attributes."""

        for k, v in attrs.items():
            self.default_vertex_attributes[k] = v

    def update_default_face_attributes(self, attrs: dict[str, float]) -> None:
        """Merge attrs into the default face attributes."""

        for k, v in attrs.items():
            self.default_face_attributes[k] = v

    def update_default_edge_attributes(self, attrs: dict[str, float]) -> None:
        """Merge attrs into the default edge attributes."""

        for k, v in attrs.items():
            self.default_edge_attributes[k] = v

    def vertex_attribute(self, key: int, name: str) -> float | None:
        """Return the attribute of a vertex, falling back to the default; nullopt when neither exists."""

        vd = self.vertex.get(key)

        if vd is None:
            return None

        if name in vd.attributes:
            return vd.attributes[name]

        return self.default_vertex_attributes.get(name)

    def set_vertex_attribute(self, key: int, name: str, value: float) -> None:
        """Store an attribute on a vertex."""

        vd = self.vertex.get(key)

        if vd is None:
            return

        vd.attributes[name] = value

    def face_attribute(self, fkey: int, name: str) -> float | None:
        """Return the attribute of a face, falling back to the default; nullopt when neither exists."""

        if fkey not in self.face:
            return None

        attrs = self.facedata.get(fkey)

        if attrs is not None and name in attrs:
            return attrs[name]

        return self.default_face_attributes.get(name)

    def set_face_attribute(self, fkey: int, name: str, value: float) -> None:
        """Store an attribute on a face."""

        if fkey not in self.face:
            return

        self.facedata.setdefault(fkey, {})[name] = value

    def edge_attribute(self, edge: tuple[int, int], name: str) -> float | None:
        """Return the attribute of an edge, falling back to the default; nullopt when neither exists."""

        u, v = edge
        uv = u in self.halfedge and v in self.halfedge[u]
        vu = v in self.halfedge and u in self.halfedge[v]

        if not uv and not vu:
            return None

        attrs = self.edgedata.get((u, v))

        if attrs is None:
            attrs = self.edgedata.get((v, u))

        if attrs is not None and name in attrs:
            return attrs[name]

        return self.default_edge_attributes.get(name)

    def set_edge_attribute(
        self, edge: tuple[int, int], name: str, value: float
    ) -> None:
        """Store an attribute on an edge."""

        u, v = edge
        key = (v, u) if (v, u) in self.edgedata else (u, v)
        self.edgedata.setdefault(key, {})[name] = value

    def vertices_attribute(
        self, name: str, keys: list[int] | None = None
    ) -> list[float | None]:
        """Return the attribute of every vertex in keys; keys nullptr means all; the result holds nullopt for missing values."""

        if keys is None:
            keys = self.vertices()

        out = []

        for k in keys:
            out.append(self.vertex_attribute(k, name))

        return out

    def set_vertices_attribute(
        self, name: str, value: float, keys: list[int] | None = None
    ) -> None:
        """Store an attribute on every vertex in keys; keys nullptr means all."""

        if keys is None:
            keys = self.vertices()

        for k in keys:
            self.set_vertex_attribute(k, name, value)

    def faces_attribute(
        self, name: str, keys: list[int] | None = None
    ) -> list[float | None]:
        """Return the attribute of every face in keys; keys nullptr means all; the result holds nullopt for missing values."""

        if keys is None:
            keys = self.faces()

        out = []

        for k in keys:
            out.append(self.face_attribute(k, name))

        return out

    def set_faces_attribute(
        self, name: str, value: float, keys: list[int] | None = None
    ) -> None:
        """Store an attribute on every face in keys; keys nullptr means all."""

        if keys is None:
            keys = self.faces()

        for k in keys:
            self.set_face_attribute(k, name, value)

    def edges_attribute(
        self, name: str, keys: list[tuple[int, int]] | None = None
    ) -> list[float | None]:
        """Return the attribute of every edge in keys; keys nullptr means all; the result holds nullopt for missing values."""

        if keys is None:
            keys = self.edges()

        out = []

        for e in keys:
            out.append(self.edge_attribute(e, name))

        return out

    def set_edges_attribute(
        self, name: str, value: float, keys: list[tuple[int, int]] | None = None
    ) -> None:
        """Store an attribute on every edge in keys; keys nullptr means all."""

        if keys is None:
            keys = self.edges()

        for e in keys:
            self.set_edge_attribute(e, name, value)

    def vertices_where(self, conditions: dict[str, float]) -> list[int]:
        """Return the vertices whose attributes match every (name, value) condition."""

        out = []

        for k in self.vertices():
            ok = True

            for n, v in conditions.items():
                val = self.vertex_attribute(k, n)

                if val is None or val != v:
                    ok = False
                    break

            if ok:
                out.append(k)

        return out

    def faces_where(self, conditions: dict[str, float]) -> list[int]:
        """Return the faces whose attributes match every (name, value) condition."""

        out = []

        for k in self.faces():
            ok = True

            for n, v in conditions.items():
                val = self.face_attribute(k, n)

                if val is None or val != v:
                    ok = False
                    break

            if ok:
                out.append(k)

        return out

    def edges_where(self, conditions: dict[str, float]) -> list[tuple[int, int]]:
        """Return the edges whose attributes match every (name, value) condition."""

        out = []

        for e in self.edges():
            ok = True

            for n, v in conditions.items():
                val = self.edge_attribute(e, n)

                if val is None or val != v:
                    ok = False
                    break

            if ok:
                out.append(e)

        return out

    def vertices_where_predicate(
        self, pred: Callable[[int, dict[str, float]], bool]
    ) -> list[int]:
        """Return the vertices for which pred(key, attributes) is true."""

        out = []

        for k in self.vertices():
            attrs = dict(self.default_vertex_attributes)

            for kk, vv in self.vertex[k].attributes.items():
                attrs[kk] = vv

            if pred(k, attrs):
                out.append(k)

        return out

    def faces_where_predicate(
        self, pred: Callable[[int, dict[str, float]], bool]
    ) -> list[int]:
        """Return the faces for which pred(key, attributes) is true."""

        out = []

        for k in self.faces():
            attrs = dict(self.default_face_attributes)

            for kk, vv in self.facedata.get(k, {}).items():
                attrs[kk] = vv

            if pred(k, attrs):
                out.append(k)

        return out

    def edges_where_predicate(
        self, pred: Callable[[tuple[int, int], dict[str, float]], bool]
    ) -> list[tuple[int, int]]:
        """Return the edges for which pred(edge, attributes) is true."""

        out = []

        for e in self.edges():
            attrs = dict(self.default_edge_attributes)
            data = self.edgedata.get(e)

            if data is None:
                data = self.edgedata.get((e[1], e[0]), {})

            for kk, vv in data.items():
                attrs[kk] = vv

            if pred(e, attrs):
                out.append(e)

        return out

    def face_normal_unitized(self, face_key: int, unitized: bool) -> Vector | None:
        """Return the face normal from the first three vertices; unitized false keeps twice the first-triangle area as length."""

        vertices = self.face_vertices(face_key)

        if vertices is None or len(vertices) < 3:
            return None

        p0 = self.vertex_point(vertices[0])
        p1 = self.vertex_point(vertices[1])
        p2 = self.vertex_point(vertices[2])

        if p0 is None or p1 is None or p2 is None:
            return None

        u = p1 - p0
        v = p2 - p0
        normal = u.cross(v)

        if not unitized:
            return normal

        length = normal.magnitude()

        if length > Tolerance.ZERO_TOLERANCE:
            return normal / length

        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometric Properties
    # ═══════════════════════════════════════════════════════════════════════════
    def area(self) -> float:
        """Return the total surface area of all faces."""

        total = 0.0

        for fk in self.faces():
            a = self.face_area(fk)

            if a is not None:
                total += a

        return total

    def centroid(self) -> Point:
        """Return the average of all vertex positions."""

        x = 0.0
        y = 0.0
        z = 0.0

        for vk in self.vertices():
            v = self.vertex[vk]
            x += v.x
            y += v.y
            z += v.z

        n = 1.0 if not self.vertex else float(len(self.vertex))

        return Point(x / n, y / n, z / n)

    def dihedral_angle(self, u: int, v: int) -> float | None:
        """Return the dihedral angle in degrees between the two faces sharing edge (u, v), nullopt on a boundary edge."""

        ef = self.edge_faces(u, v)

        if ef is None or len(ef) < 2:
            return None

        n0 = self.face_normal(ef[0])
        n1 = self.face_normal(ef[1])

        if n0 is None or n1 is None:
            return None

        dot = max(-1.0, min(1.0, n0.dot(n1)))

        return (PI - math.acos(dot)) * 180.0 / PI

    @staticmethod
    def _dihedral_arm(centroid: Point, mid: Point, edge: Vector) -> Vector | None:
        """Unit direction from the edge midpoint to the face centroid, in the plane perpendicular to the edge."""

        d = centroid - mid
        dot = d.dot(edge)
        d = d - edge * dot
        length = d.magnitude()

        if length < 1e-10:
            return None

        return d / length

    @staticmethod
    def _dihedral_arc(
        ep0: Point,
        ep1: Point,
        mid: Point,
        c0: Point,
        c1: Point,
        scale: float,
        arc_n: int,
    ) -> list[Point]:
        """Arc of arc_n + 1 points around mid from the arm toward c0 to the arm toward c1, empty when degenerate."""

        edge = ep1 - ep0

        if edge.magnitude() < 1e-10 or not edge.normalize_self():
            return []

        d0 = Mesh._dihedral_arm(c0, mid, edge)
        d1 = Mesh._dihedral_arm(c1, mid, edge)

        if d0 is None or d1 is None:
            return []

        theta = math.acos(max(-1.0, min(1.0, d0.dot(d1))))

        if abs(math.sin(theta)) < 1e-10:
            return []

        arc_pts = []

        for j in range(arc_n + 1):
            t = j / arc_n
            w1 = math.sin((1.0 - t) * theta) / math.sin(theta)
            w2 = math.sin(t * theta) / math.sin(theta)
            arc_pts.append(mid + (d0 * w1 + d1 * w2) * scale)

        return arc_pts

    @staticmethod
    def _dihedral_label(p: Point, angle: float, color: Color) -> Point:
        """Label point at p named by the angle."""

        pt = Point(p[0], p[1], p[2], str(angle))
        pt.pointcolor = color

        return pt

    def dihedral_angles(
        self, scale: float = 0.3, with_arcs: bool = True, with_points: bool = True
    ) -> tuple[dict[tuple[int, int], float], list[Polyline], list[Point]]:
        """Return the dihedral angles of all interior edges as (angles, arcs, points); arcs and label points are built when asked."""

        angles = {}
        arcs = []
        points = []
        arc_n = 12
        label_color = Color.yellow()

        for u, v in self.edges():
            da = self.dihedral_angle(u, v)

            if da is None:
                continue

            angles[(u, v)] = da
            ep0 = self.vertex_point(u)
            ep1 = self.vertex_point(v)
            mid = Point(
                (ep0[0] + ep1[0]) * 0.5,
                (ep0[1] + ep1[1]) * 0.5,
                (ep0[2] + ep1[2]) * 0.5,
            )

            if scale == 0.0:
                if with_points:
                    points.append(Mesh._dihedral_label(mid, da, label_color))

                continue

            ef = self.edge_faces(u, v)
            arc_pts = Mesh._dihedral_arc(
                ep0,
                ep1,
                mid,
                self.face_centroid(ef[0]),
                self.face_centroid(ef[1]),
                scale,
                arc_n,
            )

            if not arc_pts:
                continue

            if with_arcs:
                arc = Polyline(arc_pts)
                arc.name = "dihedral_e" + str(u) + "_" + str(v) + "=" + str(da)
                arc.linecolor = label_color
                arcs.append(arc)

            if with_points:
                points.append(
                    Mesh._dihedral_label(arc_pts[arc_n // 2], da, label_color)
                )

        return angles, arcs, points

    def face_area(self, face_key: int) -> float | None:
        """Return the area of a face."""

        vertices = self.face_vertices(face_key)

        if vertices is None or len(vertices) < 3:
            return 0.0

        p0 = self.vertex_point(vertices[0])

        if p0 is None:
            return None

        area = 0.0

        for i in range(1, len(vertices) - 1):
            p1 = self.vertex_point(vertices[i])
            p2 = self.vertex_point(vertices[i + 1])

            if p1 is None or p2 is None:
                return None

            u = p1 - p0
            v = p2 - p0
            area += u.cross(v).magnitude() * 0.5

        return area

    def face_centroid(self, face_key: int) -> Point | None:
        """Return the average of a face's vertex positions."""

        verts = self.face_vertices(face_key)

        if not verts:
            return None

        x = 0.0
        y = 0.0
        z = 0.0

        for vk in verts:
            p = self.vertex_point(vk)

            if p is None:
                return None

            x += p[0]
            y += p[1]
            z += p[2]

        n = float(len(verts))

        return Point(x / n, y / n, z / n)

    def face_normal(self, face_key: int) -> Vector | None:
        """Return the unit normal of a face."""
        return self.face_normal_unitized(face_key, True)

    def face_normals(self) -> dict[int, Vector]:
        """Return the unit normals of all faces."""

        normals = {}

        for face_key in self.faces():
            normal = self.face_normal(face_key)

            if normal is not None:
                normals[face_key] = normal

        return normals

    def vertex_angle_in_face(self, vertex_key: int, face_key: int) -> float | None:
        """Return the angle at a vertex inside a face."""

        vertices = self.face_vertices(face_key)

        if vertices is None or vertex_key not in vertices:
            return None

        vertex_index = vertices.index(vertex_key)
        n = len(vertices)
        center = self.vertex_point(vertex_key)
        prev_pos = self.vertex_point(vertices[(vertex_index + n - 1) % n])
        next_pos = self.vertex_point(vertices[(vertex_index + 1) % n])

        if center is None or prev_pos is None or next_pos is None:
            return None

        u = prev_pos - center
        v = next_pos - center
        u_len = u.magnitude()
        v_len = v.magnitude()

        if u_len < Tolerance.ZERO_TOLERANCE or v_len < Tolerance.ZERO_TOLERANCE:
            return 0.0

        return math.acos(max(-1.0, min(1.0, u.dot(v) / (u_len * v_len))))

    def vertex_normal(self, vertex_key: int) -> Vector | None:
        """Return the area-weighted vertex normal."""
        return self.vertex_normal_weighted(vertex_key, NormalWeighting.AREA)

    def vertex_normal_weighted(
        self, vertex_key: int, weighting: NormalWeighting
    ) -> Vector | None:
        """Return the vertex normal with the given weighting."""

        faces = self.vertex_faces(vertex_key)

        if not faces:
            return None

        normal_acc = Vector(0.0, 0.0, 0.0)

        for face_key in faces:
            fn = self.face_normal(face_key)

            if fn is None:
                continue

            weight = 1.0

            if weighting == NormalWeighting.AREA:
                weight = self.face_area(face_key)

                if weight is None:
                    weight = 1.0
            elif weighting == NormalWeighting.ANGLE:
                weight = self.vertex_angle_in_face(vertex_key, face_key)

                if weight is None:
                    weight = 1.0

            normal_acc += fn * weight

        length = normal_acc.magnitude()

        if length > Tolerance.ZERO_TOLERANCE:
            return normal_acc / length

        return None

    def vertex_normals(self) -> dict[int, Vector]:
        """Return the area-weighted normals of all vertices."""
        return self.vertex_normals_weighted(NormalWeighting.AREA)

    @staticmethod
    def _corner_angle(pts: list[Point], i: int) -> float:
        """Corner weight of vertex i in a face: its interior angle."""

        n = len(pts)
        prev = (i + n - 1) % n
        nxt = (i + 1) % n
        a = pts[prev] - pts[i]
        b = pts[nxt] - pts[i]
        a_len = a.magnitude()
        b_len = b.magnitude()

        if a_len < Tolerance.ZERO_TOLERANCE or b_len < Tolerance.ZERO_TOLERANCE:
            return 0.0

        return math.acos(max(-1.0, min(1.0, a.dot(b) / (a_len * b_len))))

    def vertex_normals_weighted(self, weighting: NormalWeighting) -> dict[int, Vector]:
        """Return the normals of all vertices with the given weighting."""

        acc = {}

        for fk in self.faces():
            vkeys = self.face[fk]

            if len(vkeys) < 3:
                continue

            pts = self.face_points(fk)

            if pts is None:
                continue

            normal = self.face_normal(fk)

            if normal is None:
                continue

            area = 0.0

            if weighting == NormalWeighting.AREA:
                area = self.face_area(fk)

            for i in range(len(vkeys)):
                weight = 1.0

                if weighting == NormalWeighting.AREA:
                    weight = area
                elif weighting == NormalWeighting.ANGLE:
                    weight = Mesh._corner_angle(pts, i)

                v = acc.setdefault(vkeys[i], Vector(0.0, 0.0, 0.0))
                v += normal * weight

        normals = {}

        for vk, v in acc.items():
            length = v.magnitude()

            if length > Tolerance.ZERO_TOLERANCE:
                normals[vk] = v / length

        return normals

    def volume(self) -> float:
        """Return the enclosed volume of a closed mesh."""

        total = 0.0

        for fk in self.faces():
            vkeys = self.face[fk]

            if len(vkeys) < 3:
                continue

            p0 = self.vertex_point(vkeys[0])

            if p0 is None:
                continue

            for i in range(1, len(vkeys) - 1):
                p1 = self.vertex_point(vkeys[i])
                p2 = self.vertex_point(vkeys[i + 1])

                if p1 is None or p2 is None:
                    continue

                total += (
                    p0[0] * (p1[1] * p2[2] - p1[2] * p2[1])
                    + p0[1] * (p1[2] * p2[0] - p1[0] * p2[2])
                    + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0])
                )

        return abs(total) / 6.0

    # ═══════════════════════════════════════════════════════════════════════════
    # Triangle BVH
    # ═══════════════════════════════════════════════════════════════════════════
    def _triangle_tasks(
        self, faces: list[list[int]]
    ) -> list[tuple[int, int, int, int, int]]:
        """Every triangle of the mesh: the stored triangulation of an n-gon, a fan from vertex 0 otherwise."""

        vkey_to_idx = self.vertex_index()
        face_keys = self.faces()
        tasks = []

        for fi in range(len(faces)):
            fv = faces[fi]

            if len(fv) < 3:
                continue

            tris = self.triangulation.get(face_keys[fi]) if len(fv) >= 5 else None

            if tris is not None:
                for j in range(len(tris)):
                    t = tris[j]
                    tasks.append(
                        (vkey_to_idx[t[0]], vkey_to_idx[t[1]], vkey_to_idx[t[2]], fi, j)
                    )

                continue

            for j in range(1, len(fv) - 1):
                tasks.append((fv[0], fv[j], fv[j + 1], fi, j))

        return tasks

    @staticmethod
    def _triangle_aabb(p0: Point, p1: Point, p2: Point) -> AABB:
        """AABB of a triangle, padded by a thousandth."""

        min_x = min(p0[0], p1[0], p2[0]) - 0.001
        min_y = min(p0[1], p1[1], p2[1]) - 0.001
        min_z = min(p0[2], p1[2], p2[2]) - 0.001
        max_x = max(p0[0], p1[0], p2[0]) + 0.001
        max_y = max(p0[1], p1[1], p2[1]) + 0.001
        max_z = max(p0[2], p1[2], p2[2]) + 0.001

        return AABB(
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
            (max_x - min_x) * 0.5,
            (max_y - min_y) * 0.5,
            (max_z - min_z) * 0.5,
        )

    @staticmethod
    def _triangle_world_size(aabbs: list[AABB]) -> float:
        """World size for the BVH Morton grid: 2.2 times the largest absolute extent, at least 10."""

        extent = 0.0

        for bb in aabbs:
            extent = max(extent, abs(bb.cx - bb.hx), abs(bb.cx + bb.hx))
            extent = max(extent, abs(bb.cy - bb.hy), abs(bb.cy + bb.hy))
            extent = max(extent, abs(bb.cz - bb.hz), abs(bb.cz + bb.hz))

        return max(2.2 * extent, 10.0)

    def build_triangle_bvh(self, force: bool = False) -> None:
        """Build and cache the BVH over the triangulated faces."""

        if self._triangle_bvh_built and not force:
            return

        self.clear_triangle_bvh()
        vertices, faces = self.to_vertices_and_faces()
        self._vertices_cache = vertices
        tasks = self._triangle_tasks(faces)

        for i0, i1, i2, face_idx, sub_idx in tasks:
            self._triangle_aabbs_cache.append(
                Mesh._triangle_aabb(vertices[i0], vertices[i1], vertices[i2])
            )
            self._triangle_indices_cache.append((i0, i1, i2))
            self._triangle_face_subidx_cache.append((face_idx, sub_idx))

        self._triangle_bvh = SpatialBVH()
        self._triangle_bvh.build_from_aabbs(
            self._triangle_aabbs_cache,
            Mesh._triangle_world_size(self._triangle_aabbs_cache),
        )
        self._triangle_bvh_built = True

    def triangle_bvh_ray_cast(
        self,
        origin: Point,
        direction: Vector,
        candidate_ids: list[int],
        find_all: bool = False,
    ) -> bool:
        """Collect the candidate triangle ids along a ray from the cached BVH; true when any."""

        self.build_triangle_bvh(False)

        if self._triangle_bvh is None:
            return False

        return self._triangle_bvh.ray_cast(origin, direction, candidate_ids, find_all)

    def get_triangle_by_id(
        self, tri_id: int
    ) -> tuple[bool, int, int, Point | None, Point | None, Point | None]:
        """Look up the face index, sub-triangle index and corners of a cached triangle id; false when out of range."""

        if tri_id < 0:
            return (False, 0, 0, None, None, None)

        if tri_id >= len(self._triangle_indices_cache) or tri_id >= len(
            self._triangle_face_subidx_cache
        ):
            return (False, 0, 0, None, None, None)

        tri = self._triangle_indices_cache[tri_id]
        face_idx, sub_idx = self._triangle_face_subidx_cache[tri_id]

        if (
            tri[0] >= len(self._vertices_cache)
            or tri[1] >= len(self._vertices_cache)
            or tri[2] >= len(self._vertices_cache)
        ):
            return (False, 0, 0, None, None, None)

        return (
            True,
            face_idx,
            sub_idx,
            self._vertices_cache[tri[0]],
            self._vertices_cache[tri[1]],
            self._vertices_cache[tri[2]],
        )

    def clear_triangle_bvh(self) -> None:
        """Drop the cached BVH, AABB tree and triangle data."""

        self._triangle_bvh_built = False
        self._triangle_bvh = None
        self._triangle_aabb_tree = None
        self._triangle_aabbs_cache = []
        self._triangle_indices_cache = []
        self._triangle_face_subidx_cache = []
        self._vertices_cache = []

    def build_triangle_aabb_tree(self, force: bool = False) -> None:
        """Build and cache the AABB tree over the triangulated faces."""

        from .spatial_aabbtree import SpatialAABBTree

        self.build_triangle_bvh(False)

        if self._triangle_aabb_tree is not None and not force:
            return

        self._triangle_aabb_tree = SpatialAABBTree()
        self._triangle_aabb_tree.build(self._triangle_aabbs_cache)

    def get_cached_bvh(self) -> SpatialBVH | None:
        """Return the cached triangle BVH, nullptr before build_triangle_bvh."""
        return self._triangle_bvh

    def get_cached_aabb_tree(self) -> Optional["SpatialAABBTree"]:
        """Return the cached triangle AABB tree, nullptr before build_triangle_aabb_tree."""
        return self._triangle_aabb_tree

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════
    def transform(self, xf: "Xform") -> bool:
        """Transform every vertex in place and drop the triangle caches; always true."""

        for vdata in self.vertex.values():
            point = vdata.position()
            point.transform(xf)
            vdata.set_position(point)

        self.clear_triangle_bvh()

        return True

    def transformed(self, xf: "Xform") -> "Mesh":
        """Return a transformed copy."""

        result = copy.deepcopy(self)
        result.transform(xf)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Cutting
    # ═══════════════════════════════════════════════════════════════════════════
    def cut_by_plane(self, plane: Plane) -> "Mesh":
        """Return the part on the side the plane normal points to, every section loop capped by one n-gon face, so a closed mesh stays closed; empty when nothing lies on that side, a copy when everything does."""

        points = {}

        for vk, vd in self.vertex.items():
            points[vk] = vd.position()

        tolerance = _cut_tolerance(points)
        distance = {}
        lowest = 0.0
        highest = 0.0

        for vk, p in points.items():
            d = (p - plane.origin).dot(plane.z_axis)
            distance[vk] = 0.0 if abs(d) <= tolerance else d
            lowest = min(lowest, distance[vk])
            highest = max(highest, distance[vk])

        if lowest >= 0.0:
            return self.duplicate()

        if highest <= 0.0:
            return Mesh()

        crossings = {}
        output = {}
        count = self._max_face

        for fk, ring in sorted(self.face.items()):
            normal = _newell_normal(_cut_points(ring, points))
            rings = _cut_rings(fk, ring, self.face_holes, normal, points)
            pieces = _cut_face(
                fk,
                rings,
                normal,
                plane,
                crossings,
                distance,
                points,
                self._max_vertex,
                tolerance,
            )

            for i in range(len(pieces)):
                if i == 0:
                    output[fk] = pieces[i]
                else:
                    output[count] = pieces[i]
                    count += 1

        for cap in _cut_caps(output, distance, points, plane):
            output[count] = cap
            count += 1

        result = _cut_result(
            output, points, self.face, self.facedata, self.triangulation
        )
        result.name = self.name
        result._objectcolor = self._objectcolor

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _colors_to_json(colors: list[Color]) -> list[float]:
        """Colors as a flat [r, g, b, a, ...] array."""

        arr = []

        for c in colors:
            arr.append(c[0])
            arr.append(c[1])
            arr.append(c[2])
            arr.append(c[3])

        return arr

    @staticmethod
    def _colors_from_json(arr) -> list[Color]:
        """Colors from a flat [r, g, b, a, ...] array."""

        colors = []

        if not isinstance(arr, list):
            return colors

        for i in range(0, len(arr) - 3, 4):
            colors.append(Color(arr[i], arr[i + 1], arr[i + 2], arr[i + 3]))

        return colors

    @staticmethod
    def _halfedge_to_json(halfedge: dict[int, dict[int, int | None]]) -> dict:
        """Halfedge connectivity keyed by vertex, null where no face lies on the left."""

        halfedge_json = {}

        for u, neighbors in halfedge.items():
            neighbor_json = {}

            for v, face_opt in neighbors.items():
                neighbor_json[str(v)] = face_opt

            halfedge_json[str(u)] = neighbor_json

        return halfedge_json

    @staticmethod
    def _triangulation_to_json(triangulation: dict[int, list[list[int]]]) -> dict:
        """Triangles keyed by face as [a, b, c] arrays."""

        triangulation_json = {}

        for fkey, tris in triangulation.items():
            triangulation_json[str(fkey)] = [[t[0], t[1], t[2]] for t in tris]

        return triangulation_json

    @staticmethod
    def _vertex_to_json(vertex: dict[int, VertexData]) -> dict:
        """Vertex positions and attributes keyed by vertex."""

        vertex_json = {}

        for key, vdata in vertex.items():
            vertex_json[str(key)] = {
                "attributes": dict(vdata.attributes),
                "x": vdata.x,
                "y": vdata.y,
                "z": vdata.z,
            }

        return vertex_json

    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        data = {}
        data["color_mode"] = self.color_mode.value
        data["default_edge_attributes"] = dict(self.default_edge_attributes)
        data["default_face_attributes"] = dict(self.default_face_attributes)
        data["default_vertex_attributes"] = dict(self.default_vertex_attributes)
        edgedata_json = {}

        for (u, v), attrs in self.edgedata.items():
            edgedata_json[str(u) + "," + str(v)] = dict(attrs)

        data["edgedata"] = edgedata_json
        face_json = {}

        for key, vertices in self.face.items():
            face_json[str(key)] = list(vertices)

        data["face"] = face_json
        face_holes_json = {}

        for fkey, rings in self.face_holes.items():
            face_holes_json[str(fkey)] = [list(r) for r in rings]

        data["face_holes"] = face_holes_json
        data["facecolors"] = Mesh._colors_to_json(self._facecolors)
        facedata_json = {}

        for key, attrs in self.facedata.items():
            facedata_json[str(key)] = dict(attrs)

        data["facedata"] = facedata_json
        data["guid"] = self.guid
        data["halfedge"] = Mesh._halfedge_to_json(
            self._compute_halfedges()
            if not self.halfedge and self.face
            else self.halfedge
        )
        data["linecolors"] = Mesh._colors_to_json(self._linecolors)
        data["max_face"] = self._max_face
        data["max_vertex"] = self._max_vertex
        data["name"] = self.name
        data["objectcolor"] = self._objectcolor.__jsondump__()
        data["pointcolors"] = Mesh._colors_to_json(self._pointcolors)
        data["triangulation"] = Mesh._triangulation_to_json(self.triangulation)
        data["type"] = "Mesh"
        data["vertex"] = Mesh._vertex_to_json(self.vertex)
        data["widths"] = list(self._widths)

        return data

    @staticmethod
    def _halfedge_from_json(halfedge_json: dict) -> dict[int, dict[int, int | None]]:
        """Halfedge connectivity from a JSON object keyed by vertex."""

        halfedge = {}

        for u_str, neighbors in halfedge_json.items():
            u = int(u_str)
            halfedge[u] = {}

            for v_str, face_val in neighbors.items():
                halfedge[u][int(v_str)] = face_val

        return halfedge

    @staticmethod
    def _vertex_from_json(vertex_json: dict) -> dict[int, VertexData]:
        """Vertex positions and attributes from a JSON object keyed by vertex."""

        vertex = {}

        for key_str, vdata in vertex_json.items():
            vertex_data = VertexData()
            vertex_data.x = vdata["x"]
            vertex_data.y = vdata["y"]
            vertex_data.z = vdata["z"]

            if "attributes" in vdata:
                vertex_data.attributes = Attributes(vdata["attributes"])

            vertex[int(key_str)] = vertex_data

        return vertex

    @staticmethod
    def _face_from_json(face_json: dict) -> dict[int, list[int]]:
        """Face rings from a JSON object keyed by face."""

        face = {}

        for key_str, vertices in face_json.items():
            face[int(key_str)] = list(vertices)

        return face

    @staticmethod
    def _triangulation_from_json(
        triangulation_json: dict,
    ) -> dict[int, list[list[int]]]:
        """Triangles from a JSON object of [a, b, c] arrays keyed by face."""

        triangulation = {}

        for fk_str, tris in triangulation_json.items():
            triangulation[int(fk_str)] = [[t[0], t[1], t[2]] for t in tris]

        return triangulation

    @staticmethod
    def _jsonload_attributes(data: dict, mesh: "Mesh") -> None:
        """Read face holes, face data, edge data and default attributes into mesh."""

        if "face_holes" in data:
            for fk_str, rings in data["face_holes"].items():
                mesh.face_holes[int(fk_str)] = [list(r) for r in rings]

        if "facedata" in data:
            for key_str, attrs in data["facedata"].items():
                mesh.facedata[int(key_str)] = dict(attrs)

        if "edgedata" in data:
            for edge_str, attrs in data["edgedata"].items():
                u_str, v_str = edge_str.split(",")
                mesh.edgedata[(int(u_str), int(v_str))] = dict(attrs)

        if "default_vertex_attributes" in data:
            mesh.default_vertex_attributes = dict(data["default_vertex_attributes"])

        if "default_face_attributes" in data:
            mesh.default_face_attributes = dict(data["default_face_attributes"])

        if "default_edge_attributes" in data:
            mesh.default_edge_attributes = dict(data["default_edge_attributes"])

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> "Mesh":
        """Deserialize from a JSON object."""

        from .file_encoders import file_decode_node

        mesh = cls()

        if "guid" in data:
            mesh.guid = data["guid"]

        if "name" in data:
            mesh.name = data["name"]

        if guid is not None:
            mesh.guid = guid

        if name is not None:
            mesh.name = name

        if "halfedge" in data:
            mesh.halfedge = Mesh._halfedge_from_json(data["halfedge"])

        if "vertex" in data:
            mesh.vertex = Mesh._vertex_from_json(data["vertex"])

        if "halfedge" not in data:
            for key in mesh.vertex:
                mesh.halfedge[key] = {}

        if mesh.vertex:
            mesh._max_vertex = max(mesh.vertex.keys()) + 1

        if "face" in data:
            mesh.face = Mesh._face_from_json(data["face"])

        if mesh.face:
            mesh._max_face = max(mesh.face.keys()) + 1

        Mesh._jsonload_attributes(data, mesh)

        if "max_vertex" in data:
            mesh._max_vertex = data["max_vertex"]

        if "max_face" in data:
            mesh._max_face = data["max_face"]

        if "pointcolors" in data:
            mesh._pointcolors = Mesh._colors_from_json(data["pointcolors"])

        if "facecolors" in data:
            mesh._facecolors = Mesh._colors_from_json(data["facecolors"])

        if "linecolors" in data:
            mesh._linecolors = Mesh._colors_from_json(data["linecolors"])

        if "widths" in data and isinstance(data["widths"], list):
            mesh._widths = list(data["widths"])

        if "objectcolor" in data:
            mesh._objectcolor = file_decode_node(data["objectcolor"])

        if "color_mode" in data:
            mesh.color_mode = (
                ColorMode(data["color_mode"])
                if data["color_mode"] in {m.value for m in ColorMode}
                else ColorMode.OBJECTCOLOR
            )

        if "triangulation" in data:
            mesh.triangulation = Mesh._triangulation_from_json(data["triangulation"])

        return mesh

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Mesh":
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: Union[str, "Path"]) -> None:
        """Write to a JSON file."""

        with open(filename, "w") as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filename: Union[str, "Path"]) -> "Mesh":
        """Read from a JSON file."""

        with open(filename) as f:
            data = json.load(f)

        return cls.__jsonload__(data)

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _colors_to_rgba(colors: list[Color], rgba) -> None:
        """Append colors as flat r, g, b, a floats."""

        for c in colors:
            rgba.append(c[0])
            rgba.append(c[1])
            rgba.append(c[2])
            rgba.append(c[3])

    @staticmethod
    def _colors_from_rgba(rgba) -> list[Color]:
        """Colors from flat r, g, b, a floats."""

        colors = []

        for i in range(0, len(rgba) - 3, 4):
            colors.append(Color(rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]))

        return colors

    @staticmethod
    def _vertices_to_proto(
        vertex: dict[int, VertexData], proto: "mesh_pb2.Mesh"
    ) -> None:
        """Write vertex positions and attributes into the proto."""

        for vkey, vdata in vertex.items():
            vertex_proto = proto.vertices[vkey]
            vertex_proto.x = vdata.x
            vertex_proto.y = vdata.y
            vertex_proto.z = vdata.z

            for k, v in vdata.attributes.items():
                vertex_proto.attributes[k] = v

    @staticmethod
    def _faces_to_proto(
        face: dict[int, list[int]],
        facedata: dict[int, dict[str, float]],
        face_holes: dict[int, list[list[int]]],
        proto: "mesh_pb2.Mesh",
    ) -> None:
        """Write face rings with their attributes and hole rings into the proto."""

        from .proto import mesh_pb2

        for fkey, fverts in face.items():
            face_proto = proto.faces[fkey]
            face_proto.vertices.extend(fverts)

            for k, v in facedata.get(fkey, {}).items():
                face_proto.attributes[k] = v

            for ring in face_holes.get(fkey, []):
                hole_proto = mesh_pb2.HoleRing()
                hole_proto.vertices.extend(ring)
                face_proto.holes.append(hole_proto)

    def to_proto(self) -> "mesh_pb2.Mesh":
        """Convert to the protobuf message."""

        from .proto import mesh_pb2

        proto = mesh_pb2.Mesh()

        if self.has_guid():
            proto.guid = self._guid

        proto.name = self.name

        Mesh._vertices_to_proto(self.vertex, proto)
        Mesh._faces_to_proto(self.face, self.facedata, self.face_holes, proto)

        for fkey, tris in self.triangulation.items():
            tri_list = proto.triangulation[fkey]

            for t in tris:
                tri_list.vertices.append(t[0])
                tri_list.vertices.append(t[1])
                tri_list.vertices.append(t[2])

        for (v1, v2), attrs in self.edgedata.items():
            edge_proto = mesh_pb2.EdgeData()
            edge_proto.vertex1 = v1
            edge_proto.vertex2 = v2

            for k, v in attrs.items():
                edge_proto.attributes[k] = v

            proto.edge_data.append(edge_proto)

        for k, v in self.default_vertex_attributes.items():
            proto.default_vertex_attributes[k] = v

        for k, v in self.default_face_attributes.items():
            proto.default_face_attributes[k] = v

        for k, v in self.default_edge_attributes.items():
            proto.default_edge_attributes[k] = v

        Mesh._colors_to_rgba(self._pointcolors, proto.pointcolors_rgba)
        Mesh._colors_to_rgba(self._facecolors, proto.facecolors_rgba)
        Mesh._colors_to_rgba(self._linecolors, proto.linecolors_rgba)
        proto.widths.extend(self._widths)
        proto.objectcolor.CopyFrom(self._objectcolor.to_proto())
        proto.color_mode = list(ColorMode).index(self.color_mode)

        return proto

    @staticmethod
    def _vertices_from_proto(proto: "mesh_pb2.Mesh") -> dict[int, VertexData]:
        """Vertex positions and attributes of the proto."""

        vertex = {}

        for vkey, vdata in proto.vertices.items():
            vd = VertexData(Point(vdata.x, vdata.y, vdata.z))
            vd.attributes = Attributes(dict(vdata.attributes))
            vertex[vkey] = vd

        return vertex

    @staticmethod
    def _faces_from_proto(proto: "mesh_pb2.Mesh", mesh: "Mesh") -> None:
        """Read face rings with their attributes and hole rings into mesh."""

        for fkey, fdata in proto.faces.items():
            mesh.face[fkey] = list(fdata.vertices)

            if fdata.attributes:
                mesh.facedata[fkey] = dict(fdata.attributes)

            if fdata.holes:
                mesh.face_holes[fkey] = [list(h.vertices) for h in fdata.holes]

    @staticmethod
    def _triangulation_from_proto(proto: "mesh_pb2.Mesh") -> dict[int, list[list[int]]]:
        """Triangles of the proto keyed by face."""

        triangulation = {}

        for fkey, tri_list in proto.triangulation.items():
            vlist = list(tri_list.vertices)
            tris = []

            for i in range(0, len(vlist) - 2, 3):
                tris.append([vlist[i], vlist[i + 1], vlist[i + 2]])

            triangulation[fkey] = tris

        return triangulation

    @classmethod
    def from_proto(cls, proto: "mesh_pb2.Mesh") -> "Mesh":
        """Construct from the protobuf message."""

        mesh = cls()

        if proto.guid:
            mesh.guid = proto.guid

        mesh.name = proto.name

        mesh.vertex = Mesh._vertices_from_proto(proto)
        Mesh._faces_from_proto(proto, mesh)
        mesh.triangulation = Mesh._triangulation_from_proto(proto)

        for edata in proto.edge_data:
            mesh.edgedata[(edata.vertex1, edata.vertex2)] = dict(edata.attributes)

        mesh.default_vertex_attributes = dict(proto.default_vertex_attributes)
        mesh.default_face_attributes = dict(proto.default_face_attributes)
        mesh.default_edge_attributes = dict(proto.default_edge_attributes)
        mesh._pointcolors = Mesh._colors_from_rgba(proto.pointcolors_rgba)
        mesh._facecolors = Mesh._colors_from_rgba(proto.facecolors_rgba)
        mesh._linecolors = Mesh._colors_from_rgba(proto.linecolors_rgba)
        mesh._widths = list(proto.widths)

        if proto.HasField("objectcolor"):
            mesh._objectcolor = Color.from_proto(proto.objectcolor)

        mesh.color_mode = (
            list(ColorMode)[proto.color_mode]
            if 0 <= proto.color_mode < 4
            else ColorMode.OBJECTCOLOR
        )

        if mesh.vertex:
            mesh._max_vertex = max(mesh.vertex.keys()) + 1

        if mesh.face:
            mesh._max_face = max(mesh.face.keys()) + 1

        return mesh

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Mesh":
        """Deserialize from protobuf bytes."""

        from .proto import mesh_pb2

        proto = mesh_pb2.Mesh()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: Union[str, "Path"]) -> None:
        """Write to a protobuf file."""

        data = self.pb_dumps()

        with open(filename, "wb") as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filename: Union[str, "Path"]) -> "Mesh":
        """Read from a protobuf file."""

        with open(filename, "rb") as f:
            data = f.read()

        return cls.pb_loads(data)

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return the "Mesh(name=..., vertices=..., faces=...)" form."""
        return f"Mesh(name={self.name}, vertices={self.number_of_vertices()}, faces={self.number_of_faces()})"

    def __repr__(self) -> str:
        """Return the multi-line form with name, vertices, faces and edges."""
        return f"Mesh(\n  name={self.name},\n  vertices={self.number_of_vertices()},\n  faces={self.number_of_faces()},\n  edges={self.number_of_edges()}\n)"

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════════════════════════
    def _directed_face_edges(self) -> set[tuple[int, int]]:
        """Return every directed edge (u, v) some face ring walks."""

        s = set()

        for verts in self.face.values():
            n = len(verts)

            for i in range(n):
                s.add((verts[i], verts[(i + 1) % n]))

        return s

    def _compute_halfedges(self) -> dict[int, dict[int, int | None]]:
        """Return the face-derived halfedge connectivity, computed without mutating."""

        he = {}

        for vkey in self.vertex:
            he[vkey] = {}

        for fkey, verts in self.face.items():
            n = len(verts)

            for i in range(n):
                u = verts[i]
                v = verts[(i + 1) % n]
                he.setdefault(u, {})[v] = fkey

                if u not in he.setdefault(v, {}):
                    he[v][u] = None

        return he
