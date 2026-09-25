from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import json
import uuid
import numpy as np
from .color import Color
from .mesh import Mesh
from .nurbscurve import NurbsCurve
from .plane import Plane
from .point import Point
from .tolerance import Tolerance
from .vector import Vector
from .xform import Xform
from . import nurbsknot

if TYPE_CHECKING:
    from pathlib import Path
    from .brep import BRep
    from .line import Line
    from .nurbssurface_trimmed import NurbsSurfaceTrimmed
    from .proto import nurbssurface_pb2


# ═══════════════════════════════════════════════════════════════════════════
# File helpers
# ═══════════════════════════════════════════════════════════════════════════
def _expand_nurbsknots(knots: list[float], mults: list[int]) -> list[float]:
    """Repeat each distinct knot by its multiplicity."""

    full = []

    for i in range(len(knots)):
        for m in range(mults[i]):
            full.append(float(knots[i]))

    return full


def _binomial(n: int, k: int) -> float:
    """C(n, k)."""

    r = 1.0

    for i in range(k):
        r = r * (n - i) / (i + 1)

    return r


def _is_rational_weights(weights: list[list[float]]) -> bool:
    """True when any weight differs from one."""

    for row in weights:
        for w in row:
            if abs(w - 1.0) > Tolerance.ZERO_TOLERANCE:
                return True

    return False


def _basis_table(knot, degree: int, base: int, t: float) -> list[list[float]]:
    """Triangular table of basis values and knot differences (Piegl & Tiller A2.3)."""

    order = degree + 1
    ndu = [[0.0] * order for _ in range(order)]
    ndu[0][0] = 1.0
    left = [0.0] * order
    right = [0.0] * order

    for j in range(1, degree + 1):
        left[j] = t - knot[base - j]
        right[j] = knot[base + j - 1] - t
        saved = 0.0

        for r in range(j):
            ndu[j][r] = right[r + 1] + left[j - r]
            temp = ndu[r][j - 1] / ndu[j][r]
            ndu[r][j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        ndu[j][j] = saved

    return ndu


def _basis_table_derivatives(
    ndu: list[list[float]], degree: int, deriv_order: int
) -> list[list[float]]:
    """Basis derivatives ders[k][j] from the triangular table (Piegl & Tiller A2.3)."""

    order = degree + 1
    ders = [[0.0] * order for _ in range(deriv_order + 1)]

    for j in range(degree + 1):
        ders[0][j] = ndu[j][degree]

    a = [[0.0] * order for _ in range(2)]

    for r in range(degree + 1):
        s1 = 0
        s2 = 1
        a[0][0] = 1.0

        for k in range(1, deriv_order + 1):
            d = 0.0
            rk = r - k
            pk = degree - k

            if r >= k:
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                d = a[s2][0] * ndu[rk][pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else degree - r

            for j in range(j1, j2 + 1):
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
                d += a[s2][j] * ndu[rk + j][pk]

            if r <= pk:
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
                d += a[s2][k] * ndu[r][pk]

            ders[k][r] = d
            s1, s2 = s2, s1

    factor = float(degree)

    for k in range(1, deriv_order + 1):
        for j in range(degree + 1):
            ders[k][j] *= factor

        factor *= degree - k

    return ders


def _surface_aabb(srf: NurbsSurface) -> tuple[list[float], list[float]]:
    """Bounding box of a 7 x 7 sample of the surface."""

    n = 6
    u0, u1 = srf.domain(0)
    v0, v1 = srf.domain(1)
    lo = [1e30, 1e30, 1e30]
    hi = [-1e30, -1e30, -1e30]

    for i in range(n + 1):
        for j in range(n + 1):
            p = srf.point_at(u0 + (u1 - u0) * i / n, v0 + (v1 - v0) * j / n)

            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])

    return lo, hi


def _aabb_overlap_pad(
    a: tuple[list[float], list[float]], b: tuple[list[float], list[float]]
) -> bool:
    """Boxes overlap once a is padded by a thousandth of its longest side."""

    m = max(a[1][0] - a[0][0], a[1][1] - a[0][1], a[1][2] - a[0][2]) * 1e-3

    for k in range(3):
        if a[0][k] - m > b[1][k] or b[0][k] - m > a[1][k]:
            return False

    return True


def _colors_to_json(colors: list[Color]) -> list[float]:
    """Flatten colors to r, g, b, a values."""

    arr = []

    for c in colors:
        arr.extend([c.r, c.g, c.b, c.a])

    return arr


def _colors_from_json(data: dict, key: str) -> list[Color]:
    """Read colors from a flat r, g, b, a list under key."""

    colors = []
    arr = data.get(key, [])

    for i in range(0, len(arr) - 3, 4):
        colors.append(Color(arr[i], arr[i + 1], arr[i + 2], arr[i + 3]))

    return colors


def _colors_to_proto(colors: list[Color], field) -> None:
    """Append colors to a repeated proto field."""

    for c in colors:
        cp = field.add()
        cp.r = c.r
        cp.g = c.g
        cp.b = c.b
        cp.a = c.a


def _colors_from_proto(field) -> list[Color]:
    """Read colors from a repeated proto field."""

    colors = []

    for c in field:
        colors.append(Color(c.r, c.g, c.b, c.a))

    return colors


class NurbsSurface:
    """A NURBS surface: OpenNURBS layout, nurbsknot count = order + cv_count - 2 per direction, homogeneous row-major CVs when rational."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(
        self,
        dimension: int = 0,
        is_rational: bool = False,
        order0: int = 0,
        order1: int = 0,
        cv_count0: int = 0,
        cv_count1: int = 0,
    ):
        """Construct an unset surface with the given layout."""

        self._guid = None  # Lazily minted GUID.
        self.name = "my_nurbssurface"  # Surface name.
        self.width = 1.0  # Display width.
        self.pointcolors: list[Color] = []  # Display color per control point.
        self.facecolors: list[Color] = []  # Display color per mesh face.
        self.linecolors: list[Color] = []  # Display color per control polygon segment.
        self.m_mesh: Mesh | None = None  # Cached mesh from mesh() or mesh_adaptive().

        self.initialize()
        self.create_raw(dimension, is_rational, order0, order1, cv_count0, cv_count1)

    def __deepcopy__(self, memo):
        """Copy with a new guid and the same data."""

        result = NurbsSurface()
        result._deep_copy_from(self)
        memo[id(self)] = result

        return result

    def duplicate(self) -> NurbsSurface:
        """Copy with a new guid and the same data."""
        return copy.deepcopy(self)

    # ═══════════════════════════════════════════════════════════════════════════
    # Static constructors
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def create(
        periodic_u: bool,
        periodic_v: bool,
        degree_u: int,
        degree_v: int,
        cv_count_u: int,
        cv_count_v: int,
        points: list[Point],
    ) -> NurbsSurface:
        """Construct a clamped or periodic uniform surface through cv_count_u x cv_count_v points in row-major order (u slowest)."""

        if degree_u < 1 or degree_v < 1:
            raise ValueError(
                f"NurbsSurface::create: degree must be >= 1, got degree_u={degree_u}, degree_v={degree_v}"
            )

        if cv_count_u < degree_u + 1:
            raise ValueError(
                f"NurbsSurface::create: cv_count_u ({cv_count_u}) must be >= degree_u+1 ({degree_u + 1})"
            )

        if cv_count_v < degree_v + 1:
            raise ValueError(
                f"NurbsSurface::create: cv_count_v ({cv_count_v}) must be >= degree_v+1 ({degree_v + 1})"
            )

        expected = cv_count_u * cv_count_v

        if len(points) != expected:
            raise ValueError(
                f"NurbsSurface::create: expected {expected} points ({cv_count_u}x{cv_count_v}), got {len(points)}"
            )

        surface = NurbsSurface()
        surface.create_raw(
            3,
            False,
            degree_u + 1,
            degree_v + 1,
            cv_count_u,
            cv_count_v,
            periodic_u,
            periodic_v,
            1.0,
            1.0,
        )

        for i in range(cv_count_u):
            for j in range(cv_count_v):
                surface.set_cv(i, j, points[i * cv_count_v + j])

        return surface

    @staticmethod
    def create_from_parameters(
        points: list[list[Point]],
        weights: list[list[float]],
        knots_u: list[float],
        knots_v: list[float],
        mults_u: list[int],
        mults_v: list[int],
        degree_u: int,
        degree_v: int,
        periodic_u: bool = False,
        periodic_v: bool = False,
    ) -> NurbsSurface:
        """Construct from points[iv][iu], weights[iv][iu], distinct knots and multiplicities per direction (OCCT convention)."""

        nv = len(points)
        nu = len(points[0]) if nv > 0 else 0
        order_u = degree_u + 1
        order_v = degree_v + 1

        if nu < order_u or nv < order_v or periodic_u or periodic_v:
            return NurbsSurface()

        if len(knots_u) != len(mults_u) or len(knots_v) != len(mults_v):
            return NurbsSurface()

        rational = _is_rational_weights(weights)
        full_u = _expand_nurbsknots(knots_u, mults_u)
        full_v = _expand_nurbsknots(knots_v, mults_v)
        kc_u = order_u + nu - 2
        kc_v = order_v + nv - 2

        if len(full_u) != kc_u + 2 or len(full_v) != kc_v + 2:
            return NurbsSurface()

        surface = NurbsSurface()

        if not surface.create_raw(3, rational, order_u, order_v, nu, nv):
            return NurbsSurface()

        for i in range(kc_u):
            surface.set_nurbsknot(0, i, full_u[i + 1])

        for i in range(kc_v):
            surface.set_nurbsknot(1, i, full_v[i + 1])

        for i in range(nu):
            for j in range(nv):
                p = points[j][i]

                if rational:
                    w = weights[j][i]
                    surface.set_cv_4d(i, j, p[0] * w, p[1] * w, p[2] * w, w)
                else:
                    surface.set_cv(i, j, p)

        return surface

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __eq__(self, other) -> bool:
        """Compare name, width, colors, layout, nurbsknots and CVs; guid ignored."""

        if not isinstance(other, NurbsSurface):
            return False

        if self.name != other.name or self.width != other.width:
            return False

        if (
            self.pointcolors != other.pointcolors
            or self.facecolors != other.facecolors
            or self.linecolors != other.linecolors
        ):
            return False

        if self.m_dim != other.m_dim or self.m_is_rat != other.m_is_rat:
            return False

        if (
            self.m_order != other.m_order
            or self.m_cv_count != other.m_cv_count
            or self.m_cv_stride != other.m_cv_stride
        ):
            return False

        if not np.array_equal(
            self.m_nurbsknot[0], other.m_nurbsknot[0]
        ) or not np.array_equal(self.m_nurbsknot[1], other.m_nurbsknot[1]):
            return False

        return np.array_equal(self.m_cv, other.m_cv)

    def __ne__(self, other) -> bool:
        """Compare name, width, colors, layout, nurbsknots and CVs; guid ignored."""
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformation
    # ═══════════════════════════════════════════════════════════════════════════
    def transform(self, xform: Xform) -> bool:
        """Transform every CV in place."""

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                p = self.get_cv(i, j)
                p.transform(xform)
                self.set_cv(i, j, p)

        return True

    def transformed(self, xform: Xform) -> NurbsSurface:
        """Return a transformed copy."""

        result = self.duplicate()
        result.transform(xform)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Initialization
    # ═══════════════════════════════════════════════════════════════════════════
    def initialize(self) -> None:
        """Reset every field to the empty invalid surface."""

        self._guid = None
        self.name = "my_nurbssurface"
        self.width = 1.0
        self.pointcolors = []
        self.facecolors = []
        self.linecolors = []
        self.m_dim = 0  # Coordinate dimension.
        self.m_is_rat = 0  # 1 when rational, 0 otherwise.
        self.m_order = [0, 0]  # Degree + 1 per direction.
        self.m_cv_count = [0, 0]  # Number of control vertices per direction.
        self.m_cv_stride = [0, 0]  # Doubles between consecutive CVs per direction.
        self.m_nurbsknot = [
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        ]  # NurbsKnot vector per direction, order + cv_count - 2 values.
        self.m_cv = np.zeros(0, dtype=np.float64)  # Flat CV array.

    def create_raw(
        self,
        dimension: int,
        is_rational: bool,
        order0: int,
        order1: int,
        cv_count0: int,
        cv_count1: int,
        is_periodic_u: bool = False,
        is_periodic_v: bool = False,
        nurbsknot_delta_u: float = 1.0,
        nurbsknot_delta_v: float = 1.0,
    ) -> bool:
        """Allocate nurbsknots (clamped or periodic uniform) and zeroed CVs; false when order < 2 or cv_count < order."""

        if (
            dimension < 1
            or order0 < 2
            or order1 < 2
            or cv_count0 < order0
            or cv_count1 < order1
        ):
            return False

        self.destroy()
        self.m_dim = dimension
        self.m_is_rat = 1 if is_rational else 0
        self.m_order = [order0, order1]
        self.m_cv_count = [cv_count0, cv_count1]
        self.m_cv_stride = [self.cv_size() * cv_count1, self.cv_size()]
        self.m_nurbsknot = [
            np.zeros(order0 + cv_count0 - 2, dtype=np.float64),
            np.zeros(order1 + cv_count1 - 2, dtype=np.float64),
        ]
        self.m_cv = np.zeros(cv_count0 * cv_count1 * self.cv_size(), dtype=np.float64)
        self._zero_cvs()

        if is_periodic_u:
            self._set_periodic_uniform_nurbsknot_vector(0, nurbsknot_delta_u)
        else:
            self._set_clamped_uniform_nurbsknot_vector(0, nurbsknot_delta_u)

        if is_periodic_v:
            self._set_periodic_uniform_nurbsknot_vector(1, nurbsknot_delta_v)
        else:
            self._set_clamped_uniform_nurbsknot_vector(1, nurbsknot_delta_v)

        return True

    def create_clamped_uniform(
        self,
        dimension: int,
        order0: int,
        order1: int,
        cv_count0: int,
        cv_count1: int,
        nurbsknot_delta0: float = 1.0,
        nurbsknot_delta1: float = 1.0,
    ) -> bool:
        """Allocate a non-rational surface with clamped uniform nurbsknots of the given spacing."""

        return self.create_raw(
            dimension,
            False,
            order0,
            order1,
            cv_count0,
            cv_count1,
            False,
            False,
            nurbsknot_delta0,
            nurbsknot_delta1,
        )

    def destroy(self) -> None:
        """Clear all data; is_valid() is false afterwards."""
        self.initialize()

    # ═══════════════════════════════════════════════════════════════════════════
    # Boolean queries
    # ═══════════════════════════════════════════════════════════════════════════
    def is_valid(self) -> bool:
        """Return whether orders >= 2, cv_count >= order, nurbsknot vectors are valid and the CV array is large enough."""

        if self.m_dim < 1 or self.m_order[0] < 2 or self.m_order[1] < 2:
            return False

        if self.m_cv_count[0] < self.m_order[0] or self.m_cv_count[1] < self.m_order[1]:
            return False

        if not self.is_valid_nurbsknot_vector(0) or not self.is_valid_nurbsknot_vector(
            1
        ):
            return False

        return len(self.m_cv) >= self.cv_count() * self.cv_size()

    def is_valid_nurbsknot_vector(self, dir: int) -> bool:
        """Return whether the nurbsknot vector in dir has the right length and is non-decreasing."""

        if dir < 0 or dir > 1:
            return False

        kc = self.nurbsknot_count(dir)

        if len(self.m_nurbsknot[dir]) != kc:
            return False

        for i in range(1, kc):
            if self.m_nurbsknot[dir][i] < self.m_nurbsknot[dir][i - 1]:
                return False

        return True

    def is_rational(self) -> bool:
        """Return whether the CVs carry weights."""
        return self.m_is_rat != 0

    def is_closed(self, dir: int) -> bool:
        """Return whether the first and last CV rows across dir coincide when clamped, else whether dir is periodic."""

        if dir < 0 or dir > 1 or not self.is_valid():
            return False

        if not self.is_clamped(dir, 2):
            return self.is_periodic(dir)

        last = self.m_cv_count[dir] - 1

        for k in range(self.m_cv_count[1 - dir]):
            a = self.get_cv(k, 0) if dir else self.get_cv(0, k)
            b = self.get_cv(k, last) if dir else self.get_cv(last, k)

            if a.distance(b) > Tolerance.ZERO_TOLERANCE:
                return False

        return True

    def is_periodic(self, dir: int) -> bool:
        """Return whether dir has uniform nurbsknot spacing and the first degree CV rows repeat the last."""

        if dir < 0 or dir > 1 or not self.is_valid():
            return False

        if not nurbsknot.is_periodic(
            self.m_order[dir], self.m_cv_count[dir], self.m_nurbsknot[dir]
        ):
            return False

        deg = self.degree(dir)
        n = self.m_cv_count[dir]

        for k in range(self.m_cv_count[1 - dir]):
            for i in range(deg):
                a = self.get_cv(k, i) if dir else self.get_cv(i, k)
                b = self.get_cv(k, n - deg + i) if dir else self.get_cv(n - deg + i, k)

                if a.distance(b) > Tolerance.ZERO_TOLERANCE:
                    return False

        return True

    def is_planar(
        self, plane: Plane | None = None, tolerance: float = Tolerance.ZERO_TOLERANCE
    ) -> bool:
        """Return whether every CV is within tolerance of one plane, written to plane when given."""

        if not self.is_valid():
            return False

        p0 = self.get_cv(0, 0)
        va = Vector(0, 0, 0)
        normal = Vector(0, 0, 0)

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                p = self.get_cv(i, j)
                v = p - p0

                if va.magnitude() < 1e-14:
                    va = v
                elif normal.magnitude() < 1e-14:
                    normal = va.cross(v)

        if normal.magnitude() < 1e-14:
            return True

        normal = normal / normal.magnitude()

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                p = self.get_cv(i, j)
                v = p - p0

                if abs(v.dot(normal)) > tolerance:
                    return False

        if plane is not None:
            NurbsSurface._assign_plane(plane, Plane.from_point_normal(p0, normal))

        return True

    def is_singular(self, side: int) -> bool:
        """Return whether a clamped side collapses to one point; side: 0 south (v0), 1 east (u1), 2 north (v1), 3 west (u0)."""

        if side < 0 or side > 3 or not self.is_valid():
            return False

        fix = 1 if side % 2 == 0 else 0
        end = 0 if side == 0 or side == 3 else 1

        if not self.is_clamped(fix, end):
            return False

        at = self.m_cv_count[fix] - 1 if end else 0
        first = self.get_cv(0, at) if fix else self.get_cv(at, 0)

        for k in range(1, self.m_cv_count[1 - fix]):
            p = self.get_cv(k, at) if fix else self.get_cv(at, k)

            if p.distance(first) > Tolerance.ZERO_TOLERANCE:
                return False

        return True

    def is_clamped(self, dir: int, end: int = 2) -> bool:
        """Return whether dir has full end multiplicity; end: 0 start, 1 end, 2 both."""

        if dir < 0 or dir > 1:
            return False

        return nurbsknot.is_clamped(
            self.m_order[dir], self.m_cv_count[dir], self.m_nurbsknot[dir], end
        )

    def is_duplicate(
        self,
        other: NurbsSurface,
        ignore_parameterization: bool,
        tolerance: float = Tolerance.ZERO_TOLERANCE,
    ) -> bool:
        """Return whether layout, CVs and weights match within tolerance; nurbsknots too unless ignore_parameterization."""

        if not self.is_valid() or not other.is_valid():
            return False

        if self.m_dim != other.m_dim or self.m_is_rat != other.m_is_rat:
            return False

        if self.m_order != other.m_order or self.m_cv_count != other.m_cv_count:
            return False

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                if self.get_cv(i, j).distance(other.get_cv(i, j)) > tolerance:
                    return False

                if abs(self.weight(i, j) - other.weight(i, j)) > tolerance:
                    return False

        if ignore_parameterization:
            return True

        for dir in range(2):
            for i in range(self.nurbsknot_count(dir)):
                if abs(self.nurbsknot(dir, i) - other.nurbsknot(dir, i)) > tolerance:
                    return False

        return True

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

    def dimension(self) -> int:
        """Return the coordinate dimension."""
        return self.m_dim

    def order(self, dir: int) -> int:
        """Return the order (degree + 1) in dir."""
        return self.m_order[dir] if dir == 0 or dir == 1 else 0

    def degree(self, dir: int) -> int:
        """Return the degree in dir."""
        return self.m_order[dir] - 1 if dir == 0 or dir == 1 else 0

    def cv_count(self, dir: int | None = None) -> int:
        """Return the number of control vertices in dir, or cv_count(0) * cv_count(1) without a direction."""

        if dir is None:
            return self.m_cv_count[0] * self.m_cv_count[1]

        return self.m_cv_count[dir] if dir == 0 or dir == 1 else 0

    def cv_size(self) -> int:
        """Return the doubles per CV: dimension + 1 when rational."""
        return self.m_dim + 1 if self.m_is_rat else self.m_dim

    def nurbsknot_count(self, dir: int) -> int:
        """Return order + cv_count - 2 in dir."""
        return self.m_order[dir] + self.m_cv_count[dir] - 2 if dir == 0 or dir == 1 else 0

    def span_count(self, dir: int) -> int:
        """Return cv_count - order + 1 in dir."""
        return self.m_cv_count[dir] - self.m_order[dir] + 1 if dir == 0 or dir == 1 else 0

    # ═══════════════════════════════════════════════════════════════════════════
    # Control vertex access
    # ═══════════════════════════════════════════════════════════════════════════
    def cv(self, i: int, j: int) -> np.ndarray | None:
        """Return the mutable pointer to CV[i][j], cv_size() doubles (x*w, y*w, z*w, w when rational), nullptr when out of range."""

        if i < 0 or i >= self.m_cv_count[0] or j < 0 or j >= self.m_cv_count[1]:
            return None

        idx = i * self.m_cv_stride[0] + j * self.m_cv_stride[1]

        return self.m_cv[idx : idx + self.cv_size()]

    def get_cv(self, i: int, j: int) -> Point:
        """Return the Euclidean CV (divided by weight when rational), origin when out of range."""

        cv_ptr = self.cv(i, j)

        if cv_ptr is None:
            return Point(0, 0, 0)

        return self._dehomogenize(cv_ptr)

    def get_cv_4d(self, i: int, j: int) -> tuple[bool, float, float, float, float]:
        """Return (ok, x, y, z, w) of the homogeneous CV, w = 1 when non-rational."""

        cv_ptr = self.cv(i, j)

        if cv_ptr is None:
            return (False, 0.0, 0.0, 0.0, 1.0)

        x = float(cv_ptr[0])
        y = float(cv_ptr[1]) if self.m_dim > 1 else 0.0
        z = float(cv_ptr[2]) if self.m_dim > 2 else 0.0
        w = float(cv_ptr[self.m_dim]) if self.m_is_rat else 1.0

        return (True, x, y, z, w)

    def set_cv(self, i: int, j: int, point: Point) -> bool:
        """Set the Euclidean CV, keeping its weight."""

        cv_ptr = self.cv(i, j)

        if cv_ptr is None:
            return False

        w = (
            cv_ptr[self.m_dim]
            if self.m_is_rat and abs(cv_ptr[self.m_dim]) > 1e-14
            else 1.0
        )
        cv_ptr[0] = point[0] * w

        if self.m_dim > 1:
            cv_ptr[1] = point[1] * w

        if self.m_dim > 2:
            cv_ptr[2] = point[2] * w

        return True

    def set_cv_4d(self, i: int, j: int, x: float, y: float, z: float, w: float) -> bool:
        """Set the homogeneous CV; w ignored when non-rational."""

        cv_ptr = self.cv(i, j)

        if cv_ptr is None:
            return False

        cv_ptr[0] = x

        if self.m_dim > 1:
            cv_ptr[1] = y

        if self.m_dim > 2:
            cv_ptr[2] = z

        if self.m_is_rat:
            cv_ptr[self.m_dim] = w

        return True

    def weight(self, i: int, j: int) -> float:
        """Return the weight of CV[i][j], 1 when non-rational."""

        cv_ptr = self.cv(i, j)

        return (
            float(cv_ptr[self.m_dim]) if self.m_is_rat and cv_ptr is not None else 1.0
        )

    def set_weight(self, i: int, j: int, weight: float) -> bool:
        """Rescale the homogeneous CV to the new weight so the Euclidean point stays; false when non-rational."""

        cv_ptr = self.cv(i, j)

        if not self.m_is_rat or cv_ptr is None:
            return False

        old_w = cv_ptr[self.m_dim] if abs(cv_ptr[self.m_dim]) > 1e-14 else 1.0
        new_w = weight if abs(weight) > 1e-14 else 1.0
        scale = new_w / old_w

        for d in range(self.m_dim):
            cv_ptr[d] *= scale

        cv_ptr[self.m_dim] = new_w

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # NurbsKnot access
    # ═══════════════════════════════════════════════════════════════════════════
    def nurbsknot(self, dir: int, nurbsknot_index: int) -> float:
        """Return the nurbsknot at nurbsknot_index in dir."""

        if (
            dir < 0
            or dir > 1
            or nurbsknot_index < 0
            or nurbsknot_index >= len(self.m_nurbsknot[dir])
        ):
            return 0.0

        return float(self.m_nurbsknot[dir][nurbsknot_index])

    def set_nurbsknot(
        self, dir: int, nurbsknot_index: int, nurbsknot_value: float
    ) -> bool:
        """Set the nurbsknot at nurbsknot_index in dir."""

        if (
            dir < 0
            or dir > 1
            or nurbsknot_index < 0
            or nurbsknot_index >= len(self.m_nurbsknot[dir])
        ):
            return False

        self.m_nurbsknot[dir][nurbsknot_index] = nurbsknot_value

        return True

    def nurbsknot_multiplicity(self, dir: int, nurbsknot_index: int) -> int:
        """Return the multiplicity of the nurbsknot at nurbsknot_index in dir."""

        if dir < 0 or dir > 1:
            return 0

        return nurbsknot.multiplicity(
            self.m_order[dir],
            self.m_cv_count[dir],
            self.m_nurbsknot[dir],
            nurbsknot_index,
        )

    def get_nurbsknots(self, dir: int) -> np.ndarray:
        """Return a copy of the nurbsknot vector in dir."""

        return (
            self.m_nurbsknot[dir].copy()
            if dir == 0 or dir == 1
            else np.zeros(0, dtype=np.float64)
        )

    def insert_nurbsknot(
        self, dir: int, nurbsknot_value: float, nurbsknot_multiplicity: int = 1
    ) -> bool:
        """Insert a nurbsknot with the given multiplicity in dir without changing the shape."""

        if (
            dir < 0
            or dir > 1
            or not self.is_valid()
            or nurbsknot_multiplicity <= 0
            or nurbsknot_multiplicity >= self.m_order[dir]
        ):
            return False

        t0, t1 = self.domain(dir)

        if nurbsknot_value < t0 or nurbsknot_value > t1:
            return False

        crv = self._to_curve(dir)

        if not crv.insert_nurbsknot(nurbsknot_value, nurbsknot_multiplicity):
            return False

        return self._from_curve(crv, dir)

    # ═══════════════════════════════════════════════════════════════════════════
    # Domain
    # ═══════════════════════════════════════════════════════════════════════════
    def domain(self, dir: int) -> tuple[float, float]:
        """Return [nurbsknot[order - 2], nurbsknot[cv_count - 1]] in dir."""

        if dir < 0 or dir > 1 or not self.is_valid():
            return (0.0, 0.0)

        return (
            float(self.m_nurbsknot[dir][self.m_order[dir] - 2]),
            float(self.m_nurbsknot[dir][self.m_cv_count[dir] - 1]),
        )

    def set_domain(self, dir: int, t0: float, t1: float) -> bool:
        """Linearly remap the nurbsknots in dir onto [t0, t1]."""

        if dir < 0 or dir > 1 or not self.is_valid() or t0 >= t1:
            return False

        d0, d1 = self.domain(dir)

        if abs(d1 - d0) < 1e-14:
            return False

        scale = (t1 - t0) / (d1 - d0)

        for i in range(len(self.m_nurbsknot[dir])):
            self.m_nurbsknot[dir][i] = t0 + (self.m_nurbsknot[dir][i] - d0) * scale

        return True

    def get_span_vector(self, dir: int) -> list[float]:
        """Return the distinct nurbsknot values inside the domain of dir."""

        spans = []

        if dir < 0 or dir > 1 or not self.is_valid():
            return spans

        spans.append(float(self.m_nurbsknot[dir][self.m_order[dir] - 2]))

        for i in range(self.m_order[dir] - 1, self.m_cv_count[dir]):
            if self.m_nurbsknot[dir][i] > spans[-1]:
                spans.append(float(self.m_nurbsknot[dir][i]))

        return spans

    # ═══════════════════════════════════════════════════════════════════════════
    # Division
    # ═══════════════════════════════════════════════════════════════════════════
    def divide_by_count_points(
        self, nu: int, nv: int
    ) -> tuple[list[list[Point]], list[list[Vector]], list[list[tuple[float, float]]]]:
        """Points, normals and (u, v) on a (nu + 1) x (nv + 1) grid over the domain."""

        grid = []
        normals = []
        params = []

        if not self.is_valid():
            return grid, normals, params

        u0, u1 = self.domain(0)
        v0, v1 = self.domain(1)

        for i in range(nu + 1):
            u = u0 + (u1 - u0) * i / nu if nu > 0 else u0
            grid.append([])
            normals.append([])
            params.append([])

            for j in range(nv + 1):
                v = v0 + (v1 - v0) * j / nv if nv > 0 else v0
                grid[i].append(self.point_at(u, v))
                normals[i].append(self.normal_at(u, v))
                params[i].append((u, v))

        return grid, normals, params

    def divide_by_count_planes(
        self, nu: int, nv: int
    ) -> tuple[list[list[Plane]], list[list[tuple[float, float]]]]:
        """Frames (x = dS/du, y = dS/dv) and (u, v) on a (nu + 1) x (nv + 1) grid over the domain."""

        grid = []
        params = []

        if not self.is_valid():
            return grid, params

        u0, u1 = self.domain(0)
        v0, v1 = self.domain(1)

        for i in range(nu + 1):
            u = u0 + (u1 - u0) * i / nu if nu > 0 else u0
            grid.append([])
            params.append([])

            for j in range(nv + 1):
                v = v0 + (v1 - v0) * j / nv if nv > 0 else v0
                derivs = self.evaluate(u, v, 1)
                x_axis = derivs[2]
                y_axis = derivs[1]

                if x_axis.magnitude() > 1e-14:
                    x_axis = x_axis.normalized()

                if y_axis.magnitude() > 1e-14:
                    y_axis = y_axis.normalized()

                grid[i].append(
                    Plane.from_frame(
                        self.point_at(u, v), x_axis, y_axis, self.normal_at(u, v)
                    )
                )
                params[i].append((u, v))

        return grid, params

    # ═══════════════════════════════════════════════════════════════════════════
    # Evaluation
    # ═══════════════════════════════════════════════════════════════════════════
    def point_at(self, u: float, v: float) -> Point:
        """Return S(u, v) by the tensor-product basis, origin when invalid."""

        if not self.is_valid():
            return Point(0, 0, 0)

        span_u = self._find_span(0, u)
        span_v = self._find_span(1, v)
        nu = nurbsknot.eval_basis(self.m_order[0], self.m_nurbsknot[0], span_u, u)
        nv = nurbsknot.eval_basis(self.m_order[1], self.m_nurbsknot[1], span_v, v)
        size = self.cv_size()
        total = [0.0] * size

        for i in range(self.m_order[0]):
            for j in range(self.m_order[1]):
                c = nu[i] * nv[j]
                cv_ptr = self.cv(span_u + i, span_v + j)

                for d in range(size):
                    total[d] += c * cv_ptr[d]

        return self._dehomogenize(total)

    def closest_parameters(self, test_point: Point) -> tuple[float, float]:
        """Return (u, v) of the closest surface point (grid seed + Newton)."""

        from .closest import Closest

        hit = Closest.surface_point(self, test_point)

        return (hit[0], hit[1])

    def closest_point(self, test_point: Point) -> Point:
        """Return the closest surface point to test_point."""

        u, v = self.closest_parameters(test_point)

        return self.point_at(u, v)

    def gaussian_curvature(self, u: float, v: float) -> float:
        """Return K = (LN - M^2) / (EG - F^2)."""

        forms = self._fundamental_forms(u, v)

        if forms is None:
            return 0.0

        E, F, G, L, M, N = forms
        denom = E * G - F * F

        if abs(denom) < Tolerance.ZERO_TOLERANCE:
            return 0.0

        return (L * N - M * M) / denom

    def mean_curvature(self, u: float, v: float) -> float:
        """Return H = (EN - 2FM + GL) / (2(EG - F^2)), sign following Su x Sv."""

        forms = self._fundamental_forms(u, v)

        if forms is None:
            return 0.0

        E, F, G, L, M, N = forms
        denom = E * G - F * F

        if abs(denom) < Tolerance.ZERO_TOLERANCE:
            return 0.0

        return (E * N - 2.0 * F * M + G * L) / (2.0 * denom)

    def normal_at(self, u: float, v: float) -> Vector:
        """Return the unit normal dS/dv x dS/du, z-axis at singular points."""

        derivs = self.evaluate(u, v, 1)

        if len(derivs) < 3:
            return Vector(0, 0, 1)

        normal = derivs[2].cross(derivs[1])
        length = normal.magnitude()

        if length < 1e-14:
            return Vector(0, 0, 1)

        return normal / length

    def frame_at(self, u: float, v: float) -> Plane:
        """Return the frame at (u, v): origin S, x-axis dS/du, y-axis dS/dv."""

        derivs = self.evaluate(u, v, 1)

        if len(derivs) < 3:
            return Plane(Point(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))

        return Plane(
            Point(derivs[0][0], derivs[0][1], derivs[0][2]), derivs[2], derivs[1]
        )

    def intersections_with_line(self, line: Line) -> list[Point]:
        """Return the points where the infinite line pierces the surface (grid seed + Newton)."""

        results = []

        if not self.is_valid():
            return results

        p0 = line.start()
        pe = line.end()
        d = pe - p0

        if d.magnitude() < 1e-14:
            return results

        d = d.normalized()
        helper = Vector(1, 0, 0) if abs(d[0]) < 0.9 else Vector(0, 1, 0)
        n1 = d.cross(helper).normalized()
        n2 = d.cross(n1).normalized()
        u0, u1 = self.domain(0)
        v0, v1 = self.domain(1)
        nu = max(12, self.cv_count(0) * 4)
        nv = max(12, self.cv_count(1) * 4)

        for a in range(nu + 1):
            for b in range(nv + 1):
                uv = [u0 + (u1 - u0) * a / nu, v0 + (v1 - v0) * b / nv]

                if not self._line_newton(uv, p0, n1, n2):
                    continue

                p = self.point_at(uv[0], uv[1])
                r = p - p0

                if abs(n1.dot(r)) > 1e-7 or abs(n2.dot(r)) > 1e-7:
                    continue

                dup = False

                for q in results:
                    if p.distance(q) < 1e-6:
                        dup = True

                if not dup:
                    results.append(p)

        return results

    def evaluate(self, u: float, v: float, num_derivs: int = 0) -> list[Vector]:
        """Return the point and partials up to num_derivs (max 2) in (k, m) loop order: [S, Sv, Svv, Su, Suv, Suu]."""

        result = []

        if not self.is_valid() or num_derivs < 0:
            return result

        n = min(num_derivs, 2)
        span_u = self._find_span(0, u)
        span_v = self._find_span(1, v)
        ders_u = self._basis_functions_derivatives(0, span_u, u, n)
        ders_v = self._basis_functions_derivatives(1, span_v, v, n)
        size = self.cv_size()
        skl = []

        for k in range(n + 1):
            for m in range(n - k + 1):
                total = [0.0] * size

                for i in range(self.m_order[0]):
                    for j in range(self.m_order[1]):
                        c = ders_u[k][i] * ders_v[m][j]
                        cv_ptr = self.cv(span_u + i, span_v + j)

                        for d in range(size):
                            total[d] += c * cv_ptr[d]

                skl.append(total)

        if self.m_is_rat:
            return self._rational_derivatives(skl, n)

        for s in skl:
            result.append(
                Vector(
                    s[0],
                    s[1] if self.m_dim > 1 else 0.0,
                    s[2] if self.m_dim > 2 else 0.0,
                )
            )

        return result

    def point_at_corner(self, u_end: int, v_end: int) -> Point:
        """Return the corner CV; u_end and v_end are 0 or 1."""

        i = 0 if u_end == 0 else self.m_cv_count[0] - 1
        j = 0 if v_end == 0 else self.m_cv_count[1] - 1

        return self.get_cv(i, j)

    def iso_curve(self, dir: int, c: float) -> NurbsCurve:
        """Return the iso-curve along dir at the other parameter c; rational surfaces give their exact rational curve."""

        if dir < 0 or dir > 1 or not self.is_valid():
            return NurbsCurve()

        crv = NurbsCurve(
            self.m_dim, self.m_is_rat != 0, self.m_order[dir], self.m_cv_count[dir]
        )

        for i in range(crv.nurbsknot_count()):
            crv.set_nurbsknot(i, self.nurbsknot(dir, i))

        other = 1 - dir
        span = self._find_span(other, c)
        basis = nurbsknot.eval_basis(
            self.m_order[other], self.m_nurbsknot[other], span, c
        )
        size = self.cv_size()

        for i in range(self.m_cv_count[dir]):
            total = [0.0] * size

            for k in range(self.m_order[other]):
                cv_ptr = self.cv(span + k, i) if dir else self.cv(i, span + k)

                for d in range(size):
                    total[d] += basis[k] * cv_ptr[d]

            p = Point(
                total[0],
                total[1] if self.m_dim > 1 else 0.0,
                total[2] if self.m_dim > 2 else 0.0,
            )

            if self.m_is_rat:
                crv.set_cv_4d(i, p[0], p[1], p[2], total[self.m_dim])
            else:
                crv.set_cv(i, p)

        return crv

    # ═══════════════════════════════════════════════════════════════════════════
    # Modifications
    # ═══════════════════════════════════════════════════════════════════════════
    def reverse(self, dir: int) -> bool:
        """Flip the parameterization in dir."""

        if dir < 0 or dir > 1 or not self.is_valid():
            return False

        nurbsknot.reverse(
            self.m_order[dir], self.m_cv_count[dir], self.m_nurbsknot[dir]
        )
        n = self.m_cv_count[dir]

        for k in range(self.m_cv_count[1 - dir]):
            for i in range(n // 2):
                a = self.cv(k, i) if dir else self.cv(i, k)
                b = self.cv(k, n - 1 - i) if dir else self.cv(n - 1 - i, k)
                tmp = a.copy()
                a[:] = b
                b[:] = tmp

        return True

    def transpose(self) -> bool:
        """Swap u and v."""

        if not self.is_valid():
            return False

        size = self.cv_size()
        new_cv = np.zeros(len(self.m_cv), dtype=np.float64)

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                dst = (j * self.m_cv_count[0] + i) * size
                new_cv[dst : dst + size] = self.cv(i, j)

        self.m_cv = new_cv
        self.m_order = [self.m_order[1], self.m_order[0]]
        self.m_cv_count = [self.m_cv_count[1], self.m_cv_count[0]]
        self.m_nurbsknot = [self.m_nurbsknot[1], self.m_nurbsknot[0]]
        self.m_cv_stride[0] = size * self.m_cv_count[1]

        return True

    def swap_coordinates(self, axis_i: int, axis_j: int) -> bool:
        """Swap two coordinate axes in every CV."""

        if axis_i < 0 or axis_i >= self.m_dim or axis_j < 0 or axis_j >= self.m_dim:
            return False

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                cv_ptr = self.cv(i, j)
                cv_ptr[axis_i], cv_ptr[axis_j] = cv_ptr[axis_j], cv_ptr[axis_i]

        return True

    def trim(self, dir: int, domain: tuple[float, float]) -> bool:
        """Restrict dir to the sub-domain."""

        if dir < 0 or dir > 1 or not self.is_valid():
            return False

        crv = self._to_curve(dir)

        if not crv.trim(domain[0], domain[1]):
            return False

        return self._from_curve(crv, dir)

    def split(self, dir: int, c: float) -> tuple[NurbsSurface, NurbsSurface]:
        """Return two surfaces split at c in dir; both invalid when c is outside the domain."""

        if dir < 0 or dir > 1 or not self.is_valid():
            return (NurbsSurface(), NurbsSurface())

        t0, t1 = self.domain(dir)

        if c <= t0 or c >= t1:
            return (NurbsSurface(), NurbsSurface())

        lo = self.duplicate()
        hi = self.duplicate()

        if not lo.trim(dir, (t0, c)) or not hi.trim(dir, (c, t1)):
            return (NurbsSurface(), NurbsSurface())

        return (lo, hi)

    def to_rational(self) -> bool:
        """Add weights of 1."""

        if self.m_is_rat:
            return True

        new_cv = np.zeros(self.cv_count() * (self.m_dim + 1), dtype=np.float64)

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                dst = (i * self.m_cv_count[1] + j) * (self.m_dim + 1)
                new_cv[dst : dst + self.m_dim] = self.cv(i, j)
                new_cv[dst + self.m_dim] = 1.0

        self.m_cv = new_cv
        self.m_is_rat = 1
        self.m_cv_stride = [(self.m_dim + 1) * self.m_cv_count[1], self.m_dim + 1]

        return True

    def to_non_rational(self) -> bool:
        """Drop weights, dividing each CV by its own."""

        if not self.m_is_rat:
            return True

        new_cv = np.zeros(self.cv_count() * self.m_dim, dtype=np.float64)

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                src = self.cv(i, j)
                dst = (i * self.m_cv_count[1] + j) * self.m_dim
                w = src[self.m_dim] if abs(src[self.m_dim]) > 1e-14 else 1.0
                new_cv[dst : dst + self.m_dim] = src[: self.m_dim] / w

        self.m_cv = new_cv
        self.m_is_rat = 0
        self.m_cv_stride = [self.m_dim * self.m_cv_count[1], self.m_dim]

        return True

    def increase_degree(self, dir: int, desired_degree: int) -> bool:
        """Elevate the degree in dir without changing the shape."""

        if (
            dir < 0
            or dir > 1
            or not self.is_valid()
            or desired_degree < self.degree(dir)
        ):
            return False

        if desired_degree == self.degree(dir):
            return True

        crv = self._to_curve(dir)

        if not crv.increase_degree(desired_degree):
            return False

        return self._from_curve(crv, dir)

    # ═══════════════════════════════════════════════════════════════════════════
    # Splitting
    # ═══════════════════════════════════════════════════════════════════════════
    def split_by_plane(
        self, plane: Plane, tolerance: float = 0.0
    ) -> list[NurbsSurfaceTrimmed]:
        """Return the trimmed faces on each side of the plane."""

        from .intersection import surface_plane_uv
        from .nurbssurface_trimmed import NurbsSurfaceTrimmed

        pcurves = []

        for pair in surface_plane_uv(self, plane, tolerance):
            pcurves.append(pair[1])

        return NurbsSurfaceTrimmed.split_by_uv_curves(self, pcurves, tolerance)

    def split_by_curves(
        self, curves: list[NurbsCurve], tolerance: float = 0.0
    ) -> list[NurbsSurfaceTrimmed]:
        """Return the trimmed faces cut by curves pulled onto the surface; off-surface curves are skipped."""

        from .closest import Closest
        from .nurbssurface_trimmed import NurbsSurfaceTrimmed

        pcurves = []

        for crv in curves:
            for pcurve in Closest.surface_curve(self, crv, 0.0, 0.0, tolerance):
                pcurves.append(pcurve)

        return NurbsSurfaceTrimmed.split_by_uv_curves(self, pcurves, tolerance)

    def split_by_line(
        self, line: Line, tolerance: float = 0.0
    ) -> list[NurbsSurfaceTrimmed]:
        """Return the trimmed faces cut by a line pulled onto the surface."""

        points = [line.start(), line.end()]

        return self.split_by_curves([NurbsCurve.create(False, 1, points)], tolerance)

    def split_by_surface(
        self, cutter: NurbsSurface, tolerance: float = 0.0
    ) -> list[NurbsSurfaceTrimmed]:
        """Return the trimmed faces cut by the surface/surface intersection."""

        from .intersection import surface_surface
        from .nurbssurface_trimmed import NurbsSurfaceTrimmed

        pcurves = []

        for triple in surface_surface(self, cutter, tolerance):
            pcurves.append(triple[1])

        return NurbsSurfaceTrimmed.split_by_uv_curves(self, pcurves, tolerance)

    def split_by_brep(
        self, brep: BRep, tolerance: float = 0.0
    ) -> list[NurbsSurfaceTrimmed]:
        """Return the trimmed faces cut by every overlapping face of the brep."""

        from .intersection import cut_curves_on_surface
        from .nurbssurface_trimmed import NurbsSurfaceTrimmed

        target_bb = _surface_aabb(self)
        pcurves = []

        for cutter in brep.m_surfaces:
            if not _aabb_overlap_pad(target_bb, _surface_aabb(cutter)):
                continue

            for pcurve in cut_curves_on_surface(self, cutter, tolerance):
                pcurves.append(pcurve)

        return NurbsSurfaceTrimmed.split_by_uv_curves(self, pcurves, tolerance)

    # ═══════════════════════════════════════════════════════════════════════════
    # Meshing
    # ═══════════════════════════════════════════════════════════════════════════
    def mesh_adaptive(
        self,
        max_angle: float = 20.0,
        max_edge_length: float = 0.0,
        min_edge_length: float = 0.0,
        max_chord_height: float = 0.0,
    ) -> Mesh:
        """Return the quadtree subdivision in UV up to depth 8, cached in m_mesh."""

        from .remesh_nurbssurface_adaptive import RemeshNurbsSurfaceAdaptive

        if self.m_mesh is None and self.is_valid():
            mesher = RemeshNurbsSurfaceAdaptive(self)
            (
                mesher.set_max_angle(max_angle)
                .set_max_edge_length(max_edge_length)
                .set_min_edge_length(min_edge_length)
                .set_max_chord_height(max_chord_height)
            )

            self.m_mesh = mesher.mesh()

        return self.m_mesh if self.m_mesh is not None else Mesh()

    def mesh(self) -> Mesh:
        """Return two triangles for a planar surface, else the span grid, cached in m_mesh."""

        from .remesh_nurbssurface_grid import RemeshNurbsSurfaceGrid

        if self.m_mesh is None and self.is_valid():
            self.m_mesh = (
                self._mesh_planar()
                if self.is_planar(None, 1e-6)
                else RemeshNurbsSurfaceGrid.from_u_v(self, 0, 0)
            )

        return self.m_mesh if self.m_mesh is not None else Mesh()

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        control_points = []

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                control_points.extend(self.cv(i, j).tolist())

        data = {
            "control_points": control_points,
            "cv_count_u": int(self.m_cv_count[0]),
            "cv_count_v": int(self.m_cv_count[1]),
            "dimension": int(self.m_dim),
            "facecolors": _colors_to_json(self.facecolors),
            "guid": self.guid,
            "is_rational": self.m_is_rat != 0,
            "linecolors": _colors_to_json(self.linecolors),
        }

        if self.m_mesh is not None and self.m_mesh.number_of_vertices() > 0:
            data["mesh"] = self.m_mesh.__jsondump__()

        data["name"] = self.name
        data["nurbsknots_u"] = self.m_nurbsknot[0].tolist()
        data["nurbsknots_v"] = self.m_nurbsknot[1].tolist()
        data["order_u"] = int(self.m_order[0])
        data["order_v"] = int(self.m_order[1])
        data["pointcolors"] = _colors_to_json(self.pointcolors)
        data["type"] = "NurbsSurface"
        data["width"] = float(self.width)

        return data

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> NurbsSurface:
        """Deserialize from a JSON object."""

        from .file_encoders import file_decode_node

        surface = cls()

        for key in ["dimension", "order_u", "order_v", "cv_count_u", "cv_count_v"]:
            if key not in data:
                return surface

        created = surface.create_raw(
            int(data["dimension"]),
            bool(data.get("is_rational", False)),
            int(data["order_u"]),
            int(data["order_v"]),
            int(data["cv_count_u"]),
            int(data["cv_count_v"]),
        )

        surface.guid = guid if guid is not None else data.get("guid", str(uuid.uuid4()))
        surface.name = name if name is not None else data.get("name", "my_nurbssurface")
        surface.width = data.get("width", 1.0)
        surface.pointcolors = _colors_from_json(data, "pointcolors")
        surface.facecolors = _colors_from_json(data, "facecolors")
        surface.linecolors = _colors_from_json(data, "linecolors")

        if data.get("mesh"):
            surface.m_mesh = file_decode_node(data["mesh"])

        if not created:
            return surface

        if "nurbsknots_u" in data:
            surface.m_nurbsknot[0] = np.array(data["nurbsknots_u"], dtype=np.float64)

        if "nurbsknots_v" in data:
            surface.m_nurbsknot[1] = np.array(data["nurbsknots_v"], dtype=np.float64)

        if "control_points" in data:
            surface.m_cv = np.array(data["control_points"], dtype=np.float64)

        return surface

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__(), separators=(",", ":"))

    @classmethod
    def file_json_loads(cls, json_string: str) -> NurbsSurface:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.__jsondump__(), f, indent=4)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> NurbsSurface:
        """Read from a JSON file."""
        with open(filename) as f:
            return cls.__jsonload__(json.load(f))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> nurbssurface_pb2.NurbsSurface:
        """Convert to the protobuf message."""

        from .proto import nurbssurface_pb2

        proto = nurbssurface_pb2.NurbsSurface()

        if self.has_guid():
            proto.guid = self.guid

        proto.name = self.name
        proto.dimension = int(self.m_dim)
        proto.is_rational = self.m_is_rat != 0
        proto.order_u = int(self.m_order[0])
        proto.order_v = int(self.m_order[1])
        proto.cv_count_u = int(self.m_cv_count[0])
        proto.cv_count_v = int(self.m_cv_count[1])
        proto.cv_stride_u = int(self.m_cv_stride[0])
        proto.cv_stride_v = int(self.m_cv_stride[1])
        proto.nurbsknots_u.extend(self.m_nurbsknot[0].tolist())
        proto.nurbsknots_v.extend(self.m_nurbsknot[1].tolist())

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                proto.cvs.extend(self.cv(i, j).tolist())

        proto.width = float(self.width)
        _colors_to_proto(self.pointcolors, proto.pointcolors)
        _colors_to_proto(self.facecolors, proto.facecolors)
        _colors_to_proto(self.linecolors, proto.linecolors)

        if self.m_mesh is not None and self.m_mesh.number_of_vertices() > 0:
            proto.cached_mesh.CopyFrom(self.m_mesh.to_proto())

        return proto

    @classmethod
    def from_proto(cls, proto: nurbssurface_pb2.NurbsSurface) -> NurbsSurface:
        """Construct from the protobuf message."""

        surface = cls()
        created = surface.create_raw(
            proto.dimension,
            proto.is_rational,
            proto.order_u,
            proto.order_v,
            proto.cv_count_u,
            proto.cv_count_v,
        )

        if proto.guid:
            surface.guid = proto.guid

        surface.name = proto.name
        surface.width = proto.width
        surface.pointcolors = _colors_from_proto(proto.pointcolors)
        surface.facecolors = _colors_from_proto(proto.facecolors)
        surface.linecolors = _colors_from_proto(proto.linecolors)

        if proto.HasField("cached_mesh") and len(proto.cached_mesh.vertices) > 0:
            surface.m_mesh = Mesh.from_proto(proto.cached_mesh)

        if not created:
            return surface

        for i in range(min(len(proto.nurbsknots_u), len(surface.m_nurbsknot[0]))):
            surface.m_nurbsknot[0][i] = proto.nurbsknots_u[i]

        for i in range(min(len(proto.nurbsknots_v), len(surface.m_nurbsknot[1]))):
            surface.m_nurbsknot[1][i] = proto.nurbsknots_v[i]

        size = surface.cv_size()
        stride_u = proto.cv_stride_u if proto.cv_stride_u > 0 else size * surface.m_cv_count[1]
        stride_v = proto.cv_stride_v if proto.cv_stride_v > 0 else size

        for i in range(surface.m_cv_count[0]):
            for j in range(surface.m_cv_count[1]):
                src = i * stride_u + j * stride_v
                dst = surface.cv(i, j)

                for d in range(size):
                    if src + d < len(proto.cvs):
                        dst[d] = proto.cvs[src + d]

        return surface

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> NurbsSurface:
        """Deserialize from protobuf bytes."""

        from .proto import nurbssurface_pb2

        proto = nurbssurface_pb2.NurbsSurface()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write to a protobuf file."""
        with open(filename, "wb") as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> NurbsSurface:
        """Read from a protobuf file."""
        with open(filename, "rb") as f:
            return cls.pb_loads(f.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "NurbsSurface(name=..., degree=(u, v), cvs=(u, v))"."""
        return f"NurbsSurface(name={self.name}, degree=({self.degree(0)},{self.degree(1)}), cvs=({self.m_cv_count[0]},{self.m_cv_count[1]}))"

    def __repr__(self) -> str:
        """Return the multi-line form with every control point."""

        rational = "true" if self.m_is_rat else "false"
        result = f"NurbsSurface(\n  name={self.name},\n  degree=({self.degree(0)},{self.degree(1)}),\n  cvs=({self.m_cv_count[0]},{self.m_cv_count[1]}),\n  rational={rational},\n  control_points=[\n"

        for i in range(self.m_cv_count[0]):
            for j in range(self.m_cv_count[1]):
                p = self.get_cv(i, j)
                result += f"    {p[0]:g}, {p[1]:g}, {p[2]:g}\n"

        result += "  ]\n)"

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════════════════════════
    def _deep_copy_from(self, src: NurbsSurface) -> None:
        """Copy every field but the guid."""

        self._guid = None
        self.name = src.name
        self.width = src.width
        self.pointcolors = list(src.pointcolors)
        self.facecolors = list(src.facecolors)
        self.linecolors = list(src.linecolors)
        self.m_dim = src.m_dim
        self.m_is_rat = src.m_is_rat
        self.m_order = list(src.m_order)
        self.m_cv_count = list(src.m_cv_count)
        self.m_cv_stride = list(src.m_cv_stride)
        self.m_nurbsknot = [src.m_nurbsknot[0].copy(), src.m_nurbsknot[1].copy()]
        self.m_cv = src.m_cv.copy()
        self.m_mesh = copy.deepcopy(src.m_mesh)

    def _zero_cvs(self) -> bool:
        """Zero every CV, false when the layout is unset."""

        self.m_cv[:] = 0.0

        if self.m_is_rat:
            for i in range(self.m_cv_count[0]):
                for j in range(self.m_cv_count[1]):
                    self.cv(i, j)[self.m_dim] = 1.0

        return True

    def _set_clamped_uniform_nurbsknot_vector(
        self, dir: int, delta: float = 1.0
    ) -> bool:
        """Fill the nurbsknot vector in dir with clamped uniform values of the given spacing."""

        if dir < 0 or dir > 1 or delta <= 0.0:
            return False

        self.m_nurbsknot[dir] = np.array(
            nurbsknot.compute_clamped_uniform(
                self.m_order[dir], self.m_cv_count[dir], delta
            ),
            dtype=np.float64,
        )

        return len(self.m_nurbsknot[dir]) > 0

    def _set_periodic_uniform_nurbsknot_vector(
        self, dir: int, delta: float = 1.0
    ) -> bool:
        """Fill the nurbsknot vector in dir with periodic uniform values of the given spacing."""

        if dir < 0 or dir > 1 or delta <= 0.0:
            return False

        self.m_nurbsknot[dir] = np.array(
            nurbsknot.compute_periodic_uniform(
                self.m_order[dir], self.m_cv_count[dir], delta
            ),
            dtype=np.float64,
        )

        return len(self.m_nurbsknot[dir]) > 0

    def _dehomogenize(self, h) -> Point:
        """Return the Euclidean point of a homogeneous CV or blend."""

        w = h[self.m_dim] if self.m_is_rat and abs(h[self.m_dim]) > 1e-14 else 1.0

        return Point(
            h[0] / w,
            h[1] / w if self.m_dim > 1 else 0.0,
            h[2] / w if self.m_dim > 2 else 0.0,
        )

    def _find_span(self, dir: int, t: float) -> int:
        """Return the span index in dir containing t."""
        return nurbsknot.find_span(self.m_order[dir], self.m_cv_count[dir], self.m_nurbsknot[dir], t)

    def _basis_functions_derivatives(
        self, dir: int, span: int, t: float, deriv_order: int
    ) -> list[list[float]]:
        """Return the basis derivatives ders[k][j] of the order functions on the span (Piegl & Tiller A2.3)."""

        order = self.m_order[dir]
        degree = order - 1
        knot = self.m_nurbsknot[dir]
        base = span + degree

        if knot[base - 1] == knot[base]:
            return [[0.0] * order for _ in range(deriv_order + 1)]

        return _basis_table_derivatives(_basis_table(knot, degree, base, t), degree, deriv_order)

    def _rational_derivatives(
        self, skl: list[list[float]], num_derivs: int
    ) -> list[Vector]:
        """Apply the rational quotient rule to homogeneous partials in (k, m) loop order (Piegl & Tiller A4.4)."""

        result = []
        n = num_derivs
        w00 = skl[0][self.m_dim]

        if abs(w00) < 1e-14:
            return [Vector(0, 0, 0) for _ in range(len(skl))]

        for k in range(n + 1):
            for m in range(n - k + 1):
                s = skl[k * (n + 1) - k * (k - 1) // 2 + m]
                a = Vector(
                    s[0],
                    s[1] if self.m_dim > 1 else 0.0,
                    s[2] if self.m_dim > 2 else 0.0,
                )

                for i in range(k + 1):
                    for j in range(m + 1):
                        if i == 0 and j == 0:
                            continue

                        c = (
                            _binomial(k, i)
                            * _binomial(m, j)
                            * skl[i * (n + 1) - i * (i - 1) // 2 + j][self.m_dim]
                        )
                        a -= (
                            result[
                                (k - i) * (n + 1) - (k - i) * (k - i - 1) // 2 + (m - j)
                            ]
                            * c
                        )

                result.append(a / w00)

        return result

    def _line_newton(self, uv: list[float], p0: Point, n1: Vector, n2: Vector) -> bool:
        """Run Newton on (n1, n2) . (S - p0) = 0 from (u, v); false when it leaves the domain or stalls."""

        u0, u1 = self.domain(0)
        v0, v1 = self.domain(1)

        for it in range(40):
            der = self.evaluate(uv[0], uv[1], 1)

            if len(der) < 3:
                return False

            r = Vector(der[0][0] - p0[0], der[0][1] - p0[1], der[0][2] - p0[2])
            f1 = n1.dot(r)
            f2 = n2.dot(r)

            if abs(f1) < 1e-12 and abs(f2) < 1e-12:
                return True

            j11 = n1.dot(der[2])
            j12 = n1.dot(der[1])
            j21 = n2.dot(der[2])
            j22 = n2.dot(der[1])
            det = j11 * j22 - j12 * j21

            if abs(det) < 1e-14:
                return False

            du = -(j22 * f1 - j12 * f2) / det
            dv = -(-j21 * f1 + j11 * f2) / det
            uv[0] += du
            uv[1] += dv

            if uv[0] < u0 or uv[0] > u1 or uv[1] < v0 or uv[1] > v1:
                return False

            if abs(du) < 1e-13 and abs(dv) < 1e-13:
                return True

        return True

    def _fundamental_forms(
        self, u: float, v: float
    ) -> tuple[float, float, float, float, float, float] | None:
        """Compute the first and second fundamental forms at (u, v); None at a singular point."""

        d = self.evaluate(u, v, 2)

        if len(d) < 6:
            return None

        sv = d[1]
        svv = d[2]
        su = d[3]
        suv = d[4]
        suu = d[5]
        cr = su.cross(sv)

        if cr.magnitude() < Tolerance.ZERO_TOLERANCE:
            return None

        n = cr.normalized()

        return (su.dot(su), su.dot(sv), sv.dot(sv), suu.dot(n), suv.dot(n), svv.dot(n))

    def _mesh_planar(self) -> Mesh:
        """Return two triangles through the four corners with one shared normal."""

        result = Mesh()
        p00 = self.point_at_corner(0, 0)
        p10 = self.point_at_corner(1, 0)
        p11 = self.point_at_corner(1, 1)
        p01 = self.point_at_corner(0, 1)
        v0 = result.add_vertex(p00)
        v1 = result.add_vertex(p10)
        v2 = result.add_vertex(p11)
        result.add_face([v0, v1, v2])

        if p00.distance(p01) < 1e-10:
            e1 = p10 - p00
            e2 = p11 - p00
            normal = e1.cross(e2)
        else:
            v3 = result.add_vertex(p01)
            result.add_face([v0, v2, v3])
            derivs = self.evaluate(0.5, 0.5, 1)
            normal = derivs[1].cross(derivs[2])

        if normal.magnitude() > 1e-15:
            normal = normal.normalized()

        for vkey in result.vertex:
            result.vertex[vkey].set_normal(normal[0], normal[1], normal[2])

        return result

    def _to_curve(self, dir: int) -> NurbsCurve:
        """Pack the CV rows across dir into one curve along dir with cv_size * cv_count(1 - dir) doubles per CV."""

        other = 1 - dir
        size = self.cv_size()
        crv = NurbsCurve(
            size * self.m_cv_count[other],
            False,
            self.m_order[dir],
            self.m_cv_count[dir],
        )

        for i in range(crv.nurbsknot_count()):
            crv.set_nurbsknot(i, self.nurbsknot(dir, i))

        for i in range(self.m_cv_count[dir]):
            dst = crv.cv(i)

            for j in range(self.m_cv_count[other]):
                src = self.cv(j, i) if dir else self.cv(i, j)
                dst[j * size : (j + 1) * size] = src

        return crv

    def _from_curve(self, crv: NurbsCurve, dir: int) -> bool:
        """Unpack a curve made by to_curve back into this surface along dir."""

        other = 1 - dir
        size = self.cv_size()

        if crv.m_is_rat or crv.m_dim != size * self.m_cv_count[other]:
            return False

        srf = NurbsSurface()
        created = (
            srf.create_raw(
                self.m_dim,
                self.m_is_rat != 0,
                crv.m_order,
                self.m_order[1],
                crv.m_cv_count,
                self.m_cv_count[1],
            )
            if dir == 0
            else srf.create_raw(
                self.m_dim,
                self.m_is_rat != 0,
                self.m_order[0],
                crv.m_order,
                self.m_cv_count[0],
                crv.m_cv_count,
            )
        )

        if not created:
            return False

        srf.m_nurbsknot[dir] = crv.m_nurbsknot.copy()
        srf.m_nurbsknot[other] = self.m_nurbsknot[other].copy()

        for i in range(crv.m_cv_count):
            src = crv.cv(i)

            for j in range(self.m_cv_count[other]):
                dst = srf.cv(j, i) if dir else srf.cv(i, j)
                dst[:] = src[j * size : (j + 1) * size]

        self.m_order[dir] = srf.m_order[dir]
        self.m_cv_count[dir] = srf.m_cv_count[dir]
        self.m_cv_stride = list(srf.m_cv_stride)
        self.m_nurbsknot[dir] = srf.m_nurbsknot[dir]
        self.m_cv = srf.m_cv

        return True

    @staticmethod
    def _assign_plane(dst: Plane, src: Plane) -> None:
        """Copy the frame of src into dst (the C++ Plane* out-parameter)."""

        dst._origin = src.origin
        dst._x_axis = src.x_axis
        dst._y_axis = src.y_axis
        dst._z_axis = src.z_axis
        dst._update_equation()
