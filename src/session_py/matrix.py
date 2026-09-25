from __future__ import annotations

import json
import math
import uuid
from typing import TYPE_CHECKING

from .tolerance import Tolerance

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import matrix_pb2


COMPARISON_TOLERANCE = Tolerance.ABSOLUTE / 10.0
MAX_DIMENSION = 2_147_483_647
PIVOT_TOLERANCE = Tolerance.ZERO_TOLERANCE / 100.0
SINGULAR_TOLERANCE = Tolerance.ZERO_TOLERANCE


def _eigen_value(pair: tuple[float, list[float]]) -> float:
    """Return the eigenvalue of an (eigenvalue, eigenvector) pair."""
    return pair[0]


def _matrix_size(rows: int, cols: int) -> int:
    """Return the validated element count for C++-compatible dimensions."""

    if rows < 0 or cols < 0:
        raise ValueError("Matrix dimensions cannot be negative")

    if rows > MAX_DIMENSION or cols > MAX_DIMENSION:
        raise ValueError("Matrix dimensions are too large")

    return rows * cols


class Matrix:
    """An NxM matrix with row-major storage."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, rows: int = 0, cols: int = 0) -> None:
        """Construct a rows x cols matrix of zeros; raises for invalid dimensions."""

        self._guid = None  # Lazily minted GUID.
        self.name = "my_matrix"  # Matrix name.
        self.rows = rows  # Row count.
        self.cols = cols  # Column count.
        self.data = [0.0] * _matrix_size(rows, cols)  # Row-major values.

    @staticmethod
    def zeros(rows: int, cols: int) -> "Matrix":
        """Construct a rows x cols zero matrix."""
        return Matrix(rows, cols)

    @staticmethod
    def identity(n: int) -> "Matrix":
        """Construct an n x n identity matrix."""

        m = Matrix(n, n)

        for i in range(n):
            m[i, i] = 1.0

        return m

    @staticmethod
    def from_vec(rows: int, cols: int, data: list[float]) -> "Matrix":
        """Construct from exact row-major data; raises when the size does not match."""

        if len(data) != _matrix_size(rows, cols):
            raise ValueError("Matrix data size does not match its dimensions")

        m = Matrix(rows, cols)
        m.data = list(data)

        return m

    @staticmethod
    def from_rows(rows_list: list[list[float]]) -> "Matrix":
        """Construct from equal-length rows."""

        r = len(rows_list)
        c = len(rows_list[0]) if r > 0 else 0

        for row in rows_list:
            if len(row) != c:
                raise ValueError("Matrix rows must have equal lengths")

        m = Matrix(r, c)

        for i in range(r):
            for j in range(c):
                m[i, j] = rows_list[i][j]

        return m

    @staticmethod
    def from_cols(cols_list: list[list[float]]) -> "Matrix":
        """Construct from equal-length columns."""

        c = len(cols_list)
        r = len(cols_list[0]) if c > 0 else 0

        for col in cols_list:
            if len(col) != r:
                raise ValueError("Matrix columns must have equal lengths")

        m = Matrix(r, c)

        for j in range(c):
            for i in range(r):
                m[i, j] = cols_list[j][i]

        return m

    def duplicate(self) -> "Matrix":
        """Copy with a new guid and the same data."""

        m = Matrix.from_vec(self.rows, self.cols, self.data)
        m.name = self.name

        return m

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
        self._guid = value

    def __getitem__(self, idx: tuple[int, int]) -> float:
        """Return the element at (row, col)."""

        r, c = idx

        return self.data[r * self.cols + c]

    def __setitem__(self, idx: tuple[int, int], value: float) -> None:
        """Set the element at (row, col)."""

        r, c = idx
        self.data[r * self.cols + c] = value

    def is_square(self) -> bool:
        """Return whether the matrix has equal row and column counts."""
        return self.rows == self.cols

    def is_symmetric(self) -> bool:
        """Return whether the matrix is square and symmetric."""

        if not self.is_square():
            return False

        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if abs(self[i, j] - self[j, i]) > COMPARISON_TOLERANCE:
                    return False

        return True

    def trace(self) -> float:
        """Return the diagonal sum; raises unless the matrix is square."""

        if not self.is_square():
            raise ValueError("Matrix trace requires a square matrix")

        s = 0.0

        for i in range(self.rows):
            s += self[i, i]

        return s

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __add__(self, other: "Matrix") -> "Matrix":
        """Add an equal-sized matrix."""

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")

        result = Matrix(self.rows, self.cols)

        for i in range(len(self.data)):
            result.data[i] = self.data[i] + other.data[i]

        return result

    def __sub__(self, other: "Matrix") -> "Matrix":
        """Subtract an equal-sized matrix."""

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for subtraction")

        result = Matrix(self.rows, self.cols)

        for i in range(len(self.data)):
            result.data[i] = self.data[i] - other.data[i]

        return result

    def __mul__(self, other: "Matrix | float") -> "Matrix":
        """Multiply by a dimension-compatible matrix or every element by a scalar."""

        if not isinstance(other, Matrix):
            result = Matrix(self.rows, self.cols)

            for i in range(len(self.data)):
                result.data[i] = self.data[i] * other

            return result

        if self.cols != other.rows:
            raise ValueError("Matrix dimensions are incompatible for multiplication")

        result = Matrix(self.rows, other.cols)

        for i in range(self.rows):
            for j in range(other.cols):
                s = 0.0

                for k in range(self.cols):
                    s += self[i, k] * other[k, j]

                result[i, j] = s

        return result

    def __eq__(self, other: object) -> bool:
        """Compare dimensions and values within the matrix comparison tolerance."""

        if not isinstance(other, Matrix):
            return False

        if self.rows != other.rows or self.cols != other.cols:
            return False

        for i in range(len(self.data)):
            if abs(self.data[i] - other.data[i]) > COMPARISON_TOLERANCE:
                return False

        return True

    def __ne__(self, other: object) -> bool:
        """Return whether two matrices differ."""
        return not self == other

    # ═══════════════════════════════════════════════════════════════════════════
    # Linear algebra
    # ═══════════════════════════════════════════════════════════════════════════
    def transpose(self) -> "Matrix":
        """Return the transpose."""

        result = Matrix(self.cols, self.rows)

        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]

        return result

    def _lu_internal(self) -> tuple["Matrix", "Matrix", "Matrix", int]:
        """Return (L, U, P, swaps) by partial pivoting."""

        n = self.rows
        u = self.duplicate()
        lower = Matrix.identity(n)
        p = Matrix.identity(n)
        swaps = 0

        for k in range(n):
            max_val = abs(u[k, k])
            max_row = k

            for i in range(k + 1, n):
                if abs(u[i, k]) > max_val:
                    max_val = abs(u[i, k])
                    max_row = i

            if max_row != k:
                for j in range(n):
                    u[k, j], u[max_row, j] = u[max_row, j], u[k, j]

                for j in range(n):
                    p[k, j], p[max_row, j] = p[max_row, j], p[k, j]

                for j in range(k):
                    lower[k, j], lower[max_row, j] = lower[max_row, j], lower[k, j]

                swaps += 1

            if abs(u[k, k]) < PIVOT_TOLERANCE:
                continue

            for i in range(k + 1, n):
                factor = u[i, k] / u[k, k]
                lower[i, k] = factor

                for j in range(k, n):
                    u[i, j] -= factor * u[k, j]

        return lower, u, p, swaps

    def lu_decompose(self) -> tuple["Matrix", "Matrix", "Matrix"]:
        """Return (L, U, P) with P * A = L * U; raises unless square."""

        if not self.is_square():
            raise ValueError("LU decomposition requires a square matrix")

        lower, u, p, _swaps = self._lu_internal()

        return lower, u, p

    def determinant(self) -> float:
        """Return the determinant; raises unless the matrix is square."""

        if not self.is_square():
            raise ValueError("Matrix determinant requires a square matrix")

        n = self.rows

        if n == 1:
            return self[0, 0]

        if n == 2:
            return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]

        _l, u, _p, swaps = self._lu_internal()
        sign = 1.0 if swaps % 2 == 0 else -1.0
        prod = 1.0

        for i in range(n):
            prod *= u[i, i]

        return sign * prod

    def inverse(self) -> "Matrix | None":
        """Return the inverse, or None for a non-square or singular matrix."""

        if not self.is_square():
            return None

        n = self.rows
        lower, u, p, _swaps = self._lu_internal()

        for i in range(n):
            if abs(u[i, i]) < PIVOT_TOLERANCE:
                return None

        result = Matrix(n, n)
        eye = Matrix.identity(n)

        for col in range(n):
            pb = [0.0] * n

            for i in range(n):
                for j in range(n):
                    pb[i] += p[i, j] * eye[j, col]

            y = [0.0] * n

            for i in range(n):
                y[i] = pb[i]

                for j in range(i):
                    y[i] -= lower[i, j] * y[j]

            x = [0.0] * n

            for i in range(n - 1, -1, -1):
                x[i] = y[i]

                for j in range(i + 1, n):
                    x[i] -= u[i, j] * x[j]

                x[i] /= u[i, i]

            for i in range(n):
                result[i, col] = x[i]

        return result

    def solve(self, b: "Matrix") -> "Matrix | None":
        """Return x with A * x = b, or None when no compatible unique solution exists."""

        if not self.is_square() or b.rows != self.rows or b.cols != 1:
            return None

        n = self.rows
        lower, u, p, _swaps = self._lu_internal()

        for i in range(n):
            if abs(u[i, i]) < PIVOT_TOLERANCE:
                return None

        pb = [0.0] * n

        for i in range(n):
            for j in range(n):
                pb[i] += p[i, j] * b[j, 0]

        y = [0.0] * n

        for i in range(n):
            y[i] = pb[i]

            for j in range(i):
                y[i] -= lower[i, j] * y[j]

        x = [0.0] * n

        for i in range(n - 1, -1, -1):
            x[i] = y[i]

            for j in range(i + 1, n):
                x[i] -= u[i, j] * x[j]

            x[i] /= u[i, i]

        result = Matrix(n, 1)

        for i in range(n):
            result[i, 0] = x[i]

        return result

    def qr_decompose(self) -> tuple["Matrix", "Matrix"]:
        """Return (Q, R) from Gram-Schmidt decomposition."""

        m = self.rows
        n = self.cols
        a_cols = []

        for j in range(n):
            col = [0.0] * m

            for i in range(m):
                col[i] = self[i, j]

            a_cols.append(col)

        q_cols = []
        r = Matrix.zeros(n, n)

        for j in range(n):
            v = list(a_cols[j])

            for i in range(j):
                rij = 0.0

                for k in range(m):
                    rij += q_cols[i][k] * v[k]

                r[i, j] = rij

                for k in range(m):
                    v[k] -= rij * q_cols[i][k]

            norm = 0.0

            for k in range(m):
                norm += v[k] * v[k]

            norm = math.sqrt(norm)
            r[j, j] = norm

            qcol = [0.0] * m

            if norm > PIVOT_TOLERANCE:
                for k in range(m):
                    qcol[k] = v[k] / norm

            q_cols.append(qcol)

        q = Matrix.zeros(m, n)

        for j in range(n):
            for i in range(m):
                q[i, j] = q_cols[j][i]

        return q, r

    def cholesky(self) -> "Matrix | None":
        """Return lower L with A = L * L^T, or None when not positive definite."""

        if not self.is_square():
            return None

        n = self.rows
        lower = Matrix(n, n)

        for i in range(n):
            for j in range(i + 1):
                s = self[i, j]

                for k in range(j):
                    s -= lower[i, k] * lower[j, k]

                if i == j:
                    if s <= 0.0:
                        return None

                    lower[i, j] = math.sqrt(s)
                else:
                    lower[i, j] = s / lower[j, j]

        return lower

    def eigenvalues(self) -> list[float]:
        """Return eigenvalues by bounded unshifted QR iteration; raises unless square."""

        if not self.is_square():
            raise ValueError("Matrix eigenvalues require a square matrix")

        n = self.rows
        a = self.duplicate()

        for _ in range(1000 * n):
            q, r = a.qr_decompose()
            a = r * q

            converged = True

            for i in range(1, n):
                if abs(a[i, i - 1]) >= COMPARISON_TOLERANCE:
                    converged = False
                    break

            if converged:
                break

        ev = [0.0] * n

        for i in range(n):
            ev[i] = a[i, i]

        return ev

    def _eigen_decompose_symmetric(self) -> list[tuple[float, list[float]]]:
        """Return (eigenvalue, eigenvector) pairs by QR iteration with accumulated Q."""

        n = self.rows
        a = self.duplicate()
        v = Matrix.identity(n)

        for _ in range(1000 * n):
            q, r = a.qr_decompose()
            a = r * q
            v = v * q

            converged = True

            for i in range(1, n):
                if abs(a[i, i - 1]) >= COMPARISON_TOLERANCE:
                    converged = False
                    break

            if converged:
                break

        pairs = []

        for i in range(n):
            evec = [0.0] * n

            for j in range(n):
                evec[j] = v[j, i]

            pairs.append((a[i, i], evec))

        return pairs

    def svd(self) -> tuple["Matrix", list[float], "Matrix"]:
        """Return (U, singular values, V^T)."""

        m = self.rows
        n = self.cols
        at = self.transpose()
        ata = at * self
        pairs = ata._eigen_decompose_symmetric()
        pairs.sort(key=_eigen_value, reverse=True)

        k = min(m, n)
        sv = []
        v_cols = []

        for i in range(k):
            sv.append(math.sqrt(max(0.0, pairs[i][0])))
            v_cols.append(pairs[i][1])

        v = Matrix.zeros(n, k)

        for j in range(k):
            for i in range(n):
                v[i, j] = v_cols[j][i]

        u = Matrix.zeros(m, k)

        for j in range(k):
            if sv[j] <= SINGULAR_TOLERANCE:
                continue

            for i in range(m):
                val = 0.0

                for t in range(n):
                    val += self[i, t] * v[t, j]

                u[i, j] = val / sv[j]

        return u, sv, v.transpose()

    # ═══════════════════════════════════════════════════════════════════════════
    # Norms
    # ═══════════════════════════════════════════════════════════════════════════
    def norm_frobenius(self) -> float:
        """Return the Frobenius norm."""

        s = 0.0

        for x in self.data:
            s += x * x

        return math.sqrt(s)

    def norm_1(self) -> float:
        """Return the maximum absolute column sum."""

        max_sum = 0.0

        for j in range(self.cols):
            col_sum = 0.0

            for i in range(self.rows):
                col_sum += abs(self[i, j])

            if col_sum > max_sum:
                max_sum = col_sum

        return max_sum

    def norm_inf(self) -> float:
        """Return the maximum absolute row sum."""

        max_sum = 0.0

        for i in range(self.rows):
            row_sum = 0.0

            for j in range(self.cols):
                row_sum += abs(self[i, j])

            if row_sum > max_sum:
                max_sum = row_sum

        return max_sum

    def rank(self) -> int:
        """Return the numerical rank."""

        _u, sv, _vt = self.svd()

        if not sv:
            return 0

        max_sv = 0.0

        for s in sv:
            max_sv = max(max_sv, s)

        threshold = max(self.rows, self.cols) * max_sv * COMPARISON_TOLERANCE
        count = 0

        for s in sv:
            if s > threshold:
                count += 1

        return count

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to an ordered JSON object."""

        return {
            "cols": self.cols,
            "data": self.data,
            "guid": self.guid,
            "name": self.name,
            "rows": self.rows,
            "type": "Matrix",
        }

    @classmethod
    def __jsonload__(
        cls,
        data: dict,
        guid: str | None = None,
        name: str | None = None,
    ) -> "Matrix":
        """Deserialize from a JSON object."""

        m = cls.from_vec(data["rows"], data["cols"], data["data"])
        m.guid = guid or data["guid"]
        m.name = name or data["name"]

        return m

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__(), separators=(",", ":"))

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Matrix":
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write JSON to a file."""
        with open(filename, "w") as f:
            json.dump(self.__jsondump__(), f, indent=4)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> "Matrix":
        """Read JSON from a file."""
        with open(filename) as f:
            return cls.__jsonload__(json.load(f))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> matrix_pb2.Matrix:
        """Convert to the protobuf message."""

        from .proto import matrix_pb2

        proto = matrix_pb2.Matrix()

        if self.has_guid():
            proto.guid = self.guid

        proto.name = self.name
        proto.rows = self.rows
        proto.cols = self.cols
        proto.data.extend(self.data)

        return proto

    @classmethod
    def from_proto(cls, proto: matrix_pb2.Matrix) -> "Matrix":
        """Construct from a shape-valid protobuf message."""

        m = cls.from_vec(proto.rows, proto.cols, proto.data)

        if proto.guid:
            m.guid = proto.guid

        m.name = proto.name

        return m

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Matrix":
        """Deserialize from protobuf bytes."""

        from .proto import matrix_pb2

        proto = matrix_pb2.Matrix()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write protobuf bytes to a file."""

        data = self.pb_dumps()

        with open(filename, "wb") as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filename: str | Path) -> "Matrix":
        """Read protobuf bytes from a file."""

        with open(filename, "rb") as f:
            data = f.read()

        return cls.pb_loads(data)

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return the compact dimension string."""
        return f"Matrix({self.rows}x{self.cols})"

    def __repr__(self) -> str:
        """Return the detailed representation."""

        rows_str = []

        for i in range(self.rows):
            row = []

            for j in range(self.cols):
                row.append(f"{self[i, j]:.6f}")

            rows_str.append("[" + ", ".join(row) + "]")

        return f"Matrix(name='{self.name}', guid='{self.guid[:8]}...', rows={self.rows}, cols={self.cols}, data=[{'; '.join(rows_str)}])"
