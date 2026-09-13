from __future__ import annotations

import json
import math
import uuid
from typing import TYPE_CHECKING
from typing import Optional
from typing import Union

from .tolerance import Tolerance

if TYPE_CHECKING:
    from pathlib import Path
    from .proto import matrix_pb2


COMPARISON_TOLERANCE = Tolerance.ABSOLUTE / 10.0
MAX_DIMENSION = 2_147_483_647
PIVOT_TOLERANCE = Tolerance.ZERO_TOLERANCE / 100.0
SINGULAR_TOLERANCE = Tolerance.ZERO_TOLERANCE


def _matrix_size(rows: int, cols: int) -> int:
    """Return the validated element count for C++-compatible dimensions."""
    if rows < 0 or cols < 0:
        raise ValueError("Matrix dimensions cannot be negative")
    if rows > MAX_DIMENSION or cols > MAX_DIMENSION:
        raise ValueError("Matrix dimensions are too large")
    return rows * cols


class Matrix:
    """An NxM matrix with row-major storage.

    Parameters
    ----------
    rows : int, optional
        Row count.
    cols : int, optional
        Column count.

    Attributes
    ----------
    name : str
        Matrix name.
    rows : int
        Row count.
    cols : int
        Column count.
    data : list[float]
        Row-major values.

    Notes
    -----
    Indexing uses ``matrix[row, col]``. Arithmetic operators add, subtract, or
    multiply matrices and preserve the named-method behavior.
    """

    def __init__(self, rows: int = 0, cols: int = 0) -> None:
        """Construct a rows x cols matrix of zeros.

        Raises
        ------
        ValueError
            If either dimension is negative or exceeds the shared API range.
        """
        self._guid = None
        self.name = "my_matrix"
        self.rows = rows
        self.cols = cols
        self.data = [0.0] * _matrix_size(rows, cols)

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

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def zeros(rows: int, cols: int) -> "Matrix":
        """Construct a zero matrix.

        Parameters
        ----------
        rows : int
            Row count.
        cols : int
            Column count.

        Returns
        -------
        Matrix
            Zero matrix with the requested shape.

        Examples
        --------
        >>> Matrix.zeros(2, 3).data
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        """
        return Matrix(rows, cols)

    @staticmethod
    def identity(n: int) -> "Matrix":
        """Construct an identity matrix.

        Parameters
        ----------
        n : int
            Row and column count.

        Returns
        -------
        Matrix
            Identity matrix of order ``n``.

        Examples
        --------
        >>> Matrix.identity(2).data
        [1.0, 0.0, 0.0, 1.0]
        """
        m = Matrix(n, n)
        for i in range(n):
            m[i, i] = 1.0
        return m

    @staticmethod
    def from_vec(rows: int, cols: int, data: list[float]) -> "Matrix":
        """Construct from row-major values.

        Parameters
        ----------
        rows : int
            Row count.
        cols : int
            Column count.
        data : list[float]
            Exact row-major values.

        Returns
        -------
        Matrix
            Matrix containing a copy of ``data``.

        Raises
        ------
        ValueError
            If the dimensions are invalid or do not match ``data``.

        Examples
        --------
        >>> Matrix.from_vec(1, 2, [3.0, 4.0]).data
        [3.0, 4.0]
        """
        if len(data) != _matrix_size(rows, cols):
            raise ValueError("Matrix data size does not match its dimensions")
        m = Matrix(rows, cols)
        m.data = list(data)
        return m

    @staticmethod
    def from_rows(rows_list: list[list[float]]) -> "Matrix":
        """Construct from equal-length rows.

        Parameters
        ----------
        rows_list : list[list[float]]
            Matrix rows.

        Returns
        -------
        Matrix
            Matrix containing the rows.

        Raises
        ------
        ValueError
            If row lengths differ.

        Examples
        --------
        >>> Matrix.from_rows([[1.0, 2.0], [3.0, 4.0]]).rows
        2
        """
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
        """Construct from equal-length columns.

        Parameters
        ----------
        cols_list : list[list[float]]
            Matrix columns.

        Returns
        -------
        Matrix
            Matrix containing the columns.

        Raises
        ------
        ValueError
            If column lengths differ.

        Examples
        --------
        >>> Matrix.from_cols([[1.0, 2.0], [3.0, 4.0]]).data
        [1.0, 3.0, 2.0, 4.0]
        """
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

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════

    def __getitem__(self, idx: tuple[int, int]) -> float:
        """Return the element at ``(row, col)``."""
        r, c = idx
        return self.data[r * self.cols + c]

    def __setitem__(self, idx: tuple[int, int], value: float) -> None:
        """Set the element at ``(row, col)``."""
        r, c = idx
        self.data[r * self.cols + c] = value

    def is_square(self) -> bool:
        """Return whether row and column counts are equal."""
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
        """Return the diagonal sum.

        Raises
        ------
        ValueError
            If the matrix is not square.
        """
        if not self.is_square():
            raise ValueError("Matrix trace requires a square matrix")
        s = 0.0
        for i in range(self.rows):
            s += self[i, i]
        return s

    def duplicate(self) -> "Matrix":
        """Return a copy with a new guid and the same data."""
        m = Matrix.from_vec(self.rows, self.cols, self.data)
        m.name = self.name
        return m

    # ═══════════════════════════════════════════════════════════════════════════
    # Operations
    # ═══════════════════════════════════════════════════════════════════════════

    def add(self, other: "Matrix") -> "Matrix":
        """Add an equal-sized matrix.

        Raises
        ------
        ValueError
            If the dimensions differ.
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")
        result = Matrix(self.rows, self.cols)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] + other.data[i]
        return result

    def subtract(self, other: "Matrix") -> "Matrix":
        """Subtract an equal-sized matrix.

        Raises
        ------
        ValueError
            If the dimensions differ.
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for subtraction")
        result = Matrix(self.rows, self.cols)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] - other.data[i]
        return result

    def scale(self, s: float) -> "Matrix":
        """Return a matrix with every value multiplied by ``s``."""
        result = Matrix(self.rows, self.cols)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] * s
        return result

    def multiply(self, other: "Matrix") -> "Matrix":
        """Multiply by a dimension-compatible matrix.

        Raises
        ------
        ValueError
            If this column count differs from the other row count.
        """
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

    def transpose(self) -> "Matrix":
        """Return the transpose."""
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════

    def __add__(self, other: "Matrix") -> "Matrix":
        """Add an equal-sized matrix."""
        return self.add(other)

    def __sub__(self, other: "Matrix") -> "Matrix":
        """Subtract an equal-sized matrix."""
        return self.subtract(other)

    def __mul__(self, other: "Matrix") -> "Matrix":
        """Multiply by a dimension-compatible matrix."""
        return self.multiply(other)

    def __eq__(self, other: object) -> bool:
        """Compare dimensions and values within the matrix tolerance."""
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
        return not self.__eq__(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # Linear algebra
    # ═══════════════════════════════════════════════════════════════════════════

    def _lu_internal(self) -> tuple["Matrix", "Matrix", "Matrix", int]:
        """(L, U, P, swaps) by partial pivoting"""
        n = self.rows
        u = self.duplicate()
        l = Matrix.identity(n)
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
                    l[k, j], l[max_row, j] = l[max_row, j], l[k, j]
                swaps += 1
            if abs(u[k, k]) < PIVOT_TOLERANCE:
                continue
            for i in range(k + 1, n):
                factor = u[i, k] / u[k, k]
                l[i, k] = factor
                for j in range(k, n):
                    u[i, j] -= factor * u[k, j]
        return l, u, p, swaps

    def lu_decompose(self) -> tuple["Matrix", "Matrix", "Matrix"]:
        """Factor the matrix so that ``P * A = L * U``.

        Returns
        -------
        tuple[Matrix, Matrix, Matrix]
            Lower-triangular, upper-triangular, and permutation matrices.

        Raises
        ------
        ValueError
            If the matrix is not square.
        """
        if not self.is_square():
            raise ValueError("LU decomposition requires a square matrix")
        l, u, p, _swaps = self._lu_internal()
        return l, u, p

    def determinant(self) -> float:
        """Return the determinant.

        Raises
        ------
        ValueError
            If the matrix is not square.
        """
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

    def inverse(self) -> Optional["Matrix"]:
        """Return the inverse, or ``None`` for a non-square or singular matrix."""
        if not self.is_square():
            return None
        n = self.rows
        l, u, p, _swaps = self._lu_internal()
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
                    y[i] -= l[i, j] * y[j]
            x = [0.0] * n
            for i in range(n - 1, -1, -1):
                x[i] = y[i]
                for j in range(i + 1, n):
                    x[i] -= u[i, j] * x[j]
                x[i] /= u[i, i]
            for i in range(n):
                result[i, col] = x[i]
        return result

    def solve(self, b: "Matrix") -> Optional["Matrix"]:
        """Solve ``A * x = b``.

        Parameters
        ----------
        b : Matrix
            Right-hand-side column vector.

        Returns
        -------
        Optional[Matrix]
            Solution column vector, or ``None`` if the shapes are invalid or
            the matrix is singular.
        """
        if not self.is_square() or b.rows != self.rows or b.cols != 1:
            return None
        n = self.rows
        l, u, p, _swaps = self._lu_internal()
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
                y[i] -= l[i, j] * y[j]
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
        """Compute a Gram-Schmidt QR decomposition.

        Returns
        -------
        tuple[Matrix, Matrix]
            Orthogonal and upper-triangular factors.
        """
        m = self.rows
        n = self.cols
        a_cols = [[0.0] * m for j in range(n)]
        for j in range(n):
            for i in range(m):
                a_cols[j][i] = self[i, j]
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

    def cholesky(self) -> Optional["Matrix"]:
        """Return lower ``L`` with ``A = L * L^T`` when it exists.

        Returns
        -------
        Optional[Matrix]
            Lower factor, or ``None`` if the matrix is non-square or not
            positive definite.
        """
        if not self.is_square():
            return None
        n = self.rows
        l = Matrix(n, n)
        for i in range(n):
            for j in range(i + 1):
                s = self[i, j]
                for k in range(j):
                    s -= l[i, k] * l[j, k]
                if i == j:
                    if s <= 0.0:
                        return None
                    l[i, j] = math.sqrt(s)
                else:
                    l[i, j] = s / l[j, j]
        return l

    def eigenvalues(self) -> list[float]:
        """Return eigenvalues computed by unshifted QR iteration.

        Raises
        ------
        ValueError
            If the matrix is not square.
        """
        if not self.is_square():
            raise ValueError("Matrix eigenvalues require a square matrix")
        n = self.rows
        a = self.duplicate()
        for _ in range(1000 * n):
            q, r = a.qr_decompose()
            a = r.multiply(q)
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
        """(eigenvalue, eigenvector) pairs by QR iteration with accumulated Q"""
        n = self.rows
        a = self.duplicate()
        v = Matrix.identity(n)
        for _ in range(1000 * n):
            q, r = a.qr_decompose()
            a = r.multiply(q)
            v = v.multiply(q)
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
        """Compute the singular value decomposition.

        Returns
        -------
        tuple[Matrix, list[float], Matrix]
            ``U``, singular values, and ``V^T`` in that order.
        """
        m = self.rows
        n = self.cols
        at = self.transpose()
        ata = at.multiply(self)
        pairs = ata._eigen_decompose_symmetric()
        pairs.sort(key=lambda pair: -pair[0])
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
                for l in range(n):
                    val += self[i, l] * v[l, j]
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
        """Return the JSON-compatible object representation."""
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
        guid: Optional[str] = None,
        name: Optional[str] = None,
    ) -> "Matrix":
        """Construct from a JSON-compatible object representation.

        Parameters
        ----------
        data : dict
            JSON-compatible matrix object.
        guid : Optional[str], optional
            GUID override.
        name : Optional[str], optional
            Name override.

        Returns
        -------
        Matrix
            Parsed matrix.

        Raises
        ------
        ValueError
            If dimensions are invalid or do not match the data.
        """
        m = cls.from_vec(data["rows"], data["cols"], data["data"])
        m.guid = guid or data["guid"]
        m.name = name or data["name"]
        return m

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> "Matrix":
        """Construct from a JSON string.

        Parameters
        ----------
        json_string : str
            Serialized matrix.

        Returns
        -------
        Matrix
            Parsed matrix.

        Raises
        ------
        ValueError
            If dimensions are invalid or do not match the data.
        json.JSONDecodeError
            If the string is not valid JSON.

        Examples
        --------
        >>> Matrix.file_json_loads(Matrix.identity(2).file_json_dumps()).rows
        2
        """
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filepath: Union[str, Path]) -> None:
        """Serialize to a JSON file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Destination path.
        """
        with open(filepath, "w") as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, Path]) -> "Matrix":
        """Construct from a JSON file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Source path.

        Returns
        -------
        Matrix
            Parsed matrix.

        Examples
        --------
        >>> from pathlib import Path
        >>> path = Path("matrix.json")
        >>> Matrix.identity(2).file_json_dump(path)
        >>> Matrix.file_json_load(path).rows
        2
        >>> path.unlink()
        """
        with open(filepath) as f:
            return cls.__jsonload__(json.load(f))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════

    def to_proto(self) -> matrix_pb2.Matrix:
        """Return the protobuf representation."""
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
        """Construct from a protobuf message.

        Parameters
        ----------
        proto : matrix_pb2.Matrix
            Source protobuf message.

        Returns
        -------
        Matrix
            Parsed matrix.

        Raises
        ------
        ValueError
            If dimensions are invalid or do not match the data.

        Examples
        --------
        >>> matrix = Matrix.identity(2)
        >>> Matrix.from_proto(matrix.to_proto()).data == matrix.data
        True
        """
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
        """Construct from protobuf bytes.

        Parameters
        ----------
        data : bytes
            Serialized protobuf message.

        Returns
        -------
        Matrix
            Parsed matrix.

        Raises
        ------
        ValueError
            If dimensions are invalid or do not match the data.

        Examples
        --------
        >>> Matrix.pb_loads(Matrix.identity(2).pb_dumps()).cols
        2
        """
        from .proto import matrix_pb2

        proto = matrix_pb2.Matrix()
        proto.ParseFromString(data)
        return cls.from_proto(proto)

    def pb_dump(self, filepath: Union[str, Path]) -> None:
        """Serialize to a protobuf file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Destination path.
        """
        data = self.pb_dumps()
        with open(filepath, "wb") as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath: Union[str, Path]) -> "Matrix":
        """Construct from a protobuf file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Source path.

        Returns
        -------
        Matrix
            Parsed matrix.

        Examples
        --------
        >>> from pathlib import Path
        >>> path = Path("matrix.bin")
        >>> Matrix.identity(2).pb_dump(path)
        >>> Matrix.pb_load(path).cols
        2
        >>> path.unlink()
        """
        with open(filepath, "rb") as f:
            data = f.read()
        return cls.pb_loads(data)

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════

    def __str__(self) -> str:
        """Return the compact shape description."""
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
