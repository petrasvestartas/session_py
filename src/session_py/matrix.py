import math
import uuid
from typing import Optional, List, Tuple


class Matrix:
    def __init__(self, rows: int, cols: int):
        self._guid = None
        self.name = "my_matrix"
        self._rows = rows
        self._cols = cols
        self.data = [0.0] * (rows * cols)

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @staticmethod
    def zeros(rows: int, cols: int) -> "Matrix":
        return Matrix(rows, cols)

    @staticmethod
    def identity(n: int) -> "Matrix":
        m = Matrix(n, n)
        for i in range(n):
            m[i, i] = 1.0
        return m

    @staticmethod
    def from_list(rows: int, cols: int, data: List[float]) -> "Matrix":
        m = Matrix(rows, cols)
        m.data = [float(v) for v in data]
        return m

    @staticmethod
    def from_rows(rows_list: List[List[float]]) -> "Matrix":
        r = len(rows_list)
        c = len(rows_list[0]) if r > 0 else 0
        m = Matrix(r, c)
        for i in range(r):
            for j in range(c):
                m[i, j] = rows_list[i][j]
        return m

    @staticmethod
    def from_cols(cols_list: List[List[float]]) -> "Matrix":
        c = len(cols_list)
        r = len(cols_list[0]) if c > 0 else 0
        m = Matrix(r, c)
        for j in range(c):
            for i in range(r):
                m[i, j] = cols_list[j][i]
        return m

    def _idx(self, r: int, c: int) -> int:
        return r * self._cols + c

    def __getitem__(self, idx):
        r, c = idx
        if not (0 <= r < self._rows and 0 <= c < self._cols):
            raise IndexError(f"Index ({r}, {c}) out of bounds for {self._rows}x{self._cols} matrix")
        return self.data[self._idx(r, c)]

    def __setitem__(self, idx, value):
        r, c = idx
        if not (0 <= r < self._rows and 0 <= c < self._cols):
            raise IndexError(f"Index ({r}, {c}) out of bounds for {self._rows}x{self._cols} matrix")
        self.data[self._idx(r, c)] = float(value)

    @property
    def is_square(self) -> bool:
        return self._rows == self._cols

    @property
    def is_symmetric(self) -> bool:
        if not self.is_square:
            return False
        for i in range(self._rows):
            for j in range(i + 1, self._cols):
                if abs(self[i, j] - self[j, i]) > 1e-10:
                    return False
        return True

    def trace(self) -> float:
        if not self.is_square:
            raise ValueError("Trace requires square matrix")
        return sum(self[i, i] for i in range(self._rows))

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self._rows != other._rows or self._cols != other._cols:
            return False
        for a, b in zip(self.data, other.data):
            if abs(a - b) > 1e-10:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"Matrix({self._rows}x{self._cols})"

    def __repr__(self):
        guid_prefix = self.guid[:8]
        rows_str = []
        for i in range(self._rows):
            row = [f"{self[i, j]:.6f}" for j in range(self._cols)]
            rows_str.append(f"[{', '.join(row)}]")
        return f"Matrix(name='{self.name}', guid='{guid_prefix}...', rows={self._rows}, cols={self._cols}, data=[{'; '.join(rows_str)}])"

    def duplicate(self) -> "Matrix":
        m = Matrix.from_list(self._rows, self._cols, self.data)
        m.name = self.name
        return m

    def add(self, other: "Matrix") -> "Matrix":
        if self._rows != other._rows or self._cols != other._cols:
            raise ValueError("Matrix dimensions must match for addition")
        result = Matrix(self._rows, self._cols)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] + other.data[i]
        return result

    def subtract(self, other: "Matrix") -> "Matrix":
        if self._rows != other._rows or self._cols != other._cols:
            raise ValueError("Matrix dimensions must match for subtraction")
        result = Matrix(self._rows, self._cols)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] - other.data[i]
        return result

    def scale(self, s: float) -> "Matrix":
        result = Matrix(self._rows, self._cols)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] * s
        return result

    def multiply(self, other: "Matrix") -> "Matrix":
        if self._cols != other._rows:
            raise ValueError(f"Cannot multiply {self._rows}x{self._cols} by {other._rows}x{other._cols}")
        result = Matrix(self._rows, other._cols)
        for i in range(self._rows):
            for j in range(other._cols):
                s = 0.0
                for k in range(self._cols):
                    s += self[i, k] * other[k, j]
                result[i, j] = s
        return result

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self.multiply(other)
        return self.scale(float(other))

    def __rmul__(self, other):
        return self.scale(float(other))

    def transpose(self) -> "Matrix":
        result = Matrix(self._cols, self._rows)
        for i in range(self._rows):
            for j in range(self._cols):
                result[j, i] = self[i, j]
        return result

    def _lu_internal(self):
        """LU with partial pivoting. Returns (L, U, P, swap_count)."""
        n = self._rows
        u = Matrix.from_list(n, n, self.data)
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
                    u.data[u._idx(k, j)], u.data[u._idx(max_row, j)] = u.data[u._idx(max_row, j)], u.data[u._idx(k, j)]
                for j in range(n):
                    p.data[p._idx(k, j)], p.data[p._idx(max_row, j)] = p.data[p._idx(max_row, j)], p.data[p._idx(k, j)]
                for j in range(k):
                    l.data[l._idx(k, j)], l.data[l._idx(max_row, j)] = l.data[l._idx(max_row, j)], l.data[l._idx(k, j)]
                swaps += 1
            if abs(u[k, k]) < 1e-14:
                continue
            for i in range(k + 1, n):
                factor = u[i, k] / u[k, k]
                l[i, k] = factor
                for j in range(k, n):
                    u.data[u._idx(i, j)] -= factor * u[k, j]
        return l, u, p, swaps

    def lu_decompose(self) -> Tuple["Matrix", "Matrix", "Matrix"]:
        if not self.is_square:
            raise ValueError("LU decomposition requires square matrix")
        l, u, p, _ = self._lu_internal()
        return l, u, p

    def determinant(self) -> float:
        if not self.is_square:
            raise ValueError("Determinant requires square matrix")
        n = self._rows
        if n == 1:
            return self[0, 0]
        if n == 2:
            return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
        _, u, _, swaps = self._lu_internal()
        sign = 1.0 if swaps % 2 == 0 else -1.0
        prod = 1.0
        for i in range(n):
            prod *= u[i, i]
        return sign * prod

    def inverse(self) -> Optional["Matrix"]:
        if not self.is_square:
            return None
        n = self._rows
        l, u, p, _ = self._lu_internal()
        for i in range(n):
            if abs(u[i, i]) < 1e-14:
                return None
        result = Matrix(n, n)
        eye = Matrix.identity(n)
        for col in range(n):
            pb = [sum(p[i, j] * eye[j, col] for j in range(n)) for i in range(n)]
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
        if not self.is_square or b._rows != self._rows or b._cols != 1:
            return None
        n = self._rows
        l, u, p, _ = self._lu_internal()
        for i in range(n):
            if abs(u[i, i]) < 1e-14:
                return None
        pb = [sum(p[i, j] * b[j, 0] for j in range(n)) for i in range(n)]
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

    def qr_decompose(self) -> Tuple["Matrix", "Matrix"]:
        m, n = self._rows, self._cols
        a_cols = [[self[i, j] for i in range(m)] for j in range(n)]
        q_cols = []
        r = Matrix.zeros(n, n)
        for j in range(n):
            v = list(a_cols[j])
            for i in range(j):
                rij = sum(q_cols[i][k] * v[k] for k in range(m))
                r[i, j] = rij
                for k in range(m):
                    v[k] -= rij * q_cols[i][k]
            norm = math.sqrt(sum(x * x for x in v))
            r[j, j] = norm
            if norm > 1e-14:
                q_cols.append([x / norm for x in v])
            else:
                q_cols.append([0.0] * m)
        q = Matrix.zeros(m, n)
        for j in range(n):
            for i in range(m):
                q[i, j] = q_cols[j][i]
        return q, r

    def cholesky(self) -> Optional["Matrix"]:
        if not self.is_square:
            return None
        n = self._rows
        l = Matrix.zeros(n, n)
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

    def eigenvalues(self) -> List[float]:
        if not self.is_square:
            raise ValueError("Eigenvalues require square matrix")
        n = self._rows
        a = Matrix.from_list(n, n, self.data)
        for _ in range(1000 * n):
            q, r = a.qr_decompose()
            a = r.multiply(q)
            if all(abs(a[i + 1, i]) < 1e-10 for i in range(n - 1)):
                break
        return [a[i, i] for i in range(n)]

    def _eigen_decompose_symmetric(self) -> List[Tuple[float, List[float]]]:
        """Returns [(eigenvalue, eigenvector)] for symmetric matrix."""
        n = self._rows
        a = Matrix.from_list(n, n, self.data)
        v = Matrix.identity(n)
        for _ in range(1000 * n):
            q, r = a.qr_decompose()
            a = r.multiply(q)
            v = v.multiply(q)
            if all(abs(a[i + 1, i]) < 1e-10 for i in range(n - 1)):
                break
        return [(a[i, i], [v[j, i] for j in range(n)]) for i in range(n)]

    def svd(self) -> Tuple["Matrix", List[float], "Matrix"]:
        m, n = self._rows, self._cols
        at = self.transpose()
        ata = at.multiply(self)
        pairs = ata._eigen_decompose_symmetric()
        pairs.sort(key=lambda x: -x[0])
        k = min(m, n)
        singular_values = []
        v_cols = []
        for i in range(k):
            ev, evec = pairs[i]
            singular_values.append(math.sqrt(max(0.0, ev)))
            v_cols.append(evec)
        v = Matrix.zeros(n, k)
        for j in range(k):
            for i in range(n):
                v[i, j] = v_cols[j][i]
        u = Matrix.zeros(m, k)
        for j in range(k):
            if singular_values[j] > 1e-12:
                for i in range(m):
                    val = sum(self[i, l] * v[l, j] for l in range(n))
                    u[i, j] = val / singular_values[j]
        vt = v.transpose()
        return u, singular_values, vt

    def norm_frobenius(self) -> float:
        return math.sqrt(sum(x * x for x in self.data))

    def norm_1(self) -> float:
        max_sum = 0.0
        for j in range(self._cols):
            col_sum = sum(abs(self[i, j]) for i in range(self._rows))
            if col_sum > max_sum:
                max_sum = col_sum
        return max_sum

    def norm_inf(self) -> float:
        max_sum = 0.0
        for i in range(self._rows):
            row_sum = sum(abs(self[i, j]) for j in range(self._cols))
            if row_sum > max_sum:
                max_sum = row_sum
        return max_sum

    def rank(self) -> int:
        _, sv, _ = self.svd()
        if not sv:
            return 0
        threshold = max(self._rows, self._cols) * max(sv) * 1e-10
        return sum(1 for s in sv if s > threshold)

    ###########################################################################################
    # Polymorphic JSON Serialization
    ###########################################################################################

    def __jsondump__(self):
        return {
            "cols": self._cols,
            "data": self.data,
            "guid": self.guid,
            "name": self.name,
            "rows": self._rows,
            "type": f"{self.__class__.__name__}",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        m = cls.from_list(data["rows"], data["cols"], data["data"])
        m.guid = guid
        m.name = name
        return m

    def json_dump(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data, data.get("guid"), data.get("name"))

    def json_dumps(self):
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, json_string):
        import json
        data = json.loads(json_string)
        return cls.__jsonload__(data, data.get("guid"), data.get("name"))

    ###########################################################################################
    # Protobuf Serialization
    ###########################################################################################

    def pb_dumps(self) -> bytes:
        from .proto import matrix_pb2
        proto = matrix_pb2.Matrix()
        proto.guid = self.guid
        proto.name = self.name
        proto.rows = self._rows
        proto.cols = self._cols
        proto.data.extend(self.data)
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Matrix":
        from .proto import matrix_pb2
        proto = matrix_pb2.Matrix()
        proto.ParseFromString(data)
        m = cls.from_list(proto.rows, proto.cols, list(proto.data))
        m.guid = proto.guid
        m.name = proto.name
        return m

    def pb_dump(self, filepath):
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath) -> "Matrix":
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())
