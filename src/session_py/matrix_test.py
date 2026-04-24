import math
from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("Matrix", "Constructor")
def test_matrix_constructor():
    from session_py import Matrix
    m = Matrix.zeros(2, 3)
    eye = Matrix.identity(3)
    ml = Matrix.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
    mr = Matrix.from_rows([[1.0, 2.0], [3.0, 4.0]])
    mc = Matrix.from_cols([[1.0, 3.0], [2.0, 4.0]])
    v00 = ml[0, 0]
    v01 = ml[0, 1]
    v10 = ml[1, 0]
    v11 = ml[1, 1]
    eq = (ml == mr)
    ne = (ml != Matrix.identity(2))
    sstr = str(m)
    srepr = repr(eye)
    d = ml.duplicate()

    MINI_CHECK(m.rows == 2 and m.cols == 3)
    MINI_CHECK(m.name == "my_matrix" and m.guid != "")
    MINI_CHECK(eye[0, 0] == 1.0 and eye[1, 1] == 1.0 and eye[2, 2] == 1.0)
    MINI_CHECK(eye[0, 1] == 0.0)
    MINI_CHECK(v00 == 1.0 and v01 == 2.0 and v10 == 3.0 and v11 == 4.0)
    MINI_CHECK(mc == ml)
    MINI_CHECK(eq)
    MINI_CHECK(ne)
    MINI_CHECK("Matrix(2x3)" in sstr)
    MINI_CHECK("Matrix(" in srepr)
    MINI_CHECK(d == ml and d.guid != ml.guid)


@MINI_TEST("Matrix", "Properties")
def test_matrix_properties():
    from session_py import Matrix
    m1 = Matrix.identity(3)
    m2 = Matrix.zeros(2, 3)
    m3 = Matrix.from_list(3, 3, [1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0])
    m4 = Matrix.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
    sq1 = m1.is_square
    sq2 = m2.is_square
    sym1 = m3.is_symmetric
    sym2 = m4.is_symmetric
    tr = m1.trace()

    MINI_CHECK(sq1)
    MINI_CHECK(not sq2)
    MINI_CHECK(sym1)
    MINI_CHECK(not sym2)
    MINI_CHECK(TOLERANCE.is_close(tr, 3.0))


@MINI_TEST("Matrix", "Add")
def test_matrix_add():
    from session_py import Matrix
    a = Matrix.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
    b = Matrix.from_list(2, 2, [5.0, 6.0, 7.0, 8.0])
    c = a.add(b)
    d = a + b

    MINI_CHECK(c[0, 0] == 6.0 and c[0, 1] == 8.0)
    MINI_CHECK(c[1, 0] == 10.0 and c[1, 1] == 12.0)
    MINI_CHECK(c == d)


@MINI_TEST("Matrix", "Subtract")
def test_matrix_subtract():
    from session_py import Matrix
    a = Matrix.from_list(2, 2, [5.0, 6.0, 7.0, 8.0])
    b = Matrix.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
    c = a.subtract(b)
    d = a - b

    MINI_CHECK(c[0, 0] == 4.0 and c[0, 1] == 4.0)
    MINI_CHECK(c[1, 0] == 4.0 and c[1, 1] == 4.0)
    MINI_CHECK(c == d)


@MINI_TEST("Matrix", "Scale")
def test_matrix_scale():
    from session_py import Matrix
    a = Matrix.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
    b = a.scale(2.0)
    c = a * 3.0
    d = 4.0 * a

    MINI_CHECK(b[0, 0] == 2.0 and b[0, 1] == 4.0 and b[1, 0] == 6.0 and b[1, 1] == 8.0)
    MINI_CHECK(c[0, 0] == 3.0 and c[1, 1] == 12.0)
    MINI_CHECK(d[0, 0] == 4.0 and d[1, 1] == 16.0)


@MINI_TEST("Matrix", "Multiply")
def test_matrix_multiply():
    from session_py import Matrix
    a = Matrix.from_list(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = Matrix.from_list(3, 2, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    c = a.multiply(b)
    d = a * b

    MINI_CHECK(c.rows == 2 and c.cols == 2)
    MINI_CHECK(TOLERANCE.is_close(c[0, 0], 58.0) and TOLERANCE.is_close(c[0, 1], 64.0))
    MINI_CHECK(TOLERANCE.is_close(c[1, 0], 139.0) and TOLERANCE.is_close(c[1, 1], 154.0))
    MINI_CHECK(c == d)


@MINI_TEST("Matrix", "Transpose")
def test_matrix_transpose():
    from session_py import Matrix
    a = Matrix.from_list(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t = a.transpose()

    MINI_CHECK(t.rows == 3 and t.cols == 2)
    MINI_CHECK(t[0, 0] == 1.0 and t[1, 0] == 2.0 and t[2, 0] == 3.0)
    MINI_CHECK(t[0, 1] == 4.0 and t[1, 1] == 5.0 and t[2, 1] == 6.0)


@MINI_TEST("Matrix", "Determinant")
def test_matrix_determinant():
    from session_py import Matrix
    a1 = Matrix.from_list(1, 1, [5.0])
    a2 = Matrix.from_list(2, 2, [4.0, 7.0, 2.0, 6.0])
    a3 = Matrix.from_list(3, 3, [1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0])
    eye3 = Matrix.identity(3)

    MINI_CHECK(TOLERANCE.is_close(a1.determinant(), 5.0))
    MINI_CHECK(TOLERANCE.is_close(a2.determinant(), 10.0))
    MINI_CHECK(TOLERANCE.is_close(a3.determinant(), 1.0))
    MINI_CHECK(TOLERANCE.is_close(eye3.determinant(), 1.0))


@MINI_TEST("Matrix", "Inverse")
def test_matrix_inverse():
    from session_py import Matrix
    a = Matrix.from_list(2, 2, [4.0, 7.0, 2.0, 6.0])
    inv = a.inverse()
    singular = Matrix.from_list(2, 2, [1.0, 2.0, 2.0, 4.0])
    inv_none = singular.inverse()
    prod = a.multiply(inv)

    MINI_CHECK(inv is not None)
    MINI_CHECK(TOLERANCE.is_close(inv[0, 0], 0.6) and TOLERANCE.is_close(inv[0, 1], -0.7))
    MINI_CHECK(TOLERANCE.is_close(inv[1, 0], -0.2) and TOLERANCE.is_close(inv[1, 1], 0.4))
    MINI_CHECK(inv_none is None)
    MINI_CHECK(TOLERANCE.is_close(prod[0, 0], 1.0) and TOLERANCE.is_close(prod[1, 1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(prod[0, 1], 0.0) and TOLERANCE.is_close(prod[1, 0], 0.0))


@MINI_TEST("Matrix", "Solve")
def test_matrix_solve():
    from session_py import Matrix
    a = Matrix.from_list(2, 2, [2.0, 1.0, 1.0, 3.0])
    b = Matrix.from_list(2, 1, [5.0, 10.0])
    x = a.solve(b)
    # 2x+y=5, x+3y=10 → x=1, y=3
    residual_0 = 2.0 * x[0, 0] + 1.0 * x[1, 0]
    residual_1 = 1.0 * x[0, 0] + 3.0 * x[1, 0]

    MINI_CHECK(x is not None)
    MINI_CHECK(TOLERANCE.is_close(x[0, 0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(x[1, 0], 3.0))
    MINI_CHECK(TOLERANCE.is_close(residual_0, 5.0))
    MINI_CHECK(TOLERANCE.is_close(residual_1, 10.0))


@MINI_TEST("Matrix", "Lu Decompose")
def test_matrix_lu_decompose():
    from session_py import Matrix
    a = Matrix.from_list(3, 3, [2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0])
    l, u, p = a.lu_decompose()
    pa = p.multiply(a)
    lu = l.multiply(u)

    MINI_CHECK(l.rows == 3 and u.cols == 3)
    MINI_CHECK(TOLERANCE.is_close(pa[0, 0], lu[0, 0]))
    MINI_CHECK(TOLERANCE.is_close(pa[0, 1], lu[0, 1]))
    MINI_CHECK(TOLERANCE.is_close(pa[1, 0], lu[1, 0]))
    MINI_CHECK(TOLERANCE.is_close(pa[2, 2], lu[2, 2]))
    # L is lower triangular
    MINI_CHECK(TOLERANCE.is_close(l[0, 1], 0.0) and TOLERANCE.is_close(l[0, 2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(l[1, 2], 0.0))


@MINI_TEST("Matrix", "Qr Decompose")
def test_matrix_qr_decompose():
    from session_py import Matrix
    a = Matrix.from_list(3, 3, [12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0])
    q, r = a.qr_decompose()
    qt = q.transpose()
    qtq = qt.multiply(q)
    qr_prod = q.multiply(r)

    MINI_CHECK(TOLERANCE.is_close(qtq[0, 0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(qtq[1, 1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(qtq[2, 2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(qtq[0, 1], 0.0) and TOLERANCE.is_close(qtq[0, 2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(qr_prod[0, 0], 12.0))
    MINI_CHECK(TOLERANCE.is_close(qr_prod[1, 1], 167.0))
    MINI_CHECK(TOLERANCE.is_close(qr_prod[2, 2], -41.0))


@MINI_TEST("Matrix", "Cholesky")
def test_matrix_cholesky():
    from session_py import Matrix
    a = Matrix.from_list(3, 3, [4.0, 2.0, 2.0, 2.0, 5.0, 3.0, 2.0, 3.0, 6.0])
    l = a.cholesky()
    lt = l.transpose()
    llt = l.multiply(lt)
    not_spd = Matrix.from_list(2, 2, [1.0, 2.0, 2.0, 1.0])
    l_none = not_spd.cholesky()

    MINI_CHECK(l is not None)
    MINI_CHECK(TOLERANCE.is_close(llt[0, 0], 4.0) and TOLERANCE.is_close(llt[0, 1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(llt[1, 0], 2.0) and TOLERANCE.is_close(llt[1, 1], 5.0))
    MINI_CHECK(TOLERANCE.is_close(llt[2, 2], 6.0))
    MINI_CHECK(l_none is None)


@MINI_TEST("Matrix", "Eigenvalues")
def test_matrix_eigenvalues():
    from session_py import Matrix
    a = Matrix.from_list(3, 3, [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0])
    evs = a.eigenvalues()
    evs_sorted = sorted(evs)

    MINI_CHECK(len(evs) == 3)
    MINI_CHECK(TOLERANCE.is_close(evs_sorted[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(evs_sorted[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(evs_sorted[2], 3.0))


@MINI_TEST("Matrix", "Svd")
def test_matrix_svd():
    from session_py import Matrix
    a = Matrix.from_list(3, 3, [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
    u, sv, vt = a.svd()
    sv_sorted = sorted(sv, reverse=True)

    MINI_CHECK(len(sv) == 3)
    MINI_CHECK(TOLERANCE.is_close(sv_sorted[0], 3.0))
    MINI_CHECK(TOLERANCE.is_close(sv_sorted[1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(sv_sorted[2], 1.0))


@MINI_TEST("Matrix", "Norms")
def test_matrix_norms():
    from session_py import Matrix
    a = Matrix.from_list(2, 2, [1.0, -2.0, 3.0, -4.0])
    nf = a.norm_frobenius()
    n1 = a.norm_1()
    ni = a.norm_inf()

    MINI_CHECK(TOLERANCE.is_close(nf, math.sqrt(30.0)))
    MINI_CHECK(TOLERANCE.is_close(n1, 6.0))
    MINI_CHECK(TOLERANCE.is_close(ni, 7.0))


@MINI_TEST("Matrix", "Rank")
def test_matrix_rank():
    from session_py import Matrix
    a = Matrix.identity(3)
    b = Matrix.from_list(3, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    c = Matrix.zeros(3, 3)

    MINI_CHECK(a.rank() == 3)
    MINI_CHECK(b.rank() == 2)
    MINI_CHECK(c.rank() == 0)


@MINI_TEST("Matrix", "Json Roundtrip")
def test_matrix_json_roundtrip():
    from session_py import Matrix
    from pathlib import Path
    a = Matrix.from_list(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a.name = "test_matrix"
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_matrix.json"
    a.file_json_dump(fname)
    loaded = Matrix.file_json_load(fname)

    MINI_CHECK(loaded.name == "test_matrix")
    MINI_CHECK(loaded.rows == 2 and loaded.cols == 3)
    MINI_CHECK(TOLERANCE.is_close(loaded[0, 0], 1.0) and TOLERANCE.is_close(loaded[1, 2], 6.0))


@MINI_TEST("Matrix", "Protobuf Roundtrip")
def test_matrix_protobuf_roundtrip():
    from session_py import Matrix
    from pathlib import Path
    a = Matrix.from_list(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a.name = "test_matrix_proto"
    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_matrix.bin"
    a.pb_dump(fname)
    loaded = Matrix.pb_load(fname)

    MINI_CHECK(loaded.name == "test_matrix_proto")
    MINI_CHECK(loaded.rows == 2 and loaded.cols == 3)
    MINI_CHECK(TOLERANCE.is_close(loaded[0, 0], 1.0) and TOLERANCE.is_close(loaded[1, 2], 6.0))


if __name__ == "__main__":
    run_all("python")
