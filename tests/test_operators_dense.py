"""domainia.operators.DenseOperator and operator_matrix: an explicit covariance matrix as an
Operator, and any Operator as its matrix."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.operators import DenseOperator, operator_matrix
from domainia.testing import ALT_WINDOW
from domainia.translate import Operator, to_coordinates, transform, translate_operator

NF, NT, DT = 4, 8, 0.5
N = NF * NT
TIME = TimeDomain(N, DT)
WDM = WDMDomain.full(wilson.Grid(NF, NT, DT))
WDM_ALT = WDMDomain.full(wilson.Grid(8, 4, DT), window=ALT_WINDOW)
DOMAINS = {
    "time": TIME,
    "fourier": FourierDomain.full(N, DT),
    "fourier-odd": FourierDomain.full(15, DT),
    "wdm": WDM,
    "wdm-alt": WDM_ALT,
}
NAMES = ("A", "E")
_rng = np.random.default_rng(53)
EPS64, EPS32 = float(np.finfo(np.float64).eps), float(np.finfo(np.float32).eps)


def _n(domain):
    """The record length ``N`` of a full-record domain."""
    return domain.cells.grid.N if isinstance(domain, WDMDomain) else domain.N


def _y(domain):
    """A two-channel record on ``domain`` (canonical: it comes from transform)."""
    time = TimeDomain(_n(domain), DT)
    return transform({k: _rng.standard_normal(time.N) for k in NAMES}, time, domain)


def _psd(n, rank, rng=_rng):
    """A random symmetric PSD ``(n, n)`` matrix of rank ``rank`` whose two channel blocks are
    correlated (no block structure)."""
    B = rng.standard_normal((n, rank)) / np.sqrt(n)
    return B @ B.T + (0.3 * np.eye(n) if rank == n else 0.0)


def _close(got, want, rel=1e-10):
    np.testing.assert_allclose(got, want, rtol=0, atol=rel * float(np.max(np.abs(want))))


@pytest.mark.parametrize("rank", ["full", "deficient"])
@pytest.mark.parametrize("name", DOMAINS)
def test_every_method_matches_the_dense_reference(name, rank):
    d = DOMAINS[name]
    n = 2 * _n(d)
    built = n if rank == "full" else n - 7
    C = _psd(n, built)
    op = DenseOperator(C, d, NAMES)
    assert isinstance(op, Operator) and op.domain == d and op.channel_names == NAMES
    w, V = np.linalg.eigh(C)
    keep = w > 1e-9 * w.max()
    assert keep.sum() == built                                      # the reference itself
    w, V = w[keep], V[:, keep]
    y = _y(d)
    r = to_coordinates(y, d, NAMES).reshape(-1)
    want = {"apply": C @ r, "solve": V @ ((V.T @ r) / w), "project": V @ (V.T @ r)}
    for method, v in want.items():
        out = getattr(op, method)(y)
        assert list(out) == list(NAMES)
        transform(out, d, d)                                        # the Operator output contract
        _close(to_coordinates(out, d, NAMES).reshape(-1), v)
    assert op.degrees_of_freedom == built
    assert op.quadratic_form(y) == pytest.approx(float(r @ want["solve"]), rel=1e-10)
    assert op.log_determinant() == pytest.approx(float(np.sum(np.log(w))), rel=1e-10, abs=1e-9)


@pytest.mark.parametrize("target", ["time", "fourier", "wdm-alt"])
def test_a_reversed_channel_dict_meets_the_same_blocks(target):
    """The rows follow ``channel_names``, not the dict's order: on the operator and on a
    translation of it (whose ``transform`` keeps the caller's order)."""
    C = _psd(2 * N, 2 * N)
    op = DenseOperator(C, WDM, NAMES)
    for o in (op, translate_operator(op, DOMAINS[target])):
        y = _y(o.domain)
        flipped = {"E": y["E"], "A": y["A"]}
        for method in ("apply", "solve", "project"):
            a, b = getattr(o, method)(y), getattr(o, method)(flipped)
            assert list(a) == list(b) == list(NAMES)
            for k in NAMES:
                np.testing.assert_array_equal(a[k], b[k])
        assert o.quadratic_form(flipped) == o.quadratic_form(y)


@pytest.mark.parametrize("name", DOMAINS)
def test_operator_matrix_of_a_dense_operator_is_its_symmetrised_matrix(name):
    d = DOMAINS[name]
    C = _psd(2 * _n(d), 2 * _n(d))
    C = C + 1e-14 * np.triu(np.ones_like(C), 1)                     # asymmetric below 1e-12
    S = DenseOperator(C, d, NAMES).matrix
    np.testing.assert_array_equal(S, (C + C.T) / 2)
    M = operator_matrix(DenseOperator(C, d, NAMES))
    if isinstance(d, FourierDomain):                              # sqrt(2) out and back
        _close(M, S, rel=1e-15)
    else:
        np.testing.assert_array_equal(M, S)


@pytest.mark.parametrize("wdm", [WDM, WDM_ALT], ids=["meyer", "alt"])
def test_operator_matrix_translates_a_time_matrix_to_w_c_w_transpose(wdm):
    """``TranslatedOperator.apply`` maps ``w`` to time (``W^T w / dt``), applies ``C``, maps
    back (``W``) and multiplies by ``exp(2 Delta) = dt``: the WDM matrix is ``W C W^T``."""
    C = _psd(N, N)
    g = wdm.cells.grid
    W = wilson.matrix.active_W(g.Nf, g.Nt, g.dt, window=wdm.window)
    got = operator_matrix(translate_operator(DenseOperator(C, TIME, ("A",)), wdm))
    _close(got, W @ C @ W.T, rel=1e-12)


def test_rtol_sets_the_kept_eigenvalues():
    w = np.array([1.0, 0.5, 1e-4, 1e-8] + [1.0] * (N - 4))
    Q = np.linalg.qr(_rng.standard_normal((N, N)))[0]
    C = (Q * w) @ Q.T
    assert DenseOperator(C, TIME, ("A",)).degrees_of_freedom == N
    assert DenseOperator(C, TIME, ("A",), rtol=1e-6).degrees_of_freedom == N - 1
    assert DenseOperator(C, TIME, ("A",), rtol=1e-3).degrees_of_freedom == N - 2


def test_the_matrix_is_validated():
    C = _psd(N, N)
    op = DenseOperator(C, TIME, ("A",))
    assert not op.matrix.flags.writeable and not np.shares_memory(op.matrix, C)
    with pytest.raises(ValueError, match=r"^matrix has shape \(32, 32\); 2 channel\(s\) of 32 coordinates "
                                         r"need shape \(64, 64\)$"):
        DenseOperator(C, TIME, NAMES)
    with pytest.raises(ValueError, match="not finite"):
        DenseOperator(np.where(np.eye(N) > 0, np.nan, C), TIME, ("A",))
    with pytest.raises(ValueError, match="not symmetric"):
        DenseOperator(C + 1e-9 * np.triu(np.ones_like(C), 1), TIME, ("A",))
    with pytest.raises(ValueError, match="not positive semi-definite"):
        DenseOperator(C - 2.0 * np.eye(N) * np.max(np.linalg.eigvalsh(C)), TIME, ("A",))
    with pytest.raises(TypeError, match="real numeric"):
        DenseOperator(C + 0j, TIME, ("A",))
    with pytest.raises(TypeError, match="real numeric"):
        DenseOperator(C.astype(object), TIME, ("A",))
    with pytest.raises(TypeError, match="masked"):
        DenseOperator(np.ma.MaskedArray(C), TIME, ("A",))
    with pytest.raises(TypeError, match="channel_names must be a tuple"):
        DenseOperator(C, TIME, "A")
    with pytest.raises(TypeError, match="^domain must be a full-record domain"):
        DenseOperator(C, "time", ("A",))
    with pytest.raises(ValueError, match="^domain: translate handles only WDMDomain.full"):
        DenseOperator(C, WDMDomain(CellSet.bands(WDM.cells.grid, (1, 2))), ("A",))
    with pytest.raises(TypeError, match="rtol"):
        DenseOperator(C, TIME, ("A",), rtol="1e-6")
    with pytest.raises(ValueError, match="rtol"):
        DenseOperator(C, TIME, ("A",), rtol=-1.0)
    y = {"A": np.zeros(N)}
    with pytest.raises(ValueError, match="channel_names are"):
        op.apply({"E": y["A"]})
    with pytest.raises(TypeError, match="mapping"):
        op.solve([y["A"]])
    with pytest.raises(TypeError, match=r"^op must implement domainia\.translate\.Operator, got ndarray$"):
        operator_matrix(C)


def test_a_float32_rank_deficient_matrix_uses_float32_eps():
    """... and stores it: ``op.rtol`` is ``K N eps32``, so a rebuild from the float64
    ``op.matrix`` (``dataclasses.replace``, or the fields passed back) keeps the same
    directions.  Before, the rebuild used float64's eps and refused the matrix."""
    n = N
    B = _rng.standard_normal((n, n // 2)).astype(np.float32)
    C = B @ B.T
    C = ((C + C.T) / 2).astype(np.float32)                           # exactly symmetric
    op = DenseOperator(C, TIME, ("A",))
    assert op.degrees_of_freedom == n // 2 and op.rtol == n * EPS32
    for again in (dataclasses.replace(op),
                  DenseOperator(op.matrix, op.domain, op.channel_names, rtol=op.rtol)):
        np.testing.assert_array_equal(again.matrix, op.matrix)
        assert again.degrees_of_freedom == op.degrees_of_freedom and again.rtol == op.rtol


def test_operator_matrix_refuses_a_str_channel_names():
    class Fake:
        domain = TIME
        channel_names = "AE"

        degrees_of_freedom = 0

        def apply(self, values):
            return values

        solve = project = apply

        def quadratic_form(self, values):
            return 0.0

        def log_determinant(self):
            return 0.0

    with pytest.raises(TypeError, match="channel_names must be a tuple"):
        operator_matrix(Fake())


def test_a_float16_matrix_keeps_float64_eps():
    """float16's eps (1e-3) would cut every direction below ``max|lambda| K N eps`` (here 6%);
    only float32 narrows the tolerance."""
    n = 2 * N
    w = np.linspace(1.0, 30.0, n)                                    # condition number 30
    Q = np.linalg.qr(_rng.standard_normal((n, n)))[0]
    C = ((Q * w) @ Q.T).astype(np.float16)
    assert DenseOperator(C, TIME, NAMES).degrees_of_freedom == n


def _diag(last):
    """``diag(1, ..., 1, last)`` on ``TIME``: ``eigh`` returns the entries exactly, and
    ``max |lambda| = 1``."""
    return np.diag(np.r_[np.ones(N - 1), last])


def test_the_default_tolerance_is_exact_at_its_boundary():
    """``tol = max|lambda| * (K N eps)``, the code's own expression: ``-tol`` is accepted and
    ``-1.01 tol`` refused; ``+tol`` is dropped and ``+1.01 tol`` kept."""
    tol = 1.0 * (N * EPS64)
    assert DenseOperator(_diag(-1.0 * tol), TIME, ("A",)).degrees_of_freedom == N - 1
    with pytest.raises(ValueError, match="^matrix is not positive semi-definite"):
        DenseOperator(_diag(-1.01 * tol), TIME, ("A",))
    assert DenseOperator(_diag(1.0 * tol), TIME, ("A",)).degrees_of_freedom == N - 1
    assert DenseOperator(_diag(1.01 * tol), TIME, ("A",)).degrees_of_freedom == N


def test_an_rtol_below_round_off_says_so():
    """``rtol=0`` on a rank-deficient ``B B^T``: round-off makes a zero eigenvalue negative,
    and the message says the default accepts it.  The seed is fixed: in about 1 draw in 1200
    (17 of 20000 seeds) round-off leaves every zero eigenvalue >= 0.  ``-I`` with ``rtol=0``
    gets the plain message."""
    n = 2 * N
    C = _psd(n, n - 7, np.random.default_rng(17))
    w = np.linalg.eigh(0.5 * C + 0.5 * C.T)[0]
    assert -(np.max(np.abs(w)) * (n * EPS64)) <= w[0] < 0.0          # the premise: round-off
    assert DenseOperator(C, TIME, NAMES).degrees_of_freedom == n - 7
    with pytest.raises(ValueError, match=r"^matrix is not positive semi-definite at rtol=0\.0: .* "
                                         r"within round-off of zero .* the default rtol=None accepts it$"):
        DenseOperator(C, TIME, NAMES, rtol=0)
    with pytest.raises(ValueError, match="^matrix is not positive semi-definite") as info:
        DenseOperator(-np.eye(N), TIME, ("A",), rtol=0)
    assert "round-off" not in str(info.value)


def test_rtol_forgives_negative_eigenvalues_down_to_minus_rtol_max_lambda():
    C = _diag(-1e-3)
    op = DenseOperator(C, TIME, ("A",), rtol=1e-2)
    assert op.degrees_of_freedom == N - 1 and op.rtol == 1e-2
    with pytest.raises(ValueError, match="^matrix is not positive semi-definite: "):
        DenseOperator(C, TIME, ("A",))


@pytest.mark.parametrize("rtol", [1, 1.0, 5])
def test_an_rtol_of_one_or_more_is_refused(rtol):
    """``max |lambda| >= |lambda_min|``, so from ``rtol = 1`` on no eigenvalue is below
    ``-tol``: ``-I`` was accepted, with no direction kept."""
    with pytest.raises(ValueError, match=r"^rtol must be < 1; got "):
        DenseOperator(-np.eye(N), TIME, ("A",), rtol=rtol)


@pytest.mark.parametrize(("rtol", "kind"), [
    ("1e-6", "str"), (True, "bool"), (np.True_, "bool"), (1e-6 + 0j, "complex"),
    (np.array(1e-6), "ndarray"),
])
def test_rtol_of_the_wrong_kind_raises_type_error(rtol, kind):
    with pytest.raises(TypeError, match=f"^rtol must be a real number, got {kind}$"):
        DenseOperator(np.eye(N), TIME, ("A",), rtol=rtol)


@pytest.mark.parametrize(("matrix", "dtype"), [
    (np.eye(N) + 0j, "complex128"), (np.eye(N, dtype=np.complex64), "complex64"),
    (np.eye(N).astype(object), "object"), (np.eye(N) > 0, "bool"),
])
def test_a_matrix_of_the_wrong_dtype_raises_type_error(matrix, dtype):
    with pytest.raises(TypeError, match=f"^matrix must be a real numeric array, got dtype {dtype}$"):
        DenseOperator(matrix, TIME, ("A",))


def test_rtol_reads_back_the_tolerance_used():
    """``op.rtol`` is never ``None``: ``K N eps`` for ``rtol=None`` (float32's eps for a float32
    matrix), else the given ``rtol``, as a Python float."""
    C = _psd(N, N)
    op = DenseOperator(C, TIME, ("A",))
    assert type(op.rtol) is float and op.rtol == N * EPS64
    assert DenseOperator(C.astype(np.float32), TIME, ("A",)).rtol == N * EPS32
    assert DenseOperator(np.eye(N, dtype=np.int64), TIME, ("A",)).rtol == N * EPS64
    for given, want in ((np.float64(1e-6), 1e-6), (np.int64(0), 0.0)):
        op = DenseOperator(C, TIME, ("A",), rtol=given)
        assert type(op.rtol) is float and op.rtol == want


def test_huge_matrices_fail_closed():
    """``max|lambda| * K N * eps`` overflowed to inf for ``-1.2e307 I`` (accepted, nothing kept);
    ``(C + C^T) / 2`` overflowed for ``diag(1e308)``; ``eigh`` returns inf for a finite matrix
    whose largest eigenvalue (here 2e308) is beyond float64."""
    time16 = TimeDomain(16, DT)
    with pytest.raises(ValueError, match="^matrix is not positive semi-definite: "):
        DenseOperator(-1.2e307 * np.eye(16), time16, ("A",))
    op = DenseOperator(np.diag(np.full(16, 1e308)), time16, ("A",))
    assert op.degrees_of_freedom == 16
    np.testing.assert_array_equal(op.matrix, 1e308 * np.eye(16))
    with pytest.raises(ValueError, match="^matrix: numpy.linalg.eigh returned a non-finite eigenvalue"):
        DenseOperator(np.full((4, 4), 5e307), TimeDomain(4, DT), ("A",))


def test_the_zero_matrix_is_refused():
    with pytest.raises(ValueError, match="^matrix is zero"):
        DenseOperator(np.zeros((N, N)), TIME, ("A",))


def test_the_symmetry_factor_follows_the_dtype():
    """``max |C - C^T| <= f max |C|`` with ``f = 1e-12``, or ``1e-12 eps32 / eps64`` (about
    5.4e-4) for a float32 matrix: a float32 product ``B D B^T`` nobody symmetrised (here
    ``D`` spans 8 decades) passes, a float64 asymmetry of 1e-11 does not, and the message
    prints the factor applied."""
    rng = np.random.default_rng(11)
    B = rng.standard_normal((N, N)).astype(np.float32)
    D = np.diag(np.logspace(0, -8, N)).astype(np.float32)
    C = B @ D @ B.T
    assert 1e-12 < np.max(np.abs(C - C.T)) / np.max(np.abs(C)) < 1e-6   # the premise
    assert DenseOperator(C, TIME, ("A",)).degrees_of_freedom >= N // 2   # 18 to 21 in 500 seeds
    C = _psd(N, N, rng)
    up = np.max(np.abs(C)) * np.triu(np.ones_like(C), 1)
    with pytest.raises(ValueError, match=r"^matrix is not symmetric: max \|C - C\^T\| is 1e-11 of "
                                         r"max \|C\| \(more than 1e-12\)$"):
        DenseOperator(C + 1e-11 * up, TIME, ("A",))
    with pytest.raises(ValueError, match=r"\(more than 0\.00054\)$"):
        DenseOperator((C + 1e-2 * up).astype(np.float32), TIME, ("A",))
