"""domainia.operators.DenseOperator and operator_matrix: an explicit covariance matrix as an
Operator, and any Operator as its matrix (spec sections 2.3, 2.4)."""

from __future__ import annotations

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


def _n(domain):
    """The record length ``N`` of a full-record domain."""
    return domain.cells.grid.N if isinstance(domain, WDMDomain) else domain.N


def _y(domain):
    """A two-channel record on ``domain`` (canonical: it comes from transform)."""
    time = TimeDomain(_n(domain), DT)
    return transform({k: _rng.standard_normal(time.N) for k in NAMES}, time, domain)


def _psd(n, rank):
    """A random symmetric PSD ``(n, n)`` matrix of rank ``rank`` whose two channel blocks are
    correlated (no block structure)."""
    B = _rng.standard_normal((n, rank)) / np.sqrt(n)
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
    with pytest.raises(TypeError, match="Operator"):
        operator_matrix(C)


def test_a_float32_rank_deficient_matrix_uses_float32_eps():
    n = N
    B = _rng.standard_normal((n, n // 2)).astype(np.float32)
    C = B @ B.T
    C = ((C + C.T) / 2).astype(np.float32)                           # exactly symmetric
    assert DenseOperator(C, TIME, ("A",)).degrees_of_freedom == n // 2


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
