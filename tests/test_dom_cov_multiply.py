"""Covariance.multiply: ``Sigma(nu) b`` in the layout of the factor's ``solve``, for every
covariance class, and ChannelStack's dense form.  ``multiply`` and the factors' ``solve``,
``whiten`` and ``quad`` check the leading shape of ``b`` (spec 2026-11-08 §2.8)."""

from __future__ import annotations

import re

import numpy as np
import pytest

import domainia.covariance.blocktri as blocktri
from domainia import wilson
from domainia.covariance import (
    BlockTridiagonalCovariance,
    ChannelStack,
    Covariance,
    DenseCovariance,
    DiagFactor,
    DiagonalCovariance,
    EvolutionaryCovariance,
    StackFactor,
)
from domainia.domains import CellSet
from domainia.spectra import GenericPSD

NF, NT, DT = 8, 16, 1 / 16
GRID = wilson.Grid(NF, NT, DT)
CELLS = CellSet.all_active(GRID)
CUT = CELLS.drop_columns(3, 2)
NU = np.array([-1.3, 0.4])
_rng = np.random.default_rng(59)


def _drifting(f, t, nu):
    alpha = nu[0] + nu[1] * (np.asarray(t, float) / (GRID.N * DT) - 0.5)
    return np.exp(alpha * np.log(np.maximum(np.asarray(f, float), 0.5)))


PSD = GenericPSD(_drifting)
#: Band sizes with an empty band: the ragged product must skip it.
SIZES = (3, 4, 0, 2, 5)


def _spd(n):
    B = _rng.standard_normal((n, n))
    return B @ B.T / n + np.eye(n)


_DIAG = [_spd(s) for s in SIZES]
_UP = [_rng.standard_normal((SIZES[m], SIZES[m + 1])) / 4 for m in range(len(SIZES) - 1)]
_V = 0.5 + _rng.random(CELLS.size)
_S = _spd(20)


def _blocks(nu, be):
    """Ragged band blocks that depend on ``nu``."""
    s = be.xp.exp(nu[0])
    return [s * be.asarray(D) for D in _DIAG], [s * be.asarray(U) for U in _UP]


COVS = {
    "diagonal-array": lambda: DiagonalCovariance(_V, CELLS.size),
    "diagonal-callable": lambda: DiagonalCovariance(lambda nu, be: be.asarray(_V) * be.xp.exp(nu[0]),
                                                    CELLS.size),
    "dense": lambda: DenseCovariance(lambda nu, be: be.asarray(_S) * be.xp.exp(nu[0]), 20, chol_impl="numpy"),
    "blocktri-ragged": lambda: BlockTridiagonalCovariance(_blocks, SIZES),
    "blocktri-padded": lambda: BlockTridiagonalCovariance(_blocks, SIZES, layout="padded"),
    "evolutionary-blocktri": lambda: EvolutionaryCovariance(PSD, CELLS),
    "evolutionary-padded": lambda: EvolutionaryCovariance(PSD, CUT, layout="padded"),
    "evolutionary-dense": lambda: EvolutionaryCovariance(PSD, CUT, assembly="dense"),
    "evolutionary-direct": lambda: EvolutionaryCovariance(PSD, CUT, builder="direct", assembly="dense"),
}


@pytest.mark.parametrize("name", COVS)
def test_multiply_is_the_dense_matrix_times_b(name):
    cov = COVS[name]()
    b = _rng.standard_normal(cov.n)
    want = cov.dense(NU) @ b
    got = cov.multiply(NU, b)
    assert got.shape == (cov.n,)
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))


def test_block_tridiagonal_multiply_never_forms_the_dense_matrix(monkeypatch):
    """With the dense routes made to raise, the ragged product still runs."""
    covs = [BlockTridiagonalCovariance(_blocks, SIZES), EvolutionaryCovariance(PSD, CELLS)]
    bs = [_rng.standard_normal(c.n) for c in covs]
    want = [c.dense(NU) @ b for c, b in zip(covs, bs, strict=True)]

    def boom(*args, **kwargs):
        raise AssertionError("the dense matrix was formed")

    monkeypatch.setattr(blocktri, "blocks_to_dense", boom)
    monkeypatch.setattr(BlockTridiagonalCovariance, "_matrix", boom)
    for c, b, w in zip(covs, bs, want, strict=True):
        with pytest.raises(AssertionError, match="dense matrix was formed"):
            c.dense(NU)                                              # the patch is live
        np.testing.assert_allclose(c.multiply(NU, b), w, rtol=0, atol=1e-12 * np.max(np.abs(w)))


def test_a_channel_stack_multiplies_each_channel_at_its_slice_of_nu():
    """``b`` is ``(channels, n_c)``; ``dense`` is the block-diagonal of the channels' dense
    matrices at ``split(nu)``, so ``multiply(nu, b).reshape(-1) == dense(nu) @ b.reshape(-1)``."""
    c0 = COVS["diagonal-callable"]()
    c1 = EvolutionaryCovariance(PSD, CELLS)
    st = ChannelStack([c0, c1], 2)
    nu = np.array([0.3, 0.0, -1.1, 0.6])
    D = st.dense(nu)
    n = CELLS.size
    assert D.shape == (2 * n, 2 * n)
    np.testing.assert_array_equal(D[:n, :n], c0.dense(nu[:2]))
    np.testing.assert_array_equal(D[n:, n:], c1.dense(nu[2:]))
    assert not D[:n, n:].any() and not D[n:, :n].any()
    b = _rng.standard_normal((2, n))
    got = st.multiply(nu, b)
    assert got.shape == (2, n)
    want = D @ b.reshape(-1)
    assert np.max(np.abs(got.reshape(-1) - want)) <= 1e-12 * np.max(np.abs(want))
    np.testing.assert_allclose(st.factor(nu).solve(got), b, rtol=0, atol=1e-10 * np.max(np.abs(b)))


class _FactorOnly(Covariance):
    """A covariance with a factor and neither ``_multiply`` nor ``_matrix``."""

    n = 3

    def _factor(self, C, nu, be):
        return DiagFactor.build(np.ones(3), be)


def test_a_covariance_with_neither_multiply_nor_a_dense_form_raises_type_error():
    with pytest.raises(TypeError, match=r"^_FactorOnly has no multiply: it implements neither "
                                        r"_multiply nor _matrix"):
        _FactorOnly().multiply(np.zeros(1), np.ones(3))


# --- the leading shape of b (spec 2026-11-08 §2.8) ------------------------------------------

N = CELLS.size
NU2 = np.array([0.3, 0.0, -1.1, 0.6])


def _two_channels():
    return ChannelStack([COVS["diagonal-callable"](), COVS["diagonal-array"]()], 2)


#: The three silent modes: ``(cov, nu, b, multiply's leading shape, the factor's)``.  Before
#: §2.8 a ``DiagonalCovariance`` broadcast a ``(1,)`` ``b``, a ``BlockTridiagonalCovariance``
#: cut an ``(n + 1,)`` one to its first ``n`` entries (both layouts), and a two-channel
#: ``ChannelStack`` dropped a third row, all without an error.
SILENT = {
    "diagonal-(1,)": (COVS["diagonal-array"], NU, np.ones(1), (N,), (N,)),
    "blocktri-ragged-(n+1,)": (COVS["blocktri-ragged"], NU, np.ones(15), (14,), (14,)),
    "blocktri-padded-(n+1,)": (COVS["blocktri-padded"], NU, np.ones(15), (14,), (14,)),
    "stack-(3,n)": (_two_channels, NU2, np.ones((3, N)), (2, N), (2,)),
}
#: The factor calls, and the method whose check fires: the ``"solve"`` and ``"whiten"``
#: forms of ``quad`` reach the check through ``solve`` and ``whiten``; ``DiagFactor``'s
#: ``"elementwise"`` form and ``StackFactor.quad`` check in ``quad``.
CALLS = {
    "solve": (lambda F, b: F.solve(b), "solve"),
    "whiten": (lambda F, b: F.whiten(b), "whiten"),
    "quad-solve": (lambda F, b: F.quad(b, form="solve"), "solve"),
    "quad-whiten": (lambda F, b: F.quad(b, form="whiten"), "whiten"),
    "quad-elementwise": (lambda F, b: F.quad(b, form="elementwise"), "quad"),
}
FACTOR_CASES = [(mode, call) for mode in SILENT for call in CALLS
                if call != "quad-elementwise" or not mode.startswith("blocktri")]


def _refusal(what, expected, b, end="$"):
    """The check's message.  Under ``jax.jit`` JAX appends a traceback note, so ``end=""``."""
    return (rf"^{re.escape(what)}: b must have leading shape {re.escape(str(expected))}, "
            rf"got shape {re.escape(str(b.shape))}{end}")


@pytest.mark.parametrize("mode", SILENT)
def test_multiply_refuses_b_of_the_wrong_leading_shape(mode):
    make, nu, b, want, _ = SILENT[mode]
    cov = make()
    with pytest.raises(ValueError, match=_refusal(f"{type(cov).__name__}.multiply", want, b)):
        cov.multiply(nu, b)


@pytest.mark.parametrize(("mode", "call"), FACTOR_CASES)
def test_the_factor_refuses_b_of_the_wrong_leading_shape(mode, call):
    make, nu, b, _, want = SILENT[mode]
    F = make().factor(nu)
    fn, method = CALLS[call]
    if isinstance(F, StackFactor) and call.startswith("quad"):
        method = "quad"                                  # StackFactor.quad checks before its channels
    with pytest.raises(ValueError, match=_refusal(f"{type(F).__name__}.{method}", want, b)):
        fn(F, b)


def test_a_channel_stack_whose_channels_differ_in_size_is_refused_when_used():
    """It builds and factorises (``CovarianceOperator`` reports its own size mismatch), but
    ``b`` has one row per channel, so ``multiply`` refuses it, and so do the factor's methods
    (the channel whose size differs from the row's)."""
    st = ChannelStack([DiagonalCovariance(_V, N), DiagonalCovariance(_V[:5], 5)], 0)
    F = st.factor(np.zeros(0))
    assert st.n == N + 5 and bool(F.ok)
    for b in (np.ones((2, N)), np.ones((2, 5))):
        with pytest.raises(ValueError, match=r"^ChannelStack: b has one row per channel, so the channels "
                                             r"must share one size; their row shapes are \(128,\), \(5,\)$"):
            st.multiply(np.zeros(0), b)
    b = np.ones((2, N))
    for fn, method in CALLS.values():
        with pytest.raises(ValueError, match=_refusal(f"DiagFactor.{method}", (5,), b[1])):
            fn(F, b)


@pytest.mark.parametrize("name", COVS)
def test_an_n_by_k_batch_of_columns_still_works(name):
    """Only the leading axes are compared: ``(n, k)`` is ``k`` columns, each as ``(n,)``."""
    cov = COVS[name]()
    B = _rng.standard_normal((cov.n, 3))
    got = cov.multiply(NU, B)
    want = cov.dense(NU) @ B
    assert got.shape == (cov.n, 3)
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))
    F = cov.factor(NU)
    for fn in (F.solve, F.whiten):
        out = fn(B)
        assert out.shape == (cov.n, 3)
        for j in range(3):
            np.testing.assert_allclose(out[:, j], fn(B[:, j]), rtol=0, atol=1e-13 * np.max(np.abs(out)))


def test_a_channel_stack_takes_a_batch_per_channel():
    st = _two_channels()
    B = _rng.standard_normal((2, N, 3))
    got = st.multiply(NU2, B)
    assert got.shape == (2, N, 3)
    np.testing.assert_allclose(got[:, :, 1], st.multiply(NU2, B[:, :, 1]), rtol=0,
                               atol=1e-15 * np.max(np.abs(got)))
    F = st.factor(NU2)
    np.testing.assert_allclose(F.solve(got), B, rtol=0, atol=1e-13 * np.max(np.abs(B)))
    assert F.whiten(B).shape == (2, N, 3)


@pytest.mark.parametrize("name", COVS)
def test_a_list_b_is_read_as_an_array(name):
    """``multiply`` and every factor method read ``b`` with ``be.asarray`` before the check
    (``DiagFactor`` and ``BlockTriCovFactor`` used to fail on ``b.ndim``)."""
    cov = COVS[name]()
    b = _rng.standard_normal(cov.n)
    F = cov.factor(NU)
    for fn in (lambda x: cov.multiply(NU, x), F.solve, F.whiten):
        want = fn(b)
        np.testing.assert_allclose(fn(list(b)), want, rtol=0, atol=1e-14 * np.max(np.abs(want)))
    assert float(F.quad(list(b))) == pytest.approx(float(F.quad(b)), rel=1e-14)


def test_a_failed_ragged_factor_still_reports_the_failure():
    """A ragged factor that failed stops at its first bad block, so its blocks do not add up
    to ``n``: its methods raise the failure, not a shape error."""
    cov = BlockTridiagonalCovariance(lambda nu, be: ([-be.asarray(D) for D in _DIAG],
                                                     [be.asarray(U) for U in _UP]), SIZES)
    F = cov.factor(NU)
    assert not bool(F.ok)
    for call in ("solve", "whiten", "quad-solve", "quad-whiten"):
        with pytest.raises(ValueError, match="^the factorisation failed"):
            CALLS[call][0](F, np.ones(cov.n))


@pytest.mark.jax
def test_the_check_runs_on_jax_and_under_jit(jax_be):
    """``multiply`` on JAX checks as on NumPy.  Shapes are static under ``jax.jit``, so the
    factors' checks run while tracing."""
    import jax
    import jax.numpy as jnp

    diag = COVS["diagonal-array"]()
    nu = jnp.asarray(NU)
    np.testing.assert_allclose(np.asarray(diag.multiply(nu, jnp.ones(N))), _V, rtol=1e-15, atol=0)
    with pytest.raises(ValueError, match=_refusal("DiagonalCovariance.multiply", (N,), np.ones(1))):
        diag.multiply(nu, jnp.ones(1))
    F = diag.factor(nu)
    solve = jax.jit(lambda F, b: F.solve(b))
    np.testing.assert_allclose(np.asarray(solve(F, jnp.ones(N))), 1 / _V, rtol=1e-15, atol=0)
    with pytest.raises(ValueError, match=_refusal("DiagFactor.solve", (N,), np.ones(1), end="")):
        solve(F, jnp.ones(1))
    with pytest.raises(ValueError, match=_refusal("DiagFactor.quad", (N,), np.ones(1), end="")):
        jax.jit(lambda F, b: F.quad(b, form="elementwise"))(F, jnp.ones(1))
    padded = COVS["blocktri-padded"]()
    C, G = padded.constants(jax_be), padded.factor(nu)
    whiten = jax.jit(lambda G, C, b: G.whiten(b, C=C))
    assert whiten(G, C, jnp.ones(14)).shape == (14,)
    with pytest.raises(ValueError, match=_refusal("BlockTriCovFactor.whiten", (14,), np.ones(15), end="")):
        whiten(G, C, jnp.ones(15))
    S = _two_channels().factor(jnp.asarray(NU2))
    with pytest.raises(ValueError, match=_refusal("StackFactor.quad", (2,), np.ones((3, N)), end="")):
        jax.jit(lambda S, b: S.quad(b))(S, jnp.ones((3, N)))
