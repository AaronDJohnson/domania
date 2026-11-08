"""Covariance.multiply: ``Sigma(nu) b`` in the layout of the factor's ``solve``, for every
covariance class, and ChannelStack's dense form (spec section 2.5)."""

from __future__ import annotations

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
