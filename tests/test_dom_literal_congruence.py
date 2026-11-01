"""L-A3: Lambda = W C W^T by the fast transform (domainia.literal.congruence) against the dense
W of wdm.matrix.active_W (domainia.testing), NumPy and jitted JAX."""

from __future__ import annotations

import numpy as np
import pytest

import wdm
from domainia.domains import CellSet
from domainia.literal import wdm_congruence, wdm_rows
from domainia.testing import dense_W
from domainia.testing import wdm_congruence as dense_congruence

GRIDS = [wdm.Grid(8, 16, 1 / 16), wdm.Grid(16, 32, 1.0)]


def _matrices(g: wdm.Grid):
    rng = np.random.default_rng(g.N)
    A = rng.standard_normal((g.N, g.N))
    return A @ A.T / g.N, A                                       # SPD and a general matrix


@pytest.mark.parametrize("g", GRIDS, ids=lambda g: f"{g.Nf}x{g.Nt}")
def test_congruence_is_the_dense_one(g):
    cells = CellSet.all_active(g)
    for C in _matrices(g):
        want = dense_congruence(C, cells)
        got = wdm_congruence(C, g)
        assert np.max(np.abs(got - want)) <= 1e-13 * np.abs(want).max()
        assert np.max(np.abs(wdm_congruence(C, g, chunk=5) - got)) <= 1e-15 * np.abs(want).max()
        rows = np.array([0, 3, g.N // 2, g.N - 1])
        assert np.max(np.abs(wdm_congruence(C, g, rows=rows) - want[rows])) <= 1e-13 * np.abs(want).max()


@pytest.mark.parametrize("g", GRIDS, ids=lambda g: f"{g.Nf}x{g.Nt}")
def test_rows_are_the_dense_rows(g):
    rows = np.arange(0, g.N, 7)
    W = dense_W(CellSet.all_active(g))
    assert np.max(np.abs(wdm_rows(g, rows) - W[rows])) <= 1e-15
    with pytest.raises(ValueError, match="rows must lie"):
        wdm_rows(g, [g.N])


def test_the_exact_covariance_is_the_congruence_of_eq54():
    """Eq. 9 applied to the time-domain Eq. 54 covariance is wdm.evolutionary.exact_covariance
    (the Gram form (W G)(W G)^T)."""
    g = GRIDS[0]

    def Sn(f, t):
        return (1.0 + np.maximum(f, 0.5) ** -1.5) * (1.0 + 0.4 * np.cos(2 * np.pi * t / (g.N * g.dt)))

    C = wdm.evolutionary.time_domain_covariance(g.N, g.dt, Sn)
    want = wdm.evolutionary.exact_covariance(g.Nf, g.Nt, g.dt, Sn)
    assert np.max(np.abs(wdm_congruence(C, g) - want)) <= 1e-13 * np.abs(want).max()


def test_bad_input_is_refused():
    g = GRIDS[0]
    with pytest.raises(ValueError, match="C must be"):
        wdm_congruence(np.eye(g.N - 1), g)
    with pytest.raises(ValueError, match="chunk"):
        wdm_congruence(np.eye(g.N), g, chunk=0)


@pytest.mark.jax
@pytest.mark.parametrize("g", GRIDS, ids=lambda g: f"{g.Nf}x{g.Nt}")
def test_jax_jit_matches_numpy(jax_be, g):
    import jax

    C, _ = _matrices(g)
    want = wdm_congruence(C, g)
    full = jax.jit(lambda C: wdm_congruence(C, g, chunk=7, be=jax_be))(jax_be.asarray(C))
    rows = jax.jit(lambda C: wdm_congruence(C, g, rows=np.array([1, 5]), be=jax_be))(jax_be.asarray(C))
    assert np.max(np.abs(np.asarray(full) - want)) <= 1e-13 * np.abs(want).max()
    assert np.max(np.abs(np.asarray(rows) - want[[1, 5]])) <= 1e-13 * np.abs(want).max()
