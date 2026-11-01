"""Surviving coverage that the Task 2 pruning dropped, ported from 7211bd5 onto synthetic
inputs: JAX parity of the time and WDM maps, the covariance evaluator paths, the dense and
complex block factors under jit, and ``linalg.blocktri_solve_packed``."""

from __future__ import annotations

import numpy as np
import pytest
from dom_db_inputs import jax_or_skip, synthetic_record

from domainia import linalg as la
from domainia import wilson
from domainia.covariance import CholFactor, Covariance, DiagonalCovariance, as_evaluator
from domainia.domains import CellSet, TimeDomain, WDMDomain
from domainia.maps import CropTime, TimeToFourier, TimeToWDM, WDMToTime, ZeroPadTime
from domainia.wilson.backend import NUMPY

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
NU0 = np.zeros(2)


def rel(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b)))


# -- JAX parity of the time and WDM maps (test_dom_maps_fourier_time::test_jax_parity,
#    test_dom_maps_wdm::test_jax_parity_under_jit_with_the_map_as_argument) ---------------


@pytest.mark.jax
def test_time_fourier_crop_and_pad_maps_under_jit_with_the_map_as_argument():
    jax, JB = jax_or_skip()
    td = TimeDomain(N, DT)
    d = synthetic_record(N, 11)
    F = TimeToFourier(td, fmin=1.0)
    out = jax.jit(lambda m, x: m.apply(x))(F, JB.asarray(d))
    assert isinstance(out, jax.Array)
    ref = F.apply(d)
    assert rel(out, ref) <= 1e-12
    for M in (ZeroPadTime(td, 2 * NF), CropTime(td, 3 * NF, (NT - 4) * NF)):
        o = jax.jit(lambda m, x: m.apply(x))(M, JB.asarray(d))
        assert isinstance(o, jax.Array) and np.array_equal(np.asarray(o), M.apply(d))


@pytest.mark.jax
def test_wdm_maps_under_jit_with_the_map_as_argument():
    jax, JB = jax_or_skip()
    td = TimeDomain(N, DT)
    cells = CellSet.all_active(GRID)
    raw = TimeToWDM(td, WDMDomain(cells))
    cut = TimeToWDM(td, WDMDomain(cells.drop_columns(3, 1)))
    d = synthetic_record(N, 12)
    for M in (raw, cut):
        out = jax.jit(lambda m, x: m.apply(x))(M.on(JB), JB.asarray(d))
        assert isinstance(out, jax.Array)
        assert rel(out, M.apply(d)) <= 1e-12
    y = raw.apply(d)
    back = jax.jit(lambda m, v: m.adjoint(v))(raw.on(JB), JB.asarray(y))
    assert isinstance(back, jax.Array)
    assert rel(back, raw.adjoint(y)) <= 1e-12
    Mi = WDMToTime(raw.dst, td)
    inv = jax.jit(lambda m, v: m.apply(v))(Mi.on(JB), JB.asarray(y))
    assert isinstance(inv, jax.Array)
    assert rel(inv, Mi.apply(y)) <= 1e-12
    assert rel(inv, d) <= 1e-12


# -- evaluator paths (test_dom_cov_base::test_variances_from_callables_and_evaluators) ----


def test_variances_from_callables_and_evaluators(backend, rng):
    base = rng.uniform(0.5, 2.0, 6)

    class Scaled:                                                   # an Evaluator
        def host_constants(self):
            return {"base": base}

        def evaluate(self, C, nu, be):
            return C["base"] * be.xp.exp(nu[0])

    nu = np.array([0.3, 0.0])
    for variance in (Scaled(), lambda nu, be: be.asarray(base) * be.xp.exp(nu[0])):
        cov = DiagonalCovariance(variance, 6)
        assert isinstance(cov, Covariance)
        F = cov.factor(backend.asarray(nu), backend)
        assert np.allclose(np.asarray(F.v), base * np.exp(0.3), rtol=1e-15, atol=0)
        assert np.allclose(cov.dense(nu), np.diag(base * np.exp(0.3)), rtol=1e-15, atol=0)
    with pytest.raises(ValueError, match="the variance has shape"):
        DiagonalCovariance(base, 5).factor(NU0)
    assert as_evaluator(base).evaluate(base, NU0, NUMPY) is base
    for variance in (base, Scaled(), lambda nu, be: be.asarray(base)):
        with pytest.raises(ValueError, match="pytree callables"):
            DiagonalCovariance(variance, 6, channel=1)


# -- dense and complex block factors under jit
#    (test_dom_linalg_jax::test_dense_cblock_schur_and_padded_factors_on_jax, minus the
#    deleted Schur and padded-marginal factors) -------------------------------------------


@pytest.mark.jax
def test_dense_and_cblock_factors_under_jit(jax_be, rng):
    import jax
    import jax.numpy as jnp

    # dense real
    A = rng.standard_normal((20, 20))
    C = A @ A.T + 20 * np.eye(20)
    r = rng.standard_normal(20)
    q = jax.jit(lambda C, r: (lambda f: (f.logdet, f.whiten(r)))(
        CholFactor.build(0.5 * (C + C.T), impl="numpy", be=jax_be)))(jnp.asarray(C), jnp.asarray(r))
    fn = CholFactor.build(0.5 * (C + C.T), impl="numpy", be=NUMPY)
    assert isinstance(q[1], jax.Array)
    assert abs(float(q[0]) - float(fn.logdet)) < 1e-12 * abs(float(fn.logdet))
    assert rel(q[1], fn.whiten(r)) < 1e-12
    # complex Hermitian block tridiagonal
    b, nb = 6, 4
    Z = rng.standard_normal((b * nb, b * nb)) + 1j * rng.standard_normal((b * nb, b * nb))
    M = Z @ Z.conj().T / (b * nb) + 2 * np.eye(b * nb)
    z = rng.standard_normal(b * nb) + 1j * rng.standard_normal(b * nb)
    out = jax.jit(lambda M, z: (lambda f: (f.logdet, la.cblock_backward(f, la.cblock_forward(f, z))))(
        la.cblock_factor(M, b)))(jnp.asarray(M), jnp.asarray(z))
    fc = la.cblock_factor(M, b)
    assert isinstance(out[1], jax.Array)
    assert abs(float(out[0]) - float(fc.logdet)) < 1e-12 * abs(float(fc.logdet))
    assert rel(out[1], la.cblock_backward(fc, la.cblock_forward(fc, z))) < 1e-12


# -- linalg.blocktri_solve_packed ---------------------------------------------------------


def _random_blocktri(rng, sizes):
    """A random SPD block-tridiagonal matrix and its ragged ``(diag, up)`` blocks."""
    n = sum(sizes)
    band = np.repeat(np.arange(len(sizes)), sizes)
    A = rng.standard_normal((n, n))
    C = A @ A.T + n * np.eye(n)
    C = np.where(np.abs(band[:, None] - band[None, :]) <= 1, C, 0.0)
    C = 0.5 * (C + C.T)
    off = np.concatenate([[0], np.cumsum(sizes)])
    diag = [C[off[i]:off[i + 1], off[i]:off[i + 1]] for i in range(len(sizes))]
    up = [C[off[i]:off[i + 1], off[i + 1]:off[i + 2]] for i in range(len(sizes) - 1)]
    return C, diag, up


@pytest.mark.parametrize("sizes", [[3, 5, 4, 0, 2], [6], [4, 4, 4, 4]])
def test_blocktri_solve_packed_is_the_ragged_solve(rng, sizes):
    C, diag, up = _random_blocktri(rng, sizes)
    b = rng.standard_normal((C.shape[0], 3))
    lay = la.BandLayout.from_sizes(sizes)
    fp = la.blocktri_factor(diag, up, layout="padded", band_layout=lay)
    fr = la.blocktri_factor(diag, up, layout="ragged")
    xp = la.blocktri_solve_packed(fp, lay.pack(b))
    assert xp.shape == (lay.nb, lay.B, 3) and np.all(xp[~lay.valid] == 0.0)   # dummies stay zero
    x = lay.unpack(xp)
    assert rel(x, la.blocktri_solve(fr, b)) <= 1e-12
    assert rel(x, np.linalg.solve(C, b)) <= 1e-12
    with pytest.raises(ValueError, match="padded layout"):
        la.blocktri_solve_packed(fr, lay.pack(b))
