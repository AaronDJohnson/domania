"""The covariance protocol, DiagFactor / CholFactor, DiagonalCovariance, DenseCovariance.

* Every factor is the dense Gaussian: solve, whiten, the three quadratic-form orders,
  log-determinant -- on both backends.
* ``ok`` is False (a caller's -inf) on a covariance that is not positive definite.
* Under JAX the constants reach the compiled factorisation as an argument.
* The ``domainia.variance`` classes are host-only: a ``DiagonalCovariance`` on one is not
  jittable and factorises eagerly on JAX.
"""

from __future__ import annotations

import numpy as np
import pytest
from dom_db_inputs import jax_or_skip

from domainia import wilson
from domainia.covariance import CholFactor, DenseCovariance, DiagFactor, DiagonalCovariance, Factor
from domainia.domains import CellSet
from domainia.spectra import GenericPSD
from domainia.variance import BandAveragedVariance, MidpointVariance

NU0 = np.zeros(2)


def _spd(rng: np.random.Generator, n: int) -> np.ndarray:
    A = rng.standard_normal((n, n))
    return A @ A.T / n + np.eye(n)


def test_diag_factor_is_the_dense_gaussian(backend, rng):
    v = rng.uniform(0.5, 2.0, 12)
    b, B = rng.standard_normal(12), rng.standard_normal((12, 3))
    F = DiagonalCovariance(v, 12).factor(backend.asarray(NU0), backend)
    assert isinstance(F, (DiagFactor, Factor)) and F.n_u == 0 and bool(F.ok)
    C = np.diag(v)
    assert np.allclose(np.asarray(F.solve(backend.asarray(b))), np.linalg.solve(C, b), rtol=1e-14)
    assert np.allclose(np.asarray(F.solve(backend.asarray(B))), np.linalg.solve(C, B), rtol=1e-14)
    want = b @ np.linalg.solve(C, b)
    for form in ("solve", "whiten", "elementwise"):
        assert float(F.quad(backend.asarray(b), form=form)) == pytest.approx(want, rel=1e-14)
    w = np.asarray(F.whiten(backend.asarray(b)))
    assert w @ w == pytest.approx(want, rel=1e-14)
    assert float(F.logdet) == pytest.approx(np.linalg.slogdet(C)[1], rel=1e-14)


@pytest.mark.parametrize("impl", ["numpy", "scipy"])
def test_chol_factor_is_the_dense_gaussian(backend, rng, impl):
    S = _spd(rng, 10)
    b, B = rng.standard_normal(10), rng.standard_normal((10, 4))
    F = DenseCovariance(S, 10, chol_impl=impl).factor(backend.asarray(NU0), backend)
    assert isinstance(F, CholFactor) and F.impl == impl and bool(F.ok)
    assert np.allclose(np.asarray(F.solve(backend.asarray(B))), np.linalg.solve(S, B), rtol=1e-12)
    want = b @ np.linalg.solve(S, b)
    for form in ("solve", "whiten"):
        assert float(F.quad(backend.asarray(b), form=form)) == pytest.approx(want, rel=1e-12)
    assert float(F.logdet) == pytest.approx(np.linalg.slogdet(S)[1], rel=1e-13)
    with pytest.raises(ValueError, match="quadratic form"):
        F.quad(backend.asarray(b), form="elementwise")


def test_symmetrize_selects_the_oracle_matrix(rng):
    """``symmetrize=False`` factorises S's lower triangle (``cho_factor``);
    ``symmetrize=True`` factorises 0.5 (S + S^T)."""
    S = _spd(rng, 8)
    junk = S + 0.1 * np.triu(rng.standard_normal((8, 8)), 1)      # upper triangle perturbed (|.| << 1: PD)
    low = np.tril(junk) + np.tril(junk, -1).T
    F0 = DenseCovariance(junk, 8, chol_impl="scipy", symmetrize=False).factor(NU0)
    assert float(F0.logdet) == pytest.approx(np.linalg.slogdet(low)[1], rel=1e-13)
    F1 = DenseCovariance(junk, 8, chol_impl="numpy", symmetrize=True).factor(NU0)
    assert float(F1.logdet) == pytest.approx(np.linalg.slogdet(0.5 * (junk + junk.T))[1], rel=1e-13)


def test_not_positive_definite_sets_ok_false(backend):
    v = np.array([1.0, -1.0, 2.0])
    with np.errstate(invalid="ignore"):
        F = DiagonalCovariance(v, 3).factor(backend.asarray(NU0), backend)
    assert not bool(F.ok)
    S = np.diag([1.0, -1.0, 2.0])
    for impl in ("numpy", "scipy"):
        F = DenseCovariance(S, 3, chol_impl=impl).factor(backend.asarray(NU0), backend)
        assert not bool(F.ok) and not np.isfinite(float(F.logdet))


@pytest.mark.jax
def test_jit_takes_the_constants_as_an_argument(jax_be):
    """A 2**20-entry constant must not be baked into the compiled program."""
    import jax.numpy as jnp

    big = np.linspace(1.0, 2.0, 1 << 20)

    class Big:
        def host_constants(self):
            return {"big": big}

        def evaluate(self, C, nu, be):
            return C["big"][:16] * be.xp.exp(nu[0])

    cov = DiagonalCovariance(Big(), 16)
    F = cov.factor(jnp.asarray([0.1, 0.0]))
    assert type(F.logdet).__module__.startswith("jax")
    assert cov.compiled_keys() == (None,)                     # one program, no static key
    fn = cov._jit_cache["factor", None]
    text = fn.lower(cov.constants(jax_be), jnp.asarray([0.1, 0.0]), None).as_text()
    assert len(text) < 200_000
    assert float(F.logdet) == pytest.approx(float(cov.factor(np.array([0.1, 0.0])).logdet), rel=1e-15)


def _generic_variances():
    cells = CellSet.all_active(wilson.Grid(8, 16, 0.5))
    psd = GenericPSD(lambda f, t, nu: (1.0 + np.abs(f)) ** -1.0 * (1.0 + nu[0] * np.asarray(t) / 64.0))
    return cells, [Var(psd, cells) for Var in (BandAveragedVariance, MidpointVariance)]


def test_generic_psd_variances_are_not_jittable():
    """The ``domainia.variance`` classes call the PSD on the host; arrays and callables
    stay jittable."""
    cells, variances = _generic_variances()
    for var in variances:
        assert DiagonalCovariance(var, cells.size).jittable is False
    assert DiagonalCovariance(np.ones(3), 3).jittable is True
    assert DiagonalCovariance(lambda nu, be: be.asarray(np.ones(3)), 3).jittable is True


@pytest.mark.jax
def test_generic_psd_variances_factor_eagerly_on_jax():
    jax, _ = jax_or_skip()
    import jax.numpy as jnp

    cells, variances = _generic_variances()
    nu = np.array([0.3])
    for var in variances:
        cov = DiagonalCovariance(var, cells.size)
        want = cov.factor(nu)
        for jit in (None, False):
            got = cov.factor(jnp.asarray(nu), jit=jit)
            assert isinstance(got.logdet, jax.Array) and bool(got.ok)
            assert np.array_equal(np.asarray(got.v), np.asarray(want.v))
            assert float(got.logdet) == pytest.approx(float(want.logdet), rel=1e-14)
