"""D-C1: the covariance protocol, DiagFactor / CholFactor, DiagonalCovariance, DenseCovariance.

* Every factor is the dense Gaussian: solve, whiten, the three quadratic-form orders,
  log-determinant -- on both backends.
* ``ok`` is False (the likelihood's -inf) on a covariance that is not positive definite.
* Under JAX the constants reach the compiled factorisation as an argument.
"""

from __future__ import annotations

import numpy as np
import pytest

from domainia.covariance import CholFactor, DenseCovariance, DiagFactor, DiagonalCovariance, Factor

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
    """``symmetrize=False`` factorises S's lower triangle (WT cho_factor);
    ``symmetrize=True`` factorises 0.5 (S + S^T) (GB)."""
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
