"""D-A5: the complex Hermitian block factor (``linalg.cblock_factor`` / ``cblock_forward`` /
``cblock_backward``) against, bit for bit, the GB ``fullcache`` formulas it ports (NumPy)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_triangular

from domainia import linalg as la


# -- complex Hermitian (GB fullcache) ---------------------------------------------------


def gb_block_factor(M, b):
    """GB ``fullcache.block_factor`` (fullcache.py:82-95), NumPy, verbatim."""
    nb = M.shape[0] // b

    def blk(i, j):
        return M[i * b:(i + 1) * b, j * b:(j + 1) * b]

    Ls, Cs = [np.linalg.cholesky(blk(0, 0))], [np.zeros((b, b), dtype=M.dtype)]
    for i in range(1, nb):
        C = np.conj(solve_triangular(Ls[-1], blk(i - 1, i), lower=True, check_finite=False)).T
        Ls.append(np.linalg.cholesky(blk(i, i) - C @ np.conj(C).T))
        Cs.append(C)
    return np.stack(Ls), np.stack(Cs)


def gb_forward(Ls, Cs, h):
    """GB ``fullcache.forward``, block branch."""
    nb, b = Ls.shape[0], Ls.shape[1]
    u = [solve_triangular(Ls[0], h[:b], lower=True, check_finite=False)]
    for i in range(1, nb):
        u.append(solve_triangular(Ls[i], h[i * b:(i + 1) * b] - Cs[i] @ u[-1], lower=True,
                                  check_finite=False))
    return np.concatenate(u)


def gb_backward(Ls, Cs, u):
    """GB ``fullcache.backward``, block branch."""
    nb, b = Ls.shape[0], Ls.shape[1]
    y = [None] * nb
    y[-1] = solve_triangular(Ls[-1], u[(nb - 1) * b:], lower=True, trans="C", check_finite=False)
    for i in range(nb - 2, -1, -1):
        y[i] = solve_triangular(Ls[i], u[i * b:(i + 1) * b] - np.conj(Cs[i + 1]).T @ y[i + 1],
                                lower=True, trans="C", check_finite=False)
    return np.concatenate(y)


def test_cblock_is_gb_block_factor_bit_for_bit():
    rng = np.random.default_rng(4)
    b, nb = 6, 4
    n = b * nb
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    M = A @ A.conj().T / n + 2 * np.eye(n)
    Ls, Cs = gb_block_factor(M, b)
    f = la.cblock_factor(M, b)
    assert np.array_equal(f.Ls, Ls) and np.array_equal(f.Cs, Cs) and bool(f.ok)
    assert f.logdet == 2.0 * np.sum(np.log(np.real(np.diagonal(Ls, axis1=-2, axis2=-1))))
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    u = gb_forward(Ls, Cs, z)
    assert np.array_equal(la.cblock_forward(f, z), u)
    assert np.array_equal(la.cblock_backward(f, u), gb_backward(Ls, Cs, u))


def test_cblock_failure_is_flagged():
    M = np.eye(8, dtype=complex)
    M[4:, 4:] *= -1.0
    f = la.cblock_factor(M, 4)
    assert not bool(f.ok) and np.isnan(f.logdet)
    with pytest.raises(ValueError, match="whole blocks"):
        la.cblock_factor(np.eye(9, dtype=complex), 4)
