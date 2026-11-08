"""D-A5: the wdm_dev EV linear-algebra tests retired from ``wdm`` in C8 (part-1 plan note 7),
re-homed on ``domainia`` (originals: tests/wilson/vendored/test_evolutionary.py,
which still runs them against the frozen ``wdm_dev_ev`` oracle), on the one factor family
of ``domainia.covariance``."""

from __future__ import annotations

import numpy as np
import pytest

from domainia.wilson.backend import NUMPY
from domainia import linalg as la
from domainia.covariance import BlockTriCovFactor, CholFactor, DiagFactor


@pytest.fixture
def rng():
    return np.random.default_rng(20261010)


def _spd(n, rng, jitter=1.0):
    A = rng.standard_normal((n, n))
    return A @ A.T / n + jitter * np.eye(n)


def _blocktri(sizes, rng):
    off = np.concatenate([[0], np.cumsum(sizes)])
    C = np.zeros((off[-1], off[-1]))
    for i, s in enumerate(sizes):
        C[off[i]:off[i + 1], off[i]:off[i + 1]] = _spd(s, rng, jitter=4.0)
    for i in range(len(sizes) - 1):
        B = 0.3 * rng.standard_normal((sizes[i], sizes[i + 1]))
        C[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = B
        C[off[i + 1]:off[i + 2], off[i]:off[i + 1]] = B.T
    blocks = [C[off[i]:off[i + 1], off[i]:off[i + 1]] for i in range(len(sizes))]
    upper = [C[off[i]:off[i + 1], off[i + 1]:off[i + 2]] for i in range(len(sizes) - 1)]
    return C, blocks, upper


def test_block_tridiagonal_solver(rng):
    C, blocks, upper = _blocktri([5, 3, 6, 4], rng)
    bt = la.blocktri_factor(blocks, upper, layout="ragged")
    b = rng.standard_normal((C.shape[0], 3))
    np.testing.assert_allclose(la.blocktri_solve(bt, b), np.linalg.solve(C, b), atol=1e-10)
    np.testing.assert_allclose(la.blocktri_solve(bt, b[:, 0]), np.linalg.solve(C, b[:, 0]), atol=1e-10)
    assert bt.logdet == pytest.approx(np.linalg.slogdet(C)[1], abs=1e-10)


def test_solver_flavours_agree_with_numpy(rng):
    C, blocks, upper = _blocktri([4, 4, 4], rng)
    d = np.diag(C).copy()
    b = rng.standard_normal(C.shape[0])
    bt = la.blocktri_factor(blocks, upper, layout="ragged")
    for f, ref in ((CholFactor.build(C, impl="scipy", be=NUMPY), C),
                   (BlockTriCovFactor(bt, bt.logdet, bt.ok, "ragged"), C),
                   (DiagFactor.build(d, NUMPY), np.diag(d))):
        np.testing.assert_allclose(f.solve(b), np.linalg.solve(ref, b), atol=1e-10)
        assert float(f.logdet) == pytest.approx(np.linalg.slogdet(ref)[1], abs=1e-10)
