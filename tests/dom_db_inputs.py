"""Shared synthetic inputs of the domainia tests: the sinusoid grid numbers (a realistic
test size), a dense covariance stand-in and the JAX helper.  Nothing here reads release
data or goldens."""

from __future__ import annotations

import numpy as np
import pytest

#: The toy sinusoid grid: N = 16384, dt = 2^-10, T = 16 s; cut and pad column counts.
SIN_NF, SIN_NT, SIN_DT, SIN_T = 64, 256, 2.0**-10, 16.0
SIN_CUT, SIN_PAD_COLUMNS, SIN_FMIN = (6, 5), 16, 4.0

__all__ = ["SIN_CUT", "SIN_DT", "SIN_FMIN", "SIN_NF", "SIN_NT", "SIN_PAD_COLUMNS", "SIN_T",
           "DenseCov", "jax_or_skip", "synthetic_record"]


def jax_or_skip():
    """``(jax, JAX backend)`` with float64 on; skips in the NumPy-only lane."""
    jax = pytest.importorskip("jax", reason="jax not installed (NumPy-only lane)")
    from domainia.wilson.backend import enable_x64, jax_backend

    enable_x64()
    return jax, jax_backend()


def synthetic_record(N: int, seed: int = 0) -> np.ndarray:
    """A seeded white-noise record of length ``N``."""
    return np.random.default_rng(seed).standard_normal(N)


class DenseCov:
    """A dense covariance stand-in: ``matvec`` (real, row-batched) and ``dense``."""

    def __init__(self, S: np.ndarray) -> None:
        self.S = np.asarray(S)

    def matvec(self, X, be=None):
        return X @ self.S

    def dense(self, be=None):
        return self.S
