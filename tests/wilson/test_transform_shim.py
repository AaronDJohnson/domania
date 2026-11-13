"""The transform against the frozen oracle vendored_2f7136c.

It matches to rounding on random grids, dt != 1, batched and unbatched.
Bit-identical up to v0.11.0; the Hz window of v0.12.0 (Paper I eq:Meyer-def: phi
flat value 1/sqrt(Delta F), kappa = sqrt(2)/Nf, lambda = 1/(sqrt(2) dt), against the
oracle's omega-normalised 1/sqrt(2 pi Delta F), 2 sqrt(pi)/Nf, sqrt(pi)/dt) is the
same product rounded differently, so the gate is ORACLE_RTOL * max|ref|.
"""

from __future__ import annotations

import numpy as np
import pytest
import vendored_2f7136c.transform as oracle

import domainia.wilson.transform as T

GRIDS = [(2, 2, 1.0), (4, 8, 1.0), (8, 16, 0.5), (8, 32, 1.7), (16, 32, 2.0**-10),
         (32, 64, 10.0), (64, 256, 2.0**-10), (6, 10, 0.3)]

#: max|got - ref| over max|ref| against the omega-era oracle.
ORACLE_RTOL = 1e-14     # Hz window (v0.12.0): measured 4.2e-16 fwd, 5.4e-16 inv


def _assert_is_the_oracle_to_rounding(got, ref):
    """The oracle's dtype, and its values to rounding."""
    assert got.dtype == ref.dtype
    assert np.max(np.abs(got - ref)) <= ORACLE_RTOL * np.max(np.abs(ref))


@pytest.mark.parametrize("grid", GRIDS)
def test_forward_family_is_the_oracle_to_rounding(grid, rng):
    Nf, Nt, dt = grid
    N = Nf * Nt
    x = rng.standard_normal(N)
    X = rng.standard_normal((3, N))
    pairs = [
        (T.forward_time(x, Nf, Nt, dt), oracle.forward_time(x, Nf, Nt, dt)),
        (T.forward(np.fft.rfft(x), Nf, Nt, dt), oracle.forward(np.fft.rfft(x), Nf, Nt, dt)),
        (T.forward_time_many(X, Nf, Nt, dt), oracle.forward_time_many(X, Nf, Nt, dt)),
        (T.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt),
         oracle.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt)),
    ]
    for got, ref in pairs:
        _assert_is_the_oracle_to_rounding(got, ref)


@pytest.mark.parametrize("grid", GRIDS)
def test_inverse_family_is_the_oracle_to_rounding(grid, rng):
    Nf, Nt, dt = grid
    w = rng.standard_normal((Nf + 1, Nt))          # junk in inactive pixels on purpose
    W = rng.standard_normal((3, Nf + 1, Nt))
    pairs = [
        (T.inverse(w, Nf, Nt, dt), oracle.inverse(w, Nf, Nt, dt)),
        (T.inverse_time(w, Nf, Nt, dt), oracle.inverse_time(w, Nf, Nt, dt)),
        (T.inverse_many(W, Nf, Nt, dt), oracle.inverse_many(W, Nf, Nt, dt)),
        (T.inverse_time_many(W, Nf, Nt, dt), oracle.inverse_time_many(W, Nf, Nt, dt)),
    ]
    for got, ref in pairs:
        _assert_is_the_oracle_to_rounding(got, ref)


def test_inputs_are_not_mutated(rng):
    Nf, Nt, dt = 8, 16, 0.5
    w = rng.standard_normal((Nf + 1, Nt))
    w0 = w.copy()
    T.inverse(w, Nf, Nt, dt)
    assert np.array_equal(w, w0)
