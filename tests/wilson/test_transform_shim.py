"""The transform against the frozen oracle vendored_2f7136c.

It matches to rounding on random grids, dt != 1,
float32 and float64, batched and unbatched.  Bit-identical up to v0.11.0; the
Hz window of v0.12.0 (Paper I eq:Meyer-def: phi flat value 1/sqrt(Delta F),
kappa = sqrt(2)/Nf, lambda = 1/(sqrt(2) dt), against the oracle's
omega-normalised 1/sqrt(2 pi Delta F), 2 sqrt(pi)/Nf, sqrt(pi)/dt) is the same
product rounded differently, so the gate is ORACLE_RTOL[dtype] * max|ref|.
float32 runs in single precision end to end, so its bits differ from the
oracle's at rounding; output dtypes are checked against domainia's table
(``test_single_precision.OUTPUT_DTYPE``), not the oracle's.
"""

from __future__ import annotations

import numpy as np
import pytest
import vendored_2f7136c.transform as oracle

import domainia.wilson.transform as T

from .test_single_precision import output_dtype

GRIDS = [(2, 2, 1.0), (4, 8, 1.0), (8, 16, 0.5), (8, 32, 1.7), (16, 32, 2.0**-10),
         (32, 64, 10.0), (64, 256, 2.0**-10), (6, 10, 0.3)]
DTYPES = [np.float64, np.float32]


#: max|got - ref| over max|ref| against the omega-era oracle, by working precision.
ORACLE_RTOL = {
    np.dtype(np.float64): 1e-14,     # Hz window (v0.12.0): measured 4.2e-16 fwd, 5.4e-16 inv
    np.dtype(np.float32): 2e-6,      # single precision throughout: measured ~3e-7 fwd and inv
}


def _x(rng, shape, dtype):
    return rng.standard_normal(shape).astype(dtype)


def _assert_is_the_oracle_to_rounding(name, got, ref, dtype):
    """The dtype of domainia's table (``test_single_precision.OUTPUT_DTYPE``), and the
    oracle's values to the working precision's rounding."""
    assert got.dtype == output_dtype(name, dtype)
    gap = np.max(np.abs(got.astype(np.complex128) - ref.astype(np.complex128)))
    assert gap <= ORACLE_RTOL[np.dtype(dtype)] * np.max(np.abs(ref.astype(np.complex128)))


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_family_is_the_oracle_to_rounding(grid, dtype, rng):
    Nf, Nt, dt = grid
    N = Nf * Nt
    x = _x(rng, N, dtype)
    X = _x(rng, (3, N), dtype)
    pairs = [
        ("forward_time", T.forward_time(x, Nf, Nt, dt), oracle.forward_time(x, Nf, Nt, dt)),
        ("forward", T.forward(np.fft.rfft(x), Nf, Nt, dt),
         oracle.forward(np.fft.rfft(x), Nf, Nt, dt)),
        ("forward_time_many", T.forward_time_many(X, Nf, Nt, dt),
         oracle.forward_time_many(X, Nf, Nt, dt)),
        ("forward_many", T.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt),
         oracle.forward_many(np.fft.rfft(X, axis=-1), Nf, Nt, dt)),
    ]
    for name, got, ref in pairs:
        _assert_is_the_oracle_to_rounding(name, got, ref, dtype)


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_inverse_family_is_the_oracle_to_rounding(grid, dtype, rng):
    Nf, Nt, dt = grid
    w = _x(rng, (Nf + 1, Nt), dtype)          # junk in inactive pixels on purpose
    W = _x(rng, (3, Nf + 1, Nt), dtype)
    pairs = [
        ("inverse", T.inverse(w, Nf, Nt, dt), oracle.inverse(w, Nf, Nt, dt)),
        ("inverse_time", T.inverse_time(w, Nf, Nt, dt), oracle.inverse_time(w, Nf, Nt, dt)),
        ("inverse_many", T.inverse_many(W, Nf, Nt, dt), oracle.inverse_many(W, Nf, Nt, dt)),
        ("inverse_time_many", T.inverse_time_many(W, Nf, Nt, dt),
         oracle.inverse_time_many(W, Nf, Nt, dt)),
    ]
    for name, got, ref in pairs:
        _assert_is_the_oracle_to_rounding(name, got, ref, dtype)


def test_inputs_are_not_mutated(rng):
    Nf, Nt, dt = 8, 16, 0.5
    w = rng.standard_normal((Nf + 1, Nt))
    w0 = w.copy()
    T.inverse(w, Nf, Nt, dt)
    assert np.array_equal(w, w0)
