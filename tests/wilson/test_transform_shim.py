"""The transform against the frozen oracle vendored_2f7136c.

It matches to rounding on random grids, dt != 1, one record and a stack of three.
Bit-identical up to v0.11.0; the Hz window of v0.12.0 (Paper I eq:Meyer-def: phi
flat value 1/sqrt(Delta F), kappa = sqrt(2)/Nf, lambda = 1/(sqrt(2) dt), against the
oracle's omega-normalised 1/sqrt(2 pi Delta F), 2 sqrt(pi)/Nf, sqrt(pi)/dt) is the
same product rounded differently, so the gate is ORACLE_RTOL * max|ref|.  The oracle is
padded: its forward output is gathered into the folded slots, and its inverse takes our
folded input unfolded.
"""

from __future__ import annotations

import numpy as np
import pytest
import vendored_2f7136c.transform as oracle

import domania.wilson.transform as T
from domania.wilson.grid import cells, unfold

GRIDS = [(2, 2, 1.0), (4, 8, 1.0), (8, 16, 0.5), (8, 32, 1.7), (16, 32, 2.0**-10),
         (32, 64, 10.0), (64, 256, 2.0**-10), (6, 10, 0.3)]

#: max|got - ref| over max|ref| against the omega-era oracle.
ORACLE_RTOL = 1e-14     # Hz window (v0.12.0): measured 4.2e-16 fwd, 5.4e-16 inv


def _assert_is_the_oracle_to_rounding(got, ref):
    """The oracle's dtype and shape, and its values to rounding."""
    assert got.dtype == ref.dtype and got.shape == ref.shape
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
        (T.forward_time(X, Nf, Nt, dt), np.stack([oracle.forward_time(r, Nf, Nt, dt) for r in X])),
        (T.forward(np.fft.rfft(X), Nf, Nt, dt),
         np.stack([oracle.forward(np.fft.rfft(r), Nf, Nt, dt) for r in X])),
    ]
    m, n = cells(Nf, Nt)
    for got, ref in pairs:
        _assert_is_the_oracle_to_rounding(got, ref[..., m, n])


@pytest.mark.parametrize("grid", GRIDS)
def test_inverse_family_is_the_oracle_to_rounding(grid, rng):
    Nf, Nt, dt = grid
    w = rng.standard_normal((Nf, Nt))
    W = rng.standard_normal((3, Nf, Nt))
    p, P = unfold(w, Nf, Nt), unfold(W, Nf, Nt)
    J = P + 7.0 * ~unfold(np.ones((Nf, Nt), bool), Nf, Nt)    # junk in the inactive cells
    pairs = [
        (T.inverse(w, Nf, Nt, dt), oracle.inverse(p, Nf, Nt, dt)),
        (T.inverse_time(w, Nf, Nt, dt), oracle.inverse_time(p, Nf, Nt, dt)),
        (T.inverse(W, Nf, Nt, dt), np.stack([oracle.inverse(r, Nf, Nt, dt) for r in P])),
        (T.inverse_time(W, Nf, Nt, dt), np.stack([oracle.inverse_time(r, Nf, Nt, dt) for r in P])),
        (T._inverse(J, Nf, Nt, dt), np.stack([oracle.inverse(r, Nf, Nt, dt) for r in P])),
    ]
    for got, ref in pairs:
        _assert_is_the_oracle_to_rounding(got, ref)


def test_leading_axes_are_the_loop_over_the_slices(rng):
    """Each kernel takes any leading axes: a (3, 2, ...) stack gives the loop over its
    slices, and an input one entry short on its last axis is refused.  The inverses refuse a
    padded and a flat grid too, naming the fix."""
    Nf, Nt, dt = 8, 16, 0.5
    x = rng.standard_normal((3, 2, Nf * Nt))
    w = rng.standard_normal((3, 2, Nf, Nt))
    for f, a in ((T.forward, np.fft.rfft(x)), (T.forward_time, x), (T.inverse, w), (T.inverse_time, w)):
        want = np.stack([np.stack([f(s, Nf, Nt, dt) for s in row]) for row in a])
        _assert_is_the_oracle_to_rounding(f(a, Nf, Nt, dt), want)
        with pytest.raises(ValueError, match="must have shape"):
            f(a[..., 1:], Nf, Nt, dt)
    for f in (T.inverse, T.inverse_time):
        with pytest.raises(ValueError, match=r"got \(3, 2, 9, 16\); fold a padded .* with "
                                             r"wilson\.grid\.fold first$"):
            f(unfold(w, Nf, Nt), Nf, Nt, dt)
        with pytest.raises(ValueError, match=r"got \(3, 2, 128\); reshape a flat .* with "
                                             r"\.reshape\(\.\.\., 8, 16\) first$"):
            f(x, Nf, Nt, dt)


def test_inputs_are_not_mutated(rng):
    Nf, Nt, dt = 8, 16, 0.5
    w = rng.standard_normal((Nf, Nt))
    w0 = w.copy()
    T.inverse(w, Nf, Nt, dt)
    assert np.array_equal(w, w0)
