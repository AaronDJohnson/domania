"""Tests for wdm.matrix."""

from __future__ import annotations

import numpy as np

from domania.wilson.matrix import build_C_time, build_C_wdm, real_to_rfft, rfft_to_real


def test_C_time_symmetric(simple_grid, lisa_psd):
    """C_time is symmetric."""
    g = simple_grid
    C = build_C_time(g.N, g.dt, lisa_psd)
    assert np.max(np.abs(C - C.T)) < 1e-12 * np.max(np.abs(C))


def test_C_time_toeplitz(simple_grid, flat_psd):
    """C_time has Toeplitz (constant-along-diagonal) structure for stationary noise."""
    g = simple_grid
    C = build_C_time(g.N, g.dt, flat_psd)
    # Top row determines the matrix (modulo symmetry); check a few diagonals.
    for offset in [0, 1, 2, g.N // 3]:
        diag = np.diag(C, k=offset)
        # Allow tiny numerical jitter from irfft.
        assert np.max(np.abs(diag - diag[0])) < 1e-12 * np.max(np.abs(C))


def test_C_wdm_diag_matches_whittle(simple_grid, flat_psd):
    """The diagonal of C_wdm equals the Whittle variance for flat PSD."""
    g = simple_grid
    C = build_C_wdm(g.Nf, g.Nt, g.dt, flat_psd)
    assert C.shape == (g.N, g.N)
    diag = np.diag(C).reshape(g.Nf, g.Nt)
    # Average over the interior bands (rows 1..Nf-1); should equal S_n / 2.
    expected = 1e-3 / 2
    interior = diag[1 : g.Nf]
    assert np.isclose(np.mean(interior), expected, rtol=0.01)


def test_rfft_real_roundtrip(simple_grid):
    """rfft_to_real and real_to_rfft are mutual inverses."""
    g = simple_grid
    rng = np.random.default_rng(0)
    x = rng.standard_normal(g.N)
    x_rfft = np.fft.rfft(x)
    x_real = rfft_to_real(x_rfft, g.N)
    assert x_real.shape == (g.N,)
    x_rfft_back = real_to_rfft(x_real, g.N)
    assert np.allclose(x_rfft_back, x_rfft, atol=1e-15)
