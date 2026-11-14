"""Tests for wdm.matrix."""

from __future__ import annotations

import numpy as np

from domainia.wilson.matrix import (
    build_C_time,
    build_C_wdm,
    build_W,
    real_to_rfft,
    rfft_to_real,
)
from domainia.wilson.transform import forward_time


def test_W_matches_FFT_transform(simple_grid):
    """W @ x == forward(rfft(x)) to machine precision, rows in folded flat order."""
    g = simple_grid
    W = build_W(g.Nf, g.Nt, g.dt)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(g.N)
    w_matrix = (W @ x).reshape(g.Nf, g.Nt)
    w_fft = forward_time(x, g.Nf, g.Nt, g.dt)
    assert np.allclose(w_matrix, w_fft, atol=1e-13)


def test_W_orthogonal(simple_grid):
    """W @ W.T = dt I (orthogonal rows)."""
    g = simple_grid
    W = build_W(g.Nf, g.Nt, g.dt)
    assert W.shape == (g.N, g.N)
    err = np.max(np.abs(W @ W.T - g.dt * np.eye(g.N)))
    assert err < 1e-12


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
