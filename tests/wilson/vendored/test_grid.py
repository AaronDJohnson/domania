"""Tests for wdm.grid."""

from __future__ import annotations

import numpy as np
import pytest

from domainia.wilson.grid import (
    band_frequencies,
    delta_F,
    delta_T,
    n_samples,
    time_pixel_centers,
    validate_grid,
)


def test_band_frequencies_endpoints(simple_grid):
    """freqs[0] == 0 and freqs[Nf] == Nyquist."""
    g = simple_grid
    freqs = band_frequencies(g.Nf, g.dt)
    assert freqs.shape == (g.Nf + 1,)
    assert freqs[0] == 0.0
    assert np.isclose(freqs[g.Nf], g.f_nyq)


def test_time_pixel_centers_shape(simple_grid):
    """time_pixel_centers returns Nt values, increasing by dT."""
    g = simple_grid
    t = time_pixel_centers(g.Nf, g.Nt, g.dt)
    assert t.shape == (g.Nt,)
    diffs = np.diff(t)
    assert np.allclose(diffs, g.dT)


def test_delta_relations(simple_grid):
    """dT * dF = 1/2 (critical sampling)."""
    g = simple_grid
    assert np.isclose(delta_T(g.Nf, g.dt) * delta_F(g.Nf, g.dt), 0.5)


def test_n_samples(simple_grid):
    g = simple_grid
    assert n_samples(g.Nf, g.Nt) == g.Nf * g.Nt


@pytest.mark.parametrize("bad", [-2, 0, 3, 7, 1])
def test_validate_grid_rejects_odd_or_nonpositive(bad):
    """Reject odd or non-positive Nf/Nt."""
    with pytest.raises(ValueError):
        validate_grid(bad, 16)
    with pytest.raises(ValueError):
        validate_grid(16, bad)


def test_validate_grid_accepts_valid():
    """Even positive ints pass validate_grid."""
    validate_grid(8, 16)
    validate_grid(2, 2)
