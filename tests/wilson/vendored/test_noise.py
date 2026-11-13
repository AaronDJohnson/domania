"""Tests for wdm.noise."""

from __future__ import annotations

import numpy as np

from domainia.wilson.noise import band_averaged_variance, empirical_variance


def test_empirical_variance_matches_the_band_average_for_a_flat_psd(simple_grid, flat_psd):
    """MC estimate of variance matches the band-averaged variance for flat PSD."""
    g = simple_grid
    var_mc = empirical_variance(g.Nf, g.Nt, g.dt, flat_psd, n_realizations=2000, seed=0)
    var_pred = band_averaged_variance(g.Nf, g.Nt, g.dt, flat_psd)
    # Compare interior bands; MC noise scales as 1/sqrt(n_realizations).
    rel_err = np.abs(var_mc[1 : g.Nf] - var_pred[1 : g.Nf]) / var_pred[1 : g.Nf]
    assert np.median(rel_err) < 0.05
