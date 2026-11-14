"""Tests for :mod:`wdm.evolutionary`, each against an explicit dense reference."""

from __future__ import annotations

import numpy as np
import pytest

# The frozen wdm_dev oracle (tests/wilson/oracle/wdm_dev_ev) against domainia.wilson's dense
# references: its packed grid, atoms and time mask against build_W and forward_time, its band
# average and exact covariance against band_averaged_variance, build_C_time and build_C_wdm.
# The oracle packs the cells in band order; tests/wilson/test_evolutionary_kernels.py checks
# the kernels against it.
from wdm_dev_ev import evolutionary as ev
from domainia.wilson.grid import active_mask, cells
from domainia.wilson.matrix import build_W
from domainia.wilson.transform import forward_time

NF, NT, DT = 8, 16, 0.5
N = NF * NT
#: ``build_W``'s folded rows in the oracle's band order (``numpy.nonzero`` of the padded mask).
BAND = np.argsort(np.ravel_multi_index(cells(NF, NT), (NF + 1, NT)), axis=None)


@pytest.fixture
def rng():
    return np.random.default_rng(20261010)


def test_packed_grid_matches_the_dense_transform(rng):
    g = ev.PackedGrid(NF, NT, DT)
    Wa = build_W(NF, NT, DT)[BAND]
    assert g.size == N
    np.testing.assert_array_equal(g.mask, active_mask(NF, NT))
    x = rng.standard_normal((3, N))
    np.testing.assert_allclose(np.stack([g.forward(r) for r in x]), x @ Wa.T, atol=1e-12)
    np.testing.assert_allclose(g.forward(x[0]), forward_time(x[0], NF, NT, DT).ravel()[BAND], atol=0)


def test_atoms_are_the_rows_of_the_transform():
    g = ev.PackedGrid(NF, NT, DT)
    Wa = build_W(NF, NT, DT)[BAND]
    for p in range(g.size):
        np.testing.assert_allclose(g.atom(p), Wa[p], atol=1e-12)


def test_apply_mask_and_energy_fraction_match_the_dense_mask(rng):
    g = ev.PackedGrid(NF, NT, DT)
    Wa = build_W(NF, NT, DT)[BAND]
    w2 = rng.uniform(0.0, 1.0, N)
    Mm = Wa @ np.diag(w2) @ Wa.T / DT
    Y = rng.standard_normal((2, N))
    np.testing.assert_allclose(g.apply_mask(Y, w2), Y @ Mm.T, atol=1e-12)
    np.testing.assert_allclose(g.energy_fraction(w2), np.diag(Mm), atol=1e-12)


# ---------------------------------------------------------------------------
# General S(f, t): band average and exact covariance
# ---------------------------------------------------------------------------

from domainia.wilson.matrix import build_C_time, build_C_wdm  # noqa: E402
from domainia.wilson.noise import band_averaged_variance  # noqa: E402

T_REC = N * DT
F_NYQ = 0.5 / DT


def _S_stat(f):
    """A coloured stationary PSD, steep at both ends."""
    u = np.asarray(f, float) / F_NYQ
    return 1.0 + 0.3 / (u + 0.05) ** 2 + 5.0 * u**4


def _S_evo(f, t):
    """Slowly modulated, elementwise in (f, t)."""
    return _S_stat(f) * (1.0 + 0.4 * np.sin(2 * np.pi * np.asarray(t, float) / T_REC))


def test_band_average_reduces_to_the_stationary_band_average():
    got = ev.band_averaged_evolutionary_variance(NF, NT, DT, lambda f, t: _S_stat(f) + 0 * t)
    ref = band_averaged_variance(NF, NT, DT, _S_stat)
    np.testing.assert_allclose(got[cells(NF, NT)], ref, rtol=1e-13)


def test_band_average_samples_the_cell_centres():
    sep = ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo)
    stat = band_averaged_variance(NF, NT, DT, _S_stat)
    m, n = cells(NF, NT)
    tn = n * NF * DT                                    # each folded slot's cell time
    np.testing.assert_allclose(sep[m, n], stat * (1 + 0.4 * np.sin(2 * np.pi * tn / T_REC)), rtol=1e-13)
    sub = ev.band_averaged_evolutionary_variance(NF, NT, DT, _S_evo, bands=[0, 3, NF])
    np.testing.assert_array_equal(sub, sep[[0, 3, NF]])


def test_stationary_limit_of_the_exact_covariance():
    Sst = lambda f, t: _S_stat(f) + 0 * t  # noqa: E731
    np.testing.assert_allclose(ev.time_domain_covariance(N, DT, Sst), build_C_time(N, DT, _S_stat),
                               atol=1e-12)
    Lam = ev.exact_covariance(NF, NT, DT, Sst)
    np.testing.assert_allclose(Lam, build_C_wdm(NF, NT, DT, _S_stat)[np.ix_(BAND, BAND)], atol=1e-12)


def test_exact_covariance_is_the_transformed_time_covariance():
    Wa = build_W(NF, NT, DT)[BAND]
    ref = Wa @ ev.time_domain_covariance(N, DT, _S_evo) @ Wa.T
    np.testing.assert_allclose(ev.exact_covariance(NF, NT, DT, _S_evo), ref, atol=1e-12 * np.abs(ref).max())
