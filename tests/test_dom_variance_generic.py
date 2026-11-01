"""BandAveragedVariance and MidpointVariance of a GenericPSD (replaces the golden
test_dom_variance), and DiagonalCovariance over them."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson
from domainia.covariance import DiagonalCovariance
from domainia.domains import CellSet
from domainia.spectra import GenericPSD
from domainia.variance import BandAveragedVariance, MidpointVariance

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
T = N * DT
GRID = wilson.Grid(NF, NT, DT)
CELLS = CellSet.all_active(GRID)
CUT = CELLS.drop_columns(3, 1)
PSD = GenericPSD(lambda f, t, nu: (1.0 + np.asarray(f, float) ** 2) ** nu[0]
                 * (1.0 + nu[1] * np.asarray(t, float) / T))
NU = np.array([-0.7, 0.3])


def test_band_average_is_the_kernel_band_variance_at_the_cells():
    full = wilson.noise.evolutionary_band_variance(NF, NT, DT, PSD.bind(NU))
    got = BandAveragedVariance(PSD, CELLS)(NU)
    np.testing.assert_array_equal(got, full[np.asarray(CELLS.m), np.asarray(CELLS.n)])


def test_midpoint_is_half_the_psd_at_the_band_centre_and_cell_time():
    m, n = np.asarray(CELLS.m), np.asarray(CELLS.n)
    f = wilson.band_frequencies(NF, DT)[m]
    t = n * wilson.delta_T(NF, DT)
    np.testing.assert_array_equal(MidpointVariance(PSD, CELLS)(NU), 0.5 * PSD(f, t, NU))


@pytest.mark.parametrize("cls", [BandAveragedVariance, MidpointVariance])
def test_cut_cells_are_the_full_vector_restricted(cls):
    full = cls(PSD, CELLS)(NU)
    np.testing.assert_array_equal(cls(PSD, CUT)(NU), full[CELLS.index_of(CUT)])


@pytest.mark.parametrize("cls", [BandAveragedVariance, MidpointVariance])
def test_only_generic_psds_are_accepted(cls):
    with pytest.raises(TypeError, match="GenericPSD"):
        cls(object(), CELLS)


def test_diagonal_covariance_of_a_variance_carries_its_cells_and_factors():
    cov = DiagonalCovariance(BandAveragedVariance(PSD, CUT), CUT.size)
    assert cov.cells == CUT
    v = BandAveragedVariance(PSD, CUT)(NU)
    F = cov.factor(NU)
    b = np.random.default_rng(1).standard_normal(CUT.size)
    assert float(F.logdet) == pytest.approx(float(np.sum(np.log(v))), rel=1e-13)
    assert float(F.quad(b)) == pytest.approx(float(np.sum(b * b / v)), rel=1e-13)


def test_callable_and_array_variances_carry_no_cells():
    assert DiagonalCovariance(np.ones(3), 3).cells is None
    assert DiagonalCovariance(lambda nu, be: be.asarray(np.ones(3)), 3).cells is None
