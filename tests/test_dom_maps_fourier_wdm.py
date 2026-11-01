"""FourierToWDM on synthetic records (replaces the GB-golden test_dom_maps_cells)."""

from __future__ import annotations

import numpy as np
import pytest
from dom_db_inputs import jax_or_skip, synthetic_record

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import FourierToWDM, TimeToWDM

NF, NT, DT = 16, 32, 0.25
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
BANDS = (3, 4, 5)


def _record() -> FourierDomain:
    return FourierDomain(N, DT, 0, N // 2 + 1, "rfft")


def _cells() -> CellSet:
    return CellSet.bands(GRID, BANDS)


def test_cells_are_the_record_scale_times_the_time_cells():
    x = synthetic_record(N, 3)
    got = FourierToWDM(_record(), WDMDomain(_cells())).apply(np.fft.rfft(x))
    want = TimeToWDM(TimeDomain(N, DT), WDMDomain(_cells())).apply(x)
    np.testing.assert_allclose(got, np.sqrt(N / (2 * DT)) * want, rtol=0, atol=1e-12 * np.max(np.abs(got)))


def test_fft_impl_matches_dense():
    z = np.fft.rfft(synthetic_record(N, 4))
    d = FourierToWDM(_record(), WDMDomain(_cells())).apply(z)
    f = FourierToWDM(_record(), WDMDomain(_cells()), "fft").apply(z)
    assert np.max(np.abs(f - d)) <= 1e-12 * np.max(np.abs(d))


def test_band_window_bins_feed_the_same_cells():
    z = np.fft.rfft(synthetic_record(N, 5))
    full = FourierToWDM(_record(), WDMDomain(_cells()))
    win = full.window
    sub = FourierToWDM(win, WDMDomain(_cells()))
    np.testing.assert_array_equal(sub.apply(z[win.k0:win.k0 + win.n]), full.apply(z))


def test_restrict_truncates_the_atoms_to_a_sub_window():
    full = FourierToWDM(_record(), WDMDomain(_cells()))
    win = full.window
    inner = FourierDomain(N, DT, win.k0 + 2, win.n - 4, "rfft")
    with pytest.raises(ValueError, match="restrict=True"):
        FourierToWDM(inner, WDMDomain(_cells()))
    m = FourierToWDM(inner, WDMDomain(_cells()), restrict=True)
    np.testing.assert_array_equal(m.C, full.C[:, 2:win.n - 2])


def test_adjoint_is_the_real_inner_product_adjoint():
    m = FourierToWDM(_record(), WDMDomain(_cells()))
    rng = np.random.default_rng(6)
    z = rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1)
    w = rng.standard_normal(_cells().size)
    lhs = float(np.dot(m.apply(z), w))
    rhs = float(np.real(np.vdot(m.adjoint(w), z)))
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_edge_bands_are_refused():
    with pytest.raises(ValueError, match="interior bands"):
        FourierToWDM(_record(), WDMDomain(CellSet.bands(GRID, (0, 1))))


@pytest.mark.jax
def test_jax_parity_with_the_map_as_a_jit_argument():
    jax, be = jax_or_skip()
    import jax.numpy as jnp

    m = FourierToWDM(_record(), WDMDomain(_cells()))
    z = np.fft.rfft(synthetic_record(N, 7))
    want = m.apply(z)
    got = jax.jit(lambda mm, zz: mm.apply(zz))(m.on(be), jnp.asarray(z))
    np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-12 * np.max(np.abs(want)))
