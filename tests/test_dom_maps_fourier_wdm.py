"""FourierToWDM on synthetic records (replaces the GB-golden test_dom_maps_cells)."""

from __future__ import annotations

import numpy as np
import pytest
from dom_db_inputs import DenseCov, jax_or_skip, synthetic_record

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import Compose, FourierCrop, FourierToWDM, FourierZeroPad, TimeToWDM

NF, NT, DT = 16, 32, 0.25
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
BANDS = (3, 4, 5)

#: The after() cases (7211bd5 test_dom_maps_cells::test_adjoint_and_the_record_scale): the
#: interior bands of an (8, 16) grid at dt = 0.5, and the same bands padded by 4 columns.
AG = wilson.Grid(8, 16, 0.5)
ACELLS = WDMDomain(CellSet.bands(AG, tuple(range(1, 8))))
PAD_GRID = wilson.Grid(8, 20, 0.5)


def _record() -> FourierDomain:
    return FourierDomain(N, DT, 0, N // 2 + 1, "rfft")


def _cells() -> CellSet:
    return CellSet.bands(GRID, BANDS)


def _crop_case(rng):
    """``(W, crop, Z, xr)``: the band-window map of ``ACELLS``, a crop of samples [16, 144) of
    a 160-sample record onto it, the record's bins ``Z`` (band-limited, no DC, no Nyquist)
    and the record ``xr``."""
    Z = np.zeros(81, complex)
    Z[8:57] = rng.standard_normal(49) + 1j * rng.standard_normal(49)    # no DC, no Nyquist
    xr = np.fft.irfft(Z, n=160)
    W = FourierToWDM(FourierDomain(AG.N, AG.dt, 0, 64, "rfft"), ACELLS)    # the band windows
    crop = FourierCrop(FourierDomain(160, AG.dt, 0, 81, "rfft"), W.src, 16)
    return W, crop, Z, xr


def _pad_case(rng):
    """``(Wp, zp, z, x)``: a band-limited 128-sample record ``x`` with no DC or Nyquist
    content, its store bins ``z = rfft(x)[1:64]``, their zero pad to the 160-sample record of
    ``PAD_GRID`` and the band-window map of that record's cells with ``n < 16`` (GB pad)."""
    X = np.zeros(65, complex)
    X[1:64] = rng.standard_normal(63) + 1j * rng.standard_normal(63)
    x = np.fft.irfft(X, n=AG.N)
    store = FourierDomain(AG.N, AG.dt, 1, 63, "rfft")
    kept = WDMDomain(CellSet.bands(PAD_GRID, tuple(range(1, 8))).drop_columns(0, 4))
    win = FourierToWDM(FourierDomain(PAD_GRID.N, AG.dt, 0, PAD_GRID.N // 2 + 1, "rfft"), kept).window
    Wp = FourierToWDM(win, kept)
    return Wp, FourierZeroPad(store, win), np.fft.rfft(x)[1:64], x


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


@pytest.mark.parametrize("impl", ["dense", "fft"])
def test_cut_cells_are_the_full_cells_restricted(impl):
    z = np.fft.rfft(synthetic_record(N, 8))
    full = _cells()
    cut = full.drop_columns(3, 2)
    got = FourierToWDM(_record(), WDMDomain(cut), impl).apply(z)
    want = FourierToWDM(_record(), WDMDomain(full), impl).apply(z)[full.index_of(cut)]
    assert got.shape == (cut.size,)
    assert np.max(np.abs(got - want)) <= 1e-14 * np.max(np.abs(want))


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
    with pytest.raises(ValueError, match="impl='dense' only"):
        FourierToWDM(inner, WDMDomain(_cells()), "fft", restrict=True)


def test_adjoint_is_the_real_inner_product_adjoint():
    m = FourierToWDM(_record(), WDMDomain(_cells()))
    rng = np.random.default_rng(6)
    z = rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1)
    w = rng.standard_normal(_cells().size)
    lhs = float(np.dot(m.apply(z), w))
    rhs = float(np.real(np.vdot(m.adjoint(w), z)))
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_push_of_white_record_bins_is_the_identity():
    """For ``z = rfft(x)`` of unit white noise ``E[z z^H] = N I`` on the interior bins, and the
    cells ``sqrt(N / (2 dt)) W x`` have covariance ``(N / (2 dt)) dt I = (N / 2) I``: the push
    ``1/2 Re(C M' C^H)`` takes ``M' = 2 I`` to ``I``."""
    m = FourierToWDM(_record(), WDMDomain(_cells()))
    P = m.push(DenseCov(2 * np.eye(m.src.n))).dense()
    assert P.shape == (_cells().size, _cells().size)
    assert np.max(np.abs(P - np.eye(_cells().size))) <= 1e-12


def test_after_a_crop_is_the_composition_and_the_cropped_transform():
    """7211bd5 test_dom_maps_cells::test_adjoint_and_the_record_scale, its golden-free half."""
    W, crop, Z, xr = _crop_case(np.random.default_rng(0))
    # after(FourierCrop): the composite matrix is the composition, and on a band-limited
    # 160-sample record it is the WDM transform of the cropped samples [16, 144)
    got = W.after(crop).apply(Z)
    assert np.max(np.abs(got - Compose(W, crop).apply(Z))) <= 1e-12 * np.max(np.abs(got))
    ref = np.sqrt(AG.N / (2 * AG.dt)) * TimeToWDM(TimeDomain(AG.N, AG.dt), ACELLS).apply(xr[16:144])
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))
    # a composite's src is the outer record, not the band window: it cannot be chained
    crop2 = FourierCrop(FourierDomain(192, AG.dt, 0, 97, "rfft"), crop.src, 8)
    with pytest.raises(ValueError, match="the band window"):
        W.after(crop).after(crop2)
    # the band window lies on the cells' grid record (N = 128), never on a composite's
    # outer record: a composite whose src has the window's bins [0, 64) is still refused
    Wc = W.after(FourierCrop(FourierDomain(160, AG.dt, 0, 64, "rfft"), W.src, 16))
    assert Wc.window == W.window == W.src and Wc.src != Wc.window
    with pytest.raises(ValueError, match="the band window"):
        Wc.after(FourierCrop(FourierDomain(192, AG.dt, 0, 64, "rfft"), Wc.src, 8))


def test_after_a_zero_pad_is_the_padded_record_transform():
    """With no DC or Nyquist bin in the store, ``A z + B conj z`` is the zero-padded record's
    rfft, and ``FourierZeroPad.scale = sqrt(N / N')`` keeps the cells in the store record's
    units: ``W.after(zp)`` is ``sqrt(N / (2 dt))`` times the padded record's WDM cells."""
    Wp, zp, z, x = _pad_case(np.random.default_rng(1))
    M = Wp.after(zp)
    assert M.src == zp.src and M.window == Wp.window == Wp.src
    got = M.apply(z)
    assert np.max(np.abs(got - Compose(Wp, zp).apply(z))) <= 1e-12 * np.max(np.abs(got))
    xpad = np.concatenate([x, np.zeros(PAD_GRID.N - AG.N)])
    ref = np.sqrt(AG.N / (2 * AG.dt)) * TimeToWDM(TimeDomain(PAD_GRID.N, AG.dt), Wp.dst).apply(xpad)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_edge_bands_are_refused():
    with pytest.raises(ValueError, match="interior bands"):
        FourierToWDM(_record(), WDMDomain(CellSet.bands(GRID, (0, 1))))


@pytest.mark.jax
def test_jax_parity_with_the_map_as_a_jit_argument():
    jax, be = jax_or_skip()
    import jax.numpy as jnp

    z = np.fft.rfft(synthetic_record(N, 7))
    W, crop, Z, _ = _crop_case(np.random.default_rng(0))
    Wp, zp, zs, _ = _pad_case(np.random.default_rng(1))
    cases = ((FourierToWDM(_record(), WDMDomain(_cells())), z),
             (FourierToWDM(_record(), WDMDomain(_cells()), "fft"), z),
             (W.after(crop), Z),
             (Wp.after(zp), zs))
    for m, v in cases:
        want = m.apply(v)
        got = jax.jit(lambda mm, vv: mm.apply(vv))(m.on(be), jnp.asarray(v))
        assert isinstance(got, jax.Array)
        np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-12 * np.max(np.abs(want)))
