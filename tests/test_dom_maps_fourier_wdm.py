"""FourierToWDM on synthetic records (replaces the GB-golden test_dom_maps_cells)."""

from __future__ import annotations

from dataclasses import replace

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
SIGMA0 = 2.5

#: The after() cases (ported from 7211bd5's test_dom_maps_cells): the interior bands of an
#: (8, 16) grid at dt = 0.5, and the same bands padded by 4 columns.
AG = wilson.Grid(8, 16, 0.5)
ACELLS = WDMDomain(CellSet.bands(AG, tuple(range(1, 8))))
PAD_GRID = wilson.Grid(8, 20, 0.5)


def _record() -> FourierDomain:
    return FourierDomain(N, DT, 0, N // 2 + 1, "rfft")


def _cells() -> CellSet:
    return CellSet.bands(GRID, BANDS)


def _units_record(units: str) -> FourierDomain:
    """The whole record's bins in ``units`` (``sigma0 = SIGMA0`` for ``"sigma0"``)."""
    return FourierDomain(N, DT, 0, N // 2 + 1, units, sigma0=SIGMA0 if units == "sigma0" else None)


def _in_units(R: np.ndarray, units: str) -> np.ndarray:
    """``z`` in ``units`` from ``R = numpy.fft.rfft(x)`` (the Fourier units table)."""
    return {"rfft": R, "dt_rfft": DT * R, "sigma0": R / SIGMA0}[units]


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


@pytest.mark.parametrize("impl", ["dense", "fft"])
@pytest.mark.parametrize("units", ["rfft", "dt_rfft", "sigma0"])
def test_cells_are_the_time_cells_in_every_unit(units, impl):
    """The cells are the Wilson coefficients of the record, whatever units
    the bins are in."""
    x = synthetic_record(N, 3)
    got = FourierToWDM(_units_record(units), WDMDomain(_cells()), impl).apply(
        _in_units(np.fft.rfft(x), units))
    want = TimeToWDM(TimeDomain(N, DT), WDMDomain(_cells())).apply(x)
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))


@pytest.mark.parametrize("units", ["rfft", "dt_rfft", "sigma0"])
def test_the_cell_matrix_is_kappa_times_the_atoms(units):
    """``C = kappa U``, ``kappa = src.rfft_scale sqrt(2 dt / N)``, ``U`` the unit-norm atoms
    on the band windows; ``restrict=True`` truncates them to ``src``."""
    src = _units_record(units)
    full = FourierToWDM(src, WDMDomain(_cells()))
    win = full.window
    atoms = wilson.transform.band_atoms(np.array(BANDS), NF, NT, DT)
    U = np.zeros((len(BANDS) * NT, win.n), complex)
    for b in range(len(BANDS)):
        U[b * NT:(b + 1) * NT, (NT // 2) * b + np.arange(NT)] = atoms[b]
    kappa = src.rfft_scale * np.sqrt(2 * DT / N)
    assert np.max(np.abs(full.C - kappa * U)) <= 1e-15 * kappa * np.max(np.abs(U))
    inner = replace(win, k0=win.k0 + 2, n=win.n - 4)
    m = FourierToWDM(inner, WDMDomain(_cells()), restrict=True)
    assert np.max(np.abs(m.C - kappa * U[:, 2:win.n - 2])) <= 1e-15 * kappa * np.max(np.abs(U))


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


def test_the_band_window_keeps_the_units_and_sigma0():
    """``window`` is ``dataclasses.replace`` of ``src``: the bins of the cells' grid record in
    ``src``'s units, ``sigma0`` included."""
    src = FourierDomain(N, DT, 0, N // 2 + 1, "sigma0", sigma0=2.5)
    win = FourierToWDM(src, WDMDomain(_cells())).window
    assert (win.N, win.dt, win.units, win.sigma0) == (N, DT, "sigma0", 2.5)
    assert win.k0 > 0 and win.k0 + win.n < N // 2 + 1
    assert FourierToWDM(win, WDMDomain(_cells())).window == win


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


@pytest.mark.parametrize("bands", [tuple(range(1, NF)), tuple(range(2, NF - 1))])
def test_push_of_white_record_bins_is_the_white_cell_covariance(bands):
    """For ``z = rfft(x)`` of white noise of variance ``sigma2``, ``E[z z^H] = N sigma2 I`` on the
    bins, and the cells ``W x`` have covariance ``W sigma2 I W^T = sigma2 dt I``: the push
    ``1/2 Re(C M' C^H)`` takes ``M' = N sigma2 I`` to ``sigma2 dt I``."""
    sigma2 = 1.7
    cells = CellSet.bands(GRID, bands)
    m = FourierToWDM(_record(), WDMDomain(cells))
    P = m.push(DenseCov(N * sigma2 * np.eye(m.src.n))).dense()
    assert P.shape == (cells.size, cells.size)
    assert np.max(np.abs(P - sigma2 * DT * np.eye(cells.size))) <= 1e-13 * sigma2 * DT


def test_after_a_crop_is_the_composition_and_the_cropped_transform():
    """Ported from 7211bd5's test_dom_maps_cells (its golden-free half)."""
    W, crop, Z, xr = _crop_case(np.random.default_rng(0))
    # after(FourierCrop): the composite matrix is the composition, and on a band-limited
    # 160-sample record it is the WDM transform of the cropped samples [16, 144)
    got = W.after(crop).apply(Z)
    assert np.max(np.abs(got - Compose(W, crop).apply(Z))) <= 1e-12 * np.max(np.abs(got))
    ref = TimeToWDM(TimeDomain(AG.N, AG.dt), ACELLS).apply(xr[16:144])
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


def test_after_a_crop_carries_the_units_factor():
    """``after`` applies kappa's ``src.rfft_scale``: on ``dt_rfft`` bins (``u = 1 / dt = 2``) the
    composite is still the composition with the crop (which has no scale), and its cells are
    the cropped record's Wilson coefficients (the crop geometry of the test above)."""
    rng = np.random.default_rng(0)
    Z = np.zeros(81, complex)
    Z[8:57] = rng.standard_normal(49) + 1j * rng.standard_normal(49)    # no DC, no Nyquist
    xr = np.fft.irfft(Z, n=160)
    W = FourierToWDM(FourierDomain(AG.N, AG.dt, 0, 64, "dt_rfft"), ACELLS)
    crop = FourierCrop(FourierDomain(160, AG.dt, 0, 81, "dt_rfft"), W.src, 16)
    z = AG.dt * Z                                                       # dt_rfft units
    got = W.after(crop).apply(z)
    want = Compose(W, crop).apply(z)
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))
    ref = TimeToWDM(TimeDomain(AG.N, AG.dt), ACELLS).apply(xr[16:144])
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_after_a_zero_pad_is_the_padded_record_transform():
    """``A z + B conj z`` is the zero-padded record's rfft, so ``W.after(zp)`` is the padded
    record's Wilson coefficients on the cells (here the GB pad: no DC or Nyquist bin in the
    store, cells with ``n < 16``)."""
    Wp, zp, z, x = _pad_case(np.random.default_rng(1))
    M = Wp.after(zp)
    assert M.src == zp.src and M.window == Wp.window == Wp.src
    got = M.apply(z)
    assert np.max(np.abs(got - Compose(Wp, zp).apply(z))) <= 1e-12 * np.max(np.abs(got))
    xpad = np.concatenate([x, np.zeros(PAD_GRID.N - AG.N)])
    ref = TimeToWDM(TimeDomain(PAD_GRID.N, AG.dt), Wp.dst).apply(xpad)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("units", ["rfft", "dt_rfft", "sigma0"])
def test_pad_then_wdm_is_the_padded_record_transform(units):
    """The headline units test: every bin of a white 128-sample record
    (non-zero DC and Nyquist), in any units, zero padded to 160 samples and analysed on the
    padded grid's cells, is the padded record's Wilson coefficients, through ``after`` and
    through ``Compose``."""
    x = synthetic_record(AG.N, 11)
    store = FourierDomain(AG.N, AG.dt, 0, AG.N // 2 + 1, units,
                          sigma0=SIGMA0 if units == "sigma0" else None)
    R = np.fft.rfft(x)
    z = {"rfft": R, "dt_rfft": AG.dt * R, "sigma0": R / SIGMA0}[units]
    cells = WDMDomain(CellSet.bands(PAD_GRID, tuple(range(1, 8))))
    win = FourierToWDM(replace(store, N=PAD_GRID.N, n=PAD_GRID.N // 2 + 1), cells).window
    W, pad = FourierToWDM(win, cells), FourierZeroPad(store, win)
    want = TimeToWDM(TimeDomain(PAD_GRID.N, AG.dt), cells).apply(
        np.concatenate([x, np.zeros(PAD_GRID.N - AG.N)]))
    for got in (W.after(pad).apply(z), Compose(W, pad).apply(z)):
        assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))


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
