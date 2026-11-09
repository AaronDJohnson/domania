"""Domains and cell sets."""

from __future__ import annotations

import copy
import inspect
import pickle

import numpy as np
import pytest

from domainia import wilson
from domainia.domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain

SIN = wilson.Grid(64, 256, 2.0**-10)


def test_time_domain():
    d = TimeDomain(16384, 2.0**-10)
    assert (d.N, d.dt, d.t0, d.size) == (16384, 2.0**-10, 0.0, 16384)
    assert np.array_equal(d.times(), np.arange(16384) * 2.0**-10)
    assert d == TimeDomain(16384, 2.0**-10) and hash(d) == hash(TimeDomain(16384, 2.0**-10))
    with pytest.raises(ValueError):
        TimeDomain(0, 1.0)
    with pytest.raises(ValueError):
        TimeDomain(8, -1.0)
    with pytest.raises(TypeError, match="N must be an integer, got float"):
        TimeDomain(8.0, 1.0)


def test_fourier_domain():
    d = FourierDomain(64, 0.5, 3, 10, "sigma0", sigma0=2.0)
    assert np.array_equal(d.bins(), np.arange(3, 13))
    assert np.array_equal(d.frequencies(), np.arange(3, 13) / 32.0)
    assert d.size == 10
    FourierDomain(64, 0.5, 23, 10, "rfft")                              # last bin is 32 = N/2
    with pytest.raises(ValueError, match="exceed"):
        FourierDomain(64, 0.5, 24, 10, "rfft")
    with pytest.raises(ValueError, match="units"):
        FourierDomain(64, 0.5, 0, 10, "hz")


def test_fourier_units_carry_sigma0_and_the_rfft_scale():
    """``rfft_scale`` is ``u`` with ``numpy.fft.rfft(x) = u z``, and
    ``sigma0`` is part of the domain's identity."""
    assert FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=2.0).rfft_scale == 2.0
    assert FourierDomain(64, 0.5, 0, 33, "rfft").rfft_scale == 1.0
    assert FourierDomain(64, 0.5, 0, 33, "dt_rfft").rfft_scale == 1 / 0.5
    assert FourierDomain.full(64, 0.5).rfft_scale == 1 / 0.5
    x = np.random.default_rng(0).standard_normal(64)
    R = np.fft.rfft(x)
    for units, z, kw in (("rfft", R, {}), ("dt_rfft", 0.5 * R, {}),
                         ("sigma0", R / 2.0, {"sigma0": 2.0})):
        d = FourierDomain(64, 0.5, 0, 33, units, **kw)
        np.testing.assert_allclose(d.rfft_scale * z, R, rtol=1e-15, atol=0)
    for bad in (None, 0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="sigma0"):
            FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=bad)
    for units in ("rfft", "dt_rfft"):
        with pytest.raises(ValueError, match="sigma0"):
            FourierDomain(64, 0.5, 0, 33, units, sigma0=2.0)
    with pytest.raises(TypeError, match="positional argument"):
        FourierDomain(64, 0.5, 0, 33, "sigma0", 2.0)                    # sigma0 is keyword-only
    a = FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=2)
    assert type(a.sigma0) is float and a.sigma0 == 2.0
    b = FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=2.0)
    assert a == b and hash(a) == hash(b)
    assert a != FourierDomain(64, 0.5, 0, 33, "sigma0", sigma0=3.0)
    assert FourierDomain(64, 0.5, 0, 33, "rfft").sigma0 is None


def test_all_active_is_the_packed_order():
    cells = CellSet.all_active(wilson.Grid(8, 16, 1.0))
    m, n = wilson.grid.packed_cells(8, 16)
    assert np.array_equal(cells.m_array, m) and np.array_equal(cells.n_array, n)
    assert np.array_equal(cells.packed_positions(), np.arange(128))
    assert cells.keep_mask().all()


def test_sinusoid_counts():
    """The sinusoid grid: 16384 cells, cut (6, 5) keeps 15681, edge-exact (8, 7) has 959
    rows over a 15425-cell interior."""
    cells = CellSet.all_active(SIN)
    assert cells.size == 16384
    cut = cells.drop_columns(6, 5)
    assert cut.size == 15681
    edge = cells.columns(8, 7)
    interior = cells.complement(edge)
    assert (edge.size, interior.size) == (959, 15425)
    assert set(edge.n) == set(wilson.grid.edge_columns(256, 8, 7))
    assert cells.complement(interior) == edge
    pad = CellSet.all_active(wilson.Grid(64, 272, 2.0**-10))
    assert pad.size == 17408


def test_gb_counts_from_the_prepared_grids():
    """The galactic-binary grids: 3 bands x NT cells per channel; cut (5, 4) and 27 edge rows."""
    for N, NT, bands in ((1576800, 240, (65, 66, 67)), (3153600, 438, (71, 72, 73)),
                         (6307200, 900, (69, 70, 71))):
        grid = wilson.Grid(N // NT, NT, 5.0)
        cells = CellSet.bands(grid, bands)
        assert cells.size == 3 * NT
        assert cells.drop_columns(5, 4).size == 3 * NT - 27
        assert cells.columns(5, 4).size == 27
        assert cells.band_sizes() == tuple((b, NT) for b in bands)
        assert np.array_equal(cells.m_array, np.repeat(bands, NT))
        assert np.array_equal(cells.n_array, np.tile(np.arange(NT), 3))


def test_positions_and_masks_agree():
    grid = wilson.Grid(8, 16, 0.5)
    cells = CellSet.all_active(grid)
    sub = cells.drop_columns(3, 1)
    pos = sub.packed_positions()
    m, n = wilson.grid.packed_cells(8, 16)
    assert np.array_equal(m[pos], sub.m_array) and np.array_equal(n[pos], sub.n_array)
    assert np.array_equal(np.flatnonzero(sub.keep_mask()), pos)
    assert CellSet.from_mask(grid, sub.keep_mask()) == sub
    assert np.array_equal(cells.index_of(sub), pos)
    with pytest.raises(ValueError, match="subset"):
        sub.index_of(cells)


def test_dc_and_nyquist_bands_have_half_the_cells():
    cells = CellSet.bands(wilson.Grid(8, 16, 1.0), (0, 1, 8))
    assert cells.band_sizes() == ((0, 8), (1, 16), (8, 8))
    assert set(cells.n_array[cells.m_array == 0]) == set(range(0, 16, 2))


def test_validation():
    g = wilson.Grid(8, 16, 1.0)
    with pytest.raises(ValueError, match="not active"):
        CellSet(g, (0,), (1,))                                          # DC, odd n
    with pytest.raises(ValueError, match="band order"):
        CellSet(g, (2, 1), (0, 0))
    with pytest.raises(ValueError, match="band order"):
        CellSet(g, (1, 1), (3, 3))                                      # repeat
    with pytest.raises(ValueError, match="outside"):
        CellSet(g, (9,), (0,))
    with pytest.raises(ValueError, match="ascending"):
        CellSet.bands(g, (3, 2))
    with pytest.raises(ValueError, match="exceeds"):
        CellSet.all_active(g).drop_columns(10, 7)
    # non-integral or non-finite indices are refused, never truncated (2.9999999 -> 2)
    for m_bad, n_bad in (((2.9999999,), (0,)), ((1,), (0.5,)), ((np.nan,), (0,))):
        with pytest.raises(ValueError, match="integer"):
            CellSet(g, m_bad, n_bad)
    # a boolean index array is a wrong kind
    with pytest.raises(TypeError, match="^m must be an integer index array, got dtype bool$"):
        CellSet(g, (True,), (0,))
    assert CellSet(g, np.array([1.0, 2.0]), np.array([3.0, 3.0])) == CellSet(g, (1, 2), (3, 3))


def test_identity_is_value_based_and_cheap():
    a = CellSet.all_active(SIN).drop_columns(6, 5)
    b = CellSet.all_active(SIN).drop_columns(6, 5)
    assert a is not b and a == b and hash(a) == hash(b)
    assert {a: 1}[b] == 1
    assert a != CellSet.all_active(SIN)
    assert "size=15681" in repr(a)
    m_in, n_in = np.array([1, 1, 2, 2]), np.array([0, 1, 0, 1])
    c = CellSet(wilson.Grid(8, 16, 1.0), m_in, n_in)
    assert m_in.flags.writeable and n_in.flags.writeable              # caller's arrays untouched
    assert not c.m_array.flags.writeable and c.m == (1, 1, 2, 2)


@pytest.mark.parametrize("roundtrip", [lambda c: pickle.loads(pickle.dumps(c)), copy.copy,
                                       copy.deepcopy], ids=["pickle", "copy", "deepcopy"])
def test_a_cell_set_survives_pickle_and_copy_as_an_equal_read_only_set(roundtrip):
    """``__reduce__`` rebuilds from ``(grid, m, n)``.  Before it, pickle and deepcopy gave
    writeable index arrays and ``copy.copy`` shared the original's."""
    cells = CellSet.all_active(wilson.Grid(8, 16, 0.5)).drop_columns(2, 1)
    back = roundtrip(cells)
    assert back is not cells and back == cells and hash(back) == hash(cells)
    for got, orig in ((back.m_array, cells.m_array), (back.n_array, cells.n_array)):
        assert not got.flags.writeable
        assert np.array_equal(got, orig) and not np.shares_memory(got, orig)
    assert np.array_equal(back.packed_positions(), cells.packed_positions())


def test_wdm_domain_and_channels():
    cells = CellSet.all_active(wilson.Grid(8, 16, 1.0))
    d = WDMDomain(cells)
    assert d.size == 128 and not hasattr(d, "scale")
    assert list(inspect.signature(WDMDomain).parameters) == ["cells", "window"]
    with pytest.raises(TypeError, match="Window"):
        WDMDomain(cells, 2.0)                       # the second field is the window now
    with pytest.raises(TypeError, match="unexpected keyword argument 'scale'"):
        WDMDomain(cells, scale=2.0)                 # the field is gone
    ch = ChannelAxis(("A", "E"))
    assert ch.size == 2 and ChannelAxis().size == 0
    with pytest.raises(ValueError, match="unique"):
        ChannelAxis(("A", "A"))
