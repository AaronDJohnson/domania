"""``domainia.spectra``, ``domainia.data``, ``domainia.translate``, ``domainia.variance`` and
``domainia.testing`` refuse a wrong kind with ``TypeError`` and a wrong value with
``ValueError``, as ``domainia.wilson`` does."""

from __future__ import annotations

import re

import numpy as np
import pytest

from domainia import testing as T
from domainia import wilson
from domainia.data import FourierBins, TimeSeries, WDMCells
from domainia.domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from domainia.spectra import GenericPSD, TabulatedSpectrum
from domainia.translate import transform
from domainia.variance import BandAveragedVariance, MidpointVariance

GRID = wilson.Grid(8, 16, 0.5)
CELLS = CellSet.all_active(GRID)
TIME = TimeDomain(128, 0.5)
WDM = WDMDomain.full(GRID)
BINS = FourierDomain(128, 0.5, 2, 10, "dt_rfft")               # absolute bins 2 .. 11
SPECTRUM = TabulatedSpectrum(BINS, np.arange(10.0)[None])


def exactly(message):
    return f"^{re.escape(message)}$"


# --- domainia.spectra ---------------------------------------------------------------------


@pytest.mark.parametrize(("domain", "kind"), [(TIME, "TimeDomain"), (GRID, "Grid"), (None, "NoneType")])
def test_a_spectrum_lives_on_a_fourier_domain(domain, kind):
    """Both used to raise a bare ``AttributeError`` (``domain.n``, ``domain.N``)."""
    message = f"TabulatedSpectrum.domain must be a FourierDomain, got {kind}"
    with pytest.raises(TypeError, match=exactly(message)):
        TabulatedSpectrum(domain, np.ones((1, 10)))
    with pytest.raises(TypeError, match=exactly(f"domain must be a FourierDomain, got {kind}")):
        SPECTRUM.on(domain)


@pytest.mark.parametrize(("values", "dtype"), [
    (np.ones((1, 10)) + 0j, "complex128"), (np.ones((1, 10), bool), "bool"),
    (np.full((1, 10), "1"), "<U1"), (np.ones((1, 10)).astype(object), "object"),
])
def test_spectrum_values_must_be_real_numbers(values, dtype):
    """A complex table used to lose its imaginary part (only a ``ComplexWarning``); a bool, str
    or object table was converted to float."""
    message = f"TabulatedSpectrum.values must be a real numeric array, got dtype {dtype}"
    with pytest.raises(TypeError, match=exactly(message)):
        TabulatedSpectrum(BINS, values)


def test_spectrum_values_of_numpy_integers_and_floats_pass():
    for values in (np.arange(10, dtype=np.int64)[None], np.arange(10, dtype=np.float64)[None]):
        s = TabulatedSpectrum(BINS, values)
        assert s.values.dtype == np.float64 and np.array_equal(s.values, SPECTRUM.values)


@pytest.mark.parametrize(("bins", "dtype"), [
    ([True], "bool"), ([2 + 0j], "complex128"), (["2"], "<U1"), (np.array([2], dtype=object), "object"),
])
def test_spectrum_bins_of_the_wrong_kind_raise_type_error(bins, dtype):
    """They used to raise ``ValueError("bins must be integers; ...")``."""
    with pytest.raises(TypeError, match=exactly(f"bins must be an integer index array, got dtype {dtype}")):
        SPECTRUM.at(bins)


def test_spectrum_bins_that_are_not_integers_stay_value_errors():
    for bins, first in (([2.5], "2.5"), ([3.0, np.nan], "nan"), ([np.inf], "inf")):
        with pytest.raises(ValueError, match=exactly(f"bins must hold integer indices, got {first}")):
            SPECTRUM.at(bins)
    want = SPECTRUM.values[:, [0, 9]]
    assert np.array_equal(SPECTRUM.at(np.array([2, 11], dtype=np.int64)), want)
    assert np.array_equal(SPECTRUM.at(np.array([2.0, 11.0], dtype=np.float64)), want)


@pytest.mark.parametrize(("fn", "kind"), [(1.0, "float"), (None, "NoneType"), ("psd", "str")])
def test_a_generic_psd_needs_a_callable(fn, kind):
    """It used to accept anything and fail at the first evaluation."""
    with pytest.raises(TypeError, match=exactly(f"GenericPSD.fn must be callable, got {kind}")):
        GenericPSD(fn)


# --- domainia.variance (already TypeError: pinned here) -----------------------------------


@pytest.mark.parametrize("cls", [BandAveragedVariance, MidpointVariance])
def test_the_variances_take_only_a_generic_psd(cls):
    with pytest.raises(TypeError, match="wrap the spectrum in domainia.spectra.GenericPSD"):
        cls(lambda f, t, nu: f, CELLS)


# --- domainia.data ------------------------------------------------------------------------

DATA = [(TimeSeries, TIME, np.zeros((1, 128))), (FourierBins, BINS, np.zeros((1, 10), complex)),
        (WDMCells, WDM, np.zeros((1, 128)))]


@pytest.mark.parametrize(("cls", "domain", "values"), DATA, ids=["TimeSeries", "FourierBins", "WDMCells"])
@pytest.mark.parametrize(("channels", "kind"), [
    (("A",), "tuple"), ("A", "str"), (np.str_("A"), "str_"), (np.float64(1.0), "float64"),
    (None, "NoneType"),
])
def test_data_channels_must_be_a_channel_axis(cls, domain, values, channels, kind):
    """A tuple, a str or ``None`` used to raise a bare ``AttributeError`` (``channels.size``),
    and a NumPy scalar was taken as a one-name axis."""
    message = f"{cls.__name__}.channels must be a ChannelAxis, got {kind}"
    with pytest.raises(TypeError, match=exactly(message)):
        cls(values, domain, channels)
    assert cls(values, domain, ChannelAxis(("A",))).channels.names == ("A",)


@pytest.mark.parametrize(("cls", "kind"), [(TimeSeries, "TimeDomain"), (FourierBins, "FourierDomain"),
                                           (WDMCells, "WDMDomain")])
def test_data_domain_messages_say_got(cls, kind):
    with pytest.raises(TypeError, match=exactly(f"{cls.__name__}.domain must be a {kind}, got Grid")):
        cls(np.zeros(128), GRID)


# --- domainia.translate -------------------------------------------------------------------

Y = {"A": np.ones(128)}
FULL_RECORD = "must be a full-record domain (TimeDomain, FourierDomain.full or WDMDomain.full), got Grid"


@pytest.mark.parametrize(("call", "message"), [
    (lambda: transform([np.ones(128)], TIME, WDM),
     "channel_data must be a mapping {channel: array}, got list"),
    (lambda: transform({1: np.ones(128)}, TIME, WDM), "channel names must be strings, got 1"),
    (lambda: transform({"A": np.ones(128).astype(object)}, TIME, WDM),
     "channel 'A': values must be numeric, got dtype object"),
    (lambda: transform(Y, GRID, WDM), f"source {FULL_RECORD}"),
], ids=["mapping", "name", "numeric", "source"])
def test_translate_kind_messages_say_got(call, message):
    with pytest.raises(TypeError, match=exactly(message)):
        call()


def test_complex_values_on_a_time_domain_stay_a_value_error():
    """Complex values are valid on a ``FourierDomain``: the domain makes them wrong."""
    with pytest.raises(ValueError, match="complex values on a real TimeDomain"):
        transform({"A": np.ones(128) + 0j}, TIME, WDM)


# --- domainia.testing ---------------------------------------------------------------------


@pytest.mark.parametrize(("cells", "kind"), [(GRID, "Grid"), (WDM, "WDMDomain"), (None, "NoneType")])
def test_the_dense_oracles_take_a_cell_set(cells, kind):
    """``dense_W``, ``wdm_congruence`` and ``eq45_dense`` used to raise a bare
    ``AttributeError`` (``cells.grid``)."""
    message = exactly(f"cells must be a CellSet, got {kind}")
    with pytest.raises(TypeError, match=message):
        T.dense_W(cells)
    with pytest.raises(TypeError, match=message):
        T.wdm_congruence(np.eye(128), cells)
    with pytest.raises(TypeError, match=message):
        T.eq45_dense(cells, lambda f, t: 1.0 + 0.0 * f * t)


def test_eq45_dense_needs_a_callable_spectrum():
    with pytest.raises(TypeError, match=exactly("Sn must be callable, got float")):
        T.eq45_dense(CELLS, 1.0)
