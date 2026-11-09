"""``domainia.spectra``, ``domainia.data``, ``domainia.translate``, ``domainia.variance``,
``domainia.linalg`` and ``domainia.testing`` refuse a wrong kind with ``TypeError`` and a wrong
value with ``ValueError``, as ``domainia.wilson`` does."""

from __future__ import annotations

import re

import numpy as np
import pytest

from domainia import wilson
from domainia.data import FourierBins, TimeSeries, WDMCells
from domainia.domains import CellSet, ChannelAxis, FourierDomain, TimeDomain, WDMDomain
from domainia.spectra import GenericPSD, TabulatedSpectrum
from domainia.translate import (TranslatedOperator, from_coordinates, log_coordinate_scale,
                                to_coordinates, transform, translate_operator)
from domainia.variance import BandAveragedVariance, MidpointVariance

GRID = wilson.Grid(8, 16, 0.5)
CELLS = CellSet.all_active(GRID)
TIME = TimeDomain(128, 0.5)
FOURIER = FourierDomain.full(128, 0.5)
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
    and a NumPy scalar was taken as a one-name axis (spec §2.6)."""
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
    (lambda: log_coordinate_scale(GRID), f"domain {FULL_RECORD}"),
    (lambda: to_coordinates(Y, TIME, "A"), "channel_names must be a tuple of channel names, got str"),
    (lambda: translate_operator(object(), WDM),
     "op must implement domainia.translate.Operator, got object"),
    (lambda: TranslatedOperator(object(), WDM),
     "native must be a native domainia.translate.Operator (translate_operator re-targets a "
     "TranslatedOperator), got object"),
], ids=["mapping", "name", "numeric", "source", "domain", "channel_names", "op", "native"])
def test_translate_kind_messages_say_got(call, message):
    with pytest.raises(TypeError, match=exactly(message)):
        call()


@pytest.mark.parametrize(("R", "dtype"), [
    (np.ones((1, 128)) + 0j, "complex128"), (np.ones((1, 128), bool), "bool"),
    (np.full((1, 128), "1"), "<U1"), (np.ones((1, 128)).astype(object), "object"),
])
def test_from_coordinates_refuses_coordinates_that_are_not_real(R, dtype):
    """Complex coordinates used to raise ``ValueError`` (spec §3.5): ``R`` is never complex."""
    message = f"R must be an array of real coordinates, got dtype {dtype}"
    with pytest.raises(TypeError, match=exactly(message)):
        from_coordinates(R, TIME, ("A",))


def test_from_coordinates_takes_numpy_integers_and_floats():
    for R in (np.ones((1, 128), dtype=np.int64), np.ones((1, 128), dtype=np.float64)):
        out = from_coordinates(R, TIME, ("A",))
        assert out["A"].dtype == np.float64 and np.array_equal(out["A"], np.ones(128))


def test_complex_values_on_a_time_domain_stay_a_value_error():
    """Complex values are valid on a ``FourierDomain``: the domain makes them wrong (spec §3.5)."""
    with pytest.raises(ValueError, match="complex values on a real TimeDomain"):
        transform({"A": np.ones(128) + 0j}, TIME, WDM)


@pytest.mark.parametrize("part", ["real", "imaginary"])
def test_to_coordinates_refuses_an_overflowing_sqrt2_scaling(part):
    """``sqrt(2) * 1.3e308`` overflows float64; it used to come back as ``inf`` (spec §2.9).
    NumPy warns about the overflow before the check runs, hence ``np.errstate``."""
    unit = 1.0 if part == "real" else 1j
    X = np.zeros(65, complex)
    X[3] = 1.3e308 * unit
    data = {"A": np.zeros(65, complex), "E": X}
    message = f"^channel 'E': sqrt\\(2\\) times the {part} part of rfft bin 3 overflows float64"
    with np.errstate(over="ignore"), pytest.raises(ValueError, match=message):
        to_coordinates(data, FOURIER, ("E", "A"))
    X[3] = 1.2e308 * unit
    R = to_coordinates(data, FOURIER, ("E", "A"))
    assert np.all(np.isfinite(R)) and np.max(np.abs(R[0])) == np.sqrt(2.0) * 1.2e308
