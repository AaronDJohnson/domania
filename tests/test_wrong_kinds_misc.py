"""``domainia.spectra``, ``domainia.data``, ``domainia.translate``, ``domainia.variance``,
``domainia.linalg`` and ``domainia.testing`` refuse a wrong kind with ``TypeError`` and a wrong
value with ``ValueError``, as ``domainia.wilson`` does."""

from __future__ import annotations

import re

import numpy as np
import pytest

from domainia import linalg as la
from domainia import testing as T
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


# --- domainia.linalg and domainia.testing -------------------------------------------------

M4 = np.eye(4, dtype=complex)
B2 = np.ones((2, 4))
G, MARGIN = 4, 2
S_I, S_G = np.ones(G), np.ones(G + 2 * MARGIN)
H = np.zeros(2 * (G + MARGIN) + 1, complex)
H[G + MARGIN] = 1.0                                           # lag 0


def _pcg(**kw):
    """PCG on ``A = 2 I`` with ``M = A^-1``: one iteration, exact."""
    args = {"tol": 1e-10, "maxit": 10} | kw
    return la.pcg(lambda X: 2.0 * X, lambda X: 0.5 * X, B2, **args)


INTEGER_SLOTS = {
    "cblock_factor b": ("b", lambda v: la.cblock_factor(M4, v)),
    "pcg maxit": ("maxit", lambda v: _pcg(maxit=v)),
    "BandLayout.from_sizes": ("sizes[0]", lambda v: la.BandLayout.from_sizes((v, 3))),
    "eq54_covariance margin": ("margin", lambda v: T.eq54_covariance(S_I, S_G, H, margin=v)),
    "dense_lambda_generic margin": ("margin", lambda v: T.dense_lambda_generic(np.eye(G), S_I, S_G, H,
                                                                                margin=v)),
}


@pytest.mark.parametrize("slot", INTEGER_SLOTS)
@pytest.mark.parametrize(("value", "kind"), [
    (2.0, "float"), (np.float64(2.0), "float64"), (True, "bool"), ("2", "str"), (None, "NoneType"),
])
def test_integer_slots_refuse_other_kinds(slot, value, kind):
    """``int()`` used to truncate ``2.5`` and take ``True`` as 1; ``pcg`` compared a float
    ``maxit``; ``margin=True`` passed as 1 and ``margin=1.0`` raised ``IndexError``."""
    name, call = INTEGER_SLOTS[slot]
    with pytest.raises(TypeError, match=exactly(f"{name} must be an integer, got {kind}")):
        call(value)


@pytest.mark.parametrize("slot", INTEGER_SLOTS)
def test_integer_slots_take_numpy_integers(slot):
    _, call = INTEGER_SLOTS[slot]
    got, want = call(np.int64(2)), call(2)
    if isinstance(got, np.ndarray):
        assert np.array_equal(got, want)
    elif isinstance(got, la.BandLayout):
        assert (got.nb, got.B, got.n) == (want.nb, want.B, want.n) and np.array_equal(got.back, want.back)
    elif isinstance(got, la.CBlockFactor):
        assert got.b == 2 and type(got.b) is int and np.array_equal(got.Ls, want.Ls)
    else:
        assert np.array_equal(got.x, want.x) and bool(np.all(got.converged))


@pytest.mark.parametrize(("name", "value", "kind"), [
    (name, value, kind) for name in ("tol", "true_rtol")
    for value, kind in ((True, "bool"), ("1e-6", "str"), (1e-6j, "complex"), (np.array(1e-6), "ndarray"))
] + [("tol", None, "NoneType")])
def test_pcg_tolerances_must_be_real_numbers(name, value, kind):
    """``float()`` used to parse ``true_rtol="1e-6"`` and take ``True`` as 1.0; ``tol=True``
    and a 0-d array passed, and a str, complex or ``None`` ``tol`` failed deep inside with
    Python's own message (``true_rtol=None`` is the documented default)."""
    with pytest.raises(TypeError, match=exactly(f"{name} must be a real number, got {kind}")):
        _pcg(**{name: value})


def test_pcg_tolerances_take_numpy_integers_and_floats():
    res = _pcg(tol=np.float64(1e-10), maxit=np.int64(10), true_rtol=np.int64(1))
    assert np.array_equal(res.x, 0.5 * B2) and bool(np.all(res.converged))
    assert np.array_equal(_pcg(tol=np.int64(0)).x, res.x)


STRING_SLOTS = {
    "pcg on_fail": ("on_fail", lambda v: _pcg(on_fail=v)),
    "blocktri_factor layout": ("layout", lambda v: la.blocktri_factor([np.eye(2)], [], layout=v)),
}


@pytest.mark.parametrize("slot", STRING_SLOTS)
@pytest.mark.parametrize(("value", "kind"), [(1, "int"), (None, "NoneType"), (b"nan", "bytes")])
def test_string_options_refuse_other_kinds(slot, value, kind):
    """They used to raise the unknown-choice ``ValueError``, which still answers an unknown
    string."""
    name, call = STRING_SLOTS[slot]
    with pytest.raises(TypeError, match=exactly(f"{name} must be a string, got {kind}")):
        call(value)
    with pytest.raises(ValueError, match=f"{name} must be "):
        call("dense")


def test_the_layouts_and_factors_are_checked():
    """Each used to raise a bare ``AttributeError``."""
    diag, up = [np.eye(2), np.eye(2)], [np.zeros((2, 2))]
    layout = la.BandLayout.from_sizes([2, 2])
    ragged = la.blocktri_factor(diag, up, layout="ragged")
    padded = la.blocktri_factor(diag, up, layout="padded", band_layout=layout)
    b = np.ones(4)
    for cells, kind in ((GRID, "Grid"), (WDM, "WDMDomain")):
        with pytest.raises(TypeError, match=exactly(f"cells must be a CellSet, got {kind}")):
            la.BandLayout.from_cells(cells)
    assert la.BandLayout.from_cells(CELLS).sizes == tuple(s for _, s in CELLS.band_sizes())
    message = exactly("band_layout must be a BandLayout, got object")
    with pytest.raises(TypeError, match=message):
        la.blocktri_factor(diag, up, layout="padded", band_layout=object())
    for fn in (la.blocktri_solve, la.blocktri_whiten):
        with pytest.raises(TypeError, match=message):
            fn(padded, b, band_layout=object())
        with pytest.raises(TypeError, match=exactly("f must be a BlockTriFactor, got CBlockFactor")):
            fn(la.cblock_factor(M4, 2), b)
        np.testing.assert_array_equal(fn(padded, b, band_layout=layout), fn(ragged, b))
    for fn in (la.blocktri_solve_packed, la.blocktri_whiten_packed):
        with pytest.raises(TypeError, match=exactly("f must be a BlockTriFactor, got object")):
            fn(object(), np.ones((2, 2, 1)))
    for fn in (la.cblock_forward, la.cblock_backward):
        with pytest.raises(TypeError, match=exactly("f must be a CBlockFactor, got BlockTriFactor")):
            fn(ragged, b)


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


@pytest.mark.parametrize("name", ["s_I", "s_G_src"])
@pytest.mark.parametrize(("dtype", "shown"), [(complex, "complex128"), (bool, "bool"), (object, "object")])
def test_eq54_spectra_must_be_real_numbers(name, dtype, shown):
    """A complex ``s_I`` or ``s_G_src`` used to lose its imaginary part (only a
    ``ComplexWarning``)."""
    args = {"s_I": S_I, "s_G_src": S_G}
    args[name] = args[name].astype(dtype)
    with pytest.raises(TypeError, match=exactly(f"{name} must be a real numeric array, got dtype {shown}")):
        T.eq54_covariance(args["s_I"], args["s_G_src"], H, margin=MARGIN)


def test_eq54_spectra_take_numpy_integers_and_floats():
    want = T.eq54_covariance(S_I, S_G, H, margin=MARGIN)
    for dtype in (np.int64, np.float64):
        got = T.eq54_covariance(S_I.astype(dtype), S_G.astype(dtype), H, margin=MARGIN)
        assert np.array_equal(got, want)


def test_the_pointwise_test_operator_messages_say_got():
    message = "PointwiseOperator needs a TimeDomain or WDMDomain, got Grid"
    with pytest.raises(TypeError, match=exactly(message)):
        T.PointwiseOperator(GRID, {"A": np.ones(128)})
    with pytest.raises(TypeError, match=exactly("variances must be a {channel: array} mapping, got list")):
        T.PointwiseOperator(TIME, [np.ones(128)])
    with pytest.raises(TypeError, match=exactly("channel names must be strings, got 1")):
        T.PointwiseOperator(TIME, {1: np.ones(128)})
    with pytest.raises(TypeError, match=exactly("values must be a mapping, got list")):
        T.PointwiseOperator(TIME, {"A": np.ones(128)}).apply([np.ones(128)])
