"""domainia.translate.transform: {channel: array} between full-record domains (spec section 3)."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.testing import ALT_WINDOW
from domainia.translate import log_coordinate_scale, transform
from domainia.wilson import MEYER

NF, NT, DT = 8, 16, 0.25
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
TIME = TimeDomain(N, DT)
FOURIER = FourierDomain.full(N, DT)
WDM = WDMDomain.full(GRID)
WDM_ALT = WDMDomain.full(wilson.Grid(16, 8, DT), window=ALT_WINDOW)
DOMAINS = {"time": TIME, "fourier": FOURIER, "wdm": WDM, "wdm-alt": WDM_ALT}
MASK = wilson.active_mask(NF, NT)
_rng = np.random.default_rng(41)
X = {"A": _rng.standard_normal(N), "E": _rng.standard_normal(N)}
#: An odd record (no Nyquist bin): only time and Fourier exist for it.
ODD_TIME, ODD_FOURIER = TimeDomain(15, 0.5), FourierDomain.full(15, 0.5)
ODD_X = {"A": np.random.default_rng(3).standard_normal(15)}
#: (time domain, data, domain) for the scale tests: every test domain and the odd record.
RECORDS = {**{k: (TIME, X, d) for k, d in DOMAINS.items()},
           "odd-time": (ODD_TIME, ODD_X, ODD_TIME), "odd-fourier": (ODD_TIME, ODD_X, ODD_FOURIER)}


def _on(domain):
    return transform(X, TIME, domain)


def _close(got, want, rel=1e-12):
    assert list(got) == list(want)
    for k in want:
        np.testing.assert_allclose(got[k], want[k], rtol=0, atol=rel * float(np.max(np.abs(want[k]))))


def _energy(domain, v):
    if isinstance(domain, FourierDomain):
        w = np.full(v.shape[-1], 2.0)
        w[0] = 1.0
        if domain.N % 2 == 0:
            w[-1] = 1.0
        return float(np.sum(w * np.abs(v) ** 2))
    return float(np.sum(np.asarray(v) ** 2))


def _with(a, idx, value):
    a = np.array(a)
    a[idx] = value
    return a


def test_the_full_domains():
    assert FOURIER == FourierDomain(N, DT, 0, N // 2 + 1, "dt_rfft")
    assert WDM == WDMDomain(CellSet.all_active(GRID)) and WDM.window is MEYER
    assert WDM_ALT.window is ALT_WINDOW and WDM_ALT != WDMDomain.full(wilson.Grid(16, 8, DT))


def test_the_definitions():
    _close(_on(FOURIER), {k: DT * np.fft.rfft(v) for k, v in X.items()}, rel=1e-14)
    _close(_on(WDM), {k: wilson.forward_time(v, NF, NT, DT) for k, v in X.items()}, rel=1e-14)
    _close(_on(WDM_ALT), {k: wilson.forward_time(v, 16, 8, DT, window=ALT_WINDOW) for k, v in X.items()},
           rel=1e-14)


@pytest.mark.parametrize("src", DOMAINS)
@pytest.mark.parametrize("dst", DOMAINS)
def test_every_pair_round_trips_and_agrees_with_the_route_through_time(src, dst):
    a, b = DOMAINS[src], DOMAINS[dst]
    there = transform(_on(a), a, b)
    _close(there, _on(b))
    _close(transform(there, b, a), _on(a))


@pytest.mark.parametrize("name", RECORDS)
def test_parseval_scales_match_log_coordinate_scale(name):
    t, data, d = RECORDS[name]
    on = transform(data, t, d)
    for k, x in data.items():
        want = math.exp(2.0 * log_coordinate_scale(d)) * float(np.sum(x ** 2))
        assert _energy(d, on[k]) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("n, dt", [(N, DT), (15, 0.5)])
def test_log_coordinate_scale_values(n, dt):
    assert log_coordinate_scale(TimeDomain(n, dt)) == 0.0
    assert log_coordinate_scale(FourierDomain.full(n, dt)) == pytest.approx(
        math.log(dt) + 0.5 * math.log(n), rel=1e-15)
    if n % 2 == 0:
        assert log_coordinate_scale(WDM) == pytest.approx(0.5 * math.log(DT), rel=1e-15)
        assert log_coordinate_scale(WDM_ALT) == log_coordinate_scale(WDM)


def test_log_coordinate_scale_refuses_what_transform_refuses():
    with pytest.raises(ValueError, match="FourierDomain.full"):
        log_coordinate_scale(FourierDomain(N, DT, 0, 10, "dt_rfft"))
    with pytest.raises(ValueError, match="WDMDomain.full"):
        log_coordinate_scale(WDMDomain(CellSet.bands(GRID, (2, 3))))
    with pytest.raises(TypeError, match="domain"):
        log_coordinate_scale(GRID)


def test_an_odd_record_has_no_nyquist_bin():
    y = transform(ODD_X, ODD_TIME, ODD_FOURIER)
    assert y["A"].shape == (8,) and y["A"][-1].imag != 0.0
    _close(transform(y, ODD_FOURIER, ODD_TIME), ODD_X)


def test_outputs_are_fresh_exact_and_in_input_order():
    src = {"E": X["E"], "A": X["A"]}
    same = transform(src, TIME, TIME)
    assert list(same) == ["E", "A"] and not np.shares_memory(same["A"], X["A"])
    assert not np.shares_memory(transform({"A": X["A"]}, TIME, TIME)["A"], X["A"])
    f = transform(src, TIME, FOURIER)
    assert all(v.dtype == np.complex128 and v[0].imag == 0.0 and v[-1].imag == 0.0 for v in f.values())
    w = transform(f, FOURIER, WDM)
    assert list(w) == ["E", "A"]
    assert all(v.dtype == np.float64 and np.all(v[~MASK] == 0.0) for v in w.values())
    assert transform({"A": np.arange(N)}, TIME, TIME)["A"].dtype == np.float64
    assert transform({"A": X["A"].astype(np.float32)}, TIME, WDM)["A"].dtype == np.float64


def test_low_precision_inputs_are_promoted_before_the_transform():
    """float32/complex64 inputs are computed on in float64/complex128, not just cast after."""
    x32 = X["A"].astype(np.float32)
    _close(transform({"A": x32}, TIME, WDM), {"A": wilson.forward_time(x32.astype(np.float64), NF, NT, DT)},
           rel=1e-14)
    rounded = _on(FOURIER)["A"].astype(np.complex64).astype(np.complex128)    # complex64 values
    _close(transform({"A": rounded.astype(np.complex64)}, FOURIER, TIME), {"A": np.fft.irfft(rounded / DT, n=N)},
           rel=1e-14)


def test_fourier_endpoints_are_exactly_real_where_the_wdm_inverse_rounds_them():
    """On (6, 12) the WDM inverse leaves ~1e-15 imaginary parts at DC and Nyquist (on the
    power-of-two grids above it leaves none): the Fourier result is still exactly real there,
    so it transforms back."""
    t, f, w = TimeDomain(72, 0.3), FourierDomain.full(72, 0.3), WDMDomain.full(wilson.Grid(6, 12, 0.3))
    x = {"A": np.random.default_rng(5).standard_normal(72)}
    y = transform(transform(x, t, w), w, f)
    assert y["A"][0].imag == 0.0 and y["A"][-1].imag == 0.0
    _close(y, transform(x, t, f))
    _close(transform(y, f, w), transform(x, t, w))


def _shape_message(name, got, kind, want):
    """The input-shape refusal itself (a kernel's own shape error also says "shape")."""
    return re.escape(f"channel {name!r}: shape {got}, but this {kind}Domain needs shape {want}")


_FOUR = _on(FOURIER)["A"]
_WDMV = _on(WDM)["A"]
REFUSALS = {
    "empty": ({}, TIME, WDM, ValueError, "empty"),
    "list": ([X["A"]], TIME, WDM, TypeError, "mapping"),
    "array": (X["A"], TIME, WDM, TypeError, "mapping"),
    "source-kind": (X, "time", WDM, TypeError, "domain"),
    "target-kind": (X, TIME, None, TypeError, "domain"),
    "partial-fourier": ({"A": _FOUR[1:11]}, FourierDomain(N, DT, 1, 10, "dt_rfft"), TIME, ValueError,
                        "FourierDomain.full"),
    "partial-fourier-n0": (X, TIME, FourierDomain(N, DT, 0, 10, "dt_rfft"), ValueError, "FourierDomain.full"),
    "fourier-units": ({"A": _FOUR}, FourierDomain(N, DT, 0, N // 2 + 1, "rfft"), TIME, ValueError,
                      "FourierDomain.full"),
    "partial-wdm": (X, TIME, WDMDomain(CellSet.bands(GRID, (2, 3))), ValueError, "WDMDomain.full"),
    "N": (X, TIME, FourierDomain.full(N + 2, DT), ValueError, "N="),
    "dt": (X, TIME, WDMDomain.full(wilson.Grid(NF, NT, 2 * DT)), ValueError, "dt="),
    "t0": (X, TIME, TimeDomain(N, DT, t0=1.0), ValueError, "t0"),
    "shape": ({"A": X["A"][:-1]}, TIME, WDM, ValueError, _shape_message("A", (127,), "Time", (128,))),
    "channel-shapes": ({"A": X["A"], "E": np.zeros(N + 1)}, TIME, WDM, ValueError,
                       _shape_message("E", (129,), "Time", (128,))),
    "(1, N)": ({"A": X["A"][None]}, TIME, WDM, ValueError, _shape_message("A", (1, 128), "Time", (128,))),
    "wdm-flat": ({"A": _WDMV.reshape(-1)}, WDM, TIME, ValueError, _shape_message("A", (144,), "WDM", (9, 16))),
    "wdm-transposed": ({"A": _WDMV.T}, WDM, TIME, ValueError, _shape_message("A", (16, 9), "WDM", (9, 16))),
    "nan": ({"A": _with(X["A"], 3, np.nan)}, TIME, WDM, ValueError,
            r"channel 'A': 1 of 128 entries are not finite \(1 nan, 0 inf\); first at index 3"),
    "inf-fourier": ({"A": _with(_FOUR, 5, np.inf)}, FOURIER, TIME, ValueError,
                    r"channel 'A': 1 of 65 entries are not finite \(0 nan, 1 inf\); first at index 5"),
    "nan-wdm": ({"A": _with(_WDMV, (2, 3), np.nan)}, WDM, TIME, ValueError,
                r"channel 'A': 1 of 144 entries are not finite \(1 nan, 0 inf\); first at index 35\. .*"
                r"\(Index 35 is WDM cell \(m=2, n=3\) of the dense \(9, 16\) layout\.\)$"),
    "masked": ({"A": np.ma.MaskedArray(X["A"], mask=_with(np.zeros(N, bool), 5, True))}, TIME, WDM,
               TypeError, "masked"),
    "masked-none": ({"A": np.ma.MaskedArray(X["A"])}, TIME, WDM, TypeError, "masked"),
    "complex-time": ({"A": X["A"] + 0j}, TIME, WDM, ValueError, "complex"),
    "complex-wdm": ({"A": _WDMV + 0j}, WDM, TIME, ValueError, "complex"),
    "strings": ({"A": np.array(["x"] * N)}, TIME, WDM, TypeError, "numeric"),
    "booleans": ({"A": np.ones(N, dtype=bool)}, TIME, WDM, TypeError, "numeric"),
    "name-kind": ({1: X["A"]}, TIME, WDM, TypeError, "string"),
    "name-empty": ({"": X["A"]}, TIME, WDM, ValueError, "non-empty"),
    "dc": ({"A": _with(_FOUR, 0, _FOUR[0] + 1e-300j)}, FOURIER, TIME, ValueError, r"bin 0 \(DC\) must be real"),
    "nyquist": ({"A": _with(_FOUR, -1, _FOUR[-1] + 1j)}, FOURIER, TIME, ValueError,
                r"bin 64 \(Nyquist\) must be real"),
    "inactive-dc": ({"A": _with(_WDMV, (0, 1), 1e-300)}, WDM, TIME, ValueError, "inactive"),
    "inactive-nyquist": ({"A": _with(_WDMV, (NF, 1), 1.0)}, WDM, TIME, ValueError, "inactive"),
}


@pytest.mark.parametrize("case", REFUSALS)
def test_every_refusal_rule(case):
    data, src, dst, err, match = REFUSALS[case]
    with pytest.raises(err, match=match):
        transform(data, src, dst)


_ZERO_INACTIVE = "zero them first, e.g. w[..., ~domainia.wilson.active_mask(8, 16)] = 0"
_REAL_ENDS = "take the real part of the DC/Nyquist bins first, e.g. X[..., [0, 64]] = X[..., [0, 64]].real"


@pytest.mark.parametrize("case, remedy", [("inactive-dc", _ZERO_INACTIVE), ("inactive-nyquist", _ZERO_INACTIVE),
                                          ("dc", _REAL_ENDS), ("nyquist", _REAL_ENDS)])
def test_the_canonical_form_refusals_name_their_remedy(case, remedy):
    data, src, dst, err, _ = REFUSALS[case]
    with pytest.raises(err) as info:
        transform(data, src, dst)
    assert remedy in str(info.value)


def _relative_size(message):
    found = re.search(r"\(([-+.e0-9]+) of the channel's largest \|value\|\)", message)
    assert found, message
    return float(found.group(1))


@pytest.mark.parametrize("size", [1e-16, 2.0])
@pytest.mark.parametrize("domain", ["wdm", "fourier"])
def test_the_canonical_form_refusals_size_the_value_against_the_channel_peak(domain, size):
    """Round-off (~1e-16 of the peak) is what the named remedy is for; a value at the peak is
    another layout or convention.  The message prints the relative size so the reader can tell."""
    if domain == "wdm":
        a, src, remedy = _with(_WDMV, (0, 1), -size * np.max(np.abs(_WDMV))), WDM, _ZERO_INACTIVE
    else:
        a, src, remedy = _with(_FOUR, 0, _FOUR[0] + 1j * size * np.max(np.abs(_FOUR))), FOURIER, _REAL_ENDS
    with pytest.raises(ValueError) as info:
        transform({"A": a}, src, TIME)
    message = str(info.value)
    assert f"if that is round-off, {remedy}; a value near the peak means another layout" in message
    assert _relative_size(message) == pytest.approx(min(size, 1.0), rel=0.15)


def test_an_odd_record_refusal_names_the_dc_remedy_only():
    y = transform(ODD_X, ODD_TIME, ODD_FOURIER)["A"]
    with pytest.raises(ValueError, match=r"bin 0 \(DC\) must be real") as info:
        transform({"A": _with(y, 0, y[0] + 1j)}, ODD_FOURIER, ODD_TIME)
    assert "take the real part of the DC bin first, e.g. X[..., 0] = X[..., 0].real" in str(info.value)


def test_the_named_remedies_make_raw_kernel_and_fft_outputs_valid_inputs():
    """Raw kernel outputs and full-FFT slices may carry ~1e-16 residue where transform needs
    exact zeros.  On (6, 12) all three forms do, so each is refused as it is; after the remedy
    its refusal names, transform takes it and agrees with its own route."""
    Nf, Nt, dt = 6, 12, 0.3
    n = Nf * Nt
    t, f, w = TimeDomain(n, dt), FourierDomain.full(n, dt), WDMDomain.full(wilson.Grid(Nf, Nt, dt))
    x = {"A": np.random.default_rng(6).standard_normal(n)}
    coeffs = wilson.forward_time(x["A"], Nf, Nt, dt)
    with pytest.raises(ValueError, match="inactive WDM cell"):
        transform({"A": coeffs}, w, t)
    coeffs[..., ~wilson.active_mask(Nf, Nt)] = 0
    _close(transform({"A": coeffs}, w, t), x)
    for spec in (dt * wilson.inverse(coeffs, Nf, Nt, dt), dt * np.fft.fft(x["A"])[: n // 2 + 1]):
        with pytest.raises(ValueError, match="must be real"):
            transform({"A": spec}, f, t)
        spec[..., [0, n // 2]] = spec[..., [0, n // 2]].real
        _close(transform({"A": spec}, f, t), x)


def test_a_non_finite_result_is_refused():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="the result on the FourierDomain is not finite"):
            transform({"A": np.full(N, 1e308)}, TIME, FOURIER)
