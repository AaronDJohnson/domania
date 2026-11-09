"""Single precision end to end in ``domainia.wilson.transform``.

A float32 / complex64 input runs in single precision: every FFT receives a single-precision
array, and every product stored into the output is formed in the working dtype.  float64 runs
are bit for bit unchanged by the casts.  ``tests/wilson/test_meyer_baseline.py`` pins them bit
for bit except the rescaled families (``forward_bands``, ``inverse_bands``, ``FourierToWDM``),
which it compares by tolerance.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest

import domainia.wilson.transform as T
from domainia.wilson.backend import NUMPY

NF, NT, DT = 8, 16, 0.5
N = NF * NT
BANDS = np.array([2, 3])
SINGLE = (np.dtype(np.float32), np.dtype(np.complex64))

#: domainia's dtype table: each entry point's output dtype for a (float32, float64) run.
OUTPUT_DTYPE = {
    "forward_time": (np.float32, np.float64),
    "forward_time_many": (np.float32, np.float64),
    "forward": (np.float32, np.float64),
    "forward_many": (np.float32, np.float64),
    "inverse": (np.complex64, np.complex128),
    "inverse_many": (np.complex64, np.complex128),
    "inverse_time": (np.float32, np.float64),
    "inverse_time_many": (np.float32, np.float64),
    "forward_bands": (np.float32, np.float64),
    "inverse_bands": (np.complex64, np.complex128),
}
FORWARD = ("forward_time", "forward_time_many", "forward", "forward_many")
INVERSE = ("inverse", "inverse_many", "inverse_time", "inverse_time_many")


def output_dtype(name: str, precision) -> np.dtype:
    """``OUTPUT_DTYPE[name]`` for a run at ``precision`` (``float32`` or ``float64``)."""
    single, double = OUTPUT_DTYPE[name]
    return np.dtype(single if np.dtype(precision) == np.float32 else double)


def _cases(precision, rng, to=np.asarray) -> dict:
    """``{entry point: call(be)}`` for every row of the table, on inputs of ``precision``
    (real inputs in it, complex ones in the matching complex dtype), converted by ``to``."""
    real = np.dtype(precision)
    cplx = np.dtype(np.complex64 if real == np.float32 else np.complex128)
    x, X = rng.standard_normal(N), rng.standard_normal((3, N))
    z, Z = np.fft.rfft(x), np.fft.rfft(X, axis=-1)
    w, W = T.forward_time(x, NF, NT, DT), T.forward_time_many(X, NF, NT, DT)
    wb = T.forward_bands(z, 0, BANDS, NF, NT, DT)
    x, X, w, W, wb = (to(a.astype(real)) for a in (x, X, w, W, wb))
    z, Z = (to(a.astype(cplx)) for a in (z, Z))
    return {
        "forward_time": lambda be: T.forward_time(x, NF, NT, DT, be=be),
        "forward_time_many": lambda be: T.forward_time_many(X, NF, NT, DT, be=be),
        "forward": lambda be: T.forward(z, NF, NT, DT, be=be),
        "forward_many": lambda be: T.forward_many(Z, NF, NT, DT, be=be),
        "inverse": lambda be: T.inverse(w, NF, NT, DT, be=be),
        "inverse_many": lambda be: T.inverse_many(W, NF, NT, DT, be=be),
        "inverse_time": lambda be: T.inverse_time(w, NF, NT, DT, be=be),
        "inverse_time_many": lambda be: T.inverse_time_many(W, NF, NT, DT, be=be),
        "forward_bands": lambda be: T.forward_bands(z, 0, BANDS, NF, NT, DT, be=be),
        "inverse_bands": lambda be: T.inverse_bands(wb, 0, N // 2 + 1, BANDS, NF, NT, DT, be=be),
    }


class _RecordingFFT:
    """An ``fft`` namespace that records ``(function, input dtype)`` for every call, then
    calls the wrapped namespace (``numpy.fft`` or ``jax.numpy.fft``)."""

    def __init__(self, fft):
        self._fft = fft
        self.calls: list[tuple[str, np.dtype]] = []

    def __getattr__(self, name):
        fn = getattr(self._fft, name)

        def recorded(a, *args, **kw):
            self.calls.append((name, np.dtype(a.dtype)))
            return fn(a, *args, **kw)

        return recorded


def _recording(be):
    """``be`` with its ``fft`` namespace wrapped in a :class:`_RecordingFFT` (and the recorder)."""
    rec = _RecordingFFT(be.fft)
    return dataclasses.replace(be, fft=rec), rec


@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_every_fft_of_a_float32_run_is_single_precision(name, rng):
    """The inverse family and both band kernels used to run complex128 FFTs on float32 input
    (float64 sign patterns, ``lam`` and window samples promoted the FFT input)."""
    be, rec = _recording(NUMPY)
    _cases(np.float32, rng)[name](be)
    assert rec.calls, f"{name} ran no FFT through be.fft"
    assert all(dt in SINGLE for _, dt in rec.calls), rec.calls


@pytest.mark.parametrize("name", FORWARD + INVERSE)
def test_every_product_stored_in_a_float32_run_is_single_precision(name, rng, monkeypatch):
    """The value handed to ``_set_rounded`` / ``_add_rounded`` is already in the working dtype.

    The forward family's FFTs were single precision already; its leak was after the FFT, where
    float64 ``kappa`` and sign arrays (and the edge ``np.sqrt(2)``) made the stored product
    float64, rounded once on store."""
    run = _cases(np.float32, rng)[name]          # its float64 inputs are built before the spies
    seen: list[np.dtype] = []
    for helper in ("_set_rounded", "_add_rounded"):
        def spy(a, idx, v, be, _helper=getattr(T, helper)):
            seen.append(np.dtype(v.dtype))
            return _helper(a, idx, v, be)

        monkeypatch.setattr(T, helper, spy)
    run(NUMPY)
    assert seen and all(dt in SINGLE for dt in seen), seen


def test_inverse_bands_accumulates_in_single_precision(backend, rng, monkeypatch):
    """What ``inverse_bands`` hands to ``be.add_at`` in a float32 run is complex64.

    It is the same ``be.add_at`` call on both backends.  Without the post-FFT
    ``phi.astype(rdtype)`` the band values are complex128, and ``add_at`` rounds them into
    the complex64 output (silently in ``np.add.at``), so the output dtype, the recorded FFT
    inputs and the float64 comparison stay green."""
    run = _cases(np.float32, rng, to=backend.asarray)["inverse_bands"]   # before the spy
    seen: list[np.dtype] = []

    def spy(self, a, idx, v, *, _add_at=type(backend).add_at, **kw):
        seen.append(np.dtype(v.dtype))
        return _add_at(self, a, idx, v, **kw)

    monkeypatch.setattr(type(backend), "add_at", spy)
    run(backend)
    assert seen and all(dt == np.complex64 for dt in seen), seen


@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_output_dtypes_follow_the_table(name, precision, rng):
    assert _cases(precision, rng)[name](NUMPY).dtype == output_dtype(name, precision)


@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_float32_runs_match_float64_to_single_precision_rounding(name):
    """The same draws at both precisions, rounded to float32 for the float32 run and left
    unrounded for the float64 one: the two results agree to single-precision rounding."""
    lo = np.asarray(_cases(np.float32, np.random.default_rng(5))[name](NUMPY))
    hi = np.asarray(_cases(np.float64, np.random.default_rng(5))[name](NUMPY))
    assert np.max(np.abs(lo - hi)) <= 2e-6 * np.max(np.abs(hi))


@pytest.mark.jax
@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_jax_float32_runs_are_single_precision(name, jax_be, rng):
    """The same recording on the JAX backend: single-precision FFT inputs and the table's
    output dtype, returned as a ``jax.Array``."""
    import jax

    be, rec = _recording(jax_be)
    out = _cases(np.float32, rng, to=jax_be.asarray)[name](be)
    assert isinstance(out, jax.Array), type(out)
    assert out.dtype == output_dtype(name, np.float32)
    assert rec.calls and all(dt in SINGLE for _, dt in rec.calls), rec.calls


# ---------------------------------------------------------------------------
# The single-precision guard (spec 2026-11-08 §1.4): a float32 run that overflows raises
# ---------------------------------------------------------------------------


def _overflowing() -> dict:
    """``{entry point: (a, call(a, be))}``: a double-precision input ``a`` that stays finite
    when rounded to single precision, and whose single-precision run overflows float32
    inside the entry point.  The double-precision run is finite."""
    rng = np.random.default_rng(3)
    flat = np.full(N, 1e37)                    # rfft bin 0 is N * 1e37 = 1.3e39
    impulse = np.zeros(N)
    impulse[37] = 1e37                         # every rfft bin has modulus 1e37
    bins = (rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1)) * 1e36
    w_flat = T.forward_time(flat, NF, NT, DT)  # its inverse has bin 0 = 1.3e39 again
    w_impulse = T.forward_time(impulse, NF, NT, DT)
    w_bands = T.forward_bands(bins * 1e3, 0, BANDS, NF, NT, 1e-4)

    def two(a):
        return np.stack([a, a])

    return {
        "forward_time": (flat, lambda a, be: T.forward_time(a, NF, NT, DT, be=be)),
        "forward_time_many": (two(flat), lambda a, be: T.forward_time_many(a, NF, NT, DT, be=be)),
        "forward": (bins, lambda a, be: T.forward(a, NF, NT, 1e4, be=be)),
        "forward_many": (two(bins), lambda a, be: T.forward_many(a, NF, NT, 1e4, be=be)),
        "inverse": (w_flat, lambda a, be: T.inverse(a, NF, NT, DT, be=be)),
        "inverse_many": (two(w_flat), lambda a, be: T.inverse_many(a, NF, NT, DT, be=be)),
        "inverse_time": (w_impulse, lambda a, be: T.inverse_time(a, NF, NT, DT, be=be)),
        "inverse_time_many": (two(w_impulse),
                              lambda a, be: T.inverse_time_many(a, NF, NT, DT, be=be)),
        "forward_bands": (bins, lambda a, be: T.forward_bands(a, 0, BANDS, NF, NT, 1e4, be=be)),
        "inverse_bands": (w_bands, lambda a, be: T.inverse_bands(a, 0, N // 2 + 1, BANDS, NF, NT,
                                                                 1e-4, be=be)),
    }


def _single(a: np.ndarray) -> np.ndarray:
    """``a`` rounded to single precision (float32, or complex64 for complex ``a``)."""
    return a.astype(np.complex64 if np.iscomplexobj(a) else np.float32)


def test_the_guard_reads_single_precision_results_only():
    """A double result is not scanned at all, so float64 runs gain no pass; a single one is,
    and its message names the entry point and float64."""
    T._check_single_output(np.array([1.0, np.inf, np.nan]), "f")      # float64: skipped
    T._check_single_output(np.array([np.inf + 0j]), "f")              # complex128: skipped
    T._check_single_output(np.ones(3, np.float32), "f")               # finite: passes
    msg = ("f: 2 of 3 entries of the single-precision result are not finite (1 nan, 1 inf), "
           "from finite input: the computation overflowed float32, whose largest value is "
           "3.4e38. Pass float64 (or complex128) input to compute in double.")
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        T._check_single_output(np.array([1.0, np.inf, np.nan], np.float32), "f")
    with pytest.raises(ValueError, match=r"^f: 1 of 2 entries .* \(0 nan, 1 inf\)"):
        T._check_single_output(np.array([1.0, np.inf], np.complex64), "f")


def test_forward_bands_of_complex64_bins_at_a_large_dt_raises():
    """At ``dt = 1e4`` the window samples (max 400) scale complex64 bins of amplitude 1e36
    past float32's 3.4e38 inside ``forward_bands``, which returned NaN cells with only
    ``RuntimeWarning``s.  The cells themselves fit float32: complex128 bins give them."""
    z, call = _overflowing()["forward_bands"]
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="^forward_bands: .*float64"):
            call(z.astype(np.complex64), NUMPY)
    w = call(z, NUMPY)
    assert w.dtype == np.float64 and np.all(np.isfinite(w))
    assert np.max(np.abs(w)) < np.finfo(np.float32).max


def test_forward_time_of_a_record_that_overflows_its_rfft_names_forward_time():
    """The rfft of a float32 record of 1e37 overflows (bin 0 is 1.3e39).  ``forward``'s input
    check used to report it as a gap: "forward: x_rfft: ... Fill or excise the gaps"."""
    x = np.full(N, 1e37, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="^forward_time: .*float64") as info:
            T.forward_time(x, NF, NT, DT)
    assert "Fill or excise" not in str(info.value)


def test_inverse_time_that_overflows_only_in_its_irfft_names_inverse_time():
    """The cells of an impulse of height 1e37: ``inverse`` is finite (every bin has modulus
    1e37), but the irfft's unnormalised sum, N * 1e37 = 1.3e39, overflows float32."""
    w, _ = _overflowing()["inverse_time"]
    w32 = w.astype(np.float32)
    assert np.all(np.isfinite(T.inverse(w32, NF, NT, DT)))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="^inverse_time: .*float64"):
            T.inverse_time(w32, NF, NT, DT)


@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_a_single_precision_run_that_overflows_raises_naming_its_entry_point(name):
    """Every entry point checks the arrays it computes.  Its input is finite in single
    precision, so the input check does not fire; the double run is finite."""
    a, call = _overflowing()[name]
    a32 = _single(a)
    assert np.all(np.isfinite(a32))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match=f"^{name}: .* single-precision result .*float64"):
            call(a32, NUMPY)
    out = call(a, NUMPY)
    assert out.dtype == output_dtype(name, np.float64) and np.all(np.isfinite(out))


@pytest.mark.jax
@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_the_guard_runs_on_concrete_jax_arrays_and_is_skipped_under_jit(name, jax_be):
    """Eager JAX runs on concrete float32 arrays, so the guard reads them on the host as it
    reads NumPy ones.  Under ``jax.jit`` the arrays are tracers with no values: nothing
    raises, and the overflow comes back as ``inf``/``nan``."""
    import jax

    a, call = _overflowing()[name]
    a32 = jax_be.asarray(_single(a))
    with pytest.raises(ValueError, match=f"^{name}: .*float64"):
        call(a32, jax_be)
    out = jax.jit(lambda v: call(v, jax_be))(a32)
    assert isinstance(out, jax.Array) and out.dtype == output_dtype(name, np.float32)
    assert not np.all(np.isfinite(np.asarray(out)))
