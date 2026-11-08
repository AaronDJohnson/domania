"""Single precision end to end in ``domainia.wilson.transform``.

A float32 / complex64 input runs in single precision: every FFT receives a single-precision
array, and every product stored into the output is formed in the working dtype.  float64 runs
are bit for bit unchanged (``tests/wilson/test_meyer_baseline.py``).
"""

from __future__ import annotations

import dataclasses

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


@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_output_dtypes_follow_the_table(name, precision, rng):
    assert _cases(precision, rng)[name](NUMPY).dtype == output_dtype(name, precision)


@pytest.mark.parametrize("name", list(OUTPUT_DTYPE))
def test_float32_runs_match_float64_to_single_precision_rounding(name):
    """Same inputs (rounded to float32 once), two precisions: the casts change rounding only."""
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
