"""``domainia.wilson`` refuses a wrong kind with ``TypeError`` and a wrong value with ``ValueError``,
as the rest of domainia does."""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from domainia import wilson
from domainia.wilson.filter import nu_d, phi_centered
from domainia.wilson.grid import _require_int, n_samples, validate_dt, validate_grid


@pytest.mark.parametrize(("args", "match"), [
    ((8.0, 16), "Nf must be an integer, got float"),
    ((8, 16.0), "Nt must be an integer, got float"),
    ((True, 16), "Nf must be an integer, got bool"),
    ((np.float64(8), 16), "Nf must be an integer, got float64"),
    (("8", 16), "Nf must be an integer, got str"),
])
@pytest.mark.parametrize("check", [validate_grid, n_samples], ids=["validate_grid", "n_samples"])
def test_grid_sizes_of_the_wrong_kind_raise_type_error(check, args, match):
    with pytest.raises(TypeError, match=match):
        check(*args)


@pytest.mark.parametrize(("args", "match"), [
    ((9, 16), "Nf must be even, got 9"),
    ((0, 16), "Nf must be positive, got 0"),
    ((8, -2), "Nt must be positive, got -2"),
])
def test_grid_sizes_out_of_range_stay_value_errors(args, match):
    with pytest.raises(ValueError, match=match):
        validate_grid(*args)
    assert n_samples(np.int64(8), 16) == 128 and type(n_samples(np.int64(8), 16)) is int


def test_require_int_and_its_callers_raise_type_error():
    with pytest.raises(TypeError, match="N must be an integer, got float"):
        _require_int("N", 32.0)
    with pytest.raises(TypeError, match="N must be an integer, got bool"):
        _require_int("N", False)
    assert _require_int("N", np.int32(32)) == 32
    with pytest.raises(TypeError, match="N must be an integer, got float"):
        wilson.matrix.rfft_to_real(np.zeros(17, complex), 32.0)
    with pytest.raises(TypeError, match="Nf must be an integer, got float"):
        wilson.delta_F(8.0, 1.0)
    with pytest.raises(ValueError, match="N must be even, got 5"):
        wilson.matrix.rfft_to_real(np.zeros(3, complex), 5)
    with pytest.raises(TypeError, match="Nf must be an integer, got float"):
        wilson.Grid(8.0, 16, 1.0)


@pytest.mark.parametrize(("dt", "kind"), [
    (True, "bool"), (np.True_, "bool"), ("0.5", "str"), (None, "NoneType"),
    (1.0 + 0.0j, "complex"), (np.complex64(1.0), "complex64"), (np.array(1.0 + 0.0j), "ndarray"),
    (Decimal("0.5"), "Decimal"), (np.array(0.5), "ndarray"), (np.array([0.5]), "ndarray"),
])
def test_dt_of_the_wrong_kind_raises_type_error(dt, kind):
    """``float()`` used to accept ``True`` (as 1.0), ``"0.5"`` and a NumPy complex scalar; the
    old check also let a ``Decimal``, a 0-d array and a size-1 array through."""
    with pytest.raises(TypeError, match=f"dt must be a real number, got {kind}"):
        validate_dt(dt)
    with pytest.raises(TypeError, match="dt must be a real number"):
        wilson.forward_time(np.zeros(128), 8, 16, dt)


@pytest.mark.parametrize("dt", [0.5, 1, np.float32(0.1), np.float64(0.25), np.int64(2), Fraction(1, 2)])
def test_real_dt_kinds_still_pass(dt):
    validate_dt(dt)


@pytest.mark.parametrize("dt", [0.0, -1.0, np.inf, np.nan])
def test_dt_out_of_range_stays_a_value_error(dt):
    with pytest.raises(ValueError, match="dt must be a positive finite number"):
        validate_dt(dt)


def test_nu_d_splits_the_kind_and_range_checks():
    x = np.array([0.5])
    with pytest.raises(TypeError, match="d must be an integer, got float"):
        nu_d(x, d=2.5)
    with pytest.raises(TypeError, match="d must be an integer, got bool"):
        nu_d(x, d=True)
    with pytest.raises(ValueError, match="d must be >= 1, got 0"):
        nu_d(x, d=0)
    with pytest.raises(ValueError, match="d must be >= 1, got -1"):
        nu_d(x, d=-1)
    assert nu_d(x, d=np.int64(4)).item() == 0.5
    with pytest.raises(TypeError, match="d must be an integer, got float"):
        phi_centered(8, 16, 1.0, d=2.5)


@pytest.mark.parametrize("N", [8.0, True])
def test_build_C_time_N_of_the_wrong_kind_raises_type_error(N):
    with pytest.raises(TypeError, match="N must be an integer"):
        wilson.matrix.build_C_time(N, 1.0, lambda f: np.ones_like(f))


def test_build_C_time_range_and_numpy_int():
    with pytest.raises(ValueError, match="N must be a positive even integer, got 5"):
        wilson.matrix.build_C_time(5, 1.0, lambda f: np.ones_like(f))
    assert wilson.matrix.build_C_time(np.int64(8), 1.0, lambda f: np.ones_like(f)).shape == (8, 8)


@pytest.mark.parametrize("P", [2.7, True])
def test_eq45_plan_P_of_the_wrong_kind_raises_type_error(P):
    with pytest.raises(TypeError, match="P must be an integer"):
        wilson.evolutionary.eq45_plan(8, 16, 1.0, P=P)


def test_eq45_plan_P_range_and_numpy_int():
    with pytest.raises(ValueError, match="P must be >= 1, got 0"):
        wilson.evolutionary.eq45_plan(8, 16, 1.0, P=0)
    assert wilson.evolutionary.eq45_plan(8, 16, 1.0, P=np.int64(4)).P == 4


# ---------------------------------------------------------------------------
# The other wilson sites
# ---------------------------------------------------------------------------

_T = wilson.transform
_Z = np.fft.rfft(np.random.default_rng(0).standard_normal(128))     # rfft bins of an 8 x 16 grid
_BANDS = np.array([2, 3])
_W_BANDS = _T.forward_bands(_Z, 0, _BANDS, 8, 16, 1.0)
_W = wilson.forward_time(np.random.default_rng(1).standard_normal(128), 8, 16, 1.0)


def _exact(message):
    """A ``match=`` pattern for exactly ``message``."""
    return f"^{re.escape(message)}$"


#: Integer slots: the call taking the slot's value, and a value it accepts.
_INT_SLOTS = {
    "forward_bands k0": (lambda v: _T.forward_bands(_Z, v, _BANDS, 8, 16, 1.0), 0),
    "inverse_bands k0": (lambda v: _T.inverse_bands(_W_BANDS, v, 65, _BANDS, 8, 16, 1.0), 0),
    "inverse_bands n_bins": (lambda v: _T.inverse_bands(_W_BANDS, 0, v, _BANDS, 8, 16, 1.0), 65),
    "atom_spectrum m": (lambda v: _T.atom_spectrum(v, 3, 8, 16, 1.0), 2),
    "atom_spectrum n": (lambda v: _T.atom_spectrum(2, v, 8, 16, 1.0), 3),
    "carrier_phase m": (lambda v: wilson.grid.carrier_phase(v, 4), 1),
    "edge_columns lead": (lambda v: wilson.grid.edge_columns(16, v, 1), 2),
    "edge_columns trail": (lambda v: wilson.grid.edge_columns(16, 2, v), 1),
    "empirical_variance n_realizations": (lambda v: wilson.empirical_variance(8, 16, 1.0, np.ones_like, v,
                                                                              seed=0), 2),
}


@pytest.mark.parametrize(("value", "kind"), [
    (2.0, "float"), (np.float64(2), "float64"), (True, "bool"), (np.True_, "bool"), ("2", "str"),
    (None, "NoneType"),
], ids=["float", "np.float64", "bool", "np.bool_", "str", "None"])
@pytest.mark.parametrize("slot", list(_INT_SLOTS))
def test_integer_slots_of_the_wrong_kind_raise_type_error(slot, value, kind):
    """``int()``, ``range()`` or a parity test took these silently (``k0="0"`` and
    ``carrier_phase(2.5, 4)`` ran), or ``True`` passed as 1."""
    call, _ = _INT_SLOTS[slot]
    with pytest.raises(TypeError, match=_exact(f"{slot.split()[-1]} must be an integer, got {kind}")):
        call(value)


@pytest.mark.parametrize("slot", list(_INT_SLOTS))
def test_integer_slots_take_a_numpy_integer(slot):
    call, good = _INT_SLOTS[slot]
    assert np.array_equal(np.asarray(call(np.int64(good))), np.asarray(call(good)))


@pytest.mark.parametrize(("call", "match"), [
    (lambda: _T.forward_bands(_Z, 50, _BANDS, 8, 16, 1.0), r"^the bins \[50, 115\) do not cover"),
    (lambda: _T.inverse_bands(_W_BANDS, 0, 20, _BANDS, 8, 16, 1.0), r"^the bins \[0, 20\) do not cover"),
    (lambda: _T.atom_spectrum(2, 16, 8, 16, 1.0), r"^n must lie in 0\.\.15"),
    (lambda: wilson.empirical_variance(8, 16, 1.0, np.ones_like, 0), "^n_realizations must be >= 1, got 0"),
], ids=["forward_bands k0", "inverse_bands n_bins", "atom_spectrum n", "empirical_variance"])
def test_integer_slots_out_of_range_stay_value_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call()


#: The three kernels that take a band list, through ``_check_interior_bands``.
_BAND_SLOTS = {
    "band_atoms": lambda b: _T.band_atoms(b, 8, 16, 1.0),
    "forward_bands": lambda b: _T.forward_bands(_Z, 0, b, 8, 16, 1.0),
    "inverse_bands": lambda b: _T.inverse_bands(_W_BANDS, 0, 65, b, 8, 16, 1.0),
}


@pytest.mark.parametrize(("bands", "dtype"), [
    ([True, True], "bool"), (["2", "3"], "<U1"), ([2 + 0j, 3 + 0j], "complex128"),
    (np.array([2, 3], object), "object"),
], ids=["bool", "str", "complex", "object"])
@pytest.mark.parametrize("entry", list(_BAND_SLOTS))
def test_bands_of_the_wrong_dtype_raise_type_error(entry, bands, dtype):
    """``np.asarray(bands, dtype=int)`` took ``[True, True]`` as band 1 and parsed ``["2", "3"]``."""
    with pytest.raises(TypeError, match=_exact(f"bands must be an integer index array, got dtype {dtype}")):
        _BAND_SLOTS[entry](bands)


@pytest.mark.parametrize(("bands", "first"), [([2.5, 3.0], "2.5"), ([2.0, np.nan], "nan")],
                         ids=["2.5", "nan"])
@pytest.mark.parametrize("entry", list(_BAND_SLOTS))
def test_bands_of_non_integral_floats_raise_value_error(entry, bands, first):
    """``[2.5, 3.0]`` used to be truncated to bands 2 and 3 silently."""
    with pytest.raises(ValueError, match=_exact(f"bands must hold integer indices, got {first}")):
        _BAND_SLOTS[entry](bands)


@pytest.mark.parametrize("entry", list(_BAND_SLOTS))
def test_bands_take_integer_and_integral_float_arrays(entry):
    want = np.asarray(_BAND_SLOTS[entry](_BANDS))
    for bands in ([2, 3], np.array([2, 3], np.int64), np.array([2.0, 3.0]), [np.float64(2), np.float64(3)]):
        assert np.array_equal(np.asarray(_BAND_SLOTS[entry](bands)), want)
    with pytest.raises(ValueError, match=r"^bands must be interior bands 1\.\.7; got \[0, 3\]$"):
        _BAND_SLOTS[entry]([0, 3])


@pytest.mark.parametrize(("t_max", "kind"), [
    ("128", "str"), (True, "bool"), (128 + 0j, "complex"), (Decimal("128"), "Decimal"),
    (np.array(128.0), "ndarray"),
], ids=["str", "bool", "complex", "Decimal", "0-d array"])
def test_eq45_plan_t_max_of_the_wrong_kind_raises_type_error(t_max, kind):
    """``float()`` parsed ``t_max="128"`` and took a 0-d array."""
    with pytest.raises(TypeError, match=_exact(f"t_max must be a real number, got {kind}")):
        wilson.evolutionary.eq45_plan(8, 16, 1.0, t_max=t_max)


def test_eq45_plan_t_max_takes_numpy_reals_and_is_stored_as_a_float():
    ref = wilson.evolutionary.eq45_plan(8, 16, 1.0)
    for t_max in (128, np.float64(128.0), np.int64(128), Fraction(128)):
        plan = wilson.evolutionary.eq45_plan(8, 16, 1.0, t_max=t_max)
        assert type(plan.t_max) is float and plan.t_max == ref.t_max == 128.0
        assert np.array_equal(plan.Tp, ref.Tp)


def test_eq45_plan_keep_message_names_the_dtype():
    with pytest.raises(TypeError, match=_exact("keep must be a boolean mask over the packed cells, "
                                               "got dtype int64")):
        wilson.evolutionary.eq45_plan(8, 16, 1.0, np.ones(128, np.int64))


#: The real-array checks: the call, an accepted float64 array, and the name in
#: the message.
_REAL_ARRAYS = {
    "inverse wmn": (lambda a: wilson.inverse(a, 8, 16, 1.0), _W, "wmn"),
    "inverse_many wmn": (lambda a: wilson.inverse_many(a, 8, 16, 1.0), _W[None], "wmn"),
    "inverse_time wmn": (lambda a: wilson.inverse_time(a, 8, 16, 1.0), _W, "wmn"),
    "inverse_time_many wmn": (lambda a: wilson.inverse_time_many(a, 8, 16, 1.0), _W[None], "wmn"),
    "inverse_bands w": (lambda a: _T.inverse_bands(a, 0, 65, _BANDS, 8, 16, 1.0), _W_BANDS, "w"),
    "sanitize_psd": (lambda a: wilson.grid.sanitize_psd(a, "demo"), np.ones(5), "demo: PSD"),
}


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64, np.bool_, np.str_, object],
                         ids=["complex128", "complex64", "bool", "str", "object"])
@pytest.mark.parametrize("slot", list(_REAL_ARRAYS))
def test_real_arrays_of_the_wrong_dtype_raise_type_error(slot, dtype):
    """Complex used to be a ``ValueError``; a bool grid ran, and a bool, str or object PSD was cast."""
    call, good, name = _REAL_ARRAYS[slot]
    bad = good.astype(dtype)
    message = f"{name} must be a real numeric array, got dtype {bad.dtype}"
    with pytest.raises(TypeError, match=_exact(message)):
        call(bad)


@pytest.mark.parametrize("slot", list(_REAL_ARRAYS))
def test_real_arrays_take_integer_and_float_dtypes(slot):
    call, good, _ = _REAL_ARRAYS[slot]
    ints = np.round(10 * good).astype(np.int64)
    assert np.array_equal(call(ints), call(ints.astype(np.float64)))


def test_float32_input_is_computed_in_double():
    """Every array input is promoted to double before the first FFT: a float32 or complex64
    input gives the double run's result on the same values, dtype included.  forward_time's
    rfft would otherwise run in single precision."""
    x = np.fft.irfft(_Z, 128)
    calls = {slot: (call, good) for slot, (call, good, _) in _REAL_ARRAYS.items()} | {
        "forward": (lambda a: wilson.forward(a, 8, 16, 1.0), _Z),
        "forward_many": (lambda a: wilson.forward_many(a, 8, 16, 1.0), _Z[None]),
        "forward_time": (lambda a: wilson.forward_time(a, 8, 16, 1.0), x),
        "forward_time_many": (lambda a: wilson.forward_time_many(a, 8, 16, 1.0), x[None]),
        "forward_bands": (lambda a: _T.forward_bands(a, 0, _BANDS, 8, 16, 1.0), _Z),
    }
    for slot, (call, good) in calls.items():
        low = good.astype(np.complex64 if good.dtype.kind == "c" else np.float32)
        got, want = call(low), call(low.astype(good.dtype))
        assert got.dtype == want.dtype in (np.float64, np.complex128), slot
        assert np.array_equal(got, want), slot


def test_the_stationary_psd_check_uses_the_sanitize_psd_message():
    psd = np.ones(65, complex)
    with pytest.raises(TypeError, match=_exact("stationary noise input: PSD must be a real numeric array, "
                                               "got dtype complex128")):
        wilson.stationary.build_K_full_active(8, 16, 1.0, psd)
    with pytest.raises(TypeError, match=_exact("stationary noise input: PSD must be a real numeric array, "
                                               "got dtype bool")):
        wilson.stationary.build_K_full_active(8, 16, 1.0, np.ones(65, bool))


class _UnhashableRamp:
    __hash__ = None

    def __call__(self, x):
        return x


@pytest.mark.parametrize(("make", "message"), [
    (lambda: wilson.RampWindow(np.sqrt, 3), "window name must be a non-empty string, got 3"),
    (lambda: wilson.RampWindow("x", "r"), "ramp must be callable, got str"),
    (lambda: wilson.RampWindow(_UnhashableRamp(), "r"),
     "ramp must be hashable (a window is a cache key), got an unhashable _UnhashableRamp: use a "
     "module-level function or a frozen dataclass instance"),
    (lambda: wilson.windows.require_window(wilson.Meyer),
     "window must be a Window instance, got the class Meyer: pass an instance, e.g. Meyer() or "
     "domainia.wilson.MEYER"),
    (lambda: wilson.windows.require_window("meyer"),
     "window must be a domainia.wilson Window (Meyer, RampWindow, SampledWindow), got str"),
], ids=["name", "callable", "hashable", "class", "non-window"])
def test_the_window_kind_messages_take_the_got_form(make, message):
    with pytest.raises(TypeError, match=_exact(message)):
        make()
