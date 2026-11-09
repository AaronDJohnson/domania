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
# The other wilson sites (spec 2026-11-08 §2.3-§2.5, §3.5)
# ---------------------------------------------------------------------------

_T, _IP = wilson.transform, wilson.inner_product
_Z = np.fft.rfft(np.random.default_rng(0).standard_normal(128))     # rfft bins of an 8 x 16 grid
_BANDS = np.array([2, 3])
_W_BANDS = _T.forward_bands(_Z, 0, _BANDS, 8, 16, 1.0)
_W = wilson.forward_time(np.random.default_rng(1).standard_normal(128), 8, 16, 1.0)
_VAR = wilson.whittle_variance(8, 16, 1.0, np.ones_like)
_SN = np.ones(65)


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


#: The loglike functions that take ``include_logdet``.
_LOGLIKE = {
    "loglike_whittle": lambda flag: _IP.loglike_whittle(_W, 0 * _W, _VAR, 8, include_logdet=flag),
    "loglike_whittle_many": lambda flag: _IP.loglike_whittle_many(_W, np.zeros((2, 9, 16)), _VAR, 8,
                                                                  include_logdet=flag),
    "loglike_band_averaged": lambda flag: _IP.loglike_band_averaged(_W, 0 * _W, _VAR, 8,
                                                                    include_logdet=flag),
    "loglike_evolutionary_whittle": lambda flag: _IP.loglike_evolutionary_whittle(_W, 0 * _W, _VAR, 8,
                                                                                  include_logdet=flag),
    "loglike_gaussian": lambda flag: _IP.loglike_gaussian(_W, 0 * _W, 8, 16, 1.0, _SN, include_logdet=flag),
    "loglike_gaussian_many": lambda flag: _IP.loglike_gaussian_many(_W, np.zeros((2, 9, 16)), 8, 16, 1.0,
                                                                    _SN, include_logdet=flag),
}


@pytest.mark.parametrize(("flag", "kind"), [
    ("no", "str"), (0, "int"), (1, "int"), (np.int64(1), "int64"), (0.5, "float"), (None, "NoneType"),
], ids=["str", "0", "1", "np.int64", "float", "None"])
@pytest.mark.parametrize("entry", list(_LOGLIKE))
def test_include_logdet_of_the_wrong_kind_raises_type_error(entry, flag, kind):
    """A truthy ``"no"`` used to include the log-determinant."""
    with pytest.raises(TypeError, match=_exact(f"include_logdet must be a bool, got {kind}")):
        _LOGLIKE[entry](flag)


@pytest.mark.parametrize("entry", list(_LOGLIKE))
def test_include_logdet_takes_a_numpy_bool(entry):
    for flag in (True, False):
        assert np.array_equal(_LOGLIKE[entry](np.bool_(flag)), _LOGLIKE[entry](flag))
    assert not np.array_equal(_LOGLIKE[entry](True), _LOGLIKE[entry](False))


_CELLS = {
    "whittle": lambda c: _IP.whittle(_W, _W, _VAR, 8, cells=c),
    "whittle_many": lambda c: _IP.whittle_many(_W[None], _W[None], _VAR, 8, cells=c),
    "loglike_whittle": lambda c: _IP.loglike_whittle(_W, 0 * _W, _VAR, 8, cells=c),
    "loglike_whittle_many": lambda c: _IP.loglike_whittle_many(_W, np.zeros((2, 9, 16)), _VAR, 8, cells=c),
}


@pytest.mark.parametrize(("cells", "kind"), [(1, "int"), (None, "NoneType"), (b"active", "bytes"),
                                             (("active",), "tuple")], ids=["int", "None", "bytes", "tuple"])
@pytest.mark.parametrize("entry", list(_CELLS))
def test_cells_of_the_wrong_kind_raise_type_error(entry, cells, kind):
    with pytest.raises(TypeError, match=_exact(f"cells must be a string, got {kind}")):
        _CELLS[entry](cells)


@pytest.mark.parametrize("entry", list(_CELLS))
def test_cells_keep_their_choices(entry):
    for cells in ("active", "interior"):
        assert np.array_equal(_CELLS[entry](np.str_(cells)), _CELLS[entry](cells))
    with pytest.raises(ValueError, match=_exact("cells must be 'active' or 'interior'; got 'edges'")):
        _CELLS[entry]("edges")


@pytest.mark.parametrize(("value", "kind"), [(1, "int"), (b"numpy", "bytes"), (True, "bool"),
                                             (np, "module")], ids=["int", "bytes", "bool", "module"])
def test_get_backend_backend_of_the_wrong_kind_raises_type_error(value, kind):
    with pytest.raises(TypeError, match=_exact(f"backend must be a string, got {kind}")):
        wilson.backend.get_backend(np.zeros(2), backend=value)


def test_get_backend_keeps_its_choices():
    get_backend, NUMPY = wilson.backend.get_backend, wilson.backend.NUMPY
    for backend in (None, "numpy", np.str_("numpy"), NUMPY):
        assert get_backend(np.zeros(2), backend=backend) is NUMPY
    message = "backend must be 'numpy', 'jax' or a Backend; got 'torch'"
    with pytest.raises(ValueError, match=_exact(message)):
        get_backend(backend="torch")


@pytest.mark.parametrize(("impl", "kind"), [(1, "int"), (None, "NoneType"), (b"numpy", "bytes")],
                         ids=["int", "None", "bytes"])
def test_cholesky_impl_of_the_wrong_kind_raises_type_error(backend, impl, kind):
    with pytest.raises(TypeError, match=_exact(f"impl must be a string, got {kind}")):
        backend.cholesky(backend.asarray(np.eye(2)), impl=impl)


def test_cholesky_keeps_its_choices(backend):
    A = backend.asarray(np.array([[4.0, 2.0], [2.0, 3.0]]))
    for impl in ("numpy", "scipy"):
        L, ok = backend.cholesky(A, impl=np.str_(impl))
        assert bool(ok) and np.allclose(np.asarray(L) @ np.asarray(L).T, np.asarray(A))
    with pytest.raises(ValueError, match=_exact("impl must be 'numpy' or 'scipy'; got 'lapack'")):
        backend.cholesky(A, impl="lapack")


#: The real-array checks of spec §3.5: the call, an accepted float64 array, and the name in
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
     "ramp must be hashable (a window is a cache key and static under jax.jit), got an unhashable "
     "_UnhashableRamp: use a module-level function or a frozen dataclass instance"),
    (lambda: wilson.windows.require_window(wilson.Meyer),
     "window must be a Window instance, got the class Meyer: pass an instance, e.g. Meyer() or "
     "domainia.wilson.MEYER"),
    (lambda: wilson.windows.require_window("meyer"),
     "window must be a domainia.wilson Window (Meyer, RampWindow, SampledWindow), got str"),
], ids=["name", "callable", "hashable", "class", "non-window"])
def test_the_window_kind_messages_take_the_got_form(make, message):
    with pytest.raises(TypeError, match=_exact(message)):
        make()


@pytest.mark.jax
def test_the_real_array_checks_read_a_jit_tracers_dtype(jax_be):
    """Under ``jax.jit`` the checks see tracers: a real grid passes, a complex one raises
    ``TypeError`` while tracing (JAX adds a note, so the message is compared whole)."""
    import jax
    import jax.numpy as jnp

    inv = jax.jit(lambda w: wilson.inverse(w, 8, 16, 1.0))
    inv_bands = jax.jit(lambda w: _T.inverse_bands(w, 0, 65, _BANDS, 8, 16, 1.0))
    fwd_bands = jax.jit(lambda z: _T.forward_bands(z, np.int64(0), _BANDS, 8, 16, 1.0))
    want_bands = _T.inverse_bands(_W_BANDS, 0, 65, _BANDS, 8, 16, 1.0)
    for got, want in ((inv(jnp.asarray(_W)), wilson.inverse(_W, 8, 16, 1.0)),
                      (inv_bands(jnp.asarray(_W_BANDS)), want_bands),
                      (fwd_bands(jnp.asarray(_Z)), _W_BANDS)):
        assert isinstance(got, jax.Array) and np.allclose(np.asarray(got), want, rtol=0, atol=1e-12)
    for jitted, arg, name in ((inv, _W, "wmn"), (inv_bands, _W_BANDS, "w")):
        with pytest.raises(TypeError) as info:
            jitted(jnp.asarray(arg) + 0j)
        assert str(info.value) == f"{name} must be a real numeric array, got dtype complex128"
