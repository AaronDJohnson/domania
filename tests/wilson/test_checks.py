"""The shared argument checks of ``domainia.wilson._checks``: each helper
returns the value it accepts and refuses every other kind with ``TypeError`` and the message
``"{name} must be <kind>, got <type or dtype>"``; a wrong value is the caller's ``ValueError``."""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from domainia import domains, wilson
from domainia.wilson import grid
from domainia.wilson._checks import (
    REAL_KINDS,
    index_array,
    require_bool,
    require_callable,
    require_dtype,
    require_instance,
    require_int,
    require_real,
    require_str,
    validate_dt,
    validate_finite,
)
from domainia.wilson.filter import nu_d
from domainia.wilson.noise import colored_noise


def _exact(message):
    """A ``match=`` pattern for exactly ``message``."""
    return f"^{re.escape(message)}$"


def test_real_kinds_are_signed_and_unsigned_integers_and_floats():
    assert REAL_KINDS == "iuf"


@pytest.mark.parametrize("value", [3, -2, np.int64(3), np.int32(-2), np.uint8(7)])
def test_require_int_returns_a_python_int(value):
    out = require_int("n", value)
    assert type(out) is int and out == int(value)


@pytest.mark.parametrize(("value", "kind"), [
    (True, "bool"), (np.True_, "bool"), (2.0, "float"), (np.float64(2), "float64"),
    ("2", "str"), (None, "NoneType"), (2 + 0j, "complex"), (np.array(2), "ndarray"),
    (np.array([2]), "ndarray"),
])
def test_require_int_refuses_every_other_kind(value, kind):
    with pytest.raises(TypeError, match=_exact(f"n must be an integer, got {kind}")):
        require_int("n", value)


@pytest.mark.parametrize("value", [1, 0.5, np.float32(0.25), np.float64(0.5), np.int64(2), np.uint16(3),
                                   Fraction(1, 2), np.inf])
def test_require_real_returns_a_python_float(value):
    out = require_real("x", value)
    assert type(out) is float and out == float(value)


@pytest.mark.parametrize(("value", "kind"), [
    (True, "bool"), (np.True_, "bool"), ("0.5", "str"), (None, "NoneType"), (1 + 0j, "complex"),
    (np.complex64(1), "complex64"), (Decimal("0.5"), "Decimal"), (np.array(0.5), "ndarray"),
    (np.array([0.5]), "ndarray"),
])
def test_require_real_refuses_every_other_kind(value, kind):
    with pytest.raises(TypeError, match=_exact(f"x must be a real number, got {kind}")):
        require_real("x", value)


@pytest.mark.parametrize("value", [10**400, Fraction(10**400)])
def test_require_real_refuses_a_real_too_large_for_a_float_with_value_error(value):
    with pytest.raises(ValueError, match=_exact("x is too large for a float")):
        require_real("x", value)


@pytest.mark.parametrize(("value", "expected"), [(True, True), (False, False), (np.True_, True),
                                                 (np.False_, False)])
def test_require_bool_returns_a_python_bool(value, expected):
    out = require_bool("flag", value)
    assert type(out) is bool and out is expected


@pytest.mark.parametrize(("value", "kind"), [
    ("no", "str"), (1, "int"), (0, "int"), (np.int64(1), "int64"), (0.5, "float"), (None, "NoneType"),
    (np.array(True), "ndarray"),
])
def test_require_bool_refuses_every_other_kind(value, kind):
    with pytest.raises(TypeError, match=_exact(f"flag must be a bool, got {kind}")):
        require_bool("flag", value)


@pytest.mark.parametrize("value", ["fft", "", np.str_("dense")])
def test_require_str_returns_the_string(value):
    assert require_str("impl", value) is value


@pytest.mark.parametrize(("value", "kind"), [
    (3, "int"), (None, "NoneType"), (b"fft", "bytes"), (["fft"], "list"), (np.float64(1), "float64"),
])
def test_require_str_refuses_every_other_kind(value, kind):
    with pytest.raises(TypeError, match=_exact(f"impl must be a string, got {kind}")):
        require_str("impl", value)


def test_require_instance_checks_against_the_class_passed_in():
    g = wilson.Grid(8, 16, 1.0)
    assert require_instance("grid", g, wilson.Grid, "a domainia.wilson.Grid") is g
    assert require_instance("x", 2.5, (int, float), "a Python int or float") == 2.5
    with pytest.raises(TypeError, match=_exact("grid must be a domainia.wilson.Grid, got tuple")):
        require_instance("grid", (8, 16, 1.0), wilson.Grid, "a domainia.wilson.Grid")
    with pytest.raises(TypeError, match=_exact("x must be a Python int or float, got str")):
        require_instance("x", "2.5", (int, float), "a Python int or float")


def test_require_callable():
    assert require_callable("psd", np.ones_like) is np.ones_like
    assert require_callable("psd", wilson.Grid) is wilson.Grid
    with pytest.raises(TypeError, match=_exact("psd must be callable, got float")):
        require_callable("psd", 1.0)
    with pytest.raises(TypeError, match=_exact("psd must be callable, got NoneType")):
        require_callable("psd", None)


@pytest.mark.parametrize("a", [[1, 2], [0.5], 1.0, np.float64(1.0), np.zeros(2), np.zeros(2, np.float32),
                               np.zeros((2, 3), np.int8), np.zeros(2, np.uint64)])
def test_require_dtype_passes_a_real_numeric_list_scalar_or_array(a):
    assert require_dtype("x", a, REAL_KINDS, "a real numeric array") is None


@pytest.mark.parametrize(("a", "dtype"), [
    ([1j], "complex128"), ([True], "bool"), (["a"], "<U1"), (1j, "complex128"),
    (np.complex64(1), "complex64"), (np.zeros(2, complex), "complex128"), (np.zeros(2, bool), "bool"),
    (np.array(["a"]), "<U1"), (np.array([b"a"]), "|S1"), (np.array([None]), "object"),
])
def test_require_dtype_refuses_a_dtype_outside_the_kinds(a, dtype):
    with pytest.raises(TypeError, match=_exact(f"x must be a real numeric array, got dtype {dtype}")):
        require_dtype("x", a, REAL_KINDS, "a real numeric array")


def test_require_dtype_takes_any_kinds_and_reads_dtype_without_converting():
    assert require_dtype("keep", np.ones(3, bool), "b", "a bool array") is None
    with pytest.raises(TypeError, match=_exact("keep must be a bool array, got dtype float64")):
        require_dtype("keep", np.ones(3), "b", "a bool array")

    class Unconvertible:
        dtype = np.dtype(np.float32)

        def __array__(self, *args, **kwargs):
            raise AssertionError("require_dtype converted an object that has a dtype")

    assert require_dtype("x", Unconvertible(), REAL_KINDS, "a real numeric array") is None


@pytest.mark.jax
def test_require_dtype_passes_a_jax_array_and_a_jit_tracer_through_untouched(jax_be):
    """A conversion would show: ``np.asarray`` on a tracer raises ``TracerArrayConversionError``,
    a ``TypeError`` without the helper's message."""
    import jax
    import jax.numpy as jnp

    from domainia.wilson.backend import is_tracer

    x = jnp.arange(3.0)
    assert require_dtype("x", x, REAL_KINDS, "a real numeric array") is None
    with pytest.raises(TypeError, match=_exact("x must be a real numeric array, got dtype complex128")):
        require_dtype("x", x + 0j, REAL_KINDS, "a real numeric array")
    traced = []

    @jax.jit
    def double(v):
        require_dtype("v", v, REAL_KINDS, "a real numeric array")
        traced.append(is_tracer(v))
        return 2.0 * v

    out = double(x)
    assert traced == [True] and isinstance(out, jax.Array)
    assert np.array_equal(np.asarray(out), [0.0, 2.0, 4.0])
    with pytest.raises(TypeError) as info:              # raised while tracing; JAX adds a note
        double(x + 0j)
    assert str(info.value) == "v must be a real numeric array, got dtype complex128"


@pytest.mark.parametrize(("value", "expected"), [
    ([1, 2], [1, 2]), (np.array([1, 2], np.int32), [1, 2]), (np.array([3], np.uint8), [3]),
    ([1, 2.0], [1, 2]), (np.array([[0.0, 4.0]], np.float32), [[0, 4]]), (3, 3), ([], []),
])
def test_index_array_returns_int64(value, expected):
    out = index_array("m", value)
    assert isinstance(out, np.ndarray) and out.dtype == np.int64
    assert out.shape == np.shape(expected) and np.array_equal(out, expected)


@pytest.mark.parametrize(("value", "first"), [
    ([1.5], "1.5"), ([2.9999999], "2.9999999"), ([0.0, np.nan], "nan"), ([np.inf], "inf"),
    ([[1.0], [-0.5]], "-0.5"),
])
def test_index_array_refuses_a_non_integral_or_non_finite_float_with_value_error(value, first):
    with pytest.raises(ValueError, match="integer") as info:
        index_array("m", value)
    assert str(info.value) == f"m must hold integer indices, got {first}"


@pytest.mark.parametrize(("value", "dtype"), [
    ([True], "bool"), (np.array([1, 0], bool), "bool"), ([1j], "complex128"), (["1"], "<U1"),
    ([b"1"], "|S1"), (np.array([1], object), "object"),
])
def test_index_array_refuses_a_non_numeric_dtype_with_type_error(value, dtype):
    with pytest.raises(TypeError, match=_exact(f"m must be an integer index array, got dtype {dtype}")):
        index_array("m", value)


def test_validate_dt_and_validate_finite():
    assert validate_dt(np.float32(0.5)) is None and validate_dt(Fraction(1, 2)) is None
    with pytest.raises(TypeError, match=_exact("dt must be a real number, got Decimal")):
        validate_dt(Decimal("0.5"))
    with pytest.raises(ValueError, match=_exact("dt must be a positive finite number, got -1")):
        validate_dt(-1)
    with pytest.raises(ValueError, match=_exact("dt is too large for a float")):
        validate_dt(10**400)
    assert validate_finite([1.0, 2.0], "x") is None and validate_finite([], "x") is None
    with pytest.raises(ValueError, match=r"^y: 1 of 3 entries are not finite \(0 nan, 1 inf\); "
                                         r"first at index 2\. "):
        validate_finite([0.0, 1j, complex(1.0, np.inf)], "y")


def test_grid_re_exports_the_checks_under_their_old_names():
    """``wilson.grid`` keeps ``_require_int``, ``validate_dt`` and ``validate_finite`` (imported
    from ``grid`` by ``matrix.py``, ``evolutionary.py`` and the tests), and its ``__all__``."""
    assert grid._require_int is require_int
    assert grid.validate_dt is validate_dt is wilson.validate_dt
    assert grid.validate_finite is validate_finite is wilson.validate_finite
    assert {"validate_dt", "validate_finite", "validate_grid", "n_samples"} <= set(grid.__all__)
    assert not [name for name in grid.__all__ if name.startswith("_")]


def _noise(N):
    return colored_noise(N, 1.0, np.ones_like, np.random.default_rng(0))


@pytest.mark.parametrize(("call", "message"), [
    (lambda: grid._require_int("N", 8.0), "N must be an integer, got float"),
    (lambda: wilson.validate_grid(8, np.float64(16)), "Nt must be an integer, got float64"),
    (lambda: wilson.n_samples(True, 16), "Nf must be an integer, got bool"),
    (lambda: nu_d(np.array([0.5]), d="4"), "d must be an integer, got str"),
    (lambda: _noise(np.array(8)), "N must be an integer, got ndarray"),
    (lambda: wilson.Meyer(2.5), "Meyer d must be an integer, got float"),
    (lambda: domains.TimeDomain(8.0, 1.0), "N must be an integer, got float"),
], ids=["_require_int", "validate_grid", "n_samples", "nu_d", "colored_noise", "Meyer", "domains._int"])
def test_the_seven_integer_checks_raise_the_helpers_message(call, message):
    """The seven copies of the integer check call ``require_int``."""
    with pytest.raises(TypeError, match=_exact(message)):
        call()


def test_the_range_checks_after_require_int_are_unchanged():
    assert type(wilson.Meyer(np.int64(3)).d) is int and wilson.Meyer(np.int64(3)) == wilson.Meyer(3)
    with pytest.raises(ValueError, match=_exact("Meyer d must be >= 1; got 0")):
        wilson.Meyer(0)
    assert type(domains._int("k0", np.int64(2), minimum=0)) is int
    with pytest.raises(ValueError, match=_exact("k0 must be >= 0; got -1")):
        domains._int("k0", -1, minimum=0)
    with pytest.raises(TypeError, match=_exact("k0 must be an integer, got bool")):
        domains._int("k0", np.True_, minimum=0)
    with pytest.raises(ValueError, match=_exact("d must be >= 1, got 0")):
        nu_d(np.array([0.5]), d=0)
    with pytest.raises(ValueError, match=_exact("N must be even, got 7")):
        _noise(7)
    assert _noise(np.int64(8)).shape == (5,)
