"""``domainia.wilson`` refuses a wrong kind with ``TypeError`` and a wrong value with ``ValueError``,
as the rest of domainia does."""

from __future__ import annotations

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
])
def test_dt_of_the_wrong_kind_raises_type_error(dt, kind):
    """``float()`` used to accept ``True`` (as 1.0), ``"0.5"`` and a NumPy complex scalar."""
    with pytest.raises(TypeError, match=f"dt must be a real number, got {kind}"):
        validate_dt(dt)
    with pytest.raises(TypeError, match="dt must be a real number"):
        wilson.forward_time(np.zeros(128), 8, 16, dt)


@pytest.mark.parametrize("dt", [0.5, 1, np.float32(0.1), np.float64(0.25), np.int64(2), np.array(0.5)])
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
