"""Admissible Wilson windows: Meyer(d), RampWindow, SampledWindow (spec section 2)."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from domainia import wilson
from domainia.testing import ALT_WINDOW, smoothstep, wide_profile
from domainia.wilson import MEYER, Meyer, RampWindow, SampledWindow, Window, WindowError
from domainia.wilson.filter import nu_d, phi_centered
from domainia.wilson.windows import sample_window

GRIDS = [(8, 16, 1.0), (16, 32, 0.25), (32, 8, 2.0), (64, 256, 2.0**-10)]


@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
@pytest.mark.parametrize("d", range(1, 9))
def test_meyer_sample_is_phi_centered_bit_for_bit(Nf, Nt, dt, d):
    assert np.array_equal(Meyer(d).sample(Nf, Nt, dt), phi_centered(Nf, Nt, dt, d))


def test_samples_are_cached_and_read_only():
    a = MEYER.sample(8, 16, 1.0)
    assert MEYER.sample(8, 16, 1.0) is a
    assert not a.flags.writeable
    with pytest.raises(ValueError):
        a[0] = 1.0


def test_meyer_identity():
    assert MEYER == Meyer() == Meyer(4) and hash(MEYER) == hash(Meyer(4))
    assert Meyer(5) != MEYER
    assert MEYER.name == "meyer(d=4)"
    assert isinstance(MEYER, Window)


@pytest.mark.parametrize("d, err", [(0, ValueError), (-1, ValueError), (2.5, TypeError), (True, TypeError)])
def test_meyer_rejects_a_bad_d(d, err):
    with pytest.raises(err):
        Meyer(d)


def test_a_ramp_window_with_the_meyer_ramp_is_meyer_bit_for_bit():
    w = RampWindow(partial(nu_d, d=4), "nu4")
    for Nf, Nt, dt in GRIDS:
        assert np.array_equal(w.sample(Nf, Nt, dt), MEYER.sample(Nf, Nt, dt))


@pytest.mark.parametrize("window", [MEYER, Meyer(1), Meyer(8), RampWindow(smoothstep, "smoothstep"),
                                    ALT_WINDOW], ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_admissible_windows_are_even_supported_and_a_partition_of_unity(window, Nf, Nt, dt):
    phi = window.sample(Nf, Nt, dt)
    h = Nt // 2
    inv_dF = 1.0 / wilson.delta_F(Nf, dt)
    assert phi.shape == (Nt,) and phi.dtype == np.float64 and phi[0] == 0.0
    assert np.array_equal(phi[h + 1:], phi[h - 1:0:-1])
    np.testing.assert_allclose(phi[h:] ** 2 + phi[:h] ** 2, inv_dF, rtol=1e-12, atol=0)


def test_the_alt_window_differs_from_meyer():
    assert not np.allclose(ALT_WINDOW.sample(8, 16, 1.0), MEYER.sample(8, 16, 1.0))


def test_evenness_failure_names_the_condition_window_and_bin():
    w = SampledWindow(lambda u: wide_profile(u) * np.where(np.asarray(u) < 0, 1.0 - 1e-3, 1.0), "lopsided")
    with pytest.raises(WindowError, match=r"lopsided.*evenness.*bin"):
        w.sample(8, 16, 1.0)


def test_support_failure_at_the_slab_edge_is_caught_at_construction():
    # A SampledWindow can never reach the sample-time bin-0 support branch: psi(+-1) is
    # checked at construction.  That branch is exercised by `Flat` in
    # test_sample_window_validates_third_party_windows; do not re-add a boxcar here.
    with pytest.raises(WindowError, match=r"boxcar.*support"):
        SampledWindow(lambda u: np.where(np.abs(np.asarray(u)) <= 1.0, 0.5, 0.0), "boxcar")


def test_support_failure_outside_the_slab_is_caught_at_construction():
    def leaky(u):
        a = np.abs(np.asarray(u, dtype=float))
        return wide_profile(u) + np.where((a >= 1.25) & (a <= 1.75), 1e-3, 0.0)

    with pytest.raises(WindowError, match=r"leaky.*support"):
        SampledWindow(leaky, "leaky")


def test_partition_failure():
    w = SampledWindow(lambda u: 1.01 * wide_profile(u), "loud")
    with pytest.raises(WindowError, match=r"loud.*partition of unity.*bin"):
        w.sample(8, 16, 1.0)


@pytest.mark.parametrize("ramp", [lambda x: np.asarray(x) ** 2,            # nu(x) + nu(1-x) != 1
                                  lambda x: 0.5 + 0.0 * np.asarray(x)])    # nu(0) != 0
def test_ramp_failures(ramp):
    with pytest.raises(WindowError, match="ramp"):
        RampWindow(ramp, "bad-ramp")


def _leaky(u):
    a = np.abs(np.asarray(u, dtype=float))
    return wide_profile(u) + np.where((a >= 1.25) & (a <= 1.75), 1e-3, 0.0)


@pytest.mark.parametrize("make, pattern", [
    (lambda: SampledWindow(lambda u: np.where(np.abs(np.asarray(u)) <= 1.0, 0.5, 0.0), "boxcar"),
     r"'boxcar'.*support.*psi\(.*1\.0"),
    (lambda: SampledWindow(_leaky, "leaky"), r"'leaky'.*support.*psi\("),
    (lambda: RampWindow(lambda x: np.asarray(x) ** 2, "bad-ramp"), r"'bad-ramp'.*ramp.*at x = "),
    (lambda: RampWindow(lambda x: 0.5 + 0.0 * np.asarray(x), "bad-ramp"), r"'bad-ramp'.*ramp.*nu\(0\) = "),
], ids=["boxcar", "leaky", "ramp-symmetry", "ramp-endpoints"])
def test_construction_failures_name_the_condition_window_and_point(make, pattern):
    with pytest.raises(WindowError, match=pattern):
        make()


def test_windows_compare_by_function_and_name():
    assert SampledWindow(wide_profile, "a") == SampledWindow(wide_profile, "a")
    assert hash(SampledWindow(wide_profile, "a")) == hash(SampledWindow(wide_profile, "a"))
    assert SampledWindow(wide_profile, "a") != SampledWindow(wide_profile, "b")
    assert SampledWindow(wide_profile, "a") != SampledWindow(lambda u: wide_profile(u), "a")
    assert RampWindow(smoothstep, "s") == RampWindow(smoothstep, "s")


@pytest.mark.parametrize("make", [lambda: SampledWindow(wide_profile, ""), lambda: SampledWindow(3, "x"),
                                  lambda: RampWindow(smoothstep, None), lambda: RampWindow("x", "x")])
def test_bad_names_and_functions_are_type_errors(make):
    with pytest.raises(TypeError):
        make()


def test_sample_window_rejects_non_windows():
    with pytest.raises(TypeError, match="Window"):
        sample_window("meyer", 8, 16, 1.0)


def test_sample_window_validates_third_party_windows():
    class Flat:
        name = "flat"

        def sample(self, Nf, Nt, dt):
            return np.ones(Nt)

    with pytest.raises(WindowError, match=r"flat.*support.*bin 0"):
        sample_window(Flat(), 8, 16, 1.0)


def test_sample_window_returns_builtin_samples_unchanged():
    assert sample_window(MEYER, 8, 16, 1.0) is MEYER.sample(8, 16, 1.0)


def test_an_invalid_grid_is_a_value_error():
    with pytest.raises(ValueError):
        MEYER.sample(7, 16, 1.0)
    with pytest.raises(ValueError):
        MEYER.sample(8, 16, -1.0)
