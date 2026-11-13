"""Admissible Wilson windows: Meyer(d), RampWindow, SampledWindow (domainia 2.0 spec, section 2)."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from functools import partial

import numpy as np
import pytest

from domainia import wilson
from domainia.testing import ALT_WINDOW, smoothstep, wide_profile
from domainia.wilson import MEYER, Meyer, RampWindow, SampledWindow, Window
from domainia.wilson.filter import nu_d, phi_centered
from domainia.wilson.windows import WindowError, require_window, sample_window

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
    inv_dF = 1.0 / wilson.grid.delta_F(Nf, dt)
    assert phi.shape == (Nt,) and phi.dtype == np.float64 and phi[0] == 0.0
    assert np.array_equal(phi[h + 1:], phi[h - 1:0:-1])
    np.testing.assert_allclose(phi[h:] ** 2 + phi[:h] ** 2, inv_dF, rtol=1e-12, atol=0)


def test_the_alt_window_differs_from_meyer():
    assert not np.allclose(ALT_WINDOW.sample(8, 16, 1.0), MEYER.sample(8, 16, 1.0))


def test_evenness_failure_names_the_condition_window_and_bin():
    w = SampledWindow(lambda u: wide_profile(u) * np.where(np.asarray(u) < 0, 1.0 - 1e-3, 1.0), "lopsided")
    with pytest.raises(WindowError,
                       match=r"'lopsided'.*evenness.*worst at bin 9: 4\.0 but its mirror bin 7 is 3\.996$"):
        w.sample(8, 16, 1.0)


def test_support_failure_at_the_slab_edge_is_caught_at_construction():
    # psi(+-1) is checked at construction, and bin 0 samples psi at exactly u = -1
    # (test_sampled_window_evaluates_the_profile_on_the_exact_u_grid), so a pointwise
    # profile that is nonzero at the slab edge fails here, not at sample time.  The
    # sample-time bin-0 branch is exercised by `Flat` in
    # test_sample_window_validates_third_party_windows.
    with pytest.raises(WindowError, match=r"'boxcar'.*support.*psi\(-1\.0\) = 0\.5,"):
        SampledWindow(lambda u: np.where(np.abs(np.asarray(u)) <= 1.0, 0.5, 0.0), "boxcar")


def test_support_failure_outside_the_slab_is_caught_at_construction():
    def leaky(u):
        a = np.abs(np.asarray(u, dtype=float))
        return wide_profile(u) + np.where((a >= 1.25) & (a <= 1.75), 1e-3, 0.0)

    with pytest.raises(WindowError, match=r"'leaky'.*support.*psi\(-1\.74\d*\) = 0\.001,"):
        SampledWindow(leaky, "leaky")


def test_partition_failure():
    w = SampledWindow(lambda u: 1.01 * wide_profile(u), "loud")
    with pytest.raises(WindowError, match=r"'loud'.*partition of unity.*worst at bin \d+: "
                                          r"phi\(f\)\^2 \+ phi\(f - dF\)\^2 = 16\.32\d*, 1/dF = 16\.0 "):
        w.sample(8, 16, 1.0)


@pytest.mark.parametrize("ramp, pattern", [
    (lambda x: np.asarray(x) ** 2,                                   # nu(x) + nu(1-x) != 1
     r"'bad-ramp'.*ramp.*nu\(x\) \+ nu\(1 - x\) - 1 = 5\.000e-01 at x = 0\.5$"),
    (lambda x: 0.5 + 0.0 * np.asarray(x),                            # nu(0) != 0
     r"'bad-ramp'.*ramp.*nu\(0\) = 0\.5 and nu\(1\) = 0\.5;"),
], ids=["symmetry", "endpoints"])
def test_ramp_failures(ramp, pattern):
    with pytest.raises(WindowError, match=pattern):
        RampWindow(ramp, "bad-ramp")


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

    with pytest.raises(WindowError, match=r"'flat'.*support.*bin 0 \(f = -dF\) is 1\.0, not 0$"):
        sample_window(Flat(), 8, 16, 1.0)


def test_sample_window_returns_builtin_samples_unchanged():
    assert sample_window(MEYER, 8, 16, 1.0) is MEYER.sample(8, 16, 1.0)


def test_an_invalid_grid_is_a_value_error():
    with pytest.raises(ValueError):
        MEYER.sample(7, 16, 1.0)
    with pytest.raises(ValueError):
        MEYER.sample(8, 16, -1.0)


# --------------------------------------------------------------------------- review fixes

def _sine(u):
    """The sine (MLT) profile cos(pi u / 2): no flat top, reaches 0 linearly at |u| = 1."""
    a = np.abs(np.asarray(u, dtype=float))
    return np.where(a < 1.0, np.cos(np.pi / 2.0 * a), 0.0)


def _meyer_profile(u):
    a = np.abs(np.asarray(u, dtype=float))
    return np.where(a < 0.25, 1.0, np.where(a < 0.75, np.cos(np.pi / 2.0 * nu_d((a - 0.25) / 0.5, 4)), 0.0))


class _Fixed:
    """A third-party Window returning one fixed array (validated on every call)."""

    def __init__(self, name, phi):
        self.name, self.phi = name, phi

    def sample(self, Nf, Nt, dt):
        return self.phi


# The old u = (k - h) / T / dF rounds u[0] off -1 on these grids (61 of 1100 probed).
ROUNDING_GRIDS = [(4, 6, 0.1), (6, 30, 0.7), (12, 100, 7.3), (100, 100, 1 / 3)]


@pytest.mark.parametrize("Nf, Nt, dt", ROUNDING_GRIDS + GRIDS)
def test_sampled_window_evaluates_the_profile_on_the_exact_u_grid(Nf, Nt, dt):
    seen = []

    def profile(u):
        seen.append(np.array(u))
        return wide_profile(u)

    SampledWindow(profile, "recorder").sample(Nf, Nt, dt)
    h = Nt // 2
    assert np.array_equal(seen[-1], (np.arange(Nt) - h) / h) and seen[-1][0] == -1.0


@pytest.mark.parametrize("Nf, Nt, dt", ROUNDING_GRIDS)
def test_a_profile_that_reaches_zero_linearly_at_the_slab_edge_is_admissible(Nf, Nt, dt):
    phi = SampledWindow(_sine, "sine").sample(Nf, Nt, dt)
    assert phi[0] == 0.0


@pytest.mark.parametrize("Nf, Nt, dt", GRIDS + ROUNDING_GRIDS + [(16, 1000, 0.3), (32, 8, 7.3)])
def test_a_sampled_meyer_profile_is_meyer_to_rounding(Nf, Nt, dt):
    a, b = SampledWindow(_meyer_profile, "meyer-profile").sample(Nf, Nt, dt), MEYER.sample(Nf, Nt, dt)
    assert np.max(np.abs(a - b)) <= 1e-15 * np.max(b)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_samples_fail_the_finiteness_condition(bad):
    phi = np.array(MEYER.sample(8, 16, 1.0))
    phi[[8 - 3, 8 + 3]] = bad                      # a symmetric pair
    with pytest.raises(WindowError, match=r"'nonfinite'.*finiteness.*bin 5 is (nan|inf)$"):
        sample_window(_Fixed("nonfinite", phi), 8, 16, 1.0)


def test_complex_samples_fail_the_real_condition():
    phi = MEYER.sample(8, 16, 1.0) * (1.0 + 1e-3j)
    with pytest.raises(WindowError, match=r"'cplx'.*real.*complex128"):
        sample_window(_Fixed("cplx", phi), 8, 16, 1.0)


def test_a_complex_profile_fails_the_real_condition():
    w = SampledWindow(lambda u: wide_profile(u) * (1.0 + 0j), "cplx-profile")
    with pytest.raises(WindowError, match=r"'cplx-profile'.*real.*complex128"):
        w.sample(8, 16, 1.0)


@pytest.mark.parametrize("reshape", [lambda p: np.tile(p, 2), lambda p: p[:-2], lambda p: p[None, :]],
                         ids=["long", "short", "2d"])
def test_wrong_shape_samples_fail_the_shape_condition(reshape):
    phi = reshape(np.array(MEYER.sample(8, 16, 1.0)))
    with pytest.raises(WindowError, match=r"'shaped'.*shape.*expected \(16,\) samples"):
        sample_window(_Fixed("shaped", phi), 8, 16, 1.0)


def test_a_wrong_shape_profile_fails_the_shape_condition_at_construction():
    with pytest.raises(WindowError, match=r"'scalar'.*shape.*64 points"):
        SampledWindow(lambda u: 0.0, "scalar")


def test_third_party_samples_are_a_read_only_copy():
    own = np.array(MEYER.sample(8, 16, 1.0))
    out = sample_window(_Fixed("fixed", own), 8, 16, 1.0)
    assert np.array_equal(out, own) and out.dtype == np.float64
    assert not out.flags.writeable and not np.shares_memory(out, own)
    assert own.flags.writeable                      # the caller's array is untouched


@pytest.mark.parametrize("grid", [(7, 16, 1.0), (8, 15, 1.0), (8, 16, -1.0), (8, 16, float("nan"))])
def test_sample_window_validates_the_grid_before_calling_a_third_party_window(grid):
    calls = []

    class Recorder:
        name = "recorder"

        def sample(self, Nf, Nt, dt):
            calls.append((Nf, Nt, dt))
            return np.array(MEYER.sample(8, 16, 1.0))

    with pytest.raises(ValueError) as info:
        sample_window(Recorder(), *grid)
    assert not isinstance(info.value, WindowError) and calls == []


def test_the_partition_check_covers_the_flat_value_at_f_0():
    # Bin h (f = 0) has no mirror, so only the partition check at j = 0 sees it.
    w = SampledWindow(lambda u: wide_profile(u) * np.where(np.asarray(u) == 0.0, 1.01, 1.0), "peak")
    with pytest.raises(WindowError, match=r"'peak'.*partition of unity.*worst at bin 8:"):
        w.sample(8, 16, 1.0)


@pytest.mark.parametrize("Nf, Nt, dt, j", [(8, 16, 1.0, j) for j in range(1, 8)]
                                          + [(32, 8, 2.0, j) for j in range(1, 4)])
def test_the_evenness_check_covers_every_mirror_pair(Nf, Nt, dt, j):
    h = Nt // 2
    phi = np.array(MEYER.sample(Nf, Nt, dt))
    phi[h + j] += 1e-3
    phi[h - j] -= 1e-3
    with pytest.raises(WindowError,
                       match=rf"'uneven'.*evenness.*worst at bin {h + j}: .* mirror bin {h - j} is"):
        sample_window(_Fixed("uneven", phi), Nf, Nt, dt)


@dataclass
class _MutableProfile:                              # eq=True, not frozen: __hash__ is None
    scale: float = 1.0

    def __call__(self, u):
        return self.scale * wide_profile(u)


@dataclass
class _MutableRamp:
    def __call__(self, x):
        return smoothstep(x)


@pytest.mark.parametrize("make", [lambda: SampledWindow(_MutableProfile(), "p"),
                                  lambda: RampWindow(_MutableRamp(), "r")], ids=["sampled", "ramp"])
def test_an_unhashable_function_is_a_type_error_at_construction(make):
    with pytest.raises(TypeError, match=r"hashable.*module-level function or a frozen dataclass"):
        make()


@pytest.mark.parametrize("obj", [Meyer, SampledWindow, RampWindow, _Fixed])
def test_require_window_rejects_a_class(obj):
    with pytest.raises(TypeError, match=r"Window.*pass an instance, e\.g\. Meyer\(\)"):
        require_window(obj)
    with pytest.raises(TypeError, match=r"Window.*instance"):
        sample_window(obj, 8, 16, 1.0)


@pytest.mark.parametrize("obj", ["meyer", None, 4, phi_centered])
def test_require_window_rejects_a_non_window(obj):
    with pytest.raises(TypeError, match="Window"):
        require_window(obj)


def test_require_window_returns_its_argument():
    third = _Fixed("fixed", np.array(MEYER.sample(8, 16, 1.0)))
    assert require_window(MEYER) is MEYER and require_window(ALT_WINDOW) is ALT_WINDOW
    assert require_window(third) is third


def test_returned_samples_cannot_be_made_writeable():
    third = _Fixed("fixed", np.array(MEYER.sample(8, 16, 1.0)))
    for phi in (MEYER.sample(8, 16, 1.0), ALT_WINDOW.sample(8, 16, 1.0), sample_window(third, 8, 16, 1.0)):
        with pytest.raises(ValueError):
            phi.setflags(write=True)
    assert MEYER.sample(8, 16, 1.0) is MEYER.sample(8, 16, 1.0)


@pytest.mark.parametrize("window", [MEYER, Meyer(7), ALT_WINDOW, RampWindow(smoothstep, "smoothstep")],
                         ids=lambda w: w.name)
@pytest.mark.parametrize("roundtrip", [copy.deepcopy, lambda w: pickle.loads(pickle.dumps(w))],
                         ids=["deepcopy", "pickle"])
def test_windows_survive_copy_and_pickle_as_equal_windows(window, roundtrip):
    other = roundtrip(window)
    assert other == window and hash(other) == hash(window)
    assert other.sample(8, 16, 1.0) is window.sample(8, 16, 1.0)       # one cache entry


def test_meyer_sample_converts_dt_to_float64_first():
    dt = np.float32(0.1)
    assert np.array_equal(MEYER.sample(8, 16, dt), phi_centered(8, 16, float(dt)))
