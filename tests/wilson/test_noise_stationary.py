"""domainia.wilson.noise and domainia.wilson.stationary against the frozen oracle.

Bit for bit up to v0.11.0.  The Hz window of v0.12.0 (Paper I eq:Meyer-def: phi flat
value 1/sqrt(Delta F), stationary prefactor 1/N, against the oracle's omega-normalised
1/sqrt(2 pi Delta F) and 2 pi/N) is the same product rounded differently, so every
quantity that goes through the window is gated at rounding (``_gap``; the oracle's K
arrays are scaled by 2 pi, a convention change).  Window-independent quantities
(``colored_noise``, the active-index lists) are still compared bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import vendored_2f7136c.noise as onoise
import vendored_2f7136c.stationary as ostat

import domainia.wilson.noise as noise
import domainia.wilson.stationary as stat

GRIDS = [(4, 8, 1.0), (8, 16, 0.5), (8, 32, 1.7), (16, 32, 2.0**-10)]
#: max|got - ref| over max|ref| against the omega-era oracle.
RTOL64 = 1e-14
TWO_PI = 2.0 * np.pi


def _gap(got, ref, scale=None):
    """max|got - ref| over ``scale`` (default max|ref|; the bare max|got - ref| if that is 0),
    in double precision."""
    got = np.asarray(got).astype(np.complex128)
    ref = np.asarray(ref).astype(np.complex128)
    scale = np.max(np.abs(ref)) if scale is None else scale
    gap = np.max(np.abs(got - ref))
    return gap / scale if scale else gap


def _psd(f):
    f = np.asarray(f, dtype=float)
    return 1.0 + 0.3 / (np.abs(f) + 0.05) ** 2 + 5.0 * f**2


def test_dropped_and_renamed_names():
    for gone in ("build_lambda_blocks", "build_lambda_blocks_packed", "unpack_diag_block",
                 "assemble_lambda", "lambda_to_psd", "lambda_to_psd_slab"):
        assert not hasattr(stat, gone), gone
    assert not hasattr(noise, "_generate_colored_noise")
    assert {"psd_from_K", "psd_from_lambda_blocks"} <= set(stat.__all__)
    assert "colored_noise" in noise.__all__


@pytest.mark.parametrize("grid", GRIDS)
def test_band_averaged_variance_is_the_oracle_to_rounding(grid):
    """band_averaged_variance (a phi^2-weighted average) to rounding."""
    Nf, Nt, dt = grid
    got = noise.band_averaged_variance(Nf, Nt, dt, _psd)
    ref = onoise.band_averaged_variance(Nf, Nt, dt, _psd)
    assert got.dtype == ref.dtype
    assert _gap(got, ref) <= RTOL64          # Hz window (v0.12.0): measured 4.8e-16


def test_colored_noise_and_empirical_variance_are_the_oracle_to_rounding():
    """colored_noise (no window) bit for bit; empirical_variance (forward transform) to rounding."""
    Nf, Nt, dt = 8, 16, 0.5
    a = noise.colored_noise(Nf * Nt, dt, _psd, np.random.default_rng(7))
    b = onoise._generate_colored_noise(Nf * Nt, dt, _psd, np.random.default_rng(7))
    assert np.array_equal(a, b)
    got = noise.empirical_variance(Nf, Nt, dt, _psd, 20, seed=3)
    ref = onoise.empirical_variance(Nf, Nt, dt, _psd, 20, seed=3)
    assert got.dtype == ref.dtype
    assert _gap(got, ref) <= RTOL64          # Hz window (v0.12.0): measured 2.1e-16


@pytest.mark.parametrize(("N", "dt", "match"), [
    (9, 1.0, "N must be even, got 9"),
    (7, 0.5, "N must be even, got 7"),
    (1, 1.0, "N must be even, got 1"),
    (0, 1.0, "N must be positive, got 0"),
    (-8, 1.0, "N must be positive, got -8"),
    (8, 0.0, "dt must be a positive finite number"),
    (8, -1.0, "dt must be a positive finite number"),
    (8, np.inf, "dt must be a positive finite number"),
    (8, np.nan, "dt must be a positive finite number"),
])
def test_colored_noise_rejects_bad_N_and_dt(N, dt, match):
    """``colored_noise`` is public, so it validates like every entry point.

    Unvalidated, an odd N has no Nyquist bin yet bin N//2 was drawn real-only,
    so the noise was not stationary (per-sample variance 0.79--1.22 of its mean
    at N=9, flat PSD, 40k draws); dt < 0 returned NaN with only a warning; dt=0
    and N=0 raised ZeroDivisionError; N=1 raised "negative dimensions".  The
    checks run before the PSD is evaluated or the generator is touched.
    """
    def psd(f):
        raise AssertionError("PSD evaluated before the input checks")

    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match=match):
        noise.colored_noise(N, dt, psd, rng)
    assert rng.bit_generator.state == state


@pytest.mark.parametrize(("N", "dt", "match"), [
    (True, 1.0, "N must be an integer, got bool"),
    (8.0, 1.0, "N must be an integer, got float"),
    (8, True, "dt must be a real number, got bool"),
    (8, "0.5", "dt must be a real number, got str"),
    (8, None, "dt must be a real number, got NoneType"),
])
def test_colored_noise_rejects_wrong_kinds_with_type_error(N, dt, match):
    """A wrong kind of ``N`` or ``dt`` is a ``TypeError`` (``dt=True`` used to pass as 1.0),
    raised before the PSD is evaluated or the generator is touched."""
    def psd(f):
        raise AssertionError("PSD evaluated before the input checks")

    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    with pytest.raises(TypeError, match=match):
        noise.colored_noise(N, dt, psd, rng)
    assert rng.bit_generator.state == state


def test_colored_noise_valid_edge_inputs_are_bitwise_the_oracle():
    """The checks only raise: N=2 (DC + Nyquist, no interior) and a NumPy-int N
    still give the oracle's draws bit for bit."""
    a = noise.colored_noise(2, 1.0, _psd, np.random.default_rng(5))
    b = onoise._generate_colored_noise(2, 1.0, _psd, np.random.default_rng(5))
    assert a.shape == (2,) and np.array_equal(a, b)
    c = noise.colored_noise(np.int64(16), 0.5, _psd, np.random.default_rng(5))
    d = onoise._generate_colored_noise(16, 0.5, _psd, np.random.default_rng(5))
    assert np.array_equal(c, d)


@pytest.mark.parametrize("grid", GRIDS)
def test_stationary_builders_are_the_oracle_to_rounding(grid, rng):
    """Lambda, its PSD inversions and apply_lambda_from_psd to rounding; K to rounding against
    2 pi times the oracle's K (prefactor 1/N, was 2 pi/N); the active-index lists bit for bit."""
    Nf, Nt, dt = grid
    N = Nf * Nt
    S_rfft = _psd(np.fft.rfftfreq(N, d=dt))
    for noise_in in (_psd, S_rfft):
        d1, o1, n1 = stat.build_lambda_blocks_full_active(Nf, Nt, dt, noise_in)
        d2, o2, n2 = ostat.build_lambda_blocks_full_active(Nf, Nt, dt, noise_in)
        # The off-diagonal blocks come from the frequency variation of S: they are 6e-6 to
        # 5e-3 of the diagonal (0 on the 4 x 8 grid), so their own max is not the scale of
        # their rounding (measured up to 2.9e-14 of it, block m = 3 on 8 x 16).  Every block
        # is gated against the diagonal's max|Lambda|.
        lam = max(np.max(np.abs(b)) for b in d2)
        assert all(a.dtype == b.dtype for a, b in zip(d1 + o1, d2 + o2, strict=True))
        # Hz window (v0.12.0): measured 1.8e-16 (diagonal), 7.7e-18 (off-diagonal)
        assert max(_gap(a, b, lam) for a, b in zip(d1 + o1, d2 + o2, strict=True)) <= RTOL64
        assert all(np.array_equal(a, b) for a, b in zip(n1, n2, strict=True))
        got = stat.assemble_lambda_full_active(d1, o1, n1, Nf, Nt)
        ref = ostat.assemble_lambda_full_active(d2, o2, n2, Nf, Nt)
        assert _gap(got, ref) <= RTOL64          # Hz window (v0.12.0): measured 1.8e-16
        K1, L1 = stat.build_K_full_active(Nf, Nt, dt, noise_in)
        K2, L2 = ostat.build_K_full_active(Nf, Nt, dt, noise_in)
        assert all(a.dtype == b.dtype for a, b in zip(K1 + L1, K2 + L2, strict=True))
        # Hz window (v0.12.0): measured 2.9e-16 (K_diag), 4.4e-16 (K_off), each of its own max
        assert max(_gap(a, TWO_PI * b) for a, b in zip(K1 + L1, K2 + L2, strict=True)) <= RTOL64
        got = stat.psd_from_K(K1, Nf, Nt, dt)
        assert _gap(got, ostat.lambda_to_psd_slab(K2, Nf, Nt, dt)) <= RTOL64   # Hz: 2.3e-16
        got = stat.psd_from_lambda_blocks(d1, o1, n1, Nf, Nt, dt)
        assert _gap(got, ostat.lambda_to_psd(d2, o2, n2, Nf, Nt, dt)) <= RTOL64  # Hz: 7.1e-16
    x = rng.standard_normal((Nf + 1, Nt))
    got = stat.apply_lambda_from_psd(x, Nf, Nt, dt, S_rfft)
    ref = ostat.apply_lambda_from_psd(x, Nf, Nt, dt, S_rfft)
    assert got.dtype == ref.dtype
    assert _gap(got, ref) <= RTOL64              # Hz window (v0.12.0): measured 2.0e-16


@pytest.mark.parametrize("grid", GRIDS)
def test_psd_round_trips(grid):
    Nf, Nt, dt = grid
    N = Nf * Nt
    S = _psd(np.fft.rfftfreq(N, d=dt))
    K, _ = stat.build_K_full_active(Nf, Nt, dt, S)
    d, o, n = stat.build_lambda_blocks_full_active(Nf, Nt, dt, S)
    assert np.max(np.abs(stat.psd_from_K(K, Nf, Nt, dt) / S - 1)) < 1e-12
    assert np.max(np.abs(stat.psd_from_lambda_blocks(d, o, n, Nf, Nt, dt) / S - 1)) < 1e-12


@pytest.mark.parametrize("dt", [1.0, 1e-33])
def test_psd_from_K_drops_bins_relative_to_the_window_peak(dt):
    """|phi|^2 peaks at 1/dF = 2 dt Nf, which is 1.6e-32 at dt = 1e-33: below the old
    absolute threshold 1e-30, so every bin was dropped and psd_from_K returned zeros."""
    Nf, Nt = 8, 16
    K, _ = stat.build_K_full_active(Nf, Nt, dt, lambda f: 1e-3 * np.ones_like(f))
    got = stat.psd_from_K(K, Nf, Nt, dt)
    assert got.shape == (Nf * Nt // 2 + 1,)
    assert np.max(np.abs(got / 1e-3 - 1)) < 1e-12


def _complex_psd_forms(Nf, Nt, dt, imag):
    """Every accepted PSD form, made complex: callable, rfft / full 1D, both 2D sizes."""
    N = Nf * Nt
    rf = _psd(np.fft.rfftfreq(N, d=dt)) + imag
    full = _psd(np.abs(np.fft.fftfreq(N, d=dt))) + imag
    return {"callable": lambda f: _psd(f) + imag, "rfft": rf, "full": full,
            "rfft matrix": np.diag(rf), "full matrix": np.diag(full)}


@pytest.mark.parametrize("imag", [1e-3j, 0j])
@pytest.mark.parametrize("build", ["build_lambda_blocks_full_active", "build_K_full_active"])
def test_complex_psd_is_rejected(build, imag):
    """A PSD is real: a complex one (even with a zero imaginary part) raises TypeError
    in every input form, instead of being truncated to its real part (1D and callables
    used to warn ComplexWarning and go on; the 2D form took np.real silently)."""
    import warnings

    Nf, Nt, dt = 8, 16, 1.0
    for form, noise_in in _complex_psd_forms(Nf, Nt, dt, imag).items():
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # a ComplexWarning must not stand in
            with pytest.raises(TypeError, match="PSD must be a real numeric array, got dtype complex128"):
                getattr(stat, build)(Nf, Nt, dt, noise_in)
                pytest.fail(f"{build} accepted a complex {form} PSD")


def test_apply_lambda_from_psd_inf_psd_is_zero_power(rng):
    """An inf PSD sample (the LISA f = 0 case) is zero power: ``sanitize_psd`` maps it to 0.0."""
    Nf, Nt, dt = 8, 16, 0.5
    S = _psd(np.fft.rfftfreq(Nf * Nt, d=dt))
    x = rng.standard_normal((Nf + 1, Nt))
    S[0] = 0.0
    want = stat.apply_lambda_from_psd(x, Nf, Nt, dt, S)
    S[0] = np.inf
    assert np.array_equal(stat.apply_lambda_from_psd(x, Nf, Nt, dt, S), want)
