"""Per-point K x K noise covariance from a model (domainia 3.0 spec, sections 3 and 6.2)."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import frequency_covariance

K, DT = 3, 5.0
#: Unequal levels: real, symmetric and positive definite.
LEVELS = np.array([[1.0, 0.6, 0.3], [0.6, 1.2, -0.4], [0.3, -0.4, 0.9]])
#: Unequal-arm delays [s] and phases [rad], which make the cross-spectra complex, at DC too.
DELAYS, PHASES = np.array([0.0, 7.3, -11.9]) * DT, np.array([0.0, 0.4, -1.1])
#: A complex mixing matrix for the loop-built model.
B = np.random.default_rng(20261114).standard_normal((K, K, 2)) @ [1.0, 1.0j]
#: The routes from a stationary model S(f) to a covariance.
ROUTES = {"frequency": lambda S, N=128: frequency_covariance(S, N=N, dt=DT)}


def psd(f):
    """A steep one-sided PSD [units^2/Hz]."""
    return 1e-2 * (1.0 + (2e-3 / (f + 1e-3)) ** 2) * (1.0 + (f / 0.05) ** 4)


def delayed(f, levels=LEVELS, delays=DELAYS, phases=PHASES, p=psd):
    """S_ab(f) = P(f) L_ab exp(i (theta_a(f) - theta_b(f))), theta_a(f) = 2 pi tau_a f + phi_a, for
    f of any shape: complex delayed cross-spectra with unequal levels, exactly Hermitian."""
    e = (slice(None), slice(None)) + (None,) * np.ndim(f)
    theta = 2 * np.pi * (delays[:, None] - delays)[e] * f + (phases[:, None] - phases)[e]
    return p(f) * levels[e] * np.exp(1j * theta)


def second(f):
    """Another such model, for linearity."""
    return delayed(f, np.array([[0.5, -0.2, 0.1], [-0.2, 0.7, 0.25], [0.1, 0.25, 0.4]]),
                   np.array([1.1, -4.0, 2.5]) * DT, np.array([0.3, 0.0, 0.7]),
                   lambda f: 3e-3 * (1.0 + (f / 0.02) ** 2))


def rounded(f):
    """S = B^H D(f) B summed entry by entry: Hermitian only to rounding, and its diagonal has
    a rounding-level imaginary part."""
    S = np.zeros((K, K) + np.shape(f), complex)
    for a in range(K):
        for b in range(K):
            S[a, b] = sum(np.conj(B[i, a]) * ((1.0 + i) * psd(f)) * B[i, b] for i in range(K))
    return S


def plus(a, b, rel):
    """``delayed`` with ``rel`` times each point's largest |S_cd| added to entry (a, b)."""
    def model(f):
        S = delayed(f)
        S[a, b] += rel * np.max(np.abs(S), axis=(0, 1))
        return S
    return model


def sigma(S, N, dt):
    """Spec 6.2's dense stationary reference, (K, K, N, N): Sigma[a, b, i, j] = R_ab((i - j) mod
    N), R_ab(tau) = Re[(1 / (N dt)) sum_k P_ab(k) e^{2 pi i k tau / N}], with P = S / 2 on the
    rfft bins, extended to all N bins by P_ab(N - k) = conj(P_ab(k))."""
    f = np.fft.rfftfreq(N, dt)
    P = 0.5 * np.asarray(S(f), complex)
    P = np.broadcast_to(P, P.shape[:2] + f.shape)
    R = np.fft.ifft(np.concatenate([P, np.conj(P[..., N - f.size:0:-1])], axis=-1)).real / dt
    i = np.arange(N)
    return R[..., (i[:, None] - i) % N]


def error(got, want):
    """Spec 3.3's metric: the largest |got_ab - want_ab| / sqrt(want_aa want_bb)."""
    d = np.abs(np.einsum("aa...->a...", want))
    return np.max(np.abs(got - want) / np.sqrt(d[:, None] * d[None]))


def test_frequency_covariance_is_the_per_bin_covariance_of_the_dense_reference():
    """E[X_a conj(X_b)] of X = dt rfft(x) under Sigma, at even and odd N, within 1e-12
    (measured: 3.5e-15 and 2.0e-15); real at DC and even-N Nyquist, where |Im S| is 33% and 49%
    of |S|; white noise has R(0) = S0 / (2 dt); a diagonal model gives Enchilada's from_psd."""
    white = lambda f: np.full((1, 1, 1), 4.0)
    np.testing.assert_allclose(sigma(white, 128, DT)[0, 0, :, 0],
                               np.r_[4.0 / (2 * DT), np.zeros(127)], rtol=1e-15, atol=1e-16)
    assert np.array_equal(ROUTES["frequency"](white), np.full((1, 1, 65), 128 * DT / 2 * 4.0 + 0j))
    level = np.array([1.0, 2.0, 0.5])
    diagonal = lambda f: np.eye(K)[:, :, None] * (level[:, None, None] * psd(f))
    for N in (128, 75):
        F = DT * np.fft.rfft(np.eye(N), axis=0)
        want = np.einsum("ki,abij,kj->abk", F, sigma(delayed, N, DT), F.conj(), optimize=True)
        got = ROUTES["frequency"](delayed, N)
        assert got.shape == (K, K, N // 2 + 1) and got.dtype == np.complex128
        assert error(got, want) < 1e-12
        assert not np.any(got[..., 0].imag) and np.any(got[..., -1].imag) == bool(N % 2)
        d = ROUTES["frequency"](diagonal, N)
        np.testing.assert_allclose(np.einsum("aak->ak", d), (N * DT / 2) * level[:, None]
                                   * psd(np.fft.rfftfreq(N, DT)), rtol=1e-15, atol=0)
        assert not np.any(d[~np.eye(K, dtype=bool)])


def test_linear_and_exactly_hermitian_and_a_zero_point_passes():
    """f(S1 + S2) = f(S1) + f(S2) to rounding (spec 3.4); a model Hermitian only to rounding, or
    off by less than the tolerance, gives an output that is exactly Hermitian with a real
    diagonal (3.1); a model that is 0 everywhere passes and gives 0."""
    S = rounded(np.fft.rfftfreq(128, DT))
    assert not np.array_equal(S, np.conj(np.swapaxes(S, 0, 1)))
    assert np.any(np.einsum("aak->ak", S).imag)
    for name, route in ROUTES.items():
        both = route(lambda f: delayed(f) + second(f))
        assert error(both, route(delayed) + route(second)) < 1e-14, name
        for C in (route(rounded), route(plus(0, 1, 5e-13)), route(plus(1, 1, 5e-13j))):
            assert np.array_equal(C, np.conj(np.swapaxes(C, 0, 1))), name
            assert not np.any(np.einsum("aa...->a...", C).imag), name
        assert not np.any(route(lambda f: 0.0 * delayed(f))), name


def test_every_refusal():
    """Each failure of the model check, made per point (spec 3.1), raises the right kind."""
    def dip(f):
        """``delayed``, 1e-6 as large at DC, with a 1e-16 asymmetry there: 1.7e-9 of that point's
        largest entry, though below 1e-15 of the model's largest."""
        S = delayed(f) * np.where(f == 0, 1e-6, 1.0)
        S[0, 1] += np.where(f == 0, 1e-16, 0.0)
        return S
    cases = [
        (np.eye(K), TypeError, "S must be callable"),
        (lambda f: delayed(f).real > 0, TypeError, "must be a numeric array, got dtype bool"),
        (lambda f: delayed(f).astype(str), TypeError, "numeric"),
        (lambda f: delayed(f).astype(object), TypeError, "numeric"),
        (psd, ValueError, r"must have shape \(K, K, "),
        (lambda f: 1.0, ValueError, r"got \(\)"),
        (lambda f: delayed(f)[:2], ValueError, r"got \(2, 3, "),
        (lambda f: delayed(f)[..., :-1], ValueError, "must have shape"),
        (plus(0, 0, np.nan), ValueError, "not finite"),
        (plus(2, 1, np.inf), ValueError, "not finite"),
        (plus(0, 1, 1e-9), ValueError, r"\|S\[0, 1\] - conj\(S\[1, 0\]\)\| is 1e-09 of"),
        (plus(1, 1, 1e-9j), ValueError, r"\|Im S\[1, 1\]\| is 1e-09 of"),
        (plus(2, 2, -3.0), ValueError, r"-Re S\[2, 2\] is 1 of"),
        (dip, ValueError, r"\|S\[0, 1\] - conj\(S\[1, 0\]\)\| is 1.7e-09 of"),
    ]
    for S, err, match in cases:
        for route in ROUTES.values():
            with pytest.raises(err, match=match):
                route(S)
    for kw, err, match in (({"N": 128.0}, TypeError, "N must be an integer"),
                           ({"N": 0}, ValueError, "N must be positive"),
                           ({"dt": 0.0}, ValueError, "dt must be")):
        with pytest.raises(err, match=match):
            frequency_covariance(delayed, **{"N": 128, "dt": DT, **kw})
