"""Per-point K x K noise covariance from a model (domainia 3.0 spec, sections 3 and 6.2)."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import frequency_covariance, testing, wdm_covariance, wilson

K, DT = 3, 5.0
#: Unequal levels: real, symmetric and positive definite.
LEVELS = np.array([[1.0, 0.6, 0.3], [0.6, 1.2, -0.4], [0.3, -0.4, 0.9]])
#: Unequal-arm delays [s] and phases [rad], which make the cross-spectra complex, at DC too.
DELAYS, PHASES = np.array([0.0, 7.3, -11.9]) * DT, np.array([0.0, 0.4, -1.1])
#: A complex mixing matrix for the loop-built model.
B = np.random.default_rng(20261114).standard_normal((K, K, 2)) @ [1.0, 1.0j]
#: The WDM grid of the stationary case, and its active cells in numpy.nonzero order (the rows
#: of testing.dense_W).
NF, NT = 16, 32
MASK = wilson.active_mask(NF, NT)
#: The routes from a stationary model S(f) to a covariance; the WDM route keeps the active cells.
ROUTES = {"frequency": lambda S, N=128: frequency_covariance(S, N=N, dt=DT),
          "wdm": lambda S: wdm_covariance((lambda f, t: S(f)) if callable(S) else S, Nf=NF,
                                          Nt=NT, dt=DT)[..., MASK]}
#: x = B(t) y for K independent channels y of PSD psd (spec 6.2), with
#: B(t) = L0 (I + sin(2 pi t / Tp) M1 + cos(4 pi t / Tp) M2).
L0 = np.linalg.cholesky(LEVELS)
M1 = np.array([[0.3, 0.1, -0.2], [0.0, -0.25, 0.15], [0.2, 0.1, 0.3]])
M2 = np.array([[-0.1, 0.2, 0.0], [0.1, 0.2, -0.1], [0.0, -0.2, 0.15]])


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


def mixing(t, Tp):
    """B(t), of shape (*t.shape, K, K)."""
    t = np.asarray(t, float)[..., None, None]
    return L0 @ (np.eye(K) + np.sin(2 * np.pi * t / Tp) * M1 + np.cos(4 * np.pi * t / Tp) * M2)


def modulated(Tp):
    """S_ab(f, t) = psd(f) [B(t) B(t)^T]_ab for broadcasting f and t: real cross-spectra."""
    def model(f, t):
        Bt = mixing(t, Tp)
        return psd(f) * np.moveaxis(Bt @ np.swapaxes(Bt, -1, -2), (-2, -1), (0, 1))
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


def same_cell(Nf, Nt, Tp):
    """The same-cell blocks of spec 6.2's dense evolutionary reference on the active cells,
    (K, K, N): (I_K x W) Sigma (I_K x W)^T with Sigma_ab[i, j] = [B(t_i) B(t_j)^T]_ab R(i - j),
    where R is the stationary reference of psd."""
    N = Nf * Nt
    W = testing.dense_W(Nf, Nt, DT, wilson.active_mask(Nf, Nt))
    R = sigma(lambda f: psd(f)[None, None], N, DT)[0, 0]
    Bt = np.moveaxis(mixing(np.arange(N) * DT, Tp), 0, -1).copy()     # Bt[a, c, i] = B_ac(t_i)
    out = 0.0
    for c in range(K):
        U = Bt[:, c, None] * W              # U[a, p, i] = W[p, i] B_ac(t_i), C-contiguous for BLAS
        out = out + np.einsum("api,bpi->abp", U @ R, U)
    return out


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


def test_wdm_covariance_of_a_stationary_model_is_the_dense_reference():
    """The same-cell blocks of (I_K x W) Sigma (I_K x W)^T, active edge-band cells included,
    within 1e-12 (measured: 3.3e-15) in two windows; every block is 0 at the inactive cells; a
    size-1 time axis gives the explicitly broadcast output, bit for bit."""
    Sigma = sigma(delayed, NF * NT, DT)
    for window in (wilson.MEYER, testing.ALT_WINDOW):
        W = testing.dense_W(NF, NT, DT, MASK, window=window)
        want = np.einsum("pi,abij,pj->abp", W, Sigma, W, optimize=True)
        got = wdm_covariance(lambda f, t: delayed(f), Nf=NF, Nt=NT, dt=DT, window=window)
        assert got.shape == (K, K, NF + 1, NT) and got.dtype == np.float64
        assert error(got[..., MASK], want) < 1e-12
        assert not np.any(got[..., ~MASK])
    full = wdm_covariance(lambda f, t: delayed(f + 0 * t), Nf=NF, Nt=NT, dt=DT, window=window)
    assert np.array_equal(got, full)


def test_wdm_covariance_of_an_evolutionary_model(monkeypatch):
    """x = B(t) y on 16 x 64, outside the columns wrap_cut_columns names: the error is below 5e-2
    at T_p = 32 dT and below 1.5e-2 at 64 dT (measured: 3.5e-2 and 9.3e-3); each diagonal
    entry, also as a K = 1 model, is noise.evolutionary_band_variance to 1e-14 (measured: bit
    for bit) and 0 at the inactive cells; smaller chunks, with partial last chunks, change no
    bit and keep every call within CHUNK points."""
    Nf, Nt = 16, 64
    mask = wilson.active_mask(Nf, Nt)
    lead, trail = wilson.evolutionary.wrap_cut_columns(Nf, Nt, DT)
    n = np.nonzero(mask)[1]
    kept = (n >= lead) & (n < Nt - trail)
    for Tp, bound in ((32, 5e-2), (64, 1.5e-2)):
        S = modulated(Tp * Nf * DT)
        got = wdm_covariance(S, Nf=Nf, Nt=Nt, dt=DT)
        assert error(got[..., mask][..., kept], same_cell(Nf, Nt, Tp * Nf * DT)[..., kept]) < bound
        for a in range(K):
            want = wilson.noise.evolutionary_band_variance(
                Nf, Nt, DT, lambda f, t, S=S, a=a: S(f, t)[a, a])
            one = wdm_covariance(lambda f, t, S=S, a=a: S(f, t)[a:a + 1, a:a + 1],
                                 Nf=Nf, Nt=Nt, dt=DT)
            for v in (got[a, a], one[0, 0]):
                np.testing.assert_allclose(v[mask], want[mask], rtol=1e-14, atol=0)
                assert not np.any(v[~mask])
    points = []

    def counted(f, t):
        points.append(np.broadcast(f, t).size)
        return S(f, t)
    models = (counted, lambda f, t: delayed(f))
    want = [wdm_covariance(s, Nf=NF, Nt=NT, dt=DT) for s in models]
    # 25 bins per band on NF x NT: one call on 2 columns, then 1 band x 5 columns or 2 bands x
    # 32 columns per call.
    for chunk, calls in ((125, 1 + 17 * 7), (2000, 1 + 9)):
        monkeypatch.setattr("domainia._covariance.CHUNK", chunk)
        points.clear()
        for s, C in zip(models, want):
            assert np.array_equal(wdm_covariance(s, Nf=NF, Nt=NT, dt=DT), C)
        assert max(points) <= chunk and len(points) == calls


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
    """Each failure of the model check, made per point (spec 3.1), raises the right kind, as do
    a bad grid, dt or window; a WDM model is checked in every chunk, and refused if its K, or
    its size-1 time axis, changes between calls."""
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
    for kw, err, match in (({"Nf": 15}, ValueError, "Nf must be even"),
                           ({"Nt": 32.0}, TypeError, "Nt must be an integer"),
                           ({"dt": 0.0}, ValueError, "dt must be"),
                           ({"window": "meyer"}, TypeError, "window must be"),
                           ({"S": lambda f, t: plus(0, 1, 1e-9)(f) if t.size > 2 else delayed(f)},
                            ValueError, r"\[chunk from band 0, column 0\] at point \(0, 0, "),
                           ({"S": lambda f, t: delayed(f)[:1, :1] if t.size > 2 else delayed(f)},
                            ValueError, "is 1 x 1, but was 3 x 3"),
                           ({"S": lambda f, t: delayed(f + 0 * t) if t.size > 2 else delayed(f)},
                            ValueError, r"must have shape \(K, K, 17, 1, 25\)")):
        with pytest.raises(err, match=match):
            wdm_covariance(**{"S": lambda f, t: delayed(f), "Nf": NF, "Nt": NT, "dt": DT, **kw})
