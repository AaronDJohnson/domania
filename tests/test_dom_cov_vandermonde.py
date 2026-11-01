"""P6 M-A: ``VandermondeExactTime`` / ``FCFactor`` -- the exact time covariance of WDM II Eq. 95 by its
Vandermonde square root, and the Fourier-mode route of ``FlatMarginal`` (spec section 11).

Grids: ``CFG`` (8 x 16, dt 1/16, f_b 0.5 Hz: N = 128, the sinusoid test grid; L changes sign) and the
manuscript-shaped ``MS256`` / ``MS1024`` (dt = 2^-10, f_b = 4 Hz, f_ref = 1: the manuscript's L range and
conditioning).  States: the truth (-2, 1.2) and the corners alpha_c in {-4, 0} x alpha_d in
{-2, -1e-9, 0, 1e-9, 2}.

* M-A1: the factor is Eq. 97 (``ExactTimeCovariance.sigma_tilde``) with the Eq. 95 nodes (Nyquist -rho);
  ln det = Weyl + R (slogdet; R(0) = 0, R even, R >= 0, the pair sum); V^-1 = dense LU at every state,
  whiten = the voice factor's G^-1, solve = C^-1 where the dense solve is reliable (alpha_c > -4);
  JAX = NumPy; the traps T = N dt and no ln dt;
* M-A2: Fourier-mode and geometric (template) solves = FC solves (incl. f_0 on a bin at alpha_d = 0);
* M-A3: FlatMarginal takes the O(N) mode route for the Fourier-band basis (both orders), falls back
  to column solves otherwise; the trap real H and b;
* M-A4: the 40-digit references of ``golden/sinusoid/fc_reference.json`` (N = 256: logdet, quad, the
  flat marginal, R), and GaussianLikelihood picks the mode route up through ``marginal_form``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from domainia import wilson as wdm
from domainia.wilson.backend import get_backend, pytree_dataclass

import domtest_inputs as T
from domainia.covariance import DiagonalCovariance, ExactTimeCovariance, FCFactor, VandermondeExactTime
from domainia.covariance.vandermonde import FCState, fourier_mode_table, pack_real, pair_sum_R
from domainia.data import TimeSeries
from domainia.density import RealGaussian
from domainia.domains import CellSet, TimeDomain
from domainia.likelihood import GaussianLikelihood, Params
from domainia.literal import time_shifted
from domainia.maps import Identity
from domainia.nuisance import FlatMarginal, FourierBandNuisance
from domainia.spectra import LogLinearPSD
from domainia.testing import dense_W

LOG_2PI = float(np.log(2.0 * np.pi))
TRUTH = (-2.0, 1.2)
CORNERS = [(ac, ad) for ac in (-4.0, 0.0) for ad in (-2.0, -1e-9, 0.0, 1e-9, 2.0)]
STATES = [TRUTH] + CORNERS
#: (N, dt, f_floor, analysis cut) -- the cut of FourierBandNuisance (n_u = 16 / 2 / 8)
GRIDS = {"CFG": (128, 1 / 16, 0.5, 1.0), "MS256": (256, 2.0 ** -10, 4.0, 4.0), "MS1024": (1024, 2.0 ** -10, 4.0, 4.0)}


def setup(name: str) -> tuple[VandermondeExactTime, ExactTimeCovariance, np.ndarray]:
    N, dt, fb, fmin = GRIDS[name]
    psd = LogLinearPSD(f_floor=fb, f_ref=1.0, T=N * dt)
    U = FourierBandNuisance(N, dt, fmin).discarded_basis().T
    return VandermondeExactTime(psd, N, dt), ExactTimeCovariance(psd, N, dt, form="dense"), U


def rel(a: Any, b: Any) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def well_conditioned(nu: tuple[float, float]) -> bool:
    """Dense float64 oracles (slogdet, LU / inverse of C) are reliable away from alpha_c = -4
    (cond C up to 4.4e11 there: the dense routes are off by up to ~1e-7 relative)."""
    return nu[0] > -4.0


# --------------------------------------------------------------------------------------------
# M-A1: the factor, its log-determinant and solves
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(GRIDS))
@pytest.mark.parametrize("nu", STATES)
def test_factor_is_eq97(name, nu):
    cov, ex, _ = setup(name)
    Sig = ex.sigma_tilde(np.asarray(nu))
    assert rel(cov.dense(nu), Sig) <= 1e-13
    F = cov.factor(np.asarray(nu))
    assert isinstance(F, FCFactor) and isinstance(F.state, FCState) and bool(F.ok) and F.n_u == 0


@pytest.mark.parametrize("ad", [-2.0, 0.0, 1.2])
def test_trap_nyquist_node_is_minus_rho(ad):
    """DC rho_0, Nyquist -rho_{N/2} (exactly real), conjugate pairs; a +rho Nyquist node is not Eq. 97
    and coincides with the DC node at alpha_d = 0 (V singular)."""
    cov, ex, _ = setup("MS256")
    N = cov.N
    x = cov.nodes(ad)
    rho = np.exp(ad * cov.L / (2.0 * N))
    assert x[0] == rho[0] and x[N // 2] == -rho[N // 2] and x[N // 2].imag == 0.0
    assert np.array_equal(x[N - np.arange(1, N // 2)], np.conj(x[1:N // 2]))
    nu = np.array([-2.0, ad])
    Vp = cov.vandermonde(ad)
    Vp[:, N // 2] = rho[N // 2] ** np.arange(N)                        # the +rho trap
    sig2 = np.exp(cov.constants()["log_half_df"] + (nu[0] - 0.5 * nu[1]) * cov.L)
    wrong = ((Vp * sig2) @ Vp.conj().T).real
    assert rel(wrong, ex.sigma_tilde(nu)) > 1e-5
    if ad == 0.0:
        assert np.array_equal(Vp[:, N // 2], Vp[:, 0])                 # two equal columns


@pytest.mark.parametrize("name", list(GRIDS))
@pytest.mark.parametrize("nu", STATES)
def test_logdet_is_weyl_plus_r(name, nu):
    cov, ex, _ = setup(name)
    nu = np.asarray(nu)
    F = cov.factor(nu)
    assert cov.weyl_logdet(nu) == ex.weyl_logdet(nu)
    assert float(F.logdet) == float(cov.weyl_logdet(nu) + F.state.R)
    R = pair_sum_R(cov.L, nu[1])
    assert abs(float(F.state.R) - R) <= 1e-14 * max(R, 1e-300) + 1e-30
    ld = np.linalg.slogdet(ex.sigma_tilde(nu))[1]
    assert abs(float(F.logdet) - ld) <= (2e-10 if well_conditioned(tuple(nu)) else 1e-6)


def test_pair_sum_r_is_even_nonnegative_and_zero_at_stationarity():
    cov, _, _ = setup("MS1024")
    assert pair_sum_R(cov.L, 0.0) == 0.0 and float(cov.factor(np.array([-2.0, 0.0])).state.R) == 0.0
    for ad in (1e-9, 0.3, 2.0):
        R = pair_sum_R(cov.L, ad)
        assert R > 0.0 and pair_sum_R(cov.L, -ad) == R
        Rk = float(cov.factor(np.array([0.0, ad])).state.R)
        assert abs(Rk - float(cov.factor(np.array([0.0, -ad])).state.R)) <= 1e-15 * R


def test_trap_no_ln_dt_in_the_time_domain_logdet():
    """``logdet`` is ln det C~ (time domain): ln det of W C~ W^T is N ln dt more (W W^T = dt I), and
    the Eq. 8 Jacobian +N/2 ln dt cancels half of it -- neither belongs to this factor."""
    N, dt, _, _ = GRIDS["CFG"]
    cov, ex, _ = setup("CFG")
    W = dense_W(CellSet.all_active(wdm.Grid(8, 16, dt)))
    assert np.allclose(W @ W.T, dt * np.eye(N), rtol=0, atol=1e-14)
    for nu in (TRUTH, (0.0, -2.0), (0.0, 0.0)):
        nu = np.asarray(nu)
        Lam = W @ ex.sigma_tilde(nu) @ W.T
        ld_cells = np.linalg.slogdet(0.5 * (Lam + Lam.T))[1]
        assert abs(ld_cells - float(cov.factor(nu).logdet) - N * np.log(dt)) <= 1e-9


@pytest.mark.parametrize("name", list(GRIDS))
@pytest.mark.parametrize("nu", STATES)
def test_solves_are_the_dense_ones(name, nu, rng):
    cov, ex, _ = setup(name)
    N = cov.N
    nu = np.asarray(nu)
    F = cov.factor(nu)
    C = cov.constants()
    X = rng.standard_normal((N, 3))
    V = cov.vandermonde(nu[1])
    assert rel(F.state.vsolve(X, C=C), np.linalg.solve(V, X.astype(complex))) <= 1e-11      # V^-1 (cond ~1e3)
    assert rel(F.state.vsolve(X[:, 0], C=C), np.linalg.solve(V, X[:, 0].astype(complex))) <= 1e-11
    assert rel(F.state.vsolve_T(X, C=C), np.linalg.solve(V.T, X.astype(complex))) <= 1e-11
    assert rel(F.whiten(X), np.linalg.solve(ex.voice_factor(nu), X)) <= 1e-12        # G^-1, G = B_c P
    assert F.whiten(X).dtype == np.float64 and F.solve(X).dtype == np.float64
    q_w, q_s = float(F.quad(X[:, 0])), float(F.quad(X[:, 0], form="solve"))
    assert abs(q_w - q_s) <= 1e-12 * q_w
    if well_conditioned(tuple(nu)):
        assert rel(F.solve(X), np.linalg.solve(ex.sigma_tilde(nu), X)) <= 1e-11


def test_alpha_d_zero_is_the_dft():
    cov, _, _ = setup("MS1024")
    x = np.random.default_rng(4).standard_normal(cov.N)
    F = cov.factor(np.array([-1.0, 0.0]))
    assert np.array_equal(F.state.bh, np.ones(cov.N)) and np.array_equal(F.state.bp, np.ones(cov.N))
    assert np.array_equal(F.state.vsolve(x, C=cov.constants()), np.fft.fft(x) / cov.N)


def test_pack_real_is_the_voice_pairing():
    z = np.fft.fft(np.random.default_rng(5).standard_normal((16, 2)), axis=0)
    p = pack_real(z)
    assert p.shape == (16, 2) and p.dtype == np.float64
    assert np.allclose(p.T @ p, (z.conj().T @ z).real, rtol=1e-14, atol=0)


def test_at_alpha_c_reuses_the_alpha_d_state():
    cov, _, _ = setup("MS256")
    x = np.random.default_rng(7).standard_normal(cov.N)
    F = cov.factor(np.array([-2.0, 1.2]))
    G, H = F.at_alpha_c(-3.1), cov.factor(np.array([-3.1, 1.2]))
    assert G.state is F.state
    assert np.array_equal(G.log_sig2, H.log_sig2) and float(G.logdet) == float(H.logdet)
    assert float(G.quad(x)) == float(H.quad(x))


def test_trap_record_clock_and_construction():
    N, dt = 128, 1 / 16
    psd = LogLinearPSD(f_floor=0.5, f_ref=1.0, T=N * dt)
    with pytest.raises(ValueError, match="T = N dt"):
        VandermondeExactTime(LogLinearPSD(f_floor=0.5, f_ref=1.0, T=(N - 1) * dt), N, dt)
    with pytest.raises(ValueError, match="log-linear PSD"):
        VandermondeExactTime(time_shifted(psd, -8 * 8 * dt), N, dt)
    with pytest.raises(ValueError, match="even"):
        VandermondeExactTime(LogLinearPSD(f_floor=0.5, f_ref=1.0, T=127 * dt), 127, dt)
    cov = VandermondeExactTime(psd, N, dt)
    assert (cov.block, cov.group, cov.jittable, cov.marginal_form) == (32, 16, True, "whiten")
    assert cov.state_key(np.array(TRUTH)) is None
    odd = VandermondeExactTime(LogLinearPSD(f_floor=0.5, f_ref=1.0, T=100 * dt), 100, dt)
    assert (odd.block, odd.group) == (25, 10)                           # divisors of N
    ex = ExactTimeCovariance(LogLinearPSD(f_floor=0.5, f_ref=1.0, T=100 * dt), 100, dt, form="dense")
    nu = np.array(TRUTH)
    assert abs(float(odd.factor(nu).logdet) - np.linalg.slogdet(ex.sigma_tilde(nu))[1]) <= 1e-10
    y = np.random.default_rng(8).standard_normal(100)
    assert rel(odd.factor(nu).solve(y), np.linalg.solve(ex.sigma_tilde(nu), y)) <= 1e-11


@pytest.mark.jax
@pytest.mark.parametrize("name", ["CFG", "MS1024"])
def test_jax_factor_is_numpy(jax_be, name, rng):
    import jax
    import jax.numpy as jnp

    cov, _, _ = setup(name)
    Cn, Cj = cov.constants(), cov.constants(jax_be)
    x = rng.standard_normal(cov.N)
    for nu in (TRUTH, (-4.0, -2.0), (0.0, 1e-9)):
        Fn, Fj = cov.factor(np.asarray(nu)), cov.factor(jnp.asarray(nu))
        assert type(Fj.logdet).__module__.startswith("jax")
        assert abs(float(Fj.logdet) - float(Fn.logdet)) <= 1e-12 * abs(float(Fn.logdet))
        assert abs(float(Fj.state.R) - float(Fn.state.R)) <= 1e-14
        assert rel(Fj.state.bh, Fn.state.bh) <= 1e-12 and rel(Fj.state.bp, Fn.state.bp) <= 1e-12
        assert rel(Fj.state.vsolve(jnp.asarray(x), C=Cj), Fn.state.vsolve(x, C=Cn)) <= 1e-12
        assert rel(Fj.whiten(jnp.asarray(x)), Fn.whiten(x)) <= 1e-12
        assert rel(Fj.solve(jnp.asarray(x)), Fn.solve(x)) <= 1e-12
        assert abs(float(Fj.at_alpha_c(-3.1).logdet) - float(Fn.at_alpha_c(-3.1).logdet)) <= 1e-12 * abs(
            float(Fn.at_alpha_c(-3.1).logdet))
    assert cov.compiled_keys() == (None,)                              # one program for every state
    leaves, tree = jax.tree_util.tree_flatten(Fj)
    assert isinstance(jax.tree_util.tree_unflatten(tree, leaves), FCFactor)
    stacked = jax.tree_util.tree_map(lambda *a: jnp.stack(a), Fj, cov.factor(jnp.asarray(TRUTH)))
    q = jax.vmap(lambda F: F.quad(jnp.asarray(x), C=Cj))(stacked)
    assert rel(q, [float(Fn.quad(x)), float(cov.factor(np.asarray(TRUTH)).quad(x))]) <= 1e-12


# --------------------------------------------------------------------------------------------
# M-A2: O(N) Fourier-mode and geometric (template) solves
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(GRIDS))
@pytest.mark.parametrize("nu", [TRUTH, (-4.0, 2.0), (0.0, -2.0), (0.0, 0.0), (-4.0, -1e-9)])
def test_fourier_mode_solves_are_fc_solves(name, nu):
    cov, _, U = setup(name)
    C = cov.constants()
    bins, vals = cov.fourier_mode_table(U)
    assert bins.shape == vals.shape == (U.shape[1], 2) and bins.dtype == np.int64
    dc = np.flatnonzero(np.all(bins == 0, axis=1))
    assert dc.size == 1 and vals[dc[0], 1] == 0.0                     # one bin: stored twice, second value 0
    F = cov.factor(np.asarray(nu))
    W = F.state.solve_fourier_modes(bins, vals, C=C)
    assert rel(W, F.state.vsolve(U, C=C)) <= 1e-13
    assert rel(F.whiten_fourier_modes(bins, vals), F.whiten(U)) <= 1e-13


def test_fourier_mode_table_refuses_other_bases(rng):
    cov, _, U = setup("CFG")
    assert fourier_mode_table(U) is not None
    assert fourier_mode_table(rng.standard_normal((cov.N, 3))) is None
    four = np.cos(2 * np.pi * np.outer(np.arange(cov.N), [3, 5]) / cov.N).sum(axis=1)   # 4 bins
    assert fourier_mode_table(four[:, None]) is None
    assert cov.fourier_mode_table(U[:-1]) is None                      # not this record's length


@pytest.mark.parametrize("name", list(GRIDS))
@pytest.mark.parametrize("nu", [TRUTH, (-4.0, 2.0), (0.0, -2.0), (0.0, 0.0), (0.0, 1e-9)])
def test_geometric_and_template_solves_are_fc_solves(name, nu):
    cov, _, _ = setup(name)
    N, dt = cov.N, cov.dt
    C = cov.constants()
    F = cov.factor(np.asarray(nu))
    V = cov.vandermonde(nu[1])
    i = np.arange(N)
    for kappa in (28.0, 7.0, 0.37 * N, 33.3, -12.25, N / 2 - 0.5):    # on-bin, low, high, off-bin, negative
        tau = np.exp(2j * np.pi * kappa * i / N)
        ell = F.state.solve_geometric(kappa)
        assert rel(ell, F.state.vsolve(tau, C=C)) <= 1e-12
        assert rel(ell, np.linalg.solve(V, tau)) <= 1e-11
        h = 1.3 * np.cos(2.0 * np.pi * (kappa / (N * dt)) * (i * dt) + 0.4)
        assert rel(F.state.solve_cosine(1.3, kappa, 0.4), F.state.vsolve(h, C=C)) <= 1e-12


def test_on_bin_template_at_alpha_d_zero_is_a_unit_vector():
    """tau = x_{j*} exactly (f_0 on a bin, alpha_d = 0): the Lagrange basis is e_{j*} with no 0/0."""
    cov, _, _ = setup("MS1024")
    F = cov.factor(np.array([-2.0, 0.0]))
    for kappa in (137.0, -137.0, 0.0):
        e = np.zeros(cov.N, dtype=complex)
        e[int(kappa) % cov.N] = 1.0
        assert np.array_equal(F.state.solve_geometric(kappa), e)


@pytest.mark.jax
@pytest.mark.parametrize("name", ["CFG", "MS1024"])
def test_jax_mode_and_template_solves_are_numpy(jax_be, name):
    import jax.numpy as jnp

    cov, _, U = setup(name)
    bins, vals = cov.fourier_mode_table(U)
    Cn, Cj = cov.constants(), cov.constants(jax_be)
    for nu in (TRUTH, (-4.0, -2.0), (0.0, 0.0)):
        Fn, Fj = cov.factor(np.asarray(nu)), cov.factor(jnp.asarray(nu))
        assert rel(Fj.state.solve_fourier_modes(jnp.asarray(bins), jnp.asarray(vals), C=Cj),
                   Fn.state.solve_fourier_modes(bins, vals, C=Cn)) <= 1e-12
        assert rel(Fj.whiten_fourier_modes(jnp.asarray(bins), jnp.asarray(vals)), Fn.whiten_fourier_modes(bins, vals)) <= 1e-12
        for kappa in (33.3, 28.0, -12.25):
            assert rel(Fj.state.solve_geometric(jnp.asarray(kappa)), Fn.state.solve_geometric(kappa)) <= 1e-12
        assert rel(Fj.state.solve_cosine(jnp.asarray(1.3), jnp.asarray(33.3), jnp.asarray(0.4)),
                   Fn.state.solve_cosine(1.3, 33.3, 0.4)) <= 1e-12


# --------------------------------------------------------------------------------------------
# M-A3: FlatMarginal's Fourier-mode route
# --------------------------------------------------------------------------------------------


def oracle(y: np.ndarray, C: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """Dense flat-prior marginal (test_dom_nuisance's oracle): (quad, logdet)."""
    Ci = np.linalg.inv(C)
    H, b = u.T @ Ci @ u, u.T @ Ci @ y
    return y @ Ci @ y - b @ np.linalg.solve(H, b), np.linalg.slogdet(C)[1] + np.linalg.slogdet(H)[1]


@pytest.mark.parametrize("name", list(GRIDS))
@pytest.mark.parametrize("nu", [TRUTH, (0.0, -2.0), (0.0, 0.0), (0.0, 1e-9), (0.0, 2.0)])
def test_flat_marginal_mode_route_is_the_dense_oracle(name, nu, rng):
    cov, ex, U = setup(name)
    y = rng.standard_normal(cov.N)
    q, ld = oracle(y, ex.sigma_tilde(np.asarray(nu)), U)
    for form in (None, "whiten", "solve"):
        fm = FlatMarginal(cov, U, form=form)
        assert fm.form == (form or "whiten") and fm.modes is not None and "mode_bins" in fm.constants()
        F = fm.factor(np.asarray(nu))
        assert F.n_u == U.shape[1] and bool(F.ok)
        assert abs(float(F.quad(y)) - q) <= 1e-11 * abs(q)
        assert abs(float(F.logdet) - ld) <= 2e-10


def test_flat_marginal_falls_back_to_column_solves(rng):
    cov, ex, _ = setup("CFG")
    U = np.linalg.qr(rng.standard_normal((cov.N, 3)))[0]
    y = rng.standard_normal(cov.N)
    q, ld = oracle(y, ex.sigma_tilde(np.asarray(TRUTH)), U)
    for form in ("whiten", "solve"):
        fm = FlatMarginal(cov, U, form=form)
        assert fm.modes is None and "mode_bins" not in fm.constants()
        F = fm.factor(np.asarray(TRUTH))
        assert abs(float(F.quad(y)) - q) <= 1e-11 * abs(q) and abs(float(F.logdet) - ld) <= 2e-10
    assert FlatMarginal(DiagonalCovariance(np.ones(4), 4), np.eye(4)[:, :1]).form == "solve"   # unchanged default


def test_trap_real_h_and_b():
    """H = Re(W^H diag(1/sigma^2) W) and b = Re(W^H diag(1/sigma^2) w): the imaginary parts are
    round-off (exactly 0 in exact arithmetic) and the route keeps real arrays only."""
    cov, _, U = setup("MS1024")
    C = cov.constants()
    nu = np.asarray(TRUTH)
    F = cov.factor(nu)
    bins, vals = cov.fourier_mode_table(U)
    W = F.state.solve_fourier_modes(bins, vals, C=C)
    w = F.state.vsolve(np.random.default_rng(6).standard_normal(cov.N), C=C)
    s2 = np.exp(F.log_sig2)
    Hc, bc = (W.conj().T / s2) @ W, (W.conj().T / s2) @ w
    assert np.max(np.abs(Hc.imag)) <= 1e-12 * np.max(np.abs(Hc.real))
    assert np.max(np.abs(bc.imag)) <= 1e-12 * np.max(np.abs(bc.real))
    Zm = F.whiten_fourier_modes(bins, vals)
    assert Zm.dtype == np.float64 and np.allclose(Zm.T @ Zm, Hc.real, rtol=1e-12, atol=0)
    Fm = FlatMarginal(cov, U).factor(nu)
    assert Fm.Zu.dtype == np.float64 and Fm.H.L.dtype == np.float64


@pytest.mark.jax
def test_jax_flat_marginal_mode_route(jax_be, rng):
    import jax.numpy as jnp

    cov, _, U = setup("MS1024")
    fm = FlatMarginal(cov, U)
    assert fm.jittable and fm.modes is not None
    y = rng.standard_normal(cov.N)
    for nu in (TRUTH, (-4.0, 2.0), (0.0, 0.0)):
        Mn, Mj = fm.factor(np.asarray(nu)), fm.factor(jnp.asarray(nu))
        assert type(Mj.logdet).__module__.startswith("jax")
        assert abs(float(Mj.quad(jnp.asarray(y))) - float(Mn.quad(y))) <= 1e-12 * abs(float(Mn.quad(y)))
        assert abs(float(Mj.logdet) - float(Mn.logdet)) <= 1e-12 * abs(float(Mn.logdet))
    assert fm.compiled_keys() == (None,)


# --------------------------------------------------------------------------------------------
# M-A4: the 40-digit references (golden/sinusoid/fc_reference.json)
# --------------------------------------------------------------------------------------------

REFERENCE = "sinusoid/fc_reference.json"


@pytree_dataclass(data=("t",), meta=())
@dataclass(frozen=True)
class Cosine:
    """``A cos(2 pi f t + phi)`` on the sample times (a pytree signal for the JAX leg)."""

    t: Any

    def __call__(self, s: Any) -> Any:
        xp = get_backend(s, self.t).xp
        return s[0] * xp.cos(2.0 * np.pi * s[1] * self.t + s[2])


def test_against_the_40_digit_reference():
    """``golden/sinusoid/fc_reference.json`` (mpmath, 40 digits; ``scripts/golden/make_fc_reference.py``)."""
    ref = T.golden_json(REFERENCE)
    N, dt = ref["N"], ref["dt"]
    assert (N, dt, ref["f_floor"], ref["f_ref"], ref["fmin"]) == (256, 2.0 ** -10, 4.0, 1.0, 4.0)
    psd = LogLinearPSD(f_floor=ref["f_floor"], f_ref=ref["f_ref"], T=N * dt)
    cov = VandermondeExactTime(psd, N, dt)
    x = np.array(ref["x"], dtype=float)
    U = FourierBandNuisance(N, dt, ref["fmin"]).discarded_basis().T
    assert {tuple(s["nu"]) for s in ref["states"]} == set(STATES)
    for s in ref["states"]:
        nu = np.array(s["nu"])
        F = cov.factor(nu)
        assert abs(float(F.logdet) - float(s["logdet"])) <= 1e-11
        assert abs(float(F.state.R) - float(s["R"])) <= 1e-14 and abs(pair_sum_R(cov.L, nu[1]) - float(s["R"])) <= 1e-14
        q = float(s["quad"])
        assert abs(float(F.quad(x)) - q) <= 1e-13 * q
        assert abs(float(F.quad(x, form="solve")) - q) <= 1e-13 * q
        qm = q - float(s["quad_H"])
        for form in ("whiten", "solve"):
            Fm = FlatMarginal(cov, U, form=form).factor(nu)
            assert abs(float(Fm.quad(x)) - qm) <= 1e-13 * qm
            assert abs(float(Fm.logdet) - (float(s["logdet"]) + float(s["logdet_H"]))) <= 1e-11


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_gaussian_likelihood_picks_the_mode_route(backend_name):
    """``GaussianLikelihood(..., nuisance=U)`` over the FC covariance: form "whiten", the mode route, and
    lnL = the 40-digit marginal Gaussian (A_n = 1, no signal) on both backends."""
    if backend_name == "jax":
        pytest.importorskip("jax", reason="jax not installed (NumPy-only lane)")
        from domainia.wilson.backend import enable_x64

        enable_x64()
    ref = T.golden_json(REFERENCE)
    N, dt = ref["N"], ref["dt"]
    cov = VandermondeExactTime(LogLinearPSD(f_floor=ref["f_floor"], f_ref=ref["f_ref"], T=N * dt), N, dt)
    U = FourierBandNuisance(N, dt, ref["fmin"]).discarded_basis().T
    time = TimeDomain(N, dt)
    lik = GaussianLikelihood(TimeSeries(np.array(ref["x"]), time), Identity(time), cov, RealGaussian(0.0, "log10_A_n"),
                             signal=Cosine(np.arange(N) * dt),
                             params=Params(("A", "f", "phi", "log10_A_n", "alpha_c", "alpha_d"), (0, 1, 2), (4, 5)),
                             nuisance=U, backend=backend_name)
    fm = lik.covs[0]
    assert isinstance(fm, FlatMarginal) and fm.form == "whiten" and fm.modes is not None and lik.traceable
    rows = np.array([[0.0, 30.0, 0.1, 0.0, *s["nu"]] for s in ref["states"]])
    n_eff = N - U.shape[1]
    want = np.array([-0.5 * (float(s["quad"]) - float(s["quad_H"]) + float(s["logdet"]) + float(s["logdet_H"])
                             + n_eff * LOG_2PI) for s in ref["states"]])
    got = np.asarray(lik(rows))
    assert np.all(np.abs(got - want) <= 1e-12 * np.abs(want))
