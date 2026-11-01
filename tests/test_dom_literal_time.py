"""L-A7: the padded likelihood through the time samples (domainia.literal.time_marginal, the
release's ZeroPadded("time_marginal") over the column-rotated Lambda) equals the dense
C_dd = [W'^T Lambda W']_dd / dt^2 in both layouts; the text's cell density (MaskedCovariance)
is it minus ln|det A| at every noise state (spec 10.4's consistency check of the 8 + 8 pads) --
to 1e-8 by QR even at p = 16, while the assembled Cholesky loses up to ~4e-2 there."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

import wdm
from dom_literal_cases import NUS, setup_case
from domainia.literal import ColumnRotated, MaskedCovariance, time_marginal

G = wdm.Grid(16, 32, 1 / 32)


def _dense_time(pad, cov, nu, x):
    """``(x C_dd^-1 x, ln|C_dd|)`` with the dense W' of the padded grid."""
    e = pad.ext
    W = wdm.matrix.active_W(e.Nf, e.Nt, e.dt)
    s0, s1 = pad.samples
    Cdd = (W.T @ cov.dense(nu) @ W)[s0:s1, s0:s1] / e.dt ** 2
    c = cho_factor(0.5 * (Cdd + Cdd.T), lower=True)
    return float(x @ cho_solve(c, x)), 2 * float(np.sum(np.log(np.diag(c[0]))))


@pytest.mark.parametrize("p", [8, 16])
@pytest.mark.parametrize("base", ["band", "tri"])
@pytest.mark.parametrize("layout", ["centered", "end"])
def test_time_marginal_is_the_dense_time_form(p, base, layout):
    pad, cov = setup_case(G, p, base, layout=layout)
    x = np.random.default_rng(7).standard_normal(G.N)
    tm = time_marginal(cov, pad)
    assert isinstance(tm.base, ColumnRotated) == (layout == "centered")
    for nu in NUS:
        quad, logdet = _dense_time(pad, cov, nu, x)
        F = tm.factor(nu)
        assert bool(F.ok)
        assert float(F.logdet) == pytest.approx(logdet, abs=1e-8)
        assert float(F.quad(x)) == pytest.approx(quad, rel=1e-10)


@pytest.mark.parametrize("p", [8, 16])
@pytest.mark.parametrize("base", ["band", "mid", "tri"])
@pytest.mark.parametrize("layout", ["centered", "end"])
def test_cells_and_time_differ_by_ln_det_A(p, base, layout):
    pad, cov = setup_case(G, p, base, layout=layout)
    x = np.random.default_rng(8).standard_normal(G.N)
    y = pad.lift(x)
    cells, tm, lnA = MaskedCovariance(cov, pad), time_marginal(cov, pad), pad.log_abs_det_lift()
    for nu in NUS:
        Fc, Ft = cells.factor(nu), tm.factor(nu)
        L_cells = -0.5 * (float(Fc.quad(y)) + float(Fc.logdet) + G.N * np.log(2 * np.pi))
        L_time = -0.5 * (float(Ft.quad(x)) + float(Ft.logdet) + G.N * np.log(2 * np.pi))
        assert L_cells - L_time == pytest.approx(-lnA, abs=1e-8)


def test_the_assembled_cholesky_is_limited_by_the_conditioning():
    """p = 16: M_KK has Nf/2 eigenvalues ~6e-9, M Lambda M reaches cond ~1e12 and the assembled
    Cholesky drifts from the exact identity by up to a few 1e-2 nat (QR: 1e-8, above)."""
    pad, cov = setup_case(G, 16, "tri")
    x = np.random.default_rng(5).standard_normal(G.N)
    y = pad.lift(x)
    chol = MaskedCovariance(cov, pad, factorization="cholesky")
    tm, lnA = time_marginal(cov, pad), pad.log_abs_det_lift()
    worst = 0.0
    for nu in NUS:
        Fc, Ft = chol.factor(nu), tm.factor(nu)
        d = -0.5 * (float(Fc.quad(y)) + float(Fc.logdet)) + 0.5 * (float(Ft.quad(x)) + float(Ft.logdet)) + lnA
        worst = max(worst, abs(d))
    assert 1e-6 < worst < 0.1


@pytest.mark.jax
@pytest.mark.parametrize("base", ["band", "tri"])
@pytest.mark.parametrize("form", ["solve", "reduce"])
def test_jax_jit_matches_numpy(jax_be, base, form):
    pad, cov = setup_case(G, 16, base)
    x = np.random.default_rng(9).standard_normal(G.N)
    tm = time_marginal(cov, pad, form=form)
    for nu in NUS:
        Fn, Fj = tm.factor(nu), tm.factor(jax_be.asarray(nu))
        assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), abs=1e-9)
        assert float(Fj.quad(jax_be.asarray(x))) == pytest.approx(float(Fn.quad(x)), rel=1e-10)


def test_bad_input_is_refused():
    pad, cov = setup_case(G, 8, "band")
    _, other = setup_case(G, 16, "band")
    with pytest.raises(ValueError, match="base covers"):
        ColumnRotated(other, pad)
