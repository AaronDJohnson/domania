"""L-A5: the text's 8 + 8 zero padding (domainia.literal.CenteredPad): geometry, the lift
x -> (W' chi x)_K, M = I - B B^T against the dense W' chi W'^T / dt, the edge set at round-off
(every kept cell on the paper's grids), ln det M_KK and ln|det A| against dense slogdets, and
time_shifted spectra."""

from __future__ import annotations

import numpy as np
import pytest

import domtest_inputs as T
from domainia import wilson as wdm
from domainia.covariance import Eq45Covariance
from domainia.domains import CellSet
from domainia.literal import EDGE_ENERGY_TOL, CenteredPad, Eq45Direct, time_shifted
from domainia.spectra import GenericPSD, LogLinearPSD
from domainia.variance import BandAveragedVariance, MidpointVariance

SMALL = [(wdm.Grid(8, 16, 1 / 16), 8), (wdm.Grid(16, 32, 1 / 32), 16)]


def _ids(case):
    g, p = case
    return f"{g.Nf}x{g.Nt}p{p}"


def test_geometry_of_the_centered_and_end_layouts():
    g = wdm.Grid(8, 16, 1 / 16)
    c, e = CenteredPad(g, p=8), CenteredPad(g, p=8, layout="end")
    assert c.ext == e.ext == wdm.Grid(8, 24, 1 / 16)
    assert (c.lead, e.lead, c.samples, e.samples) == (4, 0, (32, 160), (0, 128))
    assert c.t0 == -4 * g.dT and e.t0 == 0.0
    assert c.kept.size == e.kept.size == g.N and c.padding.size == 8 * g.Nf
    assert sorted(set(c.kept.n)) == list(range(4, 20)) and sorted(set(e.kept.n)) == list(range(16))
    assert np.array_equal(c.kept.m_array, CellSet.all_active(g).m_array)      # parity kept (p/2 even)
    assert np.allclose(c.column_times()[[0, 4, 23]], np.array([-4, 0, 19]) * g.dT)
    assert CenteredPad(wdm.Grid(64, 256, 2.0 ** -10)).ext.N == 17408
    for bad in (dict(p=7), dict(p=0), dict(p=6), dict(layout="left")):
        with pytest.raises(ValueError):
            CenteredPad(g, **bad)


@pytest.mark.parametrize("case", SMALL, ids=_ids)
@pytest.mark.parametrize("layout", ["centered", "end"])
def test_the_lift_and_the_mask_are_the_dense_ones(case, layout):
    g, p = case
    pad = CenteredPad(g, p=p, layout=layout)
    e = pad.ext
    W = wdm.matrix.active_W(e.Nf, e.Nt, e.dt)
    start, stop = pad.samples
    x = np.random.default_rng(0).standard_normal((3, g.N))
    assert np.max(np.abs(pad.lift(x) - x @ W[pad.kpos][:, start:stop].T)) <= 1e-14
    B = pad.padding_impulses()
    assert np.max(np.abs(B.T @ B - np.eye(B.shape[1]))) <= 1e-13
    M = pad.dense_mask()
    assert np.max(np.abs((np.eye(e.N) - B @ B.T) - M)) <= 1e-14
    rows = np.array([0, 5, g.N - 1])
    assert np.max(np.abs(pad.mask_rows(rows) - M[pad.kpos[rows]])) <= 1e-14
    psi = pad.leakage()
    assert np.max(np.abs(psi - np.diagonal(np.eye(e.N) - M)[pad.kpos])) <= 1e-14
    A = W[pad.kpos][:, start:stop]                                   # the lift on the data samples
    lnA = np.linalg.slogdet(A)[1]
    assert pad.log_abs_det_lift() == pytest.approx(lnA, abs=1e-9)
    assert pad.log_det_mask() == pytest.approx(2 * lnA - g.N * np.log(g.dt), abs=1e-9)
    # the dense M_KK slogdet agrees to its conditioning (p = 16: Nf/2 eigenvalues ~6e-9 -> 2e-7)
    Ms = M[np.ix_(pad.kpos, pad.kpos)]
    assert pad.log_det_mask() == pytest.approx(np.linalg.slogdet(Ms)[1], abs=1e-6)


@pytest.mark.jax
def test_the_lift_under_jit(jax_be):
    import jax

    g, p = SMALL[1]
    pad = CenteredPad(g, p=p)
    lift = pad.lift_map().on(jax_be)
    x = np.random.default_rng(1).standard_normal(g.N)
    got = jax.jit(lambda m, x: m.apply(x))(lift, jax_be.asarray(x))
    assert np.max(np.abs(np.asarray(got) - pad.lift(x))) <= 1e-14


def test_the_edge_set_at_round_off_is_every_kept_cell():
    """The Meyer atoms' power-law tails: no kept row of M is e_p to double precision on the
    sinusoid grid (64 x 256, p = 16) nor on the 3/6/12-month GB column counts (the leakage
    in columns does not depend on Nf: Nf = 32 stands in for 6570 / 7200 / 7008)."""
    assert EDGE_ENERGY_TOL == 2.0 ** -106
    sin = CenteredPad(wdm.Grid(64, 256, 2.0 ** -10))
    psi = sin.leakage()
    assert psi.min() == pytest.approx(4.734e-18, rel=1e-3) and psi.max() == pytest.approx(0.5, rel=1e-6)
    assert sin.edge_set().size == sin.edge_set(2.0 ** -60).size == 16384
    assert sin.edge_set(1e-8).size == 2113 and sin.edge_set(1e-3).size == 587       # 17 + 16 / 6 + 5 columns
    for Nt, low in ((240, 9.84e-18), (438, 4.58e-19), (900, 7.46e-22)):
        gb = CenteredPad(wdm.Grid(32, Nt, 5.0))
        assert gb.leakage().min() == pytest.approx(low, rel=1e-2)
        assert gb.edge_set().size == gb.kept.size
    with pytest.raises(ValueError, match="energy_tol"):
        sin.edge_set(-1.0)


def test_the_sinusoid_mask_determinant():
    """ln det M_KK on the manuscript grid (one 1024 x 1024 slogdet) and the lift's ln|det A|:
    the cell density is the time-sample density minus this constant."""
    pad = CenteredPad(wdm.Grid(64, 256, 2.0 ** -10))
    assert pad.log_det_mask() == pytest.approx(-604.662448, abs=1e-5)
    assert pad.log_abs_det_lift() == pytest.approx(-57084.948255, abs=1e-5)
    assert CenteredPad(wdm.Grid(64, 256, 2.0 ** -10), layout="end").log_det_mask() == pytest.approx(
        pad.log_det_mask(), abs=1e-8)


def test_time_shifted_spectra():
    g, p = SMALL[1]
    pad = CenteredPad(g, p=p)
    psd = LogLinearPSD(0.5, 1.0, g.N * g.dt)
    sh = time_shifted(psd, pad.t0)
    assert sh.capabilities == frozenset() and time_shifted(sh, 1.0).t0 == pytest.approx(pad.t0 + 1.0)
    with pytest.raises(ValueError, match="capability"):
        Eq45Covariance(sh, pad.cells, builder="gram")
    nu = np.array([-1.7, 0.9])
    nu_equiv = np.array([nu[0] + nu[1] * pad.t0 / psd.T, nu[1]])        # alpha(t + t0) as an alpha_c shift
    for V in (BandAveragedVariance, MidpointVariance):
        assert np.max(np.abs(V(sh, pad.cells)(nu) / V(psd, pad.cells)(nu_equiv) - 1)) <= 1e-14
    assert np.max(np.abs(Eq45Direct(sh, pad.cells).dense(nu) - Eq45Direct(psd, pad.cells, t0=pad.t0).dense(nu))) \
        <= 1e-15 * np.abs(Eq45Direct(psd, pad.cells, t0=pad.t0).dense(nu)).max()
    gen = time_shifted(GenericPSD(lambda f, t, nu: f + t), 2.0)
    assert gen(np.array([1.0]), np.array([3.0]), None)[0] == 6.0
    c = T.gb_case("gb_3mo")
    env = time_shifted(c.psd, -8 * c.grid.dT).envelope
    t = np.arange(5) * c.grid.dT
    assert np.array_equal(env.basis(t), c.env.basis(t - 8 * c.grid.dT)) and env.n_params == c.env.n_params
    cut, t0 = c.cells.drop_columns(5, 4), -8 * c.grid.dT              # the GB clock moves once: PSD or t0
    want = Eq45Direct(c.psd, cut, t0=t0, assembly="dense").dense(c.mu[0])
    got = Eq45Direct(time_shifted(c.psd, t0), cut, assembly="dense").dense(c.mu[0])
    assert np.max(np.abs(got - want)) <= 1e-15 * np.abs(want).max()
    with pytest.raises(NotImplementedError):
        env.ghat(np.arange(3), c.mu[0])
    with pytest.raises(TypeError, match="no time shift"):
        time_shifted(object(), 1.0)
