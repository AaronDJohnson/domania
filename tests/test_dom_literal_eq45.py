"""L-A4: Eq. (45) by the direct grid sum (domainia.literal.Eq45Direct) equals the existing
builders to round-off: the oracle wdm.evolutionary.eq45_direct (record grid, any PSD, any t0),
the bessel / gram builders (log-linear) and the separable builder (GB form, dense and
block-tridiagonal); its factors on NumPy and jitted JAX; -inf when not positive definite."""

from __future__ import annotations

import numpy as np
import pytest

import domtest_inputs as T
import wdm
from domainia.covariance import Eq45Covariance
from domainia.covariance.blocktri import blocks_to_dense, to_layout
from domainia.domains import CellSet
from domainia.literal import Eq45Direct
from domainia.spectra import GenericPSD, LogLinearPSD
from domainia.testing import eq45_dense
from wdm.backend import NUMPY

GRIDS = [wdm.Grid(8, 16, 1 / 16), wdm.Grid(16, 32, 1 / 32)]
NUS = [np.array([-1.7, 0.9]), np.array([-2.3, -1.4])]


def _psd(g):
    return LogLinearPSD(f_floor=0.5, f_ref=1.0, T=g.N * g.dt)


def _cell_sets(g):
    full = CellSet.all_active(g)
    return {"raw": full, "cut": full.drop_columns(3, 1)}


def _rel(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.abs(np.asarray(b)).max()


@pytest.mark.parametrize("g", GRIDS, ids=lambda g: f"{g.Nf}x{g.Nt}")
def test_record_grid_equals_the_direct_oracle(g):
    psd = _psd(g)
    for name, cells in _cell_sets(g).items():
        for nu in NUS:
            want = eq45_dense(cells, psd.bind(nu))
            for assembly in ("blocktri", "dense"):
                got = Eq45Direct(psd, cells, assembly=assembly).dense(nu)
                assert _rel(got, want) <= 1e-14, (name, assembly)


@pytest.mark.parametrize("g", GRIDS, ids=lambda g: f"{g.Nf}x{g.Nt}")
def test_t0_moves_every_pair_mid_time(g):
    """The padded grid's column 0 sits at t0 < 0: the oracle with S(f, t + t0)."""
    psd, cells, t0 = _psd(g), CellSet.all_active(g), -4 * g.dT
    for nu in NUS:
        want = eq45_dense(cells, lambda f, t: psd(f, t + t0, nu))
        assert _rel(Eq45Direct(psd, cells, t0=t0).dense(nu), want) <= 1e-14


@pytest.mark.parametrize("g", GRIDS, ids=lambda g: f"{g.Nf}x{g.Nt}")
def test_log_linear_equals_the_gram_and_bessel_builders(g):
    psd = _psd(g)
    for cells in _cell_sets(g).values():
        for nu in NUS:
            direct = Eq45Direct(psd, cells).dense(nu)
            for builder in ("gram", "bessel"):
                other = Eq45Covariance(psd, cells, builder=builder).dense(nu)
                assert _rel(direct, other) <= 1e-13, builder


def test_a_generic_spectrum_equals_the_oracle():
    g = GRIDS[0]
    T_obs = g.N * g.dt
    psd = GenericPSD(lambda f, t, nu: (1.0 + np.maximum(f, 0.5) ** nu[0]) * (1.0 + 0.4 * np.cos(2 * np.pi * t / T_obs)))
    cells = CellSet.all_active(g).drop_columns(2, 2)
    nu = np.array([-1.5])
    cov = Eq45Direct(psd, cells)
    assert not cov.jittable
    assert _rel(cov.dense(nu), eq45_dense(cells, psd.bind(nu))) <= 1e-14
    # A spectrum singular at f = 0 on cells without the DC band: the sum never reads S(0)
    # (padded support slots repeat a real bin, the empty band's S is masked), so the matrix
    # is finite, as the oracle's, and both assemblies factorise it.
    no_dc = CellSet.bands(g, tuple(range(1, g.Nf + 1)))
    for sing, nu_s in ((GenericPSD(lambda f, t, nu: np.abs(f) ** nu[0] + 0.0 * t), np.array([-1.5])),
                       (LogLinearPSD(f_floor=0.0, f_ref=1.0, T=T_obs), NUS[0])):
        with np.errstate(divide="ignore", invalid="ignore"):
            want, got = eq45_dense(no_dc, sing.bind(nu_s)), Eq45Direct(sing, no_dc).dense(nu_s)
            Fb, Fd = (Eq45Direct(sing, no_dc, assembly=a).factor(nu_s) for a in ("blocktri", "dense"))
        assert np.isfinite(got).all() and _rel(got, want) <= 1e-14
        assert bool(Fb.ok) and bool(Fd.ok)
        assert float(Fb.logdet) == pytest.approx(float(Fd.logdet), rel=1e-12)


@pytest.mark.parametrize("cut", [False, True])
def test_band_windows_equal_the_separable_builder(cut):
    """The GB form on the shipped 3-month case: S = s_I + g(t_bar) s_G on the band windows."""
    c = T.gb_case("gb_3mo")
    cells = c.cells.drop_columns(5, 4) if cut else c.cells
    for ch in range(2):
        want = Eq45Covariance(c.psd, cells, builder="separable", assembly="dense", channel=ch).dense(c.mu[ch])
        got = Eq45Direct(c.psd, cells, assembly="dense", channel=ch, row_chunk=48).dense(c.mu[ch])
        assert _rel(got, want) <= 1e-13, ch


def test_band_windows_blocktri_on_full_bands():
    c = T.gb_case("gb_3mo")
    sep = Eq45Covariance(c.psd, c.cells, builder="separable", assembly="blocktri", layout="ragged", channel=1)
    want = blocks_to_dense(*to_layout(*sep.blocks(c.mu[1]), sep.sizes, "ragged", NUMPY), NUMPY)
    direct = Eq45Direct(c.psd, c.cells, channel=1)
    got = blocks_to_dense(*to_layout(*direct.blocks(c.mu[1]), direct.sizes, "ragged", NUMPY), NUMPY)
    assert _rel(got, want) <= 1e-13
    with pytest.raises(ValueError, match="every column"):
        Eq45Direct(c.psd, c.cells.drop_columns(5, 4), channel=1).constants()


def test_blocktri_and_dense_factors_agree():
    g = GRIDS[1]
    psd, cells = _psd(g), CellSet.all_active(g)
    r = np.random.default_rng(1).standard_normal(cells.size)
    for nu in NUS:
        Fb = Eq45Direct(psd, cells).factor(nu)
        Fd = Eq45Direct(psd, cells, assembly="dense").factor(nu)
        assert bool(Fb.ok) and bool(Fd.ok)
        assert float(Fb.logdet) == pytest.approx(float(Fd.logdet), rel=1e-12)
        assert float(Fb.quad(r)) == pytest.approx(float(Fd.quad(r)), rel=1e-10)


def test_not_positive_definite_is_minus_infinity():
    g = GRIDS[0]
    neg = GenericPSD(lambda f, t, nu: -np.ones(np.broadcast(f, t).shape))
    assert not bool(Eq45Direct(neg, CellSet.all_active(g)).factor(np.zeros(1)).ok)
    assert not bool(Eq45Direct(neg, CellSet.all_active(g), assembly="dense").factor(np.zeros(1)).ok)


def test_bad_input_is_refused():
    g = GRIDS[0]
    with pytest.raises(ValueError, match="assembly"):
        Eq45Direct(_psd(g), CellSet.all_active(g), assembly="sparse")
    with pytest.raises(ValueError, match="row_chunk"):
        Eq45Direct(_psd(g), CellSet.all_active(g), row_chunk=0)
    with pytest.raises(TypeError, match="no direct"):
        Eq45Direct(object(), CellSet.all_active(g))


@pytest.mark.jax
def test_jax_jit_matches_numpy(jax_be):
    g = GRIDS[1]
    psd, cells = _psd(g), CellSet.all_active(g).drop_columns(3, 1)
    r = np.random.default_rng(2).standard_normal(cells.size)
    for assembly in ("blocktri", "dense"):
        cov = Eq45Direct(psd, cells, t0=-2 * g.dT, assembly=assembly)
        assert cov.jittable
        for nu in NUS:
            Fn = cov.factor(nu)
            Fj = cov.factor(jax_be.asarray(nu))
            assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-12)
            assert float(Fj.quad(jax_be.asarray(r))) == pytest.approx(float(Fn.quad(r)), rel=1e-10)
        assert len(cov.compiled_keys()) == 1


@pytest.mark.jax
def test_jax_band_windows(jax_be):
    c = T.gb_case("gb_3mo")
    cells = c.cells.drop_columns(5, 4)
    cov = Eq45Direct(c.psd, cells, assembly="dense", channel=0, t0=-8 * c.grid.dT)
    Fn, Fj = cov.factor(c.mu[0]), cov.factor(jax_be.asarray(c.mu[0]))
    assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), rel=1e-12)
