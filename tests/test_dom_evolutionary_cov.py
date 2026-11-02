"""EvolutionaryCovariance (WDM II Eq. 45) for a callable PSD: builders, assemblies and
the stationary limit (replaces test_dom_eq45)."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson
from domainia.covariance import EvolutionaryCovariance, keep_mask
from domainia.covariance.blocktri import blocks_to_dense, to_layout
from domainia.domains import CellSet, WDMDomain
from domainia.spectra import GenericPSD
from domainia.wilson.backend import NUMPY
from domainia.wilson.evolutionary import eq45_blocks_cheb, eq45_coeffs_interp, eq45_plan
from domainia.wilson.matrix import build_C_wdm_active

NF, NT, DT = 8, 16, 1 / 16
N = NF * NT
T_REC = N * DT
NU = np.array([-1.7, 0.9])


def _drifting_power_law(f, t, nu):
    """``max(f, 0.5) ** (a_c + a_d (t / T - 1/2))``: a drifting power law as a plain callable."""
    alpha = nu[0] + nu[1] * (np.asarray(t, float) / T_REC - 0.5)
    return np.exp(alpha * np.log(np.maximum(np.asarray(f, float), 0.5)))


PSD = GenericPSD(_drifting_power_law)


def _cells(Nt: int = NT) -> CellSet:
    return CellSet.all_active(wilson.Grid(NF, Nt, DT))


def _dense_blocks(cov: EvolutionaryCovariance, nu: np.ndarray) -> np.ndarray:
    d, u = to_layout(*cov.blocks(nu), cov.sizes, "ragged", NUMPY)
    return np.asarray(blocks_to_dense(d, u, NUMPY))


@pytest.mark.parametrize("which", ["all", "cut", "padded"])
def test_interp_matches_the_direct_sum(which):
    Nt = NT + 4 if which == "padded" else NT
    cells = _cells(Nt)
    if which == "cut":
        cells = cells.drop_columns(3, 1)
    t_max = 1.25 * Nt * NF * DT if which == "padded" else None     # 1.25 x the grid's T: not the default
    ref = EvolutionaryCovariance(PSD, cells, builder="direct", assembly="dense", t_max=t_max).dense(NU)
    got = _dense_blocks(EvolutionaryCovariance(PSD, cells, builder="interp", t_max=t_max), NU)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_assemblies_agree(rng):
    cells = _cells()
    blk = EvolutionaryCovariance(PSD, cells)
    den = EvolutionaryCovariance(PSD, cells, assembly="dense", chol_impl="scipy", symmetrize=False)
    y = rng.standard_normal(N)
    Fb, Fd = blk.factor(NU), den.factor(NU)
    assert np.array_equal(blk.dense(NU), den.dense(NU))
    assert float(Fb.logdet) == pytest.approx(float(Fd.logdet), rel=1e-13)
    assert float(Fb.quad(y)) == pytest.approx(float(Fd.quad(y)), rel=1e-12)


def test_interp_is_the_kernel_chebyshev_path():
    plan = eq45_plan(NF, NT, DT)
    want = eq45_blocks_cheb(plan, eq45_coeffs_interp(lambda f, t: PSD(f, t, NU), plan.f, plan.t_max,
                                                     plan.P, k=plan.k_used))
    got = EvolutionaryCovariance(PSD, _cells()).blocks(NU)
    assert all(np.array_equal(a, b) for a, b in zip(got[0], want[0], strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(got[1], want[1], strict=True))


@pytest.mark.parametrize("ac", [-2.0, -0.7])
def test_stationary_limit_is_the_exact_covariance(ac):
    nu = np.array([ac, 0.0])
    exact = build_C_wdm_active(NF, NT, DT, lambda f: PSD(f, 0.0, nu))
    got = _dense_blocks(EvolutionaryCovariance(PSD, _cells()), nu)
    assert np.max(np.abs(got - exact)) < 1e-12 * np.max(np.abs(exact))


def test_arguments_are_checked():
    cells = _cells()
    with pytest.raises(ValueError, match="unknown builder"):
        EvolutionaryCovariance(PSD, cells, builder="bessel")
    with pytest.raises(TypeError, match="callable"):
        EvolutionaryCovariance(object(), cells)
    with pytest.raises(ValueError, match="assembly"):
        EvolutionaryCovariance(PSD, cells, assembly="banded")
    with pytest.raises(ValueError, match="builder 'direct' needs assembly='dense'"):
        EvolutionaryCovariance(PSD, cells, builder="direct")
    with pytest.raises(ValueError, match="pair mid-times"):                # raised lazily, by the plan
        EvolutionaryCovariance(PSD, cells, t_max=0.5 * T_REC).blocks(NU)
    assert not EvolutionaryCovariance(PSD, cells).jittable


@pytest.mark.parametrize("wrong", ["domain", "grid"])
def test_cells_must_be_a_cell_set(wrong):
    cells = _cells()
    arg = WDMDomain(cells) if wrong == "domain" else cells.grid
    with pytest.raises(TypeError, match=r"cells must be a CellSet.*domain\.cells"):
        EvolutionaryCovariance(PSD, arg)


def test_keep_mask_selects_the_cells_in_band_order():
    cells = _cells().drop_columns(2, 2)
    keep = keep_mask(cells)
    assert int(keep.sum()) == cells.size
    np.testing.assert_array_equal(np.flatnonzero(keep), cells.packed_positions())
