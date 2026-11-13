"""The dense oracles of ``domainia.testing`` against independent constructions."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson
from domainia.wilson.evolutionary import exact_covariance, time_domain_covariance
from domainia import testing as T
from domainia.domains import CellSet

# The WDM Paper II toy models' test configuration: 8 x 16 cells, dt = 1/16, f_min = 0.5,
# f_ref = 1.
NF, NT, DT = 8, 16, 1.0 / 16
N = NF * NT
F_MIN, F_REF = 0.5, 1.0
GRID = wilson.Grid(NF, NT, DT)


def L_of(f):
    return np.log(np.maximum(np.asarray(f, dtype=float), F_MIN) / F_REF)


def drift(ac, ad):
    """``S(f, t) = (max(f, f_min)/f_ref)^(ac + ad (t/T - 1/2))``, elementwise (A_n = 1)."""
    T_obs = N * DT
    return lambda f, t: np.exp((ac + ad * (np.asarray(t) / T_obs - 0.5)) * L_of(f))


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def test_dense_W_rows_are_the_atoms(rng):
    cells = CellSet.all_active(GRID)
    W = T.dense_W(cells)
    x = rng.standard_normal(N)
    packed = wilson.grid.pack(wilson.forward_time(x, NF, NT, DT), NF, NT)
    assert np.max(np.abs(W @ x - packed)) < 1e-12
    assert np.max(np.abs(W @ W.T - DT * np.eye(N))) < 1e-14
    sub = cells.drop_columns(3, 1)
    assert np.array_equal(T.dense_W(sub), W[sub.packed_positions()])


def test_congruence_is_the_exact_covariance():
    Sn = drift(-2.0, 1.2)
    cells = CellSet.all_active(GRID)
    got = T.wdm_congruence(time_domain_covariance(N, DT, Sn), cells)
    assert rel(got, exact_covariance(NF, NT, DT, Sn)) < 1e-12


@pytest.mark.parametrize("ac", [-2.0, -0.7])
def test_eq45_is_exact_for_stationary_noise(ac):
    """Ported from the toy models' test of the same name."""
    C = T.eq45_dense(CellSet.all_active(GRID), drift(ac, 0.0))
    assert rel(C, exact_covariance(NF, NT, DT, drift(ac, 0.0))) < 1e-10


def test_eq45_differs_from_exact_at_second_order_with_drift():
    """Ported from the toy models' test of the same name (the test's small edges: 3
    leading and 1 trailing column)."""
    cells = CellSet.all_active(GRID)
    C = T.eq45_dense(cells, drift(-2.0, 1.2))
    exact = exact_covariance(NF, NT, DT, drift(-2.0, 1.2))
    interior = ~np.isin(cells.n_array, wilson.grid.edge_columns(NT, 3, 1))
    d = np.diag(exact)[interior]
    assert np.max(np.abs(np.diag(C)[interior] / d - 1)) < 0.05
