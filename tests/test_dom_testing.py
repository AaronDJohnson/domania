"""The dense oracles of ``domainia.testing`` against independent constructions."""

from __future__ import annotations

import re

import numpy as np
import pytest

from domainia import wilson
from domainia.wilson.evolutionary import exact_covariance, time_domain_covariance
from domainia import testing as T

# The WDM Paper II toy models' test configuration: 8 x 16 cells, dt = 1/16, f_min = 0.5,
# f_ref = 1.
NF, NT, DT = 8, 16, 1.0 / 16
N = NF * NT
F_MIN, F_REF = 0.5, 1.0
#: Every cell.
MASK = np.ones((NF, NT), bool)


def L_of(f):
    return np.log(np.maximum(np.asarray(f, dtype=float), F_MIN) / F_REF)


def drift(ac, ad):
    """``S(f, t) = (max(f, f_min)/f_ref)^(ac + ad (t/T - 1/2))``, elementwise (A_n = 1)."""
    T_obs = N * DT
    return lambda f, t: np.exp((ac + ad * (np.asarray(t) / T_obs - 0.5)) * L_of(f))


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def test_dense_W_rows_are_the_atoms(rng):
    W = T.dense_W(NF, NT, DT, MASK)
    x = rng.standard_normal(N)
    assert np.max(np.abs(W @ x - wilson.forward_time(x, NF, NT, DT)[MASK])) < 1e-12
    assert np.max(np.abs(W @ W.T - DT * np.eye(N))) < 1e-14
    sub = MASK.copy()
    sub[:, :3] = sub[:, -1:] = False
    assert np.array_equal(T.dense_W(NF, NT, DT, sub), W[sub[MASK]])


def test_congruence_is_the_exact_covariance():
    Sn = drift(-2.0, 1.2)
    got = T.wdm_congruence(time_domain_covariance(N, DT, Sn), NF, NT, DT, MASK)
    assert rel(got, exact_covariance(NF, NT, DT, Sn)) < 1e-12


@pytest.mark.parametrize("ac", [-2.0, -0.7])
def test_eq45_is_exact_for_stationary_noise(ac):
    """Ported from the toy models' test of the same name."""
    C = T.eq45_dense(NF, NT, DT, MASK, drift(ac, 0.0))
    assert rel(C, exact_covariance(NF, NT, DT, drift(ac, 0.0))) < 1e-10


def test_eq45_differs_from_exact_at_second_order_with_drift():
    """Ported from the toy models' test of the same name (the test's small edges: 3
    leading and 1 trailing column)."""
    C = T.eq45_dense(NF, NT, DT, MASK, drift(-2.0, 1.2))
    exact = exact_covariance(NF, NT, DT, drift(-2.0, 1.2))
    interior = ~np.isin(wilson.grid.cells(NF, NT)[1].ravel(), np.r_[0:3, NT - 1])
    d = np.diag(exact)[interior]
    assert np.max(np.abs(np.diag(C)[interior] / d - 1)) < 0.05


def test_the_oracles_compute_in_the_window_they_are_given():
    """With ALT_WINDOW: ``dense_W`` is ``build_W``, the congruence is ``W S W^T``, and Eq. 45
    of a stationary spectrum is the exact covariance; each differs from Meyer's."""
    W = T.dense_W(NF, NT, DT, MASK, window=T.ALT_WINDOW)
    np.testing.assert_array_equal(W, wilson.matrix.build_W(NF, NT, DT, window=T.ALT_WINDOW))
    S = np.diag(1.0 + np.arange(N) / N)                  # not the identity: W I W^T = dt I for any window
    alt = T.wdm_congruence(S, NF, NT, DT, MASK, window=T.ALT_WINDOW)
    np.testing.assert_allclose(alt, W @ S @ W.T, rtol=0, atol=1e-12)
    assert not np.allclose(alt, T.wdm_congruence(S, NF, NT, DT, MASK))
    got = T.eq45_dense(NF, NT, DT, MASK, drift(-1.2, 0.0), window=T.ALT_WINDOW)
    exact = wilson.matrix.build_C_wdm(NF, NT, DT, lambda f: drift(-1.2, 0.0)(f, 0.0),
                                      window=T.ALT_WINDOW)
    assert rel(got, exact) < 1e-10
    assert not np.allclose(got, T.eq45_dense(NF, NT, DT, MASK, drift(-1.2, 0.0)))


def test_the_oracles_refuse_a_mask_of_the_wrong_kind_or_value():
    """A mask that is not boolean is a wrong kind; a wrong shape (flat, or padded
    ``(Nf + 1, Nt)``) is a wrong value.  ``eq45_dense`` also needs a callable spectrum."""
    for mask, err, message in (
        (MASK.astype(int), TypeError, "mask must be a boolean cell mask, got dtype int64"),
        (None, TypeError, "mask must be a boolean cell mask, got dtype object"),
        (MASK.reshape(-1), ValueError, "mask must have shape (Nf, Nt) = (8, 16), got (128,)"),
        (np.ones((NF + 1, NT), bool), ValueError, "mask must have shape (Nf, Nt) = (8, 16), got (9, 16)"),
    ):
        match = f"^{re.escape(message)}$"
        with pytest.raises(err, match=match):
            T.dense_W(NF, NT, DT, mask)
        with pytest.raises(err, match=match):
            T.wdm_congruence(np.eye(N), NF, NT, DT, mask)
        with pytest.raises(err, match=match):
            T.eq45_dense(NF, NT, DT, mask, drift(0.0, 0.0))
    with pytest.raises(TypeError, match=r"^Sn must be callable, got float$"):
        T.eq45_dense(NF, NT, DT, MASK, 1.0)
