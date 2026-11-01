"""L-A6: (M Lambda M)_KK exactly (domainia.literal.MaskedCovariance): the literal matrix (every kept
row of M, the default) against the dense oracle M~_K Lambda M~_K^T (dense W' chi W'^T / dt), by QR
(default) and by the arrow's Cholesky; a partial edge set (the truncated mask M~) as a Schur
complement over a diagonal / block-tridiagonal / dense interior; explicit mask rows (the GB's
PadMask route) equal to the pad's; jitted JAX; -inf when not positive definite."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from domainia import wilson as wdm
from domainia.covariance import DiagonalCovariance
from dom_literal_cases import NUS, rel, setup_case
from domainia.literal import CenteredPad, MaskedCovariance, masked_covariance

CASES = [(wdm.Grid(8, 16, 1 / 16), 8), (wdm.Grid(16, 32, 1 / 32), 8)]


def _ids(case):
    g, p = case
    return f"{g.Nf}x{g.Nt}p{p}"


def _dense_lnl_terms(S, y):
    c = cho_factor(0.5 * (S + S.T), lower=True)
    return float(y @ cho_solve(c, y)), 2 * float(np.sum(np.log(np.diag(c[0]))))


@pytest.mark.parametrize("case", CASES, ids=_ids)
@pytest.mark.parametrize("base", ["band", "mid", "tri"])
def test_the_literal_matrix_is_the_dense_oracle(case, base):
    g, p = case
    pad, cov = setup_case(g, p, base)
    qr, chol = MaskedCovariance(cov, pad), MaskedCovariance(cov, pad, factorization="cholesky")
    oracle = masked_covariance(cov, pad, dense=True)
    assert qr.e.size == pad.kept.size and qr.i.size == 0                 # E = K at round-off
    y = pad.lift(np.random.default_rng(3).standard_normal(g.N))
    for nu in NUS:
        S = oracle.dense(nu)
        assert rel(qr.dense(nu), S) <= 1e-12
        quad, logdet = _dense_lnl_terms(S, y)
        for F in (qr.factor(nu), chol.factor(nu), oracle.factor(nu)):
            assert bool(F.ok)
            assert float(F.logdet) == pytest.approx(logdet, abs=1e-7)
            assert float(F.quad(y)) == pytest.approx(quad, rel=1e-8)


@pytest.mark.parametrize("base,tri_assembly", [("band", "blocktri"), ("tri", "blocktri"), ("tri", "dense")])
def test_a_partial_edge_set_is_the_truncated_mask(base, tri_assembly):
    """E = the kept cells leaking more than 0.1 % (not the literal default; the structure test):
    the matrix is M~_K Lambda M~_K^T (M~: e_p rows outside E), Schur over Lambda_II or QR."""
    g, p = CASES[1]
    pad, cov = setup_case(g, p, base, tri_assembly=tri_assembly)
    e = pad.edge_set(1e-3)
    assert 0 < e.size < pad.kept.size
    oracle = MaskedCovariance(cov, pad, edge=e, assembly="dense")
    y = pad.lift(np.random.default_rng(4).standard_normal(g.N))
    for nu in NUS:
        S = oracle.dense(nu)
        quad, logdet = _dense_lnl_terms(S, y)
        for fac in ("cholesky", "qr"):
            arrow = MaskedCovariance(cov, pad, edge=e, factorization=fac)
            assert rel(arrow.dense(nu), S) <= 1e-12
            F = arrow.factor(nu)
            assert float(F.logdet) == pytest.approx(logdet, abs=1e-8)
            assert float(F.quad(y)) == pytest.approx(quad, rel=1e-9)


@pytest.mark.parametrize("base", ["band", "tri"])
def test_explicit_mask_rows_are_the_pad(base):
    """The GB route (rows of M given, e.g. a PadMask with every row): the same matrix."""
    g, p = CASES[1]
    pad, cov = setup_case(g, p, base)
    y = pad.lift(np.random.default_rng(5).standard_normal(g.N))
    for e in (np.arange(pad.kept.size), pad.edge_set(1e-3)):
        rows = MaskedCovariance(cov, kpos=pad.kpos, rows=pad.mask_rows(e), edge=e)
        ref = MaskedCovariance(cov, pad, edge=e)
        for nu in NUS:
            assert rel(rows.dense(nu), ref.dense(nu)) <= 1e-13
            Fr, Fp = rows.factor(nu), ref.factor(nu)
            assert float(Fr.logdet) == pytest.approx(float(Fp.logdet), abs=1e-9)
            assert float(Fr.quad(y)) == pytest.approx(float(Fp.quad(y)), rel=1e-10)
    full = np.arange(pad.kept.size)
    dense_rows = MaskedCovariance(cov, kpos=pad.kpos, rows=pad.mask_rows(full), edge=full, assembly="dense")
    assert rel(dense_rows.dense(NUS[0]), masked_covariance(cov, pad, dense=True).dense(NUS[0])) <= 1e-13


def test_not_positive_definite_is_minus_infinity():
    g, p = CASES[0]
    pad = CenteredPad(g, p=p)
    neg = DiagonalCovariance(lambda nu, be: -be.xp.ones(pad.cells.size), pad.cells.size)
    for kw in (dict(), dict(factorization="cholesky"), dict(assembly="dense")):
        assert not bool(MaskedCovariance(neg, pad, **kw).factor(np.zeros(2)).ok)


def test_bad_input_is_refused():
    g, p = CASES[0]
    pad, cov = setup_case(g, p, "band")
    with pytest.raises(ValueError, match="base covers"):
        MaskedCovariance(cov, CenteredPad(g, p=4))
    with pytest.raises(ValueError, match="edge positions"):
        MaskedCovariance(cov, pad, edge=[pad.kept.size])
    with pytest.raises(ValueError, match="assembly"):
        MaskedCovariance(cov, pad, assembly="schur")
    with pytest.raises(ValueError, match="factorization"):
        MaskedCovariance(cov, pad, factorization="lu")
    with pytest.raises(ValueError, match="chol_impl"):
        MaskedCovariance(cov, pad, chol_impl="lu")
    with pytest.raises(ValueError, match="either pad"):
        MaskedCovariance(cov)
    with pytest.raises(ValueError, match="kpos and edge"):
        MaskedCovariance(cov, rows=pad.mask_rows())
    with pytest.raises(ValueError, match="one row of M each"):
        MaskedCovariance(cov, kpos=pad.kpos, rows=pad.mask_rows([0, 1]), edge=[0])


@pytest.mark.jax
@pytest.mark.parametrize("base", ["band", "tri"])
def test_jax_jit_matches_numpy(jax_be, base):
    g, p = CASES[1]
    pad, cov = setup_case(g, p, base)
    y = pad.lift(np.random.default_rng(6).standard_normal(g.N))
    for kw in (dict(), dict(factorization="cholesky"), dict(edge=pad.edge_set(1e-3), factorization="cholesky")):
        arrow = MaskedCovariance(cov, pad, **kw)
        assert arrow.jittable
        for nu in NUS:
            Fn, Fj = arrow.factor(nu), arrow.factor(jax_be.asarray(nu))
            assert float(Fj.logdet) == pytest.approx(float(Fn.logdet), abs=1e-8)
            assert float(Fj.quad(jax_be.asarray(y))) == pytest.approx(float(Fn.quad(y)), rel=1e-9)
        assert len(arrow.compiled_keys()) == 1
