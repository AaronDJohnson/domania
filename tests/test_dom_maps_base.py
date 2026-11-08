"""D-B1: the LinearMap base, PushedCovariance, Identity, Compose, Restrict, BinSelect."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from domainia import wilson as wdm
from dom_db_inputs import DenseCov, jax_or_skip

from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.maps import (BinSelect, Compose, Identity, LinearMap, PushedCovariance, Restrict,
                           domain_size)

GRID = wdm.Grid(8, 16, 0.5)


@dataclasses.dataclass(frozen=True, eq=False)
class _DenseComplex(LinearMap):
    """A test-local complex-linear map ``z -> A z`` between Fourier windows."""

    src: FourierDomain
    dst: FourierDomain
    A: np.ndarray
    kind = "complex"

    def apply(self, z, be=None):
        return z @ self.A.T

    def adjoint(self, y, be=None):
        return y @ np.conj(self.A)


def _cells():
    full = CellSet.all_active(GRID)
    return WDMDomain(full), WDMDomain(full.drop_columns(2, 1))


def test_domain_size():
    full, cut = _cells()
    assert domain_size(TimeDomain(128, 0.5)) == 128
    assert domain_size(FourierDomain(128, 0.5, 3, 20, "rfft")) == 20
    assert domain_size(full) == 128 and domain_size(cut) == 128 - 3 * 7 - 1 - 1   # DC and Nyquist lose column 0 only


def test_restrict_selects_in_dst_order_and_is_its_own_adjoint():
    full, cut = _cells()
    R = Restrict(full, cut)
    assert isinstance(R, LinearMap) and R.kind == "real" and R.shape == (len(cut.cells.m), 128)
    x = np.random.default_rng(0).standard_normal((3, 128))
    keep = ~np.isin(np.asarray(full.cells.n), wdm.grid.edge_columns(16, 2, 1))
    assert np.array_equal(R.apply(x), x[:, keep])
    y = np.random.default_rng(1).standard_normal((3, R.shape[0]))
    assert np.sum(R.apply(x) * y) == pytest.approx(np.sum(x * R.adjoint(y)), rel=1e-14)
    assert np.array_equal(R.dense(), np.eye(128)[keep])
    with pytest.raises(ValueError, match="not a subset"):
        Restrict(cut, full)


def test_restrict_checks_its_domains_even_when_pos_is_given():
    """The grid check reads the domains only, so it runs on every construction route
    (``pos`` given: ``on``, ``tree_unflatten``, ``dataclasses.replace``).  The window check
    is tests/test_dom_windows.py's."""
    full, cut = _cells()
    R = Restrict(full, cut)
    other = WDMDomain(CellSet.all_active(wdm.Grid(8, 16, 0.25)).drop_columns(2, 1))
    for kw in ({}, {"pos": R.pos}):
        with pytest.raises(ValueError, match="not a subset of src: its cells live on another grid"):
            Restrict(full, other, **kw)
    assert np.array_equal(dataclasses.replace(R).pos, R.pos)


def test_push_is_the_congruence_and_composes():
    full, cut = _cells()
    rng = np.random.default_rng(2)
    A = rng.standard_normal((128, 128))
    S = A @ A.T / 128
    R = Restrict(full, cut)
    P = R.push(DenseCov(S))
    assert isinstance(P, PushedCovariance)
    D = R.dense()
    assert np.allclose(P.dense(), D @ S @ D.T, rtol=0, atol=1e-14)
    assert np.allclose(P.diag(), np.diag(D @ S @ D.T), rtol=0, atol=1e-14)
    assert np.allclose(P.rows([0, 5]), (D @ S @ D.T)[[0, 5]], rtol=0, atol=1e-14)
    # pushes compose through matvec: R2 (R S R^T) R2^T
    cut2 = WDMDomain(cut.cells.drop_columns(1, 0))
    R2 = Restrict(cut, cut2)
    D2 = R2.dense()
    assert np.allclose(R2.push(P).dense(), D2 @ D @ S @ D.T @ D2.T, rtol=0, atol=1e-14)
    C = Compose(R2, R)
    assert C.src == full and C.dst == cut2 and C.kind == "real"
    assert np.array_equal(C.apply(np.eye(128)).T, (D2 @ D))
    assert np.allclose(C.push(DenseCov(S)).dense(), D2 @ D @ S @ D.T @ D2.T, rtol=0, atol=1e-14)
    assert np.array_equal(C.rows([1, 2]), (D2 @ D)[[1, 2]])


def test_compose_refuses_mismatched_domains_and_identity_is_neutral():
    full, cut = _cells()
    with pytest.raises(ValueError, match="cannot compose"):
        Compose(Restrict(full, cut), Restrict(full, cut))
    x = np.arange(128.0)
    I = Identity(full)
    assert I.dst == full and np.array_equal(I.apply(x), x) and np.array_equal(I.adjoint(x), x)
    assert np.array_equal(Compose(Restrict(full, cut), I).apply(x), Restrict(full, cut).apply(x))


def test_binselect_windows_and_complex_push():
    store = FourierDomain(4096, 5.0, 100, 60, "sigma0", sigma0=3.0)
    S = BinSelect(store, (110, 30))
    assert S.dst == FourierDomain(4096, 5.0, 110, 30, "sigma0", sigma0=3.0) and S.kind == "complex"
    assert BinSelect(store, S.dst).dst == S.dst
    for other in (FourierDomain(4096, 5.0, 110, 30, "sigma0", sigma0=4.0),
                  FourierDomain(4096, 5.0, 110, 30, "rfft")):
        with pytest.raises(ValueError, match="sigma0"):
            BinSelect(store, other)
    rng = np.random.default_rng(3)
    z = rng.standard_normal((2, 60)) + 1j * rng.standard_normal((2, 60))
    assert np.array_equal(S.apply(z), z[:, 10:40])
    y = rng.standard_normal((2, 30)) + 1j * rng.standard_normal((2, 30))
    lhs = np.real(np.vdot(y, S.apply(z)))
    assert lhs == pytest.approx(np.real(np.vdot(S.adjoint(y), z)), rel=1e-14)
    B = rng.standard_normal((60, 60)) + 1j * rng.standard_normal((60, 60))
    M = B @ B.conj().T
    assert np.array_equal(S.push(DenseCov(M)).dense(), M[10:40, 10:40])
    with pytest.raises(ValueError, match="not inside"):
        BinSelect(store, (150, 30))
    nominal = BinSelect(S.dst, (115, 10))           # store -> window -> nominal
    assert nominal.dst.k0 == 115 and np.array_equal(nominal.apply(S.apply(z)), z[:, 15:25])
    # Compose.rows of complex-linear maps is (outer inner)[idx], never outer conj(inner)
    A = rng.standard_normal((60, 30)) + 1j * rng.standard_normal((60, 30))
    lift = _DenseComplex(S.dst, store, A)           # window (30 bins) -> store (60 bins)
    sel_lift = Compose(S, lift)                     # real 0/1 outer rows, complex inner
    gap = np.max(np.abs(sel_lift.rows([0, 7]) - A[[10, 17]]))
    assert gap <= 1e-14 * np.max(np.abs(A))
    both = Compose(lift, sel_lift)                  # complex outer rows, complex inner
    ref = (A @ A[10:40])[[0, 7, 59]]
    for got in (both.rows([0, 7, 59]), LinearMap.rows(both, [0, 7, 59])):
        assert np.max(np.abs(got - ref)) <= 1e-13 * np.max(np.abs(ref))


def test_maps_are_frozen_dataclasses():
    full, cut = _cells()
    R = Restrict(full, cut)
    with pytest.raises(dataclasses.FrozenInstanceError):
        R.pos = None  # type: ignore[misc]


def test_push_without_matvec_is_a_type_error_naming_translate_operator():
    """Spec section 4.2's WDM -> time push (W^T Lambda W / dt^2) needs a fixed-noise-state
    ``matvec`` that no cell covariance has: a clear TypeError, not a bare AttributeError,
    from ``rows`` and ``matvec`` of a real or analysis push, naming the operator route
    (``CovarianceOperator``, then ``translate_operator``)."""
    from domainia.covariance import DiagonalCovariance
    from domainia.maps import TimeToFourier, WDMToTime

    full, _ = _cells()
    time = TimeDomain(GRID.N, GRID.dt)
    cov = DiagonalCovariance(np.ones(GRID.N), GRID.N)
    P = WDMToTime(full, time).push(cov)
    msg = (r"DiagonalCovariance has no matvec: .*domainia\.operators\.CovarianceOperator.*"
           r"domainia\.translate\.translate_operator")
    with pytest.raises(TypeError, match=msg):
        P.rows([0])
    with pytest.raises(TypeError, match=msg):
        P.dense()
    with pytest.raises(TypeError, match=msg):
        P.matvec(np.ones((1, GRID.N)))
    with pytest.raises(TypeError, match=msg):                 # an analysis push reads matvec too
        TimeToFourier(time).push(cov).rows([0])


def test_a_push_needs_a_linear_map_and_a_covariance():
    """``PushedCovariance`` refuses, at construction, a map that is not a ``LinearMap`` and a
    covariance with neither ``matvec`` nor ``dense``."""
    time = TimeDomain(GRID.N, GRID.dt)
    S = DenseCov(np.eye(GRID.N))
    with pytest.raises(TypeError, match="LinearMap"):
        PushedCovariance(object(), S)
    with pytest.raises(TypeError, match="LinearMap"):
        PushedCovariance(time, S)
    for cov in (np.eye(GRID.N), None, time):
        with pytest.raises(TypeError, match=r"matvec.*dense"):
            Identity(time).push(cov)
        with pytest.raises(TypeError, match=r"matvec.*dense"):
            PushedCovariance(Identity(time), cov)
    assert Identity(time).push(S).cov is S                       # either method is enough


@pytest.mark.jax
def test_maps_are_pytrees_passed_to_jit():
    jax, JB = jax_or_skip()
    full, cut = _cells()
    R = Restrict(full, cut).on(JB)
    assert isinstance(R.pos, jax.Array)
    leaves, tree = jax.tree_util.tree_flatten(R)
    assert len(leaves) == 1 and jax.tree_util.tree_unflatten(tree, leaves).pos is leaves[0]
    x = np.random.default_rng(4).standard_normal((3, 128))
    out = jax.jit(lambda m, v: m.apply(v))(R, JB.asarray(x))
    assert isinstance(out, jax.Array)
    assert np.array_equal(np.asarray(out), Restrict(full, cut).apply(x))
    back = jax.jit(lambda m, v: m.adjoint(v))(R, out)
    assert isinstance(back, jax.Array)
    assert np.array_equal(np.asarray(back), Restrict(full, cut).adjoint(np.asarray(out)))
    # a kept JAX backend object must not move float64 constants with x64 off (spec §3.3)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="float64"):
            Restrict(full, cut).on(JB)
    finally:
        jax.config.update("jax_enable_x64", True)
