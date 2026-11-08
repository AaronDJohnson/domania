"""D-B3: TimeToFourier, CropTime, ZeroPadTime -- the Whittle bins, the crop, the pad."""

from __future__ import annotations

import numpy as np
import pytest
from domainia import wilson
from dom_db_inputs import SIN_DT, SIN_NF, SIN_NT, SIN_PAD_COLUMNS, DenseCov

from domainia.domains import CellSet, TimeDomain, WDMDomain
from domainia.maps import Compose, CropTime, TimeToFourier, TimeToWDM, ZeroPadTime

N, DT = SIN_NF * SIN_NT, SIN_DT


def test_fourier_rows_adjoint_and_orthonormal_coordinates():
    td = TimeDomain(128, 0.5)
    F = TimeToFourier(td, fmin=0.2)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 128))
    Fd = F.dense()
    assert np.max(np.abs(x @ Fd.T - F.apply(x))) <= 1e-12
    y = rng.standard_normal((3, F.dst.n)) + 1j * rng.standard_normal((3, F.dst.n))
    lhs = np.sum(np.real(np.conj(y) * F.apply(x)))
    assert abs(lhs - np.sum(x * F.adjoint(y))) <= 1e-12 * np.sum(np.abs(y))
    O = F.orthonormal_rows()
    assert np.max(np.abs(O @ O.T - np.eye(2 * F.dst.n))) <= 1e-14
    R = F.apply(x[0])
    coords = np.sqrt(2.0 / 128) * np.stack([R.real, -R.imag], axis=1).ravel()
    assert np.max(np.abs(O @ x[0] - coords)) <= 1e-14


def test_push_of_a_stationary_covariance_is_the_whittle_diagonal():
    """F Sigma F^H = diag(N S(f_k) / (2 dt)) for the periodic stationary Sigma."""
    n, dt = 128, 0.5
    F = TimeToFourier(TimeDomain(n, dt), fmin=0.2)
    Sn = lambda f, t: np.maximum(f, 0.05) ** -1.5 + 0.0 * t  # noqa: E731
    Sigma = wilson.evolutionary.time_domain_covariance(n, dt, Sn)
    P = F.push(DenseCov(Sigma)).dense()
    fk = np.fft.rfftfreq(n, dt)[F.dst.k0:F.dst.k0 + F.dst.n]
    assert np.max(np.abs(np.real(np.diag(P)) / (n * Sn(fk, 0.0) / (2 * dt)) - 1)) <= 1e-13
    assert np.max(np.abs(P - np.diag(np.diag(P)))) <= 1e-13 * np.max(np.abs(P))


def test_crop_and_zero_pad_adjoints():
    td = TimeDomain(96, 0.25, 1.0)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 96))
    C = CropTime(td, 10, 40)
    y = rng.standard_normal((2, 40))
    assert np.sum(C.apply(x) * y) == pytest.approx(np.sum(x * C.adjoint(y)), rel=1e-14)
    assert C.dst.t0 == 1.0 + 10 * 0.25
    with pytest.raises(ValueError, match="not inside"):
        CropTime(td, 80, 40)
    with pytest.raises(TypeError):
        CropTime(td, 10.0, 40)
    with pytest.raises(ValueError, match="last axis"):
        C.apply(x[:, :60])
    Z = ZeroPadTime(td, 32)
    assert Z.dst == TimeDomain(128, 0.25, 1.0)
    with pytest.raises(ValueError, match=">= 0"):
        ZeroPadTime(td, -5)
    z = Z.apply(x)
    assert np.array_equal(z[:, :96], x) and np.all(z[:, 96:] == 0)
    w = rng.standard_normal((2, 128))
    assert np.sum(z * w) == pytest.approx(np.sum(x * Z.adjoint(w)), rel=1e-14)


def test_sinusoid_pad_record_is_bitwise_the_oracle_extension():
    """``TimeToWDM(extended) . ZeroPadTime`` is the padded transform of the WDM Paper II toy
    models bit for bit: ``g.forward_many(ext)`` with ``ext[:, :n] = X``."""
    td = TimeDomain(N, DT)
    Z = ZeroPadTime(td, SIN_PAD_COLUMNS * SIN_NF)
    g = wilson.Grid(SIN_NF, SIN_NT + SIN_PAD_COLUMNS, DT)
    M = Compose(TimeToWDM(Z.dst, WDMDomain(CellSet.all_active(g))), Z)
    X = np.random.default_rng(2).standard_normal((3, N))
    ext = np.zeros((3, g.N))
    ext[:, :N] = X
    ref = wilson.forward_time_many(ext, g.Nf, g.Nt, DT)[:, wilson.active_mask(g.Nf, g.Nt)]
    assert np.array_equal(M.apply(X), ref)
