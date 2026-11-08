"""D-B4: pad_maps / crop_maps (exact Dirichlet kernels), FourierZeroPad, FourierCrop."""

from __future__ import annotations

import numpy as np
import pytest
from dom_db_inputs import DenseCov, jax_or_skip

from domainia.domains import FourierDomain
from domainia.maps import FourierCrop, FourierZeroPad, crop_maps, pad_maps

NS = 256


def _band_limited(rng):
    """A record whose spectrum lives on bins 20..60 (no DC, no Nyquist): the maps are exact."""
    X = np.zeros(NS // 2 + 1, complex)
    X[20:61] = rng.standard_normal(41) + 1j * rng.standard_normal(41)
    return X, np.fft.irfft(X, n=NS)


SRC = FourierDomain(NS, 1.0, 20, 41, "rfft")


def test_zero_pad_reproduces_the_padded_record_rfft():
    X, x = _band_limited(np.random.default_rng(0))
    Np = 320
    padded = np.zeros(Np)
    padded[:NS] = x
    Xp = np.fft.rfft(padded)
    Z = FourierZeroPad(SRC, FourierDomain(Np, 1.0, 5, 80, "rfft"))
    assert Z.kind == "widely" and not hasattr(Z, "scale")
    assert np.max(np.abs(Z.apply(X[20:61]) - Xp[5:85])) <= 1e-14 * np.max(np.abs(Xp))


def test_crop_reproduces_the_cropped_record_rfft():
    X, x = _band_limited(np.random.default_rng(1))
    start, Nc = 24, 200
    Xc = np.fft.rfft(x[start:start + Nc])
    C = FourierCrop(SRC, FourierDomain(Nc, 1.0, 10, 70, "rfft"), start)
    assert not hasattr(C, "scale")
    assert np.max(np.abs(C.apply(X[20:61]) - Xc[10:80])) <= 1e-14 * np.max(np.abs(Xc))


@pytest.mark.parametrize("N", [256, 255])
def test_full_record_bins_pad_and_crop_to_the_exact_rfft(N):
    """Spec section 1.5: the self-conjugate source bins (DC, and Nyquist for even N) count
    once, so pad and crop of every bin of a record with non-zero DC and Nyquist content are
    the rfft of the padded or cropped series.  The output end bins are real to rounding."""
    x = np.random.default_rng(N).standard_normal(N) + 0.75             # non-zero DC
    x[::2] += 0.5                                                      # non-zero Nyquist (even N)
    X = np.fft.rfft(x)
    assert abs(X[0]) > 1 and (N % 2 or abs(X[-1]) > 1)
    src = FourierDomain(N, 1.0, 0, N // 2 + 1, "rfft")
    for Np in (N + 64, N + 65):
        want = np.fft.rfft(np.r_[x, np.zeros(Np - N)])
        got = FourierZeroPad(src, FourierDomain(Np, 1.0, 0, Np // 2 + 1, "rfft")).apply(X)
        assert np.max(np.abs(got - want)) <= 1e-13 * np.max(np.abs(want))
        assert abs(got[0].imag) <= 1e-13 * np.max(np.abs(want))
    for start, Nc in ((24, 200), (7, 141)):
        want = np.fft.rfft(x[start:start + Nc])
        got = FourierCrop(src, FourierDomain(Nc, 1.0, 0, Nc // 2 + 1, "rfft"), start).apply(X)
        assert np.max(np.abs(got - want)) <= 1e-13 * np.max(np.abs(want))
        assert abs(got[0].imag) <= 1e-13 * np.max(np.abs(want))


def test_maps_match_direct_transforms():
    """Ported from GB tests/test_matrix.py:19-40 (pad and crop; the taper map is dropped):
    Nf = 24, NT = 16, store bins 40..89, atol 1e-13."""
    rng = np.random.default_rng(1)
    Nf, NT = 24, 16
    N = NT * Nf
    j = np.arange(40, 90)
    z = np.zeros(N // 2 + 1, complex)
    z[j] = rng.standard_normal(j.size) + 1j * rng.standard_normal(j.size)
    x = np.fft.irfft(z, n=N)
    store = FourierDomain(N, 1.0, 40, 50, "rfft")
    Np = (NT + 4) * Nf
    Z = FourierZeroPad(store, FourierDomain(Np, 1.0, 30, 80, "rfft"))
    want = np.fft.rfft(np.r_[x, np.zeros(Np - N)])[30:110]
    assert np.allclose(Z.apply(z[j]), want, rtol=0, atol=1e-13)
    start, Nc = 3 * Nf, (NT - 4) * Nf
    C = FourierCrop(store, FourierDomain(Nc, 1.0, 20, 50, "rfft"), start)
    assert np.allclose(C.apply(z[j]), np.fft.rfft(x[start:start + Nc])[20:70], rtol=0, atol=1e-13)


def test_functions_are_the_oracle_kernels():
    """pad_maps/crop_maps keep the oracle signature (D-C's mask_rows calls crop_maps on chunks)."""
    A, B = pad_maps(NS, 320, np.arange(5, 85), np.arange(20, 61))
    Z = FourierZeroPad(SRC, FourierDomain(320, 1.0, 5, 80, "rfft"))
    assert np.array_equal(A, Z.A) and np.array_equal(B, Z.B)
    A2, B2 = crop_maps(NS, 24, 200, np.arange(10, 80), np.arange(20, 61))
    C = FourierCrop(SRC, FourierDomain(200, 1.0, 10, 70, "rfft"), 24)
    assert np.array_equal(A2, C.A) and np.array_equal(B2, C.B)
    # chunking the input bins is exact (elementwise kernels)
    Ac, Bc = crop_maps(NS, 24, 200, np.arange(10, 80), np.arange(20, 40))
    assert np.array_equal(Ac, A2[:, :20]) and np.array_equal(Bc, B2[:, :20])


def test_pad_and_crop_keep_the_units_and_sigma0():
    """No map rescales silently: ``dst`` must have ``src``'s ``(units, sigma0)``."""
    src = FourierDomain(NS, 1.0, 20, 41, "sigma0", sigma0=2.0)
    for units, kw in (("rfft", {}), ("sigma0", {"sigma0": 3.0})):
        with pytest.raises(ValueError, match="sigma0"):
            FourierZeroPad(src, FourierDomain(320, 1.0, 5, 80, units, **kw))
        with pytest.raises(ValueError, match="sigma0"):
            FourierCrop(src, FourierDomain(200, 1.0, 10, 70, units, **kw), 24)
    pad = FourierZeroPad(src, FourierDomain(320, 1.0, 5, 80, "sigma0", sigma0=2.0))
    crop = FourierCrop(src, FourierDomain(200, 1.0, 10, 70, "sigma0", sigma0=2.0), 24)
    assert pad.dst.sigma0 == crop.dst.sigma0 == 2.0


@pytest.mark.parametrize("cls", ["pad", "crop"])
def test_widely_linear_adjoint_and_no_push(cls):
    rng = np.random.default_rng(2)
    if cls == "pad":
        M = FourierZeroPad(SRC, FourierDomain(320, 1.0, 5, 80, "rfft"))
    else:
        M = FourierCrop(SRC, FourierDomain(200, 1.0, 10, 70, "rfft"), 24)
    z = rng.standard_normal((2, 41)) + 1j * rng.standard_normal((2, 41))
    y = rng.standard_normal((2, M.dst.n)) + 1j * rng.standard_normal((2, M.dst.n))
    lhs = np.real(np.vdot(y, M.apply(z)))
    assert abs(lhs - np.real(np.vdot(M.adjoint(y), z))) <= 1e-12 * abs(lhs)
    with pytest.raises(TypeError, match="widely linear"):
        M.push(DenseCov(np.eye(41)))
    # no complex row matrix exists (rows of the adjoint would be A[k] + conj(B[k])): refuse
    with pytest.raises(TypeError, match="widely linear"):
        M.rows([0])
    with pytest.raises(TypeError, match="widely linear"):
        M.dense()
    if cls == "crop":           # start is a sample index (CropTime's operator.index rule)
        assert type(FourierCrop(SRC, M.dst, np.int64(24)).start) is int
        with pytest.raises(TypeError):
            FourierCrop(SRC, M.dst, 24.0)


@pytest.mark.jax
def test_jax_parity():
    jax, JB = jax_or_skip()
    rng = np.random.default_rng(3)
    z = rng.standard_normal((2, 41)) + 1j * rng.standard_normal((2, 41))
    for M in (FourierZeroPad(SRC, FourierDomain(320, 1.0, 5, 80, "rfft")),
              FourierCrop(SRC, FourierDomain(200, 1.0, 10, 70, "rfft"), 24)):
        out = jax.jit(lambda m, v: m.apply(v))(M.on(JB), JB.asarray(z))
        assert isinstance(out, jax.Array)
        ref = M.apply(z)
        assert np.max(np.abs(np.asarray(out) - ref)) <= 1e-13 * np.max(np.abs(ref))
